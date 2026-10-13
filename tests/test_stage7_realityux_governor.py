"""Comprehensive test suite for RealityUX Mask Engine Stage 7 Integration:
Surgical Mouth Dynamics, Foreign Object Occlusion Governance, and Non-Target Isolation.

Validates:
1. Dynamic Mouth Aperture & Dental Restoration (MAR-gated, teeth/tongue isolation, feathering).
2. DFL XSeg Foreign Object Occlusion Masking (erosion kernel 3x3, Gaussian feathering, foreground preservation).
3. Multi-Person Tracking & Non-Target Subject Protection (track arbitration, zero-bleed affine transform projection).
4. Dual-mode GPU capability (with BiSeNet parser enabled vs small card XSeg-only fallback).
5. Backwards compatibility with standard processor Run(img1, keywords) signature.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from roop.processors.Mask_RealityUX import Mask_RealityUX
from roop.enhancers.mouth_dynamics import compute_mar
import roop.globals


# =============================================================================
# Helper Fixtures & Synthetic Generators
# =============================================================================

class MockFace:
    """Mock Face object for testing."""

    def __init__(
        self,
        bbox: list[float] | None = None,
        landmark_3d_68: np.ndarray | None = None,
        landmark_2d_106: np.ndarray | None = None,
        kps: np.ndarray | None = None,
        normed_embedding: np.ndarray | None = None,
    ) -> None:
        self.bbox = np.asarray(bbox, dtype=np.float32) if bbox is not None else np.array([50, 50, 150, 150], dtype=np.float32)
        self.landmark_3d_68 = landmark_3d_68
        self.landmark_2d_106 = landmark_2d_106
        self.kps = kps
        self.normed_embedding = normed_embedding


def make_synthetic_68_landmarks(mouth_open: bool = False) -> np.ndarray:
    """Generate 68-point landmarks in crop coordinates [0, 256]."""
    pts = np.zeros((68, 2), dtype=np.float32)
    # Facial contour
    for i in range(17):
        pts[i] = [20.0 + i * 12.0, 100.0 + (i - 8) ** 2 * 1.5]
    # Nose
    for i in range(27, 36):
        pts[i] = [128.0, 60.0 + (i - 27) * 7.0]

    # Mouth corners: 60 left, 64 right (distance = 70 px)
    pts[60] = [93.0, 180.0]
    pts[64] = [163.0, 180.0]

    if mouth_open:
        # Wide vertical opening (~30 px) -> MAR ~ 30*3 / (2*70) = 0.64 > 0.18
        pts[61] = [110.0, 165.0]
        pts[62] = [128.0, 163.0]
        pts[63] = [146.0, 165.0]
        pts[67] = [110.0, 195.0]
        pts[66] = [128.0, 197.0]
        pts[65] = [146.0, 195.0]
    else:
        # Closed mouth (minimal vertical delta ~ 2 px) -> MAR ~ 0.04 < 0.18
        pts[61] = [110.0, 179.0]
        pts[62] = [128.0, 179.0]
        pts[63] = [146.0, 179.0]
        pts[67] = [110.0, 181.0]
        pts[66] = [128.0, 181.0]
        pts[65] = [146.0, 181.0]

    return pts


# =============================================================================
# 1. Dynamic Mouth Aperture & Dental Restoration Tests
# =============================================================================

class TestRealityUXMouthDynamics:
    """Test MAR-gated mouth aperture and teeth restoration in RealityUX."""

    def test_open_mouth_restoration_triggered(self) -> None:
        rux = Mask_RealityUX()
        # Mock underlying engines to isolate RealityUX fusion logic
        rux._xseg.Run = MagicMock(return_value=np.zeros((256, 256), dtype=np.float32))
        rux._parser_enabled = False

        crop = np.full((256, 256, 3), 150, dtype=np.uint8)
        # Add white teeth simulation inside mouth cavity
        crop[170:190, 110:146] = [235, 235, 235]

        face_open = MockFace(landmark_3d_68=make_synthetic_68_landmarks(mouth_open=True))

        mask = rux.Run(
            crop,
            "",
            target_face=face_open,
            restore_original_mouth=True,
            mouth_open_threshold=0.18,
        )

        assert mask.shape == (256, 256)
        assert mask.dtype == np.float32
        assert rux._last_telemetry["mouth_restored"] is True
        assert rux._last_telemetry["mar"] > 0.18
        # In the mouth aperture (teeth region), mask should have high keep-original weight (near 1.0)
        assert mask[180, 128] > 0.40
        # Outside mouth (forehead), mask should remain 0.0 (swappable)
        assert mask[40, 128] == 0.0

    def test_closed_mouth_bypasses_restoration(self) -> None:
        rux = Mask_RealityUX()
        rux._xseg.Run = MagicMock(return_value=np.zeros((256, 256), dtype=np.float32))
        rux._parser_enabled = False

        crop = np.full((256, 256, 3), 150, dtype=np.uint8)
        face_closed = MockFace(landmark_3d_68=make_synthetic_68_landmarks(mouth_open=False))

        mask = rux.Run(
            crop,
            "",
            target_face=face_closed,
            restore_original_mouth=True,
            mouth_open_threshold=0.18,
        )

        assert rux._last_telemetry["mouth_restored"] is False
        assert rux._last_telemetry["mar"] < 0.18
        # Mouth region remains at 0.0 (swapped)
        assert mask[180, 128] == 0.0

    def test_restore_mouth_flag_disabled(self) -> None:
        rux = Mask_RealityUX()
        rux._xseg.Run = MagicMock(return_value=np.zeros((256, 256), dtype=np.float32))
        rux._parser_enabled = False

        crop = np.full((256, 256, 3), 150, dtype=np.uint8)
        face_open = MockFace(landmark_3d_68=make_synthetic_68_landmarks(mouth_open=True))

        # Explicitly disabled
        mask = rux.Run(
            crop,
            "",
            target_face=face_open,
            restore_original_mouth=False,
            mouth_open_threshold=0.18,
        )

        assert rux._last_telemetry["mouth_restored"] is False
        assert mask[180, 128] == 0.0

    def test_bisenet_semantic_oral_cavity_fusion(self) -> None:
        rux = Mask_RealityUX()
        rux._xseg.Run = MagicMock(return_value=np.zeros((256, 256), dtype=np.float32))
        rux._parser_enabled = True

        # Mock BiSeNet labels with mouth cavity class 11
        mock_labels = np.zeros((256, 256), dtype=np.int64)
        mock_labels[170:190, 110:146] = 11  # BiSeNet mouth class

        rux._parser.RunLabels = MagicMock(return_value=mock_labels)
        rux._executor = None

        crop = np.full((256, 256, 3), 150, dtype=np.uint8)
        face_open = MockFace(landmark_3d_68=make_synthetic_68_landmarks(mouth_open=True))

        mask = rux.Run(
            crop,
            "",
            target_face=face_open,
            restore_original_mouth=True,
            mouth_open_threshold=0.18,
        )

        assert rux._last_telemetry["mouth_restored"] is True
        # Mouth area has feathered original retention
        assert mask[180, 128] > 0.50


# =============================================================================
# 2. Foreign Object Occlusion Masking Tests
# =============================================================================

class TestRealityUXOcclusionGovernance:
    """Test foreign object occlusion isolation, erosion, and feathering."""

    def test_occlusion_erosion_and_feathering(self) -> None:
        rux = Mask_RealityUX()
        rux._parser_enabled = False

        # Simulate raw XSeg mask where a microphone/hand occludes part of the face
        mock_xseg = np.zeros((256, 256), dtype=np.float32)
        # Block representing microphone over chin (high values = keep original)
        mock_xseg[160:220, 100:160] = 0.85
        rux._xseg.Run = MagicMock(return_value=mock_xseg)

        crop = np.full((256, 256, 3), 128, dtype=np.uint8)

        mask = rux.Run(
            crop,
            "",
            enable_xseg_occlusion=True,
            occlusion_erosion=3,
        )

        assert rux._last_telemetry["occlusion_applied"] is True
        # Center of microphone is kept original (high mask weight)
        assert mask[190, 130] > 0.80
        # Check smooth gradient at boundary due to feathering
        assert 0.0 < mask[161, 101] <= mask[190, 130]

    def test_occlusion_disabled(self) -> None:
        rux = Mask_RealityUX()
        rux._parser_enabled = False
        mock_xseg = np.zeros((256, 256), dtype=np.float32)
        rux._xseg.Run = MagicMock(return_value=mock_xseg)

        crop = np.full((256, 256, 3), 128, dtype=np.uint8)

        mask = rux.Run(
            crop,
            "",
            enable_xseg_occlusion=False,
        )

        assert rux._last_telemetry["occlusion_applied"] is False


# =============================================================================
# 3. Multi-Person Tracking & Non-Target Face Protection Tests
# =============================================================================

class TestRealityUXNonTargetProtection:
    """Test multi-face arbitration and zero-bleed protection in crop space."""

    def test_non_target_protection_projection(self) -> None:
        rux = Mask_RealityUX()
        rux._parser_enabled = False
        rux._xseg.Run = MagicMock(return_value=np.zeros((256, 256), dtype=np.float32))

        crop = np.full((256, 256, 3), 128, dtype=np.uint8)

        # Multi-person scene: target face and adjacent actor face
        target_f = MockFace(bbox=[100, 100, 300, 300], normed_embedding=np.array([1.0, 0.0, 0.0]))
        actor_f = MockFace(bbox=[320, 120, 500, 320], normed_embedding=np.array([0.0, 1.0, 0.0]))

        # Define affine matrix M that maps full-frame [0..600] to crop [0..256]
        # Scaling by 0.5: crop_x = frame_x * 0.5, crop_y = frame_y * 0.5
        M = np.array([
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
        ], dtype=np.float32)

        # Actor box [320, 120, 500, 320] maps to crop [160, 60, 250, 160]
        mask = rux.Run(
            crop,
            "",
            target_face=target_f,
            frame_faces=[target_f, actor_f],
            M=M,
        )

        assert rux._last_telemetry["non_targets_protected"] == 1
        # Over the projected actor region [crop x=180, y=100], mask must be 1.0 (protected original)
        assert mask[100, 180] > 0.80
        # Over the target face center [crop x=50, y=50], mask must be 0.0 (swappable)
        assert mask[50, 50] == 0.0

    def test_single_face_scene_has_no_non_targets(self) -> None:
        rux = Mask_RealityUX()
        rux._parser_enabled = False
        rux._xseg.Run = MagicMock(return_value=np.zeros((256, 256), dtype=np.float32))

        crop = np.full((256, 256, 3), 128, dtype=np.uint8)
        target_f = MockFace(bbox=[100, 100, 300, 300])

        mask = rux.Run(
            crop,
            "",
            target_face=target_f,
            frame_faces=[target_f],
            M=np.eye(2, 3, dtype=np.float32),
        )

        assert rux._last_telemetry["non_targets_protected"] == 0


# =============================================================================
# 4. Small-Card Profile & Degenerate Cases
# =============================================================================

class TestRealityUXEdgeCases:
    """Test small card fallback mode, degenerate inputs, and backwards compatibility."""

    def test_small_card_profile_fallback(self) -> None:
        rux = Mask_RealityUX()
        rux._parser_enabled = False  # Sub-7GB laptop profile
        rux._xseg.Run = MagicMock(return_value=np.zeros((256, 256), dtype=np.float32))

        crop = np.full((256, 256, 3), 128, dtype=np.uint8)
        face_open = MockFace(landmark_3d_68=make_synthetic_68_landmarks(mouth_open=True))

        # Even with BiSeNet disabled on small GPU, landmarks restore mouth dynamically
        mask = rux.Run(
            crop,
            "",
            target_face=face_open,
            restore_original_mouth=True,
            mouth_open_threshold=0.18,
        )

        assert mask.shape == (256, 256)
        assert rux._last_telemetry["mouth_restored"] is True

    def test_empty_or_none_frame(self) -> None:
        rux = Mask_RealityUX()
        mask_none = rux.Run(None, "")
        assert mask_none.shape == (1, 1)

        mask_empty = rux.Run(np.zeros((0, 0, 3), dtype=np.uint8), "")
        assert mask_empty.shape == (1, 1)

    def test_backwards_compatible_call_signature(self) -> None:
        rux = Mask_RealityUX()
        rux._xseg.Run = MagicMock(return_value=np.zeros((128, 128), dtype=np.float32))
        rux._parser_enabled = False

        crop = np.full((128, 128, 3), 128, dtype=np.uint8)
        # Calling with legacy 2 arguments
        mask = rux.Run(crop, "keywords")
        assert mask.shape == (128, 128)
        assert mask.dtype == np.float32
