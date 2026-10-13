"""Unit and integration test suite for Stage 7: Surgical Mouth Dynamics,
Non-Target Isolation, and Occlusion Governance.

Covers:
1. Mouth Aspect Ratio (MAR) computation across landmark formats (68-pt, 106-pt, 5-pt, degenerate).
2. Dynamic mouth aperture & dental restoration (MouthDynamicsEngine).
3. Foreign object occlusion masking and erosion/feathering (OcclusionGovernor).
4. Multi-face track arbitration (IoU and ArcFace cosine similarity).
5. Non-target face protection mask generation and zero-bleed affine warp trimming.
6. AnatomicalEngine and PipelineManager end-to-end orchestration.
7. UltraMax parameters and REST route schemas for Stage 7 settings.
"""

from __future__ import annotations

import types
import numpy as np
import pytest

from roop.enhancers.mouth_dynamics import MouthDynamicsEngine, compute_mar
from roop.enhancers.occlusion_governor import (
    OcclusionGovernor,
    compute_cosine_similarity,
    compute_iou,
)
from roop.enhancers.anatomical_engine import AnatomicalEngine
from roop.pipeline_manager import PipelineManager
from roop.api.enhancer_routes import (
    UltraMaxParameters,
    UltraMaxUpdateParameters,
    UltraMaxRuntimeState,
)
import roop.globals


# =============================================================================
# Helper Fixtures and Synthetic Generators
# =============================================================================

class MockFace:
    """Mock face object with bounding box and landmark attributes."""

    def __init__(
        self,
        bbox: list[float] | None = None,
        landmark_3d_68: np.ndarray | None = None,
        landmark_2d_106: np.ndarray | None = None,
        kps: np.ndarray | None = None,
        embedding: np.ndarray | None = None,
        normed_embedding: np.ndarray | None = None,
    ) -> None:
        self.bbox = np.asarray(bbox, dtype=np.float32) if bbox is not None else np.array([50, 50, 150, 150], dtype=np.float32)
        self.landmark_3d_68 = landmark_3d_68
        self.landmark_2d_106 = landmark_2d_106
        self.kps = kps
        self.embedding = embedding
        self.normed_embedding = normed_embedding


def make_synthetic_68_landmarks(mouth_open: bool = False) -> np.ndarray:
    """Generate 68-point landmarks with open or closed mouth."""
    pts = np.zeros((68, 2), dtype=np.float32)
    # Face outline (0-16)
    for i in range(17):
        pts[i] = [20.0 + i * 10.0, 100.0 + (i - 8) ** 2 * 1.5]
    # Nose (27-35)
    for i in range(27, 36):
        pts[i] = [100.0, 50.0 + (i - 27) * 5.0]

    # Mouth outer lip: 48 to 59
    # Mouth inner lip: 60 to 67
    # 60: left corner, 64: right corner (horizontal distance = 60 px)
    pts[60] = [70.0, 140.0]
    pts[64] = [130.0, 140.0]

    if mouth_open:
        # Wide vertical separation (e.g., vertical delta ~ 24 px)
        # 61, 62, 63 upper lip
        pts[61] = [85.0, 128.0]
        pts[62] = [100.0, 126.0]
        pts[63] = [115.0, 128.0]
        # 67, 66, 65 lower lip
        pts[67] = [85.0, 152.0]
        pts[66] = [100.0, 154.0]
        pts[65] = [115.0, 152.0]
    else:
        # Closed mouth (minimal vertical delta ~ 2 px)
        pts[61] = [85.0, 139.0]
        pts[62] = [100.0, 139.0]
        pts[63] = [115.0, 139.0]
        pts[67] = [85.0, 141.0]
        pts[66] = [100.0, 141.0]
        pts[65] = [115.0, 141.0]

    return pts


# =============================================================================
# 1. Mouth Aspect Ratio (MAR) Calculation Tests
# =============================================================================

class TestMouthAspectRatio:
    """Test MAR calculation logic across various landmark layouts."""

    def test_mar_closed_mouth_68(self) -> None:
        pts = make_synthetic_68_landmarks(mouth_open=False)
        mar = compute_mar(pts)
        assert mar > 0.0
        # Closed mouth MAR should be below standard open threshold 0.18
        assert mar < 0.12

    def test_mar_open_mouth_68(self) -> None:
        pts = make_synthetic_68_landmarks(mouth_open=True)
        mar = compute_mar(pts)
        # Open mouth MAR should clearly exceed 0.18
        assert mar > 0.30

    def test_mar_106_point_landmarks_is_unknown_not_invented(self) -> None:
        # The 106-point layout carries no usable aperture signal (measured on
        # 298 real faces: the old formula called 100% of mouths open, r=-0.05
        # against the 68-point MAR). It must say "unknown" (0.0 -> no restore),
        # and must never be read through the 68-point indices.
        pts = np.zeros((106, 2), dtype=np.float32)
        pts[52:71, 0] = np.linspace(70, 130, 19)
        pts[52:61, 1] = 125.0
        pts[61:71, 1] = 155.0   # a "wide open" looking box
        assert compute_mar(pts) == 0.0

    def test_mar_68_branch_not_taken_by_larger_arrays(self) -> None:
        open68 = make_synthetic_68_landmarks(mouth_open=True)
        assert compute_mar(open68) > 0.30
        padded = np.zeros((106, 2), dtype=np.float32)
        padded[:68] = open68
        assert compute_mar(padded) == 0.0

    def test_mar_5_point_landmarks(self) -> None:
        pts = np.array([
            [30.0, 30.0],
            [70.0, 30.0],
            [50.0, 50.0],
            [35.0, 70.0],
            [65.0, 70.0],
        ], dtype=np.float32)
        mar = compute_mar(pts)
        assert mar == 0.10

    def test_mar_degenerate_cases(self) -> None:
        # None input
        assert compute_mar(None) == 0.0

        # Empty array
        assert compute_mar(np.zeros((0, 2))) == 0.0

        # NaN / Inf coordinates
        nan_pts = make_synthetic_68_landmarks()
        nan_pts[60] = [np.nan, 100.0]
        assert compute_mar(nan_pts) == 0.0

        inf_pts = make_synthetic_68_landmarks()
        inf_pts[64] = [np.inf, 100.0]
        assert compute_mar(inf_pts) == 0.0

        # Collocated mouth corners (zero width)
        zero_width = make_synthetic_68_landmarks()
        zero_width[64] = zero_width[60]
        assert compute_mar(zero_width) == 0.0


# =============================================================================
# 2. Mouth Dynamics Engine Tests
# =============================================================================

class TestMouthDynamicsEngine:
    """Test inner mouth mask generation and dynamic compositing."""

    def test_inner_lip_points_extraction(self) -> None:
        engine = MouthDynamicsEngine()
        pts68 = make_synthetic_68_landmarks(mouth_open=True)
        face = MockFace(landmark_3d_68=pts68)

        inner = engine.get_inner_lip_points(face, (256, 256))
        assert inner is not None
        assert inner.shape == (8, 2)

    def test_inner_mouth_mask_generation(self) -> None:
        engine = MouthDynamicsEngine()
        crop = np.full((256, 256, 3), 180, dtype=np.uint8)
        # Draw simulated white teeth in mouth area
        crop[130:150, 80:120] = [240, 240, 240]

        inner_pts = np.array([
            [70, 140], [85, 128], [100, 126], [115, 128],
            [130, 140], [115, 152], [100, 154], [85, 152],
        ], dtype=np.float32)

        mask = engine.generate_inner_mouth_mask(crop, inner_pts, feather_radius=4)
        assert mask.shape == (256, 256)
        assert mask.dtype == np.float32
        assert 0.0 <= mask.min() <= mask.max() <= 1.0
        # Mouth region should have positive weight
        assert mask[140, 100] > 0.20
        # Distant forehead region should be 0.0
        assert mask[20, 100] == 0.0

    def test_composite_mouth_aperture(self) -> None:
        engine = MouthDynamicsEngine()
        swapped = np.full((100, 100, 3), 50, dtype=np.uint8)
        original = np.full((100, 100, 3), 200, dtype=np.uint8)

        mask = np.zeros((100, 100), dtype=np.float32)
        mask[40:60, 40:60] = 1.0  # Full mouth aperture

        blended = engine.composite_mouth_aperture(swapped, original, mask)
        assert blended.shape == (100, 100, 3)
        assert blended.dtype == np.uint8
        # Center should equal original (200)
        assert blended[50, 50, 0] == 200
        # Outside should equal swapped (50)
        assert blended[10, 10, 0] == 50

    def test_process_trigger_on_open_mouth(self) -> None:
        engine = MouthDynamicsEngine(mar_threshold=0.18)
        pts_open = make_synthetic_68_landmarks(mouth_open=True)
        face_open = MockFace(landmark_3d_68=pts_open)

        swapped = np.full((256, 256, 3), 60, dtype=np.uint8)
        original = np.full((256, 256, 3), 210, dtype=np.uint8)

        restored, triggered, mar = engine.process(swapped, original, face_open)
        assert triggered is True
        assert mar > 0.18
        # Center should have been restored towards original
        assert restored[140, 100, 0] > 60

    def test_process_bypass_on_closed_mouth(self) -> None:
        engine = MouthDynamicsEngine(mar_threshold=0.18)
        pts_closed = make_synthetic_68_landmarks(mouth_open=False)
        face_closed = MockFace(landmark_3d_68=pts_closed)

        swapped = np.full((256, 256, 3), 60, dtype=np.uint8)
        original = np.full((256, 256, 3), 210, dtype=np.uint8)

        restored, triggered, mar = engine.process(swapped, original, face_closed)
        assert triggered is False
        assert mar < 0.18
        # Unchanged from swapped
        np.testing.assert_array_equal(restored, swapped)


# =============================================================================
# 3. Foreign Object Occlusion Governor Tests
# =============================================================================

class TestOcclusionGovernor:
    """Test XSeg occlusion masking, morphological erosion, and compositing."""

    def test_compute_iou(self) -> None:
        box1 = [0.0, 0.0, 10.0, 10.0]
        box2 = [0.0, 0.0, 10.0, 10.0]
        assert pytest.approx(compute_iou(box1, box2)) == 1.0

        box3 = [20.0, 20.0, 30.0, 30.0]
        assert compute_iou(box1, box3) == 0.0

        box4 = [5.0, 0.0, 15.0, 10.0]
        # Intersection = 5x10 = 50, Union = 100 + 100 - 50 = 150 -> IoU = 1/3
        assert pytest.approx(compute_iou(box1, box4), rel=1e-3) == 0.3333

    def test_compute_cosine_similarity(self) -> None:
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        assert pytest.approx(compute_cosine_similarity(v1, v2)) == 1.0

        v3 = np.array([0.0, 1.0, 0.0])
        assert pytest.approx(compute_cosine_similarity(v1, v3)) == 0.0

        v4 = np.array([-1.0, 0.0, 0.0])
        assert pytest.approx(compute_cosine_similarity(v1, v4)) == -1.0

        assert compute_cosine_similarity(None, v1) == 0.0
        assert compute_cosine_similarity(np.zeros(3), v1) == 0.0

    def test_occlusion_mask_prediction_and_erosion(self) -> None:
        gov = OcclusionGovernor()
        crop = np.full((128, 128, 3), 128, dtype=np.uint8)

        mask = gov.predict_xseg_occlusion_mask(crop, erosion_kernel=3, feather_radius=4)
        assert mask.shape == (128, 128)
        assert mask.dtype == np.float32
        assert 0.0 <= mask.min() <= mask.max() <= 1.0

    def test_composite_occlusions(self) -> None:
        gov = OcclusionGovernor()
        enhanced = np.full((100, 100, 3), 40, dtype=np.uint8)
        original = np.full((100, 100, 3), 220, dtype=np.uint8)

        # Occlusion mask representing a foreground hand/microphone
        occ_mask = np.zeros((100, 100), dtype=np.float32)
        occ_mask[30:70, 30:70] = 1.0

        composited = gov.composite_occlusions(enhanced, original, occ_mask)
        assert composited.shape == (100, 100, 3)
        assert composited[50, 50, 0] == 220
        assert composited[10, 10, 0] == 40


# =============================================================================
# 4. Multi-Face Arbitration and Non-Target Protection Tests
# =============================================================================

class TestMultiFaceArbitrationAndProtection:
    """Test non-target identification and zero-bleed protection masking."""

    def test_arbitrate_multi_face_tracks_by_embedding(self) -> None:
        gov = OcclusionGovernor()

        ref_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        target_f = MockFace(bbox=[10, 10, 80, 80], normed_embedding=np.array([0.95, 0.05, 0.0]))
        actor_f = MockFace(bbox=[120, 10, 200, 80], normed_embedding=np.array([0.0, 1.0, 0.0]))

        resolved, non_targets = gov.arbitrate_multi_face_tracks(
            detected_faces=[actor_f, target_f],
            target_reference_embedding=ref_emb,
            similarity_threshold=0.60,
        )

        assert resolved is target_f
        assert non_targets == [actor_f]

    def test_arbitrate_multi_face_tracks_by_target_face_object(self) -> None:
        gov = OcclusionGovernor()

        target_f = MockFace(bbox=[10, 10, 80, 80])
        actor_f1 = MockFace(bbox=[100, 10, 180, 80])
        actor_f2 = MockFace(bbox=[200, 10, 280, 80])

        resolved, non_targets = gov.arbitrate_multi_face_tracks(
            detected_faces=[target_f, actor_f1, actor_f2],
            target_reference_embedding=None,
            target_face=target_f,
        )

        assert resolved is target_f
        assert len(non_targets) == 2
        assert actor_f1 in non_targets
        assert actor_f2 in non_targets

    def test_generate_non_target_protection_mask(self) -> None:
        gov = OcclusionGovernor()
        non_target = MockFace(bbox=[100, 100, 200, 200])

        mask = gov.generate_non_target_protection_mask(
            frame_shape=(400, 400),
            non_target_faces=[non_target],
            feather_radius=5,
        )

        assert mask.shape == (400, 400)
        assert mask.dtype == np.float32
        # Protected center should be high weight (near 1.0)
        assert mask[150, 150] > 0.80
        # Distant coordinate should be 0.0
        assert mask[20, 20] == 0.0

    def test_apply_protection_to_target_warp_zero_bleed(self) -> None:
        gov = OcclusionGovernor()
        # Simulated target warp matte covering the frame
        target_matte = np.full((300, 300), 0.90, dtype=np.float32)

        # Protection mask over non-target face region
        protection_mask = np.zeros((300, 300), dtype=np.float32)
        protection_mask[100:200, 100:200] = 1.0

        safe_matte = gov.apply_protection_to_target_warp(target_matte, protection_mask)

        # Over the non-target face, weight must be 0.0 (zero bleed)
        assert safe_matte[150, 150] == 0.0
        # Outside, target matte retains its full weight
        assert pytest.approx(safe_matte[20, 20]) == 0.90


# =============================================================================
# 5. AnatomicalEngine & PipelineManager Integration Tests
# =============================================================================

class TestAnatomicalEngineAndPipelineManager:
    """Test unified pipeline coordination and telemetry propagation."""

    def test_anatomical_engine_process_crop(self) -> None:
        engine = AnatomicalEngine.get_instance()
        pts_open = make_synthetic_68_landmarks(mouth_open=True)
        face = MockFace(landmark_3d_68=pts_open)

        enhanced = np.full((128, 128, 3), 70, dtype=np.uint8)
        original = np.full((128, 128, 3), 190, dtype=np.uint8)

        out, telemetry = engine.process_enhanced_crop(
            enhanced_crop=enhanced,
            original_crop=original,
            target_face=face,
            restore_original_mouth=True,
            mouth_open_threshold=0.18,
            enable_xseg_occlusion=False,
            occlusion_erosion=3,
        )

        assert out.shape == (128, 128, 3)
        assert telemetry["mouth_restored"] is True
        assert telemetry["mar"] > 0.18
        assert telemetry["occlusion_applied"] is False

    def test_anatomical_engine_protection_pipeline(self) -> None:
        engine = AnatomicalEngine.get_instance()
        target = MockFace(bbox=[20, 20, 80, 80])
        actor = MockFace(bbox=[120, 20, 180, 80])

        full_matte = np.ones((200, 200), dtype=np.float32)

        safe_matte, prot_mask = engine.arbitrate_and_protect_frame(
            frame_shape=(200, 200),
            detected_faces=[target, actor],
            target_face=target,
            target_matte=full_matte,
        )

        assert prot_mask[50, 150] > 0.50
        assert safe_matte[50, 150] == 0.0
        assert safe_matte[50, 50] > 0.0

    def test_pipeline_manager_global_fallbacks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(roop.globals, "restore_original_mouth", True)
        monkeypatch.setattr(roop.globals, "mouth_open_threshold", 0.18)
        monkeypatch.setattr(roop.globals, "enable_xseg_occlusion", False)
        monkeypatch.setattr(roop.globals, "occlusion_erosion", 3)

        pm = PipelineManager.get_instance()
        pts_open = make_synthetic_68_landmarks(mouth_open=True)
        face = MockFace(landmark_3d_68=pts_open)

        enhanced = np.full((128, 128, 3), 80, dtype=np.uint8)
        original = np.full((128, 128, 3), 200, dtype=np.uint8)

        out, telem = pm.process_enhancer_crop(
            enhanced_crop=enhanced,
            original_crop=original,
            target_face=face,
        )

        assert telem["mouth_restored"] is True
        assert telem["mar"] > 0.18


# =============================================================================
# 6. UltraMax Parameters and API Schema Validation Tests
# =============================================================================

class TestUltraMaxApiSchemas:
    """Test Stage 7 REST schemas, field constraints, and runtime synchronization."""

    def test_default_parameters(self) -> None:
        params = UltraMaxParameters()
        assert params.restore_original_mouth is False
        assert params.mouth_open_threshold == 0.18
        assert params.enable_xseg_occlusion is False
        assert params.occlusion_erosion == 3

    def test_validation_constraints(self) -> None:
        # Invalid MAR threshold (< 0.05 or > 0.5)
        with pytest.raises(ValueError):
            UltraMaxParameters(mouth_open_threshold=0.01)

        with pytest.raises(ValueError):
            UltraMaxParameters(mouth_open_threshold=0.90)

        # Invalid occlusion erosion (< 1 or > 5)
        with pytest.raises(ValueError):
            UltraMaxParameters(occlusion_erosion=0)

        with pytest.raises(ValueError):
            UltraMaxParameters(occlusion_erosion=10)

    def test_runtime_state_sync_to_globals(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = UltraMaxRuntimeState()
        patch = UltraMaxUpdateParameters(
            restore_original_mouth=True,
            mouth_open_threshold=0.22,
            enable_xseg_occlusion=True,
            occlusion_erosion=4,
        )
        updated = state.update_settings(patch)

        assert updated.restore_original_mouth is True
        assert updated.mouth_open_threshold == 0.22
        assert updated.enable_xseg_occlusion is True
        assert updated.occlusion_erosion == 4

        # Verify synced to roop.globals
        assert roop.globals.restore_original_mouth is True
        assert roop.globals.mouth_open_threshold == 0.22
        assert roop.globals.enable_xseg_occlusion is True
        assert roop.globals.occlusion_erosion == 4
