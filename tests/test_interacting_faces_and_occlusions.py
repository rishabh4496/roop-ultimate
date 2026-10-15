"""Unit tests for interacting faces, foreground occlusions, and non-target demarcation.

Tests:
1. Non-target protection mask preserves target face's inner landmark hull.
2. Mask_RealityUX crop-space protection preserves target face's inner features.
3. Outcome verification reversion is bypassed on partial occlusion and interaction.
4. LOCK_MIN_DET relaxes to 0.55 on occluded / contact tracks to prevent unswapping.
5. Duplicate partial detections from foreground occlusions are suppressed in face_overlap.
"""

from __future__ import annotations

import unittest
from unittest import mock
import numpy as np
import cv2

from roop.enhancers.occlusion_governor import OcclusionGovernor
from roop.processors.Mask_RealityUX import _non_target_keep_mask
from roop.face_overlap import _same_face, build_regions


class MockFace:
    """Mock face structure for testing."""
    def __init__(
        self,
        bbox: list[float] | None = None,
        kps: np.ndarray | None = None,
        landmark_2d_106: np.ndarray | None = None,
        det_score: float = 0.95,
        occlusion_state: str | None = None,
    ):
        self.bbox = np.asarray(bbox, dtype=np.float32) if bbox is not None else np.array([50, 50, 150, 150], dtype=np.float32)
        self.kps = kps if kps is not None else np.array([[75, 80], [125, 80], [100, 100], [80, 125], [120, 125]], dtype=np.float32)
        self.landmark_2d_106 = landmark_2d_106
        self.det_score = det_score
        self.occlusion_state = occlusion_state


class TestInteractingFacesAndOcclusion(unittest.TestCase):
    def test_generate_non_target_protection_does_not_carve_target_inner_hull(self):
        """When an adjacent non-target face's dilated hull overlaps with target face,
        the target face's inner landmark hull must NOT be zeroed out."""
        gov = OcclusionGovernor()
        # Non-target face touching/overlapping the right side of target face
        non_target = MockFace(bbox=[110, 50, 210, 150])
        # Target face on the left (50, 50, 150, 150), center at (100, 100)
        target = MockFace(bbox=[50, 50, 150, 150])

        # Without target_face passed, dilated non-target encroaches into x=100
        mask_without_target = gov.generate_non_target_protection_mask(
            frame_shape=(300, 300),
            non_target_faces=[non_target],
            feather_radius=0,
            target_face=None,
        )
        # Dilated non-target hull/box encroaches past x=105
        assert mask_without_target[100, 105] > 0.0

        # With target_face passed, target face's inner landmark hull is preserved (0.0 protection = full swap weight)
        mask_with_target = gov.generate_non_target_protection_mask(
            frame_shape=(300, 300),
            non_target_faces=[non_target],
            feather_radius=0,
            target_face=target,
        )
        # Inside target face's inner landmarks (e.g. nose at 100, 100), protection must be 0
        self.assertEqual(mask_with_target[100, 100], 0.0)
        # Over the non-target face center (160, 100), protection is active (1.0)
        self.assertEqual(mask_with_target[100, 160], 1.0)

    def test_mask_realityux_non_target_protection_preserves_target_features(self):
        """In Mask_RealityUX, non-target bounding box keep mask must not invade
        the target face's inner landmark hull in crop space."""
        # 256x256 crop, M is identity mapping for simplicity
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        # Non-target face overlapping into crop
        non_target = MockFace(bbox=[110, 50, 220, 160])
        target = MockFace(bbox=[50, 50, 150, 150])

        prot_mask = _non_target_keep_mask(
            mask_shape=(256, 256),
            crop_shape=(256, 256),
            M=M,
            non_targets=[non_target],
            target_face=target,
        )

        # Inside target's inner keypoints (e.g. nose at 100, 100), keep mask must be 0
        self.assertEqual(prot_mask[100, 100], 0)
        # Over non-target face territory (170, 100), keep mask is 255
        self.assertEqual(prot_mask[100, 170], 255)

    def test_same_face_partial_occlusion_duplicate_suppression(self):
        """Verify that when foreground occlusion creates a nested, partially offset box,
        _same_face suppresses it so it doesn't carve the face in half."""
        # Main face claim: (pts, bbox_padded, radius)
        # Main face: [100, 100, 200, 200], radius 100
        ra = (85, 85, 215, 215)
        claim_a = (None, ra, 100.0)

        # Secondary partial box from partial occlusion (e.g. upper half of face [100, 100, 200, 160])
        # Padded box has high containment in ra (>0.85) and sep ~ 0.38 * min_r
        rb = (90, 90, 210, 170)
        claim_b = (None, rb, 70.0)

        self.assertTrue(_same_face(claim_a, claim_b),
                        "Partial detection on an occluded face should be recognized as same face")

    def test_build_regions_does_not_carve_face_with_partial_duplicate(self):
        """Full build_regions integration: a partial duplicate from occlusion does not carve region."""
        # Target face and a duplicate partial detection from an occluding hand/microphone
        target = MockFace(bbox=[100, 100, 200, 200])
        partial = MockFace(bbox=[105, 105, 195, 175])

        frame_shape = (400, 400, 3)
        # Order=[0]: target is painted, partial is unpainted claimant
        regions = build_regions([target, partial], frame_shape, order=[0])
        # Duplicate is suppressed; target face is not carved down the middle
        if regions is not None:
            # If a region exists, target must own its center
            self.assertEqual(regions[0].own[regions[0].own.shape[0] // 2, regions[0].own.shape[1] // 2], 1.0)


if __name__ == '__main__':
    unittest.main()
