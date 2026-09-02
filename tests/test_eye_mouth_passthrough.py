"""Tests for EAR blink detection, multi-scale eyelid blending, and teeth/inner-mouth passthrough."""

import os
import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / 'app'
for p in (str(REPO_ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from roop.processors.frame import face_swapper


def create_mock_68_landmarks(
    left_eye_open: bool = True,
    right_eye_open: bool = True,
    mouth_open: bool = False
) -> np.ndarray:
    """Synthesize 68-point facial landmark coordinates (256x256 space)."""
    pts = np.zeros((68, 2), dtype=np.float32)

    # Jaw (0..16)
    for i in range(17):
        pts[i] = [40 + i * 10, 100 + abs(i - 8) * 10]

    # Eyebrows (17..26)
    for i in range(5):
        pts[17 + i] = [55 + i * 12, 70]
        pts[22 + i] = [145 + i * 12, 70]

    # Nose (27..35)
    for i in range(9):
        pts[27 + i] = [128, 85 + i * 7]

    # Left eye (36..41) - center roughly (80, 95), width 30
    pts[36] = [65, 95]    # lateral corner
    pts[39] = [95, 95]    # medial corner
    if left_eye_open:
        pts[37] = [75, 87]
        pts[38] = [85, 87]
        pts[40] = [85, 103]
        pts[41] = [75, 103]
    else:
        # Closed eye: squeezed vertically
        pts[37] = [75, 94.5]
        pts[38] = [85, 94.5]
        pts[40] = [85, 95.5]
        pts[41] = [75, 95.5]

    # Right eye (42..47) - center roughly (176, 95), width 30
    pts[42] = [161, 95]   # medial corner
    pts[45] = [191, 95]   # lateral corner
    if right_eye_open:
        pts[43] = [171, 87]
        pts[44] = [181, 87]
        pts[46] = [181, 103]
        pts[47] = [171, 103]
    else:
        # Closed eye: squeezed vertically
        pts[43] = [171, 94.5]
        pts[44] = [181, 94.5]
        pts[46] = [181, 95.5]
        pts[47] = [171, 95.5]

    # Outer mouth (48..59)
    pts[48] = [95, 180]
    pts[54] = [161, 180]

    # Inner mouth (60..67)
    pts[60] = [105, 180]  # left corner
    pts[64] = [151, 180]  # right corner
    if mouth_open:
        # Vertical separation = 18 pixels (> 8 pixels)
        pts[61] = [118, 172]
        pts[62] = [128, 171]  # upper center
        pts[63] = [138, 172]
        pts[65] = [138, 188]
        pts[66] = [128, 189]  # lower center
        pts[67] = [118, 188]
    else:
        # Closed mouth: vertical separation = 2 pixels (<= 8 pixels)
        pts[61] = [118, 179]
        pts[62] = [128, 179]
        pts[63] = [138, 179]
        pts[65] = [138, 181]
        pts[66] = [128, 181]
        pts[67] = [118, 181]

    return pts


class EyeMouthPassthroughTest(unittest.TestCase):
    def setUp(self):
        self.shape = (256, 256)
        # Mock target actor image (e.g. natural skin tone BGR = 120, 160, 210)
        self.target_crop = np.full((256, 256, 3), (120, 160, 210), dtype=np.uint8)
        # Mock swapped image (contrasting green tone BGR = 40, 210, 40)
        self.swapped_crop = np.full((256, 256, 3), (40, 210, 40), dtype=np.uint8)

    def test_ear_calculation_open_and_closed_eyes(self):
        """Confirm EAR formula differentiates open eyes (> 0.25) from closed eyes (< 0.21)."""
        open_pts = create_mock_68_landmarks(left_eye_open=True, right_eye_open=True)
        l_ear_open, r_ear_open, mean_open = face_swapper.compute_eye_aspect_ratios(open_pts)

        self.assertGreater(l_ear_open, 0.25, "Open left eye EAR should be well above threshold")
        self.assertGreater(r_ear_open, 0.25, "Open right eye EAR should be well above threshold")

        closed_pts = create_mock_68_landmarks(left_eye_open=False, right_eye_open=False)
        l_ear_closed, r_ear_closed, mean_closed = face_swapper.compute_eye_aspect_ratios(closed_pts)

        self.assertLess(l_ear_closed, 0.21, "Closed left eye EAR must be below 0.21 threshold")
        self.assertLess(r_ear_closed, 0.21, "Closed right eye EAR must be below 0.21 threshold")

    def test_ear_triggers_eyelid_passthrough_mask(self):
        """Validating that EAR correctly triggers the passthrough mask when fed closed-eye coordinates."""
        closed_pts = create_mock_68_landmarks(left_eye_open=False, right_eye_open=False)
        mask_closed = face_swapper.build_blink_eyelid_mask(closed_pts, self.shape, ear_threshold=0.21)

        # Confirm mask is triggered and active in the eye regions
        left_eye_region = mask_closed[85:105, 65:95]
        right_eye_region = mask_closed[85:105, 160:190]
        self.assertGreater(float(left_eye_region.mean()), 0.50,
                           "Eyelid passthrough mask should be active over closed left eye")
        self.assertGreater(float(right_eye_region.mean()), 0.50,
                           "Eyelid passthrough mask should be active over closed right eye")

        # Open eyes must NOT trigger the passthrough mask
        open_pts = create_mock_68_landmarks(left_eye_open=True, right_eye_open=True)
        mask_open = face_swapper.build_blink_eyelid_mask(open_pts, self.shape, ear_threshold=0.21)
        self.assertEqual(float(np.sum(mask_open)), 0.0,
                         "Open eyes should produce an empty eyelid passthrough mask")

    def test_multiscale_eyelid_blend_preserves_target_blinks(self):
        """Confirm multi-scale blending composites target eyelid seamlessly onto swapped face."""
        closed_pts = create_mock_68_landmarks(left_eye_open=False, right_eye_open=True)
        eyelid_mask = face_swapper.build_blink_eyelid_mask(closed_pts, self.shape)

        blended = face_swapper.blend_eyelid_multiscale(self.target_crop, self.swapped_crop, eyelid_mask)

        # The closed left eye region in blended crop should match target crop (B~120, R~210, G~160)
        left_eye_blended = blended[90:100, 75:85]
        self.assertGreater(float(left_eye_blended[..., 0].mean()), 100.0,
                           "Target actor's eyelid blue channel should be preserved")
        self.assertGreater(float(left_eye_blended[..., 2].mean()), 180.0,
                           "Target actor's eyelid red channel should be preserved")
        self.assertLess(float(left_eye_blended[..., 1].mean()), 175.0,
                        "Swapped green eye (210) should be attenuated toward target eyelid (160)")

        # The open right eye should remain the swapped face (green channel high ~210)
        right_eye_blended = blended[90:100, 170:180]
        self.assertGreater(float(right_eye_blended[..., 1].mean()), 180.0,
                           "Open eye should retain swapped face buffer")

    def test_restorer_attenuation_on_closed_eye_bbox(self):
        """Confirm GPEN/restorer enhancement is attenuated on the closed eye bounding box."""
        closed_pts = create_mock_68_landmarks(left_eye_open=False, right_eye_open=True)
        is_blinking, att_mask = face_swapper.get_closed_eyes_attenuation(closed_pts, self.shape)

        self.assertTrue(is_blinking, "Blink state must be detected")
        # Attenuation mask should be near 0 over closed eye bbox
        left_eye_att = att_mask[90:100, 70:90]
        self.assertLess(float(left_eye_att.mean()), 0.30,
                        "Restorer enhancement should be attenuated over closed eye")

        # Rest of face should have normal enhancement (1.0)
        forehead_att = att_mask[30:50, 100:150]
        self.assertGreater(float(forehead_att.mean()), 0.95,
                           "Rest of face should receive full restoration")

    def test_inner_mouth_passthrough_when_lip_separation_greater_than_8px(self):
        """Confirm inner mouth mask activates and feathers by 3px when lip separation > 8px."""
        open_mouth_pts = create_mock_68_landmarks(mouth_open=True)
        mouth_mask = face_swapper.extract_inner_mouth_mask(open_mouth_pts, self.shape, min_separation=8.0, feather_px=3)

        # Check mask activation inside oral cavity
        cavity_center = mouth_mask[178:182, 125:131]
        self.assertGreater(float(cavity_center.mean()), 0.80,
                           "Oral cavity center should have high passthrough weight")

        # Check 3-pixel contour feathering
        # Boundary contour pixels should have smooth intermediate values (0.05 < value < 0.95)
        boundary_px = mouth_mask[172, 128]
        self.assertTrue(0.05 < boundary_px < 0.95,
                        f"Boundary should be softly feathered by 3px, got {boundary_px}")

        # Check passthrough composite preserves target mouth
        composite = face_swapper.blend_inner_mouth_passthrough(self.target_crop, self.swapped_crop, mouth_mask)
        mouth_pixels = composite[178:182, 125:131]
        self.assertGreater(float(mouth_pixels[..., 0].mean()), 100.0,
                           "Target actor's native oral cavity should be passed through")

    def test_inner_mouth_passthrough_inactive_when_mouth_closed(self):
        """Confirm inner mouth mask is completely disabled when lip separation <= 8px."""
        closed_mouth_pts = create_mock_68_landmarks(mouth_open=False)
        mouth_mask = face_swapper.extract_inner_mouth_mask(closed_mouth_pts, self.shape, min_separation=8.0)

        self.assertEqual(float(np.sum(mouth_mask)), 0.0,
                         "Closed mouth (sep <= 8px) must produce an empty inner mouth mask")

    def test_unified_facial_dynamics_pipeline(self):
        """Confirm unified apply_facial_dynamics integrates both blinks and inner-mouth retention."""
        pts = create_mock_68_landmarks(left_eye_open=False, right_eye_open=True, mouth_open=True)
        out, meta = face_swapper.apply_facial_dynamics(self.target_crop, self.swapped_crop, pts)

        self.assertTrue(meta['is_blinking'])
        self.assertTrue(meta['mouth_open'])
        self.assertLess(meta['left_ear'], 0.21)
        self.assertGreater(meta['right_ear'], 0.25)
        self.assertGreater(meta['lip_separation'], 8.0)

        # Both left eye and inner mouth should be blended from target
        left_eye_pixels = out[90:100, 75:85]
        self.assertGreater(float(left_eye_pixels[..., 0].mean()), 100.0)

        mouth_pixels = out[178:182, 125:131]
        self.assertGreater(float(mouth_pixels[..., 0].mean()), 100.0)


if __name__ == '__main__':
    unittest.main()
