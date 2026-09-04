"""Unit tests for motion-blur harmonization and Eye Aspect Ratio (EAR) blink preservation."""

import os
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / 'app'
for p in (str(APP_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from roop import motion_blur
from roop import eyelid_preserver
from roop.processors.frame import face_swapper


def create_mock_68_landmarks(
    left_eye_open: bool = True,
    right_eye_open: bool = True,
    mouth_open: bool = False
) -> np.ndarray:
    """Synthesize 68-point facial landmark coordinates (256x256 space)."""
    pts = np.zeros((68, 2), dtype=np.float32)

    # Jawline (0..16)
    for i in range(17):
        pts[i] = [40 + i * 10, 100 + abs(i - 8) * 10]

    # Eyebrows (17..26)
    for i in range(5):
        pts[17 + i] = [55 + i * 12, 70]
        pts[22 + i] = [145 + i * 12, 70]

    # Nose (27..35)
    for i in range(9):
        pts[27 + i] = [128, 85 + i * 7]

    # Left eye (36..41) - center ~ (80, 95), width 30
    pts[36] = [65, 95]    # lateral corner (p1)
    pts[39] = [95, 95]    # medial corner (p4)
    if left_eye_open:
        pts[37] = [75, 87]   # p2
        pts[38] = [85, 87]   # p3
        pts[40] = [85, 103]  # p5
        pts[41] = [75, 103]  # p6
    else:
        # Closed eye: tight vertical slit
        pts[37] = [75, 94.5]
        pts[38] = [85, 94.5]
        pts[40] = [85, 95.5]
        pts[41] = [75, 95.5]

    # Right eye (42..47) - center ~ (176, 95), width 30
    pts[42] = [161, 95]   # medial corner (p1)
    pts[45] = [191, 95]   # lateral corner (p4)
    if right_eye_open:
        pts[43] = [171, 87]   # p2
        pts[44] = [181, 87]   # p3
        pts[46] = [181, 103]  # p5
        pts[47] = [171, 103]  # p6
    else:
        pts[43] = [171, 94.5]
        pts[44] = [181, 94.5]
        pts[46] = [181, 95.5]
        pts[47] = [171, 95.5]

    # Outer mouth (48..59)
    pts[48] = [95, 180]
    pts[54] = [161, 180]

    # Inner mouth (60..67)
    pts[60] = [105, 180]
    pts[64] = [151, 180]
    if mouth_open:
        pts[61] = [118, 172]
        pts[62] = [128, 171]
        pts[63] = [138, 172]
        pts[65] = [138, 188]
        pts[66] = [128, 189]
        pts[67] = [118, 188]
    else:
        pts[61] = [118, 179]
        pts[62] = [128, 179]
        pts[63] = [138, 179]
        pts[65] = [138, 181]
        pts[66] = [128, 181]
        pts[67] = [118, 181]

    return pts


class MotionBlurAndBlinkPreservationTest(unittest.TestCase):
    def setUp(self):
        self.shape = (256, 256)
        # Synthetic distinct target actor crop (natural warm skin tone: B=120, G=160, R=210)
        self.target_crop = np.full((256, 256, 3), (120, 160, 210), dtype=np.uint8)
        # Synthetic distinct swapped face crop (contrasting green tone: B=40, G=210, R=40)
        self.swapped_crop = np.full((256, 256, 3), (40, 210, 40), dtype=np.uint8)

    # ==========================================================================
    # 1. Motion Blur Estimation & Directional Convolution Tests
    # ==========================================================================

    def test_laplacian_variance_blur_metric(self):
        """Confirm B_target = Var(Laplacian(I_target_gray)) discriminates sharp vs blurred images."""
        # Create a textured image with high-frequency edges
        np.random.seed(42)
        sharp_img = np.random.randint(40, 220, (128, 128, 3), dtype=np.uint8)
        sharp_metric = motion_blur.compute_blur_metric(sharp_img)

        # Apply strong blur to the same image
        blurred_img = cv2.GaussianBlur(sharp_img, (15, 15), 4.0)
        blurred_metric = motion_blur.compute_blur_metric(blurred_img)

        self.assertGreater(sharp_metric, 500.0,
                           "Sharp image should produce high Laplacian variance")
        self.assertLess(blurred_metric, 100.0,
                        "Blurred image must fall below 100.0 blur threshold")
        self.assertLess(blurred_metric, sharp_metric * 0.1,
                        "Blur metric must drop substantially after low-pass filtering")

    def test_optical_flow_vector_calculation(self):
        """Confirm optical flow accurately recovers translation vector (u, v) between frames."""
        np.random.seed(101)
        h, w = 128, 128
        base_frame = cv2.GaussianBlur(
            np.random.randint(50, 200, (h, w, 3), dtype=np.uint8),
            (7, 7), 2.0
        )

        # Known displacement vector (u=6.0, v=-4.0)
        gt_u, gt_v = 6.0, -4.0
        M = np.float32([[1, 0, gt_u], [0, 1, gt_v]])
        shifted_frame = cv2.warpAffine(base_frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        est_u, est_v = motion_blur.calculate_optical_flow_vector(base_frame, shifted_frame)

        self.assertAlmostEqual(est_u, gt_u, delta=1.5,
                               msg=f"Estimated u={est_u} should match ground truth u={gt_u}")
        self.assertAlmostEqual(est_v, gt_v, delta=1.5,
                               msg=f"Estimated v={est_v} should match ground truth v={gt_v}")

    def test_motion_blur_kernel_construction(self):
        """Confirm anisotropic motion blur kernel K has correct length, orientation, and normalization."""
        # Test horizontal motion: u = 8.0, v = 0.0 => length = 8.0, angle = 0
        k_h = motion_blur.construct_motion_blur_kernel_from_vector(8.0, 0.0)
        self.assertAlmostEqual(float(k_h.sum()), 1.0, places=5,
                               msg="Kernel weights must sum to 1.0 (energy conservation)")
        self.assertEqual(k_h.shape[0] % 2, 1, "Kernel dimensions must be odd")
        self.assertEqual(k_h.shape[0], k_h.shape[1], "Kernel must be square")

        # In horizontal kernel, non-center rows should have zero or near-zero energy
        c = k_h.shape[0] // 2
        center_row_energy = float(k_h[c, :].sum())
        self.assertGreater(center_row_energy, 0.90,
                           "Horizontal motion kernel should concentrate energy along the center row")

        # Test vertical motion: u = 0.0, v = 8.0 => length = 8.0, angle = pi/2
        k_v = motion_blur.construct_motion_blur_kernel_from_vector(0.0, 8.0)
        self.assertAlmostEqual(float(k_v.sum()), 1.0, places=5)
        center_col_energy = float(k_v[:, c].sum())
        self.assertGreater(center_col_energy, 0.90,
                           "Vertical motion kernel should concentrate energy along the center column")

        # Test sub-pixel motion (length < 1.0): must return identity kernel [[1.0]]
        k_sub = motion_blur.construct_motion_blur_kernel_from_vector(0.3, 0.4)
        self.assertEqual(k_sub.shape, (1, 1))
        self.assertAlmostEqual(float(k_sub[0, 0]), 1.0)

    def test_motion_blur_harmonization_adaptive_gating(self):
        """Confirm motion blur is applied only when target crop is blurred, and bypassed when sharp."""
        # 1. Target is sharp (B_target >= 100.0)
        np.random.seed(202)
        sharp_target = np.random.randint(40, 220, (128, 128, 3), dtype=np.uint8)
        sharp_swapped = sharp_target.copy()

        out_sharp, meta_sharp = motion_blur.harmonize_motion_blur(
            swapped_face=sharp_swapped,
            target_crop=sharp_target,
            flow_vector=(8.0, 0.0),
            blur_threshold=100.0
        )
        self.assertFalse(meta_sharp['is_motion_blurred'],
                         "Sharp target crop should bypass motion blur convolution")
        np.testing.assert_array_equal(out_sharp, sharp_swapped,
                                      "Swapped face must remain identical when target is sharp")

        # 2. Target is blurred (B_target < 100.0)
        blurred_target = cv2.GaussianBlur(sharp_target, (15, 15), 4.0)
        out_blurred, meta_blurred = motion_blur.harmonize_motion_blur(
            swapped_face=sharp_swapped,
            target_crop=blurred_target,
            flow_vector=(8.0, 0.0),
            blur_threshold=100.0
        )
        self.assertTrue(meta_blurred['is_motion_blurred'],
                        "Blurred target crop must trigger directional motion blur convolution")
        self.assertIsNotNone(meta_blurred['kernel'], "Kernel must be populated")

        # The convolved output must have lower sharpness than the input
        in_var = motion_blur.compute_blur_metric(sharp_swapped)
        out_var = motion_blur.compute_blur_metric(out_blurred)
        self.assertLess(out_var, in_var * 0.5,
                        "Convolved swapped patch must have significantly reduced Laplacian variance")

    def test_motion_blur_harmonizer_state_lifecycle(self):
        """Confirm MotionBlurHarmonizer tracks motion vectors across consecutive frames per track_id."""
        harmonizer = motion_blur.MotionBlurHarmonizer(blur_threshold=100.0)

        # Frame 0: First observation (no previous frame, target blurred)
        frame0_target = np.full((128, 128, 3), 120, dtype=np.uint8)
        swapped = np.full((128, 128, 3), 180, dtype=np.uint8)
        out0, meta0 = harmonizer.harmonize(swapped, frame0_target, track_id=1)
        self.assertFalse(meta0['is_motion_blurred'],
                         "First frame has no temporal history, should not blur without flow")

        # Frame 1: Target moves with high velocity and is blurred
        frame1_target = frame0_target.copy()
        frame1_target[40:80, 40:80] = 200
        frame1_target = cv2.GaussianBlur(frame1_target, (7, 7), 1.5)
        # Warmup frame 0 with feature
        frame0_target_feat = cv2.GaussianBlur(frame0_target.copy(), (7, 7), 1.5)
        harmonizer.reset(track_id=1)
        harmonizer.harmonize(swapped, frame0_target_feat, track_id=1)

        # Now frame 1 with optical flow vector directly provided
        out1, meta1 = harmonizer.harmonize(swapped, frame1_target, track_id=1, flow_vector=(7.0, 0.0))
        self.assertTrue(meta1['is_motion_blurred'],
                        "Consecutive frame with motion must trigger blur harmonization")
        self.assertEqual(meta1['flow_vector'], (7.0, 0.0))

        # Reset clears state
        harmonizer.reset(track_id=1)
        self.assertNotIn(1, harmonizer._states, "Track ID state must be cleared after reset")

    # ==========================================================================
    # 2. Eye Aspect Ratio (EAR) Blink Gating Tests
    # ==========================================================================

    def test_ear_mathematical_formula(self):
        """Confirm calculate_ear strictly evaluates EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)."""
        # Exact synthetic coordinates with known distances:
        # width = ||p1 - p4|| = 30.0
        # vertical 1 = ||p2 - p6|| = 10.0
        # vertical 2 = ||p3 - p5|| = 10.0
        # Expected EAR = (10.0 + 10.0) / (2.0 * 30.0) = 20.0 / 60.0 = 0.333333
        eye_open = np.array([
            [0.0, 0.0],    # p1
            [10.0, -5.0],  # p2
            [20.0, -5.0],  # p3
            [30.0, 0.0],   # p4
            [20.0, 5.0],   # p5
            [10.0, 5.0],   # p6
        ], dtype=np.float32)

        ear_open = eyelid_preserver.calculate_ear(eye_open)
        self.assertAlmostEqual(ear_open, 1.0 / 3.0, places=5,
                               msg=f"Open eye EAR must equal 1/3 (got {ear_open})")

        # Closed eye: vertical separation = 2.0
        # Expected EAR = (2.0 + 2.0) / (2.0 * 30.0) = 4.0 / 60.0 = 0.066667
        eye_closed = np.array([
            [0.0, 0.0],    # p1
            [10.0, -1.0],  # p2
            [20.0, -1.0],  # p3
            [30.0, 0.0],   # p4
            [20.0, 1.0],   # p5
            [10.0, 1.0],   # p6
        ], dtype=np.float32)

        ear_closed = eyelid_preserver.calculate_ear(eye_closed)
        self.assertAlmostEqual(ear_closed, 4.0 / 60.0, places=5,
                               msg=f"Closed eye EAR must equal 4/60 (got {ear_closed})")

    def test_ear_blink_gating_threshold_018(self):
        """Confirm blink gating triggers when EAR < 0.18 and is inactive when EAR >= 0.18."""
        self.assertTrue(eyelid_preserver.is_eye_blinking(0.179, threshold=0.18))
        self.assertFalse(eyelid_preserver.is_eye_blinking(0.180, threshold=0.18))
        self.assertFalse(eyelid_preserver.is_eye_blinking(0.250, threshold=0.18))

        # Test with full 68-point landmarks
        open_landmarks = create_mock_68_landmarks(left_eye_open=True, right_eye_open=True)
        l_ear_o, r_ear_o, _ = eyelid_preserver.compute_eye_aspect_ratios(open_landmarks)
        self.assertGreater(l_ear_o, 0.18, "Open eye EAR must be above 0.18")
        self.assertGreater(r_ear_o, 0.18, "Open eye EAR must be above 0.18")

        closed_landmarks = create_mock_68_landmarks(left_eye_open=False, right_eye_open=False)
        l_ear_c, r_ear_c, _ = eyelid_preserver.compute_eye_aspect_ratios(closed_landmarks)
        self.assertLess(l_ear_c, 0.18, "Closed eye EAR must be below 0.18")
        self.assertLess(r_ear_c, 0.18, "Closed eye EAR must be below 0.18")

    def test_eye_elliptical_mask_generation(self):
        """Confirm elliptical mask is generated centered around landmarks p1-p6 with soft feathering."""
        closed_landmarks = create_mock_68_landmarks(left_eye_open=False, right_eye_open=False)
        mask = eyelid_preserver.build_blink_eyelid_mask(
            closed_landmarks, self.shape, ear_threshold=0.18
        )

        # Mask should be active in left eye region (center ~ (80, 95))
        left_eye_center_val = float(mask[95, 80])
        self.assertGreater(left_eye_center_val, 0.85,
                           "Mask must have high opacity at left eye center")

        # Mask should be active in right eye region (center ~ (176, 95))
        right_eye_center_val = float(mask[95, 176])
        self.assertGreater(right_eye_center_val, 0.85,
                           "Mask must have high opacity at right eye center")

        # Mask must be completely 0 outside face eye regions (e.g. forehead or chin)
        self.assertEqual(float(mask[30, 128]), 0.0, "Forehead should have zero mask weight")
        self.assertEqual(float(mask[220, 128]), 0.0, "Chin should have zero mask weight")

        # Open eyes produce strictly zero mask
        open_landmarks = create_mock_68_landmarks(left_eye_open=True, right_eye_open=True)
        open_mask = eyelid_preserver.build_blink_eyelid_mask(
            open_landmarks, self.shape, ear_threshold=0.18
        )
        self.assertEqual(float(np.sum(open_mask)), 0.0, "Open eyes must produce empty mask")

    def test_eyelid_blend_opacity_95_percent(self):
        """Confirm original target eyelids are blended at exactly 95% opacity over the swapped face."""
        closed_landmarks = create_mock_68_landmarks(left_eye_open=False, right_eye_open=True)
        mask = eyelid_preserver.build_blink_eyelid_mask(
            closed_landmarks, self.shape, ear_threshold=0.18
        )

        # Target crop BGR: (120, 160, 210)
        # Swapped crop BGR: (40, 210, 40)
        # At 95% opacity:
        # B = 0.05 * 40 + 0.95 * 120 = 2.0 + 114.0 = 116.0
        # G = 0.05 * 210 + 0.95 * 160 = 10.5 + 152.0 = 162.5
        # R = 0.05 * 40 + 0.95 * 210 = 2.0 + 199.5 = 201.5
        blended = eyelid_preserver.blend_eyelid_preservation(
            target_crop=self.target_crop,
            swapped_crop=self.swapped_crop,
            eyelid_mask=mask,
            opacity=0.95
        )

        # Check left closed eye center (80, 95) where mask == 1.0
        left_eye_px = blended[95, 80].astype(np.float32)
        self.assertAlmostEqual(float(left_eye_px[0]), 116.0, delta=2.0,
                               msg=f"Blue channel should match 95% opacity blend (got {left_eye_px[0]})")
        self.assertAlmostEqual(float(left_eye_px[1]), 162.5, delta=2.0,
                               msg=f"Green channel should match 95% opacity blend (got {left_eye_px[1]})")
        self.assertAlmostEqual(float(left_eye_px[2]), 201.5, delta=2.0,
                               msg=f"Red channel should match 95% opacity blend (got {left_eye_px[2]})")

        # Check right open eye center (176, 95): mask == 0.0, so 100% swapped face
        right_eye_px = blended[95, 176]
        np.testing.assert_array_equal(right_eye_px, np.array([40, 210, 40], dtype=np.uint8),
                                      "Open eye must retain 100% swapped face")

        # Check corner (outside face): 100% swapped face
        corner_px = blended[10, 10]
        np.testing.assert_array_equal(corner_px, np.array([40, 210, 40], dtype=np.uint8),
                                      "Non-eye region must retain 100% swapped face")

    # ==========================================================================
    # 3. Integrated Face Swapper Gating & Dynamics Tests
    # ==========================================================================

    def test_face_swapper_facial_dynamics_integration(self):
        """Confirm apply_facial_dynamics in face_swapper uses EAR < 0.18 and blends eyelids."""
        pts = create_mock_68_landmarks(left_eye_open=False, right_eye_open=True, mouth_open=False)
        out, meta = face_swapper.apply_facial_dynamics(self.target_crop, self.swapped_crop, pts)

        self.assertTrue(meta['is_blinking'], "Blink must be flagged for closed left eye")
        self.assertLess(meta['left_ear'], 0.18, "Left ear must be below 0.18")
        self.assertGreater(meta['right_ear'], 0.18, "Right ear must be above 0.18")

        # Closed eye region should preserve target actor's eyelid color
        left_eye_out = out[95, 80].astype(np.float32)
        self.assertAlmostEqual(float(left_eye_out[0]), 116.0, delta=3.0)

        # Open eye region should retain swapped face color
        right_eye_out = out[95, 176]
        np.testing.assert_array_equal(right_eye_out, np.array([40, 210, 40], dtype=np.uint8))

    def test_unified_blur_and_blink_end_to_end(self):
        """Confirm both motion-blur harmonization and eyelid preservation execute coherently."""
        # Setup: Target crop has both motion blur and a closed eye
        np.random.seed(303)
        # Texture base
        texture = np.random.randint(60, 200, (128, 128, 3), dtype=np.uint8)
        # Low-pass filter to simulate motion blur on target crop
        blurred_target = cv2.GaussianBlur(texture, (11, 11), 3.0)

        # Swapped face is sharp
        sharp_swapped = texture.copy()

        # Closed left eye landmarks scaled to 128x128
        landmarks_128 = create_mock_68_landmarks(left_eye_open=False, right_eye_open=True) * 0.5

        # 1. Eyelid preservation
        eyelid_preserved, blink_meta = eyelid_preserver.preserve_eyelids(
            target_crop=blurred_target,
            swapped_crop=sharp_swapped,
            landmarks_68=landmarks_128,
            ear_threshold=0.18,
            opacity=0.95
        )
        self.assertTrue(blink_meta['is_blinking'])

        # 2. Motion blur harmonization prior to alpha compositing
        harmonized, blur_meta = motion_blur.harmonize_motion_blur(
            swapped_face=eyelid_preserved,
            target_crop=blurred_target,
            flow_vector=(6.0, 0.0),
            blur_threshold=100.0
        )
        self.assertTrue(blur_meta['is_motion_blurred'])

        # Final checks:
        # Output shape matches input
        self.assertEqual(harmonized.shape, (128, 128, 3))
        # Final output sharpness is reduced to match target scene blur
        self.assertLess(
            motion_blur.compute_blur_metric(harmonized),
            motion_blur.compute_blur_metric(sharp_swapped)
        )


if __name__ == '__main__':
    unittest.main()
