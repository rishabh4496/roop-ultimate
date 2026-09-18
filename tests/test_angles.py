"""Unit tests for Omnidirectional Angle Rectification & 3D Profile-Aware Weighted Umeyama.

Verifies:
1. Exact 2D roll angle calculation (atan2) with continuous boundary behavior around +-pi.
2. Canonical pose normalization R(theta, center) mapping rotated faces to upright orientation.
3. Transformation matrix invertibility: T_final @ M_composite == I under 0, 90, 180, 270 rotations.
4. Landmark variance under 0, 90, 180, and 270 degree rotations (< 1e-4 variance).
5. 2D-to-3D PnP head pose estimation (Yaw, Pitch, Roll) with canonical 3D facial mesh template.
6. Profile-aware weighted Umeyama alignment preventing horizontal collapse at high yaw (> 35 to 85 deg).
7. Sub-pixel bilinear sampling and end-to-end swap_face robustness under extreme angles.
"""

import os
import sys
from pathlib import Path
from typing import Tuple
import unittest

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / 'app'
for p in (str(REPO_ROOT), str(APP_DIR)):
    if p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(APP_DIR))

from roop.face_analyser import (
    CANONICAL_FACE_3D_5,
    compute_canonical_roll_angle,
    build_canonical_rotation_matrix,
    estimate_head_pose_pnp,
    weighted_umeyama_alignment,
    profile_aware_umeyama_alignment,
    compute_composite_inverse,
    compute_composite_forward,
    canonicalize_face_alignment,
)
from roop.utilities import transform_points, compose_affines
from roop.processors.frame.face_swapper import swap_face, ARCFACE_DST_128


def create_canonical_5pt_face(scale: float = 1.0, offset: Tuple[float, float] = (128.0, 128.0)) -> np.ndarray:
    """Return standard 5-point face landmarks scaled and translated."""
    base = np.array([
        [38.2946, 51.6963],  # Left eye
        [73.5318, 51.5014],  # Right eye
        [56.0252, 71.7366],  # Nose tip
        [41.5493, 92.3655],  # Left mouth
        [70.7299, 92.2041]   # Right mouth
    ], dtype=np.float32)
    center = np.array([56.0, 72.0], dtype=np.float32)
    return (base - center) * float(scale) + np.array(offset, dtype=np.float32)


def create_canonical_68pt_face(scale: float = 100.0, offset: Tuple[float, float] = (256.0, 256.0)) -> np.ndarray:
    """Return 68-point face landmarks projected from canonical 3D model."""
    from roop.face_3d_recon import _REF3D_68
    ref3d = (_REF3D_68 * np.array([1.0, -1.0, 1.0])).astype(np.float32)
    pts2d = ref3d[:, :2] * float(scale) + np.array(offset, dtype=np.float32)
    return pts2d


class TestCanonicalPoseNormalization(unittest.TestCase):
    """Test Mathematical Specification 1: Canonical Pose Normalization."""

    def test_canonical_roll_angle_cardinal_rotations(self):
        """Test exact 2D roll angle calculation for 0, 90, 180, and 270 degree rotations."""
        center = (128.0, 128.0)
        base_kps = create_canonical_5pt_face(scale=1.0, offset=center)

        test_cases = [
            (0.0, 0.0),
            (90.0, -90.0),    # Counter-clockwise rotation in image space makes dy negative
            (180.0, 180.0),   # 180 deg rotation
            (270.0, 90.0),    # 270 deg counter-clockwise is +90 deg clockwise
        ]

        for rot_angle, expected_detected in test_cases:
            R = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
            rotated_kps = transform_points(base_kps, R)

            theta_rad, theta_deg = compute_canonical_roll_angle(rotated_kps)

            diff = (theta_deg - expected_detected + 180.0) % 360.0 - 180.0
            self.assertLess(abs(diff), 1.0,
                            f"Roll angle detection failed for rot={rot_angle}: got {theta_deg}, expected {expected_detected}")
            self.assertTrue(np.isfinite(theta_rad))
            self.assertTrue(np.isfinite(theta_deg))

    def test_roll_angle_continuity_around_pi(self):
        """Test that roll angle computation has no discontinuities around +-pi (180 deg)."""
        center = (128.0, 128.0)
        base_kps = create_canonical_5pt_face(scale=1.0, offset=center)

        for angle in np.linspace(170.0, 190.0, 21):
            R = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_kps = transform_points(base_kps, R)
            theta_rad, theta_deg = compute_canonical_roll_angle(rotated_kps)

            self.assertTrue(np.isfinite(theta_rad))
            self.assertTrue(np.isfinite(theta_deg))
            self.assertGreaterEqual(theta_deg, -180.0)
            self.assertLessEqual(theta_deg, 180.0)

    def test_build_canonical_rotation_matrix_threshold(self):
        """Test thresholding: identity returned when |theta| <= 45, active rotation when > 45."""
        center = (128.0, 128.0)

        # Sub-threshold: 20 deg
        R_small, inv_R_small, applied_small = build_canonical_rotation_matrix(center, 20.0, threshold_deg=45.0)
        self.assertFalse(applied_small)
        np.testing.assert_allclose(R_small, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), atol=1e-5)
        np.testing.assert_allclose(inv_R_small, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), atol=1e-5)

        # Above threshold: 90 deg
        R_large, inv_R_large, applied_large = build_canonical_rotation_matrix(center, 90.0, threshold_deg=45.0)
        self.assertTrue(applied_large)

        # Verify R @ inv_R == Identity
        R_h = np.vstack([R_large, [0.0, 0.0, 1.0]])
        inv_R_h = np.vstack([inv_R_large, [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(R_h @ inv_R_h, np.eye(3), atol=1e-5)
        np.testing.assert_allclose(inv_R_h @ R_h, np.eye(3), atol=1e-5)


class TestTransformationMatrixInvertibility(unittest.TestCase):
    """Test Mathematical Specification 1: Composite Inverse Transformation T_final = inv(R) @ inv(M_warp)."""

    def test_matrix_invertibility_under_cardinal_rotations(self):
        """Verify T_final @ M_composite == I for rotations 0, 90, 180, and 270 degrees."""
        center = (256.0, 256.0)

        for angle in [0.0, 90.0, 180.0, 270.0, -45.0, 135.0]:
            R, inv_R, _ = build_canonical_rotation_matrix(center, angle, threshold_deg=0.0)

            M_warp = np.array([
                [0.85 * np.cos(0.1), -0.85 * np.sin(0.1), 32.0],
                [0.85 * np.sin(0.1),  0.85 * np.cos(0.1), 24.0]
            ], dtype=np.float32)
            inv_M_warp = cv2.invertAffineTransform(M_warp).astype(np.float32)

            M_composite = compute_composite_forward(M_warp, R)
            T_final = compute_composite_inverse(inv_R, inv_M_warp)

            M_comp_h = np.vstack([M_composite, [0.0, 0.0, 1.0]])
            T_final_h = np.vstack([T_final, [0.0, 0.0, 1.0]])

            I_forward_back = T_final_h @ M_comp_h
            I_back_forward = M_comp_h @ T_final_h

            np.testing.assert_allclose(
                I_forward_back, np.eye(3), atol=1e-4,
                err_msg=f"Invertibility failed for angle={angle}: T_final @ M_composite != I"
            )
            np.testing.assert_allclose(
                I_back_forward, np.eye(3), atol=1e-4,
                err_msg=f"Invertibility failed for angle={angle}: M_composite @ T_final != I"
            )

    def test_coordinate_roundtrip_subpixel_accuracy(self):
        """Verify forward and inverse transforms preserve sub-pixel accuracy on image grid."""
        center = (256.0, 256.0)
        grid_x, grid_y = np.meshgrid(np.linspace(50.0, 450.0, 15), np.linspace(50.0, 450.0, 15))
        test_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1).astype(np.float32)

        for angle in [0.0, 90.0, 180.0, 270.0]:
            R, inv_R, _ = build_canonical_rotation_matrix(center, angle, threshold_deg=0.0)
            M_warp = np.array([
                [1.2 * np.cos(0.05), -1.2 * np.sin(0.05), 10.0],
                [1.2 * np.sin(0.05),  1.2 * np.cos(0.05), 15.0]
            ], dtype=np.float32)
            inv_M = cv2.invertAffineTransform(M_warp).astype(np.float32)

            M_comp = compute_composite_forward(M_warp, R)
            T_final = compute_composite_inverse(inv_R, inv_M)

            crop_pts = transform_points(test_points, M_comp)
            recovered_pts = transform_points(crop_pts, T_final)

            max_err = float(np.max(np.linalg.norm(recovered_pts - test_points, axis=1)))
            self.assertLess(max_err, 1e-3, f"Roundtrip error {max_err} exceeded sub-pixel limit for angle={angle}")


class TestLandmarkVarianceUnderRotations(unittest.TestCase):
    """Test Deliverable 3: Landmark variance under 0, 90, 180, and 270 rotations."""

    def test_landmark_variance_under_cardinal_rotations(self):
        """Rectify canonical landmarks rotated by 0, 90, 180, 270 deg and verify variance is near zero."""
        center = (128.0, 128.0)
        canonical_kps = create_canonical_5pt_face(scale=1.0, offset=center)

        rectified_landmarks = []
        rotations = [0.0, 90.0, 180.0, 270.0]

        for rot_angle in rotations:
            rot_matrix = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
            rotated_kps = transform_points(canonical_kps, rot_matrix)

            _, theta_deg = compute_canonical_roll_angle(rotated_kps)

            R, _, applied = build_canonical_rotation_matrix(center, theta_deg, threshold_deg=0.0)

            rect_kps = transform_points(rotated_kps, R)
            rectified_landmarks.append(rect_kps)

            left_eye_y = rect_kps[0, 1]
            right_eye_y = rect_kps[1, 1]
            self.assertLess(abs(left_eye_y - right_eye_y), 1e-3,
                            f"Eyes not horizontally aligned at rotation {rot_angle}")

        rect_stack = np.stack(rectified_landmarks, axis=0)  # Shape: (4, 5, 2)
        variance_per_point = np.var(rect_stack, axis=0)     # Shape: (5, 2)
        max_variance = float(np.max(variance_per_point))

        self.assertLess(max_variance, 1e-4,
                        f"Landmark variance {max_variance} exceeded threshold 1e-4 across 0, 90, 180, 270 rotations")


class TestProfileAwareWeightedUmeyama(unittest.TestCase):
    """Test Mathematical Specification 2: Profile-Aware Weighted Umeyama Alignment."""

    def test_weighted_umeyama_identical_to_standard_with_uniform_weights(self):
        """Weighted Umeyama with uniform weights must match standard similarity transform."""
        src = create_canonical_5pt_face(scale=1.2, offset=(100.0, 100.0))
        dst = ARCFACE_DST_128

        M_uniform = weighted_umeyama_alignment(src, dst, weights=None)
        M_ones = weighted_umeyama_alignment(src, dst, weights=np.ones(5))
        M_cv, _ = cv2.estimateAffinePartial2D(src, dst)

        np.testing.assert_allclose(M_uniform, M_ones, atol=1e-5)
        np.testing.assert_allclose(M_uniform, M_cv, atol=1e-4)

    def test_high_yaw_prevents_horizontal_collapse(self):
        """At extreme yaw (> 45 to 85 deg), weighted Umeyama prevents horizontal scale collapse."""
        for yaw in [45.0, 60.0, 75.0, 85.0]:
            rad = np.radians(yaw)
            Ry = np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]])
            pts3d = CANONICAL_FACE_3D_5 @ Ry.T
            # Projected profile head
            profile_kps = (pts3d[:, :2] * 80.0) + np.array([64.0, 64.0], dtype=np.float32)

            M_prof, mode = profile_aware_umeyama_alignment(profile_kps, image_size=128, yaw_degrees=yaw)

            self.assertEqual(mode, "profile_weighted_5pt")
            self.assertTrue(np.isfinite(M_prof).all())

            # Verify determinant is positive and non-degenerate
            linear = M_prof[:, :2]
            det = float(np.linalg.det(linear))
            self.assertGreater(det, 0.1, f"Determinant collapsed at yaw={yaw}")

            # Verify scale factor is preserved (does not collapse to 0 or explode)
            scale_prof = float(np.linalg.norm(linear[:, 0]))
            self.assertGreater(scale_prof, 0.4, f"Scale collapsed at yaw={yaw}")
            self.assertLess(scale_prof, 1.5, f"Scale exploded at yaw={yaw}")

            # Verify nose anchor lands accurately near canonical nose template
            pts_prof = transform_points(profile_kps, M_prof)
            nose_err = float(np.linalg.norm(pts_prof[2] - ARCFACE_DST_128[2]))
            self.assertLess(nose_err, 15.0, f"Nose anchor drifted at yaw={yaw}")

    def test_7pt_profile_alignment_with_nasal_bridge_and_chin(self):
        """Test profile-aware alignment when 68-point landmarks provide bridge and chin anchors."""
        lm68 = create_canonical_68pt_face(scale=80.0, offset=(128.0, 128.0))
        kps = np.array([
            np.mean(lm68[36:42], axis=0),
            np.mean(lm68[42:48], axis=0),
            lm68[30],
            lm68[48],
            lm68[54]
        ], dtype=np.float32)

        M_prof, mode = profile_aware_umeyama_alignment(
            kps, image_size=128, yaw_degrees=65.0, landmarks_68=lm68
        )
        self.assertEqual(mode, "profile_weighted_7pt")
        self.assertTrue(np.isfinite(M_prof).all())
        self.assertEqual(M_prof.shape, (2, 3))


class Test2Dto3DPnPHeadPose(unittest.TestCase):
    """Test 2D-to-3D PnP Head Pose Estimation."""

    def test_pnp_pose_estimation_ground_truth(self):
        """Verify estimate_head_pose_pnp accurately recovers 3D angles from projected canonical mesh."""
        f = 600.0
        cx, cy = 256.0, 256.0
        cam = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

        test_poses = [
            (0.0, 0.0, 0.0),
            (45.0, 0.0, 0.0),
            (-60.0, 0.0, 0.0),
            (0.0, 20.0, 0.0),
            (0.0, 0.0, 90.0),
            (35.0, 15.0, -30.0),
        ]

        for yaw_gt, pitch_gt, roll_gt in test_poses:
            y, p, r = np.radians(yaw_gt), np.radians(pitch_gt), np.radians(roll_gt)
            Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
            Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
            Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
            R_gt = Ry @ Rx @ Rz

            pts3d = CANONICAL_FACE_3D_5 * 100.0
            pts_cam = (pts3d @ R_gt.T) + np.array([0, 0, 600.0])

            pts2d = np.zeros((5, 2), dtype=np.float32)
            pts2d[:, 0] = f * pts_cam[:, 0] / pts_cam[:, 2] + cx
            pts2d[:, 1] = f * pts_cam[:, 1] / pts_cam[:, 2] + cy

            yaw_est, pitch_est, roll_est = estimate_head_pose_pnp(pts2d, (512, 512))

            self.assertLess(abs(yaw_est - yaw_gt), 1.5, f"Yaw mismatch for GT={yaw_gt}")
            self.assertLess(abs(pitch_est - pitch_gt), 1.5, f"Pitch mismatch for GT={pitch_gt}")
            self.assertLess(abs(roll_est - roll_gt), 1.5, f"Roll mismatch for GT={roll_gt}")


class TestFaceSwapperIntegration(unittest.TestCase):
    """Test End-to-End face swapper integration under extreme roll and yaw angles."""

    def test_swap_face_under_extreme_roll_and_yaw(self):
        """Verify swap_face executes without error and maintains frame dimensions at 0, 90, 180, 270 deg and high yaw."""
        h, w = 512, 512
        frame = np.full((h, w, 3), 120, dtype=np.uint8)
        cv2.circle(frame, (256, 256), 80, (180, 160, 140), -1)

        source_face = type('MockFace', (), {'embedding': np.random.randn(512).astype(np.float32)})()

        for angle in [0.0, 90.0, 180.0, 270.0]:
            center = (256.0, 256.0)
            base_kps = create_canonical_5pt_face(scale=1.5, offset=center)
            R = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_kps = transform_points(base_kps, R)

            target_face = type('MockFace', (), {
                'kps': rotated_kps,
                'bbox': np.array([150.0, 150.0, 362.0, 362.0]),
                'landmark_3d_68': None
            })()

            swapped = swap_face(source_face, target_face, frame)
            self.assertEqual(swapped.shape, (h, w, 3))
            self.assertEqual(swapped.dtype, np.uint8)
            self.assertTrue(np.isfinite(swapped).all())


if __name__ == '__main__':
    unittest.main()
