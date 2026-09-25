"""Tests for neural environment re-lighting and normal harmonization."""

import os
import sys
import unittest
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import roop.globals
from roop.light_harmonizer import (
    estimate_surface_normals,
    decompose_intrinsic_components,
    restore_eye_catchlights,
    cast_shadow_occlusion,
    apply_light_harmonization,
)


class TestLightHarmonizer(unittest.TestCase):
    def setUp(self):
        # Create synthetic face crops for testing
        self.h, self.w = 256, 256
        # Base skin tone in BGR
        self.target_crop = np.full((self.h, self.w, 3), [160, 180, 220], dtype=np.uint8)
        # Add key light gradient from top-right to bottom-left
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        grad = ((xx / self.w) * 0.5 + ((self.h - yy) / self.h) * 0.5)[:, :, None]
        self.target_crop = np.clip(self.target_crop.astype(np.float32) * (0.6 + 0.8 * grad), 0, 255).astype(np.uint8)

        # Swapped face crop (different identity, cooler tone, frontal illumination)
        self.swapped_crop = np.full((self.h, self.w, 3), [190, 180, 175], dtype=np.uint8)

        # Mock 5-point keypoints
        self.kps = np.array([
            [self.w * 0.36, self.h * 0.42],  # left eye
            [self.w * 0.64, self.h * 0.42],  # right eye
            [self.w * 0.50, self.h * 0.58],  # nose
            [self.w * 0.38, self.h * 0.74],  # left mouth
            [self.w * 0.62, self.h * 0.74],  # right mouth
        ], dtype=np.float32)

        # Mock 68-point 3D landmarks
        self.lm68 = np.zeros((68, 3), dtype=np.float32)
        self.lm68[36:42, :2] = [[self.w * 0.34, self.h * 0.42], [self.w * 0.36, self.h * 0.40],
                                 [self.w * 0.38, self.h * 0.40], [self.w * 0.40, self.h * 0.42],
                                 [self.w * 0.38, self.h * 0.44], [self.w * 0.36, self.h * 0.44]]
        self.lm68[42:48, :2] = [[self.w * 0.60, self.h * 0.42], [self.w * 0.62, self.h * 0.40],
                                 [self.w * 0.64, self.h * 0.40], [self.w * 0.66, self.h * 0.42],
                                 [self.w * 0.64, self.h * 0.44], [self.w * 0.62, self.h * 0.44]]
        self.lm68[27:31, :2] = [[self.w * 0.50, self.h * 0.46], [self.w * 0.50, self.h * 0.50],
                                 [self.w * 0.50, self.h * 0.54], [self.w * 0.50, self.h * 0.58]]
        self.lm68[48:68, :2] = [self.w * 0.50, self.h * 0.74]
        self.lm68[8, :2] = [self.w * 0.50, self.h * 0.88]

    def test_surface_normal_properties(self):
        """Test that surface normals are unit-normalized and geometrically consistent."""
        normals = estimate_surface_normals(self.target_crop, landmarks=self.lm68, kps=self.kps)
        self.assertEqual(normals.shape, (self.h, self.w, 3))
        self.assertEqual(normals.dtype, np.float32)

        # Norm must be 1.0 everywhere
        norms = np.linalg.norm(normals, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

        # Z-component must be positive (facing camera)
        self.assertTrue(np.all(normals[:, :, 2] > 0.0))

        # Left cheek normals tilt left (Nx < 0), right cheek normals tilt right (Nx > 0)
        left_cheek_nx = normals[int(self.h * 0.55), int(self.w * 0.25), 0]
        right_cheek_nx = normals[int(self.h * 0.55), int(self.w * 0.75), 0]
        self.assertLess(left_cheek_nx, 0.0)
        self.assertGreater(right_cheek_nx, 0.0)

    def test_intrinsic_decomposition(self):
        """Test decomposition into diffuse albedo, shading, and light parameters."""
        normals = estimate_surface_normals(self.target_crop, kps=self.kps)
        albedo, shading, light_params = decompose_intrinsic_components(self.target_crop, normals)

        self.assertEqual(albedo.shape, (self.h, self.w, 3))
        self.assertEqual(shading.shape, (self.h, self.w, 1))

        # Values should be bounded [0, 1]
        self.assertTrue(np.all(albedo >= 0.0) and np.all(albedo <= 1.0))
        self.assertTrue(np.all(shading >= 0.0) and np.all(shading <= 1.0))

        # Check light parameters
        self.assertIn('key_dir', light_params)
        self.assertIn('key_intensity', light_params)
        self.assertIn('key_color', light_params)
        self.assertIn('ambient_level', light_params)

        key_dir = light_params['key_dir']
        self.assertAlmostEqual(float(np.linalg.norm(key_dir)), 1.0, places=4)
        self.assertGreater(key_dir[2], 0.0)  # Front-facing light
        self.assertGreater(light_params['ambient_level'], 0.0)

    def test_eye_catchlight_restoration(self):
        """Test isolating and restoring corneal specular catchlights."""
        # Inject specular catchlight into target eye
        tgt_with_glint = self.target_crop.copy()
        eye_y, eye_x = int(self.kps[0][1]), int(self.kps[0][0])
        tgt_with_glint[eye_y:eye_y+3, eye_x:eye_x+3] = [255, 255, 255]

        restored = restore_eye_catchlights(
            self.swapped_crop, tgt_with_glint, landmarks=self.lm68, kps=self.kps, strength=1.0
        )
        self.assertEqual(restored.shape, self.swapped_crop.shape)
        # Eye region should have increased brightness where glint was transferred
        swp_eye_val = float(np.mean(self.swapped_crop[eye_y:eye_y+3, eye_x:eye_x+3]))
        res_eye_val = float(np.mean(restored[eye_y:eye_y+3, eye_x:eye_x+3]))
        self.assertGreater(res_eye_val, swp_eye_val)

        # Zero strength is no-op
        zero_res = restore_eye_catchlights(
            self.swapped_crop, tgt_with_glint, landmarks=self.lm68, kps=self.kps, strength=0.0
        )
        np.testing.assert_array_equal(zero_res, self.swapped_crop)

    def test_dynamic_shadow_casting(self):
        """Test casting directional soft shadows from occluding objects."""
        # Create an occluder mask (e.g. microphone silhouette near chin)
        occluder = np.zeros((self.h, self.w), dtype=np.float32)
        cv2.circle(occluder, (int(self.w * 0.45), int(self.h * 0.70)), 20, 1.0, -1)

        # Light coming from top-left (Lx > 0, Ly > 0, Lz > 0)
        light_dir = np.array([0.5, 0.5, 0.7], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)

        shadowed = cast_shadow_occlusion(
            self.swapped_crop, occluder, light_dir, shadow_strength=0.8
        )
        self.assertEqual(shadowed.shape, self.swapped_crop.shape)

        # Pixels where shadow was cast should be darker than original swapped crop
        self.assertLess(float(np.mean(shadowed)), float(np.mean(self.swapped_crop)))

        # Zero strength or empty mask is no-op
        empty_occ = np.zeros((self.h, self.w), dtype=np.float32)
        no_shadow = cast_shadow_occlusion(
            self.swapped_crop, empty_occ, light_dir, shadow_strength=0.8
        )
        np.testing.assert_array_equal(no_shadow, self.swapped_crop)

    def test_apply_light_harmonization_disabled_is_exact_noop(self):
        """When light_harmonizer is False, function returns the input object directly."""
        roop.globals.light_harmonizer = False
        res = apply_light_harmonization(self.swapped_crop, self.target_crop)
        self.assertIs(res, self.swapped_crop)

    def test_apply_light_harmonization_enabled(self):
        """When light_harmonizer is True, harmonizes swapped crop with scene lighting."""
        roop.globals.light_harmonizer = True
        roop.globals.light_harmonizer_key_intensity = 1.2
        roop.globals.light_harmonizer_ambient_bias = 0.1
        roop.globals.light_harmonizer_eye_specular = 0.8
        roop.globals.light_harmonizer_shadow_occlusion = 0.5

        target_face_mock = {
            'kps': self.kps,
            'landmark_3d_68': self.lm68,
        }

        harmonized = apply_light_harmonization(
            self.swapped_crop, self.target_crop, target_face=target_face_mock
        )
        self.assertEqual(harmonized.shape, self.swapped_crop.shape)
        self.assertEqual(harmonized.dtype, np.uint8)
        # Harmonized output should be modified from raw swapped crop
        self.assertFalse(np.array_equal(harmonized, self.swapped_crop))

        # Reset global
        roop.globals.light_harmonizer = False


if __name__ == '__main__':
    unittest.main()
