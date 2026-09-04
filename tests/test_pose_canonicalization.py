"""Synthetic invariants for roll canonicalisation and profile alignment.

Run from ``app`` with:
    env/Scripts/python.exe -m unittest tests.test_pose_canonicalization
"""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

# Permit both ``python tests/test_*.py`` and unittest discovery from ``app``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.face_analyser import canonicalize_face_alignment
from roop.utilities import rotation_affines, transform_points


def _aspect_ratio(matrix):
    """The ratio of affine singular values; one means no anisotropic squeeze."""
    singular_values = np.linalg.svd(np.asarray(matrix, dtype=np.float32)[:, :2],
                                    compute_uv=False)
    return float(singular_values.max() / singular_values.min())


class PoseCanonicalizationTest(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((240, 240, 3), dtype=np.uint8)
        self.bbox = np.array([55, 35, 185, 210], dtype=np.float32)
        self.upright_kps = np.array([
            [88,  95], [150,  95], [119, 125], [96, 164], [143, 164],
        ], dtype=np.float32)

    def test_roll_60_is_uprighted_and_inverse_is_exact(self):
        # Synthesize the detector's geometry after a +60 degree target roll.
        rolled, _ = rotation_affines(self.bbox, 60.0)
        rolled_kps = transform_points(self.upright_kps, rolled)
        face = {
            'bbox': self.bbox,
            'kps': rolled_kps,
            'roll_deg': 60.0,
            '_adaptive_yaw': 0.0,
            '_adaptive_pitch': 0.0,
        }
        _crop, paste_matrix, info = canonicalize_face_alignment(
            self.image, face, 128, 'arcface')

        self.assertTrue(info['applied_roll_prerotation'])
        self.assertEqual(info['alignment_kind'], 'five_point')
        self.assertTrue(np.allclose(
            transform_points(transform_points(self.upright_kps, rolled),
                             info['pre_rotation']),
            self.upright_kps, atol=1e-4))
        # ``paste_matrix`` contains the pre-rotation. Its inverse is the exact
        # canonical-to-target path used immediately before alpha compositing.
        back = cv2.invertAffineTransform(paste_matrix)
        canonical = transform_points(self.upright_kps, paste_matrix)
        self.assertTrue(np.allclose(transform_points(canonical, back),
                                    self.upright_kps, atol=1e-4))
        self.assertLessEqual(_aspect_ratio(paste_matrix), 1.05)

    def test_yaw_70_uses_profile_anchors_without_aspect_squeeze(self):
        # Ear/tragus, nose tip, chin centre. The shallow ear-to-nose baseline
        # mimics a 70 degree profile where five-point eye alignment collapses.
        anchors = np.array([[66, 126], [122, 122], [124, 190]], dtype=np.float32)
        face = {
            'bbox': self.bbox,
            'kps': self.upright_kps,
            'profile_anchors': anchors,
            'roll_deg': 0.0,
            '_adaptive_yaw': 70.0,
            '_adaptive_pitch': 4.0,
        }
        crop, matrix, info = canonicalize_face_alignment(self.image, face, 256, 'arcface')

        self.assertEqual(crop.shape[:2], (256, 256))
        self.assertEqual(info['alignment_kind'], 'profile_3pt')
        self.assertLessEqual(_aspect_ratio(matrix), 1.05)


if __name__ == '__main__':
    unittest.main()
