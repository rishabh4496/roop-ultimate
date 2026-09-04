"""Synthetic invariants for roll canonicalisation and profile alignment.

Run from ``app`` with:
    env/Scripts/python.exe -m unittest tests.test_pose_canonicalization
"""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from skimage.transform import SimilarityTransform

# Permit both ``python tests/test_*.py`` and unittest discovery from ``app``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.face_analyser import canonicalize_face_alignment, face_yaw_pitch
from roop.face_util import swap_template_points
from roop.utilities import rotation_affines, transform_points
from tests.facegeom import project_kps


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

    def test_profile_anchors_cannot_displace_the_model_template(self):
        # An isotropic matrix alone is not enough: the former test accepted a
        # perfectly square crop with eyes and mouth in the wrong positions.
        anchors = np.array([[66, 126], [122, 122], [124, 190]], dtype=np.float32)
        face = {
            'bbox': self.bbox,
            'kps': project_kps(70, scale=80, cx=120, cy=120),
            'profile_anchors': anchors,
            'roll_deg': 0.0,
            '_adaptive_yaw': 70.0,
            '_adaptive_pitch': 4.0,
        }
        crop, matrix, info = canonicalize_face_alignment(self.image, face, 256, 'arcface')

        self.assertEqual(crop.shape[:2], (256, 256))
        self.assertEqual(info['alignment_kind'], 'five_point')
        fit = SimilarityTransform()
        fit.estimate(face['kps'], swap_template_points(256, 'arcface'))
        np.testing.assert_allclose(matrix, fit.params[:2], atol=1e-5)
        self.assertLessEqual(_aspect_ratio(matrix), 1.05)

    def test_pose_matches_known_head_angles_even_with_dense_landmarks(self):
        for yaw in (-75, -40, 0, 40, 75):
            for pitch in (-30, 0, 30):
                for roll in (-60, 0, 60):
                    with self.subTest(yaw=yaw, pitch=pitch, roll=roll):
                        face = {'kps': project_kps(yaw, pitch, roll),
                                'landmark_3d_68': np.zeros((68, 3)),
                                'landmark_2d_106': np.zeros((106, 2))}
                        np.testing.assert_allclose(face_yaw_pitch(face),
                                                   (yaw, pitch), atol=0.05)

    def test_all_angles_use_model_template_and_one_pixel_resample(self):
        image = np.random.default_rng(3).integers(0, 256, (240, 240, 3), dtype=np.uint8)
        for yaw in (-75, -45, 0, 45, 75):
            for pitch in (-40, 0, 40):
                for roll in (-120, 0, 60, 179):
                    for mode in ('arcface', 'arcface_112_v1', 'mtcnn_512'):
                        with self.subTest(yaw=yaw, pitch=pitch, roll=roll, mode=mode):
                            kps = project_kps(yaw, pitch, roll, scale=80, cx=120, cy=120)
                            face = {'bbox': self.bbox, 'kps': kps}
                            fit = SimilarityTransform()
                            fit.estimate(kps, swap_template_points(128, mode))
                            expected_matrix = fit.params[:2]
                            expected = cv2.warpAffine(image, expected_matrix, (128, 128),
                                                      borderMode=cv2.BORDER_REPLICATE)
                            dst = np.empty_like(expected)
                            crop, matrix, info = canonicalize_face_alignment(
                                image, face, 128, mode, dst=dst)
                            self.assertIs(crop, dst)
                            np.testing.assert_allclose(matrix, expected_matrix, atol=1e-5)
                            np.testing.assert_array_equal(crop, expected)
                            self.assertAlmostEqual(_aspect_ratio(matrix), 1.0, places=5)
                            np.testing.assert_allclose(
                                transform_points(transform_points(kps, matrix), info['inv_paste_matrix']),
                                kps, atol=1e-4)

    def test_reused_track_id_and_crop_size_do_not_contaminate_preview(self):
        face = {'bbox': self.bbox, 'kps': self.upright_kps,
                '_track_id': 0, 'frame_idx': 3}
        _, expected, _ = canonicalize_face_alignment(self.image, face, 128, 'arcface')
        other = dict(face, kps=self.upright_kps + [8, -4], frame_idx=4)
        canonicalize_face_alignment(self.image, other, 256, 'mtcnn_512')
        _, actual, _ = canonicalize_face_alignment(self.image, face, 128, 'arcface')
        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
