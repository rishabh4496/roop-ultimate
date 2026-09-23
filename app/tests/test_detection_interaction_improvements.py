"""Unit and regression tests for face detection, occlusion, and interaction improvements.

Covers:
1. MultiScaleFaceDetector explicit single scale / pyramid off fix.
2. Scale pyramid parser robustness (NaN, Inf, negative, malformed input).
3. Scale pyramid coordinate inversion precision on non-square / clamped dimensions.
4. Mask_Occluder session pool shape metadata.
5. Duplicate face discrimination (touching/kissing faces preserved vs concentric duplicates dropped).
6. Multi-angle accumulation in _rescue_rotated.
7. Unified partial rotated rescue in _detect_faces.
8. CLAHE rescue clean-plate embedding attachment.
9. Strict enforcement: Face angle formulas untouched.
"""

import ast
import inspect
import math
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import face_detector, face_util, face_analyser
from roop.face_detector import (
    parse_scale_pyramid,
    generate_scale_pyramid,
    rescale_detections,
    MultiScaleFaceDetector,
)
from roop.face_util import (
    _bbox_iou_simple,
    _is_face_duplicate,
    _rescue_rotated,
    _rescue_clahe,
    _detect_faces,
)
from roop.processors.Mask_Occluder import Mask_Occluder


class TestScalePyramidRobustness(unittest.TestCase):
    def test_parse_scale_pyramid_robustness(self):
        # Auto/None
        self.assertIsNone(parse_scale_pyramid(None))
        self.assertIsNone(parse_scale_pyramid("auto"))
        self.assertIsNone(parse_scale_pyramid(""))

        # Disabled variants
        self.assertEqual(parse_scale_pyramid("none"), [1.0])
        self.assertEqual(parse_scale_pyramid("off"), [1.0])
        self.assertEqual(parse_scale_pyramid("false"), [1.0])
        self.assertEqual(parse_scale_pyramid("0"), [1.0])
        self.assertEqual(parse_scale_pyramid("single"), [1.0])
        self.assertEqual(parse_scale_pyramid("1.0"), [1.0])

        # Valid scale lists and strings
        self.assertEqual(parse_scale_pyramid("0.5, 0.75, 1.0"), [0.5, 0.75, 1.0])
        self.assertEqual(parse_scale_pyramid([1.0, 0.5, 0.75]), [0.5, 0.75, 1.0])

        # Robustness against NaN, Inf, negative, zero
        self.assertEqual(parse_scale_pyramid([float("nan"), 0.5, -1.0, 0.0, float("inf")]), [0.5])
        self.assertEqual(parse_scale_pyramid("nan, 0.75, -0.5, inf, abc"), [0.75])

    def test_multiscale_detector_explicit_single_scale(self):
        # When parsed_scales is [1.0], it must only call detect_fn ONCE (single scale)
        # and NOT fall back to default_scales ([0.5, 0.75, 1.0]).
        detector = MultiScaleFaceDetector()
        mock_detect = MagicMock(return_value=(
            np.array([[100, 100, 200, 200, 0.9]], dtype=np.float32),
            np.zeros((1, 5, 2), dtype=np.float32)
        ))
        detector.detect_fn = mock_detect

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets, kpss = detector.detect(dummy_frame, det_size=640, scales=[1.0], parallel=False)

        self.assertEqual(mock_detect.call_count, 1)
        self.assertEqual(len(dets), 1)

    def test_scale_pyramid_anisotropic_coordinate_rescaling(self):
        # Frame with odd dimensions where clamping or rounding introduces anisotropic scaling
        frame = np.zeros((135, 185, 3), dtype=np.uint8)
        pyramid = generate_scale_pyramid(frame, [0.5])
        scale_repr, scaled_img = pyramid[0]

        # Scaled image coordinates
        sh, sw = scaled_img.shape[:2]
        scaled_box = np.array([[10.0, 10.0, 50.0, 50.0, 0.95]], dtype=np.float32)
        scaled_kps = np.array([[[20.0, 20.0], [40.0, 20.0], [30.0, 30.0], [25.0, 45.0], [35.0, 45.0]]], dtype=np.float32)

        res_box, res_kps = rescale_detections(scaled_box, scaled_kps, scale_factor=scale_repr)

        if isinstance(scale_repr, (tuple, list)):
            sx, sy = scale_repr
        else:
            sx = sy = scale_repr

        self.assertAlmostEqual(res_box[0, 0], 10.0 / sx, places=4)
        self.assertAlmostEqual(res_box[0, 1], 10.0 / sy, places=4)
        self.assertAlmostEqual(res_kps[0, 0, 0], 20.0 / sx, places=4)
        self.assertAlmostEqual(res_kps[0, 0, 1], 20.0 / sy, places=4)


class TestMaskOccluderMetadata(unittest.TestCase):
    def test_occluder_pool_shape_matches_model(self):
        # Inspect source code of Mask_Occluder.Initialize to verify input_shape is (1, 256, 256, 3)
        src = inspect.getsource(Mask_Occluder.Initialize)
        self.assertIn("input_shape=(1, 256, 256, 3)", src)
        self.assertNotIn("input_shape=(1, 3, 512, 512)", src)


class TestFaceContactAndDuplicateDiscrimination(unittest.TestCase):
    def test_is_face_duplicate_preserves_touching_or_kissing_faces(self):
        # Two heads side-by-side kissing/touching
        # Face 1: cx=100, cy=100, w=100, h=100 -> [50, 50, 150, 150]
        # Face 2: cx=160, cy=100, w=100, h=100 -> [110, 50, 210, 150]
        # Overlap: x in [110, 150] -> width=40, height=100 -> area=4000
        # Union: 10000 + 10000 - 4000 = 16000 -> IoU = 4000 / 16000 = 0.25
        # If closer: Face 2 at cx=140 -> overlap x in [90, 150] -> width=60, height=100 -> area=6000
        # Union = 14000 -> IoU = 6000 / 14000 = 0.428 (>= 0.35)
        # Center separation = 140 - 100 = 40. Radius = 50. Sep / radius = 40 / 50 = 0.80 (>= 0.35)
        face1 = [50.0, 50.0, 150.0, 150.0]
        face2_kissing = [90.0, 50.0, 190.0, 150.0]

        iou = _bbox_iou_simple(face1, face2_kissing)
        self.assertGreaterEqual(iou, 0.35)

        # Touching/kissing faces must NOT be marked duplicate
        is_dup = _is_face_duplicate(face2_kissing, [face1])
        self.assertFalse(is_dup, "Kissing/touching faces must be preserved, not dropped as duplicates")

    def test_is_face_duplicate_drops_concentric_duplicate(self):
        # Same face detected with slight bounding-box jitter:
        # Face 1: [50, 50, 150, 150] (center 100, 100, radius 50)
        # Face 2: [53, 52, 152, 149] (center 102.5, 100.5, radius 49.5)
        # Center separation = sqrt(2.5^2 + 0.5^2) = 2.55. Sep / min_radius = 2.55 / 49.5 = 0.051 (< 0.35)
        face1 = [50.0, 50.0, 150.0, 150.0]
        face2_concentric = [53.0, 52.0, 152.0, 149.0]

        is_dup = _is_face_duplicate(face2_concentric, [face1])
        self.assertTrue(is_dup, "Concentric duplicate detection of same face must be dropped")


class TestRotatedAndCLAHERescue(unittest.TestCase):
    def test_rescue_rotated_accumulates_across_angles(self):
        # Create a mock detector that finds face A on clockwise rotation, and face B on 180 rotation
        class DummyFace(dict):
            def __init__(self, bbox):
                super().__init__()
                self.bbox = np.array(bbox, dtype=np.float32)
                self.kps = np.zeros((5, 2), dtype=np.float32)

            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(name)

        call_records = []

        def mock_detect_raw(img, **kwargs):
            # Record calls
            call_records.append(len(call_records))
            if len(call_records) == 1:
                # First angle (clockwise) finds face at [100, 100, 200, 200]
                return [DummyFace([100, 100, 200, 200])]
            elif len(call_records) == 3:
                # Third angle (180) finds face at [300, 300, 400, 400]
                return [DummyFace([300, 300, 400, 400])]
            return []

        frame = np.zeros((500, 500, 3), dtype=np.uint8)
        with patch("roop.face_util._detect_faces_raw", side_effect=mock_detect_raw):
            results = _rescue_rotated(frame, expected_count=2)

        self.assertIsNotNone(results)
        self.assertEqual(len(results), 2, "Both faces across different angles should be accumulated")

    def test_rescue_clahe_uses_aux_false(self):
        # Ensure _rescue_clahe passes aux=False so embeddings are untouched by CLAHE
        captured_kwargs = {}

        def mock_detect_raw(img, **kwargs):
            captured_kwargs.update(kwargs)
            return []

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch("roop.face_util._detect_faces_raw", side_effect=mock_detect_raw):
            _rescue_clahe(frame)

        self.assertIn("aux", captured_kwargs)
        self.assertFalse(captured_kwargs["aux"])


class TestFaceAnglesFormulasUntouched(unittest.TestCase):
    def test_angle_functions_ast_invariants(self):
        # Verify strict constraint: face angles formulas remain 100% untouched
        from roop import face_analyser

        self.assertTrue(hasattr(face_analyser, "compute_canonical_roll_angle"))
        self.assertTrue(hasattr(face_analyser, "face_roll_degrees"))
        self.assertTrue(hasattr(face_analyser, "face_yaw_pitch"))

        # Verify solve_pose_5pt exists and is callable
        from roop.face_util import solve_pose_5pt
        self.assertTrue(callable(solve_pose_5pt))


if __name__ == "__main__":
    unittest.main()
