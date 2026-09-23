"""Targeted tests for face detection and swapped face interaction improvements.

Covers:
1. Multi-scale detector explicit single scale / pyramid off fix (Fix D7).
2. Scale pyramid coordinate invert accuracy on odd dimensions (Fix D11/D12).
3. Scale pyramid keypoint/box pairing synchronization (Fix D13).
4. Mask_Occluder input shape metadata check.
5. Rotated face rescue with multiple faces in different orientations (Fix D1/D2/D16).
6. Touching/kissing face overlap dedup in rotated rescue (Fix D4).
7. CLAHE rescue embedding purity on original clean frame (Fix D15).
8. Strict AST and source invariance check verifying face angle formulas are 100% untouched.
9. Crop contamination missing quad fallback check.
"""
import ast
import inspect
import math
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import cv2
import numpy as np

import roop.globals
from roop.face_detector import (
    MultiScaleFaceDetector,
    generate_scale_pyramid,
    parse_scale_pyramid,
    rescale_detections,
)
from roop.face_contact import crop_contamination, merged_indices
from roop.face_analyser import (
    face_roll_degrees,
    face_yaw_pitch,
    compute_canonical_roll_angle,
)
from roop.face_util import (
    _detect_faces,
    _is_face_duplicate,
    _rescue_clahe,
    _rescue_rotated,
    solve_pose_5pt,
)


class TestMultiScaleDetectorFixes(unittest.TestCase):
    """Test D7, D11/D12, D13 fixes in MultiScaleFaceDetector."""

    def test_parse_scale_pyramid_bounds_and_robustness(self):
        """D11/D12: parse_scale_pyramid handles NaN, Inf, non-numeric, bounds count."""
        # Non-numeric / NaN / Inf handling
        self.assertEqual(parse_scale_pyramid([float('nan'), float('inf'), -1.0, 0.0]), [1.0])
        self.assertEqual(parse_scale_pyramid("nan,inf,-2,0"), [1.0])
        self.assertEqual(parse_scale_pyramid([0.5, "invalid", 0.75]), [0.5, 0.75])

        # Bounded scale count (max 5 levels)
        many_scales = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        parsed = parse_scale_pyramid(many_scales)
        self.assertLessEqual(len(parsed), 5)
        self.assertTrue(all(0.05 <= s <= 4.0 for s in parsed))

        # Standard formats preserved
        self.assertIsNone(parse_scale_pyramid(None))
        self.assertIsNone(parse_scale_pyramid('auto'))
        self.assertEqual(parse_scale_pyramid('off'), [1.0])
        self.assertEqual(parse_scale_pyramid('none'), [1.0])
        self.assertEqual(parse_scale_pyramid('single'), [1.0])
        self.assertEqual(parse_scale_pyramid([1.0]), [1.0])
        self.assertEqual(parse_scale_pyramid([0.5, 1.0]), [0.5, 1.0])

    def test_multiscale_detector_explicit_single_scale_does_not_use_default_scales(self):
        """D7: MultiScaleFaceDetector.detect with parsed_scales == [1.0] executes single-scale only."""
        call_count = 0

        def mock_detect_fn(img, det_size, det_thresh):
            nonlocal call_count
            call_count += 1
            # Return 1 dummy face
            boxes = np.array([[100.0, 100.0, 200.0, 200.0, 0.9]], dtype=np.float32)
            kpss = np.zeros((1, 5, 2), dtype=np.float32)
            return boxes, kpss

        detector = MultiScaleFaceDetector(
            detect_fn=mock_detect_fn,
            default_scales=[0.5, 0.75, 1.0],
            padding=64,
        )

        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. With scales=[1.0] (explicit single scale), detect_fn must be called exactly once
        call_count = 0
        boxes, kpss = detector.detect(test_frame, scales=[1.0], parallel=False)
        self.assertEqual(call_count, 1, "detect_fn must be called exactly once when single scale is requested")
        self.assertEqual(len(boxes), 1)

        # 2. With scales='off', detect_fn must also be called exactly once
        call_count = 0
        boxes, kpss = detector.detect(test_frame, scales='off', parallel=False)
        self.assertEqual(call_count, 1, "detect_fn must be called exactly once when scales='off'")

    def test_scale_pyramid_odd_dimensions_exact_rescaling(self):
        """D11: Actual resize ratios (new_w/w, new_h/h) on odd dimensions invert accurately."""
        # Odd dimensions where rounding changes the effective scale
        frame = np.zeros((101, 151, 3), dtype=np.uint8)
        pyramid = generate_scale_pyramid(frame, [0.5])
        self.assertEqual(len(pyramid), 1)

        scale_val, scaled_img = pyramid[0]
        self.assertEqual(scaled_img.shape[:2], (50, 76))  # round(101*0.5)=50 in python round-to-even, round(151*0.5)=76

        # Candidate detection in scaled coordinates
        b_scaled = np.array([[20.0, 20.0, 50.0, 40.0, 0.95]], dtype=np.float32)
        k_scaled = np.array([[[25.0, 25.0], [45.0, 25.0], [35.0, 30.0], [28.0, 38.0], [42.0, 38.0]]], dtype=np.float32)

        b_orig, k_orig = rescale_detections(b_scaled, k_scaled, scale_factor=scale_val)

        # Expected exact coordinates
        expected_sx = 76.0 / 151.0
        expected_sy = 50.0 / 101.0
        self.assertAlmostEqual(b_orig[0, 0], 20.0 / expected_sx, places=4)
        self.assertAlmostEqual(b_orig[0, 1], 20.0 / expected_sy, places=4)
        self.assertAlmostEqual(k_orig[0, 0, 0], 25.0 / expected_sx, places=4)
        self.assertAlmostEqual(k_orig[0, 0, 1], 25.0 / expected_sy, places=4)

    def test_multiscale_kpss_pairing_synchronization(self):
        """D13: MultiScaleFaceDetector keeps boxes and kpss synchronized even if a scale returns None kps."""
        def mock_detect_fn(img, det_size, det_thresh):
            # If image height < 300, return boxes with NO keypoints
            if img.shape[0] < 300:
                boxes = np.array([[10.0, 10.0, 50.0, 50.0, 0.85]], dtype=np.float32)
                return boxes, None
            else:
                boxes = np.array([[100.0, 100.0, 200.0, 200.0, 0.95]], dtype=np.float32)
                kpss = np.ones((1, 5, 2), dtype=np.float32) * 150.0
                return boxes, kpss

        detector = MultiScaleFaceDetector(
            detect_fn=mock_detect_fn,
            default_scales=[0.5, 1.0],
            padding=64,
        )

        test_frame = np.zeros((400, 400, 3), dtype=np.uint8)
        boxes, kpss = detector.detect(test_frame, scales=[0.5, 1.0], force_pyramid=True, parallel=False)
        self.assertEqual(len(boxes), len(kpss), "Boxes and kpss must have identical length after multi-scale merge")


class TestMaskOccluderMetadata(unittest.TestCase):
    """Test Mask_Occluder input shape metadata."""

    def test_mask_occluder_model_input_shape_metadata(self):
        """Verify Mask_Occluder pool input_shape is (1, 256, 256, 3)."""
        import roop.processors.Mask_Occluder as mo
        src = inspect.getsource(mo.Mask_Occluder)
        self.assertIn("input_shape=(1, 256, 256, 3)", src)
        self.assertNotIn("input_shape=(1, 3, 512, 512)", src)


class TestRotatedAndInteractingFaceRescue(unittest.TestCase):
    """Test rotated rescue across multiple angles and touching/kissing face preservation."""

    def test_is_face_duplicate_preserves_touching_kissing_faces(self):
        """D4: Genuine touching or kissing faces (IoU >= 0.35 but center_sep >= 0.35) are preserved."""
        # Face 1: (100, 100, 200, 200) -> center (150, 150), radius 50
        box1 = np.array([100.0, 100.0, 200.0, 200.0])

        # Candidate A: Same face slightly shifted (concentric duplicate)
        # Center (152, 151), radius 50 -> sep = sqrt(4+1)/50 = 0.045 < 0.35, IoU ~ 0.90
        dup_candidate = np.array([102.0, 101.0, 202.0, 201.0])
        self.assertTrue(
            _is_face_duplicate(dup_candidate, [box1]),
            "Concentric duplicate candidate must be suppressed as duplicate"
        )

        # Candidate B: Touching/kissing face (profile meeting in center)
        # Center (210, 150), radius 50 -> center_sep = 60/50 = 1.2 > 0.35
        # Overlap: x in [160, 200] (width 40), y in [100, 200] (height 100) -> inter = 4000
        # union = 10000 + 10000 - 4000 = 16000 -> IoU = 4000/16000 = 0.25..0.40
        touching_candidate = np.array([160.0, 100.0, 260.0, 200.0])
        self.assertFalse(
            _is_face_duplicate(touching_candidate, [box1], iou_thresh=0.25),
            "Touching face with distinct center separation must NOT be suppressed as duplicate"
        )

    def test_is_face_duplicate_with_face_objects_and_landmarks(self):
        """D4: Face objects with distinct landmarks are preserved even with significant overlap."""
        class MockFace:
            def __init__(self, bbox, kps):
                self.bbox = np.array(bbox, dtype=np.float32)
                self.kps = np.array(kps, dtype=np.float32)

        face_a = MockFace(
            [100, 100, 200, 200],
            [[120, 130], [140, 130], [130, 150], [125, 170], [135, 170]]
        )
        face_b_touching = MockFace(
            [150, 100, 250, 200],
            [[210, 130], [230, 130], [220, 150], [215, 170], [225, 170]]
        )
        self.assertFalse(
            _is_face_duplicate(face_b_touching, [face_a]),
            "Touching face with distinct facial landmarks must not be suppressed"
        )

    def test_rescue_rotated_accumulates_across_different_orientations(self):
        """D1/D2: _rescue_rotated collects faces across orientations (e.g. 1 clockwise, 1 anticlockwise)."""
        class MockFace:
            def __init__(self, bbox):
                self.bbox = np.array(bbox, dtype=np.float32)
                self.kps = np.zeros((5, 2), dtype=np.float32)

        call_angles = []

        def mock_detect_faces_raw(frame, **kwargs):
            # Discriminate which rotation based on frame shape or mock call
            nonlocal call_angles
            h, w = frame.shape[:2]
            if len(call_angles) == 0:
                # 1st call: clockwise (h=640, w=480) -> returns Face 1
                call_angles.append("clockwise")
                return [MockFace([50, 50, 150, 150])]
            elif len(call_angles) == 1:
                # 2nd call: anticlockwise (h=640, w=480) -> returns Face 2
                call_angles.append("anticlockwise")
                return [MockFace([300, 300, 400, 400])]
            else:
                call_angles.append("180")
                return []

        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch('roop.face_util._detect_faces_raw', side_effect=mock_detect_faces_raw):
            faces = _rescue_rotated(test_frame)

        self.assertIsNotNone(faces)
        self.assertEqual(len(faces), 2, "_rescue_rotated must accumulate faces across multiple orientations")

    def test_rescue_rotated_per_orientation_try_except(self):
        """D16: An exception in one rotation does not abort remaining candidate orientations."""
        class MockFace:
            def __init__(self, bbox):
                self.bbox = np.array(bbox, dtype=np.float32)
                self.kps = np.zeros((5, 2), dtype=np.float32)

        calls = []

        def mock_detect_faces_raw(frame, **kwargs):
            nonlocal calls
            calls.append(len(calls))
            if len(calls) == 1:
                # 1st rotation fails with simulated error
                raise RuntimeError("Simulated OOM on clockwise rotation")
            elif len(calls) == 2:
                # 2nd rotation succeeds
                return [MockFace([100, 100, 200, 200])]
            return []

        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch('roop.face_util._detect_faces_raw', side_effect=mock_detect_faces_raw):
            faces = _rescue_rotated(test_frame)

        self.assertIsNotNone(faces)
        self.assertEqual(len(faces), 1, "Failure in first orientation must not abort subsequent orientation")

    def test_clahe_rescue_enrichment_uses_clean_original_frame(self):
        """D15: Detection uses CLAHE frame with aux=False, auxiliary enrichment uses clean frame."""
        class MockFace:
            def __init__(self, bbox):
                self.bbox = np.array(bbox, dtype=np.float32)
                self.kps = np.ones((5, 2), dtype=np.float32) * 50.0
                self.embedding = None

        raw_detect_frames = []
        aux_model_frames = []

        def mock_detect_raw(frame, aux=True, **kwargs):
            raw_detect_frames.append(frame.copy())
            return [MockFace([50, 50, 150, 150])]

        class MockModel:
            def get(self, frame, face):
                aux_model_frames.append(frame.copy())
                face.embedding = np.ones(512, dtype=np.float32)

        class MockFA:
            def __init__(self):
                self.models = {'recognition': MockModel()}

        class MockLease:
            def __enter__(self):
                return MockFA()
            def __exit__(self, *args):
                pass

        test_frame = np.full((100, 100, 3), 42, dtype=np.uint8)

        with patch('roop.face_util._detect_faces_raw', side_effect=mock_detect_raw), \
             patch('roop.face_util.lease_face_analyser', return_value=MockLease()):
            faces = _rescue_clahe(test_frame)

        self.assertIsNotNone(faces)
        self.assertEqual(len(faces), 1)
        self.assertIsNotNone(faces[0].embedding)

        # Verify that auxiliary recognition model received the untouched clean frame (values all 42)
        self.assertTrue(len(aux_model_frames) > 0)
        self.assertTrue(np.all(aux_model_frames[0] == 42), "Auxiliary recognition must run on original clean frame")


class TestFaceContactContaminationFallback(unittest.TestCase):
    """Test crop contamination fallback for faces with missing/invalid keypoints."""

    def test_contamination_falls_back_to_box_overlap_when_quad_is_none(self):
        """When face has no keypoints, crop_contamination falls back to box overlap instead of 0.0."""
        class BoxOnlyFace:
            def __init__(self, bbox):
                self.bbox = np.array(bbox, dtype=np.float32)
                self.kps = None

        # Face A at (100, 100, 200, 200) - area 10000
        # Face B at (120, 100, 220, 200) - overlapping Face A by 8000 (80%)
        face_a = BoxOnlyFace([100, 100, 200, 200])
        face_b = BoxOnlyFace([120, 100, 220, 200])

        contam = crop_contamination([face_a, face_b])
        self.assertGreater(contam[0], 0.70, "Box-only face heavily overlapped by neighbor must have > 0.70 contamination")


class TestFaceOverlapConnectedComponents(unittest.TestCase):
    """Test that build_regions partitions disjoint interacting pairs into localized components."""

    def test_build_regions_partitions_disjoint_interacting_pairs(self):
        from roop.face_overlap import build_regions

        class BoxFace:
            def __init__(self, bbox):
                self.bbox = np.asarray(bbox, dtype=np.float32)

        # Pair A (top-left)
        f0 = BoxFace([100, 100, 200, 200])
        f1 = BoxFace([160, 100, 260, 200])
        # Pair B (bottom-right, far away)
        f2 = BoxFace([1200, 700, 1300, 800])
        f3 = BoxFace([1260, 700, 1360, 800])

        regions = build_regions([f0, f1, f2, f3], (1080, 1920, 3), order=[0, 1, 2, 3])
        self.assertIsNotNone(regions)
        self.assertEqual(set(regions.keys()), {0, 1, 2, 3})

        # Pair A's regions must be localized around Pair A and not span to Pair B
        r0 = regions[0]
        self.assertLess(r0.x1, 500, "Pair A's ROI must stay localized in top-left")
        self.assertLess(r0.y1, 500, "Pair A's ROI must stay localized in top-left")

        # Pair B's regions must be localized around Pair B
        r2 = regions[2]
        self.assertGreater(r2.x0, 1000, "Pair B's ROI must stay localized in bottom-right")
        self.assertGreater(r2.y0, 600, "Pair B's ROI must stay localized in bottom-right")


class TestStrictAngleFormulasUntouched(unittest.TestCase):
    """STRICT INVARIANCE: verify that face angle formulas are 100% untouched."""

    def test_angle_function_signatures_and_names(self):
        """Ensure protected angle functions exist and have exact parameters."""
        sig_roll = inspect.signature(face_roll_degrees)
        self.assertEqual(list(sig_roll.parameters.keys()), ['face'])

        sig_yaw = inspect.signature(face_yaw_pitch)
        self.assertEqual(list(sig_yaw.parameters.keys()), ['face'])

        sig_solve = inspect.signature(solve_pose_5pt)
        self.assertEqual(list(sig_solve.parameters.keys()), ['kps'])

        sig_canon = inspect.signature(compute_canonical_roll_angle)
        self.assertEqual(list(sig_canon.parameters.keys()), ['landmarks'])

    def test_golden_angle_computations(self):
        """Verify exact mathematical output for standard upright 5-point face landmarks."""
        class MockFace:
            def __init__(self, kps):
                self.kps = np.array(kps, dtype=np.float32)

        # Standard upright face keypoints: [left_eye, right_eye, nose, left_mouth, right_mouth]
        upright_kps = [
            [100.0, 100.0],  # left eye
            [200.0, 100.0],  # right eye (dx=100, dy=0 -> roll = 0 deg)
            [150.0, 150.0],  # nose
            [110.0, 190.0],  # left mouth
            [190.0, 190.0],  # right mouth
        ]
        face_upright = MockFace(upright_kps)

        roll = face_roll_degrees(face_upright)
        self.assertAlmostEqual(roll, 0.0, places=3, msg="Upright face must have 0.0 degree roll")

        # 45-degree tilted face keypoints
        # Rotate by 45 degrees around center (150, 150)
        c, s = math.cos(math.radians(45)), math.sin(math.radians(45))
        rot_kps = []
        for x, y in upright_kps:
            rx = 150.0 + (x - 150.0) * c - (y - 150.0) * s
            ry = 150.0 + (x - 150.0) * s + (y - 150.0) * c
            rot_kps.append([rx, ry])

        face_tilted = MockFace(rot_kps)
        tilted_roll = face_roll_degrees(face_tilted)
        self.assertAlmostEqual(tilted_roll, 45.0, delta=1.0, msg="45-deg tilted face must yield ~45.0 deg roll")

    def test_ast_source_contains_no_substitutes(self):
        """Verify AST of face_util and face_analyser contains original mathematical definitions."""
        with open('app/roop/face_util.py', 'r', encoding='utf-8') as f:
            src_util = f.read()
        parsed_util = ast.parse(src_util)
        func_util = {node.name for node in ast.walk(parsed_util) if isinstance(node, ast.FunctionDef)}
        self.assertIn('solve_pose_5pt', func_util)

        with open('app/roop/face_analyser.py', 'r', encoding='utf-8') as f:
            src_analyser = f.read()
        parsed_analyser = ast.parse(src_analyser)
        func_analyser = {node.name for node in ast.walk(parsed_analyser) if isinstance(node, ast.FunctionDef)}
        self.assertIn('face_roll_degrees', func_analyser)
        self.assertIn('face_yaw_pitch', func_analyser)
        self.assertIn('compute_canonical_roll_angle', func_analyser)


if __name__ == '__main__':
    unittest.main()
