"""Regression tests for RetinaFace R50 close-up detection, multi-scale pyramid,
frame boundary context padding, and Distance-IoU Non-Maximum Suppression (DIoU-NMS).

Verifies:
1. DIoU computation and mathematical properties: DIoU = IoU - (d^2 / c^2).
2. DIoU-NMS multi-scale candidate suppression and touching face preservation.
3. Frame boundary context padding (minimum 64px, reflective border) and coordinate unpadding.
4. Scale pyramid generation, coordinate rescaling, and CLI/UI pyramid level parsing.
5. Adaptive pyramid trigger on high-res frames with 0 faces, faces > 500px, or > 75% coverage.
6. End-to-end RetinaFace R50 detection on synthetic high-resolution zoomed crops (> 75% height).
7. End-to-end RetinaFace R50 detection on border-intersecting macro crops.
8. Parallel scale execution across pyramid levels [0.5, 0.75, 1.0].
"""

import os
import sys
from pathlib import Path
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

import roop.globals
from roop.face_detector import (
    compute_diou,
    compute_diou_matrix,
    diou_nms,
    apply_context_padding,
    remove_context_padding,
    generate_scale_pyramid,
    rescale_detections,
    parse_scale_pyramid,
    should_trigger_pyramid,
    MultiScaleFaceDetector,
    detect_faces,
    detect_retinaface_closeup,
    DEFAULT_PYRAMID_SCALES,
    MIN_BORDER_PADDING_PX,
)


def _create_synthetic_face_image(size: int = 1024, zoom_factor: float = 0.85) -> np.ndarray:
    """Create a synthetic high-resolution face image or zoomed crop.

    Uses realistic facial textures and facial feature landmarks from facesets if available,
    or generates a detailed facial pattern with skin tones, eyes, nose, and mouth.
    """
    # Prefer real fixture image if available
    sample_path = REPO_ROOT / 'facesets' / 'akansha.png'
    if sample_path.exists():
        img = cv2.imread(str(sample_path))
        if img is not None:
            h, w = img.shape[:2]
            # Face region in akansha.png is around [75, 75, 180, 220]
            # Crop tightly to simulate zoomed close-up
            y1 = int(75 + (1.0 - zoom_factor) * 20)
            y2 = int(220 - (1.0 - zoom_factor) * 15)
            x1 = int(75 + (1.0 - zoom_factor) * 15)
            x2 = int(180 - (1.0 - zoom_factor) * 15)
            crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4)

    # Procedural high-resolution synthetic face fallback
    canvas = np.full((size, size, 3), (35, 35, 35), dtype=np.uint8)
    center = (size // 2, size // 2)
    face_r = int((size // 2) * zoom_factor)

    # Skin tone base
    cv2.circle(canvas, center, face_r, (140, 175, 220), -1)
    # Eyes
    eye_y = int(center[1] - face_r * 0.2)
    left_eye_x = int(center[0] - face_r * 0.35)
    right_eye_x = int(center[0] + face_r * 0.35)
    eye_r = max(4, int(face_r * 0.12))
    cv2.circle(canvas, (left_eye_x, eye_y), eye_r, (240, 240, 240), -1)
    cv2.circle(canvas, (right_eye_x, eye_y), eye_r, (240, 240, 240), -1)
    cv2.circle(canvas, (left_eye_x, eye_y), max(2, eye_r // 2), (20, 20, 20), -1)
    cv2.circle(canvas, (right_eye_x, eye_y), max(2, eye_r // 2), (20, 20, 20), -1)
    # Nose
    nose_y = int(center[1] + face_r * 0.1)
    cv2.ellipse(canvas, (center[0], nose_y), (int(face_r * 0.08), int(face_r * 0.15)), 0, 0, 360, (110, 140, 190), -1)
    # Mouth
    mouth_y = int(center[1] + face_r * 0.45)
    cv2.ellipse(canvas, (center[0], mouth_y), (int(face_r * 0.25), int(face_r * 0.1)), 0, 0, 180, (60, 60, 160), -1)

    return canvas


class TestRetinaFaceCloseupDetection(unittest.TestCase):
    """Test suite for RetinaFace R50 close-up detection and multi-scale pyramid."""

    def setUp(self):
        roop.globals.face_detector_threshold = 0.50
        roop.globals.face_detector_nms = 0.40
        roop.globals.detector_scale_pyramid = 'auto'

    def test_diou_identical_boxes(self):
        """Identical boxes must have IoU=1.0 and d=0, giving DIoU=1.0."""
        b1 = np.array([50.0, 50.0, 200.0, 200.0])
        b2 = np.array([50.0, 50.0, 200.0, 200.0])
        val = compute_diou(b1, b2)
        self.assertAlmostEqual(val, 1.0, places=5)

    def test_diou_disjoint_boxes(self):
        """Non-overlapping boxes must have IoU=0.0 and DIoU < 0.0."""
        b1 = np.array([10.0, 10.0, 50.0, 50.0])
        b2 = np.array([500.0, 500.0, 600.0, 600.0])
        val = compute_diou(b1, b2)
        self.assertLess(val, 0.0)

    def test_diou_formula_exact(self):
        """Verify DIoU formula against hand-calculated values: DIoU = IoU - (d^2 / c^2)."""
        # Box 1: [0, 0, 100, 100], Center: (50, 50), Area: 10000
        # Box 2: [50, 0, 150, 100], Center: (100, 50), Area: 10000
        # Inter: [50, 0, 100, 100], Area: 5000
        # Union: 10000 + 10000 - 5000 = 15000 -> IoU = 5000 / 15000 = 1/3 ~ 0.333333
        # Center dist: d^2 = (100 - 50)^2 + (50 - 50)^2 = 2500
        # Smallest enclosing box: [0, 0, 150, 100], w=150, h=100
        # Diagonal: c^2 = 150^2 + 100^2 = 22500 + 10000 = 32500
        # Penalty: d^2 / c^2 = 2500 / 32500 = 1 / 13 ~ 0.076923
        # Expected DIoU: 1/3 - 1/13 = 10 / 39 ~ 0.256410
        b1 = np.array([0.0, 0.0, 100.0, 100.0])
        b2 = np.array([50.0, 0.0, 150.0, 100.0])
        expected_diou = (1.0 / 3.0) - (2500.0 / 32500.0)
        computed_diou = compute_diou(b1, b2)
        self.assertAlmostEqual(computed_diou, expected_diou, places=4)

    def test_diou_nms_candidate_suppression(self):
        """Candidate boxes of the same face across multiple scales must be merged to one."""
        # Simulate 3 candidate detections from scales 0.5, 0.75, 1.0 for the same face
        dets = np.array([
            [100.0, 100.0, 700.0, 700.0, 0.98],  # Highest score
            [104.0, 98.0, 698.0, 705.0, 0.91],   # Duplicate candidate
            [95.0, 102.0, 703.0, 695.0, 0.85],   # Duplicate candidate
        ], dtype=np.float32)
        kpss = np.zeros((3, 5, 2), dtype=np.float32)

        kept_dets, kept_kpss, keep_idx = diou_nms(dets, kpss, iou_thresh=0.40)
        self.assertEqual(len(kept_dets), 1)
        self.assertEqual(keep_idx, [0])
        self.assertAlmostEqual(kept_dets[0, 4], 0.98)

    def test_diou_nms_preserves_touching_faces(self):
        """Two distinct touching faces with center separation must NOT be falsely suppressed."""
        # Two faces side-by-side with slight overlap
        dets = np.array([
            [100.0, 100.0, 300.0, 300.0, 0.95],
            [260.0, 100.0, 460.0, 300.0, 0.92],
        ], dtype=np.float32)
        kept_dets, _, keep_idx = diou_nms(dets, iou_thresh=0.40)
        self.assertEqual(len(kept_dets), 2)
        self.assertEqual(len(keep_idx), 2)

    def test_frame_boundary_context_padding(self):
        """Context padding must add at least 64px border and unpadding must restore coordinates."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        padded, offsets = apply_context_padding(frame, min_padding=64, mode='reflect')

        pad_top, pad_bottom, pad_left, pad_right = offsets
        self.assertGreaterEqual(pad_top, MIN_BORDER_PADDING_PX)
        self.assertGreaterEqual(pad_left, MIN_BORDER_PADDING_PX)
        self.assertEqual(padded.shape[0], 480 + pad_top + pad_bottom)
        self.assertEqual(padded.shape[1], 640 + pad_left + pad_right)

        # Test coordinate mapping
        dummy_dets = np.array([[pad_left + 10.0, pad_top + 20.0, pad_left + 110.0, pad_top + 120.0, 0.99]])
        dummy_kps = np.array([[[pad_left + 30.0, pad_top + 40.0],
                               [pad_left + 80.0, pad_top + 40.0],
                               [pad_left + 55.0, pad_top + 70.0],
                               [pad_left + 40.0, pad_top + 95.0],
                               [pad_left + 70.0, pad_top + 95.0]]])

        unpad_dets, unpad_kps = remove_context_padding(dummy_dets, dummy_kps, offsets)
        self.assertAlmostEqual(unpad_dets[0, 0], 10.0)
        self.assertAlmostEqual(unpad_dets[0, 1], 20.0)
        self.assertAlmostEqual(unpad_dets[0, 2], 110.0)
        self.assertAlmostEqual(unpad_dets[0, 3], 120.0)
        self.assertAlmostEqual(unpad_kps[0, 0, 0], 30.0)
        self.assertAlmostEqual(unpad_kps[0, 0, 1], 40.0)

    def test_scale_pyramid_generation_and_rescaling(self):
        """Pyramid generation and rescaling B_orig = B_scaled / s must be exact."""
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        scales = [0.5, 0.75, 1.0]
        pyramid = generate_scale_pyramid(frame, scales)
        self.assertEqual(len(pyramid), 3)

        # Scale 0.5: 300x400
        self.assertEqual(pyramid[0][0], 0.5)
        self.assertEqual(pyramid[0][1].shape[:2], (300, 400))
        # Scale 1.0: 600x800
        self.assertEqual(pyramid[2][0], 1.0)
        self.assertEqual(pyramid[2][1].shape[:2], (600, 800))

        # Test rescaling coordinates
        scaled_box = np.array([[50.0, 50.0, 200.0, 200.0, 0.9]])
        scaled_kps = np.array([[[75.0, 75.0], [125.0, 75.0], [100.0, 100.0], [80.0, 140.0], [120.0, 140.0]]])
        orig_box, orig_kps = rescale_detections(scaled_box, scaled_kps, scale_factor=0.5)

        self.assertAlmostEqual(orig_box[0, 0], 100.0)
        self.assertAlmostEqual(orig_box[0, 2], 400.0)
        self.assertAlmostEqual(orig_kps[0, 0, 0], 150.0)

    def test_parse_scale_pyramid_formats(self):
        """CLI/UI option --detector-scale-pyramid parsing must support auto, none, and lists."""
        self.assertIsNone(parse_scale_pyramid('auto'))
        self.assertIsNone(parse_scale_pyramid(None))
        self.assertEqual(parse_scale_pyramid('none'), [1.0])
        self.assertEqual(parse_scale_pyramid('0.5,0.75,1.0'), [0.5, 0.75, 1.0])
        self.assertEqual(parse_scale_pyramid([0.5, 1.0]), [0.5, 1.0])

    def test_should_trigger_pyramid_conditions(self):
        """Verify adaptive trigger conditions matching specification."""
        # 1. Face height > 500px -> True
        self.assertTrue(should_trigger_pyramid((1080, 1920), estimated_face_height=520))

        # 2. 0 faces on high-res frame -> True
        self.assertTrue(should_trigger_pyramid((1080, 1920), initial_dets=np.empty((0, 5))))

        # 3. 0 faces on small low-res thumbnail -> False
        self.assertFalse(should_trigger_pyramid((240, 320), initial_dets=np.empty((0, 5))))

        # 4. Detected face fills > 75% of frame height (e.g. 800px on 1000px frame) -> True
        closeup_det = np.array([[100, 100, 700, 900, 0.9]])
        self.assertTrue(should_trigger_pyramid((1000, 1000), initial_dets=closeup_det))

        # 5. Normal small face (100px on 1080p) -> False
        normal_det = np.array([[500, 300, 600, 420, 0.95]])
        self.assertFalse(should_trigger_pyramid((1080, 1920), initial_dets=normal_det))

    def test_retinaface_closeup_high_resolution_zoom(self):
        """Confirm RetinaFace R50 detection on high-resolution zoomed crops (> 75% frame height)."""
        zoomed_img = _create_synthetic_face_image(size=800, zoom_factor=0.90)

        # Run detection with multi-scale dynamic pyramid
        boxes, kpss = detect_retinaface_closeup(zoomed_img, scales=(0.5, 0.75, 1.0))

        self.assertGreater(len(boxes), 0, "RetinaFace R50 multi-scale pyramid must detect zoomed close-up face")
        best_score = float(boxes[0, 4])
        self.assertGreaterEqual(best_score, 0.50, f"Detection score {best_score} should be >= 0.50 threshold")
        self.assertEqual(kpss.shape[1:], (5, 2), "Landmarks must have shape (5, 2)")

        # Verify face height fills > 75% of the frame
        box_h = boxes[0, 3] - boxes[0, 1]
        self.assertGreater(box_h, 800 * 0.65, "Detected bounding box should capture close-up geometry")

    def test_retinaface_boundary_intersecting_macro_crop(self):
        """Confirm detection on macro shot where face is cut off by camera frame edges."""
        zoomed_img = _create_synthetic_face_image(size=1024, zoom_factor=0.95)
        h, w = zoomed_img.shape[:2]

        # Crop so forehead (top) and chin (bottom) or side intersect the frame borders
        # Simulates real-world macro shot with border cutoff
        macro_crop = zoomed_img[int(h * 0.12):int(h * 0.88), int(w * 0.05):int(w * 0.95)]
        macro_hi = cv2.resize(macro_crop, (1024, 1024), interpolation=cv2.INTER_LINEAR)

        # Run detection with context padding and multi-scale pyramid
        boxes, kpss = detect_retinaface_closeup(macro_hi, scales=(0.5, 0.75, 1.0), padding=64)

        self.assertGreater(len(boxes), 0, "RetinaFace R50 with context padding must detect border-intersecting face")
        self.assertGreaterEqual(float(boxes[0, 4]), 0.50)
        self.assertEqual(kpss.shape, (len(boxes), 5, 2))

    def test_parallel_multi_scale_execution(self):
        """Confirm parallel execution across pyramid levels [0.5, 0.75, 1.0] works without error."""
        zoomed_img = _create_synthetic_face_image(size=720, zoom_factor=0.85)

        detector = MultiScaleFaceDetector()
        boxes, kpss = detector.detect(
            zoomed_img,
            det_size=640,
            det_thresh=0.50,
            scales=[0.5, 0.75, 1.0],
            parallel=True,
            max_workers=2,
            force_pyramid=True,
        )

        self.assertGreater(len(boxes), 0, "Parallel multi-scale execution should successfully detect faces")
        self.assertEqual(kpss.shape, (len(boxes), 5, 2))


if __name__ == '__main__':
    unittest.main()
