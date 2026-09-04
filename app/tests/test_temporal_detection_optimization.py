import os
import sys
import unittest
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.face_analyser import (
    TemporalFaceDetector,
    compute_histogram_signature,
    compare_histogram_difference,
    scale_face_coordinates,
    detect_faces_half_res,
    get_many_faces,
    find_similar_faces,
    reset_temporal_detector
)

def make_synthetic_face(bbox, det_score=0.95, emb_seed=1):
    rng = np.random.RandomState(emb_seed)
    emb = rng.randn(512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = x2 - x1, y2 - y1
    kps = np.array([
        [cx - w * 0.2, cy - h * 0.1],
        [cx + w * 0.2, cy - h * 0.1],
        [cx, cy],
        [cx - w * 0.15, cy + h * 0.2],
        [cx + w * 0.15, cy + h * 0.2]
    ], dtype=np.float32)
    return {
        'bbox': np.array(bbox, dtype=np.float32),
        'kps': kps,
        'det_score': float(det_score),
        'embedding': emb
    }

class TemporalDetectionOptimizationTest(unittest.TestCase):

    def test_histogram_scene_cut_detection(self):
        # Two identical frames -> diff = 0.0
        frame_a = np.zeros((240, 320, 3), dtype=np.uint8)
        frame_a[50:150, 50:150] = (0, 255, 0)
        hist_a = compute_histogram_signature(frame_a)
        self.assertIsNotNone(hist_a)
        diff_same = compare_histogram_difference(hist_a, hist_a)
        self.assertAlmostEqual(diff_same, 0.0, places=4)

        # Completely different scene (scene cut) -> diff > 0.4
        frame_b = np.full((240, 320, 3), 255, dtype=np.uint8)
        frame_b[50:150, 50:150] = (0, 0, 255)
        hist_b = compute_histogram_signature(frame_b)
        diff_cut = compare_histogram_difference(hist_a, hist_b)
        self.assertGreater(diff_cut, 0.4, f"Scene cut diff should be > 0.4, got {diff_cut}")

    def test_scale_face_coordinates(self):
        face = make_synthetic_face([100.0, 100.0, 200.0, 200.0])
        inv_scale = 2.5
        scale_face_coordinates(face, inv_scale)
        self.assertTrue(np.allclose(face['bbox'], [250.0, 250.0, 500.0, 500.0]))
        self.assertAlmostEqual(float(face['kps'][2, 0]), 375.0)
        self.assertAlmostEqual(float(face['kps'][2, 1]), 375.0)

    def test_dynamic_detection_interval_lifecycle(self):
        # Create a mock detector to verify interval schedule
        det = TemporalFaceDetector(interval=5, confidence_thresh=0.65, iou_thresh=0.65)
        full_detect_count = [0]
        crop_predict_count = [0]

        real_detect_half = det.detect

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        face_f0 = make_synthetic_face([100.0, 100.0, 200.0, 200.0], det_score=0.98, emb_seed=10)

        # Mock the raw detect and crop verification
        def mock_detect_raw(frame, max_dim=640):
            full_detect_count[0] += 1
            # Return current face location
            return [make_synthetic_face(face_f0['bbox'].copy(), det_score=0.98, emb_seed=10)]

        def mock_predict_verify(frame, current_idx):
            # Simulate successful 15% crop prediction
            crop_predict_count[0] += 1
            f = make_synthetic_face(face_f0['bbox'].copy(), det_score=0.98, emb_seed=10)
            f['_track_id'] = 1
            return [f]

        # Patch internal calls for deterministic interval verification
        import roop.face_analyser
        orig_detect_half = roop.face_analyser.detect_faces_half_res
        roop.face_analyser.detect_faces_half_res = mock_detect_raw
        det._predict_and_verify = mock_predict_verify

        try:
            # Frame 0: Full detection
            det.detect(dummy_frame, frame_index=0)
            self.assertEqual(full_detect_count[0], 1)
            self.assertEqual(crop_predict_count[0], 0)

            # Frames 1, 2, 3, 4: Predictive 15% crop (skips full-frame detect)
            for f in range(1, 5):
                det.detect(dummy_frame, frame_index=f)
            self.assertEqual(crop_predict_count[0], 4)
            self.assertEqual(full_detect_count[0], 1, "Frames 1..4 must skip full detection")

            # Frame 5: Re-triggers full detection to correct drift
            det.detect(dummy_frame, frame_index=5)
            self.assertEqual(full_detect_count[0], 2, "Frame 5 must trigger full detection")

            # Frames 6..9: Predictive again
            for f in range(6, 10):
                det.detect(dummy_frame, frame_index=f)
            self.assertEqual(full_detect_count[0], 2)
            self.assertEqual(crop_predict_count[0], 8)

            # Frame 10: Full detection
            det.detect(dummy_frame, frame_index=10)
            self.assertEqual(full_detect_count[0], 3)
        finally:
            roop.face_analyser.detect_faces_half_res = orig_detect_half

    def test_drift_and_pan_fallback_triggers_full_detection(self):
        # When crop prediction drops below IoU 0.65 or confidence 0.65, full detection must be re-triggered
        det = TemporalFaceDetector(interval=5, confidence_thresh=0.65, iou_thresh=0.65)
        full_detect_count = [0]

        def mock_detect_raw(frame, max_dim=640):
            full_detect_count[0] += 1
            return [make_synthetic_face([100.0, 100.0, 200.0, 200.0], det_score=0.98, emb_seed=10)]

        def mock_predict_drift(frame, current_idx):
            # Simulates drift / camera pan where IoU drops < 0.65 -> returns None
            return None

        import roop.face_analyser
        orig_detect_half = roop.face_analyser.detect_faces_half_res
        roop.face_analyser.detect_faces_half_res = mock_detect_raw
        det._predict_and_verify = mock_predict_drift

        try:
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Frame 0: Full detection
            det.detect(dummy_frame, frame_index=0)
            self.assertEqual(full_detect_count[0], 1)

            # Frame 1: Drift occurred -> must fallback to full detection immediately
            det.detect(dummy_frame, frame_index=1)
            self.assertEqual(full_detect_count[0], 2, "Drift must immediately re-trigger full detection")
        finally:
            roop.face_analyser.detect_faces_half_res = orig_detect_half

    def test_find_similar_faces_filtering(self):
        reset_temporal_detector()
        target_face = make_synthetic_face([100, 100, 200, 200], emb_seed=1)
        dissimilar_face = make_synthetic_face([300, 100, 400, 200], emb_seed=99)

        # Mock get_many_faces
        import roop.face_analyser
        orig_get_many = roop.face_analyser.get_many_faces

        def mock_get_many(frame, **kwargs):
            return [target_face, dissimilar_face]

        roop.face_analyser.get_many_faces = mock_get_many
        try:
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            matched = find_similar_faces(dummy_frame, [target_face], threshold=0.65)
            self.assertEqual(len(matched), 1)
            self.assertTrue(np.allclose(matched[0]['bbox'], target_face['bbox']))
        finally:
            roop.face_analyser.get_many_faces = orig_get_many

if __name__ == '__main__':
    unittest.main()
