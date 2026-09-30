"""Tests for Stage 3: Adaptive SCRFD Detector Optimization and Validation Suite."""

import os
import sys
import math
import unittest
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
APP = os.path.join(ROOT, "app")
for p in (ROOT, APP):
    if p not in sys.path:
        sys.path.insert(0, p)

from roop.adaptive_detector import (
    AdaptiveDetectorConfig,
    LandmarkGeometryValidator,
    AdaptiveTrackState,
    AdaptivePlan,
    AdaptiveFaceDetector,
    get_adaptive_face_detector,
    reset_adaptive_face_detector,
)
from insightface.app.common import Face


def create_synthetic_face(
    bbox=(100, 100, 200, 200),
    det_score=0.92,
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    embedding_seed=42
) -> Face:
    """Helper to construct a mock InsightFace Face object with geometrically plausible landmarks."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    w, h = (x2 - x1), (y2 - y1)

    # 5 Keypoints: left_eye, right_eye, nose, left_mouth, right_mouth
    kps = np.array([
        [cx - w * 0.20, cy - h * 0.15],
        [cx + w * 0.20, cy - h * 0.15],
        [cx, cy],
        [cx - w * 0.15, cy + h * 0.25],
        [cx + w * 0.15, cy + h * 0.25],
    ], dtype=np.float32)

    rng = np.random.RandomState(embedding_seed)
    emb = rng.randn(512).astype(np.float32)
    norm = np.linalg.norm(emb)
    emb_norm = emb / norm if norm > 1e-6 else emb

    f = Face(bbox=np.array(bbox, dtype=np.float32), kps=kps, det_score=float(det_score))
    f.embedding = emb
    return f


class TestAdaptiveDetector(unittest.TestCase):

    def setUp(self):
        reset_adaptive_face_detector()
        self.config = AdaptiveDetectorConfig(
            enabled=True,
            full_recovery_interval=6,
            low_motion_threshold=0.05,
            large_motion_threshold=0.18,
            confidence_drop_threshold=0.15,
            min_confidence=0.55,
            roi_rescue_enabled=True,
            scene_cut_threshold=0.40,
            max_coast_frames=4,
        )
        self.detector = AdaptiveFaceDetector(self.config)

    def test_landmark_geometry_validator_valid(self):
        kps = np.array([
            [100.0, 100.0],
            [150.0, 100.0],
            [125.0, 125.0],
            [110.0, 150.0],
            [140.0, 150.0],
        ], dtype=np.float32)
        bbox = np.array([80.0, 80.0, 170.0, 170.0], dtype=np.float32)
        is_valid, reason = LandmarkGeometryValidator.validate(kps, bbox, (480, 640))
        self.assertTrue(is_valid, f"Expected valid geometry, got: {reason}")
        self.assertEqual(reason, "valid")

    def test_landmark_geometry_validator_collapsed(self):
        # Eyes collapsed together (< 3px apart)
        kps = np.array([
            [100.0, 100.0],
            [101.0, 100.0],  # only 1px apart
            [100.5, 120.0],
            [90.0, 140.0],
            [110.0, 140.0],
        ], dtype=np.float32)
        is_valid, reason = LandmarkGeometryValidator.validate(kps)
        self.assertFalse(is_valid)
        self.assertIn("interocular_collapsed", reason)

    def test_landmark_geometry_validator_inverted_axis(self):
        # Mouth above eyes (upside down / inverted labels)
        kps = np.array([
            [100.0, 150.0],  # Left eye below mouth
            [150.0, 150.0],  # Right eye below mouth
            [125.0, 125.0],
            [110.0, 100.0],  # Left mouth
            [140.0, 100.0],  # Right mouth
        ], dtype=np.float32)
        is_valid, reason = LandmarkGeometryValidator.validate(kps)
        self.assertFalse(is_valid)
        self.assertIn("inverted_facial_axis", reason)

    def test_landmark_geometry_validator_non_finite(self):
        kps = np.array([
            [np.nan, 100.0],
            [150.0, 100.0],
            [125.0, 125.0],
            [110.0, 150.0],
            [140.0, 150.0],
        ], dtype=np.float32)
        is_valid, reason = LandmarkGeometryValidator.validate(kps)
        self.assertFalse(is_valid)
        self.assertEqual(reason, "non_finite_coordinates")

    def test_low_motion_reuse_triggers_coast(self):
        dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        face0 = create_synthetic_face([100, 100, 200, 200], det_score=0.95)

        # Frame 0: Initial full detection
        faces0 = self.detector.detect(
            dummy_frame, frame_idx=0,
            det_fn=lambda fr: [face0]
        )
        self.assertEqual(len(faces0), 1)
        self.assertEqual(getattr(faces0[0], "_detection_mode"), "full")
        self.assertEqual(self.detector.telemetry["full_detections"], 1)

        # Frame 1: Tiny motion (displacement 1px, normalized ~0.007 < 0.05)
        det_called = [False]
        def mock_det_fail(fr):
            det_called[0] = True
            return [face0]

        faces1 = self.detector.detect(
            dummy_frame, frame_idx=1,
            det_fn=mock_det_fail
        )
        # Full detector should NOT have been called
        self.assertFalse(det_called[0], "Full detector was called unnecessarily during low motion")
        self.assertEqual(len(faces1), 1)
        self.assertEqual(getattr(faces1[0], "_detection_mode"), "coast_reuse")
        self.assertTrue(getattr(faces1[0], "_adaptive_reused", False))
        self.assertEqual(self.detector.telemetry["coast_reuses"], 1)

    def test_confidence_drop_triggers_detection(self):
        dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        face0 = create_synthetic_face([100, 100, 200, 200], det_score=0.95)

        # Frame 0
        self.detector.detect(dummy_frame, frame_idx=0, det_fn=lambda fr: [face0])

        # Manually simulate confidence drop in track state
        track = list(self.detector.tracks.values())[0]
        track.confidence = 0.45  # below min_confidence 0.55

        plan = self.detector.plan_strategy(dummy_frame, frame_idx=1)
        # Should trigger ROI rescue or full detection, NOT coast reuse
        self.assertIn(plan.action, ("ROI_RESCUE", "FULL_FRAME"))

    def test_face_count_change_triggers_full_detection(self):
        dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        face0 = create_synthetic_face([100, 100, 200, 200], det_score=0.95)

        # Frame 0: 1 face
        self.detector.detect(dummy_frame, frame_idx=0, det_fn=lambda fr: [face0])

        # Frame 1: Caller expects 2 faces
        plan = self.detector.plan_strategy(dummy_frame, frame_idx=1, expected_count=2)
        self.assertEqual(plan.action, "FULL_FRAME")
        self.assertIn("face_count_change", plan.reason)

    def test_large_motion_triggers_detection(self):
        dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        face0 = create_synthetic_face([100, 100, 200, 200], det_score=0.95)

        # Frame 0
        self.detector.detect(dummy_frame, frame_idx=0, det_fn=lambda fr: [face0])

        # Inject large velocity (50px displacement on 141px diagonal = ~0.35 > 0.18 threshold)
        track = list(self.detector.tracks.values())[0]
        track.velocity = np.array([45.0, 35.0, 0.0, 0.0], dtype=np.float32)

        plan = self.detector.plan_strategy(dummy_frame, frame_idx=1)
        self.assertEqual(plan.action, "FULL_FRAME")
        self.assertIn("large_motion", plan.reason)

    def test_occlusion_triggers_detection(self):
        dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        face_a = create_synthetic_face([100, 100, 200, 200], det_score=0.95, embedding_seed=1)
        face_b = create_synthetic_face([250, 100, 350, 200], det_score=0.95, embedding_seed=2)

        # Frame 0: Two well-separated faces
        self.detector.detect(dummy_frame, frame_idx=0, det_fn=lambda fr: [face_a, face_b])
        self.assertEqual(len(self.detector.tracks), 2)

        # Shift face_b so it heavily overlaps face_a (IoU > 0.12)
        tracks = list(self.detector.tracks.values())
        tracks[1].bbox = np.array([120, 110, 220, 210], dtype=np.float32)

        plan = self.detector.plan_strategy(dummy_frame, frame_idx=1)
        self.assertEqual(plan.action, "FULL_FRAME")
        self.assertIn("occlusion_overlap", plan.reason)

    def test_roi_rescue_missing_face(self):
        dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        face0 = create_synthetic_face([100, 100, 200, 200], det_score=0.95)

        # Frame 0: Full detect
        self.detector.detect(dummy_frame, frame_idx=0, det_fn=lambda fr: [face0])

        # Frame 1: Track dropped confidence so it needs rescue
        track = list(self.detector.tracks.values())[0]
        track.confidence = 0.40

        roi_called = [False]
        rescued_face = create_synthetic_face([102, 101, 202, 201], det_score=0.92)

        def mock_roi_fn(fr, box):
            roi_called[0] = True
            return [rescued_face]

        full_det_called = [False]
        def mock_full_fn(fr):
            full_det_called[0] = True
            return [face0]

        faces1 = self.detector.detect(
            dummy_frame, frame_idx=1,
            det_fn=mock_full_fn,
            roi_det_fn=mock_roi_fn
        )
        self.assertTrue(roi_called[0], "ROI detector was not called for rescue")
        self.assertFalse(full_det_called[0], "Full detector was called despite successful ROI rescue")
        self.assertEqual(len(faces1), 1)
        self.assertEqual(getattr(faces1[0], "_detection_mode"), "roi_rescue")
        self.assertEqual(self.detector.telemetry["roi_rescues_succeeded"], 1)

    def test_scene_cut_triggers_reset(self):
        frame_a = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_b = np.full((480, 640, 3), 255, dtype=np.uint8)  # Complete change -> cut
        face0 = create_synthetic_face([100, 100, 200, 200], det_score=0.95)

        # Frame 0
        self.detector.detect(frame_a, frame_idx=0, det_fn=lambda fr: [face0])
        self.assertEqual(len(self.detector.tracks), 1)

        # Frame 1: Scene cut
        plan = self.detector.plan_strategy(frame_b, frame_idx=1)
        self.assertEqual(plan.action, "FULL_FRAME")
        self.assertEqual(plan.reason, "scene_cut")

    def test_configurable_frequency(self):
        dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        face0 = create_synthetic_face([100, 100, 200, 200], det_score=0.95)

        # Frame 0: full
        self.detector.detect(dummy_frame, frame_idx=0, det_fn=lambda fr: [face0])

        # Frames 1..5: coast reuse (interval is 6)
        for i in range(1, 5):
            plan = self.detector.plan_strategy(dummy_frame, frame_idx=i)
            self.assertEqual(plan.action, "COAST_REUSE", f"Frame {i} should coast")
            # Update track state to avoid max coast
            track = list(self.detector.tracks.values())[0]
            track.last_seen_frame = i

        # Frame 6: interval elapsed (6 - 0 >= 6) -> periodic recovery
        plan6 = self.detector.plan_strategy(dummy_frame, frame_idx=6)
        self.assertEqual(plan6.action, "FULL_FRAME")
        self.assertEqual(plan6.reason, "periodic_recovery")

    def test_difficult_face_profile_and_small_face(self):
        dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        # Construct profile face with asymmetric landmarks (yaw ~ 55 degrees)
        profile_face = create_synthetic_face([100, 100, 200, 200], det_score=0.88)
        profile_face.kps = np.array([
            [125.0, 130.0],  # Left eye near nose
            [180.0, 130.0],  # Right eye wide
            [130.0, 150.0],  # Nose close to left eye (yawed)
            [135.0, 180.0],
            [170.0, 180.0],
        ], dtype=np.float32)

        # Construct tiny face (diagonal 35px < 60px)
        small_face = create_synthetic_face([300, 300, 325, 325], det_score=0.90)

        faces = self.detector.detect(
            dummy_frame, frame_idx=0,
            det_fn=lambda fr: [profile_face, small_face]
        )
        self.assertEqual(len(faces), 2)
        telemetry = self.detector.get_telemetry()
        self.assertGreaterEqual(telemetry["profile_detections"], 1)
        self.assertGreaterEqual(telemetry["small_face_detections"], 1)


if __name__ == "__main__":
    unittest.main()
