"""Behavioral contract for Phase 3 temporal tracking and detection policy."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.temporal_tracker import TemporalFaceTracker  # noqa: E402


def _emb(axis, size=8):
    out = np.zeros(size, dtype=np.float32)
    out[axis] = 1.0
    return out


def _face(x, y=100.0, size=80.0, identity=0, score=0.95, yaw=0.0,
          mask=None):
    box = np.array([x, y, x + size, y + size], dtype=np.float32)
    kps = np.array([[x + size * 0.30, y + size * 0.35],
                    [x + size * 0.70, y + size * 0.35],
                    [x + size * 0.50, y + size * 0.52],
                    [x + size * 0.34, y + size * 0.72],
                    [x + size * 0.66, y + size * 0.72]], dtype=np.float32)
    return {
        "bbox": box,
        "kps": kps,
        "landmark_2d_106": kps.copy(),
        "pose": np.array([0.0, yaw, 0.0], dtype=np.float32),
        "embedding": _emb(identity),
        "det_score": score,
        "mask": mask,
    }


def _step(tracker, frame, faces, mode="full"):
    return tracker.update(faces, frame, (480, 640, 3), detection_mode=mode)


class PersistentTrackScenarios(unittest.TestCase):

    def test_stationary_face_keeps_one_stable_id_and_all_state_fields(self):
        tracker = TemporalFaceTracker(full_interval=20)
        for frame in range(8):
            plan = tracker.plan(frame, (480, 640, 3))
            _step(tracker, frame, [_face(200)], plan.mode)
        self.assertEqual(len(tracker.tracks), 1)
        track = tracker.tracks[0]
        self.assertEqual(track.track_id, 0)
        self.assertEqual(track.status, "stable")
        self.assertIsNotNone(track.landmarks)
        self.assertIsNotNone(track.pose)
        self.assertIsNotNone(track.identity_embedding)
        self.assertEqual(track.previous_frame_index, 6)
        self.assertEqual(track.last_frame_index, 7)
        self.assertEqual(track.velocity.shape, (4,))

    def test_fast_motion_releases_smoothing_and_keeps_velocity(self):
        tracker = TemporalFaceTracker(full_interval=20)
        for frame, x in enumerate((40.0, 75.0, 180.0, 300.0)):
            mode = tracker.plan(frame, (480, 640, 3)).mode
            _step(tracker, frame, [_face(x)], mode)
        track = tracker.tracks[0]
        self.assertEqual(track.track_id, 0)
        self.assertGreater(float(track.velocity[0]), 20.0)
        # Fast motion uses the release alpha; it must not remain near the old
        # position as a fixed EMA would.
        self.assertGreater(float(track.bbox[0]), 240.0)

    def test_head_rotation_smooths_pose_and_landmarks_without_new_track(self):
        tracker = TemporalFaceTracker(full_interval=20)
        for frame, yaw in enumerate((0.0, 12.0, 35.0, 58.0, 75.0)):
            mode = tracker.plan(frame, (480, 640, 3)).mode
            _step(tracker, frame, [_face(180, yaw=yaw)], mode)
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(tracker.tracks[0].track_id, 0)
        self.assertGreater(float(tracker.tracks[0].pose[1]), 30.0)
        self.assertEqual(tracker.tracks[0].landmarks.shape, (5, 2))

    def test_previous_mask_is_carried_between_observations(self):
        tracker = TemporalFaceTracker(full_interval=20)
        first = np.zeros((8, 8), dtype=np.float32)
        second = np.ones((8, 8), dtype=np.float32)
        _step(tracker, 0, [_face(180, mask=first)], "full")
        _step(tracker, 1, [_face(182, mask=second)], "roi")
        np.testing.assert_array_equal(tracker.tracks[0].previous_mask, first)

    def test_two_faces_crossing_keep_identity_by_global_assignment(self):
        tracker = TemporalFaceTracker(full_interval=20)
        a_positions = np.linspace(80.0, 300.0, 9)
        b_positions = np.linspace(300.0, 80.0, 9)
        observed = {}
        for frame, (ax, bx) in enumerate(zip(a_positions, b_positions)):
            mode = tracker.plan(frame, (480, 640, 3)).mode
            result = _step(tracker, frame,
                           [_face(ax, identity=0), _face(bx, identity=1)], mode)
            observed[frame] = dict(result["assignments"])
        self.assertEqual(len(tracker.tracks), 2)
        self.assertEqual(observed[0][0], 0)
        self.assertEqual(observed[0][1], 1)
        self.assertEqual(observed[8][0], 0)
        self.assertEqual(observed[8][1], 1)

    def test_two_touching_faces_do_not_merge_or_swap(self):
        tracker = TemporalFaceTracker(full_interval=20)
        for frame in range(6):
            mode = tracker.plan(frame, (480, 640, 3)).mode
            result = _step(tracker, frame,
                           [_face(200, identity=0), _face(225, identity=1)], mode)
            self.assertEqual(result["assignments"].get(0), 0)
            self.assertEqual(result["assignments"].get(1), 1)
        self.assertEqual([t.track_id for t in tracker.tracks], [0, 1])

    def test_temporary_occlusion_predicts_then_recovers_same_id(self):
        tracker = TemporalFaceTracker(full_interval=20, max_misses=3)
        _step(tracker, 0, [_face(150)], "full")
        _step(tracker, 1, [_face(170)], "roi")
        _step(tracker, 2, [], "roi_fallback_full")
        _step(tracker, 3, [], "roi_fallback_full")
        self.assertEqual(tracker.tracks[0].status, "uncertain")
        self.assertIsNotNone(tracker.tracks[0].predicted_bbox)
        result = _step(tracker, 4, [_face(210)], "full")
        self.assertEqual(result["assignments"].get(0), 0)
        self.assertEqual(tracker.tracks[0].status, "stable")

    def test_leaving_and_reentering_emits_lifecycle_and_reuses_id(self):
        tracker = TemporalFaceTracker(full_interval=2, max_misses=2, reid_age=20)
        _step(tracker, 0, [_face(120)], "full")
        _step(tracker, 1, [_face(125)], "roi")
        _step(tracker, 2, [], "full")
        _step(tracker, 3, [], "full")
        _step(tracker, 4, [], "full")
        self.assertEqual(tracker.tracks[0].status, "lost")
        self.assertEqual(tracker.events[0]["type"], "left")
        result = _step(tracker, 5, [_face(500)], "full")
        self.assertEqual(result["assignments"].get(0), 0)
        self.assertTrue(any(e["type"] == "recovered" for e in result["events"]))
        self.assertEqual(tracker.tracks[0].track_id, 0)

    def test_new_face_appearing_gets_new_track(self):
        tracker = TemporalFaceTracker(full_interval=20)
        _step(tracker, 0, [_face(80, identity=0)], "full")
        result = _step(tracker, 1,
                       [_face(85, identity=0), _face(500, identity=1)], "roi")
        self.assertEqual(result["assignments"].get(0), 0)
        self.assertEqual(result["assignments"].get(1), 1)
        self.assertEqual(len(tracker.tracks), 2)
        self.assertEqual([t.track_id for t in tracker.tracks], [0, 1])


class DetectionPolicyTest(unittest.TestCase):

    def test_stable_roi_periodic_recovery_and_lost_full_fallback(self):
        tracker = TemporalFaceTracker(full_interval=4, max_misses=1)
        self.assertEqual(tracker.plan(0, (480, 640, 3)).mode, "full")
        _step(tracker, 0, [_face(200)], "full")
        _step(tracker, 1, [_face(205)], "roi")
        self.assertEqual(tracker.plan(2, (480, 640, 3)).mode, "roi")
        self.assertEqual(tracker.plan(4, (480, 640, 3)).mode, "full")
        _step(tracker, 4, [], "full")
        self.assertEqual(tracker.plan(5, (480, 640, 3)).mode, "full")
        _step(tracker, 5, [], "full")
        self.assertEqual(tracker.plan(6, (480, 640, 3)).mode, "coast")
        self.assertEqual(tracker.plan(9, (480, 640, 3)).mode, "full")

    def test_roi_policy_reduces_full_frame_calls_without_disabling_recovery(self):
        tracker = TemporalFaceTracker(full_interval=8)
        for frame in range(40):
            plan = tracker.plan(frame, (480, 640, 3))
            _step(tracker, frame, [_face(120 + frame * 2)], plan.mode)
        stats = tracker.stats
        self.assertEqual(stats["full_detections"], 5)
        self.assertEqual(stats["roi_detections"], 35)
        self.assertEqual(stats["full_detections"] + stats["roi_detections"], 40)


if __name__ == "__main__":
    unittest.main()
