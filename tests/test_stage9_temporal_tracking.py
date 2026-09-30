"""Tests for Stage 9 — Temporal Face Tracking and Flicker Control.

Validates:
1. Formal Multi-State Lifecycle:
   - TENTATIVE, STABLE, CROSSING, OCCLUDED, COASTING, PROFILE_TURNING, LOST, RECOVERED, TERMINATED.
2. Complete State Retention per Track:
   - bbox, landmarks (kps & lm106), embedding, confidence, velocity, pose,
     last_seen_frame, source_assignment, mask_state.
3. Edge Case Robustness:
   - 1. Temporary detector dropout
   - 2. Face crossing another face (interactivity & identity switch prevention)
   - 3. Face count changes
   - 4. Occlusion handling
   - 5. Profile transition (yaw > 55 deg)
   - 6. Rapid movement (adaptive process noise release)
   - 7. Target face disappearing
   - 8. Target face returning (Re-ID archive memory reactivation)
4. Confidence-Aware Interpolation & Discontinuity Guards:
   - Shot cut refusal, velocity jump refusal, identity mismatch refusal,
     Cubic Hermite spline trajectory continuity.
5. Explicit Temporal Quality Metrics:
   - Landmark jitter variance, identity stability, mask popping rate, motion fidelity.
"""

from __future__ import annotations

import unittest
import numpy as np

from roop.temporal_state_machine import (
    ConfidenceAwareInterpolator,
    RobustFaceTrack,
    TemporalQualityMetrics,
    TemporalStateMachineTracker,
    TrackState,
)


def _mock_emb(axis: int, dim: int = 512) -> np.ndarray:
    """Generate a clean unit-normalized ArcFace embedding vector."""
    v = np.zeros(dim, dtype=np.float32)
    v[axis % dim] = 1.0
    return v


def _mock_face(
    x: float,
    y: float = 120.0,
    size: float = 100.0,
    identity: int = 0,
    score: float = 0.95,
    yaw: float = 0.0,
    mask: np.ndarray = None
) -> dict:
    """Create a mock Face dictionary."""
    box = np.array([x, y, x + size, y + size], dtype=np.float32)
    kps = np.array([
        [x + size * 0.30, y + size * 0.35],
        [x + size * 0.70, y + size * 0.35],
        [x + size * 0.50, y + size * 0.52],
        [x + size * 0.35, y + size * 0.72],
        [x + size * 0.65, y + size * 0.72]
    ], dtype=np.float32)
    lm106 = np.repeat(kps[:1], 106, axis=0)  # mock 106 points

    return {
        "bbox": box,
        "kps": kps,
        "landmark_2d_106": lm106,
        "pose": np.array([0.0, yaw, 0.0], dtype=np.float32),
        "embedding": _mock_emb(identity),
        "det_score": float(score),
        "mask": mask if mask is not None else np.ones((64, 64), dtype=np.float32),
        "source_assignment": identity
    }


class TestTrackLifecycleAndStateRetention(unittest.TestCase):
    """Validates state machine transitions and complete 9-attribute state retention."""

    def test_track_progression_from_tentative_to_stable(self):
        """Track starts as TENTATIVE and promotes to STABLE after MIN_HITS_STABLE."""
        tracker = TemporalStateMachineTracker(max_lost=20, max_coast=10)

        # Frame 0: 1st observation -> TENTATIVE
        res0 = tracker.update([_mock_face(100.0)], frame_index=0)
        self.assertEqual(len(tracker.tracks), 1)
        track = tracker.tracks[0]
        self.assertEqual(track.state, TrackState.TENTATIVE)
        self.assertEqual(track.hits, 1)

        # Frame 1: 2nd observation -> still TENTATIVE
        tracker.update([_mock_face(102.0)], frame_index=1)
        self.assertEqual(track.hits, 2)
        self.assertEqual(track.state, TrackState.TENTATIVE)

        # Frame 2: 3rd observation -> STABLE
        tracker.update([_mock_face(104.0)], frame_index=2)
        self.assertEqual(track.hits, 3)
        self.assertEqual(track.state, TrackState.STABLE)

    def test_complete_nine_attribute_retention(self):
        """Track maintains all 9 required attributes with non-null values."""
        tracker = TemporalStateMachineTracker()
        face = _mock_face(150.0, y=100.0, size=80.0, identity=1, yaw=15.0)
        tracker.update([face], frame_index=0)
        track = tracker.tracks[0]
        track.source_assignment = 1

        # 1. bbox
        self.assertEqual(track.bbox.shape, (4,))
        # 2. landmarks
        self.assertIsNotNone(track.kps)
        self.assertEqual(track.kps.shape, (5, 2))
        self.assertIsNotNone(track.landmark_2d_106)
        # 3. embedding
        self.assertIsNotNone(track.embedding)
        self.assertEqual(track.embedding.shape, (512,))
        # 4. confidence
        self.assertGreater(track.confidence, 0.7)
        # 5. velocity
        self.assertEqual(track.velocity.shape, (4,))
        # 6. pose
        self.assertEqual(track.pose.shape, (3,))
        self.assertAlmostEqual(track.pose[1], 15.0)
        # 7. last_seen_frame
        self.assertEqual(track.last_seen_frame, 0)
        # 8. source_assignment
        self.assertEqual(track.source_assignment, 1)
        # 9. mask_state
        self.assertIn("mask", track.mask_state)
        self.assertIn("stability", track.mask_state)


class TestEightEdgeCases(unittest.TestCase):
    """Validates the 8 core edge cases specified in Stage 9."""

    def setUp(self):
        self.tracker = TemporalStateMachineTracker(max_lost=25, max_coast=10, reid_age=50)

    def test_case_1_temporary_detector_dropout(self):
        """Temporary detector dropout: track enters COASTING, synthesizes Face, and recovers."""
        # Frames 0-2: Established track
        for f in range(3):
            self.tracker.update([_mock_face(100.0 + f * 5.0, identity=0)], frame_index=f)
        track = self.tracker.tracks[0]
        self.assertEqual(track.state, TrackState.STABLE)

        # Frames 3-4: Detector dropout (hand crosses face, no detections returned)
        out3 = self.tracker.update([], frame_index=3)
        self.assertEqual(track.state, TrackState.COASTING)
        self.assertEqual(len(out3), 1, "Coasted synthetic face must be returned during dropout")
        self.assertTrue(out3[0].get("_coasted"))
        self.assertEqual(out3[0].get("_track_id"), 0)

        out4 = self.tracker.update([], frame_index=4)
        self.assertEqual(track.coasted_run, 2)

        # Frame 5: Detector recovers
        out5 = self.tracker.update([_mock_face(125.0, identity=0)], frame_index=5)
        self.assertEqual(track.state, TrackState.STABLE)
        self.assertEqual(track.misses, 0)
        self.assertEqual(out5[0].get("_track_id"), 0)

    def test_case_2_face_crossing_another_face(self):
        """Two faces cross with overlapping bounding boxes: embeddings freeze, zero identity switch."""
        # Person 0 moves Left-to-Right: 80 -> 240
        # Person 1 moves Right-to-Left: 240 -> 80
        a_x = np.linspace(80.0, 240.0, 9)
        b_x = np.linspace(240.0, 80.0, 9)

        assignments = {0: [], 1: []}

        for f, (ax, bx) in enumerate(zip(a_x, b_x)):
            f_a = _mock_face(ax, identity=0)
            f_b = _mock_face(bx, identity=1)
            out = self.tracker.update([f_a, f_b], frame_index=f)

            # Map detection back to ground truth identity
            for face in out:
                tid = face.get("_track_id")
                emb = face.get("embedding")
                if emb[0] == 1.0:
                    assignments[0].append(tid)
                elif emb[1] == 1.0:
                    assignments[1].append(tid)

        # Both persons must maintain their exact track_id without switching
        self.assertEqual(set(assignments[0]), {0}, "Person 0 suffered identity switch!")
        self.assertEqual(set(assignments[1]), {1}, "Person 1 suffered identity switch!")
        self.assertGreater(self.tracker.stats["crossings_detected"], 0)

    def test_case_3_face_count_changes(self):
        """Dynamic actor entry and exit cleanly manages track count."""
        # Frame 0: 1 face
        self.tracker.update([_mock_face(100.0, identity=0)], frame_index=0)
        self.assertEqual(len(self.tracker.tracks), 1)

        # Frame 1: 2nd face enters
        self.tracker.update([_mock_face(102.0, identity=0), _mock_face(400.0, identity=1)], frame_index=1)
        self.assertEqual(len(self.tracker.tracks), 2)
        self.assertIn(0, self.tracker.tracks)
        self.assertIn(1, self.tracker.tracks)

        # Frames 2-30: 2nd face exits scene (misses exceed max_lost)
        for f in range(2, 35):
            self.tracker.update([_mock_face(100.0 + f, identity=0)], frame_index=f)

        # Track 1 must be retired, Track 0 remains active
        self.assertEqual(len(self.tracker.tracks), 1)
        self.assertIn(0, self.tracker.tracks)

    def test_case_4_occlusion_handling(self):
        """Severe occlusion freezes embedding and prevents identity corruption."""
        for f in range(3):
            self.tracker.update([_mock_face(100.0, identity=0)], frame_index=f)
        track = self.tracker.tracks[0]
        orig_emb = track.embedding.copy()

        # Occlusion frame with contaminated/corrupted embedding
        bad_face = _mock_face(100.0, identity=0, score=0.45)
        bad_face["embedding"] = _mock_emb(99)  # completely foreign embedding
        track.state = TrackState.OCCLUDED
        track.update_embedding(bad_face["embedding"])

        # Embedding must NOT have updated during occlusion
        np.testing.assert_array_equal(track.embedding, orig_emb)

    def test_case_5_profile_transition(self):
        """Head yaw > 55 deg enters PROFILE_TURNING state without dropping track."""
        for f, yaw in enumerate((0.0, 25.0, 50.0, 68.0, 75.0)):
            self.tracker.update([_mock_face(150.0, yaw=yaw, identity=0)], frame_index=f)
        track = self.tracker.tracks[0]
        self.assertEqual(track.track_id, 0)
        self.assertEqual(track.state, TrackState.PROFILE_TURNING)
        self.assertGreater(self.tracker.stats["profile_turns_handled"], 0)

    def test_case_6_rapid_movement(self):
        """Whip pan velocity > 15 px/frame releases lag and adapts Kalman velocity."""
        # Stationary initialization
        self.tracker.update([_mock_face(50.0)], frame_index=0)
        self.tracker.update([_mock_face(52.0)], frame_index=1)
        self.tracker.update([_mock_face(54.0)], frame_index=2)

        # Sudden rapid motion jump
        self.tracker.update([_mock_face(180.0)], frame_index=3)
        track = self.tracker.tracks[0]
        # Kalman velocity must capture high speed (> 20 px/frame)
        self.assertGreater(float(track.velocity[0]), 20.0)

    def test_case_7_target_face_disappearing(self):
        """Face leaving scene moves to Re-ID memory archive rather than leaking memory."""
        for f in range(3):
            self.tracker.update([_mock_face(100.0, identity=0)], frame_index=f)
        self.assertEqual(len(self.tracker.tracks), 1)

        # Run past max_lost (30 frames) with no detection
        for f in range(3, 40):
            self.tracker.update([], frame_index=f)

        # Track removed from active tracks and placed into reid_archive
        self.assertEqual(len(self.tracker.tracks), 0)
        self.assertIn(0, self.tracker.reid_archive)
        self.assertEqual(self.tracker.reid_archive[0]["source_assignment"], 0)

    def test_case_8_target_face_returning(self):
        """Actor walks off-screen at frame 5 and returns at frame 25: recovers original ID."""
        for f in range(4):
            self.tracker.update([_mock_face(100.0, identity=0)], frame_index=f)
        self.assertEqual(self.tracker.tracks[0].track_id, 0)

        # Actor disappears for 30 frames
        for f in range(4, 38):
            self.tracker.update([], frame_index=f)
        self.assertEqual(len(self.tracker.tracks), 0)
        self.assertIn(0, self.tracker.reid_archive)

        # Actor returns at frame 38 at different position (x=450)
        out38 = self.tracker.update([_mock_face(450.0, identity=0)], frame_index=38)
        self.assertEqual(len(out38), 1)
        # Must be assigned original track_id 0 (NOT new track_id 1)!
        self.assertEqual(out38[0].get("_track_id"), 0)
        self.assertEqual(self.tracker.tracks[0].state, TrackState.RECOVERED)
        self.assertEqual(self.tracker.stats["returning_faces_recovered"], 1)


class TestConfidenceAwareInterpolation(unittest.TestCase):
    """Validates discontinuity rejection and Cubic Hermite trajectory smoothing."""

    def test_shot_cut_discontinuity_refusal(self):
        """Interpolation across a scene cut boundary must be refused."""
        f_a = _mock_face(100.0)
        f_b = _mock_face(110.0)
        cuts = {5}  # Cut at frame 5

        is_disc, reason = ConfidenceAwareInterpolator.check_discontinuity(
            f_a, f_b, span_frames=4, cuts=cuts, frame_lo=3, frame_hi=7
        )
        self.assertTrue(is_disc)
        self.assertIn("shot_cut", reason)

    def test_excessive_velocity_discontinuity_refusal(self):
        """Interpolation across teleportation / impossible travel speed is refused."""
        f_a = _mock_face(50.0)
        f_b = _mock_face(950.0)  # Teleport across 1080p frame in 2 frames

        is_disc, reason = ConfidenceAwareInterpolator.check_discontinuity(
            f_a, f_b, span_frames=2
        )
        self.assertTrue(is_disc)
        self.assertIn("excessive_travel", reason)

    def test_identity_drift_discontinuity_refusal(self):
        """Interpolation between two different people's embeddings is refused."""
        f_a = _mock_face(100.0, identity=0)
        f_b = _mock_face(110.0, identity=1)  # Orthogonal embedding

        is_disc, reason = ConfidenceAwareInterpolator.check_discontinuity(
            f_a, f_b, span_frames=3
        )
        self.assertTrue(is_disc)
        self.assertIn("identity_drift", reason)

    def test_cubic_hermite_endpoint_velocity_matching(self):
        """Cubic Hermite interpolation produces continuous velocities at endpoints."""
        f_a = _mock_face(100.0)
        f_b = _mock_face(200.0)
        emb = _mock_emb(0)
        vel_a = np.array([10.0, 0.0, 0.0, 0.0], dtype=np.float32)
        vel_b = np.array([10.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Midpoint fraction t=0.5
        mid = ConfidenceAwareInterpolator.interpolate_face(
            f_a, f_b, fraction=0.5, track_embedding=emb, vel_a=vel_a, vel_b=vel_b
        )
        mid_x = (mid["bbox"][0] + mid["bbox"][2]) * 0.5
        # For equal endpoint velocities, midpoint should be exactly 150.0
        self.assertAlmostEqual(float(mid_x), 200.0, delta=1.0)
        self.assertTrue(mid["_interpolated"])


class TestTemporalQualityMetrics(unittest.TestCase):
    """Validates quantitative temporal quality measurement functions."""

    def test_landmark_jitter_variance(self):
        """Noisy sequence has significantly higher jitter variance than smoothed sequence."""
        t = np.linspace(0, 10, 50)
        smooth_kps = np.stack([100.0 + 20.0 * np.sin(t), 100.0 + 10.0 * np.cos(t)], axis=-1)[:, None, :]  # (50, 1, 2)

        rng = np.random.default_rng(42)
        noisy_kps = smooth_kps + rng.normal(0.0, 2.5, size=smooth_kps.shape).astype(np.float32)

        var_smooth = TemporalQualityMetrics.compute_landmark_jitter(smooth_kps)
        var_noisy = TemporalQualityMetrics.compute_landmark_jitter(noisy_kps)

        self.assertLess(var_smooth, 0.5)
        self.assertGreater(var_noisy, 5.0)

    def test_identity_stability_score(self):
        """100% stability when zero identity flips occur; drops when flips happen."""
        stable_seq = [0, 0, 0, 0, 0, 0, 0, 0]
        unstable_seq = [0, 1, 0, 1, 0, 1, 0, 1]

        self.assertEqual(TemporalQualityMetrics.compute_identity_stability(stable_seq), 100.0)
        self.assertLess(TemporalQualityMetrics.compute_identity_stability(unstable_seq), 10.0)

    def test_mask_popping_rate(self):
        """Detects sudden drops in mask IoU."""
        m_base = np.ones((64, 64), dtype=np.float32)
        m_popped = np.zeros((64, 64), dtype=np.float32)

        stable_masks = [m_base, m_base, m_base]
        popping_masks = [m_base, m_popped, m_base]

        self.assertEqual(TemporalQualityMetrics.compute_mask_popping_rate(stable_masks), 0)
        self.assertEqual(TemporalQualityMetrics.compute_mask_popping_rate(popping_masks), 2)

    def test_motion_fidelity_correlation(self):
        """Calculates Pearson correlation between true motion and tracked motion."""
        true_motion = np.linspace(0, 100, 30)[:, None]
        lagged_motion = true_motion + 2.0

        corr = TemporalQualityMetrics.compute_motion_fidelity(true_motion, lagged_motion)
        self.assertGreater(corr, 0.99)


if __name__ == "__main__":
    unittest.main()
