"""Regression contracts for the opt-in Phase 6 temporal identity layer."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.temporal_identity import (  # noqa: E402
    TemporalIdentityStabilizer,
)


class TemporalIdentityTest(unittest.TestCase):
    def setUp(self):
        self.layer = TemporalIdentityStabilizer(
            enabled=True, switch_frames=3, transition_frames=4,
            geometry_alpha=0.25, output_strength=0.6, mask_strength=0.5)
        self.landmarks = np.asarray([
            [10, 10], [20, 10], [15, 15], [12, 20], [18, 20],
        ], dtype=np.float32)

    def test_alternating_bank_candidates_do_not_switch(self):
        self.layer.update_geometry(4, 0, self.landmarks,
                                   target_embedding=np.ones(4), confidence=0.9,
                                   source_identity=0,
                                   identity_embedding=np.asarray([0, 0, 1, 0]))
        chosen = []
        for frame, candidate in enumerate((0, 1, 0, 1, 0, 1, 1, 1), 1):
            chosen.append(self.layer.propose_source(4, candidate)[0])
        self.assertEqual(chosen[:6], [0, 0, 0, 0, 0, 0])
        self.assertEqual(chosen[-1:], [1])

    def test_alternating_identity_candidates_do_not_switch(self):
        stabilizer = TemporalIdentityStabilizer(
            enabled=True, switch_frames=3, transition_frames=2)
        for candidate in (0, 1, 0, 1, 0):
            self.assertEqual(stabilizer.propose_identity(8, candidate), 0)
        self.assertEqual(stabilizer.propose_identity(8, 1), 0)
        self.assertEqual(stabilizer.propose_identity(8, 1), 0)
        self.assertEqual(stabilizer.propose_identity(8, 1), 1)

    def test_major_pose_transition_commits_and_crossfades(self):
        state = self.layer.update_geometry(2, 0, self.landmarks,
                                           target_embedding=np.ones(4),
                                           confidence=0.9, source_identity=0)
        self.layer.update_pose(2, {"yaw": 0, "pitch": 0, "roll": 0,
                                   "confidence": 0.9})
        self.layer.propose_source(2, 0)
        moved = self.layer.update_geometry(
            2, 1, self.landmarks + np.asarray([30, 0], dtype=np.float32),
            target_embedding=np.ones(4), confidence=0.9, source_identity=0)
        moved = self.layer.update_pose(2, {"yaw": 50, "pitch": 0, "roll": 0,
                                           "confidence": 0.9})
        self.assertTrue(moved.major_pose_transition)
        selected, alpha = self.layer.propose_source(2, 1)
        self.assertEqual(selected, 1)
        self.assertAlmostEqual(alpha, 0.25)
        self.assertEqual(self.layer.propose_source(2, 1)[0], 1)

    def test_geometry_and_embedding_are_confidence_weighted(self):
        state = self.layer.update_geometry(
            1, 0, self.landmarks, target_embedding=np.asarray([1, 0, 0, 0]),
            confidence=1.0, source_identity=0)
        first = state.landmarks.copy()
        state = self.layer.update_geometry(
            1, 1, self.landmarks + 1.0,
            target_embedding=np.asarray([0, 1, 0, 0]), confidence=0.1,
            source_identity=0)
        self.assertTrue(np.all(state.landmarks > first))
        self.assertTrue(np.allclose(np.linalg.norm(state.target_embedding), 1.0))
        self.assertGreater(state.target_embedding[0], state.target_embedding[1])

    def test_output_blends_low_frequency_but_keeps_current_texture(self):
        first = np.full((64, 64, 3), 100, dtype=np.uint8)
        self.layer.blend_output(3, first, confidence=1.0)
        checker = np.indices((64, 64)).sum(axis=0) % 2
        current = np.full((64, 64, 3), 160, dtype=np.uint8)
        current[checker == 1] = 220
        result = self.layer.blend_output(3, current, confidence=0.2)
        self.assertGreater(float(result.mean()), 100.0)
        self.assertGreater(float(result.max() - result.min()), 20.0)
        self.assertEqual(self.layer.snapshot(3)["previous_output"].shape, (256, 256, 3))

    def test_mask_reveals_new_occluder_without_full_frame_blur(self):
        previous = np.zeros((8, 8), dtype=np.float32)
        current = np.ones((8, 8), dtype=np.float32)
        self.layer.stabilize_mask(5, previous, confidence=1.0)
        result = self.layer.stabilize_mask(5, current, confidence=0.2)
        self.assertTrue(np.all(result > 0.0))
        self.assertTrue(np.all(result < 1.0))

    def test_lowpass_identity_keeps_current_detail_and_is_bounded(self):
        exact = TemporalIdentityStabilizer(
            enabled=True, output_strength=0.6, lowpass_size=0)
        reduced = TemporalIdentityStabilizer(
            enabled=True, output_strength=0.6, lowpass_size=64)
        rng = np.random.default_rng(14)
        first = rng.integers(20, 220, (256, 256, 3), dtype=np.uint8)
        current = rng.integers(20, 220, (256, 256, 3), dtype=np.uint8)
        exact.blend_output(3, first, confidence=0.2)
        reduced.blend_output(3, first, confidence=0.2)
        exact_result = exact.blend_output(3, current, confidence=0.2,
                                           motion=0.05)
        reduced_result = reduced.blend_output(3, current, confidence=0.2,
                                              motion=0.05)
        self.assertEqual(reduced_result.shape, current.shape)
        self.assertEqual(reduced_result.dtype, np.uint8)
        self.assertTrue(np.all(np.isfinite(reduced_result)))
        # This is a low-frequency approximation, not a byte-equivalence claim;
        # bound the change well below a visible full-frame replacement.
        mae = float(np.mean(np.abs(
            reduced_result.astype(np.float32) - exact_result.astype(np.float32))))
        self.assertLess(mae, 3.0)
        self.assertGreater(float(np.std(reduced_result)), 20.0)

    def test_zero_lowpass_size_retains_reference_output(self):
        rng = np.random.default_rng(15)
        first = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
        current = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
        reference = TemporalIdentityStabilizer(enabled=True, lowpass_size=0)
        explicit = TemporalIdentityStabilizer(enabled=True, lowpass_size=-1)
        reference.blend_output(9, first, confidence=0.8)
        explicit.blend_output(9, first, confidence=0.8)
        expected = reference.blend_output(9, current, confidence=0.8)
        actual = explicit.blend_output(9, current, confidence=0.8)
        self.assertTrue(np.array_equal(expected, actual))

    def test_lowpass_size_is_read_from_environment(self):
        previous = os.environ.get("ROOP_TEMPORAL_IDENTITY_LOWPASS_SIZE")
        try:
            os.environ["ROOP_TEMPORAL_IDENTITY_LOWPASS_SIZE"] = "96"
            layer = TemporalIdentityStabilizer.from_env()
            self.assertEqual(layer.lowpass_size, 96)
        finally:
            if previous is None:
                os.environ.pop("ROOP_TEMPORAL_IDENTITY_LOWPASS_SIZE", None)
            else:
                os.environ["ROOP_TEMPORAL_IDENTITY_LOWPASS_SIZE"] = previous

    def test_state_contains_requested_temporal_fields(self):
        self.layer.update_geometry(8, 0, self.landmarks,
                                   target_embedding=np.ones(4), confidence=0.8,
                                   source_identity=2,
                                   identity_embedding=np.asarray([0, 1, 0, 0]))
        snap = self.layer.snapshot(8)
        for key in ("source_identity", "selected_source_index",
                    "identity_embedding", "target_embedding", "pose",
                    "landmarks", "alignment_transform", "swap_confidence",
                    "output_face_confidence", "previous_output",
                    "previous_mask", "previous_lighting"):
            self.assertIn(key, snap)
        self.assertEqual(snap["source_identity"], 2)
        self.assertTrue(np.allclose(snap["identity_embedding"], [0, 1, 0, 0]))

    def test_disabled_layer_is_a_noop(self):
        layer = TemporalIdentityStabilizer(enabled=False)
        crop = np.full((8, 8, 3), 90, dtype=np.uint8)
        self.assertIs(layer.blend_output(1, crop), crop)
        self.assertEqual(layer.propose_source(1, 2), (2, 1.0))

    def test_runtime_hooks_are_present_without_changing_default_switches(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "roop", "ProcessMgr.py"),
                  encoding="utf-8") as fh:
            process_mgr = fh.read()
        with open(os.path.join(root, "roop", "procmgr_tracking.py"),
                  encoding="utf-8") as fh:
            tracking = fh.read()
        self.assertIn("TemporalIdentityStabilizer.from_env()", process_mgr)
        self.assertIn("blend_output", process_mgr)
        self.assertIn("update_alignment", process_mgr)
        self.assertIn("update_geometry", tracking)
        self.assertIn("propose_source", tracking)
        self.assertIn("propose_identity", tracking)
        self.assertIn("ordered output history", process_mgr)
        with open(os.path.join(root, "roop", "temporal_identity.py"),
                  encoding="utf-8") as fh:
            temporal_identity = fh.read()
        self.assertIn('enabled=_bool("ROOP_TEMPORAL_IDENTITY", False)',
                      temporal_identity)


if __name__ == "__main__":
    unittest.main()
