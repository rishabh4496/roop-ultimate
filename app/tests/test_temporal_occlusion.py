"""Regression contracts for Phase 7 temporal occlusion state."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.temporal_occlusion import (  # noqa: E402
    TemporalOcclusionEngine,
    build_face_support,
)


class TemporalOcclusionTest(unittest.TestCase):
    def setUp(self):
        self.engine = TemporalOcclusionEngine(
            enabled=True, stable_frames=3, refresh_frames=5,
            event_threshold=0.12, leave_alpha=0.35, enter_alpha=0.90)
        self.support = np.ones((32, 32), dtype=np.float32)
        self.clear = np.zeros_like(self.support)
        self.occluded = np.ones_like(self.support)

    def _observe(self, index, mask, **kwargs):
        decision = self.engine.prepare(
            7, index, self.support, observation=np.zeros((32, 32, 3), np.uint8),
            confidence=1.0, **kwargs)
        self.assertEqual(decision.mode, "analyze")
        return self.engine.observe(7, index, self.support, mask,
                                   confidence=1.0, **kwargs)

    def test_state_contains_separate_requested_masks_and_confidence(self):
        self._observe(0, self.clear)
        state = self.engine.snapshot(7)
        for key in ("face_mask", "visible_face_mask", "occlusion_mask",
                    "previous_mask", "predicted_mask", "confidence"):
            self.assertIn(key, state)
        self.assertEqual(state["track_id"], 7)
        self.assertEqual(state["face_mask"].shape, self.support.shape)

    def test_entering_event_preserves_object_pixels(self):
        self._observe(0, self.clear)
        result = self._observe(1, self.occluded)
        self.assertEqual(self.engine.snapshot(7)["event"], "entering")
        self.assertTrue(np.all(result > 0.8))

    def test_stable_occlusion_uses_cheap_propagation(self):
        self._observe(0, self.clear)
        for index in range(1, 5):
            decision = self.engine.prepare(
                7, index, self.support,
                observation=np.zeros((32, 32, 3), np.uint8), confidence=1.0)
            self.assertEqual(decision.mode, "analyze")
            self.engine.observe(7, index, self.support, self.occluded,
                                confidence=1.0)
        decision = self.engine.prepare(
            7, 5, self.support,
            observation=np.zeros((32, 32, 3), np.uint8), confidence=1.0)
        self.assertEqual(decision.mode, "propagate")
        self.assertEqual(decision.reason, "occlusion_stable")
        propagated = self.engine.propagate(7, 5, decision, confidence=1.0)
        self.assertTrue(np.allclose(propagated, self.occluded))
        self.assertEqual(self.engine.snapshot(7)["analysis_mode"],
                         "cheap_propagation")

    def test_motion_or_appearance_change_reenters_analysis(self):
        self._observe(0, self.clear)
        for index in range(1, 5):
            self._observe(index, self.occluded)
        decision = self.engine.prepare(
            7, 5, self.support,
            observation=np.full((32, 32, 3), 255, np.uint8),
            confidence=1.0)
        self.assertEqual(decision.mode, "analyze")
        self.assertEqual(decision.reason, "occlusion_event_reanalysis")

    def test_leaving_event_fades_restore_instead_of_popping(self):
        self._observe(0, self.occluded)
        leaving = self._observe(1, self.clear)
        next_frame = self._observe(2, self.clear)
        self.assertGreater(float(leaving.mean()), 0.0)
        self.assertGreater(float(next_frame.mean()), 0.0)
        self.assertLess(float(next_frame.mean()), float(leaving.mean()))

    def test_tracks_are_isolated(self):
        self.engine.prepare(1, 0, self.support)
        self.engine.observe(1, 0, self.support, self.occluded, confidence=1.0)
        other_support = np.zeros_like(self.support)
        other_support[8:24, 8:24] = 1.0
        self.engine.prepare(2, 0, other_support)
        self.engine.observe(2, 0, other_support, self.clear, confidence=0.4)
        first = self.engine.snapshot(1)
        second = self.engine.snapshot(2)
        self.assertTrue(np.all(first["occlusion_mask"] > 0.9))
        self.assertAlmostEqual(float(second["occlusion_mask"].sum()), 0.0)
        self.assertNotEqual(first["track_id"], second["track_id"])

    def test_face_support_is_crop_local_and_soft_at_boundary(self):
        points = np.asarray([[8, 8], [24, 8], [24, 24], [8, 24]], np.float32)
        support = build_face_support(
            landmarks=points, matrix=np.asarray([[1, 0, 0], [0, 1, 0]], np.float32),
            shape=(32, 32))
        self.assertIsNotNone(support)
        self.assertGreater(float(support[16, 16]), 0.9)
        self.assertLess(float(support[1, 1]), 0.1)

    def test_disabled_engine_is_a_noop(self):
        engine = TemporalOcclusionEngine(enabled=False)
        decision = engine.prepare(1, 0, self.support)
        self.assertEqual(decision.mode, "disabled")
        self.assertIs(engine.observe(1, 0, self.support, self.occluded),
                      self.occluded)

    def test_runtime_hooks_are_present_without_changing_default_switch(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "roop", "ProcessMgr.py"), encoding="utf-8") as fh:
            process_mgr = fh.read()
        with open(os.path.join(root, "roop", "procmgr_masking.py"), encoding="utf-8") as fh:
            masking = fh.read()
        with open(os.path.join(root, "roop", "temporal_occlusion.py"), encoding="utf-8") as fh:
            temporal_occlusion = fh.read()
        self.assertIn("TemporalOcclusionEngine.from_env()", process_mgr)
        self.assertIn("ROOP_TEMPORAL_OCCLUSION", temporal_occlusion)
        self.assertIn("_region_owner_in_crop", masking)
        self.assertIn("build_face_support", masking)


if __name__ == "__main__":
    unittest.main()
