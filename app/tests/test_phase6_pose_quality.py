"""Dependency-light contracts for the Phase 6 real-photo evaluation tool."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.phase6_pose_quality import coverage_report, summarize_selection  # noqa: E402


class Phase6PoseQualityTest(unittest.TestCase):
    def test_coverage_does_not_claim_unrepresented_axes(self):
        report = coverage_report({"sources": [
            {"geometry": {"yaw": -90, "pitch": 0, "roll": 0}},
            {"geometry": {"yaw": 0, "pitch": 0, "roll": 0}},
            {"geometry": {"yaw": 90, "pitch": 0, "roll": 0}},
        ]})
        self.assertTrue(report["has_profile"])
        self.assertFalse(report["has_pitch"])
        self.assertFalse(report["has_inversion"])

    def test_selection_summary_uses_pose_match_tolerance(self):
        summary = summarize_selection([
            {"target_yaw": 0, "target_pitch": 0, "source_yaw": 10,
             "source_pitch": 5, "yaw_error": 10, "pitch_error": 5,
             "needs_3d": False},
            {"target_yaw": 60, "target_pitch": 0, "source_yaw": 30,
             "source_pitch": 0, "yaw_error": -30, "pitch_error": 0,
             "needs_3d": True},
        ])
        self.assertEqual(summary["valid"], 2)
        self.assertEqual(summary["match_rate"], 0.5)
        self.assertEqual(summary["needs_3d_rate"], 0.5)

    def test_empty_selection_is_explicitly_unmeasured(self):
        summary = summarize_selection([])
        self.assertIsNone(summary["match_rate"])
        self.assertIsNone(summary["needs_3d_rate"])

    def test_benchmark_identity_uses_v2_global_vector_when_present(self):
        source = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "angle_bench.py")
        with open(source, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("source_identity=None", text)
        self.assertIn("src_embed = source_identity", text)


if __name__ == "__main__":
    unittest.main()
