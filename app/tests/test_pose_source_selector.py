"""Phase 5 tests for pose-aware source selection and safe geometry hints."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.face_3d_recon import pose_warp_plan  # noqa: E402
from roop.pose_source_selector import (  # noqa: E402
    PoseEstimate,
    annotate_face_pose,
    estimate_target_pose,
    select_pose_aware_source,
)


PROPS = {
    "face_width_height": 0.78,
    "eye_distance_face_width": 0.36,
    "eye_mouth_distance_face_height": 0.34,
    "mouth_width_face_width": 0.30,
}


def _entry(yaw, pitch=0.0, roll=0.0, quality=0.95, expression=None,
           lighting=None):
    return {
        "geometry": {
            "yaw": yaw, "pitch": pitch, "roll": roll,
            "face_scale": {"pixels": 180.0, "relative_height": 0.25},
            "facial_proportions": dict(PROPS),
        },
        "quality": {"score": quality},
        "identity": {"quality_confidence": quality},
        "expression": expression or {"descriptor": [0.35, 0.35, 0.10, 0.30]},
        "appearance": lighting or {
            "luminance": {"mean": 0.5}, "color_temperature": 1.0,
            "skin_color_bgr": {"mean": [120.0, 140.0, 160.0]},
        },
    }


def _target(yaw=0.0, pitch=0.0, roll=0.0, confidence=0.95,
            expression=None, appearance=None):
    return PoseEstimate(
        yaw=yaw, pitch=pitch, roll=roll, face_scale=180.0,
        relative_scale=0.25, proportions=dict(PROPS),
        expression=expression or {"descriptor": [0.35, 0.35, 0.10, 0.30]},
        confidence=confidence, off_axis=float(np.hypot(yaw, pitch)),
        perspective_risk=0.05, inverted=abs(roll) >= 135.0, available=True,
    )


class PoseSourceSelectionTest(unittest.TestCase):

    def test_prefers_matching_lateral_pose_over_frontal(self):
        metadata = {"sources": [_entry(0), _entry(60), _entry(-60)]}
        result = select_pose_aware_source(metadata, _target(yaw=58))
        self.assertEqual(result.index, 1)
        self.assertEqual(result.reason, "source_pose_sufficient")
        self.assertFalse(result.needs_3d)

    def test_between_bank_poses_requests_a_conservative_3d_bridge(self):
        result = select_pose_aware_source(
            {"sources": [_entry(0), _entry(60)]}, _target(yaw=30))
        self.assertTrue(result.needs_3d)
        self.assertEqual(result.reason, "between_source_poses")

    def test_profile_without_profile_source_requests_3d(self):
        result = select_pose_aware_source(
            {"sources": [_entry(0), _entry(30)]}, _target(yaw=75))
        self.assertTrue(result.needs_3d)
        self.assertEqual(result.reason, "source_pose_gap")

    def test_inverted_orientation_is_never_hidden_by_a_good_yaw_match(self):
        result = select_pose_aware_source(
            {"sources": [_entry(0, roll=0), _entry(0, roll=170)]},
            _target(roll=178))
        self.assertEqual(result.index, 1)
        self.assertTrue(result.needs_3d)
        self.assertEqual(result.reason, "inverted")

    def test_expression_and_lighting_break_a_pose_tie(self):
        open_expression = {"descriptor": [0.75, 0.75, 0.65, 0.30]}
        dark = {"luminance": {"mean": 0.18}, "color_temperature": 0.82,
                "skin_color_bgr": {"mean": [90.0, 100.0, 110.0]}}
        result = select_pose_aware_source(
            {"sources": [_entry(30, expression=open_expression),
                         _entry(30, lighting=dark)]},
            _target(yaw=30, expression=open_expression, appearance=dark))
        self.assertEqual(result.index, 0)

    def test_hysteresis_keeps_a_near_tie_from_flipping(self):
        result = select_pose_aware_source(
            {"sources": [_entry(0), _entry(60)]}, _target(yaw=30),
            previous_index=0, switch_margin=0.5)
        self.assertEqual(result.index, 0)
        self.assertFalse(result.switched)

    def test_low_confidence_pose_uses_quality_fallback_hint(self):
        result = select_pose_aware_source(
            {"sources": [_entry(0), _entry(60)]}, _target(yaw=60, confidence=0.2))
        self.assertTrue(result.needs_3d)
        self.assertEqual(result.reason, "low_pose_confidence")

    def test_yaw_grid_covers_frontal_to_profile_without_frontal_bias(self):
        metadata = {"sources": [_entry(yaw) for yaw in
                                 (-90, -75, -60, -45, -30, 0, 30, 45, 60, 75, 90)]}
        for yaw in (0, 30, 45, 60, 75, 89):
            with self.subTest(yaw=yaw):
                result = select_pose_aware_source(metadata, _target(yaw=yaw))
                selected_yaw = metadata["sources"][result.index]["geometry"]["yaw"]
                self.assertLessEqual(abs(selected_yaw - yaw), 15)

    def test_pitch_and_roll_are_used_when_yaw_is_equal(self):
        metadata = {"sources": [
            _entry(60, pitch=40, roll=25),
            _entry(60, pitch=-45, roll=-30),
        ]}
        result = select_pose_aware_source(
            metadata, _target(yaw=60, pitch=-38, roll=-24))
        self.assertEqual(result.index, 1)


class TargetPoseEstimateTest(unittest.TestCase):

    def test_estimate_records_scale_roll_and_inversion(self):
        face = {
            "bbox": np.asarray([100, 80, 300, 280], np.float32),
            "kps": np.asarray([[150, 145], [250, 145], [200, 190],
                               [165, 230], [235, 230]], np.float32),
            "det_score": 0.95,
            "landmark_confidence": 0.95,
            "roll_deg": 178.0,
        }
        pose = estimate_target_pose(face, frame_shape=(720, 1280, 3))
        self.assertGreater(pose.face_scale, 190.0)
        self.assertAlmostEqual(pose.relative_scale, 200.0 / 720.0, places=5)
        self.assertTrue(pose.inverted)
        self.assertGreater(pose.confidence, 0.7)

    def test_annotation_is_detached_and_reused(self):
        face = {"bbox": [0, 0, 100, 100], "kps": [[25, 30], [75, 30],
                [50, 50], [32, 70], [68, 70]], "det_score": 0.9}
        first = annotate_face_pose(face, frame_shape=(200, 200, 3))
        second = estimate_target_pose(face)
        self.assertEqual(first.as_dict(), second.as_dict())
        self.assertIsNot(first.as_dict(), face["_pose_v5"])


class PoseWarpPlanTest(unittest.TestCase):

    def test_frontal_to_lateral_does_not_mirror_identity_details(self):
        plan = pose_warp_plan(0, 0, 60, 0)
        self.assertFalse(plan["flip"])
        self.assertLessEqual(abs(plan["yaw_shear"]), 0.20)

    def test_opposite_strong_views_justify_mirroring(self):
        plan = pose_warp_plan(-60, 0, 60, 0)
        self.assertTrue(plan["flip"])
        self.assertAlmostEqual(plan["yaw_delta"], 0.0, places=6)

    def test_pitch_compensation_is_more_tightly_bounded_than_yaw(self):
        plan = pose_warp_plan(0, -60, 0, 60)
        self.assertLessEqual(abs(plan["pitch_shear"]), 0.08)
        self.assertLessEqual(abs(plan["pitch_shear"]), abs(plan["yaw_shear"]) + 0.08)

    def test_same_pose_is_a_noop_plan(self):
        plan = pose_warp_plan(45, 20, 45, 20)
        self.assertFalse(plan["flip"])
        self.assertEqual(plan["magnitude"], 0.0)


if __name__ == "__main__":
    unittest.main()
