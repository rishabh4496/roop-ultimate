"""kps_pose_ratios — the 5-keypoint yaw / pitch proxies.

These exist because the heuristic they replaced (nose asymmetry between the two
eyes) is NOT monotonic in yaw: it peaks around 30-40 deg and falls back to
exactly 0.0 at a true 90 deg profile, because a symmetric face at 90 deg
projects both eyes onto the same x, which puts the nose equidistant from both.
That silently classified every near-profile face as frontal.

test_monotonic_in_yaw is the regression guard for that class of bug: it asserts
the property (monotonicity over the whole range), not a magic number.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.face_util import kps_pose_ratios          # noqa: E402
from tests.facegeom import project_kps              # noqa: E402

YAWS = [0, 10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90]


class TestYawRatio(unittest.TestCase):
    def test_monotonic_in_yaw(self):
        """Strictly decreasing from frontal to profile — the property the old
        asymmetry heuristic violated."""
        ratios = [kps_pose_ratios(project_kps(y))[0] for y in YAWS]
        for (y0, r0), (y1, r1) in zip(zip(YAWS, ratios), zip(YAWS[1:], ratios[1:])):
            self.assertLess(r1, r0, f"yaw_ratio rose from {y0} to {y1} deg "
                                    f"({r0:.4f} -> {r1:.4f}); it must fall "
                                    f"monotonically all the way to 90 deg")

    def test_old_asymmetry_heuristic_really_was_blind(self):
        """Documents the original bug so nobody reintroduces the measure."""
        def asymmetry(kps):
            d_left = abs(kps[2][0] - kps[0][0])
            d_right = abs(kps[2][0] - kps[1][0])
            return abs(d_left - d_right) / (d_left + d_right + 1e-9)

        self.assertGreater(asymmetry(project_kps(30)), 0.25)   # fired mid-angle
        self.assertLess(asymmetry(project_kps(85)), 0.25)      # missed near-profile
        self.assertAlmostEqual(asymmetry(project_kps(90)), 0.0, places=6)

    def test_roll_invariant(self):
        """A tilted head must not change the yaw estimate — the ratio is built
        from distances only, so roll cancels."""
        for yaw in (0, 45, 90):
            values = [kps_pose_ratios(project_kps(yaw, 0, roll))[0]
                      for roll in (-40, -20, 0, 20, 40)]
            self.assertAlmostEqual(max(values) - min(values), 0.0, places=5,
                                   msg=f"yaw_ratio moved with roll at yaw={yaw}")

    def test_gate_separates_profile_from_mid_angle(self):
        """The 0.55 masking gate must fire for every profile pose and stay
        silent for every clearly mid-angle one, across pitch and roll."""
        for yaw in (55, 65, 75, 85, 90):
            for pitch in (-25, -10, 0, 10, 25):
                for roll in (-30, 0, 30):
                    r = kps_pose_ratios(project_kps(yaw, pitch, roll))[0]
                    self.assertLess(r, 0.55, f"missed profile yaw={yaw} "
                                             f"pitch={pitch} roll={roll}")
        for yaw in (0, 20, 40):
            for pitch in (-25, 0, 25):
                for roll in (-30, 0, 30):
                    r = kps_pose_ratios(project_kps(yaw, pitch, roll))[0]
                    self.assertGreaterEqual(r, 0.55, f"false positive at yaw={yaw} "
                                                     f"pitch={pitch} roll={roll}")


class TestPitchRatio(unittest.TestCase):
    def test_monotonic_in_pitch_at_low_yaw(self):
        """Rises as the head tilts back. Only asserted at low yaw — see
        test_pitch_ratio_degenerates_at_profile."""
        for yaw in (0, 20, 40):
            pitches = [-40, -25, -10, 0, 10, 25, 40]
            values = [kps_pose_ratios(project_kps(yaw, p))[1] for p in pitches]
            for (p0, v0), (p1, v1) in zip(zip(pitches, values),
                                          zip(pitches[1:], values[1:])):
                self.assertLess(v0, v1, f"pitch_ratio fell from {p0} to {p1} deg "
                                        f"at yaw={yaw}")

    def test_pitch_ratio_degenerates_at_profile(self):
        """At 90 deg the nose and the eye->mouth axis all lie in the sagittal
        plane, which is parallel to the image plane, so pitch barely moves the
        projection. Documented as a known limit: high yaw is covered by
        yaw_ratio, which is why this is acceptable rather than a bug."""
        span = max(kps_pose_ratios(project_kps(90, p))[1] for p in (-40, 0, 40)) \
            - min(kps_pose_ratios(project_kps(90, p))[1] for p in (-40, 0, 40))
        self.assertLess(span, 0.05, "pitch_ratio unexpectedly informative at 90 "
                                    "deg — if this now works, widen its use")

    def test_extreme_pitch_gate(self):
        """The 0.32 / 0.70 bounds must catch a steeply pitched frontal head —
        the case whose exact check was dead by default because landmark_3d_68
        is not loaded unless an optional pose feature is on."""
        for pitch in (-45, -35, 35, 45):
            r = kps_pose_ratios(project_kps(0, pitch))[1]
            self.assertFalse(0.32 < r < 0.70, f"missed extreme pitch {pitch} deg")
        for pitch in (-20, -10, 0, 10, 20):
            r = kps_pose_ratios(project_kps(0, pitch))[1]
            self.assertTrue(0.32 < r < 0.70, f"false positive at pitch {pitch} deg")

    def test_roll_invariant(self):
        for roll in (-40, 0, 40):
            self.assertAlmostEqual(
                kps_pose_ratios(project_kps(30, 20, roll))[1],
                kps_pose_ratios(project_kps(30, 20, 0))[1], places=5)


class TestDegenerateInput(unittest.TestCase):
    """Must return (None, None) rather than raising or emitting NaN — callers
    treat None as 'unknown' and fall through to their other checks."""

    def test_bad_inputs(self):
        for name, bad in (
            ("none", None),
            ("empty", np.zeros((0, 2), np.float32)),
            ("wrong count", np.zeros((3, 2), np.float32)),
            ("all zero", np.zeros((5, 2), np.float32)),
            ("all identical", np.full((5, 2), 7.0, np.float32)),
        ):
            with self.subTest(name):
                self.assertEqual(kps_pose_ratios(bad), (None, None))

    def test_collapsed_eye_to_mouth_axis(self):
        """Eyes and mouth on the same point — the divide-by-zero case."""
        kps = np.array([[10, 10], [20, 10], [15, 10], [10, 10], [20, 10]], np.float32)
        self.assertEqual(kps_pose_ratios(kps), (None, None))

    def test_never_returns_nan(self):
        for yaw in YAWS:
            for pitch in (-40, 0, 40):
                for value in kps_pose_ratios(project_kps(yaw, pitch)):
                    self.assertTrue(np.isfinite(value))


if __name__ == "__main__":
    unittest.main()
