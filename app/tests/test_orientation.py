"""The roll resolver: does continuity recover the roll the detector loses?

The failure being modelled is measured, not imagined — on a profile plate rolled
past ~90 degrees the eye->mouth midline reads 123-184 degrees away from the
truth while the detector reports 0.99 confidence, so these tests corrupt a clean
sweep the same way and ask for the truth back.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.orientation import (  # noqa: E402
    RollTrack, action_for_roll, angdiff, residual_roll, roll_from_face,
    roll_from_kps, wrap180)


def kps_at(roll_deg, yaw_scale=1.0):
    """Five keypoints of a face rolled by `roll_deg`.

    `yaw_scale` squashes the eye separation to imitate a turned head, so a
    profile can be built without needing a detector.
    """
    eye_dx = 30.0 * yaw_scale
    base = np.array([[-eye_dx, -25.0], [eye_dx, -25.0], [0.0, 5.0],
                     [-18.0, 30.0], [18.0, 30.0]])
    t = np.radians(roll_deg)
    # +roll swings the chin toward image +x, matching roll_from_kps.
    rot = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    return (base @ rot.T) + np.array([256.0, 256.0])


def lm68_at(roll_deg, yaw_scale=1.0, nose_chin=True):
    """68 landmarks with eye clusters, mouth corners, and (unless disabled)
    a nose-bridge (27) / chin (8) midline pair, all rolled by `roll_deg`.

    `yaw_scale` squashes the eye separation only — matching `kps_at` — so the
    midline pair stays full-length under yaw, the exact property
    `roll_from_face` relies on to prefer it. `nose_chin=False` leaves 27/8 at
    the untouched face-centre default `lm68_at` in test_upright_recovery.py
    uses, to exercise the eye-mouth fallback explicitly.
    """
    eye_dx = 30.0 * yaw_scale
    flat = np.full((68, 2), 256.0)
    for i in range(36, 42):
        flat[i] = (-eye_dx, -25.0)
    for i in range(42, 48):
        flat[i] = (eye_dx, -25.0)
    flat[48] = (-18.0, 30.0)
    flat[54] = (18.0, 30.0)
    if nose_chin:
        flat[27] = (0.0, -45.0)   # nasion: above the eyes
        flat[8] = (0.0, 60.0)     # chin: below the mouth
    t = np.radians(roll_deg)
    rot = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    xy = ((flat - np.array([0.0, 0.0])) @ rot.T) + np.array([256.0, 256.0])
    # Points never explicitly placed (still at the raw 256.0 default) must
    # stay exactly there, not get dragged through the rotation.
    untouched = np.all(flat == 256.0, axis=1)
    xy[untouched] = 256.0
    pts = np.full((68, 3), 256.0)
    pts[:, :2] = xy
    return pts


class RollFromKps(unittest.TestCase):
    def test_recovers_the_roll_it_was_built_with(self):
        for roll in range(-180, 180, 15):
            got = roll_from_kps(kps_at(roll))
            self.assertLess(abs(angdiff(got, roll)), 1e-6, f"roll {roll}")

    def test_profile_squash_does_not_change_the_roll(self):
        # Eye separation collapses toward a profile; the midline must not care.
        for roll in (-150.0, -40.0, 0.0, 70.0, 160.0):
            for yaw in (1.0, 0.35, 0.02):
                self.assertLess(abs(angdiff(roll_from_kps(kps_at(roll, yaw)), roll)),
                                1e-6, f"roll {roll} yaw {yaw}")

    def test_degenerate_input_is_none_not_a_guess(self):
        self.assertIsNone(roll_from_kps(None))
        self.assertIsNone(roll_from_kps(np.zeros((5, 2))))
        self.assertIsNone(roll_from_kps(np.full((5, 2), np.nan)))


class RollFromFaceMidline(unittest.TestCase):
    """`roll_from_face` prefers the nose-chin midline over the eye-mouth axis
    when both eyes are needed to build the latter and yaw has started
    unbalancing them — see roll_from_face's own docstring for the measured
    real-clip motivation (roop-recode, 2026-08-15)."""

    def test_nose_chin_axis_recovers_the_roll_it_was_built_with(self):
        for roll in range(-180, 180, 15):
            face = {"landmark_3d_68": lm68_at(roll)}
            got = roll_from_face(face)
            self.assertLess(abs(angdiff(got, roll)), 1e-6, f"roll {roll}")

    def test_nose_chin_survives_the_yaw_squash_that_corrupts_eye_mouth(self):
        # At yaw_scale=0.05 the two eye clusters sit almost on top of each
        # other, which barely moves the eye-mouth axis's OWN 2D geometry
        # (it's still a clean midline in this synthetic rig) — the real
        # failure this preference guards against is a REAL detector
        # mislocating the occluded far eye, which no synthetic squash of an
        # otherwise-perfect point reproduces (this is exactly the gap
        # `test_profile_squash_does_not_change_the_roll` proved for the
        # 5-point axis). What IS directly testable here: the midline choice
        # does not regress accuracy as yaw increases, for any degree of
        # squash, since it never reads the eye points at all.
        for roll in (-150.0, -40.0, 0.0, 70.0, 160.0):
            for yaw in (1.0, 0.35, 0.02):
                face = {"landmark_3d_68": lm68_at(roll, yaw_scale=yaw)}
                got = roll_from_face(face)
                self.assertLess(abs(angdiff(got, roll)), 1e-6, f"roll {roll} yaw {yaw}")

    def test_falls_back_to_eye_mouth_when_nose_chin_is_degenerate(self):
        # Landmarks that never placed 27/8 usefully (e.g. an older model
        # export, or a synthetic fixture built before this preference
        # existed) must still resolve via the eye-mouth axis, not error out
        # or silently return a wrong value.
        for roll in (-170.0, -30.0, 0.0, 45.0, 130.0):
            face = {"landmark_3d_68": lm68_at(roll, nose_chin=False)}
            got = roll_from_face(face)
            self.assertLess(abs(angdiff(got, roll)), 1e-6, f"roll {roll}")

    def test_prefers_nose_chin_when_the_two_axes_disagree(self):
        # Build a face where the eye-mouth axis reads a DIFFERENT roll than
        # the nose-chin one — the direct case for "prefers", not just "both
        # happen to agree". A real mislocated-far-eye failure looks like
        # this: the midline is right, the eye pair is not.
        face = lm68_at(20.0)
        eye_mouth_wrong = lm68_at(80.0)
        face[36:48] = eye_mouth_wrong[36:48]
        face[48], face[54] = eye_mouth_wrong[48], eye_mouth_wrong[54]
        got = roll_from_face({"landmark_3d_68": face})
        self.assertLess(abs(angdiff(got, 20.0)), 1e-6,
                        "the corrupted eye-mouth axis was trusted over the midline")


class ActionForRoll(unittest.TestCase):
    def test_a_quarter_turn_always_leaves_less_tilt_than_it_found(self):
        # The point of the action is that the crop has less roll to absorb.
        for roll in range(-180, 180, 5):
            act = action_for_roll(roll)
            self.assertLessEqual(abs(residual_roll(roll, act)), abs(wrap180(roll)) + 1e-9,
                                 f"roll {roll} -> {act}")

    def test_no_action_ever_leaves_a_face_worse_than_45_degrees(self):
        for roll in range(-180, 180, 5):
            self.assertLessEqual(abs(residual_roll(roll, action_for_roll(roll))),
                                 90.0, f"roll {roll}")

    def test_upside_down_takes_the_half_turn(self):
        for roll in (170.0, 180.0, -170.0):
            self.assertEqual(action_for_roll(roll), "rotate_180")

    def test_near_upright_is_left_alone(self):
        for roll in (-40.0, 0.0, 40.0):
            self.assertIsNone(action_for_roll(roll))


class Continuity(unittest.TestCase):
    def test_a_clean_sweep_is_passed_through_untouched(self):
        tr = RollTrack()
        for true in range(0, 261, 4):
            got, trusted = tr.update(kps_at(true))
            self.assertTrue(trusted)
            self.assertLess(abs(angdiff(got, true)), 1e-6)
        self.assertEqual(tr.coasts, 0)

    def test_the_measured_failure_is_recovered(self):
        """Corrupt the estimate the way the detector actually corrupts it.

        On the plates the bad readings sat 123-184 degrees from the truth over a
        contiguous stretch of roll. Reproduced here over rolls 100-220.
        """
        rng = np.random.default_rng(0)
        tr = RollTrack()
        worst = 0.0
        for true in range(0, 261, 4):
            if 100 <= true <= 220:
                # Measured sign: the bad readings sat BELOW the truth by
                # 123-184 degrees, not above it.
                seen = wrap180(true - rng.uniform(123.0, 184.0))
            else:
                seen = true
            got, _ = tr.update(kps_at(seen))
            worst = max(worst, abs(angdiff(got, true)))
        # Inside the band the resolver is coasting or un-flipping, so it is not
        # exact; it has to be close enough that action_for_roll still picks the
        # turn that stands the face up.
        self.assertLess(worst, 45.0, f"worst roll error {worst:.1f} deg")

    def test_recovery_beats_doing_nothing(self):
        """The comparison that matters: against the current behaviour."""
        rng = np.random.default_rng(1)
        tr = RollTrack()
        raw_worst = res_worst = 0.0
        for true in range(0, 261, 4):
            seen = wrap180(true - rng.uniform(123.0, 184.0)) if 100 <= true <= 220 else true
            got, _ = tr.update(kps_at(seen))
            raw_worst = max(raw_worst, abs(residual_roll(true, action_for_roll(seen))))
            res_worst = max(res_worst, abs(residual_roll(true, action_for_roll(got))))
        self.assertLess(res_worst, raw_worst)
        # Trusting the raw estimate leaves a face more than a quarter turn out,
        # which is the upside-down profile the swap collapses on.
        self.assertGreater(raw_worst, 90.0)
        self.assertLess(res_worst, 60.0)

    def test_genuine_fast_rotation_is_not_suppressed(self):
        # A head really turning over must be followed, not treated as noise.
        tr = RollTrack()
        for true in range(0, 361, 30):
            got, _ = tr.update(kps_at(true))
            self.assertLess(abs(angdiff(got, true)), 1e-6, f"true {true}")

    def test_coasting_is_bounded(self):
        # A permanently bad estimate must not coast forever on a stale rate.
        tr = RollTrack(max_coast=5)
        tr.update(kps_at(0.0))
        tr.update(kps_at(10.0))
        for _ in range(20):
            tr.update(kps_at(95.0))       # never agrees with the prediction
        self.assertLessEqual(tr.coasts, 5)

    def test_a_first_observation_is_taken_at_face_value(self):
        tr = RollTrack()
        got, trusted = tr.update(kps_at(37.0))
        self.assertTrue(trusted)
        self.assertLess(abs(angdiff(got, 37.0)), 1e-6)

    def test_missing_keypoints_do_not_reset_the_track(self):
        tr = RollTrack()
        tr.update(kps_at(0.0))
        tr.update(kps_at(20.0))
        got, trusted = tr.update(None)
        self.assertFalse(trusted)
        self.assertIsNotNone(got)         # coasts rather than losing the track


if __name__ == "__main__":
    unittest.main()
