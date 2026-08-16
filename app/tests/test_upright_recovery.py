"""An inverted face must not be mistaken for an upright one.

The measured failure these guard (tests/frontal_roll_video.py builds the clip):
between roll ~140 and ~220 the detector does not lose the face. It reports
~0.98 confidence and returns a self-consistent set of 5 keypoints describing an
UPRIGHT face, with the two "mouth" points sitting on the forehead. Every
consumer of those keypoints then agrees the face needs no turning:

  * autorotate declines to turn it, and the swapper is handed a crop with the
    eyes where the mouth belongs;
  * `norm_crop` fits the crushed points to the arcface template, so the
    recognition embedding of an inverted face scores ~0.0 cosine against the
    SAME person upright — below the 0.128 two-different-people floor — and the
    face stops matching the selected target at all.

Worst error of each candidate axis over a full turn: 5 detector keypoints
172 deg, 2D-106 chin->forehead 179 deg, 3D-68 eye-mid->mouth-mid 5.4 deg. The
first two also fail in the same direction, so their agreement is not evidence.

Asserted as CONSEQUENCES — the action actually returned, the module list
actually requested — rather than against the convention the code happens to
use, because a test written against the convention agrees with the bug.
"""

import ast
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.face_util import (  # noqa: E402
    _axis_from_68, face_down_axis, face_roll_tilt, face_rotation_action)
from roop.orientation import roll_from_face, wrap180  # noqa: E402

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeFace:
    """Minimal stand-in: the axis helpers only ever getattr these three."""

    def __init__(self, kps=None, lm68=None, bbox=None):
        self.kps = kps
        self.landmark_3d_68 = lm68
        self.bbox = bbox if bbox is not None else np.array([0.0, 0.0, 100.0, 100.0])


def _rot(pts, deg, centre=(256.0, 256.0)):
    t = np.radians(deg)
    # +roll swings the chin toward image +x, matching roll_from_kps.
    m = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    return (np.asarray(pts, float) - centre) @ m.T + centre


def kps_at(roll_deg):
    base = np.array([[226.0, 231.0], [286.0, 231.0], [256.0, 261.0],
                     [238.0, 286.0], [274.0, 286.0]])
    return _rot(base, roll_deg)


def lm68_at(roll_deg):
    """68 landmarks whose eye centres and mouth corners sit where a real face's
    do; the points this code never reads are left at the face centre."""
    pts = np.full((68, 3), 256.0)
    flat = np.full((68, 2), 256.0)
    for i in range(36, 42):
        flat[i] = (226.0, 231.0)
    for i in range(42, 48):
        flat[i] = (286.0, 231.0)
    flat[48] = (238.0, 286.0)
    flat[54] = (274.0, 286.0)
    pts[:, :2] = _rot(flat, roll_deg)
    return pts


class AxisPreference(unittest.TestCase):
    def test_68_axis_recovers_the_roll_it_was_built_with(self):
        for roll in range(-180, 180, 15):
            face = FakeFace(lm68=lm68_at(roll))
            dx, dy = _axis_from_68(face)
            got = np.degrees(np.arctan2(dx, dy))
            self.assertLess(abs(wrap180(got - roll)), 1e-6, f"roll {roll}")

    def test_the_68_axis_wins_when_the_keypoints_claim_upright(self):
        # The measured failure: the head is at 180, the keypoints say ~+8.
        face = FakeFace(kps=kps_at(8.0), lm68=lm68_at(180.0))
        tilt = face_roll_tilt(face)
        self.assertGreater(abs(tilt), 170.0,
                           f"lying keypoints were believed over the landmarks (tilt {tilt})")

    def test_an_inverted_face_is_actually_turned_over(self):
        # The consequence that matters: a half turn, not None.
        face = FakeFace(kps=kps_at(8.0), lm68=lm68_at(180.0))
        self.assertEqual(face_rotation_action(face, (512, 512)), "rotate_180")

    def test_a_genuinely_upright_face_is_left_alone(self):
        face = FakeFace(kps=kps_at(0.0), lm68=lm68_at(0.0))
        self.assertIsNone(face_rotation_action(face, (512, 512)))

    def test_keypoints_are_the_fallback_not_the_preference(self):
        # No landmark model in the pipeline: the 5 keypoints must still work.
        face = FakeFace(kps=kps_at(95.0), lm68=None)
        self.assertAlmostEqual(face_roll_tilt(face), 95.0, places=4)
        self.assertEqual(face_rotation_action(face, (512, 512)), "rotate_clockwise")

    def test_malformed_landmarks_fall_through_rather_than_crash(self):
        for bad in (None, np.zeros((12, 3)), np.full((68, 3), np.nan)):
            face = FakeFace(kps=kps_at(95.0), lm68=bad)
            self.assertIsNone(_axis_from_68(face))
            self.assertAlmostEqual(face_roll_tilt(face), 95.0, places=4)

    def test_no_axis_at_all_is_none_not_a_guess(self):
        self.assertIsNone(face_down_axis(FakeFace(kps=None, lm68=None)))


class LatchAgrees(unittest.TestCase):
    """The track latch must read the same axis, or it resolves continuity over
    a signal that is 180 degrees wrong and coasts through good frames."""

    def test_roll_from_face_prefers_the_landmarks(self):
        face = FakeFace(kps=kps_at(8.0), lm68=lm68_at(180.0))
        self.assertGreater(abs(roll_from_face(face)), 170.0)

    def test_roll_from_face_falls_back_to_kps(self):
        self.assertAlmostEqual(roll_from_face(FakeFace(kps=kps_at(-60.0))),
                               -60.0, places=4)

    def test_roll_from_face_handles_a_dict_face(self):
        # The tracking pre-pass hands these round as insightface dict-faces.
        self.assertAlmostEqual(roll_from_face({"kps": kps_at(30.0)}), 30.0, places=4)


class ModelIsRequested(unittest.TestCase):
    """The axis above only exists if the 68-point model is in the analysis
    pipeline. It is skipped by default as an optimisation, so autorotate has to
    ask for it explicitly — without that the fix above is silently inert, which
    is exactly how this bug reached a user.
    """

    def test_autorotate_pulls_in_landmark_3d_68(self):
        with open(os.path.join(APP, "roop", "ProcessMgr.py"), encoding="utf-8") as fh:
            src = fh.read()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
            if "landmark_3d_68" not in body:
                continue
            if "autorotate_faces" in ast.dump(node.test):
                return
        self.fail("nothing inserts landmark_3d_68 on account of autorotate_faces; "
                  "the 68-point axis is unavailable in the default configuration "
                  "and inverted faces go back to being swapped upside down")


if __name__ == "__main__":
    unittest.main()
