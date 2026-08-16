"""How wrong the 5-point pose solve is on a head that is not the reference head.

This is the measurement that decides whether the three angle layers
(`yaw_align='pose'`, the hidden-surface trim, the off-axis fade) can be on by
default. They share one pose number, which is a virtue right up until that number
is wrong, at which point all three are wrong together and in the same direction.

`solve_pose_5pt` fits ONE reference head by weak perspective. With five points
there is no room to also solve for the head's shape, so any difference between
that reference and the person in frame comes out as pose error. The dominant
direction is nose protrusion, because in five keypoints the nose tip is the only
point off the eye/mouth plane — it is the entire depth cue, so "how far does the
nose stand out" and "how far is the head turned" are the same measurement.

The numbers here are not a bug report against the solver; five coplanar-ish points
cannot do better in one frame. They are the operating limit, recorded so that
anything keying on absolute pose is designed knowing it, and so that a future
per-identity calibration has a baseline to beat.

Deliberately NOT asserted as tight bounds on the error — that would be a test that
fails when someone improves the solver. What is asserted is the SHAPE of the
problem, which is what the design has to survive: the error is large, it is
systematic per person rather than random per frame, and it is big enough to move a
face across the engagement bands the angle layers gate on.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.facegeom import HEAD_5PT, rotation                       # noqa: E402
from roop.face_util import solve_pose_5pt, solve_pose_jaw_5pt


def project(head, yaw, pitch=0.0, roll=0.0, scale=200.0, cx=256.0, cy=256.0):
    q = np.asarray(head, dtype=np.float64) @ rotation(yaw, pitch, roll).T
    return np.column_stack([cx + scale * q[:, 0], cy - scale * q[:, 1]])


def nose(factor):
    """The reference head with the nose tip `factor` times as far forward."""
    h = np.array(HEAD_5PT, dtype=np.float64, copy=True)
    h[2, 2] *= factor
    return h


def eyes(factor):
    h = np.array(HEAD_5PT, dtype=np.float64, copy=True)
    h[0, 0] *= factor
    h[1, 0] *= factor
    return h


def solved_yaw(head, yaw, pitch=0.0):
    out = solve_pose_5pt(project(head, yaw, pitch))
    return None if out is None else out[0]


class ReferenceHeadIsExact(unittest.TestCase):
    """The baseline the rest of the file is measured against.

    Without this the other tests could be passing because the solver is broken
    for everybody, which would be a different (and easier) problem.
    """

    def test_exact_on_the_head_it_was_built_from(self):
        for yaw in (0, 15, 30, 45, 60, 75, 88):
            for pitch in (0, -20, 20):
                with self.subTest(yaw=yaw, pitch=pitch):
                    got = solve_pose_5pt(project(HEAD_5PT, yaw, pitch))
                    self.assertAlmostEqual(got[0], yaw, delta=0.05)
                    self.assertAlmostEqual(got[1], pitch, delta=0.05)


class NoseDepthDominatesTheYawEstimate(unittest.TestCase):
    """The headline number: a different nose is read as a different pose."""

    # Measured, and pinned so a change to the reference head or the solver has
    # to come past this table. Tolerance is loose (2 deg) because the point is
    # the magnitude, not the third decimal.
    TABLE = {
        # factor: {true yaw: solved yaw}
        1.4: {15: 24.6, 30: 44.6, 45: 59.6, 60: 71.3, 75: 81.1},
        1.0: {15: 15.0, 30: 30.0, 45: 45.0, 60: 60.0, 75: 75.0},
        0.6: {15: 4.5, 30: 9.7, 45: 16.5, 60: 27.1, 75: 47.8},
    }

    def test_the_measured_table_still_holds(self):
        for factor, row in self.TABLE.items():
            head = nose(factor)
            for yaw, want in row.items():
                with self.subTest(nose=factor, yaw=yaw):
                    self.assertAlmostEqual(solved_yaw(head, yaw), want, delta=2.0)

    def test_the_error_is_far_larger_than_detector_noise(self):
        """A per-person bias of 15 deg cannot be averaged away; 1 px of keypoint
        jitter moves the same estimate by about 1 deg. Temporal smoothing helps
        the second and does nothing at all about the first — which is why this is
        a design constraint and not a filtering problem."""
        bias = abs(solved_yaw(nose(1.4), 30) - 30.0)

        rng = np.random.default_rng(0)
        base = project(HEAD_5PT, 30)
        jitter = np.std([solve_pose_5pt(base + rng.normal(0, 1.0, base.shape))[0]
                         for _ in range(200)])

        self.assertGreater(bias, 10.0)
        self.assertGreater(bias, 5.0 * jitter)




class OtherShapeDifferencesAreSecondOrder(unittest.TestCase):
    """Eye separation and face length also differ between people. Recorded so the
    nose is not blamed for their share — they are visibly smaller effects, which
    is why a fix should start at the nose."""

    def test_eye_separation_costs_a_couple_of_degrees(self):
        for factor in (0.85, 1.15):
            for yaw in (30, 60):
                with self.subTest(eyes=factor, yaw=yaw):
                    self.assertLess(abs(solved_yaw(eyes(factor), yaw) - yaw), 3.0)

    def test_a_longer_face_biases_pitch_not_yaw(self):
        """Mouth corners 0.10 lower than the reference. The jaw-aware solve exists
        partly to absorb this — it reads the difference as a slightly open mouth,
        which is the right thing to do with it."""
        h = np.array(HEAD_5PT, dtype=np.float64, copy=True)
        h[3, 1] -= 0.10
        h[4, 1] -= 0.10
        yaw, pitch, _ = solve_pose_5pt(project(h, 0))
        self.assertLess(abs(yaw), 0.5)
        self.assertGreater(abs(pitch), 5.0)             # jaw-blind: reads as pitch

        jaw_yaw, jaw_pitch = solve_pose_jaw_5pt(project(h, 0))[:2]
        self.assertLess(abs(jaw_yaw), 0.5)
        self.assertLess(abs(jaw_pitch), abs(pitch))     # jaw-aware: absorbed


if __name__ == '__main__':
    unittest.main()
