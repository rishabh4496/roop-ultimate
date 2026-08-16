"""An open mouth reads as head pitch — unless the solve knows about jaws.

Two of the five arcface keypoints are the MOUTH CORNERS, and the jaw is a moving
part. Anything that solves head pose from those five points is therefore solving
for a rigid head using two points that are not rigidly attached to it. Measured
on a dead-frontal synthetic head as the jaw drops, as a fraction of the
eye-to-mouth distance:

    jaw drop     0%     15%     30%     45%     60%
    solved pitch 0.0   -11.0   -18.7   -24.1   -28.0 deg
    crop scale  1.000   0.937   0.878   0.821   0.699

So a wide-open mouth reports up to 28 degrees of pitch that is not there and,
because the alignment fits all five points, shrinks the crop by up to 30%. On
video that is the face BREATHING in size as someone talks. It is also why an
open-mouth frame looks bad in every swapper at once: the model never sees a
correctly scaled crop, whichever model it is.

`solve_pose_5pt` still reports it that way, deliberately — it is the cheap hot
path, and the gates that only need "is this face roughly frontal" are tuned
around that behaviour. ("Every gate keyed on it is tuned around it" is what this
said, and it was not true: the mouth and eye RESTORE fades were not, in two
directions at once. See TheRestoreFadesAskAboutTheHeadNotTheMouth below — that
is a consumer asking how far the HEAD is turned, and it now uses the jaw-aware
solve too.) What changed first is
that the pose-matched ALIGNMENT templates no longer use it. They were the one
consumer that could not survive it: 'pose' mode projects the reference head at
the angles it is handed, so a phantom -28 degrees moved the template to a place
the head is not, and it did so on ordinary talking frames rather than on rare
ones. Sampling this project's own test clip, most of the faces engaging the
template were engaging it on phantom pitch. `solve_pose_jaw_5pt` gives those two
callers the head's real angles plus the jaw, and the template is projected with
the mouth where the mouth actually is.

WHY THE OBVIOUS FIX DOES NOT WORK, and what does. Adding a jaw column to the
linear solve changes nothing, and the reason is exact rather than approximate:
after centring — which any pose solve must do, translation not being a pose —
the jaw displacement direction lies EXACTLY in the span of the reference head's
own coordinates, residual 4e-16. Look at the reference's y column against the
jaw pattern and it is obvious: both eyes sit at one y, both mouth corners at
another, and dropping the jaw is indistinguishable from a linear reweighting of
that axis, which is what a pitch change is. Both facts are pinned below.

What breaks the tie is the ROTATION CONSTRAINT: a jaw drop leaves a residual no
rigid rotation can produce. Driving that inconsistency to zero is a small
one-dimensional problem in the jaw, which is what the shipped solver does.
"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals                                              # noqa: E402
import roop.face_util as face_util                                # noqa: E402
from roop.face_util import _project_reference, solve_pose_5pt, solve_pose_jaw_5pt, estimate_norm, _REF5_IMAGE


def kps(yaw, pitch, roll=0.0, jaw=0.0, scale=180.0, cx=400.0, cy=300.0):
    """5 keypoints of the reference head, with the jaw dropped by `jaw` x the
    eye-to-mouth distance.

    Built by rolling the projection in 2-D rather than by asking
    `_project_reference` for the roll, so the roll here is independent of the
    convention the solver decomposes back out and a sign error in either one
    cannot cancel.
    """
    p = _project_reference(yaw, pitch, jaw)
    c, s = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    p = p @ np.array([[c, -s], [s, c]]).T
    p = p - p.mean(axis=0)
    return (p * scale + np.array([cx, cy])).astype(np.float64)


def crop_scale(k, size=256):
    M = estimate_norm(k, size)
    return math.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]))


class ModeSwitch(unittest.TestCase):
    pass



class TheEffectIsReal(ModeSwitch):
    def test_a_dropped_jaw_is_reported_as_pitch(self):
        """Frontal head, mouth opening, nothing else moving."""
        pitches = [solve_pose_5pt(kps(0, 0, jaw=j))[1]
                   for j in (0.0, 0.15, 0.30, 0.45, 0.60)]
        self.assertAlmostEqual(pitches[0], 0.0, delta=0.05)
        self.assertLess(pitches[-1], -20.0,
                        f'expected a large phantom pitch, got {pitches[-1]:.1f}')
        # Monotonic, so it tracks the mouth rather than being noise.
        for a, b in zip(pitches, pitches[1:]):
            self.assertLess(b, a + 1e-6)

    def test_the_crop_shrinks_as_the_mouth_opens(self):
        """The part a viewer sees: the pasted face changes size while talking.

        Still true with the alignment off, and deliberately so — that crop IS
        standard arcface alignment, which is what the swap models were trained
        with, so 'correcting' it in the default path would move every frame away
        from the training distribution rather than toward it.
        """
        scales = [crop_scale(kps(0, 0, jaw=j)) for j in (0.0, 0.3, 0.6)]
        self.assertGreater(scales[0] / scales[-1], 1.25,
                           f'crop scale swing only {scales[0]/scales[-1]:.3f}x')


class TheJawAwareSolveSeparatesThem(unittest.TestCase):
    """What `solve_pose_jaw_5pt` has to get right for the templates to be safe."""

    def test_a_talking_frontal_head_reads_as_frontal(self):
        for j in (0.0, 0.15, 0.30, 0.45, 0.60):
            yaw, pitch, roll, jaw = solve_pose_jaw_5pt(kps(0, 0, jaw=j))
            self.assertAlmostEqual(pitch, 0.0, delta=1.0,
                                   msg=f'jaw {j}: phantom pitch {pitch:.1f} deg')
            self.assertAlmostEqual(jaw, j, delta=0.02,
                                   msg=f'jaw {j}: solved {jaw:.3f}')

    # Jaw values deliberately chosen NOT to coincide with the solver's internal
    # scan grid. This is load-bearing: a grid whose jaws land on the scan points
    # passes at ANY quality, because the scan lands on the answer and the
    # refinement never has to work. Measured — with on-grid jaws the whole sweep
    # reports 0.0000 deg even with the refinement cut to two steps.
    OFF_GRID_JAWS = (-0.213, -0.077, 0.031, 0.187, 0.334, 0.471, 0.628, 0.742)

    def test_it_recovers_pose_and_jaw_together(self):
        """Both at once is the case that matters — a head that is turned AND
        tilted AND talking is most of this project's footage.

        Excludes |yaw| < 5, where the two are genuinely not separable; that
        region has its own test below."""
        worst = 0.0
        where = None
        # Stepped finely enough to have teeth: on a yaw-15 x pitch-20 grid,
        # cutting the refinement to two Gauss-Newton steps still scores 0.021
        # deg and slips through. At yaw 10 x pitch 10 it scores 0.183.
        for yaw in range(5, 91, 10):
            for pitch in range(-40, 41, 10):
                for roll in (-90, -45, 0, 45, 90):
                    for j in self.OFF_GRID_JAWS:
                        got = solve_pose_jaw_5pt(kps(yaw, pitch, roll, j))
                        self.assertIsNotNone(got)
                        err = max(abs(got[0] - yaw), abs(got[1] - pitch),
                                  abs(got[3] - j) * 100.0)
                        if err > worst:
                            worst, where = err, (yaw, pitch, roll, j, got)
        # Tight on purpose. The measured worst here is 0.0000 deg; a loose bound
        # would let the refinement be cut back to two Gauss-Newton steps (0.18
        # deg) without anything noticing.
        self.assertLess(worst, 0.05, f'worst {worst:.4f} at {where}')

    def test_a_converged_root_is_not_discarded_as_a_tie(self):
        """The tie-break may only fire between readings that genuinely cannot be
        told apart.

        This is a regression guard with a specific history: the "both fit
        exactly, keep the closed mouth" floor was first set at 1e-8. A candidate
        that had converged to within 0.06 of the right jaw scores about 1e-9 —
        twenty-one orders of magnitude worse than the exact root sitting beside
        it, and no tie at all — but the floor called it perfect, stopped the
        comparison and kept it. Fourteen poses on the grid were misread by up to
        5.9 degrees with the right answer already computed and thrown away.
        """
        # Read through the module, not a from-import: a value bound at import
        # time cannot see the constant it is meant to be guarding.
        self.assertLess(face_util._JAW_TIE_FLOOR, 1e-15,
                        'the tie floor must sit near the floor of an EXACT fit '
                        '(~1e-30), not at a merely small number')
        # The exact poses the 1e-8 floor misread, so the behaviour is guarded
        # and not just the constant.
        for roll in (-90, -45, 0, 45, 90):
            for j in (0.65, 0.75):
                got = solve_pose_jaw_5pt(kps(0, 40, roll, j))
                self.assertAlmostEqual(got[1], 40.0, delta=1.0,
                                       msg=f'roll {roll} jaw {j} -> pitch '
                                           f'{got[1]:.2f}, jaw {got[3]:.3f}')
                self.assertAlmostEqual(got[3], j, delta=0.01)


    def test_it_is_not_blind_at_45_degrees_of_roll(self):
        """A regression guard with a specific history. The rotation
        inconsistency was first written as |a0|^2 - |a1|^2 alone, which an
        in-plane roll turns through 2*theta — so it is identically ZERO at 45
        degrees and the search ran off to the clamp. A face lying on its side is
        exactly the footage this was reported on."""
        for roll in (-90, -75, -45, -30, 0, 30, 45, 75, 90):
            yaw, pitch, _, jaw = solve_pose_jaw_5pt(kps(30, -20, roll, 0.4))
            self.assertAlmostEqual(yaw, 30.0, delta=1.0, msg=f'roll {roll}')
            self.assertAlmostEqual(pitch, -20.0, delta=1.0, msg=f'roll {roll}')
            self.assertAlmostEqual(jaw, 0.4, delta=0.02, msg=f'roll {roll}')

    def test_the_yaw_zero_ambiguity_resolves_to_the_closed_mouth(self):
        """At yaw 0 there are genuinely TWO exact readings of the same five
        points — the nose's sideways offset, which separates a tilt from a jaw
        once the head has turned, is zero. Measured on an exact projection at
        pitch +10 with the mouth 5% closed: jaw -0.05 fits to 1.1e-31 and
        jaw +0.83 with pitch +66.7 fits to 2.1e-31.

        Nothing numerical can choose, so a prior does, and this pins which way
        it points: the more closed mouth, because reading an ordinary face as a
        head tilted 57 degrees further back is far more damaging than leaving a
        genuinely tilted one uncorrected.
        """
        for pitch in (5, 10, 20, 30):
            yaw, got_pitch, _, jaw = solve_pose_jaw_5pt(kps(0, pitch, jaw=-0.05))
            self.assertLess(abs(got_pitch - pitch), 6.0,
                            f'pitch {pitch}: took the wide-mouth reading '
                            f'({got_pitch:.1f} deg, jaw {jaw:.2f})')

    def test_degenerate_input_returns_none(self):
        self.assertIsNone(solve_pose_jaw_5pt(None))
        self.assertIsNone(solve_pose_jaw_5pt(np.zeros((5, 2))))
        self.assertIsNone(solve_pose_jaw_5pt(np.full((5, 2), np.nan)))
        self.assertIsNone(solve_pose_jaw_5pt(np.zeros((4, 2))))


class TheAlignmentNoLongerActsOnPhantomPitch(ModeSwitch):
    pass





def _restore_fade(yaw, pitch):
    """The fade in apply_mouth_area / apply_eyes_area, as shipped."""
    m = max(abs(yaw), abs(pitch))
    if m <= 25.0:
        return 1.0
    return max(0.0, min(1.0, (38.0 - m) / 13.0))


class TheRestoreFadesAskAboutTheHeadNotTheMouth(unittest.TestCase):
    """The mouth and eye restores fade out past ~25 deg, because that is where
    the plate's features stop sitting where the swap's do and pasting them
    doubles the feature.

    That is a question about the HEAD, so it cannot be answered by
    `solve_pose_5pt` — two of its five points are the mouth corners. The
    docstring at the top of this file says every gate keyed on it is tuned
    around the phantom pitch; this gate was not, in two different directions.
    """

    def test_a_frontal_talking_head_no_longer_fades_its_own_restore(self):
        """0.765 at full opening, sliding continuously as the mouth moves — so
        the restore strength modulated with speech, on the exact footage the
        feature exists for. The EYE restore too, which has no jaw in it."""
        for j in (0.0, 0.15, 0.30, 0.45, 0.60):
            y, p, _r, _j = solve_pose_jaw_5pt(kps(0, 0, jaw=j))
            self.assertEqual(_restore_fade(y, p), 1.0,
                             f'jaw {j} fades the restore on a frontal head')

    def test_the_fade_no_longer_moves_the_wrong_way_on_a_turned_head(self):
        """The sharper half. At a true yaw of 35 the fade SHOULD bite — and fed
        the jaw-blind solve, opening the mouth made the restore stronger
        (0.231 -> 0.388), because the phantom pitch drags the solved yaw down
        (35.0 -> 33.0) while staying under it, so max(|yaw|, |pitch|) falls."""
        blind = [_restore_fade(*solve_pose_5pt(kps(35, 0, jaw=j))[:2])
                 for j in (0.0, 0.60)]
        self.assertGreater(blind[1], blind[0] + 0.1,
                           'the jaw-blind fade no longer inverts — retune this test')
        aware = [_restore_fade(*solve_pose_jaw_5pt(kps(35, 0, jaw=j))[:2])
                 for j in (0.0, 0.60)]
        self.assertAlmostEqual(aware[0], aware[1], places=2,
                               msg='the fade still moves when only the mouth does')

    def test_a_real_turn_still_fades(self):
        """The guard on the fix: taking the jaw out must not take the fade out.
        It is there to stop a doubled lip and that failure is real."""
        self.assertEqual(_restore_fade(*solve_pose_jaw_5pt(kps(50, 0, jaw=0.6))[:2]), 0.0)
        self.assertLess(_restore_fade(*solve_pose_jaw_5pt(kps(35, 0))[:2]), 0.3)


class WhyALinearSolveCannotDoIt(unittest.TestCase):
    """The degeneracy, stated as arithmetic so nobody re-derives it by trying."""

    def test_the_jaw_direction_lies_in_the_reference_span(self):
        X = np.asarray(_REF5_IMAGE, dtype=np.float64)
        jaw = np.array([0.0, 0.0, 0.0, 1.0, 1.0])      # mouth corners only
        # Centred, because a pose solve removes translation first.
        Xc = X - X.mean(axis=0)
        jc = jaw - jaw.mean()
        coef, *_ = np.linalg.lstsq(Xc, jc, rcond=None)
        residual = float(np.linalg.norm(Xc @ coef - jc))
        self.assertLess(residual, 1e-9,
                        f'the jaw direction is NOT in the reference span '
                        f'(residual {residual:.2e}) — if this ever becomes true, '
                        f'a joint linear solve for pose+jaw becomes possible')

    def test_so_a_joint_linear_solve_adds_nothing(self):
        """Built the way it would be built, and shown to return the jaw-blind
        answer to the digit — so the idea is closed off, not merely untried."""
        X = np.asarray(_REF5_IMAGE, dtype=np.float64)
        delta = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        D = np.zeros((10, 8))
        for i in range(5):
            D[2 * i, 0:3] = X[i]
            D[2 * i, 6] = delta[i]
            D[2 * i + 1, 3:6] = X[i]
            D[2 * i + 1, 7] = delta[i]
        dpinv = np.linalg.pinv(D)

        for jaw in (0.0, 0.3, 0.6):
            k = kps(0, 0, jaw=jaw)
            x = np.asarray(k) - np.asarray(k).mean(axis=0)
            th = dpinv @ x.reshape(-1)
            a0x, a0y, a0z = th[0], th[1], th[2]
            a1x, a1y, a1z = th[3], th[4], th[5]
            n1 = math.hypot(math.hypot(a0x, a0y), a0z)
            r1 = (a0x / n1, a0y / n1, a0z / n1)
            dot = a1x * r1[0] + a1y * r1[1] + a1z * r1[2]
            r2 = (a1x - dot * r1[0], a1y - dot * r1[1], a1z - dot * r1[2])
            n2 = math.hypot(math.hypot(*r2[:2]), r2[2])
            r2 = (r2[0] / n2, r2[1] / n2, r2[2] / n2)
            # r3 = r1 x r2; pitch = asin(-r3y), as in solve_pose_5pt.
            r3y = r1[2] * r2[0] - r1[0] * r2[2]
            joint_pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r3y))))
            self.assertAlmostEqual(
                joint_pitch, solve_pose_5pt(k)[1], delta=0.05,
                msg=f'the joint solve differs at jaw={jaw} — if so, revisit')

    def test_the_rotation_constraint_is_what_breaks_the_tie(self):
        """And the constrained solve, which is the shipped one, does not agree
        with either — that difference IS the fix."""
        k = kps(0, 0, jaw=0.45)
        self.assertLess(solve_pose_5pt(k)[1], -20.0)
        self.assertAlmostEqual(solve_pose_jaw_5pt(k)[1], 0.0, delta=1.0)


class TheAlignmentIsThePlainFit(unittest.TestCase):
    def test_estimate_norm_is_exactly_the_plain_similarity_fit(self):
        """Load-bearing since the pose-matched template was deleted: there is
        one alignment path now, and it must be the ordinary least-squares fit
        with no branch — including on a head that is turned AND talking."""
        from skimage import transform as trans
        from roop.face_util import arcface_dst
        for yaw in (0, 30, 60, 90):
            for j in (0.0, 0.3, 0.6):
                k = kps(yaw, -20, 0.0, j)
                t = trans.SimilarityTransform()
                t.estimate(k, arcface_dst * 2.0)
                np.testing.assert_array_equal(estimate_norm(k, 224),
                                              t.params[0:2, :])


if __name__ == '__main__':
    unittest.main()
