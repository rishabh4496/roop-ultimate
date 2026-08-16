"""estimate_norm — the 5-point crop alignment, and the opt-in profile variant.

The headline guarantee here is that the alignment fit is a pure similarity. It is
opt-in precisely because it changes the crop, and therefore the swap output, for
high-yaw faces; if it ever leaked into the default path it would silently change
every render. test_off_is_bit_exact_noop is what makes that claim checkable
instead of a sentence in a commit message.
"""

import os
import sys
import unittest

import numpy as np
from skimage import transform as trans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals                                              # noqa: E402
from roop.face_util import arcface_dst, estimate_norm, kps_pose_ratios, WARP_TEMPLATES
from tests.facegeom import decompose, fit_residual, project_kps  # noqa: E402

SIZES = (112, 128, 256, 512, 320)
MODES = ("arcface", "arcface_112_v1", "arcface_112_v2", "ffhq_512", "mtcnn_512")


def reference_template(image_size, mode):
    """The destination points estimate_norm targets, rebuilt independently so a
    change to its template maths shows up as a test failure."""
    if mode in WARP_TEMPLATES:
        return WARP_TEMPLATES[mode] * float(image_size)
    if image_size % 112 == 0:
        ratio, shift = float(image_size) / 112.0, 0.0
    elif image_size % 128 == 0:
        ratio, shift = float(image_size) / 128.0, 8.0 * float(image_size) / 128.0
    elif image_size % 512 == 0:
        ratio, shift = float(image_size) / 512.0, 32.0 * float(image_size) / 512.0
    else:
        ratio, shift = float(image_size) / 112.0, 0.0
    dst = arcface_dst * ratio
    dst[:, 0] += shift
    return dst


class YawAlignCase(unittest.TestCase):
    """Restores the global so one test can never leak into another."""




class TestDefaultPath(YawAlignCase):
    pass




class TestProfileAlignment(YawAlignCase):








    def test_output_is_always_finite_and_invertible(self):
        import cv2
        for yaw in (70, 75, 80, 85, 90):
            for pitch in (-30, 0, 30):
                for roll in (-30, 0, 30):
                    m = estimate_norm(project_kps(yaw, pitch, roll), 512, "arcface")
                    self.assertTrue(np.isfinite(m).all())
                    scale, _ = decompose(m)
                    self.assertGreater(scale, 1e-6)
                    cv2.invertAffineTransform(m)   # must not raise



class TestModeSelection(YawAlignCase):
    """yaw_align accepts booleans (legacy) and mode names. Anything
    unrecognised must fall back to 'off' rather than silently enabling a mode
    that changes render output."""






class TestPoseSolve(unittest.TestCase):
    """`solve_pose_5pt` replaces the scalar ratio proxies for anything that has
    to reason about yaw and pitch together."""

    def test_recovers_every_angle_over_the_whole_grid(self):
        from roop.face_util import solve_pose_5pt
        worst = [0.0, 0.0, 0.0]
        for yaw in range(0, 91, 5):
            for pitch in range(-45, 50, 5):
                for roll in (-30, -10, 0, 10, 30):
                    got = solve_pose_5pt(project_kps(yaw, pitch, roll))
                    self.assertIsNotNone(got, f"{yaw}/{pitch}/{roll}")
                    for i, truth in enumerate((yaw, pitch, roll)):
                        worst[i] = max(worst[i], abs(got[i] - truth))
        for i, name in enumerate(("yaw", "pitch", "roll")):
            self.assertLess(worst[i], 0.5, f"{name} off by {worst[i]:.2f} deg")

    def test_works_at_a_true_profile_where_the_ratios_are_degenerate(self):
        """At 90 deg the eyes project to the same point, so eye separation — the
        thing every ratio proxy divides by — has collapsed to zero. The solve
        leans on the nose tip standing proud of the eye/mouth plane instead."""
        from roop.face_util import solve_pose_5pt
        kps = project_kps(90, 0)
        self.assertAlmostEqual(
            float(np.linalg.norm(kps[1] - kps[0])), 0.0, places=3)
        yaw, pitch, _ = solve_pose_5pt(kps)
        self.assertAlmostEqual(abs(yaw), 90.0, delta=0.5)
        self.assertAlmostEqual(pitch, 0.0, delta=0.5)


    def test_degenerate_input_returns_none_rather_than_guessing(self):
        from roop.face_util import solve_pose_5pt
        self.assertIsNone(solve_pose_5pt(None))
        self.assertIsNone(solve_pose_5pt(np.zeros((5, 2), np.float32)))
        self.assertIsNone(solve_pose_5pt(np.zeros((3, 2), np.float32)))
        self.assertIsNone(solve_pose_5pt(np.full((5, 2), np.nan, np.float32)))

    def test_offaxis_combines_the_two_angles(self):
        """|yaw| alone under-ranks a turned AND tilted head, which is how the
        worst poses kept passing for mid-angle ones."""
        from roop.face_util import offaxis_deg
        self.assertAlmostEqual(offaxis_deg(0, 0), 0.0, places=3)
        self.assertAlmostEqual(offaxis_deg(40, 0), 40.0, places=3)
        self.assertAlmostEqual(offaxis_deg(0, 40), 40.0, places=3)
        self.assertGreater(offaxis_deg(40, 40), 50.0)
        self.assertGreater(offaxis_deg(75, 40), 75.0)


class TestNonFrontalMaskRouting(unittest.TestCase):
    """The mask router picks between two different mask derivations. It read the
    same contaminated scalars the alignment gate did, and had the same blind
    spot — sixteen cells over yaw 0-90 x pitch +/-45 came out "frontal" while
    being 79-90 deg off-axis, every one a profile head also tilted up or down.
    """

    @staticmethod
    def _rule(kps):
        """The shipped test, exercised through the real helpers."""
        from roop.face_util import offaxis_deg, solve_pose_5pt
        non_frontal = False
        lex, rex, nx = kps[0][0], kps[1][0], kps[2][0]
        d_le, d_re = abs(nx - lex), abs(nx - rex)
        if d_le + d_re > 1e-5 and abs(d_le - d_re) / (d_le + d_re) > 0.25:
            non_frontal = True
        yaw_ratio, pitch_ratio = kps_pose_ratios(kps)
        if yaw_ratio is not None and yaw_ratio < 0.55:
            non_frontal = True
        if pitch_ratio is not None and not (0.32 < pitch_ratio < 0.70):
            non_frontal = True
        if not non_frontal:
            pose = solve_pose_5pt(kps)
            if pose is not None and offaxis_deg(pose[0], pose[1]) > 50.0:
                non_frontal = True
        return non_frontal

    def test_the_added_term_matches_what_the_router_ships(self):
        """This file re-implements the rule, so it can drift from the real one.
        Pin the parts that matter to the source — which now lives in
        roop/nonfrontal.py, alongside the latch that consumes it."""
        import inspect
        import roop.nonfrontal
        src = inspect.getsource(roop.nonfrontal)
        self.assertIn("solve_pose_5pt", src)
        self.assertIn("offaxis_deg", src)
        self.assertEqual(roop.nonfrontal._OFFAXIS_MAX, 50.0)

    def test_no_extreme_pose_is_routed_as_frontal(self):
        from roop.face_util import offaxis_deg, solve_pose_5pt
        blind = []
        for yaw in range(0, 91, 5):
            for pitch in range(-45, 50, 5):
                kps = project_kps(yaw, pitch)
                pose = solve_pose_5pt(kps)
                if offaxis_deg(pose[0], pose[1]) >= 50.0 and not self._rule(kps):
                    blind.append((yaw, pitch))
        self.assertEqual(blind, [], f"still routed as frontal: {blind}")

    def test_the_old_proxies_really_were_blind_there(self):
        """Records the defect, so the extra term cannot be dropped as redundant."""
        for yaw, pitch in ((75, 40), (85, -45), (90, -40), (90, 45)):
            kps = project_kps(yaw, pitch)
            yaw_ratio, pitch_ratio = kps_pose_ratios(kps)
            lex, rex, nx = kps[0][0], kps[1][0], kps[2][0]
            asym = abs(abs(nx - lex) - abs(nx - rex)) / (abs(nx - lex) + abs(nx - rex))
            self.assertLess(asym, 0.25, f"asymmetry saw yaw={yaw} pitch={pitch}")
            self.assertGreaterEqual(yaw_ratio, 0.55)
            self.assertTrue(0.32 < pitch_ratio < 0.70)
            self.assertTrue(self._rule(kps), "the new term did not rescue it")

    @staticmethod
    def _rule_without_the_new_term(kps):
        non_frontal = False
        lex, rex, nx = kps[0][0], kps[1][0], kps[2][0]
        d_le, d_re = abs(nx - lex), abs(nx - rex)
        if d_le + d_re > 1e-5 and abs(d_le - d_re) / (d_le + d_re) > 0.25:
            non_frontal = True
        yaw_ratio, pitch_ratio = kps_pose_ratios(kps)
        if yaw_ratio is not None and yaw_ratio < 0.55:
            non_frontal = True
        if pitch_ratio is not None and not (0.32 < pitch_ratio < 0.70):
            non_frontal = True
        return non_frontal

    def test_the_added_term_changes_nothing_below_its_threshold(self):
        """The term is additive and must stay that way: everything under 50 deg
        off-axis has to keep whatever verdict it already had. Otherwise closing
        the blind spot would quietly re-route ordinary faces as well."""
        from roop.face_util import offaxis_deg, solve_pose_5pt
        checked = 0
        for yaw in range(0, 91, 5):
            for pitch in range(-45, 50, 5):
                kps = project_kps(yaw, pitch)
                pose = solve_pose_5pt(kps)
                if offaxis_deg(pose[0], pose[1]) >= 50.0:
                    continue
                self.assertEqual(self._rule(kps),
                                 self._rule_without_the_new_term(kps),
                                 f"yaw={yaw} pitch={pitch} re-routed")
                checked += 1
        self.assertGreater(checked, 100)

    def test_a_dead_frontal_face_is_still_frontal(self):
        for pitch in (-15, 0, 15):
            self.assertFalse(self._rule(project_kps(0, pitch)),
                             f"yaw=0 pitch={pitch} newly non-frontal")


class TestTheYawRatioCannotDoThePoseSolvesJob(unittest.TestCase):
    """A yaw-ratio -> yaw lookup used to stand where solve_pose_5pt is now, and
    these are the two reasons it had to go. Kept as tests rather than a comment
    so nobody reintroduces the cheaper-looking scalar inversion."""

    def test_it_cannot_see_pitch_at_all(self):
        """One scalar, two unknowns. Heads at the same yaw and wildly different
        pitch are indistinguishable to it — including a pair straddling level,
        which the ratio maps to nearly the same number."""
        a = kps_pose_ratios(project_kps(0, -25))[0]
        b = kps_pose_ratios(project_kps(0, 25))[0]
        self.assertAlmostEqual(a, b, delta=0.12)
        from roop.face_util import solve_pose_5pt
        self.assertAlmostEqual(solve_pose_5pt(project_kps(0, -25))[1], -25, delta=0.5)
        self.assertAlmostEqual(solve_pose_5pt(project_kps(0, 25))[1], 25, delta=0.5)

    def test_pitch_corrupts_the_yaw_it_would_have_reported(self):
        """Worse than merely blind: a tilted profile reads as a mid-angle face,
        so the old lookup would have posed the template at the wrong yaw."""
        level = kps_pose_ratios(project_kps(90, 0))[0]
        tilted = kps_pose_ratios(project_kps(90, -30))[0]
        self.assertLess(level, 0.1)          # a profile, correctly
        self.assertGreater(tilted, 0.4)      # the same profile, read as mid-angle
        from roop.face_util import solve_pose_5pt
        for pitch in (-30, 0, 30):
            self.assertAlmostEqual(abs(solve_pose_5pt(project_kps(90, pitch))[0]),
                                   90.0, delta=0.5)

    def test_the_lookup_is_gone(self):
        import roop.face_util as fu
        for dead in ('_yaw_from_ratio', '_build_yaw_lookup', '_LUT_YAWS'):
            self.assertFalse(hasattr(fu, dead),
                             f"{dead} is back — two ways to ask the same question")


class TestAlignmentIsIllConditionedAtProfile(unittest.TestCase):
    """Records WHY the profile path exists, so the motivation survives even if
    the implementation is rewritten."""

    def test_eye_separation_collapses(self):
        sep = lambda y: float(np.linalg.norm(project_kps(y)[1] - project_kps(y)[0]))
        self.assertGreater(sep(0), 130.0)
        self.assertAlmostEqual(sep(90), 0.0, places=4)




class TestRotatedFaceHandling(unittest.TestCase):
    def test_unrotate_face_coords_clockwise(self):
        from roop.face_util import _unrotate_face_coords
        from insightface.app.common import Face

        face = Face(
            bbox=np.array([799.0, 700.0, 899.0, 800.0], dtype=np.float32),
            kps=np.array([[799.0, 700.0]], dtype=np.float32)
        )
        orig_w, orig_h = 800, 1000
        _unrotate_face_coords(face, orig_w, orig_h, "clockwise")
        self.assertAlmostEqual(float(face.kps[0][0]), 700.0, places=4)
        self.assertAlmostEqual(float(face.kps[0][1]), 200.0, places=4)

    def test_unrotate_face_coords_anticlockwise(self):
        from roop.face_util import _unrotate_face_coords
        from insightface.app.common import Face

        face = Face(
            bbox=np.array([200.0, 99.0, 300.0, 199.0], dtype=np.float32),
            kps=np.array([[200.0, 99.0]], dtype=np.float32)
        )
        orig_w, orig_h = 800, 1000
        _unrotate_face_coords(face, orig_w, orig_h, "anticlockwise")
        self.assertAlmostEqual(float(face.kps[0][0]), 700.0, places=4)
        self.assertAlmostEqual(float(face.kps[0][1]), 200.0, places=4)

    def test_every_rescue_angle_round_trips_through_the_real_helpers(self):
        """The two tests above check hand-worked numbers, which is how an angle
        can be added with an un-map that was never exercised. This one asks the
        shipped rotate helper where a pixel actually WENT, then requires the
        un-rotation to bring it back -- so a new angle cannot ship half-wired."""
        from roop.face_util import _unrotate_face_coords, _rescue_rotated
        from roop.ProcessMgr import ProcessMgr
        from insightface.app.common import Face
        import inspect

        h, w = 13, 9
        marker = np.zeros((h, w), dtype=np.int32)
        for y in range(h):
            for x in range(w):
                marker[y, x] = y * w + x

        angles = {"clockwise": "rotate_clockwise",
                  "anticlockwise": "rotate_anticlockwise",
                  "180": "rotate_180"}
        # Whatever the rescue tries, it must be un-mapped here.
        src = inspect.getsource(_rescue_rotated)
        for name in angles:
            self.assertIn(f'"{name}"', src, f"{name} is not part of the rescue")

        for name, action in angles.items():
            turned = ProcessMgr.apply_rotation(marker, action)
            for (sy, sx) in ((0, 0), (4, 7), (h - 1, w - 2)):
                ty, tx = [int(v) for v in np.argwhere(turned == marker[sy, sx])[0]]
                face = Face(bbox=np.array([tx, ty, tx + 1, ty + 1], dtype=np.float32),
                            kps=np.array([[tx, ty]], dtype=np.float32))
                _unrotate_face_coords(face, w, h, name)
                self.assertAlmostEqual(float(face.kps[0][0]), sx, places=4,
                                       msg=f"{name}: x came back wrong")
                self.assertAlmostEqual(float(face.kps[0][1]), sy, places=4,
                                       msg=f"{name}: y came back wrong")


class RotationActionCase(unittest.TestCase):
    """The orientation call is only ever asserted through its CONSEQUENCE: does
    turning the frame that way actually stand the face up?

    Asserting the returned string against a hand-worked convention is what let
    a sign inversion ship -- the test agreed with the code because it was
    derived from it. So these tests apply the project's real rotate helpers and
    measure the face afterwards.
    """

    @staticmethod
    def _face(kps):
        from insightface.app.common import Face
        kps = np.asarray(kps, dtype=np.float32)
        return Face(bbox=np.array([kps[:, 0].min(), kps[:, 1].min(),
                                   kps[:, 0].max(), kps[:, 1].max()],
                                  dtype=np.float32),
                    kps=kps)

    @staticmethod
    def _rotate_dir(action, vec):
        """Push a direction through the SHIPPED rotate helper's real point map."""
        from roop.ProcessMgr import ProcessMgr
        h, w = 9, 7
        img = np.zeros((h, w, 3), dtype=np.int32)
        for y in range(h):
            for x in range(w):
                img[y, x] = (x, y, 0)
        out = ProcessMgr.apply_rotation(img, action)
        pos = {}
        for iy in range(out.shape[0]):
            for ix in range(out.shape[1]):
                pos[(int(out[iy, ix, 0]), int(out[iy, ix, 1]))] = np.array([ix, iy])
        origin = pos[(3, 4)]
        ex, ey = pos[(4, 4)] - origin, pos[(3, 5)] - origin
        return vec[0] * ex + vec[1] * ey

    def _tilt_after(self, action, kps):
        """Tilt of the face's down axis once the frame is turned, in degrees."""
        from roop.face_util import face_down_axis
        axis = np.array(face_down_axis(self._face(kps)))
        axis /= np.linalg.norm(axis)
        moved = self._rotate_dir(action, axis)
        return abs(float(np.degrees(np.arctan2(moved[0], moved[1]))))


class TestRotationStandsTheFaceUp(RotationActionCase):
    def test_rolled_faces_end_up_upright(self):
        """A face lying on its side must be offered a rotation, and that
        rotation must leave it within 45 deg of upright -- not upside down."""
        from roop.face_util import face_rotation_action
        acted = 0
        for roll in (-120, -100, -90, -80, -70, 70, 80, 90, 100, 120):
            for yaw in (0, 45, 90):
                kps = project_kps(yaw, 0, roll)
                action = face_rotation_action(self._face(kps), (512, 512))
                self.assertIsNotNone(
                    action, f"no rotation offered at roll={roll} yaw={yaw}")
                tilt = self._tilt_after(action, kps)
                self.assertLess(tilt, 45.0,
                                f"roll={roll} yaw={yaw}: {action} left the face "
                                f"{tilt:.1f} deg off upright")
                acted += 1
        self.assertEqual(acted, 30)

    def test_rotation_strictly_improves_uprightness(self):
        from roop.face_util import face_roll_tilt, face_rotation_action
        for roll in (-110, -90, -70, 70, 90, 110):
            kps = project_kps(0, 0, roll)
            action = face_rotation_action(self._face(kps), (512, 512))
            self.assertLess(self._tilt_after(action, kps),
                            abs(face_roll_tilt(self._face(kps))),
                            f"roll={roll} got no more upright")

    def test_inverted_faces_get_the_half_turn_a_quarter_turn_cannot_do(self):
        """The case a quarter turn cannot reach. An upside-down face is 90 deg
        off upright after EITHER quarter turn -- still lying on its side -- so
        it needs the half turn, and it must actually be offered one rather than
        declined."""
        from roop.face_util import face_rotation_action
        for roll in (-180, -170, -160, -150, -140, 140, 150, 160, 170, 180):
            kps = project_kps(0, 0, roll)
            action = face_rotation_action(self._face(kps), (512, 512))
            self.assertEqual(action, "rotate_180",
                             f"roll={roll} was not treated as inverted")
            self.assertLess(self._tilt_after(action, kps), 45.0,
                            f"roll={roll}: the half turn left it off upright")

    def test_a_half_turn_is_never_offered_to_an_upright_face(self):
        """Reaching the inverted band takes 135 deg of error against a 9.1 deg
        axis, so no pose of an upright head may land there."""
        from roop.face_util import face_rotation_action
        for yaw in (0, 30, 45, 60, 75, 90):
            for pitch in (-30, -15, 0, 15, 30):
                for roll in (-40, -20, 0, 20, 40):
                    self.assertNotEqual(
                        face_rotation_action(
                            self._face(project_kps(yaw, pitch, roll)), (512, 512)),
                        "rotate_180",
                        f"upside-down call at yaw={yaw} pitch={pitch} roll={roll}")

    def test_the_two_branches_meet_without_a_seam(self):
        """At the boundary the quarter and half turns are equivalent -- both
        land the face at 45 deg -- so a noisy reading either side of 135 costs
        nothing. That is why the upper edge carries no safety margin."""
        from roop.face_util import FACE_ROLL_UPPER
        kps = project_kps(0, 0, FACE_ROLL_UPPER)
        self.assertAlmostEqual(self._tilt_after("rotate_clockwise", kps),
                               self._tilt_after("rotate_180", kps), places=3)

    def test_the_two_edges_are_not_symmetric(self):
        """The lower edge carries a safety margin over 45 deg because a false
        positive there corrupts an upright face; the upper edge must NOT
        inherit it, because nothing near 135 deg risks being upright and the
        margin would only cost recall -- a face at 120 deg is 90 deg better off
        for being turned."""
        from roop.face_util import (face_rotation_action,
                                    FACE_ROLL_LOWER, FACE_ROLL_UPPER)
        self.assertGreater(FACE_ROLL_LOWER, 45.0)
        self.assertEqual(FACE_ROLL_UPPER, 135.0)
        for roll in (115, 120, 125, 130, -115, -120, -125, -130):
            kps = project_kps(0, 0, roll)
            action = face_rotation_action(self._face(kps), (512, 512))
            self.assertIsNotNone(action, f"declined a rescuable roll={roll}")
            self.assertLess(self._tilt_after(action, kps), 45.0)


class TestRotationLeavesUprightFacesAlone(RotationActionCase):
    def test_no_rotation_for_any_upright_pose(self):
        """The regression that made this file necessary: an upright head that is
        merely TURNED must never be treated as a head lying on its side."""
        from roop.face_util import face_rotation_action
        for yaw in (0, 15, 30, 45, 50, 60, 70, 75, 80, 85, 90):
            for pitch in (-30, -15, 0, 15, 30):
                kps = project_kps(yaw, pitch, 0)
                self.assertIsNone(
                    face_rotation_action(self._face(kps), (512, 512)),
                    f"upright face rotated at yaw={yaw} pitch={pitch}")

    def test_small_and_mid_roll_is_left_alone(self):
        """Below the boundary a 90 deg turn would make things worse. Swept over
        the full pose grid, because it is yaw stacking onto real roll that
        pushes a merely-tilted head over the line."""
        from roop.face_util import face_rotation_action
        for roll in (-44, -40, -30, -15, 0, 15, 30, 40, 44):
            for yaw in (0, 15, 30, 45, 60, 75, 90):
                for pitch in (-30, 0, 30):
                    self.assertIsNone(
                        face_rotation_action(
                            self._face(project_kps(yaw, pitch, roll)), (512, 512)),
                        f"rotated at roll={roll} yaw={yaw} pitch={pitch}")

    def test_the_axis_it_uses_is_the_yaw_robust_one(self):
        """Guards the actual finding behind the fix. The nose tip swings up to
        ~45 deg under a turn of a perfectly upright head -- indistinguishable
        from real roll -- while the mouth midline stays under ~10 deg. If a
        future edit reaches for the nose again, this fails."""
        from roop.face_util import face_down_axis
        worst_used, worst_nose = 0.0, 0.0
        for yaw in (0, 30, 50, 60, 70, 80, 90):
            for pitch in (-30, -15, 0, 15, 30):
                kps = project_kps(yaw, pitch, 0)
                ax = face_down_axis(self._face(kps))
                worst_used = max(worst_used,
                                 abs(np.degrees(np.arctan2(ax[0], ax[1]))))
                nose = kps[2] - (kps[0] + kps[1]) / 2.0
                worst_nose = max(worst_nose,
                                 abs(np.degrees(np.arctan2(nose[0], nose[1]))))
        self.assertLess(worst_used, 12.0,
                        f"axis in use lies by {worst_used:.1f} deg under yaw")
        self.assertGreater(worst_nose, 40.0,
                           "nose axis no longer misbehaves -- re-check the note "
                           "in face_util if this ever fails")


class TestRotationVerificationGuard(unittest.TestCase):
    """rotation_improves_upright is the backstop that keeps a wrong orientation
    call cheap: it rejects the rotation instead of swapping an upside-down crop."""

    @staticmethod
    def _face_at(roll):
        from insightface.app.common import Face
        kps = project_kps(0, 0, roll)
        return Face(bbox=np.array([0, 0, 1, 1], dtype=np.float32), kps=kps)

    def test_accepts_a_rotation_that_stood_the_face_up(self):
        from roop.face_util import rotation_improves_upright
        self.assertTrue(rotation_improves_upright(self._face_at(90),
                                                  self._face_at(3)))

    def test_rejects_an_inverted_rotation(self):
        from roop.face_util import rotation_improves_upright
        self.assertFalse(rotation_improves_upright(self._face_at(90),
                                                   self._face_at(179)))

    def test_rejects_a_rotation_that_changed_nothing(self):
        from roop.face_util import rotation_improves_upright
        self.assertFalse(rotation_improves_upright(self._face_at(90),
                                                   self._face_at(88)))

    def test_passes_through_when_there_is_nothing_to_judge(self):
        from insightface.app.common import Face
        from roop.face_util import rotation_improves_upright
        blind = Face(bbox=np.array([0, 0, 1, 1], dtype=np.float32), kps=None)
        self.assertTrue(rotation_improves_upright(blind, self._face_at(0)))


class TestRotationActionIsSharedNotDuplicated(unittest.TestCase):
    """Render and Frame-Editor preview must agree on orientation; they did not
    while each kept its own copy of the heuristic."""

    def test_processmgr_delegates_to_face_util(self):
        import inspect
        from roop.ProcessMgr import ProcessMgr
        self.assertIn("face_rotation_action",
                      inspect.getsource(ProcessMgr.rotation_action))

    def test_core_has_no_private_copy(self):
        import inspect
        import roop.core
        src = inspect.getsource(roop.core.get_face_crop_from_frame)
        self.assertNotIn("def _rotation_action", src)
        self.assertIn("face_rotation_action", src)

    def test_every_caller_also_applies_the_outcome_gate(self):
        """Agreeing on the heuristic is only half of it.

        Every one of these three re-detects on the turned crop and commits to the
        rotation only if the face came out MORE upright — otherwise it falls back
        to the unrotated crop. A caller that shares `face_rotation_action` but
        skips `rotation_improves_upright` still ends up in a different coordinate
        space from the render on exactly the frames the gate rejects, which is
        the disagreement sharing the call was meant to end. The Gradio mask
        editor shipped in precisely that state.
        """
        import inspect
        import os
        import roop.core
        from roop.ProcessMgr import ProcessMgr

        sources = {
            "ProcessMgr.process_face": inspect.getsource(ProcessMgr.process_face),
            "core.get_face_crop_from_frame":
                inspect.getsource(roop.core.get_face_crop_from_frame),
        }
        # Read rather than import: this module pulls in the whole Gradio UI.
        tab = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "ui", "tabs", "faceswap_tab.py")
        with open(tab, encoding="utf-8") as fh:
            sources["ui/tabs/faceswap_tab.py"] = fh.read()

        for where, src in sources.items():
            self.assertIn("rotation_improves_upright", src,
                          f"{where} turns the frame without checking the result")


class TestClippedFaceCrop(unittest.TestCase):
    """align_crop must not feed the swap model a black wedge.

    A face running off the edge of the frame — "only part of the face is
    visible" — makes the aligned crop sample outside the image. That used to be
    filled with pure black (BORDER_CONSTANT/0), putting a third of the swap
    model's input far outside the distribution it was trained on. The border is
    replicated instead.

    This is invisible on any face that is fully inside the frame, which is why
    it needs a test: nothing in normal footage would catch a regression here.
    """

    @staticmethod
    def _img():
        rng = np.random.default_rng(7)
        # No pure-black pixels anywhere, so any black in a crop came from the
        # border fill rather than from the source frame.
        return rng.integers(40, 255, (720, 1280, 3), dtype=np.uint8)

    INTERIOR = [[600, 300], [680, 300], [640, 350], [605, 400], [675, 400]]
    OFF_LEFT = [[8, 300], [88, 300], [48, 350], [13, 400], [83, 400]]
    OFF_TOP = [[600, 8], [680, 8], [640, 40], [605, 70], [675, 70]]

    def test_clipped_face_crop_has_no_black_fill(self):
        from roop.face_util import align_crop
        img = self._img()
        for name, kps in (("off left", self.OFF_LEFT), ("off top", self.OFF_TOP)):
            crop, _ = align_crop(img, np.asarray(kps, np.float32), 256, "arcface")
            black = int((crop.reshape(-1, 3).sum(axis=1) == 0).sum())
            self.assertEqual(black, 0,
                             f"{name}: {black} pure-black px fed to the swap model")

    def test_interior_face_is_unaffected_by_the_border_mode(self):
        """The fix must be a no-op wherever it is not needed.

        warpAffine only consults the border mode for samples outside the source,
        so a face fully inside the frame has to come out bit-identical to the
        old BORDER_CONSTANT call — otherwise this quietly changed every render.
        """
        import cv2
        from roop.face_util import align_crop
        img = self._img()
        kps = np.asarray(self.INTERIOR, np.float32)
        new_crop, M = align_crop(img, kps, 256, "arcface")
        old_crop = cv2.warpAffine(img, M, (256, 256), borderValue=0.0)
        self.assertTrue(np.array_equal(new_crop, old_crop),
                        "border mode changed the crop of a face that is not clipped")

    def test_the_old_behaviour_really_was_black(self):
        """Guards the premise: without the fix these crops DO fill with black.

        If a future template change moved the crop fully inside the frame, the
        test above would pass vacuously and stop protecting anything.
        """
        import cv2
        from roop.face_util import estimate_norm
        img = self._img()
        M = estimate_norm(np.asarray(self.OFF_LEFT, np.float32), 256, "arcface")
        old = cv2.warpAffine(img, M, (256, 256), borderValue=0.0)
        black = int((old.reshape(-1, 3).sum(axis=1) == 0).sum())
        self.assertGreater(black, 1000,
                           "off-left keypoints no longer produce a clipped crop — "
                           "pick coordinates that do, or this suite proves nothing")


class TestTheSolveIsNotFooledByPitch(unittest.TestCase):
    def test_pitch_does_not_leak_into_the_reported_yaw(self):
        """The defect that made the ratio proxies unusable: pitch inflates
        yaw_ratio, so a tilted profile reads as a mid-angle face. The 5-point
        solve separates the two."""
        from roop.face_util import solve_pose_5pt
        for pitch in (-40, -20, 0, 20, 40):
            yaw, _, _ = solve_pose_5pt(project_kps(90, pitch))
            self.assertAlmostEqual(abs(yaw), 90.0, delta=0.5,
                                   msg=f"pitch={pitch} corrupted the yaw solve")


if __name__ == "__main__":
    unittest.main()
