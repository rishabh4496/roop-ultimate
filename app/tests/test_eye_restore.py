"""apply_eyes_area — bringing the target's own eyes back over the swap.

Properties, not pixel values. What must hold:

  * OFF IS FREE. At strength 0, or with no keypoints, the frame comes back
    bit-identical. This one is load-bearing: the call sits in the per-face
    paste path that every frame of every render goes through, so a version
    that "mostly" leaves the frame alone would put a floor under the whole
    pipeline's output and be invisible until someone diffed two renders.

  * IT ACTS WHERE THE EYES ARE. Pixels at the eye keypoints move toward the
    plate; pixels far from them do not move at all. An ellipse centred on the
    wrong keypoint, or radii computed off the wrong axis, still "works" — it
    just restores the cheek — and nothing else in the pipeline would complain.

  * IT SCALES WITH THE FACE. Radii are fractions of the interocular distance,
    so the same settings must cover the same proportion of a face at any size
    on screen. A pixel-denominated version looks fine on the clip it was tuned
    on and wrong on every other.

  * IT STANDS DOWN OFF-AXIS. Past ~38 deg of yaw or pitch the far eye is
    edge-on and the near one sits over the nose bridge; pasting the plate
    there doubles the socket, so the fade must reach exactly zero rather than
    merely getting small.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.ProcessMgr import ProcessMgr                  # noqa: E402


class _Face:
    def __init__(self, kps):
        self.kps = np.asarray(kps, dtype=np.float32)


class _Options:
    show_face_area_overlay = False


class _Mgr:
    """Just enough ProcessMgr to call the method under test."""
    options = _Options()
    apply_eyes_area = ProcessMgr.apply_eyes_area
    apply_color_transfer = ProcessMgr.apply_color_transfer


def scene(w=400, h=300, eye_y=120, eye_dx=60):
    """A grey 'swapped' frame and a white 'plate', with eyes either side."""
    swapped = np.full((h, w, 3), 60, dtype=np.uint8)
    plate = np.full((h, w, 3), 200, dtype=np.uint8)
    cx = w // 2
    face = _Face([[cx - eye_dx / 2, eye_y], [cx + eye_dx / 2, eye_y],
                  [cx, eye_y + 30], [cx - 20, eye_y + 55], [cx + 20, eye_y + 55]])
    return swapped, plate, face, (cx, eye_y, eye_dx)


class OffIsFree(unittest.TestCase):
    def test_zero_strength_is_bit_identical(self):
        swapped, plate, face, _ = scene()
        before = swapped.copy()
        out = _Mgr().apply_eyes_area(swapped, plate, face, strength=0.0)
        np.testing.assert_array_equal(out, before)

    def test_a_face_without_keypoints_is_bit_identical(self):
        swapped, plate, _, _ = scene()
        before = swapped.copy()
        out = _Mgr().apply_eyes_area(swapped, plate, _Face([]), strength=1.0)
        np.testing.assert_array_equal(out, before)

    def test_a_collapsed_eye_separation_is_bit_identical(self):
        """A true profile: both eye points land together. There is no second
        eye to restore and anything pasted lands on the side of the nose."""
        swapped, plate, _, _ = scene()
        before = swapped.copy()
        face = _Face([[200, 120], [200.5, 120], [205, 150], [195, 175], [210, 175]])
        out = _Mgr().apply_eyes_area(swapped, plate, face, strength=1.0)
        np.testing.assert_array_equal(out, before)


class ItActsOnTheEyes(unittest.TestCase):
    def test_the_eye_centres_move_toward_the_plate(self):
        swapped, plate, face, (cx, eye_y, eye_dx) = scene()
        out = _Mgr().apply_eyes_area(swapped.copy(), plate, face, strength=1.0)
        for sign in (-1, 1):
            x = int(cx + sign * eye_dx / 2)
            self.assertGreater(
                int(out[eye_y, x, 0]), 120,
                'the pixel at an eye keypoint should be most of the way to the '
                'plate at full strength')

    def test_far_from_the_eyes_nothing_moves(self):
        swapped, plate, face, _ = scene()
        out = _Mgr().apply_eyes_area(swapped.copy(), plate, face, strength=1.0)
        for pt in [(10, 10), (290, 390), (250, 200)]:
            self.assertEqual(
                int(out[pt[0], pt[1], 0]), 60,
                f'{pt} is far from either eye and must be untouched')

    def test_strength_is_monotonic(self):
        swapped, plate, face, (cx, eye_y, eye_dx) = scene()
        x = int(cx - eye_dx / 2)
        vals = [int(_Mgr().apply_eyes_area(swapped.copy(), plate, face, strength=s)[eye_y, x, 0])
                for s in (0.25, 0.5, 1.0)]
        self.assertEqual(vals, sorted(vals),
                         'more strength must mean more of the plate, not less')

    def test_the_region_scales_with_the_face(self):
        """Same settings, a face twice as wide — the restored region must be
        about twice as wide too, because the radii are fractions of the
        interocular distance rather than pixel counts."""
        widths = []
        for dx in (60, 120):
            swapped, plate, face, (cx, eye_y, _) = scene(eye_dx=dx)
            out = _Mgr().apply_eyes_area(swapped.copy(), plate, face, strength=1.0)
            row = out[eye_y, :, 0].astype(int)
            widths.append(int((row > 61).sum()))
        self.assertGreater(widths[1], widths[0] * 1.6,
                           f'region did not scale with the face: {widths}')


class ItStandsDownOffAxis(unittest.TestCase):
    def test_past_the_fade_window_nothing_moves(self):
        swapped, plate, face, _ = scene()
        before = swapped.copy()
        for yaw, pitch in ((45.0, 0.0), (0.0, 60.0), (-52.0, 0.0)):
            out = _Mgr().apply_eyes_area(swapped.copy(), plate, face,
                                         strength=1.0, yaw=yaw, pitch=pitch)
            np.testing.assert_array_equal(
                out, before,
                f'yaw={yaw} pitch={pitch} is past the fade window and must be a no-op')

    def test_the_fade_is_gradual_not_a_cliff(self):
        """A hard cutoff pops between two looks on a head parked near the
        threshold — the same reason the mask router had to be given a latch."""
        swapped, plate, face, (cx, eye_y, eye_dx) = scene()
        x = int(cx - eye_dx / 2)
        vals = [int(_Mgr().apply_eyes_area(swapped.copy(), plate, face,
                                           strength=1.0, yaw=y)[eye_y, x, 0])
                for y in (20.0, 28.0, 34.0)]
        self.assertEqual(vals, sorted(vals, reverse=True),
                         f'restore should taper as the head turns, got {vals}')
        self.assertGreater(vals[0], vals[2], 'the fade did not engage at all')


if __name__ == '__main__':
    unittest.main()
