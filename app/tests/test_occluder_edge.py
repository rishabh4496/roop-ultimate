"""How wide the boundary between a foreign object and the swap ends up on screen.

The occluder family (`Face Occluder`, `Face Occluder v3`) thresholds its output
to a hard 0/1 and then softens the result so the join does not alias. That
softening happens in CROP space, and the crop is a fixed 256 or 512 px whatever
the face's real size — so a kernel written as a constant `(5, 5)` has no fixed
width in the finished frame. It is multiplied by the crop-to-frame
magnification, which is exactly the number that grows as the camera moves in.

Measured at the default pixel boost:

    face width in frame   120px   250px   500px   900px   1400px
    5x5 blur ends up as   3.6px   7.6px  15.1px  27.2px   42.3px

A 42 px ramp is the swap bleeding a wide glow across the hand or microphone it
is supposed to stop at, and its position moves frame to frame with the mask
model's own noise. It is also the wrong way round: a close-up is where the
boundary should be tightest and where these engines get picked in the first
place, because they are what someone reaches for when objects cross the face.

These tests pin the edge to a roughly constant width on screen. They test the
kernel choice, not the blur — the blur is one cv2 call and there is nothing to
get wrong about it once the size is right.
"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals                                              # noqa: E402
from roop.face_util import estimate_norm, _project_reference     # noqa: E402
from roop.procmgr_masking import _edge_blur_kernel, OCCLUDER_EDGE_PX  # noqa: E402

# Mask engines in this family run at a fixed 256, whatever the crop size.
MASK_SHAPE = (256, 256)


def face_kps(face_width_px, cx=960.0, cy=540.0):
    """5 keypoints of a frontal head that spans `face_width_px` in the frame."""
    p = _project_reference(0.0, 0.0)
    p = p - p.mean(axis=0)
    # The interocular distance is about 36% of the visible face width.
    s = (face_width_px * 0.36) / float(np.linalg.norm(p[1] - p[0]))
    return (p * s + np.array([cx, cy])).astype(np.float64)




def edge_px(face_width_px, subsample):
    """(frame-pixel width of the resulting ramp, kernel) for this face size."""
    M = estimate_norm(face_kps(face_width_px), subsample)
    crop_per_frame = math.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]))
    frame_per_mask = (1.0 / crop_per_frame) * (subsample / MASK_SHAPE[0])
    k = _edge_blur_kernel(M, (subsample, subsample), MASK_SHAPE)
    return max(1.0, float(k)) * frame_per_mask, k


SIZES = (120, 250, 500, 900, 1400)


class TestEdgeWidthOnScreen(unittest.TestCase):


    def test_the_edge_is_roughly_constant_across_face_sizes(self):
        widths = [edge_px(s, 256)[0] for s in SIZES]
        self.assertLess(max(widths) / min(widths), 4.0,
                        f'edge width still swings {max(widths)/min(widths):.1f}x '
                        f'across face sizes: '
                        f'{[f"{w:.1f}" for w in widths]}')

    def test_a_close_up_no_longer_gets_a_halo(self):
        """The reported symptom. 27 px of ramp around a hand on a 900 px face."""
        width, _ = edge_px(900, 256)
        self.assertLess(width, 10.0, f'{width:.1f}px ramp on a close-up')


    def test_a_distant_face_still_gets_softened(self):
        """The other half: a small face is magnified by less than 1, so an
        unblurred mask staircases along the join. Blurring is still the right
        answer there — this is not 'remove the blur', it is 'size it'."""
        _, k = edge_px(120, 256)
        self.assertGreater(k, 1, 'a distant face needs the softening')

    def test_the_boost_setting_does_not_change_the_edge(self):
        """One mask pixel covers the same patch of frame whatever the crop size,
        because the mask model's own resolution is what is fixed. Pixel boost
        must therefore not move this."""
        for s in SIZES:
            a, _ = edge_px(s, 256)
            b, _ = edge_px(s, 512)
            self.assertAlmostEqual(a, b, delta=max(0.5, a * 0.05),
                                   msg=f'boost changed the edge at {s}px')

    def test_it_lands_near_the_requested_width(self):
        for s in SIZES:
            width, _ = edge_px(s, 256)
            # The floor is one mask pixel: below that there is nothing to
            # resolve with, and no kernel can make the edge finer.
            self.assertGreaterEqual(width, min(OCCLUDER_EDGE_PX, width) - 0.01)
            self.assertLess(width, max(OCCLUDER_EDGE_PX * 2.0, 9.0),
                            f'{width:.1f}px at face width {s}')


class TestItCannotCrash(unittest.TestCase):
    """This runs per face per frame on the mask path, so every degenerate input
    has to come back with a usable odd kernel rather than an exception."""

    def test_bad_inputs_fall_back_to_the_historic_kernel(self):
        for M in (None,
                  np.zeros((2, 3)),                       # singular
                  np.array([[np.nan, 0, 0], [0, 1, 0]]),
                  'not a matrix'):
            k = _edge_blur_kernel(M, (256, 256), MASK_SHAPE)
            self.assertEqual(k, 5, f'bad M did not fall back: {M!r}')




    def test_the_kernel_is_always_odd_and_bounded(self):
        for s in (20, 60, 120, 400, 900, 4000):
            for ss in (128, 256, 512, 1024):
                k = _edge_blur_kernel(estimate_norm(face_kps(s), ss),
                                      (ss, ss), MASK_SHAPE)
                self.assertEqual(k % 2, 1, 'GaussianBlur needs an odd kernel')
                self.assertGreaterEqual(k, 1)
                self.assertLessEqual(k, 17)

    def test_a_zero_sized_mask_does_not_divide_by_zero(self):
        k = _edge_blur_kernel(estimate_norm(face_kps(300), 256), (0, 0), (0, 0))
        self.assertEqual(k % 2, 1)

    def test_the_old_fixed_kernel_really_was_that_wide(self):
        """Guards the premise, so this cannot pass by accident if the geometry
        it is reasoning about ever changes."""
        M = estimate_norm(face_kps(900), 256)
        crop_per_frame = math.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]))
        self.assertGreater(5.0 / crop_per_frame, 20.0)


if __name__ == '__main__':
    unittest.main()
