"""The oral-cavity restore must read the INNER lip ring of InsightFace's 106
landmarks, never the outer upper edge.

2026-09-27, Love.mp4: the landmark fallback took indices 66..71, three of which
(67, 68, 71) sit on the OUTER upper-lip edge. Its hull was the upper-lip
vermilion, a closed mouth measured "open", and every frame pasted the target's
own upper lip over the swap: "upper lip colour does not match the faceset".
The geometry below is laid out the way 2d106det places the points on a real
face (plotted on frame 3024).
"""
import os
import sys
import unittest

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import oral_cavity as oc  # noqa: E402

CX, CY, HALF = 100.0, 100.0, 20.0     # mouth centre and half-width in pixels


def _face(inner_gap):
    """A 106-point face whose lips are 8px thick; `inner_gap` px between them."""
    pts = np.zeros((106, 2), np.float32)
    top = CY - inner_gap / 2.0          # inner edge of the upper lip
    bot = CY + inner_gap / 2.0          # inner edge of the lower lip

    def at(i, fx, y):
        pts[i] = (CX + fx * HALF, y)

    # outer ring: corners, upper edge 8px above the inner upper lip, lower edge
    at(52, -1.0, CY); at(61, 1.0, CY)
    for i, fx in zip((64, 63, 71, 67, 68), (-0.7, -0.4, -0.15, 0.1, 0.5)):
        at(i, fx, top - 8.0)
    for i, fx in zip((58, 59, 53, 56, 55), (0.7, 0.4, 0.0, -0.4, -0.7)):
        at(i, fx, bot + 8.0)
    # inner ring: upper 65 66 62 70 69 / lower 57 60 54
    for i, fx in zip((65, 66, 62, 70, 69), (-0.85, -0.5, 0.0, 0.5, 0.85)):
        at(i, fx, top)
    for i, fx in zip((54, 60, 57), (-0.5, 0.0, 0.5)):
        at(i, fx, bot)

    class F:
        landmark_2d_106 = pts
        bbox = np.array([CX - 60, CY - 70, CX + 60, CY + 50], np.float32)
    return F()


FRAME = np.zeros((220, 220, 3), np.uint8)


class InnerLipRing(unittest.TestCase):
    def test_closed_mouth_is_not_open(self):
        _, m = oc.detect_oral_cavity_mask(FRAME, _face(inner_gap=1.0))
        self.assertFalse(m['is_open'])

    def test_closed_mouth_restore_is_a_no_op(self):
        swapped = np.full_like(FRAME, 200)
        plate = np.full_like(FRAME, 40)
        out = oc.reconstruct_inner_mouth_geometry(swapped, plate, _face(inner_gap=1.0))
        self.assertTrue(np.array_equal(out, swapped))

    def test_open_mouth_mask_stays_between_the_lips(self):
        face = _face(inner_gap=12.0)
        mask, m = oc.detect_oral_cavity_mask(FRAME, face)
        self.assertTrue(m['is_open'])
        self.assertGreater(mask[int(CY), int(CX)], 0)          # the aperture
        upper_lip_y = int(CY - 6.0 - 4.0)                      # inside the upper lip
        self.assertEqual(mask[upper_lip_y, int(CX)], 0)

    def _open_mouth_frames(self, swap_flat):
        rng = np.random.default_rng(0)
        plate = np.full((220, 220, 3), 90, np.uint8)
        teeth = rng.integers(60, 250, (40, 60, 3), dtype=np.uint8)
        plate[80:120, 70:130] = teeth                        # textured open mouth
        swapped = plate.copy()
        if swap_flat:
            swapped[80:120, 70:130] = 110                    # collapsed: a flat line
        else:
            swapped[80:120, 70:130] = rng.integers(60, 250, (40, 60, 3), dtype=np.uint8)
        return plate, swapped

    def test_a_good_swapped_mouth_is_left_alone(self):
        """weeds.mp4: pasting the plate's teeth over good swapped teeth gave
        grey, mottled, doubled teeth. 0 of 174 open mouths needed repair."""
        plate, swapped = self._open_mouth_frames(swap_flat=False)
        out = oc.reconstruct_inner_mouth_geometry(swapped, plate, _face(inner_gap=12.0))
        self.assertTrue(np.array_equal(out, swapped))

    def test_a_collapsed_swapped_mouth_is_repaired(self):
        plate, swapped = self._open_mouth_frames(swap_flat=True)
        out = oc.reconstruct_inner_mouth_geometry(swapped, plate, _face(inner_gap=12.0))
        self.assertFalse(np.array_equal(out, swapped))

    def test_the_outer_upper_edge_is_never_read(self):
        # Moving the outer upper lip must not change the mask.
        a, _ = oc.detect_oral_cavity_mask(FRAME, _face(inner_gap=12.0))
        f = _face(inner_gap=12.0)
        for i in (64, 63, 71, 67, 68):
            f.landmark_2d_106[i, 1] -= 15.0
        b, _ = oc.detect_oral_cavity_mask(FRAME, f)
        self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
