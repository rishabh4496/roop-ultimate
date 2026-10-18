"""Stage 8 geometry: faces cut by the frame border, and heads rolled +-60 degrees.

No models. These pin the arithmetic that every detector result then flows through:
the ROI window, the alignment warp and its border mode.

What is and is not true of the shipped code (checked 2026-10-19, see CHANGELOG):

  * `_roi_window` clips to the frame, so a box that hangs 40% off any edge never
    yields a negative origin or an empty / out-of-range slice.
  * `align_crop` samples outside the frame for such a face and fills with
    BORDER_REPLICATE, not black. The comment there records a measured 34% black
    wedge with the default border; nothing pinned it.
  * A similarity alignment removes in-plane roll exactly: a head rolled +-60 degrees
    lands on the template keypoints, with no shear (the matrix is a pure similarity).
  * The inverse warp back to the frame cannot write the synthetic border: those
    pixels map to coordinates outside the frame canvas.
"""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import face_util                                        # noqa: E402
from roop.face_util import align_crop, estimate_norm, swap_template_points   # noqa: E402

W, H = 640, 480
FACE = 200.0                       # face box edge, pixels

_TEMPLATE = np.asarray(swap_template_points(256, "arcface"), np.float64)


def _kps(cx, cy, roll_deg=0.0, size=FACE):
    """The ArcFace template put through a known similarity: `size` px across, rolled
    `roll_deg`, centred at (cx, cy). An alignment must therefore recover the template
    EXACTLY (an arbitrary 5-point layout can only be fitted in the least-squares sense)."""
    t = np.radians(roll_deg)
    r = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    centred = (_TEMPLATE - _TEMPLATE.mean(axis=0)) * (size / 256.0)
    return (centred @ r.T + [cx, cy]).astype(np.float32)


def _box(cx, cy, size=FACE):
    return (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)


# face centre such that 40% of the box is beyond each edge
CLIPPED = {
    "left":   _box(-0.1 * FACE, H / 2),
    "right":  _box(W + 0.1 * FACE, H / 2),
    "top":    _box(W / 2, -0.1 * FACE),
    "bottom": _box(W / 2, H + 0.1 * FACE),
}
CENTRES = {"left": (-0.1 * FACE, H / 2), "right": (W + 0.1 * FACE, H / 2),
           "top": (W / 2, -0.1 * FACE), "bottom": (W / 2, H + 0.1 * FACE)}


class RoiWindowAtTheBorder(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(3)
        self.frame = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)

    def test_a_face_cut_by_40_percent_at_each_edge_gets_a_valid_in_frame_window(self):
        for edge, box in CLIPPED.items():
            for pad in (0.55, 1.0, 1.35):
                crop, x0, y0 = face_util._roi_window(self.frame, box, pad, 160)
                h, w = crop.shape[:2]
                self.assertGreater(h, 0, edge)
                self.assertGreater(w, 0, edge)
                self.assertGreaterEqual(x0, 0, edge)
                self.assertGreaterEqual(y0, 0, edge)
                self.assertLessEqual(x0 + w, W, edge)
                self.assertLessEqual(y0 + h, H, edge)
                np.testing.assert_array_equal(crop, self.frame[y0:y0 + h, x0:x0 + w])

    def test_the_visible_part_of_the_face_is_inside_the_window(self):
        for edge, (x1, y1, x2, y2) in CLIPPED.items():
            crop, x0, y0 = face_util._roi_window(self.frame, (x1, y1, x2, y2), 1.0, 160)
            h, w = crop.shape[:2]
            vx1, vy1, vx2, vy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            self.assertLessEqual(x0, vx1 + 1, edge)
            self.assertLessEqual(y0, vy1 + 1, edge)
            self.assertGreaterEqual(x0 + w, vx2 - 1, edge)
            self.assertGreaterEqual(y0 + h, vy2 - 1, edge)

    def test_a_box_entirely_outside_the_frame_is_refused_not_sliced(self):
        self.assertIsNone(face_util._roi_window(self.frame, _box(-400, 100), 0.0, 1))
        self.assertIsNone(face_util._roi_window(self.frame, (10, 10, 10, 90), 1.0, 160))

    def test_offsetting_a_result_back_to_frame_space_is_exact(self):
        class _F:
            pass
        for edge, (x1, y1, x2, y2) in CLIPPED.items():
            crop, x0, y0 = face_util._roi_window(self.frame, (x1, y1, x2, y2), 1.0, 160)
            f = _F()
            f.bbox = np.array([5.0, 6.0, 55.0, 66.0], np.float32)
            f.kps = np.array([[10, 10], [40, 10], [25, 30], [15, 50], [35, 50]], np.float32)
            face_util._offset_face_coords(f, x0, y0)
            np.testing.assert_allclose(f.bbox, [5 + x0, 6 + y0, 55 + x0, 66 + y0])
            np.testing.assert_allclose(f.kps[2], [25 + x0, 30 + y0])


class AlignmentOfAClippedFace(unittest.TestCase):

    SIZE = 256

    def _plate(self):
        # no black anywhere: a smooth mid-grey ramp, so any pure-black pixel is a border fill
        ramp = np.tile(np.linspace(90, 180, W, dtype=np.float32), (H, 1))
        return np.dstack([ramp, ramp, ramp]).astype(np.uint8)

    def test_the_border_pixels_are_the_frame_edge_not_black(self):
        plate = self._plate()
        for edge, (cx, cy) in CENTRES.items():
            kps = _kps(cx, cy)
            crop, M = align_crop(plate, kps, self.SIZE, mode="arcface")
            black = float((crop.max(axis=2) < 40).mean())
            self.assertLess(black, 0.001, "%s: %.1f%% of the crop is black" % (edge, 100 * black))
            const = cv2.warpAffine(plate, M, (self.SIZE, self.SIZE), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            self.assertGreater(float((const.max(axis=2) < 40).mean()), 0.10,
                               "%s: premise -- the face really does leave the frame" % edge)

    def test_the_clipped_faces_keypoints_still_land_on_the_template(self):
        """Negative / out-of-range keypoint coordinates are handled by the translation
        column of M, so a face half off the frame aligns exactly like one in the middle."""
        dst = swap_template_points(self.SIZE, "arcface")
        for edge, (cx, cy) in CENTRES.items():
            kps = _kps(cx, cy)
            _, M = align_crop(self._plate(), kps, self.SIZE, mode="arcface")
            mapped = cv2.transform(kps[None].astype(np.float32), M)[0]
            np.testing.assert_allclose(mapped, dst, atol=1.0, err_msg=edge)


class RollPlusMinusSixty(unittest.TestCase):

    SIZE = 256

    def test_alignment_removes_the_roll_and_does_not_shear(self):
        dst = swap_template_points(self.SIZE, "arcface")
        for roll in (60.0, -60.0, 0.0):
            kps = _kps(W / 2, H / 2, roll)
            M = estimate_norm(kps, self.SIZE, mode="arcface")
            mapped = cv2.transform(kps[None].astype(np.float32), M)[0]
            np.testing.assert_allclose(mapped, dst, atol=1.0, err_msg="roll %+d" % roll)
            sv = np.linalg.svd(np.asarray(M)[:, :2], compute_uv=False)
            self.assertAlmostEqual(float(sv[0] / sv[1]), 1.0, places=3,
                                   msg="roll %+d: a similarity must not stretch one axis" % roll)

    def test_roll_is_recovered_from_the_matrix(self):
        for roll in (60.0, -60.0):
            M = estimate_norm(_kps(W / 2, H / 2, roll), self.SIZE, mode="arcface")
            angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
            self.assertAlmostEqual(abs(angle), abs(roll), delta=1.0)
            self.assertGreater(angle * roll, 0, "sign: the warp turns the face back by its roll")

    def test_a_clipped_and_rolled_face_aligns_too(self):
        dst = swap_template_points(self.SIZE, "arcface")
        for roll in (60.0, -60.0):
            for edge, (cx, cy) in CENTRES.items():
                kps = _kps(cx, cy, roll)
                M = estimate_norm(kps, self.SIZE, mode="arcface")
                mapped = cv2.transform(kps[None].astype(np.float32), M)[0]
                np.testing.assert_allclose(mapped, dst, atol=1.0, err_msg="%s roll %+d" % (edge, roll))


class InverseWarpCannotWriteTheSyntheticBorder(unittest.TestCase):
    """Pixels the alignment invented (outside the frame) map back to coordinates
    outside the canvas, so pasting the crop onto the frame never writes them."""

    def test_paste_back_only_touches_pixels_that_exist(self):
        size = 256
        for edge, (cx, cy) in CENTRES.items():
            kps = _kps(cx, cy)
            M = estimate_norm(kps, size, mode="arcface")
            ones = np.full((size, size), 255, np.uint8)
            canvas = cv2.warpAffine(ones, cv2.invertAffineTransform(M), (W, H),
                                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            self.assertEqual(canvas.shape, (H, W))
            self.assertGreater(int((canvas > 0).sum()), 0, edge)
            # what the crop covers of the frame is only the in-frame part of the face
            ys, xs = np.nonzero(canvas)
            self.assertGreaterEqual(xs.min(), 0)
            self.assertLess(xs.max(), W)
            self.assertGreaterEqual(ys.min(), 0)
            self.assertLess(ys.max(), H)


if __name__ == "__main__":
    unittest.main()
