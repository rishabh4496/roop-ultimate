"""Stage 7 in the UltraMax enhancer hook: the mouth must be restored WHERE THE MOUTH IS.

`Mask_RealityUX` was fixed on 2026-10-14 (c72e97a) to map the detector's landmarks --
FRAME pixels -- into the aligned crop before drawing the inner-lip polygon. The
UltraMax hook (`Enhance_UltraMax` -> `PipelineManager.process_enhancer_crop` ->
`AnatomicalEngine` -> `MouthDynamicsEngine.process`) drew the same frame coordinates
into the 256/512 crop, so on a face anywhere but the frame's top-left corner the
polygon was clipped onto the crop's edge and the cavity mask was empty. It was
recorded as an open owner item in RECODE_STATUS 10-14.

The 59 existing Stage 7 tests passed throughout because they hand the engine
landmarks already in crop space, which is the one case that was never broken.
Everything here uses a real frame->crop similarity, as `estimate_norm` produces.
"""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))

from roop.enhancers.anatomical_engine import AnatomicalEngine          # noqa: E402
from roop.enhancers.mouth_dynamics import (MouthDynamicsEngine,        # noqa: E402
                                           compute_mar, frame_pts_to_crop)

# frame -> crop similarity: a face ~330 px across in a 1080p frame cut into 256 px
SCALE = 0.55
M256 = np.array([[SCALE, 0.0, -420.0], [0.0, SCALE, -180.0]], np.float32)
M512 = (M256 * 2.0).astype(np.float32)


def _invert(M):
    return cv2.invertAffineTransform(np.asarray(M, np.float32))


def _lm68_in_frame(M, centre=(128.0, 172.0), half_w=22.0, half_h=9.0):
    """68 landmarks whose inner lip (60-67) is an open mouth centred at `centre` IN THE CROP,
    expressed in FRAME pixels (what the detector returns). Other points are filler."""
    cx, cy = centre
    crop = np.zeros((68, 2), np.float32)
    crop[:] = (cx, cy - 60.0)
    crop[60] = (cx - half_w, cy)
    crop[61] = (cx - half_w / 2, cy - half_h)
    crop[62] = (cx, cy - half_h)
    crop[63] = (cx + half_w / 2, cy - half_h)
    crop[64] = (cx + half_w, cy)
    crop[65] = (cx + half_w / 2, cy + half_h)
    crop[66] = (cx, cy + half_h)
    crop[67] = (cx - half_w / 2, cy + half_h)
    inv = _invert(M)
    return cv2.transform(crop[None], inv)[0].astype(np.float32), crop


class _Face(dict):
    """Stand-in for an insightface Face (a dict with attribute access)."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


def _scene(size=256, M=M256):
    """(face, swapped_crop, original_crop, mouth_region_mask) for an open-mouth face."""
    k = size / 256.0
    lm_frame, lm_crop = _lm68_in_frame(M, centre=(128.0 * k, 172.0 * k),
                                       half_w=22.0 * k, half_h=9.0 * k)
    original = np.full((size, size, 3), 110, np.uint8)
    cv2.fillPoly(original, [np.round(lm_crop[60:68]).astype(np.int32)], (235, 235, 240))  # teeth
    swapped = np.full((size, size, 3), 60, np.uint8)
    region = np.zeros((size, size), np.uint8)
    cv2.fillPoly(region, [np.round(lm_crop[60:68]).astype(np.int32)], 255)
    face = _Face(landmark_3d_68=np.hstack([lm_frame, np.zeros((68, 1), np.float32)]))
    return face, swapped, original, region > 0


class FramePointsToCrop(unittest.TestCase):

    def test_lip_corners_land_where_the_crop_has_them(self):
        face, _, _, _ = _scene()
        pts = MouthDynamicsEngine().get_inner_lip_points(face, (256, 256))
        self.assertGreater(float(pts[:, 0].max()), 256.0, "premise: raw points are frame pixels")
        mapped = frame_pts_to_crop(pts, M256)
        np.testing.assert_allclose(mapped[0], (128 - 22, 172), atol=0.01)
        np.testing.assert_allclose(mapped[4], (128 + 22, 172), atol=0.01)

    def test_no_matrix_means_crop_space_already(self):
        pts = np.array([[10.0, 20.0], [30.0, 40.0]], np.float32)
        self.assertIs(frame_pts_to_crop(pts, None), pts)

    def test_an_unusable_matrix_is_none_not_a_guess(self):
        pts = np.ones((4, 2), np.float32)
        self.assertIsNone(frame_pts_to_crop(pts, np.full((2, 3), np.nan)))
        self.assertIsNone(frame_pts_to_crop(pts, np.eye(2)))
        self.assertIsNone(frame_pts_to_crop(np.full((4, 2), np.inf), M256))

    def test_a_3x3_homogeneous_matrix_is_accepted(self):
        pts = np.array([[900.0, 500.0]], np.float32)
        full = np.vstack([M256, [0, 0, 1]])
        np.testing.assert_allclose(frame_pts_to_crop(pts, full), frame_pts_to_crop(pts, M256))


class UltraMaxHookRestoresTheRealMouth(unittest.TestCase):

    def _run(self, size, M, stamp):
        face, swapped, original, region = _scene(size, M)
        if stamp:
            face["_aligned_M"] = M
        out, tel = AnatomicalEngine().process_enhanced_crop(
            swapped, original, face, restore_original_mouth=True, mouth_open_threshold=0.18)
        return out, tel, region

    def test_mouth_interior_comes_from_the_original_when_M_is_stamped(self):
        out, tel, region = self._run(256, M256, stamp=True)
        self.assertTrue(tel["mouth_restored"], tel)
        core = cv2.erode(region.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        self.assertGreater(core.sum(), 50)
        # original teeth ~235, swapped plate 60: the cavity must be (almost) all original
        self.assertGreater(float(out[core].mean()), 200.0)

    def test_skin_outside_the_mouth_stays_the_swapped_face(self):
        out, _, region = self._run(256, M256, stamp=True)
        far = ~cv2.dilate(region.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
        self.assertLess(float(np.abs(out[far].astype(np.float32) - 60.0).max()), 1.0)

    def test_without_M_the_mask_misses_the_mouth_which_is_the_old_defect(self):
        """Frame-space polygon drawn into the crop: the restore lands on the crop edge."""
        out, tel, region = self._run(256, M256, stamp=False)
        core = cv2.erode(region.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        self.assertLess(float(out[core].mean()), 100.0,
                        "if this fails the old behaviour is no longer reproducible -- re-derive the scene")

    def test_the_same_scene_at_512_restores_the_same_fraction_of_the_mouth(self):
        o256, _, r256 = self._run(256, M256, stamp=True)
        o512, _, r512 = self._run(512, M512, stamp=True)
        f256 = float((o256[r256].astype(np.float32) > 150).mean())
        f512 = float((o512[r512].astype(np.float32) > 150).mean())
        self.assertGreater(f256, 0.8)
        self.assertAlmostEqual(f256, f512, delta=0.08)

    def test_feather_is_the_same_fraction_of_the_face_at_both_crop_sizes(self):
        """Seam width ~ proportional to the crop, i.e. invariant to face resolution."""
        widths = {}
        for size, M in ((256, M256), (512, M512)):
            face, swapped, original, region = _scene(size, M)
            eng = MouthDynamicsEngine()
            lm = frame_pts_to_crop(eng.get_inner_lip_points(face, (size, size)), M)
            feather = max(1, int(round(4 * size / 256.0)))
            mask = eng.generate_inner_mouth_mask(original, lm, feather_radius=feather)
            row = mask[int(172 * size / 256.0)]
            ramp = np.flatnonzero((row > 0.05) & (row < 0.95))
            widths[size] = len(ramp) / size
        self.assertAlmostEqual(widths[256], widths[512], delta=0.012)

    def test_a_closed_mouth_is_left_alone(self):
        face, swapped, original, _ = _scene()
        lm = np.asarray(face["landmark_3d_68"]).copy()
        # collapse the vertical opening: upper == lower lip
        lm[65:68, 1] = lm[[63, 62, 61], 1]
        face["landmark_3d_68"] = lm
        self.assertLess(compute_mar(lm), 0.18)
        face["_aligned_M"] = M256
        out, tel = AnatomicalEngine().process_enhanced_crop(
            swapped, original, face, restore_original_mouth=True, mouth_open_threshold=0.18)
        self.assertFalse(tel["mouth_restored"])
        np.testing.assert_array_equal(out, swapped)


class ProcessMgrStampsTheMatrix(unittest.TestCase):
    """The enhancer reads `_aligned_M` off the face; ProcessMgr must set it beside
    `aligned_img` (same try, same space) and clear it after the enhancer returns."""

    def test_stamp_and_clear_sit_with_aligned_img(self):
        src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "app", "roop", "ProcessMgr.py"), encoding="utf-8").read()
        i = src.index("target_face.aligned_img = _ctx if _use_ctx else aligned_img")
        self.assertIn("target_face._aligned_M = _M_e if _use_ctx else M", src[i:i + 600])
        j = src.index("target_face.aligned_img = None", i)
        self.assertIn("target_face._aligned_M = None", src[j:j + 200])


if __name__ == "__main__":
    unittest.main()
