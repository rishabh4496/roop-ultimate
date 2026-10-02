"""AdaFace reads a crop built from the DETECTOR's keypoints, not the 68-landmark-refined ones.

Measured on 16 real clips (docs/development/RECOGNIZER_CALIBRATION.md): aligning from the refined
`face.kps` costs AdaFace AUC 0.9830 vs 0.9883 and raises false accepts at w600k's false-reject rate from
8.5% to 19.0%. buffalo_l embeds before the refinement so w600k never saw the problem. These tests pin the
properties that make the fix safe: the crop is taken BEFORE the refinement, only when AdaFace is on, under
its own key (the existing source-crop key is the swap input of BlendSwap/UniFace and must keep following the
refined points), preferred by AdaFace, and released once the embedding is cached.
"""
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import roop.face_util as fu
    import roop.recognizer_adaface as ada
    import roop.globals
    _IMPORT_ERROR = None
except ImportError as exc:                       # light profile
    _IMPORT_ERROR = exc


def setUpModule():
    if _IMPORT_ERROR is not None:
        raise unittest.SkipTest(f"face stack not importable here: {_IMPORT_ERROR}")


class Face(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


DET_KPS = np.array([[40, 52], [74, 51], [57, 72], [43, 93], [72, 92]], np.float32)


def frame():
    rng = np.random.RandomState(0)
    return cv2_blur(rng.randint(0, 256, (200, 200, 3), dtype=np.uint8))


def cv2_blur(img):
    import cv2
    return cv2.GaussianBlur(img, (0, 0), 2.0)


def make_face(kps=DET_KPS, with_68=True):
    f = Face(bbox=np.array([20, 20, 120, 130], np.float32), kps=kps.copy())
    if with_68:
        lm = np.zeros((68, 3), np.float32)
        lm[36:42, :2] = [46, 56]      # left eye
        lm[42:48, :2] = [80, 55]      # right eye
        lm[30, :2] = [63, 76]         # nose tip
        lm[48, :2] = [47, 97]         # mouth corners
        lm[54, :2] = [78, 96]
        f['landmark_3d_68'] = lm
    return f


class _AdaOn:
    """Context: AdaFace enabled (the env flag is read at import, so patch the module attribute)."""

    def __init__(self, on=True):
        self._p = mock.patch.object(ada, "_ENABLED", on)

    def __enter__(self):
        self._p.start()

    def __exit__(self, *a):
        self._p.stop()


class TestStash(unittest.TestCase):
    def test_off_by_default_costs_nothing_and_adds_no_key(self):
        faces = [make_face()]
        with _AdaOn(False):
            fu._stash_recognition_crops(frame(), faces)
        self.assertNotIn(fu.REC_CROP_KEY, faces[0])

    def test_crop_is_the_detector_keypoint_alignment(self):
        img, faces = frame(), [make_face()]
        with _AdaOn(True):
            fu._stash_recognition_crops(img, faces)
        expected, _ = fu.align_crop(img, DET_KPS, 112, mode="arcface_112_v2")
        np.testing.assert_array_equal(faces[0][fu.REC_CROP_KEY], expected)
        self.assertEqual(faces[0][fu.REC_CROP_KEY].shape, (112, 112, 3))

    def test_a_face_without_keypoints_is_skipped_not_an_error(self):
        faces = [Face(bbox=np.zeros(4, np.float32)), make_face()]
        with _AdaOn(True):
            fu._stash_recognition_crops(frame(), faces)
        self.assertNotIn(fu.REC_CROP_KEY, faces[0])
        self.assertIn(fu.REC_CROP_KEY, faces[1])

    def test_one_bad_face_does_not_stop_the_rest(self):
        bad = make_face()
        bad['kps'] = np.zeros((3, 3), np.float32)               # malformed
        good = make_face()
        with _AdaOn(True):
            fu._stash_recognition_crops(frame(), [bad, good])
        self.assertIn(fu.REC_CROP_KEY, good)

    def test_the_two_key_names_cannot_drift_apart(self):
        self.assertEqual(fu.REC_CROP_KEY, ada._REC_CROP_KEY)
        self.assertNotEqual(fu.REC_CROP_KEY, ada._CROP_KEY)


class TestOrdering(unittest.TestCase):
    """The crop has to come from the keypoints as they were BEFORE the refinement overwrites them."""

    def enrich(self, img, face, refine=True):
        with _AdaOn(True), mock.patch.object(roop.globals, "refine_landmarks", refine), \
                mock.patch.object(fu, "UPRIGHT_REMEASURE", False), \
                mock.patch.object(fu, "_lm68_should_measure", return_value=False), \
                mock.patch.object(fu.face_contact, "suppress_merged", side_effect=lambda f: (f, 0)), \
                mock.patch.object(fu.face_contact, "stamp_contamination"):
            return fu._enrich_detected_faces(img, [face])[0]

    def test_refinement_moves_kps_but_the_recognition_crop_keeps_the_detectors(self):
        img, face = frame(), make_face()
        out = self.enrich(img, face)
        self.assertFalse(np.allclose(out["kps"], DET_KPS), "the refinement should have moved the keypoints")
        detector_crop, _ = fu.align_crop(img, DET_KPS, 112, mode="arcface_112_v2")
        refined_crop, _ = fu.align_crop(img, out["kps"], 112, mode="arcface_112_v2")
        np.testing.assert_array_equal(out[fu.REC_CROP_KEY], detector_crop)
        self.assertGreater(float(np.abs(detector_crop.astype(int) - refined_crop.astype(int)).mean()), 0.5,
                           "the two crops must actually differ or this test proves nothing")

    def test_with_refinement_off_nothing_is_stashed(self):
        out = self.enrich(frame(), make_face(), refine=False)
        np.testing.assert_array_equal(out["kps"], DET_KPS)
        self.assertNotIn(fu.REC_CROP_KEY, out)

    def test_a_later_shift_of_the_keypoints_cannot_make_the_crop_stale(self):
        """ROI detection shifts kps after the fact; a crop is an image, so it is unaffected."""
        img, face = frame(), make_face()
        out = self.enrich(img, face)
        before = out[fu.REC_CROP_KEY].copy()
        fu._offset_face_coords(out, 500.0, 300.0)
        np.testing.assert_array_equal(out[fu.REC_CROP_KEY], before)


class TestAdaFacePrefersIt(unittest.TestCase):
    def setUp(self):
        self.seen = []
        p = mock.patch.object(ada, "embed_crop", side_effect=lambda c: (self.seen.append(c.copy()) or np.ones(512, np.float32)))
        p.start()
        self.addCleanup(p.stop)

    def crop(self, value):
        return np.full((112, 112, 3), value, np.uint8)

    def test_detector_crop_beats_the_refined_one(self):
        face = make_face()
        face[ada._CROP_KEY], face[ada._REC_CROP_KEY] = self.crop(10), self.crop(200)
        ada.face_embedding(face)
        self.assertEqual(int(self.seen[0][0, 0, 0]), 200)

    def test_falls_back_to_the_refined_crop_then_to_aligning_from_the_frame(self):
        face = make_face()
        face[ada._CROP_KEY] = self.crop(10)
        ada.face_embedding(face)
        self.assertEqual(int(self.seen[0][0, 0, 0]), 10)
        bare = make_face()
        img = frame()
        ada.face_embedding(bare, img)
        expected, _ = fu.align_crop(img, bare["kps"], 112, mode=ada.ALIGN_MODE)
        np.testing.assert_array_equal(self.seen[1], expected)

    def test_no_crop_and_no_frame_is_still_none(self):
        self.assertIsNone(ada.face_embedding(make_face()))
        self.assertEqual(self.seen, [])

    def test_the_big_crop_is_released_once_the_embedding_is_cached(self):
        face = make_face()
        face[ada._REC_CROP_KEY] = self.crop(200)
        first = ada.face_embedding(face)
        self.assertNotIn(ada._REC_CROP_KEY, face)
        self.assertIs(ada.face_embedding(face), first)               # cached; the crop is not needed again
        self.assertEqual(len(self.seen), 1)

    def test_the_refined_source_crop_is_not_released(self):
        """That key is the swap input of image-source swap models; it must survive."""
        face = make_face()
        face[ada._CROP_KEY] = self.crop(10)
        ada.face_embedding(face)
        self.assertIn(ada._CROP_KEY, face)

    def test_a_failed_embedding_keeps_the_crop_so_a_retry_is_possible(self):
        face = make_face()
        face[ada._REC_CROP_KEY] = self.crop(200)
        with mock.patch.object(ada, "embed_crop", side_effect=RuntimeError("session died")):
            self.assertIsNone(ada.face_embedding(face))
        self.assertIn(ada._REC_CROP_KEY, face)


class TestSourceCropKeyUntouched(unittest.TestCase):
    def test_attach_source_crops_still_follows_the_refined_keypoints_and_adds_no_rec_key(self):
        img, face = frame(), make_face()
        refined = DET_KPS + 3.0
        face["kps"] = refined
        fu._attach_source_crops(face, img)
        expected, _ = fu.align_crop(img, refined, 112, mode="arcface_112_v2")
        np.testing.assert_array_equal(face["_src_crop_arcface_112_v2"], expected)
        self.assertNotIn(fu.REC_CROP_KEY, face)


if __name__ == "__main__":
    unittest.main()
