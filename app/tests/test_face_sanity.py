"""face_sanity: every Face the analyser returns is either usable or visibly dropped.

Uses InsightFace's own Face, because its `__getattr__` (None for a missing key instead of
AttributeError) is the reason `getattr(face, 'det_score', 1.0)` is None and `None >= 0.65`
raised TypeError at three sites before a8df052.
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insightface.app.common import Face                       # noqa: E402

from roop import face_sanity                                   # noqa: E402
from roop.face_sanity import sanitize_face_attributes, sanitize_faces  # noqa: E402


def _face(**over):
    rng = np.random.RandomState(0)
    data = dict(
        bbox=np.array([10, 20, 110, 140], np.float32),
        kps=rng.rand(5, 2).astype(np.float32) * 100,
        det_score=0.91,
        landmark_2d_106=rng.rand(106, 2).astype(np.float32) * 100,
        landmark_3d_68=rng.rand(68, 3).astype(np.float32) * 100,
        pose=np.array([1.0, -2.0, 3.0], np.float32),
        embedding=rng.randn(512).astype(np.float32),
    )
    data.update(over)
    return Face({k: v for k, v in data.items() if v is not ...})


class _Base(unittest.TestCase):
    def setUp(self):
        face_sanity.reset()


class DetScore(_Base):
    def test_none_nan_inf_and_text_become_zero(self):
        for bad in (None, float('nan'), float('inf'), -float('inf'), 'high', [0.9]):
            f = _face(det_score=bad)
            self.assertTrue(sanitize_face_attributes(f), bad)
            self.assertEqual(f.det_score, 0.0, bad)
            self.assertIsInstance(f.det_score, float)

    def test_missing_key_is_the_none_case(self):
        f = _face(det_score=...)                  # key absent: Face.__getattr__ gives None
        self.assertIsNone(f.det_score)
        self.assertTrue(sanitize_face_attributes(f))
        self.assertEqual(f.det_score, 0.0)
        self.assertEqual(face_sanity.snapshot()['det_score_missing'], 1)

    def test_a_good_score_is_left_exactly_alone(self):
        for good in (0.91, np.float32(0.91), np.float64(0.5), 1, 0.0):
            f = _face(det_score=good)
            before = f.det_score
            sanitize_face_attributes(f)
            self.assertIs(f.det_score, before)
        self.assertFalse(face_sanity.snapshot())

    def test_the_comparison_that_used_to_raise_now_works(self):
        f = _face(det_score=None)
        sanitize_face_attributes(f)
        self.assertFalse(f.det_score >= 0.65)     # was: TypeError


class Dropped(_Base):
    def test_unusable_box_drops_the_face(self):
        for bad in (None, ..., np.array([np.nan, 0, 10, 10]), np.array([5, 5, 5, 20]),
                    np.array([10, 10, 5, 5]), np.zeros(3), np.array([0, 0, np.inf, 10])):
            self.assertFalse(sanitize_face_attributes(_face(bbox=bad)), repr(bad))
        self.assertEqual(face_sanity.snapshot()['dropped_bbox'], 7)

    def test_unusable_keypoints_drop_the_face(self):
        bad_kps = np.full((5, 2), np.nan, np.float32)
        for bad in (None, ..., bad_kps, np.zeros((4, 2)), np.zeros(10)):
            self.assertFalse(sanitize_face_attributes(_face(kps=bad)), repr(bad))

    def test_sanitize_faces_filters_and_keeps_order(self):
        a, b, c = _face(), _face(bbox=np.zeros(4)), _face(det_score=None)
        out = sanitize_faces([a, b, c])
        self.assertEqual(len(out), 2)
        self.assertIs(out[0], a)
        self.assertIs(out[1], c)
        self.assertEqual(c.det_score, 0.0)


class Nulled(_Base):
    def test_unusable_optional_fields_become_none_not_a_drop(self):
        nan106 = np.full((106, 2), np.nan, np.float32)
        f = _face(landmark_2d_106=nan106, landmark_3d_68=np.zeros((67, 3)),
                  pose=np.array([np.nan, 0, 0]))
        self.assertTrue(sanitize_face_attributes(f))
        self.assertIsNone(f.landmark_2d_106)
        self.assertIsNone(f.landmark_3d_68)
        self.assertIsNone(f.pose)
        self.assertEqual(f.det_score, 0.91)       # untouched
        snap = face_sanity.snapshot()
        self.assertEqual((snap['nulled_landmark_2d_106'], snap['nulled_landmark_3d_68'],
                          snap['nulled_pose']), (1, 1, 1))

    def test_nan_zero_and_wrong_shape_embeddings_are_nulled(self):
        for bad in (np.full(512, np.nan, np.float32), np.zeros(512, np.float32),
                    np.ones((2, 512), np.float32), np.ones(8, np.float32)):
            f = _face(embedding=bad)
            self.assertTrue(sanitize_face_attributes(f))
            self.assertIsNone(f.embedding)
            self.assertIsNone(f.normed_embedding)       # InsightFace derives it from embedding
        self.assertNotIn('validator_error', face_sanity.snapshot())

    def test_a_stored_normed_embedding_is_checked_too(self):
        d = dict(bbox=np.array([0, 0, 9, 9.]), kps=np.zeros((5, 2)), det_score=0.9,
                 embedding=np.ones(512, np.float32),
                 normed_embedding=np.full(512, np.nan, np.float32))
        self.assertTrue(sanitize_face_attributes(d))
        self.assertIsNone(d['normed_embedding'])
        self.assertIsNotNone(d['embedding'])

    def test_no_embedding_at_all_is_a_mode_not_damage(self):
        f = _face(embedding=...)
        self.assertTrue(sanitize_face_attributes(f))
        self.assertIsNone(f.embedding)
        self.assertNotIn('nulled_embedding', face_sanity.snapshot())

    def test_a_complete_face_is_untouched(self):
        f = _face()
        before = {k: np.array(v, copy=True) for k, v in f.items()}
        self.assertTrue(sanitize_face_attributes(f))
        for k, v in before.items():
            np.testing.assert_array_equal(np.asarray(f[k]), v)
        self.assertFalse(face_sanity.snapshot())


class Shapes(_Base):
    def test_plain_objects_and_dicts_work_as_well_as_Face(self):
        class Obj:
            pass
        o = Obj()
        o.bbox, o.kps, o.det_score = np.array([0, 0, 9, 9.]), np.zeros((5, 2)), None
        self.assertTrue(sanitize_face_attributes(o))
        self.assertEqual(o.det_score, 0.0)
        d = dict(bbox=np.array([0, 0, 9, 9.]), kps=np.zeros((5, 2)), det_score=float('nan'))
        self.assertTrue(sanitize_face_attributes(d))
        self.assertEqual(d['det_score'], 0.0)

    def test_a_validator_error_never_becomes_a_crash(self):
        with mock.patch.object(face_sanity, '_usable_box', side_effect=RuntimeError('boom')):
            self.assertTrue(sanitize_face_attributes(_face()))
        self.assertEqual(face_sanity.snapshot()['validator_error'], 1)


class Reporting(_Base):
    def test_damage_is_printed_once_and_clean_runs_print_nothing(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            sanitize_faces([_face(), _face()])
        self.assertEqual(buf.getvalue(), '')
        with redirect_stdout(buf):
            sanitize_faces([_face(bbox=np.zeros(4))])
            sanitize_faces([_face(bbox=np.zeros(4))])
        self.assertEqual(buf.getvalue().count('[FaceSanity]'), 1)
        self.assertIn('dropped_bbox=', buf.getvalue())


class TheChokePoint(_Base):
    """The validator has to run where every analyser path exits, or it protects nothing."""

    def test_enrich_detected_faces_returns_sanitized_faces(self):
        from roop import face_util
        good, bad_box, no_score = _face(), _face(bbox=np.zeros(4)), _face(det_score=None)
        with mock.patch.object(face_util.face_contact, 'suppress_merged',
                               side_effect=lambda f: (f, 0)), \
                mock.patch.object(face_util, '_lm68_should_measure', return_value=False), \
                mock.patch.object(face_util, 'UPRIGHT_REMEASURE', False), \
                mock.patch.object(face_util.face_contact, 'stamp_contamination'), \
                mock.patch.object(face_util.roop.globals, 'refine_landmarks', False, create=True):
            out = face_util._enrich_detected_faces(np.zeros((8, 8, 3), np.uint8),
                                                   [good, bad_box, no_score])
        self.assertEqual(out, [good, no_score])
        self.assertEqual(no_score.det_score, 0.0)

    def test_enrich_of_nothing_is_an_empty_list(self):
        from roop import face_util
        with mock.patch.object(face_util.face_contact, 'stamp_contamination'):
            self.assertEqual(face_util._enrich_detected_faces(np.zeros((8, 8, 3), np.uint8), []), [])


if __name__ == '__main__':
    unittest.main()
