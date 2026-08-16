"""AdaFace as a matching-only recogniser.

Two properties matter more than anything the model does:

  1. OFF BY DEFAULT IS INERT. AdaFace is opt-in; with ROOP_ADAFACE unset every
     helper must behave exactly as the old code did — w600k distances, unscaled
     thresholds. Otherwise enabling a research feature would silently change
     everyone's renders.

  2. METRICS NEVER MIX. AdaFace distances live on a different scale from w600k.
     Comparing an AdaFace distance against max_face_distance, or letting half a
     run use one metric and half the other, is worse than consistently using the
     weaker recogniser. begin_run() is therefore all-or-nothing and the tuned
     veto constants are rescaled with the metric, not left behind.

These run without the model file present.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import recognizer_adaface as ada          # noqa: E402


class _FakeFace(dict):
    """Stands in for insightface's Face — a dict subclass with attributes."""

    def __init__(self, embedding=None):
        super().__init__()
        self.embedding = embedding
        self.kps = None


class TestDisabledIsInert(unittest.TestCase):
    def setUp(self):
        ada._run_active = False

    def test_identity_distance_falls_back_to_w600k(self):
        a = _FakeFace(np.array([1.0, 0.0, 0.0], np.float32))
        b = _FakeFace(np.array([0.0, 1.0, 0.0], np.float32))
        d = ada.identity_distance(a, b)
        self.assertAlmostEqual(d, 1.0, places=5)   # orthogonal -> cosine dist 1

    def test_identical_embeddings_are_zero_distance(self):
        e = np.array([0.3, 0.4, 0.5], np.float32)
        self.assertAlmostEqual(ada.identity_distance(_FakeFace(e), _FakeFace(e)),
                               0.0, places=6)

    def test_missing_embedding_returns_none_not_zero(self):
        """A missing embedding must not read as a perfect match."""
        self.assertIsNone(ada.identity_distance(_FakeFace(None),
                                                _FakeFace(np.zeros(3, np.float32))))

    def test_threshold_and_scale_are_no_ops(self):
        self.assertEqual(ada.active_threshold(0.75), 0.75)
        for v in (0.85, 1.0, 0.15):
            self.assertEqual(ada.scale(v, 0.75), v)

    def test_begin_run_declines_without_opt_in(self):
        self.assertFalse(ada.begin_run([_FakeFace(np.zeros(3, np.float32))]))
        self.assertFalse(ada.ready())


class TestActiveScaling(unittest.TestCase):
    """When AdaFace drives, w600k-tuned constants must move with the metric."""

    def setUp(self):
        self._saved = (ada._run_active, ada.DIST_THRESHOLD)
        ada._run_active = True
        ada.DIST_THRESHOLD = 0.5

    def tearDown(self):
        ada._run_active, ada.DIST_THRESHOLD = self._saved

    def test_match_threshold_switches_scale(self):
        self.assertEqual(ada.active_threshold(0.75), 0.5)

    def test_veto_keeps_its_ratio_to_the_threshold(self):
        # The veto is deliberately LOOSER than the match gate; that relationship
        # is what carries hard frames, so it must survive the metric change.
        w600k_thr, w600k_veto = 0.75, 0.85
        scaled = ada.scale(w600k_veto, w600k_thr)
        self.assertAlmostEqual(scaled / ada.active_threshold(w600k_thr),
                               w600k_veto / w600k_thr, places=6)
        self.assertGreater(scaled, ada.active_threshold(w600k_thr))

    def test_scale_is_safe_when_threshold_is_zero(self):
        self.assertEqual(ada.scale(0.85, 0), 0.85)


class TestContract(unittest.TestCase):
    def test_alignment_matches_the_cached_crop_key(self):
        """face_embedding reuses _attach_source_crops' cached crop, so the two
        must agree on the alignment template or the embedding is subtly wrong."""
        self.assertEqual(ada.ALIGN_MODE, 'arcface_112_v2')
        self.assertIn(ada.ALIGN_MODE, ada._CROP_KEY)

    def test_input_size_matches_the_verified_graph(self):
        self.assertEqual(ada.INPUT_SIZE, 112)

    def test_threshold_is_separate_from_max_face_distance(self):
        """Reusing the w600k threshold across metrics is the failure this guards.

        Checked structurally: the module must own its threshold and must never
        READ the w600k one (CFG.max_face_distance / face_distance_threshold).
        Mentioning it in a comment is fine — consuming it is not.
        """
        src = open(os.path.join(os.path.dirname(__file__), '..', 'roop',
                                'recognizer_adaface.py'), encoding='utf-8').read()
        self.assertIn("ROOP_ADAFACE_DIST", src)
        self.assertNotIn("CFG.max_face_distance", src)
        self.assertNotIn("face_distance_threshold", src)


if __name__ == '__main__':
    unittest.main()
