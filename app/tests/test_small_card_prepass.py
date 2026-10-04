"""The sub-7GB pre-pass stops paying for rescues that cannot succeed.

Sampling the live temporal pre-pass on the RTX 3060 Laptop 6GB (2026-09-29,
py-spy, 40 s) put 72% of its wall clock in rescue detection and 28% in the first
detector call. A ROI crop around one tracked face inherited expected_count=2
from the global target list, so every crop paid the three-turn partial-miss
rescue for a person who was not in it; and a person out of shot re-ran the whole
ladder on every frame. These tests pin the SHAPE of the fix - that the passes
are not run - not a speed, which this host cannot measure reliably.
"""
import contextlib
import os
import sys
import types
import unittest
from unittest.mock import patch

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import numpy as np

import roop.globals
from roop import face_util
from roop.face_util import RescueBackoff


@contextlib.contextmanager
def _no_aux_models():
    """The rotated rescues lease an analyser to run the aux models on the faces they keep
    (face_util._rotated_pass); these tests stub the detector and care about which faces
    come back, so the lease yields an analyser with no models."""
    yield types.SimpleNamespace(models={})



class _Face:
    def __init__(self, x=10):
        self.bbox = np.array([x, 10, x + 60, 90], dtype=np.float32)
        self.kps = np.array([[x + 15, 30], [x + 45, 30], [x + 30, 50],
                             [x + 18, 70], [x + 42, 70]], dtype=np.float32)


def _frame():
    return np.zeros((240, 320, 3), dtype=np.uint8)


class _TwoTargets(unittest.TestCase):
    """Two selected people: the condition that made every one-face frame 'short'."""

    def setUp(self):
        self._saved = getattr(roop.globals, 'TARGET_FACE_GROUP', None)
        roop.globals.TARGET_FACE_GROUP = [0, 1]
        self._enrich = patch('roop.face_util._enrich_detected_faces',
                             side_effect=lambda frame, faces: faces)
        self._enrich.start()

    def tearDown(self):
        self._enrich.stop()
        roop.globals.TARGET_FACE_GROUP = self._saved


class RescueSwitch(_TwoTargets):
    def test_default_still_runs_the_partial_miss_rescue(self):
        """One face of two expected: today's behaviour is unchanged."""
        with patch('roop.face_util._detect_faces_raw', return_value=[_Face()]) as raw:
            face_util.get_all_faces(_frame())
        self.assertEqual(raw.call_count, 4, 'first pass + the three cardinal turns')
        self.assertEqual(face_util.last_rescue_outcome(), (True, 0))

    def test_rescue_false_is_one_detector_call(self):
        with patch('roop.face_util._detect_faces_raw', return_value=[_Face()]) as raw:
            faces = face_util.get_all_faces(_frame(), rescue=False)
        self.assertEqual(raw.call_count, 1)
        self.assertEqual(len(faces), 1)
        self.assertEqual(face_util.last_rescue_outcome(), (False, 0))

    def test_roi_crop_does_not_rescue_for_a_person_who_is_not_in_it(self):
        with patch('roop.face_util._detect_faces_raw', return_value=[_Face()]) as raw:
            faces = face_util.get_all_faces_in_roi(_frame(), (100, 60, 180, 180), rescue=False)
        self.assertEqual(raw.call_count, 1)
        self.assertEqual(len(faces), 1)

    def test_roi_default_is_untouched(self):
        with patch('roop.face_util._detect_faces_raw', return_value=[_Face()]) as raw:
            face_util.get_all_faces_in_roi(_frame(), (100, 60, 180, 180))
        self.assertEqual(raw.call_count, 4)

    def test_empty_frame_ladder_is_skipped_when_rescue_is_off(self):
        with patch('roop.face_util._detect_faces_raw', return_value=[]) as raw:
            self.assertEqual(face_util.get_all_faces(_frame(), rescue=False), [])
        self.assertEqual(raw.call_count, 1)

    def test_a_rescue_that_finds_a_face_reports_the_gain(self):
        calls = {'n': 0}

        def raw(frame, **kw):
            calls['n'] += 1
            return [_Face(10)] if calls['n'] == 1 else (
                [_Face(200)] if calls['n'] == 2 else [])
        with patch('roop.face_util._detect_faces_raw', side_effect=raw), \
                patch('roop.face_util.lease_face_analyser', _no_aux_models):
            faces = face_util.get_all_faces(_frame())
        self.assertEqual(len(faces), 2)
        attempted, gained = face_util.last_rescue_outcome()
        self.assertTrue(attempted)
        self.assertEqual(gained, 1)


class Backoff(unittest.TestCase):
    def _futile(self, b, idx):
        ok = b.allow(idx)
        b.record(idx, ok, (True, 0) if ok else (False, 0))
        return ok

    def test_futile_rescues_back_off_to_every_n(self):
        b = RescueBackoff(every=8, patience=2)
        ran = [i for i in range(40) if self._futile(b, i)]
        self.assertEqual(ran[:3], [0, 1, 9], 'two tries, then wait `every` frames')
        self.assertLessEqual(len(ran), 2 + 40 // 8 + 1)

    def test_a_gain_resets_it(self):
        b = RescueBackoff(every=8, patience=2)
        for i in range(2):
            self._futile(b, i)
        b.record(2, b.allow(2), (True, 1))
        self.assertTrue(all(b.allow(i) for i in (3, 4, 5)),
                        'a rescue that WORKS (a rotated face) is never throttled')

    def test_a_frame_that_needed_no_rescue_resets_it(self):
        b = RescueBackoff(every=8, patience=2)
        for i in range(2):
            self._futile(b, i)
        b.record(2, True, (False, 0))          # allowed, none needed: faces are back
        self.assertTrue(b.allow(3))

    def test_a_hard_cut_resets_it(self):
        b = RescueBackoff(every=8, patience=2)
        for i in range(2):
            self._futile(b, i)
        self.assertFalse(b.allow(3))
        self.assertTrue(b.allow(4, cut=True))

    def test_skipped_frames_are_counted(self):
        b = RescueBackoff(every=8, patience=2)
        for i in range(20):
            self._futile(b, i)
        self.assertGreater(b.skipped, 10)


class Tier(unittest.TestCase):
    def test_only_a_sub_7gb_card_takes_the_path(self):
        with patch.dict(os.environ, {'ROOP_VRAM_GB': '6'}, clear=False):
            os.environ.pop('ROOP_SMALL_CARD_PREPASS', None)
            self.assertTrue(face_util.small_card_prepass_active())
        for gb in ('7', '12', '24'):
            with patch.dict(os.environ, {'ROOP_VRAM_GB': gb}, clear=False):
                os.environ.pop('ROOP_SMALL_CARD_PREPASS', None)
                self.assertFalse(face_util.small_card_prepass_active(),
                                 f'{gb} GB must keep the exhaustive scan (the 4070)')

    def test_full_restores_the_exhaustive_scan(self):
        with patch.dict(os.environ, {'ROOP_VRAM_GB': '6',
                                     'ROOP_SMALL_CARD_PREPASS': 'full'}):
            self.assertFalse(face_util.small_card_prepass_active())

    def test_unknown_vram_is_not_a_small_card(self):
        with patch('roop.session_pool._detect_vram_gb', return_value=0.0):
            os.environ.pop('ROOP_SMALL_CARD_PREPASS', None)
            self.assertFalse(face_util.small_card_prepass_active())


if __name__ == '__main__':
    unittest.main()
