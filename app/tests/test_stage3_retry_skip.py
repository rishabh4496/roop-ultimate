"""Stage 3 does not re-detect a frame the temporal pre-pass already scanned.

With "Retry rotated" chosen, every frame that swaps nothing used to run `retry_rotated`: up to two
more full detections plus swap attempts. In temporal mode the pre-pass has already detected the
frame (upright, with its roll-aware ROI and rescue ladder), so for a covered frame that is the same
work again. Measured on five clips (docs/perf/stage3_plan) it recovered no swap.

A frame the pre-pass did NOT scan - past `_temporal_covered`, or with temporal mode off - still
retries, because there the retry is the only second look the frame gets.
"""
import contextlib
import io
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals                                                   # noqa: E402
from roop.ProcessMgr import ProcessMgr, eNoFaceAction                 # noqa: E402


def _mgr(temporal=True, covered=100, cache=True):
    m = ProcessMgr.__new__(ProcessMgr)
    m.input_face_datas = [object()]
    m.options = SimpleNamespace(show_face_masking=False, max_num_reuse_frame=0)
    m._active_ranges_mask = None
    m._temporal_mode = temporal
    m._temporal_faces = {} if cache else None
    m._temporal_covered = covered
    m._tls = SimpleNamespace()
    m.last_swapped_frame = None
    m.num_frames_no_face = 0
    m._temporal_engine = lambda name: None
    m._publish_live = lambda fr: None
    m.swap_faces = lambda frame, temp, stabilize=False, frame_idx=None: (0, frame)
    m.retries = []
    m.retry_rotated = lambda fr: (m.retries.append(1), fr)[1]
    return m


class RetrySkip(unittest.TestCase):
    def setUp(self):
        self._was = (roop.globals.no_face_action, roop.globals.processing)
        roop.globals.no_face_action = eNoFaceAction.RETRY_ROTATED
        roop.globals.processing = True          # pause_aware skips the body outside a run
        self.frame = np.zeros((4, 4, 3), np.uint8)

    def tearDown(self):
        roop.globals.no_face_action, roop.globals.processing = self._was

    def _run(self, m, idx):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            res = m.process_frame(self.frame, idx)
        return res, out.getvalue()

    def test_a_scanned_frame_with_nothing_to_swap_is_returned_untouched_without_a_retry(self):
        m = _mgr()
        res, text = self._run(m, 10)
        self.assertIs(res, self.frame)
        self.assertEqual(m.retries, [])
        self.assertIn('[Stage3] "Retry rotated" is skipped', text)

    def test_the_notice_prints_once(self):
        m = _mgr()
        _, first = self._run(m, 10)
        _, second = self._run(m, 11)
        self.assertIn('[Stage3]', first)
        self.assertNotIn('[Stage3]', second)
        self.assertEqual(m._retry_skipped_prepass, 2)

    def test_a_frame_past_the_scan_still_retries(self):
        m = _mgr(covered=50)
        self._run(m, 60)
        self.assertEqual(m.retries, [1])

    def test_without_temporal_mode_the_retry_is_untouched(self):
        for kw in (dict(temporal=False), dict(cache=False)):
            with self.subTest(**kw):
                m = _mgr(**kw)
                self._run(m, 10)
                self.assertEqual(m.retries, [1])

    def test_no_frame_index_means_no_skip(self):
        m = _mgr()
        self._run(m, None)
        self.assertEqual(m.retries, [1])

    def test_a_swapped_frame_is_unaffected(self):
        m = _mgr()
        swapped = np.ones((4, 4, 3), np.uint8)
        m.swap_faces = lambda frame, temp, stabilize=False, frame_idx=None: (1, swapped)
        res, text = self._run(m, 10)
        self.assertIs(res, swapped)
        self.assertEqual(text, '')

    def test_other_policies_are_unchanged(self):
        roop.globals.no_face_action = eNoFaceAction.SKIP_FRAME
        m = _mgr()
        res, _ = self._run(m, 10)
        self.assertIsNone(res)
        self.assertEqual(m.retries, [])

    def test_prepass_scanned_edges(self):
        m = _mgr(covered=10)
        self.assertTrue(m._prepass_scanned(9))
        self.assertFalse(m._prepass_scanned(10))
        self.assertFalse(m._prepass_scanned(None))


if __name__ == '__main__':
    unittest.main()
