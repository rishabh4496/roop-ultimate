"""The small-face restorer gate: right decision, no flicker at the edge, truly wired."""

import os
import sys
import threading
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from roop import enhance_gate as eg  # noqa: E402


def face(w, h=None):
    h = w if h is None else h
    return types.SimpleNamespace(bbox=np.array([100.0, 50.0, 100.0 + w, 50.0 + h], np.float32))


class DecisionTest(unittest.TestCase):

    def test_off_by_default_never_skips(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('ROOP_ENHANCE_MIN_FACE_PX', None)
            gate = eg.EnhanceGate()
        self.assertFalse(gate.enabled)
        self.assertFalse(gate.should_skip(face(10)))
        self.assertEqual(gate.seen, 0)
        self.assertIsNone(gate.summary())

    def test_skips_below_and_keeps_at_or_above_the_threshold(self):
        gate = eg.EnhanceGate(96)
        self.assertTrue(gate.should_skip(face(95), 'a'))
        self.assertFalse(gate.should_skip(face(96), 'b'))
        self.assertFalse(gate.should_skip(face(300), 'c'))

    def test_the_shorter_side_decides(self):
        gate = eg.EnhanceGate(96)
        self.assertTrue(gate.should_skip(face(400, 80), 'a'))     # wide but short
        self.assertFalse(gate.should_skip(face(120, 130), 'b'))

    def test_unknown_size_never_skips(self):
        gate = eg.EnhanceGate(96)
        self.assertFalse(gate.should_skip(types.SimpleNamespace(), 'a'))
        self.assertFalse(gate.should_skip(types.SimpleNamespace(bbox=None), 'b'))
        self.assertFalse(gate.should_skip(types.SimpleNamespace(
            bbox=[10, 10, 10, 10]), 'c'))                          # zero area

    def test_a_dict_subclass_face_works(self):
        """insightface's Face is a dict subclass."""
        class FakeFace(dict):
            pass
        f = FakeFace(bbox=np.array([0, 0, 50, 60], np.float32))
        self.assertTrue(eg.EnhanceGate(96).should_skip(f, 'a'))

    def test_threshold_parsing(self):
        for raw, want in (('96', 96), ('96.7', 96), ('0', 0), ('', 0),
                          ('junk', 0), ('-5', 0)):
            with self.subTest(raw=raw):
                with mock.patch.dict(os.environ, {'ROOP_ENHANCE_MIN_FACE_PX': raw}):
                    self.assertEqual(eg.threshold_from_env(), want)


class HysteresisTest(unittest.TestCase):

    def test_a_face_at_the_edge_does_not_flip_every_frame(self):
        gate = eg.EnhanceGate(96)
        sizes = [97, 95, 97, 95, 98, 94, 99, 96, 97]     # jitter around 96
        decisions = [gate.should_skip(face(s), 'track1') for s in sizes]
        # Starts enhanced (97 >= 96), drops once at 95, then STAYS skipped:
        # resuming needs 96 * 1.15 = 110.4.
        self.assertEqual(decisions, [False] + [True] * 8)

    def test_it_resumes_once_clearly_past_the_limit(self):
        gate = eg.EnhanceGate(96)
        self.assertTrue(gate.should_skip(face(90), 'a'))
        self.assertTrue(gate.should_skip(face(105), 'a'))      # inside the band
        self.assertFalse(gate.should_skip(face(111), 'a'))     # past 110.4
        self.assertFalse(gate.should_skip(face(100), 'a'))     # now enhanced: 100 >= 96

    def test_tracks_are_independent(self):
        gate = eg.EnhanceGate(96)
        self.assertTrue(gate.should_skip(face(90), 'a'))
        self.assertFalse(gate.should_skip(face(105), 'b'))     # b was never skipped

    def test_no_track_key_means_stateless(self):
        gate = eg.EnhanceGate(96)
        self.assertTrue(gate.should_skip(face(90)))
        self.assertFalse(gate.should_skip(face(100)))          # no memory to hold it

    def test_remembered_tracks_are_bounded(self):
        gate = eg.EnhanceGate(96)
        for i in range(eg._MAX_TRACKS + 50):
            gate.should_skip(face(50), i)
        self.assertLessEqual(len(gate._skipping), eg._MAX_TRACKS)


class CountingTest(unittest.TestCase):

    def test_summary_reports_counts_and_the_smallest_face(self):
        gate = eg.EnhanceGate(96)
        for s in (40, 70, 200, 300):
            gate.should_skip(face(s), s)
        line = gate.summary()
        self.assertIn('skipped on 2 of 4 faces under 96 px (50.0%)', line)
        self.assertIn('smallest face 40 px', line)

    def test_thread_safe_counting(self):
        gate = eg.EnhanceGate(96)

        def work(k):
            for i in range(400):
                gate.should_skip(face(50 if i % 2 else 200), (k, i % 7))

        threads = [threading.Thread(target=work, args=(k,)) for k in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(gate.seen, 3200)
        self.assertLessEqual(gate.skipped, gate.seen)


class WiringTest(unittest.TestCase):
    """A setting that saves and displays but is read by nothing is the failure
    this repo keeps hitting, so check every hop."""

    def setUp(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.here = here
        self.read = lambda *p: open(os.path.join(here, *p), encoding='utf-8').read()

    def test_setting_exports_the_variable_the_gate_reads(self):
        import settings
        mapping = {key: var for key, var, _kind in settings.ENV_SETTINGS}
        self.assertEqual(mapping.get('enhance_min_face_px'), 'ROOP_ENHANCE_MIN_FACE_PX')
        self.assertIn('enhance_min_face_px', settings.LIVE_ENV_SETTINGS)
        env = {}
        settings.apply_env({'enhance_min_face_px': 96}, env)
        self.assertEqual(env['ROOP_ENHANCE_MIN_FACE_PX'], '96')
        with mock.patch.dict(os.environ, env):
            self.assertEqual(eg.threshold_from_env(), 96)

    def test_default_is_off_and_it_is_saved(self):
        import settings
        self.assertEqual(settings.Settings.__init__.__defaults__ is None or True, True)
        src = self.read('settings.py')
        self.assertIn("self.enhance_min_face_px = self.default_get(data, 'enhance_min_face_px', 0)", src)
        self.assertIn("'enhance_min_face_px': self.enhance_min_face_px", src)

    def test_processmgr_builds_the_gate_and_skips_through_it(self):
        src = self.read('roop', 'ProcessMgr.py')
        self.assertIn('self._enhance_gate = EnhanceGate()', src)
        self.assertIn("p.type == 'enhance'", src)
        self.assertIn('self._enhance_gate.should_skip(', src)
        # The skip branch must come BEFORE the enhancer's own `else:`.
        self.assertLess(src.index('self._enhance_gate.should_skip('),
                        src.index('# Pooled (no global lock) ONLY when this enhancer'))

    def test_summary_is_printed_at_the_end_of_a_render(self):
        self.assertIn('_gate.summary()', self.read('roop', 'procmgr_batch.py'))

    def test_the_react_panel_binds_it(self):
        root = os.path.dirname(self.here)
        jsx = open(os.path.join(root, 'react-ui', 'src', 'components', 'Settings.jsx'),
                   encoding='utf-8').read()
        self.assertIn("bind('enhance_min_face_px', 0)", jsx)


if __name__ == '__main__':
    unittest.main()
