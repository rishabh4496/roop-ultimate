"""Stage 11 plumbing: request -> roop.globals, the Face Bank's per-sample scores, the confidence lane
route, and the UI-frame <-> pipeline-frame conversion at the real ProcessMgr sites.

The pure maths lives in test_roto_polygons / test_expression_fidelity / test_identity_lock. What is
pinned here is that the pieces are CONNECTED: a field the UI sends reaches the global the pipeline
reads, a list drawn for one clip cannot outlive its request, and a zone's frame range means the same
frame to the UI that drew it and to the render that applies it.
"""
import os
import shutil
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals as G  # noqa: E402

KEYS = ('pupil_gaze_retention', 'micro_expression_restore', 'inner_mouth_preserve', 'identity_lock',
        'identity_lock_margin', 'identity_lock_hold', 'identity_pins', 'roto_polygons',
        'roto_preview_frame', 'roto_frame_base')


class _Saved(unittest.TestCase):
    def setUp(self):
        self._saved = {k: getattr(G, k, None) for k in KEYS}

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(G, k, v)


class RequestReachesGlobals(_Saved):
    @classmethod
    def setUpClass(cls):
        import api
        cls.api = api

    def test_every_field_lands_on_its_global(self):
        self.api._apply_stage11_settings({
            'pupil_gaze_retention': 0.6, 'micro_expression_restore': 0.4, 'inner_mouth_preserve': True,
            'identity_lock': True, 'identity_lock_margin': 0.12, 'identity_lock_hold': 5,
            'identity_pins': [{'frame': 9, 'point': [10, 10], 'source_index': 1}],
            'roto_polygons': [{'points': [[0, 0], [10, 0], [10, 10]], 'mode': 'include'}],
        }, preview_frame=17)
        self.assertEqual((G.pupil_gaze_retention, G.micro_expression_restore), (0.6, 0.4))
        self.assertTrue(G.inner_mouth_preserve and G.identity_lock)
        self.assertEqual((G.identity_lock_margin, G.identity_lock_hold), (0.12, 5))
        self.assertEqual(G.identity_pins[0]['source_index'], 1)
        self.assertEqual(G.roto_polygons[0]['mode'], 'include')
        self.assertEqual(G.roto_preview_frame, 17)

    def test_values_are_clamped_and_garbage_falls_back(self):
        self.api._apply_stage11_settings({'pupil_gaze_retention': 7, 'micro_expression_restore': -3,
                                          'identity_lock_margin': 'x', 'identity_lock_hold': 999})
        self.assertEqual((G.pupil_gaze_retention, G.micro_expression_restore), (1.0, 0.0))
        self.assertEqual(G.identity_lock_margin, 0.08)
        self.assertEqual(G.identity_lock_hold, 30)

    def test_a_request_without_the_lists_clears_them(self):
        self.api._apply_stage11_settings({'identity_pins': [{'frame': 1, 'point': [1, 1], 'source_index': 0}],
                                          'roto_polygons': [{'points': [[0, 0], [9, 0], [9, 9]]}]})
        self.assertTrue(G.identity_pins and G.roto_polygons)
        self.api._apply_stage11_settings({})                      # the NEXT clip's request
        self.assertEqual((G.identity_pins, G.roto_polygons), ([], []))

    def test_a_render_clears_the_preview_frame(self):
        self.api._apply_stage11_settings({}, preview_frame=5)
        self.assertEqual(G.roto_preview_frame, 5)
        self.api._apply_stage11_settings({})
        self.assertIsNone(G.roto_preview_frame)

    def test_both_preview_and_render_call_the_helper(self):
        src = open(os.path.join(os.path.dirname(self.api.__file__), 'api.py'), encoding='utf-8').read()
        import re
        calls = re.findall(r'^[ \t]+_apply_stage11_settings\(payload.*$', src, re.M)   # call sites, not the def
        self.assertEqual(len(calls), 2, calls)
        self.assertIn('_apply_stage11_settings(payload, preview_frame=frame)', src)


class SettingsRoundTrip(_Saved):
    """The inspector persists with POST /api/settings before it starts a run. The route keeps only keys
    that exist on Settings, so a control whose key was never added there is saved by nobody and reads back
    as its default on the next launch - exactly how a control that "does nothing" gets built."""

    @classmethod
    def setUpClass(cls):
        import api
        cls.api = api

    def setUp(self):
        super().setUp()
        import settings as S
        self.S = S
        self.dir = tempfile.mkdtemp(prefix='s11cfg_')
        self.path = os.path.join(self.dir, 'config.yaml')
        self.cfg = S.Settings(self.path)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        super().tearDown()

    def test_the_six_keys_exist_with_noop_defaults(self):
        cfg = self.cfg
        self.assertEqual((cfg.pupil_gaze_retention, cfg.micro_expression_restore), (0.0, 0.0))
        self.assertEqual((cfg.inner_mouth_preserve, cfg.identity_lock), (False, False))
        self.assertEqual((cfg.identity_lock_margin, cfg.identity_lock_hold), (0.08, 3))

    def test_saved_through_the_route_and_read_back_by_a_fresh_load(self):
        body = {'pupil_gaze_retention': 0.5, 'micro_expression_restore': 0.25, 'inner_mouth_preserve': True,
                'identity_lock': True, 'identity_lock_margin': 0.12, 'identity_lock_hold': 5}
        with mock.patch.object(G, 'CFG', self.cfg):
            self.assertEqual(self.api.save_settings(dict(body)), {'status': 'success'})
        self.assertEqual(self.cfg.pupil_gaze_retention, 0.5)
        fresh = self.S.Settings(self.path)                       # a new process reading the file
        for key, want in body.items():
            self.assertEqual(getattr(fresh, key), want, key)

    def test_out_of_range_values_are_refused_not_stored(self):
        for bad in ({'pupil_gaze_retention': 2.0}, {'micro_expression_restore': -0.1},
                    {'identity_lock_margin': 0.9}, {'identity_lock_hold': 0}, {'identity_lock_hold': 31},
                    {'identity_lock_hold': True}, {'pupil_gaze_retention': 'high'}):
            with mock.patch.object(G, 'CFG', self.cfg):
                r = self.api.save_settings(dict(bad))
            self.assertEqual(getattr(r, 'status_code', None), 400, bad)
        self.assertEqual(self.cfg.pupil_gaze_retention, 0.0)
        self.assertEqual(self.cfg.identity_lock_hold, 3)

    def test_the_apply_helper_falls_back_to_the_saved_value_when_a_request_omits_it(self):
        self.cfg.pupil_gaze_retention = 0.7
        with mock.patch.object(G, 'CFG', self.cfg):
            self.api._apply_stage11_settings({})
        self.assertEqual(G.pupil_gaze_retention, 0.7)


class FrameNumbering(_Saved):
    """A zone's frame range is in the UI's numbering; the render counts from 0 inside the trim."""

    def _mgr(self):
        from roop.ProcessMgr import ProcessMgr
        m = ProcessMgr.__new__(ProcessMgr)
        m._tls = threading.local()
        return m

    def test_render_index_maps_through_the_clip_start(self):
        G.roto_frame_base = 1                                      # untrimmed: index 0 is UI frame 1
        mgr = self._mgr()
        self.assertEqual(mgr._roto_frame_ui(0), 1)
        G.roto_frame_base = 121                                    # trimmed to start at pipeline frame 120
        self.assertEqual(mgr._roto_frame_ui(0), 121)
        self.assertEqual(mgr._roto_frame_ui(9), 130)

    def test_preview_uses_the_frame_it_was_asked_to_draw(self):
        G.roto_preview_frame = 42
        self.assertEqual(self._mgr()._roto_frame_ui(None), 42)

    def test_the_real_exclusion_site_honours_the_range_at_its_boundaries(self):
        mgr = self._mgr()
        G.roto_frame_base = 11                                     # index 0 -> UI 11
        G.roto_polygons = [{'id': 'z', 'mode': 'exclude', 'feather': 0, 'pin': None, 'frames': (13, 14),
                            'points': [{'x': 10, 'y': 10, 'hin': None, 'hout': None},
                                       {'x': 50, 'y': 10, 'hin': None, 'hout': None},
                                       {'x': 50, 'y': 50, 'hin': None, 'hout': None},
                                       {'x': 10, 'y': 50, 'hin': None, 'hout': None}]}]
        plate = np.full((80, 80, 3), 200, np.uint8)
        for idx, expect_active in ((1, False), (2, True), (3, True), (4, False)):     # UI 12, 13, 14, 15
            result = np.full((80, 80, 3), 40, np.uint8)
            out = mgr._apply_roto_exclusions(result, plate, idx)
            self.assertEqual(bool((out[30, 30] == 200).all()), expect_active, f'index {idx} (UI {idx + 11})')

    def test_no_zones_is_the_same_array(self):
        mgr = self._mgr()
        G.roto_polygons = []
        r = np.zeros((10, 10, 3), np.uint8)
        self.assertIs(mgr._apply_roto_exclusions(r, r.copy(), 3), r)

    def test_the_identity_lock_is_off_unless_asked_and_per_thread(self):
        mgr = self._mgr()
        G.identity_lock = False
        self.assertIsNone(mgr._identity_lock())
        G.identity_lock, G.identity_lock_margin, G.identity_lock_hold = True, 0.08, 3
        here = mgr._identity_lock()
        self.assertIsNotNone(here)
        self.assertIs(here, mgr._identity_lock())
        seen = []
        t = threading.Thread(target=lambda: seen.append(mgr._identity_lock()))
        t.start()
        t.join()
        self.assertIsNot(seen[0], here)
        G.identity_lock_hold = 6                                    # a changed setting rebuilds the bank
        self.assertEqual(mgr._identity_lock().cfg.hold, 6)


class FaceBankScores(unittest.TestCase):
    def _members(self, c, base, frames, extra=None):
        rng = np.random.default_rng(c + 5)
        out = []
        for f in frames:
            v = base + rng.normal(0, 0.03, base.shape)
            if extra is not None and f in extra:
                v = extra[f]
            out.append({'embedding': (v / np.linalg.norm(v)).astype(np.float32), 'frame_idx': f,
                        'off_axis': 0.0, 'det_score': 0.9, 'yaw': 0.0, 'crop': np.zeros((8, 8, 3), np.uint8),
                        'face': object()})
        return out

    def test_scores_are_per_sample_with_a_runner_up(self):
        from roop.face_bank import FaceBank
        a = np.array([1.0, 0, 0, 0], np.float32)
        b = np.array([0, 1.0, 0, 0], np.float32)
        blend = (a * 0.62 + b * 0.78)                              # a crop that is half the neighbour
        faces = self._members(0, a, [0, 10, 20, 30], extra={20: blend}) + self._members(1, b, [5, 15, 25])
        bank = FaceBank()
        chars = bank._aggregate_characters(faces, [0] * 4 + [1] * 3)
        by = {c['person_id']: c for c in chars}
        scores = {int(f): (own, other) for f, own, other in by['tp_0']['frame_scores']}
        self.assertEqual(sorted(scores), [0, 10, 20, 30])
        clean_margin = scores[0][1] - scores[0][0]
        dip_margin = scores[20][1] - scores[20][0]
        self.assertGreater(clean_margin, 0.6)
        self.assertLess(dip_margin, clean_margin - 0.4)            # the contaminated sample is the close one
        from roop import identity_lock as il
        lane = il.confidence_lane([(f, o, r) for f, o, r in by['tp_0']['frame_scores']], threshold=0.75)
        self.assertIn(21, lane['amber'])                          # frame 20 in UI numbering
        self.assertNotIn(1, lane['amber'])

    def test_a_single_person_has_no_runner_up(self):
        from roop.face_bank import FaceBank
        a = np.array([1.0, 0, 0, 0], np.float32)
        chars = FaceBank()._aggregate_characters(self._members(0, a, [0, 10]), [0, 0])
        self.assertTrue(all(row[2] is None for row in chars[0]['frame_scores']))


class ConfidenceRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import api
        from fastapi.testclient import TestClient
        from roop.face_bank import get_face_bank
        cls.api, cls.client, cls.get_bank = api, TestClient(api.app), staticmethod(get_face_bank)

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='idconf_')
        self.video = os.path.join(self.dir, 'clip.mp4')
        open(self.video, 'wb').close()
        entry = mock.Mock()
        entry.filename, entry.total_frames, entry.media_id = self.video, 100, 'm-1'
        self.patches = [mock.patch.object(self.api, 'list_files_process', [entry]),
                        mock.patch.object(self.api, '_target_index_for_ref', lambda *a, **k: (0, 'm-1'))]
        for p in self.patches:
            p.start()
        bank = self.get_bank()
        self._saved = (bank.characters, bank.last_video_path, set(bank.scene_cuts), set(bank.sample_frames),
                       bank.total_frames)
        bank.last_video_path = self.video
        bank.sample_frames = set(range(0, 100, 10))
        bank.scene_cuts = set()
        bank.characters = [{'person_id': 'tp_0', 'name': 'Actor 1', 'frames': [0, 10, 20],
                            'frame_scores': [[0, 0.20, 0.80], [10, 0.62, 0.90], [20, 0.30, 0.32]]}]

    def tearDown(self):
        for p in self.patches:
            p.stop()
        bank = self.get_bank()
        bank.characters, bank.last_video_path, bank.scene_cuts, bank.sample_frames, bank.total_frames = self._saved
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_lane_is_in_ui_numbering_with_the_amber_samples_named(self):
        body = self.client.get('/api/target/timeline_tracks?index=0&threshold=0.75').json()
        lane = body['identity_confidence'][0]
        self.assertEqual(body['identity_confidence_source'], 'face_bank_sampled')
        self.assertEqual([p[0] for p in lane['points']], [1, 11, 21])
        self.assertEqual(lane['amber'], [11, 21])          # far from its mean (0.62) / barely ahead of the runner-up
        self.assertEqual(lane['threshold'], 0.75)

    def test_the_callers_threshold_moves_the_amber_set(self):
        loose = self.client.get('/api/target/timeline_tracks?index=0&threshold=1.5').json()['identity_confidence'][0]
        self.assertEqual(loose['amber'], [21])             # 0.62 is no longer near a 1.5 threshold; the close runner-up still is
        self.assertEqual(loose['threshold'], 1.5)

    def test_no_scores_means_an_empty_lane_and_source_none(self):
        self.get_bank().characters = [{'person_id': 'tp_0', 'name': 'A', 'frames': [0]}]
        body = self.client.get('/api/target/timeline_tracks?index=0').json()
        self.assertEqual((body['identity_confidence'], body['identity_confidence_source']), ([], 'none'))

    def test_a_bad_threshold_falls_back_to_the_configured_one(self):
        body = self.client.get('/api/target/timeline_tracks?index=0&threshold=-3').json()
        self.assertGreater(body['identity_threshold'], 0)


if __name__ == '__main__':
    unittest.main()
