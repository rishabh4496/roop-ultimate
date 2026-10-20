"""stage2_frame_plan: a read-only index of the frames that hold a face matched to a source.

It is a view over `_temporal_faces` + `_track_source_map` (the real Stage 2 -> Stage 3
hand-off), so these tests check three things: the index says what those two say, it never
holds on to a Face (the swap pass evicts them frame by frame), and it is built and reported
where the pre-pass finishes.
"""
import contextlib
import gc
import io
import os
import sys
import unittest
import weakref
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals                                              # noqa: E402
from roop.procmgr_tracking import TrackingMixin                  # noqa: E402
from roop.stage2_plan import (VerifiedTargetFace, build_stage2_plan,   # noqa: E402
                              check, summary)


class _F(dict):
    """A Face-like dict (what temporal faces are): attribute access onto its keys."""
    __getattr__ = dict.get


def face(frame_x, tid, score=0.9, **extra):
    d = dict(bbox=np.array([frame_x, 10, frame_x + 50, 60], np.float64),
             det_score=score, _track_id=tid, pose=np.array([1.0, 2.0, 3.0], np.float64),
             landmark_2d_106=np.zeros((106, 2), np.float32),
             embedding=np.ones(512, np.float32))
    d.update(extra)
    return _F(d)


SOURCES = {0: (0, np.ones(512, np.float32)), 1: (1, None), 2: (None, None), 3: 2}


class BuildPlan(unittest.TestCase):
    def test_only_frames_with_a_matched_face_are_in_the_plan(self):
        faces = {
            0: [face(0, 0)],                         # matched -> source 0
            1: [face(1, 2)],                         # track has no source
            2: [],                                   # nobody
            3: [face(3, 1), face(300, 2)],           # one matched, one not
            4: [face(4, None)],                      # untracked
            5: [face(5, 3)],                         # bare-int binding
        }
        plan, stats = build_stage2_plan(faces, SOURCES)
        self.assertEqual(sorted(plan), [0, 3, 5])
        self.assertEqual([(r.track_id, r.source_index) for r in plan[3]], [(1, 1)])
        self.assertEqual(plan[5][0].source_index, 2)
        self.assertEqual(stats, dict(frames_with_faces=5, frames_matched=3, faces=6, matched=3,
                                     synthetic=0, unmatched_faces=2, untracked_faces=1))

    def test_a_record_is_float32_small_and_independent_of_the_face(self):
        f = face(7, 0, score=0.8)
        plan, _ = build_stage2_plan({7: [f]}, SOURCES)
        r = plan[7][0]
        self.assertIsInstance(r, VerifiedTargetFace)
        self.assertEqual((r.frame_idx, r.det_score), (7, 0.8))
        self.assertEqual(r.bbox.dtype, np.float32)
        self.assertEqual(r.pose.dtype, np.float32)
        f['bbox'][:] = 0                             # mutating the face must not move the record
        self.assertEqual(float(r.bbox[0]), 7.0)
        self.assertFalse(hasattr(r, 'embedding') or hasattr(r, 'landmark_2d_106'))

    def test_gap_filled_and_coasted_faces_are_marked_synthetic(self):
        faces = {1: [face(1, 0, _interpolated=True)], 2: [face(2, 0, _coasted=True)], 3: [face(3, 0)]}
        plan, stats = build_stage2_plan(faces, SOURCES)
        self.assertEqual([plan[i][0].synthetic for i in (1, 2, 3)], [True, True, False])
        self.assertEqual(stats['synthetic'], 2)

    def test_bad_scores_and_missing_pose_are_survivable(self):
        faces = {1: [face(1, 0, score=None)], 2: [face(2, 0, score=float('nan'), pose=None)]}
        plan, _ = build_stage2_plan(faces, SOURCES)
        self.assertEqual(plan[1][0].det_score, 0.0)
        self.assertEqual(plan[2][0].det_score, 0.0)
        self.assertIsNone(plan[2][0].pose)

    def test_the_plan_does_not_keep_faces_alive(self):
        f = face(1, 0)
        ref = weakref.ref(f)
        faces = {1: [f]}
        plan, _ = build_stage2_plan(faces, SOURCES)
        del f
        faces.clear()                                # the swap pass evicting its cache
        gc.collect()
        self.assertIsNone(ref(), 'the plan pinned a Face')
        self.assertIn(1, plan)

    def test_empty_inputs(self):
        self.assertEqual(build_stage2_plan({}, {})[0], {})
        self.assertEqual(build_stage2_plan({1: None}, SOURCES)[0], {})


class Invariants(unittest.TestCase):
    def test_a_matched_frame_outside_the_active_mask_is_flagged(self):
        plan, _ = build_stage2_plan({2: [face(2, 0)], 5: [face(5, 0)]}, SOURCES)
        mask = np.array([False, False, True, False, False, False, False])
        self.assertEqual(check(plan, mask, 7), dict(outside_active=1, past_scan=0))

    def test_a_matched_frame_past_the_scan_is_flagged(self):
        plan, _ = build_stage2_plan({2: [face(2, 0)], 9: [face(9, 0)]}, SOURCES)
        self.assertEqual(check(plan, None, 5), dict(outside_active=0, past_scan=1))

    def test_clean_plan_has_no_problems(self):
        plan, _ = build_stage2_plan({1: [face(1, 0)]}, SOURCES)
        self.assertEqual(check(plan, np.ones(4, bool), 4), dict(outside_active=0, past_scan=0))

    def test_summary_reads_the_numbers_and_shouts_only_on_a_problem(self):
        plan, stats = build_stage2_plan({0: [face(0, 0)], 1: [face(1, 2)], 2: [face(2, 0)]}, SOURCES)
        line = summary(stats, dict(outside_active=0, past_scan=0), 10)
        self.assertIn('2 frame(s) hold a face matched to a source', line)
        self.assertIn('1 more hold only unmatched faces (1)', line)
        self.assertIn('7 of 10 active-range frame(s) hold no face', line)
        self.assertNotIn('WARNING', line)
        bad = summary(stats, dict(outside_active=3, past_scan=0), None)
        self.assertIn('WARNING: 3 matched frame(s) lie OUTSIDE', bad)
        self.assertNotIn('active-range', bad)


class _Mgr(TrackingMixin):
    def __init__(self):
        self.progress_gradio = None
        self._shot_boundaries = set()
        self._track_scanned = 0
        self.stage2_frame_plan = {}


class BuiltWherePrepassEnds(unittest.TestCase):
    def _run(self, mask=None, source_map=None, faces=None):
        mgr = _Mgr()
        mgr._active_ranges_mask = mask
        faces = faces if faces is not None else {0: [face(0, 0)], 1: [face(1, 2)], 2: [face(2, 1)]}

        def fake_tracks(self_, *a, **k):
            self_._track_source_map = SOURCES if source_map is None else source_map
            self_._track_scanned = 3
            return [{'id': 0}, {'id': 1}, {'id': 2}]

        out = io.StringIO()
        was, roop.globals.processing = roop.globals.processing, True
        try:
            with contextlib.redirect_stdout(out), \
                    mock.patch.object(_Mgr, '_precompute_tracks', fake_tracks), \
                    mock.patch.object(_Mgr, '_build_temporal_faces', lambda s, t, g: faces):
                mgr._precompute_temporal(None, [0, 1, 2], 0, 3, 3)
        finally:
            roop.globals.processing = was
        return mgr, out.getvalue()

    def test_precompute_temporal_publishes_and_reports_the_plan(self):
        mgr, text = self._run(mask=np.array([True, True, True]))
        self.assertEqual(sorted(mgr.stage2_frame_plan), [0, 2])
        self.assertEqual(mgr.stage2_frame_plan[2][0].source_index, 1)
        self.assertIn('[Stage2] plan: 2 frame(s) hold a face matched to a source', text)
        self.assertNotIn('WARNING', text.split('[Stage2]')[1].splitlines()[0])

    def test_a_match_outside_the_mask_is_reported_where_the_prepass_ends(self):
        _, text = self._run(mask=np.array([True, False, False]))
        self.assertIn('WARNING: 1 matched frame(s) lie OUTSIDE the active ranges', text)

    def test_a_failing_index_never_disables_the_prepass(self):
        with mock.patch('roop.procmgr_tracking.build_stage2_plan', side_effect=RuntimeError('boom')):
            mgr, text = self._run(mask=None)
        self.assertEqual(mgr.stage2_frame_plan, {})
        self.assertNotIn('falling back', text)       # temporal mode was not abandoned

    def test_no_mask_means_no_active_range_clause(self):
        _, text = self._run(mask=None)
        self.assertNotIn('active-range frame', text)


if __name__ == '__main__':
    unittest.main()
