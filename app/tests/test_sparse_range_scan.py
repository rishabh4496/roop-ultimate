"""Unit tests for the Stage 1 Sparse Auto-Range Pre-Scan (roop/sparse_range_scan.py).

Nothing here needs a GPU. The demuxer is driven by a FakeCap that records every call, the
detector by a probe double, and one test decodes a real MJPG file with cv2.VideoCapture to
prove sequential grab/retrieve lines up with frame indices.
"""

import os
import tempfile
import threading
import time
import unittest

import cv2
import numpy as np

import roop.globals
from roop import sparse_range_scan as srs
from roop.sparse_range_scan import (
    CUT_DELTA_Y, MERGE_GAP, build_active_ranges, dilation_pad, det_score_floor,
    frame_is_positive, luma_delta, luma_thumb, run_demux, scan_active_ranges,
    similarity_floor, sparse_stride, unit_embeddings,
)
from settings import Settings

DARK, BRIGHT = 20, 220            # luma difference 200 >> CUT_DELTA_Y


class FakeCap:
    """cv2.VideoCapture look-alike over constant-colour frames, recording every call."""

    def __init__(self, values, fps=30.0, size=(64, 36)):
        self.values = list(values)
        self.fps = fps
        self.size = size
        self.pos = 0
        self._cur = None
        self.sets, self.grabs, self.retrieves = [], 0, []
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_COUNT: len(self.values), cv2.CAP_PROP_FPS: self.fps,
                cv2.CAP_PROP_FRAME_WIDTH: self.size[0], cv2.CAP_PROP_FRAME_HEIGHT: self.size[1],
                cv2.CAP_PROP_POS_FRAMES: self.pos}.get(prop, 0)

    def set(self, prop, value):
        self.sets.append((prop, int(value)))
        self.pos = int(value)
        return True

    def grab(self):
        if self.pos >= len(self.values):
            return False
        self._cur = self.pos
        self.pos += 1
        self.grabs += 1
        return True

    def retrieve(self):
        self.retrieves.append(self._cur)
        w, h = self.size
        return True, np.full((h, w, 3), self.values[self._cur], dtype=np.uint8)

    def release(self):
        self.released = True


class FakeProbe:
    """Detector backend double: positive when the frame is bright. Records thread names."""
    targeted = False

    def __init__(self, workers=2, gate=None, boom=False):
        self.workers = workers
        self.gate = gate
        self.boom = boom
        self.events = []
        self._lock = threading.Lock()

    def prepare(self):
        with self._lock:
            self.events.append(('prepare', threading.current_thread().name))
        return self.workers

    def positive(self, frame):
        name = threading.current_thread().name
        with self._lock:
            self.events.append(('positive', name))
        if self.gate is not None and name.startswith('sparse_det'):
            self.gate.wait(10)
        if self.boom and name.startswith('sparse_det'):
            raise RuntimeError('detector exploded')
        return float(frame.mean()) > 128


def _scan(values, probe=None, fps=30.0, **kw):
    cap = FakeCap(values, fps=fps)
    probe = probe or FakeProbe()
    out = scan_active_ranges('', capture=cap, detector=probe, **kw)
    return out, cap, probe


class StrideAndPadTest(unittest.TestCase):
    def test_stride_follows_fps(self):
        table = {24: 4, 25: 5, 29.97: 5, 30: 6, 50: 10, 60: 12, 120: 12, 5: 1, 1: 1, 0.5: 1}
        for fps, want in table.items():
            self.assertEqual(sparse_stride(fps), want, fps)

    def test_stride_bad_fps_uses_25(self):
        for bad in (0, -3, float('nan'), float('inf'), None, 'x'):
            self.assertEqual(sparse_stride(bad), 5, bad)

    def test_pad_is_ceil_1_25_s_plus_2(self):
        self.assertEqual({s: dilation_pad(s) for s in (1, 4, 5, 6, 12)},
                         {1: 4, 4: 7, 5: 9, 6: 10, 12: 17})

    def test_floors(self):
        self.assertAlmostEqual(det_score_floor(0.60), 0.45)
        self.assertAlmostEqual(det_score_floor(0.50), 0.35)
        self.assertAlmostEqual(det_score_floor(0.30), 0.25)       # never below 0.25
        self.assertAlmostEqual(similarity_floor(0.35), 0.23)
        self.assertAlmostEqual(similarity_floor(0.20), 0.15)      # never below 0.15


class CutDetectionTest(unittest.TestCase):
    def test_thumbnail_is_64x36_luma(self):
        t = luma_thumb(np.full((1080, 1920, 3), 100, np.uint8))
        self.assertEqual(t.shape, (36, 64))
        self.assertAlmostEqual(float(t.mean()), 100.0, places=0)

    def test_delta_scale(self):
        a = luma_thumb(np.zeros((90, 160, 3), np.uint8))
        b = luma_thumb(np.full((90, 160, 3), 255, np.uint8))
        self.assertAlmostEqual(luma_delta(a, a), 0.0)
        self.assertAlmostEqual(luma_delta(a, b), 255.0, places=0)
        self.assertGreater(255.0, CUT_DELTA_Y)


class PositiveGateTest(unittest.TestCase):
    class F(dict):
        __getattr__ = dict.get

    def test_universal_uses_relaxed_det_floor(self):
        floor = det_score_floor(0.60)
        self.assertTrue(frame_is_positive([self.F(det_score=0.46)], floor))
        self.assertFalse(frame_is_positive([self.F(det_score=0.40)], floor))
        self.assertFalse(frame_is_positive([], floor))

    def test_targeted_uses_cosine_not_det_score(self):
        rng = np.random.default_rng(0)
        target = rng.normal(size=512)
        other = rng.normal(size=512)
        tm = unit_embeddings([self.F(embedding=target)])
        floor = similarity_floor(0.35)
        near = target + 0.5 * rng.normal(size=512)            # steep-profile style: lower, still close
        self.assertTrue(frame_is_positive([self.F(embedding=near, det_score=0.1)], 0.45, tm, floor))
        self.assertFalse(frame_is_positive([self.F(embedding=other, det_score=0.99)], 0.45, tm, floor))
        # a face with no embedding can not vouch for the target
        self.assertFalse(frame_is_positive([self.F(det_score=0.99)], 0.45, tm, floor))
        # any one matching face is enough
        self.assertTrue(frame_is_positive(
            [self.F(embedding=other), self.F(embedding=target)], 0.45, tm, floor))

    def test_unit_embeddings_skips_missing_and_zero(self):
        self.assertIsNone(unit_embeddings([]))
        self.assertIsNone(unit_embeddings([self.F(det_score=1.0), self.F(embedding=np.zeros(4))]))
        m = unit_embeddings([self.F(embedding=np.array([3.0, 4.0]))])
        self.assertAlmostEqual(float(np.linalg.norm(m[0])), 1.0, places=5)


class BuildRangesTest(unittest.TestCase):
    def test_single_hit_dilated_by_m(self):
        ranges, mask = build_active_ranges([60], 200, 6)          # M = 10
        self.assertEqual(ranges, [(50, 70)])
        self.assertEqual(int(mask.sum()), 21)
        self.assertTrue(mask[50] and mask[70] and not mask[49] and not mask[71])

    def test_adjacent_samples_form_one_island(self):
        ranges, _ = build_active_ranges([30, 36, 42], 300, 6)
        self.assertEqual(ranges, [(20, 52)])

    def test_gap_of_exactly_45_coalesces_46_does_not(self):
        # islands (0,0) and (b,b), S=6, M=10: dilated (0,10) and (b-10, b+10); gap = b-20.
        r, _ = build_active_ranges([0, 65], 400, 6)
        self.assertEqual(r, [(0, 75)])                            # 55-10-... gap 45 -> merge
        r, _ = build_active_ranges([0, 66], 400, 6)
        self.assertEqual(r, [(0, 10), (56, 76)])                  # gap 46 -> separate
        self.assertEqual(MERGE_GAP, 45)

    def test_clamped_to_clip(self):
        ranges, _ = build_active_ranges([2, 197], 200, 6)
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], 199)

    def test_trim_bounds_clamp_and_filter(self):
        ranges, mask = build_active_ranges([5, 100], 200, 6, trim_in=20, trim_out=90)
        self.assertEqual(ranges, [])                              # both outside the trim
        self.assertFalse(mask.any())
        ranges, _ = build_active_ranges([25], 200, 6, trim_in=20, trim_out=90)
        self.assertEqual(ranges, [(20, 35)])

    def test_flicker_between_samples_merges(self):
        # sample 18 missed: islands (6,12) and (24,30) -> dilated ranges overlap -> one range
        ranges, _ = build_active_ranges([6, 12, 24, 30], 300, 6)
        self.assertEqual(ranges, [(0, 40)])

    def test_empty_is_empty_not_an_error(self):
        ranges, mask = build_active_ranges([], 300, 6)
        self.assertEqual(ranges, [])
        self.assertEqual(mask.shape, (300,))
        self.assertFalse(mask.any())

    def test_zero_length_clip(self):
        ranges, mask = build_active_ranges([10], 0, 6)
        self.assertEqual(ranges, [])
        self.assertEqual(len(mask), 0)

    def test_scene_cuts_expand_to_shot_boundaries(self):
        # Island [40, 50], S=6 (pad=10) -> normally (30, 60).
        # With scene cuts at 25 and 75, it expands to cover [25, 74].
        ranges, mask = build_active_ranges([40, 50], 100, 6, scene_cuts=[25, 75])
        self.assertEqual(ranges, [(25, 74)])
        self.assertTrue(mask[25] and mask[74])
        self.assertFalse(mask[24] or mask[75])

    def test_scene_cuts_without_prior_cut_retains_dilation_pad(self):
        # Island [40, 50], S=6 (pad=10) -> normally (30, 60).
        # Cut at 75 (post only): start stays max(0, 40-10)=30, end expands to 75-1=74.
        ranges, _ = build_active_ranges([40, 50], 100, 6, scene_cuts=[75])
        self.assertEqual(ranges, [(30, 74)])

    def test_scene_cuts_without_post_cut_retains_dilation_pad(self):
        # Island [40, 50], S=6 (pad=10) -> normally (30, 60).
        # Cut at 25 (prior only): start expands to 25, end stays min(99, 50+10)=60.
        ranges, _ = build_active_ranges([40, 50], 100, 6, scene_cuts=[25])
        self.assertEqual(ranges, [(25, 60)])


class DemuxTest(unittest.TestCase):
    def _run(self, values, stride, frame_start=0, total=None):
        cap = FakeCap(values)
        items = []
        info = run_demux(cap, frame_start, total or len(values), stride,
                         lambda it: items.append(it) or True, lambda: False)
        return cap, items, info

    def test_no_seek_and_retrieve_only_samples(self):
        cap, items, info = self._run([DARK] * 40, 6)
        self.assertEqual(cap.sets, [])                            # no seeking at all
        self.assertEqual(cap.grabs, 40)                           # every frame decoded once
        grid = [0, 6, 12, 18, 24, 30, 36, 39]                     # every S-th frame + the last
        mids = [3, 9, 15, 21, 27, 33, 38]                         # held one interval for a possible cut
        self.assertEqual(cap.retrieves, sorted(grid + mids))      # nothing else is ever retrieved
        self.assertEqual([i for k, i, _ in items if k == 'sample'], grid)
        self.assertEqual(info['cuts'], 0)
        self.assertEqual([i for k, i, _ in items if k == 'mid'], [])   # no cut -> no midpoint detected

    def test_single_initial_seek_when_trimmed(self):
        cap, items, _ = self._run([DARK] * 60, 6, frame_start=10, total=30)
        self.assertEqual(cap.sets, [(cv2.CAP_PROP_POS_FRAMES, 10)])
        self.assertEqual([i for k, i, _ in items], [0, 6, 12, 18, 24, 29])   # clip-relative
        self.assertEqual(sorted(set(cap.retrieves) - {13, 19, 25, 31, 37}),
                         [10, 16, 22, 28, 34, 39])                           # absolute reads (+ midpoints)

    def test_hard_cut_retrieves_and_emits_midpoint(self):
        values = [DARK] * 57 + [BRIGHT] * 63                      # cut lands inside (54, 60]
        cap, items, info = self._run(values, 6)
        self.assertEqual(info['cuts'], 1)
        self.assertEqual(info['mids'], 1)
        mids = [(i, f) for k, i, f in items if k == 'mid']
        self.assertEqual([i for i, _ in mids], [57])              # t - floor(S/2) = 60 - 3
        self.assertGreater(float(mids[0][1].mean()), 128)
        # the midpoint is emitted before the sample that revealed the cut
        order = [(k, i) for k, i, _ in items]
        self.assertLess(order.index(('mid', 57)), order.index(('sample', 60)))
        self.assertIn(57, cap.retrieves)

    def test_gradual_change_is_not_a_cut(self):
        values = [int(20 + 0.5 * i) for i in range(100)]          # +3 luma per 6-frame stride
        cap, items, info = self._run(values, 6)
        self.assertEqual(info['cuts'], 0)
        self.assertTrue(all(k == 'sample' for k, _, _ in items))

    def test_stride_one_and_two(self):
        cap, items, info = self._run([DARK] * 4 + [BRIGHT] * 4, 1)
        self.assertEqual(info['mids'], 0)                         # nothing between adjacent frames
        self.assertEqual(cap.retrieves, list(range(8)))
        cap, items, info = self._run([DARK] * 3 + [BRIGHT] * 5, 2)
        self.assertEqual(info['mids'], 1)
        self.assertEqual([i for k, i, _ in items if k == 'mid'], [3])        # 4 - floor(2/2)

    def test_last_frame_off_grid_midpoint(self):
        # grid 0,6,12 then last=14 (gap 2): cut between 12 and 14 -> midpoint 14 - 1 = 13
        cap, items, _ = self._run([DARK] * 13 + [BRIGHT] * 2, 6)
        self.assertEqual([(k, i) for k, i, _ in items][-3:], [('sample', 12), ('mid', 13), ('sample', 14)])

    def test_stop_flag_ends_demux(self):
        cap = FakeCap([DARK] * 100)
        state = {'n': 0}

        def stopped():
            state['n'] += 1
            return state['n'] > 10
        info = run_demux(cap, 0, 100, 6, lambda it: True, stopped)
        self.assertTrue(info['stopped'])
        self.assertLess(cap.grabs, 15)

    def test_short_video_ends_cleanly(self):
        cap = FakeCap([DARK] * 20)
        info = run_demux(cap, 0, 100, 6, lambda it: True, lambda: False)
        self.assertEqual(info['decoded'], 20)
        self.assertFalse(info['stopped'])


class ScanTest(unittest.TestCase):
    def test_ranges_from_samples(self):
        (ranges, mask, stats), cap, _ = _scan([DARK] * 60 + [BRIGHT] * 60)
        self.assertEqual(stats['stride'], 6)
        self.assertEqual(stats['pad'], 10)
        self.assertEqual(ranges, [(50, 119)])
        self.assertEqual(mask.shape, (120,))
        self.assertEqual(int(mask.sum()), 70)
        self.assertEqual(stats['mode'], 'sparse_auto_range')
        self.assertEqual(cap.sets, [])
        self.assertTrue(cap.released)

    def test_midpoint_finds_the_cut_the_stride_stepped_over(self):
        # bright from frame 57: samples 54 dark / 60 bright; only the midpoint (57) is positive
        (ranges, _, stats), _, _ = _scan([DARK] * 57 + [BRIGHT] * 63)
        self.assertEqual(ranges, [(47, 119)])
        self.assertEqual(stats['mids_inspected'], 1)

    def test_trimmed_interval_is_clip_relative(self):
        values = [DARK] * 40 + [BRIGHT] * 40 + [DARK] * 40
        (ranges, mask, _), cap, _ = _scan(values, frame_start=20, frame_end=100)
        self.assertEqual(len(mask), 80)
        # bright is abs 40..79 = rel 20..59. Both edges are cuts the grid steps over, so the
        # midpoints (rel 21 entering, rel 57 leaving) are the first and last positives: 21-10, 57+10.
        self.assertEqual(ranges, [(11, 67)])
        self.assertEqual(cap.sets, [(cv2.CAP_PROP_POS_FRAMES, 20)])

    def test_nothing_found_is_empty_list_no_exception(self):
        (ranges, mask, stats), _, _ = _scan([DARK] * 120)
        self.assertEqual(ranges, [])
        self.assertEqual(mask.shape, (120,))
        self.assertFalse(mask.any())
        self.assertEqual(stats['mode'], 'no_target_found')
        self.assertEqual(stats['active_frames'], 0)

    def test_barrier_prepare_and_warmup_precede_every_worker(self):
        probe = FakeProbe(workers=3)
        _scan([DARK] * 60 + [BRIGHT] * 60, probe=probe)
        main = threading.current_thread().name
        self.assertEqual(probe.events[0], ('prepare', main))
        warm = probe.events[1:1 + 3]
        self.assertEqual(warm, [('positive', main)] * 3)          # one warm call per pool instance
        self.assertTrue(all(n.startswith('sparse_det') for k, n in probe.events[4:] if k == 'positive'))
        self.assertGreater(len([e for e in probe.events[4:] if e[0] == 'positive']), 10)

    def test_work_queue_is_bounded(self):
        gate = threading.Event()
        probe = FakeProbe(workers=2, gate=gate)
        cap = FakeCap([DARK] * 3000)
        box = {}
        t = threading.Thread(target=lambda: box.update(
            out=scan_active_ranges('', capture=cap, detector=probe)), daemon=True)
        t.start()
        time.sleep(0.6)                                           # workers are stuck on the gate
        stuck_at = len(cap.retrieves)
        gate.set()
        t.join(30)
        self.assertFalse(t.is_alive())
        # queue 4 + 2 in flight + 1 blocked in put = 7 samples, each with its midpoint retrieved
        # ahead of time: ~14 retrieves. An unbounded reader would have made ~1000 by now.
        self.assertLessEqual(stuck_at, 18)
        self.assertEqual(box['out'][2]['mode'], 'no_target_found')

    def test_detector_errors_and_no_positive_fail_safe_to_full(self):
        (ranges, mask, stats), _, _ = _scan([DARK] * 120, probe=FakeProbe(boom=True))
        self.assertEqual(ranges, [(0, 119)])
        self.assertTrue(mask.all())
        self.assertEqual(stats['mode'], 'fail_safe_full')
        self.assertGreater(stats['detector_errors'], 0)

    def test_stop_requested_fails_safe_even_with_positives(self):
        n = {'c': 0}

        def stop():
            n['c'] += 1
            return n['c'] > 60
        (ranges, mask, stats), _, _ = _scan([BRIGHT] * 600, should_stop=stop)
        self.assertEqual(ranges, [(0, 599)])
        self.assertEqual(stats['mode'], 'fail_safe_full')
        self.assertEqual(stats['reason'], 'stopped')

    def test_seek_that_lands_elsewhere_fails_safe(self):
        class OffByTwo(FakeCap):
            def get(self, prop):
                return self.pos + 2 if prop == cv2.CAP_PROP_POS_FRAMES and self.sets else super().get(prop)
        cap = OffByTwo([BRIGHT] * 200)
        ranges, mask, stats = scan_active_ranges('', capture=cap, detector=FakeProbe(),
                                                 frame_start=20, frame_end=200)
        self.assertEqual(ranges, [(0, 179)])
        self.assertEqual(stats['mode'], 'fail_safe_full')
        self.assertIn('seek', stats['reason'])

    def test_short_clip_is_processed_in_full(self):
        (ranges, mask, stats), _, probe = _scan([DARK] * 10)
        self.assertEqual(ranges, [(0, 9)])
        self.assertEqual(stats['mode'], 'full_short_clip')
        self.assertEqual(probe.events, [])                        # nothing was built or run

    def test_missing_file_fails_safe(self):
        ranges, mask, stats = scan_active_ranges('Z:/no/such/file.mp4', frame_count=50)
        self.assertEqual(ranges, [(0, 49)])
        self.assertEqual(stats['error'], 'file_not_found')

    def test_progress_callback_runs_on_calling_thread(self):
        seen = []
        _scan([DARK] * 120, progress_cb=lambda d, t: seen.append((d, t, threading.current_thread().name)))
        self.assertTrue(seen)
        self.assertEqual({n for _, _, n in seen}, {threading.current_thread().name})
        self.assertEqual(seen[-1][0], seen[-1][1])                # ends at samples_total


class RealVideoTest(unittest.TestCase):
    """Decode a real MJPG file: sequential grab/retrieve must line up with frame indices."""

    def _write(self, path, values, fps=30):
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (160, 90))
        if not w.isOpened():
            self.skipTest('MJPG writer unavailable')
        for v in values:
            w.write(np.full((90, 160, 3), v, np.uint8))
        w.release()

    def test_real_capture_alignment_and_trim(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'cut.avi')
            self._write(path, [DARK] * 30 + [BRIGHT] * 30)
            ranges, _, stats = scan_active_ranges(path, detector=FakeProbe(workers=2))
            self.assertEqual(stats['stride'], 6)
            self.assertEqual(ranges, [(20, 59)])                  # first bright sample 30, minus M=10
            # trimmed: abs 10..50 -> bright from rel 20; the cut between samples 18 and 24 is
            # probed at its midpoint 21, which is bright: 21 - 10 = 11
            ranges, mask, _ = scan_active_ranges(path, frame_start=10, frame_end=50,
                                                 detector=FakeProbe(workers=2))
            self.assertEqual(len(mask), 40)
            self.assertEqual(ranges, [(11, 39)])


class FaceAnalyserProbeTest(unittest.TestCase):
    def test_prepare_degrades_to_universal_without_recognition(self):
        from roop import face_util
        orig = (face_util.prescan_pool_size, face_util.prescan_has_recognition)
        face_util.prescan_pool_size = lambda: 2
        face_util.prescan_has_recognition = lambda: False
        try:
            probe = srs.FaceAnalyserProbe(0.45, np.eye(2, 512, dtype=np.float32), 0.23)
            self.assertEqual(probe.prepare(), 2)
            self.assertFalse(probe.targeted)
        finally:
            face_util.prescan_pool_size, face_util.prescan_has_recognition = orig

    def _probe_with(self, per_turn, **kw):
        """FaceAnalyserProbe whose detector returns `per_turn[turn]` (a list of faces); records calls."""
        from roop import face_util
        calls = []

        def fake_detect(frame, det_thresh, embed=False, turn=None):
            calls.append((turn, det_thresh, embed))
            return per_turn.get(turn, [])
        saved = (face_util.prescan_pool_size, face_util.prescan_has_recognition,
                 face_util.detect_faces_for_prescan)
        face_util.prescan_pool_size = lambda: 1
        face_util.prescan_has_recognition = lambda: True
        face_util.detect_faces_for_prescan = fake_detect
        self.addCleanup(lambda: (setattr(face_util, 'prescan_pool_size', saved[0]),
                                 setattr(face_util, 'prescan_has_recognition', saved[1]),
                                 setattr(face_util, 'detect_faces_for_prescan', saved[2])))
        probe = srs.FaceAnalyserProbe(**kw)
        probe.prepare()
        return probe, calls

    FRAME = np.zeros((8, 8, 3), np.uint8)
    E = np.eye(1, 512, dtype=np.float32)[0]

    def test_positive_asks_detector_for_relaxed_floor_and_embeddings_only_when_targeted(self):
        good = [PositiveGateTest.F(det_score=0.5, embedding=self.E)]
        uni, calls = self._probe_with({None: good}, det_floor=0.45)
        self.assertTrue(uni.positive(self.FRAME))
        tgt, calls_t = self._probe_with({None: good}, det_floor=0.45,
                                        target_matrix=self.E[None, :], sim_floor=0.23)
        self.assertTrue(tgt.positive(self.FRAME))
        self.assertEqual(calls, [(None, 0.45, False)])
        self.assertEqual(calls_t, [(None, 0.45, True)])

    def test_turns_are_tried_in_order_only_when_the_upright_pass_is_not_positive(self):
        probe, calls = self._probe_with({"anticlockwise": [PositiveGateTest.F(det_score=0.9)]},
                                        det_floor=0.35)
        self.assertTrue(probe.positive(self.FRAME))
        self.assertEqual([t for t, _, _ in calls], [None, "clockwise", "anticlockwise"])   # stops at the hit
        probe, calls = self._probe_with({None: [PositiveGateTest.F(det_score=0.9)]}, det_floor=0.35)
        probe.positive(self.FRAME)
        self.assertEqual([t for t, _, _ in calls], [None])

    def test_junk_detection_does_not_switch_the_rescue_off(self):
        # angle_inverted.mp4 frames 299-300: a 0.42 upright detection below the 0.45 floor made the
        # frame non-empty; the real inverted face was one turn away. Rescue keys on "not positive".
        junk = [PositiveGateTest.F(det_score=0.42)]
        real = [PositiveGateTest.F(det_score=0.93)]
        probe, calls = self._probe_with({None: junk, "180": real}, det_floor=0.45)
        self.assertTrue(probe.positive(self.FRAME))
        self.assertEqual([t for t, _, _ in calls], [None, "clockwise", "anticlockwise", "180"])

    def test_targeted_wrong_identity_still_tries_every_turn(self):
        other = np.eye(2, 512, dtype=np.float32)[1]
        stranger = [PositiveGateTest.F(det_score=0.99, embedding=other)]
        probe, calls = self._probe_with({None: stranger}, det_floor=0.45,
                                        target_matrix=self.E[None, :], sim_floor=0.23)
        self.assertFalse(probe.positive(self.FRAME))
        self.assertEqual([t for t, _, _ in calls], [None, "clockwise", "anticlockwise", "180"])

    def test_rescue_off_runs_one_pass(self):
        probe, calls = self._probe_with({}, det_floor=0.45, rescue=False)
        self.assertFalse(probe.positive(self.FRAME))
        self.assertEqual([t for t, _, _ in calls], [None])

    def test_stage1_never_reaches_the_cudnn_probe(self):
        # cudnn_algo.probe is only reached for SUSPECT_MODEL_KEYS (enhancers) via
        # precision_policy; the analyser's keys are not among them.
        from roop import cudnn_algo
        for key in ('recognition:buffalo_l', 'face_detection', 'detector:face-analysis'):
            self.assertNotIn(key, cudnn_algo.SUSPECT_MODEL_KEYS)


class SparseRangeScanSettingsWiringTest(unittest.TestCase):
    def test_sparse_auto_range_default_in_settings(self):
        s = Settings.__new__(Settings)
        data = {}
        s.sparse_auto_range = s.default_get(data, 'sparse_auto_range', False)
        self.assertFalse(s.sparse_auto_range)

    def test_sparse_auto_range_enabled_in_settings(self):
        s = Settings.__new__(Settings)
        data = {'sparse_auto_range': True}
        s.sparse_auto_range = s.default_get(data, 'sparse_auto_range', False)
        self.assertTrue(s.sparse_auto_range)

    def test_sparse_auto_range_default_in_globals(self):
        self.assertFalse(getattr(roop.globals, 'sparse_auto_range', True))


if __name__ == '__main__':
    unittest.main()
