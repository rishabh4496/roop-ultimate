"""Frame-exact positioning of the Stage 2 pre-pass capture (roop/range_seek.py).

The fixtures carry their own frame number in the picture (three base-16 digits in flat
blocks, readable after lossy coding), so "the frame the pre-pass got for index k is frame
k" is checked against the pixels, not against another decode that could share the error.

The variable-frame-rate clip is the one that matters: a time seek lands a frame LATE on
about a third of ranges there, and `get(POS_FRAMES)` cannot see it (cv2 echoes the
requested position; the pipe reader used to return a constant 0.0).
"""
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals                                            # noqa: E402
from roop import range_seek                                    # noqa: E402
from roop.range_seek import RangeSeeker, seeker_for, timestamps_on_grid  # noqa: E402

try:
    from roop.ffmpeg_path import ffmpeg_binary
    _FFMPEG = ffmpeg_binary()
except Exception:                                               # pragma: no cover
    _FFMPEG = None

_GEQ = ("geq=lum='if(lt(Y,H/2),if(lt(X,W/2),mod(N,16)*13+24,mod(floor(N/16),16)*13+24),"
        "if(lt(X,W/2),mod(floor(N/256),16)*13+24,24))':cb=128:cr=128")
_VFR_PTS = "setpts='(N/30+0.02*floor(N/37)+0.013*sin(N/9))/TB'"
FRAMES = 900


def _make_clip(path, vfr):
    vf = _GEQ + ("," + _VFR_PTS if vfr else "")
    cmd = [_FFMPEG, '-y', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
           '-i', 'color=c=gray:s=160x96:r=30', '-frames:v', str(FRAMES), '-vf', vf,
           '-c:v', 'libx264', '-crf', '14', '-g', '120', '-bf', '3', '-pix_fmt', 'yuv420p']
    if vfr:
        cmd += ['-fps_mode', 'vfr']
    subprocess.run(cmd + [path], check=True, capture_output=True)


def frame_number(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape

    def digit(block):
        return int(round((float(block.mean()) - 9.3) / 15.14))
    return digit(g[:h // 2, :w // 2]) + 16 * digit(g[:h // 2, w // 2:]) + 256 * digit(g[h // 2:, :w // 2])


class TimestampGrid(unittest.TestCase):
    def test_constant_rate_is_on_the_grid(self):
        self.assertTrue(timestamps_on_grid([k / 30.0 for k in range(300)], 30.0))

    def test_decode_order_with_b_frames_is_still_constant_rate(self):
        pts = [k / 24.0 for k in range(240)]
        random.Random(1).shuffle(pts)
        self.assertTrue(timestamps_on_grid(pts, 24.0))

    def test_ntsc_rate_rounding_is_tolerated(self):
        fps = 24000.0 / 1001.0
        pts = [round(k / fps, 3) for k in range(3000)]          # millisecond timestamps
        self.assertTrue(timestamps_on_grid(pts, fps))

    def test_a_jump_in_the_timestamps_is_not(self):
        pts = [k / 30.0 + (0.5 if k >= 100 else 0.0) for k in range(300)]
        self.assertFalse(timestamps_on_grid(pts, 30.0))

    def test_slow_drift_off_the_nominal_rate_is_not(self):
        # the container says 30 fps, the stream plays at 29.5: a time seek drifts a frame
        # every ~60 frames
        pts = [k / 29.5 for k in range(600)]
        self.assertFalse(timestamps_on_grid(pts, 30.0))

    def test_too_little_to_judge_is_on_the_grid(self):
        self.assertTrue(timestamps_on_grid([], 30.0))
        self.assertTrue(timestamps_on_grid([0.0], 30.0))
        self.assertTrue(timestamps_on_grid([0.0, 0.5], 0.0))


class _Cap:
    """cv2.VideoCapture protocol over `n` numbered frames; records every call."""

    def __init__(self, n=1000, grab=True):
        self.n, self.at, self.calls = n, 0, []
        if not grab:
            self.grab = None

    def set(self, prop, value):
        self.calls.append(('set', int(value)))
        self.at = int(value)

    def grab(self):
        self.calls.append(('grab',))
        if self.at >= self.n:
            return False
        self.at += 1
        return True

    def read(self):
        self.calls.append(('read',))
        if self.at >= self.n:
            return False, None
        self.at += 1
        return True, np.full((2, 2, 3), self.at - 1, np.int32)

    def get(self, prop):
        return 0.0


class SeekerPolicy(unittest.TestCase):
    def test_cfr_seeks_and_tracks_position_itself(self):
        cap = _Cap()
        s = RangeSeeker(cap, seq_gap_max=0)
        self.assertTrue(s.goto(100))
        self.assertEqual(cap.calls, [('set', 100)])
        for _ in range(5):
            s.read()
        self.assertEqual(s.pos, 105)
        self.assertTrue(s.goto(400))
        self.assertEqual(cap.calls[-1], ('set', 400))
        self.assertEqual(s.stats['walked'], 0)

    def test_already_there_is_a_no_op(self):
        cap = _Cap()
        s = RangeSeeker(cap, seq_gap_max=0)
        s.goto(10)
        s.read()
        cap.calls.clear()
        self.assertTrue(s.goto(11))
        self.assertEqual(cap.calls, [])

    def test_sequential_never_seeks_forward(self):
        cap = _Cap()
        s = RangeSeeker(cap, sequential=True)
        for target in (30, 40, 500):
            self.assertTrue(s.goto(target))
            ok, fr = s.read()
            self.assertEqual(int(fr[0, 0, 0]), target)
        self.assertFalse(any(c[0] == 'set' for c in cap.calls))
        self.assertEqual(s.stats['walks'], 3)

    def test_sequential_walks_with_read_when_the_capture_has_no_grab(self):
        cap = _Cap(grab=False)
        s = RangeSeeker(cap, sequential=True)
        self.assertTrue(s.goto(25))
        self.assertEqual(s.read()[1][0, 0, 0], 25)
        self.assertFalse(any(c[0] in ('grab', 'set') for c in cap.calls))

    def test_small_gap_is_walked_large_gap_is_seeked(self):
        cap = _Cap()
        s = RangeSeeker(cap, seq_gap_max=100)
        s.goto(500)                         # first move: gap 500 > 100 -> seek
        s.read()
        self.assertEqual(cap.calls[0], ('set', 500))
        cap.calls.clear()
        s.goto(551)                         # gap 50 -> walk
        self.assertFalse(any(c[0] == 'set' for c in cap.calls))
        s.read()
        cap.calls.clear()
        s.goto(900)                         # gap 348 -> seek
        self.assertEqual(cap.calls, [('set', 900)])

    def test_backward_move_restarts_from_zero_and_lands_exactly(self):
        cap = _Cap()
        s = RangeSeeker(cap, sequential=True)
        s.goto(200)
        s.read()
        self.assertTrue(s.goto(50))
        self.assertEqual(s.read()[1][0, 0, 0], 50)
        self.assertIn(('set', 0), cap.calls)

    def test_backward_on_a_constant_rate_source_is_one_direct_seek(self):
        cap = _Cap()
        s = RangeSeeker(cap, seq_gap_max=0)
        s.goto(500)
        s.read()
        cap.calls.clear()
        self.assertTrue(s.goto(100))
        self.assertEqual(cap.calls, [('set', 100)])
        self.assertEqual(s.read()[1][0, 0, 0], 100)

    def test_end_of_stream_while_walking_is_reported(self):
        s = RangeSeeker(_Cap(n=40), sequential=True)
        self.assertFalse(s.goto(100))
        self.assertEqual(s.stats['failed'], 1)

    def test_env_override_of_the_walk_threshold(self):
        with mock.patch.dict(os.environ, {'ROOP_RANGE_SEEK_SEQ_GAP': '250'}):
            self.assertEqual(RangeSeeker(_Cap()).seq_gap_max, 250)
        with mock.patch.dict(os.environ, {'ROOP_RANGE_SEEK_SEQ_GAP': 'junk'}):
            self.assertEqual(RangeSeeker(_Cap()).seq_gap_max, range_seek.SEQ_GAP_DEFAULT)

    def test_nothing_to_seek_skips_the_frame_rate_probe(self):
        with mock.patch.object(range_seek, 'source_is_cfr') as probe:
            s = seeker_for(_Cap(), 'x.mp4', fps=30.0, n_ranges=1, first_start=0)
        probe.assert_not_called()
        self.assertFalse(s.sequential)

    def test_unknown_frame_rate_keeps_the_seek(self):
        with mock.patch.object(range_seek, 'source_is_cfr', return_value=None):
            self.assertFalse(seeker_for(_Cap(), 'x.mp4', fps=30.0, n_ranges=3).sequential)
        with mock.patch.object(range_seek, 'source_is_cfr', return_value=False):
            self.assertTrue(seeker_for(_Cap(), 'x.mp4', fps=30.0, n_ranges=3).sequential)


@unittest.skipUnless(_FFMPEG and shutil.which(_FFMPEG) or (_FFMPEG and os.path.isfile(_FFMPEG)),
                     'ffmpeg not available')
class RealClips(unittest.TestCase):
    RANGES = [(40, 70), (133, 160), (301, 330), (517, 540), (640, 662), (777, 800), (850, 870)]

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix='roop_range_seek_')
        cls.cfr = os.path.join(cls.dir, 'cfr.mp4')
        cls.vfr = os.path.join(cls.dir, 'vfr.mp4')
        try:
            _make_clip(cls.cfr, vfr=False)
            _make_clip(cls.vfr, vfr=True)
        except Exception as exc:                                 # no libx264 in this build
            shutil.rmtree(cls.dir, ignore_errors=True)
            raise unittest.SkipTest(f'cannot build the fixture clips: {exc}')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def _open_cv2(self, path):
        return cv2.VideoCapture(path)

    def _open_pipe(self, path):
        from roop.nvdec_reader import FFmpegVideoReader
        c = cv2.VideoCapture(path)
        w, h, fps = int(c.get(3)), int(c.get(4)), c.get(cv2.CAP_PROP_FPS)
        c.release()
        return FFmpegVideoReader(path, w, h, fps, hwaccel=None, prefetch_depth=2)

    def _landed(self, cap_factory, path, seeker_kw):
        """Frame number the capture delivers first, for every range, via a RangeSeeker."""
        cap = cap_factory(path)
        s = seeker_for(cap, path, n_ranges=len(self.RANGES), first_start=self.RANGES[0][0], **seeker_kw)
        got = []
        for a, b in self.RANGES:
            self.assertTrue(s.goto(a))
            first = None
            for k in range(a, min(b, a + 3) + 1):
                ok, fr = s.read()
                self.assertTrue(ok)
                if first is None:
                    first = frame_number(fr)
                self.assertEqual(frame_number(fr), k, f'range starting {a}: frame {k} is wrong')
            got.append(first)
        cap.release()
        return s, got

    def _naive_misses(self, cap_factory, path):
        misses = 0
        for a, _ in self.RANGES:
            cap = cap_factory(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, a)
            ok, fr = cap.read()
            misses += int(not ok or frame_number(fr) != a)
            cap.release()
        return misses

    def test_the_fixture_numbers_its_frames(self):
        cap = cv2.VideoCapture(self.vfr)
        ids = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            ids.append(frame_number(fr))
        self.assertEqual(ids, list(range(FRAMES)))

    def test_cfr_is_detected_and_seeked_exactly_cv2(self):
        s, got = self._landed(self._open_cv2, self.cfr, {})
        self.assertFalse(s.sequential)
        self.assertEqual(s.stats['walked'], 0)
        self.assertEqual(got, [a for a, _ in self.RANGES])

    def test_cfr_is_detected_and_seeked_exactly_pipe(self):
        s, got = self._landed(self._open_pipe, self.cfr, {})
        self.assertFalse(s.sequential)
        self.assertEqual(got, [a for a, _ in self.RANGES])

    def test_vfr_is_detected(self):
        s = seeker_for(cv2.VideoCapture(self.vfr), self.vfr, n_ranges=3, first_start=10)
        self.assertTrue(s.sequential)

    def test_vfr_ranges_start_on_the_right_frame_cv2(self):
        if self._naive_misses(self._open_cv2, self.vfr) == 0:
            self.skipTest('this decoder seeks the VFR fixture exactly; nothing to prove here')
        s, got = self._landed(self._open_cv2, self.vfr, {})
        self.assertTrue(s.sequential)
        self.assertEqual(got, [a for a, _ in self.RANGES])
        self.assertEqual(s.stats['seeks'], 0)

    def test_vfr_ranges_start_on_the_right_frame_pipe(self):
        if self._naive_misses(self._open_pipe, self.vfr) == 0:
            self.skipTest('this decoder seeks the VFR fixture exactly; nothing to prove here')
        s, got = self._landed(self._open_pipe, self.vfr, {})
        self.assertTrue(s.sequential)
        self.assertEqual(got, [a for a, _ in self.RANGES])

    def test_pipe_reader_reports_where_it_is(self):
        cap = self._open_pipe(self.cfr)
        try:
            self.assertEqual(cap.get(cv2.CAP_PROP_POS_FRAMES), 0.0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 120)
            self.assertEqual(cap.get(cv2.CAP_PROP_POS_FRAMES), 120.0)
            ok, fr = cap.read()
            self.assertTrue(ok)
            self.assertEqual(frame_number(fr), 120)
            self.assertEqual(cap.get(cv2.CAP_PROP_POS_FRAMES), 121.0)
            self.assertTrue(cap.grab())
            self.assertEqual(cap.get(cv2.CAP_PROP_POS_FRAMES), 122.0)
            ok, fr = cap.read()
            self.assertEqual(frame_number(fr), 122)
        finally:
            cap.release()

    def test_probe_is_cached_per_file(self):
        range_seek._CFR_CACHE.clear()
        with mock.patch.object(range_seek, '_probe_pts', wraps=range_seek._probe_pts) as probe:
            self.assertIs(range_seek.source_is_cfr(self.cfr, 30.0), True)
            self.assertIs(range_seek.source_is_cfr(self.cfr, 30.0), True)
            self.assertIs(range_seek.source_is_cfr(self.vfr, 30.0), False)
        self.assertEqual(probe.call_count, 2)

    def test_missing_file_is_unknown_not_an_error(self):
        self.assertIsNone(range_seek.source_is_cfr(os.path.join(self.dir, 'nope.mp4'), 30.0))
        self.assertIsNone(range_seek.source_is_cfr(None, 30.0))


if __name__ == '__main__':
    unittest.main()
