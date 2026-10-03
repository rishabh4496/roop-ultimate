"""The live pipeline monitor: exact FPS, honest starvation verdicts, never fatal."""

import os
import queue
import sys
import threading
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import pipeline_monitor as pm  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(every=100, state=None, clock=None):
    clock = clock or Clock()
    out = []
    mon = pm.PipelineMonitor(every, state or (lambda: {'input': (3, 4), 'output': (0, 4)}),
                             emit=out.append, clock=clock)
    return mon, clock, out


def run_frames(mon, clock, n, dt):
    for _ in range(n):
        clock.t += dt
        mon.frame_done()


class WindowTest(unittest.TestCase):

    def test_one_line_per_window_with_exact_fps(self):
        mon, clock, out = make(every=100)
        mon.start()
        run_frames(mon, clock, 250, 0.1)          # 10 fps
        self.assertEqual(len(out), 2)
        self.assertIn('frames 100', out[0])
        self.assertIn('frames 200', out[1])
        self.assertIn('10.00 fps (avg 10.00)', out[0])
        self.assertIn('10.00 fps (avg 10.00)', out[1])

    def test_window_fps_tracks_a_speed_change_while_avg_lags(self):
        mon, clock, out = make(every=100)
        mon.start()
        run_frames(mon, clock, 100, 0.1)          # 10 fps
        run_frames(mon, clock, 100, 0.05)         # 20 fps
        self.assertIn('20.00 fps (avg 13.33)', out[1])

    def test_default_cadence_is_one_hundred(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('ROOP_PIPELINE_LOG_EVERY', None)
            self.assertEqual(pm.log_every_from_env(), 100)
        with mock.patch.dict(os.environ, {'ROOP_PIPELINE_LOG_EVERY': '25'}):
            self.assertEqual(pm.log_every_from_env(), 25)
        with mock.patch.dict(os.environ, {'ROOP_PIPELINE_LOG_EVERY': 'junk'}):
            self.assertEqual(pm.log_every_from_env(), 100)
        with mock.patch.dict(os.environ, {'ROOP_PIPELINE_LOG_EVERY': '0'}):
            self.assertEqual(pm.log_every_from_env(), 0)

    def test_zero_disables_everything_including_sampling(self):
        calls = []
        mon, clock, out = make(every=0, state=lambda: calls.append(1) or {})
        run_frames(mon, clock, 300, 0.1)
        self.assertEqual(out, [])
        self.assertEqual(calls, [])
        self.assertIsNone(mon.finish())


class VerdictTest(unittest.TestCase):

    def window(self, state):
        mon, clock, out = make(every=100, state=state)
        mon.start()
        run_frames(mon, clock, 100, 0.1)
        return out[0]

    def test_a_full_input_queue_is_healthy(self):
        # A GPU-bound pipeline SHOULD have a reader that is ahead of it.
        line = self.window(lambda: {'input': (4, 4), 'output': (0, 4)})
        self.assertTrue(line.endswith('| OK'), line)

    def test_an_empty_input_queue_is_decode_starvation(self):
        line = self.window(lambda: {'input': (0, 4), 'output': (0, 4)})
        self.assertIn('DECODE-STARVED', line)
        self.assertIn('empty 100%', line)

    def test_a_full_output_queue_is_encoder_backpressure(self):
        line = self.window(lambda: {'input': (3, 4), 'output': (4, 4)})
        self.assertIn('ENCODER-BOUND', line)

    def test_multi_queue_sides_use_the_fraction_of_total_capacity(self):
        # 20 per-thread queues, 18 of 20 slots used = 90% = full.
        line = self.window(lambda: {'input': (5, 20), 'output': (18, 20)})
        self.assertIn('ENCODER-BOUND', line)
        line = self.window(lambda: {'input': (5, 20), 'output': (17, 20)})
        self.assertTrue(line.endswith('| OK'), line)

    def test_starvation_must_persist_to_be_called(self):
        """An occasionally empty queue is not a verdict; a quarter of samples is."""
        seq = iter([(0, 4)] * 10 + [(3, 4)] * 90)
        line = self.window(lambda: {'input': next(seq), 'output': (0, 4)})
        self.assertTrue(line.endswith('| OK'), line)
        seq = iter([(0, 4)] * 30 + [(3, 4)] * 70)
        line = self.window(lambda: {'input': next(seq), 'output': (0, 4)})
        self.assertIn('DECODE-STARVED', line)

    def test_zero_capacity_is_not_reported_as_empty(self):
        line = self.window(lambda: {'input': (0, 0), 'output': (0, 0)})
        self.assertTrue(line.endswith('| OK'), line)


class UnitTest(unittest.TestCase):

    def test_a_non_frame_unit_is_named_in_the_line(self):
        mon, clock, out = make(every=100, state=lambda: {
            'input': (1, 2), 'output': (0, 2), 'unit': 'chunks'})
        mon.start()
        run_frames(mon, clock, 100, 0.1)
        self.assertIn('in 1.0/2 chunks', out[0])
        self.assertIn('out 0.0/2 chunks', out[0])

    def test_no_unit_means_plain_frames(self):
        mon, clock, out = make(every=100)
        mon.start()
        run_frames(mon, clock, 100, 0.1)
        self.assertIn('in 3.0/4 (empty', out[0])
        self.assertNotIn('chunks', out[0])


class SafetyTest(unittest.TestCase):

    def test_a_failing_state_probe_never_stops_the_render(self):
        def boom():
            raise RuntimeError('probe died')
        mon, clock, out = make(every=100, state=boom)
        mon.start()
        run_frames(mon, clock, 200, 0.1)
        self.assertEqual(len(out), 2)
        self.assertIn('queues n/a', out[0])
        self.assertIn('10.00 fps', out[0])

    def test_a_failing_emit_never_stops_the_render(self):
        clock = Clock()
        mon = pm.PipelineMonitor(100, lambda: {}, emit=lambda s: 1 / 0, clock=clock)
        mon.start()
        run_frames(mon, clock, 100, 0.1)             # must not raise
        self.assertEqual(len(mon.lines), 1)

    def test_thread_safe_frame_counting(self):
        mon = pm.PipelineMonitor(100, lambda: {'input': (1, 4), 'output': (0, 4)},
                                 emit=lambda s: None)
        mon.start()

        def worker():
            for _ in range(500):
                mon.frame_done()

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(mon.frames, 4000)
        self.assertEqual(len(mon.lines), 40)          # exactly one line per 100


class FinishTest(unittest.TestCase):

    def test_summary_covers_the_whole_run_including_the_partial_window(self):
        mon, clock, out = make(every=100, state=lambda: {'input': (0, 4), 'output': (0, 4)})
        mon.start()
        run_frames(mon, clock, 150, 0.1)
        summary = mon.finish()
        self.assertIn('done: 150 frames in 15.0s = 10.00 fps', summary)
        self.assertIn('decode-starved 100%', summary)

    def test_nothing_ran_nothing_to_say(self):
        mon, clock, out = make()
        self.assertIsNone(mon.finish())


class QueueStateSelectionTest(unittest.TestCase):
    """ProcessMgr must read the queues of the path that is actually running."""

    def setUp(self):
        try:
            from roop.ProcessMgr import ProcessMgr
        except Exception as exc:
            self.skipTest('ProcessMgr unavailable: %s' % exc)
        self.fn = ProcessMgr._pipeline_queue_state

    @staticmethod
    def q(n, cap):
        item = queue.Queue(cap)
        for _ in range(n):
            item.put(0)
        return item

    def ns(self, **kw):
        base = dict(frames_queue=None, processed_queue=None, _runtime_scheduler=None,
                    _runtime_read_queue=None, _runtime_write_queue=None)
        base.update(kw)
        return types.SimpleNamespace(**base)

    def test_threaded_workers_sum_their_per_thread_queues(self):
        state = self.fn(self.ns(frames_queue=[self.q(1, 3), self.q(2, 3)],
                                processed_queue=[self.q(0, 3), self.q(1, 3)]))
        self.assertEqual(state, {'input': (3, 6), 'output': (1, 6)})

    def test_parallel_stab_ignores_the_unused_per_thread_lists(self):
        """The per-thread lists exist on EVERY render; on this path they are
        structural zeros that would read as permanent decode starvation."""
        state = self.fn(self.ns(frames_queue=[self.q(0, 3)] * 4,
                                processed_queue=[self.q(0, 3)] * 4,
                                _runtime_read_queue=self.q(1, 2),
                                _runtime_write_queue=self.q(2, 2)))
        # One slot on this path is a whole chunk, and the state must say so.
        self.assertEqual(state, {'input': (1, 2), 'output': (2, 2), 'unit': 'chunks'})

    def test_the_one_owner_stream_reads_its_live_queues(self):
        sched = types.SimpleNamespace(live_queues=(self.q(2, 4), self.q(1, 4)))
        state = self.fn(self.ns(frames_queue=[self.q(0, 3)], _runtime_scheduler=sched,
                                _runtime_read_queue=self.q(0, 2)))
        self.assertEqual(state, {'input': (2, 4), 'output': (1, 4)})

    def test_idle_scheduler_without_live_queues_falls_through(self):
        sched = types.SimpleNamespace(live_queues=None)
        state = self.fn(self.ns(frames_queue=[self.q(1, 3)], processed_queue=[self.q(0, 3)],
                                _runtime_scheduler=sched))
        self.assertEqual(state, {'input': (1, 3), 'output': (0, 3)})


class FacesTest(unittest.TestCase):
    """fps alone misreads a render: the cost of a frame is its face count."""

    def test_window_reports_faces_per_frame_and_per_second(self):
        mon, clock, out = make(every=100)
        mon.start()
        for _ in range(100):
            clock.t += 0.1                            # 10 fps
            mon.frame_done(work=2)
        self.assertIn('| 2.00 faces/frame, 20.0 faces/s', out[0])

    def test_faces_per_second_stays_flat_when_fps_halves_as_faces_double(self):
        mon, clock, out = make(every=100)
        mon.start()
        for _ in range(100):
            clock.t += 0.05                           # 20 fps, 1 face a frame
            mon.frame_done(work=1)
        for _ in range(100):
            clock.t += 0.10                           # 10 fps, 2 faces a frame
            mon.frame_done(work=2)
        self.assertIn('20.00 fps', out[0])
        self.assertIn('10.00 fps', out[1])
        self.assertTrue(out[0].endswith('1.00 faces/frame, 20.0 faces/s'))
        self.assertTrue(out[1].endswith('2.00 faces/frame, 20.0 faces/s'))

    def test_a_path_that_cannot_count_faces_prints_the_old_line(self):
        mon, clock, out = make(every=100)
        mon.start()
        run_frames(mon, clock, 100, 0.1)
        self.assertNotIn('faces', out[0])

    def test_closing_summary_carries_the_whole_run_figure(self):
        mon, clock, out = make(every=100)
        mon.start()
        for _ in range(50):
            clock.t += 0.1
            mon.frame_done(work=3)
        line = mon.finish()
        self.assertTrue(line.endswith('3.00 faces/frame, 30.0 faces/s'))


class WiringTest(unittest.TestCase):
    """The hook must be where every path passes: update_progress, even with no bar."""

    def test_update_progress_ticks_the_monitor_before_anything_else(self):
        try:
            from roop.ProcessMgr import ProcessMgr
        except Exception as exc:
            self.skipTest('ProcessMgr unavailable: %s' % exc)
        seen = []
        holder = types.SimpleNamespace(_pipeline_monitor=types.SimpleNamespace(
            frame_done=lambda work=0: seen.append(work)))
        ProcessMgr.update_progress(holder, None)      # progress=None returns early
        self.assertEqual(seen, [0])                   # no thread-local tally: nothing to report

    def test_update_progress_reports_only_the_faces_this_thread_painted_since_last_time(self):
        try:
            from roop.ProcessMgr import ProcessMgr
        except Exception as exc:
            self.skipTest('ProcessMgr unavailable: %s' % exc)
        seen = []
        holder = types.SimpleNamespace(
            _tls=threading.local(),
            _pipeline_monitor=types.SimpleNamespace(frame_done=lambda work=0: seen.append(work)))
        holder._tls.faces_painted = 5                 # a warm-up frame painted 5 ...
        holder._tls.faces_reported = 5                # ... which _process_block wrote off
        holder._tls.faces_painted += 2                # the real frame painted 2
        ProcessMgr.update_progress(holder, None)
        ProcessMgr.update_progress(holder, None)      # a frame with no faces reports 0, not 2 again
        self.assertEqual(seen, [2, 0])

    def test_monitor_is_created_after_prepasses_and_finished_in_cleanup(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(here, 'roop', 'procmgr_batch.py'), encoding='utf-8').read()
        self.assertLess(src.index("phase4:before-main-processing"),
                        src.index("PipelineMonitor("))
        self.assertIn("_pipeline_monitor.finish()", src)
        self.assertIn("self._runtime_read_queue = None", src)


if __name__ == '__main__':
    unittest.main()
