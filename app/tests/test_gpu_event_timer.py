"""Non-blocking CUDA event timing (roop/gpu_event_timer.py).

`torch.cuda.Event.elapsed_time` raises "device not ready" while the end event is still queued, and the old
`StageProfiler.end()` called it immediately and swallowed the error, so its GPU column was empty on every
asynchronous stage. The fix records on the hot path and reads later, only for pairs `query()` says are finished, and
NEVER synchronizes. The fake-CUDA tests prove the logic (including that `synchronize` is never reached); the
`gpu` tests prove it on a real device, where the old behaviour is reproducible.
"""
import os
import sys
import time
import unittest
from unittest import mock

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import gpu_event_timer as get  # noqa: E402
from roop import stage_telemetry as st  # noqa: E402


class _FakeEvent:
    """Mirrors CUDA: `elapsed_time` raises until BOTH events have completed; `query` never blocks."""
    clock = 0.0

    def __init__(self, enable_timing=False):
        self.recorded_at = None
        self.done = False

    def record(self, stream=None):
        self.recorded_at = _FakeEvent.clock
        self.done = False

    def query(self):
        return self.done

    def elapsed_time(self, end):
        if not (self.done and end.done):
            raise RuntimeError("CUDA error: device not ready")
        return (end.recorded_at - self.recorded_at) * 1000.0


class _FakeCuda:
    Event = _FakeEvent

    @staticmethod
    def synchronize(*_a, **_k):
        raise AssertionError("the timer must never synchronize the device")


class _FakeTorch:
    cuda = _FakeCuda


def _timer(**kw):
    return get.GpuEventTimer(torch_module=_FakeTorch, min_interval_s=0.25, **kw)


def _span(timer, name, t0, t1):
    _FakeEvent.clock = t0
    tok = timer.begin(name)
    _FakeEvent.clock = t1
    timer.end(tok)
    return tok


def _finish(*tokens):
    for tok in tokens:
        tok.start.done = tok.end.done = True


class FakeCudaLogic(unittest.TestCase):
    def test_a_pair_the_gpu_has_not_finished_is_left_alone_not_read_and_not_waited_on(self):
        t = _timer()
        tok = _span(t, "warp", 0.0, 0.004)
        self.assertEqual(t.resolve(now=1.0), 0)                 # would have raised "device not ready" if read
        self.assertEqual(t.snapshot()["pending"], 1)
        self.assertEqual(t.errors, 0)
        _finish(tok)
        self.assertEqual(t.resolve(now=2.0), 1)
        agg = t.snapshot()["names"]["warp"]
        self.assertEqual((agg["samples"], agg["avg_ms"]), (1, 4.0))

    def test_lag_holds_back_the_newest_pairs_until_older_ones_exist(self):
        t = _timer(lag=2)
        a, b, c = (_span(t, "trt", i, i + 0.002) for i in range(3))
        _finish(a, b, c)                                         # all three are complete on the GPU
        self.assertEqual(t.resolve(now=1.0), 1)                  # only batch t-2 (seq 1) is old enough
        self.assertEqual(t.snapshot()["names"]["trt"]["samples"], 1)
        self.assertEqual(t.resolve(now=2.0, force=True), 2)      # end of run: nothing newer is coming
        self.assertEqual(t.snapshot()["names"]["trt"]["samples"], 3)

    def test_a_retire_fence_gates_the_read(self):
        class Fence:
            done = False
            def query(self):
                return self.done
        t = _timer()
        fence = Fence()
        tok = _span(t, "triton", 0.0, 0.001)
        tok.fence = fence                                        # end() stores it; set directly for the test
        _finish(tok)
        self.assertEqual(t.resolve(now=1.0), 0)                  # GPU work is done, the encode is not
        fence.done = True
        self.assertEqual(t.resolve(now=2.0), 1)

    def test_resolve_is_throttled_to_4hz_unless_forced(self):
        t = _timer()
        t1 = _span(t, "warp", 0.0, 0.001)
        _finish(t1)
        self.assertEqual(t.resolve(now=10.0), 1)
        t2 = _span(t, "warp", 1.0, 1.001)
        _finish(t2)
        self.assertEqual(t.resolve(now=10.1), 0)                 # 100 ms later: inside the 250 ms window
        self.assertEqual(t.resolve(now=10.3), 1)
        t3 = _span(t, "warp", 2.0, 2.001)
        _finish(t3)
        self.assertEqual(t.resolve(now=10.31, force=True), 1)

    def test_an_exhausted_pool_drops_the_new_sample_and_counts_it(self):
        t = _timer(pairs=2)
        a, b = _span(t, "x", 0, 1), _span(t, "x", 1, 2)
        self.assertIsNone(t.begin("x"))                          # never allocates, never blocks
        self.assertEqual(t.snapshot()["dropped"], 1)
        _finish(a, b)
        t.resolve(now=1.0)
        self.assertIsNotNone(t.begin("x"))                       # pairs return to the pool once read

    def test_nothing_in_the_timer_ever_reaches_synchronize(self):
        t = _timer(lag=1)
        toks = [_span(t, "n", i, i + 0.001) for i in range(4)]
        _finish(*toks)
        t.resolve(now=1.0)
        t.resolve(now=2.0, force=True)
        t.snapshot()                                             # _FakeCuda.synchronize raises if it is called

    def test_unavailable_without_cuda_is_inert_not_an_error(self):
        t = get.GpuEventTimer(torch_module=None)
        with mock.patch.object(get, "_cuda_torch", return_value=None):
            t = get.GpuEventTimer()
        self.assertFalse(t.available)
        self.assertIsNone(t.begin("x"))
        t.end(None)
        self.assertEqual(t.resolve(force=True), 0)
        self.assertFalse(t.snapshot()["available"])


class TelemetryHook(unittest.TestCase):
    def setUp(self):
        st.reset()
        self.addCleanup(st.reset)
        self.addCleanup(lambda: get.register(None))
        st._vram.stop()
        st._vram._probe = lambda: None
        mock.patch.object(st._vram, "start", lambda: st._vram.reset()).start()
        self.addCleanup(mock.patch.stopall)

    def _stage3(self):
        st.begin_run(10)
        st.skip(st.STAGE_1_SPARSE_PRESCAN)
        st.skip(st.STAGE_2_ACTIVE_PREPASS)
        st.enter(st.STAGE_3_MAIN_SWAP, 10)

    def test_the_telemetry_tick_reads_the_registered_timer_and_nothing_else_does(self):
        t = _timer()
        tok = _span(t, "warp", 0.0, 0.0035)
        _finish(tok)
        get.register(t)
        self._stage3()
        s3 = st.snapshot()["stages"][st.STAGE_3_MAIN_SWAP]
        self.assertEqual(s3["gpu_events"]["names"]["warp"]["avg_ms"], 3.5)
        self.assertEqual(s3["gpu_events"]["timing"], "cuda-event")

    def test_no_timer_means_no_block_in_the_frame_and_none_in_the_report(self):
        self._stage3()
        self.assertNotIn("gpu_events", st.snapshot()["stages"][st.STAGE_3_MAIN_SWAP])
        self.assertIsNone(st.build_summary()["gpu_event_timing"])

    def test_the_report_forces_a_final_read(self):
        t = _timer(lag=2)
        toks = [_span(t, "trt", i, i + 0.002) for i in range(3)]
        _finish(*toks)
        get.register(t)
        self._stage3()
        self.assertEqual(st.build_summary()["gpu_event_timing"]["names"]["trt"]["samples"], 3)

    def test_ops_say_what_their_timing_is(self):
        self._stage3()
        st.record_op("swap", 0.010)
        ops = st.snapshot()["stages"][st.STAGE_3_MAIN_SWAP]["ops"]
        self.assertEqual(ops["model"]["timing"], st.OPS_TIMING)


def _real_cuda():
    try:
        import torch
        return torch if torch.cuda.is_available() else None
    except Exception:
        return None


@pytest.mark.gpu
class RealDevice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = _real_cuda()
        if cls.torch is None:
            raise unittest.SkipTest("UNVERIFIED - no CUDA device here")

    def _busy(self):
        x = self.torch.randn(4096, 4096, device="cuda")
        self.torch.cuda.synchronize()
        return x

    def test_the_old_call_raises_and_the_timer_does_not(self):
        torch, x = self.torch, self._busy()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timer = get.GpuEventTimer(pairs=4, torch_module=torch)
        tok = timer.begin("matmul")
        s.record()
        for _ in range(40):
            x @ x
        e.record()
        timer.end(tok)
        self.assertFalse(e.query())                              # the GPU is still busy
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            s.elapsed_time(e)                                    # what StageProfiler.end() did, then swallowed
        with mock.patch.object(torch.cuda, "synchronize", side_effect=AssertionError("synchronized")):
            self.assertEqual(timer.resolve(force=True), 0)       # in flight: left pending, no raise, no wait
            deadline = time.perf_counter() + 30
            while timer.resolve(force=True) == 0:
                self.assertLess(time.perf_counter(), deadline, "GPU never finished")
        agg = timer.snapshot()["names"]["matmul"]
        self.assertEqual(agg["samples"], 1)
        self.assertGreater(agg["avg_ms"], 0.0)
        self.assertEqual(timer.errors, 0)

    def test_stage_profiler_event_only_mode_now_reports_gpu_time(self):
        from roop.stage_profiler import StageProfiler
        torch, x = self.torch, self._busy()
        prof = StageProfiler(gpu_sync=False)
        token = prof.begin("swap")
        for _ in range(40):
            x @ x
        prof.end(token)
        with mock.patch.object(torch.cuda, "synchronize", side_effect=AssertionError("synchronized")):
            deadline = time.perf_counter() + 30
            while True:
                report = prof.report()
                if report["stages"]["swap"]["gpu_event_samples"] or time.perf_counter() > deadline:
                    break
        swap = report["canonical_stages"]["swap"]
        self.assertEqual(report["mode"], "event-only")
        self.assertEqual(swap["gpu_event_samples"], 1)           # was 0: the read raised and was swallowed
        self.assertGreater(swap["gpu_event_ms_per_call"], 0.0)
        self.assertEqual(report["gpu_event_timing"]["errors"], 0)

    def test_a_long_profiled_run_recycles_its_pairs_instead_of_dropping_samples(self):
        from roop.stage_profiler import StageProfiler
        torch, x = self.torch, self._busy()
        prof = StageProfiler(gpu_sync=False)
        calls = 3000                                             # more than a pool, at ~1500 stage calls/s
        for i in range(calls):
            token = prof.begin("blend")
            x[:8, :8] @ x[:8, :8]
            prof.end(token)
            if i % 3 == 0:
                time.sleep(0.001)
        with mock.patch.object(torch.cuda, "synchronize", side_effect=AssertionError("synchronized")):
            deadline = time.perf_counter() + 30
            while True:
                report = prof.report()
                if report["stages"]["blend"]["gpu_event_samples"] + report["gpu_event_timing"]["dropped"] >= calls \
                        or time.perf_counter() > deadline:
                    break
        self.assertEqual(report["gpu_event_timing"]["dropped"], 0, report["gpu_event_timing"])
        self.assertEqual(report["stages"]["blend"]["gpu_event_samples"], calls)


if __name__ == "__main__":
    unittest.main()
