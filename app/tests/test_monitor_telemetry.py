"""Phase 15 monitor: absence of a measurement must not read as a measurement.

Every defect covered here was measured on BOTH validation GPUs: the summary
reported `end_to_end_fps` 0.0, `cpu_utilization_pct` 0.0,
`gpu_utilization_pct` None and `bottleneck` "I/O-bound" on runs whose decode
stage cost 3.3 ms of a 244.8 ms frame.  None of those were true; each was a
default standing in for a value that was never collected.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.runtime_optimizer import RuntimeMonitor


# Shape taken from a real 600-frame RTX 4070 run (p15_adap_a).
UNINSTRUMENTED = {
    "stage_seconds": {"track_detect": 33.1, "mask": 30.2, "swap": 23.3,
                      "enhance": 12.1, "decode": 2.0, "encode": 0.9,
                      "frame_total": 146.9},
    "stage_latency_ms": {"decode": 3.29, "mask": 50.29, "swap": 38.75,
                         "track_detect": 55.19, "frame_total": 244.82},
    "queue_depths": {},
    "cpu_utilization_pct": None,
    "gpu_utilization_pct": None,
    "worker_utilization_pct": None,
    "vram_pressure_pct": 45.2,
    "ram_utilization_pct": 66.3,
}


class TestBottleneckHonesty(unittest.TestCase):
    def test_uninstrumented_run_is_not_called_io_bound(self):
        verdict = RuntimeMonitor.classify_bottleneck(UNINSTRUMENTED)
        self.assertNotIn("I/O-bound", verdict)

    def test_uninstrumented_run_names_the_dominant_stage(self):
        verdict = RuntimeMonitor.classify_bottleneck(UNINSTRUMENTED)
        self.assertIn("stage-bound", verdict)
        self.assertIn("track_detect", verdict)
        self.assertIn("no queue/utilization telemetry", verdict)

    def test_the_aggregate_stage_is_never_named_as_the_bottleneck(self):
        """frame_total is the sum of the others and would always win."""
        self.assertNotIn(
            "frame_total", RuntimeMonitor.classify_bottleneck(UNINSTRUMENTED))

    def test_no_stage_data_at_all_is_unknown(self):
        self.assertEqual(
            RuntimeMonitor.classify_bottleneck(
                {"queue_depths": {}, "gpu_utilization_pct": None,
                 "cpu_utilization_pct": None, "worker_utilization_pct": None}),
            "unknown (insufficient telemetry)")

    def test_real_io_bound_is_still_reported_when_measured(self):
        """A genuine idle-queue reading with telemetry keeps its verdict."""
        summary = dict(UNINSTRUMENTED)
        summary["queue_depths"] = {"input": 0.0, "output": 0.0}
        summary["gpu_utilization_pct"] = 60.0
        summary["cpu_utilization_pct"] = 30.0
        self.assertEqual(RuntimeMonitor.classify_bottleneck(summary), "I/O-bound")

    def test_gpu_bound_still_wins_when_measured(self):
        summary = dict(UNINSTRUMENTED)
        summary["gpu_utilization_pct"] = 92.0
        summary["queue_depths"] = {"input": 2.0, "output": 1.0}
        self.assertEqual(RuntimeMonitor.classify_bottleneck(summary), "GPU-bound")

    def test_vram_and_ram_pressure_take_priority(self):
        vram = dict(UNINSTRUMENTED, vram_pressure_pct=91.0)
        self.assertEqual(RuntimeMonitor.classify_bottleneck(vram), "VRAM-bound")
        ram = dict(UNINSTRUMENTED, ram_utilization_pct=95.0)
        self.assertEqual(RuntimeMonitor.classify_bottleneck(ram), "RAM-bound")


class TestProcessCpuSampling(unittest.TestCase):
    def test_process_handle_is_reused_so_cpu_percent_can_be_a_delta(self):
        """A fresh psutil.Process always returns 0.0 for cpu_percent(None)."""
        try:
            import psutil
        except ImportError:
            self.skipTest("psutil unavailable")
        monitor = RuntimeMonitor.__new__(RuntimeMonitor)
        monitor.hardware = None
        first = monitor._resource_snapshot()
        handle = getattr(monitor, "_psutil_process", None)
        self.assertIsNotNone(handle, "the Process handle was not retained")
        # Burn measurable CPU so the second delta is non-zero.
        total = 0
        for i in range(2_000_000):
            total += i
        second = monitor._resource_snapshot()
        self.assertIs(getattr(monitor, "_psutil_process"), handle)
        self.assertIsNotNone(second.get("cpu_utilization_pct"))
        self.assertGreater(second["cpu_utilization_pct"], 0.0,
                           "process CPU still reads 0.0 across two snapshots")
        self.assertIsNotNone(first)


if __name__ == "__main__":
    unittest.main()


class TestSelfDrivenSampling(unittest.TestCase):
    """The monitor must not depend on the pipeline calling record_frame.

    It did, and the render path never calls it, so every run produced a single
    sample taken at finish(). That one sample is why process CPU read 0.0 (a
    delta needs two reads) and why the adaptive controller -- which requires
    three consecutive windows -- never acted on either validation GPU.
    """

    def _monitor(self):
        from roop.runtime_optimizer import RuntimeMonitor
        monitor = RuntimeMonitor(enabled=True, diagnostics=False)
        monitor.sample_interval = 0.25
        return monitor

    def test_samples_accumulate_without_record_frame(self):
        import time as _time
        monitor = self._monitor()
        monitor.start()
        try:
            _time.sleep(1.1)
        finally:
            summary = monitor.finish()
        self.assertGreaterEqual(
            len(monitor.samples), 3,
            "fewer than three windows: the adaptive controller could never act")
        self.assertIsNotNone(summary)

    def test_sampler_thread_is_stopped_by_finish(self):
        monitor = self._monitor()
        monitor.start()
        thread = getattr(monitor, "_sampler_thread", None)
        self.assertIsNotNone(thread)
        monitor.finish()
        self.assertIsNone(getattr(monitor, "_sampler_thread", None))
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())

    def test_disabled_monitor_starts_no_thread(self):
        from roop.runtime_optimizer import RuntimeMonitor
        monitor = RuntimeMonitor(enabled=False)
        monitor.start()
        self.assertIsNone(getattr(monitor, "_sampler_thread", None))


class TestAdaptiveHookReachesTheProductionPath(unittest.TestCase):
    """The frame/queue hook must exist on the parallel stabilization writer.

    `_runtime_adaptive_boundary` is the only caller of `record_frame` and the
    only place `SafeAdaptiveController` is consulted. It was wired into the
    SEQUENTIAL encoder loop only, while production renders through the
    parallel stabilization writer -- so on the real path the monitor saw
    frames=0, no queue depths, no worker utilization, and the controller was
    never consulted at all.
    """

    def test_parallel_writer_calls_the_boundary_hook(self):
        import inspect
        from roop import ProcessMgr as pm
        source = inspect.getsource(pm.ProcessMgr._run_stab_parallel)
        self.assertIn("_runtime_adaptive_boundary()", source,
                      "the parallel stabilization writer does not drive the "
                      "runtime monitor / adaptive controller")

    def test_sequential_path_still_calls_it(self):
        import inspect
        from roop import ProcessMgr as pm
        source = inspect.getsource(pm.ProcessMgr)
        self.assertGreaterEqual(
            source.count("self._runtime_adaptive_boundary()"), 2,
            "both the sequential and parallel writers must drive the monitor")
