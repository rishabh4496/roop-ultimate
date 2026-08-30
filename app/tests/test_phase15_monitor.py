import unittest

from roop.runtime_optimizer import (
    HardwareProfile,
    RuntimeMonitor,
    RuntimeTuning,
    SafeAdaptiveController,
)


def _hardware(vram=12.0, nvenc=True):
    return HardwareProfile(
        gpu_name="test GPU", gpu_vendor="nvidia", architecture="Ada Lovelace",
        compute_capability="8.9", vram_total_gb=vram,
        vram_available_gb=max(1.0, vram - 2.0), cuda_available=True,
        nvenc_available=nvenc, cpu_physical_cores=8, cpu_logical_cores=16,
        ram_total_gb=32.0, ram_available_gb=16.0)


class Phase15MonitorTests(unittest.TestCase):
    def test_disabled_monitor_is_a_noop(self):
        monitor = RuntimeMonitor(enabled=False)
        monitor.start()
        monitor.record_stage("encode", 1.0)
        self.assertIsNone(monitor.record_frame())
        self.assertEqual(monitor.summary()["samples"], [])

    def test_rolling_summary_exposes_stage_and_resource_metrics(self):
        monitor = RuntimeMonitor(enabled=True, diagnostics=False)
        monitor._resource_snapshot = lambda: {
            "cpu_utilization_pct": 40.0, "gpu_utilization_pct": 82.0,
            "vram_pressure_pct": 50.0, "ram_utilization_pct": 45.0}
        monitor.start()
        monitor.record_stage("decode", 0.02, calls=2)
        monitor.record_stage("encode", 0.001, calls=2)
        monitor.record_frame({"input": 1, "output": 2}, 50.0)
        result = monitor.finish({"input": 0, "output": 1}, 25.0)
        self.assertEqual(result["decode_fps"], result["stage_fps"]["decode"])
        self.assertEqual(result["encode_fps"], result["stage_fps"]["encode"])
        self.assertIn("stage_latency_ms", result)
        self.assertEqual(result["bottleneck"], "decode-bound")
        self.assertEqual(result["queue_depths"]["input"], 0.0)

    def test_summary_exposes_memory_pressure_dimensions(self):
        monitor = RuntimeMonitor(enabled=True, diagnostics=False)
        monitor._resource_snapshot = lambda: {
            "process_rss_gb": 1.25,
            "ram_total_gb": 15.8,
            "ram_available_gb": 5.6,
            "ram_committed_gb": 10.4,
            "ram_commit_limit_gb": 23.8,
            "swap_used_gb": 0.1,
            "swap_utilization_pct": 0.4,
        }
        monitor.start()
        result = monitor.finish()
        self.assertEqual(result["process_rss_gb"], 1.25)
        self.assertEqual(result["ram_available_gb"], 5.6)
        self.assertEqual(result["ram_committed_gb"], 10.4)
        self.assertEqual(result["swap_utilization_pct"], 0.4)

    def test_adaptation_requires_hysteresis_and_changes_future_work_only(self):
        tuning = RuntimeTuning(batch_size=3, tile_batch_size=2,
                               in_flight_frames=2, queue_depth=2)
        controller = SafeAdaptiveController(
            _hardware(), tuning, enabled=True)
        summary = {"bottleneck": "VRAM-bound", "vram_pressure_pct": 95.0,
                   "ram_utilization_pct": 40.0}
        self.assertIsNone(controller.update(summary, safe_boundary=True))
        self.assertIsNone(controller.update(summary, safe_boundary=True))
        action = controller.update(summary, safe_boundary=True)
        self.assertEqual(action["scope"], "next_work")
        self.assertEqual(controller.tuning.batch_size, 2)
        self.assertEqual(controller.tuning.in_flight_frames, 1)
        self.assertEqual(action["safe_boundary"], True)

    def test_explicit_codec_is_never_overridden(self):
        controller = SafeAdaptiveController(
            _hardware(), RuntimeTuning(encoder="libx264"),
            settings={"output_video_codec": "libx264"}, enabled=True)
        summary = {"bottleneck": "encode-bound", "vram_pressure_pct": 20.0,
                   "ram_utilization_pct": 20.0}
        for _ in range(4):
            self.assertIsNone(controller.update(summary, safe_boundary=True))
        self.assertEqual(controller.tuning.encoder, "libx264")


if __name__ == "__main__":
    unittest.main()
