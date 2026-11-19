"""Unit and integration tests for benchmark stage timing and GPU telemetry harness."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import unittest
from unittest.mock import patch, MagicMock

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from bench_telemetry import (
    STAGE_GROUPS,
    STAGE_TO_GROUP,
    compute_percentiles,
    GpuTelemetrySampler,
    StageTelemetry,
    BenchmarkTelemetryContext,
    hook_telemetry,
    unhook_telemetry,
    format_timing_table,
    compare_telemetry,
    format_comparison_table,
)
from roop import procmgr_runtime


class TestBenchTelemetry(unittest.TestCase):
    def test_stage_groups_contract(self):
        """Verify all 8 canonical stage groups and required stages are present."""
        expected_groups = [
            "video_decode",
            "detection_landmarks",
            "pose_source_selection",
            "swapper_inference",
            "mask_generation",
            "color_transfer_blending",
            "face_restoration",
            "video_encode_mux",
        ]
        for grp in expected_groups:
            self.assertIn(grp, STAGE_GROUPS)
            self.assertIn("name", STAGE_GROUPS[grp])
            self.assertIn("stages", STAGE_GROUPS[grp])

        required_stages = [
            "decode", "track_decode",
            "detect", "detection", "track_detect", "alignment",
            "faceset_lookup",
            "swap",
            "mask", "mask_refine", "mask_stabilize", "occlusion_analysis", "interaction_occlusion",
            "lighting", "blend", "realism", "identity_detail", "detail_transfer",
            "enhance", "enhance_recombine", "stabilize",
            "encode", "encode_finalize",
        ]
        for stage in required_stages:
            self.assertIn(stage, STAGE_TO_GROUP, f"Stage {stage} not indexed in STAGE_TO_GROUP")

    def test_compute_percentiles_empty_and_populated(self):
        """Test distribution percentiles computation on empty and populated lists."""
        empty_res = compute_percentiles([])
        self.assertEqual(empty_res["count"], 0)
        self.assertEqual(empty_res["p50"], 0.0)
        self.assertEqual(empty_res["p95"], 0.0)

        # 1 to 100 values
        values = list(range(1, 101))
        res = compute_percentiles(values)
        self.assertEqual(res["count"], 100)
        self.assertAlmostEqual(res["p50"], 50.5, delta=1.0)
        self.assertAlmostEqual(res["p95"], 95.05, delta=1.5)
        self.assertEqual(res["min"], 1.0)
        self.assertEqual(res["max"], 100.0)

    def test_gpu_telemetry_sampler_lifecycle(self):
        """Test GPU sampler starts, records samples, and stops without leaks."""
        sampler = GpuTelemetrySampler(device_id=0, interval_s=0.02)
        sampler.start()
        time.sleep(0.08)
        sampler.stop()

        stats = sampler.get_stats()
        self.assertIn("source", stats)
        self.assertIn("peak_vram_mb", stats)
        self.assertIn("peak_gpu_util_pct", stats)
        self.assertIn("mean_gpu_util_pct", stats)
        self.assertIn("total_vram_mb", stats)
        self.assertGreaterEqual(stats["samples_count"], 1)

    def test_gpu_telemetry_sampler_fallback_on_nvml_error(self):
        """Verify sampler falls back gracefully to torch or 0.0 when NVML fails."""
        with patch("pynvml.nvmlInit", side_effect=Exception("No NVML")):
            sampler = GpuTelemetrySampler(device_id=0, interval_s=0.02)
            self.assertIn(sampler.source, ("torch", "fallback"))
            vram, util = sampler.sample_once()
            self.assertGreaterEqual(vram, 0.0)
            self.assertGreaterEqual(util, 0.0)
            stats = sampler.get_stats()
            self.assertIsInstance(stats["peak_vram_mb"], (int, float))
            self.assertIsInstance(stats["mean_gpu_util_pct"], (int, float))

    def test_stage_telemetry_hook_into_procmgr_runtime(self):
        """Verify StageTelemetry hooks cleanly into procmgr_runtime.set_stage_sink."""
        orig_sink = procmgr_runtime._STAGE_SINK
        try:
            telemetry = StageTelemetry()
            telemetry.start()

            # Confirm sink is attached
            self.assertEqual(procmgr_runtime._STAGE_SINK, telemetry.record_stage)

            # Trigger _prof
            with procmgr_runtime._prof("detect"):
                time.sleep(0.005)

            with procmgr_runtime._prof("swap"):
                time.sleep(0.005)

            telemetry.stop()

            # Confirm sink is restored
            self.assertEqual(procmgr_runtime._STAGE_SINK, orig_sink)

            summary = telemetry.get_summary()
            self.assertIn("detect", summary["stages"])
            self.assertIn("swap", summary["stages"])
            self.assertEqual(summary["stages"]["detect"]["calls"], 1)
            self.assertEqual(summary["stages"]["swap"]["calls"], 1)
            self.assertGreater(summary["stages"]["detect"]["total_ms"], 1.0)
        finally:
            procmgr_runtime.set_stage_sink(orig_sink)

    def test_stage_telemetry_with_frame_scope(self):
        """Test per-frame latency tracking via frame_scope context manager."""
        telemetry = StageTelemetry()
        with telemetry:
            for frame_idx in range(5):
                with telemetry.frame_scope(frame_idx):
                    telemetry.record_stage("decode", 0.002)
                    telemetry.record_stage("detect", 0.004)
                    telemetry.record_stage("swap", 0.006)
                    telemetry.record_stage("mask", 0.001)
                    telemetry.record_stage("encode", 0.002)

        summary = telemetry.get_summary()
        self.assertEqual(summary["total_frames"], 5)
        self.assertGreater(summary["total_fps"], 0.0)

        # Check stage groups
        groups = summary["stage_groups"]
        self.assertIn("video_decode", groups)
        self.assertIn("detection_landmarks", groups)
        self.assertIn("swapper_inference", groups)
        self.assertIn("mask_generation", groups)
        self.assertIn("video_encode_mux", groups)

        self.assertEqual(groups["swapper_inference"]["calls"], 5)
        self.assertAlmostEqual(groups["swapper_inference"]["p50_ms"], 6.0, delta=1.0)
        self.assertAlmostEqual(groups["detection_landmarks"]["p50_ms"], 4.0, delta=1.0)

    def test_stage_telemetry_multithreaded_safety(self):
        """Verify concurrent stage recording across multiple worker threads."""
        telemetry = StageTelemetry()
        telemetry.start()

        def worker_task(worker_id: int):
            for i in range(10):
                telemetry.record_stage("detect", 0.001)
                telemetry.record_stage("swap", 0.002)
                telemetry.record_stage("frame_total", 0.005)

        threads = [threading.Thread(target=worker_task, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        telemetry.stop()
        summary = telemetry.get_summary()

        self.assertEqual(summary["stages"]["detect"]["calls"], 40)
        self.assertEqual(summary["stages"]["swap"]["calls"], 40)
        self.assertEqual(summary["stages"]["frame_total"]["calls"], 40)
        self.assertEqual(summary["total_frames"], 40)

    def test_table_formatting_and_comparison(self):
        """Test ASCII formatting for stage breakdown and comparison tables."""
        telemetry = StageTelemetry()
        with telemetry:
            telemetry.record_stage("decode", 0.003)
            telemetry.record_stage("detect", 0.005)
            telemetry.record_stage("swap", 0.010)
            telemetry.record_stage("encode", 0.002)
            telemetry.record_stage("frame_total", 0.020)

        summary = telemetry.get_summary(total_frames=1)
        table_str = format_timing_table(summary)
        self.assertIn("STAGE-BY-STAGE TIMING BREAKDOWN", table_str)
        self.assertIn("Swapper Inference", table_str)
        self.assertIn("Video Decode", table_str)

        # Baseline comparison test
        baseline = dict(summary)
        # Modify current for comparison
        current = json.loads(json.dumps(baseline))
        current["total_fps"] = baseline["total_fps"] * 1.2
        current["peak_vram_mb"] = baseline["peak_vram_mb"] + 150.0

        comp_dict = compare_telemetry(current, baseline)
        self.assertIn("fps", comp_dict)
        self.assertAlmostEqual(comp_dict["fps"]["pct_change"], 20.0, delta=1.0)

        comp_table_str = format_comparison_table(current, baseline)
        self.assertIn("BENCHMARK REGRESSION / SPEEDUP COMPARISON", comp_table_str)
        self.assertIn("Throughput (FPS)", comp_table_str)

    def test_no_forbidden_strings(self):
        """Invariant: ensure forbidden strings are never introduced into bench_telemetry."""
        forbidden = [
            "-".join(["roop", "unleashed"]),
            "_".join(["roop", "unleashed"]),
            "C0" + "untFloyd",
            "s0" + "md3v",
        ]
        target_file = os.path.join(HERE, "bench_telemetry.py")
        self.assertTrue(os.path.exists(target_file))
        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read().lower()
        for needle in forbidden:
            self.assertNotIn(
                needle.lower(),
                content,
                f"Forbidden string '{needle}' found in {os.path.basename(target_file)}",
            )


if __name__ == "__main__":
    unittest.main()
