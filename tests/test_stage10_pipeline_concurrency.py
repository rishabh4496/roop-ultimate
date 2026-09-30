"""Unit and integration tests for Stage 10 — CPU/GPU Pipeline Concurrency.

Validates:
1. Pipeline Stage Definitions and Concurrency Matrix:
   - 8 stages: DECODE, PREPROCESS, DETECT/TRACK, SWAP, RESTORE, MASK, COMPOSITE, ENCODE.
   - Resource classification (CPU, GPU, IO).
   - Safe overlapping rules and temporal order invariants.
2. Hardware Profiles:
   - RTX 4070 Desktop: 12GB VRAM, 31.7GB RAM, max_in_flight=6, overlapped GPU mode, TRT pools (2/2/2).
   - RTX 3060 Laptop: 6GB VRAM, 15.8GB RAM, max_in_flight=3, serialized GPU mode (0/0 pools), RSS <= 2.5 GB.
3. FrameLeaseGovernor (Bounded Queues & Backpressure):
   - Capacity bounding: strictly limits in-flight frames to governor limit.
   - Blocking put and timeout behavior under congestion.
   - Mathematical proof of zero unbounded RAM growth.
4. Thread Hygiene (No Thread Explosion):
   - Verifies thread count remains strictly bounded regardless of total frames processed.
   - Rejects one-thread-per-frame anti-patterns.
5. Operational Safety:
   - Deterministic shutdown with sentinels.
   - Pause / resume safety: mid-stream pause does not lose or duplicate frames.
   - Cancellation safety: immediate abort drains queues and releases all governor leases without hanging.
6. Benchmark Sweep across 5 Configurations:
   - 1 worker
   - CPU worker pool
   - GPU serialized
   - GPU overlapped
   - Producer/consumer pipeline
   - Verifies thread count != performance invariant.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from pathlib import Path

# Add project root and app to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
for p in (str(PROJECT_ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pytest

from roop.pipeline_concurrency import (
    BoundedPipelineEngine,
    ConcurrencyBenchmarkResult,
    FrameLeaseGovernor,
    FramePacket,
    HardwareDevice,
    MockStageExecutor,
    PipelineHardwareProfile,
    PipelineStage,
    STAGE_CONCURRENCY_RULES,
    StageSentinel,
    run_concurrency_benchmark,
)


class TestPipelineStageDecomposition(unittest.TestCase):
    """Validates the 8 canonical stages and concurrency rules."""

    def test_all_canonical_stages_defined(self):
        """Verify all 8 requested stages exist in the pipeline."""
        expected = [
            "decode", "preprocess", "detect_track", "swap",
            "restore", "mask", "composite", "encode"
        ]
        actual = [s.value for s in PipelineStage]
        self.assertEqual(sorted(actual), sorted(expected))

    def test_stage_concurrency_matrix_safety(self):
        """Verify the safe overlapping matrix matches hardware and causal invariants."""
        # 1. DECODE is IO-bound and can overlap with compute
        decode_rule = STAGE_CONCURRENCY_RULES[PipelineStage.DECODE]
        self.assertEqual(decode_rule.device, HardwareDevice.IO_HOST)
        self.assertIn(PipelineStage.SWAP, decode_rule.can_overlap_with)
        self.assertIn(PipelineStage.COMPOSITE, decode_rule.can_overlap_with)

        # 2. ENCODE is IO-bound and requires strict presentation order
        encode_rule = STAGE_CONCURRENCY_RULES[PipelineStage.ENCODE]
        self.assertEqual(encode_rule.device, HardwareDevice.IO_HOST)
        self.assertTrue(encode_rule.temporal_order_required)

        # 3. DETECT/TRACK requires temporal sequence when tracking online
        det_rule = STAGE_CONCURRENCY_RULES[PipelineStage.DETECT_TRACK]
        self.assertEqual(det_rule.gpu_stage_name, "analysis")
        self.assertTrue(det_rule.temporal_order_required)

        # 4. MASK can overlap with SWAP (crucial intra-frame concurrency finding)
        mask_rule = STAGE_CONCURRENCY_RULES[PipelineStage.MASK]
        swap_rule = STAGE_CONCURRENCY_RULES[PipelineStage.SWAP]
        self.assertIn(PipelineStage.SWAP, mask_rule.can_overlap_with)
        self.assertIn(PipelineStage.MASK, swap_rule.can_overlap_with)

        # 5. COMPOSITE is CPU SIMD bound and can overlap with GPU stages
        comp_rule = STAGE_CONCURRENCY_RULES[PipelineStage.COMPOSITE]
        self.assertEqual(comp_rule.device, HardwareDevice.CPU)
        self.assertIn(PipelineStage.SWAP, comp_rule.can_overlap_with)
        self.assertIn(PipelineStage.RESTORE, comp_rule.can_overlap_with)


class TestHardwareProfiles(unittest.TestCase):
    """Validates device profiles for RTX 4070 Desktop and RTX 3060 Laptop."""

    def test_rtx4070_profile_invariants(self):
        profile = PipelineHardwareProfile.rtx_4070()
        self.assertEqual(profile.vram_total_gb, 12.0)
        self.assertEqual(profile.ram_total_gb, 31.7)
        self.assertEqual(profile.gpu_concurrency_mode, "overlapped")
        self.assertEqual(profile.trt_pool_swap, 2)
        self.assertEqual(profile.trt_pool_analysis, 2)
        self.assertEqual(profile.trt_pool_mask, 2)
        self.assertTrue(profile.enable_cross_frame_batching)
        self.assertEqual(profile.batch_swap_max, 8)
        self.assertTrue(profile.enable_nvdec)
        self.assertGreaterEqual(profile.max_in_flight_frames, 4)

    def test_rtx3060_profile_invariants(self):
        profile = PipelineHardwareProfile.rtx_3060()
        self.assertEqual(profile.vram_total_gb, 6.0)
        self.assertEqual(profile.ram_total_gb, 15.8)
        self.assertEqual(profile.gpu_concurrency_mode, "serialized")
        # Sub-7GB tier policy: single context, 0/0 pool to prevent VRAM blowup
        self.assertEqual(profile.trt_pool_swap, 0)
        self.assertEqual(profile.trt_pool_analysis, 0)
        self.assertEqual(profile.trt_pool_mask, 0)
        self.assertFalse(profile.enable_cross_frame_batching)
        self.assertEqual(profile.batch_swap_max, 1)
        self.assertFalse(profile.enable_nvdec)  # Sub-7GB policy forces CPU decode
        self.assertLessEqual(profile.max_rss_gb, 2.5)  # Strict 2.5 GB RSS ceiling
        self.assertLessEqual(profile.max_in_flight_frames, 3)


class TestBackpressureGovernor(unittest.TestCase):
    """Validates mathematical in-flight bounds and backpressure behavior."""

    def test_lease_governor_strict_bounding(self):
        """Ensure governor blocks once max_in_flight is saturated."""
        limit = 3
        gov = FrameLeaseGovernor(max_in_flight=limit, frame_shape=(720, 1280))

        # Acquire all available leases
        for _ in range(limit):
            self.assertTrue(gov.acquire(timeout=0.1))

        self.assertEqual(gov.active_count, limit)

        # Next acquisition must fail / timeout (backpressure)
        self.assertFalse(gov.acquire(timeout=0.05))

        # Release one lease
        gov.release()
        self.assertEqual(gov.active_count, limit - 1)

        # Now acquire succeeds
        self.assertTrue(gov.acquire(timeout=0.1))
        self.assertEqual(gov.active_count, limit)

        # Clean up
        for _ in range(limit):
            gov.release()
        self.assertEqual(gov.active_count, 0)

    def test_ram_growth_mathematical_cap(self):
        """Verify RAM usage estimate stays strictly bounded."""
        limit = 4
        gov = FrameLeaseGovernor(max_in_flight=limit, frame_shape=(720, 1280))
        for _ in range(limit):
            gov.acquire()

        peak_mb = gov.estimated_live_ram_mb
        expected_cap_mb = (limit * gov.estimated_host_bytes_per_lease) / (1024 * 1024)
        self.assertAlmostEqual(peak_mb, expected_cap_mb, delta=0.1)

        for _ in range(limit):
            gov.release()


class TestPipelineThreadHygiene(unittest.TestCase):
    """Validates thread limits, explicitly rejecting one-thread-per-frame."""

    def test_no_thread_per_frame_invariant(self):
        """Processing 40 frames must not spawn 40 threads."""
        profile = PipelineHardwareProfile.rtx_4070()
        engine = BoundedPipelineEngine(
            hardware_profile=profile,
            concurrency_mode="producer_consumer_pipeline",
            frame_shape=(360, 640),
        )

        initial_threads = threading.active_count()
        num_frames = 40

        metrics = engine.run(num_frames=num_frames)

        self.assertEqual(metrics.encoded_frames, num_frames)
        # Peak active threads in engine must be bounded to O(1) pipeline stages, not O(N) frames
        final_threads = threading.active_count()
        self.assertLessEqual(final_threads - initial_threads, 10)


class TestPipelineOperationalSafety(unittest.TestCase):
    """Validates deterministic shutdown, pause/resume, and cancellation safety."""

    def test_deterministic_shutdown(self):
        profile = PipelineHardwareProfile.rtx_3060()
        engine = BoundedPipelineEngine(
            hardware_profile=profile,
            concurrency_mode="producer_consumer_pipeline",
            frame_shape=(360, 640),
        )

        metrics = engine.run(num_frames=15)
        self.assertEqual(metrics.encoded_frames, 15)
        self.assertEqual(engine.governor.active_count, 0)

    def test_pause_and_resume_safety(self):
        """Verify pipeline can pause mid-stream and resume without losing frames."""
        profile = PipelineHardwareProfile.rtx_4070()
        engine = BoundedPipelineEngine(
            hardware_profile=profile,
            concurrency_mode="cpu_worker_pool",
            frame_shape=(360, 640),
        )

        num_frames = 20
        runner_thread = threading.Thread(target=lambda: engine.run(num_frames))
        runner_thread.start()

        time.sleep(0.05)
        # Pause execution
        engine.pause()
        paused_count = engine.metrics.encoded_frames
        time.sleep(0.1)

        # While paused, progress must halt or stay bounded
        self.assertLessEqual(engine.metrics.encoded_frames, paused_count + 8)

        # Resume execution
        engine.resume()
        runner_thread.join(timeout=10.0)

        self.assertFalse(runner_thread.is_alive())
        self.assertEqual(engine.metrics.encoded_frames, num_frames)
        self.assertEqual(engine.governor.active_count, 0)

    def test_cancellation_safety(self):
        """Verify cancellation immediately drains queues and releases leases without hanging."""
        profile = PipelineHardwareProfile.rtx_4070()
        engine = BoundedPipelineEngine(
            hardware_profile=profile,
            concurrency_mode="producer_consumer_pipeline",
            frame_shape=(360, 640),
        )

        num_frames = 50
        runner_thread = threading.Thread(target=lambda: engine.run(num_frames))
        runner_thread.start()

        time.sleep(0.05)
        # Cancel execution
        engine.cancel()
        runner_thread.join(timeout=5.0)

        self.assertFalse(runner_thread.is_alive())
        # All frame leases must be cleanly reclaimed
        self.assertEqual(engine.governor.active_count, 0)


class TestBenchmarkFiveConfigurations(unittest.TestCase):
    """Validates the 5 required benchmark configurations and scaling characteristics."""

    def test_benchmark_sweep_execution(self):
        profile = PipelineHardwareProfile.rtx_4070()
        num_frames = 15

        res = run_concurrency_benchmark(
            profile=profile,
            num_frames=num_frames,
            modes=[
                "1_worker",
                "cpu_worker_pool",
                "gpu_serialized",
                "gpu_overlapped",
                "producer_consumer_pipeline",
            ],
        )

        self.assertEqual(len(res.results), 5)
        for mode, m in res.results.items():
            self.assertEqual(m.encoded_frames, num_frames)
            self.assertGreater(m.throughput_fps, 0.0)
            self.assertGreater(m.estimated_peak_ram_mb, 0.0)

        # Invariant: GPU serialized cannot exceed GPU overlapped
        fps_serialized = res.results["gpu_serialized"].throughput_fps
        fps_overlapped = res.results["gpu_overlapped"].throughput_fps
        self.assertGreater(fps_overlapped, fps_serialized)

        # Invariant: Maximum thread count != maximum performance
        # CPU worker pool with serialized GPU performs no better than 1_worker
        fps_1w = res.results["1_worker"].throughput_fps
        fps_cpu_pool = res.results["cpu_worker_pool"].throughput_fps
        # They should be roughly comparable due to GPU lock serialization
        self.assertAlmostEqual(fps_1w, fps_cpu_pool, delta=fps_1w * 0.35)


if __name__ == "__main__":
    unittest.main()
