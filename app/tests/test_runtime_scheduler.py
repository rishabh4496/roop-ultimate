"""Contract tests for Gate E's unified scheduler.

These tests exercise scheduling and admission with synthetic callbacks. They do
not present as RTX measurements; physical GPU validation remains a separate
benchmark for each target.
"""

import os
import sys
import time
import unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.runtime_optimizer import HardwareProfile, RuntimeTuning, WorkloadProfile
from roop.runtime_scheduler import SchedulerBudget, UnifiedRuntimeScheduler


def _hardware(vram, ram_available=16.0):
    return HardwareProfile(
        gpu_name="synthetic NVIDIA GPU", gpu_vendor="nvidia",
        architecture="test", compute_capability="8.6",
        vram_total_gb=vram, vram_available_gb=vram * 0.7,
        cuda_available=True, tensorrt_available=True,
        cuda_version="test", tensorrt_version="test",
        onnxruntime_version="test", nvdec_available=True,
        nvenc_available=True, cpu_physical_cores=8,
        cpu_logical_cores=16, ram_total_gb=16.0,
        ram_available_gb=ram_available)


def _workload(width=1280, height=720, stabilization=False):
    return WorkloadProfile(input_width=width, input_height=height,
                           output_width=width, output_height=height,
                           faces_per_frame=2, enhancement_enabled=True,
                           stabilization_enabled=stabilization)


class RuntimeSchedulerTests(unittest.TestCase):
    def test_budget_is_derived_from_workload_and_detected_memory(self):
        small = SchedulerBudget.from_profile(
            _hardware(6.0, ram_available=4.0), _workload(3840, 2160, True),
            RuntimeTuning(worker_count=8, queue_depth=4, in_flight_frames=4,
                          ram_buffer_mb=1536))
        desktop = SchedulerBudget.from_profile(
            _hardware(12.0, ram_available=20.0), _workload(1280, 720),
            RuntimeTuning(worker_count=8, queue_depth=4, in_flight_frames=4,
                          ram_buffer_mb=4096))
        self.assertLessEqual(small.queue_capacity, 4)
        self.assertLessEqual(small.in_flight_limit, 4)
        self.assertNotEqual(small.frame_bytes, desktop.frame_bytes)
        self.assertGreater(small.stabilization_chunk_frames, 0)
        self.assertGreaterEqual(small.estimated_host_bytes, small.frame_bytes * 2)
        self.assertFalse(small.pinned_host_memory)

    def test_pipeline_keeps_encode_order_with_bounded_queues(self):
        scheduler = UnifiedRuntimeScheduler(
            _hardware(12.0), _workload(),
            RuntimeTuning(worker_count=4, queue_depth=2, in_flight_frames=4,
                          ram_buffer_mb=1024))
        source = list(range(24))
        output = []

        def decode():
            return source.pop(0) if source else None

        def process(frame, index):
            # Complete out of order so the encode reorder buffer is exercised.
            time.sleep((index % 3) * 0.001)
            return frame * 2

        def encode(frame, index):
            output.append((index, frame))

        result = scheduler.run(decode, process, encode)
        self.assertEqual([index for index, _ in output], list(range(24)))
        self.assertEqual([frame for _, frame in output], [n * 2 for n in range(24)])
        self.assertEqual(result["decoded"], 24)
        self.assertEqual(result["processed"], 24)
        self.assertEqual(result["encoded"], 24)
        self.assertLessEqual(result["max_queue_depths"].get("decode", 0), 2)
        self.assertLessEqual(result["max_queue_depths"].get("encode", 0), 2)

    def test_aggregate_frame_leases_never_exceed_four(self):
        """Two queue maxsizes are not a memory bound; decode waiting, CUDA
        work, and encode waiting must be capped together."""
        scheduler = UnifiedRuntimeScheduler(
            _hardware(12.0), _workload(3840, 2160),
            RuntimeTuning(worker_count=32, queue_depth=32, in_flight_frames=32,
                          ram_buffer_mb=4096))
        source = list(range(24))
        output = []

        def decode():
            return source.pop(0) if source else None

        def process(frame, index):
            return frame

        def encode(frame, index):
            time.sleep(0.001)
            output.append(index)

        result = scheduler.run(decode, process, encode)
        self.assertLessEqual(result["queue_capacity"], 4)
        self.assertLessEqual(result["in_flight_limit"], 4)
        self.assertLessEqual(result["max_active_frames"], 4)
        self.assertEqual(result["active_frames"], 0)
        self.assertEqual(output, list(range(24)))

    def test_pressure_reduces_future_admission_without_destroying_resources(self):
        scheduler = UnifiedRuntimeScheduler(
            _hardware(12.0), _workload(),
            RuntimeTuning(worker_count=4, queue_depth=3, in_flight_frames=4,
                          ram_buffer_mb=1024))
        original = scheduler.effective_inflight
        with patch.object(scheduler, "_resource_snapshot", return_value={
                "time": time.time(), "vram_free_gb": 0.5,
                "vram_total_gb": 12.0, "ram_available_gb": 12.0,
                "ram_total_gb": 16.0}):
            scheduler.observe({"decode": 3}, force=True)
            scheduler.observe({"decode": 3}, force=True)
        self.assertLessEqual(scheduler.effective_inflight, original)
        self.assertGreaterEqual(scheduler.effective_inflight, 1)
        self.assertEqual(scheduler.metrics.errors, [])

    def test_stateful_stabilization_uses_the_ordered_stream(self):
        scheduler = UnifiedRuntimeScheduler(
            _hardware(12.0), _workload(),
            RuntimeTuning(worker_count=4, queue_depth=3, in_flight_frames=4))
        self.assertTrue(scheduler.frame_pipeline_allowed(stateful_stabilization=True))
        self.assertTrue(scheduler.frame_pipeline_allowed(stateful_stabilization=False))


if __name__ == "__main__":
    unittest.main()
