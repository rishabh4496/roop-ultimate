"""The bottleneck classifier must read the queue names its callers actually send.

There are two callers and they use two vocabularies:

  * `UnifiedRuntimeScheduler.run_stream` reports its own queues as
    ``{"decode": ..., "encode": ...}``;
  * `ProcessMgr._runtime_queue_snapshot` -- the path a real render takes --
    reports the same two ends as ``{"input": ..., "output": ...}``.

`classify_bottleneck` only looked up the first spelling, with a default of 0.
On the production path that made BOTH queue branches structurally unreachable
while looking measured: an absent key came back as 0, which is indistinguishable
from a genuinely empty queue. It is the same shape as Phase 15's `0.0` CPU and
`None` GPU reading as measurements -- a default standing in for a number that
was never taken -- and it silently narrowed the classifier to its utilization
branches on every render.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.runtime_optimizer import HardwareProfile, RuntimeTuning, WorkloadProfile
from roop.runtime_scheduler import UnifiedRuntimeScheduler, _queue_depth


def _scheduler():
    hardware = HardwareProfile(
        gpu_name="synthetic NVIDIA GPU", gpu_vendor="nvidia",
        architecture="test", compute_capability="8.6",
        vram_total_gb=12.0, vram_available_gb=8.0,
        cuda_available=True, tensorrt_available=True,
        cuda_version="test", tensorrt_version="test",
        onnxruntime_version="test", nvdec_available=True,
        nvenc_available=True, cpu_physical_cores=8, cpu_logical_cores=16,
        ram_total_gb=32.0, ram_available_gb=24.0)
    workload = WorkloadProfile(input_width=1280, input_height=720,
                               output_width=1280, output_height=720,
                               faces_per_frame=2, enhancement_enabled=True)
    return UnifiedRuntimeScheduler(
        hardware, workload,
        RuntimeTuning(worker_count=4, queue_depth=2, in_flight_frames=4,
                      ram_buffer_mb=1024))


def _snapshot(queue_depths):
    """The shape classify_bottleneck reads, with no resource pressure set."""
    return {"resource_samples": [{"queue_depths": dict(queue_depths),
                                  "vram_pressure_pct": 0.0,
                                  "gpu_utilization_pct": 10.0,
                                  "cpu_utilization_pct": 10.0}]}


class QueueDepthLookup(unittest.TestCase):
    def test_an_unreported_queue_has_no_depth(self):
        self.assertIsNone(_queue_depth({}, ("encode", "output")))
        self.assertIsNone(_queue_depth({"decode": 3}, ("encode", "output")))

    def test_a_reported_zero_is_a_measurement_not_an_absence(self):
        self.assertEqual(_queue_depth({"encode": 0}, ("encode", "output")), 0)

    def test_either_spelling_resolves(self):
        self.assertEqual(_queue_depth({"output": 7}, ("encode", "output")), 7)
        self.assertEqual(_queue_depth({"input": 5}, ("decode", "input")), 5)

    def test_the_primary_spelling_wins_when_both_are_present(self):
        self.assertEqual(
            _queue_depth({"encode": 1, "output": 9}, ("encode", "output")), 1)


class ClassifierReadsProcessMgrVocabulary(unittest.TestCase):
    def setUp(self):
        self.scheduler = _scheduler()
        self.capacity = self.scheduler.queue_capacity

    def test_a_full_output_queue_is_encode_bound(self):
        # ProcessMgr calls this end "output". Before the fix this returned
        # anything but "encode-bound", because "encode" was never in the dict.
        verdict = self.scheduler.classify_bottleneck(
            _snapshot({"input": 1, "output": self.capacity}))
        self.assertEqual(verdict, "encode-bound")

    def test_a_full_encode_queue_is_still_encode_bound(self):
        verdict = self.scheduler.classify_bottleneck(
            _snapshot({"decode": 1, "encode": self.capacity}))
        self.assertEqual(verdict, "encode-bound")

    def test_a_starved_input_queue_is_decode_bound(self):
        self.scheduler.metrics.decoded = 10
        verdict = self.scheduler.classify_bottleneck(
            _snapshot({"input": 0, "output": 0}))
        self.assertEqual(verdict, "decode-bound")

    def test_queues_nobody_reported_do_not_produce_a_queue_verdict(self):
        # With no queue evidence at all the classifier must fall through to the
        # utilization branches rather than infer an empty decode queue.
        self.scheduler.metrics.decoded = 10
        verdict = self.scheduler.classify_bottleneck(_snapshot({}))
        self.assertNotIn(verdict, ("decode-bound", "encode-bound"))

    def test_a_healthy_pipeline_is_not_called_queue_bound(self):
        self.scheduler.metrics.decoded = 10
        verdict = self.scheduler.classify_bottleneck(
            _snapshot({"input": 1, "output": 1}))
        self.assertNotIn(verdict, ("decode-bound", "encode-bound"))


class ProcessMgrSnapshotMatchesTheClassifier(unittest.TestCase):
    """Pin the actual key names, so the two sides cannot drift apart again."""

    def test_process_mgr_reports_input_and_output(self):
        import inspect
        from roop.ProcessMgr import ProcessMgr
        src = inspect.getsource(ProcessMgr._runtime_queue_snapshot)
        self.assertIn("'input'", src)
        self.assertIn("'output'", src)

    def test_the_classifier_accepts_those_names(self):
        scheduler = _scheduler()
        depths = {"input": 0, "output": scheduler.queue_capacity}
        self.assertIsNotNone(_queue_depth(depths, ("encode", "output")))
        self.assertIsNotNone(_queue_depth(depths, ("decode", "input")))


if __name__ == "__main__":
    unittest.main()
