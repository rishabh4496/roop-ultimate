import os
import tempfile
import unittest

from roop.runtime_optimizer import (
    HardwareProfile,
    ProfileStore,
    RuntimeAutotuner,
    RuntimeOptimizer,
    RuntimeTuning,
    WorkloadProfile,
)


def _hardware(vram=12.0):
    return HardwareProfile(
        gpu_name="test GPU", gpu_vendor="nvidia", architecture="Ada Lovelace",
        compute_capability="8.9", vram_total_gb=vram,
        vram_available_gb=max(1.0, vram - 2.0), cuda_available=True,
        cuda_version="12.8", driver_version="test", tensorrt_available=True,
        tensorrt_version="10", onnxruntime_version="1", nvenc_available=True,
        nvdec_available=True, nvenc_codecs=("h264_nvenc", "hevc_nvenc"),
        cpu_physical_cores=8, cpu_logical_cores=16, ram_total_gb=32,
        ram_available_gb=16)


class Phase14AutotunerTests(unittest.TestCase):
    def test_staged_search_is_bounded_and_uses_end_to_end_score(self):
        tuner = RuntimeAutotuner()
        base = RuntimeTuning(worker_count=4, queue_depth=2, ram_buffer_mb=1024)
        workload = WorkloadProfile(input_width=1280, input_height=720,
                                   faces_per_frame=2, enhancement_enabled=True)
        calls = []

        def measure(candidate, warmup):
            calls.append((candidate, warmup))
            # Only queue depth 3 wins; this checks that later stages can build
            # from the current best rather than a fixed hardware guess.
            return {"end_to_end_fps": 12 if candidate.get("queue_depth") == 3 else 10,
                    "peak_vram_gb": 5, "peak_ram_gb": 8,
                    "stable": True, "quality_regression": False}

        settings = {"provider": "cuda", "trt_precision": "fp32",
                    "perf_trt_pool": 1, "perf_batch_swap": "on",
                    "max_threads": 4, "cpu_ort_intra_threads": 1,
                    "cpu_ort_inter_threads": 1, "cpu_opencv_threads": 1,
                    "output_video_codec": "libx264",
                    "perf_encoder_preset": "medium"}
        tuning, report = tuner.tune(base, _hardware(), workload, settings, measure,
                                    warmup_frames=7, max_candidates=10)
        self.assertLessEqual(len(calls), 10)
        self.assertEqual(calls[0][1], 7)
        self.assertEqual(report["baseline_fps"], 10)
        self.assertEqual(report["best_fps"], 12)
        self.assertEqual(tuning.queue_depth, 3)
        self.assertIn("candidates_tested", report)

    def test_quality_or_instability_rejects_faster_candidate(self):
        tuner = RuntimeAutotuner()
        safe = tuner.score(tuner_measure(10, stable=True), _hardware())
        bad = tuner.score(tuner_measure(20, stable=False), _hardware())
        quality = tuner.score(tuner_measure(20, quality=True), _hardware())
        self.assertGreater(safe, 0)
        self.assertEqual(bad, 0)
        self.assertEqual(quality, 0)

    def test_explicit_codec_removes_encoder_trials(self):
        tuner = RuntimeAutotuner()
        base = RuntimeTuning(encoder="hevc_nvenc")
        workload = WorkloadProfile(input_width=1280, input_height=720)
        candidates = tuner.candidates(base, _hardware(), workload,
                                      {"output_video_codec": "hevc_nvenc"})
        self.assertFalse(any(item.get("stage") == "encoder" for item in candidates))

    def test_profile_round_trip_includes_new_workload_and_autotune_fields(self):
        optimizer = RuntimeOptimizer(profile_dir=tempfile.mkdtemp())
        optimizer.hardware_profiler._profile = _hardware()
        workload = WorkloadProfile(input_width=1920, input_height=1080,
                                   tracking_enabled=True, mask_enabled=True,
                                   enhancement_resolution=512,
                                   output_codec="hevc_nvenc")
        profile = optimizer.autotune_profile(
            workload,
            lambda candidate, warmup: {"end_to_end_fps": 10,
                                       "peak_vram_gb": 4,
                                       "peak_ram_gb": 8},
            max_candidates=1, save=True)
        loaded = optimizer.store.load(profile.cache_key)
        self.assertEqual(loaded["workload"]["output_codec"], "hevc_nvenc")
        self.assertTrue(loaded["autotune"]["selected"])


def tuner_measure(fps, stable=True, quality=False):
    return type("M", (), {"end_to_end_fps": fps, "peak_vram_gb": 4,
                           "peak_ram_gb": 8, "startup_seconds": 0,
                           "stable": stable, "quality_regression": quality})()


if __name__ == "__main__":
    unittest.main()
