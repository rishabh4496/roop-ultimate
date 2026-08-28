import json
import os
import tempfile
import unittest
from unittest.mock import patch

from roop.runtime_optimizer import (
    AutoTuner,
    CUDAGraphInvalidation,
    CUDAGraphManager,
    CUDAGraphRunner,
    HardwareProfile,
    ProfileStore,
    ResourceManager,
    RuntimeOptimizer,
    RuntimeProfile,
    RuntimeTuning,
    TensorRTEngineManager,
    WorkloadProfile,
)


def _hardware(vram, physical=24, logical=32, nvenc=True):
    return HardwareProfile(
        gpu_name="Test NVIDIA GPU",
        gpu_vendor="nvidia",
        compute_capability="8.9",
        vram_total_gb=vram,
        vram_available_gb=max(0.5, vram - 2.0),
        cuda_available=True,
        cuda_version="12.8",
        driver_version="999.1",
        tensorrt_available=True,
        tensorrt_version="10.0",
        onnxruntime_version="1.23.2",
        nvdec_available=nvenc,
        nvenc_available=nvenc,
        cpu_physical_cores=physical,
        cpu_logical_cores=logical,
        ram_total_gb=32.0,
        ram_available_gb=16.0,
        platform="Windows-test",
    )


def _workload(faces=1, enhanced=True, stabilized=True, width=1280, height=720):
    return WorkloadProfile(
        input_width=width,
        input_height=height,
        output_width=width,
        output_height=height,
        faces_per_frame=faces,
        enhancement_enabled=enhanced,
        enhancement_model="UltraMax" if enhanced else "",
        stabilization_enabled=stabilized,
        upscaling_enabled=False,
        temporal_detection_enabled=stabilized,
        video_length_frames=1000,
        fps=25.0,
        estimated_complexity=3.0 if enhanced else 1.5,
    )


class RuntimeOptimizerTests(unittest.TestCase):
    def test_stream_policy_is_bounded_and_small_gpu_is_serial(self):
        manager = CUDAGraphManager()
        small = manager.stream_policy({}, _workload(faces=2),
                                     _hardware(6.0, physical=8, logical=16),
                                     independent_work=4)
        desktop = manager.stream_policy({}, _workload(faces=2),
                                        _hardware(12.0), independent_work=4)
        self.assertEqual(small['stream_count'], 1)
        self.assertEqual(small['auxiliary_streams'], 0)
        self.assertFalse(small['safe_to_overlap'])
        self.assertEqual(desktop['stream_count'], 2)
        self.assertEqual(desktop['auxiliary_streams'], 1)
        self.assertTrue(desktop['safe_to_overlap'])

    def test_stream_policy_serializes_shared_mutable_buffers(self):
        policy = CUDAGraphManager.stream_policy(
            {}, _workload(faces=2), _hardware(12.0), independent_work=2,
            shared_mutable_buffers=True)
        self.assertEqual(policy['stream_count'], 1)
        self.assertFalse(policy['safe_to_overlap'])

    def test_graph_readiness_is_opt_in_and_reports_stream_policy(self):
        manager = CUDAGraphManager()
        workload = _workload()
        off = manager.readiness({}, workload, _hardware(12.0))
        on = manager.readiness({'trt_cuda_graph': True}, workload,
                               _hardware(12.0))
        self.assertFalse(off['requested'])
        self.assertFalse(off['safe'])
        self.assertTrue(on['requested'])
        self.assertTrue(on['safe'])
        self.assertIn('stream_policy', on)

    def test_graph_replay_rejects_identity_change(self):
        runner = CUDAGraphRunner(('model', 'shape-256'))
        runner.captured = True
        runner.graph = object()
        runner.static_inputs = ()
        with self.assertRaises(CUDAGraphInvalidation):
            runner.replay((), key=('model', 'shape-512'))

    def test_both_hardware_profiles_have_bounded_different_policies(self):
        tuner = AutoTuner()
        small, *_ = tuner.tune(_hardware(6.0, physical=8, logical=16), _workload(), {})
        desktop, *_ = tuner.tune(_hardware(12.0), _workload(faces=2), {})
        self.assertEqual(small.trt_context_count, 1)
        self.assertEqual(small.detector_pool_size, 0)
        self.assertEqual(small.swapper_pool_size, 0)
        self.assertGreaterEqual(desktop.trt_context_count, 1)
        self.assertGreater(desktop.detector_pool_size, small.detector_pool_size)
        for name, (lo, hi) in ResourceManager.BOUNDS.items():
            self.assertGreaterEqual(getattr(small, name), lo)
            self.assertLessEqual(getattr(small, name), hi)
            self.assertGreaterEqual(getattr(desktop, name), lo)
            self.assertLessEqual(getattr(desktop, name), hi)

    def test_explicit_settings_are_reported_and_never_applied_as_auto(self):
        settings = {
            "max_threads": 3,
            "_threads_auto": False,
            "auto_thread_selection": True,
            "perf_trt_pool": "4",
            "perf_detector_pool": "1",
            "cpu_opencv_threads": "4",
            "perf_batch_swap": "off",
            "perf_encoder_preset": "p7",
            "output_video_codec": "libx264",
            "provider": "tensorrt",
            "trt_precision": "fp32",
        }
        optimizer = RuntimeOptimizer(settings=settings)
        with patch.object(optimizer.hardware_profiler, "profile", return_value=_hardware(12.0)):
            profile = optimizer.build_profile(_workload(), save=False)
        self.assertEqual(profile.precision, "fp32")
        self.assertIn("max_threads", profile.explicit_settings)
        self.assertIn("perf_trt_pool", profile.explicit_settings)
        self.assertIn("cpu_opencv_threads", profile.explicit_settings)
        self.assertIn("perf_batch_swap", profile.explicit_settings)
        self.assertEqual(profile.tuning.worker_count, 3)
        self.assertEqual(profile.tuning.swapper_pool_size, 4)
        self.assertEqual(profile.tuning.detector_pool_size, 1)
        self.assertEqual(profile.tuning.batch_size, 1)
        self.assertEqual(profile.tuning.opencv_threads, 4)
        self.assertEqual(profile.tuning.encoder_preset, "p7")
        self.assertNotIn("ROOP_RUNTIME_CV_THREADS", RuntimeOptimizer.apply_environment(profile, settings))
        self.assertNotIn("ROOP_RUNTIME_BATCH_SIZE", RuntimeOptimizer.apply_environment(profile, settings))

    def test_runtime_environment_does_not_replace_explicit_environment(self):
        settings = {"cpu_opencv_threads": "auto", "perf_batch_swap": "auto"}
        optimizer = RuntimeOptimizer(settings=settings)
        with patch.object(optimizer.hardware_profiler, "profile", return_value=_hardware(12.0)):
            profile = optimizer.build_profile(_workload(), save=False)
        with patch.dict(os.environ, {"ROOP_RUNTIME_QUEUE_DEPTH": "99"}, clear=False):
            applied = RuntimeOptimizer.apply_environment(profile, settings)
            self.assertNotIn("ROOP_RUNTIME_QUEUE_DEPTH", applied)
            self.assertEqual(os.environ["ROOP_RUNTIME_QUEUE_DEPTH"], "99")

    def test_cache_key_contains_runtime_and_workload_identity(self):
        manager = TensorRTEngineManager()
        hardware = _hardware(12.0)
        first = manager.cache_key(hardware, _workload(width=1280), {"swap_model": "realswap"}, "mixed")
        second = manager.cache_key(hardware, _workload(width=1920, height=1080), {"swap_model": "realswap"}, "mixed")
        third = manager.cache_key(hardware, _workload(width=1280), {"swap_model": "hyperswap"}, "mixed")
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)

    def test_cache_key_changes_when_cuda_schedule_changes(self):
        manager = TensorRTEngineManager()
        hardware = _hardware(12.0)
        workload = _workload()
        serial = manager.cache_key(
            hardware, workload,
            {"swap_model": "realswap", "trt_auxiliary_streams": 0}, "mixed")
        overlapped = manager.cache_key(
            hardware, workload,
            {"swap_model": "realswap", "trt_auxiliary_streams": 1}, "mixed")
        self.assertNotEqual(serial, overlapped)

    def test_profile_store_round_trips_atomically(self):
        hardware = _hardware(6.0, physical=8, logical=16)
        workload = _workload()
        profile = RuntimeProfile(
            schema_version=1,
            created_at=1.0,
            hardware=hardware,
            workload=workload,
            tuning=RuntimeTuning(),
            precision="mixed",
            provider="tensorrt",
            cache_key="test-key",
        )
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(directory)
            path = store.save(profile)
            loaded = store.load("test-key")
            self.assertTrue(path.exists())
            self.assertEqual(loaded["cache_key"], "test-key")
            self.assertEqual(json.loads(path.read_text())["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
