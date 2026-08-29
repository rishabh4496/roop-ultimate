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
    HardwareProfiler,
    ProfileStore,
    ResourceManager,
    RuntimeOptimizer,
    RuntimeProfile,
    RuntimeTuning,
    PrecisionSelector,
    TensorRTEngineManager,
    WorkloadProfile,
    apply_cpu_affinity,
    small_card_decode_policy,
    small_card_enhancer_policy,
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
        fp16_supported=True,
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
    def _hybrid_hardware(self):
        return HardwareProfile(
            **{**_hardware(12.0).__dict__,
               "cpu_performance_indices": tuple(range(16)),
               "cpu_efficiency_indices": tuple(range(16, 32)),
               "cpu_performance_cores": 8,
               "cpu_efficiency_cores": 16,
               "cpu_topology_source": "test-cpu-set",
               "os_affinity_supported": True})

    def test_cpu_distribution_candidates_use_measured_logical_topology(self):
        tuner = AutoTuner()
        hardware = self._hybrid_hardware()
        with patch.dict(os.environ, {"ROOP_CPU_DISTRIBUTION": "p_only"}, clear=False):
            p_only, *_ = tuner.tune(hardware, _workload(), {})
        with patch.dict(os.environ, {"ROOP_CPU_DISTRIBUTION": "p_priority_e",
                                     "ROOP_CPU_E_LIMIT": "2"}, clear=False):
            p_priority, *_ = tuner.tune(hardware, _workload(), {})
        with patch.dict(os.environ, {"ROOP_CPU_DISTRIBUTION": "p_plus_e"}, clear=False):
            p_plus_e, *_ = tuner.tune(hardware, _workload(), {})
        self.assertEqual((p_only.worker_count, p_only.cpu_performance_threads,
                          p_only.cpu_efficiency_threads), (16, 16, 0))
        self.assertEqual((p_priority.worker_count, p_priority.cpu_performance_threads,
                          p_priority.cpu_efficiency_threads), (18, 16, 2))
        self.assertEqual((p_plus_e.worker_count, p_plus_e.cpu_performance_threads,
                          p_plus_e.cpu_efficiency_threads), (32, 16, 16))

    def test_cpu_affinity_uses_measured_indices_and_limited_efficiency_cores(self):
        hardware = self._hybrid_hardware()
        with patch.dict(os.environ, {"ROOP_CPU_DISTRIBUTION": "p_priority_e",
                                     "ROOP_CPU_E_LIMIT": "2"}, clear=False), \
                patch("psutil.Process") as process:
            result = apply_cpu_affinity(hardware)
        self.assertTrue(result["applied"])
        self.assertEqual(result["indices"], tuple(range(18)))
        process.return_value.cpu_affinity.assert_called_once_with(list(range(18)))

    def test_small_card_precision_is_reported_as_effective_fp32(self):
        selector = PrecisionSelector()
        self.assertEqual(selector.select({"trt_precision": "mixed"},
                                         _hardware(6.0)), "fp32")
        self.assertEqual(selector.select({"trt_precision": "fp16"},
                                         _hardware(6.0)), "fp32")
        self.assertEqual(selector.select({"trt_precision": "mixed"},
                                         _hardware(12.0)), "mixed")

    def test_small_card_graph_is_explicitly_not_admitted(self):
        manager = CUDAGraphManager()
        result = manager.readiness(
            {"trt_cuda_graph": True}, _workload(), _hardware(6.0))
        self.assertTrue(result["requested"])
        self.assertFalse(result["safe"])
        self.assertIn("sub-7GB", result["reason"])

    def test_small_card_enhancer_policy_is_hardware_adaptive(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ROOP_SMALL_CARD_ENHANCER", None)
            small = small_card_enhancer_policy(_hardware(6.0), "GPEN 256 Pro")
            desktop = small_card_enhancer_policy(_hardware(12.0), "GPEN 256 Pro")
        self.assertTrue(small["changed"])
        self.assertEqual(small["effective"], "None")
        self.assertFalse(desktop["changed"])
        self.assertEqual(desktop["effective"], "GPEN 256 Pro")

    def test_small_card_enhancer_keep_override_is_explicit(self):
        with patch.dict(os.environ, {"ROOP_SMALL_CARD_ENHANCER": "keep"}):
            result = small_card_enhancer_policy(_hardware(6.0), "GPEN 256 Pro")
        self.assertFalse(result["changed"])
        self.assertEqual(result["effective"], "GPEN 256 Pro")

    def test_small_card_auto_decode_prefers_lower_rss_cpu_path(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ROOP_NVDEC", None)
            os.environ.pop("ROOP_SMALL_CARD_NVDEC", None)
            result = small_card_decode_policy(_hardware(6.0))
        self.assertTrue(result["changed"])
        self.assertEqual(result["effective"], "0")
        self.assertIn("RSS", result["reason"])

    def test_small_card_decode_explicit_nvdec_is_not_overridden(self):
        with patch.dict(os.environ, {"ROOP_NVDEC": "1"}):
            result = small_card_decode_policy(_hardware(6.0))
        self.assertFalse(result["changed"])
        self.assertEqual(result["effective"], "1")

    def test_desktop_auto_decode_is_unchanged(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ROOP_NVDEC", None)
            os.environ.pop("ROOP_SMALL_CARD_NVDEC", None)
            result = small_card_decode_policy(_hardware(12.0))
        self.assertFalse(result["changed"])
        self.assertEqual(result["effective"], "auto")

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
        self.assertEqual(small.upscale_tile_batch_size, 1)
        self.assertEqual(small.face_concurrency, 1)
        self.assertEqual(small.in_flight_frames, 1)
        self.assertGreaterEqual(desktop.face_concurrency, 2)
        for name, (lo, hi) in ResourceManager.BOUNDS.items():
            self.assertGreaterEqual(getattr(small, name), lo)
            self.assertLessEqual(getattr(small, name), hi)
            self.assertGreaterEqual(getattr(desktop, name), lo)
            self.assertLessEqual(getattr(desktop, name), hi)

    def test_upscale_batch_hint_is_workload_specific(self):
        tuner = AutoTuner()
        desktop, *_ = tuner.tune(_hardware(12.0),
                                 _workload(faces=1, enhanced=False), {})
        upscale, *_ = tuner.tune(
            _hardware(12.0),
            WorkloadProfile(input_width=1280, input_height=720,
                            output_width=2560, output_height=1440,
                            upscaling_enabled=True), {})
        self.assertEqual(desktop.upscale_tile_batch_size, 1)
        self.assertEqual(upscale.upscale_tile_batch_size, 1)

    def test_upscale_hint_is_not_leaked_from_face_profile(self):
        optimizer = RuntimeOptimizer()
        with patch.object(optimizer.hardware_profiler, "profile",
                          return_value=_hardware(12.0)):
            profile = optimizer.build_profile(_workload(), save=False)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ROOP_RUNTIME_UPSCALE_TILE_BATCH", None)
            applied = RuntimeOptimizer.apply_environment(profile, {})
        self.assertNotIn("ROOP_RUNTIME_UPSCALE_TILE_BATCH", applied)
        self.assertNotIn("ROOP_RUNTIME_UPSCALE_TILE_BATCH", os.environ)

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

    def test_trt_auxiliary_stream_auto_sentinel_does_not_break_profile(self):
        settings = {"trt_auxiliary_streams": -1, "provider": "tensorrt",
                    "trt_precision": "mixed"}
        optimizer = RuntimeOptimizer(settings=settings)
        with patch.object(optimizer.hardware_profiler, "profile",
                          return_value=_hardware(12.0)):
            profile = optimizer.build_profile(_workload(), save=False)
        self.assertEqual(profile.tuning.cuda_auxiliary_streams, 0)
        self.assertIn("trt_auxiliary_streams", profile.automatic_settings)

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

    def test_cache_key_isolates_cpu_distribution_ab_candidates(self):
        manager = TensorRTEngineManager()
        hardware = self._hybrid_hardware()
        workload = _workload()
        with patch.dict(os.environ, {"ROOP_CPU_DISTRIBUTION": "p_only"}, clear=False):
            p_only = manager.cache_key(hardware, workload, {}, "mixed")
        with patch.dict(os.environ, {"ROOP_CPU_DISTRIBUTION": "p_plus_e"}, clear=False):
            p_plus_e = manager.cache_key(hardware, workload, {}, "mixed")
        self.assertNotEqual(p_only, p_plus_e)

    def test_cache_key_isolated_by_architecture_and_vram(self):
        manager = TensorRTEngineManager()
        workload = _workload()
        ampere = HardwareProfile(
            gpu_name="same graph device", gpu_vendor="nvidia",
            architecture="Ampere", compute_capability="8.6",
            vram_total_gb=6.0, cuda_available=True,
            cuda_version="12.8", driver_version="616.56",
            tensorrt_available=True, tensorrt_version="10.9",
            onnxruntime_version="1.23.2")
        ada = HardwareProfile(
            gpu_name="same graph device", gpu_vendor="nvidia",
            architecture="Ada Lovelace", compute_capability="8.9",
            vram_total_gb=12.0, cuda_available=True,
            cuda_version="12.8", driver_version="616.56",
            tensorrt_available=True, tensorrt_version="10.9",
            onnxruntime_version="1.23.2")
        self.assertNotEqual(manager.cache_key(ampere, workload, {}, "mixed"),
                            manager.cache_key(ada, workload, {}, "mixed"))

    def test_cache_key_changes_for_model_revision(self):
        manager = TensorRTEngineManager()
        hardware = _hardware(12.0)
        workload = _workload()
        first = manager.cache_key(
            hardware, workload, {"swap_model": "realswap", "model_hash": "a"}, "mixed")
        second = manager.cache_key(
            hardware, workload, {"swap_model": "realswap", "model_hash": "b"}, "mixed")
        self.assertNotEqual(first, second)

    def test_architecture_mapping_uses_compute_capability_not_model_name(self):
        from roop.runtime_optimizer import HardwareProfiler
        self.assertEqual(HardwareProfiler._architecture((8, 6)), "Ampere")
        self.assertEqual(HardwareProfiler._architecture((8, 9)), "Ada Lovelace")
        self.assertEqual(HardwareProfiler._architecture((9, 0)), "Hopper")
        self.assertEqual(HardwareProfiler._architecture((12, 0)), "SM 12.0")

    def test_profile_serializes_capabilities_without_claiming_unknown_modes(self):
        hardware = _hardware(6.0)
        payload = hardware.as_dict()
        self.assertIn("capabilities", payload)
        self.assertFalse(payload["capabilities"]["bf16"])
        self.assertFalse(payload["capabilities"]["fp8"])
        self.assertEqual(payload["vram_tier"], "small")

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


class FFmpegCapabilityDetectionTests(unittest.TestCase):
    """NVDEC/NVENC must survive an unhelpful PATH.

    Regression for a false negative found on the physical RTX 3060 Laptop: the
    profiler resolved ffmpeg with `shutil.which` alone, so a process that did
    not inherit Pinokio's PATH recorded `nvenc_available=False` on a machine
    whose NVENC works. That silently downgrades the chosen encoder from
    hevc_nvenc to libx264 and hides the nvenc candidates from the autotuner.
    """

    def test_resolves_ffmpeg_when_not_on_path(self):
        with patch("roop.runtime_optimizer.shutil.which", return_value=None):
            resolved = HardwareProfiler._resolve_ffmpeg()
        if resolved is None:
            self.skipTest("no bundled ffmpeg on this host to fall back to")
        self.assertTrue(os.path.isfile(resolved), resolved)

    def test_capabilities_are_not_silently_false_with_a_real_binary(self):
        ffmpeg = HardwareProfiler._resolve_ffmpeg()
        if not ffmpeg:
            self.skipTest("ffmpeg unavailable on this host")
        nvdec, nvenc, dec, enc = HardwareProfiler._ffmpeg_capabilities(
            ffmpeg, True)
        # Whatever this host reports, the probe must actually have run: an
        # encoder list that parses to nothing while ffmpeg exists is the
        # signature of the bug (an empty probe, not an absent engine).
        self.assertTrue(
            HardwareProfiler._command(ffmpeg, "-hide_banner", "-encoders"),
            "the encoder probe returned nothing for a resolvable ffmpeg")
        if enc:
            self.assertTrue(nvenc)
        if dec:
            self.assertTrue(nvdec)

    def test_no_cuda_means_no_hardware_video(self):
        self.assertEqual(
            HardwareProfiler._ffmpeg_capabilities("ffmpeg", False),
            (False, False, (), ()))


if __name__ == "__main__":
    unittest.main()
