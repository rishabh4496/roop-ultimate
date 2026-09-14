from __future__ import annotations

import importlib
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch


builder = importlib.import_module("tools.build_trt_engines")


class _Response:
    def __init__(self, payload: bytes):
        self._payload = payload
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size: int = -1):
        payload, self._payload = self._payload, b""
        return payload


class _FakeSessionOptions:
    def __init__(self):
        self.graph_optimization_level = None


class _FakeGraphOptimizationLevel:
    ORT_ENABLE_ALL = "all"


class _FakeSession:
    calls = []

    def __init__(self, path, sess_options, providers):
        self.calls.append((path, sess_options, providers))

    def get_providers(self):
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider"]


class _FakeOrt:
    SessionOptions = _FakeSessionOptions
    GraphOptimizationLevel = _FakeGraphOptimizationLevel
    InferenceSession = _FakeSession


class BuildTrtEnginesTests(unittest.TestCase):
    def setUp(self):
        self._offline = builder._OFFLINE
        _FakeSession.calls.clear()

    def tearDown(self):
        builder._OFFLINE = self._offline

    def test_registry_contains_the_three_requested_models_and_profiles(self):
        self.assertEqual(
            set(builder.MODEL_REGISTRY),
            {"inswapper_128.onnx", "GPEN-BFR-512.onnx", "scrfd_2.5g_kps.onnx"},
        )
        self.assertEqual(
            builder.MODEL_REGISTRY["inswapper_128.onnx"]["shapes"]["max"],
            "target:8x3x128x128,source:8x512",
        )
        self.assertEqual(
            builder.MODEL_REGISTRY["scrfd_2.5g_kps.onnx"]["shapes"]["min"],
            "input:1x3x640x640",
        )

    def test_workspace_matches_the_two_supported_gpu_tiers(self):
        self.assertEqual(builder._workspace_bytes(12 * 1024**3), 4096 * 1024**2)
        self.assertEqual(builder._workspace_bytes(6 * 1024**3), 1536 * 1024**2)

    def test_workspace_override_is_explicit_and_has_a_safe_floor(self):
        with patch.dict(os.environ, {"ROOP_TRT_WORKSPACE_BYTES": str(2 * 1024**3)}):
            self.assertEqual(builder._workspace_bytes(6 * 1024**3), 2 * 1024**3)
        with patch.dict(os.environ, {"ROOP_TRT_WORKSPACE_BYTES": "not-an-int"}):
            with self.assertRaises(ValueError):
                builder._workspace_bytes(12 * 1024**3)

    def test_existing_model_is_not_downloaded_again(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "model.onnx"
            destination.write_bytes(b"already here")
            with patch.object(builder.urllib.request, "urlopen") as urlopen:
                result = builder.download_file("https://example.invalid/model.onnx", destination)
            self.assertEqual(result, destination)
            urlopen.assert_not_called()

    def test_offline_mode_rejects_a_missing_model_without_network_access(self):
        builder._OFFLINE = True
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "missing.onnx"
            with patch.object(builder.urllib.request, "urlopen") as urlopen:
                with self.assertRaisesRegex(RuntimeError, "Offline mode"):
                    builder.download_file("https://example.invalid/model.onnx", destination)
            urlopen.assert_not_called()

    def test_download_is_atomic_and_removes_partial_file_on_success(self):
        builder._OFFLINE = False
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "model.onnx"
            with patch.object(builder.urllib.request, "urlopen", return_value=_Response(b"onnx")):
                result = builder.download_file("https://example.invalid/model.onnx", destination)
            self.assertEqual(result.read_bytes(), b"onnx")
            self.assertFalse(Path(str(destination) + ".part").exists())

    def test_download_tries_the_public_fallback_after_a_primary_failure(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "model.onnx"
            with patch.object(
                builder.urllib.request,
                "urlopen",
                side_effect=[urllib.error.HTTPError("https://primary", 401, "auth", {}, None), _Response(b"onnx")],
            ) as urlopen:
                result = builder.download_file(
                    "https://primary/model.onnx",
                    destination,
                    ("https://fallback/model.onnx",),
                )
            self.assertEqual(result.read_bytes(), b"onnx")
            self.assertEqual(urlopen.call_count, 2)
            self.assertFalse(Path(str(destination) + ".part").exists())

    def test_provider_options_include_cache_identity_and_profiles(self):
        with tempfile.TemporaryDirectory() as root:
            models = Path(root) / "models"
            models.mkdir()
            model = models / "GPEN-BFR-512.onnx"
            model.write_bytes(b"test model")
            old_models, old_cache = builder.MODELS_DIR, builder.CACHE_DIR
            builder.MODELS_DIR = models
            builder.CACHE_DIR = models / "trt_cache"
            try:
                with patch.object(builder, "_total_vram_bytes", return_value=6 * 1024**3):
                    options = builder._provider_options(
                        "GPEN-BFR-512.onnx",
                        builder.MODEL_REGISTRY["GPEN-BFR-512.onnx"],
                    )
            finally:
                builder.MODELS_DIR, builder.CACHE_DIR = old_models, old_cache
            self.assertEqual(options["trt_max_workspace_size"], 1536 * 1024**2)
            self.assertTrue(options["trt_engine_cache_enable"])
            self.assertTrue(options["trt_timing_cache_enable"])
            self.assertEqual(options["trt_profile_opt_shapes"], "input:2x3x512x512")
            self.assertIn("GPEN-BFR-512", options["trt_engine_cache_prefix"])

    def test_provider_options_remap_a_single_profile_to_the_real_graph_input_name(self):
        with tempfile.TemporaryDirectory() as root:
            models = Path(root) / "models"
            models.mkdir()
            model = models / "scrfd_2.5g_kps.onnx"
            model.write_bytes(b"test model")
            old_models, old_cache = builder.MODELS_DIR, builder.CACHE_DIR
            builder.MODELS_DIR = models
            builder.CACHE_DIR = models / "trt_cache"
            try:
                with patch.object(builder, "_model_input_names", return_value=("input.1",)):
                    with patch.object(builder, "_total_vram_bytes", return_value=12 * 1024**3):
                        options = builder._provider_options(
                            "scrfd_2.5g_kps.onnx",
                            builder.MODEL_REGISTRY["scrfd_2.5g_kps.onnx"],
                        )
            finally:
                builder.MODELS_DIR, builder.CACHE_DIR = old_models, old_cache
            self.assertEqual(options["trt_profile_min_shapes"], "input.1:1x3x640x640")
            self.assertEqual(options["trt_profile_opt_shapes"], "input.1:1x3x640x640")

    def test_compile_engine_requires_tensor_rt_to_be_active(self):
        with tempfile.TemporaryDirectory() as root:
            models = Path(root) / "models"
            models.mkdir()
            model = models / "GPEN-BFR-512.onnx"
            model.write_bytes(b"test model")
            old_models, old_cache = builder.MODELS_DIR, builder.CACHE_DIR
            builder.MODELS_DIR = models
            builder.CACHE_DIR = models / "trt_cache"
            try:
                with patch.object(builder, "_total_vram_bytes", return_value=12 * 1024**3):
                    compile_ort = _FakeOrt
                    builder.compile_engine(
                        "GPEN-BFR-512.onnx",
                        builder.MODEL_REGISTRY["GPEN-BFR-512.onnx"],
                        compile_ort,
                    )
            finally:
                builder.MODELS_DIR, builder.CACHE_DIR = old_models, old_cache
            self.assertEqual(len(_FakeSession.calls), 1)
            _, session_options, providers = _FakeSession.calls[0]
            self.assertEqual(session_options.graph_optimization_level, "all")
            self.assertEqual(providers[0][0], "TensorrtExecutionProvider")
            self.assertEqual(providers[1][0], "CUDAExecutionProvider")
            self.assertEqual(providers[0][1]["trt_max_workspace_size"], 4096 * 1024**2)


if __name__ == "__main__":
    unittest.main()
