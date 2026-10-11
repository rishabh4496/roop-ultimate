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
            builder.MODEL_REGISTRY["inswapper_128.onnx"]["shapes"],
            {"min": "target:1x3x128x128,source:1x512",
             "opt": "target:4x3x128x128,source:4x512",
             "max": "target:8x3x128x128,source:8x512"},
        )
        self.assertEqual(
            builder.MODEL_REGISTRY["scrfd_2.5g_kps.onnx"]["shapes"],
            {"min": "input:1x3x640x640", "opt": "input:4x3x640x640", "max": "input:8x3x640x640"},
        )

    def test_the_batch_models_are_built_from_a_verified_dynamic_graph(self):
        for name in ("inswapper_128.onnx", "scrfd_2.5g_kps.onnx", "GPEN-BFR-512.onnx"):
            self.assertTrue(builder.MODEL_REGISTRY[name]["dynamic"], name)
        self.assertNotIn("dynamic", builder.NATIVE_MODEL_REGISTRY["xseg.onnx"])

    def test_workspace_matches_the_two_supported_gpu_tiers(self):
        self.assertEqual(builder._workspace_bytes(12 * 1024**3), 2048 * 1024**2)    # RTX 4070
        self.assertEqual(builder._workspace_bytes(6 * 1024**3), 1024 * 1024**2)     # RTX 3060 laptop

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
                config = builder.MODEL_REGISTRY["GPEN-BFR-512.onnx"]
                resolved = builder.ResolvedModel(model, dict(config["shapes"]), True, "test")
                with patch.object(builder, "_total_vram_bytes", return_value=6 * 1024**3):
                    options = builder._provider_options(resolved, config)
            finally:
                builder.MODELS_DIR, builder.CACHE_DIR = old_models, old_cache
            self.assertEqual(options["trt_max_workspace_size"], 1024 * 1024**2)
            self.assertIs(options["trt_fp16_enable"], True)
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
                config = builder.MODEL_REGISTRY["scrfd_2.5g_kps.onnx"]
                resolved = builder.ResolvedModel(model, dict(config["shapes"]), True, "test")
                with patch.object(builder, "_model_input_names", return_value=("input.1",)):
                    with patch.object(builder, "_total_vram_bytes", return_value=12 * 1024**3):
                        options = builder._provider_options(resolved, config)
            finally:
                builder.MODELS_DIR, builder.CACHE_DIR = old_models, old_cache
            self.assertEqual(options["trt_profile_min_shapes"], "input.1:1x3x640x640")
            self.assertEqual(options["trt_profile_opt_shapes"], "input.1:4x3x640x640")
            self.assertEqual(options["trt_profile_max_shapes"], "input.1:8x3x640x640")

    # -- precision ----------------------------------------------------------------------------

    def test_fp16_is_the_default_and_the_swapper_is_the_documented_exception(self):
        reg = builder.MODEL_REGISTRY
        with patch.dict(os.environ, {}, clear=False):
            for key in ("ROOP_TRT_FP16", "ROOP_SWAP_FP16"):
                os.environ.pop(key, None)
            self.assertTrue(builder._fp16_for(reg["scrfd_2.5g_kps.onnx"]))
            self.assertTrue(builder._fp16_for(reg["GPEN-BFR-512.onnx"]))
            self.assertFalse(builder._fp16_for(reg["inswapper_128.onnx"]))   # overflows in FP16
            self.assertFalse(builder._fp16_for(builder.NATIVE_MODEL_REGISTRY["xseg.onnx"]))
        with patch.dict(os.environ, {"ROOP_SWAP_FP16": "1"}):
            self.assertTrue(builder._fp16_for(reg["inswapper_128.onnx"]))
            self.assertFalse(builder._fp16_for(builder.NATIVE_MODEL_REGISTRY["xseg.onnx"]))
        with patch.dict(os.environ, {"ROOP_TRT_FP16": "0", "ROOP_SWAP_FP16": "1"}):
            self.assertFalse(builder._fp16_for(reg["scrfd_2.5g_kps.onnx"]))

    # -- dynamic batch resolution -------------------------------------------------------------

    def _in_models_dir(self, root, name, payload=b"x"):
        models = Path(root) / "models"
        models.mkdir()
        (models / name).write_bytes(payload)
        old = builder.MODELS_DIR, builder.CACHE_DIR
        builder.MODELS_DIR, builder.CACHE_DIR = models, models / "trt_cache"
        self.addCleanup(lambda: setattr(builder, "MODELS_DIR", old[0]))
        self.addCleanup(lambda: setattr(builder, "CACHE_DIR", old[1]))
        return models

    def test_a_dynamic_model_is_built_from_the_verified_graph_and_keeps_its_profile(self):
        with tempfile.TemporaryDirectory() as root:
            models = self._in_models_dir(root, "inswapper_128.onnx")
            dynamic = models / "inswapper_128_dynamic.onnx"
            with patch("convert_dynamic_shapes.prepare_dynamic_model", return_value=dynamic) as prepare:
                resolved = builder._resolve_model("inswapper_128.onnx", builder.MODEL_REGISTRY["inswapper_128.onnx"])
            self.assertEqual(resolved.path, dynamic)
            self.assertTrue(resolved.dynamic)
            self.assertEqual(resolved.shapes["max"], "target:8x3x128x128,source:8x512")
            self.assertEqual(prepare.call_args.kwargs["batches"], [1, 4, 8])

    def test_a_model_that_cannot_be_made_dynamic_is_built_static_batch_1_and_labelled(self):
        with tempfile.TemporaryDirectory() as root:
            models = self._in_models_dir(root, "GPEN-BFR-512.onnx")
            with patch("convert_dynamic_shapes.prepare_dynamic_model",
                       side_effect=RuntimeError("not batch-convertible: batch 4 ...")):
                resolved = builder._resolve_model("GPEN-BFR-512.onnx", builder.MODEL_REGISTRY["GPEN-BFR-512.onnx"])
            self.assertEqual(resolved.path, models / "GPEN-BFR-512.onnx")
            self.assertFalse(resolved.dynamic)
            self.assertEqual(resolved.shapes, {t: "input:1x3x512x512" for t in ("min", "opt", "max")})
            self.assertIn("STATIC batch 1", resolved.note)

    def test_profile_batches_cover_min_opt_max_and_1_4_8_inside_the_range(self):
        profile = builder.MODEL_REGISTRY["inswapper_128.onnx"]["shapes"]
        self.assertEqual(builder._profile_batches(profile), [1, 4, 8])
        self.assertEqual(builder._profile_batches(builder.MODEL_REGISTRY["GPEN-BFR-512.onnx"]["shapes"]), [1, 2, 4])
        self.assertEqual(builder._profile_batches(builder._static_profile(profile)), [1])

    # -- compile_engine -----------------------------------------------------------------------

    class _Session:
        """Stands in for a strict TensorRT session; writes an engine file on its first run, as TensorRT does."""

        def __init__(self, engine_dir, prefix, write=True, nan_at=None):
            self.engine_dir, self.prefix, self.write, self.nan_at = engine_dir, prefix, write, nan_at
            self.ran = []

        def get_providers(self):
            return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

        def get_inputs(self):
            import types
            return [types.SimpleNamespace(name="input", type="tensor(float)")]

        def run(self, _outputs, feeds):
            import numpy as np
            batch = feeds["input"].shape[0]
            self.ran.append(batch)
            if self.write:
                (self.engine_dir / (self.prefix + "_fake.engine")).write_bytes(b"engine")
            out = np.zeros((batch, 3), dtype=np.float32)
            if self.nan_at == batch:
                out[0, 0] = np.nan
            return [out]

    def _compile(self, root, session_factory, name="GPEN-BFR-512.onnx", vram=12 * 1024**3):
        models = self._in_models_dir(root, name)
        cache = models / "trt_cache"
        cache.mkdir()
        captured = {}

        def factory(path, providers=None, sess_options=None, tag=None, **_kw):
            captured.update(path=path, providers=providers, tag=tag)
            prefix = providers[0][1]["trt_engine_cache_prefix"]
            captured["session"] = session_factory(cache, prefix)
            return captured["session"]

        config = builder.MODEL_REGISTRY[name]
        with patch.object(builder, "_total_vram_bytes", return_value=vram), \
                patch("convert_dynamic_shapes.prepare_dynamic_model", side_effect=RuntimeError("static")), \
                patch("roop.strict_session.create_strictly_accelerated_session", side_effect=factory):
            row = builder.compile_engine(name, config, _FakeOrt)
        return row, captured

    def test_compile_engine_goes_through_the_strict_factory_and_runs_the_whole_profile(self):
        with tempfile.TemporaryDirectory() as root:
            row, captured = self._compile(root, lambda cache, prefix: self._Session(cache, prefix))
            self.assertEqual(captured["providers"][0][0], "TensorrtExecutionProvider")
            self.assertEqual(captured["providers"][1][0], "CUDAExecutionProvider")
            self.assertEqual(captured["providers"][0][1]["trt_max_workspace_size"], 2048 * 1024**2)
            self.assertEqual(captured["tag"], "build:GPEN-BFR-512")
            self.assertEqual(captured["session"].ran, [1])           # static fallback => one batch
            self.assertEqual(row["engines_new"], 1)
            self.assertEqual(row["precision"], "FP16")

    def test_compile_engine_uses_the_laptop_workspace_on_a_6gb_card(self):
        with tempfile.TemporaryDirectory() as root:
            _row, captured = self._compile(root, lambda cache, prefix: self._Session(cache, prefix),
                                           vram=6 * 1024**3)
            self.assertEqual(captured["providers"][0][1]["trt_max_workspace_size"], 1024 * 1024**2)

    def test_compile_engine_fails_when_no_engine_file_was_written(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, "no engine file"):
                self._compile(root, lambda cache, prefix: self._Session(cache, prefix, write=False))

    def test_compile_engine_fails_on_non_finite_output(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, "NaN/Inf"):
                self._compile(root, lambda cache, prefix: self._Session(cache, prefix, nan_at=1))

    def test_compile_engine_requires_tensorrt_to_be_the_first_active_provider(self):
        class CudaFirst(self._Session):
            def get_providers(self):
                return ["CUDAExecutionProvider", "TensorrtExecutionProvider"]

        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, "not the active provider"):
                self._compile(root, lambda cache, prefix: CudaFirst(cache, prefix))


if __name__ == "__main__":
    unittest.main()
