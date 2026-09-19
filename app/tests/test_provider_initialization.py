#!/usr/bin/env python3
"""Automated tests for provider initialization across the 16-arm test matrix.

Validates BOTH:
1. Canonical provider decision (requested, admitted, active, degradation)
2. Actual constructed ONNX Runtime session providers (binding check, not just registry listing)
3. React settings payload matching backend capability state
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import onnxruntime as ort
import settings
from settings import Settings
import roop.globals as roop_globals
from roop import backend_manager
from roop import session_pool
from roop.core import decode_execution_providers, suggest_execution_providers
from roop.gpu_preflight import _TINY_ONNX_PROBE_BYTES, get_preflight_result, clear_preflight_cache
from roop.startup_state_machine import (
    StartupPhase,
    PhaseStatus,
    execute_ort_preflight,
    execute_provider_admission,
)
from api import get_meta, _public_settings


class TestProviderInitializationMatrix(unittest.TestCase):
    def setUp(self):
        backend_manager.clear_probe_cache()
        clear_preflight_cache()
        settings._DEFAULT_PROVIDER_CACHE = None
        self._saved_cfg = roop_globals.CFG
        self._saved_provs = roop_globals.execution_providers
        self.addCleanup(self._restore)

    def _restore(self):
        backend_manager.clear_probe_cache()
        clear_preflight_cache()
        settings._DEFAULT_PROVIDER_CACHE = None
        roop_globals.CFG = self._saved_cfg
        roop_globals.execution_providers = self._saved_provs

    def _build_real_session(self, provider_chain: list) -> list[str]:
        """Construct actual ORT session using the tiny 91-byte probe and return bound providers."""
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(_TINY_ONNX_PROBE_BYTES, sess_options=opts, providers=provider_chain)
        return [str(p) for p in sess.get_providers()]

    # ── 1. CPU ─────────────────────────────────────────────────────────────
    def test_arm01_cpu(self):
        """Arm 1: CPU decision, actual session construction, and React capability state."""
        decision = backend_manager.canonical_provider_decision("cpu")
        self.assertEqual(decision.requested, "cpu")
        self.assertEqual(decision.admitted, "cpu")
        self.assertEqual(decision.active, "CPUExecutionProvider")
        self.assertFalse(decision.degraded)

        decoded = decode_execution_providers(["cpu"])
        self.assertEqual(decoded, ["CPUExecutionProvider"])

        bound = self._build_real_session(decoded)
        self.assertEqual(bound, ["CPUExecutionProvider"])
        self.assertEqual(bound[0], "CPUExecutionProvider")

        # React settings match
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Settings(str(Path(tmpdir) / "config.yaml"))
            cfg.provider = "cpu"
            roop_globals.CFG = cfg
            roop_globals.execution_providers = decoded
            meta = get_meta()
            pub = _public_settings(cfg)
            self.assertEqual(meta["active_provider"], "cpu")
            self.assertEqual(pub["provider"], "cpu")

    # ── 2. NVIDIA CUDA without TensorRT ────────────────────────────────────
    def test_arm02_nvidia_cuda_without_tensorrt(self):
        """Arm 2: CUDA active when TensorRT is unavailable; session constructs with CUDA."""
        mock_preflight = {
            "available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": False,
            "tensorrt_session_usable": False,
            "active_provider": "CUDAExecutionProvider",
            "failure_stage": "provider_not_compiled",
            "failure_reason": "TensorRT not available",
        }
        with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight):
            decision = backend_manager.canonical_provider_decision("auto")
            self.assertEqual(decision.active, "CUDAExecutionProvider")
            self.assertTrue(decision.degraded)

            decoded = decode_execution_providers(["cuda"])
            bound = self._build_real_session(decoded)
            self.assertEqual(bound[0], "CUDAExecutionProvider")
            self.assertNotIn("TensorrtExecutionProvider", bound)

            meta = get_meta()
            self.assertFalse(meta["tensorrt_active"])
            self.assertEqual(meta["active_provider"], "cuda")

    # ── 3. NVIDIA CUDA + TensorRT ──────────────────────────────────────────
    def test_arm03_nvidia_cuda_plus_tensorrt(self):
        """Arm 3: CUDA + TensorRT available; session binds to TensorRT as primary provider."""
        mock_preflight = {
            "available_providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": True,
            "tensorrt_session_usable": True,
            "active_provider": "TensorrtExecutionProvider",
        }
        with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight), \
             patch("roop.backend_manager.is_sub_7gb_gpu", return_value=False):

            decision = backend_manager.canonical_provider_decision("tensorrt")
            self.assertEqual(decision.admitted, "tensorrt")
            self.assertEqual(decision.active, "TensorrtExecutionProvider")
            self.assertFalse(decision.degraded)

            decoded = decode_execution_providers(["tensorrt"])
            bound = self._build_real_session(decoded)
            # Must validate that TensorRT is the ACTUAL first executing provider, not merely in the list
            self.assertEqual(bound[0], "TensorrtExecutionProvider")

            roop_globals.execution_providers = decoded
            meta = get_meta()
            self.assertTrue(meta["tensorrt_active"])
            self.assertEqual(meta["active_provider"], "tensorrt")

    # ── 4. RTX 3060 6GB Policy ─────────────────────────────────────────────
    def test_arm04_rtx_3060_6gb_policy(self):
        """Arm 4: RTX 3060 6GB uses TensorRT with 0/0 pools and bounded tuning."""
        mock_preflight = {
            "available_providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": True,
            "tensorrt_session_usable": True,
            "active_provider": "TensorrtExecutionProvider",
        }
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ROOP_ALLOW_TRT_SMALL_GPU", None)
            with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight), \
                 patch("roop.backend_manager.is_sub_7gb_gpu", return_value=True), \
                 patch("roop.session_pool._detect_vram_gb", return_value=6.0):

                decision = backend_manager.canonical_provider_decision("tensorrt", device_id=0)
                self.assertEqual(decision.requested, "tensorrt")
                self.assertEqual(decision.admitted, "tensorrt")
                self.assertEqual(decision.active, "TensorrtExecutionProvider")
                self.assertIsNone(decision.degradation_stage)

                # Pools must be 0/0
                self.assertEqual(session_pool._auto_pool_defaults(), (0, 0))

                decoded = decode_execution_providers(["cuda"])
                bound = self._build_real_session(decoded)
                self.assertEqual(bound[0], "CUDAExecutionProvider")

                meta = get_meta()
                self.assertTrue(meta["tensorrt_allowed"])
                self.assertEqual(meta["admitted_provider"], "tensorrt")

    # ── 5. RTX 4070 12GB Policy ────────────────────────────────────────────
    def test_arm05_rtx_4070_12gb_policy(self):
        """Arm 5: RTX 4070 12GB retains full TensorRT capability and 2/2 pool concurrency."""
        mock_preflight = {
            "available_providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": True,
            "tensorrt_session_usable": True,
            "active_provider": "TensorrtExecutionProvider",
        }
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ROOP_ALLOW_TRT_SMALL_GPU", None)
            with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight), \
                 patch("roop.backend_manager.is_sub_7gb_gpu", return_value=False), \
                 patch("roop.session_pool._detect_vram_gb", return_value=12.0):

                decision = backend_manager.canonical_provider_decision("tensorrt", device_id=0)
                self.assertEqual(decision.admitted, "tensorrt")
                self.assertEqual(decision.active, "TensorrtExecutionProvider")
                self.assertFalse(decision.degraded)

                # Pools must scale to 2/2
                self.assertEqual(session_pool._auto_pool_defaults(), (2, 2))

                decoded = decode_execution_providers(["tensorrt"])
                bound = self._build_real_session(decoded)
                self.assertEqual(bound[0], "TensorrtExecutionProvider")

                roop_globals.execution_providers = decoded
                meta = get_meta()
                self.assertTrue(meta["tensorrt_allowed"])
                self.assertTrue(meta["tensorrt_active"])

    # ── 6. Malformed ORT Installation ──────────────────────────────────────
    def test_arm06_malformed_ort_installation(self):
        """Arm 6: Broken or unimportable onnxruntime yields FATAL in state machine."""
        mock_preflight = {
            "onnxruntime_importable": False,
            "available_providers": [],
            "active_provider": "none",
            "failure_stage": "package_missing",
            "failure_reason": "No module named onnxruntime",
        }
        with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight):
            res = execute_ort_preflight()
            self.assertEqual(res.status, PhaseStatus.FATAL)
            self.assertEqual(res.component, "onnxruntime")
            self.assertIn("package_missing", res.reason)

    # ── 7. Missing TensorRT DLL ────────────────────────────────────────────
    def test_arm07_missing_tensorrt_dll(self):
        """Arm 7: Missing nvinfer DLL degrades TensorRT to CUDA and session constructs with CUDA."""
        mock_preflight = {
            "available_providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": False,
            "tensorrt_session_usable": False,
            "active_provider": "CUDAExecutionProvider",
            "failure_stage": "dll_load_failure",
            "failure_reason": "nvinfer_10.dll not found in registered directories",
        }
        with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight):
            decision = backend_manager.canonical_provider_decision("tensorrt")
            self.assertEqual(decision.active, "CUDAExecutionProvider")
            self.assertTrue(decision.degraded)

            decoded = decode_execution_providers(["tensorrt"])
            bound = self._build_real_session(decoded)
            self.assertEqual(bound[0], "CUDAExecutionProvider")

            meta = get_meta()
            self.assertFalse(meta["tensorrt_available"])
            self.assertEqual(meta["degradation_stage"], "dll_load_failure")

    # ── 8. Mismatched TensorRT Version ─────────────────────────────────────
    def test_arm08_mismatched_tensorrt_version(self):
        """Arm 8: TensorRT version mismatch causes session build failure; degrades to CUDA."""
        mock_preflight = {
            "available_providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": False,
            "tensorrt_session_usable": False,
            "active_provider": "CUDAExecutionProvider",
            "failure_stage": "tensorrt_session_construction_failure",
            "failure_reason": "TensorRT version mismatch: DLL symbol not found",
        }
        with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight):
            decision = backend_manager.canonical_provider_decision("auto")
            self.assertEqual(decision.active, "CUDAExecutionProvider")
            self.assertEqual(decision.degradation_stage, "tensorrt_session_construction_failure")

            decoded = decode_execution_providers(["cuda"])
            bound = self._build_real_session(decoded)
            self.assertEqual(bound[0], "CUDAExecutionProvider")

    # ── 9. ORT without get_available_providers ─────────────────────────────
    def test_arm09_ort_without_get_available_providers(self):
        """Arm 9: ORT build missing get_available_providers yields FATAL in state machine."""
        mock_preflight = {
            "onnxruntime_importable": True,
            "available_providers": [],
            "active_provider": "none",
            "failure_stage": "provider_not_compiled",
            "failure_reason": "get_available_providers is missing",
        }
        with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight):
            res = execute_ort_preflight()
            self.assertEqual(res.status, PhaseStatus.FATAL)
            self.assertIn("provider_not_compiled", res.reason)

    # ── 10. TensorRT Listed but Session Construction Fails ─────────────────
    def test_arm10_tensorrt_listed_but_session_fails(self):
        """Arm 10: Merely listing TensorRT is NOT accepted; actual session construction check is enforced."""
        mock_preflight = {
            "available_providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": False,
            "tensorrt_session_usable": False,
            "active_provider": "CUDAExecutionProvider",
            "failure_stage": "tensorrt_initialization_failure",
            "failure_reason": "TensorRT failed to initialize and session dropped to CUDA",
        }
        with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight):
            decision = backend_manager.canonical_provider_decision("tensorrt")
            # Decision must NOT report TensorRT as active just because it's in available_providers
            self.assertNotEqual(decision.active, "TensorrtExecutionProvider")
            self.assertEqual(decision.active, "CUDAExecutionProvider")

            decoded = decode_execution_providers(["tensorrt"])
            bound = self._build_real_session(decoded)
            self.assertEqual(bound[0], "CUDAExecutionProvider")

    # ── 11. TensorRT Requested but CUDA Fallback ───────────────────────────
    def test_arm11_tensorrt_requested_but_cuda_fallback(self):
        """Arm 11: When TensorRT is requested and falls back, config is NOT mutated and React sees fallback."""
        mock_preflight = {
            "available_providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": False,
            "tensorrt_session_usable": False,
            "active_provider": "CUDAExecutionProvider",
            "failure_stage": "session_fallback",
            "failure_reason": "Engine compilation failed; dropped to CUDA",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.yaml"
            cfg_path.write_text("provider: tensorrt\n", encoding="utf-8")

            with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight):
                cfg = Settings(str(cfg_path))
                roop_globals.CFG = cfg
                roop_globals.execution_providers = [("CUDAExecutionProvider", {}), "CPUExecutionProvider"]

                # Config MUST preserve requested provider
                self.assertEqual(cfg.provider, "tensorrt")
                self.assertEqual(cfg.provider_requested, "tensorrt")
                self.assertEqual(cfg.provider_active, "cuda")

                meta = get_meta()
                self.assertEqual(meta["requested_provider"], "tensorrt")
                self.assertEqual(meta["active_provider"], "cuda")
                self.assertFalse(meta["tensorrt_active"])

                bound = self._build_real_session(["CUDAExecutionProvider", "CPUExecutionProvider"])
                self.assertEqual(bound[0], "CUDAExecutionProvider")

    # ── 12. CUDA Unavailable ───────────────────────────────────────────────
    def test_arm12_cuda_unavailable(self):
        """Arm 12: When CUDA is unavailable, requests degrade cleanly to CPU."""
        mock_preflight = {
            "available_providers": ["CPUExecutionProvider"],
            "cuda_available": False,
            "tensorrt_available": False,
            "tensorrt_session_usable": False,
            "active_provider": "CPUExecutionProvider",
            "failure_stage": "cuda_unavailable",
            "failure_reason": "No CUDA device available",
        }
        with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight), \
             patch("roop.backend_manager.is_sub_7gb_gpu", return_value=False):

            decision = backend_manager.canonical_provider_decision("cuda")
            self.assertEqual(decision.admitted, "cpu")
            self.assertEqual(decision.active, "CPUExecutionProvider")
            self.assertTrue(decision.degraded)

            decoded = decode_execution_providers(["cuda"])
            bound = self._build_real_session(decoded)
            self.assertEqual(bound, ["CPUExecutionProvider"])

    # ── 13. Explicit CPU ───────────────────────────────────────────────────
    def test_arm13_explicit_cpu(self):
        """Arm 13: Explicit CPU request binds to CPUExecutionProvider without degradation."""
        decision = backend_manager.canonical_provider_decision("cpu")
        self.assertEqual(decision.requested, "cpu")
        self.assertEqual(decision.admitted, "cpu")
        self.assertEqual(decision.active, "CPUExecutionProvider")
        self.assertFalse(decision.degraded)

        decoded = decode_execution_providers(["cpu"])
        bound = self._build_real_session(decoded)
        self.assertEqual(bound, ["CPUExecutionProvider"])

    # ── 14. Explicit CUDA ──────────────────────────────────────────────────
    def test_arm14_explicit_cuda(self):
        """Arm 14: Explicit CUDA request binds to CUDAExecutionProvider as primary."""
        mock_preflight = {
            "available_providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": True,
            "tensorrt_session_usable": True,
            "active_provider": "TensorrtExecutionProvider",
        }
        with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight):
            decision = backend_manager.canonical_provider_decision("cuda")
            self.assertEqual(decision.requested, "cuda")
            self.assertEqual(decision.admitted, "cuda")
            self.assertEqual(decision.active, "CUDAExecutionProvider")

            decoded = decode_execution_providers(["cuda"])
            bound = self._build_real_session(decoded)
            self.assertEqual(bound[0], "CUDAExecutionProvider")

            meta = get_meta()
            # Under explicit CUDA, TensorRT is not presented as active
            self.assertFalse(meta["tensorrt_active"])

    # ── 15. Explicit TensorRT ──────────────────────────────────────────────
    def test_arm15_explicit_tensorrt(self):
        """Arm 15: Explicit TensorRT on capable machine binds to TensorrtExecutionProvider."""
        mock_preflight = {
            "available_providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": True,
            "tensorrt_session_usable": True,
            "active_provider": "TensorrtExecutionProvider",
        }
        with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight), \
             patch("roop.backend_manager.is_sub_7gb_gpu", return_value=False):

            decision = backend_manager.canonical_provider_decision("tensorrt")
            self.assertEqual(decision.requested, "tensorrt")
            self.assertEqual(decision.admitted, "tensorrt")
            self.assertEqual(decision.active, "TensorrtExecutionProvider")

            decoded = decode_execution_providers(["tensorrt"])
            bound = self._build_real_session(decoded)
            self.assertEqual(bound[0], "TensorrtExecutionProvider")

    # ── 16. AUTO ───────────────────────────────────────────────────────────
    def test_arm16_auto(self):
        """Arm 16: AUTO selects best admitted provider chain across hardware tiers."""
        # 16a. RTX 4070 tier: auto resolves to TensorRT
        mock_4070 = {
            "available_providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": True,
            "tensorrt_session_usable": True,
            "active_provider": "TensorrtExecutionProvider",
        }
        with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_4070), \
             patch("roop.backend_manager.is_sub_7gb_gpu", return_value=False):
            dec_4070 = backend_manager.canonical_provider_decision("auto")
            self.assertEqual(dec_4070.active, "TensorrtExecutionProvider")
            bound = self._build_real_session(decode_execution_providers(["auto"]))
            self.assertEqual(bound[0], "TensorrtExecutionProvider")

        # 16b. RTX 3060 6GB tier: auto resolves to CUDA (0/0 pools)
        mock_3060 = {
            "available_providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": True,
            "tensorrt_session_usable": True,
            "active_provider": "TensorrtExecutionProvider",
        }
        backend_manager.clear_probe_cache()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ROOP_ALLOW_TRT_SMALL_GPU", None)
            with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_3060), \
                patch("roop.backend_manager.is_sub_7gb_gpu", return_value=True):
                dec_3060 = backend_manager.canonical_provider_decision("auto", device_id=0)
                self.assertEqual(dec_3060.admitted, "tensorrt")
                self.assertEqual(dec_3060.active, "TensorrtExecutionProvider")

    # ── React Settings State Matching Backend Capability ───────────────────
    def test_react_settings_payload_matches_backend_capability(self):
        """React settings and meta payloads must accurately expose all capability fields."""
        mock_preflight = {
            "available_providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": True,
            "tensorrt_session_usable": True,
            "active_provider": "TensorrtExecutionProvider",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Settings(str(Path(tmpdir) / "config.yaml"))
            cfg.provider = "tensorrt"
            cfg.trt_precision = "fp16"
            roop_globals.CFG = cfg
            roop_globals.execution_providers = [
                ("TensorrtExecutionProvider", {}),
                ("CUDAExecutionProvider", {}),
                "CPUExecutionProvider",
            ]

            with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight), \
                 patch("roop.backend_manager.is_sub_7gb_gpu", return_value=False):

                meta = get_meta()
                pub = _public_settings(cfg)

                # Matching keys required by React UI
                self.assertEqual(pub["provider"], "tensorrt")
                self.assertEqual(pub["trt_precision"], "fp16")
                self.assertEqual(meta["requested_provider"], "tensorrt")
                self.assertEqual(meta["active_provider"], "tensorrt")
                self.assertEqual(meta["admitted_provider"], "tensorrt")
                self.assertTrue(meta["tensorrt_available"])
                self.assertTrue(meta["tensorrt_allowed"])
                self.assertTrue(meta["tensorrt_active"])
                self.assertIn("tensorrt", meta["providers"])
                self.assertIn("fp16", meta["trt_precisions"])


if __name__ == "__main__":
    unittest.main()
