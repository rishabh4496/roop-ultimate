#!/usr/bin/env python3
"""Tests for sub-7GB TensorRT admission policy across hardware tiers (6GB, 8GB, 12GB, CPU)."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import settings
from settings import Settings
import roop.globals as roop_globals
from roop import backend_manager
from roop import session_pool


class TestSub7GbTensorRTPolicy(unittest.TestCase):
    def setUp(self):
        backend_manager.clear_probe_cache()
        settings._DEFAULT_PROVIDER_CACHE = None
        self._saved_cfg = roop_globals.CFG
        self._saved_provs = roop_globals.execution_providers
        self.addCleanup(self._restore)

    def _restore(self):
        backend_manager.clear_probe_cache()
        settings._DEFAULT_PROVIDER_CACHE = None
        roop_globals.CFG = self._saved_cfg
        roop_globals.execution_providers = self._saved_provs

    def test_6gb_gpu_default_admits_tensorrt_with_safe_pools(self):
        """A 6GB RTX card may use qualified TensorRT without desktop pools."""
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

                self.assertTrue(backend_manager.is_sub_7gb_gpu(0))
                self.assertTrue(backend_manager.allow_small_gpu_trt())
                self.assertTrue(backend_manager.is_trt_allowed_for_device(0))
                self.assertTrue(backend_manager._small_gpu(0))

                decision = backend_manager.canonical_provider_decision("tensorrt", device_id=0)
                self.assertEqual(decision.requested, "tensorrt")
                self.assertEqual(decision.admitted, "tensorrt")
                self.assertEqual(decision.active, "TensorrtExecutionProvider")
                self.assertIsNone(decision.degradation_stage)
                self.assertIsNone(decision.degradation_reason)

                adm = backend_manager.provider_admission("tensorrt", device_id=0)
                self.assertTrue(adm["admitted"])
                self.assertEqual(adm["admitted_provider"], "tensorrt")
                self.assertTrue(adm["tensorrt_allowed"])
                self.assertTrue(adm["is_sub_7gb_gpu"])
                self.assertTrue(adm["override"])

                # Resource pools must be 0/0 on 6GB card
                swp_pool, det_pool = session_pool._auto_pool_defaults()
                self.assertEqual((swp_pool, det_pool), (0, 0))

    def test_6gb_gpu_legacy_override_is_not_required_for_tensorrt(self):
        """Legacy opt-in variables do not control 6GB TensorRT admission."""
        mock_preflight = {
            "available_providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": True,
            "tensorrt_session_usable": True,
            "active_provider": "TensorrtExecutionProvider",
        }
        with patch.dict(os.environ, {"ROOP_ALLOW_TRT_SMALL_GPU": "1"}):
            with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight), \
                 patch("roop.backend_manager.is_sub_7gb_gpu", return_value=True), \
                 patch("roop.session_pool._detect_vram_gb", return_value=6.0):

                self.assertTrue(backend_manager.is_sub_7gb_gpu(0))
                self.assertTrue(backend_manager.allow_small_gpu_trt())
                self.assertTrue(backend_manager.is_trt_allowed_for_device(0))
                self.assertTrue(backend_manager._small_gpu(0))

                decision = backend_manager.canonical_provider_decision("tensorrt", device_id=0)
                self.assertEqual(decision.requested, "tensorrt")
                self.assertEqual(decision.admitted, "tensorrt")
                self.assertEqual(decision.active, "TensorrtExecutionProvider")
                self.assertIsNone(decision.degradation_stage)

                adm = backend_manager.provider_admission("tensorrt", device_id=0)
                self.assertTrue(adm["admitted"])
                self.assertEqual(adm["admitted_provider"], "tensorrt")
                self.assertTrue(adm["tensorrt_allowed"])
                self.assertTrue(adm["override"])

                # 6GB card MUST NOT inherit RTX 4070 pools (2/2).
                swp_pool, det_pool = session_pool._auto_pool_defaults()
                self.assertEqual((swp_pool, det_pool), (0, 0))

    def test_8gb_gpu_admits_tensorrt_with_standard_pools(self):
        """An 8GB GPU (e.g. RTX 3070 8GB) admits TensorRT without opt-in and scales to 2/2 pools."""
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
                 patch("roop.session_pool._detect_vram_gb", return_value=8.0):

                self.assertFalse(backend_manager.is_sub_7gb_gpu(0))
                self.assertTrue(backend_manager.is_trt_allowed_for_device(0))
                self.assertFalse(backend_manager._small_gpu(0))

                decision = backend_manager.canonical_provider_decision("tensorrt", device_id=0)
                self.assertEqual(decision.requested, "tensorrt")
                self.assertEqual(decision.admitted, "tensorrt")
                self.assertEqual(decision.active, "TensorrtExecutionProvider")

                adm = backend_manager.provider_admission("tensorrt", device_id=0)
                self.assertTrue(adm["admitted"])
                self.assertTrue(adm["tensorrt_allowed"])
                self.assertFalse(adm["is_sub_7gb_gpu"])

                swp_pool, det_pool = session_pool._auto_pool_defaults()
                self.assertEqual((swp_pool, det_pool), (2, 2))

    def test_12gb_gpu_retains_full_tensorrt_capability(self):
        """An RTX 4070 (12GB) retains full TensorRT capability and 2/2 pooled concurrency."""
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

                self.assertFalse(backend_manager.is_sub_7gb_gpu(0))
                self.assertTrue(backend_manager.is_trt_allowed_for_device(0))

                decision = backend_manager.canonical_provider_decision("tensorrt", device_id=0)
                self.assertEqual(decision.requested, "tensorrt")
                self.assertEqual(decision.admitted, "tensorrt")
                self.assertEqual(decision.active, "TensorrtExecutionProvider")

                adm = backend_manager.provider_admission("tensorrt", device_id=0)
                self.assertTrue(adm["admitted"])
                self.assertTrue(adm["tensorrt_allowed"])
                self.assertFalse(adm["is_sub_7gb_gpu"])

                swp_pool, det_pool = session_pool._auto_pool_defaults()
                self.assertEqual((swp_pool, det_pool), (2, 2))

    def test_cpu_only_falls_back_to_cpu_cleanly(self):
        """On a CPU-only environment without CUDA, auto and tensorrt requests degrade to CPU."""
        mock_preflight = {
            "available_providers": ["CPUExecutionProvider"],
            "cuda_available": False,
            "tensorrt_available": False,
            "tensorrt_session_usable": False,
            "active_provider": "CPUExecutionProvider",
            "failure_stage": "cuda_unavailable",
            "failure_reason": "No CUDA-capable GPU detected",
        }
        with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight), \
             patch("roop.backend_manager.is_sub_7gb_gpu", return_value=False), \
             patch("roop.session_pool._detect_vram_gb", return_value=0.0):

            decision = backend_manager.canonical_provider_decision("auto")
            self.assertEqual(decision.admitted, "tensorrt")
            self.assertEqual(decision.active, "CPUExecutionProvider")
            self.assertTrue(decision.degraded)

            trt_decision = backend_manager.canonical_provider_decision("tensorrt")
            self.assertEqual(trt_decision.active, "CPUExecutionProvider")
            self.assertTrue(trt_decision.degraded)

            swp_pool, det_pool = session_pool._auto_pool_defaults()
            self.assertEqual((swp_pool, det_pool), (0, 0))

    def test_qualified_tensorrt_remains_active_on_small_gpu(self):
        """A usable TensorRT session is not downgraded solely for being sub-7GB."""
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
                 patch("roop.backend_manager.is_sub_7gb_gpu", return_value=True):

                decision = backend_manager.canonical_provider_decision("tensorrt", device_id=0)
                self.assertEqual(decision.admitted, "tensorrt")
                self.assertEqual(decision.active, "TensorrtExecutionProvider")
                self.assertEqual(decision.active_chain, ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"))


if __name__ == "__main__":
    unittest.main()
