#!/usr/bin/env python3
"""Tests for app/roop/gpu_preflight.py authoritative preflight logic."""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch, MagicMock

import torch
from roop import gpu_preflight


class TestGpuPreflight(unittest.TestCase):
    def setUp(self):
        gpu_preflight.clear_preflight_cache()
        self.addCleanup(gpu_preflight.clear_preflight_cache)

    def test_required_schema_keys_present(self):
        """Preflight must always return the exact required schema keys."""
        required_keys = {
            "onnxruntime_importable",
            "onnxruntime_version",
            "onnxruntime_path",
            "available_providers",
            "cuda_available",
            "tensorrt_available",
            "tensorrt_session_usable",
            "active_provider",
            "failure_stage",
            "failure_reason",
        }
        res = gpu_preflight.run_gpu_preflight()
        self.assertEqual(set(res.keys()), required_keys)
        self.assertIsInstance(res["onnxruntime_importable"], bool)
        self.assertIsInstance(res["onnxruntime_version"], str)
        self.assertIsInstance(res["onnxruntime_path"], str)
        self.assertIsInstance(res["available_providers"], list)
        self.assertIsInstance(res["cuda_available"], bool)
        self.assertIsInstance(res["tensorrt_available"], bool)
        self.assertIsInstance(res["tensorrt_session_usable"], bool)
        self.assertIsInstance(res["active_provider"], str)

    def test_package_missing_failure(self):
        """When onnxruntime is absent or fails to import, failure_stage is package_missing."""
        with patch.dict(sys.modules, {"onnxruntime": None}):
            res = gpu_preflight.run_gpu_preflight(force_probe=True)
            self.assertFalse(res["onnxruntime_importable"])
            self.assertEqual(res["failure_stage"], "package_missing")
            self.assertIn("not installed", res["failure_reason"])

    def test_namespace_package_failure(self):
        """When onnxruntime has __file__ None (namespace package), failure_stage is package_missing."""
        mock_ort = types.ModuleType("onnxruntime")
        mock_ort.__file__ = None
        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            res = gpu_preflight.run_gpu_preflight(force_probe=True)
            self.assertFalse(res["onnxruntime_importable"])
            self.assertEqual(res["failure_stage"], "package_missing")
            self.assertIn("namespace package", res["failure_reason"])

    def test_provider_not_compiled_failure(self):
        """When get_available_providers is missing, failure_stage is provider_not_compiled."""
        mock_ort = types.ModuleType("onnxruntime")
        mock_ort.__file__ = "/env/site-packages/onnxruntime/__init__.py"
        mock_ort.__version__ = "1.23.2"
        # No get_available_providers attribute
        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            res = gpu_preflight.run_gpu_preflight(force_probe=True)
            self.assertTrue(res["onnxruntime_importable"])
            self.assertEqual(res["failure_stage"], "provider_not_compiled")
            self.assertIn("get_available_providers", res["failure_reason"])

    def test_cuda_unavailable_failure(self):
        """When CUDA device is not visible, failure_stage is cuda_unavailable."""
        mock_ort = types.ModuleType("onnxruntime")
        mock_ort.__file__ = "/env/site-packages/onnxruntime/__init__.py"
        mock_ort.__version__ = "1.23.2"
        mock_ort.get_available_providers = lambda: [
            "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"
        ]

        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            with patch("torch.cuda.is_available", return_value=False):
                res = gpu_preflight.run_gpu_preflight(force_probe=True)
                self.assertFalse(res["cuda_available"])
                self.assertFalse(res["tensorrt_session_usable"])
                self.assertEqual(res["failure_stage"], "cuda_unavailable")

    def test_dll_load_failure(self):
        """When TensorRT DLLs are missing on Windows, failure_stage is dll_load_failure."""
        mock_ort = types.ModuleType("onnxruntime")
        mock_ort.__file__ = "/env/site-packages/onnxruntime/__init__.py"
        mock_ort.__version__ = "1.23.2"
        mock_ort.get_available_providers = lambda: [
            "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"
        ]

        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.device_count", return_value=1):
                    with patch("sys.platform", "win32"):
                        with patch.object(gpu_preflight, "_check_tensorrt_dlls", return_value=False):
                            res = gpu_preflight.run_gpu_preflight(force_probe=True)
                            self.assertEqual(res["failure_stage"], "dll_load_failure")
                            self.assertFalse(res["tensorrt_session_usable"])

    def test_tensorrt_initialization_failure_drops_to_cuda(self):
        """When TensorRT EP fails to initialize in session and drops to CUDA."""
        mock_ort = types.ModuleType("onnxruntime")
        mock_ort.__file__ = "/env/site-packages/onnxruntime/__init__.py"
        mock_ort.__version__ = "1.23.2"
        mock_ort.get_available_providers = lambda: [
            "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"
        ]

        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        mock_ort.InferenceSession = MagicMock(return_value=mock_session)
        mock_ort.SessionOptions = MagicMock()

        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.device_count", return_value=1):
                    with patch.object(gpu_preflight, "_check_tensorrt_dlls", return_value=True):
                        res = gpu_preflight.run_gpu_preflight(force_probe=True)
                        self.assertFalse(res["tensorrt_session_usable"])
                        self.assertEqual(res["active_provider"], "CUDAExecutionProvider")
                        self.assertEqual(res["failure_stage"], "tensorrt_initialization_failure")

    def test_tensorrt_session_construction_failure(self):
        """When InferenceSession constructor raises an exception."""
        mock_ort = types.ModuleType("onnxruntime")
        mock_ort.__file__ = "/env/site-packages/onnxruntime/__init__.py"
        mock_ort.__version__ = "1.23.2"
        mock_ort.get_available_providers = lambda: [
            "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"
        ]

        def fail_session(*args, **kwargs):
            raise RuntimeError("ORT C++ engine allocation failed")

        mock_ort.InferenceSession = fail_session
        mock_ort.SessionOptions = MagicMock()

        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.device_count", return_value=1):
                    with patch.object(gpu_preflight, "_check_tensorrt_dlls", return_value=True):
                        res = gpu_preflight.run_gpu_preflight(force_probe=True)
                        self.assertFalse(res["tensorrt_session_usable"])
                        self.assertEqual(res["failure_stage"], "tensorrt_session_construction_failure")
                        self.assertIn("ORT C++ engine allocation failed", res["failure_reason"])


if __name__ == "__main__":
    unittest.main()
