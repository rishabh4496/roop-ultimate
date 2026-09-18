#!/usr/bin/env python3
"""Tests for app/verify_ort.py verification logic."""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import verify_ort


class TestVerifyOrt(unittest.TestCase):
    def test_healthy_environment_verifies_successfully(self):
        """A healthy environment with callable get_available_providers passes without error."""
        mock_ort = types.ModuleType("onnxruntime")
        mock_ort.__file__ = "/env/lib/site-packages/onnxruntime/__init__.py"
        mock_ort.__version__ = "1.23.2"
        mock_ort.get_available_providers = lambda: [
            "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"
        ]

        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            # Should not raise SystemExit
            verify_ort.verify_onnxruntime()

    def test_missing_get_available_providers_fails_loudly(self):
        """An onnxruntime module without get_available_providers must fail loudly with exit code 1."""
        broken_ort = types.ModuleType("onnxruntime")
        broken_ort.__file__ = "/env/lib/site-packages/onnxruntime/__init__.py"
        # No get_available_providers attribute

        with patch.dict(sys.modules, {"onnxruntime": broken_ort}):
            with self.assertRaises(SystemExit) as cm:
                verify_ort.verify_onnxruntime()
            self.assertEqual(cm.exception.code, 1)

    def test_uncallable_get_available_providers_fails_loudly(self):
        """A non-callable get_available_providers attribute must fail loudly with exit code 1."""
        broken_ort = types.ModuleType("onnxruntime")
        broken_ort.__file__ = "/env/lib/site-packages/onnxruntime/__init__.py"
        broken_ort.get_available_providers = "not_a_function"

        with patch.dict(sys.modules, {"onnxruntime": broken_ort}):
            with self.assertRaises(SystemExit) as cm:
                verify_ort.verify_onnxruntime()
            self.assertEqual(cm.exception.code, 1)

    def test_namespace_package_no_file_fails_loudly(self):
        """An unpopulated namespace package (__file__ is None) must fail loudly with exit code 1."""
        namespace_ort = types.ModuleType("onnxruntime")
        namespace_ort.__file__ = None

        with patch.dict(sys.modules, {"onnxruntime": namespace_ort}):
            with self.assertRaises(SystemExit) as cm:
                verify_ort.verify_onnxruntime()
            self.assertEqual(cm.exception.code, 1)

    def test_local_file_shadowing_detected(self):
        """A local onnxruntime.py file must be detected and fail loudly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shadow_file = os.path.join(tmpdir, "onnxruntime.py")
            Path(shadow_file).write_text("# shadow", encoding="utf-8")
            with self.assertRaises(SystemExit) as cm:
                verify_ort.detect_shadowing(tmpdir)
            self.assertEqual(cm.exception.code, 1)

    def test_local_directory_shadowing_detected(self):
        """A local onnxruntime directory outside site-packages must be detected and fail loudly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shadow_dir = os.path.join(tmpdir, "onnxruntime")
            os.makedirs(shadow_dir, exist_ok=True)
            with self.assertRaises(SystemExit) as cm:
                verify_ort.detect_shadowing(tmpdir)
            self.assertEqual(cm.exception.code, 1)

    def test_numpy_2_fails_loudly(self):
        """NumPy 2.x ABI incompatibility must fail loudly with exit code 1."""
        mock_np = types.ModuleType("numpy")
        mock_np.__version__ = "2.0.1"

        with patch.dict(sys.modules, {"numpy": mock_np}):
            with self.assertRaises(SystemExit) as cm:
                verify_ort.verify_numpy()
            self.assertEqual(cm.exception.code, 1)

    def test_numpy_1_26_4_passes(self):
        """NumPy 1.26.4 compatibility passes without error."""
        mock_np = types.ModuleType("numpy")
        mock_np.__version__ = "1.26.4"

        with patch.dict(sys.modules, {"numpy": mock_np}):
            verify_ort.verify_numpy()


if __name__ == "__main__":
    unittest.main()
