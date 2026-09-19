#!/usr/bin/env python3
"""Tests for app/verify_ort.py verification logic."""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
import importlib.metadata
from importlib.machinery import ModuleSpec
from pathlib import Path
from unittest.mock import patch, MagicMock

import verify_ort


class TestVerifyOrt(unittest.TestCase):
    @staticmethod
    def _healthy_ort():
        mock_ort = types.ModuleType("onnxruntime")
        distribution = importlib.metadata.distribution("onnxruntime-gpu")
        module_path = str(distribution.locate_file("onnxruntime/__init__.py"))
        mock_ort.__file__ = module_path
        mock_ort.__version__ = distribution.version
        mock_ort.__spec__ = ModuleSpec(
            "onnxruntime", loader=None, origin=module_path, is_package=True
        )
        mock_ort.__path__ = [os.path.dirname(module_path)]
        mock_ort.get_available_providers = lambda: [
            "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"
        ]
        return mock_ort

    def test_healthy_environment_verifies_successfully(self):
        """A healthy environment with callable get_available_providers passes without error."""
        mock_ort = self._healthy_ort()

        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            # Should not raise SystemExit
            verify_ort.verify_onnxruntime()

    def test_missing_get_available_providers_fails_loudly(self):
        """An onnxruntime module without get_available_providers must fail loudly with exit code 1."""
        broken_ort = self._healthy_ort()
        del broken_ort.get_available_providers
        # No get_available_providers attribute

        with patch.dict(sys.modules, {"onnxruntime": broken_ort}):
            with self.assertRaises(SystemExit) as cm:
                verify_ort.verify_onnxruntime()
            self.assertEqual(cm.exception.code, 1)

    def test_uncallable_get_available_providers_fails_loudly(self):
        """A non-callable get_available_providers attribute must fail loudly with exit code 1."""
        broken_ort = self._healthy_ort()
        broken_ort.get_available_providers = "not_a_function"

        with patch.dict(sys.modules, {"onnxruntime": broken_ort}):
            with self.assertRaises(SystemExit) as cm:
                verify_ort.verify_onnxruntime()
            self.assertEqual(cm.exception.code, 1)

    def test_namespace_package_no_file_fails_loudly(self):
        """An unpopulated namespace package (__file__ is None) must fail loudly with exit code 1."""
        namespace_ort = self._healthy_ort()
        namespace_ort.__file__ = None
        namespace_ort.__spec__ = ModuleSpec("onnxruntime", loader=None, origin=None, is_package=True)

        with patch.dict(sys.modules, {"onnxruntime": namespace_ort}):
            with self.assertRaises(SystemExit) as cm:
                verify_ort.verify_onnxruntime()
            self.assertEqual(cm.exception.code, 1)

    def test_sys_path_package_shadowing_detected(self):
        """A second normal onnxruntime package root is also a shadow artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "onnxruntime"), exist_ok=True)
            Path(os.path.join(tmpdir, "onnxruntime", "__init__.py")).write_text(
                "__version__ = 'shadow'", encoding="utf-8"
            )
            with self.assertRaises(SystemExit) as cm:
                verify_ort.detect_shadowing(tmpdir)
            self.assertEqual(cm.exception.code, 1)

    def test_namespace_directory_without_init_detected(self):
        """A package directory without __init__.py must be rejected explicitly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "onnxruntime"), exist_ok=True)
            with self.assertRaises(SystemExit) as cm:
                verify_ort.detect_shadowing(tmpdir)
            self.assertEqual(cm.exception.code, 1)

    def test_stale_bytecode_shadowing_detected(self):
        """A stale onnxruntime bytecode artifact must not be accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pycache = os.path.join(tmpdir, "__pycache__")
            os.makedirs(pycache, exist_ok=True)
            Path(os.path.join(pycache, "onnxruntime.cpython-310.pyc")).write_bytes(b"stale")
            with self.assertRaises(SystemExit) as cm:
                verify_ort.detect_shadowing(tmpdir)
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
