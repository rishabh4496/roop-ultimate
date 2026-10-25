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


def _ort_gpu_installed() -> bool:
    try:
        importlib.metadata.distribution("onnxruntime-gpu")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


# `_healthy_ort` copies the REAL onnxruntime-gpu distribution's version and file layout into a stand-in module, so the
# four tests that use it cannot run where that package is absent (the CI light profile installs no GPU stack). They
# are skipped there, not weakened: a skip reads as "unverified" in the summary, which is what it is.
needs_ort_gpu_dist = unittest.skipUnless(
    _ort_gpu_installed(), "onnxruntime-gpu is not installed here; these tests copy its real distribution metadata")


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

    @needs_ort_gpu_dist
    def test_healthy_environment_verifies_successfully(self):
        """A healthy environment with callable get_available_providers passes without error."""
        mock_ort = self._healthy_ort()

        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            # Should not raise SystemExit
            verify_ort.verify_onnxruntime()

    @needs_ort_gpu_dist
    def test_missing_get_available_providers_fails_loudly(self):
        """An onnxruntime module without get_available_providers must fail loudly with exit code 1."""
        broken_ort = self._healthy_ort()
        del broken_ort.get_available_providers
        # No get_available_providers attribute

        with patch.dict(sys.modules, {"onnxruntime": broken_ort}):
            with self.assertRaises(SystemExit) as cm:
                verify_ort.verify_onnxruntime()
            self.assertEqual(cm.exception.code, 1)

    @needs_ort_gpu_dist
    def test_uncallable_get_available_providers_fails_loudly(self):
        """A non-callable get_available_providers attribute must fail loudly with exit code 1."""
        broken_ort = self._healthy_ort()
        broken_ort.get_available_providers = "not_a_function"

        with patch.dict(sys.modules, {"onnxruntime": broken_ort}):
            with self.assertRaises(SystemExit) as cm:
                verify_ort.verify_onnxruntime()
            self.assertEqual(cm.exception.code, 1)

    @needs_ort_gpu_dist
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

    def test_install_extras_missing_sam2_fails_loudly(self):
        """A failed `--no-deps` sam2 install must not reach the install marker."""
        with patch.dict(sys.modules, {"sam2": None, "onnxconverter_common": types.ModuleType("x")}):
            with self.assertRaises(SystemExit) as cm:
                verify_ort.verify_install_extras()
            self.assertEqual(cm.exception.code, 1)

    def test_install_extras_pass_when_importable(self):
        fake = {name: types.ModuleType(name) for name in ("sam2", "onnxconverter_common")}
        with patch.dict(sys.modules, fake):
            verify_ort.verify_install_extras()

    def test_extras_are_opt_in_for_verify_environment(self):
        """Start-up verification (--require-complete) must not start requiring SAM 2."""
        import inspect
        params = inspect.signature(verify_ort.verify_environment).parameters
        self.assertIs(params["require_extras"].default, False)


class TestActiveProviderGate(unittest.TestCase):
    """The preflight session must have run on the GPU on an NVIDIA machine.

    A CUDA/TensorRT session that drops to CPU raises nothing, and the package
    checks still pass; `gpu_preflight` then reports CPUExecutionProvider. Only
    "none" used to be fatal, so a sub-7 GB card could be committed as verified on
    the CPU."""

    nvidia = types.SimpleNamespace(vendor="nvidia")
    other = types.SimpleNamespace(vendor="apple")

    def test_cpu_on_an_nvidia_machine_is_fatal_and_says_why(self):
        pre = {"active_provider": "CPUExecutionProvider", "failure_stage": "dll_load_failure",
               "failure_reason": "cudnn64_9.dll not found"}
        with self.assertRaises(SystemExit) as cm:
            verify_ort.verify_active_provider(self.nvidia, pre)
        self.assertEqual(cm.exception.code, 1)

    def test_the_message_names_the_provider_stage_and_reason(self):
        pre = {"active_provider": "CPUExecutionProvider", "failure_stage": "dll_load_failure",
               "failure_reason": "cudnn64_9.dll not found"}
        import contextlib
        import io
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            verify_ort.verify_active_provider(self.nvidia, pre)
        text = err.getvalue()
        for needle in ("CPUExecutionProvider", "dll_load_failure", "cudnn64_9.dll not found"):
            self.assertIn(needle, text)

    def test_gpu_providers_pass_and_are_returned(self):
        for name in ("TensorrtExecutionProvider", "CUDAExecutionProvider"):
            self.assertEqual(verify_ort.verify_active_provider(self.nvidia, {"active_provider": name}), name)

    def test_cuda_after_a_tensorrt_failure_still_passes(self):
        """A sub-7 GB card legitimately runs CUDA only; the TensorRT stage failing must not fail the install here."""
        pre = {"active_provider": "CUDAExecutionProvider", "failure_stage": "tensorrt_initialization_failure"}
        self.assertEqual(verify_ort.verify_active_provider(self.nvidia, pre), "CUDAExecutionProvider")

    def test_none_is_fatal_on_every_vendor(self):
        for hw in (self.nvidia, self.other):
            for pre in ({"active_provider": "none"}, {}):
                with self.assertRaises(SystemExit):
                    verify_ort.verify_active_provider(hw, pre)

    def test_cpu_is_accepted_where_there_is_no_nvidia_gpu(self):
        self.assertEqual(
            verify_ort.verify_active_provider(self.other, {"active_provider": "CPUExecutionProvider"}),
            "CPUExecutionProvider")

    def test_verify_environment_actually_calls_the_gate(self):
        """Defined-and-never-called is how this project's checks keep dying."""
        import inspect
        self.assertIn("verify_active_provider(hardware, preflight)", inspect.getsource(verify_ort.verify_environment))


if __name__ == "__main__":
    unittest.main()
