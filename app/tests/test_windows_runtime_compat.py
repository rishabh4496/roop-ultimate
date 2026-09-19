"""Tests for the Windows NVIDIA binary/runtime compatibility verifier."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import windows_runtime_compat as compat


def _write_pe(path: Path, machine: int = 0x8664) -> None:
    data = bytearray(128)
    data[0:2] = b"MZ"
    data[0x3C:0x40] = (64).to_bytes(4, "little")
    data[64:68] = b"PE\0\0"
    data[68:70] = machine.to_bytes(2, "little")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class WindowsRuntimeCompatibilityTests(unittest.TestCase):
    def _fixture(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        ort = root / "site-packages" / "onnxruntime"
        capi = ort / "capi"
        torch = root / "site-packages" / "torch"
        torch_lib = torch / "lib"
        trt = root / "site-packages" / "tensorrt"
        trt_libs = root / "site-packages" / "tensorrt_libs"
        cuda = root / "site-packages" / "nvidia" / "cuda_runtime" / "bin"
        cudnn = root / "site-packages" / "nvidia" / "cudnn" / "bin"
        for directory in (capi, torch_lib, trt, trt_libs, cuda, cudnn):
            directory.mkdir(parents=True, exist_ok=True)

        files = {
            capi / "onnxruntime.dll": 0x8664,
            capi / "onnxruntime_providers_cuda.dll": 0x8664,
            capi / "onnxruntime_providers_tensorrt.dll": 0x8664,
            capi / "onnxruntime_providers_shared.dll": 0x8664,
            torch_lib / "torch_cuda.dll": 0x8664,
            torch_lib / "c10_cuda.dll": 0x8664,
            trt / "tensorrt_bindings.pyd": 0x8664,
            trt_libs / "nvinfer_10.dll": 0x8664,
            trt_libs / "nvinfer_plugin_10.dll": 0x8664,
            trt_libs / "nvonnxparser_10.dll": 0x8664,
            cuda / "cudart64_12.dll": 0x8664,
            cuda / "cublas64_12.dll": 0x8664,
            cuda / "cublasLt64_12.dll": 0x8664,
            cudnn / "cudnn64_9.dll": 0x8664,
        }
        for path, machine in files.items():
            _write_pe(path, machine)
        modules = {
            "onnxruntime": str(ort / "__init__.py"),
            "torch": str(torch / "__init__.py"),
            "tensorrt": str(trt / "__init__.py"),
            "tensorrt_libs": str(trt_libs / "__init__.py"),
        }
        hardware = SimpleNamespace(
            system="Windows", vendor="nvidia", vram_mb=(12288,),
            gpu_names=("RTX test",), architecture="AMD64",
        )
        return tmp, root, (capi, torch_lib, trt, trt_libs, cuda, cudnn), modules, hardware

    @staticmethod
    def _fake_torch():
        return SimpleNamespace(
            version=SimpleNamespace(cuda="12.8"),
            cuda=SimpleNamespace(is_available=lambda: True),
        )

    def test_non_nvidia_is_not_applicable(self):
        hardware = SimpleNamespace(system="Windows", vendor="cpu", vram_mb=())
        report = compat.verify_windows_nvidia_runtime(hardware=hardware)
        self.assertEqual(report["status"], "not_applicable")
        self.assertFalse(report["applicable"])

    def test_amd_is_not_applicable(self):
        hardware = SimpleNamespace(system="Windows", vendor="amd", vram_mb=(8192,))
        report = compat.verify_windows_nvidia_runtime(hardware=hardware)
        self.assertEqual(report["status"], "not_applicable")
        self.assertFalse(report["applicable"])

    def test_owned_x64_components_pass_and_report_selected_paths(self):
        tmp, root, dirs, modules, hardware = self._fixture()
        self.addCleanup(tmp.cleanup)
        with patch.object(compat, "_active_roots", return_value=[str(root)]), \
             patch.object(compat, "_package_dirs", return_value=([str(item) for item in dirs], modules)), \
             patch.object(compat, "_path_directories", return_value=[]), \
             patch.object(compat, "_nvidia_smi", return_value={"path": "smi", "driver": "x", "gpu": "RTX", "vram_mb": "12288"}), \
             patch.object(compat, "_module_file", side_effect=lambda name: modules.get(name)), \
             patch.object(compat, "_load_native", return_value=(True, None)), \
             patch.object(compat.sys, "platform", "win32"), \
             patch.object(compat.os, "add_dll_directory", side_effect=lambda value: object()), \
             patch.dict(compat.sys.modules, {"torch": self._fake_torch()}), \
             patch.object(compat.importlib.metadata, "version", side_effect=lambda name: "1.23.2" if name == "onnxruntime-gpu" else "10.9.0.34"):
            report = compat.verify_windows_nvidia_runtime(hardware=hardware, require_tensorrt=True)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["python_architecture"], compat._normalise_architecture(compat.platform.machine()))
        self.assertEqual(report["torch_wheel_architecture"], "x64")
        self.assertEqual(report["onnxruntime_wheel_architecture"], "x64")
        self.assertEqual(report["tensorrt_python_bindings_architecture"], "x64")
        self.assertTrue(report["selected_dll_paths"]["onnxruntime_provider_tensorrt"].endswith("onnxruntime_providers_tensorrt.dll"))
        self.assertTrue(report["environment_consistent"])

    def test_x86_native_component_fails_closed(self):
        tmp, root, dirs, modules, hardware = self._fixture()
        self.addCleanup(tmp.cleanup)
        _write_pe(dirs[0] / "onnxruntime_providers_tensorrt.dll", 0x014C)
        with patch.object(compat, "_active_roots", return_value=[str(root)]), \
             patch.object(compat, "_package_dirs", return_value=([str(item) for item in dirs], modules)), \
             patch.object(compat, "_path_directories", return_value=[]), \
             patch.object(compat, "_nvidia_smi", return_value={"path": "smi", "driver": "x", "gpu": "RTX", "vram_mb": "12288"}), \
             patch.object(compat, "_module_file", side_effect=lambda name: modules.get(name)), \
             patch.object(compat, "_load_native", return_value=(True, None)), \
             patch.object(compat.sys, "platform", "win32"), \
             patch.object(compat.os, "add_dll_directory", side_effect=lambda value: object()), \
             patch.dict(compat.sys.modules, {"torch": self._fake_torch()}), \
             patch.object(compat.importlib.metadata, "version", return_value="1"):
            report = compat.verify_windows_nvidia_runtime(hardware=hardware, require_tensorrt=True)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("architecture mismatch" in reason for reason in report["failure_reasons"]))

    def test_duplicate_cuda_versions_are_reported_without_path_mutation(self):
        tmp, root, dirs, modules, hardware = self._fixture()
        self.addCleanup(tmp.cleanup)
        _write_pe(dirs[4] / "cudart64_11.dll", 0x8664)
        original_path = os.environ.get("PATH", "")
        with patch.object(compat, "_active_roots", return_value=[str(root)]), \
             patch.object(compat, "_package_dirs", return_value=([str(item) for item in dirs], modules)), \
             patch.object(compat, "_path_directories", return_value=[]), \
             patch.object(compat, "_nvidia_smi", return_value={"path": "smi", "driver": "x", "gpu": "RTX", "vram_mb": "12288"}), \
             patch.object(compat, "_module_file", side_effect=lambda name: modules.get(name)), \
             patch.object(compat, "_load_native", return_value=(True, None)), \
             patch.object(compat.sys, "platform", "win32"), \
             patch.object(compat.os, "add_dll_directory", side_effect=lambda value: object()), \
             patch.dict(compat.sys.modules, {"torch": self._fake_torch()}), \
             patch.object(compat.importlib.metadata, "version", return_value="1"):
            report = compat.verify_windows_nvidia_runtime(hardware=hardware, require_tensorrt=True)
        self.assertEqual(report["status"], "degraded")
        self.assertIn("cuda_runtime", report["duplicate_dll_versions"])
        self.assertEqual(os.environ.get("PATH", ""), original_path)

    def test_foreign_path_candidate_is_detected_and_cannot_be_selected_as_valid(self):
        tmp, root, dirs, modules, hardware = self._fixture()
        self.addCleanup(tmp.cleanup)
        foreign = Path(tmp.name) / "system-cuda"
        _write_pe(foreign / "cudart64_12.dll", 0x8664)
        owned_cuda = dirs[4] / "cudart64_12.dll"
        owned_cuda.unlink()
        with patch.object(compat, "_active_roots", return_value=[str(root / "site-packages")]), \
             patch.object(compat, "_package_dirs", return_value=([str(item) for item in dirs] + [str(foreign)], modules)), \
             patch.object(compat, "_path_directories", return_value=[str(foreign)]), \
             patch.object(compat, "_nvidia_smi", return_value={"path": "smi", "driver": "x", "gpu": "RTX", "vram_mb": "12288"}), \
             patch.object(compat, "_module_file", side_effect=lambda name: modules.get(name)), \
             patch.object(compat, "_load_native", return_value=(True, None)), \
             patch.object(compat.sys, "platform", "win32"), \
             patch.object(compat.os, "add_dll_directory", side_effect=lambda value: object()), \
             patch.dict(compat.sys.modules, {"torch": self._fake_torch()}), \
             patch.object(compat.importlib.metadata, "version", return_value="1"):
            report = compat.verify_windows_nvidia_runtime(hardware=hardware, require_tensorrt=True)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("outside the active environment" in reason for reason in report["failure_reasons"]))
        self.assertTrue(report["path_contamination"])


if __name__ == "__main__":
    unittest.main()
