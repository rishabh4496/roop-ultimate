#!/usr/bin/env python3
"""Windows NVIDIA binary/runtime compatibility verification.

This module is deliberately independent from the model stack.  It inspects the
native files that the active Python environment is expected to use, registers
only environment-owned DLL directories with ``os.add_dll_directory``, and
reports the selected path, PE architecture, load result, and contamination
signals for every critical CUDA, cuDNN, TensorRT, PyTorch, and ORT component.

It never edits the process PATH.  PATH is an input to the diagnostic report,
not a repair mechanism.
"""
from __future__ import annotations

import ctypes
import glob
import importlib
import importlib.metadata
import os
import platform
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


TENSORRT_EP = "TensorrtExecutionProvider"
CUDA_EP = "CUDAExecutionProvider"

_MACHINE_NAMES = {
    0x014C: "x86",
    0x8664: "x64",
    0xAA64: "arm64",
}

_COMPONENTS = {
    "onnxruntime_provider_cuda": ("onnxruntime_providers_cuda.dll",),
    "onnxruntime_provider_tensorrt": ("onnxruntime_providers_tensorrt.dll",),
    "onnxruntime_provider_shared": ("onnxruntime_providers_shared.dll",),
    "tensorrt_core": ("nvinfer.dll", "nvinfer_[0-9]*.dll"),
    "tensorrt_plugin": ("nvinfer_plugin.dll", "nvinfer_plugin_[0-9]*.dll"),
    "tensorrt_parser": ("nvonnxparser.dll", "nvonnxparser_[0-9]*.dll"),
    "cuda_runtime": ("cudart64_*.dll",),
    "cuda_cublas": ("cublas64_*.dll",),
    "cuda_cublaslt": ("cublasLt64_*.dll",),
    "cudnn": ("cudnn*.dll",),
    "torch_cuda": ("torch_cuda*.dll", "c10_cuda.dll"),
}

_PROCESS_DLL_HANDLES: list[Any] = []


def _normalise_path(value: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(value))))


def _normalise_architecture(value: str | None) -> str:
    text = (value or "").strip().lower().replace("-", "_")
    if text in {"amd64", "x86_64", "x64", "64bit"}:
        return "x64"
    if text in {"x86", "i386", "i686", "32bit"}:
        return "x86"
    if text in {"arm64", "aarch64"}:
        return "arm64"
    return text or "unknown"


def _under(path: str, roots: Iterable[str]) -> bool:
    candidate = _normalise_path(path)
    for root in roots:
        try:
            if os.path.commonpath((candidate, _normalise_path(root))) == _normalise_path(root):
                return True
        except ValueError:
            continue
    return False


def _active_roots() -> list[str]:
    roots = [sys.prefix, os.path.dirname(sys.executable)]
    for entry in sys.path:
        # A foreign interpreter's site-packages can appear on sys.path after a
        # broken activation.  It is evidence of contamination, not an owned
        # runtime directory, so only retain entries under the active prefix.
        if entry and os.path.isdir(entry) and _under(entry, [sys.prefix]):
            roots.append(entry)
    result: list[str] = []
    seen: set[str] = set()
    for root in roots:
        key = _normalise_path(root)
        if key not in seen:
            result.append(os.path.abspath(root))
            seen.add(key)
    return result


def _add_dir(directories: list[str], directory: str | None) -> None:
    if not directory or not os.path.isdir(directory):
        return
    value = os.path.abspath(directory)
    key = _normalise_path(value)
    if not any(_normalise_path(item) == key for item in directories):
        directories.append(value)


def _module_file(name: str) -> str | None:
    try:
        module = sys.modules.get(name) or importlib.import_module(name)
    except Exception:
        return None
    value = getattr(module, "__file__", None)
    return os.path.abspath(value) if value else None


def _package_dirs() -> tuple[list[str], dict[str, str | None]]:
    """Return preferred active-environment DLL directories and module paths."""
    directories: list[str] = []
    modules = {
        "onnxruntime": _module_file("onnxruntime"),
        "torch": _module_file("torch"),
        "tensorrt": _module_file("tensorrt"),
        "tensorrt_libs": _module_file("tensorrt_libs"),
    }

    ort = modules["onnxruntime"]
    if ort:
        _add_dir(directories, os.path.dirname(ort))
        _add_dir(directories, os.path.join(os.path.dirname(ort), "capi"))
    torch = modules["torch"]
    if torch:
        _add_dir(directories, os.path.join(os.path.dirname(torch), "lib"))
    trt = modules["tensorrt"]
    if trt:
        _add_dir(directories, os.path.dirname(trt))
        _add_dir(directories, os.path.join(os.path.dirname(os.path.dirname(trt)), "tensorrt_libs"))
    trt_libs = modules["tensorrt_libs"]
    if trt_libs:
        _add_dir(directories, os.path.dirname(trt_libs))

    for site in list(sys.path):
        if not site or not os.path.isdir(site):
            continue
        _add_dir(directories, os.path.join(site, "tensorrt"))
        _add_dir(directories, os.path.join(site, "tensorrt_libs"))
        nvidia_root = os.path.join(site, "nvidia")
        if os.path.isdir(nvidia_root):
            for current, _dirs, _files in os.walk(nvidia_root):
                if os.path.basename(current).lower() in {"bin", "lib"}:
                    _add_dir(directories, current)

    # Explicit environment variables are reported and searched after active
    # environment directories, never promoted ahead of them.
    for variable in ("CUDA_PATH", "CUDNN_PATH", "TENSORRT_PATH", "TRT_PATH"):
        value = os.environ.get(variable)
        if value:
            _add_dir(directories, value)
            _add_dir(directories, os.path.join(value, "bin"))
            _add_dir(directories, os.path.join(value, "lib"))
            _add_dir(directories, os.path.join(value, "lib64"))
    return directories, modules


def process_local_dll_directories(directories: Iterable[str]) -> list[str]:
    """Register active-environment directories without changing PATH."""
    registered: list[str] = []
    owned = _active_roots()
    for directory in directories:
        if not os.path.isdir(directory) or not _under(directory, owned):
            continue
        value = os.path.abspath(directory)
        if value in registered:
            continue
        if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
            try:
                handle = os.add_dll_directory(value)
                _PROCESS_DLL_HANDLES.append(handle)
            except (OSError, AttributeError) as exc:
                # The verifier records this through the caller.  A failed
                # registration must not be hidden as a successful load.
                raise RuntimeError(f"could not register DLL directory {value}: {exc}") from exc
        registered.append(value)
    return registered


def _path_directories() -> list[str]:
    return [os.path.abspath(item) for item in os.environ.get("PATH", "").split(os.pathsep) if item and os.path.isdir(item)]


def _pe_architecture(path: str) -> str:
    try:
        with open(path, "rb") as handle:
            if handle.read(2) != b"MZ":
                return "not_pe"
            handle.seek(0x3C)
            offset_data = handle.read(4)
            if len(offset_data) != 4:
                return "invalid_pe"
            handle.seek(struct.unpack("<I", offset_data)[0])
            if handle.read(4) != b"PE\0\0":
                return "invalid_pe"
            machine_data = handle.read(2)
            if len(machine_data) != 2:
                return "invalid_pe"
            return _MACHINE_NAMES.get(struct.unpack("<H", machine_data)[0], "unknown")
    except OSError as exc:
        return f"unreadable:{exc}"


def _load_native(path: str) -> tuple[bool, str | None]:
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        return True, "WinDLL unavailable on this host"
    try:
        loader(path)
        return True, None
    except OSError as exc:
        return False, str(exc)


def _nvidia_smi() -> dict[str, str]:
    executable = shutil.which("nvidia-smi")
    if not executable and os.name == "nt":
        for root in (os.environ.get("ProgramW6432"), os.environ.get("ProgramFiles")):
            if root:
                candidate = os.path.join(root, "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")
                if os.path.isfile(candidate):
                    executable = candidate
                    break
    result = {"path": executable or "none", "driver": "none", "gpu": "none", "vram_mb": "0"}
    if not executable:
        return result
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        row = [item.strip() for item in completed.stdout.splitlines()[0].split(",")] if completed.stdout.strip() else []
        if row:
            result["gpu"] = row[0]
        if len(row) > 1:
            result["driver"] = row[1]
        if len(row) > 2:
            result["vram_mb"] = row[2]
    except (OSError, IndexError, subprocess.SubprocessError):
        pass
    return result


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"
    except Exception as exc:
        return f"error: {exc}"


def _find_candidates(patterns: Iterable[str], directories: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for directory in directories:
        for pattern in patterns:
            for path in sorted(glob.glob(os.path.join(directory, pattern))):
                key = _normalise_path(path)
                if key not in seen and os.path.isfile(path):
                    result.append(os.path.abspath(path))
                    seen.add(key)
    return result


def _artifact(path: str | None, expected_arch: str, owned_roots: list[str], required: bool, *, load: bool = True) -> dict[str, Any]:
    if not path:
        return {
            "path": None, "exists": False, "required": required, "owned_by_environment": False,
            "architecture": "missing", "architecture_ok": False, "loadable": False,
            "load_error": "not found",
        }
    arch = _pe_architecture(path)
    loadable, load_error = _load_native(path) if load else (False, "not loaded because the candidate is outside the active environment")
    return {
        "path": path,
        "exists": True,
        "required": required,
        "owned_by_environment": _under(path, owned_roots),
        "architecture": arch,
        "architecture_ok": arch == expected_arch,
        "loadable": loadable,
        "load_error": load_error,
    }


def _component_report(
    name: str,
    patterns: tuple[str, ...],
    directories: list[str],
    path_directories: list[str],
    owned_roots: list[str],
    expected_arch: str,
    required: bool,
) -> dict[str, Any]:
    candidates = _find_candidates(patterns, directories)
    selected = candidates[0] if candidates else None
    details = _artifact(
        selected,
        expected_arch,
        owned_roots,
        required,
        load=bool(selected and _under(selected, owned_roots)),
    )
    candidate_details = [
        _artifact(path, expected_arch, owned_roots, required, load=_under(path, owned_roots))
        for path in candidates
    ]
    basename_map: dict[str, list[str]] = {}
    for candidate in candidates:
        basename_map.setdefault(os.path.basename(candidate).lower(), []).append(candidate)
    duplicate_files = {key: value for key, value in basename_map.items() if len(value) > 1}
    versioned_duplicates = {}
    if name in {"tensorrt_core", "tensorrt_plugin", "tensorrt_parser", "cuda_runtime", "cuda_cublas", "cuda_cublaslt"}:
        if len({os.path.basename(path).lower() for path in candidates}) > 1:
            versioned_duplicates = {"candidates": candidates}
    foreign_candidates = [path for path in candidates if not _under(path, owned_roots)]
    path_candidates = [path for path in candidates if any(_normalise_path(os.path.dirname(path)) == _normalise_path(directory) for directory in path_directories)]
    return {
        "required": required,
        "selected": details,
        "candidate_details": candidate_details,
        "candidates": candidates,
        "foreign_candidates": foreign_candidates,
        "path_candidates": path_candidates,
        "duplicate_files": duplicate_files,
        "versioned_duplicates": versioned_duplicates,
    }


def verify_windows_nvidia_runtime(*, hardware: Any | None = None, require_tensorrt: bool | None = None) -> dict[str, Any]:
    """Return a fail-closed compatibility report for Windows NVIDIA systems."""
    vendor = str(getattr(hardware, "vendor", "nvidia") if hardware is not None else "nvidia").lower()
    system = str(getattr(hardware, "system", platform.system()))
    vram = [int(value) for value in (getattr(hardware, "vram_mb", ()) or ()) if int(value) > 0]
    if require_tensorrt is None:
        require_tensorrt = vendor == "nvidia" and max(vram, default=0) >= 7 * 1024

    base: dict[str, Any] = {
        "applicable": system == "Windows" and vendor == "nvidia",
        "status": "not_applicable",
        "failure_reasons": [],
        "warnings": [],
        "python_architecture": _normalise_architecture(platform.machine()),
        "expected_native_architecture": _normalise_architecture(platform.machine()),
        "torch_wheel_architecture": "not_checked",
        "onnxruntime_wheel_architecture": "not_checked",
        "tensorrt_python_bindings_architecture": "not_checked",
        "tensorrt_native_dll_architecture": "not_checked",
        "cuda_runtime_dlls": {},
        "cudnn_dlls": {},
        "onnxruntime_tensorrt_provider_dll": {},
        "selected_dll_paths": {},
        "dll_directories": [],
        "path_entries": _path_directories(),
        "path_contamination": [],
        "duplicate_dll_versions": {},
        "nvidia_smi": _nvidia_smi(),
        "environment_roots": _active_roots(),
        "environment_consistent": True,
    }
    if not base["applicable"]:
        return base

    expected_arch = base["expected_native_architecture"]
    if expected_arch != "x64":
        base["failure_reasons"].append(f"unsupported Windows NVIDIA Python architecture: {expected_arch}")

    preferred_dirs, modules = _package_dirs()
    path_dirs = base["path_entries"]
    all_dirs = list(preferred_dirs)
    for directory in path_dirs:
        _add_dir(all_dirs, directory)
    base["dll_directories"] = preferred_dirs
    owned_roots = base["environment_roots"]
    try:
        base["registered_dll_directories"] = process_local_dll_directories(preferred_dirs)
    except RuntimeError as exc:
        base["failure_reasons"].append(str(exc))
        base["registered_dll_directories"] = []

    torch_file = modules.get("torch")
    ort_file = modules.get("onnxruntime")
    trt_file = modules.get("tensorrt")
    torch_native = _find_candidates(
        ("torch*.dll", "c10*.dll"),
        [os.path.join(os.path.dirname(torch_file), "lib"), os.path.dirname(torch_file)] if torch_file else [],
    )
    ort_native = _find_candidates(("onnxruntime*.dll",), [os.path.join(os.path.dirname(ort_file), "capi"),] if ort_file else [])
    trt_bindings = _find_candidates(
        ("*.pyd", "nvinfer*.dll", "nvonnxparser*.dll"),
        [os.path.dirname(trt_file),] if trt_file else [],
    )
    base["module_paths"] = {"torch": torch_file, "onnxruntime": ort_file, "tensorrt": trt_file}
    base["torch_wheel_architecture"] = _normalise_architecture(_pe_architecture(torch_native[0])) if torch_native else "missing"
    base["onnxruntime_wheel_architecture"] = _normalise_architecture(_pe_architecture(ort_native[0])) if ort_native else "missing"
    if trt_bindings:
        base["tensorrt_python_bindings_architecture"] = _normalise_architecture(_pe_architecture(trt_bindings[0]))
    elif trt_file:
        # Current NVIDIA TensorRT wheels expose a Python wrapper and keep the
        # binding implementation in tensorrt-cu12-libs.  The native TensorRT
        # DLL architecture is therefore the binding architecture in this
        # layout, even though no .pyd is present beside __init__.py.
        binding_native = _find_candidates(
            ("nvinfer_[0-9]*.dll", "nvinfer_plugin_[0-9]*.dll", "nvonnxparser_[0-9]*.dll"),
            [os.path.join(os.path.dirname(trt_file), "..", "tensorrt_libs")],
        )
        base["tensorrt_python_bindings_architecture"] = (
            _normalise_architecture(_pe_architecture(binding_native[0])) if binding_native else "missing"
        )
    else:
        base["tensorrt_python_bindings_architecture"] = "missing"
    if base["torch_wheel_architecture"] not in {expected_arch, "missing"}:
        base["failure_reasons"].append("PyTorch native wheel architecture does not match the Python process")
    if base["onnxruntime_wheel_architecture"] not in {expected_arch, "missing"}:
        base["failure_reasons"].append("ONNX Runtime native wheel architecture does not match the Python process")
    if base["tensorrt_python_bindings_architecture"] not in {expected_arch, "missing"}:
        base["failure_reasons"].append("TensorRT Python bindings architecture does not match the Python process")

    specs = {
        name: _component_report(
            name, patterns, all_dirs, path_dirs, owned_roots, expected_arch,
            required=(name in {"onnxruntime_provider_cuda", "onnxruntime_provider_shared", "cuda_runtime", "cuda_cublas", "cuda_cublaslt", "cudnn", "torch_cuda"}
                      or require_tensorrt and name in {"onnxruntime_provider_tensorrt", "tensorrt_core", "tensorrt_plugin", "tensorrt_parser"}),
        )
        for name, patterns in _COMPONENTS.items()
    }
    base["components"] = specs
    base["cuda_runtime_dlls"] = {key: specs[key] for key in ("cuda_runtime", "cuda_cublas", "cuda_cublaslt")}
    base["cudnn_dlls"] = specs["cudnn"]
    base["onnxruntime_tensorrt_provider_dll"] = specs["onnxruntime_provider_tensorrt"]
    base["selected_dll_paths"] = {
        key: value["selected"]["path"] for key, value in specs.items() if value["selected"]["path"]
    }

    for name, report in specs.items():
        selected = report["selected"]
        if report["required"] and not selected["exists"]:
            base["failure_reasons"].append(f"required native component missing: {name}")
        if selected["exists"] and not selected["owned_by_environment"]:
            base["failure_reasons"].append(f"selected native component is outside the active environment: {name}={selected['path']}")
        if selected["exists"] and not selected["architecture_ok"]:
            base["failure_reasons"].append(f"native architecture mismatch: {name}={selected['architecture']}, expected {expected_arch}")
        if selected["exists"] and not selected["loadable"]:
            base["failure_reasons"].append(f"native DLL load failure: {name}: {selected['load_error']}")
        for candidate in report["candidate_details"]:
            if candidate["owned_by_environment"] and candidate["required"] and not candidate["architecture_ok"]:
                base["failure_reasons"].append(
                    f"native architecture mismatch: {name} candidate={candidate['path']} "
                    f"architecture={candidate['architecture']}, expected {expected_arch}"
                )
            if candidate["owned_by_environment"] and candidate["required"] and not candidate["loadable"]:
                base["failure_reasons"].append(
                    f"native DLL load failure: {name} candidate={candidate['path']}: {candidate['load_error']}"
                )
        if report["foreign_candidates"]:
            base["path_contamination"].append({"component": name, "paths": report["foreign_candidates"]})
        if report["duplicate_files"]:
            base["duplicate_dll_versions"][name] = report["duplicate_files"]
        if report["versioned_duplicates"]:
            base["duplicate_dll_versions"][name] = report["versioned_duplicates"]

    base["tensorrt_native_dll_architecture"] = _normalise_architecture(
        specs["tensorrt_core"]["selected"]["architecture"]
    ) if specs["tensorrt_core"]["selected"]["exists"] else "missing"
    base["cuda_version"] = "unknown"
    try:
        torch = sys.modules.get("torch") or importlib.import_module("torch")
        base["cuda_version"] = str(getattr(getattr(torch, "version", None), "cuda", None) or "unknown")
        if base["torch_wheel_architecture"] == "missing" and bool(torch.cuda.is_available()):
            base["failure_reasons"].append("PyTorch CUDA is available but no native torch CUDA DLL was found")
    except Exception as exc:
        base["warnings"].append(f"could not inspect torch CUDA metadata: {exc}")
    base["onnxruntime_version"] = _distribution_version("onnxruntime-gpu")
    base["tensorrt_version"] = _distribution_version("tensorrt-cu12")

    if base["duplicate_dll_versions"]:
        base["warnings"].append("duplicate native DLL basenames were found across search roots")
    if base["path_contamination"]:
        base["warnings"].append("PATH contains foreign native runtime candidates; process-local directories were preferred")
    base["environment_consistent"] = not bool(base["failure_reasons"])
    if base["failure_reasons"]:
        base["status"] = "failed"
    elif base["warnings"]:
        base["status"] = "degraded"
    else:
        base["status"] = "passed"
    return base


if __name__ == "__main__":  # pragma: no cover
    import json
    print(json.dumps(verify_windows_nvidia_runtime(), indent=2, sort_keys=True))
