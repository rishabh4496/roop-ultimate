#!/usr/bin/env python3
"""Fail-closed verification of the provisioned Python runtime.

This verifier is intentionally executed from ``app/`` by Pinokio. It validates
the imported modules and their distributions, rather than treating a created
venv or a successful package-manager command as proof of installation.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
import subprocess
import sys
from typing import Any


ORT_DISTRIBUTIONS = {
    "onnxruntime",
    "onnxruntime-gpu",
    "onnxruntime-directml",
    "onnxruntime-rocm",
    "onnxruntime-silicon",
}


def _normalise(name: str) -> str:
    return name.replace("_", "-").replace(".", "-").lower()


def _fatal(message: str) -> None:
    print(f"[FATAL] {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def _distributions() -> dict[str, str]:
    try:
        result: dict[str, str] = {}
        for distribution in importlib.metadata.distributions():
            name = distribution.metadata.get("Name")
            if name:
                result[_normalise(name)] = distribution.version
        return result
    except Exception as exc:
        _fatal(f"Could not inspect installed Python distributions: {exc}")
    raise AssertionError("unreachable")


def detect_shadowing(app_dir: str) -> None:
    """Detect local files/directories and conflicting ORT distributions."""
    cwd = os.getcwd()
    check_paths = [cwd]
    if app_dir not in check_paths:
        check_paths.append(app_dir)

    for path in check_paths:
        py_shadow = os.path.join(path, "onnxruntime.py")
        if os.path.isfile(py_shadow):
            _fatal(f"Local file shadows onnxruntime package: {py_shadow}")

        dir_shadow = os.path.join(path, "onnxruntime")
        if os.path.isdir(dir_shadow) and "site-packages" not in os.path.normpath(dir_shadow).lower():
            _fatal(f"Local directory shadows onnxruntime package: {dir_shadow}")

    installed = _distributions()
    ort_packages = sorted(name for name in ORT_DISTRIBUTIONS if name in installed)
    if len(ort_packages) > 1:
        _fatal(
            "Conflicting ONNX Runtime distributions are installed: "
            + ", ".join(ort_packages)
            + ". Exactly one ORT distribution is allowed."
        )


def verify_numpy() -> None:
    """InsightFace native bindings require the pinned NumPy 1.x ABI."""
    try:
        import numpy as np
    except Exception as exc:
        _fatal(f"Failed to import NumPy: {exc}")
    if np.__version__ != "1.26.4":
        _fatal(f"Incompatible NumPy version {np.__version__}; required 1.26.4")
    print(f"[Verify Runtime] numpy      : {np.__version__}", flush=True)


def verify_onnxruntime() -> list[str]:
    """Import ORT and require a real provider API and non-empty result."""
    try:
        import onnxruntime as ort
    except Exception as exc:
        _fatal(f"Failed to import onnxruntime: {exc}")

    ort_file = getattr(ort, "__file__", None)
    if not ort_file:
        _fatal(
            "onnxruntime resolved to an unpopulated namespace package "
            "(__file__ is None). Stale or corrupted package artifacts detected."
        )
    version = getattr(ort, "__version__", "unknown")
    if not isinstance(version, str) or not version.strip() or version == "unknown":
        _fatal(f"imported onnxruntime ({ort_file}) does not expose a usable __version__")
    provider_api = getattr(ort, "get_available_providers", None)
    if not callable(provider_api):
        _fatal(
            f"imported onnxruntime ({ort_file}) does not expose callable "
            "get_available_providers"
        )
    try:
        providers = list(provider_api())
    except Exception as exc:
        _fatal(f"ort.get_available_providers() raised an exception: {exc}")
    if not providers:
        _fatal("onnxruntime reports an empty list of execution providers")

    print(f"[Verify Runtime] ort.__file__: {ort_file}", flush=True)
    print(f"[Verify Runtime] ort.version : {version}", flush=True)
    print(f"[Verify Runtime] ort.api     : callable", flush=True)
    print(f"[Verify Runtime] providers   : {providers}", flush=True)
    return providers


def _hardware() -> Any:
    try:
        from provision_runtime import detect_hardware
        return detect_hardware()
    except Exception as exc:
        _fatal(f"Could not inspect the machine GPU state: {exc}")
    raise AssertionError("unreachable")


def verify_distribution_contract(hardware: Any) -> None:
    installed = _distributions()
    actual = sorted(name for name in ORT_DISTRIBUTIONS if name in installed)
    if len(actual) != 1:
        _fatal(
            "Expected exactly one ONNX Runtime distribution, found "
            + (", ".join(actual) if actual else "none")
        )

    if hardware.vendor == "nvidia":
        expected = "onnxruntime-gpu"
    elif hardware.vendor == "amd" and hardware.system != "Darwin":
        expected = "onnxruntime-directml" if os.name == "nt" else "onnxruntime-rocm"
    elif hardware.system == "Darwin" and hardware.architecture == "arm64":
        expected = "onnxruntime-silicon"
    else:
        expected = "onnxruntime"
    if actual[0] != expected:
        _fatal(f"ONNX Runtime distribution is {actual[0]}, expected {expected} for {hardware.vendor}/{hardware.system}")
    print(f"[Verify Runtime] ort.dist    : {actual[0]}=={installed[actual[0]]}", flush=True)


def _nvidia_driver(hardware: Any) -> str:
    executable = getattr(hardware, "nvidia_smi", None) or shutil.which("nvidia-smi")
    if not executable:
        _fatal("NVIDIA hardware was detected but nvidia-smi is not available")
    result = subprocess.run(
        [executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        _fatal(f"nvidia-smi failed: {result.stderr.strip() or 'no GPU data returned'}")
    line = result.stdout.splitlines()[0].strip()
    print(f"[Verify Runtime] nvidia-smi: {line}", flush=True)
    return line


def verify_torch_and_gpu(hardware: Any) -> None:
    try:
        import torch
    except Exception as exc:
        _fatal(f"PyTorch is not importable: {exc}")
    print(f"[Verify Runtime] torch      : {torch.__version__}", flush=True)
    if hardware.vendor == "nvidia":
        _nvidia_driver(hardware)
        if not bool(torch.cuda.is_available()):
            _fatal("NVIDIA hardware is visible to nvidia-smi but torch.cuda.is_available() is false")
        print(f"[Verify Runtime] torch.cuda : {torch.version.cuda or 'unknown'}", flush=True)
    elif hardware.vendor == "amd" and os.name == "nt":
        if "torch-directml" not in _distributions():
            _fatal("AMD hardware requires the torch-directml distribution")


def verify_tensorrt_package(hardware: Any) -> None:
    if hardware.vendor != "nvidia":
        print("[Verify Runtime] tensorrt   : not required for non-NVIDIA runtime", flush=True)
        return
    installed = _distributions()
    missing = [name for name in ("tensorrt-cu12", "tensorrt-cu12-libs", "tensorrt-cu12-bindings")
               if _normalise(name) not in installed]
    if missing:
        _fatal("NVIDIA runtime is missing TensorRT distributions: " + ", ".join(missing))
    try:
        import tensorrt
    except Exception as exc:
        _fatal(f"TensorRT package is installed but cannot be imported: {exc}")
    version = getattr(tensorrt, "__version__", "unknown")
    print(f"[Verify Runtime] tensorrt  : {version}", flush=True)


def verify_complete_marker() -> None:
    marker = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          ".pinokio-install-complete.json")
    incomplete = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              ".pinokio-install-incomplete.json")
    if os.path.exists(incomplete):
        _fatal("installation is explicitly marked incomplete; run Pinokio Install or Update")
    if not os.path.isfile(marker):
        _fatal("installation-complete marker is missing; app/env is not proof of a valid install")


def verify_environment(*, require_complete: bool = False) -> None:
    if require_complete:
        verify_complete_marker()
    app_dir = os.path.dirname(os.path.abspath(__file__))
    detect_shadowing(app_dir)
    verify_numpy()
    hardware = _hardware()
    print(f"[Verify Runtime] hardware   : {hardware.vendor}/{hardware.system}", flush=True)
    verify_distribution_contract(hardware)
    providers = verify_onnxruntime()
    verify_torch_and_gpu(hardware)
    verify_tensorrt_package(hardware)
    print(f"[Verify Runtime] verified    : providers={providers}", flush=True)
    print("[OK] Python runtime installation contract verified.", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    verify_environment(require_complete=args.require_complete)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
