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

from ort_package_detector import (
    ORTInspectionError,
    ORT_DISTRIBUTIONS,
    inspect_onnxruntime,
    scan_shadow_paths,
)


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
    """Run the explicit pre-import namespace/shadow scan."""
    namespace_paths, shadow_paths, package_roots = scan_shadow_paths([app_dir])
    if namespace_paths:
        _fatal("onnxruntime package directory is missing __init__.py: " + ", ".join(namespace_paths))
    if shadow_paths:
        _fatal("shadowed ONNX Runtime artifacts found on import paths: " + ", ".join(shadow_paths))
    installed = _distributions()
    ort_packages = sorted(name for name in ORT_DISTRIBUTIONS if name in installed)
    if len(ort_packages) != 1:
        _fatal(
            "Expected exactly one ONNX Runtime distribution, found "
            + (", ".join(ort_packages) if ort_packages else "none")
        )
    expected_root = os.path.dirname(
        str(importlib.metadata.distribution(ort_packages[0]).locate_file("onnxruntime/__init__.py"))
    )
    foreign_roots = [root for root in package_roots if os.path.normcase(os.path.realpath(root))
                     != os.path.normcase(os.path.realpath(expected_root))]
    if foreign_roots:
        _fatal("duplicate or shadowed onnxruntime package roots found: " + ", ".join(foreign_roots))


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
        report = inspect_onnxruntime(module=ort)
    except ORTInspectionError as exc:
        _fatal(str(exc))
    except Exception as exc:
        _fatal(f"Failed to import or inspect onnxruntime: {exc}")

    print(f"Python executable        : {sys.executable}", flush=True)
    print(f"sys.prefix              : {sys.prefix}", flush=True)
    print(f"sys.path                : {sys.path}", flush=True)
    print(f"onnxruntime.__file__    : {report.module_path}", flush=True)
    print(f"onnxruntime spec.origin : {report.spec_origin}", flush=True)
    print(f"onnxruntime distribution: {report.distribution_location}", flush=True)
    print(f"onnxruntime version     : {report.module_version}", flush=True)

    provider_api = getattr(ort, "get_available_providers", None)
    if not callable(provider_api):
        _fatal(
            f"imported onnxruntime ({report.module_path}) does not expose callable "
            "get_available_providers"
        )
    try:
        providers = list(provider_api())
    except Exception as exc:
        _fatal(f"ort.get_available_providers() raised an exception: {exc}")
    if not providers:
        _fatal("onnxruntime reports an empty list of execution providers")

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
