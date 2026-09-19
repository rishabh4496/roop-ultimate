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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ort_package_detector import (
    ORTInspectionError,
    ORT_DISTRIBUTIONS,
    inspect_onnxruntime,
    scan_shadow_paths,
)
from install_state import (
    INSTALLER_VERSION,
    InstallationStateError,
    atomic_write_json,
    check_ready,
    repository_commit,
)


_LAST_ORT_REPORT = None


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
    global _LAST_ORT_REPORT
    try:
        import onnxruntime as ort
        report = inspect_onnxruntime(module=ort)
        _LAST_ORT_REPORT = report
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


def verify_distribution_contract(hardware: Any) -> dict[str, str]:
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
    return {"name": actual[0], "version": installed[actual[0]]}


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


def verify_torch_and_gpu(hardware: Any) -> dict[str, str]:
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
    gpu_name = (hardware.gpu_names[0] if getattr(hardware, "gpu_names", ()) else "none")
    return {
        "pytorch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda or "none"),
        "gpu_name": gpu_name,
        "gpu_vendor": str(hardware.vendor),
    }


def verify_tensorrt_package(hardware: Any) -> dict[str, str]:
    if hardware.vendor != "nvidia":
        print("[Verify Runtime] tensorrt   : not required for non-NVIDIA runtime", flush=True)
        return {"version": "not_required"}
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
    return {"version": str(version)}


def verify_complete_marker() -> None:
    try:
        check_ready()
    except InstallationStateError as exc:
        _fatal(str(exc))


def _write_manifest(path: str, manifest: dict[str, Any]) -> None:
    destination = os.path.abspath(path)
    atomic_write_json(Path(destination), manifest)
    print(f"[Verify Runtime] manifest  : {destination}", flush=True)


def verify_environment(*, require_complete: bool = False, manifest: bool = False) -> dict[str, Any]:
    if require_complete:
        verify_complete_marker()
    app_dir = os.path.dirname(os.path.abspath(__file__))
    detect_shadowing(app_dir)
    verify_numpy()
    hardware = _hardware()
    print(f"[Verify Runtime] hardware   : {hardware.vendor}/{hardware.system}", flush=True)
    ort_distribution = verify_distribution_contract(hardware)
    providers = verify_onnxruntime()
    torch_info = verify_torch_and_gpu(hardware)
    tensorrt_info = verify_tensorrt_package(hardware)
    print(f"[Verify Runtime] verified    : providers={providers}", flush=True)
    print("[OK] Python runtime installation contract verified.", flush=True)
    if not manifest:
        return {}

    from roop.gpu_preflight import get_preflight_result
    preflight = get_preflight_result(force_probe=True)
    vram_mb = [int(value) for value in getattr(hardware, "vram_mb", ()) if int(value) > 0]
    max_vram_mb = max(vram_mb, default=0)
    trt_required = hardware.vendor == "nvidia" and max_vram_mb >= 7 * 1024
    if trt_required and not preflight.get("tensorrt_session_usable", False):
        _fatal(
            "TensorRT minimal session verification failed on a GPU where TensorRT is required: "
            + str(preflight.get("failure_reason") or preflight.get("failure_stage") or "unknown")
        )
    active_provider = str(preflight.get("active_provider") or "none")
    if active_provider == "none":
        _fatal("runtime preflight did not construct an active ORT provider session")
    trt_result = (
        "passed" if preflight.get("tensorrt_session_usable") else
        "not_required_by_sub_7gb_policy" if hardware.vendor == "nvidia" and max_vram_mb < 7 * 1024 else
        "not_required" if hardware.vendor != "nvidia" else
        f"failed:{preflight.get('failure_stage') or 'unknown'}"
    )
    manifest_data: dict[str, Any] = {
        "schema": 3,
        "state": "verified",
        "verification_passed": True,
        "installer_version": INSTALLER_VERSION,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": str(hardware.system),
        "architecture": str(hardware.architecture),
        "gpu_vendor": torch_info["gpu_vendor"],
        "gpu_name": torch_info["gpu_name"],
        "gpu_vram_mb": vram_mb,
        "cuda_version": torch_info["cuda_version"],
        "pytorch_version": torch_info["pytorch_version"],
        "onnxruntime_version": str(_LAST_ORT_REPORT.module_version if _LAST_ORT_REPORT else "unknown"),
        "onnxruntime_path": str(_LAST_ORT_REPORT.module_path if _LAST_ORT_REPORT else "unknown"),
        "onnxruntime_distribution": ort_distribution,
        "tensorrt_version": tensorrt_info["version"],
        "ort_provider_list": providers,
        "tensorrt_session_test_result": trt_result,
        "tensorrt_session_usable": bool(preflight.get("tensorrt_session_usable", False)),
        "active_provider": active_provider,
        "installation_timestamp": datetime.now(timezone.utc).isoformat(),
        "repository_commit": repository_commit(),
        "dependency_verification_status": "passed",
        "runtime_verification_status": "passed",
        "runtime_preflight": preflight,
    }
    return manifest_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--manifest-out", default=None)
    args = parser.parse_args(argv)
    result = verify_environment(
        require_complete=args.require_complete,
        manifest=args.manifest_out is not None,
    )
    if args.manifest_out:
        _write_manifest(args.manifest_out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
