"""Standalone diagnostic mode for Roop Ultimate runtime environment.

Gathers and prints detailed diagnostic telemetry for Python, ONNX Runtime,
CUDA, TensorRT, Provider Decision, Config, and Result.
Does NOT start React, Gradio, FastAPI workers, or models.

Exit codes:
0 = PASS
1 = DEGRADED
2 = FAIL
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _get_driver_version() -> str:
    try:
        from roop.backend_manager import _driver_from_smi
        driver = _driver_from_smi()
        if driver and driver != "none":
            return driver
    except Exception:
        pass

    import shutil
    import subprocess
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            res = subprocess.run([smi], capture_output=True, text=True, timeout=2)
            for line in res.stdout.splitlines():
                if "Driver Version:" in line:
                    parts = line.split("Driver Version:")
                    if len(parts) > 1:
                        return parts[1].split()[0].strip()
        except Exception:
            pass
    return "none"


def _get_registered_dll_dirs() -> List[str]:
    dirs: List[str] = []
    if sys.platform == "win32":
        try:
            from roop.gpu_preflight import register_gpu_runtime_dirs
            dirs = register_gpu_runtime_dirs()
        except Exception:
            pass
    return dirs


def _get_trt_provider_lib(dll_dirs: List[str]) -> str:
    # Check if ONNX Runtime TensorRT provider library or nvinfer DLL is found
    names_to_check = (
        "onnxruntime_providers_tensorrt.dll",
        "nvinfer_10.dll",
        "nvinfer.dll",
    )
    for d in dll_dirs:
        if not os.path.isdir(d):
            continue
        for name in names_to_check:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    try:
        import onnxruntime as ort
        ort_dir = os.path.dirname(ort.__file__)
        for name in names_to_check:
            p = os.path.join(ort_dir, "capi", name)
            if os.path.isfile(p):
                return p
    except Exception:
        pass
    return "available via registered DLL directories" if dll_dirs else "none"


def collect_runtime_diagnostics() -> Tuple[Dict[str, Any], int]:
    """Gather diagnostic info across all required sections and determine status and exit code."""
    data: Dict[str, Any] = {}
    fatal_reasons: List[str] = []
    degraded_reasons: List[str] = []

    # 1. Python
    py_version = sys.version.replace("\n", " ").strip()
    py_executable = sys.executable
    data["python"] = {
        "version": py_version,
        "executable": py_executable,
    }

    import numpy as np
    np_ver = getattr(np, "__version__", "unknown")
    try:
        if int(np_ver.split(".")[0]) >= 2:
            fatal_reasons.append(f"NumPy version {np_ver} >= 2.0.0 is incompatible with InsightFace")
    except Exception:
        pass

    # 2. ONNX Runtime
    ort_path = "none"
    ort_version = "none"
    ort_api = "missing"
    ort_providers: List[str] = []
    try:
        import onnxruntime as ort
        ort_path = getattr(ort, "__file__", "none")
        ort_version = getattr(ort, "__version__", "unknown")
        if hasattr(ort, "get_available_providers"):
            ort_api = "present"
            ort_providers = list(ort.get_available_providers())
        else:
            ort_api = "missing"
            fatal_reasons.append("ONNX Runtime missing get_available_providers API")
        if not ort_providers:
            fatal_reasons.append("ONNX Runtime exposes no execution providers")
    except Exception as exc:
        fatal_reasons.append(f"Failed to import onnxruntime: {exc}")

    data["onnxruntime"] = {
        "module_path": ort_path,
        "version": ort_version,
        "get_available_providers_api": ort_api,
        "available_providers": ort_providers,
    }

    # 3. CUDA
    torch_version = "none"
    cuda_runtime_version = "none"
    gpu_name = "none"
    compute_capability = "none"
    driver_version = _get_driver_version()
    has_cuda = False

    try:
        import torch
        torch_version = getattr(torch, "__version__", "none")
        cuda_runtime_version = getattr(torch.version, "cuda", "none") or "none"
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            has_cuda = True
            gpu_name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            compute_capability = f"{major}.{minor}"
        else:
            from roop.startup_state_machine import has_nvidia_hardware
            if has_nvidia_hardware():
                fatal_reasons.append("NVIDIA GPU detected but PyTorch reports CUDA is unavailable")
    except Exception as exc:
        fatal_reasons.append(f"PyTorch import failed: {exc}")

    data["cuda"] = {
        "pytorch_version": torch_version,
        "cuda_runtime_version": cuda_runtime_version,
        "gpu_name": gpu_name,
        "compute_capability": compute_capability,
        "driver": driver_version,
    }

    # 4. TensorRT
    trt_pkg = "not installed"
    trt_version = "none"
    try:
        import tensorrt as trt
        trt_pkg = "installed"
        trt_version = getattr(trt, "__version__", "none")
    except ImportError:
        trt_pkg = "not installed"
    except Exception as exc:
        trt_pkg = f"error: {exc}"

    dll_dirs = _get_registered_dll_dirs()
    trt_provider_lib = _get_trt_provider_lib(dll_dirs)

    # Minimal TensorRT ORT session probe
    trt_session_result = "not tested"
    try:
        from roop.gpu_preflight import get_preflight_result
        preflight = get_preflight_result()
        if preflight.get("active_provider") == "TensorrtExecutionProvider" and preflight.get("tensorrt_session_usable"):
            trt_session_result = "verified active"
        elif preflight.get("active_provider") == "CUDAExecutionProvider":
            trt_session_result = f"fell back to CUDA ({preflight.get('failure_reason') or 'initialization failed'})"
            if trt_pkg == "installed":
                degraded_reasons.append(f"TensorRT probe fell back to CUDA: {preflight.get('failure_reason')}")
        else:
            trt_session_result = preflight.get("failure_reason") or f"active provider: {preflight.get('active_provider')}"
    except Exception as exc:
        trt_session_result = f"probe failed: {exc}"

    data["tensorrt"] = {
        "python_package": trt_pkg,
        "version": trt_version,
        "dll_directories": dll_dirs,
        "provider_library": trt_provider_lib,
        "minimal_session_result": trt_session_result,
    }

    # Native compatibility is a separate, path-aware check.  It is deliberately
    # advisory for CPU/AMD systems and fail-closed for applicable Windows NVIDIA
    # installations, where a provider list alone is not proof of a usable EP.
    try:
        from provision_runtime import detect_hardware
        from windows_runtime_compat import verify_windows_nvidia_runtime
        compat_hardware = detect_hardware()
        compat_vram = [int(value) for value in getattr(compat_hardware, "vram_mb", ()) if int(value) > 0]
        binary_compatibility = verify_windows_nvidia_runtime(
            hardware=compat_hardware,
            require_tensorrt=compat_hardware.vendor == "nvidia" and max(compat_vram, default=0) >= 7 * 1024,
        )
        if binary_compatibility.get("status") == "failed":
            fatal_reasons.extend(binary_compatibility.get("failure_reasons", []))
        elif binary_compatibility.get("status") == "degraded":
            degraded_reasons.extend(binary_compatibility.get("warnings", []))
    except Exception as exc:
        applicable_nvidia = False
        try:
            from provision_runtime import detect_hardware
            detected_hardware = detect_hardware()
            applicable_nvidia = detected_hardware.system == "Windows" and detected_hardware.vendor == "nvidia"
        except Exception:
            pass
        binary_compatibility = {
            "applicable": applicable_nvidia,
            "status": "unavailable",
            "failure_reasons": [str(exc)],
            "warnings": [],
            "selected_dll_paths": {},
        }
        if applicable_nvidia:
            fatal_reasons.append(f"Binary compatibility probe unavailable: {exc}")
        else:
            degraded_reasons.append(f"Binary compatibility probe unavailable: {exc}")
    data["binary_compatibility"] = binary_compatibility

    # 5. Provider Decision
    requested = "auto"
    admitted = "auto"
    active = "unknown"
    fallback_reason = "none"
    try:
        from roop.backend_manager import canonical_provider_decision
        decision = canonical_provider_decision()
        requested = decision.requested
        admitted = decision.admitted
        active = decision.active.replace("ExecutionProvider", "").lower()
        fallback_reason = decision.degradation_reason or "none"
        if decision.degraded:
            degraded_reasons.append(f"Provider degraded: requested {requested}, active {active} ({fallback_reason})")
    except Exception as exc:
        fatal_reasons.append(f"Provider decision failed: {exc}")

    data["provider_decision"] = {
        "requested": requested,
        "admitted": admitted,
        "active": active,
        "fallback_reason": fallback_reason,
    }

    # 6. Config
    config_provider = "none"
    config_precision = "none"
    config_builder_opt = "none"
    try:
        from settings import Settings
        cfg_file = "config.yaml" if os.path.isfile("config.yaml") else "default_config.yaml"
        cfg = Settings(cfg_file)
        config_provider = getattr(cfg, "provider", "auto")
        config_precision = getattr(cfg, "trt_precision", "mixed")
        config_builder_opt = getattr(cfg, "trt_builder_optimization_level", 3)
    except Exception as exc:
        degraded_reasons.append(f"Config load issue: {exc}")

    data["config"] = {
        "provider": config_provider,
        "trt_precision": config_precision,
        "trt_builder_optimization_level": config_builder_opt,
    }

    # 7. Result calculation
    if fatal_reasons:
        overall = "FAIL"
        code = 2
    elif degraded_reasons:
        overall = "DEGRADED"
        code = 1
    else:
        overall = "PASS"
        code = 0

    data["result"] = {
        "status": overall,
        "exit_code": code,
        "fatal_reasons": fatal_reasons,
        "degraded_reasons": degraded_reasons,
    }

    return data, code


def format_diagnostics_report(data: Dict[str, Any]) -> str:
    """Format collected telemetry matching exact prompt headings and fields."""
    py = data["python"]
    ort = data["onnxruntime"]
    cuda = data["cuda"]
    trt = data["tensorrt"]
    binary = data["binary_compatibility"]
    dec = data["provider_decision"]
    cfg = data["config"]
    res = data["result"]

    dll_str = ", ".join(trt["dll_directories"]) if trt["dll_directories"] else "none"
    providers_str = ", ".join(ort["available_providers"]) if ort["available_providers"] else "none"

    lines = [
        "=== Python ===",
        f"python version: {py['version']}",
        f"python executable: {py['executable']}",
        "",
        "=== ONNX Runtime ===",
        f"module path: {ort['module_path']}",
        f"version: {ort['version']}",
        f"get_available_providers API: {ort['get_available_providers_api']}",
        f"available providers: {providers_str}",
        "",
        "=== CUDA ===",
        f"PyTorch version: {cuda['pytorch_version']}",
        f"CUDA runtime version: {cuda['cuda_runtime_version']}",
        f"GPU name: {cuda['gpu_name']}",
        f"compute capability: {cuda['compute_capability']}",
        f"driver: {cuda['driver']}",
        "",
        "=== TensorRT ===",
        f"TensorRT Python package: {trt['python_package']}",
        f"TensorRT version: {trt['version']}",
        f"TensorRT DLL directories: {dll_str}",
        f"TensorRT provider library: {trt['provider_library']}",
        f"minimal TensorRT ORT session result: {trt['minimal_session_result']}",
        "",
        "=== Binary Compatibility ===",
        f"status: {binary.get('status', 'unknown')}",
        f"Python architecture: {binary.get('python_architecture', 'unknown')}",
        f"PyTorch wheel architecture: {binary.get('torch_wheel_architecture', 'unknown')}",
        f"ONNX Runtime wheel architecture: {binary.get('onnxruntime_wheel_architecture', 'unknown')}",
        f"TensorRT bindings architecture: {binary.get('tensorrt_python_bindings_architecture', 'unknown')}",
        f"TensorRT native DLL architecture: {binary.get('tensorrt_native_dll_architecture', 'unknown')}",
        f"PATH contamination: {binary.get('path_contamination', []) or 'none'}",
        f"duplicate DLL versions: {binary.get('duplicate_dll_versions', {}) or 'none'}",
        "selected critical DLL paths:",
        *[
            f"  {name}: {path}"
            for name, path in sorted((binary.get("selected_dll_paths") or {}).items())
        ],
        "",
        "=== Provider Decision ===",
        f"requested: {dec['requested']}",
        f"admitted: {dec['admitted']}",
        f"active: {dec['active']}",
        f"fallback reason: {dec['fallback_reason']}",
        "",
        "=== Config ===",
        f"provider: {cfg['provider']}",
        f"trt_precision: {cfg['trt_precision']}",
        f"trt_builder_optimization_level: {cfg['trt_builder_optimization_level']}",
        "",
        "=== Result ===",
        f"{res['status']}",
    ]
    return "\n".join(lines)


def run_diagnose_runtime() -> int:
    """Execute standalone diagnostic probe, print report, and return exit code."""
    data, code = collect_runtime_diagnostics()
    report = format_diagnostics_report(data)
    print(report, flush=True)
    return code
