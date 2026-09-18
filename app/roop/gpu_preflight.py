#!/usr/bin/env python3
"""Single authoritative Windows GPU runtime preflight.

Determines the concrete usability of CUDA and TensorRT execution providers
before any production model session is instantiated.

Distinguishes:
- package missing
- provider not compiled into ORT
- DLL load failure
- CUDA unavailable
- TensorRT initialization failure
- TensorRT session construction failure
"""
from __future__ import annotations

import glob
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Tiny in-memory 1-node Relu ONNX model (91 bytes).
# Generated via onnx.helper: Float tensor X [1, 4] -> Relu -> Y [1, 4], opset 13.
# Zero external package dependencies needed to deserialize or construct in ORT.
_TINY_ONNX_PROBE_BYTES = (
    b"\x08\n:?\n\x0c\n\x01X\x12\x01Y\"\x04Relu\x12\x05probeZ\x13\n\x01X\x12\x0e\n\x0c\x08"
    b"\x01\x12\x08\n\x02\x08\x01\n\x02\x08\x04b\x13\n\x01Y\x12\x0e\n\x0c\x08\x01\x12\x08\n"
    b"\x02\x08\x01\n\x02\x08\x04B\x04\n\x00\x10\r"
)

_REGISTERED_DLL_DIRS: List[str] = []
_PREFLIGHT_CACHE: Optional[Dict[str, Any]] = None


def register_gpu_runtime_dirs() -> List[str]:
    """Register TensorRT and CUDA DLL directories so ONNX Runtime can load them.

    Must be called BEFORE onnxruntime creates any GPU inference session.
    Safe and idempotent.
    """
    global _REGISTERED_DLL_DIRS
    if _REGISTERED_DLL_DIRS:
        return list(_REGISTERED_DLL_DIRS)

    dll_dirs: List[str] = []

    def _add(directory: Optional[str]) -> None:
        if directory and os.path.isdir(directory) and directory not in dll_dirs:
            dll_dirs.append(directory)

    # 1. TensorRT and Torch library directories
    for module_name, resolver in (
        ("tensorrt", lambda m: os.path.join(
            os.path.dirname(os.path.dirname(getattr(m, "__file__", ""))), "tensorrt_libs")),
        ("tensorrt_libs", lambda m: os.path.dirname(getattr(m, "__file__", ""))),
        ("torch", lambda m: os.path.join(os.path.dirname(getattr(m, "__file__", "")), "lib")),
    ):
        try:
            module = __import__(module_name)
            if getattr(module, "__file__", None):
                _add(resolver(module))
        except Exception:
            continue

    # 2. Pip-installed NVIDIA runtime wheels (nvidia/*/bin)
    try:
        import nvidia
        roots = ([os.path.dirname(getattr(nvidia, "__file__", ""))]
                 if getattr(nvidia, "__file__", None)
                 else list(getattr(nvidia, "__path__", [])))
        for root in roots:
            for current, _dirs, _files in os.walk(root):
                if os.path.basename(current).lower() == "bin":
                    _add(current)
    except Exception:
        pass

    # 3. Register via os.add_dll_directory and prepend to PATH on Windows
    for directory in dll_dirs:
        try:
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(directory)
        except Exception:
            pass
        os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")

    _REGISTERED_DLL_DIRS = list(dll_dirs)
    return list(_REGISTERED_DLL_DIRS)


def _check_tensorrt_dlls(search_dirs: List[str]) -> bool:
    """Verify that essential TensorRT DLLs are resolvable on Windows."""
    if sys.platform != "win32":
        return True

    roots = list(search_dirs)
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry and os.path.isdir(entry) and entry not in roots:
            roots.append(entry)

    # Look for nvinfer_*.dll or nvinfer.dll
    for root in roots:
        try:
            matches = glob.glob(os.path.join(root, "nvinfer*.dll"))
            if matches:
                return True
        except OSError:
            continue
    return False


def run_gpu_preflight(force_probe: bool = False) -> Dict[str, Any]:
    """Execute authoritative GPU runtime preflight checks (A-H).

    Returns a standardized dictionary with keys:
    - onnxruntime_importable: bool
    - onnxruntime_version: str
    - onnxruntime_path: str
    - available_providers: list[str]
    - cuda_available: bool
    - tensorrt_available: bool
    - tensorrt_session_usable: bool
    - active_provider: str
    - failure_stage: str | None
    - failure_reason: str | None
    """
    global _PREFLIGHT_CACHE
    if _PREFLIGHT_CACHE is not None and not force_probe:
        return dict(_PREFLIGHT_CACHE)

    # Always ensure DLL directories are registered first on Windows
    registered_dirs = register_gpu_runtime_dirs()

    result: Dict[str, Any] = {
        "onnxruntime_importable": False,
        "onnxruntime_version": "",
        "onnxruntime_path": "",
        "available_providers": [],
        "cuda_available": False,
        "tensorrt_available": False,
        "tensorrt_session_usable": False,
        "active_provider": "none",
        "failure_stage": None,
        "failure_reason": None,
    }

    # A. Check if onnxruntime is importable
    try:
        import onnxruntime as ort
    except (ImportError, ModuleNotFoundError) as exc:
        result["failure_stage"] = "package_missing"
        result["failure_reason"] = f"onnxruntime is not installed: {exc}"
        _PREFLIGHT_CACHE = dict(result)
        return result
    except Exception as exc:
        result["failure_stage"] = "package_missing"
        result["failure_reason"] = f"onnxruntime import failed: {exc}"
        _PREFLIGHT_CACHE = dict(result)
        return result

    ort_file = getattr(ort, "__file__", None)
    if not ort_file:
        result["failure_stage"] = "package_missing"
        result["failure_reason"] = (
            "onnxruntime resolved to an unpopulated namespace package (__file__ is None)"
        )
        _PREFLIGHT_CACHE = dict(result)
        return result

    result["onnxruntime_importable"] = True
    result["onnxruntime_version"] = str(getattr(ort, "__version__", ""))
    result["onnxruntime_path"] = str(ort_file)

    # B. Check get_available_providers()
    lister = getattr(ort, "get_available_providers", None)
    if not callable(lister):
        result["failure_stage"] = "provider_not_compiled"
        result["failure_reason"] = (
            "imported onnxruntime module does not expose callable get_available_providers()"
        )
        _PREFLIGHT_CACHE = dict(result)
        return result

    try:
        providers = [str(p) for p in lister()]
        result["available_providers"] = providers
    except Exception as exc:
        result["failure_stage"] = "provider_not_compiled"
        result["failure_reason"] = f"ort.get_available_providers() raised: {exc}"
        _PREFLIGHT_CACHE = dict(result)
        return result

    # C. Check CUDA provider
    has_cuda_ep = "CUDAExecutionProvider" in providers

    # D. Check TensorRT provider
    has_trt_ep = "TensorrtExecutionProvider" in providers

    # F. Check CUDA device visibility
    cuda_device_ok = False
    try:
        import torch
        cuda_device_ok = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        cuda_device_ok = False
    result["cuda_available"] = cuda_device_ok

    # E. Check TensorRT dependent DLLs on Windows
    trt_dlls_ok = True
    if has_trt_ep and sys.platform == "win32":
        trt_dlls_ok = _check_tensorrt_dlls(registered_dirs)
        if not trt_dlls_ok:
            result["failure_stage"] = "dll_load_failure"
            result["failure_reason"] = (
                "TensorRT runtime DLLs (nvinfer_10.dll / nvinfer*.dll) were not found "
                "in registered directories or PATH"
            )

    # If CUDA is unavailable on the machine:
    if not cuda_device_ok and (has_cuda_ep or has_trt_ep):
        if not result["failure_stage"]:
            result["failure_stage"] = "cuda_unavailable"
            result["failure_reason"] = "No CUDA-capable GPU detected by PyTorch runtime"

    # If TensorRT is not compiled into ORT:
    if not has_trt_ep and not result["failure_stage"]:
        result["failure_stage"] = "provider_not_compiled"
        result["failure_reason"] = "TensorrtExecutionProvider is not compiled into this onnxruntime build"

    # G & H. Test actual session construction with tiny deterministic ONNX probe
    # Try TensorRT first if supported and DLLs are present
    session_probed = False
    if has_trt_ep and cuda_device_ok and trt_dlls_ok:
        try:
            sess_opts = ort.SessionOptions()
            sess_opts.log_severity_level = 3  # Error only, keep preflight quiet
            sess = ort.InferenceSession(
                _TINY_ONNX_PROBE_BYTES,
                sess_options=sess_opts,
                providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            active = [str(p) for p in sess.get_providers()]
            session_probed = True
            if "TensorrtExecutionProvider" in active:
                result["tensorrt_available"] = True
                result["tensorrt_session_usable"] = True
                result["active_provider"] = "TensorrtExecutionProvider"
                result["failure_stage"] = None
                result["failure_reason"] = None
            elif "CUDAExecutionProvider" in active:
                result["tensorrt_available"] = False
                result["tensorrt_session_usable"] = False
                result["active_provider"] = "CUDAExecutionProvider"
                result["failure_stage"] = "tensorrt_initialization_failure"
                result["failure_reason"] = "TensorRT failed to initialize and session dropped to CUDA"
            else:
                result["tensorrt_available"] = False
                result["tensorrt_session_usable"] = False
                result["active_provider"] = active[0] if active else "CPUExecutionProvider"
                result["failure_stage"] = "tensorrt_initialization_failure"
                result["failure_reason"] = f"TensorRT failed to initialize and session dropped to {result['active_provider']}"
        except Exception as exc:
            result["tensorrt_available"] = False
            result["tensorrt_session_usable"] = False
            result["failure_stage"] = "tensorrt_session_construction_failure"
            result["failure_reason"] = f"TensorRT session construction failed: {exc}"

    # If TensorRT did not become active, probe CUDA session if available
    if result["active_provider"] == "none" and has_cuda_ep and cuda_device_ok:
        try:
            sess_opts = ort.SessionOptions()
            sess_opts.log_severity_level = 3
            sess = ort.InferenceSession(
                _TINY_ONNX_PROBE_BYTES,
                sess_options=sess_opts,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            active = [str(p) for p in sess.get_providers()]
            if "CUDAExecutionProvider" in active:
                result["active_provider"] = "CUDAExecutionProvider"
            elif active:
                result["active_provider"] = active[0]
            else:
                result["active_provider"] = "CPUExecutionProvider"
        except Exception:
            result["active_provider"] = "CPUExecutionProvider"

    # Fallback to CPU if still unset
    if result["active_provider"] == "none":
        result["active_provider"] = (
            "CPUExecutionProvider" if "CPUExecutionProvider" in providers else "none"
        )

    _PREFLIGHT_CACHE = dict(result)
    return result


def get_preflight_result(force_probe: bool = False) -> Dict[str, Any]:
    """Return cached authoritative preflight result, probing on first call."""
    return run_gpu_preflight(force_probe=force_probe)


def clear_preflight_cache() -> None:
    """Clear cached preflight result (used for tests)."""
    global _PREFLIGHT_CACHE
    _PREFLIGHT_CACHE = None
