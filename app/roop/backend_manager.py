"""Runtime hardware/provider resolution.

This module deliberately has no model or UI dependencies.  It answers one
question for the rest of the pipeline: which ONNX Runtime providers are
actually usable on this machine, and in what order should they be attempted?
Provider *availability* is not enough (CUDA can be listed while its DLLs or
device are unavailable), so capability checks are kept here and are cached for
the lifetime of the process.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Iterable, List, Dict, Tuple


_lock = threading.Lock()
_probe_cache: Dict[Tuple[str, int], bool] = {}


def _name(value) -> str:
    return value[0] if isinstance(value, (tuple, list)) else str(value)


def _available() -> List[str]:
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except Exception:
        return []


def provider_available(name: str, available: Iterable[str] | None = None) -> bool:
    """Return whether *name* is listed by the installed ORT build."""
    wanted = name.lower().replace("executionprovider", "")
    return any(wanted == p.lower().replace("executionprovider", "")
               for p in (available if available is not None else _available()))


def provider_usable(name: str, device_id: int = 0,
                    available: Iterable[str] | None = None) -> bool:
    """Cheap, side-effect-light capability probe, cached per device.

    A provider that is not listed cannot work. CUDA/TRT additionally require a
    visible CUDA device.  The probe intentionally does not instantiate a
    production model: doing that here would build TensorRT engines during UI
    startup and would make every provider query expensive.
    """
    canonical = _name(name)
    key = (canonical.lower(), int(device_id))
    with _lock:
        if key in _probe_cache:
            return _probe_cache[key]
    ok = provider_available(canonical, available)
    if ok and canonical.lower().startswith(("cuda", "tensorrt", "rocm")):
        try:
            import torch
            ok = bool(torch.cuda.is_available())
            if ok:
                count = int(torch.cuda.device_count())
                ok = 0 <= int(device_id) < max(1, count)
        except Exception:
            ok = False
    if ok and canonical.lower().startswith("dml"):
        # ORT's DML provider is self-contained; the listing is the reliable
        # check and importing torch must not make DirectML appear unavailable.
        ok = True
    with _lock:
        _probe_cache[key] = ok
    return ok


_HIERARCHY = {
    "auto": ("TensorrtExecutionProvider", "CUDAExecutionProvider",
             "ROCMExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"),
    "tensorrt": ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"),
    "cuda": ("CUDAExecutionProvider", "CPUExecutionProvider"),
    "rocm": ("ROCMExecutionProvider", "CPUExecutionProvider"),
    "dml": ("DmlExecutionProvider", "CPUExecutionProvider"),
    "directml": ("DmlExecutionProvider", "CPUExecutionProvider"),
    "coreml": ("CoreMLExecutionProvider", "CPUExecutionProvider"),
    "cpu": ("CPUExecutionProvider",),
}


def resolve_provider_names(requested: Iterable[str] | None,
                           device_id: int = 0) -> List[str]:
    """Resolve a requested backend to a validated, ordered provider chain."""
    requested = list(requested or ("cpu",))
    available = _available()
    # Accept both encoded names and the short names used by settings.yaml.
    short = _name(requested[0]).lower().replace("executionprovider", "")
    candidates = _HIERARCHY.get(short, tuple(_name(p) for p in requested))
    resolved: List[str] = []
    for candidate in candidates:
        if provider_usable(candidate, device_id, available) and candidate not in resolved:
            resolved.append(candidate)
    if not resolved:
        # CPU is the only safe universal last resort.  Keep this explicit so a
        # broken GPU runtime is visible in diagnostics rather than silently
        # producing an empty providers list and a later opaque crash.
        resolved = ["CPUExecutionProvider"] if provider_available("CPUExecutionProvider", available) else []
    return resolved


def diagnostic_report(device_id: int = 0, requested: str | None = None) -> dict:
    """Return JSON-safe diagnostics for logs and the diagnostics panel."""
    available = _available()
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        gpu = torch.cuda.get_device_name(device_id) if cuda else ""
        vram_gb = (torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
                   if cuda else 0.0)
    except Exception:
        cuda, gpu, vram_gb = False, "", 0.0
    configured = requested or os.environ.get("ROOP_EXECUTION_PROVIDER", "auto")
    return {
        "configured": configured,
        "available_providers": available,
        "cuda_visible": cuda,
        "gpu": gpu,
        "vram_gb": round(vram_gb, 2),
        "resolved": resolve_provider_names([configured], device_id),
    }


def clear_probe_cache() -> None:
    with _lock:
        _probe_cache.clear()


def cache_namespace(precision: str, device_id: int = 0) -> str:
    """Build a stable TensorRT cache namespace for this runtime and GPU.

    TensorRT engine filenames include a graph hash, but the parent directory
    must still separate precision and runtime ABI.  This namespace prevents a
    CUDA/TRT upgrade or device swap from reusing a stale engine.
    """
    precision = str(precision or "mixed").lower()
    try:
        import onnxruntime as ort
        ort_ver = str(getattr(ort, "__version__", "unknown"))
    except Exception:
        ort_ver = "unknown"
    gpu = "cpu"
    sm = "na"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(device_id)
            sm = "sm%02d%02d" % tuple(torch.cuda.get_device_capability(device_id))
    except Exception:
        pass
    raw = f"{precision}_{gpu}_{sm}_ort{ort_ver}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def trt_tuning_namespace(builder_optimization_level: int = 3,
                         auxiliary_streams: int = -1,
                         cuda_graph: bool = False) -> str:
    """Return the cache suffix for TensorRT build/runtime tuning knobs.

    These options can change the generated engine or its execution schedule,
    while TensorRT's graph filename does not necessarily encode all of them.
    Keeping them in the parent directory prevents an A/B benchmark or a
    changed default from accidentally reusing an engine built with another
    tuning profile.
    """
    return f"_b{int(builder_optimization_level)}_a{int(auxiliary_streams)}_g{int(bool(cuda_graph))}"
