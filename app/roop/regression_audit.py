"""Cross-hardware regression-audit contracts.

This module is deliberately an audit layer, not a second execution pipeline.
It describes the supported backend/workflow matrix, records what the current
machine can expose, and identifies cache entries that must not be trusted.
Availability is never reported as validation: a backend is ``available_not_validated``
until a real workload evidence record is supplied by the regression suite.

The same report shape is used on Windows/NVIDIA, ROCm/AMD, DirectML, Apple
CoreML, and CPU hosts.  Missing hardware is represented as ``unavailable``;
it is not converted into a false pass and no result is copied between hosts.
"""

from __future__ import annotations

import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


QUALITY_MODES = ("FAST", "BALANCED", "REALISTIC", "MAX QUALITY")

# These are the user-visible paths accepted by the existing video benchmark.
# Keep this list explicit so a renamed branch cannot silently disappear from
# Phase 15 coverage.
ENHANCERS = (
    "None", "Adaptive", "GFPGAN", "Codeformer", "Codeformer (fp16)",
    "DMDNet", "GPEN 256", "GPEN 256 Pro", "GPEN Realistic", "GPEN",
    "GPEN 256 Ultra", "GPEN 1024", "GPEN 2048", "UltraMax", "Restoreformer++",
    "KEEP (sidecar)",
)

BACKENDS = (
    {"id": "cuda_fp32", "provider": "cuda", "execution_provider": "CUDAExecutionProvider", "family": "nvidia", "precision": "fp32"},
    {"id": "cuda_fp16", "provider": "cuda", "execution_provider": "CUDAExecutionProvider", "family": "nvidia", "precision": "fp16"},
    {"id": "cuda_mixed", "provider": "cuda", "execution_provider": "CUDAExecutionProvider", "family": "nvidia", "precision": "mixed"},
    {"id": "tensorrt_fp32", "provider": "tensorrt", "execution_provider": "TensorrtExecutionProvider", "family": "nvidia", "precision": "fp32"},
    {"id": "tensorrt_fp16", "provider": "tensorrt", "execution_provider": "TensorrtExecutionProvider", "family": "nvidia", "precision": "fp16"},
    {"id": "tensorrt_mixed", "provider": "tensorrt", "execution_provider": "TensorrtExecutionProvider", "family": "nvidia", "precision": "mixed"},
    {"id": "rocm_fp32", "provider": "rocm", "execution_provider": "ROCMExecutionProvider", "family": "amd", "precision": "fp32"},
    {"id": "rocm_fp16", "provider": "rocm", "execution_provider": "ROCMExecutionProvider", "family": "amd", "precision": "fp16"},
    {"id": "rocm_mixed", "provider": "rocm", "execution_provider": "ROCMExecutionProvider", "family": "amd", "precision": "mixed"},
    {"id": "directml_fp32", "provider": "directml", "execution_provider": "DmlExecutionProvider", "family": "directml", "precision": "fp32"},
    {"id": "directml_fp16", "provider": "directml", "execution_provider": "DmlExecutionProvider", "family": "directml", "precision": "fp16"},
    {"id": "coreml_fp32", "provider": "coreml", "execution_provider": "CoreMLExecutionProvider", "family": "apple", "precision": "fp32"},
    {"id": "coreml_fp16", "provider": "coreml", "execution_provider": "CoreMLExecutionProvider", "family": "apple", "precision": "fp16"},
    {"id": "cpu_fp32", "provider": "cpu", "execution_provider": "CPUExecutionProvider", "family": "cpu", "precision": "fp32"},
)

WORKFLOWS = (
    "image_swap", "video_swap", "multiple_faces", "faceset_creation",
    "old_faceset_loading", "new_faceset_loading", "every_enhancer",
    "every_quality_mode", "provider_fallback", "configuration_loading",
    "preview_path", "batch_path", "long_video",
)

LIFECYCLE_CHECKS = (
    "startup", "shutdown", "model_release", "memory_release",
    "repeated_jobs", "switching_gpus", "switching_providers",
    "switching_precision",
)


def _import_ort():
    try:
        import onnxruntime as ort
        return ort
    except Exception:
        return None


def _import_torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def core_enhancer_names(core_path: str | os.PathLike | None = None) -> list[str]:
    """Discover exact enhancer branches from the production selector.

    ``core.py`` intentionally uses exact strings.  Keeping the discovered set
    in the report catches a newly added legacy/config path that a hand-written
    benchmark allowlist would otherwise omit.
    """
    if core_path is None:
        core_path = Path(__file__).with_name("core.py")
    try:
        source = Path(core_path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return []
    names = set(re.findall(r"roop\.globals\.selected_enhancer\s*==\s*'([^']+)'", source))
    names.add("None")
    return sorted(names)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _torch_facts(torch_module) -> dict:
    if torch_module is None:
        return {"imported": False, "cuda": False, "rocm": False, "devices": 0}
    version = getattr(torch_module, "version", None)
    hip = str(getattr(version, "hip", "") or "")
    cuda = False
    devices = 0
    names = []
    try:
        cuda_api = torch_module.cuda
        cuda = bool(cuda_api.is_available())
        devices = int(cuda_api.device_count()) if cuda else 0
        names = [str(cuda_api.get_device_name(i)) for i in range(devices)]
    except Exception:
        pass
    return {
        "imported": True,
        "version": str(getattr(torch_module, "__version__", "unknown")),
        "cuda": cuda,
        "cuda_version": str(getattr(version, "cuda", "") or ""),
        "rocm": bool(hip),
        "hip_version": hip,
        "devices": devices,
        "device_names": names,
    }


def _status_for_backend(row: Mapping[str, Any], available: Iterable[str], torch_facts: Mapping[str, Any], provider_usable=None) -> tuple[str, str]:
    from roop.backend_manager import provider_available

    listed = provider_available(row["execution_provider"], available)
    if not listed:
        return "unavailable", "execution provider is not exposed by this ORT build"
    if row["family"] == "nvidia" and not torch_facts.get("cuda"):
        return "unavailable", "provider is listed but no CUDA device is visible"
    if row["family"] == "amd" and not torch_facts.get("rocm"):
        return "unavailable", "ROCm provider is listed but torch exposes no HIP runtime"
    if provider_usable is not None:
        try:
            if not bool(provider_usable(row["execution_provider"], 0, available)):
                return "unavailable", "provider capability probe rejected device 0"
        except Exception as exc:
            return "probe_error", "provider capability probe failed: %s" % exc
    return "available_not_validated", "provider is exposed; real workload evidence is still required"


def runtime_capabilities(*, ort_module=None, torch_module=None, provider_usable=None) -> dict:
    """Collect portable runtime facts without constructing a production model."""
    ort_module = _import_ort() if ort_module is None else ort_module
    torch_module = _import_torch() if torch_module is None else torch_module
    available = []
    if ort_module is not None:
        try:
            available = [str(p) for p in ort_module.get_available_providers()]
        except Exception:
            available = []
    torch_facts = _torch_facts(torch_module)
    backend_rows = []
    for row in BACKENDS:
        status, reason = _status_for_backend(row, available, torch_facts, provider_usable)
        backend_rows.append(dict(row, status=status, reason=reason))
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "onnxruntime": str(getattr(ort_module, "__version__", "unavailable")),
        "available_providers": available,
        "torch": torch_facts,
        "backends": backend_rows,
    }


def _cache_entry(path: Path, root: Path, status: str, reasons: list[str]) -> dict:
    try:
        size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.is_dir() else path.stat().st_size
    except OSError:
        size = None
    return {
        "path": str(path),
        "relative": str(path.relative_to(root)) if path != root else ".",
        "status": status,
        "reasons": reasons,
        "bytes": size,
    }


def inspect_cache_roots(roots: Iterable[str | os.PathLike], active_namespaces: Iterable[str] = ()) -> dict:
    """Find stale/unscoped engine and runtime-profile artifacts, without deleting.

    A directory is trusted only when it starts with an active namespace. Names
    containing ``drvunknown`` are always untrusted because driver identity is
    now part of the TensorRT namespace. Legacy precision-only directories are
    reported for review rather than silently reused or removed.
    """
    namespaces = tuple(str(value) for value in active_namespaces if value)
    entries = []
    roots_seen = []
    for raw_root in roots:
        root = Path(raw_root)
        roots_seen.append(str(root))
        if not root.exists():
            continue
        try:
            children = list(root.iterdir())
        except OSError:
            continue
        for child in children:
            lower = child.name.lower()
            relevant = child.is_dir() or child.suffix.lower() in (".engine", ".plan", ".timing", ".cache", ".json")
            if not relevant:
                continue
            reasons = []
            if "drvunknown" in lower:
                reasons.append("missing driver identity in cache namespace")
            if child.is_dir() and lower in {"fp16", "fp32", "mixed", "bf16", "int8", "fp8"}:
                reasons.append("legacy precision-only namespace")
            if "runtime_profile" in str(root).lower() and "drvunknown" in lower:
                reasons.append("runtime profile was recorded before driver-key fix")
            if reasons:
                entries.append(_cache_entry(child, root, "stale_candidate", reasons))
                continue
            if child.is_dir() and namespaces and not any(child.name.startswith(ns) for ns in namespaces):
                descendants = list(child.rglob("*.engine")) + list(child.rglob("*.timing"))
                if descendants:
                    entries.append(_cache_entry(child, root, "unscoped_candidate", ["engine artifacts are outside the active runtime namespace"]))
    return {
        "roots": roots_seen,
        "active_namespaces": list(namespaces),
        "entries": entries,
        "stale_candidates": sum(row["status"] == "stale_candidate" for row in entries),
        "unscoped_candidates": sum(row["status"] == "unscoped_candidate" for row in entries),
        "action": "review and rebuild in the active namespace; no automatic deletion performed",
    }


def coverage_rows() -> list[dict]:
    """Return every required backend/workflow and lifecycle evidence slot."""
    rows = []
    for backend in BACKENDS:
        for workflow in WORKFLOWS:
            rows.append({"id": "%s:%s" % (backend["id"], workflow), "backend": backend["id"], "workflow": workflow, "status": "not_run", "evidence": None})
    for check in LIFECYCLE_CHECKS:
        rows.append({"id": "lifecycle:%s" % check, "workflow": check, "status": "not_run", "evidence": None})
    return rows


def summarize_coverage(rows: Iterable[Mapping[str, Any]]) -> dict:
    counts = {}
    for row in rows:
        status = str(row.get("status", "not_run"))
        counts[status] = counts.get(status, 0) + 1
    return {"total": sum(counts.values()), "by_status": counts, "complete": counts.get("pass", 0) == sum(counts.values()) and bool(counts)}


def build_report(*, cache_roots: Iterable[str | os.PathLike] = (), active_namespaces: Iterable[str] = (), ort_module=None, torch_module=None, provider_usable=None) -> dict:
    """Build a machine-readable audit report with explicit missing evidence."""
    rows = coverage_rows()
    report = {
        "schema": "roop-phase15-cross-hardware-audit-v1",
        "runtime": runtime_capabilities(ort_module=ort_module, torch_module=torch_module, provider_usable=provider_usable),
        "cache_audit": inspect_cache_roots(cache_roots, active_namespaces),
        "enhancers": {
            "required": list(ENHANCERS),
            "source_discovered": core_enhancer_names(),
            "missing_from_audit": sorted(set(core_enhancer_names()) - set(ENHANCERS)),
            "status": "not_run",
            "evidence": None,
        },
        "quality_modes": {"required": list(QUALITY_MODES), "status": "not_run", "evidence": None},
        "coverage": rows,
        "coverage_summary": summarize_coverage(rows),
        "rules": {
            "availability_is_not_validation": True,
            "unavailable_hardware_is_not_pass": True,
            "cross_host_results_must_not_be_copied": True,
            "cache_entries_are_not_deleted_by_audit": True,
        },
    }
    return _json_safe(report)


__all__ = [
    "BACKENDS", "ENHANCERS", "LIFECYCLE_CHECKS", "QUALITY_MODES",
    "WORKFLOWS", "build_report", "core_enhancer_names", "coverage_rows", "inspect_cache_roots",
    "runtime_capabilities", "summarize_coverage",
]
