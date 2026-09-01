"""Authoritative, JSON-safe runtime state for the web and terminal clients.

This module is an observation layer.  It reads state that the existing API and
processing runtime already own; it does not select providers, change tuning, or
parse terminal output in the browser.  A missing source is represented by an
explicit sentinel instead of a made-up zero or empty label.
"""

from __future__ import annotations

import re
import os
import sys
import threading
import time
from typing import Any, Mapping, Optional


UNKNOWN = "UNKNOWN"
NOT_AVAILABLE = "NOT AVAILABLE"
NOT_APPLICABLE = "NOT APPLICABLE"
SCHEMA_VERSION = 1

_FRAME_RE = re.compile(r"(\d[\d,]*)\s*/\s*(\d[\d,]*)")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps\b", re.IGNORECASE)
_RESOURCE_TTL = 2.0
_resource_lock = threading.Lock()
_resource_cache = {"at": 0.0, "value": None, "process": None}


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _text(value: Any) -> str:
    if value is None:
        return UNKNOWN
    value = str(value).strip()
    return value or UNKNOWN


def _provider_name(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    name = str(value or "").strip()
    if not name:
        return UNKNOWN
    lowered = name.lower().replace("executionprovider", "")
    return {
        "cuda": "cuda",
        "tensorrt": "tensorrt",
        "cpu": "cpu",
        "rocm": "rocm",
        "dml": "dml",
    }.get(lowered, lowered or UNKNOWN)


def _manager():
    # A telemetry poll must not import the processing graph or initialize the
    # ML stack just to answer an idle request.  `run.py` imports `core` before
    # serving the API, so an already-loaded module is sufficient during real
    # application use; tests and lightweight API imports correctly report the
    # runtime manager as unavailable.
    core = sys.modules.get("roop.core")
    return getattr(core, "process_mgr", None) if core is not None else None


def _copy_resources(value: Mapping[str, Any]) -> dict:
    return {
        "gpu": value.get("gpu", UNKNOWN),
        "vram": dict(value.get("vram") or {}),
        "cpu": dict(value.get("cpu") or {}),
        "memory": dict(value.get("memory") or {}),
        **({"gpu_utilization_pct": value["gpu_utilization_pct"]}
           if "gpu_utilization_pct" in value else {}),
    }


def _active_provider() -> str:
    try:
        roop_globals = sys.modules.get("roop.globals")
        if roop_globals is None:
            return UNKNOWN
        providers = getattr(roop_globals, "execution_providers", None)
        if providers:
            return _provider_name(providers)
        cfg = getattr(roop_globals, "CFG", None)
        return _provider_name(getattr(cfg, "provider", None))
    except Exception:
        return UNKNOWN


def _resource_snapshot() -> dict:
    """Return cached read-only host/GPU measurements.

    The 2-second cache is intentionally shorter than the existing diagnostics
    cache's 5-second window but still keeps `/api/progress` polls from issuing
    a driver call on every request.  The runtime monitor's measured values are
    overlaid separately when that opt-in monitor is active.
    """
    now = time.monotonic()
    with _resource_lock:
        if (_resource_cache["value"] is not None and
                now - _resource_cache["at"] < _RESOURCE_TTL):
            return _copy_resources(_resource_cache["value"])

    value = {
        "gpu": UNKNOWN,
        "vram": {"used_gb": UNKNOWN, "free_gb": UNKNOWN, "total_gb": UNKNOWN},
        "cpu": {"utilization_pct": UNKNOWN, "logical_threads": UNKNOWN},
        "memory": {
            "process_rss_gb": UNKNOWN,
            "used_gb": UNKNOWN,
            "available_gb": UNKNOWN,
            "total_gb": UNKNOWN,
            "utilization_pct": UNKNOWN,
        },
    }

    try:
        import psutil
        value["cpu"]["utilization_pct"] = round(float(psutil.cpu_percent()), 1)
        logical = psutil.cpu_count(logical=True)
        if logical:
            value["cpu"]["logical_threads"] = int(logical)
        memory = psutil.virtual_memory()
        value["memory"].update({
            "used_gb": round(memory.used / 2**30, 2),
            "available_gb": round(memory.available / 2**30, 2),
            "total_gb": round(memory.total / 2**30, 2),
            "utilization_pct": round(float(memory.percent), 1),
        })
        process = _resource_cache.get("process")
        if process is None or process.pid != os.getpid():
            process = psutil.Process()
            _resource_cache["process"] = process
        value["memory"]["process_rss_gb"] = round(
            process.memory_info().rss / 2**30, 3)
    except Exception:
        pass

    # Torch is already loaded by the processing application before a real run.
    # Do not import it from an idle progress poll: that would turn a cheap
    # telemetry request into a multi-second ML-stack initialization.
    if "torch" in sys.modules:
        try:
            import torch
        except Exception:
            torch = None
    else:
        torch = None
    try:
        if torch is None:
            raise RuntimeError("torch not initialized by the processing runtime")
        if torch.cuda.is_available():
            value["gpu"] = _text(torch.cuda.get_device_name(0))
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            used_bytes = total_bytes - free_bytes
            value["vram"] = {
                "used_gb": round(used_bytes / 2**30, 2),
                "free_gb": round(free_bytes / 2**30, 2),
                "total_gb": round(total_bytes / 2**30, 2),
            }
    except Exception:
        pass

    with _resource_lock:
        _resource_cache.update({"at": now, "value": dict(value)})
    return _copy_resources(value)


def _monitor_sample(manager) -> Mapping[str, Any]:
    monitor = getattr(manager, "_runtime_monitor", None)
    if monitor is None:
        return {}
    try:
        with monitor._lock:
            samples = list(monitor._samples)
        return samples[-1] if samples else {}
    except Exception:
        return {}


def _profile_data(manager) -> tuple[Mapping[str, Any], Any]:
    profile = getattr(manager, "runtime_profile", None)
    if profile is None:
        return {}, None
    try:
        return profile.as_dict(), profile
    except Exception:
        return {}, profile


def _frame_values(progress: Mapping[str, Any], run_stats: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    desc = str(progress.get("desc") or "")
    match = _FRAME_RE.search(desc)
    done = total = None
    if match:
        try:
            done = int(match.group(1).replace(",", ""))
            total = int(match.group(2).replace(",", ""))
        except ValueError:
            done = total = None
    if done is None:
        done = run_stats.get("frames_done")
    if total is None:
        total = run_stats.get("frames_total")
    fraction = _number(progress.get("progress"))
    fps = None
    fps_match = _FPS_RE.search(desc)
    if fps_match:
        fps = _number(fps_match.group(1))
    return fraction if fraction is not None else UNKNOWN, done if done is not None else UNKNOWN, total if total is not None else UNKNOWN, fps if fps is not None else UNKNOWN


def snapshot(progress: Optional[Mapping[str, Any]] = None,
             run_stats: Optional[Mapping[str, Any]] = None,
             output: Optional[Mapping[str, Any]] = None,
             eta_s: Any = None,
             live_seq: Any = None,
             parts: Optional[list] = None) -> dict:
    """Build one JSON-safe state object consumed by API clients.

    ``progress`` and ``run_stats`` are passed by the API because those objects
    are owned by `api.py`.  Supplying no values is supported for diagnostics
    and tests, but it deliberately reports missing job facts as sentinels.
    """
    progress = progress or {}
    run_stats = run_stats or {}
    output = output or {}
    manager = _manager()
    profile_data, profile = _profile_data(manager)
    provider = _active_provider()
    fraction, frame_done, frame_total, fps = _frame_values(progress, run_stats)
    eta = _number(eta_s)
    resources = _resource_snapshot()
    sample = _monitor_sample(manager)

    # RuntimeMonitor is optional.  Its values are preferred when available
    # because they are the runtime's own sampled measurements; otherwise the
    # cached host/GPU probes remain the only verified values.
    if sample:
        cpu = dict(resources["cpu"])
        memory = dict(resources["memory"])
        vram = dict(resources["vram"])
        for target, source in ((cpu, "cpu_utilization_pct"),
                               (memory, "process_rss_gb"),
                               (memory, "ram_used_gb"),
                               (memory, "ram_available_gb"),
                               (memory, "ram_total_gb"),
                               (memory, "ram_utilization_pct"),
                               (vram, "vram_free_gb"),
                               (vram, "vram_total_gb")):
            if sample.get(source) is not None:
                key = {
                    "cpu_utilization_pct": "utilization_pct",
                    "process_rss_gb": "process_rss_gb",
                    "ram_used_gb": "used_gb",
                    "ram_available_gb": "available_gb",
                    "ram_total_gb": "total_gb",
                    "ram_utilization_pct": "utilization_pct",
                    "vram_free_gb": "free_gb",
                    "vram_total_gb": "total_gb",
                }[source]
                target[key] = round(float(sample[source]), 2)
        resources = {"gpu": resources["gpu"], "vram": vram,
                     "cpu": cpu, "memory": memory}
        if sample.get("gpu_utilization_pct") is not None:
            resources["gpu_utilization_pct"] = round(float(sample["gpu_utilization_pct"]), 1)

    if profile is not None:
        hardware = getattr(profile, "hardware", None)
        gpu_name = getattr(hardware, "gpu_name", "")
        if gpu_name:
            resources["gpu"] = str(gpu_name)

    active = bool(progress.get("processing"))
    paused = bool(progress.get("paused"))
    error = str(progress.get("error") or "").strip()
    desc = str(progress.get("desc") or "").strip()
    if error:
        status_code = "ERROR"
    elif paused:
        status_code = "PAUSED"
    elif active:
        status_code = "PROCESSING"
    elif desc.lower() == "done":
        status_code = "DONE"
    elif desc.lower().startswith("stopped"):
        status_code = "STOPPED"
    else:
        status_code = "IDLE"

    configured_model = UNKNOWN
    configured_precision = UNKNOWN
    quality_profile = UNKNOWN
    try:
        roop_globals = sys.modules.get("roop.globals")
        if roop_globals is None:
            raise RuntimeError("application globals not initialized")
        cfg = getattr(roop_globals, "CFG", None)
        configured_model = _text(getattr(cfg, "swap_model", None))
        quality_profile = _text(getattr(roop_globals, "adaptive_enhancer_profile", None))
        if provider == "tensorrt":
            configured_precision = _text(getattr(cfg, "trt_precision", None))
        elif provider != UNKNOWN:
            configured_precision = NOT_APPLICABLE
    except Exception:
        pass

    effective_precision = _text(profile_data.get("precision")) if profile_data else configured_precision
    effective_provider = _text(profile_data.get("provider")) if profile_data else provider
    if effective_provider != UNKNOWN:
        effective_provider = _provider_name(effective_provider)

    pool = {
        "swap": UNKNOWN, "detector": UNKNOWN, "detmask": UNKNOWN,
        "enhancer": UNKNOWN, "expression": UNKNOWN,
    }
    tuning = profile_data.get("tuning") or {}
    for key, field in (("swap", "swapper_pool_size"),
                       ("detector", "detector_pool_size"),
                       ("detmask", "detmask_pool_size"),
                       ("enhancer", "enhancer_pool_size"),
                       ("expression", "expression_pool_size")):
        if field in tuning:
            pool[key] = tuning[field]

    workers = {"configured": UNKNOWN, "active": UNKNOWN, "recommended": UNKNOWN}
    queue = {"input": UNKNOWN, "output": UNKNOWN, "capacity": UNKNOWN, "in_flight": UNKNOWN}
    if manager is not None:
        for key, attr in (("configured", "num_threads"),):
            value = getattr(manager, attr, None)
            if value is not None:
                workers[key] = int(value)
        if profile:
            workers["recommended"] = getattr(profile.tuning, "worker_count", UNKNOWN)
        scheduler = getattr(manager, "_runtime_scheduler", None)
        if scheduler is not None:
            workers["active"] = getattr(scheduler, "worker_count", UNKNOWN)
            queue["capacity"] = getattr(scheduler, "queue_capacity", UNKNOWN)
            queue["in_flight"] = getattr(scheduler, "effective_inflight", UNKNOWN)
        try:
            depths = manager._runtime_queue_snapshot()
            queue.update({"input": depths.get("input", UNKNOWN),
                          "output": depths.get("output", UNKNOWN)})
        except Exception:
            pass

    errors = [error] if error else []
    if sample:
        errors.extend(str(item) for item in (sample.get("errors") or []) if item)
    if manager is not None:
        summary = getattr(manager, "_runtime_scheduler_summary", None) or {}
        errors.extend(str(item) for item in (summary.get("errors") or []) if item)

    return {
        "schema_version": SCHEMA_VERSION,
        "job": {
            "id": NOT_AVAILABLE,
            "processing": active,
            "paused": paused,
            "started_at": (_number(run_stats.get("start"))
                           if _number(run_stats.get("start")) is not None else UNKNOWN),
            "output": _text(output.get("path")),
        },
        "frame_progress": {
            "fraction": fraction,
            "done": frame_done,
            "total": frame_total,
            "stage": _text(desc),
            "live_seq": (int(live_seq) if _number(live_seq) is not None else UNKNOWN),
        },
        "fps": fps,
        "eta_s": eta if eta is not None else UNKNOWN,
        "provider": effective_provider,
        "model": configured_model,
        "precision": effective_precision,
        "gpu": resources["gpu"],
        "vram": resources["vram"],
        "cpu": resources["cpu"],
        "memory": resources["memory"],
        "pool": pool,
        "workers": workers,
        "queue": queue,
        "profile": {
            "runtime": _text(profile_data.get("cache_key")),
            "quality": quality_profile,
        },
        "status": {"code": status_code, "message": _text(desc)},
        "warnings": UNKNOWN,
        "errors": errors,
        "observed_at": time.time(),
    }


def reset_resource_cache() -> None:
    """Reset only the observation cache; useful for deterministic tests."""
    with _resource_lock:
        _resource_cache.update({"at": 0.0, "value": None, "process": None})
