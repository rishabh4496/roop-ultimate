"""Authoritative, JSON-safe runtime state for the web and terminal clients.

This module is an observation layer.  It reads state that the existing API and
processing runtime already own; it does not select providers, change tuning, or
parse terminal output in the browser.  A missing source is represented by an
explicit sentinel instead of a made-up zero or empty label.
"""

from __future__ import annotations

import re
import os
import platform
import sys
import threading
import time
from typing import Any, Mapping, Optional


UNKNOWN = "UNKNOWN"
NOT_AVAILABLE = "NOT AVAILABLE"
NOT_APPLICABLE = "NOT APPLICABLE"
SCHEMA_VERSION = 1

SECTION_NAMES = (
    "SYSTEM", "HARDWARE", "PROVIDER", "MODEL", "PRECISION",
    "PROCESSING", "POOLING", "QUEUE", "PROFILE", "PERFORMANCE",
    "WARNINGS", "ERRORS", "PROJECT", "CHECKPOINT",
)

_FRAME_RE = re.compile(r"(\d[\d,]*)\s*/\s*(\d[\d,]*)")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps\b", re.IGNORECASE)
_RESOURCE_TTL = 2.0
_resource_lock = threading.Lock()
_resource_cache = {"at": 0.0, "value": None, "process": None}


def classify_log(message: Any) -> tuple[str, str]:
    """Classify an existing terminal line without changing its text.

    The prefixes and terms below are already emitted by the processing and
    queue code.  Classification adds machine-readable context to admitted
    log entries; it does not create a new diagnostic value or parse a value
    that was not already present in the line.
    """
    text = str(message or "").strip()
    lowered = text.lower()
    if re.search(r"error|exception|traceback|failed|failure|abort", lowered):
        return "ERRORS", "ERROR"
    if re.search(r"warning|warn|above the measured-safe|fallback|unavailable|skipped", lowered):
        return "WARNINGS", "WARNING"
    checks = (
        ("CHECKPOINT", r"checkpoint|partial output|segment|part \d+ written|resume"),
        ("QUEUE", r"\bqueue\b|queueing|job"),
        ("PROVIDER", r"provider|onnx backend|executionprovider"),
        ("PRECISION", r"precision|fp16|fp32|bf16|mixed"),
        ("POOLING", r"pool|context|in-flight|inflight|worker"),
        ("PROFILE", r"profile|autotune|calibrat|tuning"),
        ("HARDWARE", r"gpu|vram|cuda|nvdec|nvenc|cpu hardware|ram"),
        ("PERFORMANCE", r"fps|throughput|stage timing|runtime monitor|bottleneck|elapsed"),
        ("PROJECT", r"project"),
        ("PROCESSING", r"processing|upscal|interpolat|encod|combining|mux|paused|stopped|done"),
        ("SYSTEM", r"system host|environment|backend|initialization|keepawake"),
    )
    for category, pattern in checks:
        if re.search(pattern, lowered):
            return category, "INFO"
    return "PROCESSING", "INFO"


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


def _section(status: str, values: Mapping[str, Any], source: str) -> dict:
    """Build the stable named section shape used by terminal and React."""
    return {"status": status, "source": source, "values": dict(values)}


def _queue_snapshot() -> Optional[dict]:
    """Read the queue's compact live projection without copying job payloads."""
    module = sys.modules.get("routes_queue")
    queue = getattr(module, "_queue", None) if module is not None else None
    lock = getattr(module, "_lock", None) if module is not None else None
    if not isinstance(queue, Mapping):
        return None
    try:
        if lock is not None:
            lock.acquire()
        jobs = list(queue.get("jobs") or [])
        current_id = queue.get("current")
        current = next((job for job in jobs if job.get("id") == current_id), None)
        counts = {}
        for job in jobs:
            state = str(job.get("state") or job.get("status") or UNKNOWN)
            counts[state] = counts.get(state, 0) + 1
        return {
            "running": bool(queue.get("running")),
            "paused": bool(queue.get("paused")),
            "current_id": current_id or NOT_AVAILABLE,
            "current_state": (str(current.get("state") or current.get("status"))
                              if current else NOT_AVAILABLE),
            "current_target": (str(current.get("target_name") or UNKNOWN)
                               if current else NOT_AVAILABLE),
            "current_project_id": (str(current.get("project_id") or UNKNOWN)
                                   if current else NOT_APPLICABLE),
            "job_count": len(jobs),
            "state_counts": counts,
        }
    except Exception:
        return None
    finally:
        if lock is not None:
            try:
                lock.release()
            except RuntimeError:
                pass


def _project_snapshot(project_id: str) -> tuple[Optional[dict], Optional[dict]]:
    """Return compact project/checkpoint state from the durable record."""
    if not project_id:
        return None, None
    try:
        import project_checkpoint
        record = project_checkpoint.load(project_id)
        checkpoint = record.get("checkpoint") or {}
        partial = record.get("partial_output") or {}
        project = {
            "id": str(record.get("id") or project_id),
            "name": str(record.get("name") or UNKNOWN),
            "state": str(record.get("state") or UNKNOWN),
            "application_version": str(
                (record.get("application") or {}).get("version") or UNKNOWN),
            "updated_at": record.get("updated_at", UNKNOWN),
        }
        checkpoint_view = {
            "sequence": checkpoint.get("sequence", UNKNOWN),
            "safe_frame": checkpoint.get("safe_frame", UNKNOWN),
            "next_frame": checkpoint.get("next_frame", UNKNOWN),
            "segment_count": len(checkpoint.get("segments") or []),
            "manifest": checkpoint.get("manifest") or NOT_APPLICABLE,
            "partial_file_count": len(partial.get("files") or []),
            "integrity": partial.get("integrity", UNKNOWN),
            "written_at": checkpoint.get("written_at", UNKNOWN),
        }
        return project, checkpoint_view
    except Exception:
        return {
            "id": str(project_id),
            "name": UNKNOWN,
            "state": UNKNOWN,
            "application_version": UNKNOWN,
            "updated_at": UNKNOWN,
        }, {"status": "unreadable", "reason": "project checkpoint could not be read"}


def _pause_state(progress: Mapping[str, Any]) -> dict:
    runtime = sys.modules.get("roop.procmgr_runtime")
    controller = getattr(runtime, "pause_controller", None)
    if controller is not None:
        try:
            return dict(controller.snapshot())
        except Exception:
            pass
    return {
        "requested": bool(progress.get("pause_requested")),
        "acknowledged": bool(progress.get("paused")),
        "active_work": UNKNOWN,
        "pending_output": UNKNOWN,
    }


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
             parts: Optional[list] = None,
             log_lines: Optional[list[Mapping[str, Any]]] = None) -> dict:
    """Build one JSON-safe state object consumed by API clients.

    ``progress`` and ``run_stats`` are passed by the API because those objects
    are owned by `api.py`.  Supplying no values is supported for diagnostics
    and tests, but it deliberately reports missing job facts as sentinels.
    """
    progress = progress or {}
    run_stats = run_stats or {}
    output = output or {}
    manager = _manager()
    queue_view = _queue_snapshot()
    api_module = sys.modules.get("api")
    project_id = str(getattr(api_module, "_active_project_id", "") or "")
    if not project_id and queue_view:
        project_id = str(queue_view.get("current_project_id") or "")
        if project_id in (UNKNOWN, NOT_APPLICABLE):
            project_id = ""
    project_view, checkpoint_view = _project_snapshot(project_id)
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
    pause = _pause_state(progress)
    pause_requested = bool(pause.get("requested"))
    pause_acknowledged = bool(pause.get("acknowledged"))
    error = str(progress.get("error") or "").strip()
    desc = str(progress.get("desc") or "").strip()
    if error:
        status_code = "ERROR"
    elif pause_requested and not pause_acknowledged:
        status_code = "PAUSE_REQUESTED"
    elif paused or pause_acknowledged:
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

    warning_entries = []
    log_error_entries = []
    if log_lines is not None:
        for line in log_lines:
            message = line.get("msg", "") if isinstance(line, Mapping) else line
            category, level = classify_log(message)
            if category == "WARNINGS":
                warning_entries.append({
                    "seq": line.get("seq", UNKNOWN) if isinstance(line, Mapping) else UNKNOWN,
                    "message": str(message),
                    "level": level,
                })
            elif category == "ERRORS":
                log_error_entries.append({
                    "seq": line.get("seq", UNKNOWN) if isinstance(line, Mapping) else UNKNOWN,
                    "message": str(message),
                    "level": level,
                })

    requested_provider = UNKNOWN
    model_values = {"swap_model": configured_model}
    try:
        roop_globals = sys.modules.get("roop.globals")
        cfg = getattr(roop_globals, "CFG", None) if roop_globals else None
        requested_provider = _text(getattr(cfg, "provider", None))
        for name in ("selected_enhancer", "mask_engine", "mask_engine_2",
                     "detector_engine", "upscale_model_after", "interp_after_swap",
                     "recognizer"):
            if cfg is not None and hasattr(cfg, name):
                model_values[name] = _text(getattr(cfg, name, None))
    except Exception:
        pass

    available_provider_values = UNKNOWN
    try:
        roop_globals = sys.modules.get("roop.globals")
        providers = getattr(roop_globals, "execution_providers", None) if roop_globals else None
        if providers:
            available_provider_values = [str(item) for item in providers]
    except Exception:
        pass

    runtime_summary = getattr(manager, "_runtime_summary", None) if manager is not None else None
    scheduler_summary = getattr(manager, "_runtime_scheduler_summary", None) if manager is not None else None
    stage_profile = getattr(manager, "_stage_profile_report", None) if manager is not None else None
    if not isinstance(stage_profile, Mapping):
        stage_profile = {}
    if not isinstance(runtime_summary, Mapping):
        runtime_summary = {}
    if not isinstance(scheduler_summary, Mapping):
        scheduler_summary = {}

    def available(value: Any) -> bool:
        if value is None or value == UNKNOWN or value == NOT_AVAILABLE:
            return False
        if isinstance(value, Mapping):
            return any(available(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return bool(value)
        return True

    processing_values = {
        "status": status_code,
        "stage": _text(desc),
        "fraction": fraction,
        "frames_done": frame_done,
        "frames_total": frame_total,
        "fps": fps,
        "eta_s": eta if eta is not None else UNKNOWN,
        "pause": pause,
    }
    pooling_values = {"pool": pool, "workers": workers}
    profile_values = {
        "runtime_cache_key": _text(profile_data.get("cache_key")),
        "quality": quality_profile,
        "explicit_settings": list(profile_data.get("explicit_settings") or []),
        "automatic_settings": list(profile_data.get("automatic_settings") or []),
        "reasons": list(profile_data.get("reasons") or []),
    }
    performance_values = {
        "fps": fps,
        "eta_s": eta if eta is not None else UNKNOWN,
        "stage_timing": stage_profile,
        "runtime_monitor": runtime_summary,
        "scheduler": scheduler_summary,
        "bottleneck": (scheduler_summary.get("bottleneck") or
                       runtime_summary.get("bottleneck") or UNKNOWN),
    }
    sections = {
        "SYSTEM": _section("AVAILABLE", {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pid": os.getpid(),
        }, "host runtime"),
        "HARDWARE": _section("AVAILABLE" if available(resources) else "UNKNOWN",
                              {"gpu": resources["gpu"], "vram": resources["vram"],
                               "cpu": resources["cpu"], "memory": resources["memory"]},
                              "runtime resource probe"),
        "PROVIDER": _section("AVAILABLE" if available(effective_provider) else "UNKNOWN", {
            "requested": requested_provider,
            "effective": effective_provider,
            "available": available_provider_values,
        }, "runtime globals/profile"),
        "MODEL": _section("AVAILABLE" if available(model_values) else "UNKNOWN",
                           model_values, "runtime configuration"),
        "PRECISION": _section("AVAILABLE" if available(effective_precision) else "UNKNOWN", {
            "configured": configured_precision,
            "effective": effective_precision,
        }, "runtime configuration/profile"),
        "PROCESSING": _section("AVAILABLE", processing_values, "progress controller"),
        "POOLING": _section("AVAILABLE" if available(pooling_values) else "UNKNOWN",
                             pooling_values, "runtime profile/scheduler"),
        "QUEUE": _section("AVAILABLE" if queue_view else "UNKNOWN",
                           queue_view or {"state": UNKNOWN}, "durable queue"),
        "PROFILE": _section("AVAILABLE" if profile_data else "UNKNOWN",
                            profile_values, "runtime profile"),
        "PERFORMANCE": _section("AVAILABLE" if available(performance_values) else "UNKNOWN",
                                 performance_values, "runtime monitor/profile"),
        "WARNINGS": _section("AVAILABLE" if log_lines is not None else "UNKNOWN", {
            "count": len(warning_entries) if log_lines is not None else UNKNOWN,
            "items": warning_entries if log_lines is not None else UNKNOWN,
        }, "structured terminal log"),
        "ERRORS": _section("AVAILABLE" if (log_lines is not None or errors) else "UNKNOWN", {
            "count": (len(errors) + len(log_error_entries)
                      if log_lines is not None or errors else UNKNOWN),
            "items": ([*errors, *log_error_entries]
                      if log_lines is not None or errors else UNKNOWN),
        }, "progress/runtime/structured terminal log"),
        "PROJECT": _section("AVAILABLE" if project_view else "NOT_APPLICABLE",
                            project_view or {"id": NOT_APPLICABLE}, "durable project record"),
        "CHECKPOINT": _section("AVAILABLE" if checkpoint_view else "NOT_APPLICABLE",
                               checkpoint_view or {"state": NOT_APPLICABLE},
                               "durable project checkpoint"),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "job": {
            "id": NOT_AVAILABLE,
            "processing": active,
            "paused": pause_acknowledged,
            "pause_requested": pause_requested,
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
        "pause": pause,
        "warnings": warning_entries if log_lines is not None else UNKNOWN,
        "errors": errors,
        "sections": sections,
        "observed_at": time.time(),
    }


def reset_resource_cache() -> None:
    """Reset only the observation cache; useful for deterministic tests."""
    with _resource_lock:
        _resource_cache.update({"at": 0.0, "value": None, "process": None})
