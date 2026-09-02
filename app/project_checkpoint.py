"""Durable project state for safe processing recovery.

The queue remembers what should run.  A project remembers what was actually
being rendered, including the immutable inputs and the output segments that
survived the last safe boundary.  Project files are deliberately small JSON
records; media and model files remain in their existing locations.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import tempfile
import time
import uuid
from typing import Any, Callable, Mapping


PROJECT_SCHEMA_VERSION = 1
COMPATIBILITY = {
    "project_schema": PROJECT_SCHEMA_VERSION,
    "processing_contract": "segmented-video-v1",
}
PROJECTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "projects")
STATES = ("PAUSED", "INTERRUPTED", "RECOVERABLE", "FAILED", "COMPLETED", "PROCESSING")

# Roughly 1.2 s of backoff in total: long enough to outlast an antivirus or
# indexer handle, short enough that a real permissions fault still surfaces
# promptly rather than stalling a render.
_REPLACE_ATTEMPTS = 8


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _atomic_write(path: str, value: Mapping[str, Any]) -> None:
    """Replace one JSON record only after its contents have reached disk.

    THE RENAME IS RETRIED, BECAUSE ON WINDOWS IT FAILS TRANSIENTLY.  `os.replace`
    raises PermissionError (WinError 5 / WinError 32) whenever any other process
    holds a handle to either file for even a moment -- Defender scanning the
    freshly written temporary, the Search indexer touching the destination, a
    backup agent.  Nothing in this application is holding them.

    Observed on the physical RTX 3060 host: a project's PROCESSING state update
    raised WinError 5, the exception escaped the render worker, and the run died
    while `_progress["processing"]` stayed true -- the UI showed a job
    generating forever with no error and no output.  A single unretried rename
    was therefore able to wedge the whole application.

    The retry window is short and bounded; a genuine, persistent failure (a
    read-only directory, a real permissions problem) still raises, so this
    cannot mask a broken projects directory.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(value, fh, indent=1, sort_keys=True, default=_json_default)
            fh.flush()
            os.fsync(fh.fileno())
        delay = 0.02
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == _REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 0.4)
    finally:
        try:
            os.remove(temporary)
        except OSError:
            pass


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str) -> dict:
    """Return an identity that detects replacement, edits, and truncation."""
    absolute = os.path.abspath(os.path.expanduser(str(path)))
    stat = os.stat(absolute)
    return {
        "path": absolute,
        "name": os.path.basename(absolute),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": file_sha256(absolute),
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def project_path(project_id: str) -> str:
    safe = os.path.basename(str(project_id))
    if safe != str(project_id) or not safe:
        raise ValueError("invalid project id")
    return os.path.join(PROJECTS_DIR, safe + ".json")


_PROVIDER_ALIASES = {
    "CUDAExecutionProvider": "cuda",
    "TensorrtExecutionProvider": "tensorrt",
    "TensorRTExecutionProvider": "tensorrt",
    "CPUExecutionProvider": "cpu",
    "CoreMLExecutionProvider": "coreml",
}


def normalize_provider(value: Any) -> str:
    """Reduce anything ORT calls a provider to this project's short name.

    The shapes that actually reach here are not interchangeable and only one
    of them used to be handled:

        "tensorrt"                                  cfg.provider
        "TensorrtExecutionProvider"                 a bare EP name
        ("TensorrtExecutionProvider", {...opts})    an EP with options
        [("Tensorrt...", {...}), "CPUExecution..."] roop_globals.execution_providers

    The last one is what `api.py` passes as `effective_provider` on every real
    render, and a single unwrap of it yields the *tuple*, not the name — which
    `str()` then rendered as the whole `('TensorrtExecutionProvider', {...})`
    literal, engine-cache path and all. Validation recomputes the identity
    without an effective provider, gets the short "tensorrt", and the two can
    never be equal. Every project written under TensorRT was therefore reported
    RECOVERABLE while being permanently refused with "runtime provider differs
    from the checkpoint" — a checkpoint that could not be resumed by the very
    machine that wrote it, seconds earlier.

    So: unwrap until a string, then alias. `test_project_provider_identity.py`
    fails on the single-unwrap version.
    """
    for _ in range(4):
        if isinstance(value, (list, tuple)):
            value = value[0] if len(value) else ""
        else:
            break
    text = str(value or "")
    # Records written before the unwrap was fixed hold the stringified tuple.
    # Reading the EP name back out of it is lossless, so those projects become
    # resumable again without rewriting any file on disk — which matters,
    # because a migration that rewrites a checkpoint is a migration that can
    # corrupt one.
    if text[:1] in "([":
        match = re.match(r"[(\[]\s*['\"]([A-Za-z]+ExecutionProvider)['\"]", text)
        text = match.group(1) if match else text
    return _PROVIDER_ALIASES.get(text, text)


def runtime_identity(payload: Mapping[str, Any], cfg: Any, effective_provider: Any = None) -> dict:
    hardware = dict(getattr(cfg, "hardware", {}) or {})
    provider = normalize_provider(
        effective_provider or payload.get("provider") or getattr(cfg, "provider", ""))
    return {
        "provider": provider,
        "precision": str(payload.get("trt_precision") or getattr(cfg, "trt_precision", "")),
        "models": {
            key: payload.get(key, getattr(cfg, key, ""))
            for key in ("swap_model", "enhancer", "mask_engine", "mask_engine_2",
                        "detector_engine", "upscale_model_after", "interp_after_swap",
                        "recognizer")
        },
        "hardware": {
            "signature": str(getattr(cfg, "hardware_signature", "") or ""),
            "profile_key": hardware.get("hardware_profile_key"),
            "gpu": hardware.get("gpu") or hardware.get("gpu_name"),
            "vram_tier": hardware.get("vram_tier"),
            "vram_gb": hardware.get("vram_gb") or hardware.get("vram_total_gb"),
            "ram_gb": hardware.get("ram_gb") or hardware.get("ram_total_gb"),
        },
        "platform": platform.platform(),
    }


def new_project(*, job_id: str | None, name: str, payload: Mapping[str, Any],
                sources: list[dict], target: dict, frame_start: int,
                frame_end: int, output: dict, cfg: Any,
                target_faces: list[dict] | None = None,
                app_version: str = "") -> dict:
    now = time.time()
    project_id = uuid.uuid4().hex[:16]
    payload_copy = json.loads(json.dumps(dict(payload), default=_json_default))
    record = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "id": project_id,
        "job_id": job_id,
        "name": name or target.get("name") or project_id,
        "state": "PROCESSING",
        "created_at": now,
        "updated_at": now,
        "application": {
            "compatibility": dict(COMPATIBILITY),
            "version": str(app_version or ""),
        },
        "inputs": {
            "sources": sources,
            "target": target,
            "frame_start": int(frame_start),
            "frame_end": int(frame_end),
            "target_faces": target_faces or [],
        },
        "settings": {
            "payload": payload_copy,
            "fingerprint": fingerprint(payload_copy),
        },
        "runtime": runtime_identity(payload_copy, cfg),
        "output": output,
        "checkpoint": {
            "sequence": 0,
            "safe_frame": int(frame_start),
            "next_frame": int(frame_start),
            "segments": [],
            "manifest": "",
            "written_at": now,
        },
        "partial_output": {"files": [], "integrity": "unknown"},
        "error": "",
    }
    save(record)
    return record


def save(record: Mapping[str, Any]) -> dict:
    value = dict(record)
    value["updated_at"] = time.time()
    _atomic_write(project_path(value["id"]), value)
    return value


def load(project_id: str) -> dict:
    with open(project_path(project_id), encoding="utf-8") as fh:
        return json.load(fh)


def list_projects() -> list[dict]:
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    result = []
    for name in os.listdir(PROJECTS_DIR):
        if not name.endswith(".json") or name.startswith("."):
            continue
        try:
            result.append(load(name[:-5]))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    result.sort(key=lambda item: item.get("updated_at", 0), reverse=True)
    return result


def update_state(project_id: str, state: str, error: str = "") -> dict | None:
    if state not in STATES:
        raise ValueError(f"unknown project state: {state}")
    try:
        record = load(project_id)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    record["state"] = state
    record["error"] = str(error or "")
    return save(record)


def update_runtime(project_id: str, runtime: Mapping[str, Any]) -> dict | None:
    try:
        record = load(project_id)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    record["runtime"] = dict(runtime)
    return save(record)


def update_checkpoint(project_id: str, *, safe_frame: int, next_frame: int,
                      segments: list[dict], manifest: str = "",
                      partial_files: list[dict] | None = None,
                      state: str | None = None) -> dict | None:
    try:
        record = load(project_id)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    previous = record.get("checkpoint") or {}
    record["checkpoint"] = {
        "sequence": int(previous.get("sequence", 0) or 0) + 1,
        "safe_frame": int(safe_frame),
        "next_frame": int(next_frame),
        "segments": json.loads(json.dumps(segments, default=_json_default)),
        "manifest": os.path.abspath(manifest) if manifest else "",
        "manifest_identity": (file_identity(manifest)
                              if manifest and os.path.isfile(manifest) else None),
        "written_at": time.time(),
    }
    if partial_files is not None:
        record["partial_output"] = {
            "files": json.loads(json.dumps(partial_files, default=_json_default)),
            "integrity": "verified",
        }
    if state:
        record["state"] = state
    return save(record)


def _same_file(want: Mapping[str, Any], reasons: list[str], label: str) -> None:
    path = want.get("path")
    if not path or not os.path.isfile(path):
        reasons.append(f"{label} is missing: {path or 'no path recorded'}")
        return
    try:
        actual = file_identity(path)
    except OSError as exc:
        reasons.append(f"{label} cannot be read: {exc}")
        return
    for key in ("size", "sha256"):
        if actual.get(key) != want.get(key):
            reasons.append(f"{label} changed ({key} mismatch)")
            break


def validate(record: Mapping[str, Any], cfg: Any, current_payload: Mapping[str, Any] | None = None,
             check_partial: bool = True) -> list[str]:
    reasons = []
    if record.get("schema_version") != PROJECT_SCHEMA_VERSION:
        reasons.append("project checkpoint schema is not supported by this application")
    compatibility = record.get("application", {}).get("compatibility") or {}
    for key, value in COMPATIBILITY.items():
        if compatibility.get(key) != value:
            reasons.append(f"incompatible checkpoint {key}")

    inputs = record.get("inputs") or {}
    for index, source in enumerate(inputs.get("sources") or []):
        _same_file(source, reasons, f"source {index + 1}")
    _same_file(inputs.get("target") or {}, reasons, "target")

    saved_payload = (record.get("settings") or {}).get("payload") or {}
    payload = current_payload if current_payload is not None else saved_payload
    if fingerprint(payload) != (record.get("settings") or {}).get("fingerprint"):
        reasons.append("processing settings differ from the checkpoint")

    saved_runtime = record.get("runtime") or {}
    current_runtime = runtime_identity(payload, cfg)
    # The saved side goes through the same normalisation as the live side, so a
    # record written before normalize_provider was fixed compares on the EP name
    # it always meant rather than on the literal it was accidentally stored as.
    if normalize_provider(saved_runtime.get("provider")) != current_runtime.get("provider"):
        reasons.append("runtime provider differs from the checkpoint")
    if saved_runtime.get("precision") != current_runtime.get("precision"):
        reasons.append("runtime precision differs from the checkpoint")
    saved_models = saved_runtime.get("models") or {}
    for key, value in saved_models.items():
        if value != (current_runtime.get("models") or {}).get(key):
            reasons.append(f"model configuration '{key}' differs from the checkpoint")
    saved_hw = saved_runtime.get("hardware") or {}
    current_hw = current_runtime.get("hardware") or {}
    if saved_hw.get("signature") and saved_hw.get("signature") != current_hw.get("signature"):
        reasons.append("hardware profile differs from the checkpoint")
    if saved_runtime.get("platform") and saved_runtime.get("platform") != current_runtime.get("platform"):
        reasons.append("operating-system/runtime platform differs from the checkpoint")

    saved_output = record.get("output") or {}
    current_output = {
        "directory": os.path.abspath(str(getattr(cfg, "output_path", "") or
                                          saved_output.get("directory", ""))),
        "format": str(getattr(cfg, "output_video_format", "mp4") or "mp4"),
        "codec": str(getattr(cfg, "output_video_codec", "") or ""),
        "quality": getattr(cfg, "video_quality", None),
        "method": str(payload.get("output_method") or getattr(cfg, "output_method", "File")),
        "template": str(getattr(cfg, "output_template", "") or ""),
    }
    for key in ("directory", "format", "codec", "quality", "method", "template"):
        if saved_output.get(key) != current_output.get(key):
            reasons.append(f"output configuration '{key}' differs from the checkpoint")

    if check_partial:
        manifest_identity = (record.get("checkpoint") or {}).get("manifest_identity")
        if manifest_identity:
            _same_file(manifest_identity, reasons, "checkpoint manifest")
        partial = record.get("partial_output") or {}
        for item in partial.get("files") or []:
            _same_file(item, reasons, f"checkpoint output {item.get('name') or item.get('path')}")
    return reasons


def summarize(record: Mapping[str, Any], validation: list[str] | None = None) -> dict:
    checkpoint = record.get("checkpoint") or {}
    return {
        "id": record.get("id"),
        "job_id": record.get("job_id"),
        "name": record.get("name"),
        "state": record.get("state"),
        "updated_at": record.get("updated_at"),
        "safe_frame": checkpoint.get("safe_frame"),
        "next_frame": checkpoint.get("next_frame"),
        "segments": len(checkpoint.get("segments") or []),
        "recoverable": not validation,
        "validation_errors": list(validation or []),
        "error": record.get("error", ""),
    }


__all__ = [
    "COMPATIBILITY", "PROJECT_SCHEMA_VERSION", "PROJECTS_DIR", "STATES",
    "file_identity", "file_sha256", "fingerprint", "list_projects", "load",
    "new_project", "normalize_provider", "project_path", "runtime_identity",
    "save", "summarize",
    "update_checkpoint", "update_runtime", "update_state", "validate",
]
