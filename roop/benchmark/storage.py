"""Durable, local benchmark-history storage.

History is intentionally machine-local: a recommendation measured on one GPU
must not be silently transferred to a different device.  Writes are atomic and
protected by a small cross-process lock so UI and worker processes can record
their results safely.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_HISTORY_PATH = PROJECT_ROOT / ".roop" / "benchmark_history.json"
_LOCK_TIMEOUT_SECONDS = 5.0
_LOCK_POLL_SECONDS = 0.05


class BenchmarkStorageError(RuntimeError):
    """Raised when persistent benchmark history cannot be safely updated."""


def _history_path(storage_path: str | os.PathLike[str] | None) -> Path:
    """Resolve a caller-provided history path or the project-local default."""
    return (
        Path(storage_path).expanduser().resolve()
        if storage_path
        else BENCHMARK_HISTORY_PATH
    )


class _HistoryLock:
    """Portable lock-file guard for the read-modify-write transaction."""

    def __init__(
        self, history_path: Path, timeout: float = _LOCK_TIMEOUT_SECONDS
    ) -> None:
        self._path = history_path.with_suffix(history_path.suffix + ".lock")
        self._timeout = timeout
        self._acquired = False

    def __enter__(self) -> "_HistoryLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                descriptor = os.open(
                    str(self._path), os.O_CREAT | os.O_EXCL | os.O_WRONLY
                )
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(str(os.getpid()))
                self._acquired = True
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise BenchmarkStorageError(
                        "Timed out waiting for benchmark history to become available."
                    )
                time.sleep(_LOCK_POLL_SECONDS)
            except OSError as exc:
                raise BenchmarkStorageError(
                    "Unable to lock benchmark history: %s" % exc
                ) from exc

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._acquired:
            try:
                self._path.unlink(missing_ok=True)
            except OSError as exc:
                LOGGER.warning(
                    "Could not remove benchmark history lock %s: %s", self._path, exc
                )


def _load_history_unlocked(history_path: Path) -> list[dict[str, Any]]:
    """Read the history file, refusing to overwrite malformed user data."""
    if not history_path.exists():
        return []
    try:
        with history_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkStorageError(
            "Benchmark history could not be read: %s" % exc
        ) from exc
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise BenchmarkStorageError(
            "Benchmark history has an invalid schema; it was not modified."
        )
    return payload


def _write_history_unlocked(history_path: Path, history: list[dict[str, Any]]) -> None:
    """Atomically replace the JSON history file after validating serialization."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = history_path.with_name(
        ".%s.%s.tmp" % (history_path.name, uuid.uuid4().hex)
    )
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                history,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, history_path)
    except (OSError, TypeError, ValueError) as exc:
        raise BenchmarkStorageError(
            "Benchmark history could not be saved: %s" % exc
        ) from exc
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    """Copy a JSON object field or raise a concise caller-facing error."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("%s must be a mapping" % field_name)
    return dict(value)


def _finite_float(value: Any, field_name: str, default: float = 0.0) -> float:
    """Normalise a numeric benchmark metric for standards-compliant JSON."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("%s must be a number" % field_name)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be a number" % field_name) from exc
    if not math.isfinite(number):
        raise ValueError("%s must be finite" % field_name)
    return number


def _timestamp(value: Any) -> str:
    """Return a timezone-aware ISO-8601 run timestamp."""
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be an ISO-8601 string") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.isoformat()


def _run_id(value: Any) -> str:
    """Use a supplied UUID when valid, otherwise create the required UUID4."""
    if isinstance(value, str):
        try:
            return str(uuid.UUID(value))
        except ValueError:
            pass
    return str(uuid.uuid4())


def _normalise_result(data: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the public benchmark-record schema and safe defaults."""
    device_specs = _mapping(data.get("device_specs"), "device_specs")
    active_models = _mapping(data.get("active_models"), "active_models")
    workload = _mapping(data.get("workload"), "workload")
    metrics = _mapping(data.get("metrics"), "metrics")
    settings = _mapping(data.get("recommended_settings"), "recommended_settings")

    target_faces = workload.get("target_faces", 1)
    if isinstance(target_faces, bool):
        raise ValueError("workload.target_faces must be an integer")
    try:
        target_faces = int(target_faces)
    except (TypeError, ValueError) as exc:
        raise ValueError("workload.target_faces must be an integer") from exc
    if target_faces < 1:
        raise ValueError("workload.target_faces must be at least 1")

    execution_threads = settings.get("execution_threads", 0)
    if isinstance(execution_threads, bool):
        raise ValueError("recommended_settings.execution_threads must be an integer")
    try:
        execution_threads = int(execution_threads)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "recommended_settings.execution_threads must be an integer"
        ) from exc
    if execution_threads < 0:
        raise ValueError("recommended_settings.execution_threads cannot be negative")

    provider_options = _mapping(
        settings.get("provider_options"), "recommended_settings.provider_options"
    )
    return {
        "run_id": _run_id(data.get("run_id")),
        "timestamp": _timestamp(data.get("timestamp")),
        "device_specs": device_specs,
        "active_models": {
            "swapper": str(active_models.get("swapper", "")),
            "enhancer": str(active_models.get("enhancer", "")),
            "mask_engine": str(active_models.get("mask_engine", "")),
        },
        "workload": {"target_faces": target_faces},
        "metrics": {
            "avg_fps": _finite_float(metrics.get("avg_fps"), "metrics.avg_fps"),
            "p1_low_fps": _finite_float(
                metrics.get("p1_low_fps"), "metrics.p1_low_fps"
            ),
            "peak_vram_mb": _finite_float(
                metrics.get("peak_vram_mb"), "metrics.peak_vram_mb"
            ),
            "peak_cpu_pct": _finite_float(
                metrics.get("peak_cpu_pct"), "metrics.peak_cpu_pct"
            ),
        },
        "recommended_settings": {
            "execution_threads": execution_threads,
            "execution_provider": str(settings.get("execution_provider", "")),
            "temp_format": str(settings.get("temp_format", "")),
            "provider_options": provider_options,
        },
        "applied": bool(data.get("applied", False)),
    }


def save_benchmark_result(
    data: Mapping[str, Any],
    storage_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Validate and atomically append one benchmark run to local history.

    The persisted, normalised record is returned so callers can immediately
    keep its generated ``run_id`` for a later applied-status update.
    """
    if not isinstance(data, Mapping):
        raise ValueError("data must be a mapping")
    record = _normalise_result(data)
    history_path = _history_path(storage_path)
    with _HistoryLock(history_path):
        history = _load_history_unlocked(history_path)
        history.append(record)
        _write_history_unlocked(history_path, history)
    return record


def load_benchmark_history(
    storage_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Return benchmark runs in chronological insertion order.

    An unreadable history is left untouched and represented as an empty result,
    allowing the benchmark UI to remain available while logs retain the cause.
    Saving after corruption still raises ``BenchmarkStorageError`` rather than
    overwriting the recoverable file.
    """
    history_path = _history_path(storage_path)
    try:
        with _HistoryLock(history_path):
            return _load_history_unlocked(history_path)
    except BenchmarkStorageError as exc:
        LOGGER.error("Unable to load benchmark history: %s", exc)
        return []


def get_latest_optimal_settings(
    storage_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    """Return a copy of the most recent recommendation, if one exists."""
    for record in reversed(load_benchmark_history(storage_path)):
        settings = record.get("recommended_settings")
        if isinstance(settings, Mapping):
            return dict(settings)
    return None


def update_setting_status(
    run_id: str,
    applied: bool = True,
    storage_path: str | os.PathLike[str] | None = None,
) -> bool:
    """Set whether a run's recommended settings were applied.

    Returns ``True`` only when a matching run was persisted.  An unknown run ID
    is not an error because history may have been rotated by another process.
    """
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    history_path = _history_path(storage_path)
    with _HistoryLock(history_path):
        history = _load_history_unlocked(history_path)
        for record in history:
            if record.get("run_id") == run_id:
                record["applied"] = bool(applied)
                _write_history_unlocked(history_path, history)
                return True
    return False


__all__ = [
    "BENCHMARK_HISTORY_PATH",
    "BenchmarkStorageError",
    "get_latest_optimal_settings",
    "load_benchmark_history",
    "save_benchmark_result",
    "update_setting_status",
]
