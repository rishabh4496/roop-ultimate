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
# Keep this checkout's profiles separate from every other roop installation.
# ``app/roop/benchmark/storage.py`` -> project root is three parents upward.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = PROJECT_ROOT           # backwards-compatible public name
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
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise BenchmarkStorageError(
            "Benchmark history has an invalid schema; it was not modified."
        )
    try:
        for item in payload:
            _normalise_result(item, generate_metadata=False)
    except ValueError as exc:
        raise BenchmarkStorageError(
            "Benchmark history has an invalid benchmark record: %s" % exc
        ) from exc
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


_METRIC_KEYS = {"avg_fps", "p1_low_fps", "peak_vram_mb", "peak_cpu_pct"}
_MODEL_KEYS = {"swapper", "enhancer", "mask_engine"}
_PRESET_KEYS = {"max_throughput", "balanced", "quiet"}
_PRESET_VALUE_KEYS = {"threads", "provider_options", "temp_format"}
_BOTTLE_NECKS = {
    "GPU Compute Bound",
    "GPU VRAM Bound",
    "CPU Bound",
    "Disk I/O Bound",
}
_STATUSES = {"declined", "accepted", "pending"}
_RECORD_KEYS = {
    "run_id",
    "timestamp",
    "device_specs",
    "active_models",
    "workload",
    "baseline_metrics",
    "best_metrics",
    "presets",
    "bottleneck",
    "status",
}


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    """Copy one required JSON object field."""
    if not isinstance(value, Mapping):
        raise ValueError("%s must be a mapping" % field_name)
    return dict(value)


def _exact_mapping(
    value: Any, field_name: str, expected_keys: set[str]
) -> dict[str, Any]:
    mapping = _mapping(value, field_name)
    if set(mapping) != expected_keys:
        raise ValueError(
            "%s must contain exactly: %s" % (field_name, ", ".join(sorted(expected_keys)))
        )
    return mapping


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


def _timestamp(value: Any, *, generate: bool) -> str:
    """Return a timezone-aware ISO-8601 run timestamp."""
    if value is None and generate:
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


def _run_id(value: Any, *, generate: bool) -> str:
    """Validate a UUID4, or allocate one only for a newly saved record."""
    if value is None and generate:
        return str(uuid.uuid4())
    if not isinstance(value, str):
        raise ValueError("run_id must be a UUID4 string")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise ValueError("run_id must be a UUID4 string") from exc
    if parsed.version != 4:
        raise ValueError("run_id must be a UUID4 string")
    return str(parsed)


def _normalise_metrics(value: Any, field_name: str) -> dict[str, float]:
    metrics = _exact_mapping(value, field_name, _METRIC_KEYS)
    normalised = {
        key: _finite_float(metrics[key], "%s.%s" % (field_name, key))
        for key in sorted(_METRIC_KEYS)
    }
    if any(number < 0.0 for number in normalised.values()):
        raise ValueError("%s values cannot be negative" % field_name)
    return normalised


def _normalise_presets(value: Any) -> dict[str, dict[str, Any]]:
    presets = _exact_mapping(value, "presets", _PRESET_KEYS)
    normalised: dict[str, dict[str, Any]] = {}
    for name in sorted(_PRESET_KEYS):
        preset = _exact_mapping(presets[name], "presets.%s" % name, _PRESET_VALUE_KEYS)
        threads = preset["threads"]
        if isinstance(threads, bool):
            raise ValueError("presets.%s.threads must be an integer" % name)
        try:
            threads = int(threads)
        except (TypeError, ValueError) as exc:
            raise ValueError("presets.%s.threads must be an integer" % name) from exc
        if threads < 1:
            raise ValueError("presets.%s.threads must be at least one" % name)
        if not isinstance(preset["provider_options"], Mapping):
            raise ValueError("presets.%s.provider_options must be a mapping" % name)
        if not isinstance(preset["temp_format"], str) or not preset["temp_format"]:
            raise ValueError("presets.%s.temp_format must be a non-empty string" % name)
        normalised[name] = {
            "threads": threads,
            "provider_options": dict(preset["provider_options"]),
            "temp_format": preset["temp_format"],
        }
    return normalised


def _normalise_result(
    data: Mapping[str, Any], *, generate_metadata: bool = True
) -> dict[str, Any]:
    """Validate and normalise the public, strict benchmark-profile schema."""
    if not isinstance(data, Mapping):
        raise ValueError("data must be a mapping")
    unexpected = set(data) - _RECORD_KEYS
    missing = (_RECORD_KEYS - {"run_id", "timestamp"}) - set(data)
    if unexpected or missing:
        details = []
        if unexpected:
            details.append("unexpected: %s" % ", ".join(sorted(unexpected)))
        if missing:
            details.append("missing: %s" % ", ".join(sorted(missing)))
        raise ValueError("invalid benchmark record keys (%s)" % "; ".join(details))

    device_specs = _mapping(data.get("device_specs"), "device_specs")
    active_models = _exact_mapping(data.get("active_models"), "active_models", _MODEL_KEYS)
    workload = _exact_mapping(data.get("workload"), "workload", {"target_faces", "test_mode"})
    target_faces = workload["target_faces"]
    if isinstance(target_faces, bool):
        raise ValueError("workload.target_faces must be an integer")
    try:
        target_faces = int(target_faces)
    except (TypeError, ValueError) as exc:
        raise ValueError("workload.target_faces must be an integer") from exc
    if target_faces < 1:
        raise ValueError("workload.target_faces must be at least 1")
    if workload["test_mode"] not in {"quick", "full"}:
        raise ValueError("workload.test_mode must be 'quick' or 'full'")
    if data["bottleneck"] not in _BOTTLE_NECKS:
        raise ValueError("bottleneck is not a recognised benchmark bottleneck")
    if data["status"] not in _STATUSES:
        raise ValueError("status must be declined, accepted, or pending")
    for name, model in active_models.items():
        if not isinstance(model, str):
            raise ValueError("active_models.%s must be a string" % name)

    record = {
        "run_id": _run_id(data.get("run_id"), generate=generate_metadata),
        "timestamp": _timestamp(data.get("timestamp"), generate=generate_metadata),
        "device_specs": device_specs,
        "active_models": dict(active_models),
        "workload": {"target_faces": target_faces, "test_mode": workload["test_mode"]},
        "baseline_metrics": _normalise_metrics(data["baseline_metrics"], "baseline_metrics"),
        "best_metrics": _normalise_metrics(data["best_metrics"], "best_metrics"),
        "presets": _normalise_presets(data["presets"]),
        "bottleneck": data["bottleneck"],
        "status": data["status"],
    }
    try:
        json.dumps(record, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("benchmark record must be JSON serialisable") from exc
    return record


def save_benchmark_result(
    data: Mapping[str, Any],
    storage_path: str | os.PathLike[str] | None = None,
) -> str:
    """Validate and atomically append one benchmark run to local history.

    Returns the generated or validated UUID4 run identifier.
    """
    if not isinstance(data, Mapping):
        raise ValueError("data must be a mapping")
    record = _normalise_result(data)
    history_path = _history_path(storage_path)
    with _HistoryLock(history_path):
        history = _load_history_unlocked(history_path)
        history.append(record)
        _write_history_unlocked(history_path, history)
    return record["run_id"]


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


def get_latest_profile(
    storage_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    """Return the latest complete profile, or ``None`` before the first run."""
    history = load_benchmark_history(storage_path)
    return dict(history[-1]) if history else None


def get_latest_optimal_settings(
    storage_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any] | None:
    """Return a copy of the most recent recommendation, if one exists."""
    profile = get_latest_profile(storage_path)
    if profile is None:
        return None
    preset = profile["presets"]["balanced"]
    return {
        "execution_threads": preset["threads"],
        "execution_provider": "",
        "temp_format": preset["temp_format"],
        "provider_options": dict(preset["provider_options"]),
    }


def update_setting_status(
    run_id: str,
    applied: bool = True,
    storage_path: str | os.PathLike[str] | None = None,
) -> bool:
    """Compatibility alias for applying or declining a stored profile."""
    return update_profile_status(
        run_id, "accepted" if applied else "declined", storage_path
    )


def update_profile_status(
    run_id: str,
    status: str,
    storage_path: str | os.PathLike[str] | None = None,
) -> bool:
    """Persist a user's accepted, declined, or pending-profile decision."""
    _run_id(run_id, generate=False)
    if status not in _STATUSES:
        raise ValueError("status must be declined, accepted, or pending")
    history_path = _history_path(storage_path)
    with _HistoryLock(history_path):
        history = _load_history_unlocked(history_path)
        for record in history:
            if record.get("run_id") == run_id:
                record["status"] = status
                _write_history_unlocked(history_path, history)
                return True
    return False


__all__ = [
    "BENCHMARK_HISTORY_PATH",
    "BenchmarkStorageError",
    "get_latest_profile",
    "get_latest_optimal_settings",
    "load_benchmark_history",
    "save_benchmark_result",
    "update_setting_status",
]
