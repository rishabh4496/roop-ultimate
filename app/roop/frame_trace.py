"""Bounded, frame-indexed host timing samples for the live Processing HUD.

The profiler records only calls made inside ``frame_trace_scope``. Stage values
are accumulated per source frame from the same blocking host timers used by the
runtime dashboard; a missing stage stays absent rather than being interpolated.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

_LIMIT = 480
_local = threading.local()
_lock = threading.Lock()
_frames: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()
_epoch = 0


def current_source_frame() -> Optional[int]:
    """Return the active scoped source frame, when this worker is being traced."""
    return getattr(_local, "source_frame", None)


@contextmanager
def frame_trace_scope(source_frame: Optional[int]) -> Iterator[None]:
    """Attribute nested profiler calls on this worker to a 1-based source frame."""
    previous = getattr(_local, "source_frame", None)
    try:
        _local.source_frame = int(source_frame) if source_frame is not None else None
    except (TypeError, ValueError):
        _local.source_frame = None
    try:
        yield
    finally:
        _local.source_frame = previous


def record_stage(stage: str, seconds: float) -> None:
    """Add one measured profiler call to the active source frame, if scoped."""
    source_frame = getattr(_local, "source_frame", None)
    if source_frame is None:
        return
    try:
        elapsed_ms = max(0.0, float(seconds) * 1000.0)
    except (TypeError, ValueError):
        return
    name = str(stage)
    with _lock:
        row = _frames.get(source_frame)
        if row is None:
            row = {"frame": source_frame, "stage_ms": {}, "frame_ms": None}
            _frames[source_frame] = row
        if name == "frame_total":
            row["frame_ms"] = round(elapsed_ms, 3)
        else:
            stages = row["stage_ms"]
            stages[name] = round(float(stages.get(name, 0.0)) + elapsed_ms, 3)
        _frames.move_to_end(source_frame)
        while len(_frames) > _LIMIT:
            _frames.popitem(last=False)


def get_recent_frame_trace(limit: int = 160) -> list[Dict[str, Any]]:
    """Return a JSON-safe, source-frame ordered tail of the bounded trace."""
    return get_frame_trace_snapshot(limit)[1]


def get_frame_trace_snapshot(limit: int = 160) -> tuple[int, list[Dict[str, Any]]]:
    """Read the trace generation and its rows atomically for telemetry clients."""
    bounded = max(1, min(int(limit), _LIMIT))
    with _lock:
        epoch = _epoch
        rows = list(_frames.values())
    rows.sort(key=lambda row: row["frame"])
    return epoch, [
        {"frame": row["frame"], "stage_ms": dict(row["stage_ms"]), "frame_ms": row["frame_ms"]}
        for row in rows[-bounded:]
    ]


def clear_frame_trace() -> None:
    global _epoch
    with _lock:
        _frames.clear()
        _epoch += 1


def get_frame_trace_epoch() -> int:
    """Return the generation of the active per-render trace buffer."""
    with _lock:
        return _epoch
