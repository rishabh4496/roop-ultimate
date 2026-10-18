"""What the pipeline is doing RIGHT NOW that the progress bar cannot show.

WHY. A cold TensorRT engine build takes 2-18 minutes (`predictor.verify_and_warmup`), the
engine canary runs a reference engine beside it, and the VRAM governor can throttle in-flight
frames mid-render. None of that is a frame counter, so to the React client a render in its first
minutes is indistinguishable from a hung one - and every request it makes meanwhile hits its
15 s deadline. This registry gives those phases a name and an out-of-band channel
(`GET /api/v1/pipeline/status`, `WS /ws/v1/pipeline/status`, `routes_pipeline_status.py`).

DESIGN (same constraint as `routes_telemetry`): the pipeline runs on plain threads and cannot
await, so it only WRITES here (a lock and a sequence number); the web side samples. Nothing in
the hot path blocks on a client, and a registry nobody reads costs a dict update per phase.

Kinds (the strings clients match on):
    STAGE_COMPILING_TRT     a model's first inference (engine build or cache load) is running
    STAGE_RUNNING_CANARY    the startup fidelity canary is running for a model
    VRAM_GOVERNOR_THROTTLE  free VRAM fell under the floor; in-flight frames were reduced
    SEGMENT_PROGRESS        one slice of a split render changed state
    SEGMENTS_JOINED         a split render's segments were joined (or the join failed)
"""
from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Iterator, List, Optional

STAGE_COMPILING_TRT = "STAGE_COMPILING_TRT"
STAGE_RUNNING_CANARY = "STAGE_RUNNING_CANARY"
VRAM_GOVERNOR_THROTTLE = "VRAM_GOVERNOR_THROTTLE"
SEGMENT_PROGRESS = "SEGMENT_PROGRESS"
SEGMENTS_JOINED = "SEGMENTS_JOINED"

_EVENT_LIMIT = 200

_lock = threading.RLock()
_seq = 0
_active: Dict[int, Dict[str, Any]] = {}
_events: Deque[Dict[str, Any]] = deque(maxlen=_EVENT_LIMIT)
_segments: Dict[str, Dict[str, Any]] = {}
_governor: Dict[str, Any] = {"throttles": 0, "last": None}
_token = 0


def _bump() -> int:
    global _seq
    _seq += 1
    return _seq


def _emit(kind: str, **detail: Any) -> Dict[str, Any]:
    event = {"seq": _bump(), "time": time.time(), "kind": kind, **detail}
    _events.append(event)
    return event


def begin(kind: str, **detail: Any) -> int:
    """Mark `kind` as running. Returns a token for `end`."""
    global _token
    with _lock:
        _token += 1
        token = _token
        _active[token] = {"kind": kind, "since": time.time(), **detail}
        _emit(kind, state="started", **detail)
        return token


def end(token: int, ok: bool = True, **detail: Any) -> None:
    with _lock:
        entry = _active.pop(token, None)
        if entry is None:
            return
        elapsed = time.time() - entry["since"]
        extra = {k: v for k, v in entry.items() if k not in ("kind", "since")}
        _emit(entry["kind"], state="finished" if ok else "failed",
              elapsed_s=round(elapsed, 3), **extra, **detail)


@contextlib.contextmanager
def stage(kind: str, **detail: Any) -> Iterator[None]:
    """`with stage(STAGE_COMPILING_TRT, model="gpen"):` - ended on error too."""
    token = begin(kind, **detail)
    ok = False
    try:
        yield
        ok = True
    finally:
        end(token, ok=ok)


def governor_throttle(**detail: Any) -> None:
    """The VRAM governor reduced work (called by `UnifiedRuntimeScheduler`)."""
    with _lock:
        _governor["throttles"] += 1
        _governor["last"] = {"time": time.time(), **detail}
        _emit(VRAM_GOVERNOR_THROTTLE, **detail)


def segment_update(group: str, index: int, count: int, state: str, **detail: Any) -> None:
    """Per-segment state for a split render (`state` is the queue job state)."""
    with _lock:
        g = _segments.setdefault(str(group), {"count": int(count), "segments": {}})
        g["count"] = int(count)
        g["segments"][int(index)] = {"state": state, "updated": time.time(), **detail}
        g["done"] = sum(1 for s in g["segments"].values() if s["state"] == "COMPLETED")
        _emit(SEGMENT_PROGRESS, group=str(group), index=int(index), count=int(count),
              state=state, **detail)


def segments_joined(group: str, ok: bool, **detail: Any) -> None:
    with _lock:
        g = _segments.setdefault(str(group), {"count": 0, "segments": {}})
        g["joined"] = {"ok": bool(ok), "time": time.time(), **detail}
        _emit(SEGMENTS_JOINED, group=str(group), ok=bool(ok), **detail)


def seq() -> int:
    with _lock:
        return _seq


def snapshot(events: int = 50) -> Dict[str, Any]:
    with _lock:
        active: List[Dict[str, Any]] = [dict(v, elapsed_s=round(time.time() - v["since"], 3))
                                        for v in _active.values()]
        return {
            "seq": _seq,
            "time": time.time(),
            "active": active,
            "compiling": any(a["kind"] == STAGE_COMPILING_TRT for a in active),
            "canary": any(a["kind"] == STAGE_RUNNING_CANARY for a in active),
            "governor": dict(_governor),
            "segments": {k: {"count": v["count"], "done": v.get("done", 0),
                             "joined": v.get("joined"),
                             "segments": {str(i): dict(s) for i, s in sorted(v["segments"].items())}}
                         for k, v in _segments.items()},
            # `[-0:]` is the whole list, so 0 must be spelled out.
            "events": list(_events)[-int(events):] if int(events) > 0 else [],
        }


def reset() -> None:
    """Tests only."""
    global _seq, _token
    with _lock:
        _seq = 0
        _token = 0
        _active.clear()
        _events.clear()
        _segments.clear()
        _governor.update({"throttles": 0, "last": None})
