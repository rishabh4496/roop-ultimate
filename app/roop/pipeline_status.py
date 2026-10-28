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
import os
from collections import deque
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional

# The `stage` field every compile event carries, in the words the brief and the terminal use. The
# `kind` string below is what existing clients match on, so it is NOT renamed.
TRT_COMPILE = "TRT_COMPILE"
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
_last_compile: Dict[str, Any] = {}
_listeners: List[Callable[[Dict[str, Any]], None]] = []
_token = 0


def _bump() -> int:
    global _seq
    _seq += 1
    return _seq


def _emit(kind: str, **detail: Any) -> Dict[str, Any]:
    event = {"seq": _bump(), "time": time.time(), "kind": kind, **detail}
    _events.append(event)
    for listener in list(_listeners):
        try:
            listener(event)
        except Exception as exc:         # a display fault must never stop an engine build
            from roop.degrade import swallowed
            swallowed("roop/pipeline_status.py:listener", exc, "listener skipped")
    return event


def subscribe(listener: Callable[[Dict[str, Any]], None]) -> Callable[[], None]:
    """Call `listener(event)` for every event, on the emitting thread, under the registry lock.

    The terminal's TensorRT spinner is the one subscriber: it must NOT block (it only flips a field
    and wakes its own thread). Returns the function that unsubscribes."""
    with _lock:
        _listeners.append(listener)

    def unsubscribe() -> None:
        with _lock:
            if listener in _listeners:
                _listeners.remove(listener)
    return unsubscribe


def begin(kind: str, **detail: Any) -> int:
    """Mark `kind` as running. Returns a token for `end`."""
    global _token
    if kind == STAGE_COMPILING_TRT:
        detail.setdefault("stage", TRT_COMPILE)
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
        extra.update(detail)             # a finishing detail (cache_hit) may refine a started one
        _emit(entry["kind"], state="finished" if ok else "failed",
              elapsed_s=round(elapsed, 3), **extra)
        if entry["kind"] == STAGE_COMPILING_TRT:
            _last_compile.clear()
            _last_compile.update(extra, ok=bool(ok), elapsed_s=round(elapsed, 3), time=time.time())


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


def session_precision(session: Any) -> str:
    """'FP16' | 'BF16' | 'FP32' for a TensorRT session, 'n/a' when it is not on TensorRT (or cannot say).

    Read from the provider options the session was BUILT with (`trt_fp16_enable` / `trt_bf16_enable`),
    not from the global setting: xseg / 2d106det / 1k3d68 / RF++ are FP32 under an FP16 global, and the
    React banner names the precision of the engine that is actually compiling. "Mixed" is FP16 with
    FP32 islands, so it reads FP16 here.
    """
    try:
        options = (session.get_provider_options() or {}).get("TensorrtExecutionProvider")
    except Exception as exc:
        from roop.degrade import swallowed
        swallowed("roop/pipeline_status.py:session_precision", exc, "precision shown as n/a")
        return "n/a"
    if options is None:                 # absent = not a TensorRT session; {} = TensorRT at its defaults (FP32)
        return "n/a"
    on = lambda key: str(options.get(key, "")).strip().lower() in ("1", "true")
    if on("trt_bf16_enable"):
        return "BF16"
    return "FP16" if on("trt_fp16_enable") else "FP32"


def session_trt_info(session: Any) -> Dict[str, Any]:
    """`{precision, workspace_mb, cache_path}` of a TensorRT session, read from the options it was BUILT with.

    The same source as `session_precision` for the same reason: the global setting is not what this
    engine got. `workspace_mb` / `cache_path` are None when the session is not on TensorRT or the
    option was not set (an absent workspace means TensorRT's own default, which is not a number we
    can state, so it is not guessed)."""
    info: Dict[str, Any] = {"precision": session_precision(session), "workspace_mb": None, "cache_path": None}
    if info["precision"] == "n/a":
        return info
    try:
        options = (session.get_provider_options() or {}).get("TensorrtExecutionProvider") or {}
    except Exception as exc:
        from roop.degrade import swallowed
        swallowed("roop/pipeline_status.py:session_trt_info", exc, "workspace and cache shown as unknown")
        return info
    try:
        raw = options.get("trt_max_workspace_size")
        if raw not in (None, "", "0", 0):
            info["workspace_mb"] = int(round(int(raw) / (1024 * 1024)))
    except (TypeError, ValueError):
        pass
    cache = options.get("trt_engine_cache_path")
    if cache and str(options.get("trt_engine_cache_enable", "1")).strip().lower() in ("1", "true"):
        info["cache_path"] = str(cache)
    return info


def engine_cache_files(cache_path: Optional[str]) -> Optional[Dict[str, tuple]]:
    """`{file name: (size, mtime_ns)}` of the engine plans under `cache_path`, or None when it cannot be read.

    ORT's TensorRT provider writes a `.engine` when it builds one and only READS an existing one on a
    cache hit, so a build is exactly "a plan appeared or changed". None (unknown) is kept apart from
    `{}` (an empty cache, so the first build is a miss)."""
    if not cache_path:
        return None
    try:
        out: Dict[str, tuple] = {}
        for entry in os.scandir(cache_path):
            if entry.is_file() and entry.name.endswith((".engine", ".plan", ".trt")):
                st = entry.stat()
                out[entry.name] = (st.st_size, st.st_mtime_ns)
        return out
    except OSError:
        return None


def cache_hit(before: Optional[Dict[str, tuple]], after: Optional[Dict[str, tuple]]) -> Optional[bool]:
    """True: loaded from the cache. False: a plan was built. None: the cache could not be inspected."""
    if before is None or after is None:
        return None
    return all(before.get(name) == sig for name, sig in after.items())


def format_receipt(info: Dict[str, Any]) -> str:
    """The terminal's one-line receipt for a finished compile: `[TRT CACHE HIT: 0.18s] gpen_256 FP16`."""
    model = str(info.get("model") or "model")
    precision = str(info.get("precision") or "")
    tail = f" {model}" + (f" {precision}" if precision and precision != "n/a" else "")
    elapsed = float(info.get("elapsed_s") or 0.0)
    if not info.get("ok", True):
        return f"[TRT BUILD FAILED after {elapsed:.1f}s]{tail}"
    hit = info.get("cache_hit")
    if hit is True:
        return f"[TRT CACHE HIT: {elapsed:.2f}s]{tail}"
    if hit is False:
        return f"[TRT CACHE MISS: built in {elapsed:.1f}s]{tail}"
    return f"[TRT ENGINE READY: {elapsed:.1f}s, cache not inspectable]{tail}"


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


ENGINE_IDLE = "IDLE"
ENGINE_COMPILING = "COMPILING_TRT"
ENGINE_CANARY = "RUNNING_CANARY"
ENGINE_LABELS = {
    ENGINE_COMPILING: "Compiling TensorRT Engine...",
    ENGINE_CANARY: "Verifying TensorRT Engine...",
}
_ENGINE_STATE_OF = {STAGE_COMPILING_TRT: ENGINE_COMPILING, STAGE_RUNNING_CANARY: ENGINE_CANARY}


def telemetry_fields() -> Dict[str, Any]:
    """The compile / canary state as FLAT, change-only fields for the `/ws/telemetry` frame.

    `/ws/telemetry` is the socket the React shell already holds open for the whole session, and its
    sampler only sends when a frame differs from the last one. So nothing here may tick: `elapsed_s`
    is deliberately absent (the client counts from `engine_since`), and every value is a primitive,
    so the client's shallow compare stays quiet between stages.

    A compile outranks a canary (a canary's reference engine build is the slower of the two, but the
    compile is what blocks the first frame); when both run, the compile is named and `engine_active`
    counts both. `engine_label` is '' when idle so a client can use it as its own truth test.
    """
    with _lock:
        rows = sorted((v for v in _active.values() if v["kind"] in _ENGINE_STATE_OF),
                      key=lambda v: (v["kind"] != STAGE_COMPILING_TRT, v["since"]))
        if not rows:
            return {"engine_state": ENGINE_IDLE, "engine_label": "", "engine_model": "",
                    "engine_precision": "", "engine_since": None, "engine_active": 0}
        head = rows[0]
        state = _ENGINE_STATE_OF[head["kind"]]
        precision = str(head.get("precision") or "")
        return {"engine_state": state, "engine_label": ENGINE_LABELS[state],
                "engine_model": str(head.get("model") or ""),
                "engine_precision": "" if precision == "n/a" else precision,
                "engine_since": round(float(head["since"]), 3), "engine_active": len(rows)}


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
            "last_compile": dict(_last_compile) or None,
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
        _last_compile.clear()
        _governor.update({"throttles": 0, "last": None})
