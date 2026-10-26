"""Non-blocking CUDA event timing: record now, read later, never synchronize.

WHY. `torch.cuda.Event.elapsed_time` raises `RuntimeError: CUDA error: device not ready` while the end event is
still queued behind GPU work (reproduced on the 4070: `end.query()` is False right after `record()`, and the call
raises). `StageProfiler.end()` called it immediately and swallowed the error, so its "event-only" GPU column was
empty on every asynchronous stage while the report said it was measuring. The cure that costs a render is a
`torch.cuda.synchronize()` per batch (1-3 ms each, and it serialises every stream). This module takes the third
way: pre-allocated start/end pairs are RECORDED on the hot path (a stream-ordered enqueue, no host wait) and READ
later, only once `end.query()` (and an optional retire fence such as an encode-done event) says the GPU has
already passed them, on a throttled cadence by whoever is listening.

    timer = GpuEventTimer(pairs=256)
    tok = timer.begin("warp")          # hot path: one event record
    ... enqueue GPU work ...
    timer.end(tok, fence=encode_done)  # hot path: one event record; `fence` is read later, never waited on
    timer.resolve()                    # telemetry thread, <= 4 Hz: query()-gated elapsed_time, no sync
    timer.snapshot()                   # {"warp": {"samples": n, "avg_ms": .., "last_ms": .., "max_ms": ..}, ...}

`lag` holds a pair back until `lag` newer pairs have begun (batch t-2 while t is in flight), so a read never races
the batch the pipeline is still filling. A pool that is exhausted drops the NEW measurement and counts it
(`dropped`); the hot path never allocates, blocks or waits. Nothing here ever calls `synchronize`.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from roop.degrade import swallowed as _swallowed

# The telemetry cadence the UI samples at (routes_telemetry.SAMPLE_HZ = 4): resolving more often only burns the
# thread that is supposed to be idle.
RESOLVE_INTERVAL_S = 0.25


class _Pair:
    __slots__ = ("start", "end", "name", "seq", "fence")

    def __init__(self, start: Any, end: Any) -> None:
        self.start, self.end = start, end
        self.name: Optional[str] = None
        self.seq = 0
        self.fence: Any = None


class GpuEventTimer:
    """A pool of pre-allocated CUDA event pairs with deferred, query-gated reads."""

    def __init__(self, pairs: int = 256, lag: int = 0, torch_module: Any = None,
                 min_interval_s: float = RESOLVE_INTERVAL_S) -> None:
        self.lag = max(0, int(lag))
        self.min_interval_s = float(min_interval_s)
        self._torch = torch_module if torch_module is not None else _cuda_torch()
        self._lock = threading.Lock()
        self._free: Deque[_Pair] = deque()
        self._pending: List[_Pair] = []
        self._seq = 0
        self._last_resolve: Optional[float] = None
        self._agg: Dict[str, Dict[str, float]] = {}
        self.dropped = 0
        self.errors = 0
        if self._torch is not None:
            try:
                for _ in range(max(1, int(pairs))):
                    self._free.append(_Pair(self._torch.cuda.Event(enable_timing=True),
                                            self._torch.cuda.Event(enable_timing=True)))
            except Exception as exc:  # noqa: BLE001 - no CUDA context: the timer is simply unavailable
                _swallowed("roop/gpu_event_timer.py:preallocate", exc, "GPU event timing unavailable")
                self._free.clear()
                self._torch = None

    @property
    def available(self) -> bool:
        return self._torch is not None

    # ── hot path: enqueue only ──────────────────────────────────────────────
    def begin(self, name: str, stream: Any = None) -> Optional[_Pair]:
        """Record the start event; None when unavailable or the pool is exhausted (counted in `dropped`)."""
        if self._torch is None:
            return None
        with self._lock:
            if not self._free:
                self.dropped += 1
                return None
            pair = self._free.popleft()
            self._seq += 1
            pair.name, pair.seq, pair.fence = str(name), self._seq, None
        try:
            pair.start.record(stream) if stream is not None else pair.start.record()
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            _swallowed("roop/gpu_event_timer.py:begin", exc, "one GPU timing sample lost")
            self._release(pair)
            return None
        return pair

    def end(self, pair: Optional[_Pair], stream: Any = None, fence: Any = None) -> None:
        """Record the end event and queue the pair for a later read. `fence` is any object with `.query()` (an
        encode-done event, a writer's completion event): the pair is not read before it reports done."""
        if pair is None:
            return
        try:
            pair.end.record(stream) if stream is not None else pair.end.record()
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            _swallowed("roop/gpu_event_timer.py:end", exc, "one GPU timing sample lost")
            self._release(pair)
            return
        pair.fence = fence
        with self._lock:
            self._pending.append(pair)

    # ── telemetry thread: read what is already finished ─────────────────────
    def resolve(self, now: Optional[float] = None, force: bool = False) -> int:
        """Fold every pair whose GPU work has already completed into the aggregates; return how many.

        Throttled to `min_interval_s` (4 Hz) unless `force`. `force` also drops the `lag` hold, for the end of a
        run when nothing newer is coming; it still reads only pairs that `query()` reports complete."""
        now = time.perf_counter() if now is None else now
        if not force and self._last_resolve is not None and now - self._last_resolve < self.min_interval_s:
            return 0
        self._last_resolve = now
        with self._lock:
            pending, self._pending = self._pending, []
            newest = self._seq
        keep: List[_Pair] = []
        done = 0
        for pair in pending:
            if not force and pair.seq > newest - self.lag:
                keep.append(pair)
                continue
            try:
                ready = pair.end.query() and (pair.fence is None or pair.fence.query())
                if not ready:
                    keep.append(pair)
                    continue
                self._fold(pair.name, float(pair.start.elapsed_time(pair.end)))
                done += 1
            except Exception as exc:  # noqa: BLE001
                self.errors += 1
                _swallowed("roop/gpu_event_timer.py:resolve", exc, "one GPU timing sample lost")
            self._release(pair)
        if keep:
            with self._lock:
                self._pending = keep + self._pending
        return done

    def _fold(self, name: Optional[str], ms: float) -> None:
        with self._lock:
            a = self._agg.setdefault(name or "?", {"samples": 0, "total_ms": 0.0, "last_ms": 0.0, "max_ms": 0.0})
            a["samples"] += 1
            a["total_ms"] += ms
            a["last_ms"] = ms
            a["max_ms"] = max(a["max_ms"], ms)

    def _release(self, pair: _Pair) -> None:
        pair.name, pair.fence = None, None
        with self._lock:
            self._free.append(pair)

    def snapshot(self) -> Dict[str, Any]:
        """JSON-safe per-name aggregates plus the honesty counters a report must show next to them."""
        with self._lock:
            names = {k: {"samples": int(v["samples"]), "total_ms": v["total_ms"],
                         "avg_ms": round(v["total_ms"] / v["samples"], 4) if v["samples"] else None,
                         "last_ms": round(v["last_ms"], 4), "max_ms": round(v["max_ms"], 4)}
                     for k, v in self._agg.items()}
            return {"available": self._torch is not None, "timing": "cuda-event", "names": names,
                    "pending": len(self._pending), "dropped": self.dropped, "errors": self.errors}

    def reset(self) -> None:
        with self._lock:
            self._agg.clear()
            self.dropped = 0
            self.errors = 0


def _cuda_torch() -> Any:
    try:
        import torch
        return torch if torch.cuda.is_available() else None
    except Exception as exc:  # noqa: BLE001
        _swallowed("roop/gpu_event_timer.py:torch", exc, "GPU event timing unavailable")
        return None


# The timer the report reads. Whoever owns GPU-resident stages registers theirs; nothing registers one today
# except ROOP_PROFILE_DETAIL's StageProfiler, because the production Stage 3 operations are blocking ORT/cv2 calls
# whose host wall time IS their duration (see stage_telemetry.OPS_TIMING).
_registered: Optional[GpuEventTimer] = None
_registered_lock = threading.Lock()


def register(timer: Optional[GpuEventTimer]) -> None:
    global _registered
    with _registered_lock:
        _registered = timer


def registered() -> Optional[GpuEventTimer]:
    with _registered_lock:
        return _registered


def resolve_registered(now: Optional[float] = None, force: bool = False) -> Optional[Dict[str, Any]]:
    """The telemetry tick: resolve the registered timer (self-throttled to 4 Hz) and return its snapshot."""
    timer = registered()
    if timer is None:
        return None
    try:
        timer.resolve(now=now, force=force)
        return timer.snapshot()
    except Exception as exc:  # noqa: BLE001 - a telemetry fault must not break a render, but must not be invisible
        _swallowed("roop/gpu_event_timer.py:resolve_registered", exc, "GPU event block left out of the telemetry")
        return None


__all__ = ["GpuEventTimer", "RESOLVE_INTERVAL_S", "register", "registered", "resolve_registered"]
