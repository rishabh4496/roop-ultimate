"""Retry a Stage 3 frame that died of a GPU memory error, alone, after flushing the allocator caches.

Before this module a frame whose swap, mask or restore raised a CUDA / TensorRT out-of-memory error was written
UNSWAPPED: `ProcessMgr`'s per-frame handler logs it, `audit_frame_failed` counts it, and the original frame goes to
the encoder. That keeps a long render alive, but a transient spike (several workers each holding a multi-face frame's
activations at once, an enhancer arena at its peak) cost the frame it happened on, and on a card near its limit it
cost every frame that landed on the same spike.

WHAT THIS DOES, AND WHAT IT DELIBERATELY DOES NOT

  recover  an out-of-memory error on a frame -> every other worker drains out of its frame and waits at the gate,
           the torch caching allocator is emptied (`gc.collect`, `torch.cuda.empty_cache`), and the frame is run
           ONCE MORE with nothing else on the GPU. That is the "downshift concurrency" of the brief taken to its
           limit (one frame), and it is the only thing that can actually relieve a transient: the memory another
           worker holds at that moment is released by that worker finishing, not by anything this thread can free.
  refuse   anything that is not a memory error. A deterministic failure (a bad input shape, a model bug, a
           corrupt context after `cudaErrorIllegalAddress`) fails identically alone, so it keeps the existing path:
           log, count, write the original. A shape-binding error is classified and counted separately so the audit
           names it; it is not retried.
  not built  a per-frame switch to a secondary execution provider. A CPU session of the swapper or a restorer is a
           second multi-hundred-MB model load (seconds per face), and the project decided against a runtime ladder
           on purpose (`create_strictly_accelerated_session` builds once; a session that came up on CPU while CUDA
           was requested was an invisible 10-50x slowdown until `assert_session_providers`). The build-time ladder
           `backend_manager.build_session_with_fallback` is unchanged.
  cannot   shrink an ONNX Runtime arena. ORT keeps freed blocks in its own arena (kNextPowerOfTwo); only the torch
           allocator (the zero-copy ring, GPEN / RF++ / UltraMax GPU post-processing, SAM 2) is emptied here.

The gate costs one lock round trip per frame (a frame is 100+ ms). `ROOP_GPU_RECOVERY=0` runs the old path.

Pure Python plus an optional torch import: `tests/test_gpu_recovery.py`.
"""
from __future__ import annotations

import contextlib
import gc
import threading
import time
from collections import Counter
from typing import Any, Callable, Optional

from roop.degrade import swallowed
from roop.env import env_bool

__all__ = ["classify", "flush_gpu_caches", "RecoveryGate", "gate", "run_frame", "stats", "reset", "enabled",
           "summary_lines"]

# lower-cased substrings of the messages ORT, TensorRT, cuDNN, cuBLAS and torch raise when the DEVICE is out of memory
_OOM_MARKERS = (
    "out of memory",
    "outofmemory",                      # TensorRT "Error Code 2: OutOfMemory", torch.cuda.OutOfMemoryError
    "cudaerrormemoryallocation",
    "cuda_error_out_of_memory",
    "cudnn_status_alloc_failed",
    "cublas_status_alloc_failed",
    "cublas_status_not_initialized",    # what cuBLAS reports when its workspace allocation failed
    "failed to allocate memory",
    "bfcarena",
    "allocate memory for requested size",
    "cuda failure 2",
    "cuda error: out of memory",
)
# a shape or binding the engine refuses; deterministic, so counted but never retried
_SHAPE_MARKERS = (
    "invalid dimensions for input",
    "got invalid dimensions",
    "invalid input shape",
    "shape mismatch",
    "binding",
    "profile",
    "dimensions of tensor",
)


def enabled() -> bool:
    return env_bool("ROOP_GPU_RECOVERY", True)


def classify(exc: BaseException) -> Optional[str]:
    """'oom' | 'shape' | None. Walks the cause chain: ORT wraps the CUDA error."""
    seen = 0
    err: Optional[BaseException] = exc
    while err is not None and seen < 6:
        if isinstance(err, MemoryError):
            return "oom"
        if type(err).__name__ == "OutOfMemoryError":
            return "oom"
        text = f"{type(err).__name__}: {err}".lower()
        if any(m in text for m in _OOM_MARKERS):
            return "oom"
        if any(m in text for m in _SHAPE_MARKERS) and ("tensorrt" in text or "onnxruntime" in text
                                                         or "invalid" in text or "trt" in text):
            return "shape"
        err = err.__cause__ or err.__context__
        seen += 1
    return None


def flush_gpu_caches() -> dict:
    """Empty every allocator cache this process owns that CAN be emptied. Returns what was done."""
    done = {"gc": False, "torch_cache": False}
    try:
        gc.collect()
        done["gc"] = True
    except Exception as err:                                  # noqa: BLE001
        swallowed("gpu_recovery.gc", err, "continuing")
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            done["torch_cache"] = True
    except Exception as err:                                  # noqa: BLE001
        swallowed("gpu_recovery.empty_cache", err, "continuing")
    return done


class RecoveryGate:
    """Frames run shared; one recovery runs exclusive. A waiting recovery blocks NEW frames (no starvation)."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._active = 0
        self._waiting = 0
        self._exclusive = False

    @contextlib.contextmanager
    def shared(self):
        with self._cond:
            while self._exclusive or self._waiting:
                self._cond.wait(0.5)
            self._active += 1
        try:
            yield
        finally:
            with self._cond:
                self._active -= 1
                self._cond.notify_all()

    @contextlib.contextmanager
    def exclusive(self, abort: Optional[Callable[[], bool]] = None, poll: float = 0.25):
        """Wait for every shared holder to leave. Yields True when exclusive, False if `abort()` fired first
        (the render was stopped while a paused worker was still inside its frame)."""
        with self._cond:
            self._waiting += 1
            try:
                while self._exclusive or self._active:
                    if abort is not None and abort():
                        break
                    self._cond.wait(poll)
                got = not (self._exclusive or self._active)
                if got:
                    self._exclusive = True
            finally:
                self._waiting -= 1
                self._cond.notify_all()
        try:
            yield got
        finally:
            if got:
                with self._cond:
                    self._exclusive = False
                    self._cond.notify_all()


gate = RecoveryGate()
_lock = threading.Lock()
_counts: Counter = Counter()
_first: dict = {}


def reset() -> None:
    with _lock:
        _counts.clear()
        _first.clear()


def stats() -> dict:
    with _lock:
        return {"recovered": _counts["recovered"], "failed_after_retry": _counts["failed_after_retry"],
                "oom_seen": _counts["oom"], "shape_seen": _counts["shape"], "flushes": _counts["flush"],
                "wait_s": round(_counts["wait_ms"] / 1000.0, 3), "first": dict(_first)}


def summary_lines() -> list:
    s = stats()
    if not (s["oom_seen"] or s["shape_seen"]):
        return []
    out = []
    if s["oom_seen"]:
        out.append(f"GPU memory errors: {s['oom_seen']} frame(s) hit one; {s['recovered']} recovered after the "
                   f"allocator caches were flushed and the frame was rerun alone, {s['failed_after_retry']} still "
                   f"failed and were written untouched ({s['wait_s']} s spent draining the other workers).")
    if s["shape_seen"]:
        out.append(f"Shape/binding errors: {s['shape_seen']} frame(s), not retried (deterministic).")
    for kind, text in s["first"].items():
        out.append(f"  first {kind}: {text}")
    return out


def _note(kind: str, exc: BaseException) -> None:
    with _lock:
        _counts[kind] += 1
        _first.setdefault(kind, f"{type(exc).__name__}: {str(exc)[:140]}")


def run_frame(first: Callable[[], Any], retry: Optional[Callable[[], Any]] = None, site: str = "frame",
              abort: Optional[Callable[[], bool]] = None,
              before_wait: Optional[Callable[[], None]] = None) -> Any:
    """Run `first()`; on a device out-of-memory error rerun `retry()` (default `first`) alone, once.

    `retry` must build its input afresh: the failed attempt may have mutated the frame in place. Anything that is
    not an out-of-memory error, a retry that fails again, and a stopped render all re-raise the ORIGINAL error so
    the caller's existing handler (log, count, write the original frame) runs exactly as before.

    `before_wait` runs after this frame has left the shared gate and before it waits for the others: release anything
    another worker may be blocked on that only this frame would have produced (the stabilisation warm-up dedup slot),
    or the drain would wait for a frame that is waiting for us.
    """
    if not enabled():
        return first()
    with gate.shared():
        try:
            return first()
        except Exception as exc:                              # noqa: BLE001 - classified below, else re-raised
            kind = classify(exc)
            if kind is None:
                raise
            _note(kind, exc)
            if kind != "oom":
                raise
            failure = exc
    if before_wait is not None:
        try:
            before_wait()
        except Exception as err:                              # noqa: BLE001
            swallowed(f"gpu_recovery.{site}.before_wait", err, "continuing to the drain")
    started = time.perf_counter()
    with gate.exclusive(abort=abort) as alone:
        with _lock:
            _counts["wait_ms"] += (time.perf_counter() - started) * 1000.0
        if not alone:
            raise failure
        flush_gpu_caches()
        with _lock:
            _counts["flush"] += 1
        try:
            result = (retry or first)()
        except Exception as again:                            # noqa: BLE001 - the original is the informative one
            _note("failed_after_retry", again)
            swallowed(f"gpu_recovery.{site}", again, "retry alone also failed; the original frame is written",
                      strict=False)
            raise failure from again
        _note("recovered", failure)
        swallowed(f"gpu_recovery.{site}", failure, "flushed caches and reran the frame alone: recovered",
                  strict=False)
        return result
