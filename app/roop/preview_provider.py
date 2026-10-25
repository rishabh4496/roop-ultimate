"""Time-bounded, LRU-cached random-access frames for interactive HTTP previews.

WHO ASKS. `#/faceswap` and `#/extras` fetch single frames over HTTP while the user drags the playhead,
holds the magnifier loupe, or drives the split wipe / diff map: `/api/target/preview`, the filmstrip
grid and `/api/preview`. They are random access, bursty, and a newer request makes every older one
worthless.

WHAT WENT WRONG. Those requests decoded through `capturer.get_video_frame`: one shared `cv2.VideoCapture`
behind one lock, a fallback that shells out to ffmpeg with no timeout, and no limit on how long a request
may wait for either. A single slow seek (4K HEVC, a long GOP, a hung ffmpeg) held every later request until
the React client's 15 s deadline cut them off, which is what the user sees as "Frame unavailable - Retry".

WHAT THIS DOES.
* Every request has a deadline (``DEADLINE_S`` = 2 s, far inside the 15 s API default). Past it the caller
  gets ``PreviewTimeout`` carrying ``retry_after_ms``; the routes turn that into the same
  ``503 {busy: true, retry_after_ms}`` the React frame requester already treats as "ask again", not as a
  failure.
* Decoding runs on this provider's OWN daemon thread, so a request thread never sits inside a decoder lock.
  A decode that outlives its request keeps going and lands in the cache, so the retry is a hit.
* Newest request first. A queued request nobody is waiting for any more is dropped without decoding, and a
  decode whose every waiter has been superseded (``cancel``) is skipped before it starts.
* Identical in-flight requests share one decode.
* A 32-frame LRU, under a byte budget (``ROOP_PREVIEW_CACHE_MB``, default 256) so 32 frames of 4K cannot
  take 800 MB on the 16 GB machine. Frames go in and out as COPIES: the preview path draws mask outlines and
  face boxes onto the array it is handed, and a shared array would burn that overlay into every later reader.

WHAT THIS DOES NOT DO. It does not own a decoder: it calls the ``decode`` function it is given (the app
passes ``capturer.get_video_frame``). The Stage 3 render does not touch that decoder or its lock (the render
reads through ``video_io_hw`` / its own pipe; ``capturer._capture_lock`` is taken only inside capturer.py),
so a scrub cannot queue behind Stage 3, and Stage 3 cannot queue behind a scrub. What was missing was the
bound on waiting, which is what this adds.
"""
from __future__ import annotations

import itertools
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, TimeoutError as _FutureTimeout
from typing import Any, Callable, Dict, Hashable, List, Optional

from roop.degrade import swallowed as _swallowed

CACHE_FRAMES = 32
DEADLINE_S = 2.0
RETRY_AFTER_MS = 500


class FrameSuperseded(Exception):
    """A queued frame request was dropped because the caller asked for a newer frame meanwhile."""


class PreviewTimeout(Exception):
    """No frame within the deadline. The decode may still be running and will land in the cache."""

    def __init__(self, path: str, frame: int, waited_s: float, retry_after_ms: int = RETRY_AFTER_MS) -> None:
        super().__init__(f"no frame {frame} of {os.path.basename(str(path))} within {waited_s:.1f}s")
        self.path, self.frame, self.waited_s, self.retry_after_ms = path, frame, waited_s, int(retry_after_ms)


def _never() -> bool:
    return False


def _budget_bytes() -> int:
    try:
        return max(0, int(os.environ.get("ROOP_PREVIEW_CACHE_MB", "256"))) * 1024 * 1024
    except ValueError:
        return 256 * 1024 * 1024


def _capturer_decode(path: str, frame: int, cancel: Optional[Callable[[], bool]]):
    from roop.capturer import get_video_frame
    return get_video_frame(path, frame, cancel=cancel) if cancel is not None else get_video_frame(path, frame)


class _Job:
    __slots__ = ("key", "path", "frame", "future", "waiters", "cancels", "real_cancel", "started")

    def __init__(self, key: Hashable, path: str, frame: int) -> None:
        self.key, self.path, self.frame = key, path, frame
        self.future: Future = Future()
        self.waiters = 0
        self.cancels: Dict[int, Callable[[], bool]] = {}
        self.real_cancel = False          # some waiter supplied a cancel; else the decoder is called without one
        self.started = False

    def stale(self) -> bool:
        """True only when there is at least one waiter and ALL of them have been superseded."""
        cancels = list(self.cancels.values())
        return bool(cancels) and all(c() for c in cancels)


class InteractiveFramePreviewProvider:
    """``get_frame(path, frame)`` -> BGR ndarray | None, 1-based like ``capturer.get_video_frame``.

    Raises ``PreviewTimeout`` past the deadline and ``FrameSuperseded`` when ``cancel()`` says so. Thread safe."""

    def __init__(self, decode: Optional[Callable[[str, int, Optional[Callable[[], bool]]], Any]] = None, *,
                 cache_frames: int = CACHE_FRAMES, deadline_s: float = DEADLINE_S,
                 budget_bytes: Optional[int] = None, name: str = "preview-decode") -> None:
        self._decode = decode or _capturer_decode
        self.cache_frames = max(1, int(cache_frames))
        self.deadline_s = float(deadline_s)
        self._budget = _budget_bytes() if budget_bytes is None else int(budget_bytes)
        self._name = name
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._pending: List[_Job] = []                  # newest last; popped from the end
        self._inflight: Dict[Hashable, _Job] = {}
        self._cache: "OrderedDict[Hashable, Any]" = OrderedDict()
        self._cache_bytes = 0
        self._ids = itertools.count(1)
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self._counters = {"hits": 0, "decodes": 0, "timeouts": 0, "superseded": 0, "dropped": 0, "shared": 0}

    # ── cache ────────────────────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _key(path: str, frame: int) -> Hashable:
        try:
            return (os.path.abspath(path), os.stat(path).st_mtime_ns, int(frame))
        except OSError:
            return (os.path.abspath(path), None, int(frame))          # None mtime: never cached

    def _cache_get(self, key: Hashable):
        if key[1] is None:
            return None
        with self._lock:
            frame = self._cache.get(key)
            if frame is None:
                return None
            self._cache.move_to_end(key)
            self._counters["hits"] += 1
            return frame.copy()

    def _cache_put(self, key: Hashable, frame) -> None:
        if key[1] is None or frame is None:
            return
        size = int(getattr(frame, "nbytes", 0))
        if self._budget <= 0 or size > self._budget:
            return
        mine = frame.copy()
        with self._lock:
            old = self._cache.pop(key, None)
            if old is not None:
                self._cache_bytes -= int(old.nbytes)
            self._cache[key] = mine
            self._cache_bytes += size
            while len(self._cache) > self.cache_frames or (self._cache_bytes > self._budget and len(self._cache) > 1):
                _, evicted = self._cache.popitem(last=False)
                self._cache_bytes -= int(evicted.nbytes)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._cache_bytes = 0

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counters, cached=len(self._cache), cached_bytes=self._cache_bytes,
                        queued=len(self._pending), inflight=len(self._inflight))

    # ── requests ─────────────────────────────────────────────────────────────────────────────────────────
    def get_frame(self, path: str, frame: int, cancel: Optional[Callable[[], bool]] = None,
                  deadline_s: Optional[float] = None):
        budget = self.deadline_s if deadline_s is None else float(deadline_s)
        started = time.monotonic()
        end = started + max(0.0, budget)
        key = self._key(path, frame)
        mine = cancel or _never
        while True:
            hit = self._cache_get(key)
            if hit is not None:
                return hit
            if mine():
                self._count("superseded")
                raise FrameSuperseded(frame)
            remaining = end - time.monotonic()
            if remaining <= 0:
                self._count("timeouts")
                raise PreviewTimeout(path, frame, time.monotonic() - started)
            job, token = self._attach(key, path, frame, cancel)
            try:
                result = job.future.result(timeout=remaining)
            except _FutureTimeout:
                self._count("timeouts")
                raise PreviewTimeout(path, frame, time.monotonic() - started) from None
            except FrameSuperseded:
                if mine():
                    self._count("superseded")
                    raise
                continue                          # the decode was skipped for OTHER waiters; we still want it
            finally:
                self._detach(job, token)
            return None if result is None else result.copy()

    def _count(self, name: str) -> None:
        with self._lock:
            self._counters[name] += 1

    def _attach(self, key: Hashable, path: str, frame: int, cancel: Optional[Callable[[], bool]]):
        with self._cv:
            if self._closed:
                raise RuntimeError("preview provider is closed")
            job = self._inflight.get(key)
            if job is None or job.future.done():
                job = _Job(key, path, frame)
                self._inflight[key] = job
                self._pending.append(job)
                self._ensure_thread()
            else:
                self._counters["shared"] += 1
            token = next(self._ids)
            job.waiters += 1
            job.cancels[token] = cancel or _never
            job.real_cancel = job.real_cancel or cancel is not None
            self._cv.notify()
            return job, token

    def _detach(self, job: _Job, token: int) -> None:
        with self._cv:
            job.waiters -= 1
            job.cancels.pop(token, None)

    # ── worker ───────────────────────────────────────────────────────────────────────────────────────────
    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
            self._thread.start()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._pending and not self._closed:
                    self._cv.wait()
                if self._closed:
                    return
                job = self._pending.pop()                         # newest first: it is the one on screen
                if job.waiters <= 0:                              # every asker gave up before we started
                    self._inflight.pop(job.key, None)
                    self._counters["dropped"] += 1
                    job.future.set_exception(FrameSuperseded(job.frame))
                    continue
                job.started = True
            self._execute(job)

    def _execute(self, job: _Job) -> None:
        try:
            if job.stale():
                raise FrameSuperseded(job.frame)
            frame = self._cache_get(job.key)                      # an earlier job for this frame may have landed
            if frame is None:
                frame = self._decode(job.path, job.frame, job.stale if job.real_cancel else None)
                with self._lock:
                    self._counters["decodes"] += 1
                self._cache_put(job.key, frame)
            outcome: Any = (frame, None)
        except FrameSuperseded as exc:                            # flow control, not an error: say nothing
            outcome = (None, exc)
        except Exception as exc:                                  # delivered to the waiters; must not kill the worker
            _swallowed("roop/preview_provider.py:decode", exc, "the asking request gets this error; the worker lives on")
            outcome = (None, exc)
        finally:
            with self._lock:
                if self._inflight.get(job.key) is job:
                    del self._inflight[job.key]
        if outcome[1] is not None:
            job.future.set_exception(outcome[1])
        else:
            job.future.set_result(outcome[0])
