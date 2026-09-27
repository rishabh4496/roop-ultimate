"""Zero-copy shared-memory frame transport between processes.

:class:`SharedMemoryRingBuffer`
    One ``multiprocessing.shared_memory.SharedMemory`` block holds ``slots``
    frame buffers plus a small int64 header. Producers get a NumPy view of a
    free slot (``np.ndarray(shape, dtype, buffer=shm.buf, offset=...)``),
    write into it in place, and commit it with a sequence number; consumers
    get a view of the oldest committed slot and release it when done. No frame
    is ever copied between processes.

    With several consumers, slots are released out of order, which a plain
    circular buffer cannot represent. So the ring is a FIFO ring of *slot
    indices* over a pool of slots: ``free``/``ready`` semaphores count slots in
    each state, a lock guards the header (head/tail counters, per-slot state
    and sequence number), and commit order is delivery order.

:class:`FramePipeline`
    Ingestion (a decode process), N inference processes, and assembly (the
    calling process, in frame order) connected by two rings.

Shutdown and deadlocks
    roop-ultimate's rule for queue hand-offs: bound every wait a dead
    counterpart would park forever; never drop the message that tells a live
    counterpart to stop. So every blocking call takes a timeout, the pipeline
    polls child liveness between waits, ``close()`` wakes consumers with a
    token that each re-posts, and ``abort()`` (callable from any process)
    wakes every waiter with :class:`RingAborted`.

Cleanup guarantees
    * Normal exit, exceptions, SIGINT / SIGTERM (and SIGBREAK on Windows):
      :func:`cleanup_all` closes every segment this process mapped and
      unlinks every segment it created (``atexit`` + signal handlers,
      installed by :func:`install_cleanup_handlers`).
    * Hard crashes (SIGSEGV, SIGKILL, TerminateProcess): no Python code can
      run, and a Python-level SIGSEGV handler is actively harmful (the fault
      re-executes; the process can hang instead of dying), so none is
      installed. The guarantee comes from below Python: on Windows a named
      mapping is destroyed by the OS when its last handle closes; on POSIX,
      ``multiprocessing``'s resource tracker (a separate process) unlinks
      segments whose creator died. ``faulthandler`` is enabled for a
      traceback. Both behaviours are tested by killing a process outright.
    * Attaching processes must not unlink: on POSIX before Python 3.13 an
      attach also registered the segment with the attacher's resource
      tracker, which unlinked it when the ATTACHER exited; attachments are
      unregistered from the tracker to avoid that.
"""
from __future__ import annotations

import atexit
import faulthandler
import gc
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

FREE, WRITING, READY, READING = 0, 1, 2, 3
_HEAD, _TAIL, _CLOSED, _ABORTED = 0, 1, 2, 3
_FIXED = 4  # header words before the per-slot arrays

# --------------------------------------------------------------------------- lifecycle
_REGISTRY_LOCK = threading.Lock()
_OWNED: dict[str, shared_memory.SharedMemory] = {}
_ATTACHED: dict[str, shared_memory.SharedMemory] = {}
_HANDLERS_INSTALLED = False


def _close(shm: shared_memory.SharedMemory) -> None:
    try:
        shm.close()
    except BufferError:
        # A NumPy view still exports the buffer; drop stray views and retry.
        gc.collect()
        try:
            shm.close()
        except BufferError:
            logger.warning("shared memory %s still has live views; leaving it mapped", shm.name)


def cleanup_all() -> None:
    """Close every segment this process mapped; unlink the ones it created."""
    with _REGISTRY_LOCK:
        attached, owned = list(_ATTACHED.values()), list(_OWNED.values())
        _ATTACHED.clear()
        _OWNED.clear()
    for shm in attached:
        _close(shm)
    for shm in owned:
        _close(shm)
        try:
            shm.unlink()  # no-op on Windows: the OS frees it with the last handle
        except FileNotFoundError:
            pass


def install_cleanup_handlers() -> None:
    """``atexit`` + SIGINT/SIGTERM(/SIGBREAK) handlers that run :func:`cleanup_all`
    and then defer to the previous handler. Idempotent; main thread only."""
    global _HANDLERS_INSTALLED
    if _HANDLERS_INSTALLED:
        return
    _HANDLERS_INSTALLED = True
    atexit.register(cleanup_all)
    if not faulthandler.is_enabled():
        try:
            faulthandler.enable()
        except (RuntimeError, ValueError, OSError):  # no usable stderr
            pass
    if threading.current_thread() is not threading.main_thread():
        return
    names = ["SIGINT", "SIGTERM"] + (["SIGBREAK"] if sys.platform == "win32" else [])
    for name in names:
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        previous = signal.getsignal(signum)

        def handler(sig: int, frame: Any, _previous: Any = previous) -> None:
            cleanup_all()
            if callable(_previous):
                _previous(sig, frame)
            elif _previous == signal.SIG_IGN:
                return
            else:
                signal.signal(sig, signal.SIG_DFL)
                os.kill(os.getpid(), sig)

        signal.signal(signum, handler)


def create_owned(size: int) -> shared_memory.SharedMemory:
    """A new segment this process owns: closed and unlinked by :func:`cleanup_all`
    (normal exit, exceptions, SIGINT/SIGTERM/SIGBREAK)."""
    shm = shared_memory.SharedMemory(create=True, size=size)
    with _REGISTRY_LOCK:
        _OWNED[shm.name] = shm
    return shm


def attach_tracked(name: str) -> shared_memory.SharedMemory:
    """Attach to another process's segment; closed (never unlinked) by :func:`cleanup_all`."""
    shm = shared_memory.SharedMemory(name=name)
    _untrack(shm)
    with _REGISTRY_LOCK:
        _ATTACHED[shm.name] = shm
    return shm


def release(shm: shared_memory.SharedMemory, *, unlink: bool) -> None:
    """Close (and optionally unlink) a segment now and forget it."""
    with _REGISTRY_LOCK:
        _OWNED.pop(shm.name, None)
        _ATTACHED.pop(shm.name, None)
    _close(shm)
    if unlink:
        try:
            shm.unlink()
        except FileNotFoundError:
            pass


def _untrack(shm: shared_memory.SharedMemory) -> None:
    if os.name == "nt":
        return
    try:
        from multiprocessing import resource_tracker

        resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - best effort; only affects POSIX < 3.13
        pass


# --------------------------------------------------------------------------- ring
class RingAborted(RuntimeError):
    """The ring was aborted by some process."""


@dataclass(frozen=True)
class Slot:
    """A borrowed slot: ``array`` is a view into shared memory."""

    index: int
    seq: int
    array: np.ndarray


class SharedMemoryRingBuffer:
    """Fixed-shape frame slots in one shared-memory block.

    Create it in the parent, pass it to ``multiprocessing.Process`` args (the
    sync primitives travel with it), and it attaches on the other side.

    Args:
        slots: Number of frame buffers.
        frame_shape: Shape of one frame, e.g. ``(1080, 1920, 3)``.
        dtype: Element type (``uint8`` for BGR frames).
        ctx: A multiprocessing context (default: ``spawn``, the only one on Windows).
    """

    def __init__(self, slots: int, frame_shape: tuple[int, ...], dtype: Any = np.uint8,
                 ctx: Any = None) -> None:
        if slots < 1:
            raise ValueError("slots must be >= 1")
        ctx = ctx or mp.get_context("spawn")
        self.slots = int(slots)
        self.frame_shape = tuple(int(d) for d in frame_shape)
        self.dtype = np.dtype(dtype)
        self.frame_nbytes = int(np.prod(self.frame_shape)) * self.dtype.itemsize
        self._header_words = _FIXED + 3 * self.slots  # fixed + ring + state + seq
        self._data_offset = -(-self._header_words * 8 // 64) * 64  # 64-byte aligned
        size = self._data_offset + self.slots * self.frame_nbytes
        install_cleanup_handlers()
        self._shm = shared_memory.SharedMemory(create=True, size=size)
        self.name = self._shm.name
        with _REGISTRY_LOCK:
            _OWNED[self.name] = self._shm
        self._owner = True
        self._lock = ctx.Lock()
        self._free = ctx.Semaphore(self.slots)
        self._ready = ctx.Semaphore(0)
        self._header[:] = 0

    # pickling: carry the name and primitives, re-attach on the other side
    def __getstate__(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("slots", "frame_shape", "dtype", "frame_nbytes",
                                              "_header_words", "_data_offset", "name", "_lock",
                                              "_free", "_ready")}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        install_cleanup_handlers()
        self._shm = shared_memory.SharedMemory(name=self.name)
        _untrack(self._shm)
        with _REGISTRY_LOCK:
            _ATTACHED[self.name] = self._shm
        self._owner = False

    # ------------------------------------------------------------------ views
    @property
    def _header(self) -> np.ndarray:
        return np.ndarray((self._header_words,), dtype=np.int64, buffer=self._shm.buf)

    def view(self, index: int) -> np.ndarray:
        """Zero-copy view of slot ``index``."""
        return np.ndarray(self.frame_shape, dtype=self.dtype, buffer=self._shm.buf,
                          offset=self._data_offset + index * self.frame_nbytes)

    def _state(self, h: np.ndarray) -> np.ndarray:
        return h[_FIXED + self.slots:_FIXED + 2 * self.slots]

    def _seq(self, h: np.ndarray) -> np.ndarray:
        return h[_FIXED + 2 * self.slots:_FIXED + 3 * self.slots]

    def _ring(self, h: np.ndarray) -> np.ndarray:
        return h[_FIXED:_FIXED + self.slots]

    @property
    def aborted(self) -> bool:
        return bool(self._header[_ABORTED])

    @property
    def closed(self) -> bool:
        return bool(self._header[_CLOSED])

    def pending(self) -> int:
        """Committed slots not yet taken by a consumer."""
        with self._lock:
            h = self._header
            return int(h[_TAIL] - h[_HEAD])

    # ------------------------------------------------------------------ producer
    def acquire_write(self, timeout: float | None = None) -> Slot:
        """Borrow a free slot to write into. Raises ``TimeoutError`` / :class:`RingAborted`."""
        if not self._free.acquire(timeout=timeout):
            self._check_abort()
            raise TimeoutError("no free slot")
        if self.aborted:
            self._free.release()  # pass the wake-up on
            raise RingAborted("ring aborted")
        with self._lock:
            h = self._header
            state = self._state(h)
            free = np.nonzero(state == FREE)[0]
            if free.size == 0:  # cannot happen while the semaphore is honest
                raise RuntimeError("free semaphore out of step with slot states")
            index = int(free[0])
            state[index] = WRITING
        return Slot(index, -1, self.view(index))

    def commit(self, slot: Slot | int, seq: int) -> None:
        """Publish a written slot with its sequence number (FIFO delivery)."""
        index = slot.index if isinstance(slot, Slot) else int(slot)
        with self._lock:
            h = self._header
            if self._state(h)[index] != WRITING:
                raise RuntimeError(f"slot {index} was not borrowed for writing")
            self._state(h)[index] = READY
            self._seq(h)[index] = seq
            ring = self._ring(h)
            ring[h[_TAIL] % self.slots] = index
            h[_TAIL] += 1
        self._ready.release()

    def put(self, frame: np.ndarray, seq: int, timeout: float | None = None) -> None:
        """Copy ``frame`` into a slot and commit it (for callers without a view)."""
        slot = self.acquire_write(timeout)
        slot.array[...] = frame
        self.commit(slot, seq)

    def close(self) -> None:
        """No more commits: consumers drain what is left, then get ``None``."""
        with self._lock:
            self._header[_CLOSED] = 1
        self._ready.release()  # one token; each consumer that finds the ring empty re-posts it

    def abort(self) -> None:
        """Wake every waiter in every process with :class:`RingAborted`."""
        with self._lock:
            self._header[_ABORTED] = 1
        self._ready.release()
        self._free.release()

    def _check_abort(self) -> None:
        if self.aborted:
            raise RingAborted("ring aborted")

    # ------------------------------------------------------------------ consumer
    def acquire_read(self, timeout: float | None = None) -> Slot | None:
        """Borrow the oldest committed slot; ``None`` once closed and drained.

        Raises ``TimeoutError`` if nothing arrives in time, :class:`RingAborted` on abort.
        """
        if not self._ready.acquire(timeout=timeout):
            self._check_abort()
            raise TimeoutError("no frame committed")
        if self.aborted:
            self._ready.release()
            raise RingAborted("ring aborted")
        with self._lock:
            h = self._header
            if h[_HEAD] == h[_TAIL]:
                if h[_CLOSED]:
                    self._ready.release()
                    return None
                raise RuntimeError("ready semaphore out of step with the ring")
            index = int(self._ring(h)[h[_HEAD] % self.slots])
            h[_HEAD] += 1
            self._state(h)[index] = READING
            seq = int(self._seq(h)[index])
        return Slot(index, seq, self.view(index))

    def release(self, slot: Slot | int) -> None:
        """Return a read slot to the free pool."""
        index = slot.index if isinstance(slot, Slot) else int(slot)
        with self._lock:
            state = self._state(self._header)
            if state[index] != READING:
                raise RuntimeError(f"slot {index} was not borrowed for reading")
            state[index] = FREE
        self._free.release()

    def get(self, timeout: float | None = None) -> tuple[int, np.ndarray] | None:
        """Copy out the oldest frame and release its slot: ``(seq, frame)`` or ``None``."""
        slot = self.acquire_read(timeout)
        if slot is None:
            return None
        try:
            return slot.seq, slot.array.copy()
        finally:
            self.release(slot)

    # ------------------------------------------------------------------ teardown
    def destroy(self) -> None:
        """Close this process's mapping; the creator also unlinks."""
        with _REGISTRY_LOCK:
            (_OWNED if self._owner else _ATTACHED).pop(self.name, None)
        _close(self._shm)
        if self._owner:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> SharedMemoryRingBuffer:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.destroy()


# --------------------------------------------------------------------------- pipeline
class PipelineError(RuntimeError):
    """A pipeline stage failed; the message carries the child's traceback."""


class PipelineCancelled(PipelineError):
    """``run()`` was cancelled through its ``cancel`` event."""


def _patient(call: Callable[[float], Any], timeout: float) -> Any:
    """Retry a timed wait until it succeeds or the ring aborts.

    A timeout inside a stage is not an error (decoding or inference can be
    slow); the PARENT watches liveness and aborts the rings if a stage dies,
    which turns every wait into :class:`RingAborted`.
    """
    while True:
        try:
            return call(timeout)
        except TimeoutError:
            continue


def _ingest_main(frames: Callable[[], Iterable[tuple[int, np.ndarray]]], raw: SharedMemoryRingBuffer,
                 errors: Any, timeout: float) -> None:
    try:
        for seq, frame in frames():
            slot = _patient(raw.acquire_write, timeout)
            slot.array[...] = frame
            raw.commit(slot, seq)
        raw.close()
    except RingAborted:
        pass
    except BaseException:  # noqa: BLE001 - report everything, then abort both sides
        errors.put(("ingest", traceback.format_exc()))
        raw.abort()


def _worker_main(worker: Callable[[np.ndarray, np.ndarray, int], None],
                 init: Callable[[], Any] | None, raw: SharedMemoryRingBuffer,
                 out: SharedMemoryRingBuffer, errors: Any, timeout: float, wid: int) -> None:
    try:
        state = init() if init is not None else None
        while True:
            src = _patient(raw.acquire_read, timeout)
            if src is None:
                break
            dst = _patient(out.acquire_write, timeout)
            try:
                worker(src.array, dst.array, src.seq) if state is None else \
                    worker(src.array, dst.array, src.seq, state)  # type: ignore[call-arg]
            finally:
                raw.release(src)
            out.commit(dst, src.seq)
    except RingAborted:
        pass
    except BaseException:  # noqa: BLE001
        errors.put((f"worker {wid}", traceback.format_exc()))
        raw.abort()
        out.abort()


class FramePipeline:
    """Ingest process -> N inference processes -> ordered assembly in the caller.

    Args:
        frames: Picklable zero-argument callable returning an iterable of
            ``(seq, frame)`` with ``seq`` = 0, 1, 2, ... (run in the ingest process).
        worker: Picklable ``worker(src, dst, seq[, state])`` writing the result
            for ``src`` into ``dst`` (both zero-copy views).
        frame_shape / out_shape: Input and output frame shapes.
        workers: Inference processes.
        slots: Slots per ring.
        init: Optional picklable per-worker initializer (e.g. to build ONNX
            sessions); its return value is passed to ``worker`` as ``state``.
        timeout: Seconds any single wait may take before liveness is re-checked.
    """

    def __init__(self, frames: Callable[[], Iterable[tuple[int, np.ndarray]]],
                 worker: Callable[..., None], frame_shape: tuple[int, ...],
                 out_shape: tuple[int, ...] | None = None, workers: int = 2, slots: int = 8,
                 init: Callable[[], Any] | None = None, timeout: float = 1.0,
                 stall_timeout: float = 120.0) -> None:
        self.frames = frames
        self.worker = worker
        self.frame_shape = frame_shape
        self.out_shape = out_shape or frame_shape
        self.workers = workers
        self.slots = max(slots, workers + 1)
        self.init = init
        self.timeout = timeout
        self.stall_timeout = stall_timeout

    def run(self, sink: Callable[[int, np.ndarray], None],
            cancel: threading.Event | None = None) -> int:
        """Run to completion, calling ``sink(seq, frame)`` in sequence order.

        ``frame`` is a view valid only during the call. Returns frames delivered.
        Setting ``cancel`` stops within one ``timeout``: the rings are aborted,
        the children joined (terminated if they do not exit), and every segment
        released, then :class:`PipelineCancelled` is raised.

        Raises:
            PipelineError: a stage raised, died, or made no progress for
                ``stall_timeout`` seconds.
        """
        ctx = mp.get_context("spawn")
        errors = ctx.Queue()
        raw = SharedMemoryRingBuffer(self.slots, self.frame_shape, ctx=ctx)
        out = SharedMemoryRingBuffer(self.slots, self.out_shape, ctx=ctx)
        procs = [ctx.Process(target=_ingest_main, args=(self.frames, raw, errors, self.timeout),
                             name="face-engine-ingest", daemon=True)]
        procs += [ctx.Process(target=_worker_main,
                              args=(self.worker, self.init, raw, out, errors, self.timeout, i),
                              name=f"face-engine-worker-{i}", daemon=True)
                  for i in range(self.workers)]
        delivered = 0
        pending: dict[int, np.ndarray] = {}
        try:
            for p in procs:
                p.start()
            workers_done = False
            last_progress = time.monotonic()
            while True:
                if cancel is not None and cancel.is_set():
                    raise PipelineCancelled(f"cancelled after {delivered} frames")
                self._raise_child_errors(errors, raw, out)
                dead = [p for p in procs if p.exitcode not in (None, 0)]
                if dead:
                    time.sleep(0.2)  # let a reported traceback arrive first
                    self._raise_child_errors(errors, raw, out)
                    raise PipelineError(f"{dead[0].name} died with exit code {dead[0].exitcode}")
                if not workers_done and not any(p.is_alive() for p in procs[1:]):
                    workers_done = True
                    self._raise_child_errors(errors, raw, out)
                    out.close()
                try:
                    slot = out.acquire_read(self.timeout)
                except TimeoutError:
                    if time.monotonic() - last_progress > self.stall_timeout:
                        raise PipelineError(f"no frame for {self.stall_timeout}s") from None
                    continue
                if slot is None:
                    break
                last_progress = time.monotonic()
                try:
                    if slot.seq == delivered:
                        sink(slot.seq, slot.array)
                        delivered += 1
                    else:
                        # Out of order: copy out and free the slot at once. Holding it
                        # while waiting for an earlier frame could leave every slot
                        # full of later frames and no room for the missing one.
                        pending[slot.seq] = slot.array.copy()
                finally:
                    out.release(slot)
                while delivered in pending:
                    sink(delivered, pending.pop(delivered))
                    delivered += 1
            if pending:
                raise PipelineError(f"frames missing before {sorted(pending)[:5]}")
            self._raise_child_errors(errors, raw, out)
            return delivered
        except RingAborted:
            # A stage aborted the rings; its traceback is (about to be) on the queue.
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                self._raise_child_errors(errors, raw, out)
                time.sleep(0.05)
            raise PipelineError("a stage aborted the pipeline without reporting why") from None
        except BaseException:
            raw.abort()
            out.abort()
            raise
        finally:
            for p in procs:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
            raw.destroy()
            out.destroy()

    @staticmethod
    def _raise_child_errors(errors: Any, raw: SharedMemoryRingBuffer,
                            out: SharedMemoryRingBuffer) -> None:
        try:
            stage, tb = errors.get_nowait()
        except Exception:  # noqa: BLE001 - queue.Empty
            return
        raw.abort()
        out.abort()
        raise PipelineError(f"{stage} failed:\n{tb}")


@dataclass(frozen=True)
class VideoFrames:
    """Picklable frame source for :class:`FramePipeline`: frames ``[start, end)``
    of a video, numbered from 0."""

    path: str
    start: int = 0
    end: int | None = None
    hwaccel: str | None = None

    def __call__(self) -> Iterable[tuple[int, np.ndarray]]:
        from face_engine.media.capturer import VideoSource

        source = VideoSource(self.path, hwaccel=self.hwaccel)
        for frame in source.frames(self.start, self.end):
            yield frame.index - self.start, frame.image
