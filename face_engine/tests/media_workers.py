"""Top-level callables for FramePipeline tests (spawned processes import them by name)."""
from __future__ import annotations

import os
import sys
import time

import numpy as np


def invert(src: np.ndarray, dst: np.ndarray, seq: int) -> None:
    np.subtract(255, src, out=dst)


def slow_invert(src: np.ndarray, dst: np.ndarray, seq: int) -> None:
    # Uneven latency so workers finish out of order.
    time.sleep(0.002 * ((seq * 7) % 5))
    np.subtract(255, src, out=dst)


def fail_at_five(src: np.ndarray, dst: np.ndarray, seq: int) -> None:
    if seq == 5:
        raise ValueError("simulated inference failure at frame 5")
    dst[...] = src


def die_at_five(src: np.ndarray, dst: np.ndarray, seq: int) -> None:
    if seq == 5:
        os._exit(3)  # no Python cleanup at all, like a native crash
    dst[...] = src


class Counting:
    """Picklable frame source: ``n`` frames whose pixels equal their index."""

    def __init__(self, n: int, shape: tuple[int, ...]) -> None:
        self.n, self.shape = n, shape

    def __call__(self):  # type: ignore[no-untyped-def]
        for i in range(self.n):
            yield i, np.full(self.shape, i % 256, np.uint8)


def hold_ring_forever() -> None:
    """Child for the hard-kill test: create a ring, print its name, wait."""
    from face_engine.media.ipc_pool import SharedMemoryRingBuffer

    ring = SharedMemoryRingBuffer(2, (64, 64, 3))
    print(ring.name, flush=True)
    time.sleep(600)


def cleanup_on_sigint() -> None:
    """Child for the signal test: SIGINT must close + unlink before the default handler runs."""
    import signal
    from multiprocessing import shared_memory

    from face_engine.media import ipc_pool

    ring = ipc_pool.SharedMemoryRingBuffer(2, (64, 64, 3))
    name = ring.name
    try:
        signal.raise_signal(signal.SIGINT)
    except KeyboardInterrupt:
        pass
    released = not ipc_pool._OWNED
    try:
        shared_memory.SharedMemory(name=name).close()
        attachable = True
    except FileNotFoundError:
        attachable = False
    print(f"released={released} attachable={attachable}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    {"hold": hold_ring_forever, "sigint": cleanup_on_sigint}[sys.argv[1]]()
