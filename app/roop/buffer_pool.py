"""Fixed, reusable NumPy pinned-memory buffers for decoded frames and RGB crops.

Pre-allocates page-locked host memory via PyTorch CUDA pinned memory (or standard
NumPy as fallback) to eliminate continuous memory reallocations and RSS sawteeth
in core video processing worker loops.
"""

from __future__ import annotations

import os
import queue
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np

_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    pass


def is_pinned_supported() -> bool:
    """Check if CUDA pinned host memory is available."""
    if not _HAS_TORCH:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def allocate_pinned_buffer(shape: Tuple[int, ...], dtype: Any = np.uint8) -> np.ndarray:
    """Allocate a fixed pinned-memory NumPy buffer.
    
    Uses PyTorch page-locked pinned host memory when CUDA is available,
    falling back gracefully to standard numpy array.
    """
    if is_pinned_supported() and dtype == np.uint8:
        try:
            tensor = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
            return tensor.numpy()
        except Exception:
            pass
    return np.empty(shape, dtype=dtype)


class PinnedBufferPool:
    """Fixed-capacity pool of pre-allocated pinned-memory buffers for decoded frames."""

    def __init__(self, shape: Tuple[int, ...], capacity: int = 4, dtype: Any = np.uint8):
        self.shape = tuple(int(x) for x in shape)
        self.capacity = max(1, int(capacity))
        self.dtype = dtype
        self._pool: queue.Queue = queue.Queue(maxsize=self.capacity)
        self._lock = threading.Lock()
        self._total_allocated = 0

        # Pre-allocate the fixed buffers
        for _ in range(self.capacity):
            buf = allocate_pinned_buffer(self.shape, self.dtype)
            self._pool.put(buf)
            self._total_allocated += 1

    def acquire(self, timeout: Optional[float] = 0.5) -> np.ndarray:
        """Acquire a pre-allocated pinned buffer from the pool."""
        try:
            return self._pool.get(timeout=timeout)
        except queue.Empty:
            # Fallback allocation if pool is momentarily exhausted
            return allocate_pinned_buffer(self.shape, self.dtype)

    def release(self, buf: np.ndarray) -> None:
        """Return a buffer back to the pool."""
        if buf is None or buf.shape != self.shape:
            return
        try:
            self._pool.put_nowait(buf)
        except queue.Full:
            # Drop excess
            pass

    def clear(self) -> None:
        """Drain the pool."""
        while not self._pool.empty():
            try:
                self._pool.get_nowait()
            except queue.Empty:
                break


class _ThreadLocalCropBuffers(threading.local):
    """Thread-local cache of pre-allocated pinned buffers for standard RGB crop sizes."""

    def __init__(self):
        super().__init__()
        self._crops: Dict[int, np.ndarray] = {}

    def get_crop_buffer(self, size: int) -> np.ndarray:
        buf = self._crops.get(size)
        if buf is None or buf.shape != (size, size, 3):
            buf = allocate_pinned_buffer((size, size, 3), dtype=np.uint8)
            self._crops[size] = buf
        return buf


_TLS_CROP_BUFFERS = _ThreadLocalCropBuffers()


def get_crop_buffer(size: int = 512) -> np.ndarray:
    """Get a pre-allocated, reusable pinned-memory crop buffer for this thread."""
    return _TLS_CROP_BUFFERS.get_crop_buffer(int(size))


_GLOBAL_FRAME_POOLS: Dict[Tuple[int, int], PinnedBufferPool] = {}
_GLOBAL_FRAME_POOLS_LOCK = threading.Lock()


def get_frame_buffer_pool(height: int, width: int, capacity: int = 4) -> PinnedBufferPool:
    """Get or create a frame buffer pool for the specified video dimensions."""
    key = (int(height), int(width))
    with _GLOBAL_FRAME_POOLS_LOCK:
        pool = _GLOBAL_FRAME_POOLS.get(key)
        if pool is None or pool.capacity < capacity:
            pool = PinnedBufferPool(shape=(height, width, 3), capacity=capacity)
            _GLOBAL_FRAME_POOLS[key] = pool
        return pool


def release_frame_buffer_pools() -> None:
    """Release all cached frame buffer pools."""
    with _GLOBAL_FRAME_POOLS_LOCK:
        for pool in _GLOBAL_FRAME_POOLS.values():
            pool.clear()
        _GLOBAL_FRAME_POOLS.clear()
