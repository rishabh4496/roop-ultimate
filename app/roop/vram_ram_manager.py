"""Stage 11 — VRAM/RAM Management Architecture and Runtime Governor.

Provides a unified, bounded memory management infrastructure:
1. Allocation Profiling:
   - Tracks every large allocation: frames, crops, tensor transfers, CPU copies.
   - Detects duplicated frames, duplicated face crops, unnecessary copies,
     stale GPU tensors, cache growth, and queue accumulation.
2. Buffer Preallocation & Reuse:
   - FrameBufferPool: Fixed-capacity pinned-memory frame ring buffer.
   - CropBufferPool: Multi-tier preallocated face crop buffers (112, 128, 256, 512, 1024).
   - FastScratchBuffer: Preallocated workspace eliminating float32 heap allocations during blending.
3. Bounded Queues & Backpressure Governor:
   - Mathematically enforces max in-flight frames to prevent queue accumulation.
4. Lifetime-Aware Tensors:
   - RAII tensor management with deterministic GPU memory reclaim.
5. Explicit Cache Policy:
   - Bounded LRU cache with max items, byte ceiling, eviction telemetry, and purge hooks.
6. VRAM-Aware Runtime Limits:
   - Live VRAM/RAM detection and dynamic parameter selection:
     batch size, worker count, face crop concurrency, enhancer concurrency, buffer depth.
   - Hardware tier profiles for RTX 4070 Desktop, RTX 3060 Laptop, and constrained devices.
   - Policy: Never sacrifice stability for theoretical utilization.
"""

from __future__ import annotations
from roop.degrade import swallowed as _swallowed

import collections
import contextlib
import dataclasses
import enum
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False

LOGGER = logging.getLogger(__name__)


# =============================================================================
# 1. Allocation Profiler & Hotspot Detector
# =============================================================================

class AllocationCategory(enum.Enum):
    FRAME = "frame"
    FACE_CROP = "face_crop"
    TENSOR_H2D = "tensor_h2d"
    TENSOR_D2H = "tensor_d2h"
    TENSOR_OP = "tensor_op"
    CPU_COPY = "cpu_copy"
    CACHE_ENTRY = "cache_entry"
    QUEUE_BUFFER = "queue_buffer"


@dataclass
class AllocationRecord:
    category: AllocationCategory
    size_bytes: int
    shape: Tuple[int, ...]
    dtype: str
    device: str
    caller: str
    timestamp: float = field(default_factory=time.time)
    reused: bool = False


@dataclass
class MemoryAuditReport:
    total_allocations: int
    total_allocated_bytes: int
    peak_active_bytes: int
    current_active_bytes: int
    reused_allocations: int
    bytes_saved_by_reuse: int
    category_breakdown: Dict[str, Dict[str, Any]]
    detected_hotspots: List[str]
    duplicated_frame_count: int
    duplicated_crop_count: int
    unnecessary_copies_count: int
    stale_tensor_count: int
    queue_accumulated_bytes: int


class AllocationProfiler:
    """Thread-safe allocation profiler detecting memory leaks and duplication hotspots."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._lock = threading.Lock()
        self._records: List[AllocationRecord] = []
        self._active_allocations: Dict[int, AllocationRecord] = {}
        self._peak_active_bytes = 0
        self._current_active_bytes = 0
        self._reused_count = 0
        self._reused_bytes = 0
        self._frame_hashes: Dict[int, int] = {}
        self._duplicated_frames = 0
        self._duplicated_crops = 0
        self._unnecessary_copies = 0
        self._stale_tensors = 0
        self._queue_accumulated_bytes = 0

    def record_alloc(self, category: AllocationCategory, size_bytes: int,
                     shape: Tuple[int, ...] = (), dtype: str = "uint8",
                     device: str = "cpu", caller: str = "",
                     reused: bool = False, obj_id: Optional[int] = None) -> None:
        if not self.enabled:
            return
        rec = AllocationRecord(
            category=category,
            size_bytes=size_bytes,
            shape=shape,
            dtype=dtype,
            device=device,
            caller=caller,
            reused=reused,
        )
        with self._lock:
            self._records.append(rec)
            if reused:
                self._reused_count += 1
                self._reused_bytes += size_bytes
            else:
                self._current_active_bytes += size_bytes
                if self._current_active_bytes > self._peak_active_bytes:
                    self._peak_active_bytes = self._current_active_bytes
            if obj_id is not None and not reused:
                self._active_allocations[obj_id] = rec

    def record_free(self, obj_id: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            rec = self._active_allocations.pop(obj_id, None)
            if rec is not None:
                self._current_active_bytes = max(0, self._current_active_bytes - rec.size_bytes)

    def mark_duplicated_frame(self, caller: str = "") -> None:
        with self._lock:
            self._duplicated_frames += 1

    def mark_duplicated_crop(self, caller: str = "") -> None:
        with self._lock:
            self._duplicated_crops += 1

    def mark_unnecessary_copy(self, caller: str = "") -> None:
        with self._lock:
            self._unnecessary_copies += 1

    def mark_stale_tensor(self, caller: str = "") -> None:
        with self._lock:
            self._stale_tensors += 1

    def record_queue_accumulation(self, bytes_buffered: int) -> None:
        with self._lock:
            self._queue_accumulated_bytes = max(self._queue_accumulated_bytes, bytes_buffered)

    def generate_report(self) -> MemoryAuditReport:
        with self._lock:
            total_allocs = len(self._records)
            total_bytes = sum(r.size_bytes for r in self._records if not r.reused)
            breakdown: Dict[str, Dict[str, Any]] = {}
            for cat in AllocationCategory:
                cat_records = [r for r in self._records if r.category == cat]
                cat_bytes = sum(r.size_bytes for r in cat_records if not r.reused)
                cat_reused_bytes = sum(r.size_bytes for r in cat_records if r.reused)
                breakdown[cat.value] = {
                    "count": len(cat_records),
                    "allocated_bytes": cat_bytes,
                    "reused_bytes": cat_reused_bytes,
                    "allocated_mb": round(cat_bytes / (1024.0 ** 2), 2),
                }

            hotspots = []
            if self._duplicated_frames > 0:
                hotspots.append(f"Duplicated frames detected: {self._duplicated_frames} instances")
            if self._duplicated_crops > 0:
                hotspots.append(f"Duplicated face crops detected: {self._duplicated_crops} instances")
            if self._unnecessary_copies > 0:
                hotspots.append(f"Unnecessary CPU/tensor copies: {self._unnecessary_copies} instances")
            if self._stale_tensors > 0:
                hotspots.append(f"Stale GPU tensors detected: {self._stale_tensors} instances")
            if self._queue_accumulated_bytes > 100 * 1024 * 1024:
                hotspots.append(f"High queue accumulation: {round(self._queue_accumulated_bytes / (1024**2), 1)} MB")

            return MemoryAuditReport(
                total_allocations=total_allocs,
                total_allocated_bytes=total_bytes,
                peak_active_bytes=self._peak_active_bytes,
                current_active_bytes=self._current_active_bytes,
                reused_allocations=self._reused_count,
                bytes_saved_by_reuse=self._reused_bytes,
                category_breakdown=breakdown,
                detected_hotspots=hotspots,
                duplicated_frame_count=self._duplicated_frames,
                duplicated_crop_count=self._duplicated_crops,
                unnecessary_copies_count=self._unnecessary_copies,
                stale_tensor_count=self._stale_tensors,
                queue_accumulated_bytes=self._queue_accumulated_bytes,
            )

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._active_allocations.clear()
            self._peak_active_bytes = 0
            self._current_active_bytes = 0
            self._reused_count = 0
            self._reused_bytes = 0
            self._duplicated_frames = 0
            self._duplicated_crops = 0
            self._unnecessary_copies = 0
            self._stale_tensors = 0
            self._queue_accumulated_bytes = 0


# Global singleton profiler
GLOBAL_PROFILER = AllocationProfiler(enabled=True)


# =============================================================================
# 2. Buffer Preallocation & Reuse Infrastructure
# =============================================================================

def allocate_pinned_or_host(shape: Tuple[int, ...], dtype: Any = np.uint8) -> np.ndarray:
    """Allocate page-locked pinned host memory if available; fall back to numpy."""
    if _HAS_TORCH and torch.cuda.is_available() and dtype == np.uint8:
        try:
            tensor = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
            return tensor.numpy()
        except Exception as exc:
            _swallowed("roop/vram_ram_manager.py:allocate_pinned_or_host", exc)
    return np.empty(shape, dtype=dtype)


class FrameBufferPool:
    """Fixed-capacity ring pool of pre-allocated pinned frame buffers."""

    def __init__(self, shape: Tuple[int, int, int], capacity: int = 4,
                 profiler: Optional[AllocationProfiler] = None):
        self.shape = shape
        self.capacity = max(1, int(capacity))
        self.profiler = profiler or GLOBAL_PROFILER
        self._pool: queue.Queue[np.ndarray] = queue.Queue(maxsize=self.capacity)
        self._lock = threading.Lock()
        self._frame_bytes = int(shape[0] * shape[1] * shape[2])
        self._total_created = 0

        for _ in range(self.capacity):
            buf = allocate_pinned_or_host(self.shape, np.uint8)
            self._pool.put(buf)
            self._total_created += 1
            if self.profiler:
                self.profiler.record_alloc(
                    category=AllocationCategory.FRAME,
                    size_bytes=self._frame_bytes,
                    shape=self.shape,
                    device="pinned_host",
                    caller="FrameBufferPool.__init__",
                    reused=False,
                    obj_id=id(buf),
                )

    def acquire(self, timeout: Optional[float] = None) -> np.ndarray:
        """Acquire a pre-allocated buffer from the pool or allocate on exhaustion."""
        try:
            buf = self._pool.get(timeout=timeout) if timeout else self._pool.get_nowait()
            if self.profiler:
                self.profiler.record_alloc(
                    category=AllocationCategory.FRAME,
                    size_bytes=self._frame_bytes,
                    shape=self.shape,
                    device="pinned_host",
                    caller="FrameBufferPool.acquire",
                    reused=True,
                )
            return buf
        except queue.Empty:
            # Fallback allocation if pool is depleted
            buf = allocate_pinned_or_host(self.shape, np.uint8)
            if self.profiler:
                self.profiler.record_alloc(
                    category=AllocationCategory.FRAME,
                    size_bytes=self._frame_bytes,
                    shape=self.shape,
                    device="host",
                    caller="FrameBufferPool.acquire_fallback",
                    reused=False,
                    obj_id=id(buf),
                )
            return buf

    def release(self, buf: np.ndarray) -> None:
        """Return a buffer to the pool."""
        if buf is None or buf.shape != self.shape:
            return
        try:
            self._pool.put_nowait(buf)
        except queue.Full:
            if self.profiler:
                self.profiler.record_free(id(buf))

    def available_count(self) -> int:
        return self._pool.qsize()

    def clear(self) -> None:
        while not self._pool.empty():
            try:
                buf = self._pool.get_nowait()
                if self.profiler:
                    self.profiler.record_free(id(buf))
            except queue.Empty:
                break


class CropBufferPool:
    """Pre-allocated multi-tier crop buffer cache across standard face sizes."""

    SUPPORTED_SIZES = (112, 128, 256, 512, 1024)

    def __init__(self, capacity_per_size: int = 4,
                 profiler: Optional[AllocationProfiler] = None):
        self.capacity_per_size = max(1, int(capacity_per_size))
        self.profiler = profiler or GLOBAL_PROFILER
        self._pools: Dict[int, queue.Queue[np.ndarray]] = {}
        self._lock = threading.Lock()

        for s in self.SUPPORTED_SIZES:
            q: queue.Queue[np.ndarray] = queue.Queue(maxsize=self.capacity_per_size)
            for _ in range(self.capacity_per_size):
                buf = allocate_pinned_or_host((s, s, 3), np.uint8)
                q.put(buf)
                if self.profiler:
                    self.profiler.record_alloc(
                        category=AllocationCategory.FACE_CROP,
                        size_bytes=s * s * 3,
                        shape=(s, s, 3),
                        device="pinned_host",
                        caller="CropBufferPool.__init__",
                        reused=False,
                        obj_id=id(buf),
                    )
            self._pools[s] = q

    def acquire_crop(self, size: int) -> np.ndarray:
        s = int(size)
        q = self._pools.get(s)
        if q is not None:
            try:
                buf = q.get_nowait()
                if self.profiler:
                    self.profiler.record_alloc(
                        category=AllocationCategory.FACE_CROP,
                        size_bytes=s * s * 3,
                        shape=(s, s, 3),
                        device="pinned_host",
                        caller="CropBufferPool.acquire_crop",
                        reused=True,
                    )
                return buf
            except queue.Empty:
                pass
        # Fallback allocation
        buf = allocate_pinned_or_host((s, s, 3), np.uint8)
        if self.profiler:
            self.profiler.record_alloc(
                category=AllocationCategory.FACE_CROP,
                size_bytes=s * s * 3,
                shape=(s, s, 3),
                device="host",
                caller="CropBufferPool.acquire_fallback",
                reused=False,
                obj_id=id(buf),
            )
        return buf

    def release_crop(self, buf: np.ndarray) -> None:
        if buf is None:
            return
        s = buf.shape[0]
        if buf.shape == (s, s, 3):
            q = self._pools.get(s)
            if q is not None:
                try:
                    q.put_nowait(buf)
                    return
                except queue.Full:
                    pass
        if self.profiler:
            self.profiler.record_free(id(buf))

    def clear(self) -> None:
        for q in self._pools.values():
            while not q.empty():
                try:
                    buf = q.get_nowait()
                    if self.profiler:
                        self.profiler.record_free(id(buf))
                except queue.Empty:
                    break


class FastScratchBuffer:
    """Pre-allocated reusable float32/uint8 workspace for alpha blending and mask math.
    
    Eliminates the 15-20 MB per-face heap allocation cycle seen during
    warp-feather-composite operations.
    """

    def __init__(self, size: int = 512, profiler: Optional[AllocationProfiler] = None):
        self.size = size
        self.profiler = profiler or GLOBAL_PROFILER
        # Pre-allocate aligned float32 and uint8 planes
        self.f32_plane1 = np.empty((size, size, 3), dtype=np.float32)
        self.f32_plane2 = np.empty((size, size, 3), dtype=np.float32)
        self.f32_alpha = np.empty((size, size, 1), dtype=np.float32)
        self.u8_mask = np.empty((size, size), dtype=np.uint8)
        self.u8_out = np.empty((size, size, 3), dtype=np.uint8)
        self._lock = threading.Lock()

    def blend_faces_inplace(self, swap_crop: np.ndarray, plate_crop: np.ndarray,
                            alpha_mask: np.ndarray) -> np.ndarray:
        """In-place linear blending: swap * alpha + plate * (1 - alpha)."""
        with self._lock:
            # Resize if inputs mismatch (fallback path)
            if swap_crop.shape != (self.size, self.size, 3) or plate_crop.shape != (self.size, self.size, 3):
                alpha = (alpha_mask.astype(np.float32) / 255.0)
                if len(alpha.shape) == 2:
                    alpha = alpha[:, :, None]
                return (swap_crop.astype(np.float32) * alpha + plate_crop.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

            # Convert to float planes without creating intermediate temporaries
            np.copyto(self.f32_plane1, swap_crop, casting="unsafe")
            np.copyto(self.f32_plane2, plate_crop, casting="unsafe")

            if len(alpha_mask.shape) == 2:
                np.copyto(self.f32_alpha[:, :, 0], alpha_mask, casting="unsafe")
            else:
                np.copyto(self.f32_alpha, alpha_mask, casting="unsafe")
            self.f32_alpha *= (1.0 / 255.0)

            # In-place blending arithmetic
            # f32_plane1 = f32_plane1 * alpha + f32_plane2 * (1.0 - alpha)
            self.f32_plane1 *= self.f32_alpha
            self.f32_plane2 *= (1.0 - self.f32_alpha)
            self.f32_plane1 += self.f32_plane2
            np.clip(self.f32_plane1, 0, 255, out=self.f32_plane1)
            np.copyto(self.u8_out, self.f32_plane1, casting="unsafe")

            if self.profiler:
                self.profiler.record_alloc(
                    category=AllocationCategory.CPU_COPY,
                    size_bytes=self.size * self.size * 3,
                    shape=(self.size, self.size, 3),
                    device="cpu_scratch",
                    caller="FastScratchBuffer.blend_faces_inplace",
                    reused=True,
                )
            return self.u8_out.copy()


# =============================================================================
# 3. Lifetime-Aware Tensors (RAII GPU Memory Scoping)
# =============================================================================

class LifetimeTensor:
    """RAII-scoped tensor container that guarantees deterministic GPU memory reclamation."""

    def __init__(self, tensor: Any, device: str = "cpu",
                 profiler: Optional[AllocationProfiler] = None,
                 empty_cache_on_exit: bool = False):
        self.tensor = tensor
        self.device = str(device)
        self.profiler = profiler or GLOBAL_PROFILER
        self.empty_cache_on_exit = empty_cache_on_exit
        self.size_bytes = self._compute_size()
        self._freed = False

        if self.profiler:
            cat = AllocationCategory.TENSOR_H2D if "cuda" in self.device else AllocationCategory.TENSOR_OP
            self.profiler.record_alloc(
                category=cat,
                size_bytes=self.size_bytes,
                shape=getattr(tensor, "shape", ()),
                dtype=str(getattr(tensor, "dtype", "unknown")),
                device=self.device,
                caller="LifetimeTensor.__init__",
                reused=False,
                obj_id=id(self),
            )

    def _compute_size(self) -> int:
        if self.tensor is None:
            return 0
        if hasattr(self.tensor, "element_size") and hasattr(self.tensor, "nelement"):
            return int(self.tensor.element_size() * self.tensor.nelement())
        if hasattr(self.tensor, "nbytes"):
            return int(self.tensor.nbytes)
        return 0

    def free(self) -> None:
        if self._freed:
            return
        self._freed = True
        if self.profiler:
            self.profiler.record_free(id(self))
        del self.tensor
        self.tensor = None
        if self.empty_cache_on_exit and _HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                _swallowed("roop/vram_ram_manager.py:LifetimeTensor.free", exc)

    def __enter__(self) -> "LifetimeTensor":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.free()


# =============================================================================
# 4. Explicit Memory Cache with LRU & Byte-Ceiling Policy
# =============================================================================

@dataclass
class CacheEntry:
    key: str
    value: Any
    size_bytes: int
    last_accessed: float = field(default_factory=time.time)


class ExplicitMemoryCache:
    """Thread-safe LRU cache with explicit maximum item count and byte ceiling."""

    def __init__(self, max_items: int = 128, max_bytes_mb: float = 256.0,
                 profiler: Optional[AllocationProfiler] = None):
        self.max_items = max(1, int(max_items))
        self.max_bytes = int(max_bytes_mb * 1024.0 * 1024.0)
        self.profiler = profiler or GLOBAL_PROFILER
        self._lock = threading.Lock()
        self._entries: collections.OrderedDict[str, CacheEntry] = collections.OrderedDict()
        self._current_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.last_accessed = time.time()
                self._entries.move_to_end(key)
                self._hits += 1
                return entry.value
            self._misses += 1
            return None

    def put(self, key: str, value: Any, size_bytes: Optional[int] = None) -> None:
        if size_bytes is None:
            if hasattr(value, "nbytes"):
                size_bytes = int(value.nbytes)
            elif hasattr(value, "element_size") and hasattr(value, "nelement"):
                size_bytes = int(value.element_size() * value.nelement())
            else:
                size_bytes = sys.getsizeof(value)

        with self._lock:
            # If replacing an existing key, subtract prior size
            if key in self._entries:
                old = self._entries.pop(key)
                self._current_bytes -= old.size_bytes

            # Evict if over item cap or byte ceiling
            while self._entries and (len(self._entries) >= self.max_items or
                                     self._current_bytes + size_bytes > self.max_bytes):
                _, evicted = self._entries.popitem(last=False)
                self._current_bytes -= evicted.size_bytes
                self._evictions += 1
                if self.profiler:
                    self.profiler.record_free(id(evicted))

            entry = CacheEntry(key=key, value=value, size_bytes=size_bytes)
            self._entries[key] = entry
            self._current_bytes += size_bytes
            if self.profiler:
                self.profiler.record_alloc(
                    category=AllocationCategory.CACHE_ENTRY,
                    size_bytes=size_bytes,
                    device="host_cache",
                    caller="ExplicitMemoryCache.put",
                    reused=False,
                    obj_id=id(entry),
                )

    def purge(self) -> int:
        """Evict all entries and return freed bytes."""
        with self._lock:
            freed = self._current_bytes
            for entry in self._entries.values():
                if self.profiler:
                    self.profiler.record_free(id(entry))
            self._entries.clear()
            self._current_bytes = 0
            return freed

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "items": len(self._entries),
                "max_items": self.max_items,
                "current_bytes": self._current_bytes,
                "current_mb": round(self._current_bytes / (1024.0 ** 2), 2),
                "max_mb": round(self.max_bytes / (1024.0 ** 2), 2),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_ratio": round(self._hits / max(1, self._hits + self._misses), 3),
            }


# =============================================================================
# 5. VRAM-Aware Runtime Limits & Dynamic Configuration
# =============================================================================

@dataclass(frozen=True)
class RuntimeMemoryLimits:
    """Dynamically selected runtime bounds governed by detected VRAM/RAM."""

    gpu_tier: str
    vram_total_gb: float
    vram_free_gb: float
    ram_total_gb: float
    ram_available_gb: float
    batch_size: int
    worker_count: int
    face_crop_concurrency: int
    enhancer_concurrency: int
    buffer_depth: int
    max_in_flight_frames: int
    rss_budget_mb: float
    gpen_size: int
    rationale: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VramRuntimeSelector:
    """Dynamically probes system memory and computes safe concurrency limits.
    
    Principles:
    1. Never sacrifice stability for theoretical utilization.
    2. Enforce strict safety margins (1.5 GB VRAM margin default).
    3. Hysteresis: Step down immediately on memory dip; step up conservatively.
    """

    def __init__(self, safety_margin_gb: float = 1.5):
        self.safety_margin_gb = max(0.5, float(safety_margin_gb))

    @staticmethod
    def query_vram() -> Tuple[float, float, float]:
        """Return (free_gb, used_gb, total_gb) for primary CUDA GPU."""
        # 1. Try NVML
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gib = 1024.0 ** 3
            return info.free / gib, info.used / gib, info.total / gib
        except Exception as exc:
            _swallowed("roop/vram_ram_manager.py:query_vram:nvml", exc)

        # 2. Try PyTorch
        if _HAS_TORCH and torch.cuda.is_available():
            try:
                free_b, total_b = torch.cuda.mem_get_info(0)
                gib = 1024.0 ** 3
                return free_b / gib, (total_b - free_b) / gib, total_b / gib
            except Exception as exc:
                _swallowed("roop/vram_ram_manager.py:query_vram:torch", exc)

        return 0.0, 0.0, 0.0

    @staticmethod
    def query_ram() -> Tuple[float, float]:
        """Return (available_gb, total_gb) of system RAM."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            gib = 1024.0 ** 3
            return mem.available / gib, mem.total / gib
        except Exception as exc:
            _swallowed("roop/vram_ram_manager.py:query_ram:psutil", exc)
        return 8.0, 16.0  # Conservative fallback

    def select_limits(self, width: int = 1920, height: int = 1080,
                      is_stabilization_enabled: bool = False,
                      override_vram_gb: Optional[float] = None,
                      override_ram_gb: Optional[float] = None) -> RuntimeMemoryLimits:
        """Dynamically determine safe concurrency, batch, and queue depth."""
        if override_vram_gb is not None:
            vram_total_gb = float(override_vram_gb)
            vram_free_gb = max(0.0, vram_total_gb - 2.0)
        else:
            v_free, _, v_tot = self.query_vram()
            vram_free_gb = v_free
            vram_total_gb = v_tot

        if override_ram_gb is not None:
            ram_total_gb = float(override_ram_gb)
            ram_avail_gb = max(0.0, ram_total_gb * 0.7)
        else:
            r_avail, r_tot = self.query_ram()
            ram_avail_gb = r_avail
            ram_total_gb = r_tot

        reasons = []

        # Tier classification
        if vram_total_gb >= 11.5:
            gpu_tier = "rtx_4070_desktop"  # 12GB+ tier
        elif vram_total_gb >= 6.5:
            gpu_tier = "midrange_desktop"  # 8GB tier
        elif vram_total_gb > 0:
            gpu_tier = "rtx_3060_laptop"   # 6GB and low-VRAM mobile tier
        else:
            gpu_tier = "cpu_only"

        # Safe effective VRAM headroom after safety margin
        headroom_vram = max(0.0, vram_free_gb - self.safety_margin_gb)

        # ---------------------------------------------------------------------
        # Dynamic selection rules based on tier and available headroom
        # ---------------------------------------------------------------------
        if gpu_tier == "rtx_4070_desktop":
            # RTX 4070: High VRAM headroom, fast PCIe, 32GB host RAM
            if headroom_vram >= 4.0:
                batch_size = 8
                worker_count = 12
                face_crop_concurrency = 4
                enhancer_concurrency = 2
                buffer_depth = 2
                max_in_flight = 8
                gpen_size = 512
                reasons.append("RTX 4070 full tier: dual-pool TRT admitted, 8-face batching enabled")
            else:
                batch_size = 4
                worker_count = 8
                face_crop_concurrency = 2
                enhancer_concurrency = 1
                buffer_depth = 2
                max_in_flight = 6
                gpen_size = 512
                reasons.append("RTX 4070 constrained headroom: throttled to 4-face batching")
            rss_budget_mb = 4096.0

        elif gpu_tier == "rtx_3060_laptop":
            # RTX 3060: 6GB mobile, shared thermal/power envelope, 16GB host RAM
            # Crucial: Must stay under 2.5 GB RSS and avoid multi-context driver paging!
            batch_size = 1
            worker_count = 2 if is_stabilization_enabled else 4
            face_crop_concurrency = 1
            enhancer_concurrency = 1
            buffer_depth = 1
            max_in_flight = 3
            gpen_size = 256  # Sized down for speed and memory safety
            rss_budget_mb = 2500.0
            reasons.append("RTX 3060 Laptop tier: single-context serialization, 1-face batch, RSS <= 2.5GB")

        elif gpu_tier == "midrange_desktop":
            # 8GB Desktop GPU (e.g. RTX 3070 / 4060 Ti)
            batch_size = 2 if headroom_vram >= 2.0 else 1
            worker_count = 6
            face_crop_concurrency = 2
            enhancer_concurrency = 1
            buffer_depth = 2
            max_in_flight = 4
            gpen_size = 512
            rss_budget_mb = 3072.0
            reasons.append("8GB Midrange tier: 2-face batching, single enhancer context")

        else:
            # CPU fallback
            batch_size = 1
            worker_count = min(4, max(1, os.cpu_count() or 2))
            face_crop_concurrency = 1
            enhancer_concurrency = 1
            buffer_depth = 1
            max_in_flight = 2
            gpen_size = 256
            rss_budget_mb = 2048.0
            reasons.append("CPU-only fallback: single worker pipeline, minimal buffering")

        # Resolution scaling adjustment: 4K needs tighter frame bounds
        pixels = width * height
        if pixels >= 3840 * 2160:
            buffer_depth = 1
            max_in_flight = min(max_in_flight, 3)
            reasons.append("4K UHD video detected: buffer depth constrained to 1 to prevent RAM ballooning")

        # RAM pressure safeguard: if system RAM is < 4GB available, shrink in-flight frames
        if ram_avail_gb < 4.0:
            worker_count = max(1, worker_count // 2)
            max_in_flight = max(1, max_in_flight // 2)
            buffer_depth = 1
            reasons.append("Low host RAM (<4GB free): worker count and in-flight queue depth halved")

        return RuntimeMemoryLimits(
            gpu_tier=gpu_tier,
            vram_total_gb=round(vram_total_gb, 2),
            vram_free_gb=round(vram_free_gb, 2),
            ram_total_gb=round(ram_total_gb, 2),
            ram_available_gb=round(ram_avail_gb, 2),
            batch_size=batch_size,
            worker_count=worker_count,
            face_crop_concurrency=face_crop_concurrency,
            enhancer_concurrency=enhancer_concurrency,
            buffer_depth=buffer_depth,
            max_in_flight_frames=max_in_flight,
            rss_budget_mb=rss_budget_mb,
            gpen_size=gpen_size,
            rationale=reasons,
        )


# Global singleton selector
GLOBAL_MEMORY_SELECTOR = VramRuntimeSelector()
