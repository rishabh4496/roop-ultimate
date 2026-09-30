"""Tests for Stage 11 — VRAM/RAM Management Architecture and Runtime Limits.

Covers:
1. AllocationProfiler tracking, reporting, and hotspot detection.
2. FrameBufferPool preallocation, acquisition, reuse, and capacity limits.
3. CropBufferPool multi-tier sizing (112, 128, 256, 512, 1024) and reuse.
4. FastScratchBuffer zero-allocation in-place blending vs naive reference.
5. LifetimeTensor RAII scoping and deterministic cleanup.
6. ExplicitMemoryCache LRU eviction, byte ceiling, and purge.
7. VramRuntimeSelector hardware profile tuning:
   - RTX 4070 Desktop (12GB VRAM / 32GB RAM / pooled execution)
   - RTX 3060 Laptop (6GB VRAM / 16GB RAM / single-context safety tier)
   - 4K resolution throttle & low-RAM safety degradation
8. vram_governor integration with governed runtime limits.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pytest

# Ensure app path is available
ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app"
if str(APP_PATH) not in sys.path:
    sys.path.insert(0, str(APP_PATH))

from roop.vram_ram_manager import (
    AllocationCategory,
    AllocationProfiler,
    CropBufferPool,
    ExplicitMemoryCache,
    FastScratchBuffer,
    FrameBufferPool,
    LifetimeTensor,
    RuntimeMemoryLimits,
    VramRuntimeSelector,
)
from roop.vram_governor import (
    governed_buffer_depth,
    governed_enhancer_concurrency,
    governed_face_concurrency,
    governed_memory_limits,
    governed_worker_count,
)


class TestAllocationProfiler(unittest.TestCase):
    """Test allocation profiling and memory hotspot detection."""

    def setUp(self):
        self.profiler = AllocationProfiler(enabled=True)

    def test_record_alloc_and_free(self):
        self.profiler.record_alloc(
            category=AllocationCategory.FRAME,
            size_bytes=1000,
            shape=(10, 10, 10),
            dtype="uint8",
            device="host",
            caller="test",
            reused=False,
            obj_id=123,
        )
        report = self.profiler.generate_report()
        self.assertEqual(report.total_allocations, 1)
        self.assertEqual(report.current_active_bytes, 1000)
        self.assertEqual(report.peak_active_bytes, 1000)

        # Record free
        self.profiler.record_free(123)
        report2 = self.profiler.generate_report()
        self.assertEqual(report2.current_active_bytes, 0)
        self.assertEqual(report2.peak_active_bytes, 1000)

    def test_reused_allocation_tracking(self):
        self.profiler.record_alloc(
            category=AllocationCategory.FACE_CROP,
            size_bytes=500,
            reused=True,
            caller="test_reuse",
        )
        report = self.profiler.generate_report()
        self.assertEqual(report.reused_allocations, 1)
        self.assertEqual(report.bytes_saved_by_reuse, 500)
        self.assertEqual(report.current_active_bytes, 0)

    def test_hotspot_detection(self):
        self.profiler.mark_duplicated_frame("procmgr_batch")
        self.profiler.mark_duplicated_crop("align_crop")
        self.profiler.mark_unnecessary_copy("cv2.cvtColor")
        self.profiler.mark_stale_tensor("PyTorch_cache")
        self.profiler.record_queue_accumulation(150 * 1024 * 1024)

        report = self.profiler.generate_report()
        self.assertEqual(report.duplicated_frame_count, 1)
        self.assertEqual(report.duplicated_crop_count, 1)
        self.assertEqual(report.unnecessary_copies_count, 1)
        self.assertEqual(report.stale_tensor_count, 1)
        self.assertTrue(len(report.detected_hotspots) >= 4)


class TestBufferPools(unittest.TestCase):
    """Test pre-allocated frame and crop buffer pools."""

    def test_frame_buffer_pool_lifecycle(self):
        profiler = AllocationProfiler(enabled=True)
        pool = FrameBufferPool(shape=(360, 640, 3), capacity=2, profiler=profiler)
        self.assertEqual(pool.available_count(), 2)

        # Acquire 1
        buf1 = pool.acquire()
        self.assertEqual(buf1.shape, (360, 640, 3))
        self.assertEqual(pool.available_count(), 1)

        # Acquire 2
        buf2 = pool.acquire()
        self.assertEqual(pool.available_count(), 0)

        # Acquire 3 (fallback allocation beyond capacity)
        buf3 = pool.acquire()
        self.assertEqual(buf3.shape, (360, 640, 3))

        # Release
        pool.release(buf1)
        self.assertEqual(pool.available_count(), 1)
        pool.release(buf2)
        self.assertEqual(pool.available_count(), 2)

        pool.clear()
        self.assertEqual(pool.available_count(), 0)

    def test_crop_buffer_pool_tiers(self):
        profiler = AllocationProfiler(enabled=True)
        crop_pool = CropBufferPool(capacity_per_size=2, profiler=profiler)

        # Test standard sizes
        for size in (112, 128, 256, 512, 1024):
            crop = crop_pool.acquire_crop(size)
            self.assertEqual(crop.shape, (size, size, 3))
            self.assertEqual(crop.dtype, np.uint8)
            crop_pool.release_crop(crop)

        # Test non-standard fallback size
        crop_nonstandard = crop_pool.acquire_crop(384)
        self.assertEqual(crop_nonstandard.shape, (384, 384, 3))
        crop_pool.release_crop(crop_nonstandard)
        crop_pool.clear()


class TestFastScratchBuffer(unittest.TestCase):
    """Test fast zero-allocation scratch blending."""

    def test_blend_faces_inplace_accuracy(self):
        scratch = FastScratchBuffer(size=256)
        swap = np.full((256, 256, 3), 200, dtype=np.uint8)
        plate = np.full((256, 256, 3), 50, dtype=np.uint8)
        alpha = np.full((256, 256), 128, dtype=np.uint8)  # ~0.5 blend

        result = scratch.blend_faces_inplace(swap, plate, alpha)

        # Reference mathematical blend: 200 * (128/255) + 50 * (127/255) = 100.39 + 24.90 = ~125
        ref_alpha = 128.0 / 255.0
        ref = int(round(200.0 * ref_alpha + 50.0 * (1.0 - ref_alpha)))

        self.assertEqual(result.shape, (256, 256, 3))
        self.assertEqual(result.dtype, np.uint8)
        diff = np.abs(result.astype(int) - ref)
        self.assertLessEqual(np.max(diff), 1)


class TestLifetimeTensor(unittest.TestCase):
    """Test RAII lifetime-aware tensor disposal."""

    def test_lifetime_tensor_context(self):
        arr = np.zeros((100, 100), dtype=np.float32)
        profiler = AllocationProfiler(enabled=True)

        with LifetimeTensor(arr, device="cpu", profiler=profiler) as lt:
            self.assertIsNotNone(lt.tensor)
            self.assertEqual(lt.size_bytes, 40000)
            rep = profiler.generate_report()
            self.assertEqual(rep.current_active_bytes, 40000)

        # Exited context
        self.assertTrue(lt._freed)
        self.assertIsNone(lt.tensor)
        rep2 = profiler.generate_report()
        self.assertEqual(rep2.current_active_bytes, 0)


class TestExplicitMemoryCache(unittest.TestCase):
    """Test explicit LRU cache with item count and byte limits."""

    def test_lru_eviction_by_count(self):
        cache = ExplicitMemoryCache(max_items=3, max_bytes_mb=10.0)
        cache.put("a", np.zeros((10, 10), dtype=np.uint8), size_bytes=100)
        cache.put("b", np.zeros((10, 10), dtype=np.uint8), size_bytes=100)
        cache.put("c", np.zeros((10, 10), dtype=np.uint8), size_bytes=100)

        # Access "a" so "b" becomes the least recently used
        _ = cache.get("a")

        # Put "d" -> "b" should be evicted
        cache.put("d", np.zeros((10, 10), dtype=np.uint8), size_bytes=100)

        self.assertIsNotNone(cache.get("a"))
        self.assertIsNone(cache.get("b"))
        self.assertIsNotNone(cache.get("c"))
        self.assertIsNotNone(cache.get("d"))

    def test_byte_ceiling_eviction(self):
        # 1000 bytes maximum
        cache = ExplicitMemoryCache(max_items=10, max_bytes_mb=0.001)
        cache.put("k1", "data1", size_bytes=600)
        self.assertEqual(cache.stats()["current_bytes"], 600)

        # Putting another 600 bytes exceeds 1000 bytes, evicting k1
        cache.put("k2", "data2", size_bytes=600)
        self.assertIsNone(cache.get("k1"))
        self.assertIsNotNone(cache.get("k2"))
        self.assertEqual(cache.stats()["evictions"], 1)

    def test_purge(self):
        cache = ExplicitMemoryCache(max_items=10, max_bytes_mb=10.0)
        cache.put("x", 123, size_bytes=200)
        freed = cache.purge()
        self.assertEqual(freed, 200)
        self.assertEqual(cache.stats()["items"], 0)


class TestVramRuntimeSelector(unittest.TestCase):
    """Test dynamic selection of runtime limits across hardware profiles."""

    def setUp(self):
        self.selector = VramRuntimeSelector(safety_margin_gb=1.5)

    def test_rtx_4070_desktop_profile(self):
        # Simulate RTX 4070 (12GB VRAM, 32GB RAM)
        limits = self.selector.select_limits(
            width=1920, height=1080,
            override_vram_gb=12.0,
            override_ram_gb=32.0,
        )
        self.assertEqual(limits.gpu_tier, "rtx_4070_desktop")
        self.assertGreaterEqual(limits.batch_size, 4)
        self.assertGreaterEqual(limits.worker_count, 8)
        self.assertEqual(limits.enhancer_concurrency, 2)
        self.assertGreaterEqual(limits.max_in_flight_frames, 6)
        self.assertEqual(limits.rss_budget_mb, 4096.0)

    def test_rtx_3060_laptop_profile(self):
        # Simulate RTX 3060 Laptop (6GB VRAM, 16GB RAM)
        limits = self.selector.select_limits(
            width=1920, height=1080,
            override_vram_gb=6.0,
            override_ram_gb=16.0,
        )
        self.assertEqual(limits.gpu_tier, "rtx_3060_laptop")
        self.assertEqual(limits.batch_size, 1)  # Strictly serialized
        self.assertLessEqual(limits.worker_count, 4)
        self.assertEqual(limits.enhancer_concurrency, 1)  # Single context
        self.assertEqual(limits.buffer_depth, 1)
        self.assertEqual(limits.gpen_size, 256)
        self.assertEqual(limits.rss_budget_mb, 2500.0)  # Hard RSS cap

    def test_4k_resolution_throttling(self):
        # 4K UHD video (3840x2160) on 4070
        limits = self.selector.select_limits(
            width=3840, height=2160,
            override_vram_gb=12.0,
            override_ram_gb=32.0,
        )
        self.assertEqual(limits.buffer_depth, 1)
        self.assertLessEqual(limits.max_in_flight_frames, 3)

    def test_low_host_ram_safeguard(self):
        # Low available host RAM (< 4GB)
        limits = self.selector.select_limits(
            width=1920, height=1080,
            override_vram_gb=12.0,
            override_ram_gb=3.5,  # Constrained RAM
        )
        self.assertEqual(limits.buffer_depth, 1)
        self.assertLessEqual(limits.max_in_flight_frames, 4)


class TestVramGovernorIntegration(unittest.TestCase):
    """Test vram_governor integration with dynamic runtime limits."""

    def test_governed_helpers(self):
        limits = governed_memory_limits()
        self.assertIsInstance(limits, RuntimeMemoryLimits)

        w = governed_worker_count(20)
        self.assertLessEqual(w, limits.worker_count)

        d = governed_buffer_depth(4)
        self.assertLessEqual(d, limits.buffer_depth)

        f = governed_face_concurrency(8)
        self.assertLessEqual(f, limits.face_crop_concurrency)

        e = governed_enhancer_concurrency(4)
        self.assertLessEqual(e, limits.enhancer_concurrency)


if __name__ == "__main__":
    unittest.main()
