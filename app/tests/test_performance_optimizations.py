"""Comprehensive test suite for the three high-performance systems optimizations:
1. Garbage Collection (GC) Lifecycle Management
2. Non-Target & Empty Frame Early Fast-Path
3. Frame Buffer Pre-Allocation with Pinned Memory
"""

import gc
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from roop.buffer_pool import (
    PinnedBufferPool,
    allocate_pinned_buffer,
    get_crop_buffer,
    get_frame_buffer_pool,
    is_pinned_supported,
    release_frame_buffer_pools,
)
from roop.face_analyser import (
    check_face_matches_target,
    compute_cosine_similarity,
    evaluate_fast_path,
    has_face,
)
import roop.core


class TestBufferPool(unittest.TestCase):
    """Test frame buffer pre-allocation and pinned buffer pooling."""

    def tearDown(self):
        release_frame_buffer_pools()

    def test_allocate_pinned_buffer(self):
        buf = allocate_pinned_buffer((720, 1280, 3), dtype=np.uint8)
        self.assertEqual(buf.shape, (720, 1280, 3))
        self.assertEqual(buf.dtype, np.uint8)
        self.assertTrue(buf.flags.writeable)

    def test_pinned_buffer_pool_acquire_and_release(self):
        pool = PinnedBufferPool(shape=(100, 100, 3), capacity=2)
        buf1 = pool.acquire()
        buf2 = pool.acquire()
        self.assertIsNot(buf1, buf2)
        self.assertEqual(buf1.shape, (100, 100, 3))
        self.assertEqual(buf2.shape, (100, 100, 3))

        # Releasing buf1 back to pool
        pool.release(buf1)
        buf1_reacquired = pool.acquire()
        self.assertIs(buf1, buf1_reacquired)

    def test_get_crop_buffer_thread_local(self):
        crop1 = get_crop_buffer(512)
        self.assertEqual(crop1.shape, (512, 512, 3))
        crop2 = get_crop_buffer(512)
        # Reuses same pre-allocated buffer for this thread
        self.assertIs(crop1, crop2)

        crop_small = get_crop_buffer(256)
        self.assertEqual(crop_small.shape, (256, 256, 3))

    def test_get_frame_buffer_pool_cache(self):
        p1 = get_frame_buffer_pool(720, 1280, capacity=4)
        p2 = get_frame_buffer_pool(720, 1280, capacity=4)
        self.assertIs(p1, p2)
        release_frame_buffer_pools()


class TestFaceAnalyserFastPath(unittest.TestCase):
    """Test fast-path evaluation and bypass routing for 0 faces and non-target tracks."""

    def test_cosine_similarity(self):
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(compute_cosine_similarity(v1, v2), 1.0, places=4)

        v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(compute_cosine_similarity(v1, v3), 0.0, places=4)

        v_zero = np.zeros(3, dtype=np.float32)
        self.assertEqual(compute_cosine_similarity(v1, v_zero), 0.0)

    def test_check_face_matches_target(self):
        target_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        target = SimpleNamespace(embedding=target_emb)

        # Matching face (distance = 0.0 <= 0.65 threshold)
        matching_face = SimpleNamespace(embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32))
        matched, score = check_face_matches_target(matching_face, [target], threshold=0.65)
        self.assertTrue(matched)
        self.assertAlmostEqual(score, 1.0, places=4)

        # Dissimilar face (distance = 1.0 > 0.65 threshold)
        dissimilar_face = SimpleNamespace(embedding=np.array([0.0, 1.0, 0.0], dtype=np.float32))
        matched, score = check_face_matches_target(dissimilar_face, [target], threshold=0.65)
        self.assertFalse(matched)
        self.assertAlmostEqual(score, 0.0, places=4)

    def test_evaluate_fast_path_empty_frame(self):
        # Frame with None or 0 precomputed faces -> immediate bypass
        should_bypass, faces = evaluate_fast_path(None)
        self.assertTrue(should_bypass)
        self.assertEqual(len(faces), 0)

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        should_bypass, faces = evaluate_fast_path(dummy_frame, precomputed_faces=[])
        self.assertTrue(should_bypass)
        self.assertEqual(len(faces), 0)

    def test_evaluate_fast_path_non_target_faces(self):
        target = SimpleNamespace(embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32))
        non_target_face = SimpleNamespace(embedding=np.array([0.0, 1.0, 0.0], dtype=np.float32))

        # Face present but similarity < threshold -> should bypass
        should_bypass, faces = evaluate_fast_path(
            np.zeros((100, 100, 3), dtype=np.uint8),
            target_faces=[target],
            threshold=0.65,
            precomputed_faces=[non_target_face]
        )
        self.assertTrue(should_bypass)
        self.assertEqual(len(faces), 1)

    def test_evaluate_fast_path_matched_face(self):
        target = SimpleNamespace(embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32))
        target_match = SimpleNamespace(embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32))

        # Face matches target -> do NOT bypass (should proceed to swap)
        should_bypass, faces = evaluate_fast_path(
            np.zeros((100, 100, 3), dtype=np.uint8),
            target_faces=[target],
            threshold=0.65,
            precomputed_faces=[target_match]
        )
        self.assertFalse(should_bypass)
        self.assertEqual(len(faces), 1)


class TestGCLifecycleAndCoreFastPath(unittest.TestCase):
    """Test GC lifecycle management and core.py fast path."""

    def test_core_fast_path_bypass(self):
        target = SimpleNamespace(embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32))
        non_target = SimpleNamespace(embedding=np.array([0.0, 1.0, 0.0], dtype=np.float32))

        with patch("roop.face_analyser.get_all_faces", return_value=[non_target]):
            bypass = roop.core.fast_path_bypass(
                np.zeros((100, 100, 3), dtype=np.uint8),
                target_faces=[target],
                threshold=0.65
            )
            self.assertTrue(bypass)

    def test_gc_disabled_during_active_work(self):
        gc.enable()
        self.assertTrue(gc.isenabled())
        gc.disable()
        self.assertFalse(gc.isenabled())
        gc.enable()
        self.assertTrue(gc.isenabled())


if __name__ == "__main__":
    unittest.main()
