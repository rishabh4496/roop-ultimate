"""Tests for Stage 7 — XSeg 3 Mask Quality and Performance Optimization.

Validates:
1. XSeg3BufferPool: Reusable buffers, zero-allocation pre-processing, LUT normalization.
2. smoothstep_threshold: Cubic Hermite curve, continuity, zero popping.
3. refine_xseg3_mask: Morphological closing, guided-filter edge snapping, mouth protection, confidence fallback.
4. XSeg3MaskCache: Conditional reuse, pose/geometry/confidence/occlusion invalidation, affine warping.
5. XSeg3TemporalStabilizer: Frame-to-frame EMA smoothing, strict contiguity, sudden-flux reset.
6. Mask_XSeg3 end-to-end integration and exception safety.
"""

import math
import os
import unittest
import numpy as np
import cv2

from roop.xseg3_optimizer import (
    BUFFER_POOL,
    MASK_CACHE,
    TEMPORAL_STABILIZER,
    XSeg3BufferPool,
    XSeg3MaskCache,
    XSeg3TemporalStabilizer,
    smoothstep_threshold,
    refine_xseg3_mask,
)
from roop.processors.Mask_XSeg3 import Mask_XSeg3


class MockFace:
    """Mock Face object for testing."""
    def __init__(
        self,
        kps=None,
        pose=None,
        det_score=0.98,
        track_id=1,
        landmark_2d_106=None,
        bbox=None
    ):
        if kps is None:
            # 5 standard facial landmarks in 512x512 space:
            # [left_eye, right_eye, nose, left_mouth, right_mouth]
            self.kps = np.array([
                [192.0, 240.0],
                [320.0, 240.0],
                [256.0, 310.0],
                [208.0, 390.0],
                [304.0, 390.0],
            ], dtype=np.float32)
        else:
            self.kps = np.asarray(kps, dtype=np.float32)

        self.pose = pose if pose is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.det_score = float(det_score)
        self._temporal_confidence = float(det_score)
        self._track_id = track_id
        self.track_id = track_id
        self.bbox = bbox if bbox is not None else np.array([128.0, 128.0, 384.0, 384.0], dtype=np.float32)
        self.landmark_2d_106 = landmark_2d_106


class TestXSeg3BufferPool(unittest.TestCase):
    """Test buffer pool allocation and zero-copy/LUT operations."""

    def setUp(self):
        self.pool = XSeg3BufferPool()

    def test_buffer_persistence(self):
        b1 = self.pool.get_input_buffer((1, 256, 256, 3))
        b2 = self.pool.get_input_buffer((1, 256, 256, 3))
        self.assertIs(b1, b2, "Input buffer must be cached thread-locally")
        self.assertEqual(b1.shape, (1, 256, 256, 3))
        self.assertEqual(b1.dtype, np.float32)

    def test_resize_buffer_persistence(self):
        r1 = self.pool.get_resize_buffer((256, 256, 3))
        r2 = self.pool.get_resize_buffer((256, 256, 3))
        self.assertIs(r1, r2, "Resize buffer must be cached thread-locally")
        self.assertEqual(r1.shape, (256, 256, 3))
        self.assertEqual(r1.dtype, np.uint8)

    def test_prepare_model_input_normalization(self):
        # Create synthetic test pattern with known values
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        img[:, :] = [0, 128, 255]
        out = self.pool.prepare_model_input(img)
        self.assertEqual(out.shape, (1, 256, 256, 3))
        self.assertEqual(out.dtype, np.float32)
        # Check normalization matches / 255.0
        self.assertAlmostEqual(out[0, 0, 0, 0], 0.0 / 255.0, places=5)
        self.assertAlmostEqual(out[0, 0, 0, 1], 128.0 / 255.0, places=5)
        self.assertAlmostEqual(out[0, 0, 0, 2], 255.0 / 255.0, places=5)

    def test_cached_kernels(self):
        self.assertIsNotNone(XSeg3BufferPool.KERNEL_ELLIPSE_3)
        self.assertEqual(XSeg3BufferPool.KERNEL_ELLIPSE_3.shape, (3, 3))
        self.assertIsNotNone(XSeg3BufferPool.KERNEL_RECT_3)
        self.assertEqual(XSeg3BufferPool.KERNEL_RECT_3.shape, (3, 3))


class TestSmoothstepThreshold(unittest.TestCase):
    """Test cubic Hermite smoothstep thresholding."""

    def test_boundary_values(self):
        x = np.array([-1.0, 0.0, 0.20, 0.35, 0.50, 0.80, 1.5], dtype=np.float32)
        out = smoothstep_threshold(x, lo=0.20, hi=0.50)
        self.assertEqual(out[0], 0.0)
        self.assertEqual(out[1], 0.0)
        self.assertEqual(out[2], 0.0)
        # Midpoint of [0.20, 0.50] is 0.35 -> t = 0.5 -> 3(0.25) - 2(0.125) = 0.5
        self.assertAlmostEqual(out[3], 0.5, places=5)
        self.assertEqual(out[4], 1.0)
        self.assertEqual(out[5], 1.0)
        self.assertEqual(out[6], 1.0)

    def test_strict_monotonicity(self):
        x = np.linspace(0.0, 1.0, 100, dtype=np.float32)
        out = smoothstep_threshold(x, lo=0.25, hi=0.45)
        diffs = np.diff(out)
        self.assertTrue(np.all(diffs >= -1e-7), "Smoothstep must be monotonically non-decreasing")

    def test_derivative_smoothness(self):
        # Near lo=0.20, derivative should approach zero smoothly
        eps = 1e-4
        y_lo_plus = smoothstep_threshold(np.array([0.20 + eps], dtype=np.float32), lo=0.20, hi=0.50)[0]
        deriv_lo = y_lo_plus / eps
        # t = eps / 0.30 -> S'(t) = 6t(1-t) -> 0 as t -> 0
        self.assertLess(deriv_lo, 0.01, "First derivative at lo must be near 0 (smooth entry)")


class TestRefineXSeg3Mask(unittest.TestCase):
    """Test confidence-aware refinement, guided filtering, morphology, and mouth protection."""

    def setUp(self):
        self.guide_frame = np.full((512, 512, 3), 120, dtype=np.uint8)
        # Add high-contrast edge simulating hair or glasses frame
        self.guide_frame[:, 256:] = 30
        self.face = MockFace()

    def test_output_shape_and_range(self):
        raw_mask = np.random.uniform(0.0, 1.0, (256, 256)).astype(np.float32)
        refined = refine_xseg3_mask(
            raw_mask,
            self.guide_frame,
            target_face=self.face,
            confidence=0.95
        )
        self.assertEqual(refined.shape, (512, 512))
        self.assertEqual(refined.dtype, np.float32)
        self.assertGreaterEqual(float(refined.min()), 0.0)
        self.assertLessEqual(float(refined.max()), 1.0)

    def test_morphological_hole_closing(self):
        # Mask with single-pixel hole on cheek specular reflection
        raw_mask = np.ones((256, 256), dtype=np.float32)
        raw_mask[128, 128] = 0.0  # isolated hole
        refined = refine_xseg3_mask(
            raw_mask,
            self.guide_frame,
            target_face=self.face,
            enable_guided_filter=False,
            fill_specular_holes=True
        )
        # In refined 512x512 space, (256, 256) should be filled
        self.assertGreater(refined[256, 256], 0.8, "Morphological closing must fill isolated specular pinhole")

    def test_mouth_protection(self):
        # Simulate false-positive occlusion in the mouth region (kps[3] and kps[4])
        raw_mask = np.zeros((256, 256), dtype=np.float32)
        # Mouth region in 256 space is around y=195, x=128
        raw_mask[185:205, 115:140] = 0.55  # false-positive occluder (e.g. teeth)
        refined = refine_xseg3_mask(
            raw_mask,
            self.guide_frame,
            target_face=self.face,
            enable_guided_filter=False
        )
        # Refined mask in mouth center should be damped
        mouth_y = int(390)
        mouth_x = int(256)
        self.assertLess(refined[mouth_y, mouth_x], 0.35, "Mouth cavity false positive should be damped")

    def test_confidence_fallback_on_degraded_face(self):
        raw_mask = np.ones((256, 256), dtype=np.float32)  # global occluder
        # Run with low confidence 0.30
        refined = refine_xseg3_mask(
            raw_mask,
            self.guide_frame,
            target_face=self.face,
            confidence=0.30,
            enable_guided_filter=False
        )
        # Far corner (0, 0) is well outside landmark convex hull
        self.assertLess(refined[0, 0], 0.5, "Degraded confidence must restrict occluder to facial hull")


class TestXSeg3MaskCache(unittest.TestCase):
    """Test geometry-aware conditional mask reuse cache."""

    def setUp(self):
        self.cache = XSeg3MaskCache(enabled=True)
        self.cache.clear()
        self.crop = np.full((256, 256, 3), 128, dtype=np.uint8)
        self.mask = np.full((256, 256), 0.7, dtype=np.float32)
        self.face = MockFace(track_id="track_42", pose=[0.0, 0.0, 0.0], det_score=0.95)

    def test_cold_cache_miss(self):
        can_reuse, mask, reason = self.cache.evaluate_reuse(
            track_id="track_42",
            current_kps=self.face.kps,
            target_face=self.face,
            crop_bgr=self.crop,
            frame_idx=0
        )
        self.assertFalse(can_reuse)
        self.assertIsNone(mask)
        self.assertEqual(reason, "cache_cold")

    def test_cache_hit_on_static_face(self):
        # Update cache with initial observation
        self.cache.update(
            track_id="track_42",
            mask_256=self.mask,
            current_kps=self.face.kps,
            target_face=self.face,
            crop_bgr=self.crop,
            frame_idx=0
        )
        # Next frame with identical pose/geometry
        can_reuse, cached_mask, reason = self.cache.evaluate_reuse(
            track_id="track_42",
            current_kps=self.face.kps,
            target_face=self.face,
            crop_bgr=self.crop,
            frame_idx=1
        )
        self.assertTrue(can_reuse)
        self.assertIsNotNone(cached_mask)
        self.assertIn("cache_hit", reason)

    def test_cache_miss_on_pose_change(self):
        self.cache.update(
            track_id="track_42",
            mask_256=self.mask,
            current_kps=self.face.kps,
            target_face=self.face,
            crop_bgr=self.crop,
            frame_idx=0
        )
        # Pose rotates by 4.0 degrees yaw (> 2.5 degree tolerance)
        turned_face = MockFace(track_id="track_42", pose=[0.0, 4.0, 0.0], det_score=0.95)
        can_reuse, _, reason = self.cache.evaluate_reuse(
            track_id="track_42",
            current_kps=self.face.kps,
            target_face=turned_face,
            crop_bgr=self.crop,
            frame_idx=1
        )
        self.assertFalse(can_reuse)
        self.assertIn("pose_delta", reason)

    def test_cache_miss_on_geometry_change(self):
        self.cache.update(
            track_id="track_42",
            mask_256=self.mask,
            current_kps=self.face.kps,
            target_face=self.face,
            crop_bgr=self.crop,
            frame_idx=0
        )
        # Landmark shifts by 6.0 pixels (> 1.8 px tolerance)
        shifted_kps = self.face.kps + 6.0
        can_reuse, _, reason = self.cache.evaluate_reuse(
            track_id="track_42",
            current_kps=shifted_kps,
            target_face=self.face,
            crop_bgr=self.crop,
            frame_idx=1
        )
        self.assertFalse(can_reuse)
        self.assertIn("geom_delta", reason)

    def test_cache_miss_on_occlusion_flux(self):
        self.cache.update(
            track_id="track_42",
            mask_256=self.mask,
            current_kps=self.face.kps,
            target_face=self.face,
            crop_bgr=self.crop,
            frame_idx=0
        )
        # A bright hand enters face center (fingerprint changes dramatically)
        hand_crop = self.crop.copy()
        hand_crop[64:192, 64:192] = 240
        can_reuse, _, reason = self.cache.evaluate_reuse(
            track_id="track_42",
            current_kps=self.face.kps,
            target_face=self.face,
            crop_bgr=hand_crop,
            frame_idx=1
        )
        self.assertFalse(can_reuse)
        self.assertIn("occlusion_flux", reason)

    def test_cache_affine_warping(self):
        # Test affine warp on small translation
        self.cache.update(
            track_id="track_42",
            mask_256=self.mask,
            current_kps=self.face.kps,
            target_face=self.face,
            crop_bgr=self.crop,
            frame_idx=0
        )
        # Small displacement of 1.0 px (within 1.8 px tolerance)
        slight_kps = self.face.kps + 1.0
        can_reuse, warped_mask, reason = self.cache.evaluate_reuse(
            track_id="track_42",
            current_kps=slight_kps,
            target_face=self.face,
            crop_bgr=self.crop,
            frame_idx=1
        )
        self.assertTrue(can_reuse)
        self.assertEqual(warped_mask.shape, (256, 256))
        self.assertIn("cache_hit_warped", reason)


class TestXSeg3TemporalStabilizer(unittest.TestCase):
    """Test motion-compensated temporal stabilizer."""

    def setUp(self):
        self.stab = XSeg3TemporalStabilizer(alpha=0.80, enabled=True)
        self.stab.clear()

    def test_initial_frame(self):
        m0 = np.full((100, 100), 0.5, dtype=np.float32)
        res = self.stab.stabilize(track_id="track_1", current_mask=m0, frame_idx=0)
        np.testing.assert_array_equal(res, m0)

    def test_contiguous_smoothing(self):
        m0 = np.full((100, 100), 0.4, dtype=np.float32)
        m1 = np.full((100, 100), 0.6, dtype=np.float32)
        self.stab.stabilize(track_id="track_1", current_mask=m0, frame_idx=0)
        smoothed = self.stab.stabilize(track_id="track_1", current_mask=m1, frame_idx=1)
        # Expected: alpha * m1 + (1 - alpha) * m0 = 0.8 * 0.6 + 0.2 * 0.4 = 0.48 + 0.08 = 0.56
        self.assertAlmostEqual(float(smoothed[50, 50]), 0.56, places=4)

    def test_non_contiguous_skip_resets(self):
        m0 = np.full((100, 100), 0.2, dtype=np.float32)
        m10 = np.full((100, 100), 0.9, dtype=np.float32)
        self.stab.stabilize(track_id="track_1", current_mask=m0, frame_idx=0)
        # Frame jumps from 0 to 10 -> should NOT blend
        res = self.stab.stabilize(track_id="track_1", current_mask=m10, frame_idx=10)
        self.assertAlmostEqual(float(res[50, 50]), 0.9, places=4)

    def test_sudden_flux_resets(self):
        # Occlusion onset (hand sweeps across face): mask changes by > 0.35 MAD
        m0 = np.zeros((100, 100), dtype=np.float32)
        m1 = np.ones((100, 100), dtype=np.float32)
        self.stab.stabilize(track_id="track_1", current_mask=m0, frame_idx=0)
        res = self.stab.stabilize(track_id="track_1", current_mask=m1, frame_idx=1)
        # Because MAD is 1.0 > 0.35, it should reset directly to m1 without lag/ghosting
        self.assertAlmostEqual(float(res[50, 50]), 1.0, places=4)


class TestMaskXSeg3Processor(unittest.TestCase):
    """Test Mask_XSeg3 processor lifecycle and invocation."""

    def test_processor_attributes(self):
        p = Mask_XSeg3()
        self.assertEqual(p.processorname, 'mask_xseg3')
        self.assertEqual(p.type, 'mask')

    def test_empty_frame_handling(self):
        p = Mask_XSeg3()
        out = p.Run(None)
        self.assertIsNone(out)
        out2 = p.Run(np.zeros((0, 0, 3), dtype=np.uint8))
        self.assertEqual(out2.size, 0)


if __name__ == '__main__':
    unittest.main()
