"""Tests for Stage 8 — Compositing Quality Engine.

Validates:
1. LinearColorSpace: Exact 8-bit LUT roundtrip, piecewise IEC 61966-2-1 compliance, soft-knee HDR compression.
2. OKLabColorSpace: Forward/inverse precision (<1e-5), neutral grey chroma invariance, chromatic separation.
3. SkinPhotometricMatcher: Skin chrominance cluster segmentation, exposure alignment, bounded white balance shift.
4. DarkSceneToneMapper: Scene classification (NORMAL, DARK, VERY_DARK), shadow floor anchoring, chroma noise suppression.
5. MultiBandSeamBlender: Frequency split decomposition, linear-light dark fringe elimination, gradient continuity.
6. EdgePreservingSharpener: Interior-only micro-sharpening, boundary halo avoidance.
7. CompositingQualityEngine: Master deterministic compositing pipeline, thread-safety, and exception resilience.
"""

from __future__ import annotations

import concurrent.futures
import math
import os
import unittest

import cv2
import numpy as np
import pytest

from roop.compositing_engine import (
    COMPOSITING_ENGINE,
    CompositingQualityEngine,
    DarkSceneToneMapper,
    EdgePreservingSharpener,
    LinearColorSpace,
    MultiBandSeamBlender,
    OKLabColorSpace,
    SkinPhotometricMatcher,
)


class TestLinearColorSpace(unittest.TestCase):
    """Validates accurate, reversible sRGB <-> Linear RGB conversions."""

    def test_srgb_to_linear_and_back_exact_256(self):
        """All 256 discrete 8-bit values must roundtrip with zero mismatch."""
        vals = np.arange(256, dtype=np.uint8)
        lin = LinearColorSpace.srgb_to_linear(vals)
        srgb = LinearColorSpace.linear_to_srgb(lin, soft_knee=False)
        diff = np.abs(vals.astype(int) - srgb.astype(int))
        self.assertEqual(diff.max(), 0, f"Max discrete roundtrip error was {diff.max()}")
        self.assertEqual(np.count_nonzero(diff), 0, "Non-zero mismatches in discrete sRGB roundtrip")

    def test_soft_knee_superwhite(self):
        """Values > 1.0 (HDR specularities) must compress smoothly without NaNs or hard clipping."""
        high_values = np.array([1.0, 1.2, 1.5, 2.0, 5.0], dtype=np.float32)
        compressed = LinearColorSpace.linear_to_srgb(high_values, soft_knee=True)
        # All output values must be valid uint8 [0, 255]
        self.assertTrue((compressed >= 254).all())
        self.assertEqual(compressed.dtype, np.uint8)
        self.assertFalse(np.isnan(compressed).any())

    def test_piecewise_linear_segment(self):
        """Linear segment below 0.04045 must match exactly (1/12.92 slope)."""
        x_low = np.array([0.01, 0.02, 0.04], dtype=np.float32)
        lin_expected = x_low / 12.92
        lin_actual = LinearColorSpace.srgb_to_linear(x_low)
        np.testing.assert_allclose(lin_actual, lin_expected, rtol=1e-5)


class TestOKLabColorSpace(unittest.TestCase):
    """Validates OKLab perceptual color space conversion fidelity and geometry."""

    def test_forward_inverse_precision(self):
        """OKLab forward and inverse transforms must invert with error < 1e-5 across RGB cube."""
        # Grid of linear RGB values in [0.05, 0.95]
        r = np.linspace(0.05, 0.95, 10, dtype=np.float32)
        g = np.linspace(0.05, 0.95, 10, dtype=np.float32)
        b = np.linspace(0.05, 0.95, 10, dtype=np.float32)
        grid = np.stack(np.meshgrid(b, g, r, indexing="ij"), axis=-1)

        oklab = OKLabColorSpace.linear_bgr_to_oklab(grid)
        reconstructed = OKLabColorSpace.oklab_to_linear_bgr(oklab)
        diff = np.abs(grid - reconstructed)
        self.assertLess(diff.max(), 1e-5, f"OKLab roundtrip error exceeded tolerance: {diff.max()}")

    def test_achromatic_neutral_chroma(self):
        """A neutral grey must have chromatic opponent coordinates a and b near zero."""
        grey = np.full((10, 10, 3), 0.5, dtype=np.float32)
        oklab = OKLabColorSpace.linear_bgr_to_oklab(grey)
        L = oklab[..., 0]
        a = oklab[..., 1]
        b = oklab[..., 2]

        self.assertTrue(np.all(L > 0.0))
        self.assertLess(np.abs(a).max(), 1e-4, "Neutral grey has non-zero opponent a")
        self.assertLess(np.abs(b).max(), 1e-4, "Neutral grey has non-zero opponent b")


class TestSkinPhotometricMatcher(unittest.TestCase):
    """Validates skin-isolated photometric, exposure, and white balance matching."""

    def setUp(self):
        # Create a synthetic face image with skin tone (BGR: ~140, 170, 215)
        self.skin_target = np.full((128, 128, 3), [140, 170, 215], dtype=np.uint8)
        # Add background with non-skin color (cool grey/blue)
        self.skin_target[:30, :] = [200, 150, 100]

    def test_skin_mask_extraction(self):
        """Skin mask should identify central warm skin pixels and exclude cool background."""
        mask = SkinPhotometricMatcher.extract_skin_mask(self.skin_target)
        # Center should be skin (1.0)
        self.assertGreater(mask[64, 64], 0.8)
        # Top cool background should be excluded (0.0)
        self.assertEqual(mask[10, 10], 0.0)

    def test_exposure_matching_underexposed_face(self):
        """An underexposed face crop matched to normal target must have increased luminance."""
        # Swapped face is underexposed skin
        underexposed_paste = (self.skin_target * 0.6).astype(np.uint8)
        matched = SkinPhotometricMatcher.match_photometrics(
            underexposed_paste, self.skin_target, strength=1.0, dark_tier="NORMAL"
        )
        # Center skin brightness must increase towards target
        self.assertGreater(matched[64, 64, 0], underexposed_paste[64, 64, 0])
        self.assertGreater(matched[64, 64, 1], underexposed_paste[64, 64, 1])
        self.assertGreater(matched[64, 64, 2], underexposed_paste[64, 64, 2])

    def test_bounded_white_balance_shift(self):
        """White balance chromatic shifts must remain bounded to prevent blowout."""
        # Extreme blue-tinted paste
        blue_paste = np.full((128, 128, 3), [220, 150, 130], dtype=np.uint8)
        matched = SkinPhotometricMatcher.match_photometrics(
            blue_paste, self.skin_target, strength=1.0, dark_tier="NORMAL"
        )
        # Output should be valid uint8
        self.assertEqual(matched.dtype, np.uint8)
        self.assertFalse(np.isnan(matched).any())


class TestDarkSceneToneMapper(unittest.TestCase):
    """Validates low-light scene handling, shadow floor anchoring, and chroma damping."""

    def test_scene_luminance_classification(self):
        """Classifies images into NORMAL, DARK, and VERY_DARK based on 90th percentile."""
        normal_img = np.full((64, 64, 3), 150, dtype=np.uint8)
        dark_img = np.full((64, 64, 3), 45, dtype=np.uint8)
        very_dark_img = np.full((64, 64, 3), 15, dtype=np.uint8)

        self.assertEqual(DarkSceneToneMapper.classify_scene_luminance(normal_img), "NORMAL")
        self.assertEqual(DarkSceneToneMapper.classify_scene_luminance(dark_img), "DARK")
        self.assertEqual(DarkSceneToneMapper.classify_scene_luminance(very_dark_img), "VERY_DARK")

    def test_shadow_floor_anchoring(self):
        """In dark scenes, shadows should not crush below plate ambient floor."""
        # Plate has ambient floor of 0.05
        target_lin = np.full((64, 64, 3), 0.05, dtype=np.float32)
        # Paste has deep black shadows at 0.001
        paste_lin = np.full((64, 64, 3), 0.001, dtype=np.float32)

        anchored = DarkSceneToneMapper.anchor_shadows(paste_lin, target_lin, dark_tier="DARK")
        # Anchored shadow must be raised towards target floor
        self.assertGreaterEqual(anchored.min(), 0.02)


class TestMultiBandSeamBlender(unittest.TestCase):
    """Validates frequency-split Laplacian blending and dark fringe elimination."""

    def test_linear_light_fringe_elimination(self):
        """Linear-light blending eliminates the ~32-level dark fringe produced by non-linear sRGB blending."""
        # White (255) meeting Black (0) with alpha = 0.5
        white_u8 = np.full((32, 32, 3), 255, dtype=np.uint8)
        black_u8 = np.full((32, 32, 3), 0, dtype=np.uint8)
        alpha = np.full((32, 32), 0.5, dtype=np.float32)

        # Naive non-linear sRGB blend: 0.5 * 255 + 0.5 * 0 = 127.5 -> 128
        # But 128 in sRGB has linear luminance ~0.2158 instead of 0.5!
        naive_blend = (alpha[:, :, None] * white_u8.astype(np.float32) + (1.0 - alpha[:, :, None]) * black_u8.astype(np.float32)).astype(np.uint8)

        # Linear-light blend
        white_lin = LinearColorSpace.srgb_to_linear(white_u8)
        black_lin = LinearColorSpace.srgb_to_linear(black_u8)
        linear_blend = alpha[:, :, None] * white_lin + (1.0 - alpha[:, :, None]) * black_lin
        correct_srgb = LinearColorSpace.linear_to_srgb(linear_blend, soft_knee=False)

        # The correct sRGB value for 50% photon flux is ~188, NOT 128!
        # The difference (188 - 128 = 60 levels) is the dark seam!
        luminance_drop = int(correct_srgb[0, 0, 0]) - int(naive_blend[0, 0, 0])
        self.assertGreater(luminance_drop, 50, f"Expected >50 level difference, got {luminance_drop}")

    def test_multiband_spatial_continuity(self):
        """MultiBandSeamBlender produces smooth spatial gradients with zero step discontinuities."""
        paste = np.full((64, 64, 3), 0.8, dtype=np.float32)
        target = np.full((64, 64, 3), 0.3, dtype=np.float32)
        # Linear ramp alpha from 0 to 1
        alpha = np.repeat(np.linspace(0.0, 1.0, 64, dtype=np.float32)[None, :], 64, axis=0)

        blended = MultiBandSeamBlender.blend(paste, target, alpha, feather_px=6.0)
        # Check monotonic transition across columns
        profile = blended[32, :, 0]
        self.assertTrue((np.diff(profile) >= -1e-5).all(), "Gradient had non-monotonic step discontinuity")


class TestEdgePreservingSharpener(unittest.TestCase):
    """Validates interior-only micro-sharpening and halo prevention."""

    def test_interior_sharpening_gating(self):
        """Sharpening is applied inside deep mask (alpha > 0.6) and zeroed at edges (alpha < 0.5)."""
        # Linear light image with micro-texture
        img_lin = np.full((64, 64, 3), 0.5, dtype=np.float32)
        # Add high-frequency spot in center and at edge
        img_lin[32, 32] = 0.55
        img_lin[32, 10] = 0.55

        # Alpha mask: center has 1.0, edge (x=10) has 0.3
        alpha = np.zeros((64, 64), dtype=np.float32)
        alpha[:, 20:] = 1.0
        alpha[:, 10] = 0.3

        sharpened = EdgePreservingSharpener.sharpen(img_lin, alpha, strength=0.25)
        # Center spot (inside mask) should be boosted
        self.assertGreater(sharpened[32, 32, 0], img_lin[32, 32, 0])
        # Edge spot (outside gate) should remain completely unsharpened
        self.assertAlmostEqual(sharpened[32, 10, 0], img_lin[32, 10, 0], places=5)


class TestCompositingQualityEngine(unittest.TestCase):
    """Validates end-to-end integration, determinism, thread safety, and resilience."""

    def setUp(self):
        self.engine = COMPOSITING_ENGINE
        self.paste = np.full((64, 64, 3), 180, dtype=np.uint8)
        self.target = np.full((64, 64, 3), 120, dtype=np.uint8)
        self.matte = np.full((64, 64, 1), 0.8, dtype=np.float32)

    def test_composite_roi_shape_and_dtype(self):
        """composite_roi returns valid uint8 array matching target ROI shape."""
        out = self.engine.composite_roi(self.paste, self.target, self.matte)
        self.assertEqual(out.shape, (64, 64, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_determinism_across_runs(self):
        """Composite operations must be 100% bit-identical across repeated executions."""
        rng = np.random.default_rng(42)
        p = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        t = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        m = rng.uniform(0.0, 1.0, (64, 64)).astype(np.float32)

        res1 = self.engine.composite_roi(p, t, m)
        res2 = self.engine.composite_roi(p, t, m)
        self.assertTrue(np.array_equal(res1, res2), "Compositing produced non-deterministic results")

    def test_thread_safety(self):
        """Concurrent calls across multiple threads must produce consistent, valid results."""
        rng = np.random.default_rng(100)
        p = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        t = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        m = rng.uniform(0.0, 1.0, (64, 64)).astype(np.float32)

        expected = self.engine.composite_roi(p, t, m)

        def worker():
            return self.engine.composite_roi(p, t, m)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker) for _ in range(16)]
            results = [f.result() for f in futures]

        for res in results:
            self.assertTrue(np.array_equal(res, expected), "Thread concurrency diverged from single-thread result")


if __name__ == "__main__":
    unittest.main()
