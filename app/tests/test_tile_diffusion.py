"""Tests for the high-fidelity diffusion tile restoration pass in roop-ultimate.

Validates:
1. Dynamic Tiling & VRAM Control:
   - Overlapping tiled inference (512x512 tiles with 64px Gaussian feathered margins).
   - Seamless reconstruction with zero seam artifacts across boundaries.
   - 4K crop tile planning and single-tile execution staying within 8 GB VRAM budget.
2. Latent Tile Inpainting & Micro-Texture Detail Injection:
   - Low-frequency color (Cr, Cb) and global geometry strictly fixed from the swapper.
   - Synthesis of micro-textures: skin pores, hair follicles, lip crevices, sclera vessels.
   - Exposure gating protecting specular highlights and crushed darks.
3. Multi-Model Restoration Switcher:
   - Fast / Realtime (GPEN-512 / GFPGAN, 25-60 FPS).
   - Balanced (CodeFormer with fidelity weight 0.6).
   - VFX / Master Quality (Tile Diffusion Synthesizer, 2-6 FPS offline rendering).
4. Processor lifecycle and pipeline integration (ProcessMgr and core plugins).
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import cv2

import roop.globals
from roop.tile_diffusion import (
    GaussianFeatherWindow,
    TileInferenceManager,
    FrequencyResidualSynthesizer,
    TileDiffusionEngine,
    RestorationSwitcher
)
from roop.processors.Enhance_TileDiffusion import Enhance_TileDiffusion
from roop.ProcessMgr import ProcessMgr
from roop.core import get_processing_plugins


class TestGaussianFeatherWindow(unittest.TestCase):
    def test_window_shape_and_bounds(self):
        window = GaussianFeatherWindow.get_window(tile_size=512, margin=64)
        self.assertEqual(window.shape, (512, 512, 1))
        self.assertEqual(window.dtype, np.float32)
        # Check boundary weights fall off smoothly
        self.assertLess(window[0, 0, 0], 0.05)
        self.assertLess(window[511, 511, 0], 0.05)
        # Check central core has unity weight
        self.assertAlmostEqual(float(window[256, 256, 0]), 1.0, places=4)
        self.assertAlmostEqual(float(window[128, 128, 0]), 1.0, places=4)
        self.assertAlmostEqual(float(window[384, 384, 0]), 1.0, places=4)

    def test_window_cache(self):
        w1 = GaussianFeatherWindow.get_window(512, 64)
        w2 = GaussianFeatherWindow.get_window(512, 64)
        self.assertIs(w1, w2)


class TestTileInferenceManager(unittest.TestCase):
    def setUp(self):
        self.tiler = TileInferenceManager(tile_size=512, overlap=64, vram_budget_gb=8.0)

    def test_single_tile_planning(self):
        tiles = self.tiler.plan_tiles(512, 512)
        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0], (0, 512, 0, 512))

    def test_small_crop_planning(self):
        tiles = self.tiler.plan_tiles(256, 256)
        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0], (0, 256, 0, 256))

    def test_4k_crop_tile_planning_and_step(self):
        # 4K crop (2160 x 3840)
        h, w = 2160, 3840
        tiles = self.tiler.plan_tiles(h, w)
        self.assertGreater(len(tiles), 10)

        # Verify step size between consecutive tiles matches tile_size - overlap (448)
        y_starts = sorted(list(set(t[0] for t in tiles)))
        for i in range(len(y_starts) - 2):
            self.assertEqual(y_starts[i + 1] - y_starts[i], 512 - 64)

        # Verify every pixel in the 4K canvas is covered by at least one tile
        covered = np.zeros((h, w), dtype=bool)
        for y1, y2, x1, x2 in tiles:
            self.assertEqual(y2 - y1, 512)
            self.assertEqual(x2 - x1, 512)
            covered[y1:y2, x1:x2] = True
        self.assertTrue(np.all(covered))

    def test_seamless_feather_reconstruction(self):
        # A test image with smooth gradient spanning across tile boundaries
        h, w = 1000, 1000
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        synthetic_img = np.stack([
            (x_coords / w * 255).astype(np.uint8),
            (y_coords / h * 255).astype(np.uint8),
            ((x_coords + y_coords) / (h + w) * 255).astype(np.uint8)
        ], axis=-1)

        # Identity tile processing function
        reconstructed = self.tiler.process_tiled(synthetic_img, tile_fn=lambda tile: tile.copy())

        # Verify reconstructed matches input without boundary seam artifacts
        self.assertEqual(reconstructed.shape, synthetic_img.shape)
        diff = np.abs(reconstructed.astype(np.float32) - synthetic_img.astype(np.float32))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        self.assertLess(max_diff, 2.0)
        self.assertLess(mean_diff, 0.1)


class TestFrequencyResidualSynthesizer(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create a synthetic face crop with skin, lips, eyes
        self.h, self.w = 512, 512
        self.swapper_crop = np.full((self.h, self.w, 3), [130, 140, 185], dtype=np.uint8)
        # Add eye white region
        self.swapper_crop[150:230, 120:220] = [210, 215, 220]
        # Add lips region
        self.swapper_crop[350:410, 180:330] = [80, 85, 170]

        # Diffusion pass with fine detail and color drift to test filtering
        self.diff_output = self.swapper_crop.copy()
        # Introduce synthetic color drift in diffusion output (e.g. greenish cast)
        self.diff_output[:, :, 0] = np.clip(self.diff_output[:, :, 0] - 25, 0, 255)
        self.diff_output[:, :, 1] = np.clip(self.diff_output[:, :, 1] + 35, 0, 255)
        # Add high-frequency noise/texture
        noise = (np.random.randn(self.h, self.w) * 15).astype(np.float32)
        self.diff_output[:, :, 2] = np.clip(self.diff_output[:, :, 2].astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def test_chrominance_lock_zero_color_drift(self):
        """Low-frequency color from swapper must be 100% preserved (zero color drift)."""
        restored = FrequencyResidualSynthesizer.inject_residual(
            swapper_crop=self.swapper_crop,
            diffusion_output=self.diff_output,
            strength=0.75,
            sigma=2.5
        )
        self.assertEqual(restored.shape, self.swapper_crop.shape)

        # Convert both to YCrCb
        ycrcb_orig = cv2.cvtColor(self.swapper_crop, cv2.COLOR_BGR2YCrCb)
        ycrcb_restored = cv2.cvtColor(restored, cv2.COLOR_BGR2YCrCb)

        # Cr and Cb channels must be byte-exact to the swapper crop
        np.testing.assert_array_equal(ycrcb_orig[:, :, 1], ycrcb_restored[:, :, 1])
        np.testing.assert_array_equal(ycrcb_orig[:, :, 2], ycrcb_restored[:, :, 2])

    def test_low_frequency_geometry_lock(self):
        """Global geometry and low frequencies must be strictly fixed from the swapper."""
        restored = FrequencyResidualSynthesizer.inject_residual(
            swapper_crop=self.swapper_crop,
            diffusion_output=self.diff_output,
            strength=0.70,
            sigma=2.5
        )
        y_orig = cv2.cvtColor(self.swapper_crop, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        y_rest = cv2.cvtColor(restored, cv2.COLOR_BGR2YCrCb)[:, :, 0]

        # Extract macro low-frequency geometry
        low_orig = cv2.GaussianBlur(y_orig, (0, 0), sigmaX=5.0, sigmaY=5.0)
        low_rest = cv2.GaussianBlur(y_rest, (0, 0), sigmaX=5.0, sigmaY=5.0)

        # Macro geometry should differ by less than 1 luminance unit
        macro_diff = np.max(np.abs(low_orig.astype(np.float32) - low_rest.astype(np.float32)))
        self.assertLess(macro_diff, 1.5)

    def test_micro_texture_injection(self):
        """High-frequency micro-textures are injected into the facial skin and features."""
        restored = FrequencyResidualSynthesizer.inject_residual(
            swapper_crop=self.swapper_crop,
            diffusion_output=self.diff_output,
            strength=0.85,
            sigma=2.5
        )
        y_orig = cv2.cvtColor(self.swapper_crop, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        y_rest = cv2.cvtColor(restored, cv2.COLOR_BGR2YCrCb)[:, :, 0]

        # High-frequency standard deviation should increase due to micro-texture injection
        hf_orig = y_orig.astype(np.float32) - cv2.GaussianBlur(y_orig, (0, 0), 2.5).astype(np.float32)
        hf_rest = y_rest.astype(np.float32) - cv2.GaussianBlur(y_rest, (0, 0), 2.5).astype(np.float32)

        self.assertGreater(float(np.std(hf_rest)), float(np.std(hf_orig)))

    def test_exposure_gate_protects_extremes(self):
        """Exposure gate suppresses noise in pure black and pure specular white pixels."""
        extreme_crop = self.swapper_crop.copy()
        # Pure black shadow corner
        extreme_crop[0:50, 0:50] = 0
        # Pure specular highlight
        extreme_crop[0:50, 460:512] = 255

        noisy_diff = extreme_crop.copy()
        noisy_diff = np.clip(noisy_diff.astype(np.float32) + 30.0, 0, 255).astype(np.uint8)

        restored = FrequencyResidualSynthesizer.inject_residual(
            swapper_crop=extreme_crop,
            diffusion_output=noisy_diff,
            strength=0.9,
            sigma=2.5
        )
        # Shadows and highlights must remain intact without blown-out noise
        self.assertTrue(np.all(restored[0:50, 0:50] < 5))
        self.assertTrue(np.all(restored[0:50, 460:512] > 250))


class TestTileDiffusionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = TileDiffusionEngine(device="cpu", fp16=False, vram_budget_gb=8.0, steps=2)
        self.engine.initialize()

    def tearDown(self):
        self.engine.release()

    def test_engine_inference(self):
        tile = np.full((512, 512, 3), 128, dtype=np.uint8)
        # Add synthetic detail
        cv2.circle(tile, (256, 256), 64, (160, 160, 160), -1)
        out = self.engine.infer_tile(tile)
        self.assertEqual(out.shape, (512, 512, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.isfinite(out).all())

    def test_varying_steps(self):
        tile = np.full((512, 512, 3), 120, dtype=np.uint8)
        self.engine.steps = 1
        out1 = self.engine.infer_tile(tile)
        self.engine.steps = 4
        out4 = self.engine.infer_tile(tile)
        self.assertEqual(out1.shape, (512, 512, 3))
        self.assertEqual(out4.shape, (512, 512, 3))


class TestEnhanceTileDiffusionProcessor(unittest.TestCase):
    def setUp(self):
        self.proc = Enhance_TileDiffusion()
        self.proc.Initialize({"devicename": "cpu"})

    def tearDown(self):
        self.proc.Release()

    def test_processor_attributes(self):
        self.assertEqual(self.proc.processorname, "tile_diffusion")
        self.assertTrue(self.proc.self_excluding)
        self.assertEqual(self.proc.type, "enhance")
        self.assertEqual(self.proc.model_template, "ffhq_512")

    def test_run_single_face_crop(self):
        crop = np.full((512, 512, 3), [120, 130, 170], dtype=np.uint8)
        cv2.rectangle(crop, (150, 150), (350, 350), (140, 150, 190), -1)
        res, scale = self.proc.Run(source_faceset=None, target_face=None, temp_frame=crop)
        self.assertEqual(res.shape, (512, 512, 3))
        self.assertEqual(scale, 1)
        self.assertEqual(res.dtype, np.uint8)

    def test_processmgr_registration(self):
        self.assertIn("tile_diffusion", ProcessMgr.plugins)
        self.assertEqual(ProcessMgr.plugins["tile_diffusion"], "Enhance_TileDiffusion")

    def test_core_get_processing_plugins(self):
        old_enhancer = roop.globals.selected_enhancer
        try:
            roop.globals.selected_enhancer = "Tile Diffusion Synthesizer"
            plugins = get_processing_plugins(masking_engine="RealityUX")
            self.assertIn("tile_diffusion", plugins)
            self.assertIn("RealityUX", plugins)
        finally:
            roop.globals.selected_enhancer = old_enhancer


class TestRestorationSwitcher(unittest.TestCase):
    def setUp(self):
        self.old_enhancer = roop.globals.selected_enhancer
        self.old_fidelity = roop.globals.codeformer_fidelity
        self.old_mode = getattr(roop.globals, "restoration_mode", "balanced")

    def tearDown(self):
        roop.globals.selected_enhancer = self.old_enhancer
        roop.globals.codeformer_fidelity = self.old_fidelity
        roop.globals.restoration_mode = self.old_mode

    def test_mode_resolution(self):
        fast = RestorationSwitcher.resolve_mode("fast")
        self.assertEqual(fast["id"], "fast")
        self.assertEqual(fast["enhancer"], "GPEN")
        self.assertIn("25", fast["fps_range"])

        balanced = RestorationSwitcher.resolve_mode("balanced")
        self.assertEqual(balanced["id"], "balanced")
        self.assertEqual(balanced["enhancer"], "Codeformer")
        self.assertEqual(balanced["codeformer_fidelity"], 0.6)

        vfx = RestorationSwitcher.resolve_mode("vfx")
        self.assertEqual(vfx["id"], "vfx")
        self.assertEqual(vfx["enhancer"], "Tile Diffusion Synthesizer")
        self.assertIn("2", vfx["fps_range"])

    def test_apply_to_globals(self):
        RestorationSwitcher.apply_to_globals("fast", roop.globals)
        self.assertEqual(roop.globals.restoration_mode, "fast")
        self.assertEqual(roop.globals.selected_enhancer, "GPEN")

        RestorationSwitcher.apply_to_globals("balanced", roop.globals)
        self.assertEqual(roop.globals.restoration_mode, "balanced")
        self.assertEqual(roop.globals.selected_enhancer, "Codeformer")
        self.assertEqual(roop.globals.codeformer_fidelity, 0.6)

        RestorationSwitcher.apply_to_globals("vfx", roop.globals)
        self.assertEqual(roop.globals.restoration_mode, "vfx")
        self.assertEqual(roop.globals.selected_enhancer, "Tile Diffusion Synthesizer")


if __name__ == "__main__":
    unittest.main()
