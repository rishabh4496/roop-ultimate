import os
import unittest

import cv2
import numpy as np

from roop import globals as roop_globals
from roop.procmgr_color import ColorTransferMixin
from roop.procmgr_masking import MaskingMixin
from roop.temporal_compositing import composite_multiband


class TestOcclusionColorPipeline(unittest.TestCase):
    def setUp(self):
        self._saved = {
            "mask_erode_dilate_radius": getattr(roop_globals, "mask_erode_dilate_radius", 0),
            "skin_tone_warmth": getattr(roop_globals, "skin_tone_warmth", 0.0),
            "saturation_match": getattr(roop_globals, "saturation_match", 0.0),
            "color_transfer_mode": getattr(roop_globals, "color_transfer_mode", "rct"),
        }

    def tearDown(self):
        for key, value in self._saved.items():
            setattr(roop_globals, key, value)

    def test_signed_radius_erodes_and_dilates_landmark_contour(self):
        theta = np.linspace(0, 2 * np.pi, 68, endpoint=False)
        landmarks = np.column_stack((80 + 40 * np.cos(theta),
                                     80 + 52 * np.sin(theta))).astype(np.float32)
        mask = MaskingMixin()

        roop_globals.mask_erode_dilate_radius = 0
        neutral = mask.create_landmark_mask(landmarks, (160, 160, 3), 0)
        roop_globals.mask_erode_dilate_radius = 8
        dilated = mask.create_landmark_mask(landmarks, (160, 160, 3), 0)
        roop_globals.mask_erode_dilate_radius = -8
        eroded = mask.create_landmark_mask(landmarks, (160, 160, 3), 0)

        self.assertGreater(int(dilated.sum()), int(neutral.sum()))
        self.assertLess(int(eroded.sum()), int(neutral.sum()))

    def test_neutral_photometric_controls_are_bit_exact(self):
        source = np.full((96, 96, 3), (70, 110, 165), dtype=np.uint8)
        target = np.full((96, 96, 3), (90, 135, 190), dtype=np.uint8)
        roop_globals.color_transfer_mode = "none"
        roop_globals.skin_tone_warmth = 0
        roop_globals.saturation_match = 0

        out = ColorTransferMixin().apply_color_transfer(source, target)
        np.testing.assert_array_equal(out, source)

    def test_warmth_and_saturation_change_only_photometric_output(self):
        source = np.full((128, 128, 3), (55, 92, 145), dtype=np.uint8)
        target = np.full((128, 128, 3), (90, 145, 205), dtype=np.uint8)
        roop_globals.color_transfer_mode = "none"
        roop_globals.skin_tone_warmth = 100
        roop_globals.saturation_match = 1.0

        out = ColorTransferMixin().apply_color_transfer(source, target)
        self.assertEqual(out.shape, source.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreater(float(np.abs(out.astype(np.int16) - source).mean()), 0.0)
        self.assertTrue(np.isfinite(out).all())

    def test_multiband_composite_has_cpu_fallback(self):
        old = os.environ.get("ROOP_GPU_LAPLACIAN_BLEND")
        os.environ["ROOP_GPU_LAPLACIAN_BLEND"] = "0"
        try:
            target = np.zeros((32, 32, 3), dtype=np.uint8)
            paste = np.full((32, 32, 3), 255, dtype=np.uint8)
            alpha = np.zeros((32, 32), dtype=np.float32)
            alpha[8:24, 8:24] = 1.0
            out = composite_multiband(paste, target, alpha, {"feather_px": 2})
            self.assertEqual(out.shape, target.shape)
            self.assertEqual(out.dtype, np.uint8)
            self.assertGreater(int(out[16, 16, 0]), int(out[0, 0, 0]))
        finally:
            if old is None:
                os.environ.pop("ROOP_GPU_LAPLACIAN_BLEND", None)
            else:
                os.environ["ROOP_GPU_LAPLACIAN_BLEND"] = old


if __name__ == "__main__":
    unittest.main()
