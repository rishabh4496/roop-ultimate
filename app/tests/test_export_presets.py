"""crop_to_fill — the centered-crop math behind one-click export presets.

No ffmpeg execution here, same convention as test_enhancer_align.py testing
affine math without running a model: this is pure arithmetic, so it is tested
as pure arithmetic.
"""

import os
import sys
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

from roop.util_ffmpeg import crop_to_fill                       # noqa: E402
from routes_export import PRESETS                               # noqa: E402


class TestCropToFill(unittest.TestCase):
    SOURCES = {
        "portrait_9x16": (1080, 1920),
        "landscape_16x9": (1920, 1080),
        "square_1x1": (1000, 1000),
        "classic_4x3": (1200, 900),
    }

    def test_crop_is_nonnegative_and_bounded(self):
        for name, (sw, sh) in self.SOURCES.items():
            for key, preset in PRESETS.items():
                l, r, t, b = crop_to_fill(sw, sh, preset["w"], preset["h"])
                for pct in (l, r, t, b):
                    self.assertGreaterEqual(pct, 0.0, f"{name}/{key}: negative crop {pct}")
                self.assertLess(l + r, 100.0, f"{name}/{key}: cropped the whole width away")
                self.assertLess(t + b, 100.0, f"{name}/{key}: cropped the whole height away")

    def test_crop_is_centered(self):
        for name, (sw, sh) in self.SOURCES.items():
            for key, preset in PRESETS.items():
                l, r, t, b = crop_to_fill(sw, sh, preset["w"], preset["h"])
                self.assertAlmostEqual(l, r, places=6, msg=f"{name}/{key}: left != right")
                self.assertAlmostEqual(t, b, places=6, msg=f"{name}/{key}: top != bottom")

    def test_only_one_axis_is_ever_trimmed(self):
        for name, (sw, sh) in self.SOURCES.items():
            for key, preset in PRESETS.items():
                l, r, t, b = crop_to_fill(sw, sh, preset["w"], preset["h"])
                self.assertTrue((l == 0.0 and r == 0.0) or (t == 0.0 and b == 0.0),
                                f"{name}/{key}: both axes trimmed ({l},{r},{t},{b})")

    def test_matching_ratio_is_a_noop(self):
        self.assertEqual(crop_to_fill(1920, 1080, 16, 9), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(crop_to_fill(1080, 1920, 9, 16), (0.0, 0.0, 0.0, 0.0))
        # Equivalent ratios (not just identical numbers) must also no-op.
        self.assertEqual(crop_to_fill(3840, 2160, 16, 9), (0.0, 0.0, 0.0, 0.0))

    def test_wider_than_target_trims_width_only(self):
        # 16:9 source into a 9:16 target: much wider than the target ratio.
        l, r, t, b = crop_to_fill(1920, 1080, 9, 16)
        self.assertGreater(l, 0.0)
        self.assertEqual((t, b), (0.0, 0.0))

    def test_taller_than_target_trims_height_only(self):
        # 9:16 source into a 16:9 target: much taller than the target ratio.
        l, r, t, b = crop_to_fill(1080, 1920, 16, 9)
        self.assertGreater(t, 0.0)
        self.assertEqual((l, r), (0.0, 0.0))

    def test_degenerate_input_returns_zero(self):
        self.assertEqual(crop_to_fill(0, 1080, 9, 16), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(crop_to_fill(1080, 0, 9, 16), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(crop_to_fill(1080, 1920, 0, 16), (0.0, 0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
