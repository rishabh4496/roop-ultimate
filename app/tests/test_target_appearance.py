"""Phase 10 target-conditioned lighting/color tests and regressions."""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals as g
from roop.appearance_conditioning import (DARK, NORMAL, VERY_DARK,
                                           TargetAppearanceStabilizer,
                                           analyze_target_appearance,
                                           classify_low_light,
                                           protect_restorer_output)
from roop.procmgr_color import ColorTransferMixin
from roop.procmgr_merger import MergerMixin


class _Color(ColorTransferMixin):
    pass


class _Merger(MergerMixin):
    def __init__(self):
        self.seen = []

    def apply_sharpen(self, image, strength):
        self.seen.append(("sharpen", float(strength)))
        return image

    def apply_clarity(self, image, strength):
        self.seen.append(("clarity", float(strength)))
        return image


def _target(value=150, cast=None, gradient=False):
    h = w = 160
    if gradient:
        x = np.linspace(0.35, 1.0, w, dtype=np.float32)[None, :]
        y = np.linspace(0.85, 1.0, h, dtype=np.float32)[:, None]
        field = x * y
        image = np.full((h, w, 3), value, np.float32) * field[:, :, None]
    else:
        image = np.full((h, w, 3), value, np.float32)
    if cast == "warm":
        image[:, :, 0] *= 0.62
        image[:, :, 2] *= 1.18
    elif cast == "blue":
        image[:, :, 0] *= 1.25
        image[:, :, 2] *= 0.58
    return np.clip(image, 0, 255).astype(np.uint8)


class TargetAppearanceTest(unittest.TestCase):

    def setUp(self):
        self._old = {
            "color_transfer_mode": g.color_transfer_mode,
            "target_conditioned_appearance": g.target_conditioned_appearance,
            "target_conditioned_appearance_strength": g.target_conditioned_appearance_strength,
        }
        g.color_transfer_mode = "rct"
        g.target_conditioned_appearance = True
        g.target_conditioned_appearance_strength = 0.75

    def tearDown(self):
        for key, value in self._old.items():
            setattr(g, key, value)

    def test_low_light_tiers_cover_normal_dark_and_very_dark(self):
        self.assertEqual(analyze_target_appearance(_target(175))["tier"], NORMAL)
        self.assertEqual(analyze_target_appearance(_target(70))["tier"], DARK)
        self.assertEqual(analyze_target_appearance(_target(28))["tier"], VERY_DARK)
        self.assertEqual(classify_low_light({"mean": 0.8, "p50": 0.8, "p90": 0.9}), NORMAL)

    def test_disabled_conditioning_keeps_legacy_color_path_bit_identical(self):
        source = _target(150, cast="warm")
        target = _target(100, cast="blue")
        appearance = analyze_target_appearance(target)
        g.target_conditioned_appearance = False
        legacy = _Color().apply_color_transfer(source, target)
        disabled = _Color().apply_color_transfer(source, target, appearance=appearance)
        np.testing.assert_array_equal(disabled, legacy)

    def test_spatial_illumination_preserves_target_shadow_direction(self):
        source = _target(175)
        target = _target(175, gradient=True)
        appearance = analyze_target_appearance(target)
        output = _Color().apply_color_transfer(source, target, appearance=appearance)
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).astype(np.float32)
        target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Measure inside the face support; the outer crop is deliberately
        # protected from appearance changes because it may contain hair/background.
        left_right = float(gray[:, 88:112].mean() - gray[:, 48:72].mean())
        target_left_right = float(target_gray[:, 88:112].mean() - target_gray[:, 48:72].mean())
        self.assertGreater(left_right, 4.0)
        self.assertGreater(left_right / max(1.0, target_left_right), 0.35)

    def test_blue_and_warm_scene_casts_are_followed_without_whitening(self):
        source = _target(150, cast="warm")
        for cast in ("warm", "blue"):
            target = _target(120, cast=cast)
            appearance = analyze_target_appearance(target)
            output = _Color().apply_color_transfer(source, target, appearance=appearance)
            out_b, _, out_r = [float(v) for v in output.mean(axis=(0, 1))]
            tar_b, _, tar_r = [float(v) for v in target.mean(axis=(0, 1))]
            self.assertLess(abs((out_r - out_b) - (tar_r - tar_b)), 42.0)
            # A blue cast must remain blue relative to a neutral face.
            if cast == "blue":
                self.assertGreater(out_b - out_r, 8.0)

    def test_very_dark_restorer_is_pulled_back_toward_target_conditioned_input(self):
        reference = _target(28)
        bright = _target(220)
        protected = protect_restorer_output(bright, reference, VERY_DARK)
        self.assertLess(float(protected.mean()), 75.0)
        self.assertGreater(float(protected.mean()), float(reference.mean()))

    def test_low_resolution_and_motion_blur_remain_finite_and_conservative(self):
        target = cv2.GaussianBlur(cv2.resize(_target(42, cast="blue"), (48, 48)),
                                  (9, 9), 2.0)
        appearance = analyze_target_appearance(target)
        source = cv2.resize(_target(42), (48, 48), interpolation=cv2.INTER_AREA)
        output = _Color().apply_color_transfer(source, target,
                                               appearance=appearance)
        self.assertEqual(output.shape, target.shape)
        self.assertTrue(np.isfinite(output).all())
        self.assertEqual(appearance["tier"], VERY_DARK)

    def test_temporal_stabilizer_reduces_stable_light_color_jitter(self):
        stabilizer = TargetAppearanceStabilizer(enabled=True, alpha=0.30)
        raw, filtered = [], []
        for i in range(16):
            image = _target(150, cast="warm" if i % 2 else "blue")
            appearance = analyze_target_appearance(image)
            raw.append(float(appearance["color_temperature"]))
            filtered.append(float(stabilizer.update(7, appearance)["color_temperature"]))
        self.assertLess(float(np.mean(np.abs(np.diff(filtered[2:])))),
                        float(np.mean(np.abs(np.diff(raw[2:])))))

    def test_merger_sharpen_and_clarity_are_reduced_in_dark_scene(self):
        old = {name: getattr(g, name) for name in (
            "merger_sharpen", "merger_clarity", "merger_hist_match",
            "merger_motion_blur", "merger_grain_match", "merger_degrade")}
        try:
            g.merger_sharpen = 0.8
            g.merger_clarity = 0.8
            for name in ("merger_hist_match", "merger_motion_blur",
                         "merger_grain_match", "merger_degrade"):
                setattr(g, name, 0.0)
            merger = _Merger()
            merger.apply_merger_post(_target(50), _target(50), {"tier": DARK})
            self.assertEqual(merger.seen, [("clarity", 0.4), ("sharpen", 0.4)])
        finally:
            for name, value in old.items():
                setattr(g, name, value)


if __name__ == "__main__":
    unittest.main()
