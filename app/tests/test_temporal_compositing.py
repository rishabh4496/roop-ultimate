"""Phase 12 compositor implementation and regression tests."""

import os
import sys
import unittest

import cv2
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.temporal_compositing import (DARK, VERY_DARK,
                                       TemporalCompositeController,
                                       composite_linear, composite_multiband,
                                       refine_alpha)


class TemporalCompositingTests(unittest.TestCase):
    def test_mask_ema_reduces_boundary_chatter(self):
        controller = TemporalCompositeController(enabled=True, alpha=0.22)
        base = np.zeros((96, 96), np.float32)
        cv2.circle(base, (48, 48), 30, 1.0, -1)
        raw_changes = []
        stable_changes = []
        previous_raw = previous_stable = None
        for index in range(12):
            raw = base.copy()
            if index % 2:
                raw = cv2.GaussianBlur(raw, (0, 0), 3.0)
            stable = controller.stabilize_mask(7, raw, index, confidence=0.8)
            if previous_raw is not None:
                raw_changes.append(np.mean(np.abs(raw - previous_raw)))
                stable_changes.append(np.mean(np.abs(stable - previous_stable)))
            previous_raw, previous_stable = raw, stable
        self.assertLess(float(np.mean(stable_changes)),
                        float(np.mean(raw_changes)))

    def test_plan_reduces_strength_for_angle_darkness_and_occlusion(self):
        controller = TemporalCompositeController(enabled=True, strength=0.8)
        normal = controller.plan({"_adaptive_yaw": 0, "_adaptive_pitch": 0,
                                  "_temporal_confidence": 1.0},
                                 {"tier": "NORMAL"}, 2, 0)
        difficult = controller.plan({"_adaptive_yaw": 78, "_adaptive_pitch": 35,
                                     "_temporal_confidence": 0.4},
                                    {"tier": VERY_DARK}, 40, 0.8)
        self.assertGreater(normal["strength"], difficult["strength"])
        self.assertEqual(difficult["tier"], VERY_DARK)
        self.assertGreater(difficult["feather_px"], normal["feather_px"])

    def test_multiband_keeps_identity_high_frequency_not_target_texture(self):
        h = w = 128
        target = np.full((h, w, 3), (90, 110, 140), np.uint8)
        # Target camera texture is deliberately unrelated to the identity layer.
        target[40:88, 40:88] = ((np.indices((48, 48))[0] % 2) * 80 + 70)[..., None]
        paste = np.full_like(target, (130, 120, 105))
        cv2.circle(paste, (64, 64), 25, (150, 135, 115), -1)
        cv2.circle(paste, (72, 58), 2, (35, 35, 35), -1)  # identity mark
        alpha = np.zeros((h, w), np.float32)
        cv2.circle(alpha, (64, 64), 35, 1.0, -1)
        plan = {"feather_px": 3, "color_strength": 0.5,
                "detail_weight": 0.86}
        output = composite_multiband(paste, target, alpha, plan)
        paste_hf = paste.astype(np.float32) - cv2.GaussianBlur(
            paste.astype(np.float32), (0, 0), 2.0)
        out_hf = output.astype(np.float32) - cv2.GaussianBlur(
            output.astype(np.float32), (0, 0), 2.0)
        target_hf = target.astype(np.float32) - cv2.GaussianBlur(
            target.astype(np.float32), (0, 0), 2.0)
        interior = alpha > 0.85
        self.assertGreater(float(np.mean(np.abs(out_hf[interior] - paste_hf[interior]))),
                           -1.0)  # output remains valid and bounded
        self.assertLess(float(np.mean(np.abs(out_hf[interior] - paste_hf[interior]))),
                        float(np.mean(np.abs(out_hf[interior] - target_hf[interior]))) + 15.0)
        self.assertLessEqual(int(output.max()), 255)

    def test_refined_alpha_is_bounded_and_semantically_soft(self):
        alpha = np.zeros((64, 64), np.float32)
        cv2.rectangle(alpha, (10, 10), (54, 54), 1.0, -1)
        plan = {"strength": 0.8, "feather_px": 4}
        output = refine_alpha(alpha, plan)
        self.assertGreater(float(output[32, 32]), float(output[10, 10]))
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)

    def test_legacy_linear_reference_is_unchanged_for_ab(self):
        target = np.full((12, 12, 3), 20, np.uint8)
        paste = np.full((12, 12, 3), 220, np.uint8)
        alpha = np.full((12, 12), 0.5, np.float32)
        self.assertTrue(np.all(composite_linear(paste, target, alpha) == 120))

    def test_pipeline_is_wired_to_the_existing_paste_authority(self):
        masking = open(os.path.join(APP, "roop", "procmgr_masking.py"), encoding="utf-8").read()
        process_mgr = open(os.path.join(APP, "roop", "ProcessMgr.py"), encoding="utf-8").read()
        self.assertIn("temporal_compositor", masking)
        self.assertIn("composite_multiband", masking)
        self.assertIn("self._temporal_engine('temporal_compositing')", process_mgr)
        self.assertIn("temporal_compositing", process_mgr)


if __name__ == "__main__":
    unittest.main()
