"""Unit tests for Component 4 (Temporal EMA for Canonical Masks & Color Transfer)
and Component 5 (Restorer High-Frequency De-Shimmer).
"""

import math
import os
import sys
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.procmgr_masking import CanonicalMaskEMA
from roop.procmgr_color import ColorTransferMixin
from roop.temporal_color import TemporalColorEMA, rct_stats, lct_stats, rct_affine, lct_affine
from roop.temporal_compositing import TemporalCompositeController
from roop.enhancer_stabilizer import TemporalEnhancerBlend
from roop.video_temporal import VideoTemporalState


class TestCanonicalMaskEMA(unittest.TestCase):
    def test_canonical_smoothing_reduces_chatter_without_motion(self):
        ema = CanonicalMaskEMA(alpha=0.25, enabled=True, canonical_size=128)
        # Create base mask with jittering boundary
        base = np.zeros((128, 128), dtype=np.float32)
        cv2.circle(base, (64, 64), 40, 1.0, -1)

        rng = np.random.RandomState(42)
        raw_diffs = []
        smooth_diffs = []
        prev_raw = None
        prev_smooth = None

        for i in range(10):
            jitter = base.copy()
            if i % 2 == 1:
                jitter = cv2.GaussianBlur(jitter, (0, 0), 2.5)
            smoothed = ema.smooth(jitter, track_id=1, frame_index=i, motion=0.0)
            if prev_raw is not None:
                raw_diffs.append(float(np.mean(np.abs(jitter - prev_raw))))
                smooth_diffs.append(float(np.mean(np.abs(smoothed - prev_smooth))))
            prev_raw = jitter
            prev_smooth = smoothed

        self.assertLess(np.mean(smooth_diffs), np.mean(raw_diffs) * 0.75)

    def test_camera_motion_warping_prevents_screen_space_lag(self):
        """Under camera motion, screen-space masks move across the frame.

        Warping to canonical space before EMA ensures the mask is smoothed relative to the face,
        preventing trailing ghost artifacts across the screen.
        """
        ema = CanonicalMaskEMA(alpha=0.20, enabled=True, canonical_size=128)
        frame_w, frame_h = 640, 480
        crop_size = 128

        # Simulate camera panning: face moves 30 pixels right per frame in screen coordinates
        prev_smoothed_screen = None
        for i in range(5):
            screen_cx = 150 + i * 30
            screen_cy = 200

            # Affine matrix from screen to canonical crop (centered at screen_cx, screen_cy)
            M = np.array([
                [1.0, 0.0, crop_size / 2.0 - screen_cx],
                [0.0, 1.0, crop_size / 2.0 - screen_cy]
            ], dtype=np.float32)
            IM = cv2.invertAffineTransform(M)

            # Ground truth face mask in screen space
            screen_mask = np.zeros((frame_h, frame_w), dtype=np.float32)
            cv2.circle(screen_mask, (screen_cx, screen_cy), 40, 1.0, -1)

            smoothed_screen = ema.smooth(screen_mask, track_id=10, frame_index=i,
                                         M=M, IM=IM, motion=0.05)

            # The smoothed mask in screen space should stay centered on the current face position
            center_val = smoothed_screen[screen_cy, screen_cx]
            self.assertGreater(center_val, 0.85, f"Center of moving face dropped to {center_val} on frame {i}")

            # Check that the mask didn't leave a ghost trail at the starting position from frame 0
            if i >= 3:
                trail_val = smoothed_screen[200, 150]
                self.assertLess(trail_val, 0.05, f"Ghost trail retained at starting position: {trail_val}")

    def test_motion_gated_factor_increases_adaptation_speed(self):
        ema_static = CanonicalMaskEMA(alpha=0.15, enabled=True, canonical_size=64)
        ema_moving = CanonicalMaskEMA(alpha=0.15, enabled=True, canonical_size=64)

        mask1 = np.ones((64, 64), dtype=np.float32) * 0.2
        mask2 = np.ones((64, 64), dtype=np.float32) * 0.8

        # Seed frame 0
        ema_static.smooth(mask1, track_id=1, frame_index=0, motion=0.0)
        ema_moving.smooth(mask1, track_id=1, frame_index=0, motion=0.10)

        # Update frame 1
        res_static = ema_static.smooth(mask2, track_id=1, frame_index=1, motion=0.0)
        res_moving = ema_moving.smooth(mask2, track_id=1, frame_index=1, motion=0.10)

        # High motion should adapt much closer to mask2 (0.8) than static
        self.assertGreater(float(res_moving.mean()), float(res_static.mean()))

    def test_entering_occluder_adapts_rapidly(self):
        """When an occluder enters the face (mask drops), it must reveal quickly to avoid painting over hands."""
        ema = CanonicalMaskEMA(alpha=0.20, enabled=True, canonical_size=64)
        face_mask = np.ones((64, 64), dtype=np.float32)
        ema.smooth(face_mask, track_id=1, frame_index=0)

        # Hand covers right half: alpha drops from 1.0 to 0.0
        occluded = face_mask.copy()
        occluded[:, 32:] = 0.0

        res = ema.smooth(occluded, track_id=1, frame_index=1)
        # The occluded region should drop sharply (>= 0.75 adaptation)
        self.assertLess(float(res[:, 32:].mean()), 0.35)

    def test_reset_clears_history(self):
        ema = CanonicalMaskEMA(alpha=0.25, enabled=True, canonical_size=64)
        mask = np.ones((64, 64), dtype=np.float32)
        ema.smooth(mask, track_id=1, frame_index=0)
        self.assertIn(1, ema._states)
        ema.reset()
        self.assertEqual(len(ema._states), 0)


class TestTemporalColorEMA(unittest.TestCase):
    def test_motion_gating_adapts_faster_under_movement(self):
        eng = TemporalColorEMA(enabled=True, alpha=0.15)
        # Create statistics
        s_m = np.array([120.0, 130.0, 140.0], dtype=np.float32)
        s_s = np.array([20.0, 15.0, 15.0], dtype=np.float32)
        t_m1 = np.array([100.0, 128.0, 128.0], dtype=np.float32)
        t_s1 = np.array([22.0, 14.0, 14.0], dtype=np.float32)
        t_m2 = np.array([115.0, 128.0, 128.0], dtype=np.float32)

        # Frame 0
        eng.smooth_stats(track_id=1, frame_index=0, kind="rct",
                         source_mean=s_m, source_spread=s_s,
                         target_mean=t_m1, target_spread=t_s1, motion=0.0)

        # Clone state for moving comparison
        eng_moving = TemporalColorEMA(enabled=True, alpha=0.15)
        eng_moving.smooth_stats(track_id=1, frame_index=0, kind="rct",
                                source_mean=s_m, source_spread=s_s,
                                target_mean=t_m1, target_spread=t_s1, motion=0.10)

        # Frame 1: target shifts toward t_m2
        res_static = eng_static = eng.smooth_stats(track_id=1, frame_index=1, kind="rct",
                                                   source_mean=s_m, source_spread=s_s,
                                                   target_mean=t_m2, target_spread=t_s1, motion=0.0)
        res_moving = eng_moving.smooth_stats(track_id=1, frame_index=1, kind="rct",
                                             source_mean=s_m, source_spread=s_s,
                                             target_mean=t_m2, target_spread=t_s1, motion=0.10)

        # Moving response should adapt more toward t_m2[0] = 115.0 than static
        self.assertGreater(float(res_moving[2][0]), float(res_static[2][0]))

    def test_smooth_stats_rct_reduces_variance(self):
        eng = TemporalColorEMA(enabled=True, alpha=0.20)
        s_m = np.array([128.0, 128.0, 128.0], dtype=np.float32)
        s_s = np.array([20.0, 10.0, 10.0], dtype=np.float32)

        rng = np.random.RandomState(99)
        raw_targets = []
        smooth_targets = []

        for i in range(20):
            # Target mean with jitter
            jitter = float(rng.normal(0, 3.0))
            t_m = np.array([120.0 + jitter, 135.0, 130.0], dtype=np.float32)
            t_s = s_s.copy()
            res = eng.smooth_stats(track_id=5, frame_index=i, kind="rct",
                                   source_mean=s_m, source_spread=s_s,
                                   target_mean=t_m, target_spread=t_s, motion=0.0)
            raw_targets.append(float(t_m[0]))
            smooth_targets.append(float(res[2][0]))

        self.assertLess(np.std(np.diff(smooth_targets)), np.std(np.diff(raw_targets)))

    def test_plate_mean_jump_triggers_reset(self):
        """A jump larger than CUT_MEAN (18.0 LAB levels) must restart instead of smearing old grade."""
        eng = TemporalColorEMA(enabled=True, alpha=0.15)
        s_m = np.array([128.0, 128.0, 128.0], dtype=np.float32)
        s_s = np.array([20.0, 10.0, 10.0], dtype=np.float32)

        t_m_dark = np.array([60.0, 128.0, 128.0], dtype=np.float32)
        t_m_bright = np.array([140.0, 128.0, 128.0], dtype=np.float32)  # delta = 80 > 18

        eng.smooth_stats(track_id=1, frame_index=0, kind="rct",
                         source_mean=s_m, source_spread=s_s,
                         target_mean=t_m_dark, target_spread=s_s)

        # On frame 1, scene cut jumps to bright
        res = eng.smooth_stats(track_id=1, frame_index=1, kind="rct",
                               source_mean=s_m, source_spread=s_s,
                               target_mean=t_m_bright, target_spread=s_s)

        # Target mean should be exactly t_m_bright (reset immediately, zero lag/smear)
        np.testing.assert_allclose(res[2], t_m_bright, atol=1e-3)

    def test_procmgr_color_integration(self):
        """ColorTransferMixin._video_color_stats passes motion to temporal engine."""
        class MockMgr(ColorTransferMixin):
            def __init__(self):
                self._tls = SimpleNamespace()
                self._temporal_color_ema = TemporalColorEMA(enabled=True, alpha=0.20)
                self._shot_boundaries = set()

        mgr = MockMgr()
        mgr._tls.video_color_context = None
        mgr._tls.current_track_id = 7
        mgr._tls.frame_idx = 1
        mgr._tls.current_motion = 0.05

        s_m = np.array([128.0, 128.0, 128.0], dtype=np.float32)
        s_s = np.array([20.0, 10.0, 10.0], dtype=np.float32)
        t_m = np.array([125.0, 130.0, 130.0], dtype=np.float32)
        t_s = np.array([21.0, 11.0, 11.0], dtype=np.float32)

        # Prime frame 0
        mgr._tls.frame_idx = 0
        mgr._video_color_stats('rct', s_m, s_s, t_m, t_s)

        # Frame 1
        mgr._tls.frame_idx = 1
        t_m2 = np.array([130.0, 130.0, 130.0], dtype=np.float32)
        res = mgr._video_color_stats('rct', s_m, s_s, t_m2, t_s, motion=0.05)
        # Should be smoothed
        self.assertLess(float(res[2][0]), 130.0)
        self.assertGreater(float(res[2][0]), 125.0)


class TestRestorerHighFrequencyDeShimmer(unittest.TestCase):
    def _synthetic_face_crop(self, size=256, seed=123):
        """Create a synthetic 256 or 512 face crop with smooth skin, distinct eyes and mouth."""
        rng = np.random.RandomState(seed)
        crop = np.full((size, size, 3), 160, dtype=np.uint8)

        # Cheeks / skin tone
        crop[:, :, 0] = 140
        crop[:, :, 1] = 160
        crop[:, :, 2] = 185

        # Left eye (y: 28-38%, x: 25-40%)
        ey1, ey2 = int(0.28 * size), int(0.38 * size)
        ex1, ex2 = int(0.25 * size), int(0.40 * size)
        crop[ey1:ey2, ex1:ex2] = (40, 40, 40)

        # Right eye (y: 28-38%, x: 60-75%)
        ex3, ex4 = int(0.60 * size), int(0.75 * size)
        crop[ey1:ey2, ex3:ex4] = (40, 40, 40)

        # Mouth (y: 65-80%, x: 35-65%)
        my1, my2 = int(0.65 * size), int(0.80 * size)
        mx1, mx2 = int(0.35 * size), int(0.65 * size)
        crop[my1:my2, mx1:mx2] = (70, 70, 160)

        return crop

    def test_deshimmer_reduces_skin_pore_variance(self):
        """On static skin texture, generative pore noise shimmer is significantly reduced across frames."""
        vstate = VideoTemporalState(fps=30.0)
        vstate.index = 0
        base_crop = self._synthetic_face_crop(size=256, seed=1)

        rng = np.random.RandomState(42)
        raw_skin_diffs = []
        smooth_skin_diffs = []
        prev_raw = None
        prev_smooth = None

        # Forehead patch (static skin)
        fy1, fy2, fx1, fx2 = 20, 60, 80, 170

        for idx in range(8):
            vstate.index = idx
            # Add independent generative pore shimmer
            shimmer = (rng.randn(256, 256, 3) * 6.0).astype(np.float32)
            cur_crop = np.clip(base_crop.astype(np.float32) + shimmer, 0, 255).astype(np.uint8)

            guide = base_crop.copy()
            out_crop = vstate.texture(tid=1, key=('texture', 'gpen_realistic', 0, (256, 256)),
                                      crop=cur_crop, guide=guide, keep=None)

            if prev_raw is not None:
                raw_diffs = float(np.mean(np.abs(cur_crop[fy1:fy2, fx1:fx2].astype(np.float32) -
                                                 prev_raw[fy1:fy2, fx1:fx2].astype(np.float32))))
                smooth_diffs = float(np.mean(np.abs(out_crop[fy1:fy2, fx1:fx2].astype(np.float32) -
                                                    prev_smooth[fy1:fy2, fx1:fx2].astype(np.float32))))
                raw_skin_diffs.append(raw_diffs)
                smooth_skin_diffs.append(smooth_diffs)

            prev_raw = cur_crop
            prev_smooth = out_crop

        # High-frequency shimmer should be reduced by at least 40% on static skin
        self.assertLess(np.mean(smooth_skin_diffs), np.mean(raw_skin_diffs) * 0.60)

    def test_blinking_remains_crisp_zero_ghosting(self):
        """When an eye blinks (eyelid closes), gate drops to 0 so the closed eyelid has zero ghosting of the open eye."""
        vstate = VideoTemporalState(fps=30.0)
        vstate.index = 0
        open_crop = self._synthetic_face_crop(size=256, seed=10)

        # Closed eye crop (eyelid skin replaces dark eye)
        closed_crop = open_crop.copy()
        ey1, ey2 = int(0.28 * 256), int(0.38 * 256)
        ex1, ex2 = int(0.25 * 256), int(0.40 * 256)
        closed_crop[ey1:ey2, ex1:ex2] = (140, 160, 185)

        # Feed frame 0: open eye
        vstate.index = 0
        vstate.texture(tid=1, key=('texture', 'ultramax', 0, (256, 256)),
                       crop=open_crop, guide=open_crop, keep=None)

        # Frame 1: person blinks!
        vstate.index = 1
        blink_out = vstate.texture(tid=1, key=('texture', 'ultramax', 0, (256, 256)),
                                   crop=closed_crop, guide=closed_crop, keep=None)

        # The eyelid area in blink_out must be crisp closed eyelid skin (mean ~ 160), NOT smeared with dark open eye (~40)
        eyelid_pixels = blink_out[ey1:ey2, ex1:ex2]
        self.assertGreater(float(eyelid_pixels.mean()), 145.0,
                           f"Eyelid was ghosted with dark open eye: mean {eyelid_pixels.mean()}")

    def test_speaking_remains_crisp_zero_ghosting(self):
        """When mouth opens / speaks, mouth gate drops to 0 so open mouth is not ghosted with closed mouth."""
        vstate = VideoTemporalState(fps=30.0)
        vstate.index = 0
        closed_mouth = self._synthetic_face_crop(size=256, seed=20)

        # Open mouth with visible teeth (bright white teeth inside mouth)
        open_mouth = closed_mouth.copy()
        my1, my2 = int(0.68 * 256), int(0.78 * 256)
        mx1, mx2 = int(0.40 * 256), int(0.60 * 256)
        open_mouth[my1:my2, mx1:mx2] = (240, 240, 240)

        # Frame 0: closed mouth
        vstate.index = 0
        vstate.texture(tid=2, key=('texture', 'codeformer', 0, (256, 256)),
                       crop=closed_mouth, guide=closed_mouth, keep=None)

        # Frame 1: mouth speaks / opens
        vstate.index = 1
        speak_out = vstate.texture(tid=2, key=('texture', 'codeformer', 0, (256, 256)),
                                   crop=open_mouth, guide=open_mouth, keep=None)

        teeth_pixels = speak_out[my1:my2, mx1:mx2]
        # Teeth must remain crisp white (mean > 220), not darkened or ghosted
        self.assertGreater(float(teeth_pixels.mean()), 220.0,
                           f"Teeth were ghosted/darkened by previous closed mouth: mean {teeth_pixels.mean()}")

    def test_enhancer_stabilizer_dynamic_motion_gating(self):
        """TemporalEnhancerBlend drops gate to 0 on blinking/moving areas while stabilizing static areas."""
        teb = TemporalEnhancerBlend(blend=0.30, enabled=True)
        open_crop = self._synthetic_face_crop(size=256, seed=30)
        closed_crop = open_crop.copy()
        ey1, ey2 = int(0.28 * 256), int(0.38 * 256)
        ex1, ex2 = int(0.25 * 256), int(0.40 * 256)
        closed_crop[ey1:ey2, ex1:ex2] = (140, 160, 185)

        # Frame 0
        teb.blend_crop(open_crop, track_id=1, frame_index=0)

        # Frame 1: blink
        out = teb.blend_crop(closed_crop, track_id=1, frame_index=1)

        eyelid_pixels = out[ey1:ey2, ex1:ex2]
        self.assertGreater(float(eyelid_pixels.mean()), 145.0,
                           f"Enhancer stabilizer ghosted the closed eyelid: mean {eyelid_pixels.mean()}")

    def test_process_mgr_restorer_list_contains_all_three_generative_models(self):
        """Verify ProcessMgr recognizes gpen_realistic, ultramax, and codeformer for texture de-shimmer."""
        with open(os.path.join(APP, "roop", "ProcessMgr.py"), "r", encoding="utf-8") as f:
            code = f.read()

        self.assertIn("gpen_realistic", code)
        self.assertIn("ultramax", code)
        self.assertIn("codeformer", code)
        # Verify the specific line exists
        self.assertIn("('gpen_realistic', 'ultramax', 'codeformer')", code)


if __name__ == "__main__":
    unittest.main()
