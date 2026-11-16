"""Enhance-before-mask temporal blend, velocity motion blur, skin grain transfer, Range Lock spans, and the
track -> faceset resolution the four Batch Matrix strategies share.

Modules under test: ``roop.enhancer_stabilizer``, ``roop.realism_matcher``, ``roop.batch_matrix``,
``TrajectoryCache.track_spans`` and ``routes_trajectory.spans_payload``. Everything here is synthetic and CPU-only:
what it proves is the MECHANISM (variance goes down, the PSD survives, a reset really resets, the resolution is
order-independent). Whether the defaults look right on real footage is a separate question that needs the regression
benchmark and a counterbalanced A/B (AGENTS.md); nothing below claims it.
"""
import hashlib
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import batch_matrix as bm                                     # noqa: E402
from roop import realism_matcher as rm                                  # noqa: E402
from roop.enhancer_stabilizer import TemporalEnhancerBlend              # noqa: E402
from roop.trajectory_cache import ROW, TrajectoryCache                  # noqa: E402

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── synthetic material ───────────────────────────────────────────────────────────────────────────────
def _texture(h=256, w=256, seed=1, sigma=3.0):
    """Smooth, structured, mid-grey BGR texture (so flow has something to lock onto and nothing clips)."""
    rng = np.random.default_rng(seed)
    chans = []
    for c in range(3):
        n = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma)
        n = (n - n.min()) / max(float(n.max() - n.min()), 1e-6)
        chans.append(100.0 + 60.0 * n + 5.0 * c)
    return np.stack(chans, axis=-1).astype(np.float32)


def _u8(x):
    return np.clip(x + 0.5, 0, 255).astype(np.uint8)


def _redraw(base, seed, sigma=7.0):
    """What a restorer does every frame: the same face plus an independent high-frequency texture draw."""
    rng = np.random.default_rng(seed)
    n = cv2.GaussianBlur(rng.standard_normal(base.shape[:2]).astype(np.float32), (0, 0), 0.7)
    n *= sigma / max(float(n.std()), 1e-6)
    return _u8(base + n[..., None])


def _hf_temporal_std(frames):
    """Mean over pixels of the temporal std of the high-pass band: the shimmer the eye reads as boiling skin."""
    hp = [f.astype(np.float32) - cv2.GaussianBlur(f.astype(np.float32), (0, 0), 1.5) for f in frames]
    return float(np.mean(np.std(np.stack(hp), axis=0)))


def _shift(img, dx):
    m = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def _rmse(a, b):
    return float(np.sqrt(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)))


# ═══════════════════════════════ temporal enhancer blend ═════════════════════════════════════════════
class TemporalEnhancerBlendTests(unittest.TestCase):
    N, SKIP = 32, 6

    def _run(self, frames, blend=0.20, **kw):
        eng = TemporalEnhancerBlend(blend=blend, **kw)
        outs = [eng.blend_crop(f, track_id=1, frame_index=i) for i, f in enumerate(frames)]
        return eng, outs

    def test_static_face_loses_the_expected_share_of_high_frequency_variance(self):
        """An EMA with weight w on white noise leaves sqrt((1-w)/(1+w)) of its std: 0.816 at the default 0.20. The
        band must fall by about that much: clearly less shimmer, and no more smoothing than the setting asks for."""
        base = _texture()
        frames = [_redraw(base, 100 + i) for i in range(self.N)]
        eng, outs = self._run(frames)
        ratio = _hf_temporal_std(outs[self.SKIP:]) / _hf_temporal_std(frames[self.SKIP:])
        self.assertLess(ratio, 0.88, f"high-frequency shimmer only fell to {ratio:.3f} of its input")
        self.assertGreater(ratio, 0.70, f"smoothed harder than blend=0.20 allows ({ratio:.3f})")
        s = eng.stats()
        self.assertEqual(s["applied"], self.N - 1)            # frame 0 seeds; every other frame was blended
        self.assertEqual(s["seeded"], 1)

    def test_blend_scales_the_reduction(self):
        base = _texture()
        frames = [_redraw(base, 200 + i) for i in range(self.N)]
        r = {}
        for w in (0.10, 0.20, 0.40):
            _, outs = self._run(frames, blend=w)
            r[w] = _hf_temporal_std(outs[self.SKIP:]) / _hf_temporal_std(frames[self.SKIP:])
        self.assertGreater(r[0.10], r[0.20])
        self.assertGreater(r[0.20], r[0.40])

    def test_moving_head_is_registered_before_it_is_averaged(self):
        """5 px/frame of head motion over sharp, high-contrast content. The flow-registered blend must land closer to
        the noise-free truth than the raw frames, and clearly closer than the same EMA WITHOUT registration, which
        ghosts. (Over a smooth texture the two are indistinguishable, which is why this one is sharp.)"""
        base = np.clip(128.0 + (_texture(sigma=1.5) - 128.0) * 2.0, 0, 255)
        truth = [_shift(base, 5.0 * i) for i in range(self.N)]
        frames = [_redraw(t, 300 + i) for i, t in enumerate(truth)]
        _, outs = self._run(frames)
        naive, prev = [], None
        for f in frames:
            out = f.astype(np.float32) if prev is None else 0.8 * f.astype(np.float32) + 0.2 * prev
            naive.append(out)
            prev = out
        sl = slice(self.SKIP, None)
        e_in = np.mean([_rmse(f, t) for f, t in zip(frames[sl], truth[sl])])
        e_flow = np.mean([_rmse(f, t) for f, t in zip(outs[sl], truth[sl])])
        e_naive = np.mean([_rmse(f, t) for f, t in zip(naive[sl], truth[sl])])
        self.assertLess(e_flow, e_in, "registered blend is no closer to the truth than the raw frames")
        self.assertLess(e_flow, e_naive - 0.3, "flow registration did no better than an unregistered EMA (ghosting)")

    def test_disabled_or_zero_blend_returns_the_input_object(self):
        crop = _u8(_texture())
        off = TemporalEnhancerBlend(blend=0.2, enabled=False)
        self.assertIs(off.blend_crop(crop, 1, 0), crop)
        zero = TemporalEnhancerBlend(blend=0.0)
        self.assertIs(zero.blend_crop(crop, 1, 0), crop)
        self.assertIs(zero.blend_crop(crop, 1, 1), crop)

    def test_non_contiguous_frame_reseeds_and_passes_the_crop_through(self):
        eng = TemporalEnhancerBlend()
        base = _texture()
        eng.blend_crop(_redraw(base, 1), 1, 10)
        eng.blend_crop(_redraw(base, 2), 1, 11)
        jump = _redraw(base, 3)
        out = eng.blend_crop(jump, 1, 40)                       # a parallel block's first frame / a seek
        self.assertIs(out, jump)
        self.assertEqual(eng.stats()["reset_noncontiguous"], 1)
        self.assertEqual(eng.stats()["applied"], 1)

    def test_the_pipelines_cut_set_resets_the_history(self):
        """``cut=True`` is the authoritative signal (``ProcessMgr._shot_boundaries``)."""
        eng = TemporalEnhancerBlend()
        base = _texture()
        for i in range(5):
            eng.blend_crop(_redraw(base, i), 1, i)
        first = _redraw(base, 50)
        out = eng.blend_crop(first, 1, 5, cut=True)
        self.assertIs(out, first, "the first frame of a new shot was blended with the old one")
        self.assertEqual(eng.stats()["reset_shot"], 1)
        eng.blend_crop(_redraw(base, 51), 1, 6)                   # and it blends normally inside the new shot
        self.assertEqual(eng.stats()["applied"], 5)               # 4 in shot one + this one

    def test_an_exposure_step_the_cut_set_missed_is_caught_photometrically(self):
        eng = TemporalEnhancerBlend()
        a = _texture(seed=11)
        for i in range(5):
            eng.blend_crop(_redraw(a, i), 1, i)
        bright = _redraw(np.clip(a + 40.0, 0, 255), 50)
        self.assertIs(eng.blend_crop(bright, 1, 5), bright)
        self.assertEqual(eng.stats()["reset_shot"], 1)

    def test_photometric_line_has_a_known_blind_spot_that_only_the_cut_set_covers(self):
        """A different shot at similar lighting reads to the flow as head motion. This pins that limit so nobody
        relies on the photometric line alone: the pipeline passes ``cut`` for exactly this case."""
        eng = TemporalEnhancerBlend()
        a, b = _texture(seed=11), _texture(seed=99)
        for i in range(5):
            eng.blend_crop(_redraw(a, i), 1, i)
        eng.blend_crop(_redraw(b, 50), 1, 5)
        self.assertEqual(eng.stats()["reset_shot"], 0)

    def test_heavy_occlusion_drops_the_history(self):
        eng = TemporalEnhancerBlend()
        base = _texture()
        eng.blend_crop(_redraw(base, 1), 1, 0)
        eng.blend_crop(_redraw(base, 2), 1, 1)
        occluded = np.ones((64, 64), np.float32)                  # 1 = the plate is restored over the crop
        cur = _redraw(base, 3)
        out = eng.blend_crop(cur, 1, 2, keep=occluded)
        self.assertIs(out, cur)
        self.assertEqual(eng.stats()["reset_occlusion"], 1)
        eng.blend_crop(_redraw(base, 4), 1, 3, keep=np.zeros((64, 64), np.float32))
        self.assertEqual(eng.stats()["applied"], 2)               # frame 1 (before) and frame 3 (clear again)

    def test_a_clean_face_is_not_occluded_just_because_the_crop_is_mostly_background(self):
        """The defect the first real regression render exposed: the paste restores the plate everywhere outside the face
        (neck, hair, background), which is most of an aligned crop. Read over the whole crop that is "occluded" on every
        clean frame and the blend never applied (299 of 312 faces reset). It must be read inside the face interior."""
        from roop.enhancer_stabilizer import occluded_share
        restore = np.ones((256, 256), np.float32)                 # 1 = the plate is restored ...
        cv2.ellipse(restore, (128, 128), (80, 100), 0, 0, 360, 0.0, -1)      # ... except where the swap is kept
        region = np.zeros((256, 256), np.uint8)
        cv2.fillConvexPoly(region, np.int32([[80, 100], [176, 100], [160, 190], [96, 190]]), 1)
        self.assertGreater(occluded_share(restore), 0.5, "premise: most of the crop is plate")
        self.assertEqual(occluded_share(restore, region), 0.0)
        eng = TemporalEnhancerBlend()
        base = _texture()
        for i in range(6):
            eng.blend_crop(_redraw(base, i), 1, i, keep=restore, region=region)
        s = eng.stats()
        self.assertEqual(s["reset_occlusion"], 0)
        self.assertEqual(s["applied"], 5)
        # Without the region the same mask resets every frame: the behaviour that shipped inert.
        whole = TemporalEnhancerBlend()
        for i in range(6):
            whole.blend_crop(_redraw(base, i), 1, i, keep=restore)
        self.assertEqual(whole.stats()["applied"], 0)

    def test_a_hand_over_the_face_interior_does_reset(self):
        from roop.enhancer_stabilizer import occluded_share
        restore = np.ones((256, 256), np.float32)
        cv2.ellipse(restore, (128, 128), (80, 100), 0, 0, 360, 0.0, -1)
        region = np.zeros((256, 256), np.uint8)
        cv2.fillConvexPoly(region, np.int32([[80, 100], [176, 100], [160, 190], [96, 190]]), 1)
        covered = restore.copy()
        covered[150:200, :] = 1.0                                  # a hand across the lower half of the interior
        self.assertGreater(occluded_share(covered, region), 0.35)
        eng = TemporalEnhancerBlend()
        base = _texture()
        eng.blend_crop(_redraw(base, 0), 1, 0, keep=restore, region=region)
        eng.blend_crop(_redraw(base, 1), 1, 1, keep=restore, region=region)
        eng.blend_crop(_redraw(base, 2), 1, 2, keep=covered, region=region)
        self.assertEqual(eng.stats()["reset_occlusion"], 1)

    def test_a_region_of_another_size_is_resampled_not_rejected(self):
        from roop.enhancer_stabilizer import occluded_share
        keep = np.zeros((256, 256), np.float32)
        keep[:, :128] = 1.0
        region = np.ones((64, 64), np.uint8)
        self.assertAlmostEqual(occluded_share(keep, region), 0.5, delta=0.02)

    def test_light_occlusion_does_not_reset(self):
        eng = TemporalEnhancerBlend()
        base = _texture()
        keep = np.zeros((64, 64), np.float32)
        keep[:10] = 1.0                                           # 16 % of the crop
        for i in range(4):
            eng.blend_crop(_redraw(base, i), 1, i, keep=keep)
        self.assertEqual(eng.stats()["reset_occlusion"], 0)
        self.assertEqual(eng.stats()["applied"], 3)

    def test_shape_change_and_missing_frame_index_never_raise(self):
        eng = TemporalEnhancerBlend()
        eng.blend_crop(_u8(_texture(256, 256)), 1, 0)
        small = _u8(_texture(128, 128))
        self.assertIs(eng.blend_crop(small, 1, 1), small)
        self.assertEqual(eng.stats()["reset_shape"], 1)
        crop = _u8(_texture())
        self.assertIs(eng.blend_crop(crop, 1, None), crop)
        self.assertEqual(eng.stats()["skipped_no_key"], 1)

    def test_tracks_do_not_share_history(self):
        eng = TemporalEnhancerBlend()
        a, b = _texture(seed=1), _texture(seed=2)
        eng.blend_crop(_redraw(a, 1), "A", 0)
        eng.blend_crop(_redraw(b, 1), "B", 0)
        eng.blend_crop(_redraw(a, 2), "A", 1)
        eng.blend_crop(_redraw(b, 2), "B", 1)
        self.assertEqual(eng.stats()["applied"], 2)
        self.assertEqual(eng.stats()["reset_shot"], 0)

    def test_clone_for_block_has_its_own_history_but_shares_the_counters(self):
        eng = TemporalEnhancerBlend()
        base = _texture()
        eng.blend_crop(_redraw(base, 1), 1, 0)
        clone = eng.clone_for_block()
        self.assertEqual(len(clone._states), 0)
        clone.blend_crop(_redraw(base, 2), 1, 1)                   # a clone starts empty: this one only seeds
        self.assertEqual(clone.stats()["seeded"], 2)
        self.assertEqual(eng.stats()["seeded"], 2, "a clone's work vanished from the shared report")

    def test_warmup_frames_follow_the_blend(self):
        self.assertEqual(TemporalEnhancerBlend(blend=0.2).warmup_frames(0.01), 3)
        self.assertGreater(TemporalEnhancerBlend(blend=0.5).warmup_frames(0.01),
                           TemporalEnhancerBlend(blend=0.2).warmup_frames(0.01))

    def test_summary_line_says_what_happened(self):
        eng = TemporalEnhancerBlend()
        self.assertIsNone(eng.summary_line())
        base = _texture()
        for i in range(3):
            eng.blend_crop(_redraw(base, i), 1, i)
        line = eng.summary_line()
        self.assertIn("applied 2/3", line)
        self.assertIn("[EnhancerBlend]", line)


# ═══════════════════════════════ trajectory-plan velocity ═══════════════════════════════════════════
_KPS = np.float32([[-30, -20], [30, -20], [0, 0], [-20, 30], [20, 30]])


def _plan(tracks, frame_start=0):
    """``tracks``: {track: [(frame, cx, cy), ...]} -> a TrajectoryCache holding those rows (absolute frames)."""
    entries = sorted(((f, t, cx, cy) for t, pts in tracks.items() for f, cx, cy in pts))
    rows = np.zeros(len(entries), dtype=ROW)
    for i, (f, t, cx, cy) in enumerate(entries):
        rows["frame"][i], rows["track"][i] = f + frame_start, t
        rows["kps"][i] = _KPS + np.float32([cx, cy])
    total = max(f for f, *_ in entries) + 1
    return TrajectoryCache(rows, {"frame_start": frame_start, "total_frames": total})


class PlanVelocityTests(unittest.TestCase):
    def test_central_difference_of_constant_motion(self):
        plan = _plan({1: [(f, 100 + 5 * f, 200) for f in range(10)]})
        vx, vy = rm.plan_velocity(plan, 1, 4, absolute=True)
        self.assertAlmostEqual(vx, 5.0, places=4)
        self.assertAlmostEqual(vy, 0.0, places=4)

    def test_one_sided_at_the_ends_of_a_track(self):
        plan = _plan({1: [(f, 100, 50 - 4 * f) for f in range(6)]})
        self.assertAlmostEqual(rm.plan_velocity(plan, 1, 0, absolute=True)[1], -4.0, places=4)
        self.assertAlmostEqual(rm.plan_velocity(plan, 1, 5, absolute=True)[1], -4.0, places=4)

    def test_a_jump_is_a_cut_not_a_velocity(self):
        plan = _plan({1: [(0, 100, 100), (1, 102, 100), (2, 600, 400), (3, 602, 400)]})
        self.assertIsNone(rm.plan_velocity(plan, 1, 1, absolute=True))
        self.assertIsNone(rm.plan_velocity(plan, 1, 2, absolute=True))

    def test_missing_track_plan_or_neighbours_are_none(self):
        plan = _plan({1: [(0, 100, 100)], 2: [(0, 10, 10), (1, 14, 10)]})
        self.assertIsNone(rm.plan_velocity(plan, 1, 0, absolute=True))       # a one-frame track
        self.assertIsNone(rm.plan_velocity(plan, 9, 0, absolute=True))
        self.assertIsNone(rm.plan_velocity(None, 1, 0))
        self.assertIsNone(rm.plan_velocity(plan, None, 0))
        self.assertIsNone(rm.plan_velocity(plan, 2, None))

    def test_relative_frames_count_from_the_job_start(self):
        plan = _plan({1: [(f, 100 + 6 * f, 100) for f in range(8)]}, frame_start=30)
        self.assertAlmostEqual(rm.plan_velocity(plan, 1, 3, absolute=False)[0], 6.0, places=4)
        self.assertAlmostEqual(rm.plan_velocity(plan, 1, 33, absolute=True)[0], 6.0, places=4)

    def test_velocity_is_per_track(self):
        plan = _plan({1: [(f, 100 + 8 * f, 100) for f in range(6)], 2: [(f, 300, 100 + 1 * f) for f in range(6)]})
        self.assertAlmostEqual(rm.plan_velocity(plan, 1, 2, absolute=True)[0], 8.0, places=4)
        self.assertAlmostEqual(rm.plan_velocity(plan, 2, 2, absolute=True)[1], 1.0, places=4)


# ═══════════════════════════════ velocity motion blur ═══════════════════════════════════════════════
def _dir_energy(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx)), float(np.mean(gy * gy))


M2 = np.float64([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])     # the aligned crop is 2x the frame: 1 px/frame -> 2 px


class VelocityBlurTests(unittest.TestCase):
    def setUp(self):
        rm.reset()
        self.crop = _u8(_texture(256, 256, seed=5, sigma=1.2))

    def _blur(self, v, **kw):
        return rm.velocity_blur(self.crop, v, M2, plate_width=256, **kw)

    def test_slow_head_is_left_untouched(self):
        out, info = self._blur((2.0, 0.0))
        self.assertIs(out, self.crop)
        self.assertEqual(info["reason"], "below threshold")

    def test_the_threshold_is_exclusive(self):
        out, _ = self._blur((3.0, 0.0))
        self.assertIs(out, self.crop)

    def test_just_above_the_threshold_is_continuous(self):
        out, info = self._blur((3.05, 0.0))
        diff = np.abs(out.astype(np.int16) - self.crop.astype(np.int16))
        self.assertLess(float(diff.max()), 6.0, "sharpness popped when the speed crossed 3 px/frame")
        self.assertTrue(info["applied"] or info["reason"])

    def test_horizontal_motion_blurs_horizontally(self):
        out, info = self._blur((6.0, 0.0))
        self.assertTrue(info["applied"])
        ex0, ey0 = _dir_energy(self.crop)
        ex1, ey1 = _dir_energy(out)
        # Isotropic texture loses SOME cross-axis gradient too (its fine detail has an x component), so the property
        # is anisotropy: the gradient along the motion collapses far more than the one across it.
        self.assertLess(ex1 / ex0, 0.40, "the gradient along the motion was not reduced")
        self.assertGreater((ey1 / ey0) / (ex1 / ex0), 2.0, "the blur was not directional")

    def test_vertical_motion_blurs_vertically(self):
        out, _ = self._blur((0.0, -6.0))
        ex0, ey0 = _dir_energy(self.crop)
        ex1, ey1 = _dir_energy(out)
        self.assertLess(ey1 / ey0, 0.40)
        self.assertGreater((ex1 / ex0) / (ey1 / ey0), 2.0)

    def test_diagonal_motion_blurs_both_axes_alike(self):
        out, _ = self._blur((6.0, 6.0))
        ex0, ey0 = _dir_energy(self.crop)
        ex1, ey1 = _dir_energy(out)
        self.assertLess(ex1 / ex0, 0.50)
        self.assertLess(ey1 / ey0, 0.50)
        self.assertAlmostEqual(ex1 / ex0, ey1 / ey0, delta=0.06)

    def test_kernel_length_follows_speed_and_the_crop_scale(self):
        _, slow = self._blur((6.0, 0.0))
        _, fast = self._blur((12.0, 0.0))
        self.assertGreater(fast["length"], slow["length"])
        big = _u8(_texture(512, 512, seed=5, sigma=1.2))
        _, wide = rm.velocity_blur(big, (6.0, 0.0), M2, plate_width=256)
        self.assertAlmostEqual(wide["length"], slow["length"] * 2.0, delta=0.5)

    def test_mean_brightness_is_preserved(self):
        out, _ = self._blur((9.0, 3.0))
        self.assertAlmostEqual(float(out.mean()), float(self.crop.mean()), delta=0.6)

    def test_stands_down_for_the_plate_motion_blur_and_zero_strength(self):
        out, info = self._blur((9.0, 0.0), defer_to_plate_blur=True)
        self.assertIs(out, self.crop)
        self.assertIn("merger_motion_blur", info["reason"])
        out, _ = self._blur((9.0, 0.0), strength=0.0)
        self.assertIs(out, self.crop)

    def test_no_velocity_or_matrix_is_a_counted_no_op(self):
        out, _ = rm.velocity_blur(self.crop, None, M2)
        self.assertIs(out, self.crop)
        out, _ = rm.velocity_blur(self.crop, (9.0, 0.0), None)
        self.assertIs(out, self.crop)
        self.assertIn("2 no plan/velocity", " ".join(rm.summary_lines()))

    def test_the_counter_line_reports_what_ran(self):
        self._blur((9.0, 0.0))
        self._blur((1.0, 0.0))
        line = [x for x in rm.summary_lines() if "velocity blur" in x][0]
        self.assertIn("1/2 face(s) blurred", line)


# ═══════════════════════════════ skin grain transfer ════════════════════════════════════════════════
def _ellipse(h=256, w=256):
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (w // 2, h // 2), (w * 40 // 100, h * 46 // 100), 0, 0, 360, 1.0, -1)
    return m


def _colored_noise(shape, seed, sigma, blur=0.7):
    rng = np.random.default_rng(seed)
    n = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), blur)
    return n * (sigma / max(float(n.std()), 1e-6))


def _radial_psd(patch, bins=10):
    x = patch.astype(np.float64) - patch.mean()
    n = x.shape[0]
    win = np.outer(np.hanning(n), np.hanning(n))
    p = np.abs(np.fft.fftshift(np.fft.fft2(x * win))) ** 2
    yy, xx = np.indices(p.shape)
    r = np.hypot(yy - n // 2, xx - n // 2) / (n / 2.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = np.array([p[(r >= lo) & (r < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])])
    return out / out.sum()


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _luma(img):
    return 0.114 * img[..., 0].astype(np.float32) + 0.587 * img[..., 1] + 0.299 * img[..., 2]


class GrainTransferTests(unittest.TestCase):
    SIGMA_PLATE = 5.0

    def setUp(self):
        rm.reset()
        base = np.full((256, 256, 3), 128.0, np.float32)
        base += cv2.GaussianBlur(np.random.default_rng(0).standard_normal((256, 256)).astype(np.float32),
                                 (0, 0), 25)[..., None] * 6.0
        self.skin = _ellipse()
        self.noise = _colored_noise((256, 256), 7, self.SIGMA_PLATE)
        self.plate = _u8(base + self.noise[..., None])               # grainy footage
        self.swap = _u8(base + _colored_noise((256, 256), 8, 0.4)[..., None])   # the clean restorer output
        self.centre = (slice(80, 176), slice(80, 176))

    def _added(self, **kw):
        out, info = rm.grain_transfer(self.swap, self.plate, self.skin, **kw)
        return out, info, out.astype(np.float32) - self.swap.astype(np.float32)

    def test_adds_grain_to_a_clean_swap_on_a_grainy_plate(self):
        out, info, add = self._added(strength=0.35)
        self.assertTrue(info["applied"], info)
        self.assertGreater(info["sigma_plate"], info["sigma_swap"])
        self.assertFalse(np.array_equal(out, self.swap))
        centre = add[..., 0][self.centre]
        self.assertGreater(float(centre.std()), 0.5)

    def test_added_noise_is_zero_mean_and_does_not_move_the_low_band(self):
        out, _, add = self._added(strength=0.35)
        self.assertLess(abs(float(add[..., 0][self.centre].mean())), 0.4)
        lo = lambda x: cv2.GaussianBlur(x.astype(np.float32), (0, 0), 6.0)
        self.assertLess(float(np.abs(lo(out) - lo(self.swap)).mean()), 0.25, "the grain shifted tone / lighting")

    def test_added_amplitude_is_strength_times_the_deficit(self):
        from roop.grain_match import deficit_sigma, noise_sigma
        s_t = noise_sigma(self.plate, self.skin > 0.6)
        s_p = noise_sigma(self.swap, self.skin > 0.6)
        _, info, add = self._added(strength=0.35)
        want = deficit_sigma(s_t, s_p, 0.35)
        self.assertAlmostEqual(info["added"], want, delta=0.35 * want)
        measured = float(add[..., 0][self.centre].std())
        self.assertAlmostEqual(measured, info["added"], delta=0.30 * info["added"])

    def test_strength_scales_the_amount_linearly(self):
        _, _, a1 = self._added(strength=0.35)
        _, _, a2 = self._added(strength=0.70)
        r = float(a2[..., 0][self.centre].std()) / float(a1[..., 0][self.centre].std())
        self.assertAlmostEqual(r, 2.0, delta=0.3)

    def test_preserves_the_plates_noise_spectrum_better_than_white_noise(self):
        """The point of lifting the plate's OWN high-pass instead of drawing Gaussian noise: the added grain has the
        plate's spectrum. White noise of the same variance (what ``merger_grain_match`` adds) does not."""
        before = hashlib.sha1(self.plate.tobytes()).hexdigest()
        _, _, add = self._added(strength=0.35)
        self.assertEqual(hashlib.sha1(self.plate.tobytes()).hexdigest(), before, "the plate was modified")
        # What the stage claims to transfer is the plate's skin high-pass layer, so that is the reference spectrum.
        hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), rm.GRAIN_SIGMA)
        plate_psd = _radial_psd(hp(_luma(self.plate))[self.centre])
        added_psd = _radial_psd(add[..., 0][self.centre])
        # The control: white Gaussian noise of the SAME variance, as merger_grain_match draws.
        std = float(add[..., 0][self.centre].std())
        white = np.random.default_rng(3).standard_normal((256, 256)).astype(np.float32) * std
        white_psd = _radial_psd(hp(white)[self.centre])
        c_added, c_white = _cos(added_psd, plate_psd), _cos(white_psd, plate_psd)
        self.assertGreater(c_added, 0.95, f"added grain spectrum diverges from the plate's ({c_added:.3f})")
        self.assertGreater(c_added, c_white + 0.10,
                           f"no better than white noise ({c_added:.3f} vs {c_white:.3f})")

    def test_grain_stays_inside_the_skin(self):
        out, _, _ = self._added(strength=0.35)
        far = (slice(0, 24), slice(0, 24))
        self.assertTrue(np.array_equal(out[far], self.swap[far]), "grain leaked outside the skin region")

    def test_strength_zero_is_the_identity(self):
        out, info = rm.grain_transfer(self.swap, self.plate, self.skin, strength=0.0)
        self.assertIs(out, self.swap)
        self.assertEqual(info["reason"], "strength 0")

    def test_a_swap_already_as_noisy_as_the_plate_gets_nothing(self):
        noisy = _u8(self.swap.astype(np.float32) + _colored_noise((256, 256), 9, 8.0)[..., None])
        out, info = rm.grain_transfer(noisy, self.plate, self.skin, strength=0.35)
        self.assertIs(out, noisy)
        self.assertIn("already at least as noisy", info["reason"])
        self.assertIn("1 already at least as noisy", " ".join(rm.summary_lines()))

    def test_no_skin_or_a_sliver_of_skin_is_a_counted_no_op(self):
        out, info = rm.grain_transfer(self.swap, self.plate, None, strength=0.35)
        self.assertIs(out, self.swap)
        tiny = np.zeros((256, 256), np.float32)
        tiny[100:106, 100:106] = 1.0
        out, info = rm.grain_transfer(self.swap, self.plate, tiny, strength=0.35)
        self.assertIs(out, self.swap)
        self.assertIn("skin region", info["reason"])

    def test_a_clean_plate_gets_nothing(self):
        clean_plate = _u8(self.swap.astype(np.float32))
        out, info = rm.grain_transfer(self.swap, clean_plate, self.skin, strength=0.35)
        self.assertIs(out, self.swap)

    def test_a_rerun_adds_identical_grain(self):
        a, _, _ = self._added(strength=0.35)
        b, _, _ = self._added(strength=0.35)
        self.assertTrue(np.array_equal(a, b))

    def test_plate_of_another_size_is_resampled_not_rejected(self):
        small = cv2.resize(self.plate, (128, 128), interpolation=cv2.INTER_AREA)
        out, info = rm.grain_transfer(self.swap, small, self.skin, strength=0.35)
        self.assertEqual(out.shape, self.swap.shape)


# ═══════════════════════════ Range Lock spans + track -> faceset resolution ═══════════════════════════
class TrackSpanTests(unittest.TestCase):
    def test_a_track_that_leaves_and_returns_has_two_ranges(self):
        plan = _plan({1: [(f, 100, 100) for f in list(range(0, 10)) + list(range(20, 30))],
                      2: [(f, 50, 50) for f in range(5, 16)]})
        spans = {s["track"]: s for s in plan.track_spans()}
        self.assertEqual(spans[1]["ranges"], [[0, 9], [20, 29]])
        self.assertEqual(spans[1]["frames"], 20)
        self.assertEqual((spans[1]["first"], spans[1]["last"]), (0, 29))
        self.assertEqual(spans[2]["ranges"], [[5, 15]])
        self.assertEqual([s["track"] for s in plan.track_spans()], [1, 2])      # ascending

    def test_a_one_frame_hole_is_bridged_by_default_and_a_wider_one_is_not(self):
        plan = _plan({1: [(f, 0, 0) for f in (0, 1, 2, 4, 5, 9, 10)]})
        self.assertEqual(plan.track_spans(max_gap=1)[0]["ranges"], [[0, 5], [9, 10]])
        self.assertEqual(plan.track_spans(max_gap=0)[0]["ranges"], [[0, 2], [4, 5], [9, 10]])
        self.assertEqual(plan.track_spans(max_gap=10)[0]["ranges"], [[0, 10]])

    def test_relative_spans_count_from_the_first_frame_of_the_job(self):
        plan = _plan({1: [(f, 0, 0) for f in range(0, 5)]}, frame_start=100)
        self.assertEqual(plan.track_spans(relative=False)[0]["ranges"], [[100, 104]])
        self.assertEqual(plan.track_spans(relative=True)[0]["ranges"], [[0, 4]])

    def test_an_empty_plan_has_no_spans(self):
        self.assertEqual(TrajectoryCache(np.zeros(0, dtype=ROW)).track_spans(), [])

    def test_spans_payload_lists_unbound_tracks_with_the_reason(self):
        pytest = __import__("pytest")
        pytest.importorskip("fastapi")
        from routes_trajectory import spans_payload
        plan = _plan({1: [(f, 0, 0) for f in range(5)], 2: [(f, 0, 0) for f in range(3, 8)]})
        entries = bm.gallery_entries([{"id": os.path.abspath("alice.fsz"), "name": "alice.fsz"}])
        binds = bm.resolve_track_facesets({1: (0, None), 2: (None, None)}, entries)
        out = spans_payload(plan, binds)
        by = {t["track"]: t for t in out["tracks"]}
        self.assertEqual((out["assigned"], out["named"]), (1, 1))
        self.assertEqual(by[1]["source_index"], 0)
        self.assertIsNone(by[2]["source_index"])
        self.assertTrue(by[1]["faceset_path"].endswith("alice.fsz"))
        self.assertIsNone(by[2]["faceset_id"])
        self.assertTrue(by[2]["reason"])
        self.assertEqual(by[2]["ranges"], [[3, 7]])

    def test_the_timeline_publishes_the_tracks_in_ui_numbering(self):
        from roop import timeline_tracks as tt
        tt.begin_run("clip.mp4", 100, 50)
        tt.publish_track_spans([{"track": 3, "frames": 5, "ranges": [(0, 4)], "faceset_id": "x", "faceset_path": "x",
                                 "via": "prepass", "reason": None}])
        maps = tt.render_maps("clip.mp4")
        self.assertEqual(maps["tracks"][0]["ranges"], [[101, 105]])
        self.assertEqual(maps["tracks"][0]["faceset_id"], "x")
        self.assertEqual(tt.render_maps("other.mp4")["tracks"], [])
        tt.begin_run("clip.mp4", 0, 10)
        self.assertEqual(tt.render_maps("clip.mp4")["tracks"], [])             # a new run forgets the last one


def _gallery(*names):
    return [{"id": os.path.abspath(n), "path": os.path.abspath(n), "name": n} for n in names]


class TrackFacesetResolutionTests(unittest.TestCase):
    def setUp(self):
        self.entries = bm.gallery_entries(_gallery("alice.fsz", "bob.fsz", "carol.fsz"))
        self.ids = [e["id"] for e in self.entries]

    def _payload(self, **extra):
        p = {"face_mapping": [1, 0], "source_mapping_ids": [self.ids[1], self.ids[0]],
             "source_mapping_names": ["bob.fsz", "alice.fsz"], "swap_mode": "multi_person"}
        p.update(extra)
        return p

    def test_all_four_strategies_resolve_the_same_tracks_to_the_same_facesets(self):
        person = {10: 0, 11: 1, 12: 0}
        expect = {10: self.ids[1], 11: self.ids[0], 12: self.ids[1]}
        results = []
        for strategy in ("one_to_many", "grouped", "matrix", "recipes"):
            got = bm.plan_track_facesets(self._payload(batch_strategy=strategy), self.entries, person)
            self.assertEqual({t: b.faceset_id for t, b in got.items()}, expect, strategy)
            results.append({t: b.as_dict() for t, b in got.items()})
        self.assertTrue(all(r == results[0] for r in results), "strategies disagree about the same payload")
        tracks = bm.plan_track_facesets(
            {"batch_strategy": "tracks",
             "track_bindings": {"10": "bob.fsz", "11": self.ids[0], "12": self.ids[1]}}, self.entries)
        self.assertEqual({t: b.faceset_id for t, b in tracks.items()}, expect)

    def test_the_answer_does_not_depend_on_dict_or_track_order(self):
        a = bm.plan_track_facesets(self._payload(), self.entries, {10: 0, 11: 1, 12: 0}, tracks=[12, 11, 10])
        b = bm.plan_track_facesets(self._payload(), self.entries, {12: 0, 10: 0, 11: 1}, tracks=[10, 12, 11])
        self.assertEqual(list(a), [10, 11, 12])
        self.assertEqual(list(a), list(b))
        self.assertEqual({t: x.as_dict() for t, x in a.items()}, {t: x.as_dict() for t, x in b.items()})

    def test_ids_beat_positions_when_the_gallery_is_reordered(self):
        reordered = bm.gallery_entries(_gallery("carol.fsz", "alice.fsz", "bob.fsz"))
        got = bm.plan_track_facesets(self._payload(), reordered, {10: 0, 11: 1})
        self.assertEqual(got[10].faceset_id, os.path.abspath("bob.fsz"))
        self.assertEqual(got[11].faceset_id, os.path.abspath("alice.fsz"))
        self.assertEqual(got[10].via, "id")

    def test_names_then_raw_index_are_the_fallbacks_in_that_order(self):
        no_ids = self._payload()
        del no_ids["source_mapping_ids"]
        self.assertEqual(bm.plan_track_facesets(no_ids, self.entries, {1: 0})[1].via, "name")
        bare = {"face_mapping": [2], "swap_mode": "selected"}
        got = bm.plan_track_facesets(bare, self.entries, {1: 0})
        self.assertEqual(got[1].via, "index")
        self.assertEqual(got[1].faceset_id, self.ids[2])

    def test_unresolved_tracks_are_never_faceset_zero(self):
        payload = self._payload(face_mapping=[1, -1], source_mapping_ids=[self.ids[1], None])
        got = bm.plan_track_facesets(payload, self.entries, {10: 0, 11: 1, 12: None, 13: 7})
        self.assertEqual(got[10].faceset_id, self.ids[1])
        for t in (11, 12, 13):
            self.assertIsNone(got[t].faceset_id, t)
            self.assertEqual(got[t].via, "none")
            self.assertTrue(got[t].reason)

    def test_explicit_bindings_refuse_a_gallery_position_and_an_ambiguous_name(self):
        twin = bm.gallery_entries([{"id": os.path.abspath("a/x.fsz"), "name": "x.fsz"},
                                   {"id": os.path.abspath("b/x.fsz"), "name": "x.fsz"}])
        got = bm.plan_track_facesets({"track_bindings": {"1": 0, "2": "x.fsz", "3": "nobody.fsz"}}, twin)
        self.assertIsNone(got[1].faceset_id)
        self.assertIn("not a stable reference", got[1].reason)
        self.assertIn("2 facesets are named", got[2].reason)
        self.assertIn("no faceset matches", got[3].reason)
        exact = bm.plan_track_facesets({"track_bindings": {"2": os.path.abspath("b/x.fsz")}}, twin)
        self.assertEqual(exact[2].source_index, 1)

    def test_a_memory_faceset_has_an_id_but_no_path(self):
        entries = bm.gallery_entries([{"id": "memory:abc123", "name": "Face 1"}])
        got = bm.resolve_track_facesets({5: (0, None)}, entries)
        self.assertEqual(got[5].faceset_id, "memory:abc123")
        self.assertIsNone(got[5].faceset_path)

    def test_the_prepass_map_is_resolved_against_the_runs_own_list(self):
        from types import MappingProxyType
        tsm = MappingProxyType({7: (1, np.zeros(4)), 3: (0, np.zeros(4)), 5: (None, np.zeros(4)),
                                9: (4, np.zeros(4)), 2: 1})
        got = bm.resolve_track_facesets(tsm, self.entries)
        self.assertEqual(list(got), [2, 3, 5, 7, 9])                       # ascending, whatever the map's order
        self.assertEqual(got[3].faceset_id, self.ids[0])
        self.assertEqual(got[7].faceset_id, self.ids[1])
        self.assertEqual(got[2].faceset_id, self.ids[1])                   # a bare int works too
        self.assertIsNone(got[5].faceset_id)
        self.assertIn("no source", got[5].reason)
        self.assertIsNone(got[9].faceset_id)
        self.assertIn("not in this run", got[9].reason)
        self.assertEqual(got[3].via, "prepass")

    def test_an_assigned_faceset_without_an_id_is_still_an_assignment(self):
        """The defect the first real regression render exposed: the benchmark builds its source FaceSet directly, so it
        has faces but no ``_source_id`` / ``_source_path``. The pre-pass assigned track 0 to it ("1 matched to a source")
        and the record said the track was unbound because the faceset had no NAME. Assigned is not nameable."""
        class FS:                                                  # what FaceSet() looks like with a face appended
            faces = [object()]
        entries = bm.gallery_entries([FS()])
        got = bm.resolve_track_facesets({0: (0, None)}, entries)
        self.assertEqual(got[0].source_index, 0)
        self.assertIsNone(got[0].faceset_id)
        self.assertIsNone(got[0].faceset_path)
        self.assertIn("no id or path", got[0].reason)
        self.assertEqual(got[0].via, "prepass")

    def test_an_empty_run_faceset_is_not_a_binding(self):
        entries = bm.gallery_entries([object(), self.entries[0]])          # a skipped person's empty FaceSet()
        got = bm.resolve_track_facesets({1: (0, None), 2: (1, None)}, entries)
        self.assertIsNone(got[1].faceset_id)
        self.assertIsNone(got[1].source_index)
        self.assertIn("empty faceset", got[1].reason)
        self.assertEqual(got[2].faceset_id, self.ids[0])

    def test_active_bindings_are_replaced_and_cleared(self):
        bm.set_active(bm.resolve_track_facesets({1: (0, None)}, self.entries), "p1")
        self.assertEqual(list(bm.active()), [1])
        self.assertEqual(bm.active_json()[0]["faceset_id"], self.ids[0])
        bm.clear_active()
        self.assertEqual(bm.active(), {})

    def test_strategy_is_read_from_the_payload_or_inferred_from_its_shape(self):
        self.assertEqual(bm.strategy_of({"batch_strategy": "recipes"}), "matrix")
        self.assertEqual(bm.strategy_of({"batch_strategy": "multi-track"}), "tracks")
        self.assertEqual(bm.strategy_of({"track_bindings": {"1": "a"}}), "tracks")
        self.assertEqual(bm.strategy_of({"face_mapping": [0]}), "one_to_many")


# ═══════════════════════════ where the stages sit, and that they are wired ═══════════════════════════
def _src(*parts):
    with open(os.path.join(APP, *parts), encoding="utf-8") as fh:
        return fh.read()


class PipelineOrderTests(unittest.TestCase):
    """A source-order pin, not a behavioural proof: it fails if a refactor moves a stage across a neighbour. The
    behaviour of each stage is covered above; this is the claim that they run in THIS order, per face, per frame."""

    def test_enhance_then_temporal_blend_then_realism_then_mask_and_paste(self):
        src = _src("roop", "ProcessMgr.py")
        order = ["elif (p.type == 'enhance'",           # the restorer, inside the processor loop
                 "_es.apply(",                           # whole-crop motion-adaptive blend
                 "_hf.stabilize(",                       # flow-warped HF carry
                 "_te.blend_crop(",                      # flow-registered whole-crop EMA (this change)
                 "_rm.velocity_blur(",                   # head-velocity motion blur
                 "self.apply_merger_post(",              # the DFL merger ops (sharpen, plate blur, grain, ...)
                 "_rm.grain_transfer(",                  # sensor grain last, so nothing blurs or sharpens it
                 "# ── Apply manual mask in canonical face-crop space"]
        at = []
        for marker in order:
            self.assertEqual(src.count(marker), 1 if marker != "self.apply_merger_post(" else 2, marker)
            at.append(src.index(marker))
        self.assertEqual(at, sorted(at), "a stage moved across its neighbour: %s" % list(zip(order, at)))

    def test_nothing_enhances_after_the_paste(self):
        paste = _src("roop", "procmgr_masking.py")
        self.assertNotIn("p.type == 'enhance'", paste)
        self.assertNotIn("Enhance_", paste)
        self.assertNotIn("face_enhancer", paste)

    def test_the_blend_is_cloned_per_block_and_counted_in_the_warmup(self):
        self.assertIn("'hf_stabilizer', 'temporal_enhancer'", _src("roop", "ProcessMgr.py"))
        self.assertIn("'_temporal_enhancer'", _src("roop", "ProcessMgr.py"))
        stab = _src("roop", "procmgr_stabilization.py")
        self.assertEqual(stab.count("_temporal_enhancer"), 2)
        self.assertIn("_temporal_enhancer", _src("roop", "scene_detector.py"))

    def test_the_stage_is_gated_on_an_enhancer_being_in_the_chain(self):
        src = _src("roop", "ProcessMgr.py")
        self.assertIn("self._temporal_enhancer.enabled = _has_enhancer", src)


class SettingsWiringTests(unittest.TestCase):
    KEYS = {"temporal_enhancer_blend": 0.20, "velocity_motion_blur": 1.0, "grain_match_strength": 0.35}

    def test_settings_default_to_the_shipped_values_and_globals_stay_neutral(self):
        import tempfile
        from settings import Settings
        import roop.globals as g
        with tempfile.TemporaryDirectory() as d:
            cfg = Settings(os.path.join(d, "config.yaml"))
        for key, value in self.KEYS.items():
            self.assertEqual(getattr(cfg, key), value, key)
            self.assertEqual(float(getattr(g, key)), 0.0, f"roop.globals.{key} must default to neutral")

    def test_they_are_saved(self):
        src = _src("settings.py")
        for key in self.KEYS:
            self.assertIn(f"'{key}': self.{key}", src)

    def test_the_request_helper_applies_them_to_globals_for_both_render_and_preview(self):
        api = _src("api.py")
        body = api[api.index("def _apply_merger_settings(payload):"):api.index("def _apply_eye_restore_settings")]
        for key in self.KEYS:
            self.assertIn(f'"{key}"', body)

    def test_they_are_part_of_the_baseline_signature(self):
        from roop.benchmark.regression import SIGNATURE_KEYS
        for key in self.KEYS:
            self.assertIn(key, SIGNATURE_KEYS)


if __name__ == "__main__":
    unittest.main()
