"""Stage 18: flow-advected mask fusion, hysteresis and the guided matte.

Masks: 1 = keep the original (occluded), 0 = swap. Each behavioural claim has its
control: the same input through a plain threshold / a plain EMA must show the
defect the fuser exists to remove.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.mask_fusion import (FlowHysteresisFuser, FusionMaskStabilizer,          # noqa: E402
                              guided_matte, hysteresis_ramp)

N = 64


def _crop(seed=0):
    # Coarse random blocks, upsampled: strong contrast (grey std ~70) so that "the
    # picture changed here" is a large, unambiguous change, not a few grey levels.
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8), (N, N), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(img)


def _flat(p):
    return np.full((N, N), p, np.float32)


class HysteresisRampTest(unittest.TestCase):

    def test_a_clear_pixel_needs_the_high_threshold(self):
        r = hysteresis_ramp(_flat(0.50), np.zeros((N, N), bool))
        self.assertLess(float(r.mean()), 0.05)
        r = hysteresis_ramp(_flat(0.70), np.zeros((N, N), bool))
        self.assertGreater(float(r.mean()), 0.95)

    def test_an_occluded_pixel_stays_until_the_low_threshold(self):
        on = np.ones((N, N), bool)
        self.assertGreater(float(hysteresis_ramp(_flat(0.45), on).mean()), 0.9)
        self.assertLess(float(hysteresis_ramp(_flat(0.30), on).mean()), 0.05)

    def test_a_probability_hovering_between_the_thresholds_does_not_chatter(self):
        """The defect: P wobbles 0.44-0.56 around the cutoff. A plain 0.5 threshold
        flips on every wobble; the hysteresis output flips at most once."""
        rng = np.random.default_rng(1)
        seq = 0.5 + 0.06 * np.sign(rng.standard_normal(40))
        plain = [bool(p > 0.5) for p in seq]
        state = np.zeros((1, 1), bool)
        outs = []
        for p in seq:
            o = hysteresis_ramp(np.array([[p]], np.float32), state)
            state = o > 0.5
            outs.append(bool(state[0, 0]))
        flips = lambda s: sum(a != b for a, b in zip(s[:-1], s[1:]))
        self.assertGreater(flips(plain), 8, 'control: the plain threshold must chatter')
        self.assertLessEqual(flips(outs), 1)


class FuserTest(unittest.TestCase):

    def test_a_steady_mask_passes_through_unchanged_in_shape(self):
        f = FlowHysteresisFuser()
        crop = _crop()
        m = np.zeros((N, N), np.float32)
        m[:, 32:] = 1.0
        for k in range(5):
            out = f.update(m, crop, key=1, frame_index=k)
        self.assertGreater(float(out[:, 40:].mean()), 0.95)
        self.assertLess(float(out[:, :24].mean()), 0.05)

    def test_a_single_frame_dropout_of_an_occluder_is_bridged(self):
        """An occluder the segmenter loses for one frame must not flash the swap on
        over it."""
        f = FlowHysteresisFuser(alpha=0.7)
        crop = _crop()
        on, off = _flat(0.0), _flat(0.0)
        on[10:50, 10:50] = 1.0
        for k in range(4):
            f.update(on, crop, key=1, frame_index=k)
        dropped = f.update(off, crop, key=1, frame_index=4)
        self.assertGreater(float(dropped[20:40, 20:40].mean()), 0.5,
                           'the occluded region must survive one lost frame')
        plain = off                                       # control: a stateless segmenter
        self.assertEqual(float(plain[20:40, 20:40].mean()), 0.0)

    def test_disoccluded_skin_resets_at_once_where_the_picture_changed(self):
        """The flow-consistency weight: where the crop no longer matches the warped
        history, the history is not trusted. Same masks, the picture either does or
        does not change under the occluder."""
        on = _flat(0.0)
        on[10:50, 10:50] = 1.0
        off = _flat(0.0)
        crop_a, crop_b = _crop(0), _crop(0)
        crop_b[10:50, 10:50] = 255 - crop_a[10:50, 10:50]   # the object left: very different pixels there

        def run(second_crop):
            f = FlowHysteresisFuser(alpha=0.9, sigma=14.0)
            for k in range(4):
                f.update(on, crop_a, key=1, frame_index=k)
            return f.update(off, second_crop, key=1, frame_index=4)

        ghost_same = float(run(crop_a)[20:40, 20:40].mean())       # picture unchanged: history persists
        ghost_changed = float(run(crop_b)[20:40, 20:40].mean())    # picture changed: history dropped
        self.assertGreater(ghost_same, 0.5, 'control: with a consistent picture the history holds')
        self.assertLess(ghost_changed, 0.1)

    def test_a_repeated_call_for_the_same_frame_is_idempotent(self):
        f = FlowHysteresisFuser()
        crop = _crop()
        m = _flat(0.0)
        m[20:40, 20:40] = 1.0
        for k in range(3):
            f.update(m, crop, key=1, frame_index=k)
        a = f.update(m, crop, key=1, frame_index=3)
        b = f.update(a, crop, key=1, frame_index=3)           # ProcessMgr hands the result back in
        np.testing.assert_array_equal(a, b)

    def test_a_gap_in_frames_starts_over_rather_than_blending_stale_history(self):
        f = FlowHysteresisFuser()
        crop = _crop()
        full = np.ones((N, N), np.float32)
        for k in range(3):
            f.update(full, crop, key=1, frame_index=k)
        out = f.update(_flat(0.0), crop, key=1, frame_index=9)   # 6 frames later
        self.assertLess(float(out.mean()), 0.05)

    def test_tracks_do_not_share_state(self):
        f = FlowHysteresisFuser()
        crop = _crop()
        for k in range(3):
            f.update(np.ones((N, N), np.float32), crop, key=1, frame_index=k)
        out = f.update(_flat(0.0), crop, key=2, frame_index=3)
        self.assertLess(float(out.mean()), 0.05)

    def test_bad_input_is_returned_not_raised(self):
        f = FlowHysteresisFuser()
        out = f.update(np.zeros((N, N, 3), np.float32), _crop(), key=1, frame_index=0)
        self.assertEqual(out.shape, (N, N, 3))
        self.assertGreaterEqual(f.skipped, 1)


class GuidedMatteTest(unittest.TestCase):

    def test_the_mask_edge_snaps_to_the_image_edge(self):
        """A blurry mask boundary 2 px off a sharp image step: the guided matte moves
        it onto the step (radius 4). Control: the input itself is off."""
        guide = np.zeros((N, N), np.uint8)
        guide[:, 32:] = 220
        mask = np.zeros((N, N), np.float32)
        mask[:, 34:] = 1.0
        mask = cv2.GaussianBlur(mask, (0, 0), 3.0)
        out = guided_matte(mask, guide, radius=4, eps=1e-3)
        edge = lambda m: int(np.argmax(m[N // 2] > 0.5))
        self.assertAlmostEqual(edge(mask), 34, delta=1)
        self.assertAlmostEqual(edge(out), 32, delta=1)

    def test_a_flat_guide_leaves_the_mask_smoothed_not_destroyed(self):
        mask = _flat(0.0)
        mask[:, 20:] = 1.0
        out = guided_matte(mask, np.full((N, N), 90, np.uint8))
        self.assertGreater(float(out[:, 40:].mean()), 0.9)
        self.assertLess(float(out[:, :10].mean()), 0.1)


class FusionStabilizerTest(unittest.TestCase):

    KPS = np.array([[20, 24], [44, 24], [32, 34], [24, 46], [40, 46]], np.float32)

    def test_it_is_a_drop_in_for_the_parent_contract(self):
        s = FusionMaskStabilizer(strength=0.5, motion_beta=0.0)
        crop = _crop()
        m = np.zeros((N, N, 1), np.float32)               # XSeg hands back (H, W, 1)
        m[10:50, 10:50] = 1.0
        out = None
        for t in range(4):
            out = s.apply(m, self.KPS, t, guide=crop, track_id=7)
        self.assertEqual(out.shape, m.shape)
        self.assertEqual(out.dtype, np.float32)

    def test_without_a_guide_or_track_id_it_is_the_parent(self):
        s = FusionMaskStabilizer(strength=0.5, motion_beta=0.0)
        m = _flat(0.5)
        seeded = s.apply(m, self.KPS, 0, guide=None, track_id=None)
        np.testing.assert_array_equal(seeded, m)          # parent seeds by passing through


if __name__ == '__main__':
    unittest.main()
