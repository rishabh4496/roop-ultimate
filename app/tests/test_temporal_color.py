"""roop/temporal_color.py: the split transform must equal the shipped one, and the recurrence must be safe."""
import os
import sys

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import temporal_color as tc  # noqa: E402


def _crop(seed, size=128, gain=1.0, cast=(0, 0, 0)):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = np.stack([110 + 0.3 * xx, 130 + 0.2 * yy, 150 + 0.1 * (xx + yy)], axis=2)
    img = base * gain + np.array(cast, np.float32) + rng.normal(0, 6, (size, size, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def _mixin():
    try:
        from roop.procmgr_color import ColorTransferMixin
    except Exception as exc:                                # pragma: no cover - light profile
        pytest.skip(f"production colour mixin needs the full stack: {type(exc).__name__}")
    return ColorTransferMixin()


class TestSplitTransformEqualsShipped:
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_lct_is_bit_identical(self, seed):
        src, tgt = _crop(seed, gain=0.8, cast=(10, -5, 0)), _crop(seed + 10, gain=1.1, cast=(-6, 4, 8))
        s_lab, s, t = tc.lct_stats(src, tgt)
        A, b = tc.lct_affine(s, t)
        assert np.array_equal(tc.apply_affine(src, A, b, s_lab), _mixin()._color_transfer_lct(src, tgt))

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_rct_is_within_one_level(self, seed):
        src, tgt = _crop(seed, gain=0.8), _crop(seed + 10, gain=1.1, cast=(-6, 4, 8))
        s_lab, s, t = tc.rct_stats(src, tgt)
        A, b = tc.rct_affine(s, t)
        ours = tc.apply_affine(src, A, b, s_lab).astype(np.int16)
        ref = _mixin()._color_transfer_rct(src, tgt).astype(np.int16)
        assert np.abs(ours - ref).max() <= 2             # one LAB level can be two BGR levels after the matrix


def _stats(kind, src, tgt):
    return (tc.lct_stats if kind == "lct" else tc.rct_stats)(src, tgt)[1:]


class TestRecurrence:
    def _engine(self, mode="transform", alpha=0.15):
        return tc.TemporalColorEMA(True, alpha, mode)

    def _feed(self, eng, frames, kind="lct", track=1, start=0):
        out = []
        for i, (s, t) in enumerate(frames):
            out.append(eng.smooth(track, start + i, kind, *_stats(kind, s, t)))
        return out

    def test_first_frame_is_the_unsmoothed_transform(self):
        s, t = _crop(1, gain=0.8), _crop(2, gain=1.1)
        A0, b0 = tc.lct_affine(*_stats("lct", s, t))
        A, b = self._engine().smooth(1, 0, "lct", *_stats("lct", s, t))
        assert np.array_equal(A, A0) and np.array_equal(b, b0)

    def test_disabled_ordered_off_or_untracked_is_always_the_raw_transform(self):
        s, t = _crop(1), _crop(2, gain=1.05)
        raw = tc.lct_affine(*_stats("lct", s, t))
        for eng, tid in ((tc.TemporalColorEMA(False), 1), (self._engine(), None)):
            for i in range(3):
                got = eng.smooth(tid, i, "lct", *_stats("lct", s, t))
                assert np.array_equal(got[0], raw[0])
        eng = self._engine()
        eng.set_ordered(False)
        assert np.array_equal(eng.smooth(1, 5, "lct", *_stats("lct", s, t))[0], raw[0])

    def test_jitter_is_reduced(self):
        rng = np.random.default_rng(5)
        frames = [(_crop(100 + i, gain=0.8), _crop(200 + i, gain=1.0 + 0.02 * rng.standard_normal())) for i in range(60)]
        eng = self._engine()
        smooth = [b for _, b in self._feed(eng, frames)]
        raw = [tc.lct_affine(*_stats("lct", s, t))[1] for s, t in frames]
        assert np.std(np.diff(smooth, axis=0)) < 0.5 * np.std(np.diff(raw, axis=0))
        assert eng.blended >= 55 and eng.restarts == 1

    @pytest.mark.parametrize("mode", ["stats", "transform"])
    def test_converges_to_a_constant_target(self, mode):
        s, t = _crop(1, gain=0.8), _crop(2, gain=1.1)
        A_raw, b_raw = tc.lct_affine(*_stats("lct", s, t))
        out = self._feed(self._engine(mode), [(s, t)] * 40)
        assert np.allclose(out[-1][1], b_raw, atol=1e-3)

    def test_a_cut_restarts_instead_of_blending(self):
        eng = self._engine()
        dark = [(_crop(1, gain=0.8), _crop(2, gain=0.6)) for _ in range(5)]
        bright = [(_crop(3, gain=0.8), _crop(4, gain=1.7)) for _ in range(3)]
        self._feed(eng, dark)
        r0 = eng.restarts
        A, b = eng.smooth(1, 5, "lct", *_stats("lct", *bright[0]))
        assert eng.restarts == r0 + 1
        assert np.array_equal(b, tc.lct_affine(*_stats("lct", *bright[0]))[1])

    @pytest.mark.parametrize("bad_index", [3, 2, 9])          # repeated, backwards, a gap larger than MAX_GAP
    def test_out_of_order_frames_restart_the_track(self, bad_index):
        eng = self._engine()
        pair = (_crop(1, gain=0.8), _crop(2, gain=1.05))
        self._feed(eng, [pair] * 4)                           # frames 0..3
        r0 = eng.restarts
        eng.smooth(1, bad_index, "lct", *_stats("lct", *pair))
        assert eng.restarts == r0 + 1

    def test_a_short_gap_is_blended_with_the_weight_of_the_frames_it_skipped(self):
        eng = self._engine(alpha=0.15)
        s, t = _crop(1, gain=0.8), _crop(2, gain=1.1)
        eng.smooth(1, 0, "lct", *_stats("lct", s, t))
        t2 = _crop(2, gain=1.14)
        A1, b1 = eng.smooth(1, 2, "lct", *_stats("lct", s, t2))
        first = tc.lct_affine(*_stats("lct", s, t))[1]
        now = tc.lct_affine(*_stats("lct", s, t2))[1]
        a = 1 - 0.85 ** 2
        assert np.allclose(b1, (1 - a) * first + a * now, atol=1e-4)

    def test_tracks_are_independent(self):
        eng = self._engine()
        pair_a, pair_b = (_crop(1, gain=0.8), _crop(2, gain=1.0)), (_crop(3, gain=0.8), _crop(4, gain=1.25))
        for i in range(4):
            ba = eng.smooth(1, i, "lct", *_stats("lct", *pair_a))[1]
            bb = eng.smooth(2, i, "lct", *_stats("lct", *pair_b))[1]
        assert np.allclose(ba, tc.lct_affine(*_stats("lct", *pair_a))[1], atol=1e-3)
        assert np.allclose(bb, tc.lct_affine(*_stats("lct", *pair_b))[1], atol=1e-3)

    def test_clone_for_block_starts_empty_and_warmup_matches_the_alpha(self):
        eng = self._engine(alpha=0.15)
        self._feed(eng, [(_crop(1), _crop(2))] * 3)
        clone = eng.clone_for_block()
        assert clone.states == {} and clone.enabled and clone.alpha == eng.alpha
        assert eng.warmup_frames(0.01) == 29                  # ceil(ln 0.01 / ln 0.85)
        eng.reset()
        assert eng.states == {}


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())
