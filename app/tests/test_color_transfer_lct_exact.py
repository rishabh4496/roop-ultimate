"""`_color_transfer_lct` and the grayscale guard are bit-identical to the implementation they replaced.

The 'lighting' stage ran 6.7-6.9 ms per call (two calls per swapped face: the 256 px swap crop and
the 512 px post-enhance match) and the 512 px LCT alone was 5.2 ms against a 3 ms budget. The hot
operations were full-image passes that did not need to be: a BGR->LAB conversion of the whole TARGET
when only every 16th pixel is read, `np.clip` + `.astype` allocating 3 MB temporaries per call (ten
worker threads), and a second float buffer. The fix keeps every number the same:

  * the target's LAB is computed on the strided subset only (BGR->LAB is per pixel);
  * the 3x3 statistics, eigen-decompositions and matrix are untouched;
  * the float32 transform output is clamped in place and truncated into a reused per-thread buffer,
    which is exactly `np.clip(x, 0, 255).astype(np.uint8)`.

What is pinned: the output is `np.array_equal` to the reference below (the previous code, verbatim)
on real-looking crops at every size the pipeline feeds it, on shapes whose flat stride wraps, on a
non-contiguous target, on noise and flat images; results do not depend on buffer reuse across calls
or threads; and the guard still returns the same decision.
"""
import os
import sys
import threading

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.procmgr_color import ColorTransferMixin                       # noqa: E402


def reference_lct(source, target):
    """The implementation before this change, verbatim."""
    s_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    t_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    s_f = s_lab.astype(np.float32)
    s_sub = s_lab.reshape(-1, 3)[::16].astype(np.float32)
    t_sub = t_lab.reshape(-1, 3)[::16].astype(np.float32)
    s_mean, t_mean = s_sub.mean(0), t_sub.mean(0)
    eps = np.eye(3, dtype=np.float32) * 1e-4
    s_centered = s_sub - s_mean
    t_centered = t_sub - t_mean
    denominator_s = max(1, s_centered.shape[0] - 1)
    denominator_t = max(1, t_centered.shape[0] - 1)
    Cs = (s_centered.T @ s_centered) / denominator_s + eps
    Ct = (t_centered.T @ t_centered) / denominator_t + eps
    w_s, V_s = np.linalg.eigh(Cs)
    minv_s = (V_s * (1.0 / np.sqrt(np.clip(w_s, 1e-6, None)))) @ V_s.T
    w_t, V_t = np.linalg.eigh(Ct)
    msqrt_t = (V_t * np.sqrt(np.clip(w_t, 0, None))) @ V_t.T
    A = msqrt_t @ minv_s
    offset = t_mean - s_mean @ A.T
    M = np.hstack([A, offset.reshape(3, 1)])
    out_lab = cv2.transform(s_f, M)
    out_u8 = np.clip(out_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_u8, cv2.COLOR_LAB2BGR)


def reference_guard_is_gray(source):
    bg = cv2.absdiff(source[:, :, 0], source[:, :, 1])
    gr = cv2.absdiff(source[:, :, 1], source[:, :, 2])
    return float(cv2.mean(bg)[0]) < 5.0 and float(cv2.mean(gr)[0]) < 5.0


def _scene(seed, size):
    """A face-like crop: smooth skin-toned gradient, darker features, sensor noise."""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    base = np.stack([150 + 40 * yy, 165 + 30 * xx, 205 - 25 * yy], axis=-1)
    base -= 60 * np.exp(-(((xx - 0.35) ** 2 + (yy - 0.4) ** 2) / 0.004))[..., None]
    base -= 50 * np.exp(-(((xx - 0.65) ** 2 + (yy - 0.4) ** 2) / 0.004))[..., None]
    base += rng.randn(size, size, 3) * 4
    return np.clip(base, 0, 255).astype(np.uint8)


def _pair(seed, size):
    target = _scene(seed, size)
    gain = np.array([0.9, 1.0, 1.1], np.float32) * (0.85 + 0.03 * (seed % 9))
    return np.clip(target.astype(np.float32) * gain, 0, 255).astype(np.uint8), target


@pytest.fixture()
def lct():
    return ColorTransferMixin()._color_transfer_lct


class TestLctIsBitIdentical:

    @pytest.mark.parametrize('size', [64, 128, 192, 256, 384, 512, 640, 1024])
    def test_face_like_crops_at_every_pipeline_size(self, lct, size):
        for seed in range(4):
            s, t = _pair(seed, size)
            assert np.array_equal(lct(s, t), reference_lct(s, t)), (size, seed)

    @pytest.mark.parametrize('shape', [(250, 250, 3), (255, 259, 3), (64, 33, 3), (17, 5, 3),
                                       (480, 640, 3), (1, 64, 3)])
    def test_shapes_whose_flat_stride_wraps(self, lct, shape):
        rng = np.random.RandomState(shape[0] * 31 + shape[1])
        s = rng.randint(0, 256, shape, dtype=np.uint8)
        t = rng.randint(0, 256, shape, dtype=np.uint8)
        assert np.array_equal(lct(s, t), reference_lct(s, t))

    def test_noise_flat_and_near_gray(self, lct):
        rng = np.random.RandomState(7)
        flat = np.full((256, 256, 3), 128, np.uint8)
        near_gray = rng.randint(100, 110, (256, 256, 3), dtype=np.uint8)
        noise = rng.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        for s, t in ((flat, noise), (near_gray, noise), (noise, flat), (noise, near_gray),
                     (flat, flat), (noise, noise)):
            assert np.array_equal(lct(s, t), reference_lct(s, t))

    def test_extreme_colours_that_saturate_the_clamp(self, lct):
        """Where clip + truncate matter most: the transform pushes values off both ends."""
        rng = np.random.RandomState(11)
        s = np.where(rng.rand(256, 256, 1) > 0.5, 255, 0).astype(np.uint8) * np.ones((1, 1, 3), np.uint8)
        t = rng.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        t[..., 0] //= 8
        assert np.array_equal(lct(s, t), reference_lct(s, t))
        assert np.array_equal(lct(t, s), reference_lct(t, s))

    def test_a_non_contiguous_target_view(self, lct):
        rng = np.random.RandomState(13)
        big = rng.randint(0, 256, (300, 600, 3), dtype=np.uint8)
        s = rng.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        t = big[10:266, 20:276]
        assert not t.flags['C_CONTIGUOUS']
        assert np.array_equal(lct(s, t), reference_lct(s, np.ascontiguousarray(t)))

    def test_neither_input_is_modified(self, lct):
        s, t = _pair(3, 256)
        s0, t0 = s.copy(), t.copy()
        lct(s, t)
        assert np.array_equal(s, s0) and np.array_equal(t, t0)

    def test_the_result_is_a_fresh_array_each_call(self, lct):
        """The scratch buffers are reused across calls; the returned image must not be."""
        s, t = _pair(1, 256)
        a = lct(s, t)
        b = lct(s, t)
        assert a is not b and not np.shares_memory(a, b)
        s2, t2 = _pair(5, 256)
        lct(s2, t2)
        assert np.array_equal(a, reference_lct(s, t)), 'a later call changed an earlier result'

    def test_alternating_sizes_on_one_thread_as_the_pipeline_does(self, lct):
        """Swap crop (256) then post-enhance match (512), per face, on the same worker."""
        for seed in range(6):
            for size in (256, 512):
                s, t = _pair(seed, size)
                assert np.array_equal(lct(s, t), reference_lct(s, t))

    def test_concurrent_threads_do_not_share_scratch(self, lct):
        pairs = [_pair(i, 256 if i % 2 else 512) for i in range(12)]
        expected = [reference_lct(s, t) for s, t in pairs]
        got = [None] * len(pairs)
        errors = []

        def work(i):
            try:
                for _ in range(8):
                    got[i] = lct(*pairs[i])
            except Exception as exc:                      # noqa: BLE001
                errors.append(exc)
        threads = [threading.Thread(target=work, args=(i,)) for i in range(len(pairs))]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert not errors
        assert all(np.array_equal(g, e) for g, e in zip(got, expected))


class TestGrayscaleGuard:
    """`apply_color_transfer` skips a grayscale source; the decision must not move."""

    def _skips(self, source):
        # Mirrors apply_color_transfer's guard exactly, through the public method with mode 'lct'.
        import roop.globals as g
        saved = (getattr(g, 'color_transfer_mode', None), getattr(g, 'skin_tone_warmth', 0.0),
                 getattr(g, 'saturation_match', 0.0), getattr(g, 'target_conditioned_appearance', False))
        g.color_transfer_mode, g.skin_tone_warmth, g.saturation_match = 'lct', 0.0, 0.0
        g.target_conditioned_appearance = False
        try:
            out = ColorTransferMixin().apply_color_transfer(source, _scene(2, source.shape[0]))
            return out is source
        finally:
            (g.color_transfer_mode, g.skin_tone_warmth, g.saturation_match,
             g.target_conditioned_appearance) = saved

    def test_gray_is_skipped_and_colour_is_not(self):
        gray = np.repeat(np.random.RandomState(0).randint(0, 256, (256, 256, 1), dtype=np.uint8), 3, axis=2)
        assert self._skips(gray) and reference_guard_is_gray(gray)
        colour = _scene(4, 256)
        assert not self._skips(colour) and not reference_guard_is_gray(colour)

    @pytest.mark.parametrize('tint', [0, 3, 4, 5, 6, 12])
    def test_the_threshold_sits_where_it_did(self, tint):
        rng = np.random.RandomState(tint)
        g = rng.randint(40, 200, (128, 128, 1), dtype=np.uint8)
        img = np.repeat(g, 3, axis=2).astype(np.int16)
        img[:, :, 0] += tint
        img = np.clip(img, 0, 255).astype(np.uint8)
        assert self._skips(img) == reference_guard_is_gray(img)
