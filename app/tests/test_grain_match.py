"""roop/grain_match.py: the estimator, the variance arithmetic, and the wiring into the real ``paste_upscale``.

The estimator and the arithmetic are pure NumPy/OpenCV. The wiring tests drive ``MaskingMixin.paste_upscale`` (the
production paste) on a synthetic plate whose finest-scale noise is KNOWN, with ``ROOP_GRAIN_MATCH=0`` as the control:
a feature that is read, defined and never run would pass every test that only imports it.
"""
import os
import sys
import threading
import types

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import grain_match as gm  # noqa: E402


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("ROOP_GRAIN_MATCH", raising=False)
    gm.reset()


def _plate(h, w, sigma, seed):
    yy, xx = np.mgrid[0:h, 0:w]
    base = 90.0 + 0.12 * xx + 0.08 * yy
    base[:, w // 2:] += 25.0                                       # an edge, so the estimator is not tested on flat ground
    noise = np.random.default_rng(seed).normal(0, sigma, (h, w, 1)) if sigma > 0 else 0.0
    return np.clip(np.repeat((base[:, :, None] + noise), 3, axis=2), 0, 255).round().astype(np.uint8)


class TestEstimator:
    @pytest.mark.parametrize("sigma", [1.0, 2.0, 4.0, 8.0])
    def test_recovers_white_noise_level_across_an_edge(self, sigma):
        est = gm.noise_sigma(_plate(300, 300, sigma, 3))
        assert abs(est - sigma) / sigma < 0.05, (sigma, est)

    def test_clean_image_reads_at_the_quantisation_floor(self):
        assert gm.noise_sigma(_plate(300, 300, 0.0, 1)) < 0.35

    def test_too_few_pixels_is_none_not_zero(self):
        img = _plate(100, 100, 3.0, 2)
        where = np.zeros((100, 100), bool)
        where[:5, :5] = True
        assert gm.noise_sigma(img, where) is None

    def test_float_and_uint8_agree(self):
        img = _plate(200, 200, 3.0, 4)
        assert gm.noise_sigma(img.astype(np.float32)) == pytest.approx(gm.noise_sigma(img), abs=1e-6)


class TestArithmetic:
    def test_variances_add_not_standard_deviations(self):
        assert gm.deficit_sigma(5.0, 2.0) == pytest.approx(np.sqrt(21.0))

    def test_no_deficit_when_the_paste_is_noisier(self):
        assert gm.deficit_sigma(1.0, 3.0) == 0.0

    def test_strength_scales_the_added_sigma(self):
        assert gm.deficit_sigma(5.0, 2.0, 0.5) == pytest.approx(0.5 * np.sqrt(21.0))
        assert gm.deficit_sigma(5.0, 2.0, 0.0) == 0.0

    def test_env_strength(self, monkeypatch):
        assert gm.strength() == 1.0
        monkeypatch.setenv("ROOP_GRAIN_MATCH", "0")
        assert gm.strength() == 0.0
        monkeypatch.setenv("ROOP_GRAIN_MATCH", "7")
        assert gm.strength() == 1.0                                 # clamped


class TestMatch:
    H = W = 160

    def _matte(self):
        m = np.zeros((self.H, self.W), np.float32)
        cv2.ellipse(m, (self.W // 2, self.H // 2), (55, 65), 0, 0, 360, 1.0, -1)
        return cv2.GaussianBlur(m, (15, 15), 0)

    def test_reaches_the_plate_floor_and_leaves_outside_the_matte_alone(self):
        target = _plate(self.H, self.W, 5.0, 11)
        paste = _plate(self.H, self.W, 1.0, 12).astype(np.float32)
        matte = self._matte()
        out, info = gm.match(paste, target, matte, frame_index=3, track_id=1)
        assert info["applied"] and info["added"] == pytest.approx(np.sqrt(25.0 - 1.0), rel=0.1)
        core = cv2.erode((matte > 0.95).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        assert gm.noise_sigma(out, core) == pytest.approx(gm.noise_sigma(target, core), rel=0.08)
        assert np.array_equal(out[matte == 0], paste[matte == 0])

    def test_noisier_paste_is_returned_untouched(self):
        target = _plate(self.H, self.W, 0.0, 21)
        paste = _plate(self.H, self.W, 4.0, 22).astype(np.float32)
        out, info = gm.match(paste, target, self._matte())
        assert out is paste and not info["applied"] and "already" in info["reason"]

    def test_tiny_matte_says_so(self):
        m = np.zeros((self.H, self.W), np.float32)
        m[80, 80] = 1.0
        out, info = gm.match(_plate(self.H, self.W, 1, 1).astype(np.float32), _plate(self.H, self.W, 5, 2), m)
        assert not info["applied"] and "fewer than" in info["reason"]

    def test_same_inputs_same_grain_and_new_frame_new_grain(self):
        target = _plate(self.H, self.W, 5.0, 31)
        paste = _plate(self.H, self.W, 0.0, 32).astype(np.float32)
        a, _ = gm.match(paste, target, self._matte(), frame_index=7, track_id=2, origin=(10, 20))
        b, _ = gm.match(paste, target, self._matte(), frame_index=7, track_id=2, origin=(10, 20))
        c, _ = gm.match(paste, target, self._matte(), frame_index=8, track_id=2, origin=(10, 20))
        assert np.array_equal(a, b) and not np.array_equal(a, c)

    def test_grain_is_zero_mean(self):
        target = _plate(self.H, self.W, 6.0, 41)
        paste = _plate(self.H, self.W, 0.0, 42).astype(np.float32)
        out, _ = gm.match(paste, target, self._matte(), frame_index=1)
        assert abs(float(np.mean(out - paste))) < 0.1

    def test_counters_say_what_happened(self):
        target, paste = _plate(self.H, self.W, 5.0, 51), _plate(self.H, self.W, 0.0, 52).astype(np.float32)
        gm.match(paste, target, self._matte(), frame_index=1)
        gm.match(target.astype(np.float32), paste.astype(np.uint8), self._matte(), frame_index=2)
        text = " ".join(gm.summary_lines())
        assert "1/2 face(s) got grain" in text and "1 already at least as noisy" in text
        gm.reset()
        assert gm.summary_lines() == []


# --------------------------------------------------------------------------------------------------------------
# The wiring: the REAL paste_upscale, with the feature off as the control.
# --------------------------------------------------------------------------------------------------------------
def _stub():
    try:
        from roop.procmgr_masking import MaskingMixin
    except Exception as exc:                                         # pragma: no cover - light profile
        pytest.skip(f"production paste needs the full stack: {type(exc).__name__}")

    class Stub(MaskingMixin):
        def __init__(self):
            self.options = types.SimpleNamespace(show_face_area_overlay=False, blend_ratio=1.0)
            self._tls = threading.local()
    return Stub()


H, W, CROP, FACE = 480, 640, 512, 230.0


def _matrix():
    s = CROP * 0.8 / FACE
    cx, cy = W * 0.5, H * 0.5
    t = np.array([CROP / 2.0, CROP / 2.0]) - s * np.array([cx, cy])
    return np.array([[s, 0, t[0]], [0, s, t[1]]], np.float32), (cx, cy)


def _hull(cx, cy):
    a = np.linspace(0, 2 * np.pi, 106, endpoint=False)
    return np.stack([cx + FACE * 0.46 * np.cos(a), cy + FACE * 0.55 * np.sin(a)], 1).astype(np.float32)


def _paste(plate, crop, monkeypatch, grain, frame_index=5):
    monkeypatch.setenv("ROOP_GRAIN_MATCH", grain)
    m, (cx, cy) = _matrix()
    return _stub().paste_upscale(crop, crop, m, plate.copy(), 1, [0, 0, 0, 0, 20.0, 10.0], face_landmarks=_hull(cx, cy),
                                 inplace=True, track_id=1, frame_index=frame_index)


def _core():
    c = np.zeros((H, W), np.uint8)
    cv2.circle(c, (W // 2, H // 2), 55, 1, -1)
    return c.astype(bool)


class TestPasteUpscaleWiring:
    def test_grain_brings_a_clean_swap_up_to_a_grainy_plate_and_off_is_the_control(self, monkeypatch):
        plate = _plate(H, W, 5.0, 61)
        crop = _plate(CROP, CROP, 0.0, 62)
        on = _paste(plate, crop, monkeypatch, "1")
        off = _paste(plate, crop, monkeypatch, "0")
        target = gm.noise_sigma(plate, _core())
        assert target > 4.0
        assert gm.noise_sigma(off, _core()) < 1.0                            # the control: the swap really is clean
        assert gm.noise_sigma(on, _core()) == pytest.approx(target, rel=0.10)  # ...and the feature closed the gap
        assert any("got grain" in line for line in gm.summary_lines())

    def test_a_plate_cleaner_than_the_swap_is_bit_identical_to_off(self, monkeypatch):
        plate = _plate(H, W, 0.0, 71)
        crop = _plate(CROP, CROP, 4.0, 72)
        on = _paste(plate, crop, monkeypatch, "1")
        off = _paste(plate, crop, monkeypatch, "0")
        assert np.array_equal(on, off)

    def test_pixels_outside_the_face_are_the_plate(self, monkeypatch):
        plate = _plate(H, W, 5.0, 81)
        on = _paste(plate, _plate(CROP, CROP, 0.0, 82), monkeypatch, "1")
        outside = np.ones((H, W), bool)
        outside[H // 2 - 200:H // 2 + 200, W // 2 - 200:W // 2 + 200] = False
        assert np.array_equal(on[outside], plate[outside])

    def test_the_render_is_repeatable(self, monkeypatch):
        plate, crop = _plate(H, W, 5.0, 91), _plate(CROP, CROP, 0.0, 92)
        a = _paste(plate, crop, monkeypatch, "1")
        b = _paste(plate, crop, monkeypatch, "1")
        c = _paste(plate, crop, monkeypatch, "1", frame_index=6)
        assert np.array_equal(a, b) and not np.array_equal(a, c)


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())
