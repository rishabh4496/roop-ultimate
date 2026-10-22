"""ROOP_PASTE_INTERP selects the filter of the face's inverse warp, and the default is exactly the shipped bicubic."""
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

try:
    from roop import procmgr_masking as pm
except Exception as _exc:                                            # pragma: no cover - light profile
    pytest.skip(f"production paste needs the full stack: {type(_exc).__name__}", allow_module_level=True)


class Stub(pm.MaskingMixin):
    def __init__(self):
        self.options = types.SimpleNamespace(show_face_area_overlay=False, blend_ratio=1.0)
        self._tls = threading.local()


def _paste(monkeypatch, interp):
    if interp is None:
        monkeypatch.delenv("ROOP_PASTE_INTERP", raising=False)
    else:
        monkeypatch.setenv("ROOP_PASTE_INTERP", interp)
    monkeypatch.setenv("ROOP_GRAIN_MATCH", "0")
    rng = np.random.default_rng(4)
    frame = (rng.random((480, 640, 3)) * 90 + 80).astype(np.uint8)
    # a crop with real high-frequency structure, so the two kernels have something to disagree about
    yy, xx = np.mgrid[0:512, 0:512]
    crop = np.repeat((128 + 70 * np.sin(xx / 3.1) * np.cos(yy / 2.7))[:, :, None], 3, axis=2).astype(np.uint8)
    s = 512 * 0.8 / 230.0
    M = np.array([[s, 0, 256 - s * 320], [0, s, 256 - s * 240]], np.float32)
    a = np.linspace(0, 2 * np.pi, 106, endpoint=False)
    lm = np.stack([320 + 105 * np.cos(a), 240 + 125 * np.sin(a)], 1).astype(np.float32)
    seen = []
    real = cv2.warpAffine

    def spy(src, m, dsize, *a, **k):
        seen.append(k.get("flags"))
        return real(src, m, dsize, *a, **k)

    monkeypatch.setattr(pm.cv2, "warpAffine", spy)
    out = Stub().paste_upscale(crop, crop, M, frame.copy(), 1, [0, 0, 0, 0, 25.0, 10.0], face_landmarks=lm, inplace=True)
    return out, seen


class TestPasteInterp:
    def test_default_is_bicubic(self, monkeypatch):
        out, flags = _paste(monkeypatch, None)
        assert cv2.INTER_CUBIC in flags and cv2.INTER_LANCZOS4 not in flags
        same, _ = _paste(monkeypatch, "cubic")
        assert np.array_equal(out, same)

    def test_a_typo_is_bicubic_not_a_different_filter(self, monkeypatch):
        out, flags = _paste(monkeypatch, "lanczos")
        assert cv2.INTER_LANCZOS4 not in flags
        assert np.array_equal(out, _paste(monkeypatch, None)[0])

    def test_lanczos4_is_used_and_changes_the_pixels_a_little(self, monkeypatch):
        cubic, _ = _paste(monkeypatch, None)
        lanc, flags = _paste(monkeypatch, "Lanczos4")                   # case-insensitive
        assert cv2.INTER_LANCZOS4 in flags and cv2.INTER_CUBIC not in flags       # the face warp changed filter
        diff = np.abs(cubic.astype(np.int16) - lanc.astype(np.int16))
        assert diff.max() > 0 and np.isfinite(lanc).all()
        assert diff.mean() < 2.0                                          # a different kernel, not a different picture


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())
