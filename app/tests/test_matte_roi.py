"""The face-neighbourhood matte in ``paste_upscale`` must give the SAME frame as the full-frame matte, to within the
one thing it cannot reproduce.

The canvas is only allowed to exist because every output pixel has its full support inside it, and that part is
exact: nothing is ever non-zero outside the canvas, and everywhere the matte is interior it is identical. What is NOT
reproducible is ``cv2.warpAffine``'s fixed-point rounding of source coordinates, which shifts when the destination
origin is translated: a handful of matte-edge pixels land on the other side of a 1/32-px interpolation boundary and
differ by ONE level. So the bound asserted here is max 1 level and at most ``MAX_DIFF_PIXELS`` pixels per frame (observed
2), against a render-to-render pixel noise floor of 0.71/255 mean, 22/255 max.

The scenarios are adversarial on purpose - faces touching and leaving every frame edge, rotated and scaled crops, blend
strengths from 0 to 100, with and without a landmark hull - and the fast path must actually be TAKEN in most of them (a
spy on ``_matte_roi``), or agreement would be the full-frame path agreeing with itself.
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

try:
    from roop.procmgr_masking import MaskingMixin
except Exception as _exc:                                            # pragma: no cover - light profile
    pytest.skip(f"production paste needs the full stack: {type(_exc).__name__}", allow_module_level=True)

CROP = 512
MAX_DIFF_PIXELS = 64


class Stub(MaskingMixin):
    def __init__(self):
        self.options = types.SimpleNamespace(show_face_area_overlay=False, blend_ratio=1.0)
        self._tls = threading.local()
        self.taken = 0
        self.refused = 0

    def _matte_roi(self, *a, **k):
        out = super()._matte_roi(*a, **k)
        self.taken += out is not None
        self.refused += out is None
        return out


def scenario(rng, w, h, cx, cy, face_px, roll_deg, blend, landmarks=True, skew=0.0):
    s = CROP * 0.8 / face_px
    th = np.deg2rad(roll_deg)
    c, sn = np.cos(th) * s, np.sin(th) * s
    lin = np.array([[c, -sn + skew], [sn, c]])
    t = np.array([CROP / 2.0, CROP / 2.0]) - lin @ np.array([cx, cy])
    M = np.hstack([lin, t[:, None]]).astype(np.float32)
    frame = (rng.random((h, w, 3)) * 90 + 80).astype(np.uint8)
    crop = (rng.random((CROP, CROP, 3)) * 70 + 90).astype(np.uint8)
    a = np.linspace(0, 2 * np.pi, 106, endpoint=False)
    lm = np.stack([cx + face_px * 0.46 * np.cos(a + th), cy + face_px * 0.55 * np.sin(a + th)], 1).astype(np.float32)
    return frame, crop, M, (lm if landmarks else None), [0, 0, 0, 0, float(blend), 10.0]


def paste(frame, crop, M, lm, offsets, env, monkeypatch):
    monkeypatch.setenv("ROOP_BLEND_ROI_MATTE", env)
    monkeypatch.setenv("ROOP_GRAIN_MATCH", "0")                      # isolate the matte: grain is tested elsewhere
    stub = Stub()
    out = stub.paste_upscale(crop, crop, M, frame.copy(), 1, offsets, face_landmarks=lm, inplace=True,
                             track_id=1, frame_index=3)
    return out, stub


SIZES = [(1280, 720), (1920, 1080), (640, 480)]


def _cases():
    rng = np.random.default_rng(2026)
    cases = []
    for i in range(36):
        w, h = SIZES[i % len(SIZES)]
        face = float(rng.uniform(90, 330))
        # centres from well inside to hanging off the edges
        cx = float(rng.choice([rng.uniform(0.3, 0.7) * w, rng.uniform(-0.05, 0.12) * w, rng.uniform(0.88, 1.05) * w]))
        cy = float(rng.choice([rng.uniform(0.3, 0.7) * h, rng.uniform(-0.08, 0.15) * h, rng.uniform(0.85, 1.06) * h]))
        cases.append(pytest.param((w, h, cx, cy, face, float(rng.uniform(-40, 40)), float(rng.choice([0, 10, 25, 25, 50, 100])),
                                   bool(rng.random() > 0.15), float(rng.uniform(-0.1, 0.1)), 1000 + i), id=f"s{i}"))
    return cases


class TestFastPathEqualsFullFrame:
    @pytest.mark.parametrize("p", _cases())
    def test_composited_frame_is_identical(self, p, monkeypatch):
        w, h, cx, cy, face, roll, blend, lm_on, skew, seed = p
        frame, crop, M, lm, offsets = scenario(np.random.default_rng(seed), w, h, cx, cy, face, roll, blend, lm_on, skew)
        fast, fs = paste(frame, crop, M, lm, offsets, "1", monkeypatch)
        full, ls = paste(frame, crop, M, lm, offsets, "0", monkeypatch)
        assert ls.taken == 0 and ls.refused == 0                       # the control never touched the new path
        diff = np.abs(fast.astype(np.int16) - full.astype(np.int16))
        assert diff.max() <= 1, f"max difference {int(diff.max())} levels"
        assert int((diff > 0).any(axis=2).sum()) <= MAX_DIFF_PIXELS

    def test_the_fast_path_is_actually_taken(self, monkeypatch):
        taken = 0
        for p in [c.values[0] for c in _cases()]:
            w, h, cx, cy, face, roll, blend, lm_on, skew, seed = p
            frame, crop, M, lm, offsets = scenario(np.random.default_rng(seed), w, h, cx, cy, face, roll, blend, lm_on, skew)
            _, stub = paste(frame, crop, M, lm, offsets, "1", monkeypatch)
            taken += stub.taken
        assert taken >= 28, f"only {taken}/36 scenarios used the canvas; the equality above proves little"

    def test_4k_identical_and_the_canvas_is_small(self, monkeypatch):
        frame, crop, M, lm, offsets = scenario(np.random.default_rng(7), 3840, 2160, 1900, 900, 260, 12, 25)
        fast, fs = paste(frame, crop, M, lm, offsets, "1", monkeypatch)
        full, _ = paste(frame, crop, M, lm, offsets, "0", monkeypatch)
        diff = np.abs(fast.astype(np.int16) - full.astype(np.int16))
        assert fs.taken == 1 and diff.max() <= 1 and int((diff > 0).any(axis=2).sum()) <= MAX_DIFF_PIXELS


class TestRefusals:
    def _pair(self, monkeypatch, **kw):
        frame, crop, M, lm, offsets = scenario(np.random.default_rng(5), 1280, 720, 640, 360, 230, 5, 25)
        monkeypatch.setenv("ROOP_BLEND_ROI_MATTE", "1")
        monkeypatch.setenv("ROOP_GRAIN_MATCH", "0")
        stub = Stub()
        stub.paste_upscale(crop, crop, M, frame.copy(), 1, offsets, face_landmarks=lm, inplace=True, **kw)
        return stub

    def test_other_faces_in_the_frame_use_the_full_frame_path(self, monkeypatch):
        import roop.procmgr_masking as pm
        frame, crop, M, lm, offsets = scenario(np.random.default_rng(5), 1280, 720, 640, 360, 230, 5, 25)
        monkeypatch.setenv("ROOP_GRAIN_MATCH", "0")
        stub = Stub()
        stub._tls.frame_faces = [object(), object()]
        try:
            stub.paste_upscale(crop, crop, M, frame.copy(), 1, offsets, face_landmarks=lm, inplace=True, target_face=object())
        except Exception:
            pass                                                    # the protect step needs a real manager; we only need the routing
        assert stub.taken == 0 and stub.refused == 0

    def test_compositor_region_and_env_switch_refuse(self, monkeypatch):
        stub = Stub()
        assert stub._matte_roi_eligible(None, None, None, None)
        assert not stub._matte_roi_eligible(object(), None, None, None)
        assert not stub._matte_roi_eligible(None, np.ones((4, 4)), None, None)
        assert not stub._matte_roi_eligible(None, None, types.SimpleNamespace(enabled=True), None)
        monkeypatch.setenv("ROOP_BLEND_ROI_MATTE", "0")
        assert not stub._matte_roi_eligible(None, None, None, None)

    def test_distance_edge_mode_refuses(self, monkeypatch):
        import roop.globals as g
        monkeypatch.setattr(g, "mask_edge_mode", "distance", raising=False)
        assert not Stub()._matte_roi_eligible(None, None, None, None)

    def test_a_matte_that_reaches_the_canvas_edge_falls_back(self, monkeypatch):
        stub = Stub()
        frame, crop, M, lm, offsets = scenario(np.random.default_rng(5), 1280, 720, 640, 360, 230, 5, 25)
        ell = np.full((CROP, CROP), 255, np.uint8)
        monkeypatch.setattr(stub, "blur_area", lambda m, b: np.full_like(m, 255))     # support floods the canvas
        assert stub._matte_roi(ell, cv2.invertAffineTransform(M), frame.shape, lm, None, offsets) is None


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())
