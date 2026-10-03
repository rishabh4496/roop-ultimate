"""roop.mask_roi must equal the full-frame cv2 call BIT FOR BIT.

The matte is non-zero only around a face, so the blur / erode / dilate run on its
padded bounding box. That is only a legitimate optimisation if it is exact: these tests
compare with `np.array_equal`, no tolerance, over random and adversarial masks (blobs
touching every frame edge, thin lines, several islands, one pixel, huge kernels).
"""
import os
import sys

import cv2
import numpy as np
import pytest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import mask_roi  # noqa: E402


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())


def _blob(shape, rng, kind):
    h, w = shape
    m = np.zeros(shape, np.uint8)
    if kind == "ellipse":
        cv2.ellipse(m, (int(rng.integers(w // 4, 3 * w // 4)), int(rng.integers(h // 4, 3 * h // 4))),
                    (int(rng.integers(8, w // 6)), int(rng.integers(8, h // 6))), 0, 0, 360, 255, -1)
    elif kind == "soft":
        cv2.ellipse(m, (w // 2, h // 2), (w // 8, h // 5), 0, 0, 360, 255, -1)
        m = cv2.GaussianBlur(m, (21, 21), 0)           # graded values, like the real matte
    elif kind == "edge_left_top":
        m[0:40, 0:60] = 255
    elif kind == "edge_right_bottom":
        m[h - 30:, w - 50:] = 200
    elif kind == "full_width_band":
        m[h // 2 - 3:h // 2 + 3, :] = 255
    elif kind == "islands":
        for _ in range(4):
            y, x = int(rng.integers(0, h - 20)), int(rng.integers(0, w - 20))
            m[y:y + 15, x:x + 15] = int(rng.integers(1, 256))
    elif kind == "single_pixel":
        m[int(rng.integers(0, h)), int(rng.integers(0, w))] = 255
    elif kind == "noise":
        m = (rng.random(shape) > 0.995).astype(np.uint8) * 255
    return m


KINDS = ["ellipse", "soft", "edge_left_top", "edge_right_bottom", "full_width_band",
         "islands", "single_pixel", "noise"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("k", [3, 7, 27, 51, 111])
def test_gaussian_is_bit_identical(kind, k):
    rng = np.random.default_rng(abs(hash((kind, k))) % 2 ** 32)
    m = _blob((360, 480), rng, kind)
    assert np.array_equal(mask_roi.gaussian_blur(m, (k, k)), cv2.GaussianBlur(m, (k, k), 0))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", [cv2.MORPH_ELLIPSE, cv2.MORPH_RECT])
@pytest.mark.parametrize("k", [3, 13, 27, 43])
def test_erode_and_dilate_are_bit_identical(kind, shape, k):
    rng = np.random.default_rng(abs(hash((kind, k, shape))) % 2 ** 32)
    m = _blob((360, 480), rng, kind)
    ker = cv2.getStructuringElement(shape, (k, k))
    assert np.array_equal(mask_roi.erode(m, ker), cv2.erode(m, ker, iterations=1))
    assert np.array_equal(mask_roi.dilate(m, ker), cv2.dilate(m, ker, iterations=1))


def test_the_production_feather_chain_is_bit_identical():
    """blur_area's exact sequence on a face-shaped matte at 1080p."""
    m = np.zeros((1080, 1920), np.uint8)
    cv2.ellipse(m, (900, 420), (190, 260), 0, 0, 360, 255, -1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
    ref = cv2.GaussianBlur(cv2.erode(cv2.GaussianBlur(m, (3, 3), 0), ker), (111, 111), 0)
    got = mask_roi.gaussian_blur(mask_roi.erode(mask_roi.gaussian_blur(m, (3, 3)), ker), (111, 111))
    assert np.array_equal(got, ref)


def test_empty_mask_returns_zeros_like_cv2():
    m = np.zeros((100, 120), np.uint8)
    out = mask_roi.gaussian_blur(m, (9, 9))
    assert out.shape == m.shape and out.dtype == np.uint8 and not out.any()


def test_large_support_falls_back_and_still_matches():
    m = np.full((120, 160), 255, np.uint8)
    assert np.array_equal(mask_roi.gaussian_blur(m, (15, 15)), cv2.GaussianBlur(m, (15, 15), 0))


def test_non_uint8_or_non_2d_uses_cv2_directly():
    f = np.zeros((90, 120), np.float32)
    f[40:50, 50:70] = 1.0
    assert np.array_equal(mask_roi.gaussian_blur(f, (9, 9)), cv2.GaussianBlur(f, (9, 9), 0))
    c = np.zeros((90, 120, 3), np.uint8)
    c[40:50, 50:70] = 255
    assert np.array_equal(mask_roi.gaussian_blur(c, (9, 9)), cv2.GaussianBlur(c, (9, 9), 0))


def test_it_does_not_mutate_the_input():
    m = np.zeros((200, 200), np.uint8)
    m[80:120, 80:120] = 255
    before = m.copy()
    mask_roi.gaussian_blur(m, (21, 21))
    mask_roi.erode(m, np.ones((5, 5), np.uint8))
    assert np.array_equal(m, before)


def test_env_switch_disables_it(monkeypatch):
    monkeypatch.setenv("ROOP_MASK_ROI", "0")
    assert not mask_roi.enabled()
    m = np.zeros((200, 200), np.uint8)
    m[90:110, 90:110] = 255
    assert np.array_equal(mask_roi.gaussian_blur(m, (9, 9)), cv2.GaussianBlur(m, (9, 9), 0))


def test_it_is_actually_faster_on_a_face_sized_matte():
    """The point of the module. Loose bound (measured ~3x at this size): a regression to a
    full-frame pass reads 1.0x."""
    import time
    m = np.zeros((1080, 1920), np.uint8)
    cv2.ellipse(m, (900, 420), (105, 145), 0, 0, 360, 255, -1)    # a ~250 px face

    def best(fn):
        times = []
        for _ in range(5):
            t = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t)
        return min(times)
    # The app runs OpenCV single-threaded per worker (ProcessMgr: cv2.setNumThreads(1));
    # under cv2's default pool a full-frame blur is 8x cheaper and hides what workers pay.
    prior = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        assert best(lambda: mask_roi.gaussian_blur(m, (61, 61))) < 0.75 * best(
            lambda: cv2.GaussianBlur(m, (61, 61), 0))
    finally:
        cv2.setNumThreads(prior)
