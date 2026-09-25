"""roop.enhance_blend: Restore Ultra's low-from-swap / high-from-restorer
recombination and the inner-feature region. Pure CPU."""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import enhance_blend as eb   # noqa: E402


def _img(seed, size=512, blur=1.5):
    r = np.random.default_rng(seed)
    return cv2.GaussianBlur(r.integers(0, 255, (size, size, 3), dtype=np.uint8),
                            (0, 0), blur)


def test_flat_restorer_contributes_nothing_but_the_swap_low_band():
    swap = _img(1)
    flat = np.full_like(swap, 128)
    out = eb.frequency_blend(swap, flat, weight=0.75)
    low = cv2.GaussianBlur(swap, (0, 0), eb.SPLIT_SIGMA_512,
                           borderType=cv2.BORDER_REFLECT)
    assert np.abs(out.astype(int) - low.astype(int)).max() <= 1


def test_low_band_comes_from_the_swap_not_the_restorer():
    # A restorer that re-grades the whole face (+40 levels) must not move the
    # output's tone: that shift is all low band.
    swap = _img(2)
    regraded = cv2.add(swap, np.full_like(swap, 40))
    out = eb.frequency_blend(swap, regraded, weight=1.0)
    assert abs(float(out.mean()) - float(swap.mean())) < 1.5
    assert abs(float(regraded.mean()) - float(swap.mean())) > 30


def test_weight_scales_the_restorer_detail():
    swap, rest = _img(3, blur=4.0), _img(4, blur=0.8)
    hf = [float(cv2.Laplacian(eb.frequency_blend(swap, rest, weight=w),
                              cv2.CV_32F).std()) for w in (0.0, 0.5, 1.0)]
    assert hf[0] < hf[1] < hf[2]


def test_low_resolution_swap_is_accepted_and_output_is_restorer_sized():
    swap, rest = _img(5, size=256), _img(6)
    out = eb.frequency_blend(swap, rest)
    assert out.shape == rest.shape and out.dtype == np.uint8


def test_region_zero_is_the_swap_and_region_one_is_the_blend():
    swap, rest = _img(7), _img(8)
    zero = np.zeros(swap.shape[:2], np.float32)
    one = np.ones(swap.shape[:2], np.float32)
    assert np.abs(eb.frequency_blend(swap, rest, region=zero).astype(int)
                  - swap.astype(int)).max() <= 1
    assert np.abs(eb.frequency_blend(swap, rest, region=one).astype(int)
                  - eb.frequency_blend(swap, rest).astype(int)).max() <= 1


def test_inner_feature_weight_covers_features_and_fades():
    lm = np.zeros((106, 2), np.float32)
    rng = np.random.default_rng(0)
    lm[:] = rng.uniform(150, 350, (106, 2))
    lm[33:43] = rng.uniform([180, 200], [220, 220], (10, 2))  # an eye
    m = eb.inner_feature_weight(lm, (512, 512))
    assert m.shape == (512, 512) and m.min() >= 0.0 and m.max() <= 1.0
    assert m[210, 200] > 0.9           # inside the eye
    assert m[5, 5] < 0.01              # far corner
    assert eb.inner_feature_weight(None, (512, 512)) is None
    bad = lm.copy()
    bad[0, 0] = np.nan
    assert eb.inner_feature_weight(bad, (512, 512)) is None


def load_tests(loader, tests, pattern):
    """Expose this module's bare `test_*` functions to `unittest discover`
    (see tests/unittest_shim.py). pytest never calls load_tests."""
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:  # discovery started from inside tests/
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
