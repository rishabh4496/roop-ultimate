"""Face-swap visual defects: hard obstruction edges, occlusion-contaminated colour statistics,
restorer tone steps at the paste rim, and watchdog treatment of an obstruction that comes and goes.

Each test names the defect it pins and the pre-change behaviour it would have failed on.
"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import mask_soft, occlusion_color, restorer_seam  # noqa: E402


# --------------------------------------------------------------------------- #
# mask_soft: a cliff at the threshold turns model noise into popping
# --------------------------------------------------------------------------- #
def test_soft_threshold_keeps_the_decision_boundary():
    p = np.linspace(0.0, 1.0, 1001, dtype=np.float32)
    for c in (0.35, 0.40):
        s = mask_soft.soft_threshold(p, c)
        off_boundary = np.abs(p - c) > 1e-4
        assert np.array_equal((s > 0.5)[off_boundary], (p > c)[off_boundary]), \
            'soft > 0.5 must reproduce the engine\'s old hard decision'
        assert abs(float(mask_soft.soft_threshold(np.float32(c), c)) - 0.5) < 1e-6
        # the rails, to float32 precision (the ramp ends at c -/+ 0.1)
        assert float(s[p <= c - 0.1].max()) <= 1e-6 and float(s[p >= c + 0.1].min()) >= 1.0 - 1e-6


def test_soft_threshold_removes_threshold_chatter():
    """A pixel whose probability wanders +-0.02 around the threshold flips 0/1 under the hard
    step (pre-change `(p > 0.35).astype(float32)`); the ramp turns that into a small wobble."""
    rng = np.random.default_rng(7)
    p = (0.35 + rng.normal(0.0, 0.02, size=4000)).astype(np.float32)
    hard = (p > 0.35).astype(np.float32)
    soft = mask_soft.soft_threshold(p, 0.35)
    assert hard.std() > 0.45                       # it really was flipping
    assert soft.std() < 0.5 * hard.std()


def test_soft_threshold_is_finite_on_garbage():
    out = mask_soft.soft_threshold(np.array([np.nan, np.inf, -np.inf, 0.35], np.float32), 0.35)
    assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0


def test_thin_preserving_erode_keeps_a_glasses_rim_that_cv2_erode_deletes():
    m = np.zeros((128, 128), np.float32)
    m[40:100, 20:30] = 1.0                         # thick object, 10 px wide
    m[20, 30:110] = 1.0                            # 1 px "rim"
    m[60:62, 60:120] = 1.0                         # 2 px "strand"
    el = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    plain = cv2.erode(m, el)
    assert plain[20, 30:110].sum() == 0 and plain[60:62, 60:120].sum() == 0      # premise: the old erosion kills them
    kept = mask_soft.thin_preserving_erode(m, 5)
    assert kept[20, 30:110].sum() == pytest.approx(80.0)
    assert kept[60:62, 60:120].sum() == pytest.approx(120.0)
    # the thick object still shrinks, as the erosion intended
    assert kept[40:100, 20:30].sum() < m[40:100, 20:30].sum()
    assert np.allclose(kept[45:95, 24:26], 1.0)    # ...to its core, not to nothing


def test_thin_preserving_erode_is_identity_at_k1():
    m = np.random.default_rng(1).random((32, 32)).astype(np.float32)
    assert np.array_equal(mask_soft.thin_preserving_erode(m, 1), m)


# --------------------------------------------------------------------------- #
# occlusion_color: hand/hair pixels must not be colour statistics
# --------------------------------------------------------------------------- #
def _skin_crop(size=256, seed=3, noise=3.0):
    rng = np.random.default_rng(seed)
    img = np.empty((size, size, 3), np.float32)
    img[:] = (118.0, 150.0, 196.0)                 # BGR skin
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _disk(size, r_frac=0.40):
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2.0
    return (((xx - c) ** 2 + (yy - c) ** 2) <= (r_frac * size) ** 2)


def test_clean_mask_excludes_the_obstruction_and_is_eroded():
    size = 256
    skin = _disk(size).astype(np.float32)
    keep = np.ones((size, size), np.float32)
    keep[90:170, 130:200] = 0.0                    # a hand over the cheek: the occluder says restore the plate
    m = occlusion_color.clean_stats_mask(skin, keep)
    assert m is not None
    assert not m[90:170, 130:200].any()
    assert not m[~_disk(size)].any()
    assert m.sum() < ((skin > 0.5) & (keep > 0.5)).sum()          # eroded


def test_clean_mask_none_when_nothing_survives():
    assert occlusion_color.clean_stats_mask(np.zeros((64, 64)), np.ones((64, 64))) is None
    assert occlusion_color.clean_stats_mask(np.ones((64, 64)), np.zeros((64, 64))) is None


def test_stats_diverge_fires_for_a_hand_and_stays_quiet_without_one():
    size = 256
    target = _skin_crop(size)
    hand = np.zeros((size, size), bool)
    hand[100:180, 120:200] = True
    target[hand] = (60, 105, 225)                  # a more saturated, orange-red patch inside the face
    skin = _disk(size).astype(np.float32)
    first_pass = (skin > 0.5)                      # what pass 1 samples: colour + hull, no occlusion knowledge

    keep_hand = np.ones((size, size), np.float32)
    keep_hand[hand] = 0.0
    clean = occlusion_color.clean_stats_mask(skin, keep_hand)
    assert occlusion_color.stats_diverge(target, first_pass, clean)

    clear = _skin_crop(size)                       # same face, nothing in front of it
    clean_clear = occlusion_color.clean_stats_mask(skin, np.ones((size, size), np.float32))
    assert not occlusion_color.stats_diverge(clear, first_pass, clean_clear), \
        'an un-occluded face must not trigger a second colour pass (keeps those frames bit-identical)'


def test_contaminated_statistics_tint_the_swap_and_clean_ones_do_not():
    """The defect itself, on the production transfer: a Reinhard transfer whose sampling mask
    includes the hand maps the swap onto cheek+hand statistics, so the face's own tone is wrong."""
    from roop.procmgr_color import ColorTransferMixin
    size = 256
    target = _skin_crop(size)
    hand = np.zeros((size, size), bool)
    hand[90:190, 110:215] = True
    target[hand] = (40, 95, 235)
    swap = _skin_crop(size, seed=11)
    swap[:] = np.clip(swap.astype(np.float32) * (0.82, 0.92, 1.0) + (14, 0, -18), 0, 255).astype(np.uint8)
    skin = _disk(size)
    keep = np.ones((size, size), np.float32)
    keep[hand] = 0.0
    clean = occlusion_color.clean_stats_mask(skin.astype(np.float32), keep)

    mixin = ColorTransferMixin()
    bad = mixin._color_transfer_rct(swap, target, mask=skin.astype(np.uint8))
    good = mixin._color_transfer_rct(swap, target, mask=clean.astype(np.uint8))

    truth = cv2.cvtColor(_skin_crop(size), cv2.COLOR_BGR2LAB).astype(np.float32)[skin & ~hand].mean(0)
    on_face = skin & ~hand

    def err(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)[on_face].mean(0)
        return float(np.abs(lab - truth).max())

    assert err(bad) > 3.0, 'premise: hand pixels in the sampling mask must visibly move the face tone'
    assert err(good) < 1.5, 'occlusion-aware sampling should land on the real cheek tone'


# --------------------------------------------------------------------------- #
# restorer_seam: restorer tone step at the paste rim
# --------------------------------------------------------------------------- #
def _matte(size=256, ramp=14):
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2.0
    d = 0.42 * size - np.sqrt((xx - c) ** 2 + (yy - c) ** 2)       # + inside
    return np.clip(d / ramp, 0, 1).astype(np.float32)


def _textured(size, tone, seed):
    rng = np.random.default_rng(seed)
    base = np.empty((size, size, 3), np.float32)
    base[:] = tone
    base += cv2.GaussianBlur(rng.normal(0, 10, (size, size, 3)).astype(np.float32), (0, 0), 1.2)
    return base


def test_rim_tone_lock_hands_the_rim_back_to_the_swap_but_keeps_the_interior_exact():
    size = 256
    swap = _textured(size, (120, 150, 196), 1)
    restored = swap + 18.0                         # the restorer regraded the whole patch 18 levels brighter
    a = _matte(size)
    out = restorer_seam.rim_tone_lock(restored, swap, a)
    w = restorer_seam.rim_weight(a)

    rim = (a > 0.45) & (a < 0.55)
    step_before = float(np.abs(restored - swap)[rim].mean())
    step_after = float(np.abs(out - swap)[rim].mean())
    assert step_before > 17.0 and step_after < 3.0, (step_before, step_after)

    interior = w >= 1.0
    assert interior.sum() > 5000
    assert np.array_equal(out[interior], restored[interior]), 'past the ramp the restorer output must be untouched'


def test_rim_tone_lock_keeps_the_restorers_high_band_at_the_rim():
    size = 256
    swap = _textured(size, (120, 150, 196), 1)
    rng = np.random.default_rng(5)
    detail = cv2.GaussianBlur(rng.normal(0, 14, (size, size, 3)).astype(np.float32), (0, 0), 0.8)
    restored = swap + 18.0 + detail                # brighter AND sharper
    a = _matte(size)
    out = restorer_seam.rim_tone_lock(restored, swap, a)
    rim = ((a > 0.3) & (a < 0.7))[..., None]

    def hf(x):
        return x - cv2.GaussianBlur(x, (0, 0), 6.0)

    e_out = float(np.square(hf(out) * rim).sum())
    e_res = float(np.square(hf(restored) * rim).sum())
    assert e_out > 0.9 * e_res, 'tone lock must not smooth the detail the restorer was run to add'


def test_rim_tone_lock_is_identity_when_nothing_to_fix():
    size = 128
    swap = _textured(size, (120, 150, 196), 2)
    a = _matte(size, 8)
    assert np.allclose(restorer_seam.rim_tone_lock(swap, swap, a), swap, atol=1e-3)
    assert np.array_equal(restorer_seam.rim_tone_lock(swap + 9.0, swap, a, rim_frac=0.0), swap + 9.0)


def test_rim_weight_empty_matte_is_zero():
    assert float(restorer_seam.rim_weight(np.zeros((32, 32), np.float32)).max()) == 0.0
