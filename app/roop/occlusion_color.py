"""Occlusion-aware sampling for the swap-stage colour transfer.

The first colour pass in ``ProcessMgr.process_face`` runs inside the swap stage, before any mask
engine has looked at the crop. Its sampling mask is "skin-coloured pixels inside the landmark hull"
(``ColorTransferMixin._skin_region_mask``), which is a colour test: a hand, a sleeve, a patch of
hair that happens to fall inside the YCrCb skin gate and the hull is sampled like cheek. The
transfer then maps the swap onto the mean/std of "cheek + hand", and the face drifts orange (or
green, under a coloured light) for exactly as long as the object is in front of it.

The mask stage knows better: ``_unified_keep`` is 1 only where the swap survives every engine
(face region found, nothing in front). This module turns that into a sampling mask and decides
whether re-running the transfer with it is worth doing.

Contract
--------
``stats_diverge`` is the gate. It compares the TARGET crop's own Lab mean/std under the mask the
first pass used and under the occlusion-aware one. Within tolerance the first pass was already
sampling the same skin and nothing is redone, so un-occluded frames stay bit-identical to the
pre-change pipeline.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

MEAN_TOL = 1.5          # Lab units (OpenCV 8-bit Lab, the space the Reinhard transfer works in)
STD_REL_TOL = 0.06      # relative change in a channel's std
KEEP_MIN = 0.9          # a pixel counts as "swap kept" only when the unified keep matte is this high
SKIN_MIN = 0.5          # ...and this much skin evidence
ERODE_FRAC = 0.015      # erosion radius as a fraction of the crop's short side
MIN_PIXELS = 256


MIN_EXCLUDED_FRAC = 0.01    # below this share of the first pass's sample being excluded, nothing can have changed


def excludes_sampled_pixels(sampled_mask, keep, *, keep_min: float = KEEP_MIN,
                            min_frac: float = MIN_EXCLUDED_FRAC) -> bool:
    """True when the keep matte removes at least ``min_frac`` of the pixels the first pass sampled.

    The cheap gate in front of the Lab assessment: a frame whose keep matte covers the whole sampled skin has
    no occluder or hair in it, so the first pass stands and the (comparatively expensive) statistics are never
    computed. Shapes may differ (the matte is resized to the sample).
    """
    m = np.asarray(sampled_mask, bool)
    n = int(m.sum())
    if n == 0:
        return False
    k = np.asarray(keep, np.float32)
    if k.ndim == 3:
        k = k[..., 0]
    if k.shape != m.shape:
        k = cv2.resize(k, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_LINEAR)
    return int((k[m] < keep_min).sum()) >= min_frac * n


def clean_stats_mask(skin_support, keep, *, keep_min: float = KEEP_MIN, skin_min: float = SKIN_MIN,
                     erode_frac: float = ERODE_FRAC, min_pixels: int = MIN_PIXELS) -> Optional[np.ndarray]:
    """Boolean mask of crop pixels that are skin AND not obstructed AND inside the kept face region.

    ``skin_support`` is the first pass's soft skin/hull map, ``keep`` the unified keep matte
    (1 = swap kept). Eroded so the boundary ring, where the matte and the object edge are least
    certain, does not contribute. Returns None when too little survives (caller keeps pass 1).
    """
    s = np.asarray(skin_support, np.float32)
    k = np.asarray(keep, np.float32)
    if s.ndim == 3:
        s = s[..., 0]
    if k.ndim == 3:
        k = k[..., 0]
    if s.shape != k.shape:
        k = cv2.resize(k, (s.shape[1], s.shape[0]), interpolation=cv2.INTER_LINEAR)
    m = ((s >= skin_min) & (k >= keep_min)).astype(np.uint8)
    r = max(1, int(round(erode_frac * min(m.shape[:2]))))
    for rr in (r, max(1, r // 2)):
        el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))
        e = cv2.erode(m, el, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        if int(e.sum()) >= min_pixels:
            return e
    return None


WORK_PX = 128           # statistics are low-frequency: measure them on at most this many px (short side)


def _scale(img_shape) -> float:
    return min(1.0, float(WORK_PX) / max(1, min(img_shape[0], img_shape[1])))


def _small(img, f: float, nearest: bool = False):
    if f >= 1.0:
        return img
    h, w = img.shape[:2]
    size = (max(8, int(round(w * f))), max(8, int(round(h * f))))
    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


def lab_stats(img_bgr, mask) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(mean, std) of the crop's 8-bit Lab channels over a boolean mask, or None if it is nearly empty.

    Measured on a <= ``WORK_PX`` copy: the transfer's statistics are low-frequency, and converting a full
    512 px crop to Lab for every face on every mask pass cost 9 ms of CPU, ~4x the rest of this module.
    """
    img = np.asarray(img_bgr)
    m = np.asarray(mask, bool)
    f = _scale(img.shape)
    ms = _small(m.astype(np.uint8), f, nearest=True).astype(bool)
    if int(ms.sum()) < 8:
        return None
    lab = cv2.cvtColor(_small(img, f), cv2.COLOR_BGR2LAB)
    px = lab[ms].astype(np.float32)
    return px.mean(0), px.std(0)


# Reliability guards, from running the redo on real footage (90 frames, 116 redone faces): without them one
# face had its crop brightened by 69 L levels from a 272-px sample (10% of the first pass's), and two had
# their contrast crushed by a near-flat patch (L std 0.8-1.0) because Reinhard scales by target_std/source_std.
MIN_CROP_FRAC = 0.01        # the clean sample must cover at least this share of the crop...
MIN_KEEP_RATIO = 0.35       # ...and at least this share of what the first pass sampled
MIN_L_STD = 3.0             # 8-bit Lab L std of the target over the sample: below this it is a flat patch
MAX_SHIFT_L = 18.0          # a hand/hair explains a modest tone error; more than this the parse is suspect
MAX_SHIFT_AB = 12.0


def sample_is_reliable(target_bgr, mask_before, mask_after) -> bool:
    """True when the occlusion-aware sample is large and textured enough to fit statistics to."""
    if mask_before is None or mask_after is None:
        return False
    h, w = np.asarray(mask_after).shape[:2]
    n_after = int(np.asarray(mask_after, bool).sum())
    n_before = int(np.asarray(mask_before, bool).sum())
    if n_after < max(MIN_PIXELS, int(MIN_CROP_FRAC * h * w)) or n_after < MIN_KEEP_RATIO * n_before:
        return False
    st = lab_stats(target_bgr, mask_after)
    return st is not None and float(st[1][0]) >= MIN_L_STD


def correction_is_plausible(first_pass_bgr, redone_bgr, mask) -> bool:
    """True when the redo moved the face by an amount occlusion could explain (mean |Lab delta| over ``mask``)."""
    a_img, b_img = np.asarray(first_pass_bgr), np.asarray(redone_bgr)
    f = _scale(a_img.shape)
    m = _small(np.asarray(mask, bool).astype(np.uint8), f, nearest=True).astype(bool)
    if int(m.sum()) < 8:
        return False
    a = cv2.cvtColor(_small(a_img, f), cv2.COLOR_BGR2LAB).astype(np.float32)[m]
    b = cv2.cvtColor(_small(b_img, f), cv2.COLOR_BGR2LAB).astype(np.float32)[m]
    d = np.abs(a - b).mean(0)
    return bool(d[0] <= MAX_SHIFT_L and d[1] <= MAX_SHIFT_AB and d[2] <= MAX_SHIFT_AB)


def stats_diverge(target_bgr, mask_before, mask_after, *, mean_tol: float = MEAN_TOL,
                  std_rel_tol: float = STD_REL_TOL) -> bool:
    """True when the target crop's Lab statistics differ between the two sampling masks.

    Measured on the TARGET crop, the reference the transfer maps onto, so it needs no swap output
    and costs two Lab means. Either mask unusable -> False (nothing to compare, keep pass 1).
    """
    if mask_before is None or mask_after is None:
        return False
    a = lab_stats(target_bgr, mask_before)
    b = lab_stats(target_bgr, mask_after)
    if a is None or b is None:
        return False
    if float(np.max(np.abs(a[0] - b[0]))) > mean_tol:
        return True
    rel = np.abs(a[1] - b[1]) / np.maximum(a[1], 1.0)
    return bool(float(rel.max()) > std_rel_tol)
