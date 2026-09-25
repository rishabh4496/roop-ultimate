"""How a restorer's output is recombined with the swap it restored.

The shipped combine is linear: `blend_ratio * restored + (1 - blend_ratio) *
swap` at paste time (procmgr_masking.paste_upscale). That hands the restorer's
LOW band -- its tone, shading and the face's large-scale shape -- to the
output in exactly the same proportion as its fine detail. A GAN/codebook prior
(RestoreFormer++, GPEN, CodeFormer) regrades and reshapes at those low
frequencies: that is where the smoothed "waxy" look and most identity drift
live, while the part worth keeping (pores, lash and lip edges) is high band.

`frequency_blend` splits them:

    L_swap    = Blur(I_swap)
    H_restore = I_restore - Blur(I_restore)
    I_final   = clamp(L_swap + w * H_restore)

`inner_feature_weight` optionally restricts the restorer to the sensory zones
(eyes, brows, nose, lips), feathered into the swap, for users who want the
swap's own skin everywhere else.

All CPU, all on one 512 crop: two Gaussian blurs and a few adds. A per-face GPU
round trip for ops of this size was measured 1.1-40x slower than OpenCV and
reverted (bf96c1f); the mask/composite chain around it is NumPy anyway.
"""
import cv2
import numpy as np

# Gaussian sigma of the low/high split, in pixels of a 512 crop; scaled with
# the crop. ~pores and lash edges above the cut, shading and shape below it.
#
# tests/restore_ultra_bench.py, 2026-09-25, RTX 4070 live config
# (hyperswap / XSeg / TRT), 72 real Restore Ultra calls over 8 clips, source
# akansha. id_src = recognizer cosine to the source; skin_hf = cheek band-pass
# texture / the real target footage's (1.0 = as textured as the plate):
#
#   arm                        id_src   skin_hf   id_src > shipped (paired)
#   swap, no restorer          0.6238   0.873
#   Restore Ultra (shipped)    0.5729   1.584     -
#   linear blend 0.75          0.5898   1.296     68/72  +0.0168
#   freq w0.75 sigma 3         0.6058   1.149     68/72  +0.0328
#   freq w0.75 sigma 2         0.6123   1.086     67/72  +0.0394  (uint8 path)
#   freq w0.75 sigma 5         0.5941   1.192     61/72  +0.0212
#
# The restorer was NOT waxy here -- it was 1.58x MORE textured than the real
# footage (its own finish adds 0.09 of that) -- but it cost 0.051 of identity.
# Sigma 2 keeps most of that identity AND lands closest to the plate's texture.
SPLIT_SIGMA_512 = 2.0

# insightface 2d106 groups (same indices as temporal_expression.LEFT_EYE/...)
_EYES = (tuple(range(33, 43)), tuple(range(87, 97)))
_BROWS = (tuple(range(43, 52)), tuple(range(97, 106)))
_MOUTH = (tuple(range(52, 72)),)
_NOSE = (tuple(range(72, 87)),)
INNER_GROUPS = _EYES + _BROWS + _MOUTH + _NOSE


def _blur(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma,
                            borderType=cv2.BORDER_REFLECT)


def frequency_blend(swap, restored, weight=0.75, sigma=None, region=None):
    """Low band from `swap`, `weight` x high band from `restored`.

    Both uint8 BGR; `swap` is resized to `restored` if needed. `region`
    (optional float HxW in [0,1]) is where the restorer applies at all:
    outside it the output is the swap, unchanged. Returns uint8 BGR at the
    restored crop's size.
    """
    if restored is None or swap is None:
        return restored
    h, w = restored.shape[:2]
    if sigma is None:
        sigma = SPLIT_SIGMA_512 * (w / 512.0)
    swap = np.ascontiguousarray(swap, dtype=np.uint8)
    restored = np.ascontiguousarray(restored, dtype=np.uint8)
    # The swap usually arrives at a lower resolution (256 against a 512
    # restorer output): blur it THERE at the scaled sigma and upsample the
    # already-smooth low band, instead of upsampling first and blurring 4x
    # the pixels. uint8 throughout (OpenCV's fixed-point SIMD path), the high
    # band split into its saturating positive and negative halves:
    # 7.2 -> 2.3 ms per 512 face on one thread, within 2 levels at p99 of the
    # float formulation on a worst-case random texture.
    k = w / float(swap.shape[1])
    low = _blur(swap, sigma / k) if k > 1.0 else swap
    if low.shape[:2] != (h, w):
        low = cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)
    if k <= 1.0:
        low = _blur(low, sigma)
    rb = _blur(restored, sigma)
    wt = float(weight)
    pos = cv2.addWeighted(restored, wt, rb, -wt, 0.0)
    neg = cv2.addWeighted(rb, wt, restored, -wt, 0.0)
    out = cv2.subtract(cv2.add(low, pos), neg)
    if region is not None:
        base = swap if swap.shape[:2] == (h, w) else cv2.resize(
            swap, (w, h), interpolation=cv2.INTER_CUBIC)
        a = np.asarray(region, np.float32)
        if a.shape[:2] != (h, w):
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        a = np.ascontiguousarray(a, dtype=np.float32)
        out = cv2.blendLinear(out, base, a, 1.0 - a)
    return out


def inner_feature_weight(landmarks_106, shape, feather_frac=0.035,
                         grow_frac=0.02):
    """Feathered [0,1] map over eyes, brows, nose and lips.

    `landmarks_106` must already be in the crop's pixel coordinates. Each
    group is filled as its convex hull, grown by `grow_frac` of the crop
    width (a lash line sits just outside the eye contour points), then
    feathered with a Gaussian of `feather_frac` x width, so the edge fades
    into the swapped skin instead of drawing a line around the hair or jaw.
    None if the landmarks are unusable.
    """
    pts = np.asarray(landmarks_106, np.float32) if landmarks_106 is not None else None
    if pts is None or pts.shape != (106, 2) or not np.all(np.isfinite(pts)):
        return None
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    for idx in INNER_GROUPS:
        hull = cv2.convexHull(np.round(pts[list(idx)]).astype(np.int32))
        cv2.fillConvexPoly(m, hull, 255)
    g = max(1, int(round(grow_frac * w)))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (2 * g + 1, 2 * g + 1)))
    # Feather at quarter resolution: a sigma of ~18 px is a smooth ramp, and
    # blurring it at full size was 5 ms of the 5.3 this function cost.
    f = max(1.0, feather_frac * w)
    q = cv2.resize(m, (max(1, w // 4), max(1, h // 4)),
                   interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    q = _blur(q, max(0.5, f / 4.0))
    return np.clip(cv2.resize(q, (w, h), interpolation=cv2.INTER_LINEAR),
                   0.0, 1.0)
