"""Dual-stream frequency splitting and Reinhard colour transfer.

Shared by `Enhance_UltraMax`'s dual-stream engine and driven directly by
`tests/test_restorers.py`. Kept out of the processor so the operators can be
graded on their own, without a GPU or a model load.

WHY A SPLIT AT ALL, in this repo's own measured terms. Two restorers are good
at different halves of the problem, and the numbers are already on record
(`Enhance_GPENRealistic`, high-frequency std carried through to the paste):

    swap input 2.67 | GPEN-256 2.82 | CodeFormer-512 4.11 | GPEN-512 5.14

GPEN-512 synthesises ~25% more high-frequency detail than CodeFormer. But
CodeFormer's discrete codebook is the better STRUCTURE prior -- it is what
draws a clean iris rim, a correct tooth boundary and symmetric lids -- and
GPEN is the model with the colour cast. Neither is strictly better, so taking
the low band from one and the high band from the other is not a stylistic
choice; it is picking each network's measured strength.

THE FILTER HAS TO BE EDGE-AWARE. A Gaussian split leaks structure across every
strong boundary, so the "high" band of a Gaussian split at a jawline carries
the jaw itself, and re-adding it at a gain > 1 is exactly the halo this project
already had to remove from UltraMax once (the sigma 1.0-2.5 band that drew a
second lower eyelid). A guided filter's band is bounded by the edges in its own
guide, so the high band carries pores and lashes and NOT the silhouette.

`cv2.ximgproc.guidedFilter` is contrib-only and is NOT present in the
opencv-python build this app ships (4.9.0, verified), so the filter is
implemented here on `cv2.boxFilter`. That is He et al. exactly, not an
approximation of it -- the O(1) box-filter formulation IS the published
algorithm.
"""

import cv2
import numpy as np

# Reinhard std ratios are clamped to this band. `procmgr_color` bounds its own
# LAB transfer to exactly the same [0.80, 1.20] and for the same reason: an
# unbounded ratio multiplies a colour cast instead of removing one, and on a
# low-variance crop (an evenly lit cheek filling the frame) the ratio runs away
# entirely.
_STD_LO, _STD_HI = 0.80, 1.20

# Guard for the variance divisions below. Small enough not to bias `a` on real
# image variance, large enough that a perfectly flat patch cannot divide by 0.
_EPS = 1e-6


def guided_filter(image, radius=8, eps=0.04, guide=None):
    """Edge-preserving smoother (He et al. 2010), O(1) box-filter form.

    `image` float32 in [0, 1], HxW or HxWxC. `guide` defaults to `image`
    (self-guided), which is what the frequency split uses: each stream is
    smoothed by its own edges.

    `eps` is in the SQUARED units of the guide, so it only means "0.04" for a
    [0, 1] input, and getting that wrong fails SILENTLY IN BOTH DIRECTIONS.
    Feed it [0, 255] data and the variance term dominates everywhere, `a` -> 1,
    and the filter becomes the IDENTITY: `frequency_split` then computes
    `d - d == 0` and injects no detail at all while looking like it ran
    (measured: max |guided - box| = 120 levels on a step edge, i.e. nothing was
    smoothed). Push eps far the other way and `a` -> 0 and it degenerates to a
    plain box blur, whose high band carries the silhouette and rings. Callers
    here always normalise to [0, 1] first.
    """
    I = np.ascontiguousarray(image, dtype=np.float32)
    G = I if guide is None else np.ascontiguousarray(guide, dtype=np.float32)
    if G.shape[:2] != I.shape[:2]:
        raise ValueError("guide and image must share spatial dimensions")

    # ksize must be odd; radius is the half-width, matching the reference.
    k = (int(radius) * 2 + 1, int(radius) * 2 + 1)

    def _box(x):
        return cv2.boxFilter(x, -1, k, normalize=True,
                             borderType=cv2.BORDER_REFLECT)

    mean_G = _box(G)
    corr_G = _box(G * G)
    var_G = np.maximum(corr_G - mean_G * mean_G, 0.0)

    if guide is None:
        mean_I = mean_G
        cov = var_G
    else:
        mean_I = _box(I)
        cov = _box(G * I) - mean_G * mean_I

    a = cov / (var_G + float(eps) + _EPS)
    b = mean_I - a * mean_G
    return _box(a) * G + _box(b)


def frequency_split(structural, detail, radius=8, eps=0.04, gain=1.25,
                    clamp=None):
    """Low band from `structural`, high band from `detail`.

        I_low  = GuidedFilter(I_structural)
        I_high = I_detail - GuidedFilter(I_detail)
        out    = I_low + gain * I_high

    Both inputs are uint8 BGR of identical shape; the result is uint8 BGR.

    `clamp` bounds the injected high band in 0-255 levels BEFORE the gain.
    This is not cosmetic. The two streams are separate networks run on the same
    crop, so a feature can land a pixel apart between them; without a bound, a
    disagreement at a lash line or an iris rim is re-added at `gain` and reads
    as a double edge. Same reasoning -- and the same order of magnitude -- as
    `MergerMixin.apply_clarity`'s +-18 clamp, which exists to stop exactly this
    on the L channel. None disables it.
    """
    s = np.ascontiguousarray(structural, dtype=np.float32) / 255.0
    d = np.ascontiguousarray(detail, dtype=np.float32) / 255.0
    if s.shape != d.shape:
        raise ValueError(f"stream shape mismatch: {s.shape} vs {d.shape}")

    low = guided_filter(s, radius, eps)
    high = d - guided_filter(d, radius, eps)

    if clamp is not None:
        lim = float(clamp) / 255.0
        np.clip(high, -lim, lim, out=high)

    out = low + float(gain) * high
    return cv2.convertScaleAbs(out, alpha=255.0, beta=0.0)


def frequency_split_luma(structural, detail, radius=8, eps=0.04, gain=1.25,
                         clamp=None):
    """The split, but on LUMINANCE ONLY -- structure and texture are luma.

    MEASURED, and this is why it exists. The 3-channel `frequency_split`
    above, graded on 40 aligned crops of s1.mp4 against `Enhance_UltraMax`'s
    single-stream path:

        arm                       ms/face   chroma drift
        UltraMax single              40.0           0.45
        dual, 3-channel + Reinhard  151.9           1.79

    Two separate defects, both fixed here:

    1. COLOUR. Splitting in BGR injects the detail network's CHROMA high band
       as well as its luma, which is precisely GPEN's known pink/magenta cast
       arriving through the back door -- and a Reinhard transfer cannot undo
       it, because Reinhard matches global mean/std while the cast is
       per-pixel. Restoring luminance and taking chrominance from the crop is
       the operator that measures 0.44-0.45 on both GPEN processors, and it
       is what this uses.

    2. COST. Three channels of guided filter, run over both streams, is ~82 ms
       of host work on a 512 crop -- more than either network. Luminance is one
       channel, so the filter does a third of the work.

    Detail is not lost by going luma-only: a restorer's pore, lash and lip
    texture is luminance structure. Chroma high-frequency at these scales is
    mostly the network's own tint, which is the thing being removed.
    """
    s = np.ascontiguousarray(structural, dtype=np.uint8)
    d = np.ascontiguousarray(detail, dtype=np.uint8)
    if s.shape != d.shape:
        raise ValueError(f"stream shape mismatch: {s.shape} vs {d.shape}")

    s_l = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    d_l = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    low = guided_filter(s_l, radius, eps)
    high = d_l - guided_filter(d_l, radius, eps)
    if clamp is not None:
        lim = float(clamp) / 255.0
        np.clip(high, -lim, lim, out=high)

    # The new luminance, as a SIGNED OFFSET applied to the structural image.
    # A luminance-only edit is the same delta on all three channels, so this
    # moves brightness and leaves the structural stream's hue and saturation
    # exactly where they were -- the same identity `luma_only_recolour` uses.
    delta = ((low + float(gain) * high) - s_l) * 255.0
    out = s.astype(np.float32) + delta[:, :, None]
    return cv2.convertScaleAbs(out)


def reinhard_lab(image, reference, strength=1.0, l_weight=1.0):
    """Match `image`'s LAB mean/std to `reference`'s (Reinhard et al. 2001).

    `image` and `reference` are uint8 BGR; they need not share a shape, since
    only their global statistics are used.

    The per-channel std ratio is clamped to [0.80, 1.20]. An affine rescale
    about the mean cannot blur -- it scales every deviation, high-frequency
    ones included, by one constant -- but an UNCLAMPED ratio can amplify a
    low-variance crop's noise without bound, and can multiply a cast rather
    than remove it. `procmgr_color`'s RCT already carries this clamp.

    `l_weight` scales the luminance transfer separately from chrominance. Full
    L matching drags the restored face's contrast back toward the degraded
    crop's, giving back part of what the restorer just recovered; the A/B
    channels carry the colour cast this transfer exists to remove and have no
    such downside.
    """
    src = cv2.cvtColor(np.ascontiguousarray(image, dtype=np.uint8),
                       cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(np.ascontiguousarray(reference, dtype=np.uint8),
                       cv2.COLOR_BGR2LAB).astype(np.float32)

    s_mean, s_std = cv2.meanStdDev(src)
    r_mean, r_std = cv2.meanStdDev(ref)
    s_mean = s_mean.reshape(3).astype(np.float32)
    s_std = s_std.reshape(3).astype(np.float32)
    r_mean = r_mean.reshape(3).astype(np.float32)
    r_std = r_std.reshape(3).astype(np.float32)

    ratio = np.clip(r_std / np.maximum(s_std, 1e-3), _STD_LO, _STD_HI)
    matched = (src - s_mean) * ratio + r_mean

    w = np.array([float(strength) * float(l_weight),
                  float(strength), float(strength)], dtype=np.float32)
    out = src + w * (matched - src)
    return cv2.cvtColor(np.clip(out, 0.0, 255.0).astype(np.uint8),
                        cv2.COLOR_LAB2BGR)
