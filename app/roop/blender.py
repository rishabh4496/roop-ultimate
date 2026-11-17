"""Seam colour matching and multi-band blending (Bug #2, Stage 2).

Pure OpenCV / NumPy (Apache-2.0 / BSD-3-Clause); no weights, no GPU, offline-safe.
Everything works on ROI / crop-sized arrays, never a full frame.

Two independent pieces
----------------------
``skin_lab_match``
    Reinhard transfer in CIE-Lab whose STATISTICS come only from the eroded
    ``skin`` class pixels of the crop. Eyes, brows, lips, mouth interior (teeth)
    and the sclera contribute nothing to the statistics, and the correction is
    weighted by the (softened) skin mask, so those regions are not recoloured
    either. The swapped crop and the target crop are pixel-aligned (same
    affine), so one skin mask indexes both.

``blend``  (``blend_mode``)
    ``laplacian``     4-level Laplacian-pyramid blend. Each band is mixed with the
                      same level of the mask's Gaussian pyramid, so low frequencies
                      (skin lighting) change over a wide band and high frequencies
                      (pores) over a tight one. Default.
    ``poisson_fast``  A seamless-clone style correction without the linear solve:
                      the paste-to-plate difference on the matte's edge band is
                      diffused inward by normalised Gaussian convolution at two
                      scales. This APPROXIMATES Poisson membrane interpolation; it
                      is not an exact Poisson solve.
    ``alpha_feather`` Plain ``alpha*paste + (1-alpha)*plate``. The legacy blend.

Where alpha is 0 every mode returns the plate bit-for-bit, and the Laplacian mode
is gated by ``smoothstep(alpha / 0.1)`` so its coarse bands cannot leave a step at
the edge of the matte or tint the plate outside it.
"""
from __future__ import annotations

import cv2
import numpy as np

BLEND_MODES = ('laplacian', 'poisson_fast', 'alpha_feather')
DEFAULT_BLEND_MODE = 'laplacian'

_LAB_MAX_SHIFT = np.array([14.0, 10.0, 10.0], dtype=np.float32)   # per-pixel cap on the Lab delta
_SCALE_LIMITS = ((0.85, 1.15), (0.75, 1.33), (0.75, 1.33))        # sigma ratio clamps for L, a, b


def normalize_blend_mode(value) -> str:
    v = str(value or '').strip().lower().replace('-', '_').replace(' ', '_')
    return v if v in BLEND_MODES else DEFAULT_BLEND_MODE


# --------------------------------------------------------------------------- #
# Colour / luminance transfer over the eroded skin class
# --------------------------------------------------------------------------- #
def _to_lab(img) -> np.ndarray:
    return cv2.cvtColor(np.clip(np.asarray(img, np.float32) / 255.0, 0.0, 1.0), cv2.COLOR_BGR2LAB)


def _from_lab(lab) -> np.ndarray:
    return cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2BGR) * 255.0


def eroded_skin(skin, *, erode_frac: float = 0.02, min_pixels: int = 256):
    """Skin mask eroded so boundary pixels (hair line, jaw, nostril rim) cannot skew
    the statistics. Halves the erosion once if too little survives; None if still short."""
    s = (np.asarray(skin) > 0.5).astype(np.uint8)
    h, w = s.shape[:2]
    k = max(2, int(round(erode_frac * min(h, w))))
    for kk in (k, max(1, k // 2)):
        el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kk + 1, 2 * kk + 1))
        e = cv2.erode(s, el, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        if int(e.sum()) >= min_pixels:
            return e
    return None


COLOR_MODES = ('reinhard', 'adain', 'cdf')
DEFAULT_COLOR_MODE = 'reinhard'
_ADAIN_SCALE_LIMITS = ((0.5, 2.0), (0.5, 2.0), (0.5, 2.0))   # a sanity bound, not a look: AdaIN matches sigma
_CDF_KNOTS = 33                                              # quantile knots of the per-channel mapping
_CDF_TAIL = 0.01                                             # knots span the 1st..99th percentile


def normalize_color_mode(value) -> str:
    v = str(value or '').strip().lower().replace('-', '_').replace(' ', '_')
    if v in ('histogram', 'cdf_histogram', 'histogram_match', 'cdf_match'):
        v = 'cdf'
    return v if v in COLOR_MODES else DEFAULT_COLOR_MODE


def _cdf_map(content_vals, reference_vals, query):
    """Histogram specification of ONE channel: the monotone map that sends the content's distribution
    onto the reference's, as a piecewise-linear function through ``_CDF_KNOTS`` matched quantiles.

    Where Reinhard / AdaIN match two moments, this matches the whole CDF, so a bimodal reference
    (lit cheek + shadowed cheek) stays bimodal. Values past the content's 1st / 99th percentile move by
    the end knots' offset instead of being clipped onto them.
    """
    q = np.linspace(_CDF_TAIL, 1.0 - _CDF_TAIL, _CDF_KNOTS)
    kc = np.quantile(content_vals, q).astype(np.float64)
    kr = np.quantile(reference_vals, q).astype(np.float64)
    kc = kc + np.arange(_CDF_KNOTS) * 1e-6                       # strictly increasing for np.interp
    out = np.interp(query, kc, kr)
    out = np.where(query < kc[0], query + (kr[0] - kc[0]), out)
    out = np.where(query > kc[-1], query + (kr[-1] - kc[-1]), out)
    return out.astype(np.float32)


def skin_lab_match(content, reference, skin, *, strength: float = 1.0,
                   erode_frac: float = 0.02, soft_frac: float = 0.015,
                   min_pixels: int = 256, work_px: int = 128, mode: str = DEFAULT_COLOR_MODE):
    """Match ``content`` (swapped crop) to ``reference`` (target crop) in Lab.

    Returns ``(corrected float32 BGR 0..255, info)``; ``info['applied']`` is False
    (and ``corrected`` is ``content`` unchanged) when the skin mask is too small.

    ``mode`` picks the statistics model over the eroded skin pixels (all three share the skin
    weight, the per-pixel Lab shift cap and the low-resolution workflow below):

    ``reinhard``  per-channel mean and sigma, sigma ratio clamped to ``_SCALE_LIMITS`` (default).
    ``adain``     the same affine with the sigma ratio left free (0.5..2): full mean/std transfer.
    ``cdf``       per-channel CDF (histogram) matching, ``_cdf_map``.

    The statistics, the clamped delta and the soft skin weight are all
    computed at ~``work_px`` pixels on the short side; only the resulting smooth
    colour CORRECTION is upsampled and added to the full-resolution crop. That is
    ~10x cheaper than converting both crops to Lab at full size, and it leaves the
    pores untouched: the correction is a low-frequency field by construction.
    """
    mode = normalize_color_mode(mode)
    info = {'applied': False, 'pixels': 0, 'mean_shift_lab': (0.0, 0.0, 0.0), 'mode': mode}
    strength = float(np.clip(strength, 0.0, 1.0))
    c = np.asarray(content, np.float32)
    ref = np.asarray(reference, np.float32)
    if strength <= 0.0 or c.shape[:2] != ref.shape[:2]:
        return c, info

    h, w = c.shape[:2]
    f = min(1.0, float(work_px) / max(1, min(h, w)))
    lw, lh = max(8, int(round(w * f))), max(8, int(round(h * f)))
    area = cv2.INTER_AREA
    c_lo, r_lo = cv2.resize(c, (lw, lh), interpolation=area), cv2.resize(ref, (lw, lh), interpolation=area)
    skin_lo = cv2.resize((np.asarray(skin) > 0.5).astype(np.float32), (lw, lh), interpolation=area) > 0.5
    stat = eroded_skin(skin_lo, erode_frac=erode_frac,
                       min_pixels=max(24, int(min_pixels * lw * lh / float(h * w))))
    if stat is None:
        return c, info

    lab_c, lab_r = _to_lab(c_lo), _to_lab(r_lo)
    pc, pr = lab_c[stat], lab_r[stat]
    mu_c, mu_r = pc.mean(0), pr.mean(0)
    sd_c, sd_r = pc.std(0), pr.std(0)
    if mode == 'cdf':
        raw = np.stack([_cdf_map(pc[:, i], pr[:, i], lab_c[..., i]) - lab_c[..., i] for i in range(3)], axis=-1)
    else:
        limits = _ADAIN_SCALE_LIMITS if mode == 'adain' else _SCALE_LIMITS
        scale = np.ones(3, np.float32)
        for i, (lo, hi) in enumerate(limits):
            scale[i] = np.clip(sd_r[i] / sd_c[i], lo, hi) if sd_c[i] > 1e-3 else 1.0
        raw = (lab_c - mu_c) * (scale - 1.0) + (mu_r - mu_c)
    delta = np.clip(raw, -_LAB_MAX_SHIFT, _LAB_MAX_SHIFT)
    weight = cv2.GaussianBlur(skin_lo.astype(np.float32), (0, 0), max(0.7, soft_frac * min(lh, lw)))
    # Difference of two conversions, so Lab<->BGR gamut clipping cancels.
    corr_lo = _from_lab(lab_c + (strength * weight)[..., None] * delta) - _from_lab(lab_c)
    corr = corr_lo if (lw, lh) == (w, h) else cv2.resize(corr_lo, (w, h), interpolation=cv2.INTER_LINEAR)
    info.update(applied=True, pixels=int(stat.sum() / max(f * f, 1e-6)),
                mean_shift_lab=tuple(float(v) for v in (mu_r - mu_c)))
    return np.clip(c + corr, 0.0, 255.0).astype(np.float32), info


def delta_e76(a_bgr, b_bgr) -> np.ndarray:
    """Per-pixel CIE76 colour difference between two BGR images."""
    return np.linalg.norm(_to_lab(a_bgr) - _to_lab(b_bgr), axis=-1)


# --------------------------------------------------------------------------- #
# Blenders
# --------------------------------------------------------------------------- #
def _alpha2d(alpha, shape) -> np.ndarray:
    a = np.asarray(alpha, np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    if a.shape != tuple(shape[:2]):
        raise ValueError('alpha shape %s does not match image %s' % (a.shape, tuple(shape[:2])))
    return np.clip(a, 0.0, 1.0)


def alpha_feather_blend(paste, target, alpha) -> np.ndarray:
    p, t = np.asarray(paste, np.float32), np.asarray(target, np.float32)
    a = _alpha2d(alpha, p.shape)[..., None]
    return a * p + (1.0 - a) * t


def laplacian_blend(paste, target, alpha, levels: int = 4, edge_guard: float = 0.1) -> np.ndarray:
    p, t = np.asarray(paste, np.float32), np.asarray(target, np.float32)
    a = _alpha2d(alpha, p.shape)
    h, w = a.shape
    # Every level must stay >= 2 px, or pyrDown degenerates.
    levels = int(min(levels, np.floor(np.log2(max(min(h, w), 1))) - 1))
    if levels < 1:
        return alpha_feather_blend(p, t, a)

    # The Laplacian pyramid is LINEAR, so the textbook blend
    #     sum_k up^k( a_k * L_k(paste) + (1 - a_k) * L_k(plate) )
    # equals plate + sum_k up^k( a_k * L_k(paste - plate) ): one 3-channel pyramid of the
    # difference instead of two, and exactly paste where every a_k is 1.
    gp_a, gp_d = [a], [p - t]
    for _ in range(levels):
        gp_a.append(cv2.pyrDown(gp_a[-1]))
        gp_d.append(cv2.pyrDown(gp_d[-1]))

    out = gp_a[-1][..., None] * gp_d[-1]
    for k in range(levels - 1, -1, -1):
        size = (gp_a[k].shape[1], gp_a[k].shape[0])
        band = gp_d[k] - cv2.pyrUp(gp_d[k + 1], dstsize=size)
        out = cv2.pyrUp(out, dstsize=size) + gp_a[k][..., None] * band

    g = np.clip(a / max(edge_guard, 1e-6), 0.0, 1.0)
    g = (g * g * (3.0 - 2.0 * g))[..., None]
    return t + g * out


def poisson_fast_blend(paste, target, alpha, max_correction: float = 40.0) -> np.ndarray:
    p, t = np.asarray(paste, np.float32), np.asarray(target, np.float32)
    a = _alpha2d(alpha, p.shape)
    edge = 4.0 * a * (1.0 - a)                      # 1 on the 0.5 contour, 0 where alpha is 0 or 1
    if float(edge.max()) < 1e-4:
        return a[..., None] * p + (1.0 - a[..., None]) * t
    # Normalised convolution of the edge-band difference, at <= 128 px: the correction is a smooth
    # field, so full-resolution Gaussians (sigma up to span/10) would be pure waste.
    h, w = a.shape
    f = min(1.0, 128.0 / max(1, min(h, w)))
    lw, lh = max(8, int(round(w * f))), max(8, int(round(h * f)))
    edge_lo = cv2.resize(edge, (lw, lh), interpolation=cv2.INTER_AREA)
    delta_lo = cv2.resize((t - p) * edge[..., None], (lw, lh), interpolation=cv2.INTER_AREA)
    span = float(np.sqrt(lw * lh))
    corr_lo = np.zeros_like(delta_lo)
    for sigma, weight in ((span / 40.0, 0.5), (span / 10.0, 0.5)):
        sigma = max(sigma, 0.8)
        num = cv2.GaussianBlur(delta_lo, (0, 0), sigma)
        den = cv2.GaussianBlur(edge_lo, (0, 0), sigma)
        corr_lo += weight * num / np.maximum(den, 1e-3)[..., None]
    corr = cv2.resize(corr_lo, (w, h), interpolation=cv2.INTER_LINEAR)
    corr = np.clip(corr, -max_correction, max_correction)
    ak = a[..., None]
    return ak * (p + corr) + (1.0 - ak) * t


def blend(paste, target, alpha, mode: str = DEFAULT_BLEND_MODE) -> np.ndarray:
    """Blend ``paste`` over ``target`` under ``alpha`` (float32 result, 0..255 range)."""
    mode = str(mode)
    if mode == 'laplacian':
        return laplacian_blend(paste, target, alpha)
    if mode == 'poisson_fast':
        return poisson_fast_blend(paste, target, alpha)
    if mode == 'alpha_feather':
        return alpha_feather_blend(paste, target, alpha)
    raise ValueError('unknown blend_mode %r (expected one of %s)' % (mode, ', '.join(BLEND_MODES)))


class Blender:
    """Thin holder so callers can carry a mode around: ``Blender('laplacian').blend(...)``."""

    def __init__(self, mode: str = DEFAULT_BLEND_MODE):
        self.mode = normalize_blend_mode(mode)

    def blend(self, paste, target, alpha):
        return blend(paste, target, alpha, self.mode)

    skin_lab_match = staticmethod(skin_lab_match)
