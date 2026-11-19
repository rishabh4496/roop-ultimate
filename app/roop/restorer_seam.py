"""Inward-feathered tone lock for a restorer's output patch.

A restorer (GPEN Realistic, UltraMax, CodeFormer ...) is trained to produce a plausible face, not to
keep the plate's exposure: it regrades what it is given (see the comment above the optional
``color_match_after_enhance`` pass in ``ProcessMgr``, which is off by default). The patch is then pasted
under a matte whose alpha ramps from 0 to 1 over a few percent of the face. Across that ramp the
content is *restorer output* on one side and *plate* on the other; if the restorer moved the tone, the
ramp is a visible step in lightness, and because the patch is a crop's worth of face it reads as a
rectangular or oval "sticker".

``rim_tone_lock`` fixes that in the one place it matters. Toward the matte's rim the restorer's LOW band
(lighting, tone) is handed back to the colour-matched swap while its HIGH band (pores, lashes, the
sharpness the restorer is there to add) is kept::

    out = restored + (1 - w) * (low(swap) - low(restored))        w = rim_weight(matte)

``w`` is 0 on the rim and ramps to 1 over ``rim_frac`` of the face width, inside. At ``w = 1``
(everywhere past the ramp) ``out == restored`` exactly, so the interior and every pixel of a patch that
already matched the swap are untouched.
"""
from __future__ import annotations

import cv2
import numpy as np

DEFAULT_RIM_FRAC = 0.10        # inward ramp width, as a fraction of the face (matte) width
DEFAULT_SIGMA_FRAC = 0.05      # low-band blur sigma, as a fraction of the face width
MIN_RIM_PX = 3.0


def face_span(alpha) -> float:
    """Width proxy of the pasted face in crop pixels: sqrt of the matte's half-alpha area."""
    a = np.asarray(alpha, np.float32)
    return float(np.sqrt(max(int((a > 0.5).sum()), 1)))


def rim_weight(alpha, rim_frac: float = DEFAULT_RIM_FRAC) -> np.ndarray:
    """0 on/outside the matte's half-alpha contour, smoothstep to 1 over ``rim_frac`` of the face width."""
    a = np.asarray(alpha, np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    inside = (a > 0.5).astype(np.uint8)
    if not inside.any():
        return np.zeros(a.shape, np.float32)
    dist = cv2.distanceTransform(np.pad(inside, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    r = max(MIN_RIM_PX, float(rim_frac) * face_span(a))
    t = np.clip(dist / r, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def rim_tone_lock(restored, swap, alpha, *, rim_frac: float = DEFAULT_RIM_FRAC,
                  sigma_frac: float = DEFAULT_SIGMA_FRAC) -> np.ndarray:
    """Return ``restored`` with its low band blended toward ``swap``'s toward the matte rim.

    ``restored`` and ``swap`` are same-shape BGR images in the crop the matte ``alpha`` is defined on
    (0..255, any dtype); the result is float32 in the same range. ``rim_frac <= 0`` is a no-op.
    """
    r = np.asarray(restored, np.float32)
    if rim_frac <= 0.0:
        return r
    s = np.asarray(swap, np.float32)
    if s.shape != r.shape:
        s = cv2.resize(s, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_AREA)
    a = np.asarray(alpha, np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    if a.shape != r.shape[:2]:
        a = cv2.resize(a, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_LINEAR)
    w = rim_weight(a, rim_frac)
    if float(w.min()) >= 1.0:
        return r
    sigma = float(np.clip(sigma_frac * face_span(a), 2.0, 24.0))
    # The correction is a smooth field, so measure it at reduced size and upsample it.
    f = min(1.0, 128.0 / max(1, min(r.shape[:2])))
    if f < 1.0:
        lw, lh = max(8, int(round(r.shape[1] * f))), max(8, int(round(r.shape[0] * f)))
        sg = max(0.8, sigma * f)
        low_s = cv2.GaussianBlur(cv2.resize(s, (lw, lh), interpolation=cv2.INTER_AREA), (0, 0), sg)
        low_r = cv2.GaussianBlur(cv2.resize(r, (lw, lh), interpolation=cv2.INTER_AREA), (0, 0), sg)
        corr = cv2.resize(low_s - low_r, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        corr = cv2.GaussianBlur(s, (0, 0), sigma) - cv2.GaussianBlur(r, (0, 0), sigma)
    return r + (1.0 - w)[..., None] * corr
