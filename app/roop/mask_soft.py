"""Continuous edge primitives for obstruction (occluder) masks.

Convention here is the engines' own: 1.0 = restore the plate (an obstruction), 0.0 = keep the swap.

Why this exists
---------------
Three obstruction engines turned a probability map into a hard 0/1 step before any softening:

* ``Mask_Occluder``  ``(p > 0.35)`` then a Gaussian that is skipped when its kernel is 1 px,
* ``Mask_Clip2Seg``  ``p > thresh`` assigned 1.0 / 0.0 in place,
* ``Mask_RealityUX`` ``(xseg > 0.40)`` then ``cv2.erode`` of the binary image.

A step function turns a model's frame-to-frame noise into popping: a pixel whose probability
wanders across the threshold flips between "hand" and "face", so the contour buzzes. The binary
erosion in the last one also deletes anything thinner than its kernel (glasses rims, hair strands).

``soft_threshold`` keeps the engine's operating point (0.5 at the old threshold, so ``soft > 0.5``
reproduces the old decision) and replaces the cliff with a smoothstep ramp.
``thin_preserving_erode`` shrinks thick regions as the erosion did but gives back every structure
the erosion would have removed.
"""
from __future__ import annotations

import cv2
import numpy as np

DEFAULT_RAMP_WIDTH = 0.20          # probability span of the ramp (centre +- 0.10)


def smoothstep01(x) -> np.ndarray:
    t = np.clip(np.asarray(x, np.float32), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def soft_threshold(p, center: float, width: float = DEFAULT_RAMP_WIDTH) -> np.ndarray:
    """Smoothstep replacement for ``p > center``.

    0.0 at ``center - width/2``, 1.0 at ``center + width/2``, exactly 0.5 at ``center`` — so
    ``soft_threshold(p, c) > 0.5`` is ``p > c`` and the engine's decision boundary does not move.
    """
    w = max(float(width), 1e-6)
    # In-place on one fresh float32 buffer (the naive form allocated ~6 temporaries: 1.9 ms at 512 px, vs 0.1 ms
    # for the `p > c` it replaces). NaN -> 0 by the clip, +-inf -> the rails.
    t = np.array(p, dtype=np.float32, copy=True)
    t -= np.float32(float(center) - 0.5 * w)
    t *= np.float32(1.0 / w)
    np.nan_to_num(t, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(t, 0.0, 1.0, out=t)
    # t*t*(3-2t)
    u = t * np.float32(-2.0)
    u += np.float32(3.0)
    u *= t
    u *= t
    return u


def thin_preserving_erode(soft, k: int) -> np.ndarray:
    """Grey-scale erosion by a ``k`` x ``k`` square that does not delete thin structures.

    ``eroded = erode(soft)`` shrinks every region by ~k/2 px (the intended effect on a thick hand or
    microphone: the swap is allowed to run up to the object instead of stopping short of it) but
    also removes anything narrower than ``k`` outright. The top-hat ``soft - open(soft)`` is exactly
    that removed material, so ``max(eroded, tophat)`` = thick regions shrunk, thin ones intact.
    """
    s = np.clip(np.asarray(soft, np.float32), 0.0, 1.0)
    k = int(k)
    if k <= 1:
        return s
    # Morphology on a uint8 copy: OpenCV's 8-bit rectangle erode/dilate is SIMD (0.15 ms at 512 px) where the
    # float32 path was 1.5 ms. The 1/255 quantisation is far below the mask's own noise, and 0 / 1 are exact.
    s8 = np.rint(s * 255.0).astype(np.uint8)
    el = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    eroded = cv2.erode(s8, el)
    opened = cv2.dilate(eroded, el)
    tophat = cv2.subtract(s8, opened)                # saturating: structures narrower than k
    return np.maximum(eroded, tophat).astype(np.float32) * np.float32(1.0 / 255.0)
