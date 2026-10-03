"""Exact region-of-support versions of the full-frame mask blur / erode / dilate.

WHY. The paste matte is a full-frame uint8 image that is non-zero only around one face.
The feather chain (`blur_area`: 3x3 blur, elliptical erode, a Gaussian ~10% of the face
size wide) and the landmark-hull dilate (`create_landmark_mask`) were run over the WHOLE
frame. Instrumented through a real 1080p render (2026-10-03, 90 frames, RTX 4070, one
swapped face, config as shipped)::

    GaussianBlur k105-113   1920x1080 uint8    ~77 ms/call
    dilate 39-43 ellipse    1920x1080 uint8    ~43 ms/call
    erode  27-29 ellipse    1920x1080 uint8    ~19 ms/call
    cv2 blur/morphology total: 224 CPU-ms per frame

None of that touches a pixel that is not within the kernel radius of the face. These
helpers run the SAME cv2 call on the matte's bounding box padded by twice the kernel
radius and write the result into a zero frame.

EXACT, NOT APPROXIMATE. Outside the bounding box the input is 0, and the Gaussian, the
erosion and the dilation of 0 are 0 (erode's border is +inf and dilate's is -inf, so a
crop's edge never invents a value). Inside the padded box every output pixel's window
either lies in the box or reflects (BORDER_REFLECT_101, the Gaussian's border) from a
position at least `pad - r` outside the support, which is 0. `pad = 2 * r + 2` makes
that hold at every box edge that is not the frame edge, and at a frame edge the box edge
IS the frame edge, where cv2 applies the same border rule it would have applied anyway.
`tests/test_mask_roi.py` checks `np.array_equal` against cv2 on random and adversarial
masks, kernels and frame-edge contact; there is no tolerance.

Falls back to the plain cv2 call for anything it cannot prove exact: not 2-D uint8, an
empty support (returns zeros, which is what cv2 returns), or a support so large the crop
saves nothing. ROOP_MASK_ROI=0 disables it.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

# A crop covering more than this share of the frame is not worth the bookkeeping.
_MAX_FRACTION = 0.6


def enabled() -> bool:
    return os.environ.get("ROOP_MASK_ROI", "1").strip().lower() not in ("0", "false", "off", "no")


def _roi(mask: np.ndarray, radius: int):
    """(y0, y1, x0, x1) of the non-zero support padded by 2r+2, or None to use cv2 directly.

    Returns ``()`` for an empty mask (the caller returns zeros).
    """
    if not enabled() or mask.ndim != 2 or mask.dtype != np.uint8 or mask.size == 0:
        return None
    x, y, w, h = cv2.boundingRect(mask)          # bounding box of the NON-ZERO pixels
    if w == 0 or h == 0:
        return ()
    pad = 2 * int(radius) + 2
    height, width = mask.shape
    y0, y1 = max(0, y - pad), min(height, y + h + pad)
    x0, x1 = max(0, x - pad), min(width, x + w + pad)
    if (y1 - y0) * (x1 - x0) > _MAX_FRACTION * height * width:
        return None
    return y0, y1, x0, x1


def _apply(mask: np.ndarray, radius: int, op) -> np.ndarray:
    box = _roi(mask, radius)
    if box is None:
        return op(mask)
    if box == ():
        return np.zeros_like(mask)
    y0, y1, x0, x1 = box
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = op(np.ascontiguousarray(mask[y0:y1, x0:x1]))
    return out


def gaussian_blur(mask: np.ndarray, ksize, sigma_x: float = 0) -> np.ndarray:
    """``cv2.GaussianBlur(mask, ksize, sigma_x)`` computed over the matte's support only."""
    kx, ky = (ksize, ksize) if np.isscalar(ksize) else (int(ksize[0]), int(ksize[1]))
    radius = max(kx, ky) // 2
    return _apply(mask, radius, lambda m: cv2.GaussianBlur(m, (kx, ky), sigma_x))


def erode(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """``cv2.erode(mask, kernel)`` (one iteration, default border) over the support only."""
    radius = max(kernel.shape[:2]) // 2
    return _apply(mask, radius, lambda m: cv2.erode(m, kernel, iterations=1))


def dilate(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """``cv2.dilate(mask, kernel)`` (one iteration, default border) over the support only."""
    radius = max(kernel.shape[:2]) // 2
    return _apply(mask, radius, lambda m: cv2.dilate(m, kernel, iterations=1))
