"""Colour stabilisation: tie a swapped / restored face's colour back to the target.

Host (:func:`transfer_color`, uint8 numpy, OpenCV 8-bit LAB) and CUDA
(:func:`transfer_color_cuda`, float tensors, CIE L*a*b*) implement the same
modes. OpenCV's 8-bit LAB is an affine per-channel rescaling of CIE L*a*b*
(``L * 255/100``, ``a + 128``, ``b + 128``, same sRGB gamma), and mean shifts
and standard-deviation ratios are invariant under such a rescaling, so both
paths compute the same correction; they differ only by the host path's 8-bit
rounding.

Statistics come from a region (the face interior); the correction applies to
every pixel. Nothing leaves the device on the CUDA path: a batch member whose
region is too small (< 16 px) is returned unchanged via ``torch.where``.
"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    import torch

MIN_REGION_PIXELS = 16


class ColorMode(str, Enum):
    """How a face's colour is tied back to a reference.

    NONE        the network's colour as-is.
    LAB_MEAN    shift the LAB channel means to the reference's (face region).
    REINHARD    LAB mean AND standard deviation (Reinhard et al. 2001). Matching
                L's spread also scales back texture contrast the restorer added.
    KEEP_CHROMA take L from the restored face and a/b from the reference: all
                restored luminance detail, none of its colour cast.
    """

    NONE = "none"
    LAB_MEAN = "lab_mean"
    REINHARD = "reinhard"
    KEEP_CHROMA = "keep_chroma"


def transfer_color(image: np.ndarray, reference: np.ndarray, mode: ColorMode,
                   region: np.ndarray | None = None) -> np.ndarray:
    """Match ``image``'s colour to ``reference`` in LAB (both uint8 BGR, same size).

    Statistics are taken inside ``region`` (uint8/bool mask) when given; the
    correction is applied to every pixel.
    """
    if mode is ColorMode.NONE:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    sel = np.ones(image.shape[:2], bool) if region is None else np.asarray(region) > 0
    if sel.sum() < MIN_REGION_PIXELS:
        return image
    if mode is ColorMode.KEEP_CHROMA:
        lab[:, :, 1:] = ref[:, :, 1:]
    else:
        for c in range(3):
            mu_i, mu_r = lab[:, :, c][sel].mean(), ref[:, :, c][sel].mean()
            if mode is ColorMode.REINHARD:
                sd_i, sd_r = lab[:, :, c][sel].std(), ref[:, :, c][sel].std()
                lab[:, :, c] = (lab[:, :, c] - mu_i) * (sd_r / max(sd_i, 1e-3)) + mu_r
            else:
                lab[:, :, c] += mu_r - mu_i
    return cv2.cvtColor(np.clip(np.rint(lab), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------- CUDA
def bgr_to_lab(image: torch.Tensor) -> torch.Tensor:
    """``(B, 3, H, W)`` BGR ``[0, 255]`` -> CIE L*a*b* (L in ``[0, 100]``), D65."""
    from kornia.color import rgb_to_lab

    return rgb_to_lab((image.flip(1) / 255.0).clamp(0, 1))


def lab_to_bgr(lab: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`bgr_to_lab`; returns BGR ``[0, 255]`` (clamped)."""
    from kornia.color import lab_to_rgb

    return (lab_to_rgb(lab, clip=True) * 255.0).flip(1)


def _masked_moments(lab: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample, per-channel mean and std over ``weight`` (``(B, 1, H, W)``)."""
    total = weight.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    mean = (lab * weight).sum(dim=(2, 3), keepdim=True) / total
    var = (((lab - mean) ** 2) * weight).sum(dim=(2, 3), keepdim=True) / total
    return mean, var.clamp_min(0).sqrt()


def transfer_color_cuda(image: torch.Tensor, reference: torch.Tensor, mode: ColorMode,
                        region: torch.Tensor | None = None) -> torch.Tensor:
    """Batched :func:`transfer_color` on the tensors' device.

    Args:
        image: ``(B, 3, H, W)`` BGR ``[0, 255]`` float.
        reference: Same shape (the ORIGINAL target crop, normally).
        mode: :class:`ColorMode`.
        region: ``(B or 1, 1, H, W)`` weights / 0-1 mask for the statistics;
            default the whole crop.

    Returns:
        ``(B, 3, H, W)`` BGR float ``[0, 255]``.
    """
    import torch

    if mode is ColorMode.NONE:
        return image
    lab, ref = bgr_to_lab(image.float()), bgr_to_lab(reference.float())
    b, _, h, w = lab.shape
    weight = (torch.ones((1, 1, h, w), device=lab.device) if region is None
              else region.to(device=lab.device, dtype=lab.dtype)).expand(b, 1, h, w)
    if mode is ColorMode.KEEP_CHROMA:
        out = torch.cat([lab[:, :1], ref[:, 1:]], dim=1)
    else:
        mu_i, sd_i = _masked_moments(lab, weight)
        mu_r, sd_r = _masked_moments(ref, weight)
        if mode is ColorMode.REINHARD:
            out = (lab - mu_i) * (sd_r / sd_i.clamp_min(1e-3 * 100 / 255)) + mu_r
        else:
            out = lab + (mu_r - mu_i)
    enough = (weight > 0.5).flatten(1).sum(1) >= MIN_REGION_PIXELS
    return torch.where(enough.view(b, 1, 1, 1), lab_to_bgr(out), image.float())
