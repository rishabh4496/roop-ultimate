"""Frequency-split detail injection: the restorer's texture on the swap's light.

For a swapped crop ``S`` and a restored crop ``R`` of the same face (both
``(B, 3, H, W)`` on the GPU, same template and size)::

    L_S = G_sigma * S            low band of the swap: lighting, colour, identity
    L_R = G_sigma * R
    H_R = R - L_R                restorer's high band: pores, lashes, fine lines
    H_S = S - L_S
    fused = L_S + boost * H_R + swap_detail * H_S,   clamped to [0, max_value]

``swap_detail = 0`` is the brief's formula exactly. ``swap_detail = 1 - boost``
makes ``boost`` a crossfade between the two high bands instead: ``boost = 0``
returns ``S`` unchanged, ``boost = 1`` swaps in the restorer's detail. That is
the form :mod:`face_engine.enhancers.semantic_fusion` uses per region. With the
brief's form, a low ``boost`` (0.15 over the inner mouth) would also delete
the swap's own teeth detail, leaving a blurred mouth, which is the opposite
of "attenuate the restoration there".

Why a split rather than ``alpha * R + (1 - alpha) * S``: a linear blend moves
the LOW band too, so the restorer's lighting, skin tone and whatever identity
drift it has come along at weight ``alpha``; the split takes the low band from
the swap alone (``test_ultra_quality`` measures both).

The kernel
----------
``kernel_size=None`` (default) uses ``2 * ceil(3 sigma) + 1`` (17 at 2.5). The
brief's ``(9, 9)`` at sigma 2.5 truncates the Gaussian at 1.6 sigma, and the
renormalized stub behaves like a SMALLER sigma: 27% less of a 24 px wave
reaches the detail band than a true sigma-2.5 split (``test_ultra_quality``).
Pass ``kernel_size=9`` for the brief's exact filter.

Sigma is in CROP pixels, and a crop is pasted back at its on-screen size: a
1024 crop over a face ~500 px across is shrunk ~2x, so crop-space sigma 2.5
lands at ~1.25 screen px. Detail finer than that is filtered out again by the
paste (roop-ultimate measured a texture filter at crop sigma 1.1 moving the
rendered frame by 0.4%, 2026-08-23). :func:`paste_aware_sigma` keeps the split
at a fixed size ON SCREEN instead. Measured on real footage (2026-09-28, 78
faces, :mod:`face_engine.enhancers.ultra_engine`), crop sigma 2.5 DOES survive
at the router's face sizes (skin texture 160% of the plate) and paste-aware
sigma lost identity at p10 (0.959 -> 0.922) for +6 points of texture, so fixed
2.5 is the default.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def gaussian_kernel_size(sigma: float) -> int:
    """Odd kernel covering +-3 sigma."""
    return 2 * math.ceil(3.0 * float(sigma)) + 1


def paste_aware_sigma(paste_ratio: torch.Tensor, screen_sigma: float = 1.25,
                      min_sigma: float = 1.0, max_sigma: float = 8.0) -> torch.Tensor:
    """Per-face crop-space sigma that is ``screen_sigma`` pixels in the frame.

    ``paste_ratio`` ``(N,)`` = frame pixels per crop pixel (``1 / matrix scale``;
    0.5 = the crop is shrunk 2x when pasted). The default 1.25 is the brief's
    sigma 2.5 on a 1024 crop pasted at half size.
    """
    return (screen_sigma / paste_ratio.clamp_min(1e-3)).clamp(min_sigma, max_sigma)


class FrequencySplitBlender:
    """Low band from the swap, high band from the restorer (see the module docstring).

    Args:
        sigma: Gaussian sigma of the split, crop pixels (a float, or per face
            at call time).
        kernel_size: Odd kernel side; None = ``2 * ceil(3 sigma) + 1``.
        texture_boost: Weight of the restorer's high band (a float, or a
            ``(B, 1, H, W)`` / ``(B, 1, 1, 1)`` tensor at call time).
        swap_detail: Weight of the swap's own high band; ``"complement"`` =
            ``1 - texture_boost`` (crossfade), 0.0 = the brief's formula.
        max_value: Clamp range top: 255 for this package's BGR crops, 1.0 for
            ``[0, 1]`` tensors.
        low_pass: ``"gaussian"`` or ``"bilateral"`` (edge-preserving: no halo
            of the restorer's detail across a strong edge; ~10x the cost).
        sigma_color: Bilateral range sigma, as a fraction of ``max_value``.
    """

    def __init__(self, sigma: float = 2.5, kernel_size: int | None = None,
                 texture_boost: float = 1.0, swap_detail: float | str = 0.0,
                 max_value: float = 255.0, low_pass: str = "gaussian",
                 sigma_color: float = 0.1) -> None:
        if low_pass not in ("gaussian", "bilateral"):
            raise ValueError("low_pass must be 'gaussian' or 'bilateral'")
        if kernel_size is not None and kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if not (swap_detail == "complement" or isinstance(swap_detail, (int, float))):
            raise ValueError("swap_detail must be a number or 'complement'")
        self.sigma = float(sigma)
        self.kernel_size = kernel_size
        self.texture_boost = texture_boost
        self.swap_detail = swap_detail
        self.max_value = float(max_value)
        self.low_pass = low_pass
        self.sigma_color = float(sigma_color)

    def _kernel(self, sigma: Any) -> int:
        if self.kernel_size is not None:
            return self.kernel_size
        import torch

        peak = float(sigma.max()) if torch.is_tensor(sigma) else float(sigma)  # host scalar
        return gaussian_kernel_size(peak)

    def low(self, x: torch.Tensor, sigma: Any = None, kernel: int | None = None) -> torch.Tensor:
        """The low band of ``x``; ``sigma`` a float or ``(B,)`` per-image tensor."""
        import torch
        from kornia.filters import bilateral_blur, gaussian_blur2d

        s = self.sigma if sigma is None else sigma
        k = kernel or self._kernel(s)
        if torch.is_tensor(s):
            s = s.to(device=x.device, dtype=x.dtype).reshape(-1, 1).expand(-1, 2).contiguous()
            if s.shape[0] != x.shape[0]:
                s = s.expand(x.shape[0], 2)
        else:
            s = (float(s), float(s))
        if self.low_pass == "bilateral":
            return bilateral_blur(x, (k, k), self.sigma_color * self.max_value, s)
        return gaussian_blur2d(x, (k, k), s)

    def split(self, x: torch.Tensor, sigma: Any = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``(low, high)`` with ``low + high == x`` exactly."""
        lo = self.low(x, sigma)
        return lo, x - lo

    def blend(self, swapped: torch.Tensor, restored: torch.Tensor, *,
              texture_boost: Any = None, swap_detail: Any = None,
              sigma: Any = None) -> torch.Tensor:
        """``L_S + boost * H_R + swap_detail * H_S``, clamped.

        ``swapped`` and ``restored`` are the same face on the same template
        and size. ``texture_boost`` / ``swap_detail`` override the constructor
        values (floats or broadcastable tensors, e.g. a per-pixel region map).
        """
        if swapped.shape != restored.shape:
            raise ValueError(f"shape mismatch {tuple(swapped.shape)} vs {tuple(restored.shape)}")
        s, r = swapped.float(), restored.float()
        sig = self.sigma if sigma is None else sigma
        kernel = self._kernel(sig)
        lo_s = self.low(s, sig, kernel)
        hi_r = r - self.low(r, sig, kernel)
        boost = self.texture_boost if texture_boost is None else texture_boost
        keep = self.swap_detail if swap_detail is None else swap_detail
        if isinstance(keep, str):  # "complement"
            keep = 1.0 - boost
        fused = lo_s + boost * hi_r
        if not (isinstance(keep, (int, float)) and keep == 0):
            fused = fused + keep * (s - lo_s)
        return fused.clamp(0.0, self.max_value)


def linear_blend(swapped: torch.Tensor, restored: torch.Tensor, alpha: float,
                 max_value: float = 255.0) -> torch.Tensor:
    """The naive global blend the split replaces (the reference arm)."""
    return (alpha * restored.float() + (1.0 - alpha) * swapped.float()).clamp(0.0, max_value)
