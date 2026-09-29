"""Reinhard colour transfer in CIELAB, statistics taken inside a mask.

    out = (src - mean_src) * (std_tgt / std_src) + mean_tgt      per L, a, b

`source_patch` is the SYNTHETIC crop (the swap output), `target_patch` the
target's own crop, both aligned the same way, and `mask` a weight map in that
crop space (the composite swap mask is the natural choice: it is where the two
must agree). Every pixel of the source is moved -- the paste mask decides where
the result lands.

Float LAB (cv2's float32 path: L 0..100, a/b about -127..127), not 8-bit LAB:
the 8-bit path quantises a/b to integers before the scale is applied, and a
skin tone lives in a few units of a/b.

Two measured cautions from this repo, both honoured here:
  * transfer the raw per-channel statistics; constraining the direction
    ("colour only, not luminance") in gamma RGB rotated the chroma vector and
    closed 0.5% of a gap instead of 93% (memory: luma-projection-destroys-
    colour-transfer). This works in LAB, per channel, with no projection.
  * std ratios are clamped to [1/max_scale, max_scale]: over a flat or tiny
    region the source std approaches zero and the ratio amplifies noise into
    speckle.

A CUDA torch tensor pair is processed in torch on its device (sRGB -> XYZ D65
-> LAB with OpenCV's constants), matching cv2 to ~1e-3 LAB units; numpy input
stays on the CPU (see mask_engine's module note on why).
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-6


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


def _weighted_stats(lab, w, wsum):
    """Per-channel weighted mean and std. lab (H,W,3), w (H,W,1)."""
    mean = (lab * w).sum(axis=(0, 1)) / wsum
    var = (((lab - mean) ** 2) * w).sum(axis=(0, 1)) / wsum
    return mean, np.sqrt(np.maximum(var, 0.0))


def _to_unit(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return np.clip(img.astype(np.float32), 0.0, 1.0)


def match_color_reinhard(source_patch: np.ndarray, target_patch: np.ndarray, mask: np.ndarray,
                         max_scale: float = 2.0, min_weight: float = 64.0) -> np.ndarray:
    """Recolour `source_patch` to the target's LAB statistics inside `mask`.

    Patches: (H,W,3) BGR, uint8 or float in [0,1]; the result has the source's
    dtype. mask: (H,W) or (H,W,1) weights in [0,1]. When the mask carries less
    than `min_weight` pixels of weight the source is returned unchanged -- a
    statistic over a handful of pixels is not the face's colour.
    """
    if _is_torch(source_patch):
        return _match_torch(source_patch, target_patch, mask, max_scale, min_weight)
    import cv2
    src = np.asarray(source_patch)
    tgt = np.asarray(target_patch)
    if src.ndim != 3 or src.shape[2] != 3 or src.shape != tgt.shape:
        raise ValueError(f"patches must be matching (H,W,3), got {src.shape} and {tgt.shape}")
    w = np.asarray(mask, dtype=np.float32).reshape(src.shape[:2] + (1,))
    w = np.clip(w, 0.0, 1.0)
    wsum = float(w.sum())
    if wsum < min_weight:
        return src.copy()
    src_lab = cv2.cvtColor(_to_unit(src), cv2.COLOR_BGR2LAB)
    tgt_lab = cv2.cvtColor(_to_unit(tgt), cv2.COLOR_BGR2LAB)
    s_mean, s_std = _weighted_stats(src_lab, w, wsum)
    t_mean, t_std = _weighted_stats(tgt_lab, w, wsum)
    scale = np.clip(t_std / np.maximum(s_std, _EPS), 1.0 / max_scale, max_scale)
    out_lab = ((src_lab - s_mean) * scale + t_mean).astype(np.float32)
    out_lab[..., 0] = np.clip(out_lab[..., 0], 0.0, 100.0)
    out = np.clip(cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR), 0.0, 1.0)
    if src.dtype == np.uint8:
        return (out * 255.0).round().astype(np.uint8)
    return out.astype(src.dtype)


# ── torch path ────────────────────────────────────────────────────────────────

# OpenCV's sRGB (D65) matrices, and its white point, from color_lab.cpp.
_RGB2XYZ = ((0.412453, 0.357580, 0.180423),
            (0.212671, 0.715160, 0.072169),
            (0.019334, 0.119193, 0.950227))
_WHITE = (0.950456, 1.0, 1.088754)
_XYZ2RGB = tuple(map(tuple, np.linalg.inv(np.array(_RGB2XYZ, dtype=np.float64))))


def _mat3(x, m):
    """x (...,3) times a 3x3 m^T as explicit multiply-adds.

    NOT `x @ m.T`: roop/core.py turns TF32 matmul on globally at import, and
    under TF32 a matmul keeps a 10-bit mantissa -- measured, the LAB round trip
    went 0.0035 -> 0.0085 (2.2/255). Elementwise fp32 math is not affected.
    """
    import torch
    return torch.stack([x[..., 0] * m[i][0] + x[..., 1] * m[i][1] + x[..., 2] * m[i][2]
                        for i in range(3)], dim=-1)


def bgr_to_lab_torch(bgr):
    """(...,3) BGR in [0,1] -> LAB, OpenCV's float convention."""
    import torch
    rgb = bgr.flip(-1)
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = _mat3(lin, _RGB2XYZ) / torch.tensor(_WHITE, dtype=bgr.dtype, device=bgr.device)
    d = 6.0 / 29.0
    f = torch.where(xyz > d ** 3, xyz.clamp_min(0) ** (1.0 / 3.0), xyz / (3 * d * d) + 4.0 / 29.0)
    L = torch.where(xyz[..., 1] > d ** 3, 116.0 * f[..., 1] - 16.0, 903.3 * xyz[..., 1])
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], dim=-1)


def lab_to_bgr_torch(lab):
    import torch
    L, a, b = lab.unbind(-1)
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    d = 6.0 / 29.0
    inv = lambda t: torch.where(t > d, t ** 3, 3 * d * d * (t - 4.0 / 29.0))
    y = torch.where(L > 903.3 * d ** 3, fy ** 3, L / 903.3)
    xyz = torch.stack([inv(fx), y, inv(fz)], dim=-1)
    xyz = xyz * torch.tensor(_WHITE, dtype=lab.dtype, device=lab.device)
    lin = _mat3(xyz, _XYZ2RGB).clamp(0.0, 1.0)
    rgb = torch.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1.0 / 2.4) - 0.055)
    return rgb.flip(-1)


def _match_torch(src, tgt, mask, max_scale, min_weight):
    import torch
    if tuple(src.shape) != tuple(tgt.shape) or src.dim() != 3 or src.shape[-1] != 3:
        raise ValueError(f"patches must be matching (H,W,3), got {tuple(src.shape)} and {tuple(tgt.shape)}")
    to_unit = (lambda t: t.to(torch.float32) / 255.0) if src.dtype == torch.uint8 else \
        (lambda t: t.to(torch.float32).clamp(0.0, 1.0))
    w = torch.as_tensor(mask, device=src.device, dtype=torch.float32).reshape(
        src.shape[0], src.shape[1], 1).clamp(0.0, 1.0)
    wsum = w.sum()
    if float(wsum) < min_weight:
        return src.clone()
    s_lab, t_lab = bgr_to_lab_torch(to_unit(src)), bgr_to_lab_torch(to_unit(tgt.to(src.device)))

    def stats(lab):
        mean = (lab * w).sum(dim=(0, 1)) / wsum
        std = ((((lab - mean) ** 2) * w).sum(dim=(0, 1)) / wsum).clamp_min(0).sqrt()
        return mean, std
    s_mean, s_std = stats(s_lab)
    t_mean, t_std = stats(t_lab)
    scale = (t_std / s_std.clamp_min(_EPS)).clamp(1.0 / max_scale, max_scale)
    out_lab = (s_lab - s_mean) * scale + t_mean
    out_lab[..., 0] = out_lab[..., 0].clamp(0.0, 100.0)
    out = lab_to_bgr_torch(out_lab).clamp(0.0, 1.0)
    if src.dtype == torch.uint8:
        return (out * 255.0).round().to(torch.uint8)
    return out.to(src.dtype)
