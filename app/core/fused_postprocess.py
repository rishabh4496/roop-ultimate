"""One-kernel Stage 3 post-processing on the GPU: LAB colour transfer, frequency-split blend, mask feather, grain, composite.

    out_roi = composite(target_roi, face, alpha)
    face    = clamp( blur(CT(swapped)) + w * (restored - blur(restored)) ) + grain
    alpha   = blur(xseg_mask)

All four tensors are planar ``[B, 3|1, H, W]`` in the SAME (ROI) pixel space; resampling the crops into the ROI is the
caller's earlier step (``CudaAffineBatch.paste_faces_roi``). ``CT`` is a per-channel Reinhard transfer in CIE L*a*b* using
precomputed per-face statistics, so a per-Face-ID EMA lives OUTSIDE the kernel (``LabStatsEMA``): the kernel is handed
``(src_mean, scale, tgt_mean)``.

HOW IT IS ONE PASS. A Gaussian is a stencil, and Triton exposes no shared memory, so a scratch buffer in global memory would
undo the point. Instead each program loads one haloed tile of every input (reflect-indexed, so the image border matches
OpenCV's ``BORDER_REFLECT``), does the pointwise colour transfer on the tile in registers, and blurs it with a banded Gaussian
MATRIX built from index arithmetic: ``blur(X) = G @ X @ G`` (``tl.dot``, IEEE fp32). The colour transfer therefore happens
BEFORE the blur, in the order the brief asks for, with no intermediate ever written to global memory. Only the interior of the
tile (tile minus halo) is stored.

NOT in the kernel, on purpose: anything that needs a whole-image reduction. The LAB statistics and the target's noise floor
(``lab_stats``, ``estimate_noise_sigma``) are tiny torch reductions run before it; a tile cannot know the image's median.

Not wired into the render. Production's swap / restore outputs are host arrays (ORT returns numpy), the later stages are
host-frame code, and the pipeline is GPU-bound, so moving this onto the GPU adds GPU work. See ``docs/perf/fused_postprocess/``
for the numbers; ``torch.compile(mode="max-autotune-no-cudagraphs")`` is not an option here (no C++ compiler on this Windows
install for Inductor).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:                                       # pragma: no cover - the CPU/light profile
    triton = tl = None
    HAVE_TRITON = False

# OpenCV's float RGB <-> XYZ (D65) constants, so the reference agrees with cv2.cvtColor on float32 input.
_RGB2XYZ = ((0.412453, 0.357580, 0.180423), (0.212671, 0.715160, 0.072169), (0.019334, 0.119193, 0.950227))
_XYZ2RGB = ((3.240479, -1.53715, -0.498535), (-0.969256, 1.875991, 0.041556), (0.055648, -0.204043, 1.057311))
_WHITE = (0.950456, 1.0, 1.088754)
MAX_HALO = 8        # a 32x32 tile needs >= 16 interior pixels; a 64x64 tile needs 161 KB of shared memory, Ada allows 99 KB


# ---------------------------------------------------------------------------------------------------------------------
# Pure-torch pieces: the reference, and the whole-image reductions the kernel cannot do
# ---------------------------------------------------------------------------------------------------------------------
def _srgb_to_lin(c: torch.Tensor) -> torch.Tensor:
    return torch.where(c <= 0.04045, c / 12.92, ((c.clamp_min(0) + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(c: torch.Tensor) -> torch.Tensor:
    c = c.clamp(0.0, 1.0)
    return torch.where(c <= 0.0031308, 12.92 * c, 1.055 * c.clamp_min(1e-8) ** (1 / 2.4) - 0.055)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """``[..., 3, H, W]`` RGB in [0, 1] -> CIE L*a*b* (L 0..100), matching ``cv2.cvtColor`` on float32."""
    lin = _srgb_to_lin(rgb)
    m = torch.tensor(_RGB2XYZ, dtype=lin.dtype, device=lin.device)
    xyz = torch.einsum("ij,...jhw->...ihw", m, lin)
    w = torch.tensor(_WHITE, dtype=lin.dtype, device=lin.device).view(*([1] * (xyz.dim() - 3)), 3, 1, 1)
    t = xyz / w
    f = torch.where(t > 0.008856, t.clamp_min(1e-8) ** (1 / 3), 7.787 * t + 16.0 / 116.0)
    fx, fy, fz = f.unbind(-3)
    return torch.stack((116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)), dim=-3)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    L, a, b = lab.unbind(-3)
    fy = (L + 16.0) / 116.0
    f = torch.stack((fy + a / 500.0, fy, fy - b / 200.0), dim=-3)
    t = torch.where(f > 6.0 / 29.0, f ** 3, (f - 16.0 / 116.0) / 7.787)
    w = torch.tensor(_WHITE, dtype=lab.dtype, device=lab.device).view(*([1] * (t.dim() - 3)), 3, 1, 1)
    m = torch.tensor(_XYZ2RGB, dtype=lab.dtype, device=lab.device)
    return _lin_to_srgb(torch.einsum("ij,...jhw->...ihw", m, t * w))


def gaussian_taps(sigma: float, radius: Optional[int] = None) -> Tuple[int, float]:
    """(radius, sum of the unnormalised 1-D taps) for a Gaussian of ``sigma``; radius defaults to ceil(3 sigma)."""
    r = int(radius if radius is not None else max(1, math.ceil(3.0 * float(sigma))))
    return r, float(sum(math.exp(-(k * k) / (2.0 * sigma * sigma)) for k in range(-r, r + 1)))


def _reflect(n: int, pad: int, device) -> torch.Tensor:
    """Indices of OpenCV's BORDER_REFLECT (edge pixel repeated): -1 -> 0, -2 -> 1, n -> n-1."""
    i = torch.arange(-pad, n + pad, device=device)
    i = torch.where(i < 0, -i - 1, i)
    return torch.where(i >= n, 2 * n - i - 1, i)


def blur_reference(x: torch.Tensor, sigma: float, radius: Optional[int] = None) -> torch.Tensor:
    """Separable Gaussian with reflect borders, ``[..., H, W]``, float32. The unfused baseline."""
    r, norm = gaussian_taps(sigma, radius)
    k = torch.tensor([math.exp(-(d * d) / (2.0 * sigma * sigma)) for d in range(-r, r + 1)], dtype=torch.float32,
                     device=x.device) / norm
    h, w = x.shape[-2:]
    x = x.index_select(-1, _reflect(w, r, x.device))
    x = sum(k[i] * x[..., i:i + w] for i in range(2 * r + 1))
    x = x.index_select(-2, _reflect(h, r, x.device))
    return sum(k[i] * x[..., i:i + h, :] for i in range(2 * r + 1))


def reinhard_params(src_mean, src_std, tgt_mean, tgt_std):
    """``(src_mean, scale, tgt_mean)`` for the kernel, with production ``rct``'s clamps: chroma std ratio in [0.8, 1.2]."""
    scale = tgt_std / src_std.clamp_min(1e-3)
    scale = torch.stack((scale[..., 0], scale[..., 1].clamp(0.8, 1.2), scale[..., 2].clamp(0.8, 1.2)), dim=-1)
    return src_mean, scale, tgt_mean


def lab_stats(rgb: torch.Tensor, weight: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-image LAB mean and std, ``[B, 3]`` each, over the pixels ``weight`` (``[B, 1, H, W]``) selects (all if None)."""
    lab = rgb_to_lab(rgb.float())
    if weight is None:
        return lab.mean(dim=(-2, -1)), lab.std(dim=(-2, -1)).clamp_min(1.0)
    w = weight.float().expand_as(lab)
    total = w.sum(dim=(-2, -1)).clamp_min(1e-6)
    mean = (lab * w).sum(dim=(-2, -1)) / total
    var = (((lab - mean[..., None, None]) ** 2) * w).sum(dim=(-2, -1)) / total
    return mean, var.sqrt().clamp_min(1.0)


class LabStatsEMA:
    """Per-Face-ID EMA of LAB mean / std, reset on a scene cut. ``alpha`` is the weight of the NEW frame.

    Kept because the brief asks for it; NOTE the measurement behind it: an EMA of the colour-transfer transform made the
    tone pumping WORSE (+9..20%, 402 real frames, docs/perf/color_ema/). ``alpha=1.0`` is "no smoothing".
    """

    def __init__(self, alpha: float = 0.15, cut_threshold: float = 18.0):
        self.alpha, self.cut_threshold, self.state = float(alpha), float(cut_threshold), {}

    def update(self, face_id, mean: torch.Tensor, std: torch.Tensor):
        prev = self.state.get(face_id)
        if prev is None or float((mean - prev[0]).abs().max()) > self.cut_threshold:
            self.state[face_id] = (mean.clone(), std.clone())                 # first frame, or a cut: restart
        else:
            a = self.alpha
            self.state[face_id] = ((1 - a) * prev[0] + a * mean, (1 - a) * prev[1] + a * std)
        return self.state[face_id]

    def reset(self) -> None:
        self.state.clear()


def estimate_noise_sigma(roi: torch.Tensor, weight: Optional[torch.Tensor] = None, max_samples: int = 16384) -> torch.Tensor:
    """White-noise-equivalent sigma per image in 8-bit levels, ``[B]``: Immerkaer's Laplacian + median absolute deviation.

    ``roi`` is ``[B, 3, H, W]`` in [0, 1] (or uint8). Same estimator as ``roop/grain_match.py``; ``weight`` ([B, 1, H, W]) picks
    the pixels (e.g. the ring around the face, or the face under the matte).
    """
    x = roi.float() * (255.0 if roi.dtype != torch.uint8 else 1.0)
    y = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    k = torch.tensor([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    resp = torch.nn.functional.conv2d(torch.nn.functional.pad(y[:, None], (1, 1, 1, 1), mode="replicate"), k)[:, 0]
    out = []
    for i in range(resp.shape[0]):
        v = resp[i] if weight is None else resp[i][weight[i, 0] > 0.5]
        if v.numel() > max_samples:
            v = v[:: v.numel() // max_samples]
        out.append(1.4826 * (v - v.median()).abs().median() / 6.0 if v.numel() >= 400 else v.new_zeros(()))
    return torch.stack(out)


def grain_deficit(sigma_target: torch.Tensor, sigma_face: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Sigma to ADD so the face reaches the plate's noise floor: variances add, never negative (roop/grain_match.py)."""
    return (sigma_target.square() - sigma_face.square()).clamp_min(0).sqrt() * float(strength)


def reference_postprocess(swapped, restored, xseg_mask, target_roi, *, lab=None, detail_weight=0.75, blur_sigma=2.0,
                          feather_sigma=2.0, grain_sigma=None, seed=0, bgr=False, radius=None, feather_radius=None,
                          noise: Optional[torch.Tensor] = None):
    """The SAME maths as the kernel, as a chain of ordinary torch ops. The parity oracle and the unfused benchmark arm.

    ``noise`` (``[B, 1, H, W]`` unit normal) lets a test feed the kernel's own random numbers; without it no grain is added.
    """
    sw, rs = swapped.float(), restored.float()
    if bgr:
        sw, rs = sw.flip(-3), rs.flip(-3)
    if lab is not None:
        sm, sc, tm = (t.float()[..., None, None] for t in lab)
        sw = lab_to_rgb((rgb_to_lab(sw) - sm) * sc + tm)
    face = blur_reference(sw, blur_sigma, radius) + detail_weight * (rs - blur_reference(rs, blur_sigma, radius))
    face = face.clamp(0.0, 1.0)
    if grain_sigma is not None and noise is not None:
        face = face + noise * (grain_sigma.float()[:, None, None, None] / 255.0)
    alpha = blur_reference(xseg_mask.float(), feather_sigma, feather_radius).clamp(0.0, 1.0)
    tgt = target_roi.float() / (255.0 if target_roi.dtype == torch.uint8 else 1.0)
    if bgr:
        face = face.flip(-3)
    out = tgt + alpha * (face - tgt)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------------------------------------------------
if HAVE_TRITON:
    @triton.jit
    def _srgb_to_lin_k(c):
        return tl.where(c <= 0.04045, c / 12.92, tl.exp2(2.4 * tl.log2((tl.maximum(c, 0.0) + 0.055) / 1.055)))

    @triton.jit
    def _lin_to_srgb_k(c):
        c = tl.minimum(tl.maximum(c, 0.0), 1.0)
        return tl.where(c <= 0.0031308, 12.92 * c, 1.055 * tl.exp2(tl.log2(tl.maximum(c, 1e-8)) / 2.4) - 0.055)

    @triton.jit
    def _f_k(t):
        return tl.where(t > 0.008856, tl.exp2(tl.log2(tl.maximum(t, 1e-8)) / 3.0), 7.787 * t + 16.0 / 116.0)

    @triton.jit
    def _finv_k(f):
        return tl.where(f > 6.0 / 29.0, f * f * f, (f - 16.0 / 116.0) / 7.787)

    @triton.jit
    def _band(BLOCK: tl.constexpr, RAD: tl.constexpr, SIGMA: tl.constexpr, NORM: tl.constexpr):
        i = tl.arange(0, BLOCK)
        d = (i[:, None] - i[None, :]).to(tl.float32)
        return tl.where((d >= -RAD) & (d <= RAD), tl.exp(-(d * d) / (2.0 * SIGMA * SIGMA)) / NORM, 0.0)

    @triton.jit
    def _blur_k(x, g):
        return tl.dot(tl.dot(g, x, input_precision="ieee"), g, input_precision="ieee")

    @triton.jit
    def _fused_post_kernel(
        swap_ptr, rest_ptr, mask_ptr, tgt_ptr, out_ptr, sm_ptr, sc_ptr, tm_ptr, gs_ptr,
        H, W, t_sb, t_sc, t_sy, t_sx, o_sb, o_sc, o_sy, o_sx, w_detail, seed,
        BLOCK: tl.constexpr, R: tl.constexpr, SIGMA: tl.constexpr, NORM: tl.constexpr,
        RM: tl.constexpr, SIGMA_M: tl.constexpr, NORM_M: tl.constexpr, HALO: tl.constexpr,
        DO_CT: tl.constexpr, DO_GRAIN: tl.constexpr, T_U8: tl.constexpr, O_U8: tl.constexpr, BGR: tl.constexpr,
    ):
        TILE: tl.constexpr = BLOCK - 2 * HALO
        b = tl.program_id(0)
        ty = tl.program_id(1)
        tx = tl.program_id(2)
        ys = ty * TILE - HALO + tl.arange(0, BLOCK)                       # frame rows of the haloed tile
        xs = tx * TILE - HALO + tl.arange(0, BLOCK)
        yr = tl.where(ys < 0, -ys - 1, ys)
        yr = tl.where(yr >= H, 2 * H - yr - 1, yr)                         # BORDER_REFLECT
        xr = tl.where(xs < 0, -xs - 1, xs)
        xr = tl.where(xr >= W, 2 * W - xr - 1, xr)
        off = yr[:, None] * W + xr[None, :]
        plane = H * W
        sbase = b * 3 * plane
        mbase = b * plane

        s0 = tl.load(swap_ptr + sbase + off).to(tl.float32)
        s1 = tl.load(swap_ptr + sbase + plane + off).to(tl.float32)
        s2 = tl.load(swap_ptr + sbase + 2 * plane + off).to(tl.float32)
        if BGR:
            sr, sg, sb_ = s2, s1, s0
        else:
            sr, sg, sb_ = s0, s1, s2

        if DO_CT:                                                          # Reinhard in CIE L*a*b*, per pixel, in registers
            lr = _srgb_to_lin_k(sr)
            lg = _srgb_to_lin_k(sg)
            lb = _srgb_to_lin_k(sb_)
            fx = _f_k((0.412453 * lr + 0.357580 * lg + 0.180423 * lb) / 0.950456)
            fy = _f_k(0.212671 * lr + 0.715160 * lg + 0.072169 * lb)
            fz = _f_k((0.019334 * lr + 0.119193 * lg + 0.950227 * lb) / 1.088754)
            L = (116.0 * fy - 16.0 - tl.load(sm_ptr + b * 3 + 0)) * tl.load(sc_ptr + b * 3 + 0) + tl.load(tm_ptr + b * 3 + 0)
            A = (500.0 * (fx - fy) - tl.load(sm_ptr + b * 3 + 1)) * tl.load(sc_ptr + b * 3 + 1) + tl.load(tm_ptr + b * 3 + 1)
            B_ = (200.0 * (fy - fz) - tl.load(sm_ptr + b * 3 + 2)) * tl.load(sc_ptr + b * 3 + 2) + tl.load(tm_ptr + b * 3 + 2)
            gy = (L + 16.0) / 116.0
            x_ = 0.950456 * _finv_k(gy + A / 500.0)
            y_ = _finv_k(gy)
            z_ = 1.088754 * _finv_k(gy - B_ / 200.0)
            sr = _lin_to_srgb_k(3.240479 * x_ - 1.53715 * y_ - 0.498535 * z_)
            sg = _lin_to_srgb_k(-0.969256 * x_ + 1.875991 * y_ + 0.041556 * z_)
            sb_ = _lin_to_srgb_k(0.055648 * x_ - 0.204043 * y_ + 1.057311 * z_)

        g = _band(BLOCK, R, SIGMA, NORM)
        lo_r = _blur_k(sr, g)
        lo_g = _blur_k(sg, g)
        lo_b = _blur_k(sb_, g)

        r0 = tl.load(rest_ptr + sbase + off).to(tl.float32)
        r1 = tl.load(rest_ptr + sbase + plane + off).to(tl.float32)
        r2 = tl.load(rest_ptr + sbase + 2 * plane + off).to(tl.float32)
        if BGR:
            rr, rg, rb = r2, r1, r0
        else:
            rr, rg, rb = r0, r1, r2
        fr = tl.minimum(tl.maximum(lo_r + w_detail * (rr - _blur_k(rr, g)), 0.0), 1.0)
        fg = tl.minimum(tl.maximum(lo_g + w_detail * (rg - _blur_k(rg, g)), 0.0), 1.0)
        fb = tl.minimum(tl.maximum(lo_b + w_detail * (rb - _blur_k(rb, g)), 0.0), 1.0)

        m = tl.load(mask_ptr + mbase + off).to(tl.float32)
        alpha = tl.minimum(tl.maximum(_blur_k(m, _band(BLOCK, RM, SIGMA_M, NORM_M)), 0.0), 1.0)

        gi = tl.arange(0, BLOCK)
        inner = ((gi[:, None] >= HALO) & (gi[:, None] < BLOCK - HALO) & (gi[None, :] >= HALO) & (gi[None, :] < BLOCK - HALO)
                 & (ys[:, None] >= 0) & (ys[:, None] < H) & (xs[None, :] >= 0) & (xs[None, :] < W))
        if DO_GRAIN:                                                       # luminance grain: one number added to all channels
            n = tl.randn(seed + b, (ys[:, None] * W + xs[None, :]).to(tl.int32)) * (tl.load(gs_ptr + b) / 255.0)
            fr += n
            fg += n
            fb += n

        if BGR:
            f0, f1, f2 = fb, fg, fr
        else:
            f0, f1, f2 = fr, fg, fb
        for c in tl.static_range(3):
            if c == 0:
                fc = f0
            elif c == 1:
                fc = f1
            else:
                fc = f2
            tp = tgt_ptr + b * t_sb + c * t_sc + ys[:, None] * t_sy + xs[None, :] * t_sx
            t = tl.load(tp, mask=inner, other=0.0).to(tl.float32)
            if T_U8:
                t = t / 255.0
            o = t + alpha * (fc - t)
            op = out_ptr + b * o_sb + c * o_sc + ys[:, None] * o_sy + xs[None, :] * o_sx
            if O_U8:
                tl.store(op, (tl.minimum(tl.maximum(o, 0.0), 1.0) * 255.0 + 0.5).to(tl.uint8), mask=inner)
            else:
                tl.store(op, o.to(out_ptr.dtype.element_ty), mask=inner)


def fused_postprocess(swapped: torch.Tensor, restored: torch.Tensor, xseg_mask: torch.Tensor, target_roi: torch.Tensor,
                      *, lab=None, detail_weight: float = 0.75, blur_sigma: float = 2.0, feather_sigma: float = 2.0,
                      grain_sigma: Optional[torch.Tensor] = None, seed: int = 0, bgr: bool = False,
                      out: Optional[torch.Tensor] = None, block: int = 32, num_warps: Optional[int] = None) -> torch.Tensor:
    """Run the fused kernel. ``swapped`` / ``restored`` ``[B, 3, H, W]`` and ``xseg_mask`` ``[B, 1, H, W]`` are contiguous
    float16 / float32 in [0, 1]; ``target_roi`` is ``[B, 3, H, W]`` (any strides; uint8 in [0, 255] or float in [0, 1]).
    ``out`` (default: a new tensor like ``target_roi``) may be a strided VIEW of the destination frame, e.g.
    ``frame[:, :, y0:y1, x0:x1]``, so the composite lands in the frame with no copy. ``lab`` = ``(src_mean, scale, tgt_mean)``
    ``[B, 3]`` float32 (``reinhard_params``); ``grain_sigma`` ``[B]`` in 8-bit levels (``grain_deficit``).
    """
    if not HAVE_TRITON:
        raise RuntimeError("triton is not installed (pip install triton-windows on Windows, triton on Linux)")
    for name, t in (("swapped", swapped), ("restored", restored), ("xseg_mask", xseg_mask)):
        if not (t.is_cuda and t.is_contiguous() and t.dtype in (torch.float16, torch.float32)):
            raise ValueError(f"{name} must be a contiguous float16/float32 CUDA tensor")
    B, C, H, W = swapped.shape
    if C != 3 or restored.shape != swapped.shape or tuple(xseg_mask.shape) != (B, 1, H, W) or tuple(target_roi.shape) != (B, 3, H, W):
        raise ValueError("expected swapped/restored/target_roi [B,3,H,W] and xseg_mask [B,1,H,W] in one pixel space")
    if target_roi.dtype not in (torch.uint8, torch.float16, torch.float32):
        raise ValueError("target_roi must be uint8, float16 or float32")
    if out is None:
        out = torch.empty_like(target_roi)
    if tuple(out.shape) != (B, 3, H, W) or out.dtype != target_roi.dtype:
        raise ValueError("out must match target_roi's shape and dtype")
    r, norm = gaussian_taps(blur_sigma)
    rm, norm_m = gaussian_taps(feather_sigma)
    halo = max(r, rm)
    if halo > (block - 16) // 2:
        raise ValueError(f"sigma too large for a {block}px tile (halo {halo}, at most {(block - 16) // 2}); use a smaller sigma")
    dev = swapped.device
    zeros = torch.zeros((B, 3), dtype=torch.float32, device=dev)
    sm, sc, tm = (zeros, zeros, zeros) if lab is None else tuple(t.float().contiguous().to(dev) for t in lab)
    gs = zeros[:, 0] if grain_sigma is None else grain_sigma.float().contiguous().to(dev)
    tile = block - 2 * halo
    if num_warps is None:
        num_warps = 4 if halo <= 4 else 8            # measured: halo 6 at 4 warps spills (8.97 ms vs 1.23 ms at B=4, 512x512)
    grid = (B, triton.cdiv(H, tile), triton.cdiv(W, tile))
    _fused_post_kernel[grid](
        swapped, restored, xseg_mask, target_roi, out, sm, sc, tm, gs, H, W,
        *target_roi.stride(), *out.stride(), float(detail_weight), int(seed),
        BLOCK=block, R=r, SIGMA=float(blur_sigma), NORM=norm, RM=rm, SIGMA_M=float(feather_sigma), NORM_M=norm_m, HALO=halo,
        DO_CT=lab is not None, DO_GRAIN=grain_sigma is not None, T_U8=target_roi.dtype == torch.uint8,
        O_U8=out.dtype == torch.uint8, BGR=bool(bgr), num_warps=num_warps)
    return out
