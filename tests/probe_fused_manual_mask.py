#!/usr/bin/env python3
"""What do the keep-original matte and the low-band mix cost the fused kernel, and what would the in-kernel inverse warp save?

The 2026-10-25 re-issue of the fused-kernel brief asks for (a) manual paint/erase brush strokes and mouth/eye overrides in the
mask epilogue and (b) a bicubic inverse affine warp INSIDE the kernel. (a) is `keep_mask` / `low_weight` on `fused_postprocess`.
(b) is priced here instead of built: the kernel is handed ROI-space tensors, so the warp is a separate resampling step
(`CudaAffineBatch.paste_faces_roi`); fusing it would only save that step's launch and one ROI-sized intermediate.

  kernel arms   baseline x2 (the null control), + keep_mask, + low_weight=0.8 / detail 1.0, + both
  halo arm      the same sampler at 1.78x the output pixels (the tile halo an in-kernel warp must also resample)
  warp arm      one bicubic `grid_sample` of the 8 channels the kernel reads (swapped 3 + restored 3 + xseg 1 + keep 1) from a
                512 px crop into an ROI of the same size, i.e. the work an in-kernel warp would absorb, timed by itself

Numbers: docs/perf/fused_postprocess/. TF32 off (not through core.py). Not a render measurement.

    app\\env\\Scripts\\python.exe tests\\probe_fused_manual_mask.py [--batch 4] [--size 512] [--out report.json]
"""
import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from app.core import fused_postprocess as fp  # noqa: E402


def med_ms(fn, iters=100, warm=15):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(ts), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if not (torch.cuda.is_available() and fp.HAVE_TRITON):
        print("needs CUDA and triton")
        return 2
    B, S, dev = a.batch, a.size, "cuda"
    g = torch.Generator(device="cpu").manual_seed(0)
    r = lambda *s: torch.rand(*s, generator=g).to(dev)
    sw = (0.25 + 0.5 * r(B, 3, S, S)).contiguous()
    rs = (sw + 0.1 * (r(B, 3, S, S) - 0.5)).clamp(0, 1).contiguous()
    m = torch.zeros(B, 1, S, S, device=dev)
    m[:, :, S // 6:5 * S // 6, S // 6:5 * S // 6] = 1.0
    keep = torch.zeros_like(m)
    keep[:, :, S // 4:S // 2, S // 3:2 * S // 3] = 1.0
    tgt = (r(B, 3, S, S) * 255).to(torch.uint8)
    sm, ss = fp.lab_stats(sw)
    tm, ts = fp.lab_stats(tgt.float() / 255.0)
    lab = fp.reinhard_params(sm, ss, tm, ts)
    gs = torch.full((B,), 3.0, device=dev)
    kw = dict(lab=lab, grain_sigma=gs)
    out = {"device": torch.cuda.get_device_name(0), "batch": B, "roi": S, "kernel_ms": {}}
    arms = {
        "baseline_a": dict(),
        "baseline_b_null": dict(),
        "keep_mask": dict(keep_mask=keep),
        "low0.8_detail1.0": dict(low_weight=0.8, detail_weight=1.0),
        "keep_and_low": dict(keep_mask=keep, low_weight=0.8, detail_weight=1.0),
    }
    for name, extra in arms.items():
        out["kernel_ms"][name] = med_ms(lambda e=extra: fp.fused_postprocess(sw, rs, m, tgt, **kw, **e))
    # in-kernel inverse warp: the resampling it would absorb (a rotated + scaled bicubic sample of the kernel's 8 inputs)
    stack = torch.cat((sw, rs, m, keep), dim=1).contiguous()
    theta = torch.tensor([[0.93, -0.25, 0.02], [0.25, 0.93, -0.01]], device=dev).expand(B, -1, -1).contiguous()
    grid = F.affine_grid(theta, (B, 8, S, S), align_corners=False)
    out["warp_ms"] = med_ms(lambda: F.grid_sample(stack, grid, mode="bicubic", padding_mode="zeros", align_corners=False))
    out["warp_ms_bilinear_for_reference"] = med_ms(lambda: F.grid_sample(stack, grid, mode="bilinear", padding_mode="zeros", align_corners=False))
    # An in-kernel warp must resample each tile's blur HALO too (the Gaussians read neighbours): block 32, halo 4 -> tile 24, so
    # (32/24)^2 = 1.78x the pixels. Time the same sampler at that output size: a lower bound on the fused warp's compute
    # (it ignores register pressure and the gather's loss of coalescing), set against the ONE thing fusing saves.
    sh = int(round(S * 32 / 24))
    grid_h = F.affine_grid(theta, (B, 8, sh, sh), align_corners=False)
    out["warp_with_halo_recompute_ms"] = med_ms(lambda: F.grid_sample(stack, grid_h, mode="bicubic", padding_mode="zeros", align_corners=False))
    out["intermediate_traffic_mb"] = round(2 * stack.numel() * 4 / 2 ** 20, 1)      # written by the warp, read by the kernel
    base = statistics.mean((out["kernel_ms"]["baseline_a"], out["kernel_ms"]["baseline_b_null"]))
    out["null_spread_ms"] = round(abs(out["kernel_ms"]["baseline_a"] - out["kernel_ms"]["baseline_b_null"]), 4)
    out["keep_overhead_pct_of_baseline"] = round((out["kernel_ms"]["keep_mask"] / base - 1) * 100, 1)
    out["low_overhead_pct_of_baseline"] = round((out["kernel_ms"]["low0.8_detail1.0"] / base - 1) * 100, 1)
    out["warp_as_pct_of_kernel_plus_warp"] = round(out["warp_ms"] / (out["warp_ms"] + base) * 100, 1)
    text = json.dumps(out, indent=2)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
