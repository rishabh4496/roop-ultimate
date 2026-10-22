"""The fused Stage 3 post-processing kernel against what it replaces.

    app\\env\\Scripts\\python.exe tests\\probe_fused_postprocess.py [--out report.json]

Arms, per (faces B, crop size S), times in ms for the WHOLE batch:

  fused          app.core.fused_postprocess.fused_postprocess        one Triton launch, device tensors in and out
  unfused torch  reference_postprocess                               the same maths as ~100 ordinary torch ops
  fused + host   upload the uint8 target ROI, fused, download        what a caller with host arrays (ORT returns numpy) pays
  cpu            the production host chain, per face: _color_transfer_rct (cv2 LAB), enhance_blend.frequency_blend (uint8
                 cv2), grain_match.match, a cv2-feathered numpy composite - one thread, as one worker's share

Also reported: how many CUDA kernels the unfused arm launches (torch.profiler), and the largest difference between fused and
unfused in 8-bit levels. The CPU arm is the real production functions but NOT an exact twin (8-bit LAB, no per-pixel grain
RNG match), so it prices the step; it does not prove pixel equivalence.
"""
import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "app"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

cv2.setNumThreads(1)
from app.core import fused_postprocess as fp  # noqa: E402


def gpu_time(fn, reps=30, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b))
    return statistics.median(out)


def cpu_chain(sw, rs, m, tgt):
    """sw, rs, tgt: uint8 HxWx3 BGR; m: float HxW. One face."""
    from roop.procmgr_color import ColorTransferMixin
    from roop import enhance_blend, grain_match
    mix = ColorTransferMixin()
    ct = mix._color_transfer_rct(sw, tgt)
    face = enhance_blend.frequency_blend(ct, rs, weight=0.75)
    face, _ = grain_match.match(face, tgt, m, strength=1.0)
    a = cv2.GaussianBlur(m, (0, 0), 2.0)[:, :, None]
    return np.clip(a * face.astype(np.float32) + (1 - a) * tgt.astype(np.float32), 0, 255).astype(np.uint8)


def cpu_time(fn, reps=15):
    fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not (torch.cuda.is_available() and fp.HAVE_TRITON):
        print("[PENDING] needs CUDA and triton")
        return 2
    rows = []
    for S in (256, 512):
        for B in (1, 4):
            g = torch.Generator(device="cpu").manual_seed(S + B)
            sw = (0.25 + 0.5 * torch.rand(B, 3, S, S, generator=g)).cuda().contiguous()
            rs = (sw + 0.1 * (torch.rand(B, 3, S, S, generator=g).cuda() - 0.5)).clamp(0, 1).contiguous()
            m = torch.zeros(B, 1, S, S, device="cuda")
            m[:, :, S // 8:7 * S // 8, S // 8:7 * S // 8] = 1.0
            tgt = (torch.rand(B, 3, S, S, generator=g) * 255).to(torch.uint8).cuda()
            sm, ss = fp.lab_stats(sw)
            tm, ts = fp.lab_stats(tgt.float() / 255.0)
            lab = fp.reinhard_params(sm, ss, tm, ts)
            sigma = 2.0 * S / 512.0
            kw = dict(blur_sigma=sigma, feather_sigma=sigma)
            out = torch.empty_like(tgt)

            fused = lambda: fp.fused_postprocess(sw, rs, m, tgt, lab=lab, out=out, **kw)
            unfused = lambda: fp.reference_postprocess(sw, rs, m, tgt, lab=lab, **kw)
            tgt_host = tgt.cpu().numpy()

            def with_host():
                t = torch.from_numpy(tgt_host).cuda(non_blocking=True)
                o = fp.fused_postprocess(sw, rs, m, t, lab=lab, **kw)
                return o.cpu()

            diff = float((fused().float() - (unfused().clamp(0, 1) * 255.0).round()).abs().max())
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                unfused()
                torch.cuda.synchronize()
            kernels = sum(e.count for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA)
            sw_np = (sw[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            rs_np = (rs[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            tg_np = tgt[0].permute(1, 2, 0).cpu().numpy().copy()
            m_np = m[0, 0].cpu().numpy()
            cpu_one = cpu_time(lambda: cpu_chain(sw_np, rs_np, m_np, tg_np))
            row = dict(faces=B, size=S, fused_ms=gpu_time(fused), unfused_torch_ms=gpu_time(unfused),
                       fused_with_host_ms=cpu_time(with_host, 30), cpu_ms_all_faces=cpu_one * B, cpu_ms_per_face=cpu_one,
                       unfused_cuda_kernels=int(kernels), max_diff_levels=diff)
            rows.append(row)
            print(f"B={B} S={S:3d}  fused {row['fused_ms']:6.3f} ms | unfused torch {row['unfused_torch_ms']:7.3f} ms "
                  f"({row['unfused_cuda_kernels']} kernels) | fused+host {row['fused_with_host_ms']:6.2f} ms | "
                  f"cpu {row['cpu_ms_all_faces']:6.1f} ms ({cpu_one:.1f}/face) | max diff {diff:.1f} levels", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(dict(rows=rows, device=torch.cuda.get_device_name(0)), fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
