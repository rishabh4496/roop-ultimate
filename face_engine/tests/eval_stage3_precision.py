"""Measure FP16 vs FP32 for the batched swap / restore models on real faces.

Not collected by pytest. Usage::

    python -m face_engine.tests.eval_stage3_precision CLIP [CLIP ...] [--frames 8]

Targets: faces detected on ``--frames`` evenly spaced frames of each clip.
Sources: the six faces of insightface's ``t1.jpg``. Every (source, target)
pair is swapped with the batched swapper (TensorRT; FP32 and FP16 engines),
pasted, re-detected at the swapped spot and re-embedded; identity is the
cosine to the source, leakage the cosine to the original target. Restorers
are scored by how much of each target face's identity survives restoration,
and by their output's PSNR against the FP32 output.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from face_engine.core.config import EngineConfig, Provider
from face_engine.core.execution import ExecutionEngine
from face_engine.models.zoo import build_default_registry
from face_engine.pipeline.aligner import warp_face_inverse_cuda
from face_engine.pipeline.detector import SCRFDDetector
from face_engine.processors import (
    BatchedFaceEnhancer,
    BatchedFaceSwapper,
    GPUIdentityEncoder,
)


def frames_of(path: str, count: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in np.linspace(0, max(total - 1, 0), count).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            out.append(frame)
    cap.release()
    return out


def main() -> None:
    import insightface

    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--swappers", default="hyperswap_1a_256,inswapper_128")
    ap.add_argument("--enhancers", default="gpen_bfr_512,restoreformer_plus_plus")
    args = ap.parse_args()

    reg = build_default_registry()
    cuda = ExecutionEngine(EngineConfig(providers=[Provider.CUDA, Provider.CPU]))
    trt = ExecutionEngine(EngineConfig(providers=[Provider.TENSORRT, Provider.CUDA, Provider.CPU]))
    det = SCRFDDetector(cuda, reg.ensure("scrfd_10g_bnkps", show_progress=False))
    enc = GPUIdentityEncoder(cuda, reg.ensure("arcface_w600k_r50", show_progress=False))
    t1 = cv2.imread(str(Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"))
    t1_t = torch.from_numpy(t1).cuda().permute(2, 0, 1)[None].float()
    sources = enc.embed(t1_t, det.detect_cuda(t1_t).kps)

    targets = []  # (frame tensor, kps (1,5,2), box, target embedding)
    for clip in args.clips:
        for frame in frames_of(clip, args.frames):
            f = torch.from_numpy(frame).cuda().permute(2, 0, 1)[None].float()
            d = det.detect_cuda(f)
            keep = (d.scores > 0.6) & ((d.boxes[:, 2] - d.boxes[:, 0]) > 60)
            for i in torch.nonzero(keep)[:, 0].tolist():
                targets.append((f, d.kps[i:i + 1], d.boxes[i], enc.embed(f, d.kps[i:i + 1])[0]))
    print(f"{len(targets)} target faces x {sources.shape[0]} sources", flush=True)

    def at_spot(frame: torch.Tensor, box: torch.Tensor) -> torch.Tensor | None:
        d = det.detect_cuda(frame)
        if len(d) == 0:
            return None
        j = int((d.boxes - box).abs().sum(1).argmin())
        return enc.embed(frame, d.kps[j:j + 1])[0]

    for name in args.swappers.split(","):
        base = None
        for precision in ("fp32", "fp16"):
            sw = BatchedFaceSwapper(trt, name, reg.ensure(name, show_progress=False),
                                    precision=precision)
            ident, leak, crops, bad = [], [], [], 0
            for f, kps, box, target_emb in targets:
                src = sources
                res = sw.swap(f, kps.expand(src.shape[0], -1, -1).contiguous(), source=src)
                bad += int((~res.ok).sum())
                crops.append(res.crops)
                for s in range(src.shape[0]):
                    pasted = warp_face_inverse_cuda(f, res.crops[s:s + 1], res.matrices[s:s + 1],
                                                    res.model_mask[s:s + 1]
                                                    if res.model_mask is not None else None)
                    e = at_spot(pasted, box)
                    if e is None:
                        continue
                    ident.append(float(e @ src[s]))
                    leak.append(float(e @ target_emb))
            crops_t = torch.cat(crops)
            drift = "" if base is None else \
                f"  crop |fp16-fp32| mean {float((crops_t - base).abs().mean()):.2f} lv"
            base = crops_t if base is None else base
            ident_a = np.array(ident)
            print(f"{name:18s} {precision}: identity {ident_a.mean():.4f} (p5 {np.percentile(ident_a, 5):.3f})"
                  f"  leakage {np.mean(leak):.4f}  non-finite faces {bad}  n={len(ident)}{drift}",
                  flush=True)

    for name in [n for n in args.enhancers.split(",") if n]:
        base = None
        for precision in ("fp32", "fp16"):
            en = BatchedFaceEnhancer(trt, name, reg.ensure(name, show_progress=False),
                                     precision=precision)
            kept, outs, rejected = [], [], 0
            for f, kps, box, target_emb in targets:
                r = en.enhance(f, kps)
                rejected += int((~r.ok).sum())
                outs.append(r.crops_out)
                e = at_spot(r.frames, box)
                if e is not None:
                    kept.append(float(e @ target_emb))
            o = torch.cat(outs)
            psnr = ""
            if base is not None:
                mse = float(((o - base) ** 2).mean())
                psnr = f"  PSNR vs fp32 {10 * np.log10(255 ** 2 / max(mse, 1e-9)):.2f} dB"
            base = o if base is None else base
            print(f"{name:24s} {precision}: identity kept {np.mean(kept):.4f} "
                  f"(p5 {np.percentile(kept, 5):.3f})  rejected {rejected}/{len(targets)}{psnr}",
                  flush=True)


if __name__ == "__main__":
    main()
