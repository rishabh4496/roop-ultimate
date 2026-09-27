"""Stage 3 benchmark: single-face sequential vs batched (B=4) throughput.

Not collected by pytest. Usage::

    python -m face_engine.tests.bench_stage3 [--iters 30] [--providers trt|cuda]
                                              [--precision fp32,fp16] [--stages swap,...]

Both arms process the SAME four faces of ``insightface``'s ``t1.jpg`` with
the same model, precision and providers:

* sequential: four calls, one face each, on each model's ORIGINAL graph
  (``batching=False``: static batch-1 engines, the best single-face setup);
* batched: one call with all four faces, on the verified dynamic-batch graph
  (engines built for a 1..8 profile, optimised for 4).

Running single faces through the batch-profile engines instead would be a
handicapped baseline: LivePortrait read 88 ms/face that way vs 35 ms/face on
its static engines (2026-09-28).

Arms are counterbalanced (sequential, batched, batched, sequential) after a
warm-up that absorbs TensorRT engine builds, because the first arm otherwise
pays for the build (AGENTS.md: two neutral results read +21.8% and +9.8%
without counterbalancing). Each timed call ends with ``torch.cuda.synchronize``.
"""
from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import torch

from face_engine.core.config import EngineConfig, Provider
from face_engine.core.execution import ExecutionEngine
from face_engine.models.zoo import build_default_registry
from face_engine.pipeline.aligner import warp_face_inverse_cuda
from face_engine.pipeline.detector import SCRFDDetector
from face_engine.processors import (
    BatchedExpressionRestorer,
    BatchedFaceEnhancer,
    BatchedFaceSwapper,
    GPUIdentityEncoder,
)

B = 4


def timed(fn: Callable[[], object], iters: int) -> list[float]:
    out = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1000)
    return out


def granted(*objects: object) -> str:
    """Primary provider of each object's session (catches a silent CUDA fallback)."""
    out = []
    for obj in objects:
        try:
            out.append(obj.session.primary_provider.replace("ExecutionProvider", ""))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            out.append("?")
    return "/".join(out)


def release(engine: ExecutionEngine) -> None:
    """Drop every cached session so the next stage builds on an empty card."""
    engine.release()
    torch.cuda.empty_cache()


def compare(label: str, sequential: Callable[[], object], batched: Callable[[], object],
            iters: int) -> dict[str, float]:
    for _ in range(3):  # warm-up: engine builds, allocator, cudnn plans
        sequential()
        batched()
    runs: dict[str, list[float]] = {"seq": [], "bat": []}
    for arm in ("seq", "bat", "bat", "seq"):
        runs[arm] += timed(sequential if arm == "seq" else batched, iters)
    seq, bat = statistics.median(runs["seq"]), statistics.median(runs["bat"])
    row = {"seq_ms": seq, "bat_ms": bat, "seq_fps": B * 1000 / seq, "bat_fps": B * 1000 / bat,
           "speedup": seq / bat}
    print(f"{label:44s} sequential {seq:8.2f} ms ({row['seq_fps']:7.1f} faces/s)   "
          f"batched {bat:8.2f} ms ({row['bat_fps']:7.1f} faces/s)   x{row['speedup']:.2f}",
          flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--providers", default="trt", choices=("trt", "cuda"))
    ap.add_argument("--precision", default="fp32,fp16")
    ap.add_argument("--stages", default="swap,boost,encoder,enhance,expression,chain")
    ap.add_argument("--enhancers", default="gpen_bfr_512,restoreformer_plus_plus")
    args = ap.parse_args()
    stages = set(args.stages.split(","))

    import insightface

    registry = build_default_registry()
    providers = ([Provider.TENSORRT, Provider.CUDA, Provider.CPU] if args.providers == "trt"
                 else [Provider.CUDA, Provider.CPU])
    engine = ExecutionEngine(EngineConfig(providers=providers))
    image = cv2.imread(str(Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"))
    detector = SCRFDDetector(ExecutionEngine(EngineConfig(providers=[Provider.CUDA])),
                             registry.ensure("scrfd_10g_bnkps", show_progress=False))
    frame = torch.from_numpy(image).cuda().permute(2, 0, 1)[None].float()
    kps = detector.detect_cuda(frame).kps[:B]
    arc = registry.ensure("arcface_w600k_r50", show_progress=False)
    encoder = GPUIdentityEncoder(engine, arc)
    encoder1 = GPUIdentityEncoder(engine, arc, batching=False)
    source = encoder.embed(frame, detector.detect_cuda(frame).kps[B:B + 1])
    print(f"GPU {torch.cuda.get_device_name()}  providers={args.providers}  B={B}  "
          f"iters={args.iters} x2 per arm", flush=True)

    if "encoder" in stages:
        compare("arcface encoder",
                lambda: [encoder1.embed(frame, kps[i:i + 1]) for i in range(B)],
                lambda: encoder.embed(frame, kps), args.iters)
        release(engine)

    hs = registry.ensure("hyperswap_1a_256", show_progress=False)
    for precision in args.precision.split(","):
        swapper = BatchedFaceSwapper(engine, "hyperswap_1a_256", hs, precision=precision)
        single = BatchedFaceSwapper(engine, "hyperswap_1a_256", hs, precision=precision,
                                    batching=False)
        swapper.set_source(source[0])
        single.set_source(source[0])
        print(f"  hyperswap_1a {precision}: batched graph = {swapper.batched}, "
              f"providers {granted(single, swapper)}", flush=True)
        if "swap" in stages:
            compare(f"hyperswap_1a {precision} (256)",
                    lambda one=single: [one.swap(frame, kps[i:i + 1]) for i in range(B)],
                    lambda many=swapper: many.swap(frame, kps), args.iters)
        if "boost" in stages:
            # Pixel Boost 512 = 4 tiles per face: sequential is 4 x (1 face, 4 tiles).
            compare(f"hyperswap_1a {precision} pixel boost 512",
                    lambda one=single: [one.swap(frame, kps[i:i + 1], pixel_boost=512)
                                        for i in range(B)],
                    lambda many=swapper: many.swap(frame, kps, pixel_boost=512), args.iters)
        del swapper, single
        release(engine)
        if "enhance" in stages:
            for name in args.enhancers.split(","):
                path = registry.ensure(name, show_progress=False)
                enh = BatchedFaceEnhancer(engine, name, path, precision=precision)
                enh1 = BatchedFaceEnhancer(engine, name, path, precision=precision,
                                           batching=False)
                compare(f"{name} {precision} (batched={enh.batched}, {granted(enh1, enh)})",
                        lambda one=enh1: [one.enhance(frame, kps[i:i + 1]) for i in range(B)],
                        lambda many=enh: many.enhance(frame, kps), args.iters)
                release(engine)

    if "expression" in stages:
        restorer = BatchedExpressionRestorer.from_registry(engine, registry)
        restorer1 = BatchedExpressionRestorer.from_registry(engine, registry, batching=False)
        swapper = BatchedFaceSwapper(engine, "hyperswap_1a_256",
                                     registry.ensure("hyperswap_1a_256", show_progress=False))
        swapper.set_source(source[0])
        res = swapper.swap(frame, kps, pixel_boost=512)
        compare("liveportrait expression (512 crop)",
                lambda: [restorer1.restore(res.crops[i:i + 1], res.target_crops[i:i + 1])
                         for i in range(B)],
                lambda: restorer.restore(res.crops, res.target_crops), args.iters)
        # The restorer returns its input when a model fails; a failed arm is fast.
        if restorer.failures or restorer1.failures:
            raise SystemExit(f"expression restore FAILED {restorer.failures + restorer1.failures} times: "
                             "the row above measured the fallback, not the model")
        del restorer, restorer1, swapper
        release(engine)

    if "chain" in stages:
        swapper = BatchedFaceSwapper(engine, "hyperswap_1a_256",
                                     registry.ensure("hyperswap_1a_256", show_progress=False))
        swapper.set_source(source[0])
        single = BatchedFaceSwapper(engine, "hyperswap_1a_256", hs, batching=False)
        single.set_source(source[0])
        gpen = registry.ensure("gpen_bfr_512", show_progress=False)
        enh = BatchedFaceEnhancer(engine, "gpen_bfr_512", gpen)
        enh1 = BatchedFaceEnhancer(engine, "gpen_bfr_512", gpen, batching=False)

        def chain(sw: BatchedFaceSwapper, en: BatchedFaceEnhancer, k: torch.Tensor) -> torch.Tensor:
            r = sw.swap(frame, k, pixel_boost=512)
            pasted = warp_face_inverse_cuda(frame, r.crops, r.matrices, r.model_mask)
            return en.enhance(pasted, k, reference=frame).frames

        compare("chain: swap 512 -> paste -> gpen_512 (defaults)",
                lambda: [chain(single, enh1, kps[i:i + 1]) for i in range(B)],
                lambda: chain(swapper, enh, kps), args.iters)


if __name__ == "__main__":
    main()
