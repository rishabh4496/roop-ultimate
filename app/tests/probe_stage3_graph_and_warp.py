#!/usr/bin/env python3
"""Two claims of the "VRAM-resident Stage 3" brief, priced on the real engine: CUDA-graph replay of a TensorRT swapper
engine, and the batched GPU bicubic warp against the CPU `cv2.warpAffine` loop.

  graph arm  Deserialises a compiled dynamic-batch swapper engine (default `hyperswap_1a_256_sm89_fp16_b8.engine` from
             `.cache/trt_engines`), binds static buffers per batch bucket, and times `execute_async_v3` against
             `torch.cuda.CUDAGraph.replay()` (capture on a side stream, `thread_local` mode as `face_engine/pipeline/
             tracker.py` does). Also checks replay == eager bit for bit, and reports ms per face by bucket.
  warp arm   `CudaAffineBatch.warp_frames` (bicubic `grid_sample`) and `kornia.geometry.transform.warp_affine` against a
             one-thread `cv2.warpAffine(INTER_CUBIC, BORDER_REPLICATE)` loop on 1080p frames, with pixel error, and the
             cost of getting HOST frames onto the device (today's case; NvZeroCopyReader frames are already there).

It measures the KERNELS, not the render clock. The pipeline is GPU-bound (`AGENTS.md`), and nothing here is wired into
a render: the production swapper is an ORT session, not a raw TensorRT context. Numbers: `docs/perf/stage3_resident_graph/`.

Run outside the app: TF32 is NOT enabled here, `roop/core.py` enables it at import and `CudaAffineBatch` documents ~5x
worse resampling error under it.

    app/env/Scripts/python.exe app/tests/probe_stage3_graph_and_warp.py [--engine PATH] [--buckets 1,2,4,8] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))

import cv2                                                          # noqa: E402
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

cv2.setNumThreads(1)
DEFAULT_ENGINE = os.path.join(REPO, ".cache", "trt_engines", "hyperswap_1a_256_sm89_fp16_b8.engine")


def _sync_times(fn, iters: int, warm: int) -> list:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def _pipelined(fn, iters: int = 200, warm: int = 20) -> float:
    """Back-to-back launches with one sync: separates launch overhead from GPU time."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def graph_arm(engine_path: str, buckets: list) -> list:
    import tensorrt as trt
    dev = torch.device("cuda:0")
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    with open(engine_path, "rb") as handle:
        engine = runtime.deserialize_cuda_engine(handle.read())
    if engine is None:
        raise RuntimeError(f"could not deserialize {engine_path}")
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    outputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    tail = {n: tuple(engine.get_tensor_shape(n))[1:] for n in names}
    max_batch = max(engine.get_tensor_profile_shape(inputs[0], 0)[2][0], 1)
    rows = []
    for bucket in buckets:
        row = {"batch": bucket}
        if bucket > max_batch:
            row["skipped"] = f"engine optimisation profile tops out at batch {max_batch}"
            rows.append(row)
            continue
        context = engine.create_execution_context()
        stream = torch.cuda.Stream(device=dev)
        bufs = {n: (torch.randn((bucket,) + tail[n], device=dev) * 0.3) for n in names}
        with torch.cuda.stream(stream):
            for n in inputs:
                context.set_input_shape(n, (bucket,) + tail[n])
            for n in names:
                context.set_tensor_address(n, bufs[n].data_ptr())
            eager = lambda: context.execute_async_v3(stream.cuda_stream)
            for _ in range(3):
                eager()
        stream.synchronize()
        with torch.cuda.stream(stream):
            row["eager_ms"] = round(statistics.median(_sync_times(eager, 200, 20)), 3)
            row["eager_pipelined_ms"] = round(_pipelined(eager), 3)
            eager()
        stream.synchronize()
        reference = {n: bufs[n].clone() for n in outputs}
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
                if not context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError("execute_async_v3 returned False during capture")
        except Exception as exc:                                    # record the failure, it is a result
            row["capture_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            continue
        for n in outputs:
            bufs[n].zero_()
        graph.replay()
        torch.cuda.synchronize()
        row["max_abs_replay_vs_eager"] = max(float((bufs[n] - reference[n]).abs().max()) for n in outputs)
        row["graph_ms"] = round(statistics.median(_sync_times(graph.replay, 200, 20)), 3)
        row["graph_pipelined_ms"] = round(_pipelined(graph.replay), 3)
        row["saving_ms"] = round(row["eager_pipelined_ms"] - row["graph_pipelined_ms"], 3)
        row["eager_ms_per_face"] = round(row["eager_pipelined_ms"] / bucket, 3)
        rows.append(row)
        del graph, context
    return rows


def _frame(rng, h: int, w: int) -> np.ndarray:
    base = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    return cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)


def _matrix(rng, h: int, w: int, size: int) -> np.ndarray:
    scale = 0.55 + 0.1 * rng.random()
    angle = np.deg2rad(rng.uniform(-25, 25))
    cx, cy = rng.uniform(300, w - 300), rng.uniform(250, h - 250)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) * scale
    trans = np.array([size / 2, size / 2]) - rot @ np.array([cx, cy])
    return np.hstack([rot, trans[:, None]]).astype(np.float32)


def warp_arm(buckets: list, size: int = 256, height: int = 1080, width: int = 1920) -> list:
    import kornia
    from roop.optimized_processor import CudaAffineBatch
    dev = torch.device("cuda:0")
    rng = np.random.default_rng(0)
    rows = []
    for bucket in buckets:
        frames = [_frame(rng, height, width) for _ in range(bucket)]
        mats = np.stack([_matrix(rng, height, width, size) for _ in range(bucket)])

        def cpu():
            return [cv2.warpAffine(f, m, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    for f, m in zip(frames, mats)]

        cpu_out = cpu()
        for _ in range(8):
            cpu()
        cpu_ms = statistics.median([_timed(cpu) for _ in range(40)])
        resident = torch.from_numpy(np.stack(frames)).to(dev).permute(0, 3, 1, 2).contiguous()
        g_mats = torch.from_numpy(mats).to(dev)
        reference = torch.from_numpy(np.stack(cpu_out)).to(dev).permute(0, 3, 1, 2).float()

        def affine():
            return CudaAffineBatch.warp_frames(resident.float(), g_mats, size)

        def kor():
            return kornia.geometry.transform.warp_affine(resident.float(), g_mats, (size, size), mode="bicubic",
                                                         padding_mode="border", align_corners=False)

        def with_upload():
            t = torch.from_numpy(np.stack(frames)).to(dev).permute(0, 3, 1, 2).float()
            return CudaAffineBatch.warp_frames(t, g_mats, size)

        d_aff = (affine() - reference).abs()
        d_kor = (kor() - reference).abs()
        rows.append({
            "batch": bucket,
            "cv2_loop_ms": round(cpu_ms, 3),
            "cuda_affine_resident_ms": round(statistics.median(_sync_times(affine, 60, 8)), 3),
            "cuda_affine_with_host_upload_ms": round(statistics.median(_sync_times(with_upload, 20, 4)), 3),
            "kornia_resident_ms": round(statistics.median(_sync_times(kor, 60, 8)), 3),
            "cuda_affine_vs_cv2_mean_abs_255": round(float(d_aff.mean()), 4),
            "cuda_affine_vs_cv2_max_abs_255": round(float(d_aff.max()), 2),
            "kornia_vs_cv2_mean_abs_255": round(float(d_kor.mean()), 4),
            "kornia_vs_cv2_max_abs_255": round(float(d_kor.max()), 2),
        })
        del resident, reference
        torch.cuda.empty_cache()
    return rows


def _timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1e3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--buckets", default="1,2,4,8,16")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("CUDA is required")
        return 2
    buckets = [int(x) for x in args.buckets.split(",")]
    report = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "engine": os.path.basename(args.engine)}
    if os.path.isfile(args.engine):
        report["graph"] = graph_arm(args.engine, buckets)
    else:
        report["graph"] = f"engine not found: {args.engine}"
    report["warp"] = warp_arm(buckets)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
