"""Does batching CodeFormer fp16 (UltraMax's network) make it cheaper per face?  (Stage 3 contract probe.)

The sibling of bench_restorer_batch.py (RestoreFormer++, 2026-10-03: flat, 24.4 / 25.6 / 25.2 / 25.0 ms/face at
B = 1 / 2 / 4 / 8), for the network UltraMax actually runs. That note listed CodeFormer as "not probed".

Measured 2026-10-13, RTX 4070, TensorRT "mixed" FP16 through the app's own provider policy, GPU-resident torch input,
ORT-owned CUDA output (binding a Torch-owned output corrupts faces - Enhance_UltraMax._run_cuda_postprocess), real
aligned crops, 20 calls x 3 interleaved rounds, median, ms PER FACE::

    engine profile (min/opt/max batch)      B=1      B=2      B=4      B=8
    prod  1 / 1 / 8   (what the app builds)  25.6     27.1     26.8     26.8     <- batching is 4-6% SLOWER
    opt2  1 / 2 / 8   (the Stage 3 brief)    27.3     22.5     25.0     24.4     <- see the note below

Batched outputs are bit-identical to the same faces run singly (max|d| 0.0, both engines) and match the CPU EP to a mean
of 0.002-0.007 on the [-1, 1] output, so batching is VALID here: the ONNX graph is batch-invariant. It just does not pay
on the shipped profile. The opt2 engine is faster at exactly B=2 (-12% vs prod B=1) but 6.6% slower at B=1, which is
most of a single-person clip. A second run on the SAME cached engines reproduced every cell to <1% (25.61 / 27.06 /
26.92 / 26.82 and 27.53 / 22.49 / 25.06 / 24.44); a fresh engine BUILD was not repeated, and the engine-to-engine spread on
this card is ~15% (bench_restorer_batch.py), so one build is not a result. Even taken at face value the ceiling is 12% of a ~26 ms/frame stage on frames that hold two
faces at once - about 2% of a render, under the ~5% this rig resolves.

PITFALL (cost an hour): ``astype`` on a transposed array KEEPS the transposed memory layout, and ``bind_input`` is handed
a raw ``data_ptr()``, so every tensor must be made contiguous (np.ascontiguousarray / .contiguous()) before it is bound.
The network still answers with a plausible finite image.

Usage (repo root, app stopped; needs app/output/restorer_fp16/crops.npz - tools/bench_restorer_batch.py --arms writes it)::

    app\\env\\Scripts\\python.exe tools/bench_codeformer_batch.py [--out result.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app"
for p in (str(APP), str(APP / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

BATCHES = (1, 2, 4, 8)


def _trt(chain):
    return next(p for p in chain if isinstance(p, tuple) and "tensorrt" in p[0].lower())[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    out_path = Path(args.out).resolve() if args.out else None      # the app's init changes the working directory

    # the app's init FIRST: it puts the TensorRT DLLs on PATH (AGENTS.md)
    import angle_bench as ab
    from settings import Settings
    cfg = Settings(str(APP / "config.yaml"))
    g = ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), None, None, sync_config=True)

    import numpy as np
    import onnxruntime as ort
    import torch
    from roop.precision_policy import providers_for
    from roop.utilities import get_onnx_session_options

    path = str(APP / "models" / "CodeFormer" / "codeformer.fp16.onnx")
    crops = np.load(APP / "output" / "restorer_fp16" / "crops.npz")["crops"]            # BGR uint8 512x512x3
    x_all = np.ascontiguousarray(((crops[..., ::-1].astype(np.float32) / 127.5) - 1.0)
                                 .transpose(0, 3, 1, 2).astype(np.float16))
    opts = get_onnx_session_options(optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED)
    prod = providers_for("ultramax", g.execution_providers, path)[0]
    brief = []
    for prov in prod:
        if isinstance(prov, tuple) and "tensorrt" in prov[0].lower():
            o = dict(prov[1])
            o["trt_profile_min_shapes"], o["trt_profile_opt_shapes"], o["trt_profile_max_shapes"] = (
                "x:1x3x512x512", "x:2x3x512x512", "x:8x3x512x512")
            scoped = str(o["trt_engine_cache_path"]) + "_probe_opt2_" + hashlib.sha256(b"1-2-8").hexdigest()[:6]
            os.makedirs(scoped, exist_ok=True)       # a different profile is a different engine: never share a cache directory
            o["trt_engine_cache_path"] = o["trt_timing_cache_path"] = scoped
            prov = (prov[0], o)
        brief.append(prov)

    class Arm:
        def __init__(self, label, chain):
            self.label = label
            self.s = ort.InferenceSession(path, opts, providers=chain)
            if self.s.get_providers()[0] != "TensorrtExecutionProvider":
                raise SystemExit("%s is not on TensorRT (%s)" % (label, self.s.get_providers()))
            self.w = torch.tensor([0.5], dtype=torch.float64, device="cuda")

        def tensor(self, b):
            return torch.from_numpy(np.ascontiguousarray(x_all[[i % len(x_all) for i in range(b)]])).cuda().contiguous()

        def _bind(self, xt):
            io = self.s.io_binding()
            io.bind_input("x", "cuda", 0, np.float16, tuple(xt.shape), xt.data_ptr())
            io.bind_input("w", "cuda", 0, np.float64, tuple(self.w.shape), self.w.data_ptr())
            io.bind_output("y", "cuda", 0)
            return io

        def run(self, xt):
            io = self._bind(xt)
            self.s.run_with_iobinding(io)
            io.synchronize_outputs()
            return io.copy_outputs_to_cpu()[0].astype(np.float32)

        def time_ms(self, xt, iters):
            io = self._bind(xt)
            for _ in range(3):
                self.s.run_with_iobinding(io)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                self.s.run_with_iobinding(io)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters * 1e3

    arms = [Arm("prod", prod), Arm("opt2", brief)]
    result = {"gpu": torch.cuda.get_device_name(0),
              "prod_profile": [_trt(prod).get(k) for k in ("trt_profile_min_shapes", "trt_profile_opt_shapes", "trt_profile_max_shapes")],
              "arms": {}}
    for a in arms:                                   # the first call per shape may build/tune: keep it out of the timing
        singles = np.concatenate([a.run(a.tensor(1)[:1]) if k == 0 else
                                  a.run(torch.from_numpy(np.ascontiguousarray(x_all[k:k + 1])).cuda().contiguous())
                                  for k in range(4)])
        for b in BATCHES:
            a.run(a.tensor(b))
        batch4 = a.run(a.tensor(4))
        diff = np.abs(batch4 - singles)
        result["arms"][a.label] = {"batch4_vs_singles_max_abs": float(diff.max()), "batch4_vs_singles_mean_abs": float(diff.mean()),
                                   "ms_per_face": {}}
        print("[%s] batch-4 vs the same faces alone: max|d|=%.5f mean|d|=%.6f" % (a.label, diff.max(), diff.mean()), flush=True)

    rounds = {(a.label, b): [] for a in arms for b in BATCHES}
    for _ in range(args.rounds):
        for b in BATCHES:
            for a in arms:
                rounds[(a.label, b)].append(a.time_ms(a.tensor(b), args.iters) / b)
    for a in arms:
        for b in BATCHES:
            r = rounds[(a.label, b)]
            result["arms"][a.label]["ms_per_face"][str(b)] = {"median": round(statistics.median(r), 3), "rounds": [round(v, 3) for v in r]}
        print("[%s] ms/face  %s" % (a.label, "  ".join("B=%d %.2f" % (b, statistics.median(rounds[(a.label, b)])) for b in BATCHES)), flush=True)
    text = json.dumps(result, indent=2)
    if out_path:
        out_path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
