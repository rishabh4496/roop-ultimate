"""Does batching a 512 restorer make it cheaper per face?  (Stage 6 contract probe.)

Measured 2026-10-03, RTX 4070, RestoreFormer++ (the network under Restore Ultra),
TensorRT "mixed" FP16 through the app's own provider policy, GPU-resident IOBinding,
real FFHQ-aligned 512 crops, 20 timed calls each (two runs; ms per face)::

                                          run 1     run 2
    production, fixed batch-1 graph       25.16     21.75    <- engine-to-engine spread ~15%
    dynamic-batch graph  B=1              24.40     25.10
                         B=2              25.61     25.13
                         B=4              25.21     25.03
                         B=8              24.99     24.83    <- flat within a process: no gain
    SSIM vs TensorRT FP32 (production B=1): 0.9960 / 0.9970 mean, 0.9950 / 0.9957 min
    (the gate asked for 0.998: the shipped FP16 "mixed" path is already below it at B=1)

One 512x512 conv/attention network already saturates the card at batch 1, so the
per-face time does not move with B. The shipped ONNX exports are fixed batch-1; this
tool relaxes the batch axis with the swapper's own ``_relax_batch_dim`` and lets
``trt_shape_profile`` derive the TensorRT profile, i.e. everything a batched restorer
would need, and still measures 0% per-face gain.

Usage (from the repo root, app stopped)::

    app\\env\\Scripts\\python.exe tools/bench_restorer_batch.py [--out result.json]

Needs ``ROOP_KEEP_DIR`` (or <PINOKIO_HOME>/roop-keep) with ``single/s3.mp4`` for the crops.
First run builds three TensorRT engines (a few minutes); later runs hit the cache.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app"
for p in (str(APP), str(APP / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

# FFHQ 512 five-point template (the restorers' training space)
FFHQ_512 = [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
            [201.26117, 371.41043], [313.08905, 371.15118]]


def media_dir() -> Path:
    env = os.environ.get("ROOP_KEEP_DIR")
    return Path(env) if env else REPO.parents[1] / "roop-keep"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--clip", default=str(media_dir() / "single" / "s3.mp4"))
    parser.add_argument("--model", default=str(APP / "models" / "restoreformer_plus_plus.onnx"))
    parser.add_argument("--model-key", default="restoreformer_pp")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    # the app's init FIRST: it puts the TensorRT DLLs on PATH; a bare process silently
    # falls back to CPU and reports a 4 ms model as 210 ms (AGENTS.md)
    import angle_bench as ab
    from settings import Settings
    cfg = Settings(str(APP / "config.yaml"))
    g = ab.init_pipeline(cfg.provider, cfg.swap_model, None, None, sync_config=True)

    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from skimage.transform import SimilarityTransform

    from roop.benchmark.regression import ssim as ssim_fn
    from roop.face_util import get_all_faces
    from roop.precision_policy import providers_for
    from roop.processors.FaceSwapInsightFace import _relax_batch_dim

    work = Path(tempfile.mkdtemp(prefix="restorer_batch_"))
    dynamic = work / "dynamic_batch.onnx"
    model = onnx.load(args.model)
    _relax_batch_dim(model)
    del model.graph.value_info[:]
    onnx.save(model, str(dynamic))

    template = np.array(FFHQ_512, np.float32)
    cap = cv2.VideoCapture(args.clip)
    crops, index = [], 0
    while len(crops) < 8:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        if index % 12:
            continue
        faces = get_all_faces(frame)
        if not faces:
            continue
        face = max(faces, key=lambda f: f.bbox[2] - f.bbox[0])
        tf = SimilarityTransform()
        tf.estimate(face.kps.astype(np.float32), template)
        crops.append(cv2.warpAffine(frame, tf.params[:2], (512, 512), flags=cv2.INTER_CUBIC))
    if len(crops) < 8:
        raise SystemExit(f"only {len(crops)} aligned faces found in {args.clip}")

    x_all = np.stack([((c[..., ::-1].astype(np.float32) / 127.5) - 1.0).transpose(2, 0, 1)
                      for c in crops]).astype(np.float32)

    def post(y):
        return np.clip((y.transpose(1, 2, 0) + 1) * 127.5, 0, 255).astype(np.uint8)

    def session(path, providers):
        s = ort.InferenceSession(path, ort.SessionOptions(), providers=providers)
        active = s.get_providers()[0]
        if "Tensorrt" not in active:
            raise SystemExit(f"not on TensorRT ({active}): a CPU/CUDA number would be meaningless")
        return s

    def run(sess, batch, iters):
        xt = torch.from_numpy(batch).cuda().contiguous()
        yt = torch.empty((batch.shape[0], 3, 512, 512), dtype=torch.float32, device="cuda")
        io = sess.io_binding()
        io.bind_input(sess.get_inputs()[0].name, "cuda", 0, np.float32, tuple(xt.shape),
                      xt.data_ptr())
        io.bind_output(sess.get_outputs()[0].name, "cuda", 0, np.float32, tuple(yt.shape),
                       yt.data_ptr())
        for _ in range(3):
            sess.run_with_iobinding(io)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            sess.run_with_iobinding(io)
        torch.cuda.synchronize()
        return yt.cpu().numpy(), (time.perf_counter() - t0) / iters

    result = {"gpu": torch.cuda.get_device_name(0), "model": os.path.basename(args.model)}

    fp32 = session(args.model, providers_for(args.model_key, g.execution_providers,
                                             args.model, requested="fp32")[0])
    baseline = [post(run(fp32, x_all[k:k + 1], 1)[0][0]) for k in range(8)]
    del fp32

    prod = session(args.model, providers_for(args.model_key, g.execution_providers,
                                             args.model)[0])
    outs, dt = [], 0.0
    for k in range(8):
        y, dt = run(prod, x_all[k:k + 1], args.iters)
        outs.append(post(y[0]))
    s = [ssim_fn(a, b) for a, b in zip(outs, baseline)]
    result["production_b1"] = {"ms_per_face": dt * 1e3, "ssim_vs_fp32_mean": float(np.mean(s)),
                               "ssim_vs_fp32_min": float(np.min(s))}
    del prod

    dyn = session(str(dynamic), providers_for(args.model_key, g.execution_providers,
                                              str(dynamic))[0])
    # Per-face identity of a batched call against the SAME engine at batch 1 (the acceptance test for a
    # cross-frame enhancer batcher: a face must not change because of who it was batched with).
    b1_out = [run(dyn, x_all[k:k + 1], 1)[0][0] for k in range(8)]
    for b in (1, 2, 4, 8):
        y, dt = run(dyn, x_all[:b], args.iters)
        s = [ssim_fn(post(y[k]), baseline[k]) for k in range(b)]
        same = sum(int(np.array_equal(y[k], b1_out[k])) for k in range(b))
        worst = max(float(np.abs(y[k].astype(np.float64) - b1_out[k].astype(np.float64)).max()) for k in range(b))
        result[f"dynamic_b{b}"] = {"ms_per_call": dt * 1e3, "ms_per_face": dt * 1e3 / b,
                                   "ssim_vs_fp32_mean": float(np.mean(s)),
                                   "ssim_vs_fp32_min": float(np.min(s)),
                                   "bit_identical_to_b1": f"{same}/{b}",
                                   "max_abs_diff_vs_b1": worst}
    base_ms = result["production_b1"]["ms_per_face"]
    result["best_batched_ms_per_face"] = min(result[f"dynamic_b{b}"]["ms_per_face"]
                                             for b in (2, 4, 8))
    result["latency_reduction_vs_production"] = 1 - result["best_batched_ms_per_face"] / base_ms
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
