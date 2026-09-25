"""Grade the swap network's INT8 / FP8 engines against FP16 on held-out faces.

Model-level, on the HOLDOUT set that tools/build_calibration_set.py writes from
clips that were never calibrated on. Every arm sees the identical feeds (the
live preprocessing's own blobs and identity vectors).

ARMS
  ort      the production path: ONNX Runtime + TensorRT EP at the config's
           trt_precision (what renders use today)
  fp16     native TensorRT FP16 engine (same runner as the quantized arms, so
           fp16 -> int8/fp8 isolates quantization)
  int8     native, IInt8EntropyCalibrator2 on the calibration set
  fp8      native, explicit E4M3 Q/DQ from the calibration set's amax

METRICS (per arm, over the holdout)
  id_src     cosine of the swapped crop's ArcFace embedding (the app's own
             recognizer, 112 crop = the 256 arcface crop scaled) to the source
             identity that was fed in. The product. Higher is better.
  d_id       id_src minus the fp16 arm's, per face, then averaged; and the
             share of faces where it fell by more than 0.02
  psnr       dB against the fp16 arm's output image
  mask_iou   of the net's own mask (>0.5) against fp16's
  ms/call    numpy-in/numpy-out wall time per call (what the live swap pays)
  gpu ms     enqueue-only, CUDA-synchronized, per call (the GPU work removed)
  finite     outputs with a NaN/inf (must be 0)

    app/env/Scripts/python.exe app/tests/quant_quality_bench.py
    app/env/Scripts/python.exe app/tests/quant_quality_bench.py --tiers fp16 int8 fp8 --rebuild
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def unit(v):
    v = np.asarray(v, np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def to_image(chw, denormalize):
    x = np.asarray(chw, np.float32)
    if denormalize:
        x = (x + 1.0) / 2.0
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8).transpose(1, 2, 0)[:, :, ::-1]


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="*", default=["fp16", "int8", "fp8"])
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--timing-calls", type=int, default=300)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import cv2
    import angle_bench as ab
    from settings import Settings
    from roop import trt_quant as tq
    from roop.processors.FaceSwapInsightFace import SWAP_MODELS

    cfg = Settings(os.path.join(APP, "config.yaml"))
    swap_model = str(cfg.swap_model)
    spec = SWAP_MODELS[swap_model]
    g = ab.init_pipeline(str(cfg.provider), swap_model, "None", str(cfg.mask_engine),
                         sync_config=True)
    model_path = os.path.join(APP, "models", spec["file"])
    calib = tq.CalibrationSet.load(tq.calibration_set_path(spec["file"]))
    hold = tq.CalibrationSet.load(str(tq.calibration_set_path(spec["file"])).replace(
        ".calib.npz", ".holdout.npz"))
    print(f"[quant] {swap_model}: calibration {len(calib)}, holdout {len(hold)} "
          f"({tq.device_identity()})", flush=True)

    from roop.face_util import get_first_face  # noqa: F401  (initializes the analyser)
    import roop.globals as rg
    from roop.face_analyser import get_face_analyser
    rec = get_face_analyser().models["recognition"]

    def embed(img256):
        crop = cv2.resize(img256, (112, 112), interpolation=cv2.INTER_AREA)
        return unit(rec.get_feat(crop))

    # ORT production arm: the real swapper class, with quantization off.
    from roop.processors.FaceSwapInsightFace import FaceSwapInsightFace
    rg.CFG.swap_quantization = "off"
    sw = FaceSwapInsightFace()
    sw.Initialize({"devicename": "cuda", "swap_model": swap_model})
    feeds = [hold.feed(i) for i in range(len(hold))]

    arms = {}

    def ort_call(feed):
        return sw._infer(feed)

    arms["ort"] = (ort_call, None)
    for tier in args.tiers:
        try:
            engine, tier_, why = tq.ensure_engine(model_path, tier=tier, calib=calib,
                                                  force=args.rebuild)
        except tq.QuantizationError as error:
            print(f"[quant] {tier}: REJECTED -- {error}", flush=True)
            continue
        runner = tq.NativeTRTRunner(engine)
        order = [o.name for o in sw.model_swap_insightface.get_outputs()]
        arms[tier] = ((lambda f, r=runner, o=order: [r.run(f)[r.outputs.index(n)] for n in o]), runner)
        man = json.load(open(str(engine) + ".json", encoding="utf-8"))
        print(f"[quant] {tier}: {engine.name} built/validated "
              f"({'rebuilt: ' + ', '.join(why) if why else 'cache valid'}); "
              f"layers {man.get('layer_precisions')}", flush=True)

    results = {}
    outputs = {}
    for name, (call, runner) in arms.items():
        imgs, masks, bad = [], [], 0
        for f in feeds:
            o = call(f)
            if not all(np.isfinite(x).all() for x in o):
                bad += 1
            imgs.append(to_image(o[0][0], spec.get("denormalize", False)))
            masks.append(o[1][0, 0] > 0.5 if len(o) > 1 else None)
        # timing: numpy path
        f0 = feeds[0]
        for _ in range(20):
            call(f0)
        t = time.perf_counter()
        for _ in range(args.timing_calls):
            call(f0)
        ms = (time.perf_counter() - t) / args.timing_calls * 1000
        gpu = None
        if runner is not None:
            import torch
            slot = runner._lease()
            stream = torch.cuda.current_stream()
            for n in runner.inputs:
                slot["buf"][n].copy_(torch.from_numpy(np.ascontiguousarray(f0[n])))
            for _ in range(20):
                runner._enqueue(slot, stream)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(args.timing_calls):
                runner._enqueue(slot, stream)
            torch.cuda.synchronize()
            gpu = (time.perf_counter() - t) / args.timing_calls * 1000
            runner._return(slot)
        ids = [float(np.dot(embed(im), unit(hold.embeddings[i]))) for i, im in enumerate(imgs)]
        outputs[name] = (imgs, masks, ids)
        results[name] = {"id_src": float(np.mean(ids)), "nonfinite": bad,
                         "ms_call": round(ms, 3), "gpu_ms": None if gpu is None else round(gpu, 3)}

    ref_imgs, ref_masks, ref_ids = outputs["fp16"] if "fp16" in outputs else outputs["ort"]
    for name, (imgs, masks, ids) in outputs.items():
        d = np.asarray(ids) - np.asarray(ref_ids)
        ious = [1.0 if not (a | b).any() else float((a & b).sum() / (a | b).sum())
                for a, b in zip(masks, ref_masks) if a is not None and b is not None]
        results[name].update(
            d_id=float(d.mean()), d_id_worst=float(d.min()),
            frac_drop_gt_002=float((d < -0.02).mean()),
            psnr=float(np.mean([psnr(a, b) for a, b in zip(imgs, ref_imgs)])),
            mask_iou=float(np.mean(ious)) if ious else None)

    # per-stratum identity delta for the reduced tiers
    strata = {}
    for name in args.tiers:
        if name == "fp16" or name not in outputs:
            continue
        d = np.asarray(outputs[name][2]) - np.asarray(ref_ids)
        for axis in ("pose_bin", "light_bin", "tone_bin", "occlusion_bin"):
            for value in sorted({m[axis] for m in hold.meta}):
                idx = [i for i, m in enumerate(hold.meta) if m[axis] == value]
                strata.setdefault(name, {})[f"{axis}={value}"] = (len(idx), round(float(d[idx].mean()), 4))

    ref = "fp16" if "fp16" in outputs else "ort"
    print(f"\n{'arm':<6} {'id_src':>7} {'d_id':>8} {'worst':>8} {'>0.02':>6} "
          f"{'psnr':>6} {'maskIoU':>7} {'ms/call':>8} {'gpu ms':>7} {'nonfin':>6}   (vs {ref})")
    for name, r in results.items():
        print(f"{name:<6} {r['id_src']:7.4f} {r['d_id']:+8.4f} {r['d_id_worst']:+8.4f} "
              f"{r['frac_drop_gt_002']:6.1%} {r['psnr']:6.2f} "
              f"{(r['mask_iou'] or float('nan')):7.4f} {r['ms_call']:8.3f} "
              f"{(r['gpu_ms'] if r['gpu_ms'] is not None else float('nan')):7.3f} {r['nonfinite']:6d}")
    for name, rows in strata.items():
        print(f"\n{name} d_id by stratum (n, mean):")
        for key, (n, v) in rows.items():
            print(f"  {key:<28} n={n:<4} {v:+.4f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"results": results, "strata": strata,
                       "device": tq.device_identity()}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
