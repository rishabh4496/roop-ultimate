"""Scratch: cosine / SSIM of the production TRT engines vs a CPU FP32 reference, plus ms/call.

Run from repo root:  app/env/Scripts/python.exe <this file> [model ...]
Models: rfpp_mixed rfpp_fp32 ultramax gpen512 gpen256
"""
import glob
import json
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(REPO, "app")
for p in (APP, os.path.join(APP, "tests")):
    sys.path.insert(0, p)
os.chdir(APP)

import cv2
import numpy as np
from skimage.metrics import structural_similarity

import trt_precision_fidelity as fid
fid._init_pipeline()
import onnxruntime
import roop.globals as g
from roop import aux_canary as ac, precision_policy as pp

M = os.path.join(APP, "models")
ARMS = {
    "rfpp_mixed": ("restoreformer_pp", "restoreformer_plus_plus.onnx", "mixed", False, ()),
    "rfpp_fp32": ("restoreformer_pp", "restoreformer_plus_plus.onnx", "fp32", False, ()),
    "ultramax": ("ultramax", os.path.join("CodeFormer", "codeformer.fp16.onnx"), None, True, (0.5,)),
    "gpen512": ("gpen_512", "GPEN-BFR-512.onnx", None, False, ()),
    "gpen256": ("gpen_256", "gpen_bfr_256.onnx", None, False, ()),
}
only = sys.argv[1:] or list(ARMS)


def faces(size):
    out = []
    for f in sorted(glob.glob(os.path.join(APP, "facesets", "*.png")))[:6]:
        im = cv2.imread(f)
        if im is None:
            continue
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255.0
        out.append(("face:" + os.path.basename(f), im.transpose(2, 0, 1)))
    return out


def inputs(size):
    items = []
    for kind, seed in (("skin+n0.04", 1), ("skin+n0.08", 2)):
        items.append(("synth:%s#%d" % (kind, seed), ac._synth(kind, seed, size, size)))
    return items + faces(size)


def cos(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def ssim(a, b):
    a = np.clip(a[0].astype(np.float32).transpose(1, 2, 0), -1, 1)
    b = np.clip(b[0].astype(np.float32).transpose(1, 2, 0), -1, 1)
    return float(structural_similarity(a, b, channel_axis=2, data_range=2.0))


results = {}
for arm in only:
    key, rel, requested, extended, aux = ARMS[arm]
    path = os.path.join(M, rel)
    prov = pp.providers_for(key, g.execution_providers, path, requested=requested)[0]
    t0 = time.time()
    live = onnxruntime.InferenceSession(path, __import__("roop.utilities", fromlist=["x"]).get_onnx_session_options(), providers=prov)
    if live.get_providers()[0] != "TensorrtExecutionProvider":
        print(arm, "NOT ON TRT", live.get_providers(), flush=True)
        continue
    opts = live.get_provider_options().get("TensorrtExecutionProvider", {})
    ref_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED if extended else None
    ref = onnxruntime.InferenceSession(
        path, __import__("roop.utilities", fromlist=["x"]).get_onnx_session_options(optimization_level=ref_level),
        providers=["CPUExecutionProvider"])
    name = live.get_inputs()[0].name
    dt = ac._input_dtype(live.get_inputs()[0])
    shp = live.get_inputs()[0].shape
    size = shp[2] if isinstance(shp[2], int) else 512
    extra = {m.name: np.array([v], dtype=ac._input_dtype(m)) for m, v in zip(live.get_inputs()[1:], aux)}
    out_name = [live.get_outputs()[0].name]
    rows = {}
    xs = []
    for label, chw in inputs(size):
        x = np.ascontiguousarray((chw * 2.0 - 1.0)[None], dtype=dt)
        xs.append(x)
        a = live.run(out_name, {name: x, **extra})[0].astype(np.float32)   # first call builds the engine
        b = ref.run(out_name, {name: x, **extra})[0].astype(np.float32)
        rows[label] = {"cosine": round(cos(a, b), 6), "ssim": round(ssim(a, b), 5),
                       "max_abs": round(float(np.abs(a - b).max()), 4), "finite": bool(np.isfinite(a).all())}
    times = []
    for _ in range(40):
        s = time.perf_counter()
        live.run(out_name, {name: xs[2], **extra})
        times.append((time.perf_counter() - s) * 1000)
    cs = [r["cosine"] for r in rows.values()]
    ss = [r["ssim"] for r in rows.values()]
    results[arm] = {"fp16": opts.get("trt_fp16_enable"), "ms_median": round(statistics.median(times[5:]), 2),
                    "cos_min": min(cs), "cos_mean": round(sum(cs) / len(cs), 6), "ssim_min": min(ss),
                    "ssim_mean": round(sum(ss) / len(ss), 5), "rows": rows, "build_s": round(time.time() - t0, 1)}
    print("[%s] fp16=%s ms=%.2f cos min %.6f mean %.6f | ssim min %.4f mean %.4f | build+run %.0fs" % (
        arm, opts.get("trt_fp16_enable"), results[arm]["ms_median"], results[arm]["cos_min"],
        results[arm]["cos_mean"], results[arm]["ssim_min"], results[arm]["ssim_mean"], results[arm]["build_s"]), flush=True)
    del live, ref
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measure_cosine_%s.json" % "_".join(only))
json.dump(results, open(out, "w"), indent=1)
print("wrote", out)
