"""UltraMax (codeformer.fp16) under TensorRT: is B=N the same as N x B=1, and is it faster per face?

The brief behind this asked for a (min=1, opt=2, max=8) batch profile "so multi-face frames execute in
one batched inference pass". Two things have to be true for that to be worth anything, and both are
measured here on the real graph, through the app's own provider chain (``providers_for('ultramax', ...)``):

  parity    B=4 output vs the same four crops run one at a time (max |diff| on the float output and in
            uint8 levels after the app's own ``x*127.5+127.5`` conversion)
  speed     ms per face at B=1 (x4) vs B=4, warm, ``--iters`` repetitions, median

``--profile shipped`` uses the profile ``trt_shape_profile`` resolves today and the engine already in the
cache (no build). ``--profile 1,2,8`` rebuilds an engine for that profile in its OWN cache directory under
``--cache`` (a first-time build takes minutes and is not counted).

    app/env/Scripts/python.exe app/tests/ultramax_batch_probe.py --profile shipped
    app/env/Scripts/python.exe app/tests/ultramax_batch_probe.py --profile 1,2,8 --cache <dir>

Needs the 4070's TensorRT stack; it prints SKIP and exits 0 when TensorRT is not among the providers, so
a CI box without one does not fail on it.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODEL_REL = os.path.join("models", "CodeFormer", "codeformer.fp16.onnx")
S = 512


def init_environment():
    """The app's own bring-up (TensorRT DLL paths, provider decode), without loading any pipeline stage."""
    os.chdir(APP)
    import roop.globals as g
    from settings import Settings
    g.CFG = Settings("config.yaml")
    g.CFG.provider = "tensorrt"
    from ui.main import prepare_environment
    prepare_environment()
    from roop.core import decode_execution_providers
    g.execution_providers = decode_execution_providers(["tensorrt"])
    return g


def build_session(profile: str = "shipped", cache_dir: str | None = None):
    """An ORT session on the UltraMax graph with the shipped provider chain, optionally a different batch profile."""
    import onnxruntime as ort
    import roop.globals as g
    from roop.precision_policy import providers_for
    model_path = os.path.join(APP, MODEL_REL)
    chain, _ = providers_for("ultramax", g.execution_providers, model_path)
    patched = []
    for item in chain:
        name, opts = (item if isinstance(item, tuple) else (item, {}))
        opts = dict(opts)
        if name == "TensorrtExecutionProvider" and profile != "shipped":
            lo, opt, hi = (int(v) for v in profile.split(","))
            opts["trt_profile_min_shapes"] = f"x:{lo}x3x{S}x{S}"
            opts["trt_profile_opt_shapes"] = f"x:{opt}x3x{S}x{S}"
            opts["trt_profile_max_shapes"] = f"x:{hi}x3x{S}x{S}"
            scoped = os.path.join(cache_dir or os.path.join(APP, "models", "trt_cache"), f"probe_b{lo}_{opt}_{hi}")
            os.makedirs(scoped, exist_ok=True)
            opts["trt_engine_cache_path"] = scoped
            if opts.get("trt_timing_cache_path"):
                opts["trt_timing_cache_path"] = scoped
        patched.append((name, opts) if opts else name)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess = ort.InferenceSession(model_path, sess_options=so, providers=patched)
    return sess


def make_crops(n: int, dtype) -> np.ndarray:
    """n deterministic 512x512 face-like crops (smooth shading + texture + an edge) in [-1, 1], NCHW."""
    out = []
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / S
    for i in range(n):
        rng = np.random.default_rng(1000 + i)
        base = 0.15 * np.sin(6.0 * xx + i) + 0.2 * np.cos(4.0 * yy - i) + 0.05 * (xx - 0.5)
        tex = rng.normal(0, 0.06, (S, S)).astype(np.float32)
        edge = 0.25 * ((xx - 0.5) ** 2 + (yy - 0.45) ** 2 < 0.09)
        ch = [np.clip(base + tex + edge + 0.04 * c, -1, 1) for c in range(3)]
        out.append(np.stack(ch, 0))
    return np.stack(out, 0).astype(dtype)


def run(sess, x: np.ndarray) -> np.ndarray:
    names = [i.name for i in sess.get_inputs()]
    w = np.array([0.5], dtype=np.float64)
    return sess.run(None, {names[0]: x, names[1]: w})[0]


def to_u8(out: np.ndarray) -> np.ndarray:
    return np.clip(np.maximum(out.astype(np.float32), -1.0) * 127.5 + 127.5, 0, 255).round().astype(np.uint8)


def active_providers(sess):
    return sess.get_providers()


def measure(sess, batch: int, iters: int, warm: int = 5) -> dict:
    dtype = np.float16 if "float16" in sess.get_inputs()[0].type else np.float32
    crops = make_crops(batch, dtype)
    singles = [crops[i:i + 1] for i in range(batch)]
    for _ in range(warm):
        run(sess, crops)
        for s in singles:
            run(sess, s)
    t_batch, t_single = [], []
    for _ in range(iters):
        t0 = time.perf_counter()
        run(sess, crops)
        t_batch.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        for s in singles:
            run(sess, s)
        t_single.append((time.perf_counter() - t0) * 1e3)
    batched = run(sess, crops)
    one_by_one = np.concatenate([run(sess, s) for s in singles], 0)
    d = np.abs(batched.astype(np.float32) - one_by_one.astype(np.float32))
    u = np.abs(to_u8(batched).astype(np.int16) - to_u8(one_by_one).astype(np.int16))
    return {
        "batch": batch,
        "single_ms_per_face": statistics.median(t_single) / batch,
        "batched_ms_per_face": statistics.median(t_batch) / batch,
        "max_abs_float": float(d.max()),
        "max_levels_u8": int(u.max()),
        "frac_pixels_differing_u8": float((u > 0).mean()),
        "finite": bool(np.isfinite(batched.astype(np.float32)).all()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="shipped", help="'shipped' or 'min,opt,max' e.g. 1,2,8")
    ap.add_argument("--cache", default=None, help="parent directory for a non-shipped profile's engine cache")
    ap.add_argument("--batches", default="2,4", help="batch sizes to compare against singles")
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args(argv)
    g = init_environment()
    if not any("Tensorrt" in str(p) for p in g.execution_providers):
        print("SKIP: TensorRT is not among the execution providers on this machine")
        return 0
    t0 = time.perf_counter()
    sess = build_session(a.profile, a.cache)
    print(f"[probe] profile={a.profile} session ready in {time.perf_counter() - t0:.1f}s "
          f"providers={active_providers(sess)}")
    for b in (int(v) for v in a.batches.split(",")):
        r = measure(sess, b, a.iters)
        print(f"[probe] B={r['batch']}: {r['single_ms_per_face']:.2f} ms/face as singles, "
              f"{r['batched_ms_per_face']:.2f} ms/face batched "
              f"({r['single_ms_per_face'] / r['batched_ms_per_face']:.2f}x); "
              f"max|diff| float {r['max_abs_float']:.3e}, u8 {r['max_levels_u8']} levels, "
              f"{r['frac_pixels_differing_u8'] * 100:.3f}% px differ; finite={r['finite']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
