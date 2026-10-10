"""Which layers of the 512 restorer (RestoreFormer++, the network under Restore Ultra) cost the FP16 path its SSIM?

    app\\env\\Scripts\\python.exe tools/restorer_fp16_ranking.py validate    # all-FP16 simulation vs FP32, 32 real crops
    app\\env\\Scripts\\python.exe tools/restorer_fp16_ranking.py rank        # inject FP16 into ONE module at a time

WHY A SIMULATION. The shipped path is the ORT TensorRT provider, which fuses layers and exposes no per-layer
tensors, so its per-layer FP16 error cannot be read off. The CUDA EP cannot run this graph on this ORT build (session
creation silently falls back to CPU), but the CPU EP can (~1 s per 512 crop) and gives a TRUE FP32 reference with every
intermediate available. FP16 is modelled the way TensorRT stores it: weights, inputs and outputs of a module's nodes go
through Cast(fp16) -> Cast(fp32), so rounding AND saturation (>65504 -> inf) happen where an FP16 tensor would; the
accumulation inside a layer stays FP32 (as tensor-core convolutions do). That is a model, not the engine, so `validate`
first checks that the all-FP16 simulation reproduces the SSIM the real TensorRT FP16 engine loses (0.996 mean in
bench_restorer_batch); only if it does is the per-module ranking meaningful. Native TensorRT builds are the final judge.

A MODULE is the parent path of a node name ('/encoder/block.0/norm1/InstanceNormalization' -> '/encoder/block.0/norm1').
Ranking = the final-output error (SSIM, mean abs error in 8-bit levels) when ONLY that module is FP16 and everything else
is FP32: a per-layer FP16-vs-FP32 error that includes how much the rest of the network amplifies it.

Needs ``ROOP_KEEP_DIR`` clips for the crops (cached in app/output/restorer_fp16/crops.npz). Output: app/output/restorer_fp16/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app"
for p in (str(APP), str(APP / "tests"), str(REPO / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = APP / "output" / "restorer_fp16"
MODEL = APP / "models" / "restoreformer_plus_plus.onnx"
SINGLE_ONNX = OUT / "restoreformer_single_output.onnx"     # the same graph with only the real output (tools/restorer_native_engine.py strip)
FLOAT, FLOAT16 = 1, 10


def _init_app():
    import angle_bench as ab
    from settings import Settings
    cfg = Settings(str(APP / "config.yaml"))
    return ab.init_pipeline(cfg.provider, cfg.swap_model, None, None, sync_config=True)


def module_of(name: str) -> str:
    parts = name.rstrip("/").split("/")
    return "/".join(parts[:-1]) or name


def load_model():
    import onnx
    model = onnx.load(str(MODEL))
    inferred = onnx.shape_inference.infer_shapes(model)
    dtypes = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        dtypes[vi.name] = vi.type.tensor_type.elem_type
    for init in model.graph.initializer:
        dtypes[init.name] = init.data_type
    final = model.graph.output[0].name
    for o in list(model.graph.output)[1:]:          # leftover export intermediates; production reads output 0 only
        model.graph.output.remove(o)
    del model.graph.value_info[:]
    return model, dtypes, final


def inject(model, dtypes, selected):
    """A copy of *model* in which the nodes with indices in *selected* behave as FP16 tensors would."""
    import onnx
    from onnx import TensorProto, helper
    m = onnx.ModelProto()
    m.CopyFrom(model)
    nodes = list(m.graph.node)
    inside_out = {o for i in selected for o in nodes[i].output}
    new_nodes, quantised = [], {}

    def cast_chain(src, dst, tag):
        mid = f"{dst}__h"
        return [helper.make_node("Cast", [src], [mid], name=f"{tag}_to16", to=TensorProto.FLOAT16),
                helper.make_node("Cast", [mid], [dst], name=f"{tag}_to32", to=TensorProto.FLOAT)]

    for i, node in enumerate(nodes):
        if i not in selected:
            new_nodes.append(node)
            continue
        pre = []
        for k, t in enumerate(node.input):
            if t and dtypes.get(t) == FLOAT and t not in inside_out:
                if t not in quantised:
                    quantised[t] = f"{t}__q"
                    pre += cast_chain(t, quantised[t], f"sim{i}_in{k}")
                node.input[k] = quantised[t]
        post = []
        for k, o in enumerate(node.output):
            if o and dtypes.get(o) == FLOAT:
                raw = f"{o}__raw"
                node.output[k] = raw
                post += cast_chain(raw, o, f"sim{i}_out{k}")
        new_nodes += pre + [node] + post
    del m.graph.node[:]
    m.graph.node.extend(new_nodes)
    return m


def session(model_or_path, level=None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if level is not None:
        so.graph_optimization_level = level
    data = model_or_path if isinstance(model_or_path, (bytes, str)) else model_or_path.SerializeToString()
    return ort.InferenceSession(data, so, providers=["CPUExecutionProvider"])


def prep(crops):
    import numpy as np
    return np.stack([((c[..., ::-1].astype(np.float32) / 127.5) - 1.0).transpose(2, 0, 1) for c in crops]).astype(np.float32)


def post(y):
    import numpy as np
    return np.clip((y.transpose(1, 2, 0) + 1) * 127.5, 0, 255).astype(np.uint8)


def metrics(ys, refs):
    import cv2
    import numpy as np
    from roop.benchmark.regression import ssim
    s, mae, psnr = [], [], []
    for y, r in zip(ys, refs):
        a, b = post(y), post(r)
        s.append(float(ssim(a, b)))
        mae.append(float(np.abs(y.astype(np.float64) - r.astype(np.float64)).mean() * 127.5))
        mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
        psnr.append(99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))
    return {"ssim_mean": float(np.mean(s)), "ssim_min": float(np.min(s)), "mae_levels": float(np.mean(mae)),
            "psnr_mean": float(np.mean(psnr)), "psnr_min": float(np.min(psnr))}


def run_all(sess, x, final):
    import numpy as np
    return np.concatenate([sess.run([final], {"input": x[k:k + 1]})[0] for k in range(len(x))])


def groups_of(model, dtypes):
    nodes = list(model.graph.node)
    groups = defaultdict(list)
    for i, n in enumerate(nodes):
        if any(dtypes.get(o) == FLOAT for o in n.output) and n.op_type != "Constant":
            groups[module_of(n.name)].append(i)
    return nodes, groups


def reference(crops, final, model):
    import numpy as np
    path = OUT / "reference.npy"
    x = prep(crops)
    if path.is_file():
        return x, np.load(path)
    import onnxruntime as ort
    t0 = time.time()
    ref = run_all(session(model), x, final)
    np.save(path, ref)
    print(f"[ref] FP32 CPU reference for {len(x)} crops in {time.time() - t0:.0f}s", flush=True)
    return x, ref


def cmd_validate(args):
    import numpy as np
    _init_app()
    import bench_restorer_batch as b
    OUT.mkdir(parents=True, exist_ok=True)
    crops = b.collect_crops(32, cache=str(OUT / "crops.npz"))
    model, dtypes, final = load_model()
    x, ref = reference(crops, final, model)
    nodes, groups = groups_of(model, dtypes)
    allsel = {i for idx in groups.values() for i in idx}
    res = {"n_nodes": len(nodes), "n_modules": len(groups), "ops_in_modules": dict(Counter(nodes[i].op_type for i in allsel))}
    null = inject(model, dtypes, set())
    res["null_control"] = metrics(run_all(session(null), x[:4], final), ref[:4])
    t0 = time.time()
    sim = inject(model, dtypes, allsel)
    res["all_fp16_sim"] = metrics(run_all(session(sim), x, final), ref)
    res["all_fp16_sim"]["seconds"] = time.time() - t0
    res["real_trt_fp16_vs_trt_fp32_reference"] = {"ssim_mean": "0.996-0.997", "ssim_min": "0.995-0.9957",
                                                  "source": "bench_restorer_batch.py header, 2026-10-03, 8 crops"}
    (OUT / "validate.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


def cmd_rank(args):
    import numpy as np
    _init_app()
    import bench_restorer_batch as b
    OUT.mkdir(parents=True, exist_ok=True)
    crops = b.collect_crops(32, cache=str(OUT / "crops.npz"))
    model, dtypes, final = load_model()
    x, ref = reference(crops, final, model)
    sub = list(range(0, 32, 32 // args.crops))[:args.crops]
    xs, rs = x[sub], ref[sub]
    nodes, groups = groups_of(model, dtypes)
    path = OUT / "rank.json"
    done = json.loads(path.read_text()) if path.is_file() else {}
    todo = [g for g in groups if g not in done]
    print(f"[rank] {len(groups)} modules, {len(done)} done, {len(todo)} to go; {len(sub)} crops each", flush=True)
    for n, g in enumerate(todo):
        t0 = time.time()
        sim = inject(model, dtypes, set(groups[g]))
        m = metrics(run_all(session(sim), xs, final), rs)
        m["ops"] = dict(Counter(nodes[i].op_type for i in groups[g]))
        done[g] = m
        path.write_text(json.dumps(done, indent=1))
        if n % 10 == 0:
            print(f"[rank] {n + 1}/{len(todo)} {g}: ssim {m['ssim_mean']:.5f} mae {m['mae_levels']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("validate", "rank"))
    ap.add_argument("--crops", type=int, default=8, help="crops per module injection (of the 32)")
    args = ap.parse_args()
    return {"validate": cmd_validate, "rank": cmd_rank}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
