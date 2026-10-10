"""What precision do XSeg / w600k_r50 / 2d106det / 1k3d68 REALLY run under in the shipped TensorRT config, and does the
XSeg NHWC input cost layers?

    env/Scripts/python.exe tests/trt_precision_probe.py            # writes docs/perf/trt_precision_probe.json
    env/Scripts/python.exe tests/trt_precision_probe.py shipped    # ... trt_precision_probe_shipped.json (same code, other label)

PART A - live session facts, read from the sessions the app's own loaders build (Mask_XSeg.Initialize, the FaceAnalysis
bundle), NOT from the request: bound provider, trt_fp16_enable / layer-norm fallback / heuristics / level as the live
session reports them, and an ORT profiling run of the same chain (count nodes per provider: one TensorrtExecutionProvider
node = the whole graph is one engine, anything on CUDA/CPU is a partition boundary).

PART B - the engine ORT cached for that graph, opened with the TensorRT python API: layer list (names; the cached engine
was built with default profiling verbosity, so names only), reformat/shuffle/transpose layers, and per-layer precision
where the build recorded it. A SECOND engine is built here from the same ONNX with ProfilingVerbosity.DETAILED, the same
fp16 flag and the same builder level, so the layer information carries precisions and formats; it is a diagnostic copy,
never used for inference.
"""
import glob
import json
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LIVE_KEYS = ("trt_fp16_enable", "trt_layer_norm_fp32_fallback", "trt_build_heuristics_enable",
             "trt_builder_optimization_level", "trt_engine_cache_path", "trt_force_sequential_engine_build",
             "trt_max_workspace_size", "trt_profile_min_shapes", "trt_profile_opt_shapes", "trt_profile_max_shapes")


def _init_pipeline():
    from settings import Settings
    import angle_bench as ab
    cfg = Settings(os.path.join(APP, "config.yaml"))
    return ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), "None", "None", sync_config=True)


def live_facts(sess):
    opts = (sess.get_provider_options() or {}).get("TensorrtExecutionProvider") or {}
    return {"providers": sess.get_providers(), "inputs": [(i.name, list(i.shape), i.type) for i in sess.get_inputs()],
            "outputs": [(o.name, list(o.shape), o.type) for o in sess.get_outputs()],
            "options": {k: opts.get(k) for k in LIVE_KEYS if k in opts}}


def profile_nodes(model_path, providers, feed_shape):
    """ORT profiling of a fresh session on the SAME provider chain: node count per provider."""
    import onnxruntime
    from roop.utilities import get_onnx_session_options
    so = get_onnx_session_options()
    so.enable_profiling = True
    so.profile_file_prefix = os.path.join(APP, "output", "trt_precision_probe_prof")
    sess = onnxruntime.InferenceSession(model_path, so, providers=providers)
    name = sess.get_inputs()[0].name
    x = np.random.RandomState(0).rand(*feed_shape).astype(np.float32)
    for _ in range(3):
        sess.run(None, {name: x})
    prof = sess.end_profiling()
    try:
        events = json.load(open(prof))
    finally:
        try:
            os.unlink(prof)
        except OSError:
            pass
    c = Counter()
    for e in events:
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"):
            c[(e.get("args") or {}).get("provider", "?")] += 1
    # a node that ran 3 times is counted 3 times; report per run
    return {k: v // 3 if v >= 3 else v for k, v in c.items()}, sess.get_providers()


def find_engine(cache_dir, input_name, out_tail=None):
    """The cached engine whose first input binding is *input_name* (several graphs share a namespace dir)."""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    rt = trt.Runtime(logger)
    hits = []
    for f in sorted(glob.glob(os.path.join(cache_dir, "*.engine")), key=os.path.getmtime, reverse=True):
        try:
            eng = rt.deserialize_cuda_engine(open(f, "rb").read())
        except Exception:
            continue
        if eng is None:
            continue
        names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
        if input_name not in names:
            continue
        if out_tail is not None:        # 'data' is the input name of 2d106det, 1k3d68 and genderage: tell them apart by output
            outs = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)
                    if eng.get_tensor_mode(eng.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
            if not outs or tuple(eng.get_tensor_shape(outs[0]))[1:] != tuple(out_tail):
                continue
        hits.append((f, eng))
    return hits, rt


def inspect(eng):
    import tensorrt as trt
    insp = eng.create_engine_inspector()
    raw = insp.get_engine_information(trt.LayerInformationFormat.JSON)
    info = json.loads(raw)
    layers = info.get("Layers", [])
    names = [l if isinstance(l, str) else l.get("Name", "?") for l in layers]
    kinds = Counter()
    for l in layers:
        if isinstance(l, dict):
            kinds[l.get("LayerType", "?")] += 1
    bad = [n for n in names if any(t in n.lower() for t in ("transpose", "reformat", "shuffle", "permut", "copynode"))]
    return {"n_layers": len(layers), "layer_types": dict(kinds), "profiling_verbosity": str(eng.profiling_verbosity),
            "transpose_like": bad, "first": names[:6], "last": names[-4:],
            "io": [(eng.get_tensor_name(i), str(eng.get_tensor_mode(eng.get_tensor_name(i))),
                    list(eng.get_tensor_shape(eng.get_tensor_name(i))), str(eng.get_tensor_dtype(eng.get_tensor_name(i))),
                    str(eng.get_tensor_format(eng.get_tensor_name(i)))) for i in range(eng.num_io_tensors)],
            "raw_layers": layers}


def build_detailed(model_path, fp16, level, shape_override=None):
    """Diagnostic engine from the same ONNX: DETAILED verbosity so layer precisions/formats are recorded."""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(open(model_path, "rb").read(), os.path.dirname(model_path)):
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.builder_optimization_level = level
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    for i in range(net.num_inputs):
        t = net.get_input(i)
        if any(d < 0 for d in t.shape):
            shape = list((shape_override or {}).get(t.name) or [1 if d < 0 else d for d in t.shape])
            prof = builder.create_optimization_profile()
            prof.set_shape(t.name, shape, shape, shape)
            cfg.add_optimization_profile(prof)
    blob = builder.build_serialized_network(net, cfg)
    if blob is None:
        raise RuntimeError("detailed build failed")
    return trt.Runtime(logger).deserialize_cuda_engine(bytes(blob))


def layer_times(eng, runs=200, warm=30):
    """Per-layer GPU time of the DETAILED engine with TensorRT's own IProfiler (device buffers from torch)."""
    import tensorrt as trt
    import torch

    class P(trt.IProfiler):
        def __init__(self):
            trt.IProfiler.__init__(self)
            self.t = Counter()

        def report_layer_time(self, name, ms):
            self.t[name] += ms

    ctx = eng.create_execution_context()
    bufs = []
    for i in range(eng.num_io_tensors):
        name = eng.get_tensor_name(i)
        shape = tuple(ctx.get_tensor_shape(name))
        dt = torch.float32 if eng.get_tensor_dtype(name) == trt.DataType.FLOAT else torch.float16
        b = torch.rand(shape, dtype=dt, device="cuda") if eng.get_tensor_mode(name) == trt.TensorIOMode.INPUT else             torch.zeros(shape, dtype=dt, device="cuda")
        bufs.append(b)
        ctx.set_tensor_address(name, b.data_ptr())
    stream = torch.cuda.Stream()
    for _ in range(warm):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    wall = (time.perf_counter() - t0) * 1000.0 / runs
    prof = P()
    ctx.profiler = prof
    for _ in range(runs):
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
    per = {k: v / runs for k, v in prof.t.items()}
    return {"wall_ms_per_run": round(wall, 4), "sum_layer_ms": round(sum(per.values()), 4), "per_layer_ms": per}


def summarise(info):
    """Compact view of the detailed layer list: precision histogram and any layer that moves data."""
    prec = Counter()
    movers = []
    for l in info["raw_layers"]:
        if not isinstance(l, dict):
            continue
        outs = l.get("Outputs") or []
        for o in outs:
            prec[str(o.get("Format/Datatype", "?"))] += 1
        t = l.get("LayerType", "")
        nm = l.get("Name", "")
        if t in ("Shuffle", "Reformat") or "reformat" in nm.lower() or "transpose" in nm.lower():
            movers.append({"name": nm, "type": t, "in": [i.get("Format/Datatype") for i in l.get("Inputs") or []],
                           "out": [o.get("Format/Datatype") for o in outs]})
    return {"output_format_dtype_hist": dict(prec), "data_movers": movers}


def main():
    _init_pipeline()
    import roop.globals as g
    out = {"trt_precision": getattr(g.CFG, "trt_precision", None), "date": time.strftime("%Y-%m-%d %H:%M"), "models": {}}

    # ---- XSeg through its own loader --------------------------------------------------------------------------
    from roop.processors.Mask_XSeg import Mask_XSeg
    xs = Mask_XSeg()
    xs.Initialize({"devicename": "cuda"})
    xs.Run(np.zeros((256, 256, 3), np.uint8), "")                                  # the engine builds/loads on first inference
    targets = [("xseg", os.path.join(APP, "models", "xseg.onnx"), xs.model_xseg, (1, 256, 256, 3))]

    # ---- buffalo bundle through face_util ---------------------------------------------------------------------
    from roop import face_util
    face_util._ensure_face_analyser()
    fa = face_util.get_face_analyser()
    for task, stem, shape in (("recognition", "w600k_r50", (1, 3, 112, 112)), ("landmark_2d_106", "2d106det", (1, 3, 192, 192))):
        m = fa.models.get(task)
        if m is not None:
            targets.append((stem, getattr(m, "model_file", None), m.session, shape))
    lm68 = fa.models.get("landmark_3d_68") or getattr(fa, "lm68_model", None)
    if lm68 is not None:
        targets.append(("1k3d68", getattr(lm68, "model_file", None), lm68.session, (1, 3, 192, 192)))

    for stem, mpath, sess, shape in targets:
        rec = {"model_file": mpath}
        rec["live"] = live_facts(sess)
        # warm every session once so the engine file exists
        try:
            name = sess.get_inputs()[0].name
            sess.run(None, {name: np.zeros(shape, np.float32)})
        except Exception as exc:
            rec["warm_error"] = repr(exc)
        opts = rec["live"]["options"]
        rec["fp16"] = str(opts.get("trt_fp16_enable")).lower() in ("1", "true")
        rec["level"] = int(opts.get("trt_builder_optimization_level", 3) or 3)
        cache = opts.get("trt_engine_cache_path")
        if cache and sess.get_inputs():
            _o = sess.get_outputs()[0].shape
            hits, _rt = find_engine(cache, sess.get_inputs()[0].name,
                                    [d for d in _o[1:]] if all(isinstance(d, int) for d in _o[1:]) else None)
            rec["cached_engines"] = [os.path.basename(f) for f, _ in hits]
            if hits:
                f, eng = hits[0]
                info = inspect(eng)
                raw = info.pop("raw_layers")
                rec["cached_engine"] = {"file": os.path.basename(f), "bytes": os.path.getsize(f), **info}
                rec["cached_engine"]["layers_json_head"] = raw[:3]
        try:
            rec["ort_profile_nodes_per_provider"], _ = profile_nodes(mpath, [p for p in sess.get_providers()] if False else
                                                                      _chain_like(sess), shape)
        except Exception as exc:
            rec["profile_error"] = repr(exc)[:300]
        if mpath and os.path.exists(mpath):
            try:
                eng = build_detailed(mpath, rec["fp16"], rec["level"],
                                     {sess.get_inputs()[0].name: list(shape)})
                d = inspect(eng)
                raw = d.pop("raw_layers")
                rec["detailed_engine"] = {k: d[k] for k in ("n_layers", "layer_types", "transpose_like", "io",
                                                            "first", "last", "profiling_verbosity")}
                rec["detailed_engine"].update(summarise({"raw_layers": raw}))
                rec["detailed_engine"]["float_output_layers"] = [
                    l.get("Name") for l in raw if isinstance(l, dict) and any(
                        str(o.get("Format/Datatype", "")).startswith("Float") for o in (l.get("Outputs") or []))]
                lt = layer_times(eng)
                per = lt.pop("per_layer_ms")
                movers = {m["name"]: m["type"] for m in rec["detailed_engine"]["data_movers"]}
                by = Counter()
                for n, ms in per.items():
                    by[movers.get(n, "compute")] += ms
                lt["ms_by_kind"] = {k: round(v, 4) for k, v in by.items()}
                lt["top5_layers_ms"] = sorted(((round(v, 4), n) for n, v in per.items()), reverse=True)[:5]
                lt["input_shuffle_ms"] = {n: round(ms, 5) for n, ms in per.items() if movers.get(n) == "Shuffle"}
                order = [l.get("Name") for l in raw if isinstance(l, dict)]
                lt["head_ms"] = [(n, round(per.get(n, 0.0), 5)) for n in order[:5]]       # engine order: what the input costs
                lt["tail_ms"] = [(n, round(per.get(n, 0.0), 5)) for n in order[-4:]]
                rec["detailed_engine"]["layer_times"] = lt
            except Exception as exc:
                rec["detailed_error"] = repr(exc)[:300]
        out["models"][stem] = rec
        print("[probe] %-10s providers=%s fp16=%s level=%s layers(cached)=%s transpose_like(cached)=%s" % (
            stem, rec["live"]["providers"], rec["fp16"], rec["level"],
            (rec.get("cached_engine") or {}).get("n_layers"), (rec.get("cached_engine") or {}).get("transpose_like")), flush=True)
    label = sys.argv[1] if len(sys.argv) > 1 else ""
    out["label"] = label or "env"
    out["per_model_precision"] = {k: v["live"]["options"].get("trt_fp16_enable") for k, v in out["models"].items()}
    path = os.path.join(REPO, "docs", "perf", "trt_precision_probe%s.json" % (("_" + label) if label else ""))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1, default=str)
    print("wrote", path)
    return 0


_NUMERIC_KEYS = ("trt_builder_optimization_level", "trt_auxiliary_streams", "trt_max_partition_iterations",
                 "trt_min_subgraph_size", "trt_dla_core", "trt_max_workspace_size", "trt_onnx_model_bytes_size")


_SETTABLE_KEYS = ("device_id", "trt_fp16_enable", "trt_layer_norm_fp32_fallback", "trt_build_heuristics_enable",
                  "trt_builder_optimization_level", "trt_engine_cache_enable", "trt_engine_cache_path",
                  "trt_timing_cache_enable", "trt_timing_cache_path", "trt_force_sequential_engine_build",
                  "trt_max_workspace_size")


def _chain_like(sess):
    """Rebuild the provider chain the live session was created with (names + live options)."""
    chain = []
    allopts = sess.get_provider_options() or {}
    for p in sess.get_providers():
        o = {k: v for k, v in (allopts.get(p) or {}).items() if k in _SETTABLE_KEYS}
        for k, v in list(o.items()):                     # ORT reports booleans as '0'/'1' but accepts only 'True'/'False'
            if str(v) in ("0", "1") and k.startswith("trt_") and k not in _NUMERIC_KEYS:
                o[k] = "True" if str(v) == "1" else "False"
        chain.append((p, o) if o else p)
    return chain


if __name__ == "__main__":
    sys.exit(main())
