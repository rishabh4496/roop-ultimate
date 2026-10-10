"""Build native TensorRT engines of the 512 restorer with chosen FP32 islands inside an FP16 network.

    app\\env\\Scripts\\python.exe tools/restorer_native_engine.py strip
    app\\env\\Scripts\\python.exe tools/restorer_native_engine.py build NAME [--fp32-ops InstanceNormalization,Softmax]
                                                              [--fp32-modules /encoder/attn.0,/decoder/conv_out] [--all-fp32]
                                                              [--all-fp32] [--model PATH]

Why not ``tools/build_trt_engines.py --native``: that path classifies every node by op type (norms, softmax and the final
conv are pinned, the rest left to "auto"), tries fp8 / int8 tiers first, and runs a polygraphy validation loop that
blacklists nodes. For an A/B of WHICH layers need FP32 the builder must do exactly one thing - apply the island set it is
given, obey it, and nothing else - so this builds with the TensorRT API directly, with the same ONNX, FP16 flag, builder
level (3) and 4 GiB workspace as the production provider. FP32 I/O, static 1x3x512x512, so the engine drops into the same
harness as the ORT sessions (``app/roop/trt_native_runner.NativeEngine``).

A layer is constrained when its name is under a chosen module path (``/encoder/attn.0/...``) or starts with the name of a
chosen op type's ONNX node. ``OBEY_PRECISION_CONSTRAINTS`` makes it a requirement: the build fails rather than quietly
running the layer in FP16. Engines land in app/output/restorer_fp16/engines/NAME.engine (+ NAME.json with the island list).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app"
for p in (str(APP), str(APP / "tests"), str(REPO / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = APP / "output" / "restorer_fp16"
ENGINES = OUT / "engines"
SINGLE = OUT / "restoreformer_single_output.onnx"
SKIP_TYPES = ("SHAPE", "CONSTANT", "ASSERTION", "FILL", "NON_ZERO", "ONE_HOT", "TOPK", "ARG")


def _prep_runtime():
    # the app puts the TensorRT DLLs on PATH; a bare process cannot import tensorrt otherwise
    import angle_bench as ab
    from settings import Settings
    cfg = Settings(str(APP / "config.yaml"))
    ab.init_pipeline(cfg.provider, cfg.swap_model, None, None, sync_config=True)


def cmd_strip(args):
    import onnx
    model = onnx.load(str(APP / "models" / "restoreformer_plus_plus.onnx"))
    for o in list(model.graph.output)[1:]:
        model.graph.output.remove(o)
    del model.graph.value_info[:]
    OUT.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(SINGLE))
    print("wrote", SINGLE, "outputs:", [o.name for o in model.graph.output])


def _node_prefixes(model_path, ops):
    import onnx
    m = onnx.load(model_path, load_external_data=False)
    return {n.name for n in m.graph.node if n.op_type in ops}


def _anchor_names(model_path, ops, modules, min_rank):
    """ONNX tensor names to expose as FP32 outputs: float tensors of rank >= min_rank produced by `ops` nodes (any module, or
    only under `modules` when given). Needs shape inference for the dtypes."""
    import onnx
    m = onnx.load(model_path)
    inferred = onnx.shape_inference.infer_shapes(m)
    info = {v.name: v.type.tensor_type for v in list(inferred.graph.value_info) + list(inferred.graph.output)}
    out = []
    for n in m.graph.node:
        if n.op_type not in ops or (modules and not any(n.name.startswith(mod.rstrip("/") + "/") for mod in modules)):
            continue
        t = info.get(n.output[0])
        if t is not None and t.elem_type == 1 and len(t.shape.dim) >= min_rank:
            out.append(n.output[0])
    return out


def cmd_build(args):
    _prep_runtime()
    import tensorrt as trt
    model_path = args.model or str(SINGLE)
    if not os.path.isfile(model_path):
        raise SystemExit(f"{model_path} missing: run `strip` first")
    ops = [o for o in args.fp32_ops.split(",") if o]
    except_ops = [o for o in args.fp32_except_ops.split(",") if o]
    except_nodes = tuple(_node_prefixes(model_path, set(except_ops))) if except_ops else ()
    modules = [m for m in args.fp32_modules.split(",") if m]
    op_nodes = tuple(_node_prefixes(model_path, set(ops))) if ops else ()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(model_path):
        raise SystemExit("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.builder_optimization_level = args.opt_level
    if not args.all_fp32:
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    cache_path = OUT / ("timing.cache" if not args.no_timing_cache else f"timing_{args.name}.cache")
    blob = cache_path.read_bytes() if (cache_path.is_file() and not args.no_timing_cache) else b""
    tcache = config.create_timing_cache(blob)
    config.set_timing_cache(tcache, ignore_mismatch=False)

    anchors = [a for a in args.anchors.split(",") if a]
    if args.anchor_ops:
        anchors += _anchor_names(model_path, set(args.anchor_ops.split(",")),
                                 [m for m in args.anchor_modules.split(",") if m], 4)
    primary = network.get_output(0).name if network.num_outputs else ""
    exposed, missing = [], []
    if anchors:
        tensors = {}
        for i in range(network.num_layers):
            lyr = network.get_layer(i)
            for k in range(lyr.num_outputs):
                tensors[lyr.get_output(k).name] = lyr.get_output(k)
        for name in dict.fromkeys(anchors):
            t = tensors.get(name)
            if t is None:
                missing.append(name)
                continue
            network.mark_output(t)
            if t.dtype in (trt.float32, trt.float16):       # an index tensor (int64) stays what it is
                t.dtype = trt.float32
            exposed.append(name)
        if not exposed:
            raise SystemExit(f"{args.name}: none of the {len(anchors)} anchor tensors exist in the TensorRT network")
    constrained, failed = [], []
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        name = layer.name
        hit = any(name.startswith(m.rstrip("/") + "/") or name == m for m in modules) \
            or any(name == n or name.startswith(n + "_") or name.startswith(n + "/") for n in op_nodes)
        if except_ops:       # complement: every layer that does NOT belong to one of these op types
            hit = hit or not any(name == n or name.startswith(n + "_") or name.startswith(n + "/") for n in except_nodes)
        if not hit or any(t in str(layer.type) for t in SKIP_TYPES):
            continue
        try:
            if any(layer.get_output(k).dtype != trt.float32 and layer.get_output(k).dtype != trt.float16
                   for k in range(layer.num_outputs)):
                continue
            layer.precision = trt.float32
            for k in range(layer.num_outputs):
                layer.set_output_type(k, trt.float32)
            constrained.append(name)
        except Exception as exc:                                    # a layer that cannot take a precision is reported, not hidden
            failed.append((name, str(exc)[:80]))
    if (modules or ops or except_ops) and not constrained:
        # e.g. Git Bash rewrites a lone "/encoder" argument into a Windows path (set MSYS_NO_PATHCONV=1); building anyway would
        # silently produce a plain FP16 engine labelled as an island.
        raise SystemExit(f"{args.name}: the requested islands matched no TensorRT layer (modules={modules}, ops={ops})")
    print(f"[build] {args.name}: anchors exposed as FP32 outputs: {len(exposed)} (missing {len(missing)})", flush=True)
    print(f"[build] {args.name}: {network.num_layers} layers; constrained to FP32: {len(constrained)}; "
          f"could not constrain: {len(failed)}; fp16={'off (all FP32)' if args.all_fp32 else 'on'}", flush=True)
    t0 = time.time()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise SystemExit("build failed")
    ENGINES.mkdir(parents=True, exist_ok=True)
    (ENGINES / f"{args.name}.engine").write_bytes(bytes(plan))
    cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))
    meta = {"name": args.name, "model": model_path, "fp16": not args.all_fp32, "opt_level": args.opt_level, "no_timing_cache": bool(args.no_timing_cache), "fp32_ops": ops, "fp32_except_ops": except_ops, "fp32_modules": modules, "primary_output": primary, "anchors": len(exposed), "anchors_missing": len(missing),
            "anchor_sample": exposed[:6],
            "constrained_layers": len(constrained), "failed_layers": failed[:20], "n_failed": len(failed),
            "build_seconds": round(time.time() - t0, 1), "engine_bytes": len(bytes(plan)),
            "tensorrt": trt.__version__, "constrained_names_sample": constrained[:12]}
    (ENGINES / f"{args.name}.json").write_text(json.dumps(meta, indent=1))
    print(f"[build] {args.name}: built in {meta['build_seconds']}s, {meta['engine_bytes'] / 1e6:.0f} MB", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("strip")
    b = sub.add_parser("build")
    b.add_argument("name")
    b.add_argument("--model", default="")
    b.add_argument("--fp32-ops", default="")
    b.add_argument("--fp32-modules", default="")
    b.add_argument("--opt-level", type=int, default=3, help="TensorRT builder optimization level (production: 3)")
    b.add_argument("--no-timing-cache", action="store_true", help="fresh tactic timings: a different draw of kernel choices")
    b.add_argument("--anchors", default="", help="comma list of ONNX tensor names to expose as FP32 network outputs")
    b.add_argument("--anchor-ops", default="", help="expose the float rank>=4 outputs of these ONNX op types (e.g. Add: the residual stream)")
    b.add_argument("--anchor-modules", default="", help="restrict --anchor-ops to nodes under these module paths")
    b.add_argument("--fp32-except-ops", default="", help="the complement: FP32 for every layer NOT from these ONNX op types")
    b.add_argument("--all-fp32", action="store_true", help="no FP16 flag at all (the native FP32 reference)")
    args = ap.parse_args()
    return {"strip": cmd_strip, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
