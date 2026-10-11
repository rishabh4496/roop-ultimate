"""Convert the three auxiliary ONNX models (xseg, 2d106det, 1k3d68) to half precision, with per-strategy FP32 exclusions.

    app/env/Scripts/python.exe tools/convert_fp16.py --model all --strategy norm_head
    app/env/Scripts/python.exe tools/convert_fp16.py --model 1k3d68 --strategy head --out some/dir
    app/env/Scripts/python.exe tools/convert_fp16.py --list

The result is an ONNX file with FP16 weights and compute and FP32 graph inputs/outputs (`keep_io_types=True`), so every
caller's pre/post-processing is unchanged. It is written to `app/models/fp16/<stem>.<strategy>.onnx` (never over the shipped
FP32 file - those stay the reference the fidelity harness compares against).

This tool only CONVERTS. Whether a variant may be used is decided by `app/tests/trt_precision_fidelity.py onnx16`, against
the gates the earlier briefs set (xseg IoU >= 0.995 on >= 99% of faces and none < 0.95; landmark p95 <= 0.3 frame px).
`docs/perf/aux_fp16_onnx_2026-10-11.md` records which strategy cleared them (none, if that file says none).

STRATEGIES (op types kept in FP32 - onnxconverter_common wraps each such node in Cast pairs):
  all        nothing excluded: the plain FP16 conversion.
  norm       the normalisation statistics. XSeg's instance norm is decomposed into GlobalAveragePool / Max / Sqrt /
             Reciprocal; a variance in FP16 loses its low bits, and Reciprocal of a small one overflows.
  head       the regression/output layer: Gemm (2d106det, 1k3d68 - the 212 / 3 x 68 coordinates) and Sigmoid (XSeg's mask).
  norm_head  both.
  bn         BatchNormalization too (the landmark nets fold it into the preceding conv at export, so this only matters if an
             exporter left it unfused).
"""
import argparse
import collections
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MODELS = os.path.join(REPO, "app", "models")

# model stem -> path under app/models. The same files trt_precision_fidelity.py uses as its FP32 reference.
FILES = {"xseg": "xseg.onnx", "2d106det": "buffalo_l/2d106det.onnx", "1k3d68": "buffalo_l/1k3d68.onnx"}

_NORM = ("GlobalAveragePool", "Sqrt", "Reciprocal", "Max", "ReduceMean", "InstanceNormalization", "LayerNormalization")
_HEAD = ("Gemm", "Sigmoid", "Softmax")
STRATEGIES = {
    "all": (),
    "norm": _NORM,
    "head": _HEAD,
    "norm_head": _NORM + _HEAD,
    "bn": _NORM + _HEAD + ("BatchNormalization",),
}


def out_path(stem, strategy, out_dir=None):
    return os.path.join(out_dir or os.path.join(MODELS, "fp16"), "%s.%s.onnx" % (stem, strategy))


def _io(model):
    from onnx import TensorProto as T
    return [(i.name, T.DataType.Name(i.type.tensor_type.elem_type)) for i in list(model.graph.input) + list(model.graph.output)]


def _force_fp32_io(model):
    """onnxconverter_common leaves a graph OUTPUT float16 when its producer is an op in the block list's neighbourhood
    (observed: Gemm kept FP32 -> output FLOAT16). Every caller reads float32, so cast it back under the original name."""
    from onnx import TensorProto as T, helper
    for out in model.graph.output:
        if out.type.tensor_type.elem_type != T.FLOAT16:
            continue
        inner = out.name + "__fp16"
        for node in model.graph.node:
            for i, o in enumerate(node.output):
                if o == out.name:
                    node.output[i] = inner
        model.graph.node.append(helper.make_node("Cast", [inner], [out.name], to=T.FLOAT, name="Cast_out_" + out.name))
        out.type.tensor_type.elem_type = T.FLOAT
    return model


def convert(stem, strategy="norm_head", out_dir=None, extra_block=()):
    """Write the FP16 variant and return (path, report). Raises if the result is not a valid FP16 graph."""
    import onnx
    from onnx import TensorProto as T
    from onnxconverter_common import float16

    if stem not in FILES:
        raise ValueError("unknown model %r (choose from %s)" % (stem, ", ".join(FILES)))
    if strategy not in STRATEGIES:
        raise ValueError("unknown strategy %r (choose from %s)" % (strategy, ", ".join(STRATEGIES)))
    src = os.path.join(MODELS, FILES[stem])
    model = onnx.load(src)
    if any(i.data_type == T.FLOAT16 for i in model.graph.initializer):
        raise ValueError("%s is already half precision - convert the FP32 original" % src)
    present = collections.Counter(n.op_type for n in model.graph.node)
    block = [op for op in (STRATEGIES[strategy] + tuple(extra_block)) if present.get(op)]
    # max_finite_val: the converter's default (1e4) CLAMPS any larger constant - 1k3d68 holds a 52808.9 that FP16 (max 65504)
    # represents fine and the default silently turned into 10000. min_positive_val 1e-7 flushes the BN-fold epsilons to 1e-7;
    # they are counted, not hidden.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        half = float16.convert_float_to_float16(model, min_positive_val=1e-7, max_finite_val=65504.0,
                                                keep_io_types=True, op_block_list=block or None,
                                                disable_shape_infer=False)
    clipped = [str(w.message) for w in caught if "truncated" in str(w.message)]
    half = _force_fp32_io(half)
    onnx.checker.check_model(half)
    n16 = sum(1 for i in half.graph.initializer if i.data_type == T.FLOAT16)
    n32 = sum(1 for i in half.graph.initializer if i.data_type == T.FLOAT)
    bad_io = [(n, t) for n, t in _io(half) if t != "FLOAT"]
    if bad_io:
        raise ValueError("%s/%s: graph I/O is not FLOAT after conversion: %s" % (stem, strategy, bad_io))
    # a conversion that kept every weight FP32 would "pass" the fidelity gate for the wrong reason
    if n16 == 0:
        raise ValueError("%s/%s: no initializer became float16 - nothing was converted" % (stem, strategy))
    dst = out_path(stem, strategy, out_dir)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    onnx.save(half, dst)
    report = {"stem": stem, "strategy": strategy, "blocked_ops": block,
              "blocked_nodes": sum(present[o] for o in block), "nodes": sum(present.values()),
              "fp16_initializers": n16, "fp32_initializers": n32,
              "io": _io(half), "clamped_constants": len(clipped),
              "largest_clamped": max([float(m.split("number ")[1].split(" ")[0]) for m in clipped if "truncated to 10000" in m]
                                     or [0.0]),
              "bytes": os.path.getsize(dst), "path": dst}
    return dst, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="all", help="xseg | 2d106det | 1k3d68 | all")
    ap.add_argument("--strategy", default="norm_head", choices=sorted(STRATEGIES))
    ap.add_argument("--block", default="", help="extra op types to keep FP32, comma separated")
    ap.add_argument("--out", default=None, help="output directory (default app/models/fp16)")
    ap.add_argument("--list", action="store_true", help="print the strategies and exit")
    args = ap.parse_args(argv)
    if args.list:
        for k, v in STRATEGIES.items():
            print("%-10s %s" % (k, ", ".join(v) or "(nothing excluded)"))
        return 0
    stems = list(FILES) if args.model == "all" else [args.model]
    extra = tuple(s.strip() for s in args.block.split(",") if s.strip())
    for stem in stems:
        _, rep = convert(stem, args.strategy, args.out, extra)
        print("[fp16] %-9s %-9s fp16 weights %3d  fp32 weights %3d  kept-FP32 nodes %3d/%3d (%s)  io %s  clamped consts %d (max %.0f)  %.1f MB -> %s" % (
            stem, args.strategy, rep["fp16_initializers"], rep["fp32_initializers"], rep["blocked_nodes"], rep["nodes"],
            ",".join(rep["blocked_ops"]) or "-", [t for _, t in rep["io"]], rep["clamped_constants"], rep["largest_clamped"], rep["bytes"] / 1e6, rep["path"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
