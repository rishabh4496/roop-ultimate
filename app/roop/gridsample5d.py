"""Rewrite 5-D GridSample nodes into ops TensorRT can actually build.

Why this exists
---------------
LivePortrait's `warping_spade.onnx` warps a 5-D feature volume `(1,32,16,64,64)`
with two `GridSample` nodes. Nothing in the GPU stack will execute them:

  * TensorRT's `IGridSampleLayer` is documented 4-D only, so its parser rejects
    both nodes (`addGridSample ... nbDims == 4`, `INVALID_NODE`).
  * onnxruntime's CUDA GridSample kernel is likewise 4-D only and fails at run
    time with "Only 4-D tensor is supported".

onnxruntime's answer is to partition just those two nodes to the CPU provider.
That works, but profiling the CPU kernel on this project's shapes shows what it
costs per call:

    /dense_motion_network/GridSample   (22,4,16,64,64)   134 ms
    /GridSample                        (1,32,16,64,64)    12 ms

146 ms of a 165 ms warm warping call — against 1.5 ms for the identical maths on
the GPU. The gap is not the device alone: onnxruntime's CPU GridSample is ~20x
slower than torch's on the same shapes.

The fix
-------
Trilinear sampling is not a primitive — it is eight corner reads and a weighted
sum, all of which TensorRT builds natively. So each 5-D GridSample is replaced
in the graph by that expansion, and the whole model becomes a single TensorRT
engine with no CPU partition and no device round-trips.

Measured on an RTX 4070 (warping module only, FP16 TensorRT): 165.2 ms -> 26.8
ms, a 6.16x gain, with no loss of accuracy against an FP32 CPU reference — the
rewritten graph's error is 4.71e-02 where the stock TensorRT path's is 4.99e-02.
A whole restore call goes 237.8 ms -> 34.0 ms per face. CUDA-only machines,
which cannot run the stock module at all, get 87.8 ms.

The win exceeds the 146 ms of CPU kernel time because the partition cost more
than the kernels: profiling the provider assignment shows the stock model runs
as *nine* separate TensorRT subgraphs wrapped around the two CPU nodes, with a
device round-trip at every boundary. The rewritten model is one engine.

The expansion follows the ONNX GridSample spec exactly:

    grid[..., 0] -> x (the W axis), [..., 1] -> y (H), [..., 2] -> z (D)

    align_corners=0:  i = ((g + 1) * size - 1) / 2
    align_corners=1:  i = (g + 1) / 2 * (size - 1)

    out = sum over the 8 corners of  X[clamp(corner)] * w_x * w_y * w_z * valid

`padding_mode="zeros"` is what makes the `valid` term a plain multiply: an
out-of-range corner contributes nothing, so the index can be clamped for the
gather and the contribution zeroed afterwards.

The gather uses `GatherND` with `batch_dims=1` over a `(N, S, C)` view rather
than `GatherElements` over `(N, C, S)`. Both are correct; GatherND needs one
index per sample point instead of one per sample point *per channel*, which on
the big node is 92 MB of index tensors instead of 184 MB, and it drops the eight
`Expand` nodes that broadcasting would otherwise need.

Only `mode="bilinear"` + `padding_mode="zeros"` are rewritten. Anything else is
left untouched for onnxruntime to place as it sees fit — a wrong rewrite would
be far worse than a slow one.
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

__all__ = ["patch_gridsample_5d", "ensure_patched_model", "PATCHED_SUFFIX",
           "PATCH_VERSION"]

PATCHED_SUFFIX = "-trt.onnx"

# Bump whenever the expansion changes. The cached file is keyed on this as well
# as the source's mtime: without it, editing the rewrite above would silently
# keep serving a graph built by the previous version.
PATCH_VERSION = 3


def _static_shape(value_info):
    """A fully-resolved integer shape, or None if any dimension is symbolic."""
    dims = []
    for d in value_info.type.tensor_type.shape.dim:
        if d.HasField("dim_value") and d.dim_value > 0:
            dims.append(d.dim_value)
        else:
            return None
    return dims


def _collect_shapes(model):
    inferred = shape_inference.infer_shapes(model, strict_mode=False)
    shapes = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + \
            list(inferred.graph.output):
        s = _static_shape(vi)
        if s is not None:
            shapes[vi.name] = s
    return shapes


def _attrs(node):
    out = {}
    for a in node.attribute:
        out[a.name] = a.s.decode() if a.type == onnx.AttributeProto.STRING else a.i
    return out


class _Builder:
    """Accumulates nodes and initializers under a unique name prefix."""

    def __init__(self, prefix):
        self.p = prefix
        self.nodes = []
        self.inits = []
        self._n = 0

    def name(self, tag):
        self._n += 1
        return f"{self.p}{tag}_{self._n}"

    def const(self, tag, array, dtype=np.float32):
        n = self.name(tag)
        self.inits.append(numpy_helper.from_array(np.asarray(array, dtype), n))
        return n

    def op(self, op_type, inputs, tag, **kwargs):
        out = self.name(tag)
        self.nodes.append(helper.make_node(op_type, list(inputs), [out],
                                           name=out + "_node", **kwargs))
        return out

    def op_multi(self, op_type, inputs, tags, **kwargs):
        outs = [self.name(t) for t in tags]
        self.nodes.append(helper.make_node(op_type, list(inputs), outs,
                                           name=outs[0] + "_node", **kwargs))
        return outs


def _expand_one(node, x_shape, g_shape, prefix):
    """Build the replacement subgraph for a single 5-D GridSample node.

    Returns (nodes, initializers). The final node writes the original node's
    output name, so the rest of the graph is untouched.
    """
    n, c, d_in, h_in, w_in = x_shape
    _, d_out, h_out, w_out, _ = g_shape
    a = _attrs(node)
    align = int(a.get("align_corners", 0))
    x_name, g_name = node.input[0], node.input[1]
    y_name = node.output[0]

    s_in = d_in * h_in * w_in          # flattened input volume
    m = d_out * h_out * w_out          # flattened sample count
    b = _Builder(prefix)

    # ── grid -> three coordinate channels, each (N, M, 1) ────────────────────
    g2 = b.op("Reshape", [g_name, b.const("grid_shape", [n, m, 3], np.int64)], "grid2d")
    gx, gy, gz = b.op_multi("Split", [g2, b.const("split3", [1, 1, 1], np.int64)],
                            ["gx", "gy", "gz"], axis=2)

    one = b.const("one", 1.0)
    zero = b.const("zero", 0.0)

    # ── per-axis: unnormalize, split into lower/upper corner, weight, validity ─
    axes = {}
    for tag, g_ch, size in (("x", gx, w_in), ("y", gy, h_in), ("z", gz, d_in)):
        if align:
            scale, offset = (size - 1) / 2.0, (size - 1) / 2.0
        else:
            scale, offset = size / 2.0, (size - 1) / 2.0
        t = b.op("Mul", [g_ch, b.const(f"scale_{tag}", scale)], f"scaled_{tag}")
        coord = b.op("Add", [t, b.const(f"offset_{tag}", offset)], f"coord_{tag}")

        lower = b.op("Floor", [coord], f"floor_{tag}")
        frac = b.op("Sub", [coord, lower], f"frac_{tag}")          # weight of upper
        w_lo = b.op("Sub", [one, frac], f"wlo_{tag}")

        hi_bound = b.const(f"hi_{tag}", size - 1)
        corners = []
        for step in (0, 1):
            pos = lower if step == 0 else \
                b.op("Add", [lower, one], f"upper_{tag}")
            ge = b.op("GreaterOrEqual", [pos, zero], f"ge_{tag}{step}")
            le = b.op("LessOrEqual", [pos, hi_bound], f"le_{tag}{step}")
            in_range = b.op("And", [ge, le], f"in_{tag}{step}")
            valid = b.op("Cast", [in_range], f"valid_{tag}{step}", to=TensorProto.FLOAT)
            clamped = b.op("Clip", [pos, zero, hi_bound], f"clamp_{tag}{step}")
            # Substituting 0 for an out-of-range corner costs nothing (its weight
            # is already 0) and makes the cast below total: casting a NaN to an
            # integer is undefined and would gather from an arbitrary offset.
            safe = b.op("Where", [in_range, clamped, zero], f"safe_{tag}{step}")
            index = b.op("Cast", [safe], f"idx_{tag}{step}", to=TensorProto.INT32)
            corners.append((index, w_lo if step == 0 else frac, valid))
        axes[tag] = corners

    # ── data as (N, S, C) so one index serves every channel ──────────────────
    xf = b.op("Reshape", [x_name, b.const("x_shape", [n, c, s_in], np.int64)], "xflat")
    xt = b.op("Transpose", [xf], "xt", perm=[0, 2, 1])

    # The flattened index is built in INT32, never in float. This is not a
    # micro-optimisation: TensorRT runs this graph with FP16 enabled, and the
    # big node addresses a volume of 16*64*64 = 65536 voxels. FP16's largest
    # value is 65504 and its spacing up there is 32, so a float index would be
    # rounded to a different voxel — or overflow to inf — and the module would
    # sample essentially at random. This was measured, not anticipated: with the
    # index in float the module was 6.7x faster and produced 0.71 max abs error
    # on a [0,1] output, while agreeing to 3e-06 on CPU where everything is FP32.
    h_const = b.const("h_in", h_in, np.int32)
    w_const = b.const("w_in", w_in, np.int32)

    contribs = []
    for cz, wz, vz in axes["z"]:
        zh = b.op("Mul", [cz, h_const], "zh")                 # z * H
        for cy, wy, vy in axes["y"]:
            zy = b.op("Mul", [b.op("Add", [zh, cy], "zy"), w_const], "zyw")
            wzy = b.op("Mul", [wz, wy], "wzy")
            vzy = b.op("Mul", [vz, vy], "vzy")
            for cx, wx, vx in axes["x"]:
                lin = b.op("Add", [zy, cx], "lin")            # ... + x
                idx = b.op("Cast", [lin], "idx", to=TensorProto.INT64)
                gathered = b.op("GatherND", [xt, idx], "gathered", batch_dims=1)
                weight = b.op("Mul", [b.op("Mul", [wzy, wx], "wzyx"),
                                      b.op("Mul", [vzy, vx], "vzyx")], "weight")
                contribs.append(b.op("Mul", [gathered, weight], "contrib"))

    acc = b.op("Sum", contribs, "acc")
    acc_t = b.op("Transpose", [acc], "accT", perm=[0, 2, 1])
    b.nodes.append(helper.make_node(
        "Reshape", [acc_t, b.const("out_shape", [n, c, d_out, h_out, w_out], np.int64)],
        [y_name], name=prefix + "out_node"))
    return b.nodes, b.inits


def patch_gridsample_5d(model):
    """Replace every rewritable 5-D GridSample in `model`.

    Returns (model, count). The input model is not modified. count is how many
    nodes were rewritten — 0 means nothing was eligible and the model is
    returned unchanged.
    """
    shapes = _collect_shapes(model)
    graph = model.graph
    targets = []
    for idx, node in enumerate(graph.node):
        if node.op_type != "GridSample":
            continue
        x_shape = shapes.get(node.input[0])
        g_shape = shapes.get(node.input[1])
        if not x_shape or not g_shape or len(x_shape) != 5 or len(g_shape) != 5:
            continue
        a = _attrs(node)
        if a.get("mode", "bilinear") != "bilinear":
            continue
        if a.get("padding_mode", "zeros") != "zeros":
            continue
        targets.append((idx, node, x_shape, g_shape))

    if not targets:
        return model, 0

    # Rewrite back-to-front so the earlier indices stay valid.
    for idx, node, x_shape, g_shape in reversed(targets):
        safe = "".join(ch if ch.isalnum() else "_" for ch in (node.name or f"gs{idx}"))
        nodes, inits = _expand_one(node, x_shape, g_shape, f"gs5d_{idx}_{safe}_")
        del graph.node[idx]
        for offset, new in enumerate(nodes):
            graph.node.insert(idx + offset, new)
        graph.initializer.extend(inits)

    return model, len(targets)


def ensure_patched_model(src_path, verbose=True):
    """Return a path to a GridSample-free copy of `src_path`, building it once.

    The result is cached next to the source as `<name>-trt.onnx`, alongside a
    `.meta` stamp recording PATCH_VERSION and the source's mtime; the cache is
    reused only when both still match. Any failure returns `src_path` unchanged:
    a slow expression restore is a far better outcome than a broken one.
    """
    try:
        out_path = os.path.splitext(src_path)[0] + PATCHED_SUFFIX
        meta_path = out_path + ".meta"
        stamp = f"{PATCH_VERSION}\n{os.path.getmtime(src_path):.6f}\n"
        if os.path.exists(out_path) and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                if fh.read() == stamp:
                    return out_path

        if verbose:
            print("[Expression] Rewriting the 5-D GridSample nodes so the warping "
                  "module can build as a single TensorRT engine (one-off)...")
        model, count = patch_gridsample_5d(onnx.load(src_path))
        if count == 0:
            if verbose:
                print("[Expression] No rewritable 5-D GridSample found — using the "
                      "model as shipped.")
            return src_path
        onnx.checker.check_model(model, full_check=False)
        tmp = out_path + ".tmp"
        onnx.save(model, tmp)
        os.replace(tmp, out_path)
        with open(meta_path, "w", encoding="utf-8") as fh:
            fh.write(stamp)
        if verbose:
            print(f"[Expression] Rewrote {count} GridSample node(s) -> "
                  f"{os.path.basename(out_path)}")
        return out_path
    except Exception as e:
        print(f"[Expression] GridSample rewrite failed ({e}); falling back to the "
              f"stock model with those nodes on CPU.")
        return src_path
