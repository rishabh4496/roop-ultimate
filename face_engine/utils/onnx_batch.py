"""Derived ONNX variants: a dynamic batch dimension, and FP16 weights.

Every swap / restore export in the zoo has batch 1 fixed in its input shape,
and most also fix it inside the graph (``Reshape`` targets such as
``[1, 1024, 1, 1]``). :func:`make_batch_dynamic` rewrites both:

* graph inputs and outputs get a symbolic ``batch`` dimension;
* a ``Reshape`` whose constant target starts with ``1`` gets ``-1`` there
  (let the runtime infer the batch) when the target has no other ``-1``,
  else ``0`` (copy the input's first dimension).

A rewrite is only a hypothesis: :func:`verify_batch_equivalence` runs B
different random inputs through the original model one at a time and through
the rewritten model as one batch and compares. :func:`batched_model` refuses
(returns None) when the rewrite fails or changes the numbers. Measured
2026-09-28, B=4 against 4 x B=1:

    hyperswap_1a_256            43 reshapes rewritten   max |diff| 3.5e-6   batchable
    inswapper_128                0                      0                    batchable as shipped
    restoreformer_plus_plus      0                      1.7e-4               batchable as shipped
    arcface_w600k_r50            0                      0                    batchable as shipped
    liveportrait_motion          0                      1.7e-6               batchable as shipped
    liveportrait_appearance      1                      0                    batchable
    gpen_bfr_512 / 1024        106 / 120                fails                NOT batchable
    liveportrait_warping         6                      fails                NOT batchable

GPEN is StyleGAN2: its modulated convolutions fold the batch into the
``Conv`` ``group`` attribute (``groups = batch``), a static attribute no
reshape rewrite can reach. Those models run once per face.

Derived files live next to the source model (``<stem>.batch.onnx``,
``<stem>.fp16.onnx``) with a sidecar recording the source's size and mtime,
so an updated source model is re-derived.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

BATCH_DIM = "batch"
# Bump when the rewrite or its verification changes, so derived files are rebuilt.
_RULE = 5


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".derived.json")


def _stamp(source: Path) -> dict[str, Any]:
    st = source.stat()
    return {"source": source.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _fresh(derived: Path, source: Path, extra: dict[str, Any]) -> dict[str, Any] | None:
    side = _sidecar(derived)
    if not (derived.exists() and side.exists()):
        return None
    try:
        record = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    wanted = {**_stamp(source), **extra}
    return record if all(record.get(k) == v for k, v in wanted.items()) else None


def make_batch_dynamic(model: Any) -> int:
    """Rewrite ``model`` (an ``onnx.ModelProto``) in place; returns reshapes rewritten."""
    from onnx import numpy_helper

    graph = model.graph
    initializers = {i.name: i for i in graph.initializer}
    for tensor in list(graph.input) + list(graph.output):
        if tensor.name in initializers:
            continue
        dims = tensor.type.tensor_type.shape.dim
        if len(dims) and (dims[0].dim_value == 1 or dims[0].dim_param):
            dims[0].ClearField("dim_value")
            dims[0].dim_param = BATCH_DIM
    # Stale per-tensor shapes would contradict the new batch dimension.
    del graph.value_info[:]

    constants = {n.output[0]: n for n in graph.node if n.op_type == "Constant"}
    consumers: dict[str, list[Any]] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    rewritten = 0
    for node in graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        name = node.input[1]
        if name in initializers:
            value = numpy_helper.to_array(initializers[name])
        elif name in constants:
            value = numpy_helper.to_array(constants[name].attribute[0].t)
        else:
            continue
        if not (value.ndim == 1 and value.size and value[0] == 1):
            continue
        if any(c.op_type != "Reshape" for c in consumers[name]):
            continue
        new = value.astype(np.int64).copy()
        new[0] = 0 if (new == -1).any() else -1
        new_name = f"{name}__batch"
        if new_name not in initializers:
            init = numpy_helper.from_array(new, new_name)
            graph.initializer.append(init)
            initializers[new_name] = init
        node.input[1] = new_name
        rewritten += 1
    return rewritten


def decompose_instance_norm(model: Any) -> int:
    """Replace every ``InstanceNormalization`` with primitive ops, in place.

    ``y = scale * (x - mean) / sqrt(var + eps) + bias``, computed on ``x``
    flattened to ``(N, C, L)`` (so any rank >= 3 works; RestoreFormer++ uses
    rank-3 instance norms) and reshaped back with ``Shape(x)``. HyperSwap has
    FP16 sections: there the statistics are computed in FP32 and the result
    cast back (``(x - mean)^2`` overflows FP16's 65504 on real activations,
    which the fused kernel avoids by accumulating in FP32).

    Needed because ORT's CUDA InstanceNormalization (and TensorRT's) is wrong
    at batch > 1 on HyperSwap's generator: at B=2 on a (2, 1024, 2, 2) tensor
    the CUDA result for row 0 differed from the B=1 result by 1.73 while the
    CPU kernel agreed to 1e-6 (2026-09-28; the batched swap's rows bled into
    each other by up to 136 levels). Returns the number of nodes replaced.
    """
    from onnx import TensorProto, helper, numpy_helper

    graph = model.graph
    opset = max((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")),
                default=13)
    dtypes: dict[str, int] = {i.name: i.data_type for i in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant" and node.attribute:
            dtypes[node.output[0]] = node.attribute[0].t.data_type

    def const(name: str, value: np.ndarray) -> str:
        graph.initializer.append(numpy_helper.from_array(value, name))
        return name

    def reduce_mean(inp: str, out: str, prefix: str) -> Any:
        if opset >= 18:
            return helper.make_node("ReduceMean", [inp, const(f"{prefix}_axes_{out[-4:]}",
                                                              np.array([2], np.int64))],
                                    [out], keepdims=1)
        return helper.make_node("ReduceMean", [inp], [out], axes=[2], keepdims=1)

    replaced = 0
    nodes = list(graph.node)
    del graph.node[:]
    for node in nodes:
        if node.op_type != "InstanceNormalization":
            graph.node.append(node)
            continue
        eps = next((helper.get_attribute_value(a) for a in node.attribute if a.name == "epsilon"),
                   1e-5)
        x, scale, bias = node.input
        y = node.output[0]
        p = f"{node.name or y}__in"
        elem = dtypes.get(scale, TensorProto.FLOAT)
        half = elem == TensorProto.FLOAT16
        flat_shape = const(f"{p}_flat", np.array([0, 0, -1], np.int64))
        param_shape = const(f"{p}_pshape", np.array([1, -1, 1], np.int64))
        eps_name = const(f"{p}_eps", np.array(eps, np.float64 if elem == TensorProto.DOUBLE
                                              else np.float32))
        pre, post = [], []
        x_in, sc, bi, out = x, scale, bias, y
        if half:  # compute in FP32, cast the result back
            pre = [helper.make_node("Cast", [x], [f"{p}_x32"], to=TensorProto.FLOAT),
                   helper.make_node("Cast", [scale], [f"{p}_s32"], to=TensorProto.FLOAT),
                   helper.make_node("Cast", [bias], [f"{p}_b32"], to=TensorProto.FLOAT)]
            x_in, sc, bi, out = f"{p}_x32", f"{p}_s32", f"{p}_b32", f"{p}_y32"
            post = [helper.make_node("Cast", [out], [y], to=TensorProto.FLOAT16)]
        graph.node.extend(pre)
        graph.node.extend([
            helper.make_node("Shape", [x_in], [f"{p}_xshape"]),
            helper.make_node("Reshape", [x_in, flat_shape], [f"{p}_x3"]),
            reduce_mean(f"{p}_x3", f"{p}_mean", p),
            helper.make_node("Sub", [f"{p}_x3", f"{p}_mean"], [f"{p}_centered"]),
            helper.make_node("Mul", [f"{p}_centered", f"{p}_centered"], [f"{p}_sq"]),
            reduce_mean(f"{p}_sq", f"{p}_var", p),
            helper.make_node("Add", [f"{p}_var", eps_name], [f"{p}_var_eps"]),
            helper.make_node("Sqrt", [f"{p}_var_eps"], [f"{p}_std"]),
            helper.make_node("Div", [f"{p}_centered", f"{p}_std"], [f"{p}_norm"]),
            helper.make_node("Reshape", [sc, param_shape], [f"{p}_scale"]),
            helper.make_node("Reshape", [bi, param_shape], [f"{p}_bias"]),
            helper.make_node("Mul", [f"{p}_norm", f"{p}_scale"], [f"{p}_scaled"]),
            helper.make_node("Add", [f"{p}_scaled", f"{p}_bias"], [f"{p}_y3"]),
            helper.make_node("Reshape", [f"{p}_y3", f"{p}_xshape"], [out]),
        ])
        graph.node.extend(post)
        replaced += 1
    return replaced


def _random_feeds(session: Any, rng: np.random.Generator) -> dict[str, np.ndarray]:
    feeds = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        data = rng.random(shape, dtype=np.float32) * 2.0 - 1.0
        if len(shape) == 2:  # identity vectors: unit norm, like the real input
            data /= np.linalg.norm(data, axis=1, keepdims=True)
        feeds[inp.name] = data
    return feeds


def verify_batch_equivalence(original: Path, batched: Path, batch: int = 4,
                             providers: list[str] | None = None,
                             rtol: float = 5e-3) -> float:
    """Check that ``batched`` at batch B computes what ``original`` does per sample.

    Reference: ``original`` on the CPU, one sample at a time. Noise floor:
    ``original`` on ``providers``, one at a time, against that reference.
    Candidate: ``batched`` on ``providers`` with all B samples together. The
    candidate passes when its error is within ``2 x`` the noise floor or
    ``rtol`` of the output range, whichever is larger.

    Why a noise floor instead of a fixed tolerance: random inputs drive these
    networks far outside their training range, where even the unmodified
    model on CUDA drifts from the CPU (HyperSwap: 5e-2 of range on random
    input, 0.4 levels on real faces). A real fault is far above both: the CUDA
    InstanceNorm batch bug read 1.0 of range.

    Returns the candidate's error (relative to the output range). Raises
    ``ValueError`` on failure, or whatever the runtime raises.
    """
    import onnxruntime as ort

    providers = providers or _verification_providers()
    cpu = ort.InferenceSession(str(original), providers=["CPUExecutionProvider"])
    one = ort.InferenceSession(str(original), providers=providers)
    many = ort.InferenceSession(str(batched), providers=providers)
    rng = np.random.default_rng(0)
    feeds = [_random_feeds(one, rng) for _ in range(batch)]
    reference = [cpu.run(None, f) for f in feeds]
    singles = [one.run(None, f) for f in feeds]
    stacked = many.run(None, {k: np.concatenate([f[k] for f in feeds]) for k in feeds[0]})
    worst, floor, compared = 0.0, 0.0, 0
    for out_index, out in enumerate(stacked):
        parts = [np.asarray(r[out_index]) for r in reference]
        if parts[0].ndim == 0 or parts[0].shape[0] != 1:
            continue  # not a per-sample output
        ref = np.concatenate(parts)
        if out.shape != ref.shape:
            raise ValueError(f"output {out_index}: batched shape {out.shape} != {ref.shape}")
        scale = max(float(np.abs(ref).max()), 1e-6)
        single = np.concatenate([np.asarray(s[out_index]) for s in singles])
        floor = max(floor, float(np.abs(single - ref).max()) / scale)
        worst = max(worst, float(np.abs(out - ref).max()) / scale)
        compared += 1
    if compared == 0:
        raise ValueError("no per-sample output to compare")
    allowed = max(2.0 * floor, rtol)
    if worst > allowed:
        raise ValueError(f"batched outputs differ from the reference by {worst:.2e} of range "
                         f"(the unbatched model on the same provider: {floor:.2e})")
    return worst


def _verification_providers() -> list[str]:
    """CUDA when available: a rewrite verified on the CPU passed for HyperSwap
    while the CUDA kernels computed it wrongly (see
    :func:`decompose_instance_norm`). Verify where it will run."""
    import onnxruntime as ort

    try:
        import torch

        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    if cuda and "CUDAExecutionProvider" in ort.get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def decomposed_norms(batched: Path) -> int:
    """InstanceNorms the batch rewrite replaced (from its sidecar); 0 if unknown."""
    try:
        return int(json.loads(_sidecar(batched).read_text(encoding="utf-8"))
                   .get("instance_norms", 0))
    except (OSError, ValueError):
        return 0


def batched_model(source: Path | str, *, fp16: bool = False) -> Path | None:
    """A verified dynamic-batch copy of ``source``, or None when the model
    cannot be batched.

    ``fp16=True`` (a TensorRT FP16 engine will be built from it) also returns
    None when the rewrite replaced InstanceNormalization nodes: their FP32
    statistics do not survive a TensorRT FP16 build. Measured 2026-09-28 on
    222 real-clip faces: HyperSwap identity 0.588 -> 0.334 and RestoreFormer++
    identity kept 0.808 -> 0.033, while the ORIGINAL graphs at FP16 matched
    FP32 (0.587, and see the README). FP16 callers run the original graph,
    one face per call.
    """
    import onnx

    source = Path(source)
    derived = source.with_name(f"{source.stem}.batch.onnx")
    record = _fresh(derived, source, {"kind": "batch", "rule": _RULE})
    if record is not None:
        if fp16 and record.get("instance_norms"):
            return None
        return derived if record.get("ok") else None
    model = onnx.load(str(source))
    rewritten = make_batch_dynamic(model)
    instance_norms = decompose_instance_norm(model)
    onnx.save(model, str(derived))
    ok, detail = True, ""
    try:
        detail = f"max rel error vs CPU reference {verify_batch_equivalence(source, derived):.2e}"
    except Exception as exc:  # noqa: BLE001 - any failure means "not batchable"
        ok, detail = False, f"{type(exc).__name__}: {str(exc)[:300]}"
        logger.info("%s is not batchable: %s", source.name, detail)
    _sidecar(derived).write_text(json.dumps(
        {**_stamp(source), "kind": "batch", "ok": ok, "rewritten": rewritten,
         "instance_norms": instance_norms, "verified_on": _verification_providers()[0],
         "detail": detail, "rule": _RULE}, indent=1), encoding="utf-8")
    if not ok:
        derived.unlink(missing_ok=True)
    if fp16 and instance_norms:
        return None
    return derived if ok else None


def fp16_model(source: Path | str) -> Path:
    """FP16-weight copy (ORT's converter; inputs and outputs stay FP32)."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    source = Path(source)
    derived = source.with_name(f"{source.stem}.fp16.onnx")
    if _fresh(derived, source, {"kind": "fp16"}) is None:
        model = convert_float_to_float16(onnx.load(str(source)), keep_io_types=True)
        onnx.save(model, str(derived))
        _sidecar(derived).write_text(json.dumps({**_stamp(source), "kind": "fp16"}, indent=1),
                                     encoding="utf-8")
    return derived



_PROFILES: dict[tuple[str, int, int, int], Any] = {}


def batch_shape_profile(model_path: Path | str, *, opt: int = 4, max_batch: int = 8) -> Any:
    """TensorRT profile ``1 / opt / max_batch`` for every graph input of a batched model.

    Read from the model itself, so no caller hand-writes input names and
    shapes; cached per file, because reading a 400 MB graph on every
    inference made the batched arms 7-25x slower (first benchmark run with
    this helper, 2026-09-28). Without a profile TensorRT cannot build a dynamic-batch engine:
    the batched LivePortrait nets failed on every call ("failed to create
    engine from network"), which the expression restorer's never-cost-a-frame
    fallback turned into a fake 16.5x speedup (2026-09-28).
    """
    import onnx

    from face_engine.core.execution import ShapeProfile

    path = Path(model_path)
    key = (str(path.resolve()), path.stat().st_mtime_ns, opt, max_batch)
    if key in _PROFILES:
        return _PROFILES[key]
    model = onnx.load(str(model_path), load_external_data=False)
    initializers = {i.name for i in model.graph.initializer}
    shapes: dict[str, tuple[int, ...]] = {}
    for inp in model.graph.input:
        if inp.name in initializers:
            continue
        dims = inp.type.tensor_type.shape.dim
        rest = tuple(int(d.dim_value) for d in dims[1:])
        if any(d <= 0 for d in rest):
            raise ValueError(f"{Path(model_path).name}: input {inp.name} has a dynamic "
                             f"non-batch dimension {rest}; pass a profile explicitly")
        shapes[inp.name] = rest

    def at(b: int) -> dict[str, tuple[int, ...]]:
        return {k: (b, *v) for k, v in shapes.items()}

    profile = ShapeProfile(min_shapes=at(1), opt_shapes=at(min(opt, max_batch)),
                           max_shapes=at(max_batch))
    _PROFILES[key] = profile
    return profile
