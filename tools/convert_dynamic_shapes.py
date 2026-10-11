"""Give static batch-1 ONNX face models a dynamic batch axis, and PROVE it works.

The strict TensorRT path (``roop/trt_session_builder.py``) refuses a static
batch-1 graph (``StaticBatchError``), and a TensorRT optimization profile of
min 1 / opt 4 / max 8 cannot be built against one either. This tool rewrites the
graph so dimension 0 of every runtime input is the named parameter
``batch_size`` and writes ``<name>_dynamic.onnx`` next to the source.

What "converted" means here -- and why ``onnx.checker`` is not enough
---------------------------------------------------------------------
``onnx.checker.check_model`` validates structure. A graph that was exported at
batch 1 can pass it, load in ONNX Runtime, accept a batch of 4, and compute the
wrong thing: styled/modulated convolutions (GPEN) bake ``groups=1``, and a
``Reshape`` to ``[1, C, H, W]`` folds the batch into the channel axis. That is
this project's recurring failure -- something reports success while not
running -- so every conversion is verified numerically before it is written:

    converted(batch of N)[i]  ==  original(sample i alone)      for N in --check-batches

on the CPU provider (deterministic, no GPU needed, safe next to a running app).
A model that fails is NOT written (``--force`` writes it anyway and says so);
the exit status is non-zero. Pass/fail, the worst relative error and the reason
are printed for every model and written to ``--report``.

What the surgery does
---------------------
1. Runtime inputs only (initializers listed as inputs by older exporters are
   weights): dim 0 -> ``batch_size``.
2. Outputs (ALL of them, including internals an export leaks as extra graph
   outputs: ORT treats a declared output shape as fact and constant-folds
   ``Shape`` of it, which silently pins the batch): dim 0 of an output whose
   declared dim 0 is 1 (or already symbolic) -> ``batch_size``. An output whose dim 0 is some other number (SCRFD folds the
   batch into its anchor axis: ``[12800, 1]``) -> ``batch_size_x_rows``. The
   check below finds how the batch is folded (SCRFD interleaves it per cell:
   index = (cell * N + sample) * anchors + anchor; it is NOT batch-major) and
   prints it, because whatever decodes those outputs has to unfold it.
3. ``Reshape`` nodes whose shape operand starts with 1 and whose DATA is
   provably batched -- ONNX shape inference, run with the symbolic ``batch_size``,
   gives its dim 0 that name -- get 0 there (ONNX: "copy the input's
   dimension"), whether the shape lives in a ``Constant`` node or an
   initializer; an initializer shared with a non-Reshape consumer is copied, not
   edited. Anything whose leading axis is not provably the batch (weights, and
   the per-sample weights of a styled convolution) is left alone.

Not convertible by graph surgery: styled/modulated convolutions (GPEN). They
collapse a per-sample weight tensor into ``[out, in, k, k]`` and use ``groups=1``;
a batch needs ``groups=N``, and ONNX ``Conv.group`` is a compile-time attribute.
The tool reports such a model as failed and does not write it.
4. ``value_info`` (stale static intermediate shapes) is dropped so ORT/TensorRT
   re-infer them.

Usage::

    app/env/Scripts/python.exe tools/convert_dynamic_shapes.py            # the default model set
    app/env/Scripts/python.exe tools/convert_dynamic_shapes.py app/models/inswapper_128.onnx --check-batches 1,4,8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_MODELS = REPO_ROOT / "app" / "models"

BATCH_DIM = "batch_size"
FOLDED_DIM = "batch_size_x_rows"
SUFFIX = "_dynamic"

#: Relative max error allowed between a batched run and the per-sample runs. FP32
#: on the CPU provider; batching only changes reduction order.
DEFAULT_TOLERANCE = 1e-3

#: Per-sample outputs to verify when an export leaks internals as extra graph
#: outputs. RestoreFormer++ lists 15: the image (``2701``), two scalar losses,
#: codebook tensors and intermediate feature maps.
PRIMARY_OUTPUTS: Dict[str, Tuple[str, ...]] = {
    "restoreformer_plus_plus": ("2701",),
}

#: The face-model set the strict TensorRT path cares about. A missing file is
#: skipped with a message, not an error (scrfd_2.5g_kps is a download, not a
#: bundled model; det_10g is the SCRFD the app actually ships).
DEFAULT_MODELS: Tuple[str, ...] = (
    "inswapper_128.onnx",
    "scrfd_2.5g_kps.onnx",
    "buffalo_l/det_10g.onnx",
    "gpen_bfr_256.onnx",
    "GPEN-BFR-512.onnx",
    "restoreformer_plus_plus.onnx",
)


# --------------------------------------------------------------------------
# graph surgery
# --------------------------------------------------------------------------

def runtime_inputs(model: onnx.ModelProto) -> List[onnx.ValueInfoProto]:
    weights = {initializer.name for initializer in model.graph.initializer}
    return [value for value in model.graph.input if value.name not in weights]


def _dims(value: onnx.ValueInfoProto) -> List[Any]:
    return [d.dim_param or (d.dim_value if d.HasField("dim_value") else None)
            for d in value.type.tensor_type.shape.dim]


def _set_dim0(value: onnx.ValueInfoProto, name: str) -> None:
    dims = value.type.tensor_type.shape.dim
    if len(dims):
        dims[0].ClearField("dim_value")
        dims[0].dim_param = name


def _shape_source(model: onnx.ModelProto, name: str):
    """(kind, tensor) of a Reshape's shape operand, or (None, None)."""
    for initializer in model.graph.initializer:
        if initializer.name == name:
            return "initializer", initializer
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output and node.output[0] == name:
            for attribute in node.attribute:
                if attribute.name == "value" and attribute.HasField("t"):
                    return "constant", attribute.t
    return None, None


def _batched_tensors(model: onnx.ModelProto, dim_name: str) -> Tuple[set, str]:
    """Names of tensors whose dim 0 is provably ``dim_name``, and the rule used."""
    try:
        inferred = onnx.shape_inference.infer_shapes(model, data_prop=True)
    except Exception:                                   # e.g. a model above 2 GB
        carries = {value.name for value in runtime_inputs(model)}
        for node in model.graph.node:                   # ONNX nodes are topologically sorted
            if any(name in carries for name in node.input):
                carries.update(node.output)
        return carries, "downstream-of-input (shape inference unavailable)"
    names = set()
    for value in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        dims = value.type.tensor_type.shape.dim
        if len(dims) and dims[0].dim_param == dim_name:
            names.add(value.name)
    return names, "symbolic shape inference"


def make_batch_dynamic(model: onnx.ModelProto, dim_name: str = BATCH_DIM) -> Dict[str, Any]:
    """Rewrite ``model`` in place; return what was changed."""
    graph = model.graph
    changes: Dict[str, Any] = {"inputs": [], "outputs": [], "reshapes": 0,
                               "reshapes_copied": 0}

    for value in runtime_inputs(model):
        before = _dims(value)
        _set_dim0(value, dim_name)
        changes["inputs"].append({"name": value.name, "before": before, "after": _dims(value)})

    for value in graph.output:
        before = _dims(value)
        if not before:
            continue
        symbolic = isinstance(before[0], str)
        _set_dim0(value, dim_name if (before[0] == 1 or symbolic) else FOLDED_DIM)
        changes["outputs"].append({"name": value.name, "before": before, "after": _dims(value)})

    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for node in graph.node:
        for tensor in node.input:
            consumers.setdefault(tensor, []).append(node)

    del graph.value_info[:]                      # stale static intermediate shapes
    batched, changes["reshape_rule"] = _batched_tensors(model, dim_name)

    for node in graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2 or node.input[0] not in batched:
            continue
        kind, tensor = _shape_source(model, node.input[1])
        if tensor is None:
            continue
        shape = numpy_helper.to_array(tensor)
        if shape.ndim != 1 or shape.size < 2 or int(shape[0]) != 1:
            continue
        dynamic = np.array(shape, copy=True)
        dynamic[0] = 0                     # ONNX Reshape: 0 = copy the input's dimension
        users = consumers.get(node.input[1], [])
        if kind == "initializer" and any(user.op_type != "Reshape" for user in users):
            # Shared with something that is not a Reshape: give this node its own copy.
            fresh = numpy_helper.from_array(dynamic, node.input[1] + "__dynbatch")
            graph.initializer.append(fresh)
            node.input[1] = fresh.name
            changes["reshapes_copied"] += 1
        else:
            tensor.CopyFrom(numpy_helper.from_array(dynamic, tensor.name))
        changes["reshapes"] += 1

    return changes


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

def _concrete_input(value: onnx.ValueInfoProto, batch: int, size: int) -> Tuple[int, ...]:
    shape: List[int] = []
    for index, dim in enumerate(value.type.tensor_type.shape.dim):
        if index == 0:
            shape.append(batch)
        elif dim.HasField("dim_value") and dim.dim_value > 0:
            shape.append(int(dim.dim_value))
        else:
            shape.append(size)            # a dynamic spatial axis
    return tuple(shape)


def _feeds(inputs: Sequence[onnx.ValueInfoProto], batch: int, size: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    feeds: Dict[str, np.ndarray] = {}
    for value in inputs:
        element = value.type.tensor_type.elem_type
        dtype = onnx.helper.tensor_dtype_to_np_dtype(element)
        shape = _concrete_input(value, batch, size)
        if np.issubdtype(dtype, np.floating):
            feeds[value.name] = rng.standard_normal(shape).astype(dtype)
        else:
            feeds[value.name] = np.zeros(shape, dtype=dtype)
    return feeds


def _session(payload: bytes):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(payload, options, providers=["CPUExecutionProvider"])


def _rel_error(got: np.ndarray, want: np.ndarray) -> float:
    scale = float(np.max(np.abs(want))) + 1e-6
    return float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64))) / scale)


def _unfold(got: np.ndarray, batch: int, single_shape: Tuple[int, ...]):
    """Candidate (layout name, [N, *single_shape] array) pairs for a batched output."""
    if got.shape == (batch,) + tuple(single_shape):
        yield "leading batch axis", got
        return
    rows = single_shape[0] if single_shape else 1
    if not single_shape or got.shape[0] != batch * rows or got.shape[1:] != tuple(single_shape[1:]):
        return
    rest = tuple(single_shape[1:])
    # Folded into axis 0: rows = cells * anchors, laid out [cell][sample][anchor].
    # anchors == rows is the plain batch-major fold.
    for anchors in [rows] + [a for a in (2, 1, 3, 4, 5, 6, 8) if a != rows]:
        if rows % anchors:
            continue
        cells = rows // anchors
        arranged = got.reshape((cells, batch, anchors) + rest).transpose((1, 0, 2) + tuple(range(3, 3 + len(rest))))
        name = "batch-major fold" if anchors == rows else f"cell-major fold ({anchors} anchors per cell)"
        yield name, arranged.reshape((batch,) + tuple(single_shape))


def verify_equivalence(
    original: onnx.ModelProto,
    converted: onnx.ModelProto,
    batches: Iterable[int],
    size: int = 640,
    tolerance: float = DEFAULT_TOLERANCE,
    outputs: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """converted(batch N)[i] must equal original(sample i alone), for every N.

    Rank-0 outputs (losses, counters) have no per-sample meaning and are skipped;
    ``outputs`` restricts the check to the named per-sample outputs (an export
    that leaks internals as graph outputs, e.g. RestoreFormer++, needs this).
    """
    inputs = runtime_inputs(original)
    names = [value.name for value in inputs]
    output_names = [o.name for o in original.graph.output]
    reference = _session(original.SerializeToString())
    candidate = _session(converted.SerializeToString())
    result: Dict[str, Any] = {"ok": True, "tolerance": tolerance, "batches": {}, "worst_rel_error": 0.0,
                              "reason": "", "layouts": {}, "skipped_outputs": [], "unverified_outputs": []}

    for batch in batches:
        feeds = _feeds(inputs, batch, size, seed=1234 + batch)
        singles = []
        for index in range(batch):
            one = {name: np.ascontiguousarray(feeds[name][index:index + 1]) for name in names}
            singles.append(reference.run(None, one))
        try:
            batched = candidate.run(None, feeds)
        except Exception as error:               # a shape rejection IS the finding
            result.update(ok=False, reason=f"batch {batch}: ONNX Runtime rejected the batched input "
                                           f"({type(error).__name__}: {str(error)[:200]}); the graph bakes the "
                                           f"batch into a Reshape or a grouped Conv - styled/modulated "
                                           f"convolutions (GPEN) cannot be batched by graph surgery")
            result["batches"][str(batch)] = {"ok": False}
            return result

        worst = 0.0
        for position, (got, ref_name) in enumerate(zip(batched, output_names)):
            wanted = np.stack([single[position] for single in singles], axis=0)       # [N, *single_shape]
            single_shape = tuple(wanted.shape[1:])
            if not single_shape:
                if ref_name not in result["skipped_outputs"]:
                    result["skipped_outputs"].append(ref_name)
                continue
            if outputs is not None and ref_name not in outputs:
                if ref_name not in result["unverified_outputs"]:
                    result["unverified_outputs"].append(ref_name)
                continue
            best = None
            for layout, arranged in _unfold(got, batch, single_shape):
                error = _rel_error(arranged, wanted)
                if best is None or error < best[1]:
                    best = (layout, error)
                if error <= tolerance:
                    break
            if best is None:
                result.update(ok=False, reason=(
                    f"batch {batch}: output {ref_name!r} came back {tuple(got.shape)}; expected "
                    f"{(batch,) + single_shape} or axis 0 folded to {batch * single_shape[0]} rows"))
                result["batches"][str(batch)] = {"ok": False}
                return result
            result["layouts"][ref_name] = best[0]
            worst = max(worst, best[1])

        passed = worst <= tolerance
        result["batches"][str(batch)] = {"ok": passed, "rel_error": worst}
        result["worst_rel_error"] = max(result["worst_rel_error"], worst)
        if not passed:
            result.update(ok=False, reason=(
                f"batch {batch}: batched output differs from per-sample runs "
                f"(relative error {worst:.3g} > {tolerance:g}); the graph loads and accepts a batch but "
                f"does not compute one - typically a Reshape/Conv(groups) baked at batch 1"))
            return result
    return result


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def dynamic_path(source: Path) -> Path:
    return source.with_name(source.stem + SUFFIX + source.suffix)


def convert_model(
    source: Path,
    *,
    batches: Sequence[int] = (1, 4, 8),
    size: int = 640,
    tolerance: float = DEFAULT_TOLERANCE,
    verify: bool = True,
    force: bool = False,
    out_dir: Optional[Path] = None,
    outputs: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Convert one model. ``report['written']`` is the path, or None when it failed verification."""
    started = time.time()
    report: Dict[str, Any] = {"source": str(source), "written": None, "ok": False, "reason": ""}
    original = onnx.load(str(source))
    converted = onnx.load(str(source))
    outputs = outputs or PRIMARY_OUTPUTS.get(source.stem)
    report["changes"] = make_batch_dynamic(converted)

    if verify:
        report["verification"] = verify_equivalence(original, converted, batches, size, tolerance, outputs)
        report["ok"] = bool(report["verification"]["ok"])
        report["reason"] = report["verification"]["reason"]
    else:
        report["ok"] = True
        report["reason"] = "verification skipped (--no-verify)"

    target = dynamic_path(source)
    if out_dir is not None:
        target = Path(out_dir) / target.name
    if report["ok"] or force:
        target.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(converted, str(target))
        onnx.checker.check_model(str(target))                 # path form: also valid above 2 GB
        report["written"] = str(target)
        if not report["ok"]:
            report["reason"] += "  [written anyway: --force]"
    report["seconds"] = round(time.time() - started, 1)
    return report


def prepare_dynamic_model(source: Path, **kwargs: Any) -> Path:
    """Path of a verified ``_dynamic.onnx`` for ``source``, converting it when missing/stale.

    Raises RuntimeError with the verification reason when the model cannot be
    made batch-dynamic by graph surgery. Used by tools/build_trt_engines.py.
    """
    source = Path(source)
    target = dynamic_path(source)
    if target.is_file() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    report = convert_model(source, **kwargs)
    if not report["written"] or not report["ok"]:
        raise RuntimeError(f"{source.name}: not batch-convertible: {report['reason']}")
    return Path(report["written"])


def _resolve(name: str) -> Optional[Path]:
    direct = Path(name)
    if direct.is_file():
        return direct
    for root in (APP_MODELS, REPO_ROOT / "models"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="ONNX files (default: the face-model set under app/models)")
    parser.add_argument("--check-batches", default="1,4,8", help="batch sizes to verify (default 1,4,8)")
    parser.add_argument("--size", type=int, default=640, help="size for a dynamic spatial axis (SCRFD: 640)")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--no-verify", action="store_true", help="skip the numeric check (NOT recommended)")
    parser.add_argument("--force", action="store_true", help="write a model even if verification failed")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    batches = tuple(int(item) for item in args.check_batches.split(",") if item.strip())
    names = args.models or list(DEFAULT_MODELS)
    reports: List[Dict[str, Any]] = []
    failed = 0
    for name in names:
        source = _resolve(name)
        if source is None:
            print(f"[skip] {name}: not found under app/models or models/")
            reports.append({"source": name, "ok": None, "reason": "not found"})
            continue
        if source.stem.endswith(SUFFIX):
            print(f"[skip] {source.name}: already a {SUFFIX} model")
            continue
        print(f"[....] {source.name}")
        try:
            report = convert_model(source, batches=batches, size=args.size, tolerance=args.tolerance,
                                   verify=not args.no_verify, force=args.force, out_dir=args.out_dir)
        except Exception as error:
            report = {"source": str(source), "ok": False, "written": None,
                      "reason": f"{type(error).__name__}: {error}"}
        reports.append(report)
        verification = report.get("verification", {})
        if report["ok"]:
            worst = verification.get("worst_rel_error")
            print(f"[ OK ] {source.name} -> {Path(report['written']).name}  "
                  f"(batches {','.join(verification.get('batches', {}))}; worst rel error "
                  f"{worst:.2e}; {report['changes']['reshapes']} Reshape shapes, "
                  f"{len(report['changes']['inputs'])} input(s), {report['seconds']}s)")
        else:
            failed += 1
            print(f"[FAIL] {source.name}: {report['reason']}"
                  + (f"\n       written anyway: {report['written']}" if report.get("written") else ""))
    if args.report:
        args.report.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
    print(f"\n{sum(1 for r in reports if r.get('ok'))} converted, {failed} failed verification, "
          f"{sum(1 for r in reports if r.get('ok') is None)} not found")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
