"""Precision-aware ONNX graph annotations and native TensorRT compilation.

The ONNX Runtime TensorRT execution provider is intentionally kept as the
application's compatibility path.  Per-layer precision constraints are a
native TensorRT feature, so this module provides the explicit builder path
used by ``tools/build_trt_engines.py --native`` and by callers that need
precision segmentation.

Graph surgery is deliberately metadata-only: node names are made stable and
the annotations are stored in ONNX metadata.  No operator is replaced and no
initializer is rewritten.  The native builder applies the annotations to the
corresponding ``ILayer`` objects after parsing.  This avoids attaching
unknown attributes to standard ONNX operators, which would make otherwise
valid models unparsable by TensorRT.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

logger = logging.getLogger("roop.trt_graph_surgery")

PRECISION_FP32 = "fp32"
PRECISION_FP16 = "fp16"
PRECISION_AUTO = "auto"
PRECISION_ORDER = ("fp8", "int8", "fp16", "fp32")
COSINE_THRESHOLD = 0.99

_NORMALIZATION_OPS = frozenset(
    {"InstanceNormalization", "LayerNormalization", "GroupNormalization", "GroupNorm"}
)
_REDUCIBLE_OPS = frozenset(
    {
        "Conv", "ConvTranspose", "MatMul", "Gemm", "Gelu", "Relu", "LeakyRelu",
        "Sigmoid", "Tanh", "Silu", "Swish", "Elu", "Clip", "Add", "Mul",
    }
)
_EMBEDDING_TOKENS = frozenset(
    {"arcface", "embedding", "latent", "identity", "id_embed", "source_vector", "emap"}
)
_ATTENTION_TOKENS = frozenset(
    {"attention", "self_attn", "cross_attn", "selfattention", "crossattention", "qkv"}
)


class TensorRTGraphError(RuntimeError):
    """Raised when a graph cannot be prepared or compiled safely."""


@dataclass(frozen=True)
class LayerPrecisionAnnotation:
    node_name: str
    op_type: str
    precision: str = PRECISION_AUTO
    output_precision: str | None = None
    reason: str = "intermediate layer may use reduced precision"
    reduced_precision_allowed: bool = True
    tensor_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphSurgeryResult:
    source_path: str
    prepared_path: str
    model_family: str
    annotations: tuple[LayerPrecisionAnnotation, ...]
    tensor_to_node: Mapping[str, str]
    source_digest: str

    @property
    def sensitive(self) -> tuple[LayerPrecisionAnnotation, ...]:
        return tuple(item for item in self.annotations if item.precision != PRECISION_AUTO)

    @property
    def annotation_by_name(self) -> dict[str, LayerPrecisionAnnotation]:
        return {item.node_name: item for item in self.annotations}

    def node_for_tensor(self, tensor_name: str) -> str | None:
        return self.tensor_to_node.get(str(tensor_name))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_model_family(model_family: str | None, model_path: str | os.PathLike[str] | None = None) -> str:
    value = f"{model_family or ''} {model_path or ''}".lower().replace("\\", "/")
    if "inswapper" in value or "swap" in value or "arcface" in value:
        return "inswapper"
    if "restoreformer" in value or "restore_ultra" in value:
        return "restoreformer_pp"
    if "gpen" in value:
        return "gpen"
    if "xseg" in value or "dfl" in value:
        return "dfl_xseg"
    if "scrfd" in value or "detector" in value:
        return "scrfd"
    return re.sub(r"[^a-z0-9]+", "_", str(model_family or "unknown").lower()).strip("_") or "unknown"


def _node_text(node: Any) -> str:
    values = [getattr(node, "name", ""), getattr(node, "op", ""), getattr(node, "op_type", "")]
    for tensor in getattr(node, "inputs", ()) or ():
        values.append(getattr(tensor, "name", ""))
    for tensor in getattr(node, "outputs", ()) or ():
        values.append(getattr(tensor, "name", ""))
    return " ".join(str(item or "") for item in values).lower()


def _op_type(node: Any) -> str:
    return str(getattr(node, "op", getattr(node, "op_type", "")) or "")


def _is_embedding_matmul(node: Any) -> bool:
    text = _node_text(node)
    return _op_type(node) in {"MatMul", "Gemm"} and any(token in text for token in _EMBEDDING_TOKENS)


def _is_attention_softmax(node: Any) -> bool:
    # Treat every Softmax as sensitive.  A detector's class-probability
    # softmax is safe in FP32 too, and this conservative rule covers exports
    # that omit useful attention names from their nodes.
    return _op_type(node) == "Softmax" or (
        "softmax" in _node_text(node) and any(token in _node_text(node) for token in _ATTENTION_TOKENS)
    )


def _annotation_for(node: Any, output_names: set[str], normalization_precision: str) -> LayerPrecisionAnnotation:
    op_type = _op_type(node)
    node_name = str(getattr(node, "name", "") or "")
    tensor_names = tuple(
        str(getattr(tensor, "name", ""))
        for tensor in (getattr(node, "outputs", ()) or ())
        if getattr(tensor, "name", None)
    )
    if op_type in _NORMALIZATION_OPS or "groupnorm" in node_name.lower():
        return LayerPrecisionAnnotation(
            node_name, op_type, normalization_precision, normalization_precision,
            "normalization statistics are precision-sensitive", False, tensor_names,
        )
    if _is_attention_softmax(node):
        return LayerPrecisionAnnotation(
            node_name, op_type, PRECISION_FP32, PRECISION_FP32,
            "attention/classification softmax stability", False, tensor_names,
        )
    if _is_embedding_matmul(node):
        return LayerPrecisionAnnotation(
            node_name, op_type, PRECISION_FP32, PRECISION_FP32,
            "ArcFace/latent embedding injection", False, tensor_names,
        )
    if op_type == "Conv" and output_names.intersection(tensor_names):
        return LayerPrecisionAnnotation(
            node_name, op_type, PRECISION_FP16, PRECISION_FP16,
            "final projection convolution", True, tensor_names,
        )
    if op_type in _REDUCIBLE_OPS:
        return LayerPrecisionAnnotation(
            node_name, op_type, PRECISION_AUTO, None,
            "intermediate layer may use FP8/INT8/FP16", True, tensor_names,
        )
    return LayerPrecisionAnnotation(
        node_name, op_type, PRECISION_AUTO, None,
        "TensorRT default precision", True, tensor_names,
    )


def classify_onnx_graph(
    model_path: str | os.PathLike[str],
    *,
    model_family: str | None = None,
    normalization_precision: str = PRECISION_FP32,
) -> GraphSurgeryResult:
    """Parse an ONNX graph with ONNX and GraphSurgeon and classify every node."""
    path = Path(model_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if normalization_precision not in {PRECISION_FP32, PRECISION_FP16}:
        raise ValueError("normalization_precision must be 'fp32' or 'fp16'")
    try:
        import onnx
        import onnx_graphsurgeon as gs
    except ImportError as error:
        raise TensorRTGraphError(
            "precision-aware TensorRT builds require both onnx and onnx-graphsurgeon"
        ) from error

    model = onnx.load(str(path), load_external_data=False)
    graph = gs.import_onnx(model)
    graph_nodes = list(graph.nodes)
    output_names = {
        str(getattr(output, "name", "")) for output in (graph.outputs or ())
    }
    for index, node in enumerate(graph_nodes):
        if not getattr(node, "name", None):
            node.name = f"roop_{index:05d}_{_op_type(node) or 'node'}"
    family = canonical_model_family(model_family, path)
    annotations = tuple(_annotation_for(node, output_names, normalization_precision) for node in graph_nodes)
    tensor_to_node = {
        tensor_name: annotation.node_name
        for annotation in annotations
        for tensor_name in annotation.tensor_names
    }
    return GraphSurgeryResult(
        source_path=str(path),
        prepared_path=str(path),
        model_family=family,
        annotations=annotations,
        tensor_to_node=tensor_to_node,
        source_digest=_sha256(path),
    )


def _metadata(model: Any, result: GraphSurgeryResult) -> None:
    existing = {item.key: item.value for item in model.metadata_props}
    values = {
        "roop.precision.schema": "1",
        "roop.precision.model_family": result.model_family,
        "roop.precision.source_sha256": result.source_digest,
        "roop.precision.nodes": json.dumps(
            [
                {
                    "node": item.node_name,
                    "op": item.op_type,
                    "precision": item.precision,
                    "output_precision": item.output_precision,
                    "reason": item.reason,
                    "reduced_precision_allowed": item.reduced_precision_allowed,
                }
                for item in result.annotations
            ],
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    del model.metadata_props[:]
    for key, value in {**existing, **values}.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = str(value)


def prepare_onnx_model(
    model_path: str | os.PathLike[str],
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    model_family: str | None = None,
    normalization_precision: str = PRECISION_FP32,
) -> GraphSurgeryResult:
    """Create a cached, metadata-tagged ONNX copy without changing the source."""
    source = Path(model_path).resolve()
    result = classify_onnx_graph(
        source, model_family=model_family, normalization_precision=normalization_precision
    )
    root = Path(cache_dir or source.parent / "trt_cache") / "graph_surgery"
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        f"{result.source_digest}:{result.model_family}:{normalization_precision}".encode()
    ).hexdigest()[:16]
    destination = root / f"{source.stem}_{result.model_family}_{digest}.onnx"
    if not destination.is_file() or destination.stat().st_mtime_ns < source.stat().st_mtime_ns:
        import onnx
        import onnx_graphsurgeon as gs

        model = onnx.load(str(source), load_external_data=False)
        graph = gs.import_onnx(model)
        for index, node in enumerate(graph.nodes):
            if not getattr(node, "name", None):
                node.name = f"roop_{index:05d}_{_op_type(node) or 'node'}"
        graph.cleanup().toposort()
        tagged = gs.export_onnx(graph)
        _metadata(tagged, result)
        temporary = destination.with_suffix(destination.suffix + ".part")
        onnx.save(tagged, str(temporary))
        os.replace(temporary, destination)
    return GraphSurgeryResult(
        source_path=result.source_path,
        prepared_path=str(destination),
        model_family=result.model_family,
        annotations=result.annotations,
        tensor_to_node=result.tensor_to_node,
        source_digest=result.source_digest,
    )


def _trt_dtype(trt: Any, precision: str) -> Any:
    return getattr(trt, "float32" if precision == PRECISION_FP32 else "float16")


def configure_builder_config(
    builder: Any,
    trt: Any,
    *,
    workspace_bytes: int,
    tier: str = "fp8",
) -> Any:
    """Create an IBuilderConfig with explicit constraints and safe fallbacks."""
    config = builder.create_builder_config()
    flags = getattr(trt, "BuilderFlag", object())

    def set_flag(name: str) -> None:
        flag = getattr(flags, name, None)
        if flag is not None:
            config.set_flag(flag)

    set_flag("PREFER_PRECISION_CONSTRAINTS")
    set_flag("DIRECT_IO")
    if tier in {"fp8", "int8", "fp16"}:
        set_flag("FP16")
    if tier in {"fp8", "int8"}:
        set_flag("INT8")
    if tier == "fp8":
        set_flag("FP8")
    workspace_bytes = max(256 * 1024 * 1024, int(workspace_bytes))
    pool = getattr(getattr(trt, "MemoryPoolType", object()), "WORKSPACE", None)
    if pool is not None and hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(pool, workspace_bytes)
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = workspace_bytes
    return config


def apply_layer_precision(
    network: Any,
    trt: Any,
    annotations: Iterable[LayerPrecisionAnnotation],
    *,
    blacklist: Iterable[str] = (),
    tier: str = "fp8",
    strict: bool = True,
) -> tuple[str, ...]:
    """Apply ``ILayer.precision`` and output types for tagged nodes."""
    by_name = {item.node_name: item for item in annotations}
    blocked = {str(item) for item in blacklist}
    applied: list[str] = []
    seen: set[str] = set()
    for index in range(int(getattr(network, "num_layers", 0))):
        layer = network.get_layer(index)
        name = str(getattr(layer, "name", ""))
        annotation = by_name.get(name)
        if annotation is None:
            continue
        seen.add(name)
        if name in blocked and annotation.reduced_precision_allowed:
            annotation = LayerPrecisionAnnotation(
                annotation.node_name, annotation.op_type, PRECISION_FP32,
                PRECISION_FP32, "activation validator blacklist", False,
                annotation.tensor_names,
            )
        if annotation.precision == PRECISION_AUTO:
            continue
        precision = annotation.precision
        # The final Conv is FP16 in the normal mixed build, but the terminal
        # fallback must really be FP32 rather than carrying a stale FP16 hint.
        if tier == "fp32" and precision == PRECISION_FP16:
            precision = PRECISION_FP32
        dtype = _trt_dtype(trt, precision)
        try:
            layer.precision = dtype
            for output_index in range(int(getattr(layer, "num_outputs", 0))):
                setter = getattr(layer, "set_output_type", None)
                if callable(setter):
                    output_precision = annotation.output_precision or precision
                    if tier == "fp32" and output_precision == PRECISION_FP16:
                        output_precision = PRECISION_FP32
                    setter(output_index, _trt_dtype(trt, output_precision))
            applied.append(name)
        except Exception as error:
            raise TensorRTGraphError(
                f"could not apply {annotation.precision} to TensorRT layer {name!r}"
            ) from error
    missing = sorted(
        item.node_name for item in annotations
        if item.precision != PRECISION_AUTO and item.node_name not in seen
    )
    if strict and missing:
        raise TensorRTGraphError(
            "TensorRT parser did not expose annotated ONNX nodes: " + ", ".join(missing[:12])
        )
    return tuple(applied)


def _network_flag(trt: Any, name: str) -> int:
    return int(getattr(getattr(trt, "NetworkDefinitionCreationFlag", object()), name, 0))


def _parse_network(trt: Any, model_path: str | os.PathLike[str]) -> tuple[Any, Any, Any]:
    logger_obj = trt.Logger(getattr(trt.Logger, "WARNING", 0))
    builder = trt.Builder(logger_obj)
    network = builder.create_network(_network_flag(trt, "EXPLICIT_BATCH"))
    parser = trt.OnnxParser(network, logger_obj)
    if not parser.parse_from_file(str(model_path)):
        errors = []
        for index in range(int(getattr(parser, "num_errors", 0))):
            errors.append(str(parser.get_error(index)))
        raise TensorRTGraphError(
            f"TensorRT ONNX parse failed for {model_path}: {' | '.join(errors)}"
        )
    return builder, network, logger_obj


def _serialize(builder: Any, network: Any, config: Any) -> bytes:
    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
    else:
        engine = builder.build_engine(network, config)
        serialized = engine.serialize()
    if serialized is None:
        raise TensorRTGraphError("TensorRT returned no serialized engine")
    return bytes(serialized)


def _add_optimization_profile(
    builder: Any,
    network: Any,
    config: Any,
    profiles: Mapping[str, Mapping[str, Sequence[int]]] | None,
) -> None:
    """Add one complete profile for dynamic inputs, including inswapper's two inputs."""
    dynamic = []
    for index in range(int(getattr(network, "num_inputs", 0))):
        tensor = network.get_input(index)
        declared = tuple(int(value) for value in tensor.shape)
        if any(value < 0 for value in declared):
            dynamic.append((str(tensor.name), declared))
    if not dynamic:
        return
    profile = builder.create_optimization_profile()
    supplied = profiles or {}
    for name, declared in dynamic:
        explicit = supplied.get(name, {})
        values = []
        for key, batch in (("min", 1), ("opt", 4), ("max", 8)):
            if key in explicit:
                shape = tuple(int(item) for item in explicit[key])
            else:
                shape = tuple(
                    batch if index == 0 else (640 if len(declared) >= 4 and index >= len(declared) - 2 else 1)
                    if value < 0 else value
                    for index, value in enumerate(declared)
                )
            if len(shape) != len(declared) or any(item <= 0 for item in shape):
                raise TensorRTGraphError(f"invalid TensorRT profile for input {name!r}: {shape!r}")
            values.append(shape)
        profile.set_shape(name, values[0], values[1], values[2])
    config.add_optimization_profile(profile)


def build_native_engine(
    model_path: str | os.PathLike[str],
    engine_path: str | os.PathLike[str],
    *,
    model_family: str | None = None,
    workspace_bytes: int = 4 * 1024**3,
    blacklist: Iterable[str] = (),
    normalization_precision: str = PRECISION_FP32,
    tiers: Sequence[str] = PRECISION_ORDER,
    input_profiles: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
    validator: Callable[[Path, GraphSurgeryResult], "ValidationReport"] | None = None,
    max_recompiles: int = 8,
) -> tuple[Path, GraphSurgeryResult, "ValidationReport | None"]:
    """Build a native TensorRT engine, validating and blacklisting bad nodes."""
    try:
        import tensorrt as trt
    except ImportError as error:
        raise TensorRTGraphError("native TensorRT builds require the tensorrt Python bindings") from error

    prepared = prepare_onnx_model(
        model_path, model_family=model_family, normalization_precision=normalization_precision,
        cache_dir=Path(engine_path).resolve().parent,
    )
    blocked = {str(item) for item in blacklist}
    if validator is None and os.environ.get("ROOP_TRT_SKIP_VALIDATION", "0").lower() not in {
        "1", "true", "yes", "on"
    }:
        validator = lambda engine, graph: run_polygraphy_validation(
            graph.prepared_path,
            engine_path=engine,
            tensor_to_node=graph.tensor_to_node,
        )
    report: ValidationReport | None = None
    target = Path(engine_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(max(1, int(max_recompiles))):
        last_error: Exception | None = None
        serialized: bytes | None = None
        for tier in tiers:
            try:
                builder, network, _logger = _parse_network(trt, prepared.prepared_path)
                config = configure_builder_config(
                    builder, trt, workspace_bytes=workspace_bytes, tier=tier,
                )
                _add_optimization_profile(builder, network, config, input_profiles)
                apply_layer_precision(
                    network, trt, prepared.annotations, blacklist=blocked,
                    tier=tier, strict=True,
                )
                serialized = _serialize(builder, network, config)
                break
            except Exception as error:
                last_error = error
                logger.warning("TensorRT %s build failed for %s: %s", tier, Path(model_path).name, error)
        if serialized is None:
            raise TensorRTGraphError("all TensorRT precision tiers failed") from last_error
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(serialized)
        os.replace(temporary, target)
        if validator is None:
            return target, prepared, report
        report = validator(target, prepared)
        new_nodes = set(report.blacklisted_nodes) - blocked
        if not new_nodes:
            return target, prepared, report
        blocked.update(new_nodes)
        logger.warning("Activation validation blacklisted %d node(s); recompiling", len(new_nodes))
    raise TensorRTGraphError(
        f"TensorRT activation validation did not converge after {max_recompiles} recompiles"
    )


@dataclass(frozen=True)
class LayerMetric:
    tensor_name: str
    node_name: str | None
    cosine_similarity: float
    mae: float
    cosine_drop: float


@dataclass(frozen=True)
class ValidationReport:
    metrics: tuple[LayerMetric, ...]
    blacklisted_nodes: tuple[str, ...] = ()
    command_lines: tuple[tuple[str, ...], ...] = ()

    @property
    def passed(self) -> bool:
        return not self.blacklisted_nodes


def compare_activation_outputs(
    baseline: Mapping[str, Any],
    mixed: Mapping[str, Any],
    *,
    tensor_to_node: Mapping[str, str] | None = None,
    cosine_threshold: float = COSINE_THRESHOLD,
) -> ValidationReport:
    """Calculate cosine similarity and MAE per activation tensor."""
    import numpy as np

    metrics: list[LayerMetric] = []
    bad: set[str] = set()
    for name in sorted(set(baseline).intersection(mixed)):
        reference = np.asarray(baseline[name], dtype=np.float32).reshape(-1)
        candidate = np.asarray(mixed[name], dtype=np.float32).reshape(-1)
        if reference.shape != candidate.shape:
            continue
        ref_norm = float(np.linalg.norm(reference))
        cand_norm = float(np.linalg.norm(candidate))
        if ref_norm == 0.0 and cand_norm == 0.0:
            cosine = 1.0
        elif ref_norm == 0.0 or cand_norm == 0.0:
            cosine = 0.0
        else:
            cosine = float(np.dot(reference, candidate) / (ref_norm * cand_norm))
        mae = float(np.mean(np.abs(reference - candidate)))
        node_name = (tensor_to_node or {}).get(name)
        metric = LayerMetric(name, node_name, cosine, mae, 1.0 - cosine)
        metrics.append(metric)
        if cosine < float(cosine_threshold) and node_name:
            bad.add(node_name)
    return ValidationReport(tuple(metrics), tuple(sorted(bad)))


def polygraphy_commands(
    onnx_path: str | os.PathLike[str],
    *,
    baseline_results: str | os.PathLike[str],
    trt_results: str | os.PathLike[str],
    engine_path: str | os.PathLike[str] | None = None,
    inputs_path: str | os.PathLike[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return Polygraphy commands that expose all ONNX/TRT layer outputs."""
    # Polygraphy 0.53 does not expose a literal ``--onnx`` switch; the
    # equivalent explicit loader is ``--model-type onnx``.  Keeping the model
    # type explicit prevents a future engine filename from being interpreted
    # as an ONNX graph by accident.
    common = (
        "polygraphy", "run", str(onnx_path), "--model-type", "onnx",
        "--onnx-outputs", "mark", "all",
    )
    saved_inputs = str(inputs_path or Path(baseline_results).with_suffix(".inputs.json"))
    baseline = common + (
        "--onnxrt", "--save-inputs", saved_inputs,
        "--save-results", str(baseline_results), "--silent",
    )
    mixed_source = str(engine_path) if engine_path is not None else str(onnx_path)
    mixed_parts = ["polygraphy", "run", mixed_source]
    if engine_path is None:
        mixed_parts.extend(("--model-type", "onnx"))
    mixed_parts.extend(
        (
            "--trt", "--trt-outputs", "mark", "all", "--load-inputs", saved_inputs,
            "--save-results", str(trt_results), "--silent",
        )
    )
    mixed = tuple(mixed_parts)
    return baseline, mixed


def _load_polygraphy_results(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    """Load common Polygraphy JSON result shapes without requiring Polygraphy."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("outputs"), dict):
        return payload["outputs"]
    if isinstance(payload, dict):
        return payload
    raise TensorRTGraphError(f"unsupported Polygraphy result format: {path}")


def run_polygraphy_validation(
    onnx_path: str | os.PathLike[str],
    *,
    tensor_to_node: Mapping[str, str] | None = None,
    engine_path: str | os.PathLike[str] | None = None,
    work_dir: str | os.PathLike[str] | None = None,
    cosine_threshold: float = COSINE_THRESHOLD,
    runner: Callable[..., Any] = subprocess.run,
) -> ValidationReport:
    """Run ONNX FP32 and TensorRT Polygraphy arms and compare every output."""
    root = Path(work_dir or tempfile.mkdtemp(prefix="roop-polygraphy-"))
    root.mkdir(parents=True, exist_ok=True)
    baseline_path = root / "baseline.json"
    trt_path = root / "trt.json"
    inputs_path = root / "inputs.json"
    commands = polygraphy_commands(
        onnx_path, baseline_results=baseline_path, trt_results=trt_path,
        engine_path=engine_path, inputs_path=inputs_path,
    )
    for command in commands:
        completed = runner(command, check=False, capture_output=True, text=True)
        if int(getattr(completed, "returncode", 1)) != 0:
            raise TensorRTGraphError(
                f"Polygraphy validation failed ({' '.join(command)}): "
                f"{getattr(completed, 'stderr', '')}"
            )
    report = compare_activation_outputs(
        _load_polygraphy_results(baseline_path),
        _load_polygraphy_results(trt_path),
        tensor_to_node=tensor_to_node,
        cosine_threshold=cosine_threshold,
    )
    return ValidationReport(report.metrics, report.blacklisted_nodes, commands)


__all__ = [
    "COSINE_THRESHOLD", "GraphSurgeryResult", "LayerMetric", "LayerPrecisionAnnotation",
    "PRECISION_AUTO", "PRECISION_FP16", "PRECISION_FP32", "PRECISION_ORDER",
    "TensorRTGraphError", "ValidationReport", "apply_layer_precision",
    "build_native_engine", "canonical_model_family", "classify_onnx_graph",
    "compare_activation_outputs", "configure_builder_config", "polygraphy_commands",
    "prepare_onnx_model", "run_polygraphy_validation",
]
