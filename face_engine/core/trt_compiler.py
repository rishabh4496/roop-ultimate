"""Ahead-of-time TensorRT engine compilation, verification and lookup.

``tools/compile_engines.py`` is the CLI; this module is the library.

What gets compiled
------------------
Each :class:`EngineSpec` names a zoo model, the graph to build (the verified
dynamic-batch copy from :mod:`face_engine.utils.onnx_batch` where the model
can batch, else the original) and one optimisation profile per input, using
the graph's REAL input names and layouts:

    model               input(s)                               batch  min/opt/max
    hyperswap_1x_256    target (B,3,256,256), source (B,512)   dyn    1 / 2 / 8
    inswapper_128       target (B,3,128,128), source (B,512)   dyn    1 / 2 / 8
    xseg_3              input  (B,256,256,3)  NHWC             dyn    1 / 2 / 8
    bisenet_resnet34    input  (B,3,512,512)                   dyn    1 / 2 / 4
    gpen_bfr_512        input  (1,3,512,512)                   fixed  1
    gpen_bfr_1024       input  (1,3,1024,1024)                 fixed  1

The Stage 6 brief asked for ``source_emb``, NCHW XSeg and a 1-4 GPEN batch:
HyperSwap's embedding input is named ``source``, ``xseg_3`` is NHWC, and GPEN
cannot batch at all (its StyleGAN2 modulated convolutions fold the batch into
the ``Conv`` group count; see ``onnx_batch``), so GPEN profiles are fixed at 1
and GPEN-1024 takes 1024 px, not 512.

FP16 and the batched HyperSwap graph
------------------------------------
The batched HyperSwap / RestoreFormer++ graphs replace InstanceNormalization
(whose CUDA/TRT kernels mix batch rows) with primitive ops that must compute
their statistics in FP32. Through ONNX Runtime's TensorRT EP an FP16 engine
ran them in FP16 and HyperSwap identity collapsed (0.588 -> 0.334 on 222
real-clip swaps). Here the builder is told: every layer of a decomposed
InstanceNorm is pinned to FP32 (``layer.precision`` + output types) under
``OBEY_PRECISION_CONSTRAINTS``; the rest of the network may use FP16.

Verification (every engine, right after the build)
--------------------------------------------------
1. **Fidelity**: the engine's outputs on REAL face crops (insightface's
   ``t1.jpg``, aligned and normalised like the pipeline does) against ONNX
   Runtime FP32 on the ORIGINAL model, one sample at a time. Reported as
   mean |diff| / output range; an engine above the gate is rejected and
   deleted. A latency check alone would have passed the collapsed FP16
   HyperSwap.
2. **Latency**: context + ``execute_async_v3`` on torch CUDA memory, median
   of 50 runs at batch 1 and at the opt batch. HyperSwap-256 at batch 1 must
   be under 15 ms.

Artefacts: ``.cache/trt_engines/{model}_sm{SM}_{precision}_b{max}.engine`` and
a ``.json`` sidecar (TensorRT version, SM, graph + source stamps, profile,
pinned layers, verification). :func:`find_engine` only returns an engine
whose sidecar matches this machine, this TensorRT and the unchanged graph.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from face_engine.core.config import DEFAULT_CACHE_DIR, _env_path

logger = logging.getLogger(__name__)

ENGINE_DIR = _env_path("FACE_ENGINE_CACHE_DIR", DEFAULT_CACHE_DIR) / "trt_engines"
TIMING_CACHE = "timing_cache.bin"
FIDELITY_GATE = 0.01  # mean |engine - onnxruntime fp32| / output range
HYPERSWAP_LATENCY_GATE_MS = 15.0
_IN_MARK = "__in_"  # decompose_instance_norm's tensor-name prefix


# ---------------------------------------------------------------------------- specs
@dataclass(frozen=True)
class EngineSpec:
    """One engine to build.

    Attributes:
        model: Zoo name.
        group: ``swapper`` / ``enhancer`` / ``masker``.
        inputs: Input name -> shape without the batch dimension.
        batch: ``(min, opt, max)``; ``(1, 1, 1)`` for a fixed-batch model.
        batched_graph: Build the verified dynamic-batch copy of the model.
        fp32_layers: Layer / tensor name substrings pinned to FP32 in an FP16
            build (on top of decomposed InstanceNorms): the layers measured to
            leave FP16's range.
    """

    model: str
    group: str
    inputs: dict[str, tuple[int, ...]]
    batch: tuple[int, int, int] = (1, 2, 8)
    batched_graph: bool = True
    fp32_layers: tuple[str, ...] = ()

    @property
    def max_batch(self) -> int:
        return self.batch[2]


ENGINE_SPECS: dict[str, EngineSpec] = {s.model: s for s in (
    *(EngineSpec(f"hyperswap_{v}_256", "swapper",
                 {"source": (512,), "target": (3, 256, 256)}) for v in ("1a", "1b", "1c")),
    EngineSpec("inswapper_128", "swapper", {"target": (3, 128, 128), "source": (512,)}),
    EngineSpec("gpen_bfr_512", "enhancer", {"input": (3, 512, 512)}, (1, 1, 1), False),
    # GPEN-1024 FP16 produced NaN (Stage 6). Measured per node in FP32 on real
    # 1024 crops (2026-09-28): the 18 per-layer demodulation chains stay within
    # FP16 (sums <= ~400); the encoder's final linear reaches 1.9e5 and the
    # style pixel-norm's Pow 3.6e10 / ReduceMean 1.1e9 (its Div output ~3e-5 is
    # below FP16's smallest normal). Those two blocks run FP32.
    EngineSpec("gpen_bfr_1024", "enhancer", {"input": (3, 1024, 1024)}, (1, 1, 1), False,
               fp32_layers=("/final_linear/", "/generator/style/style.0/")),
    EngineSpec("xseg_3", "masker", {"input": (256, 256, 3)}, (1, 2, 8), False),
    # The boosted pipeline's detector (face_engine/boosted): fixed 640, batch <= 2.
    EngineSpec("retinaface_r50", "detector", {"input": (3, 640, 640)}, (1, 1, 2), False),
    EngineSpec("bisenet_resnet34", "masker", {"input": (3, 512, 512)}, (1, 2, 4), False),
)}


def select_specs(group: str) -> list[EngineSpec]:
    if group == "all":
        return list(ENGINE_SPECS.values())
    chosen = [s for s in ENGINE_SPECS.values() if s.group == group or s.model == group]
    if not chosen:
        raise KeyError(f"unknown model group {group!r}")
    return chosen


# ---------------------------------------------------------------------------- hardware
@dataclass(frozen=True)
class GpuInfo:
    device_id: int
    name: str
    sm: str  # "89" for compute capability 8.9
    total_vram_mb: int


def discover_gpu(device_id: int = 0) -> GpuInfo:
    """Compute capability from NVML (``nvmlDeviceGetCudaComputeCapability``), torch as fallback."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
        return GpuInfo(device_id, name.decode() if isinstance(name, bytes) else name,
                       f"{major}{minor}", int(total // 2 ** 20))
    except Exception:  # noqa: BLE001 - NVML unavailable: ask CUDA
        import torch

        major, minor = torch.cuda.get_device_capability(device_id)
        props = torch.cuda.get_device_properties(device_id)
        return GpuInfo(device_id, props.name, f"{major}{minor}", int(props.total_memory // 2 ** 20))


def engine_path(spec: EngineSpec, gpu: GpuInfo, precision: str,
                root: Path = ENGINE_DIR) -> Path:
    return root / f"{spec.model}_sm{gpu.sm}_{precision}_b{spec.max_batch}.engine"


def _stamp(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"file": path.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


# ---------------------------------------------------------------------------- build
def _trt() -> Any:
    import tensorrt as trt

    return trt


class _Logger:
    """TensorRT logger routed into :mod:`logging` (warnings and up)."""

    def __new__(cls) -> Any:
        trt = _trt()

        class Impl(trt.ILogger):  # type: ignore[misc]
            def __init__(self) -> None:
                trt.ILogger.__init__(self)
                self.errors: list[str] = []

            def log(self, severity: Any, msg: str) -> None:
                if severity <= trt.ILogger.Severity.ERROR:
                    self.errors.append(msg)
                    logger.error("TensorRT: %s", msg)
                elif severity == trt.ILogger.Severity.WARNING:
                    logger.debug("TensorRT: %s", msg)

        return Impl()


def pin_decomposed_norms(network: Any, extra: tuple[str, ...] = ()) -> int:
    """Pin every layer of a decomposed InstanceNorm, and every layer whose name or
    output name contains one of ``extra``, to FP32. Returns the count."""
    trt = _trt()
    floating = (trt.float32, trt.float16)
    pinned = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        outputs = [layer.get_output(j) for j in range(layer.num_outputs)]
        names = [layer.name, *(o.name for o in outputs)]
        if not any(_IN_MARK in n or any(e in n for e in extra) for n in names):
            continue
        if not all(o.dtype in floating for o in outputs):
            continue  # shape / index arithmetic
        layer.precision = trt.float32
        for j in range(layer.num_outputs):
            layer.set_output_type(j, trt.float32)
        pinned += 1
    return pinned


@dataclass
class BuildResult:
    spec: EngineSpec
    path: Path
    precision: str
    gpu: GpuInfo
    graph: str
    build_s: float
    pinned_fp32_layers: int
    engine_mb: float
    fidelity: dict[str, float] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    ok: bool = True
    problems: list[str] = field(default_factory=list)


def build_engine(spec: EngineSpec, source: Path, precision: str, gpu: GpuInfo, *,
                 workspace_bytes: int = 4 << 30, root: Path = ENGINE_DIR,
                 pin_norms: bool = True) -> BuildResult:
    """Parse, configure, build and serialize one engine (no verification).

    ``pin_norms=False`` exists for the control experiment only (an FP16 build
    of a decomposed-InstanceNorm graph without the FP32 constraints).
    """
    import torch

    from face_engine.utils.onnx_batch import batched_model

    trt = _trt()
    if precision not in ("fp16", "fp32"):
        raise ValueError("precision must be fp16 or fp32")
    root.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(gpu.device_id)
    graph = source
    if spec.batched_graph:
        derived = batched_model(source)
        if derived is None:
            raise RuntimeError(f"{spec.model}: no verified dynamic-batch graph")
        graph = derived
    log = _Logger()
    builder = trt.Builder(log)
    # TensorRT 10 networks are always explicit-batch; the flag is accepted
    # (and deprecated) for compatibility with 8.x-style call sites.
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, log)
    if not parser.parse_from_file(str(graph)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"{spec.model}: ONNX parse failed: {errors[:3]}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_bytes))
    pinned = 0
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        pinned = pin_decomposed_norms(network, spec.fp32_layers) if pin_norms else 0
        if pinned:
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    # Timing cache: layer tactic timings are reused across builds and models.
    cache_file = root / TIMING_CACHE
    blob = cache_file.read_bytes() if cache_file.exists() else b""
    timing = config.create_timing_cache(blob)
    config.set_timing_cache(timing, ignore_mismatch=False)
    profile = builder.create_optimization_profile()
    lo, opt, hi = spec.batch
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        if tensor.name not in spec.inputs:
            raise RuntimeError(f"{spec.model}: graph input {tensor.name!r} not in the spec "
                               f"({sorted(spec.inputs)})")
        rest = spec.inputs[tensor.name]
        profile.set_shape(tensor.name, (lo, *rest), (opt, *rest), (hi, *rest))
    config.add_optimization_profile(profile)
    t0 = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    if plan is None:
        raise RuntimeError(f"{spec.model}: build failed: {log.errors[-3:]}")
    cache_file.write_bytes(memoryview(config.get_timing_cache().serialize()))
    path = engine_path(spec, gpu, precision, root)
    path.write_bytes(memoryview(plan))
    return BuildResult(spec, path, precision, gpu, graph.name, build_s, pinned,
                       path.stat().st_size / 2 ** 20)


# ---------------------------------------------------------------------------- runtime
class TensorRTEngine:
    """A deserialized engine executed on torch CUDA memory.

    Exposes the subset of :class:`~face_engine.core.execution.ManagedSession`
    the batched GPU classes use (``input_names``, ``output_names``,
    ``run_binding``, ``primary_provider``, ``on_gpu``), so a compiled engine can
    stand in for an ONNX Runtime session.
    """

    primary_provider = "TensorRT (AOT engine)"
    on_gpu = True
    fell_back = False

    def __init__(self, path: Path | str, device_id: int = 0) -> None:
        import torch

        trt = _trt()
        self.path = Path(path)
        self.device = torch.device("cuda", device_id)
        key = (str(self.path.resolve()), device_id)
        if key not in _DESERIALIZED:
            torch.cuda.set_device(self.device)
            log = _Logger()
            runtime = trt.Runtime(log)
            engine = runtime.deserialize_cuda_engine(self.path.read_bytes())
            if engine is None:
                raise RuntimeError(f"cannot deserialize {self.path.name}: {log.errors[-2:]}")
            _DESERIALIZED[key] = (runtime, engine, log)
        self.runtime, self.engine, self._log = _DESERIALIZED[key]
        # One execution context per consumer: the preview and a render must not
        # share input shapes / tensor addresses.
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        mode = trt.TensorIOMode
        self.input_names = tuple(n for n in names if self.engine.get_tensor_mode(n) == mode.INPUT)
        self.output_names = tuple(n for n in names if self.engine.get_tensor_mode(n) == mode.OUTPUT)
        self._dtypes = {n: _torch_dtype(self.engine.get_tensor_dtype(n)) for n in names}
        profile = [self.engine.get_tensor_profile_shape(self.input_names[0], 0)]
        self.max_batch = int(profile[0][2][0])

    def run_binding(self, input_tensors: dict[str, Any], *, output_shapes: Any = None,
                    output_tensors: Any = None, unreturned_outputs: Any = (),
                    device_id: Any = None) -> dict[str, Any]:
        """Run on CUDA tensors; outputs are new torch tensors (shapes from the engine).

        Enqueued on the current torch stream, so no fence is needed (unlike ORT,
        which computes on its own stream). Unreturned outputs are released right
        away: the caching allocator only reuses them for later work on the same
        stream, which runs after this engine.
        """
        import torch

        stream = torch.cuda.current_stream(self.device)
        for name in self.input_names:
            t = input_tensors[name]
            if t.dtype != self._dtypes[name]:
                t = t.to(self._dtypes[name])
            t = t.contiguous()
            input_tensors[name] = t
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())
        out: dict[str, Any] = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            tensor = torch.empty(shape, dtype=self._dtypes[name], device=self.device)
            self.context.set_tensor_address(name, tensor.data_ptr())
            out[name] = tensor
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError(f"{self.path.name}: execute_async_v3 failed")
        return {k: v for k, v in out.items() if k not in unreturned_outputs}


# Deserialized engines shared within the process (weights live in VRAM once).
_DESERIALIZED: dict[tuple[str, int], tuple[Any, Any, Any]] = {}


def _torch_dtype(trt_dtype: Any) -> Any:
    import torch

    trt = _trt()
    return {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
            trt.int64: torch.int64, trt.int8: torch.int8, trt.bool: torch.bool,
            trt.uint8: torch.uint8}[trt_dtype]


# ---------------------------------------------------------------------------- verification
def _t1_faces() -> tuple[np.ndarray, np.ndarray]:
    """insightface's t1.jpg and its six faces' landmarks (the Stage 2/3 test photo)."""
    import cv2
    import insightface

    from face_engine.core.config import EngineConfig, Provider
    from face_engine.core.execution import ExecutionEngine
    from face_engine.models.zoo import build_default_registry
    from face_engine.pipeline.detector import SCRFDDetector

    image = cv2.imread(str(Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"))
    engine = ExecutionEngine(EngineConfig(providers=[Provider.CUDA, Provider.CPU]))
    detector = SCRFDDetector(engine, build_default_registry().ensure("scrfd_10g_bnkps",
                                                                     show_progress=False))
    faces = detector.detect(image)
    return image, np.stack([f.kps for f in faces])


def sample_inputs(spec: EngineSpec, batch: int) -> dict[str, np.ndarray]:
    """Real, pipeline-normalised inputs for ``spec`` (face crops of t1.jpg)."""
    from face_engine.pipeline.aligner import (
        estimate_similarity_transform,
        template_points,
        warp_face_by_translation,
    )

    image, kps = _t1_faces()
    idx = [i % len(kps) for i in range(batch)]

    def crops(size: int, template: str) -> np.ndarray:
        out = []
        for i in idx:
            m = estimate_similarity_transform(kps[i], template_points(size, template))
            out.append(warp_face_by_translation(image, m, size))
        return np.stack(out)  # (B, S, S, 3) uint8 BGR

    name = spec.model
    if name.startswith(("hyperswap", "inswapper")):
        size = 256 if name.startswith("hyperswap") else 128
        rgb = crops(size, "arcface_128")[..., ::-1].astype(np.float32) / 255.0
        target = rgb if name.startswith("inswapper") else (rgb - 0.5) / 0.5
        rng = np.random.default_rng(0)
        src = rng.standard_normal((batch, 512)).astype(np.float32)
        src /= np.linalg.norm(src, axis=1, keepdims=True)
        return {"target": np.ascontiguousarray(target.transpose(0, 3, 1, 2)), "source": src}
    if name == "xseg_3":
        return {"input": np.ascontiguousarray(crops(256, "arcface_128").astype(np.float32) / 255.0)}
    if name == "bisenet_resnet34":
        rgb = crops(512, "ffhq_512")[..., ::-1].astype(np.float32) / 255.0
        norm = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32)
        return {"input": np.ascontiguousarray(norm.transpose(0, 3, 1, 2))}
    if name == "retinaface_r50":
        import cv2

        canvas = cv2.resize(image, (640, 640)).astype(np.float32) - np.array(
            [104.0, 117.0, 123.0], np.float32)  # BGR, mean only (biubug6)
        return {"input": np.ascontiguousarray(
            np.repeat(canvas.transpose(2, 0, 1)[None], batch, axis=0))}
    if name.startswith("gpen"):
        size = spec.inputs["input"][-1]
        rgb = crops(size, "ffhq_512")[..., ::-1].astype(np.float32) / 127.5 - 1.0
        return {"input": np.ascontiguousarray(rgb.transpose(0, 3, 1, 2))}
    raise KeyError(f"no sample input for {name}")


def check_fidelity(engine: TensorRTEngine, spec: EngineSpec, source: Path,
                   batch: int) -> dict[str, float]:
    """Mean and max |engine - ORT fp32 original| / output range, per output (first output gated)."""
    import onnxruntime as ort
    import torch

    feeds = sample_inputs(spec, batch)
    reference = ort.InferenceSession(str(source), providers=["CUDAExecutionProvider",
                                                              "CPUExecutionProvider"])
    ref_names = [o.name for o in reference.get_outputs()]
    singles = [reference.run(None, {k: v[i:i + 1] for k, v in feeds.items()})
               for i in range(batch)]
    got = engine.run_binding({k: torch.from_numpy(v).to(engine.device) for k, v in feeds.items()})
    torch.cuda.synchronize()
    result: dict[str, float] = {}
    for oi, name in enumerate(ref_names):
        if name not in got:
            continue
        ref = np.concatenate([np.asarray(s[oi]) for s in singles])
        out = got[name].float().cpu().numpy().reshape(ref.shape)
        scale = max(float(ref.max() - ref.min()), 1e-6)
        diff = np.abs(out - ref)
        result[f"{name}.mean"] = float(diff.mean() / scale)
        result[f"{name}.max"] = float(diff.max() / scale)
    return result


def measure_latency(engine: TensorRTEngine, spec: EngineSpec, batch: int,
                    runs: int = 50) -> float:
    """Median ms per ``execute_async_v3`` at ``batch`` (inputs already on the GPU)."""
    import torch

    feeds = {k: torch.from_numpy(v).to(engine.device)
             for k, v in sample_inputs(spec, batch).items()}
    for _ in range(5):
        engine.run_binding(dict(feeds))
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine.run_binding(dict(feeds))
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def verify(result: BuildResult, source: Path,
           on_step: Callable[[str], None] | None = None) -> BuildResult:
    """Load the serialized engine, check fidelity and latency, write the sidecar."""
    spec = result.spec
    engine = TensorRTEngine(result.path, result.gpu.device_id)
    batch = spec.batch[1]
    fidelity = check_fidelity(engine, spec, source, batch)
    result.fidelity = fidelity
    first = next((v for k, v in fidelity.items() if k.endswith(".mean")), None)
    # "not <=" rather than ">": NaN compares False with everything, and a
    # non-finite engine (GPEN-1024 in FP16) must FAIL here, not pass.
    if first is None or not first <= FIDELITY_GATE:
        result.ok = False
        result.problems.append(f"fidelity {first} is not within {FIDELITY_GATE} of the "
                               "output range (non-finite output fails too)")
    result.latency_ms = {"b1": measure_latency(engine, spec, 1)}
    if batch > 1:
        result.latency_ms[f"b{batch}"] = measure_latency(engine, spec, batch)
    if spec.model.startswith("hyperswap") and result.latency_ms["b1"] >= HYPERSWAP_LATENCY_GATE_MS:
        result.ok = False
        result.problems.append(f"HyperSwap batch-1 latency {result.latency_ms['b1']:.2f} ms "
                               f">= {HYPERSWAP_LATENCY_GATE_MS} ms")
    if on_step:
        on_step(f"verified {result.path.name}")
    write_sidecar(result, source)
    return result


def write_sidecar(result: BuildResult, source: Path) -> None:
    trt = _trt()
    from face_engine.utils.onnx_batch import batched_model

    graph = batched_model(source) if result.spec.batched_graph else source
    record = {
        "engine": result.path.name, "model": result.spec.model, "precision": result.precision,
        "tensorrt": trt.__version__, "gpu": asdict(result.gpu), "source": _stamp(source),
        "graph": _stamp(Path(graph)) if graph else None, "inputs": result.spec.inputs,
        "batch": result.spec.batch, "pinned_fp32_layers": result.pinned_fp32_layers,
        "build_s": round(result.build_s, 1), "engine_mb": round(result.engine_mb, 1),
        "fidelity": result.fidelity, "latency_ms": result.latency_ms, "ok": result.ok,
        "problems": result.problems,
    }
    result.path.with_suffix(".json").write_text(json.dumps(record, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------- lookup
def find_engine(model: str, source: Path | str, precision: str, *, device_id: int = 0,
                root: Path = ENGINE_DIR) -> Path | None:
    """A verified engine for ``model`` built from this exact ``source`` on this GPU, or None."""
    spec = ENGINE_SPECS.get(model)
    if spec is None or not root.exists():
        return None
    try:
        trt = _trt()
        gpu = discover_gpu(device_id)
    except Exception:  # noqa: BLE001 - no TensorRT / GPU: no AOT engine
        return None
    path = engine_path(spec, gpu, precision, root)
    side = path.with_suffix(".json")
    if not (path.exists() and side.exists()):
        return None
    try:
        record = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    source = Path(source)
    if not (record.get("ok") and record.get("tensorrt") == trt.__version__
            and record.get("gpu", {}).get("sm") == gpu.sm
            and record.get("source") == _stamp(source)):
        return None
    return path


def find_engine_for(source: Path | str, precision: str, *, device_id: int = 0,
                    root: Path = ENGINE_DIR) -> Path | None:
    """:func:`find_engine` by ONNX file instead of zoo name (the masker only has paths)."""
    source = Path(source)
    for spec in ENGINE_SPECS.values():
        found = find_engine(spec.model, source, precision, device_id=device_id, root=root)
        if found is not None:
            return found
    return None


def aot_available(source: Path | str, precision: str, engine_config: Any = None,
                  device_id: int = 0) -> bool:
    """Whether :func:`aot_engine` would return an engine (sidecar checks only)."""
    import os

    if os.environ.get("FACE_ENGINE_AOT", "1") == "0":
        return False
    if engine_config is not None:
        from face_engine.core.config import Provider

        if Provider.TENSORRT not in list(engine_config.providers):
            return False
    return find_engine_for(source, precision, device_id=device_id) is not None


def aot_engine(source: Path | str, precision: str, engine_config: Any = None,
               device_id: int = 0) -> TensorRTEngine | None:
    """A compiled engine for ``source`` to use instead of an ONNX Runtime session.

    Only when the caller's :class:`~face_engine.core.config.EngineConfig` asks for
    TensorRT (a user who chose CUDA or CPU gets what they chose), and only a
    verified engine for this GPU / TensorRT / unchanged model; else None.
    ``FACE_ENGINE_AOT=0`` disables AOT engines entirely.
    """
    if not aot_available(source, precision, engine_config, device_id):
        return None
    path = find_engine_for(source, precision, device_id=device_id)
    if path is None:
        return None
    try:
        return TensorRTEngine(path, device_id)
    except Exception as exc:  # noqa: BLE001 - a bad engine falls back to ONNX Runtime
        logger.warning("AOT engine %s unusable (%s); using ONNX Runtime", path.name, exc)
        return None
