"""OptimizedInferenceSession: TensorRT -> CUDA -> CPU, with persistent device
IO binding, for the swap models (HiFiFace, HyperSwap).

Why here and not roop/core/inference_engine.py: roop/core.py is a MODULE. A
roop/core/ package next to it would shadow it and break every
`import roop.core` in the app.

What it does
------------
* Provider chain. 'tensorrt' -> [TRT, CUDA, CPU], 'cuda' -> [CUDA, CPU],
  'cpu' -> [CPU]. ORT never raises when an EP fails to register -- it logs and
  continues down the list -- and it can also DROP the CUDA EP during the first
  run (memories: ep-assertion-and-warmup, trt-allocates-on-first-inference).
  So the active provider is read after a warm-up inference, stored in
  `active_provider`, printed when it is not the requested one, and
  `strict=True` turns that into an error.
* TensorRT options: fp16 on/off, a FIXED optimization profile of batch 1
  (min = opt = max) for every input with a symbolic dim, so a shape never
  triggers an engine rebuild mid-render, and the engine + timing cache under
  `~/.cache/roop-ultimate/trt_cache/<namespace>`. The namespace carries
  precision, GPU, compute capability and the TRT/ORT versions: an FP32 and an
  FP16 engine for one model must never share a directory (the swapper's FP32
  engine once collided with the FP16 ones; memory: swapper-fp16-smudge-fix),
  and an engine is only valid for the GPU + TRT build that made it.
* Zero-copy IO binding. Output buffers are allocated ONCE as torch CUDA
  tensors and bound by pointer; `run_binding` binds the caller's CUDA tensors
  directly (no copy when dtype/shape/contiguity already match, one device copy
  into a persistent staging buffer when they do not) and runs with
  `run_with_iobinding`. Nothing crosses PCIe. ORT runs on a torch stream
  (user_compute_stream) that first waits on the caller's stream, so inputs
  written by torch kernels are complete before the model reads them.
* `convert_onnx_fp16` for a TensorRT FP16 build that fails on an FP32 graph:
  the converted model (float16 weights, float32 I/O kept) is tried once
  before giving up on TensorRT.

PRECISION IS A MEASURED TRADE HERE, not a free speed-up. See
tools/benchmark_swapper.py and the numbers recorded in RECODE_STATUS.md.
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_TORCH_DTYPES = {
    "tensor(float)": "float32", "tensor(float16)": "float16",
    "tensor(int64)": "int64", "tensor(int32)": "int32",
}


def default_cache_root() -> Path:
    """~/.cache/roop-ultimate/trt_cache, or $ROOP_TRT_CACHE_ROOT."""
    env = os.environ.get("ROOP_TRT_CACHE_ROOT")
    return Path(env) if env else Path.home() / ".cache" / "roop-ultimate" / "trt_cache"


def cache_namespace(precision: str, device_id: int = 0) -> str:
    """precision + GPU + sm + TRT + ORT: the conditions an engine is valid for."""
    import onnxruntime
    gpu, sm = "gpu", "sm0"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(device_id)
            major, minor = torch.cuda.get_device_capability(device_id)
            sm = f"sm{major}{minor}"
    except Exception:  # noqa: BLE001 - no torch: the namespace is still unique per precision
        pass
    try:
        import tensorrt
        trt = tensorrt.__version__
    except Exception:  # noqa: BLE001
        trt = "none"
    raw = f"{precision}_{gpu}_{sm}_trt{trt}_ort{onnxruntime.__version__}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)


def _prepare_runtime() -> None:
    """Put TensorRT/CUDA DLLs where ORT's provider loader finds them (Windows).

    Without it the TRT EP fails to register and ORT carries on without it --
    a bare python process benchmarks CUDA or CPU while believing it ran TRT.
    """
    try:
        from roop.trt_session_builder import prepare_tensorrt_runtime
        prepare_tensorrt_runtime()
    except Exception as error:  # noqa: BLE001 - reported by the provider check instead
        print(f"[inference_engine] TensorRT runtime prep skipped: {error}", flush=True)


def _fixed_profile(session_inputs, batch: int = 1) -> Optional[Tuple[str, str, str]]:
    """'name:1x3x256x256,...' for every input with a symbolic dim, else None."""
    parts, dynamic = [], False
    for meta in session_inputs:
        dims = []
        for i, d in enumerate(meta.shape):
            if isinstance(d, int) and d > 0:
                dims.append(str(d))
            elif i == 0:
                dims.append(str(batch))
                dynamic = True
            else:
                return None          # a non-batch symbolic dim: no fixed profile to give
        parts.append(f"{meta.name}:{'x'.join(dims)}")
    if not dynamic:
        return None
    s = ",".join(parts)
    return s, s, s


def _graph_inputs(model_path: str):
    """(name, shape) of the runtime inputs, read with onnx (no session needed)."""
    import onnx
    model = onnx.load(model_path, load_external_data=False)
    init = {i.name for i in model.graph.initializer}

    class _M:
        def __init__(self, name, shape):
            self.name, self.shape = name, shape
    out = []
    for v in model.graph.input:
        if v.name in init:
            continue
        shape = [d.dim_value if d.HasField("dim_value") else (d.dim_param or None)
                 for d in v.type.tensor_type.shape.dim]
        out.append(_M(v.name, shape))
    return out


def is_fp16_graph(model_path: str) -> bool:
    """True when most float initializers are already float16 (hyperswap is)."""
    import onnx
    model = onnx.load(model_path, load_external_data=False)
    f16 = sum(i.data_type == onnx.TensorProto.FLOAT16 for i in model.graph.initializer)
    f32 = sum(i.data_type == onnx.TensorProto.FLOAT for i in model.graph.initializer)
    return f16 > f32


def _rewire_internal_uses_of_outputs(model) -> int:
    """With keep_io_types the converter re-casts each graph output to float32
    (Cast Y_fp16 -> X). A node INSIDE the graph that also reads X then gets
    float32 next to float16 operands. hififace does this: its `mask` output
    feeds Sub_516 (1 - mask) -- "Type parameter (T) of Optype (Sub) bound to
    different types". Point those internal readers at Y, the FP16 tensor
    before the cast. Returns how many inputs were rewired."""
    outputs = {o.name for o in model.graph.output}
    cast_src = {n.output[0]: n.input[0] for n in model.graph.node
                if n.op_type == "Cast" and n.output and n.output[0] in outputs}
    rewired = 0
    for node in model.graph.node:
        if node.op_type == "Cast" and node.output and node.output[0] in cast_src:
            continue
        for i, name in enumerate(node.input):
            if name in cast_src:
                node.input[i] = cast_src[name]
                rewired += 1
    return rewired


# Kept in FP32 on top of onnxconverter-common's DEFAULT_OP_BLOCK_LIST.
# MEASURED on hififace_unofficial_256 (2026-09-29): a plain conversion loads,
# runs, and returns NaN -- its AdaIN blocks take a variance as mean(x^2),
# which overflows float16's 65504. Blocking InstanceNormalization alone still
# gave NaN; blocking these four made it finite.
FP16_EXTRA_BLOCK = ("ReduceMean", "Pow", "Sqrt", "Div")


def convert_onnx_fp16(model_path: str, output_path: str, keep_io_types: bool = True,
                      op_block_list: Optional[Sequence[str]] = None,
                      validate: bool = True) -> str:
    """Write an FP16 copy of `model_path` to `output_path`; return the path used.

    Float32 inputs/outputs are kept (`keep_io_types`), so callers feed and read
    exactly what they did before. A graph that is already FP16 is not converted
    again; its own path is returned. `op_block_list` None = the converter's
    defaults + FP16_EXTRA_BLOCK.

    `validate` runs the converted model on a random input and raises
    ValueError if any output is non-finite: an FP16 model that loads and runs
    is NOT evidence it works (the unblocked hififace conversion did both and
    returned NaN).

    Needs onnxconverter-common (install.js installs it --no-deps).
    """
    if is_fp16_graph(model_path):
        return model_path
    import onnx
    from onnxconverter_common import float16
    model = onnx.load(model_path)
    # The converter names the Casts it inserts around blocked ops from the
    # CONSUMING node's name. hififace_unofficial_256 leaves 38 node names
    # empty, so the casts collided ('_input_cast_0', '_output_cast_0') and the
    # result failed to load ("Duplicate definition of name"). Same in
    # onnxconverter-common 1.14.0 and 1.16.0. Unique names first.
    seen = set()
    for i, node in enumerate(model.graph.node):
        if not node.name or node.name in seen:
            node.name = f"{node.op_type}_{i}"
        seen.add(node.name)
    if op_block_list is None:
        op_block_list = list(float16.DEFAULT_OP_BLOCK_LIST) + list(FP16_EXTRA_BLOCK)
    converted = float16.convert_float_to_float16(model, keep_io_types=keep_io_types,
                                                 op_block_list=list(op_block_list))
    _rewire_internal_uses_of_outputs(converted)
    onnx.checker.check_model(converted)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(converted, output_path)
    if validate:
        _check_finite(output_path)
    return output_path


def _check_finite(model_path: str) -> None:
    import onnxruntime
    _prepare_runtime()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in onnxruntime.get_available_providers()]
    sess = onnxruntime.InferenceSession(model_path, providers=providers)
    rng = np.random.default_rng(0)
    feed = {}
    for meta in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in meta.shape]
        dtype = np.float16 if meta.type == "tensor(float16)" else np.float32
        x = rng.uniform(-1, 1, shape)
        if len(shape) == 2:
            x = x / np.linalg.norm(x, axis=1, keepdims=True)       # an identity vector
        feed[meta.name] = x.astype(dtype)
    for name, out in zip([o.name for o in sess.get_outputs()], sess.run(None, feed)):
        if not np.isfinite(out).all():
            raise ValueError(f"{Path(model_path).name}: output {name!r} is non-finite after "
                             "FP16 conversion; extend op_block_list")


class ProviderFallbackError(RuntimeError):
    """strict=True and the session is not running on the requested provider."""


class OptimizedInferenceSession:
    """onnxruntime.InferenceSession + provider chain + persistent IO binding.

    provider   'tensorrt' | 'cuda' | 'cpu'
    precision  'fp16' | 'fp32' -- reaches the TensorRT EP only; CUDA and CPU
               run the graph's own dtypes (memory: trt-precision-reaches-only-
               tensorrt). For FP16 off TensorRT, convert the model instead.
    """

    def __init__(self, model_path: str, provider: str = "tensorrt", precision: str = "fp16",
                 device_id: int = 0, cache_root: Optional[os.PathLike] = None,
                 batch: int = 1, strict: bool = False, warmup: bool = True,
                 fp16_model_fallback: bool = True, session_options=None) -> None:
        import onnxruntime
        provider = str(provider).lower()
        if provider not in ("tensorrt", "cuda", "cpu"):
            raise ValueError(f"provider must be tensorrt, cuda or cpu, got {provider!r}")
        if precision not in ("fp16", "fp32"):
            raise ValueError(f"precision must be fp16 or fp32, got {precision!r}")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
        self.model_path = str(model_path)
        self.requested_provider = provider
        self.precision = precision
        self.device_id = int(device_id)
        self.batch = int(batch)
        self.strict = strict
        self.cache_dir: Optional[Path] = None
        self.fallback_note: Optional[str] = None
        self._lock = threading.Lock()
        self._cache_root_override = cache_root
        self._torch = None
        self._stream = None

        if provider != "cpu":
            _prepare_runtime()
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available for provider " + provider)
            self._torch = torch
            self._device = torch.device("cuda", self.device_id)
            # ORT runs on this stream; callers' streams are ordered into it.
            self._stream = torch.cuda.Stream(device=self._device)

        if session_options is None:
            from roop.utilities import get_onnx_session_options
            session_options = get_onnx_session_options()
        self._session_options = session_options

        import time
        _t0 = time.perf_counter()
        _t0_wall = time.time()
        _vram0 = 0.0
        try:
            if self._torch is not None and self._torch.cuda.is_available():
                _vram0 = self._torch.cuda.memory_allocated(self.device_id) / (1024.0 * 1024.0)
        except Exception:
            pass

        model_arg = self.model_path
        try:
            self.session = self._build(onnxruntime, model_arg, provider)
            # Before the warm-up: a session ORT rebuilt CPU-only cannot take a
            # CUDA binding, and the error it raises then names a data transfer,
            # not the provider that was lost.
            self._check_provider()
            if warmup:
                self.warmup()
        except Exception as error:
            if not (provider == "tensorrt" and precision == "fp16" and fp16_model_fallback):
                raise
            # TensorRT refused to build this FP32 graph at FP16: try it
            # converted to FP16 once before leaving TensorRT.
            fp16_path = str(self._cache_path() / (Path(model_path).stem + ".fp16.onnx"))
            converted = convert_onnx_fp16(self.model_path, fp16_path)
            self.fallback_note = (f"TensorRT FP16 build failed ({type(error).__name__}: {error}); "
                                  f"retried with FP16 model {Path(converted).name}")
            print(f"[inference_engine] {self.fallback_note}", flush=True)
            self.session = self._build(onnxruntime, converted, provider)
            self._check_provider()
            if warmup:
                self.warmup()

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self._check_provider()        # again: the CUDA EP can be dropped DURING the first run

        try:
            _elapsed = time.perf_counter() - _t0
            _vram1 = 0.0
            if self._torch is not None and self._torch.cuda.is_available():
                _vram1 = self._torch.cuda.memory_allocated(self.device_id) / (1024.0 * 1024.0)
            _vram_cost = max(0.0, _vram1 - _vram0)
            from roop.model_lifecycle import register_model_lifecycle, check_engine_cache_status, format_shape_from_session
            active_p = getattr(self, "active_provider", provider)
            dev_str = f"cuda:{self.device_id}" if provider != "cpu" else "cpu"
            shape_str = format_shape_from_session(self.session)
            cache_status = check_engine_cache_status(Path(self.model_path).stem, self.cache_dir, _t0_wall, active_p)
            register_model_lifecycle(
                model=Path(self.model_path).stem,
                device=dev_str,
                provider=active_p,
                precision=self.precision,
                input_shape=shape_str,
                engine_cache=cache_status,
                vram_cost=_vram_cost,
                init_time=_elapsed,
                session_id=id(self.session),
                extra={"fallback_note": self.fallback_note},
            )
        except Exception:
            pass

    # -- construction ----------------------------------------------------------

    def _cache_path(self) -> Path:
        path = Path(os.fspath(self._cache_root_override)) if getattr(
            self, "_cache_root_override", None) else default_cache_root()
        path = path / cache_namespace(self.precision, self.device_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _providers(self, provider: str, model_arg: str) -> List[Any]:
        chain: List[Any] = []
        stream = str(self._stream.cuda_stream) if self._stream is not None else "0"
        if provider == "tensorrt":
            self.cache_dir = self._cache_path()
            # Booleans as 'True'/'False': ORT 1.23 REJECTS '1'/'0' for these keys,
            # and a rejected option does not raise -- ORT drops TensorRT AND
            # CUDA and retries on CPU alone ("Falling back to
            # ['CPUExecutionProvider']"). Found by this module's own smoke run.
            opts = {
                "device_id": str(self.device_id),
                "trt_fp16_enable": "True" if self.precision == "fp16" else "False",
                "trt_engine_cache_enable": "True",
                "trt_engine_cache_path": str(self.cache_dir),
                "trt_timing_cache_enable": "True",
                "trt_timing_cache_path": str(self.cache_dir),
                # TRT takes only user_compute_stream; has_user_compute_stream is
                # a CUDA-EP key and is rejected here (same silent CPU fallback)
                "user_compute_stream": stream,
            }
            profile = _fixed_profile(_graph_inputs(model_arg), self.batch)
            if profile is not None:
                opts["trt_profile_min_shapes"], opts["trt_profile_opt_shapes"], \
                    opts["trt_profile_max_shapes"] = profile
            chain.append(("TensorrtExecutionProvider", opts))
        if provider in ("tensorrt", "cuda"):
            chain.append(("CUDAExecutionProvider", {
                "device_id": str(self.device_id),
                "has_user_compute_stream": "1",
                "user_compute_stream": stream,
            }))
        chain.append("CPUExecutionProvider")
        return chain

    def _build(self, onnxruntime, model_arg: str, provider: str):
        self.providers = self._providers(provider, model_arg)
        session = onnxruntime.InferenceSession(model_arg, self._session_options,
                                               providers=self.providers)
        self._binding = None
        self._outputs: Dict[str, Any] = {}
        self._staging: Dict[str, Any] = {}
        self.session = session
        return session

    def _check_provider(self) -> None:
        active = self.session.get_providers()
        self.active_provider = {"TensorrtExecutionProvider": "tensorrt",
                                "CUDAExecutionProvider": "cuda",
                                "CPUExecutionProvider": "cpu"}.get(active[0], active[0])
        if self.active_provider != self.requested_provider:
            msg = (f"[inference_engine] {Path(self.model_path).name}: requested "
                   f"{self.requested_provider}, running on {self.active_provider} "
                   f"(active chain {active})")
            if self.strict:
                raise ProviderFallbackError(msg)
            print(msg, flush=True)

    # -- binding -------------------------------------------------------------

    @property
    def on_device(self) -> bool:
        return self._torch is not None

    def _torch_dtype(self, ort_type: str):
        return getattr(self._torch, _TORCH_DTYPES.get(ort_type, "float32"))

    def _concrete(self, shape) -> Tuple[int, ...]:
        out = []
        for i, d in enumerate(shape):
            if isinstance(d, int) and d > 0:
                out.append(d)
            elif i == 0:
                out.append(self.batch)
            else:
                raise ValueError(f"{Path(self.model_path).name}: non-batch dynamic dim in {shape}")
        return tuple(out)

    def _ensure_binding(self) -> None:
        if self._binding is not None:
            return
        torch = self._torch
        self._binding = self.session.io_binding()
        for meta in self.session.get_outputs():
            buf = torch.empty(self._concrete(meta.shape), dtype=self._torch_dtype(meta.type),
                              device=self._device)
            self._outputs[meta.name] = buf
            self._bind(self._binding.bind_output, meta.name, buf)
        for meta in self.session.get_inputs():
            self._staging[meta.name] = torch.zeros(self._concrete(meta.shape),
                                                   dtype=self._torch_dtype(meta.type),
                                                   device=self._device)
        self._bound_inputs: Dict[str, int] = {}

    def _bind(self, fn, name: str, tensor) -> None:
        np_dtype = {self._torch.float32: np.float32, self._torch.float16: np.float16,
                    self._torch.int64: np.int64, self._torch.int32: np.int32}[tensor.dtype]
        fn(name, "cuda", self.device_id, np_dtype, list(tensor.shape), tensor.data_ptr())

    def run_binding(self, inputs: Dict[str, Any], return_all: bool = False, clone: bool = False):
        """Run on device memory only. Returns output[0] (or all, by name).

        Inputs are torch CUDA tensors keyed by input name. A tensor already of
        the model's dtype, shape and contiguity on this device is bound as is
        (zero copy); otherwise it is copied device-to-device into a persistent
        staging buffer. Outputs are the session's PERSISTENT buffers: the next
        call overwrites them, so pass clone=True to keep a result.
        """
        if not self.on_device or self.active_provider == "cpu":
            raise RuntimeError(f"run_binding needs a GPU session; {Path(self.model_path).name} "
                               f"is running on {getattr(self, 'active_provider', 'cpu')}")
        torch = self._torch
        with self._lock:
            self._ensure_binding()
            for meta in self.session.get_inputs():
                if meta.name not in inputs:
                    raise ValueError(f"missing input {meta.name!r}")
                t = inputs[meta.name]
                if not torch.is_tensor(t):
                    raise TypeError(f"{meta.name}: expected a torch tensor, got {type(t).__name__}")
                stage = self._staging[meta.name]
                if tuple(t.shape) != tuple(stage.shape):
                    raise ValueError(f"{meta.name}: shape {tuple(t.shape)}, session is fixed at "
                                     f"{tuple(stage.shape)}")
                if t.device == self._device and t.dtype == stage.dtype and t.is_contiguous():
                    src = t                                   # zero copy
                else:
                    stage.copy_(t, non_blocking=True)
                    src = stage
                if self._bound_inputs.get(meta.name) != src.data_ptr():
                    self._bind(self._binding.bind_input, meta.name, src)
                    self._bound_inputs[meta.name] = src.data_ptr()
            # inputs were written on the caller's stream; ORT runs on ours
            self._stream.wait_stream(torch.cuda.current_stream(self._device))
            self.session.run_with_iobinding(self._binding)
            torch.cuda.current_stream(self._device).wait_stream(self._stream)
            outs = {name: (buf.clone() if clone else buf) for name, buf in self._outputs.items()}
        return outs if return_all else outs[self.output_names[0]]

    def run(self, feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """numpy in, numpy out -- session.run's contract, through the binding
        when on a GPU (one H2D per input, one D2H per output)."""
        if not self.on_device:
            return self.session.run(None, feed)
        torch = self._torch
        tensors = {k: torch.from_numpy(np.ascontiguousarray(v)).to(self._device, non_blocking=False)
                   for k, v in feed.items()}
        outs = self.run_binding(tensors, return_all=True)
        return [outs[name].cpu().numpy() for name in self.output_names]

    def warmup(self, cycles: int = 1) -> None:
        """Zeros through the model: TensorRT builds/loads its engine here, not
        on frame 0, and a CUDA EP that is going to be dropped is dropped here."""
        metas = self.session.get_inputs()
        if self.on_device and getattr(self, "active_provider", None) != "cpu":
            torch = self._torch
            self.input_names = [m.name for m in metas]
            self.output_names = [o.name for o in self.session.get_outputs()]
            feed = {m.name: torch.zeros(self._concrete(m.shape), dtype=self._torch_dtype(m.type),
                                        device=self._device) for m in metas}
            for _ in range(cycles):
                self.run_binding(feed)
            torch.cuda.synchronize(self._device)
        else:
            feed = {m.name: np.zeros(self._concrete(m.shape), dtype=np.float32) for m in metas}
            for _ in range(cycles):
                self.session.run(None, feed)

    def release(self) -> None:
        self._binding = None
        self._outputs.clear()
        self._staging.clear()
        self.session = None
