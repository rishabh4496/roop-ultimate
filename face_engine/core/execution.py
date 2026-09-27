"""ONNX Runtime session management with explicit provider fallback.

:class:`ExecutionEngine` owns every ``InferenceSession`` the engine creates:

* **Provider chain.** Sessions are requested with TensorRT -> CUDA -> CPU (or
  the configured order). Providers the installed onnxruntime build does not
  offer are dropped up front. If session construction raises (a TensorRT
  engine build failure, a CUDA DLL that will not load), the engine retries
  with the next shorter chain.
* **Grant verification.** ORT can silently drop an EP whose libraries fail to
  load and still return a working CPU session; ``session.get_providers()`` is
  the only tell. Every :class:`ManagedSession` records the provider that was
  actually granted, and ``EngineConfig.strict`` turns a silent fallback into
  :class:`ProviderFallbackError`.
* **Session cache.** Identical ``(model file, provider chain, shape profile)``
  requests return the same session object; nothing is re-instantiated.
* **VRAM hygiene.** :meth:`ExecutionEngine.cleanup_vram` runs ``gc.collect()``
  and ``torch.cuda.empty_cache()``; :meth:`ExecutionEngine.release` drops
  cached sessions first so their arenas can actually be freed.
"""
from __future__ import annotations

import gc
import logging
import os
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from face_engine.core.config import EngineConfig, Provider

logger = logging.getLogger(__name__)

ProviderSpec = tuple[str, dict[str, Any]]
ShapeTuple = tuple[int, ...]

_DLL_LOCK = threading.Lock()
_DLL_DIRS: list[str] = []
_DLL_DONE = threading.Event()
_DLL_HANDLES: list[object] = []


class ExecutionError(RuntimeError):
    """Base class for execution-layer failures."""


class ProviderFallbackError(ExecutionError):
    """Raised in strict mode when ORT granted a lower provider than requested."""


class SessionCreationError(ExecutionError):
    """Raised when no provider chain could build a session for a model."""


@dataclass(frozen=True)
class ShapeProfile:
    """TensorRT optimisation profile for dynamic-shape inputs.

    Each mapping is ``input name -> shape``; all three must name the same
    inputs. Rendered to the ``"name:1x3x640x640,other:..."`` form the
    TensorRT EP expects.
    """

    min_shapes: Mapping[str, ShapeTuple]
    opt_shapes: Mapping[str, ShapeTuple]
    max_shapes: Mapping[str, ShapeTuple]

    def __post_init__(self) -> None:
        names = set(self.min_shapes)
        if names != set(self.opt_shapes) or names != set(self.max_shapes):
            raise ValueError("min/opt/max shape profiles must name the same inputs")
        for name in names:
            lo, opt, hi = self.min_shapes[name], self.opt_shapes[name], self.max_shapes[name]
            if not (len(lo) == len(opt) == len(hi)):
                raise ValueError(f"profile for {name!r} has mismatched ranks")
            if any(a > b or b > c for a, b, c in zip(lo, opt, hi)):
                raise ValueError(f"profile for {name!r} must satisfy min <= opt <= max")

    @staticmethod
    def _render(shapes: Mapping[str, ShapeTuple]) -> str:
        return ",".join(f"{name}:{'x'.join(str(d) for d in shape)}"
                        for name, shape in sorted(shapes.items()))

    def trt_options(self) -> dict[str, str]:
        """TensorRT EP option entries for this profile."""
        return {
            "trt_profile_min_shapes": self._render(self.min_shapes),
            "trt_profile_opt_shapes": self._render(self.opt_shapes),
            "trt_profile_max_shapes": self._render(self.max_shapes),
        }

    def cache_key(self) -> tuple[str, str, str]:
        """Hashable identity for the session cache."""
        opts = self.trt_options()
        return (opts["trt_profile_min_shapes"], opts["trt_profile_opt_shapes"],
                opts["trt_profile_max_shapes"])

    @classmethod
    def bounding_box_mask_profile(
        cls,
        input_name: str = "input",
        *,
        min_batch: int = 1,
        opt_batch: int = 4,
        max_batch: int = 8,
        channels: int = 3,
        height: int = 256,
        width: int = 256,
    ) -> ShapeProfile:
        """Dynamic batching profile for bounding boxes / masks: [1, 3, 256, 256] to [8, 3, 256, 256]."""
        return cls(
            min_shapes={input_name: (min_batch, channels, height, width)},
            opt_shapes={input_name: (opt_batch, channels, height, width)},
            max_shapes={input_name: (max_batch, channels, height, width)},
        )

    @classmethod
    def super_resolution_profile(
        cls,
        input_name: str = "input",
        *,
        min_batch: int = 1,
        opt_batch: int = 2,
        max_batch: int = 4,
        channels: int = 3,
        height: int = 512,
        width: int = 512,
    ) -> ShapeProfile:
        """Dynamic batching profile for super-resolution: [1, 3, 512, 512] to [4, 3, 512, 512]."""
        return cls(
            min_shapes={input_name: (min_batch, channels, height, width)},
            opt_shapes={input_name: (opt_batch, channels, height, width)},
            max_shapes={input_name: (max_batch, channels, height, width)},
        )


@dataclass(frozen=True)
class ManagedSession:
    """An ``InferenceSession`` plus what was asked for and what was granted.

    Attributes:
        session: The live ORT session.
        model_path: Absolute path of the model file.
        requested: Provider names in the chain that built this session.
        granted: ``session.get_providers()`` at creation.
        wanted: The first provider of the full, filtered preference chain —
            what the caller would have got had nothing failed.
        provider_options: The options each provider in ``requested`` was
            built with (e.g. the CUDA ``gpu_mem_limit`` derived from the VRAM
            free at that moment — it differs from a later
            :meth:`ExecutionEngine.cuda_mem_limit` call).
    """

    session: ort.InferenceSession
    model_path: Path
    requested: tuple[str, ...]
    granted: tuple[str, ...]
    wanted: str
    input_names: tuple[str, ...] = field(default=())
    output_names: tuple[str, ...] = field(default=())
    provider_options: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def primary_provider(self) -> str:
        """The provider ORT actually placed first."""
        return self.granted[0]

    @property
    def fell_back(self) -> bool:
        """True when the granted primary provider is not the one wanted."""
        return self.primary_provider != self.wanted

    @property
    def on_gpu(self) -> bool:
        """True when the primary provider executes on a GPU."""
        return self.primary_provider in (Provider.TENSORRT.value, Provider.CUDA.value)

    def run(self, feeds: Mapping[str, np.ndarray],
            output_names: Sequence[str] | None = None) -> list[np.ndarray]:
        """Run the session with host-memory inputs and outputs."""
        return self.session.run(list(output_names) if output_names else None, dict(feeds))

    def run_binding(
        self,
        input_tensors: Mapping[str, Any],
        *,
        output_tensors: Mapping[str, Any] | None = None,
        output_shapes: Mapping[str, Sequence[int]] | None = None,
        device_id: int | None = None,
    ) -> dict[str, Any]:
        """Run with zero-copy IOBinding directly on CUDA memory pointers."""
        return run_binding(
            self,
            input_tensors,
            output_tensors=output_tensors,
            output_shapes=output_shapes,
            device_id=device_id,
        )


def register_gpu_runtime_dirs() -> list[str]:
    """Make TensorRT / CUDA / cuDNN libraries loadable by ONNX Runtime.

    On Windows the TensorRT EP and cuDNN are loaded lazily by name; unless
    their folders are on the DLL search path ORT drops the EP and silently
    runs on CPU. This registers, once per process, the ``tensorrt_libs``
    package, ``torch/lib`` and every ``nvidia/*/bin`` wheel folder via
    ``os.add_dll_directory`` and prepends them to this process's ``PATH``
    (the TensorRT EP resolves ``nvinfer`` through the ordinary search order).
    It then calls ``onnxruntime.preload_dlls()`` where available (ORT >= 1.21).

    A no-op off Windows apart from the ORT preload. Returns the registered
    directories.
    """
    with _DLL_LOCK:
        if _DLL_DONE.is_set():
            return list(_DLL_DIRS)
        found: list[str] = []

        def _add(directory: str | None) -> None:
            if directory and os.path.isdir(directory) and directory not in found:
                found.append(directory)

        if sys.platform == "win32":
            for module_name, sub in (("tensorrt_libs", ""), ("tensorrt", "../tensorrt_libs"),
                                     ("torch", "lib")):
                try:
                    module = __import__(module_name)
                except Exception:  # noqa: BLE001 - any import failure means "not present"
                    continue
                module_file = getattr(module, "__file__", None)
                if module_file:
                    _add(os.path.normpath(os.path.join(os.path.dirname(module_file), sub)))
            try:
                import nvidia  # type: ignore[import-not-found]
                for root in list(getattr(nvidia, "__path__", [])):
                    for current, _dirs, _files in os.walk(root):
                        if os.path.basename(current).lower() == "bin":
                            _add(current)
            except Exception:  # noqa: BLE001
                pass
            for directory in found:
                try:
                    _DLL_HANDLES.append(os.add_dll_directory(directory))
                except OSError:
                    continue
            if found:
                os.environ["PATH"] = os.pathsep.join(found + [os.environ.get("PATH", "")])
        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            try:
                preload()
            except Exception as exc:  # noqa: BLE001 - preload is best-effort
                logger.debug("onnxruntime.preload_dlls failed: %s", exc)
        _DLL_DIRS.extend(found)
        _DLL_DONE.set()
        return list(found)


def _torch_cuda() -> Any:
    """``torch.cuda`` if torch is importable with CUDA, else None."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    return torch.cuda if torch.cuda.is_available() else None


def query_vram(device_id: int = 0) -> tuple[int, int] | None:
    """``(free_bytes, total_bytes)`` for a CUDA device, or None when unknown.

    Uses ``torch.cuda.mem_get_info`` (a driver query; it does not allocate a
    tensor, but it does initialise a CUDA context on first call).
    """
    cuda = _torch_cuda()
    if cuda is None or device_id >= cuda.device_count():
        return None
    free, total = cuda.mem_get_info(device_id)
    return int(free), int(total)


class ExecutionEngine:
    """Creates, caches and releases ONNX Runtime sessions.

    Thread-safe: concurrent :meth:`get_session` calls for the same key build
    exactly one session.

    Example:
        >>> engine = ExecutionEngine()
        >>> handle = engine.get_session("models/scrfd_10g_bnkps.onnx")
        >>> handle.primary_provider
        'TensorrtExecutionProvider'
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config: EngineConfig = config or EngineConfig()
        self._sessions: dict[tuple[Any, ...], ManagedSession] = {}
        self._lock = threading.RLock()
        self._build_locks: dict[tuple[Any, ...], threading.Lock] = {}
        if self.config.register_gpu_dlls:
            register_gpu_runtime_dirs()

    # ------------------------------------------------------------------ providers
    @staticmethod
    def available_providers() -> list[str]:
        """Providers compiled into the installed onnxruntime build."""
        return list(ort.get_available_providers())

    def cuda_mem_limit(self) -> int | None:
        """Byte limit for the CUDA EP arena, derived from physical VRAM.

        ``total * vram_fraction``; with ``free_vram_fraction`` set, also capped
        at that fraction of the VRAM free right now (see
        :class:`~face_engine.core.config.CUDAOptions` for why that is off by
        default). None when VRAM cannot be queried (ORT then uses its default,
        i.e. unlimited).
        """
        opts = self.config.cuda
        if opts.gpu_mem_limit_override is not None:
            return opts.gpu_mem_limit_override
        vram = query_vram(self.config.device_id)
        if vram is None:
            return None
        free, total = vram
        limit = total * opts.vram_fraction
        if opts.free_vram_fraction is not None:
            limit = min(limit, free * opts.free_vram_fraction)
        return max(1, int(limit))

    def _provider_options(self, provider: Provider, profile: ShapeProfile | None,
                          trt_fp16: bool | None = None) -> dict[str, Any]:
        if provider is Provider.TENSORRT:
            options = self.config.tensorrt_provider_options()
            if trt_fp16 is not None and trt_fp16 != options["trt_fp16_enable"]:
                # A separate engine cache per precision: an FP16 engine must never
                # be served to a session that asked for FP32, or vice versa.
                options["trt_fp16_enable"] = trt_fp16
                options["trt_engine_cache_path"] = str(
                    Path(options["trt_engine_cache_path"]) / ("fp16" if trt_fp16 else "fp32"))
            if self.config.tensorrt.trt_engine_cache_enable:
                Path(options["trt_engine_cache_path"]).mkdir(parents=True, exist_ok=True)
            if profile is not None:
                options.update(profile.trt_options())
            return options
        if provider is Provider.CUDA:
            options: dict[str, Any] = {
                "device_id": self.config.device_id,
                "arena_extend_strategy": self.config.cuda.arena_extend_strategy,
                "cudnn_conv_algo_search": self.config.cuda.cudnn_conv_algo_search,
                "do_copy_in_default_stream": self.config.cuda.do_copy_in_default_stream,
            }
            limit = self.cuda_mem_limit()
            if limit is not None:
                options["gpu_mem_limit"] = limit
            return options
        return {}

    def resolve_providers(self, providers: Sequence[Provider] | None = None,
                          profile: ShapeProfile | None = None,
                          trt_fp16: bool | None = None) -> list[ProviderSpec]:
        """Preference-ordered ``(name, options)`` pairs this build can offer.

        Raises:
            ExecutionError: if none of the requested providers is available.
        """
        requested = list(providers) if providers is not None else list(self.config.providers)
        available = set(self.available_providers())
        chain = [(p.value, self._provider_options(p, profile, trt_fp16))
                 for p in requested if p.value in available]
        dropped = [p.value for p in requested if p.value not in available]
        if dropped:
            logger.info("providers not in this onnxruntime build: %s", ", ".join(dropped))
        if not chain:
            raise ExecutionError(
                f"none of {[p.value for p in requested]} is available; "
                f"onnxruntime offers {sorted(available)}")
        return chain

    # ------------------------------------------------------------------ sessions
    def _session_options(self) -> ort.SessionOptions:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = self.config.log_severity_level
        if self.config.intra_op_num_threads:
            options.intra_op_num_threads = self.config.intra_op_num_threads
        return options

    @staticmethod
    def _cache_key(model_path: Path, chain_names: Sequence[str],
                   profile: ShapeProfile | None, trt_fp16: bool | None) -> tuple[Any, ...]:
        stat = model_path.stat()
        # Size + mtime: a model replaced on disk under the same name is a new model.
        return (str(model_path), stat.st_size, stat.st_mtime_ns, tuple(chain_names),
                profile.cache_key() if profile is not None else None, trt_fp16)

    def get_session(self, model_path: os.PathLike[str] | str, *,
                    providers: Sequence[Provider] | None = None,
                    shape_profile: ShapeProfile | None = None,
                    trt_fp16: bool | None = None) -> ManagedSession:
        """Return a cached or newly built session for ``model_path``.

        Args:
            model_path: ONNX file.
            providers: Override the configured provider preference.
            shape_profile: TensorRT optimisation profile for dynamic inputs.
            trt_fp16: Override ``trt_fp16_enable`` for this session only (with
                its own engine cache). Swap models need ``False``: FP16
                overflows in them (see face_engine.processors.swapper).

        Raises:
            FileNotFoundError: the model file does not exist.
            SessionCreationError: every provider chain failed.
            ProviderFallbackError: strict mode and a lower provider was granted.
        """
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"model not found: {path}")
        chain = self.resolve_providers(providers, shape_profile, trt_fp16)
        key = self._cache_key(path, [name for name, _ in chain], shape_profile, trt_fp16)

        with self._lock:
            cached = self._sessions.get(key)
            if cached is not None:
                return cached
            build_lock = self._build_locks.setdefault(key, threading.Lock())

        with build_lock:
            with self._lock:
                cached = self._sessions.get(key)
                if cached is not None:
                    return cached
            handle = self._build(path, chain)
            with self._lock:
                self._sessions[key] = handle
                self._build_locks.pop(key, None)
            return handle

    def _build(self, path: Path, chain: list[ProviderSpec]) -> ManagedSession:
        wanted = chain[0][0]
        errors: list[str] = []
        for start in range(len(chain)):
            attempt = chain[start:]
            names = [name for name, _ in attempt]
            try:
                session = ort.InferenceSession(
                    str(path), sess_options=self._session_options(),
                    providers=names, provider_options=[opts for _, opts in attempt])
            except Exception as exc:  # noqa: BLE001 - ORT raises bare RuntimeError/Fail
                errors.append(f"{names[0]}: {exc}")
                logger.warning("session for %s failed on %s, falling back: %s",
                               path.name, names[0], exc)
                continue
            granted = tuple(session.get_providers())
            handle = ManagedSession(
                session=session, model_path=path, requested=tuple(names), granted=granted,
                wanted=wanted,
                input_names=tuple(i.name for i in session.get_inputs()),
                output_names=tuple(o.name for o in session.get_outputs()),
                provider_options={name: dict(opts) for name, opts in attempt})
            if handle.fell_back:
                message = (f"{path.name}: wanted {wanted}, onnxruntime granted "
                           f"{handle.primary_provider} (providers {list(granted)})")
                if self.config.strict:
                    raise ProviderFallbackError(message)
                logger.warning(message)
            return handle
        raise SessionCreationError(
            f"could not create a session for {path}: " + " | ".join(errors))

    @property
    def cached_sessions(self) -> list[ManagedSession]:
        """Snapshot of every live cached session."""
        with self._lock:
            return list(self._sessions.values())

    # ------------------------------------------------------------------ buffers
    def device_type(self, handle: ManagedSession) -> str:
        """ORT device string for tensors bound to ``handle`` ('cuda' or 'cpu')."""
        return "cuda" if handle.on_gpu else "cpu"

    def allocate(self, handle: ManagedSession, array: np.ndarray) -> ort.OrtValue:
        """Copy ``array`` into an ``OrtValue`` on the session's device."""
        return ort.OrtValue.ortvalue_from_numpy(
            np.ascontiguousarray(array), self.device_type(handle), self.config.device_id)

    def empty(self, handle: ManagedSession, shape: Sequence[int],
              dtype: np.dtype[Any] | type = np.float32) -> ort.OrtValue:
        """Allocate an uninitialised ``OrtValue`` on the session's device."""
        return ort.OrtValue.ortvalue_from_shape_and_type(
            list(shape), dtype, self.device_type(handle), self.config.device_id)

    def run_bound(self, handle: ManagedSession, inputs: Mapping[str, ort.OrtValue],
                  outputs: Mapping[str, ort.OrtValue]) -> None:
        """Run with device-resident inputs and preallocated outputs (no host copies)."""
        binding = handle.session.io_binding()
        for name, value in inputs.items():
            binding.bind_ortvalue_input(name, value)
        for name, value in outputs.items():
            binding.bind_ortvalue_output(name, value)
        binding.synchronize_inputs()
        handle.session.run_with_iobinding(binding)
        binding.synchronize_outputs()

    def run_binding(
        self,
        session: ManagedSession | ort.InferenceSession,
        input_tensors: Mapping[str, Any],
        *,
        output_tensors: Mapping[str, Any] | None = None,
        output_shapes: Mapping[str, Sequence[int]] | None = None,
        device_id: int | None = None,
    ) -> dict[str, Any]:
        """Execute inference with zero-copy IOBinding directly on CUDA memory pointers."""
        return run_binding(
            session,
            input_tensors,
            output_tensors=output_tensors,
            output_shapes=output_shapes,
            device_id=device_id if device_id is not None else self.config.device_id,
        )

    # ------------------------------------------------------------------ cleanup
    def release(self, model_path: os.PathLike[str] | str | None = None) -> int:
        """Drop cached sessions (all, or those for one model). Returns the count.

        Callers must also drop their own references for the arena to be freed.
        """
        with self._lock:
            if model_path is None:
                count = len(self._sessions)
                self._sessions.clear()
            else:
                target = str(Path(model_path).expanduser().resolve())
                keys = [k for k in self._sessions if k[0] == target]
                for k in keys:
                    del self._sessions[k]
                count = len(keys)
        self.cleanup_vram()
        return count

    @staticmethod
    def cleanup_vram() -> None:
        """Collect garbage and return torch's cached CUDA blocks to the driver.

        ORT's own arena is freed only when its session is destroyed; call
        :meth:`release` first to drop the engine's references.
        """
        gc.collect()
        cuda = _torch_cuda()
        if cuda is not None:
            cuda.empty_cache()
            if hasattr(cuda, "ipc_collect"):
                cuda.ipc_collect()

    def close(self) -> None:
        """Release every session and clean up."""
        self.release()

    def __enter__(self) -> ExecutionEngine:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

_ONNX_TO_TORCH_DTYPE: dict[str, Any] = {}
_TORCH_TO_NP_DTYPE: dict[Any, Any] = {}


def _init_torch_types() -> None:
    global _ONNX_TO_TORCH_DTYPE, _TORCH_TO_NP_DTYPE
    if _ONNX_TO_TORCH_DTYPE:
        return
    try:
        import torch
        _ONNX_TO_TORCH_DTYPE.update({
            "tensor(float)": torch.float32,
            "tensor(float16)": torch.float16,
            "tensor(double)": torch.float64,
            "tensor(int32)": torch.int32,
            "tensor(int64)": torch.int64,
            "tensor(int8)": torch.int8,
            "tensor(uint8)": torch.uint8,
            "tensor(bool)": torch.bool,
            "tensor(bfloat16)": getattr(torch, "bfloat16", torch.float16),
        })
        _TORCH_TO_NP_DTYPE.update({
            torch.float32: np.float32,
            torch.float16: np.float16,
            torch.float64: np.float64,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.int8: np.int8,
            torch.uint8: np.uint8,
            torch.bool: np.bool_,
        })
    except Exception:
        pass


def run_binding(
    session: ManagedSession | ort.InferenceSession,
    input_tensors: Mapping[str, Any],
    *,
    output_tensors: Mapping[str, Any] | None = None,
    output_shapes: Mapping[str, Sequence[int]] | None = None,
    device_id: int | None = None,
) -> dict[str, Any]:
    """Execute inference with zero-copy IOBinding directly on CUDA memory pointers.

    Pre-allocates output ``torch.empty(...)`` tensors directly on the target device
    (device="cuda" when running on GPU), binds raw CUDA memory pointers via
    ``tensor.data_ptr()``, and invokes ``session.run_with_iobinding(binding)``
    without transferring tensors back to host CPU memory.

    Memory strategy: session graphs remain warm in VRAM across frames; per-frame
    allocator thrash is eliminated, and neither ``empty_cache()`` nor ``gc.collect()``
    is invoked within the frame loop.

    Args:
        session: Live :class:`ManagedSession` or :class:`onnxruntime.InferenceSession`.
        input_tensors: Mapping of input node names to ``torch.Tensor`` buffers.
        output_tensors: Optional pre-allocated destination tensors.
        output_shapes: Optional explicit shapes for dynamic outputs.
        device_id: CUDA device ordinal (default 0 or session's configured device).

    Returns:
        Mapping of output node names to ``torch.Tensor`` results residing on
        the execution device.
    """
    import torch
    _init_torch_types()

    if isinstance(session, ManagedSession):
        ort_sess = session.session
        on_gpu = session.on_gpu
        target_dev_id = session.provider_options.get(session.primary_provider, {}).get("device_id", 0) if on_gpu else 0
    else:
        ort_sess = session
        providers = ort_sess.get_providers()
        on_gpu = len(providers) > 0 and providers[0] in (
            Provider.TENSORRT.value, Provider.CUDA.value
        )
        target_dev_id = 0

    if device_id is not None:
        target_dev_id = device_id

    first_tensor = next(iter(input_tensors.values())) if input_tensors else None
    if first_tensor is not None and getattr(first_tensor, "is_cuda", False):
        dev_type = "cuda"
        dev_id = first_tensor.device.index if first_tensor.device.index is not None else target_dev_id
    elif on_gpu:
        dev_type = "cuda"
        dev_id = target_dev_id
    else:
        dev_type = "cpu"
        dev_id = 0

    binding = ort_sess.io_binding()
    dim_map: dict[str, int] = {}
    batch_size: int | None = None

    for inp_node in ort_sess.get_inputs():
        if inp_node.name in input_tensors:
            t = input_tensors[inp_node.name]
            if not t.is_contiguous():
                t = t.contiguous()
            if dev_type == "cuda" and not t.is_cuda:
                t = t.to(device=f"cuda:{dev_id}")
            np_dtype = _TORCH_TO_NP_DTYPE.get(t.dtype, np.float32)
            binding.bind_input(
                name=inp_node.name,
                device_type=dev_type,
                device_id=dev_id,
                element_type=np_dtype,
                shape=tuple(t.shape),
                buffer_ptr=t.data_ptr(),
            )
            for sym, val in zip(inp_node.shape, t.shape):
                if isinstance(sym, str):
                    dim_map[sym] = int(val)
            if batch_size is None and len(t.shape) > 0:
                batch_size = int(t.shape[0])

    if batch_size is None and first_tensor is not None and len(first_tensor.shape) > 0:
        batch_size = int(first_tensor.shape[0])
    if batch_size is None:
        batch_size = 1

    results: dict[str, Any] = {}
    cuda_device = torch.device("cuda", dev_id) if dev_type == "cuda" else torch.device("cpu")

    for out_node in ort_sess.get_outputs():
        if output_tensors is not None and out_node.name in output_tensors:
            out_t = output_tensors[out_node.name]
            if not out_t.is_contiguous():
                out_t = out_t.contiguous()
            np_dtype = _TORCH_TO_NP_DTYPE.get(out_t.dtype, np.float32)
        else:
            out_dtype = _ONNX_TO_TORCH_DTYPE.get(out_node.type, torch.float32)
            np_dtype = _TORCH_TO_NP_DTYPE.get(out_dtype, np.float32)
            if output_shapes is not None and out_node.name in output_shapes:
                out_shape = tuple(output_shapes[out_node.name])
            else:
                resolved = []
                for idx, dim in enumerate(out_node.shape):
                    if isinstance(dim, int) and dim > 0:
                        resolved.append(dim)
                    elif isinstance(dim, str) and dim in dim_map:
                        resolved.append(dim_map[dim])
                    elif idx == 0 and batch_size is not None:
                        resolved.append(batch_size)
                    else:
                        raise ValueError(
                            f"Unable to resolve dynamic shape dimension {dim!r} at index {idx} "
                            f"for output {out_node.name!r} of model {getattr(session, 'model_path', 'unknown')}. "
                            f"Pass explicit output_shapes or output_tensors."
                        )
                out_shape = tuple(resolved)
            out_t = torch.empty(out_shape, dtype=out_dtype, device=cuda_device)

        results[out_node.name] = out_t
        binding.bind_output(
            name=out_node.name,
            device_type=dev_type,
            device_id=dev_id,
            element_type=np_dtype,
            shape=tuple(out_t.shape),
            buffer_ptr=out_t.data_ptr(),
        )

    ort_sess.run_with_iobinding(binding)
    return results
