"""Strict TensorRT engines with persistent CUDA I/O-binding buffers.

This module is the model-facing half of the high-throughput renderer.  It
builds sessions through :mod:`roop.trt_session_builder`, which supplies
explicit TensorRT profiles and rejects every provider chain that is not exactly
``["TensorrtExecutionProvider"]``.  Inputs and outputs are allocated once on
the selected CUDA device and rebound by pointer for each enqueue.

The Python ONNX Runtime version currently shipped with this project exposes
``OrtValue.ortvalue_from_shape_and_type`` and pointer-based ``bind_input`` /
``bind_output`` APIs, but does not expose the C++ ``OrtValue.device_tensor``
method.  Using persistent PyTorch CUDA allocations and passing their device
pointers to I/O binding is the supported equivalent: ORT and TensorRT read and
write device memory directly, and no host output copy is performed here.  The
buffers remain owned by this object for the entire engine lifetime.

Dynamic output dimensions that cannot be inferred from the model graph must be
provided as ``output_shapes``.  This is deliberate: silently asking ORT to
allocate an unknown output would break the persistent-buffer contract.
"""

from __future__ import annotations
from roop.degrade import swallowed as _swallowed

import os
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .optimized_processor import VramGovernor
from .trt_session_builder import (
    ProfileShapes,
    Shape,
    TensorRTSessionConfig,
    TensorRTSessionError,
    assert_strict_tensorrt_session,
    build_tensorrt_session,
    derive_profile_shapes,
    prepare_tensorrt_runtime,
)

try:
    import torch
except Exception as _degrade_error:  # pragma: no cover - CPU-only import smoke tests
    _swallowed("roop/optimized_trt_engine.py:46", _degrade_error, "fallback continued")
    torch = None  # type: ignore[assignment]

try:
    import onnxruntime as ort
except Exception as _degrade_error:  # pragma: no cover - optional until an engine is built
    _swallowed("roop/optimized_trt_engine.py:51", _degrade_error, "fallback continued")
    ort = None  # type: ignore[assignment]


TorchTensor = Any
ShapeToken = Union[int, str]
OutputShape = Tuple[ShapeToken, ...]


def _numpy_dtype(ort_type: str) -> np.dtype[Any]:
    """Map an ONNX Runtime tensor type string to a NumPy dtype."""

    mapping: Dict[str, np.dtype[Any]] = {
        "tensor(float)": np.dtype(np.float32),
        "tensor(float16)": np.dtype(np.float16),
        "tensor(double)": np.dtype(np.float64),
        "tensor(int64)": np.dtype(np.int64),
        "tensor(int32)": np.dtype(np.int32),
        "tensor(int16)": np.dtype(np.int16),
        "tensor(int8)": np.dtype(np.int8),
        "tensor(uint8)": np.dtype(np.uint8),
        "tensor(uint16)": np.dtype(np.uint16),
        "tensor(bool)": np.dtype(np.bool_),
    }
    try:
        return mapping[str(ort_type)]
    except KeyError as error:
        raise TensorRTSessionError(
            f"unsupported ONNX tensor type: {ort_type!r}"
        ) from error


def _torch_dtype(numpy_dtype: np.dtype[Any]) -> Any:
    """Map a NumPy dtype to the corresponding PyTorch CUDA dtype."""

    if torch is None:
        raise RuntimeError("PyTorch is required for persistent TensorRT buffers")
    mapping = {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float16): torch.float16,
        np.dtype(np.float64): torch.float64,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int16): torch.int16,
        np.dtype(np.int8): torch.int8,
        np.dtype(np.uint8): torch.uint8,
        np.dtype(np.uint16): torch.uint16,
        np.dtype(np.bool_): torch.bool,
    }
    try:
        return mapping[np.dtype(numpy_dtype)]
    except KeyError as error:
        raise TensorRTSessionError(
            f"unsupported CUDA buffer dtype: {numpy_dtype}"
        ) from error


def _dynamic_dimension(value: Any) -> bool:
    """Return whether an ONNX shape component is symbolic or unknown."""

    if value is None:
        return True
    if isinstance(value, int):
        return value <= 0
    return not str(value).isdigit()


def _is_batch_shape(shape: Sequence[Any]) -> bool:
    """Return whether a tensor has a leading dimension suitable for batching."""

    return len(shape) > 0


def _profile_max_shapes(
    config: TensorRTSessionConfig, model_path: Path
) -> Dict[str, Shape]:
    """Derive the same concrete maximum input shapes used by the provider."""

    minimum, optimal, maximum = config.resolved_batches()
    _minimum_shapes, _optimal_shapes, maximum_shapes = derive_profile_shapes(
        model_path,
        min_batch=minimum,
        opt_batch=optimal,
        max_batch=maximum,
        shape_overrides=config.shape_overrides,
        explicit_profiles=config.explicit_profiles,
        require_dynamic_batch=config.require_dynamic_batch,
        allow_static_batch_one=config.allow_static_batch_one,
        default_spatial_shape=config.default_spatial_shape,
    )
    return maximum_shapes


def _resolve_output_shape(
    declared: Sequence[Any],
    batch: int,
    max_batch: int,
    override: Optional[Sequence[ShapeToken]],
) -> Tuple[int, ...]:
    """Resolve an output shape for persistent allocation.

    ``"batch"`` and ``"max_batch"`` are accepted in an override to make
    detector heads such as ``[B, N, 4]`` readable.  An integer override is the
    maximum concrete value for that dimension.
    """

    if override is not None:
        if len(override) != len(declared):
            raise TensorRTSessionError(
                f"output shape override rank {len(override)} does not match "
                f"declared rank {len(declared)}"
            )
        resolved: List[int] = []
        for index, token in enumerate(override):
            if isinstance(token, str):
                normalized = token.strip().lower()
                if normalized == "batch":
                    resolved.append(int(batch))
                elif normalized in {"max_batch", "max"}:
                    resolved.append(int(max_batch))
                else:
                    try:
                        resolved.append(int(token))
                    except ValueError as error:
                        raise TensorRTSessionError(
                            "unsupported output shape token "
                            f"{token!r} at dimension {index}"
                        ) from error
            else:
                resolved.append(int(token))
        if any(value <= 0 for value in resolved):
            raise TensorRTSessionError(f"output shape must be positive: {resolved!r}")
        return tuple(resolved)

    resolved = []
    for index, dimension in enumerate(declared):
        if index == 0 and _dynamic_dimension(dimension):
            resolved.append(int(batch))
        elif not _dynamic_dimension(dimension):
            resolved.append(int(dimension))
        else:
            raise TensorRTSessionError(
                "persistent output allocation needs an output_shapes override for "
                f"dynamic dimension {index} in declared shape {tuple(declared)!r}"
            )
    return tuple(resolved)


@dataclass(frozen=True)
class TensorRTEngineConfig:
    """One strict model configuration and its persistent-buffer policy."""

    session: TensorRTSessionConfig = field(
        default_factory=TensorRTSessionConfig.from_environment
    )
    output_shapes: Mapping[str, OutputShape] = field(default_factory=dict)
    min_execution_batch: int = 2
    per_item_vram_mb: float = 96.0
    require_persistent_outputs: bool = True

    @classmethod
    def from_environment(
        cls,
        *,
        cache_path: Optional[str | os.PathLike[str]] = None,
        device_id: int = 0,
    ) -> "TensorRTEngineConfig":
        """Create a hardware-tiered configuration from repository settings."""

        return cls(
            session=TensorRTSessionConfig.from_environment(
                cache_path=cache_path,
                device_id=device_id,
            ),
            min_execution_batch=max(
                2, int(os.environ.get("ROOP_TRT_MIN_EXEC_BATCH", "2"))
            ),
            per_item_vram_mb=float(os.environ.get("ROOP_TRT_PER_ITEM_MB", "96")),
        )


@dataclass
class PersistentCudaBuffer:
    """A device allocation held for repeated ORT I/O binding."""

    name: str
    tensor: TorchTensor
    numpy_dtype: np.dtype[Any]
    max_shape: Tuple[int, ...]

    @property
    def data_ptr(self) -> int:
        """Return the stable CUDA pointer passed to ONNX Runtime."""

        return int(self.tensor.data_ptr())


class TensorRTEngine:
    """A strict TensorRT session with reusable CUDA input/output allocations."""

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        config: Optional[TensorRTEngineConfig] = None,
        *,
        session: Any = None,
    ) -> None:
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("CUDA-enabled PyTorch is required for TensorRTEngine")
        if ort is None:
            raise RuntimeError("onnxruntime-gpu is required for TensorRTEngine")
        self.model_path = Path(model_path)
        self.config = config or TensorRTEngineConfig.from_environment()
        self.session_config = self.config.session
        prepare_tensorrt_runtime()
        self.session = session or build_tensorrt_session(
            self.model_path,
            config=self.session_config,
        )
        assert_strict_tensorrt_session(self.session, self.model_path)
        self.device_id = int(self.session_config.device_id)
        self.device = torch.device(f"cuda:{self.device_id}")
        self._lock = threading.RLock()
        self._closed = False
        self._inputs = {meta.name: meta for meta in self.session.get_inputs()}
        self._outputs = {meta.name: meta for meta in self.session.get_outputs()}
        if not self._inputs or not self._outputs:
            raise TensorRTSessionError(
                f"TensorRT model has incomplete I/O: {self.model_path}"
            )
        minimum, optimal, maximum = self.session_config.resolved_batches()
        self.min_batch = int(minimum)
        self.opt_batch = int(optimal)
        self.max_batch = int(maximum)
        if self.max_batch < 2:
            raise TensorRTSessionError(
                f"{self.model_path.name} has no dynamic batch capacity; "
                "strict throughput mode requires max_batch >= 2"
            )
        self._profile_max = _profile_max_shapes(self.session_config, self.model_path)
        for meta in self._inputs.values():
            declared = tuple(meta.shape)
            if declared and not _dynamic_dimension(declared[0]):
                raise TensorRTSessionError(
                    f"{self.model_path.name} input {meta.name!r} has static batch "
                    f"{declared[0]!r}; strict TensorRT batching requires a dynamic axis"
                )
        self.governor = VramGovernor.from_environment(
            device_id=self.device_id,
            requested_batch=self.max_batch,
        )
        safe = self.governor.safe_batch(
            self.max_batch,
            per_item_mb=max(1.0, float(self.config.per_item_vram_mb)),
        )
        self.buffer_batch = max(
            int(self.config.min_execution_batch),
            min(self.max_batch, max(2, int(safe))),
        )
        self.buffer_batch = min(self.max_batch, self.buffer_batch)
        self.input_buffers: Dict[str, PersistentCudaBuffer] = {}
        self.output_buffers: Dict[str, PersistentCudaBuffer] = {}
        self._allocate_buffers()

    @property
    def active_providers(self) -> List[str]:
        """Return the validated provider chain."""

        return [str(provider) for provider in self.session.get_providers()]

    @property
    def input_names(self) -> Tuple[str, ...]:
        """Names in graph order."""

        return tuple(self._inputs)

    @property
    def output_names(self) -> Tuple[str, ...]:
        """Names in graph order."""

        return tuple(self._outputs)

    def _allocate_buffers(self) -> None:
        """Allocate the maximum safe persistent tensors exactly once."""

        with torch.cuda.device(self.device_id):
            for name, meta in self._inputs.items():
                if name not in self._profile_max:
                    raise TensorRTSessionError(
                        f"no TensorRT profile was derived for input {name!r}"
                    )
                profile_shape = tuple(int(value) for value in self._profile_max[name])
                if not _is_batch_shape(profile_shape):
                    raise TensorRTSessionError(f"input {name!r} has no batch dimension")
                shape = (self.buffer_batch,) + profile_shape[1:]
                dtype = _numpy_dtype(meta.type)
                tensor = torch.empty(
                    shape, dtype=_torch_dtype(dtype), device=self.device
                )
                self.input_buffers[name] = PersistentCudaBuffer(
                    name, tensor, dtype, shape
                )

            for name, meta in self._outputs.items():
                declared = tuple(meta.shape)
                override = self.config.output_shapes.get(name)
                shape = _resolve_output_shape(
                    declared,
                    batch=self.buffer_batch,
                    max_batch=self.buffer_batch,
                    override=override,
                )
                dtype = _numpy_dtype(meta.type)
                tensor = torch.empty(
                    shape, dtype=_torch_dtype(dtype), device=self.device
                )
                self.output_buffers[name] = PersistentCudaBuffer(
                    name, tensor, dtype, shape
                )

    def _validate_feed(self, feeds: Mapping[str, TorchTensor]) -> int:
        """Validate names, device placement, dtype, and common batch size."""

        missing = sorted(set(self._inputs) - set(feeds))
        if missing:
            raise ValueError(f"TensorRT feed is missing inputs: {missing!r}")
        unknown = sorted(set(feeds) - set(self._inputs))
        if unknown:
            raise ValueError(f"TensorRT feed contains unknown inputs: {unknown!r}")
        batch: Optional[int] = None
        for name, value in feeds.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"input {name!r} must be a torch.Tensor")
            if not value.is_cuda or value.device != self.device:
                raise ValueError(
                    f"input {name!r} must be on {self.device}, got {value.device}"
                )
            if value.ndim == 0:
                raise ValueError(f"input {name!r} has no batch dimension")
            if batch is None:
                batch = int(value.shape[0])
            elif int(value.shape[0]) != batch:
                raise ValueError("all TensorRT inputs must share the leading batch")
            expected = self.input_buffers[name]
            if tuple(int(item) for item in value.shape[1:]) != expected.max_shape[1:]:
                raise ValueError(
                    f"input {name!r} has shape {tuple(value.shape)}; expected batch x "
                    f"{expected.max_shape[1:]} for its explicit TensorRT profile"
                )
        if batch is None or batch <= 0:
            raise ValueError("TensorRT feed is empty")
        if batch > self.buffer_batch:
            raise MemoryError(
                f"batch {batch} exceeds persistent buffer capacity "
                f"{self.buffer_batch}; "
                "call run_batched so the VRAM governor can split the request"
            )
        return batch

    @staticmethod
    def _pad_tensor(value: TorchTensor, target_batch: int) -> TorchTensor:
        """Pad a CUDA batch by repeating its final sample."""

        current = int(value.shape[0])
        if current >= target_batch:
            return value
        return torch.cat(
            (
                value,
                value[-1:].expand((target_batch - current,) + tuple(value.shape[1:])),
            ),
            dim=0,
        )

    def run_gpu(
        self,
        feeds: Mapping[str, TorchTensor],
        *,
        pad_to_minimum: bool = True,
    ) -> Dict[str, TorchTensor]:
        """Execute one GPU batch and return CUDA tensor views, never NumPy arrays."""

        batch = self._validate_feed(feeds)
        run_batch = max(batch, self.min_batch, int(self.config.min_execution_batch))
        run_batch = min(run_batch, self.buffer_batch)
        if run_batch < batch:
            raise MemoryError("persistent buffer cannot hold the requested batch")
        if not pad_to_minimum:
            run_batch = batch
        padded = {
            name: self._pad_tensor(value, run_batch) if run_batch > batch else value
            for name, value in feeds.items()
        }
        with self._lock, torch.cuda.device(self.device_id):
            binding = self.session.io_binding()
            for name, value in padded.items():
                target = self.input_buffers[name].tensor
                target.narrow(0, 0, run_batch).copy_(value, non_blocking=True)
                binding.bind_input(
                    name,
                    "cuda",
                    self.device_id,
                    self.input_buffers[name].numpy_dtype,
                    tuple(int(item) for item in value.shape),
                    self.input_buffers[name].data_ptr,
                )
            for name, meta in self._outputs.items():
                target = self.output_buffers[name]
                output_shape = _resolve_output_shape(
                    tuple(meta.shape),
                    batch=run_batch,
                    max_batch=self.buffer_batch,
                    override=self.config.output_shapes.get(name),
                )
                binding.bind_output(
                    name,
                    "cuda",
                    self.device_id,
                    target.numpy_dtype,
                    output_shape,
                    target.data_ptr,
                )
            self.session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            result: Dict[str, TorchTensor] = {}
            for name, meta in self._outputs.items():
                target = self.output_buffers[name].tensor
                output_shape = _resolve_output_shape(
                    tuple(meta.shape),
                    batch=run_batch,
                    max_batch=self.buffer_batch,
                    override=self.config.output_shapes.get(name),
                )
                view = target.reshape(self.output_buffers[name].max_shape)
                if len(output_shape) > 0 and _is_batch_shape(output_shape):
                    view = view.narrow(0, 0, int(output_shape[0]))
                view = view.reshape(output_shape)
                result[name] = (
                    view.narrow(0, 0, batch) if output_shape[0] == run_batch else view
                )
            return result

    def run_batched(
        self,
        feeds: Mapping[str, TorchTensor],
        *,
        requested_batch: Optional[int] = None,
    ) -> Dict[str, TorchTensor]:
        """Run a large CUDA feed in governed chunks, shrinking after OOM."""

        self._validate_feed({name: value[:1] for name, value in feeds.items()})
        first = next(iter(feeds.values()))
        total_items = int(first.shape[0])
        for name, value in feeds.items():
            if int(value.shape[0]) != total_items:
                raise ValueError("all TensorRT inputs must share the leading batch")
            expected = self.input_buffers[name].max_shape[1:]
            if tuple(int(item) for item in value.shape[1:]) != expected:
                raise ValueError(
                    f"input {name!r} has shape {tuple(value.shape)}; expected "
                    f"batch x {expected}"
                )
        requested = min(
            self.buffer_batch,
            max(2, int(requested_batch or self.opt_batch)),
        )
        chunk_size = max(2, min(self.buffer_batch, self.governor.safe_batch(requested)))
        chunks: List[Dict[str, TorchTensor]] = []
        start = 0
        while start < total_items:
            item_count = min(chunk_size, total_items - start)
            run_size = max(2, item_count)
            while True:
                current = {
                    name: value[start : start + item_count]
                    for name, value in feeds.items()
                }
                try:
                    chunks.append(self.run_gpu(current, pad_to_minimum=True))
                    break
                except BaseException as error:
                    message = str(error).lower()
                    retryable = any(
                        token in message
                        for token in (
                            "out of memory",
                            "cudaerror",
                            "alloc_failed",
                            "resource exhausted",
                        )
                    )
                    if not retryable or run_size <= 2:
                        raise RuntimeError(
                            "strict TensorRT execution failed; no CUDA/CPU "
                            "fallback was attempted"
                        ) from error
                    self.governor.note_oom(run_size)
                    run_size = max(2, run_size // 2)
                    item_count = min(item_count, run_size)
                    chunk_size = min(chunk_size, run_size)
            start += item_count
        if not chunks:
            return {}
        result: Dict[str, TorchTensor] = {}
        for name in self.output_names:
            result[name] = torch.cat([chunk[name] for chunk in chunks], dim=0)
        return result

    def close(self) -> None:
        """Release persistent references after all CUDA work has completed."""

        if self._closed:
            return
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize(self.device_id)
        self.input_buffers.clear()
        self.output_buffers.clear()
        del self.session
        self._closed = True

    def __enter__(self) -> "TensorRTEngine":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


@dataclass(frozen=True)
class TensorRTModelSpec:
    """Factory input for one detector, swapper, or restorer model."""

    role: str
    model_path: str | os.PathLike[str]
    output_shapes: Mapping[str, OutputShape] = field(default_factory=dict)
    shape_overrides: Mapping[str, Shape] = field(default_factory=dict)
    explicit_profiles: Mapping[str, ProfileShapes] = field(default_factory=dict)


@dataclass
class TensorRTModelBundle:
    """Strictly validated model engines used by a vectorized pipeline."""

    detector: Optional[TensorRTEngine] = None
    swapper: Optional[TensorRTEngine] = None
    restorer: Optional[TensorRTEngine] = None

    def close(self) -> None:
        """Close every constructed engine."""

        for engine in (self.detector, self.swapper, self.restorer):
            if engine is not None:
                engine.close()


def build_tensorrt_engine(
    spec: TensorRTModelSpec,
    base_config: Optional[TensorRTEngineConfig] = None,
) -> TensorRTEngine:
    """Build one strict engine with role-specific profiles and output shapes."""

    source = base_config or TensorRTEngineConfig.from_environment()
    session_config = replace(
        source.session,
        shape_overrides=dict(spec.shape_overrides) or source.session.shape_overrides,
        explicit_profiles=dict(spec.explicit_profiles)
        or source.session.explicit_profiles,
    )
    config = replace(
        source,
        session=session_config,
        output_shapes=dict(spec.output_shapes) or source.output_shapes,
    )
    return TensorRTEngine(spec.model_path, config=config)


def build_model_bundle(
    *,
    detector: Optional[TensorRTModelSpec] = None,
    swapper: Optional[TensorRTModelSpec] = None,
    restorer: Optional[TensorRTModelSpec] = None,
    base_config: Optional[TensorRTEngineConfig] = None,
) -> TensorRTModelBundle:
    """Construct SCRFD/RetinaFace, Inswapper, and CodeFormer/GFPGAN engines.

    Static batch-one exports are rejected by the underlying strict factory.
    This is intentional: an engine that can only execute one face at a time
    cannot meet the batched TRT contract and must be replaced or run through
    the repository's explicit legacy path.
    """

    bundle = TensorRTModelBundle()
    try:
        if detector is not None:
            bundle.detector = build_tensorrt_engine(detector, base_config)
        if swapper is not None:
            bundle.swapper = build_tensorrt_engine(swapper, base_config)
        if restorer is not None:
            bundle.restorer = build_tensorrt_engine(restorer, base_config)
        return bundle
    except BaseException:
        bundle.close()
        raise


__all__ = [
    "OutputShape",
    "PersistentCudaBuffer",
    "TensorRTEngine",
    "TensorRTEngineConfig",
    "TensorRTModelBundle",
    "TensorRTModelSpec",
    "build_model_bundle",
    "build_tensorrt_engine",
]
