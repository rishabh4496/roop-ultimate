"""TensorRT-backed batched face swapping.

The public class in this module is deliberately small and model-facing.  The
existing frame processor can keep its richer alignment/compositing logic while
using :class:`TensorRTFaceSwapper` for the actual inswapper inference.  Inputs
and outputs stay on the selected CUDA device for the ORT I/O-binding call.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - CPU-only import environments
    torch = None

from roop.trt_engine import TensorRTInferenceSession

logger = logging.getLogger("roop.face_swapper")


INSWAPPER_DYNAMIC_SHAPE_PROFILE = {
    "min": "target:1x3x128x128,source:1x512",
    "opt": "target:4x3x128x128,source:4x512",
    "max": "target:8x3x128x128,source:8x512",
}


def _static_batch(model_path: str) -> bool:
    """Return whether the model still declares a fixed batch of one."""
    try:
        import onnx

        model = onnx.load(model_path, load_external_data=False)
        for value_info in list(model.graph.input) + list(model.graph.output):
            dims = value_info.type.tensor_type.shape.dim
            if dims and dims[0].dim_param:
                return False
            if dims and dims[0].dim_value not in (0, 1):
                return False
        return True
    except Exception:
        # A failed inspection should not prevent the regular ORT fallback from
        # trying the original model.
        return False


def _dynamic_model_copy(model_path: str) -> str:
    """Create a cached batch-dynamic inswapper graph when needed.

    The released inswapper graph is commonly exported with batch=1.  Passing a
    TensorRT profile alone cannot change that graph.  This private copy changes
    only the leading runtime dimensions and the leading batch value of constant
    Reshape shapes.  The source model is never modified.
    """
    if not os.path.isfile(model_path) or not _static_batch(model_path):
        return model_path
    if os.environ.get("ROOP_TRT_RELAX_STATIC_BATCH", "1").strip().lower() in {
        "0", "false", "no", "off"
    }:
        return model_path
    try:
        import onnx

        source = Path(model_path)
        cache_dir = source.parent / ".stage3"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / f"{source.stem}.batch.onnx"
        if cached.exists() and cached.stat().st_mtime >= source.stat().st_mtime:
            return str(cached)

        model = onnx.load(str(source), load_external_data=False)
        initializers = {item.name for item in model.graph.initializer}
        for value_info in list(model.graph.input) + list(model.graph.output):
            if value_info.name in initializers:
                continue
            dims = value_info.type.tensor_type.shape.dim
            if dims:
                dims[0].dim_param = "N"
                dims[0].ClearField("dim_value")

        constants = {
            node.output[0]: node
            for node in model.graph.node
            if node.op_type == "Constant" and node.output and node.attribute
        }
        for node in model.graph.node:
            if node.op_type != "Reshape" or len(node.input) < 2:
                continue
            constant = constants.get(node.input[1])
            if constant is None:
                continue
            attribute = next(
                (
                    item
                    for item in constant.attribute
                    if item.name == "value" and item.HasField("t")
                ),
                None,
            )
            if attribute is None:
                continue
            shape = onnx.numpy_helper.to_array(attribute.t)
            if shape.ndim != 1 or shape.size < 2 or int(shape[0]) != 1:
                continue
            dynamic_shape = np.array(shape, copy=True)
            dynamic_shape[0] = 0
            attribute.t.CopyFrom(
                onnx.numpy_helper.from_array(dynamic_shape, attribute.t.name)
            )
        onnx.save(model, str(cached))
        return str(cached)
    except Exception as exc:
        logger.warning("Could not create a dynamic inswapper graph: %s", exc)
        return model_path


class TensorRTFaceSwapper:
    """Run ``inswapper_128.onnx`` with CUDA TensorRT I/O binding.

    ``swap_batch`` accepts normalized ``(N, 3, 128, 128)`` target crops and
    ``(N, 512)`` ArcFace embeddings.  The tensors must already be contiguous
    float32 CUDA tensors on this object's device.  Keeping that requirement
    explicit prevents an accidental host round trip from being hidden inside
    the hot path.
    """

    shape_profile = dict(INSWAPPER_DYNAMIC_SHAPE_PROFILE)
    target_shape = (3, 128, 128)
    source_shape = (512,)
    max_batch = 8

    def __init__(self, model_path: str, device_id: int = 0):
        if torch is None:
            raise RuntimeError("PyTorch is required for TensorRT face swapping")
        self.original_model_path = os.path.abspath(model_path)
        self.model_path = _dynamic_model_copy(self.original_model_path)
        self.device_id = int(device_id)
        self.device = torch.device(f"cuda:{self.device_id}")
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRTFaceSwapper requires a CUDA device")

        cache_dir = os.path.abspath(
            os.environ.get(
                "ROOP_TRT_CACHE_DIR",
                os.path.join(os.path.dirname(self.model_path), "trt_cache"),
            )
        )
        self.engine = TensorRTInferenceSession(
            model_path=self.model_path,
            device_id=self.device_id,
            enable_fp16=True,
            cache_dir=cache_dir,
            dynamic_shape_profile=self.shape_profile,
        )
        self.session = self.engine.get_session()
        self.io_binding = self.session.io_binding()
        self._lock = threading.RLock()
        self._closed = False
        # Some TensorRT/ORT builds accept the dynamic input profile but expose
        # the released inswapper graph's output as batch-one at execution time.
        # Keep the fast path enabled and remember that a runtime fallback was
        # needed so subsequent calls do not repeat a costly failed launch.
        self._batch_unsupported = False
        self.target_name, self.source_name = self._resolve_input_names()
        self.output_name = self._resolve_output_name()

    def _resolve_input_names(self):
        inputs = list(self.session.get_inputs())
        by_name = {item.name: item.name for item in inputs}
        target = by_name.get("target")
        source = by_name.get("source")
        if target is None:
            target = next((item.name for item in inputs if len(item.shape or []) == 4), None)
        if source is None:
            source = next((item.name for item in inputs if len(item.shape or []) == 2), None)
        if target is None or source is None:
            raise RuntimeError(
                "inswapper must expose one rank-4 target input and one rank-2 source input"
            )
        return target, source

    def _resolve_output_name(self) -> str:
        outputs = list(self.session.get_outputs())
        if not outputs:
            raise RuntimeError("inswapper has no output tensor")
        return next((item.name for item in outputs if item.name == "output"), outputs[0].name)

    def _validate_inputs(self, target_tensors, source_embeddings) -> int:
        if target_tensors.ndim != 4 or tuple(target_tensors.shape[1:]) != self.target_shape:
            raise ValueError(
                "target_tensors must have shape (N, 3, 128, 128), got "
                f"{tuple(target_tensors.shape)}"
            )
        if source_embeddings.ndim != 2 or tuple(source_embeddings.shape[1:]) != self.source_shape:
            raise ValueError(
                "source_embeddings must have shape (N, 512), got "
                f"{tuple(source_embeddings.shape)}"
            )
        batch = int(target_tensors.shape[0])
        if batch <= 0 or batch != int(source_embeddings.shape[0]):
            raise ValueError("target and source batches must be equal and non-empty")
        if not target_tensors.is_cuda or not source_embeddings.is_cuda:
            raise ValueError("TensorRTFaceSwapper inputs must already be CUDA tensors")
        if target_tensors.device != self.device or source_embeddings.device != self.device:
            raise ValueError(
                f"TensorRTFaceSwapper inputs must be on {self.device}, got "
                f"{target_tensors.device} and {source_embeddings.device}"
            )
        return batch

    def _swap_chunk(self, target_tensors, source_embeddings):
        batch = self._validate_inputs(target_tensors, source_embeddings)
        target_tensors = target_tensors.contiguous()
        source_embeddings = source_embeddings.contiguous()
        if target_tensors.dtype != torch.float32:
            target_tensors = target_tensors.float()
        if source_embeddings.dtype != torch.float32:
            source_embeddings = source_embeddings.float()
        out_tensor = torch.empty(
            (batch, 3, 128, 128), dtype=torch.float32, device=self.device
        )
        binding = self.io_binding
        clear_inputs = getattr(binding, "clear_binding_inputs", None)
        clear_outputs = getattr(binding, "clear_binding_outputs", None)
        if clear_inputs is not None:
            clear_inputs()
        if clear_outputs is not None:
            clear_outputs()
        binding.bind_input(
            name=self.target_name,
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float32,
            shape=tuple(int(item) for item in target_tensors.shape),
            buffer_ptr=int(target_tensors.data_ptr()),
        )
        binding.bind_input(
            name=self.source_name,
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float32,
            shape=tuple(int(item) for item in source_embeddings.shape),
            buffer_ptr=int(source_embeddings.data_ptr()),
        )
        binding.bind_output(
            name=self.output_name,
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float32,
            shape=tuple(int(item) for item in out_tensor.shape),
            buffer_ptr=int(out_tensor.data_ptr()),
        )
        try:
            self.session.run_with_iobinding(binding)
        except RuntimeError as exc:
            if batch <= 1:
                raise
            self._batch_unsupported = True
            logger.warning(
                "TensorRT inswapper rejected batch=%d (%s); retrying as "
                "batch-one calls.",
                batch,
                exc,
            )
            return torch.cat(
                [
                    self._swap_chunk(
                        target_tensors[index : index + 1],
                        source_embeddings[index : index + 1],
                    )
                    for index in range(batch)
                ],
                dim=0,
            )
        return out_tensor

    def swap_batch(self, target_tensors, source_embeddings):
        """Execute one or more bounded dynamic batches with zero-copy binding."""
        if self._closed:
            raise RuntimeError("TensorRTFaceSwapper is closed")
        batch = self._validate_inputs(target_tensors, source_embeddings)
        with self._lock:
            if self._batch_unsupported and batch > 1:
                return torch.cat(
                    [
                        self._swap_chunk(
                            target_tensors[index : index + 1],
                            source_embeddings[index : index + 1],
                        )
                        for index in range(batch)
                    ],
                    dim=0,
                )
            if batch <= self.max_batch:
                return self._swap_chunk(target_tensors, source_embeddings)
            outputs = []
            for start in range(0, batch, self.max_batch):
                outputs.append(
                    self._swap_chunk(
                        target_tensors[start : start + self.max_batch],
                        source_embeddings[start : start + self.max_batch],
                    )
                )
            return torch.cat(outputs, dim=0)

    def RunBatch(self, target_tensors, source_embeddings):
        """Compatibility spelling used by processor batch dispatchers."""
        return self.swap_batch(target_tensors, source_embeddings)

    run_batch = swap_batch

    def Run(self, target_tensor, source_embedding):
        """Run one crop while retaining the batched implementation."""
        return self.swap_batch(target_tensor, source_embedding)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.io_binding = None
        self.session = None
        self.engine = None

    Release = close

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


__all__ = [
    "INSWAPPER_DYNAMIC_SHAPE_PROFILE",
    "TensorRTFaceSwapper",
]
