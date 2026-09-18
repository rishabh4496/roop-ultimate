"""GPEN-BFR-512 restoration with CUDA preprocessing and seam-safe compositing.

This module keeps the small ``sanitize_frame_output`` helper used by the legacy
frame swapper and exposes the Stage 3 GPEN implementation.  GFPGAN is not
loaded here.  The replacement model is ``GPEN-BFR-512.onnx``.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np
from roop.degrade import swallowed as _swallowed

try:
    import torch
except Exception as _degrade_error:  # pragma: no cover - CPU-only import environments
    _swallowed("roop/processors/face_enhancer.py:20", _degrade_error, "CPU-only enhancer import")
    torch = None

from roop.trt_engine import TensorRTInferenceSession

LOGGER = logging.getLogger("roop.face_enhancer")
GPEN_MODEL_NAME = "GPEN-BFR-512.onnx"
GPEN_MODEL_URL = "https://huggingface.co/countfloyd/deepfake/resolve/main/GPEN-BFR-512.onnx"
GPEN_INPUT_SIZE = 512
GPEN_INPUT_PROFILE = {
    "min": "input:1x3x512x512",
    "opt": "input:1x3x512x512",
    "max": "input:1x3x512x512",
}


def sanitize_frame_output(output: Any, original_frame: np.ndarray, *, stage: str,
                          scale: float = 1.0) -> np.ndarray:
    """Return a finite uint8 frame, or the untouched original on corruption."""
    if torch is not None and isinstance(output, torch.Tensor):
        if torch.isnan(output).any().item() or torch.isinf(output).any().item():
            LOGGER.warning("%s emitted NaN/Inf; preserving original frame", stage)
            return original_frame.copy()
        output = output.detach().float().cpu().numpy()

    array = np.asarray(output)
    if not np.isfinite(array).all():
        LOGGER.warning("%s emitted NaN/Inf; preserving original frame", stage)
        return original_frame.copy()
    return np.clip(
        array.astype(np.float32, copy=False) * scale, 0, 255
    ).astype(np.uint8)


def gaussian_eroded_boundary_mask(shape: Sequence[int], erosion_px: int = 5) -> np.ndarray:
    """Create a soft interior matte with a five-pixel eroded boundary.

    Erosion removes the crop's hard outer edge before the Gaussian transition is
    applied.  The result is one in the stable interior and fades to zero at the
    crop boundary, which avoids a visible square seam when a restored crop is
    pasted into a target canvas.
    """
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        return np.zeros((max(0, height), max(0, width)), dtype=np.float32)
    radius = max(1, int(erosion_px))
    base = np.ones((height, width), dtype=np.float32)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    # Explicit constant borders are important here. OpenCV's default border
    # handling can replicate the all-ones crop and leave an opaque outer edge,
    # which defeats the purpose of the seam guard on crops touching a canvas
    # boundary.
    eroded = cv2.erode(
        base,
        kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # sigma=radius gives a smooth, approximately radius-wide falloff while
    # retaining a full-strength interior on normal 512px crops.
    padding = max(1, radius * 3)
    padded = cv2.copyMakeBorder(
        eroded,
        padding,
        padding,
        padding,
        padding,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )
    blurred = cv2.GaussianBlur(padded, (0, 0), sigmaX=float(radius))
    blurred = blurred[padding : padding + height, padding : padding + width]
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


def blend_restored_crop(
    target_canvas: np.ndarray,
    restored_crop: np.ndarray,
    bbox: Sequence[float],
    boundary_mask: Optional[np.ndarray] = None,
    erosion_px: int = 5,
) -> np.ndarray:
    """Blend a restored BGR crop into ``target_canvas`` at ``bbox``.

    ``bbox`` uses the normal ``(x1, y1, x2, y2)`` half-open convention.  The
    crop and mask are clipped to the canvas before resizing, so detections that
    touch an image edge cannot create an out-of-bounds write.
    """
    canvas = np.asarray(target_canvas)
    result = canvas.copy()
    if canvas.ndim != 3 or canvas.shape[2] != 3:
        raise ValueError("target_canvas must be an HxWx3 BGR array")
    crop = np.asarray(restored_crop)
    if crop.ndim != 3 or crop.shape[2] != 3 or crop.size == 0:
        return result

    x1, y1, x2, y2 = [int(round(value)) for value in bbox[:4]]
    left, top = max(0, x1), max(0, y1)
    right, bottom = min(canvas.shape[1], x2), min(canvas.shape[0], y2)
    if right <= left or bottom <= top:
        return result

    width, height = right - left, bottom - top
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)
    if boundary_mask is None:
        mask = gaussian_eroded_boundary_mask((height, width), erosion_px)
    else:
        mask = np.asarray(boundary_mask, dtype=np.float32)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = np.clip(mask, 0.0, 1.0)
    mask = mask[..., None]

    base = result[top:bottom, left:right].astype(np.float32)
    restored = resized.astype(np.float32)
    result[top:bottom, left:right] = np.clip(
        base * (1.0 - mask) + restored * mask, 0.0, 255.0
    ).astype(np.uint8)
    return result


class GPENEnhancer:
    """GPEN-BFR-512 TensorRT restorer.

    Input conversion to RGB ``[-1, 1]`` is performed on the selected CUDA device
    before ORT binding.  GPEN's exported graph is batch-one, so ``restore_batch``
    accepts a batch-shaped tensor but executes bounded one-sample calls when a
    caller supplies more than one crop.
    """

    model_name = GPEN_MODEL_NAME
    input_size = GPEN_INPUT_SIZE
    input_profile = dict(GPEN_INPUT_PROFILE)
    processorname = "gpen_bfr_512"
    type = "enhance"
    model_template = "ffhq_512"

    def __init__(self, model_path: Optional[str] = None, device_id: int = 0):
        if torch is None:
            raise RuntimeError("PyTorch is required for GPEN TensorRT restoration")
        self.device_id = int(device_id)
        self.device = torch.device(f"cuda:{self.device_id}")
        if not torch.cuda.is_available():
            raise RuntimeError("GPENEnhancer requires a CUDA device")
        # GPEN-BFR-512 is exported with a batch-one graph, so keep one crop in
        # flight on both the 6GB laptop and the 12GB desktop profile.
        self.max_batch = 1
        self.model_path = os.path.abspath(model_path or os.path.join(
            os.path.dirname(__file__), "..", "models", GPEN_MODEL_NAME
        ))
        self.engine: Optional[TensorRTInferenceSession] = None
        self.session = None
        self.io_binding = None
        self.input_name = "input"
        self.output_name = "output"
        self._lock = threading.RLock()
        self._closed = False

    def initialize(self, plugin_options: Optional[dict] = None) -> "GPENEnhancer":
        self.plugin_options = plugin_options
        if self.engine is not None:
            return self
        if not os.path.isfile(self.model_path):
            try:
                from roop.utilities import conditional_download
                conditional_download(os.path.dirname(self.model_path), [GPEN_MODEL_URL])
            except Exception as exc:
                raise FileNotFoundError(self.model_path) from exc
        cache_dir = os.path.abspath(os.environ.get(
            "ROOP_TRT_CACHE_DIR",
            os.path.join(os.path.dirname(self.model_path), "trt_cache"),
        ))
        self.engine = TensorRTInferenceSession(
            model_path=self.model_path,
            device_id=self.device_id,
            enable_fp16=True,
            cache_dir=cache_dir,
            dynamic_shape_profile=self.input_profile,
        )
        self.session = self.engine.get_session()
        self.io_binding = self.session.io_binding()
        inputs = list(self.session.get_inputs())
        outputs = list(self.session.get_outputs())
        if inputs:
            self.input_name = next(
                (item.name for item in inputs if item.name == "input"), inputs[0].name
            )
        if outputs:
            self.output_name = next(
                (item.name for item in outputs if item.name == "output"), outputs[0].name
            )
        return self

    Initialize = initialize

    def _restore_one(self, normalized: "torch.Tensor"):
        if tuple(normalized.shape) != (1, 3, self.input_size, self.input_size):
            raise ValueError("GPEN input must have shape (1, 3, 512, 512)")
        if not normalized.is_cuda or normalized.device != self.device:
            raise ValueError(f"GPEN input must be a CUDA tensor on {self.device}")
        output = torch.empty_like(normalized, dtype=torch.float32)
        binding = self.io_binding
        clear_inputs = getattr(binding, "clear_binding_inputs", None)
        clear_outputs = getattr(binding, "clear_binding_outputs", None)
        if clear_inputs is not None:
            clear_inputs()
        if clear_outputs is not None:
            clear_outputs()
        binding.bind_input(
            name=self.input_name,
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float32,
            shape=tuple(int(value) for value in normalized.shape),
            buffer_ptr=int(normalized.data_ptr()),
        )
        binding.bind_output(
            name=self.output_name,
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float32,
            shape=tuple(int(value) for value in output.shape),
            buffer_ptr=int(output.data_ptr()),
        )
        self.session.run_with_iobinding(binding)
        return output

    def restore_batch(self, normalized_crops: "torch.Tensor"):
        """Restore normalized CUDA RGB tensors and return CUDA RGB tensors."""
        if self._closed:
            raise RuntimeError("GPENEnhancer is closed")
        self.initialize()
        if normalized_crops.ndim != 4 or tuple(normalized_crops.shape[1:]) != (
            3, self.input_size, self.input_size
        ):
            raise ValueError("normalized_crops must have shape (N, 3, 512, 512)")
        if normalized_crops.shape[0] <= 0:
            raise ValueError("normalized_crops cannot be empty")
        normalized_crops = normalized_crops.contiguous()
        if normalized_crops.dtype != torch.float32:
            normalized_crops = normalized_crops.float()
        with self._lock:
            outputs = [
                self._restore_one(normalized_crops[index : index + 1])
                for index in range(int(normalized_crops.shape[0]))
            ]
        return torch.cat(outputs, dim=0)

    RunBatch = restore_batch

    def enhance_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Restore one BGR crop and return a BGR uint8 crop at its input size."""
        if crop_bgr is None:
            return crop_bgr
        if np.asarray(crop_bgr).size == 0:
            return np.asarray(crop_bgr).copy()
        source = np.asarray(crop_bgr)
        input_shape = source.shape[:2]
        source_512 = cv2.resize(
            source, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC
        )
        if not source_512.flags.c_contiguous:
            source_512 = np.ascontiguousarray(source_512)
        # HWC BGR -> CHW RGB, then normalize in GPU memory. The CPU only owns
        # the original crop upload; no CPU float normalization is performed.
        tensor = torch.from_numpy(source_512[..., ::-1].copy()).to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).div(127.5).sub(1.0)
        with torch.no_grad():
            restored = self.restore_batch(tensor)[0]
            restored = restored.clamp(-1.0, 1.0).add(1.0).mul(127.5).round().to(torch.uint8)
            restored = restored.permute(1, 2, 0).flip(2).contiguous()
        output = restored.cpu().numpy()
        if output.shape[:2] != input_shape:
            output = cv2.resize(output, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_CUBIC)
        return output

    def Run(self, source_faceset=None, target_face=None, temp_frame=None):
        """Plugin-compatible single-crop restoration entry point."""
        if temp_frame is None or np.asarray(temp_frame).size == 0:
            return temp_frame, 1
        return self.enhance_crop(temp_frame), 1

    def enhance_and_blend(
        self, target_canvas: np.ndarray, crop_bgr: np.ndarray, bbox: Sequence[float]
    ) -> np.ndarray:
        restored = self.enhance_crop(crop_bgr)
        return blend_restored_crop(target_canvas, restored, bbox, erosion_px=5)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.io_binding = None
        self.session = None
        self.engine = None

    Release = close

    def __enter__(self) -> "GPENEnhancer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


TensorRTGPENEnhancer = GPENEnhancer
GPENBFR512 = GPENEnhancer

__all__ = [
    "GPENBFR512",
    "GPENEnhancer",
    "GPEN_INPUT_PROFILE",
    "GPEN_INPUT_SIZE",
    "GPEN_MODEL_NAME",
    "TensorRTGPENEnhancer",
    "blend_restored_crop",
    "gaussian_eroded_boundary_mask",
    "sanitize_frame_output",
]
