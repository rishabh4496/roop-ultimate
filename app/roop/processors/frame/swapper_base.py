"""The swapper contract: one face crop in, one swapped crop pasted back out.

`BaseFaceSwapper` splits a swap into the four steps every model in
`FaceSwapInsightFace.SWAP_MODELS` shares, so a model is a small subclass and not
another branch inside a 2000-line processor:

    initialize_session  build the inference session, read and CHECK the graph
    pre_process         aligned BGR crop + raw ArcFace embedding -> input feed
    infer               feed -> the model's image tensor (mask kept per thread)
    post_process        image tensor -> uint8 crop -> pasted into the frame

The render path does NOT go through `post_process`. ProcessMgr's paste
(procmgr_masking.paste_upscale) layers occlusion, the swap model's own mask,
colour and temporal compositing on top of a paste, and none of that belongs in a
model class. `post_process` here is the plain inverse-affine paste the contract
names, for callers that want a whole swap from one object (tools, tests).

Static shapes are enforced, not assumed. `TensorSpec` is read from the ONNX
graph at load time (`inspect_onnx_topology`) and every feed is checked against it
before it reaches the session: wrong dtype, wrong rank, a wrong static dimension
or a non-contiguous buffer is a `ValueError` at the boundary instead of an ORT
shape error, or worse, a silent copy, three calls deep.

numpy-only at import: cv2, onnx and onnxruntime load inside the methods that use
them, so the registry and these checks import in the light (no-GPU) test profile.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

Dim = Union[int, str, None]


# ── Tensor topology ──────────────────────────────────────────────────────────

_ONNX_DTYPES = {1: np.float32, 10: np.float16, 11: np.float64, 7: np.int64, 6: np.int32}


@dataclass(frozen=True)
class TensorSpec:
    """One graph input/output. A str or None dim is symbolic (e.g. 'batch_size')."""
    name: str
    shape: Tuple[Dim, ...]
    dtype: Any = np.float32

    @property
    def rank(self) -> int:
        return len(self.shape)

    def is_static(self, axis: int) -> bool:
        return isinstance(self.shape[axis], int) and self.shape[axis] > 0


def inspect_onnx_topology(model_path: str) -> Tuple[Tuple[TensorSpec, ...], Tuple[TensorSpec, ...]]:
    """(inputs, outputs) of an ONNX file, read with onnx.load.

    Weights are not loaded (`load_external_data=False`) and initializers that
    old exporters list among the graph inputs are dropped, so this is only what
    a caller actually feeds.

    Measured 2026-09-29 on the shipped files:
        hififace_unofficial_256.onnx
            IN  target float32 [batch_size, 3, 256, 256]
            IN  source float32 [batch_size, 512]
            OUT output float32 [batch_size, 3, 256, 256]
            OUT mask   float32 [batch_size, 1, 256, 256]
        hyperswap_1a_256.onnx
            IN  source float32 [1, 512]            (source FIRST)
            IN  target float32 [1, 3, 256, 256]
            OUT output float32 [1, 3, 256, 256]    ([-1, 1])
            OUT mask   float32 [1, 1, 256, 256]
    """
    import onnx

    model = onnx.load(model_path, load_external_data=False)
    initializers = {init.name for init in model.graph.initializer}

    def _spec(value_info) -> TensorSpec:
        ttype = value_info.type.tensor_type
        dims = tuple(d.dim_value if d.HasField("dim_value") else (d.dim_param or None)
                     for d in ttype.shape.dim)
        return TensorSpec(value_info.name, dims, _ONNX_DTYPES.get(ttype.elem_type, np.float32))

    inputs = tuple(_spec(v) for v in model.graph.input if v.name not in initializers)
    outputs = tuple(_spec(v) for v in model.graph.output)
    return inputs, outputs


def validate_tensor(array: np.ndarray, spec: TensorSpec, *, batch: Optional[int] = None) -> np.ndarray:
    """Check `array` against `spec`; return it C-contiguous.

    A non-contiguous array is made contiguous here (the one allowed repair: ORT
    and IO binding would otherwise copy behind our back), and the result is
    asserted contiguous. dtype, rank and every STATIC dim must match exactly --
    no casting, no reshaping. A symbolic dim accepts any positive size, except
    the batch axis, which must equal `batch` when one is given.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{spec.name}: expected np.ndarray, got {type(array).__name__}")
    if array.dtype != np.dtype(spec.dtype):
        raise ValueError(f"{spec.name}: dtype {array.dtype}, model expects {np.dtype(spec.dtype)}")
    if array.ndim != spec.rank:
        raise ValueError(f"{spec.name}: shape {array.shape} has rank {array.ndim}, "
                         f"model expects rank {spec.rank} {spec.shape}")
    for axis, (got, want) in enumerate(zip(array.shape, spec.shape)):
        if isinstance(want, int) and want > 0:
            if got != want:
                raise ValueError(f"{spec.name}: axis {axis} is {got}, model is static "
                                 f"{want} (shape {array.shape} vs {spec.shape})")
        elif got <= 0:
            raise ValueError(f"{spec.name}: axis {axis} is empty ({array.shape})")
    if batch is not None and array.shape[0] != batch:
        raise ValueError(f"{spec.name}: batch {array.shape[0]}, expected {batch}")
    array = np.ascontiguousarray(array)
    assert array.flags["C_CONTIGUOUS"], spec.name
    return array


def validate_tensor_torch(tensor, spec: TensorSpec, *, batch: Optional[int] = None):
    """validate_tensor for a torch CUDA tensor: same checks, no cast, no host
    copy. A non-contiguous tensor is made contiguous on the device."""
    import torch
    if not torch.is_tensor(tensor):
        raise ValueError(f"{spec.name}: expected a torch tensor, got {type(tensor).__name__}")
    want = {np.dtype(np.float32): torch.float32, np.dtype(np.float16): torch.float16}[np.dtype(spec.dtype)]
    if tensor.dtype != want:
        raise ValueError(f"{spec.name}: dtype {tensor.dtype}, model expects {want}")
    if tensor.dim() != spec.rank:
        raise ValueError(f"{spec.name}: shape {tuple(tensor.shape)} has rank {tensor.dim()}, "
                         f"model expects rank {spec.rank} {spec.shape}")
    for axis, (got, need) in enumerate(zip(tensor.shape, spec.shape)):
        if isinstance(need, int) and need > 0 and got != need:
            raise ValueError(f"{spec.name}: axis {axis} is {got}, model is static {need}")
    if batch is not None and tensor.shape[0] != batch:
        raise ValueError(f"{spec.name}: batch {tensor.shape[0]}, expected {batch}")
    if not tensor.is_cuda:
        raise ValueError(f"{spec.name}: tensor is on {tensor.device}, the zero-copy path needs CUDA")
    return tensor.contiguous()


# ── Execution provider names ─────────────────────────────────────────────────

_PROVIDER_ALIASES = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    "trt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
    "directml": ["DmlExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
}


def resolve_providers(execution_provider: Union[str, Sequence[Any], None]) -> list:
    """'cuda' / 'CUDAExecutionProvider' / a ready provider list -> an ORT list.

    A full ORT name gets CPU appended as the fallback; a list (which may carry
    (name, options) tuples, as roop.globals.execution_providers does) is used
    as given.
    """
    if execution_provider is None:
        return ["CPUExecutionProvider"]
    if isinstance(execution_provider, (list, tuple)):
        return list(execution_provider)
    key = str(execution_provider).strip()
    alias = _PROVIDER_ALIASES.get(key.lower())
    if alias is not None:
        return list(alias)
    if key.endswith("ExecutionProvider"):
        return [key] if key == "CPUExecutionProvider" else [key, "CPUExecutionProvider"]
    raise ValueError(f"unknown execution provider {execution_provider!r}")


# ── The contract ─────────────────────────────────────────────────────────────

class BaseFaceSwapper(ABC):
    """Abstract swapper. Subclasses implement the four steps; `swap` chains them.

    Class attributes a subclass sets (the same contract ProcessMgr reads off
    FaceSwapInsightFace, under the same names):
        model_output_size, model_mean, model_standard_deviation,
        model_denormalize, model_template
    """

    model_name: str = ""
    model_output_size: int = 256
    model_mean: Sequence[float] = (0.0, 0.0, 0.0)
    model_standard_deviation: Sequence[float] = (1.0, 1.0, 1.0)
    model_denormalize: bool = False
    model_template: str = "arcface"

    def __init__(self) -> None:
        self.session = None
        self.input_specs: Dict[str, TensorSpec] = {}
        self.output_specs: Tuple[TensorSpec, ...] = ()
        # Per thread, like FaceSwapInsightFace._mask_tls: workers share one
        # swapper, and `infer` cannot return the mask without changing its type.
        self._mask_tls = threading.local()

    # -- the four steps ------------------------------------------------------

    @abstractmethod
    def initialize_session(self, model_path: str, execution_provider: str, **kwargs) -> None:
        """Build the session and verify the graph matches this model's contract."""

    @abstractmethod
    def pre_process(self, target_crop: np.ndarray, source_embedding: np.ndarray) -> Dict[str, Any]:
        """Aligned BGR uint8 crop (S,S,3) + raw 512-d embedding -> validated feed."""

    @abstractmethod
    def infer(self, inputs: Dict[str, Any]) -> np.ndarray:
        """Run the model on a validated feed; return its image output [B,3,S,S]."""

    @abstractmethod
    def post_process(self, swap_crop: np.ndarray, affine_matrix: np.ndarray,
                     target_frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Swapped crop -> the target frame with the crop pasted back."""

    # -- shared --------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        return self.session is not None

    @property
    def last_mask(self) -> Optional[np.ndarray]:
        """The model's own face mask from this thread's last `infer`, or None."""
        return getattr(self._mask_tls, "mask", None)

    def release(self) -> None:
        self.session = None
        self.input_specs = {}
        self.output_specs = ()

    def swap(self, target_crop: np.ndarray, source_embedding: np.ndarray,
             affine_matrix: np.ndarray, target_frame: np.ndarray,
             mask: Optional[np.ndarray] = None) -> np.ndarray:
        """pre_process -> infer -> post_process. `mask` None = the model's own."""
        out = self.infer(self.pre_process(target_crop, source_embedding))
        if mask is None:
            mask = self.last_mask
        return self.post_process(out, affine_matrix, target_frame, mask)

    def to_crop(self, swap_crop: np.ndarray) -> np.ndarray:
        """Model image tensor -> BGR uint8 (S,S,3).

        The arithmetic of ProcessMgr.normalize_swap_frame (procmgr_tiling),
        followed by the uint8 cast the paste needs.
        """
        chw = np.asarray(swap_crop, dtype=np.float32)
        if chw.ndim == 4:
            if chw.shape[0] != 1:
                raise ValueError(f"post_process takes one crop, got batch {chw.shape[0]}")
            chw = chw[0]
        if chw.ndim != 3 or chw.shape[0] != 3:
            raise ValueError(f"swap_crop must be [3,S,S] or [1,3,S,S], got {np.shape(swap_crop)}")
        hwc = chw.transpose(1, 2, 0)
        if self.model_denormalize:
            hwc = (hwc + 1.0) / 2.0
        hwc = np.clip((hwc * 255.0).round(), 0, 255)[:, :, ::-1]
        return np.ascontiguousarray(hwc.astype(np.uint8))

    @staticmethod
    def paste_back(crop: np.ndarray, affine_matrix: np.ndarray, target_frame: np.ndarray,
                   mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Inverse-warp `crop` (aligned by `affine_matrix`) into a copy of the frame.

        `mask` is in CROP space, float in [0,1], shape (S,S), (1,S,S) or
        (1,1,S,S); None pastes the whole crop. Returns a new uint8 frame.
        """
        import cv2

        frame = np.asarray(target_frame)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError(f"target_frame must be uint8 (H,W,3), got {frame.dtype} {frame.shape}")
        M = np.asarray(affine_matrix, dtype=np.float64)
        if M.shape != (2, 3):
            raise ValueError(f"affine_matrix must be 2x3, got {M.shape}")
        size = crop.shape[0]
        if mask is None:
            mask = np.ones((size, size), dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32).reshape(mask.shape[-2:])
        if mask.shape != (size, size):
            raise ValueError(f"mask {mask.shape} does not match crop {crop.shape[:2]}")
        out = frame.copy()
        roi = paste_roi(M, size, frame.shape[:2])
        if roi is None:
            return out
        x0, y0, x1, y1 = roi
        # Only the crop's footprint: warping the whole frame for a 256 px face
        # cost 40 ms at 1280x886 against ~2 ms here, and every pixel outside
        # the footprint has alpha 0 anyway. NOT bit-identical to the full-frame
        # warp: re-basing the translation moves cv2's 1/32 px fixed-point
        # rounding. Measured over 200 pastes on t1.jpg: max 2 levels, mean
        # 3e-5 -- far under the 0.71/255 render-to-render noise floor.
        inv = cv2.invertAffineTransform(M)
        inv[:, 2] -= (x0, y0)
        pasted = cv2.warpAffine(crop, inv, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        alpha = cv2.warpAffine(np.clip(mask, 0.0, 1.0), inv, (x1 - x0, y1 - y0),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0.0)[:, :, None]
        region = frame[y0:y1, x0:x1].astype(np.float32)
        blended = alpha * pasted.astype(np.float32) + (1.0 - alpha) * region
        out[y0:y1, x0:x1] = np.clip(blended.round(), 0, 255).astype(np.uint8)
        return out


def paste_roi(M, size: int, frame_hw) -> Optional[Tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) frame box covering the crop's footprint under inv(M),
    one pixel of margin for bilinear taps, clipped to the frame; None if the
    footprint misses the frame entirely."""
    M = np.asarray(M, dtype=np.float64).reshape(2, 3)
    corners = np.array([[0, 0], [size, 0], [0, size], [size, size]], dtype=np.float64)
    pts = (corners - M[:, 2]) @ np.linalg.inv(M[:, :2]).T
    h, w = int(frame_hw[0]), int(frame_hw[1])
    x0 = max(int(np.floor(pts[:, 0].min())) - 1, 0)
    y0 = max(int(np.floor(pts[:, 1].min())) - 1, 0)
    x1 = min(int(np.ceil(pts[:, 0].max())) + 2, w)
    y1 = min(int(np.ceil(pts[:, 1].max())) + 2, h)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1
