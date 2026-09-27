"""Face detection: SCRFD and YOLOFace behind one interface.

Both detectors letterbox the frame into a square canvas (aspect ratio kept,
centred, zero padding), run the ONNX model through
:class:`~face_engine.core.execution.ExecutionEngine`, decode every candidate
in a vectorized pass, and suppress duplicates with
``torchvision.ops.batched_nms`` (one call per batch; the frame index is the
NMS group, so faces in different frames never suppress each other).

Input normalization is per model, and ImageNet mean/std is the wrong choice
for both. Measured 2026-09-27 on 240 frames sampled from four real clips,
each detection checked against SCRFD boxes at score 0.3 (IoU > 0.4):

    detector  normalization          confirmed  unconfirmed
    YOLOFace  x/255, RGB                   333            5   <- default
    YOLOFace  (x-127.5)/128, BGR           302            4
    YOLOFace  ImageNet, RGB                285            5
    SCRFD     (x-127.5)/128, RGB           349            0   <- default
    SCRFD     ImageNet, RGB                339            0   (kps shift 1.4% of width)

SCRFD's default matches InsightFace's own implementation box for box (min
IoU 0.948 on the insightface sample group photo). YOLOFace was trained the
Ultralytics way (RGB, /255). :class:`Normalization` still offers ImageNet
for models that need it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from face_engine.core.execution import ExecutionEngine, ManagedSession

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


class Normalization(str, Enum):
    """Pixel normalization applied to the letterboxed uint8 canvas."""

    SYMMETRIC_128 = "symmetric_128"  # (x - 127.5) / 128
    IMAGENET = "imagenet"  # (x/255 - mean) / std
    UNIT = "unit"  # x / 255


@dataclass(frozen=True)
class Face:
    """One detected face, in source-frame pixel coordinates.

    Attributes:
        bbox: ``(4,)`` float32 ``x1, y1, x2, y2``, clipped to the frame.
        kps: ``(5, 2)`` float32 landmarks — left eye, right eye, nose, left and
            right mouth corner (image left/right). NOT clipped: a face cut by
            the frame edge keeps its true landmark positions, which the
            aligner needs.
        score: Detector confidence in ``[0, 1]``.
        frame_size: ``(height, width)`` of the source frame.
    """

    bbox: np.ndarray
    kps: np.ndarray
    score: float
    frame_size: tuple[int, int]

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2],
                        dtype=np.float32)

    @property
    def kps_inside_frame(self) -> bool:
        """True when all five landmarks fall inside the frame."""
        h, w = self.frame_size
        return bool(np.all((self.kps[:, 0] >= 0) & (self.kps[:, 0] < w)
                           & (self.kps[:, 1] >= 0) & (self.kps[:, 1] < h)))


@dataclass(frozen=True)
class Letterbox:
    """How a frame was placed on the detector canvas.

    ``canvas = frame * scale + (pad_x, pad_y)``.
    """

    scale: float
    pad_x: float
    pad_y: float
    input_size: int
    frame_size: tuple[int, int]

    def to_frame(self, points: np.ndarray) -> np.ndarray:
        """Map ``(..., 2)`` canvas coordinates back to the source frame."""
        out = np.empty_like(points, dtype=np.float32)
        out[..., 0] = (points[..., 0] - self.pad_x) / self.scale
        out[..., 1] = (points[..., 1] - self.pad_y) / self.scale
        return out


def as_bgr(frame: np.ndarray | None) -> np.ndarray | None:
    """Coerce a frame to contiguous ``uint8`` 3-channel BGR, or None if unusable.

    Accepts grayscale (H, W) / (H, W, 1), BGRA, and float images in [0, 1] or
    [0, 255]. Returns None for None, empty, non-finite, or non-image arrays.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return None
    if frame.ndim == 2:
        frame = frame[:, :, None]
    if frame.ndim != 3 or frame.shape[0] < 2 or frame.shape[1] < 2:
        return None
    channels = frame.shape[2]
    if frame.dtype != np.uint8:
        data = frame.astype(np.float32)
        if not np.all(np.isfinite(data)):
            return None
        if data.max(initial=0.0) <= 1.0:
            data = data * 255.0
        frame = np.clip(data, 0, 255).astype(np.uint8)
    if channels == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif channels == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif channels != 3:
        return None
    return np.ascontiguousarray(frame)


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, Letterbox]:
    """Resize ``frame`` into a centred ``size x size`` canvas, keeping aspect ratio."""
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    rw, rh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (rw, rh), interpolation=interpolation)
    pad_x, pad_y = (size - rw) // 2, (size - rh) // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[pad_y:pad_y + rh, pad_x:pad_x + rw] = resized
    # Use the exact per-axis scale so the inverse mapping matches the resize.
    return canvas, Letterbox(scale=float(min(rw / w, rh / h)), pad_x=float(pad_x),
                             pad_y=float(pad_y), input_size=size, frame_size=(h, w))


def normalize(canvas: np.ndarray, mode: Normalization, swap_rb: bool) -> np.ndarray:
    """``(H, W, 3)`` uint8 -> ``(1, 3, H, W)`` float32 blob."""
    data = canvas[:, :, ::-1] if swap_rb else canvas
    data = data.astype(np.float32)
    if mode is Normalization.SYMMETRIC_128:
        data = (data - 127.5) / 128.0
    elif mode is Normalization.IMAGENET:
        # ImageNet statistics are defined in RGB order.
        mean = _IMAGENET_MEAN if swap_rb else _IMAGENET_MEAN[::-1]
        std = _IMAGENET_STD if swap_rb else _IMAGENET_STD[::-1]
        data = (data - mean) / std
    else:
        data = data / 255.0
    return np.ascontiguousarray(data.transpose(2, 0, 1)[None])


def batched_nms(boxes: np.ndarray, scores: np.ndarray, groups: np.ndarray,
                iou_threshold: float, device: str = "cpu") -> np.ndarray:
    """Indices kept by ``torchvision.ops.batched_nms``, highest score first."""
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    import torch
    from torchvision.ops import batched_nms as _tv_batched_nms

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    keep = _tv_batched_nms(torch.as_tensor(boxes, dtype=torch.float32, device=device),
                           torch.as_tensor(scores, dtype=torch.float32, device=device),
                           torch.as_tensor(groups, dtype=torch.int64, device=device),
                           float(iou_threshold))
    return keep.cpu().numpy()


@dataclass
class _Candidates:
    boxes: np.ndarray  # (N, 4) frame coords
    scores: np.ndarray  # (N,)
    kps: np.ndarray  # (N, 5, 2) frame coords

    @staticmethod
    def empty() -> _Candidates:
        return _Candidates(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                           np.zeros((0, 5, 2), np.float32))


@dataclass
class GPUDetections:
    """Detections for a batch of frames, as tensors on the frames' device.

    Rows are ordered by frame, then score (highest first).

    Attributes:
        boxes: ``(N, 4)`` float32 ``x1, y1, x2, y2`` in frame pixels, clipped.
        kps: ``(N, 5, 2)`` float32 landmarks (not clipped).
        scores: ``(N,)`` float32.
        frame_index: ``(N,)`` int64, the batch frame each row belongs to.
        frame_size: ``(height, width)`` shared by the batch.
        num_frames: Batch size.
    """

    boxes: Any
    kps: Any
    scores: Any
    frame_index: Any
    frame_size: tuple[int, int]
    num_frames: int

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    @property
    def device(self) -> Any:
        return self.boxes.device

    @staticmethod
    def empty(device: Any, frame_size: tuple[int, int], num_frames: int = 1) -> GPUDetections:
        import torch

        return GPUDetections(torch.zeros((0, 4), device=device),
                             torch.zeros((0, 5, 2), device=device),
                             torch.zeros((0,), device=device),
                             torch.zeros((0,), dtype=torch.int64, device=device),
                             frame_size, num_frames)

    def index(self, rows: Any) -> GPUDetections:
        """Subset by an index or boolean tensor (a boolean mask costs one sync)."""
        return GPUDetections(self.boxes[rows], self.kps[rows], self.scores[rows],
                             self.frame_index[rows], self.frame_size, self.num_frames)

    def to_faces(self) -> list[list[Face]]:
        """Explicit host copy: one :class:`Face` list per frame."""
        boxes, kps = self.boxes.cpu().numpy(), self.kps.cpu().numpy()
        scores, frames = self.scores.cpu().numpy(), self.frame_index.cpu().numpy()
        out: list[list[Face]] = [[] for _ in range(self.num_frames)]
        for i in range(len(boxes)):
            out[int(frames[i])].append(Face(bbox=boxes[i].astype(np.float32),
                                            kps=kps[i].astype(np.float32),
                                            score=float(scores[i]), frame_size=self.frame_size))
        return out


def letterbox_cuda(frames: Any, size: int) -> tuple[Any, Letterbox]:
    """GPU :func:`letterbox`: ``(B, 3, H, W)`` -> ``(B, 3, size, size)`` float32, centred.

    Downscales with ``antialias=True`` bilinear (the counterpart of
    ``INTER_AREA``). The geometry depends only on the shape, so the
    :class:`Letterbox` is computed on the host without a device read.
    """
    import torch
    import torch.nn.functional as F

    b, _, h, w = frames.shape
    scale = min(size / w, size / h)
    rw, rh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    data = frames if frames.is_floating_point() else frames.float()
    resized = F.interpolate(data, size=(rh, rw), mode="bilinear", align_corners=False,
                            antialias=scale < 1.0) if (rh, rw) != (h, w) else data
    pad_x, pad_y = (size - rw) // 2, (size - rh) // 2
    canvas = torch.zeros((b, 3, size, size), dtype=torch.float32, device=frames.device)
    canvas[:, :, pad_y:pad_y + rh, pad_x:pad_x + rw] = resized.clamp(0, 255)
    return canvas, Letterbox(scale=float(min(rw / w, rh / h)), pad_x=float(pad_x),
                             pad_y=float(pad_y), input_size=size, frame_size=(h, w))


def normalize_cuda(canvas: Any, mode: Normalization, swap_rb: bool) -> Any:
    """GPU :func:`normalize` for a ``(B, 3, H, W)`` BGR float canvas."""
    import torch

    data = canvas.flip(1) if swap_rb else canvas
    if mode is Normalization.SYMMETRIC_128:
        return ((data - 127.5) / 128.0).contiguous()
    if mode is Normalization.IMAGENET:
        mean = _IMAGENET_MEAN if swap_rb else _IMAGENET_MEAN[::-1]
        std = _IMAGENET_STD if swap_rb else _IMAGENET_STD[::-1]
        m = torch.as_tensor(mean.copy(), device=data.device).view(1, 3, 1, 1)
        s = torch.as_tensor(std.copy(), device=data.device).view(1, 3, 1, 1)
        return ((data - m) / s).contiguous()
    return (data / 255.0).contiguous()


def _to_frame_cuda(points: Any, info: Letterbox) -> Any:
    """Canvas -> frame coordinates for ``(..., 2)`` tensors (host scalars only:
    building a pad tensor from host values would be a copy and a sync per call)."""
    import torch

    return torch.stack([(points[..., 0] - info.pad_x) / info.scale,
                        (points[..., 1] - info.pad_y) / info.scale], dim=-1)


@dataclass
class BaseDetector:
    """Shared letterbox -> infer -> decode -> NMS pipeline.

    Attributes:
        engine: Session provider.
        model_path: ONNX file.
        input_size: Square canvas side; forced to the model's fixed size if it has one.
        score_threshold: Candidates below this are dropped before NMS.
        iou_threshold: ``batched_nms`` IoU threshold.
        nms_device: ``"cpu"`` or ``"cuda"``. After the score threshold only a
            few dozen boxes remain, where the host->device copy costs more than
            the NMS; ``"cpu"`` is the default for that reason.
        min_face_size: Boxes narrower or shorter than this (pixels) are dropped.
    """

    engine: ExecutionEngine
    model_path: Path | str
    input_size: int = 640
    score_threshold: float = 0.5
    iou_threshold: float = 0.4
    nms_device: str = "cpu"
    min_face_size: float = 2.0
    normalization: Normalization = Normalization.SYMMETRIC_128
    swap_rb: bool = False
    _handle: ManagedSession | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0, 1]")
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")

    @property
    def session(self) -> ManagedSession:
        if self._handle is None:
            self._handle = self.engine.get_session(self.model_path)
            fixed = self._fixed_input_size(self._handle)
            if fixed is not None and fixed != self.input_size:
                logger.info("%s takes a fixed %dpx input; using it instead of %d",
                            Path(self.model_path).name, fixed, self.input_size)
                self.input_size = fixed
        return self._handle

    @staticmethod
    def _fixed_input_size(handle: ManagedSession) -> int | None:
        shape = handle.session.get_inputs()[0].shape
        h, w = shape[2], shape[3]
        return int(h) if isinstance(h, int) and h == w and h > 0 else None

    # --------------------------------------------------------------- pipeline
    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, Letterbox]:
        _ = self.session  # resolve a fixed input size before letterboxing
        canvas, info = letterbox(frame, self.input_size)
        return normalize(canvas, self.normalization, self.swap_rb), info

    def _decode(self, outputs: list[np.ndarray], info: Letterbox) -> _Candidates:
        raise NotImplementedError

    def _candidates(self, frame: np.ndarray | None) -> _Candidates:
        bgr = as_bgr(frame)
        if bgr is None:
            return _Candidates.empty()
        blob, info = self.preprocess(bgr)
        handle = self.session
        outputs = handle.session.run(None, {handle.input_names[0]: blob})
        found = self._decode(outputs, info)
        if found.boxes.shape[0] == 0:
            return found
        h, w = info.frame_size
        boxes = found.boxes.copy()
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h)
        ok = (((boxes[:, 2] - boxes[:, 0]) >= self.min_face_size)
              & ((boxes[:, 3] - boxes[:, 1]) >= self.min_face_size)
              & np.all(np.isfinite(found.kps.reshape(len(boxes), -1)), axis=1))
        return _Candidates(boxes[ok], found.scores[ok], found.kps[ok])

    def detect(self, frame: np.ndarray | None) -> list[Face]:
        """Faces in one frame, highest score first. Never raises on bad input."""
        return self.detect_batch([frame])[0]

    def detect_batch(self, frames: list[np.ndarray | None]) -> list[list[Face]]:
        """Faces per frame; one ``batched_nms`` call covers the whole batch."""
        per_frame = [self._candidates(f) for f in frames]
        counts = [c.boxes.shape[0] for c in per_frame]
        if sum(counts) == 0:
            return [[] for _ in frames]
        boxes = np.concatenate([c.boxes for c in per_frame])
        scores = np.concatenate([c.scores for c in per_frame])
        kps = np.concatenate([c.kps for c in per_frame])
        groups = np.repeat(np.arange(len(frames)), counts)
        keep = batched_nms(boxes, scores, groups, self.iou_threshold, self.nms_device)
        results: list[list[Face]] = [[] for _ in frames]
        for i in keep:
            g = int(groups[i])
            h, w = per_frame_size(frames[g])
            results[g].append(Face(bbox=boxes[i].astype(np.float32),
                                   kps=kps[i].astype(np.float32),
                                   score=float(scores[i]), frame_size=(h, w)))
        return results

    def detect_cuda(self, frames: Any) -> GPUDetections:
        """Detect faces in ``(B, 3, H, W)`` CUDA frames without leaving the GPU.

        Letterbox, normalization, inference (``run_binding``: ORT reads and writes
        torch memory directly), anchor decoding, thresholding and
        ``torchvision.ops.batched_nms`` all run on the frames' device; nothing is
        copied to the host. The data-dependent steps (the score mask and NMS
        itself) synchronize with the device, as any variable-size result must.
        """
        import torch
        from torchvision.ops import batched_nms as _tv_batched_nms

        if frames.ndim == 3:
            frames = frames[None]
        b, _, h, w = frames.shape
        handle = self.session
        canvas, info = letterbox_cuda(frames, self.input_size)
        blob = normalize_cuda(canvas, self.normalization, self.swap_rb)
        shapes = self._output_shapes(handle, info.input_size)
        boxes_all, kps_all, scores_all, index_all = [], [], [], []
        for i in range(b):
            outputs = handle.run_binding({handle.input_names[0]: blob[i:i + 1]},
                                         output_shapes=shapes)
            boxes, kps, scores = self._decode_cuda([outputs[n] for n in handle.output_names], info)
            boxes_all.append(boxes)
            kps_all.append(kps)
            scores_all.append(scores)
            index_all.append(torch.full_like(scores, i, dtype=torch.int64))
        boxes = torch.cat(boxes_all)
        kps = torch.cat(kps_all)
        scores = torch.cat(scores_all)
        frame_index = torch.cat(index_all)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
        ok = ((scores >= self.score_threshold)
              & ((boxes[:, 2] - boxes[:, 0]) >= self.min_face_size)
              & ((boxes[:, 3] - boxes[:, 1]) >= self.min_face_size)
              & torch.isfinite(kps).flatten(1).all(1))
        keep = ok.nonzero()[:, 0]
        if keep.shape[0] == 0:
            return GPUDetections.empty(frames.device, (h, w), b)
        boxes, kps, scores, frame_index = boxes[keep], kps[keep], scores[keep], frame_index[keep]
        kept = _tv_batched_nms(boxes, scores, frame_index, float(self.iou_threshold))
        # Order by frame, then score: batched_nms returns score order over the batch.
        kept = kept[torch.argsort(frame_index[kept], stable=True)]
        return GPUDetections(boxes[kept], kps[kept], scores[kept], frame_index[kept], (h, w), b)

    def _output_shapes(self, handle: ManagedSession,
                       size: int) -> dict[str, tuple[int, ...]] | None:
        """Explicit output shapes for ``run_binding`` when the export's are not usable."""
        return None

    def _decode_cuda(self, outputs: list[Any], info: Letterbox) -> tuple[Any, Any, Any]:
        """``(boxes, kps, scores)`` for EVERY candidate, as device tensors (no threshold)."""
        raise NotImplementedError(f"{type(self).__name__} has no CUDA decoder")


def per_frame_size(frame: Any) -> tuple[int, int]:
    return int(frame.shape[0]), int(frame.shape[1])


@dataclass
class SCRFDDetector(BaseDetector):
    """SCRFD (InsightFace ``det_10g`` / ``scrfd_10g_bnkps``): 3 strides x 2 anchors.

    Outputs are 9 tensors ordered scores(8,16,32), boxes(8,16,32), kps(8,16,32);
    boxes and keypoints are distances/offsets in stride units from each anchor
    centre.
    """

    swap_rb: bool = True
    strides: tuple[int, ...] = (8, 16, 32)
    anchors_per_location: int = 2
    _centers: dict[tuple[Any, ...], Any] = field(default_factory=dict, init=False,
                                                        repr=False)

    def _anchor_centers(self, size: int, stride: int) -> np.ndarray:
        key = (size, stride)
        if key not in self._centers:
            n = size // stride
            ys, xs = np.mgrid[:n, :n]
            centers = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32) * stride
            self._centers[key] = np.repeat(centers, self.anchors_per_location, axis=0)
        return self._centers[key]

    def _decode(self, outputs: list[np.ndarray], info: Letterbox) -> _Candidates:
        levels = len(self.strides)
        if len(outputs) != 3 * levels:
            raise RuntimeError(f"SCRFD expected {3 * levels} outputs, got {len(outputs)}")
        boxes_all, scores_all, kps_all = [], [], []
        for level, stride in enumerate(self.strides):
            scores = outputs[level].reshape(-1)
            keep = scores >= self.score_threshold
            if not np.any(keep):
                continue
            centers = self._anchor_centers(info.input_size, stride)[keep]
            dist = outputs[levels + level].reshape(-1, 4)[keep] * stride
            offs = outputs[2 * levels + level].reshape(-1, 5, 2)[keep] * stride
            boxes = np.concatenate([centers - dist[:, :2], centers + dist[:, 2:]], axis=1)
            boxes_all.append(info.to_frame(boxes.reshape(-1, 2, 2)).reshape(-1, 4))
            kps_all.append(info.to_frame(centers[:, None, :] + offs))
            scores_all.append(scores[keep])
        if not boxes_all:
            return _Candidates.empty()
        return _Candidates(np.concatenate(boxes_all), np.concatenate(scores_all),
                           np.concatenate(kps_all))

    def _output_shapes(self, handle: ManagedSession,
                       size: int) -> dict[str, tuple[int, ...]]:
        """The export's output shapes are recorded for 640 only; derive them for ``size``."""
        names = handle.output_names
        levels = len(self.strides)
        shapes: dict[str, tuple[int, ...]] = {}
        for level, stride in enumerate(self.strides):
            n = (size // stride) ** 2 * self.anchors_per_location
            shapes[names[level]] = (n, 1)
            shapes[names[levels + level]] = (n, 4)
            shapes[names[2 * levels + level]] = (n, 10)
        return shapes

    def _anchor_centers_cuda(self, size: int, stride: int, device: Any) -> Any:
        import torch

        key = ("cuda", size, stride, str(device))
        cache = self._centers
        if key not in cache:
            cache[key] = torch.as_tensor(self._anchor_centers(size, stride), device=device)
        return cache[key]

    def _decode_cuda(self, outputs: list[Any], info: Letterbox) -> tuple[Any, Any, Any]:
        """Every anchor decoded (no threshold yet), so the batch needs one mask."""
        import torch

        levels = len(self.strides)
        if len(outputs) != 3 * levels:
            raise RuntimeError(f"SCRFD expected {3 * levels} outputs, got {len(outputs)}")
        boxes_all, kps_all, scores_all = [], [], []
        for level, stride in enumerate(self.strides):
            centers = self._anchor_centers_cuda(info.input_size, stride, outputs[level].device)
            dist = outputs[levels + level].reshape(-1, 4) * stride
            offs = outputs[2 * levels + level].reshape(-1, 5, 2) * stride
            boxes = torch.cat([centers - dist[:, :2], centers + dist[:, 2:]], dim=1)
            boxes_all.append(_to_frame_cuda(boxes.reshape(-1, 2, 2), info).reshape(-1, 4))
            kps_all.append(_to_frame_cuda(centers[:, None, :] + offs, info))
            scores_all.append(outputs[level].reshape(-1))
        return torch.cat(boxes_all), torch.cat(kps_all), torch.cat(scores_all)


@dataclass
class YOLOFaceDetector(BaseDetector):
    """YOLOv8-face (``yoloface_8n``): one ``(1, 20, 8400)`` output.

    Rows are ``cx, cy, w, h, score`` then 5 x ``(x, y, visibility)``, already in
    canvas pixels (anchor decoding is inside the export).
    """

    normalization: Normalization = Normalization.UNIT
    swap_rb: bool = True

    def _decode(self, outputs: list[np.ndarray], info: Letterbox) -> _Candidates:
        pred = np.asarray(outputs[0])
        if pred.ndim != 3 or pred.shape[1] < 20:
            raise RuntimeError(f"YOLOFace output has unexpected shape {pred.shape}")
        det = pred[0].T  # (N, 20)
        keep = det[:, 4] >= self.score_threshold
        if not np.any(keep):
            return _Candidates.empty()
        det = det[keep]
        cxcy, wh = det[:, 0:2], det[:, 2:4]
        corners = np.stack([cxcy - wh / 2, cxcy + wh / 2], axis=1)  # (N, 2, 2)
        kps = det[:, 5:20].reshape(-1, 5, 3)[:, :, :2]
        return _Candidates(info.to_frame(corners).reshape(-1, 4), det[:, 4].copy(),
                           info.to_frame(kps))

    def _decode_cuda(self, outputs: list[Any], info: Letterbox) -> tuple[Any, Any, Any]:
        import torch

        pred = outputs[0]
        if pred.ndim != 3 or pred.shape[1] < 20:
            raise RuntimeError(f"YOLOFace output has unexpected shape {tuple(pred.shape)}")
        det = pred[0].T
        cxcy, wh = det[:, 0:2], det[:, 2:4]
        corners = torch.stack([cxcy - wh / 2, cxcy + wh / 2], dim=1)
        kps = det[:, 5:20].reshape(-1, 5, 3)[:, :, :2]
        return (_to_frame_cuda(corners, info).reshape(-1, 4), _to_frame_cuda(kps, info),
                det[:, 4].contiguous())
