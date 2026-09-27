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
    _centers: dict[tuple[int, int], np.ndarray] = field(default_factory=dict, init=False,
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
