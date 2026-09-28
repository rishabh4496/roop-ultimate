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
        return self._select_cuda(torch.cat(boxes_all), torch.cat(kps_all), torch.cat(scores_all),
                                 torch.cat(index_all), (h, w), b, frames.device)

    def _select_cuda(self, boxes: Any, kps: Any, scores: Any, frame_index: Any,
                     frame_size: tuple[int, int], b: int, device: Any) -> GPUDetections:
        """Clip, threshold, drop tiny / non-finite, ``batched_nms``: the shared tail."""
        import torch
        from torchvision.ops import batched_nms as _tv_batched_nms

        h, w = frame_size
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
        ok = ((scores >= self.score_threshold)
              & ((boxes[:, 2] - boxes[:, 0]) >= self.min_face_size)
              & ((boxes[:, 3] - boxes[:, 1]) >= self.min_face_size)
              & torch.isfinite(kps).flatten(1).all(1))
        keep = ok.nonzero()[:, 0]
        if keep.shape[0] == 0:
            return GPUDetections.empty(device, (h, w), b)
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


@dataclass
class GPUSCRFDDetector(BaseDetector):
    """SCRFD-10G on a tight, aspect-preserving canvas, TensorRT FP16, all on the GPU.

    Unlike :class:`SCRFDDetector` (a square 640 letterbox, ONNX Runtime CUDA):

    * the long side is scaled to ``input_size`` (640) and each side is padded
      only up to a multiple of 32 (bottom / right, black), so a 1080p frame
      becomes a 384 x 640 canvas instead of 640 x 640 (40% fewer pixels);
    * inference runs on the dynamic-shape AOT engine (``tools/compile_engines.py``:
      profile 384..1024 per side, opt 640 x 640; 0.91 ms at 384 x 640 vs 3.42
      ms for ONNX Runtime CUDA FP32, RTX 4070, 2026-09-28), binding torch
      memory (``set_tensor_address``); without it, ONNX Runtime IOBinding on a
      copy of the model with symbolic H / W (``named_spatial_dims``);
    * anchor grids are cached per canvas; boxes / landmarks are decoded for
      every anchor at once and mapped back by the one scale factor.

    Input normalization is ``(rgb/255 - 0.5) / 0.5``: SCRFD's symmetric scaling
    (InsightFace uses /128; the 0.4% difference is below detection noise), not
    ImageNet mean/std (see the module docstring). ``input_format`` says what
    :meth:`detect_cuda` is given: ``"bgr255"`` (this package's frames) or
    ``"rgb01"``.
    """

    score_threshold: float = 0.45
    iou_threshold: float = 0.40
    pad_multiple: int = 32
    input_format: str = "bgr255"
    precision: str = "fp16"
    placement: str = "center"
    min_canvas: int = 384  # the engine profile's smallest side
    strides: tuple[int, ...] = (8, 16, 32)
    anchors_per_location: int = 2
    _runner: Any = field(default=None, init=False, repr=False)
    _grids: dict[tuple[Any, ...], Any] = field(default_factory=dict, init=False, repr=False)
    _canvases: dict[tuple[Any, ...], Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.input_format not in ("bgr255", "rgb01"):
            raise ValueError("input_format must be 'bgr255' or 'rgb01'")
        if self.placement not in ("top_left", "center", "center_square"):
            raise ValueError("placement must be 'top_left', 'center' or 'center_square'")

    @property
    def runner(self) -> Any:
        """The AOT TensorRT engine when compiled for this GPU, else ONNX Runtime."""
        if self._runner is None:
            from face_engine.core.trt_compiler import aot_engine
            from face_engine.utils.onnx_batch import named_spatial_dims

            self._runner = aot_engine(self.model_path, self.precision, self.engine.config)
            if self._runner is None:
                self._runner = self.engine.get_session(named_spatial_dims(self.model_path))
        return self._runner

    @property
    def uses_tensorrt_engine(self) -> bool:
        from face_engine.core.trt_compiler import TensorRTEngine

        return isinstance(self.runner, TensorRTEngine)

    def canvas_size(self, height: int, width: int) -> tuple[float, int, int, int, int]:
        """``(scale, resized_h, resized_w, canvas_h, canvas_w)`` for an ``H x W`` frame."""
        scale = float(self.input_size) / max(height, width)
        rh, rw = max(1, round(height * scale)), max(1, round(width * scale))
        if self.placement == "center_square":
            return scale, rh, rw, self.input_size, self.input_size
        m, lo = self.pad_multiple, self.min_canvas
        return scale, rh, rw, max(lo, -(-rh // m) * m), max(lo, -(-rw // m) * m)

    def _offset(self, rh: int, rw: int, ch: int, cw: int) -> tuple[int, int]:
        """Top-left corner of the resized frame on the canvas."""
        if self.placement == "top_left":
            return 0, 0
        return (ch - rh) // 2, (cw - rw) // 2

    def anchor_grid(self, canvas_h: int, canvas_w: int, stride: int, device: Any) -> Any:
        """``(N, 2)`` anchor centres in canvas pixels, in the network's order (cached)."""
        import torch

        key = (canvas_h, canvas_w, stride, str(device))
        if key not in self._grids:
            ys, xs = torch.meshgrid(torch.arange(canvas_h // stride, device=device),
                                    torch.arange(canvas_w // stride, device=device),
                                    indexing="ij")
            centers = torch.stack([xs, ys], -1).reshape(-1, 2).float() * stride
            self._grids[key] = centers.repeat_interleave(self.anchors_per_location, 0)
        return self._grids[key]

    def prepare(self, height: int, width: int, device: Any = "cuda") -> None:
        """Pre-build the anchor grids for ``H x W`` frames (a render's frame size)."""
        _, _, _, ch, cw = self.canvas_size(height, width)
        for stride in self.strides:
            self.anchor_grid(ch, cw, stride, device)

    def preprocess_cuda(self, frames: Any) -> tuple[Any, float]:
        """``(B, 3, H, W)`` -> ``((B, 3, ch, cw)`` normalized RGB canvas, scale)."""
        import torch
        import torch.nn.functional as F

        x = frames if frames.is_floating_point() else frames.float()
        b, _, h, w = x.shape
        scale, rh, rw, ch, cw = self.canvas_size(h, w)
        # Resize first, then convert / normalize the small image (9x fewer pixels
        # at 1080p). The canvas is reused: its black padding never changes.
        resized = F.interpolate(x, size=(rh, rw), mode="bilinear", align_corners=False,
                                antialias=scale < 1.0) if (rh, rw) != (h, w) else x
        key = (b, ch, cw, str(x.device))
        canvas = self._canvases.get(key)
        if canvas is None:
            canvas = self._canvases[key] = torch.full((b, 3, ch, cw), -1.0, device=x.device)
        oy, ox = self._offset(rh, rw, ch, cw)
        region = canvas[..., oy:oy + rh, ox:ox + rw]
        if self.input_format == "bgr255":
            region.copy_(resized.flip(1).clamp(0, 255) * (2.0 / 255.0) - 1.0)
        else:
            region.copy_(resized.clamp(0, 1) * 2.0 - 1.0)
        return canvas, scale

    def _anchors(self, canvas_hw: tuple[int, int], device: Any) -> tuple[Any, Any]:
        """``(centres (N, 2), strides (N, 1))`` for all levels of a canvas, cached."""
        import torch

        key = ("all", *canvas_hw, str(device))
        if key not in self._grids:
            grids = [self.anchor_grid(*canvas_hw, s, device) for s in self.strides]
            self._grids[key] = (torch.cat(grids), torch.cat([
                torch.full((g.shape[0], 1), float(s), device=device)
                for g, s in zip(grids, self.strides)]))
        return self._grids[key]

    def decode_cuda(self, outputs: list[Any], canvas_hw: tuple[int, int], scale: float,
                    frame_hw: tuple[int, int] | None = None) -> tuple[Any, Any, Any]:
        """9 outputs (scores, boxes, kps per stride) -> frame-space ``(boxes, kps, scores)``.

        Score threshold, minimum face size and finiteness are one mask over all
        anchors (elementwise, no host read); ONE device read then takes the
        survivors, which alone are decoded. Same numbers as decoding every
        anchor first, for a fraction of the work and one sync instead of two.
        ``frame_hw`` clips the boxes to the frame.
        """
        import torch

        levels = len(self.strides)
        scores = torch.cat([outputs[i].float().reshape(-1) for i in range(levels)])
        dist = torch.cat([outputs[levels + i].float().reshape(-1, 4) for i in range(levels)])
        offs = torch.cat([outputs[2 * levels + i].float().reshape(-1, 10)
                          for i in range(levels)])
        centers, strides = self._anchors(canvas_hw, scores.device)
        size = (dist[:, :2] + dist[:, 2:]) * strides / scale  # (w, h) in frame pixels
        ok = ((scores >= self.score_threshold) & (size >= self.min_face_size).all(1)
              & torch.isfinite(dist).all(1) & torch.isfinite(offs).all(1))
        keep = ok.nonzero()[:, 0]
        c, st = centers[keep], strides[keep]
        if frame_hw is not None:  # canvas -> frame: undo the placement offset
            _, rh, rw, _, _ = self.canvas_size(*frame_hw)
            oy, ox = self._offset(rh, rw, *canvas_hw)
            if oy or ox:
                c = c - torch.tensor([ox, oy], dtype=c.dtype, device=c.device)
        d = dist[keep] * st
        boxes = torch.cat([c - d[:, :2], c + d[:, 2:]], 1) / scale
        kps = (c[:, None, :] + offs[keep].reshape(-1, 5, 2) * st[:, :, None]) / scale
        if frame_hw is not None:
            h, w = frame_hw
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
        return boxes, kps, scores[keep]

    def detect_cuda(self, frames: Any) -> GPUDetections:
        import torch
        from torchvision.ops import batched_nms

        if frames.ndim == 3:
            frames = frames[None]
        b, _, h, w = frames.shape
        runner = self.runner
        canvas, scale = self.preprocess_cuda(frames)
        ch, cw = canvas.shape[-2:]
        shapes = None
        if not self.uses_tensorrt_engine:  # ORT needs the canvas's output lengths
            names = runner.output_names
            shapes = {}
            for level, stride in enumerate(self.strides):
                n = (ch // stride) * (cw // stride) * self.anchors_per_location
                shapes[names[level]] = (n, 1)
                shapes[names[3 + level]] = (n, 4)
                shapes[names[6 + level]] = (n, 10)
        parts = []
        for i in range(b):
            out = runner.run_binding({runner.input_names[0]: canvas[i:i + 1]},
                                     output_shapes=shapes)
            bx, kp, sc = self.decode_cuda([out[n] for n in runner.output_names], (ch, cw),
                                          scale, (h, w))
            parts.append((bx, kp, sc, torch.full_like(sc, i, dtype=torch.int64)))
        boxes, kps, scores, index = (torch.cat(t) for t in zip(*parts))
        if scores.shape[0] == 0:
            return GPUDetections.empty(frames.device, (h, w), b)
        kept = batched_nms(boxes, scores, index, float(self.iou_threshold))
        if b > 1:  # frame order, then score (batched_nms returns score order)
            kept = kept[torch.argsort(index[kept], stable=True)]
        return GPUDetections(boxes[kept], kps[kept], scores[kept], index[kept], (h, w), b)

    def detect_batch(self, frames: list[np.ndarray | None]) -> list[list[Face]]:
        """Host BGR frames through the GPU path (one code path)."""
        import torch

        out: list[list[Face]] = []
        for frame in frames:
            bgr = as_bgr(frame)
            if bgr is None:
                out.append([])
                continue
            t = torch.from_numpy(np.ascontiguousarray(bgr)).cuda().permute(2, 0, 1)[None]
            fmt, self.input_format = self.input_format, "bgr255"
            try:
                out.append(self.detect_cuda(t).to_faces()[0])
            finally:
                self.input_format = fmt
        return out


# ---------------------------------------------------------------------------- angle-resilient
def unrotate_points_cuda(points: Any, k: Any, height: float, width: float) -> Any:
    """Undo ``torch.rot90(image, k, dims=[-2, -1])`` on ``(..., 2)`` ``x, y`` points.

    ``height`` / ``width`` are the image's size BEFORE the rotation; points are
    continuous pixel coordinates (pixel ``i`` spans ``[i, i + 1)``). ``k`` is an
    int or a tensor broadcastable to ``points[..., 0]`` (one angle per row, no
    host read). ``torch.rot90`` turns from the first dim toward the second, so
    rotated ``(x', y')`` came from:

    ====  ===================
    k     original ``(x, y)``
    ====  ===================
    0     ``(x', y')``
    1     ``(W - y', x')``
    2     ``(W - x', H - y')``
    3     ``(y', H - x')``
    ====  ===================
    """
    import torch

    x, y = points[..., 0], points[..., 1]
    if not torch.is_tensor(k):
        k = torch.full_like(x, int(k) % 4)
    k = k.to(x.dtype) % 4
    ox = torch.where(k == 0, x, torch.where(k == 1, width - y, torch.where(k == 2, width - x, y)))
    oy = torch.where(k == 0, y, torch.where(k == 1, x, torch.where(k == 2, height - y,
                                                                    height - x)))
    return torch.stack([ox, oy], dim=-1)


def unrotate_boxes_cuda(boxes: Any, k: Any, height: float, width: float) -> Any:
    """:func:`unrotate_points_cuda` for ``(N, 4)`` boxes (exact for multiples of 90 deg)."""
    import torch

    x1, y1, x2, y2 = boxes.unbind(-1)
    corners = torch.stack([torch.stack([x1, y1], -1), torch.stack([x2, y1], -1),
                           torch.stack([x1, y2], -1), torch.stack([x2, y2], -1)], -2)
    kk = k[:, None] if torch.is_tensor(k) and k.ndim == 1 else k
    back = unrotate_points_cuda(corners, kk, height, width)
    return torch.cat([back.amin(-2), back.amax(-2)], dim=-1)


@dataclass
class DualDetections:
    """One frame's detections split at the high threshold (ByteTrack's input).

    Attributes:
        high: score ``>= score_threshold`` (0.50): new tracks start only from these.
        low: ``low_threshold <= score < score_threshold`` (0.20 .. 0.50): used
            only to keep EXISTING tracks alive (the second association).
        angle: Rotation (degrees, counter-clockwise as ``torch.rot90``) of the
            pass that produced the best face, 0 for upright.
        swept: Whether the rotation sweep ran on this frame.
        tilted: High faces whose landmarks are rolled more than
            ``max_pass_roll`` in the pass that produced them (no pass saw
            them upright; their landmarks are the least reliable).
    """

    high: GPUDetections
    low: GPUDetections
    angle: int = 0
    swept: bool = False
    tilted: int = 0


@dataclass
class SweepStats:
    """Counters for :class:`AngleResilientSCRFD`. Read them: a sweep that never
    runs and a sweep that runs every frame both look fine from the output."""

    frames: int = 0
    sweeps: int = 0
    tilt_sweeps: int = 0
    sweep_hits: int = 0
    empty_skips: int = 0
    angle_frames: dict[int, int] = field(default_factory=lambda: {0: 0, 90: 0, 180: 0, 270: 0})


@dataclass
class AngleResilientSCRFD(GPUSCRFDDetector):
    """:class:`GPUSCRFDDetector` that also finds faces lying down or upside down.

    Per frame, in order:

    1. **Primary pass**: one batch-1 inference on the tight canvas at the
       current angle: 0 deg, or the preferred angle for ``preferred_hold``
       frames after a sweep found the best face at another angle. Upright
       video pays exactly this, the plain detector's cost
       (``test_angle_resilience`` checks that no sweep runs on it).
    2. **Conditional sweep**: the three OTHER angles as one batch of three
       square ``sweep_size`` canvases (``torch.rot90`` of one centred canvas,
       so the three share a shape; the engine profile allows batch 3), only when

       * the primary pass found no face (score >= ``score_threshold``) and the
         scene is not in its "empty" state, or
       * a face it found is rolled more than ``max_pass_roll`` (45 deg) in the
         pass's own frame (the eye-midpoint -> mouth-midpoint axis). SCRFD
         at 0 deg DOES find many rotated faces (t1.jpg rotated 90 deg: 4 of 6
         at score >= 0.5; 180 deg: 2 of 6) but with landmarks 5-28% of the
         face size off, so "sweep only when nothing is found" never fired on
         them. Measured axis roll: upright t1 faces |roll| <= 33 deg; the
         same faces found in a 90 deg frame 64-124 deg, in a 180 deg frame
         ~170 deg (their eye line reads ~0 there: the network swaps the
         eyes, so the eye line cannot be the test), or
       * it found fewer faces than one of the last ``lost_window`` frames
         (a face was lost in the last 2 frames: it may have rolled over), or
       * :meth:`note_track_lost` asked for it (the tracker's view of the same).

       A scene with no face for ``empty_after`` consecutive frames is
       registered empty; it is then swept once per ``empty_rescan`` frames
       instead of every frame, until a face appears (or :meth:`reset`).
    3. Boxes and landmarks from rotated passes are mapped back to the upright
       frame on the device (:func:`unrotate_points_cuda`, one angle per row,
       no host read). Landmarks keep their semantic order (the person's left
       eye stays point 0), so the similarity fit downstream carries the roll.
    4. One ``batched_nms`` (IoU ``iou_threshold``) over every candidate with
       score >= ``low_threshold``, ranked so that a candidate upright in its
       own pass beats a tilted one whatever the scores (the same face found
       at 0 deg with a higher score but rolled landmarks loses to its
       upright rotated pass); then the split into :class:`DualDetections`
       high / low by the real score.

    Host reads per frame: the survivor positions in decode (as the parent),
    the NMS result size, and the high count that decides the sweep. Scalars
    and index lists only; no image, box or landmark data leaves the device.
    """

    score_threshold: float = 0.50
    low_threshold: float = 0.20
    iou_threshold: float = 0.45
    sweep_size: int = 640
    preferred_hold: int = 10
    lost_window: int = 2
    empty_after: int = 30
    empty_rescan: int = 15
    max_pass_roll: float = 45.0
    stats: SweepStats = field(default_factory=SweepStats, init=False)
    _preferred_k: int = field(default=0, init=False, repr=False)
    _preferred_left: int = field(default=0, init=False, repr=False)
    _recent: list[int] = field(default_factory=list, init=False, repr=False)
    _empty_streak: int = field(default=0, init=False, repr=False)
    _force_sweep: int = field(default=0, init=False, repr=False)
    _k_rows: dict[tuple[Any, ...], Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.low_threshold <= self.score_threshold:
            raise ValueError("low_threshold must be in [0, score_threshold]")
        if self.sweep_size % self.pad_multiple or self.sweep_size < self.input_size:
            raise ValueError("sweep_size must be a multiple of pad_multiple and >= input_size")

    # --------------------------------------------------------------- state
    def reset(self) -> None:
        """Forget the scene (call at a shot cut or a new clip)."""
        self._preferred_k = self._preferred_left = self._empty_streak = self._force_sweep = 0
        self._recent = []

    def note_track_lost(self, frames: int = 2) -> None:
        """Sweep on the next ``frames`` frames (a tracker lost a face)."""
        self._force_sweep = max(self._force_sweep, int(frames))

    @property
    def scene_empty(self) -> bool:
        return self._empty_streak >= self.empty_after

    def prepare(self, height: int, width: int, device: Any = "cuda") -> None:
        """Pre-build the anchor grids for ``H x W`` frames: the upright and the
        sideways tight canvases, and the ``sweep_size`` square (640 x 640, the
        largest)."""
        for hw in ((height, width), (width, height)):
            self._anchors(self.canvas_size(*hw)[3:], device)
        self._anchors((self.sweep_size, self.sweep_size), device)

    # --------------------------------------------------------------- canvases
    def _resized(self, frame: Any) -> tuple[Any, float]:
        """``(1, 3, rh, rw)`` normalized RGB in ``[-1, 1]``, and the scale."""
        import torch.nn.functional as F

        x = frame if frame.is_floating_point() else frame.float()
        _, _, h, w = x.shape
        scale, rh, rw, _, _ = self.canvas_size(h, w)
        resized = F.interpolate(x, size=(rh, rw), mode="bilinear", align_corners=False,
                                antialias=scale < 1.0) if (rh, rw) != (h, w) else x
        if self.input_format == "bgr255":
            return resized.flip(1).clamp(0, 255) * (2.0 / 255.0) - 1.0, scale
        return resized.clamp(0, 1) * 2.0 - 1.0, scale

    def _canvas(self, key: tuple[Any, ...], shape: tuple[int, ...], device: Any) -> Any:
        """A reused canvas; its black (-1) padding never changes."""
        import torch

        canvas = self._canvases.get(key)
        if canvas is None:
            canvas = self._canvases[key] = torch.full(shape, -1.0, device=device)
        return canvas

    def _infer(self, canvas: Any) -> list[Any]:
        """The 9 outputs as ``(B, anchors, k)`` per image, in anchor-grid order.

        The heads end in ``Transpose(2, 3, 0, 1) -> Reshape(-1, k)``, so a batch
        comes back ordered ``(h, w, image, anchor)``; this undoes that.
        """
        runner = self.runner
        b, _, ch, cw = canvas.shape
        shapes = None
        if not self.uses_tensorrt_engine:  # ORT needs the canvas's output lengths
            names = runner.output_names
            shapes = {}
            for level, stride in enumerate(self.strides):
                n = (ch // stride) * (cw // stride) * self.anchors_per_location * b
                shapes[names[level]] = (n, 1)
                shapes[names[3 + level]] = (n, 4)
                shapes[names[6 + level]] = (n, 10)
        out = runner.run_binding({runner.input_names[0]: canvas}, output_shapes=shapes)
        parts = []
        for j, name in enumerate(runner.output_names):
            stride = self.strides[j % len(self.strides)]
            cells = (ch // stride) * (cw // stride)
            t = out[name].float().reshape(cells, b, self.anchors_per_location, -1)
            parts.append(t.permute(1, 0, 2, 3).reshape(b, cells * self.anchors_per_location, -1))
        return parts

    def _candidates_cuda(self, outputs: list[Any], canvas_hw: tuple[int, int],
                         ks: tuple[int, ...], image_hw: tuple[int, int],
                         offset: tuple[int, int], scale: float, frame_hw: tuple[int, int],
                         square: bool) -> tuple[Any, Any, Any, Any, Any]:
        """Threshold at ``low_threshold``, decode, map back to the upright frame.

        ``ks[i]`` is image i's rotation. ``square``: the canvas held the upright
        resized image (``image_hw``) at ``offset`` = ``(oy, ox)`` and was then
        rotated whole. Otherwise the resized image was rotated and then placed
        at ``offset``. Returns ``(boxes, kps, scores, k_per_row, tilted)``.
        """
        import torch

        levels = len(self.strides)
        scores = torch.cat(outputs[:levels], 1)[..., 0]                  # (B, N)
        dist = torch.cat(outputs[levels:2 * levels], 1)                  # (B, N, 4)
        offs = torch.cat(outputs[2 * levels:], 1)                        # (B, N, 10)
        centers, strides = self._anchors(canvas_hw, scores.device)       # (N, 2), (N, 1)
        size = (dist[..., :2] + dist[..., 2:]) * strides / scale
        ok = ((scores >= self.low_threshold) & (size >= self.min_face_size).all(-1)
              & torch.isfinite(dist).all(-1) & torch.isfinite(offs).all(-1))
        img, anchor = ok.nonzero(as_tuple=True)
        c, st = centers[anchor], strides[anchor]
        d = dist[img, anchor] * st
        boxes = torch.cat([c - d[:, :2], c + d[:, 2:]], 1)
        kps = c[:, None, :] + offs[img, anchor].reshape(-1, 5, 2) * st[:, :, None]
        # Roll in the pass's own frame: the eye-mid -> mouth-mid axis vs +y.
        axis = (kps[:, 3] + kps[:, 4] - kps[:, 0] - kps[:, 1]) * 0.5
        roll = torch.rad2deg(torch.atan2(-axis[:, 0], axis[:, 1]))
        tilted = roll.abs() > self.max_pass_roll
        key = (ks, str(scores.device))
        if key not in self._k_rows:
            self._k_rows[key] = torch.tensor(ks, dtype=torch.float32, device=scores.device)
        k = self._k_rows[key][img]
        # Canvas -> frame as (p - offset) / scale in one fused op per tensor:
        # the per-frame cost here is kernel LAUNCHES (the GPU work is ~0.9 ms
        # either way), so the upright pass skips the rotation entirely.
        oy, ox = offset
        skey = ("shift", ox, oy, scale, str(boxes.device))
        if skey not in self._k_rows:
            self._k_rows[skey] = torch.tensor([ox, oy], dtype=boxes.dtype, device=boxes.device)
        shift = self._k_rows[skey]
        inv = 1.0 / scale
        if not any(ks):
            boxes = torch.sub(boxes, shift.repeat(2)).mul_(inv)
            kps = torch.sub(kps, shift).mul_(inv)
        elif square:  # rotation about the whole canvas first, then the placement
            ch, cw = canvas_hw
            boxes = (unrotate_boxes_cuda(boxes, k, ch, cw) - shift.repeat(2)).mul_(inv)
            kps = (unrotate_points_cuda(kps, k[:, None], ch, cw) - shift).mul_(inv)
        else:       # placement first, then the rotation of the resized image
            rh, rw = image_hw
            boxes = unrotate_boxes_cuda(boxes - shift.repeat(2), k, rh, rw).mul_(inv)
            kps = unrotate_points_cuda(kps - shift, k[:, None], rh, rw).mul_(inv)
        h, w = frame_hw
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
        return boxes, kps, scores[img, anchor], k, tilted

    def _primary(self, resized: Any, scale: float, k: int,
                 frame_hw: tuple[int, int]) -> tuple[Any, ...]:
        """Batch-1 pass on the tight canvas of the resized frame rotated by ``k``."""
        import torch

        rh, rw = resized.shape[-2:]
        rot = torch.rot90(resized, k, dims=[2, 3]) if k else resized
        th, tw = rot.shape[-2:]
        m, lo = self.pad_multiple, self.min_canvas
        ch, cw = max(lo, -(-th // m) * m), max(lo, -(-tw // m) * m)
        canvas = self._canvas(("primary", ch, cw, th, tw, str(resized.device)), (1, 3, ch, cw),
                              resized.device)
        oy, ox = self._offset(th, tw, ch, cw)
        canvas[..., oy:oy + th, ox:ox + tw].copy_(rot)
        return self._candidates_cuda(self._infer(canvas), (ch, cw), (k,), (rh, rw), (oy, ox),
                                     scale, frame_hw, square=False)

    def _sweep(self, resized: Any, scale: float, ks: tuple[int, ...],
               frame_hw: tuple[int, int]) -> tuple[Any, ...]:
        """The other angles as ONE batch of square canvases."""
        import torch

        rh, rw = resized.shape[-2:]
        s = self.sweep_size
        base = self._canvas(("sweep", s, rh, rw, str(resized.device)), (1, 3, s, s),
                            resized.device)
        oy, ox = (s - rh) // 2, (s - rw) // 2
        base[..., oy:oy + rh, ox:ox + rw].copy_(resized)
        batch = torch.cat([torch.rot90(base, k, dims=[2, 3]) for k in ks])  # (3, 3, S, S)
        return self._candidates_cuda(self._infer(batch), (s, s), ks, (rh, rw), (oy, ox),
                                     scale, frame_hw, square=True)

    # --------------------------------------------------------------- detection
    def _should_sweep(self, n_high: int, n_tilted: int) -> bool:
        if self._force_sweep > 0:
            return True
        if n_tilted:
            self.stats.tilt_sweeps += 1
            return True
        if self._recent and n_high < max(self._recent):
            return True  # a face was lost within the last lost_window frames
        if n_high > 0:
            return False
        if not self.scene_empty:
            return True
        due = (self._empty_streak - self.empty_after) % max(1, self.empty_rescan) == 0
        if not due:
            self.stats.empty_skips += 1
        return due

    def _split(self, boxes: Any, kps: Any, scores: Any, tilted: Any,
               frame_hw: tuple[int, int]) -> tuple[GPUDetections, GPUDetections, Any, int, int]:
        """NMS over all candidates -> (high, low, rows in rank order, #high, #tilted high).

        The NMS rank puts every candidate upright in its pass above every
        tilted one (scores are in [0, 1]); the output is then ordered by the
        real score, so high / low is a prefix split.
        """
        import torch
        from torchvision.ops import batched_nms

        dev = boxes.device
        if scores.shape[0] == 0:
            empty = GPUDetections.empty(dev, frame_hw)
            return empty, GPUDetections.empty(dev, frame_hw), scores.long(), 0, 0
        rank = scores - 2.0 * tilted.to(scores.dtype)
        ranked = batched_nms(boxes, rank, torch.zeros_like(scores, dtype=torch.int64),
                             float(self.iou_threshold))  # rank order
        kept = ranked[torch.argsort(scores[ranked], descending=True, stable=True)]
        b, k, s, t = boxes[kept], kps[kept], scores[kept], tilted[kept]
        high_mask = s >= self.score_threshold
        counts = torch.stack([high_mask.sum(), (high_mask & t).sum()])
        n_high, n_tilted = int(counts[0]), int(counts[1])  # scalars (one sync)
        index = torch.zeros(s.shape[0], dtype=torch.int64, device=dev)
        high = GPUDetections(b[:n_high], k[:n_high], s[:n_high], index[:n_high], frame_hw, 1)
        low = GPUDetections(b[n_high:], k[n_high:], s[n_high:], index[n_high:], frame_hw, 1)
        return high, low, ranked, n_high, n_tilted

    def detect_dual(self, frame: Any) -> DualDetections:
        """One frame, ``(3, H, W)`` or ``(1, 3, H, W)``; frames in video order."""
        import torch

        if frame.ndim == 3:
            frame = frame[None]
        if frame.shape[0] != 1:
            raise ValueError("detect_dual takes one frame (frames carry scene state)")
        h, w = frame.shape[-2:]
        resized, scale = self._resized(frame)
        k0 = self._preferred_k if self._preferred_left > 0 else 0
        boxes, kps, scores, ks, tilted = self._primary(resized, scale, k0, (h, w))
        high, low, ranked, n_high, n_tilted = self._split(boxes, kps, scores, tilted, (h, w))
        best_k, swept = k0, False
        if self._should_sweep(n_high, n_tilted):
            swept = True
            self.stats.sweeps += 1
            others = tuple(k for k in range(4) if k != k0)
            sb, skp, ss, sk, st = self._sweep(resized, scale, others, (h, w))
            if ss.shape[0]:
                upright_before = n_high - n_tilted
                boxes, kps = torch.cat([boxes, sb]), torch.cat([kps, skp])
                scores, ks = torch.cat([scores, ss]), torch.cat([ks, sk])
                tilted = torch.cat([tilted, st])
                high, low, ranked, n_high, n_tilted = self._split(boxes, kps, scores, tilted,
                                                                  (h, w))
                if n_high - n_tilted > upright_before:
                    self.stats.sweep_hits += 1
                if n_high:
                    best_k = int(ks[ranked[0]])  # the best-ranked face's pass
                    if best_k != k0:
                        self._preferred_k = best_k
                        self._preferred_left = self.preferred_hold + 1
        if self._preferred_left > 0:
            self._preferred_left -= 1
        self._force_sweep = max(0, self._force_sweep - 1)
        self._empty_streak = 0 if n_high else self._empty_streak + 1
        self._recent = (self._recent + [n_high])[-self.lost_window:]
        self.stats.frames += 1
        angle = 90 * best_k
        self.stats.angle_frames[angle] = self.stats.angle_frames.get(angle, 0) + 1
        return DualDetections(high, low, angle, swept, n_tilted)

    def detect_cuda(self, frames: Any) -> GPUDetections:
        """High-score faces (the :class:`GPUSCRFDDetector` interface); a batch is
        taken as consecutive frames, each through :meth:`detect_dual`."""
        import torch

        if frames.ndim == 3:
            frames = frames[None]
        b, _, h, w = frames.shape
        parts = [self.detect_dual(frames[i:i + 1]).high for i in range(b)]
        if b == 1:
            return parts[0]
        return GPUDetections(torch.cat([p.boxes for p in parts]),
                             torch.cat([p.kps for p in parts]),
                             torch.cat([p.scores for p in parts]),
                             torch.cat([torch.full_like(p.frame_index, i)
                                        for i, p in enumerate(parts)]), (h, w), b)
