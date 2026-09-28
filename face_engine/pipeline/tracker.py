"""Detection stride: full detection every N frames, optical-flow tracking between.

:class:`StridedFaceTracker` runs the detector on frame 0, N, 2N, ... and, on
the frames between, moves each face's landmarks and box with pyramidal
Lucas-Kanade optical flow (:class:`LucasKanadeTracker`), entirely on the GPU.
Detection is forced early when

* :class:`SceneCutDetector` sees a colour-histogram jump (a shot cut: the
  previous faces are meaningless in the new shot);
* tracking loses a face (too few points pass the forward-backward check, or
  the fitted motion is implausible);
* no faces are being tracked (nothing to follow, so every frame detects
  until something is found).

Landmark tracking uses optical flow rather than a landmark network: the
registry has no public ``hrffa`` release (see ``models/zoo.py``), and
``2dfan4`` (98 MB, 68 points) costs more per face than SCRFD per frame.

Each face is followed with a patch of points: its five landmarks plus a
grid inside the box. The grid's motion is fitted as one similarity
transform (weighted by the forward-backward check); the landmarks take their
own tracked position when they pass the check and the fitted motion
otherwise, so a mouth corner can move with the expression while an occluded
eye follows the head.

Host reads: every frame reads a small, fixed number of scalars (the cut
decision, and on tracked frames whether any face was lost) because those
decide control flow. No image or landmark data leaves the device.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from face_engine.pipeline.aligner import (
    estimate_similarity_transform_cuda,
    similarity_is_valid,
    transform_points_cuda,
)
from face_engine.pipeline.detector import BaseDetector, GPUDetections

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# BT.601 luma weights for BGR input.
_LUMA_BGR = (0.114, 0.587, 0.299)


class TrackerConfig(BaseModel):
    """Stride and tracking settings."""

    model_config = ConfigDict(frozen=True)

    detection_stride: int = Field(default=3, ge=1)
    # Lucas-Kanade
    window: int = Field(default=21, ge=5)  # odd patch side, pixels at each level
    levels: int = Field(default=4, ge=1)
    iterations: int = Field(default=8, ge=1)
    grid: int = Field(default=5, ge=2)  # grid x grid points per face box
    grid_extent: float = Field(default=0.6, gt=0.0, le=1.0)  # fraction of the box it spans
    fb_threshold: float = Field(default=1.0, gt=0.0)  # px, forward-backward error
    min_valid_fraction: float = Field(default=0.5, gt=0.0, le=1.0)
    max_scale_change: float = Field(default=1.25, gt=1.0)  # per frame
    # Scene cuts
    cut_threshold: float = Field(default=0.5, gt=0.0, le=1.0)
    histogram_bins: int = Field(default=32, ge=4)
    histogram_size: int = Field(default=64, ge=8)  # frames are pooled to this width
    # Identity across detections
    match_iou: float = Field(default=0.3, gt=0.0, lt=1.0)


# ---------------------------------------------------------------------------- scene cuts
class SceneCutDetector:
    """Shot-cut detector from per-channel colour histograms, on the GPU.

    Each frame is area-pooled to ``histogram_size`` px wide and binned per
    channel; the score is the total-variation distance between consecutive
    frames' normalized histograms, averaged over channels (0 = identical,
    1 = disjoint). A score above ``threshold`` is a cut.
    """

    def __init__(self, threshold: float = 0.5, bins: int = 32, size: int = 64) -> None:
        self.threshold = threshold
        self.bins = bins
        self.size = size
        self._previous: torch.Tensor | None = None

    def histogram(self, frame: torch.Tensor) -> torch.Tensor:
        """``(3, bins)`` normalized histogram of a ``(3, H, W)`` / ``(1, 3, H, W)`` frame."""
        import torch
        import torch.nn.functional as F

        f = frame if frame.ndim == 4 else frame[None]
        f = f.float()
        h, w = f.shape[-2:]
        size = (max(1, round(self.size * h / w)), self.size)
        small = F.interpolate(f, size=size, mode="area") if (h, w) != size else f
        q = (small[0].clamp(0, 255) * (self.bins / 256.0)).long().clamp(0, self.bins - 1)
        q = q.flatten(1) + torch.arange(3, device=q.device)[:, None] * self.bins
        # scatter_add, not bincount: bincount reads the max back to size its output.
        hist = torch.zeros(3 * self.bins, device=q.device).scatter_add_(
            0, q.flatten(), torch.ones(q.numel(), device=q.device)).view(3, self.bins)
        return hist / hist.sum(dim=1, keepdim=True)

    def score(self, frame: torch.Tensor) -> torch.Tensor:
        """Distance to the previous frame (0-dim tensor; 0 for the first frame)."""
        import torch

        hist = self.histogram(frame)
        previous, self._previous = self._previous, hist
        if previous is None:
            return torch.zeros((), device=hist.device)
        return 0.5 * (hist - previous).abs().sum(dim=1).mean()

    def reset(self) -> None:
        self._previous = None


# ---------------------------------------------------------------------------- optical flow
def to_gray(frame: torch.Tensor) -> torch.Tensor:
    """``(1, 3, H, W)`` / ``(3, H, W)`` BGR -> ``(1, 1, H, W)`` float32 luma."""
    f = frame if frame.ndim == 4 else frame[None]
    f = f.float()
    return (f[:, 0:1] * _LUMA_BGR[0] + f[:, 1:2] * _LUMA_BGR[1] + f[:, 2:3] * _LUMA_BGR[2])


class LucasKanadeTracker:
    """Pyramidal Lucas-Kanade (Bouguet) point tracking in pure PyTorch.

    All points are tracked together. Per level, one ``grid_sample`` cuts each
    point's template patch (plus a 1-px ring for its central-difference
    gradient, so no full-image gradient is computed); each Gauss-Newton step
    is one more ``grid_sample`` of the current frame plus a few elementwise
    ops. The iteration count is fixed (no data-dependent exit, so no host
    reads).

    Args:
        window: Odd patch side in pixels (at every pyramid level).
        levels: Pyramid levels (level L is 2^-L scale, 2x2 box downsampling).
        iterations: Gauss-Newton steps per level.
    """

    def __init__(self, window: int = 21, levels: int = 4, iterations: int = 8,
                 cuda_graphs: bool = True) -> None:
        self.window = window | 1
        self.levels = levels
        self.iterations = iterations
        self.cuda_graphs = cuda_graphs
        self._offsets: dict[tuple[str, int], Any] = {}
        self._scales: dict[tuple[str, int, int], Any] = {}
        self._graphs: dict[tuple[Any, ...], _GraphEntry] = {}

    def pyramid(self, gray: torch.Tensor) -> list[torch.Tensor]:
        """``[(1, 1, h, w), (1, 1, h/2, w/2), ...]``."""
        import torch.nn.functional as F

        out = [gray]
        while len(out) < self.levels and min(out[-1].shape[-2:]) >= 4 * self.window:
            out.append(F.avg_pool2d(out[-1], 2))
        return out

    def _window_offsets(self, device: Any, side: int) -> torch.Tensor:
        import torch

        key = (str(device), side)
        if key not in self._offsets:
            r = side // 2
            axis = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
            gy, gx = torch.meshgrid(axis, axis, indexing="ij")
            self._offsets[key] = torch.stack([gx, gy], -1).reshape(1, -1, 2)
        return self._offsets[key]

    def _sample(self, image: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """Bilinear samples of ``(1, 1, h, w)`` at ``(N, K, 2)`` pixel coords -> ``(N, K)``."""
        import torch
        import torch.nn.functional as F

        h, w = image.shape[-2:]
        key = (str(xy.device), h, w)
        scale = self._scales.get(key)
        if scale is None:  # created eagerly, never inside a graph capture
            scale = torch.tensor([2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1)], device=xy.device)
            self._scales[key] = scale
        n, k = xy.shape[:2]
        out = F.grid_sample(image, (xy * scale - 1.0).reshape(1, n, k, 2), mode="bilinear",
                            padding_mode="border", align_corners=True)
        return out.reshape(n, k)

    def track(self, previous: list[torch.Tensor], current: list[torch.Tensor],
              points: torch.Tensor) -> torch.Tensor:
        """New positions ``(N, 2)`` of ``(N, 2)`` level-0 points."""
        import torch

        side = self.window
        ring = self._window_offsets(points.device, side + 2)
        inner = self._window_offsets(points.device, side)
        n = points.shape[0]
        levels = min(len(previous), len(current))
        guess = torch.zeros_like(points)
        for level in reversed(range(levels)):
            p = points * (1.0 / (2 ** level))
            patch = self._sample(previous[level], p[:, None, :] + ring).view(n, side + 2, side + 2)
            template = patch[:, 1:-1, 1:-1].reshape(n, -1)
            ix = ((patch[:, 1:-1, 2:] - patch[:, 1:-1, :-2]) * 0.5).reshape(n, -1)
            iy = ((patch[:, 2:, 1:-1] - patch[:, :-2, 1:-1]) * 0.5).reshape(n, -1)
            grad = torch.stack([ix, iy], -1)  # (N, K, 2)
            hessian = grad.transpose(1, 2) @ grad  # (N, 2, 2)
            det = hessian[:, 0, 0] * hessian[:, 1, 1] - hessian[:, 0, 1] ** 2
            inv_det = torch.where(det.abs() > 1e-6, 1.0 / det, torch.zeros_like(det))
            inverse = torch.stack([torch.stack([hessian[:, 1, 1], -hessian[:, 0, 1]], -1),
                                   torch.stack([-hessian[:, 0, 1], hessian[:, 0, 0]], -1)],
                                  -2) * inv_det[:, None, None]
            steepest = grad @ inverse  # (N, K, 2): delta = sum_k err_k * steepest_k
            base = p[:, None, :] + inner
            for _ in range(self.iterations):
                err = template - self._sample(current[level], base + guess[:, None, :])
                guess = guess + (err[:, None, :] @ steepest)[:, 0]
            if level > 0:
                guess = guess * 2.0
        return points + guess

    def track_forward_backward(self, previous: list[torch.Tensor], current: list[torch.Tensor],
                               points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(forward, backward)``: points tracked into ``current`` and back again.

        On CUDA the pair runs as one captured CUDA graph per (point bucket,
        pyramid shape): eager LK is ~250 tiny kernels per direction and was
        launch-bound (7.7 ms for 60 or 150 points alike, RTX 4070,
        2026-09-28). Points are padded to a multiple of :data:`_POINT_BUCKET`.
        """
        if not (self.cuda_graphs and points.is_cuda) or points.shape[0] == 0:
            forward = self.track(previous, current, points)
            return forward, self.track(current, previous, forward)

        n = points.shape[0]
        capacity = -(-n // _POINT_BUCKET) * _POINT_BUCKET
        entry = self._graph(previous, current, capacity)
        for static, live in zip(entry.previous, previous):
            static.copy_(live)
        for static, live in zip(entry.current, current):
            static.copy_(live)
        entry.points[:n].copy_(points)
        entry.points[n:].copy_(points[:1].expand(capacity - n, 2))
        entry.graph.replay()
        return entry.forward[:n].clone(), entry.backward[:n].clone()

    def _graph(self, previous: list[torch.Tensor], current: list[torch.Tensor],
               capacity: int) -> _GraphEntry:
        key = (str(previous[0].device), capacity, tuple(tuple(t.shape) for t in previous),
               tuple(tuple(t.shape) for t in current))
        entry = self._graphs.get(key)
        if entry is None:
            entry = self._capture(previous, current, capacity)
            if len(self._graphs) >= _MAX_GRAPHS:
                self._graphs.pop(next(iter(self._graphs)))
            self._graphs[key] = entry
        return entry

    def prepare(self, pyramid: list[torch.Tensor], points: int) -> None:
        """Capture the graphs for up to ``points`` tracked points on this pyramid shape now.

        ``torch.cuda.graph`` synchronizes the whole DEVICE when a capture
        starts: the first tracked frame of a render stalled every CUDA stream
        of the Stage 7 pipeline once (found by ``face_engine/benchmark.py``'s
        sync counter, 2026-09-28). Capturing before the render keeps that out
        of the frame loop.
        """
        if not (self.cuda_graphs and pyramid[0].is_cuda):
            return
        for capacity in range(_POINT_BUCKET, points + _POINT_BUCKET, _POINT_BUCKET):
            self._graph(pyramid, pyramid, capacity)

    def _capture(self, previous: list[torch.Tensor], current: list[torch.Tensor],
                 capacity: int) -> _GraphEntry:
        import torch

        prev = [t.clone() for t in previous]
        cur = [t.clone() for t in current]
        pts = torch.zeros((capacity, 2), device=previous[0].device)
        side = torch.cuda.Stream(device=pts.device)
        side.wait_stream(torch.cuda.current_stream(pts.device))
        with torch.cuda.stream(side):  # warm-up: allocations, cached constants
            for _ in range(2):
                fwd = self.track(prev, cur, pts)
                self.track(cur, prev, fwd)
        torch.cuda.current_stream(pts.device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        # thread_local: in the default "global" mode a capture forbids unsafe
        # CUDA calls from EVERY thread, and NVDEC decoding on the
        # HardwareVideoDecoder thread makes them: the first capture mid-video
        # deadlocked the render (2026-09-28). Only this thread is restricted.
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            fwd = self.track(prev, cur, pts)
            bwd = self.track(cur, prev, fwd)
        return _GraphEntry(graph, prev, cur, pts, fwd, bwd)


_POINT_BUCKET = 64
_MAX_GRAPHS = 16


@dataclass
class _GraphEntry:
    graph: Any
    previous: list[Any]
    current: list[Any]
    points: Any
    forward: Any
    backward: Any


# ---------------------------------------------------------------------------- stride tracker
@dataclass
class TrackedFaces:
    """Faces for one frame, as device tensors (see :class:`GPUDetections`).

    Attributes:
        detections: Boxes, landmarks, scores (the score of the detection the
            track came from on tracked frames).
        track_ids: ``(N,)`` int64, stable across frames while a face is
            followed; a face re-detected after a cut or a loss gets a new id
            unless it overlaps its old box.
        source: ``"detect"`` (stride), ``"cut"``, ``"lost"``, ``"empty"``
            (a detection because nothing was tracked) or ``"track"``.
        frame_number: Frames seen so far, 0-based.
    """

    detections: GPUDetections
    track_ids: Any
    source: str
    frame_number: int

    @property
    def detected(self) -> bool:
        return self.source != "track"

    def __len__(self) -> int:
        return len(self.detections)


@dataclass
class TrackerStats:
    frames: int = 0
    detections: int = 0
    tracked: int = 0
    cuts: int = 0
    lost: int = 0
    by_source: dict[str, int] = field(default_factory=dict)

    @property
    def detector_saving(self) -> float:
        """Fraction of frames that did not run the detector."""
        return self.tracked / self.frames if self.frames else 0.0


class StridedFaceTracker:
    """Full detection every ``detection_stride`` frames, optical flow between.

    Feed frames in order, one at a time, as ``(3, H, W)`` or ``(1, 3, H, W)``
    BGR tensors on the GPU. :meth:`update` returns the frame's faces as device
    tensors. Call :meth:`reset` between clips.
    """

    def __init__(self, detector: BaseDetector, config: TrackerConfig | None = None) -> None:
        self.detector = detector
        self.config = config or TrackerConfig()
        c = self.config
        self.flow = LucasKanadeTracker(c.window, c.levels, c.iterations)
        self.cuts = SceneCutDetector(c.cut_threshold, c.histogram_bins, c.histogram_size)
        self.stats = TrackerStats()
        self._faces: GPUDetections | None = None
        self._ids: Any = None
        self._next_id: Any = None
        self._pyramid: list[Any] | None = None
        self._since_detection = 0

    def reset(self) -> None:
        self.cuts.reset()
        self.stats = TrackerStats()
        self._faces = self._ids = self._next_id = self._pyramid = None
        self._since_detection = 0

    # ------------------------------------------------------------------ public
    def prepare(self, height: int, width: int, max_faces: int = 8, device: Any = "cuda") -> None:
        """Pre-capture the optical-flow CUDA graphs for ``H x W`` frames and up to ``max_faces``.

        See :meth:`LucasKanadeTracker.prepare`; a face count beyond ``max_faces``
        still works, capturing its graph on first use.
        """
        import torch

        gray = torch.zeros((1, 1, height, width), device=device)
        per_face = 5 + self.config.grid ** 2
        self.flow.prepare(self.flow.pyramid(gray), max_faces * per_face)

    def update(self, frame: torch.Tensor) -> TrackedFaces:
        f = frame if frame.ndim == 4 else frame[None]
        number = self.stats.frames
        pyramid = self.flow.pyramid(to_gray(f))
        cut = bool(self.cuts.score(f) > self.config.cut_threshold)  # host read: 1 scalar

        source: str | None = None
        if self._faces is None:
            source = "detect"
        elif cut:
            source = "cut"
        elif self._since_detection + 1 >= self.config.detection_stride:
            source = "detect"
        elif len(self._faces) == 0:
            source = "empty"

        tracked: GPUDetections | None = None
        if source is None:
            assert self._pyramid is not None
            tracked, lost = self._track(self._faces, self._pyramid, pyramid)
            if bool(lost.any()):  # host read: 1 scalar
                source = "lost"
        if source is not None:
            faces = self.detector.detect_cuda(f)
            ids = self._assign_ids(faces, None if source == "cut" else self._faces)
            self._since_detection = 0
            self.stats.detections += 1
            self.stats.cuts += source == "cut"
            self.stats.lost += source == "lost"
        else:
            assert tracked is not None
            faces, ids, source = tracked, self._ids, "track"
            self._since_detection += 1
            self.stats.tracked += 1
        self.stats.frames += 1
        self.stats.by_source[source] = self.stats.by_source.get(source, 0) + 1
        self._faces, self._ids, self._pyramid = faces, ids, pyramid
        return TrackedFaces(faces, ids, source, number)

    # ------------------------------------------------------------------ tracking
    def _face_points(self, faces: GPUDetections) -> torch.Tensor:
        """``(F, 5 + g^2, 2)``: landmarks, then a grid over the box centre."""
        import torch

        g, extent = self.config.grid, self.config.grid_extent
        axis = torch.linspace(-extent / 2, extent / 2, g, device=faces.boxes.device)
        gy, gx = torch.meshgrid(axis, axis, indexing="ij")
        unit = torch.stack([gx, gy], -1).reshape(1, -1, 2)
        center = (faces.boxes[:, :2] + faces.boxes[:, 2:]) / 2
        size = (faces.boxes[:, 2:] - faces.boxes[:, :2])
        grid = center[:, None, :] + unit * size[:, None, :]
        return torch.cat([faces.kps, grid], dim=1)

    def _track(self, faces: GPUDetections, previous: list[Any],
               current: list[Any]) -> tuple[GPUDetections, torch.Tensor]:
        import torch

        c = self.config
        points = self._face_points(faces)  # (F, P, 2)
        f, p = points.shape[:2]
        flat = points.reshape(-1, 2)
        forward, backward = self.flow.track_forward_backward(previous, current, flat)
        h, w = faces.frame_size
        inside = ((forward[:, 0] >= 0) & (forward[:, 0] <= w - 1)
                  & (forward[:, 1] >= 0) & (forward[:, 1] <= h - 1))
        ok = (((backward - flat).norm(dim=1) < c.fb_threshold) & inside
              & torch.isfinite(forward).all(1)).reshape(f, p)
        moved = forward.reshape(f, p, 2)
        motion = estimate_similarity_transform_cuda(points, moved, weights=ok.float())
        rigid = transform_points_cuda(points, motion)
        # Landmarks: own flow where it passed the check, else the face's motion.
        kps = torch.where(ok[:, :5, None], moved[:, :5], rigid[:, :5])
        scale = torch.sqrt((motion[:, 0, 0] ** 2 + motion[:, 1, 0] ** 2).clamp_min(0))
        valid_fraction = ok[:, 5:].float().mean(1)
        lost = ((valid_fraction < c.min_valid_fraction) | ~similarity_is_valid(motion)
                | (scale > c.max_scale_change) | (scale < 1.0 / c.max_scale_change))
        corners = faces.boxes.reshape(-1, 2, 2)
        center = transform_points_cuda(corners.mean(1, keepdim=True), motion)[:, 0]
        half = (corners[:, 1] - corners[:, 0]) * scale[:, None] / 2
        boxes = torch.cat([center - half, center + half], 1)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
        tracked = GPUDetections(boxes, kps, faces.scores, faces.frame_index, faces.frame_size,
                                faces.num_frames)
        return tracked, lost

    # ------------------------------------------------------------------ identity
    def _assign_ids(self, faces: GPUDetections, previous: GPUDetections | None) -> torch.Tensor:
        """Mutual-best IoU match to the previous faces keeps their id; others get new ids."""
        import torch
        from torchvision.ops import box_iou

        device = faces.boxes.device
        n = len(faces)
        if self._next_id is None:
            self._next_id = torch.zeros((), dtype=torch.int64, device=device)
        ids = torch.full((n,), -1, dtype=torch.int64, device=device)
        if previous is not None and len(previous) and n:
            iou = box_iou(faces.boxes, previous.boxes)  # (n, m)
            best_prev = iou.argmax(1)
            best_new = iou.argmax(0)
            mutual = best_new[best_prev] == torch.arange(n, device=device)
            keep = mutual & (iou.gather(1, best_prev[:, None])[:, 0] >= self.config.match_iou)
            ids = torch.where(keep, self._ids[best_prev], ids)
        new = ids < 0
        ids = torch.where(new, self._next_id + torch.cumsum(new.long(), 0) - 1, ids)
        self._next_id = self._next_id + new.long().sum()
        return ids


# ---------------------------------------------------------------------------- temporal smoothing
class TemporalTrackerConfig(BaseModel):
    """:class:`TemporalFaceTracker` settings.

    Attributes:
        alpha: EMA weight of the NEW measurement: ``s = alpha * z + (1 - alpha) * s_prev``.
        iou_weight: Match cost = ``w * (1 - IoU) + (1 - w) * landmark distance``
            (mean point distance / face size, capped at 1).
        max_cost: Pairs above this cost are not matched.
        max_missed: Frames a track is carried on its motion vector while its
            face is not detected before it is dropped.
        smooth: False matches and retains without smoothing (the raw boxes).
    """

    model_config = ConfigDict(frozen=True)

    alpha: float = Field(default=0.75, gt=0.0, le=1.0)
    iou_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    max_cost: float = Field(default=0.7, gt=0.0)
    max_missed: int = Field(default=3, ge=0)
    smooth: bool = True


@dataclass
class TemporalFaces:
    """One frame's tracked faces.

    Attributes:
        detections: Smoothed (or predicted) boxes and landmarks.
        track_ids: ``(N,)`` int64 (host list), stable while a face is followed.
        predicted: ``(N,)`` bool (host list): True for a face carried on its
            motion vector because the detector missed it this frame.
    """

    detections: GPUDetections
    track_ids: list[int]
    predicted: list[bool]


class TemporalFaceTracker:
    """Frame-to-frame face tracks with EMA smoothing and short-gap retention.

    Feed each frame's :class:`GPUDetections` in order (from a detector or from
    :class:`StridedFaceTracker`). Detections are matched to tracks by IoU and
    landmark distance (computed on the GPU; the small cost matrix is read back
    for the assignment, one sync per frame), smoothed with an EMA, and a track
    whose face is missed is predicted from its last motion for up to
    ``max_missed`` frames instead of disappearing (a one-frame miss is a
    visible flicker: the face pops between swapped and original).

    Smoothing trades jitter for lag: the smoothed landmarks trail a moving face
    by ``(1 - alpha) / alpha`` frames of its motion. Whether that is a net win
    is measured, not assumed (``face_engine/README.md``).
    """

    def __init__(self, config: TemporalTrackerConfig | None = None) -> None:
        self.config = config or TemporalTrackerConfig()
        self._boxes: Any = None      # (T, 4) smoothed
        self._kps: Any = None        # (T, 5, 2) smoothed
        self._vel_boxes: Any = None  # (T, 4) per-frame motion
        self._vel_kps: Any = None
        self._scores: Any = None
        self._ids: list[int] = []
        self._missed: list[int] = []
        self._next_id = 0
        self.frames = 0

    def reset(self) -> None:
        self.__init__(self.config)  # type: ignore[misc]

    def _cost(self, boxes: Any, kps: Any) -> Any:
        """``(T, D)`` match cost between tracks and detections, on the device."""
        from torchvision.ops import box_iou

        c = self.config
        iou = box_iou(self._boxes, boxes)
        size = ((self._boxes[:, 2:] - self._boxes[:, :2]).clamp_min(1).prod(1).sqrt())[:, None]
        dist = (self._kps[:, None] - kps[None]).norm(dim=-1).mean(-1) / size
        return c.iou_weight * (1.0 - iou) + (1.0 - c.iou_weight) * dist.clamp(max=1.0)

    def update(self, detections: GPUDetections) -> TemporalFaces:
        import numpy as np
        import torch
        from scipy.optimize import linear_sum_assignment

        c = self.config
        self.frames += 1
        boxes, kps, scores = detections.boxes, detections.kps, detections.scores
        dev = boxes.device
        n_tracks, n_det = len(self._ids), int(boxes.shape[0])
        pairs: list[tuple[int, int]] = []
        if n_tracks and n_det:
            cost = self._cost(boxes, kps).cpu().numpy()  # small: tracks x detections
            rows, cols = linear_sum_assignment(cost)
            pairs = [(int(r), int(k)) for r, k in zip(rows, cols) if cost[r, k] <= c.max_cost]
        matched_t = {t for t, _ in pairs}
        matched_d = {d for _, d in pairs}
        ids: list[int] = []
        predicted: list[bool] = []
        a = c.alpha if c.smooth else 1.0
        new_state: dict[str, list[Any]] = {"b": [], "k": [], "vb": [], "vk": [], "s": []}
        missed: list[int] = []
        for t, d in pairs:
            b = a * boxes[d] + (1 - a) * self._boxes[t]
            k = a * kps[d] + (1 - a) * self._kps[t]
            new_state["b"].append(b)
            new_state["k"].append(k)
            new_state["vb"].append(b - self._boxes[t])
            new_state["vk"].append(k - self._kps[t])
            new_state["s"].append(scores[d])
            ids.append(self._ids[t])
            missed.append(0)
            predicted.append(False)
        for t in range(n_tracks):
            if t in matched_t or self._missed[t] >= c.max_missed:
                continue
            new_state["b"].append(self._boxes[t] + self._vel_boxes[t])
            new_state["k"].append(self._kps[t] + self._vel_kps[t])
            new_state["vb"].append(self._vel_boxes[t])
            new_state["vk"].append(self._vel_kps[t])
            new_state["s"].append(self._scores[t])
            ids.append(self._ids[t])
            missed.append(self._missed[t] + 1)
            predicted.append(True)
        for d in range(n_det):
            if d in matched_d:
                continue
            new_state["b"].append(boxes[d])
            new_state["k"].append(kps[d])
            new_state["vb"].append(torch.zeros_like(boxes[d]))
            new_state["vk"].append(torch.zeros_like(kps[d]))
            new_state["s"].append(scores[d])
            ids.append(self._next_id)
            self._next_id += 1
            missed.append(0)
            predicted.append(False)
        if ids:
            self._boxes = torch.stack(new_state["b"])
            self._kps = torch.stack(new_state["k"])
            self._vel_boxes = torch.stack(new_state["vb"])
            self._vel_kps = torch.stack(new_state["vk"])
            self._scores = torch.stack(new_state["s"])
        else:
            self._boxes = self._kps = self._vel_boxes = self._vel_kps = self._scores = None
        self._ids, self._missed = ids, missed
        if not ids:
            return TemporalFaces(GPUDetections.empty(dev, detections.frame_size), [], [])
        h, w = detections.frame_size
        shown = self._boxes.clone()
        shown[:, 0::2] = shown[:, 0::2].clamp(0, w)
        shown[:, 1::2] = shown[:, 1::2].clamp(0, h)
        order = np.argsort(ids, kind="stable")
        idx = torch.as_tensor(order, device=dev)
        return TemporalFaces(
            GPUDetections(shown[idx], self._kps[idx], self._scores[idx],
                          torch.zeros(len(ids), dtype=torch.int64, device=dev),
                          detections.frame_size, 1),
            [ids[i] for i in order], [predicted[i] for i in order])


# ---------------------------------------------------------------------------- ByteTrack
class ByteTrackConfig(BaseModel):
    """:class:`RobustByteTracker` settings.

    Attributes:
        high_threshold: Detections at or above this score are "high": they
            are matched first and are the only ones that start tracks.
        low_threshold: Detections in ``[low, high)`` only keep existing
            tracks alive (the second association).
        match_iou_high: Minimum IoU for the first association (ByteTrack's
            ``match_thresh`` 0.8 as a distance).
        match_iou_low: Minimum IoU for the second association (ByteTrack: 0.5).
        new_track_threshold: A high detection left unmatched starts a track
            only at this score or above.
        max_lost_frames: Frames a track is kept (and predicted) without a match.
        alpha: EMA weight of the NEW landmark measurement.
        motion_compensated_ema: Move the previous smoothed landmarks with the
            Kalman prediction (centre shift and height ratio) before blending,
            so the EMA smooths jitter about the motion instead of trailing the
            motion. Plain EMA (False) trails a face moving v px/frame by
            ``v (1 - alpha) / alpha`` (1/3 v at alpha 0.75), which is what kept
            :class:`TemporalFaceTracker`'s EMA off in the render (1.6-2.7x
            the raw error on the fastest decile of four clips, 2026-09-28).
        low_association: False disables the second association (plain
            single-threshold tracking; the control arm of the tests).
        emit_lost: Report lost tracks (at their predicted position, flagged
            ``predicted``) until they are dropped.
    """

    model_config = ConfigDict(frozen=True)

    high_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    low_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    match_iou_high: float = Field(default=0.20, ge=0.0, le=1.0)
    match_iou_low: float = Field(default=0.50, ge=0.0, le=1.0)
    new_track_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    max_lost_frames: int = Field(default=5, ge=0)
    alpha: float = Field(default=0.75, gt=0.0, le=1.0)
    motion_compensated_ema: bool = True
    low_association: bool = True
    emit_lost: bool = True


@dataclass
class ByteTracks:
    """One frame's tracks.

    Attributes:
        detections: Kalman boxes and EMA landmarks (predicted for lost tracks),
            ordered by track id; ``scores`` are each track's last matched score.
        track_ids: Host list, stable while a face is followed.
        predicted: True for a lost track reported at its predicted position.
        low_matched: True for a track kept alive by a LOW detection this frame.
    """

    detections: GPUDetections
    track_ids: list[int]
    predicted: list[bool]
    low_matched: list[bool]


@dataclass
class ByteTrackStats:
    frames: int = 0
    high_matches: int = 0
    low_matches: int = 0
    new_tracks: int = 0
    lost_events: int = 0
    removed: int = 0


class _KalmanXYAH:
    """ByteTrack's constant-velocity Kalman filter on ``(cx, cy, a, h)``, batched
    over tracks in torch (``(T, 8)`` means, ``(T, 8, 8)`` covariances).

    Noise is proportional to the box height as in ByteTrack
    (``std_weight_position`` 1/20, ``std_weight_velocity`` 1/160). The gain
    uses ``torch.linalg.solve_ex`` (no error check, so no host sync).
    """

    wp, wv = 1.0 / 20.0, 1.0 / 160.0

    def __init__(self, device: Any, dtype: Any) -> None:
        import torch

        self.F = torch.eye(8, device=device, dtype=dtype)
        self.F[:4, 4:] = torch.eye(4, device=device, dtype=dtype)
        self.H = torch.eye(4, 8, device=device, dtype=dtype)

    def _diag(self, h: Any, pos: tuple[float, float, float, float]) -> Any:
        import torch

        wp = torch.stack([pos[0] * h, pos[1] * h, torch.full_like(h, pos[2]), pos[3] * h], -1)
        return torch.diag_embed(wp ** 2)

    def initiate(self, z: Any) -> tuple[Any, Any]:
        import torch

        h = z[:, 3]
        std = torch.stack([2 * self.wp * h, 2 * self.wp * h, torch.full_like(h, 1e-2),
                           2 * self.wp * h, 10 * self.wv * h, 10 * self.wv * h,
                           torch.full_like(h, 1e-5), 10 * self.wv * h], -1)
        return torch.cat([z, torch.zeros_like(z)], 1), torch.diag_embed(std ** 2)

    def predict(self, mean: Any, cov: Any) -> tuple[Any, Any]:
        import torch

        h = mean[:, 3]
        std = torch.stack([self.wp * h, self.wp * h, torch.full_like(h, 1e-2), self.wp * h,
                           self.wv * h, self.wv * h, torch.full_like(h, 1e-5), self.wv * h], -1)
        mean = mean @ self.F.T
        cov = self.F @ cov @ self.F.T + torch.diag_embed(std ** 2)
        return mean, cov

    def update(self, mean: Any, cov: Any, z: Any) -> tuple[Any, Any]:
        import torch

        r = self._diag(mean[:, 3], (self.wp, self.wp, 1e-1, self.wp))
        s = self.H @ cov @ self.H.T + r                              # (M, 4, 4)
        pht = cov @ self.H.T                                          # (M, 8, 4)
        gain = torch.linalg.solve_ex(s, pht.transpose(1, 2), check_errors=False)[0]
        gain = gain.transpose(1, 2)                                   # (M, 8, 4)
        innovation = z - mean[:, :4]
        mean = mean + (gain @ innovation[..., None])[..., 0]
        cov = cov - gain @ s @ gain.transpose(1, 2)
        return mean, cov


def _xyah(boxes: Any) -> Any:
    import torch

    wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-3)
    return torch.cat([(boxes[:, :2] + boxes[:, 2:]) * 0.5, (wh[:, 0] / wh[:, 1])[:, None],
                      wh[:, 1:2]], 1)


def _tlbr(mean: Any) -> Any:
    import torch

    h = mean[:, 3]
    w = mean[:, 2] * h
    cx, cy = mean[:, 0], mean[:, 1]
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


class RobustByteTracker:
    """ByteTrack for faces: two-stage association, Kalman boxes, EMA landmarks.

    Per frame (:meth:`update`, frames in order):

    1. Every live track (tracked and lost) is predicted one frame ahead by
       the Kalman filter; a lost track's landmarks move with its box (centre
       shift, height ratio): the "previous velocity" prediction.
    2. **First association**: HIGH detections (score >= 0.50) vs all live
       tracks, IoU distance, Hungarian assignment, IoU >= ``match_iou_high``.
    3. **Second association**: tracks still unmatched that were TRACKED last
       frame vs the LOW detections (0.20 <= score < 0.50), IoU >=
       ``match_iou_low``. A head turning toward profile, or into shadow,
       drops SCRFD's score to 0.25-0.35 while the box stays put; this keeps
       the track (and its id) instead of losing it.
    4. Matched tracks take a Kalman update and an EMA landmark update; the
       unmatched become lost (kept ``max_lost_frames``); unmatched HIGH
       detections start tracks. A low detection never starts a track (the
       0.45-0.50 band alone held the back of a head and background specks,
       2026-09-28).

    One host read per frame: the ``tracks x detections`` IoU matrix for the
    assignment (as :class:`TemporalFaceTracker`). ``on_lost`` is called when a
    tracked face goes unmatched, e.g. ``AngleResilientSCRFD.note_track_lost``
    so the detector sweeps other rotations on the next frames.
    """

    def __init__(self, config: ByteTrackConfig | None = None,
                 on_lost: Any = None) -> None:
        self.config = config or ByteTrackConfig()
        self.on_lost = on_lost
        self.stats = ByteTrackStats()
        self._kf: _KalmanXYAH | None = None
        self._mean: Any = None   # (T, 8)
        self._cov: Any = None    # (T, 8, 8)
        self._kps: Any = None    # (T, 5, 2) EMA landmarks
        self._scores: Any = None
        self._ids: list[int] = []
        self._lost: list[int] = []   # 0 = tracked this frame
        self._next_id = 0

    def reset(self) -> None:
        self.__init__(self.config, self.on_lost)  # type: ignore[misc]

    def __len__(self) -> int:
        return len(self._ids)

    # --------------------------------------------------------------- inputs
    def _split(self, detections: Any) -> tuple[GPUDetections, GPUDetections]:
        """DualDetections -> (high, low); a plain GPUDetections is split here."""
        high = getattr(detections, "high", None)
        if high is not None:
            return high, detections.low
        s = detections.scores
        c = self.config
        return (detections.index(s >= c.high_threshold),
                detections.index((s >= c.low_threshold) & (s < c.high_threshold)))

    # --------------------------------------------------------------- update
    def update(self, detections: Any) -> ByteTracks:
        """Advance one frame with a ``DualDetections`` (or ``GPUDetections``)."""
        import torch
        from scipy.optimize import linear_sum_assignment
        from torchvision.ops import box_iou

        c = self.config
        self.stats.frames += 1
        high, low = self._split(detections)
        frame_size = high.frame_size
        dev = high.boxes.device
        dtype = torch.float32
        if self._kf is None:
            self._kf = _KalmanXYAH(dev, dtype)
        kf = self._kf
        n_high, n_low = len(high), len(low)
        n_tracks = len(self._ids)
        det_boxes = torch.cat([high.boxes, low.boxes]).to(dtype)
        det_kps = torch.cat([high.kps, low.kps]).to(dtype)
        det_scores = torch.cat([high.scores, low.scores]).to(dtype)

        # 1. predict (and carry the landmarks with the box)
        pred_kps = None
        if n_tracks:
            prev = self._mean
            self._mean, self._cov = kf.predict(self._mean, self._cov)
            ratio = (self._mean[:, 3] / prev[:, 3].clamp_min(1e-3))[:, None, None]
            pred_kps = ((self._kps - prev[:, None, :2]) * ratio + self._mean[:, None, :2])

        # 2-3. associations on one host copy of the IoU matrix
        pairs: list[tuple[int, int]] = []
        if n_tracks and (n_high + n_low):
            iou = box_iou(_tlbr(self._mean), det_boxes).cpu().numpy()
            free_t = list(range(n_tracks))
            if n_high:
                cost = 1.0 - iou[:, :n_high]
                rows, cols = linear_sum_assignment(cost)
                for r, k in zip(rows, cols):
                    if iou[r, k] >= c.match_iou_high:
                        pairs.append((int(r), int(k)))
                matched = {t for t, _ in pairs}
                free_t = [t for t in free_t if t not in matched]
            if c.low_association and n_low:
                cand = [t for t in free_t if self._lost[t] == 0]  # tracked last frame
                if cand:
                    sub = iou[cand][:, n_high:]
                    rows, cols = linear_sum_assignment(1.0 - sub)
                    for r, k in zip(rows, cols):
                        if sub[r, k] >= c.match_iou_low:
                            pairs.append((cand[int(r)], n_high + int(k)))
        self.stats.high_matches += sum(1 for _, d in pairs if d < n_high)
        self.stats.low_matches += sum(1 for _, d in pairs if d >= n_high)

        # 4. update matched, age unmatched, start new
        matched_t = {t: d for t, d in pairs}
        ids, lost, low_flag = [], [], []
        keep_rows: list[int] = []
        lost_now = 0
        for t in range(n_tracks):
            if t in matched_t:
                keep_rows.append(t)
                ids.append(self._ids[t])
                lost.append(0)
                low_flag.append(matched_t[t] >= n_high)
            else:
                if self._lost[t] == 0:
                    lost_now += 1
                if self._lost[t] + 1 > c.max_lost_frames:
                    self.stats.removed += 1
                    continue
                keep_rows.append(t)
                ids.append(self._ids[t])
                lost.append(self._lost[t] + 1)
                low_flag.append(False)
        if n_tracks:
            rows_t = torch.as_tensor(keep_rows, dtype=torch.int64, device=dev)
            mean, cov = self._mean[rows_t], self._cov[rows_t]
            kps, scores = pred_kps[rows_t], self._scores[rows_t]
            m_pos = [i for i, t in enumerate(keep_rows) if t in matched_t]
            if m_pos:
                pos = torch.as_tensor(m_pos, dtype=torch.int64, device=dev)
                d_idx = torch.as_tensor([matched_t[keep_rows[i]] for i in m_pos],
                                        dtype=torch.int64, device=dev)
                um, uc = kf.update(mean[pos], cov[pos], _xyah(det_boxes[d_idx]))
                mean, cov = mean.index_copy(0, pos, um), cov.index_copy(0, pos, uc)
                base = kps[pos] if c.motion_compensated_ema else self._kps[rows_t][pos]
                ema = c.alpha * det_kps[d_idx] + (1.0 - c.alpha) * base
                kps = kps.index_copy(0, pos, ema)
                scores = scores.index_copy(0, pos, det_scores[d_idx])
        else:
            mean = torch.zeros((0, 8), device=dev, dtype=dtype)
            cov = torch.zeros((0, 8, 8), device=dev, dtype=dtype)
            kps = torch.zeros((0, 5, 2), device=dev, dtype=dtype)
            scores = torch.zeros((0,), device=dev, dtype=dtype)
        matched_d = set(matched_t.values())
        if n_high:
            h_scores = None
            fresh = [d for d in range(n_high) if d not in matched_d]
            if fresh and c.new_track_threshold > c.high_threshold:
                h_scores = high.scores.cpu().numpy()  # only when the bar is above high
                fresh = [d for d in fresh if h_scores[d] >= c.new_track_threshold]
            if fresh:
                f_idx = torch.as_tensor(fresh, dtype=torch.int64, device=dev)
                nm, nc = kf.initiate(_xyah(det_boxes[f_idx]))
                mean, cov = torch.cat([mean, nm]), torch.cat([cov, nc])
                kps = torch.cat([kps, det_kps[f_idx]])
                scores = torch.cat([scores, det_scores[f_idx]])
                for _ in fresh:
                    ids.append(self._next_id)
                    self._next_id += 1
                    lost.append(0)
                    low_flag.append(False)
                self.stats.new_tracks += len(fresh)
        self._mean, self._cov, self._kps, self._scores = mean, cov, kps, scores
        self._ids, self._lost = ids, lost
        if lost_now:
            self.stats.lost_events += lost_now
            if self.on_lost is not None:
                self.on_lost()
        return self._emit(frame_size, dev, low_flag)

    def _emit(self, frame_size: tuple[int, int], dev: Any, low_flag: list[bool]) -> ByteTracks:
        import torch

        show = [i for i, l in enumerate(self._lost) if l == 0 or self.config.emit_lost]
        if not show:
            return ByteTracks(GPUDetections.empty(dev, frame_size), [], [], [])
        show.sort(key=lambda i: self._ids[i])
        idx = torch.as_tensor(show, dtype=torch.int64, device=dev)
        boxes = _tlbr(self._mean[idx])
        h, w = frame_size
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
        dets = GPUDetections(boxes, self._kps[idx], self._scores[idx],
                             torch.zeros(len(show), dtype=torch.int64, device=dev),
                             frame_size, 1)
        return ByteTracks(dets, [self._ids[i] for i in show],
                          [self._lost[i] > 0 for i in show], [low_flag[i] for i in show])
