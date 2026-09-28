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
