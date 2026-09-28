"""RetinaFace ResNet-50 on the GPU: TensorRT engine, vectorized anchor decoding, batched NMS.

:class:`RetinaFaceR50Detector` is a :class:`~face_engine.pipeline.detector.BaseDetector`,
so :class:`~face_engine.pipeline.tracker.StridedFaceTracker` runs it every
``detection_stride`` frames and tracks landmarks by optical flow in between
(:func:`strided_tracker`), and ``GpuFrameProcessor`` uses it through
``RenderParams.detector_model = "retinaface_r50"``.

The export (``retinaface_r50``, biubug6/Pytorch_Retinaface) and what
roop-ultimate learned running it (``app/roop/retinaface.py``):

* 640 x 640 input, BGR minus ``(104, 117, 123)``, no scaling;
* the softmax is inside the graph: the face score is ``conf[..., 1]``
  (re-applying softmax flattened every score: 0 faces at threshold 0.8);
* PriorBox anchors: min sizes (16, 32) / (64, 128) / (256, 512) on steps
  8 / 16 / 32, variances (0.1, 0.2), in (cx, cy, w, h) order.

**Letterboxed, not square-resized.** roop-ultimate squashes each frame to
640 x 640 (its note: letterboxing "suppressed scores under TensorRT on 16:9").
Measured here (2026-09-28, TensorRT FP16 and ONNX Runtime FP32 agree
exactly), against SCRFD's detections on 60 frames per clip:

    faces SCRFD found that R50 missed    squash   letterbox
    d1 (1080p, two faces, 100 frames)      41 / 63 + 21 frames w/ 2 faces    2 (94 frames w/ 2)
    Weeds / Love / d2 (720p)             13 / 7 / 8               4 / 1 / 0
    d6 (4K)                                0 (+17 duplicate boxes)  0

Squashing 16:9 by 3x / 1.7x makes faces 2:1 wide and loses the second face in
contact shots; the letterbox's extra detections over SCRFD were real faces
(upside down, turned away, behind a hand). So the frame is letterboxed
(:func:`~face_engine.pipeline.detector.letterbox_cuda`, centred, black pad).

Inference runs on the AOT TensorRT engine when ``tools/compile_engines.py``
built one (FP16, 2.15 ms per 640 frame on an RTX 4070, fidelity 3.5e-4 of
range, 2026-09-28): ``TensorRTEngine.run_binding`` sets the torch tensors'
addresses on the execution context and enqueues on the current stream.
Without an engine it falls back to ONNX Runtime IOBinding. Decoding,
thresholding and ``torchvision.ops.batched_nms`` (IoU 0.45) stay on the device.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from face_engine.pipeline.detector import (
    BaseDetector,
    Face,
    GPUDetections,
    Letterbox,
    _to_frame_cuda,
    letterbox_cuda,
)

logger = logging.getLogger(__name__)

INPUT_SIZE = 640
MEAN_BGR = (104.0, 117.0, 123.0)
MIN_SIZES = ((16, 32), (64, 128), (256, 512))
STEPS = (8, 16, 32)
VARIANCE = (0.1, 0.2)


def priors(size: int = INPUT_SIZE) -> np.ndarray:
    """``(N, 4)`` normalized ``(cx, cy, w, h)`` anchors in the network's order (biubug6 PriorBox)."""
    out = []
    for step, sizes in zip(STEPS, MIN_SIZES):
        f = int(np.ceil(size / step))
        cy, cx = np.meshgrid((np.arange(f) + 0.5) * step / size,
                             (np.arange(f) + 0.5) * step / size, indexing="ij")
        centers = np.stack([cx.reshape(-1), cy.reshape(-1)], 1)
        for_each = np.repeat(centers, len(sizes), axis=0)
        wh = np.tile(np.array(sizes, np.float64)[:, None] / size, (centers.shape[0], 2))
        out.append(np.concatenate([for_each, wh], 1))
    return np.concatenate(out).astype(np.float32)


@dataclass
class RetinaFaceR50Detector(BaseDetector):
    """RetinaFace R50 over ``(B, 3, H, W)`` BGR CUDA frames (see the module docstring)."""

    input_size: int = INPUT_SIZE
    score_threshold: float = 0.6
    iou_threshold: float = 0.45
    precision: str = "fp16"
    _runner: Any = field(default=None, init=False, repr=False)
    _priors: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _softmaxed: bool | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.input_size != INPUT_SIZE:
            raise ValueError("retinaface_r50 is calibrated at 640 x 640 only")

    @property
    def runner(self) -> Any:
        """The AOT TensorRT engine when compiled for this GPU, else the ONNX Runtime session."""
        if self._runner is None:
            from face_engine.core.trt_compiler import aot_engine

            self._runner = aot_engine(self.model_path, self.precision, self.engine.config)
            if self._runner is None:
                logger.info("retinaface_r50: no compiled engine; using ONNX Runtime")
                self._runner = self.engine.get_session(self.model_path)
        return self._runner

    @property
    def uses_tensorrt_engine(self) -> bool:
        from face_engine.core.trt_compiler import TensorRTEngine

        return isinstance(self.runner, TensorRTEngine)

    def _priors_cuda(self, device: Any) -> Any:
        import torch

        key = str(device)
        if key not in self._priors:
            self._priors[key] = torch.as_tensor(priors(), device=device)
        return self._priors[key]

    def blob_cuda(self, frames: Any) -> tuple[Any, Letterbox]:
        """``(B, 3, H, W)`` BGR -> ``(B, 3, 640, 640)`` letterboxed, mean-subtracted float32."""
        import torch

        canvas, info = letterbox_cuda(frames, INPUT_SIZE)
        mean = torch.tensor(MEAN_BGR, device=canvas.device).view(1, 3, 1, 1)
        return (canvas - mean).contiguous(), info

    def decode_cuda(self, loc: Any, conf: Any, landms: Any,
                    info: Letterbox) -> tuple[Any, Any, Any]:
        """Every anchor of ``(B, N, 4/2/10)`` outputs -> frame-space ``(boxes, kps, scores)``."""
        import torch

        h = w = INPUT_SIZE  # decode on the canvas, then undo the letterbox
        p = self._priors_cuda(loc.device)
        loc, landms = loc.float(), landms.float()
        centers = p[:, :2] + loc[..., :2] * VARIANCE[0] * p[:, 2:]
        sizes = p[:, 2:] * torch.exp(loc[..., 2:] * VARIANCE[1])
        scale = torch.tensor([w, h], dtype=torch.float32, device=loc.device)
        boxes = torch.cat([(centers - sizes / 2) * scale, (centers + sizes / 2) * scale], -1)
        pts = p[:, None, :2] + landms.reshape(*landms.shape[:2], 5, 2) * VARIANCE[0] * p[:, None, 2:]
        if self._softmaxed is None:  # one host read, once: is the softmax in the graph?
            self._softmaxed = bool((conf[0, 0].float().sum() - 1.0).abs() < 1e-3)
            if not self._softmaxed:
                logger.info("retinaface_r50 export without an in-graph softmax: applying one")
        scores = conf.float()[..., 1] if self._softmaxed else conf.float().softmax(-1)[..., 1]
        boxes = _to_frame_cuda(boxes.reshape(*boxes.shape[:-1], 2, 2), info).flatten(-2)
        return boxes, _to_frame_cuda(pts * scale, info), scores

    def detect_cuda(self, frames: Any) -> GPUDetections:
        import torch

        if frames.ndim == 3:
            frames = frames[None]
        b, _, h, w = frames.shape
        runner = self.runner
        blob, info = self.blob_cuda(frames)
        step = getattr(runner, "max_batch", 1)
        n = self._priors_cuda(frames.device).shape[0]
        names = runner.output_names
        loc, conf, landms = [], [], []
        for start in range(0, b, step):
            part = blob[start:start + step]
            k = part.shape[0]
            out = runner.run_binding({runner.input_names[0]: part}, output_shapes={
                names[0]: (k, n, 4), names[1]: (k, n, 2), names[2]: (k, n, 10)})
            loc.append(out[names[0]])
            conf.append(out[names[1]])
            landms.append(out[names[2]])
        boxes, kps, scores = self.decode_cuda(torch.cat(loc), torch.cat(conf), torch.cat(landms),
                                              info)
        index = torch.arange(b, device=frames.device).repeat_interleave(n)
        return self._select_cuda(boxes.reshape(-1, 4), kps.reshape(-1, 5, 2), scores.reshape(-1),
                                 index, (h, w), b, frames.device)

    def detect_batch(self, frames: list[np.ndarray | None]) -> list[list[Face]]:
        """Host frames through the GPU path (same decoding, one code path)."""
        import torch

        out: list[list[Face]] = []
        for frame in frames:
            if frame is None or frame.size == 0:
                out.append([])
                continue
            t = torch.from_numpy(np.ascontiguousarray(frame[..., :3])).cuda()
            out.append(self.detect_cuda(t.permute(2, 0, 1)[None]).to_faces()[0])
        return out


def strided_tracker(detector: BaseDetector, detection_stride: int = 3) -> Any:
    """R50 every ``detection_stride`` frames, landmark tracking by optical flow in between."""
    from face_engine.pipeline.tracker import StridedFaceTracker, TrackerConfig

    return StridedFaceTracker(detector, TrackerConfig(detection_stride=detection_stride))
