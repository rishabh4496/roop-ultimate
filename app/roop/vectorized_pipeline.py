"""Shared-memory video pipeline with batched CUDA transforms and NVENC output.

The pipeline has one bounded host boundary:

``NVDEC -> SharedMemoryFrameRing -> pinned/asynchronous upload -> CUDA -> TRT -> NVENC``

There are no image files, no per-frame FFmpeg processes, and no
``multiprocessing.Queue`` payloads.  The ring carries only fixed-size bytes and
small metadata records; a long-lived FFmpeg process receives packed BGR frames
on stdin and writes the final video through ``h264_nvenc`` or ``hevc_nvenc``.

The detector adapter is intentionally model-contract based.  The included
RetinaFace decoder covers the common ``loc/conf/landms`` output layout, while
SCRFD variants can provide a decoder callback because their output tensor names
and anchor layouts differ between exports.  Both paths call the strict
TensorRT engine in batches and feed the temporal :class:`FacePrepass`, so
tracking and alignment matrices remain compatible with the existing compositor.
"""

from __future__ import annotations
from roop.degrade import swallowed as _swallowed

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .env import env_bool, env_str
from .ffmpeg_path import NVENC_PRESET_DEFAULT, NVENC_PRESETS

# This pipeline is the STRICT path: it exists to prove the NVDEC -> CUDA -> TRT
# -> NVENC route runs end to end with no silent fallback, so both strict flags
# default ON here and a missing stage raises instead of degrading.
#
# `roop.optimized_prepass` reads the SAME ROOP_OPT_STRICT_TRT flag and
# deliberately defaults it OFF, because it is the general-purpose prepass and
# must still run on a batch-one detector. The difference is intentional; naming
# both defaults keeps it visible instead of buried in an inline string literal.
# See tests/test_env_flags.py.
STRICT_TRT_DEFAULT = True
STRICT_NVDEC_DEFAULT = True

from .hardware_streamer import (
    FramePacket,
    NvdecFrameSource,
    RingFrameProducer,
    SharedMemoryFrameRing,
)
from .optimized_prepass import FacePrepass, FrameAnalysis
from .optimized_processor import CudaAffineBatch, CudaFrameBridge
from .optimized_trt_engine import TensorRTEngine

try:
    import torch
    import torch.nn.functional as torch_functional
except Exception as _degrade_error:  # pragma: no cover - import-safe CPU test environments
    _swallowed("roop/vectorized_pipeline.py:60", _degrade_error, "fallback continued")
    torch = None  # type: ignore[assignment]
    torch_functional = None  # type: ignore[assignment]


UInt8Array = NDArray[np.uint8]
Float32Array = NDArray[np.float32]


def _ffmpeg_binary() -> str:
    """Resolve the repository FFmpeg binary without creating a file."""

    try:
        from roop.ffmpeg_path import ffmpeg_binary

        return str(ffmpeg_binary())
    except Exception as _degrade_error:
        _swallowed("roop/vectorized_pipeline.py:76", _degrade_error, "fallback continued")
        return shutil.which("ffmpeg") or "ffmpeg"


def _popen_kwargs() -> Dict[str, Any]:
    """Use a hidden child process on Windows."""

    return {"creationflags": 0x08000000} if os.name == "nt" else {}


def _positive_int(value: Any, default: int) -> int:
    """Parse a positive integer for bounded allocations."""

    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(default))


@dataclass(frozen=True)
class PipelineConfig:
    """Transport, batching, and encoder settings for both GPU tiers."""

    batch_size: int = 8
    ring_capacity: int = 4
    device_id: int = 0
    strict_nvdec: bool = True
    strict_trt: bool = True
    encoder: str = "h264_nvenc"
    encoder_preset: str = "p4"
    quality: int = 18
    detection_batch_size: int = 8
    colorspace: str = "bt709"

    @classmethod
    def from_environment(cls, device_id: int = 0) -> "PipelineConfig":
        """Select bounded defaults for the RTX 4070 and RTX 3060 profiles."""

        total_gb = 0.0
        try:
            if torch is not None and torch.cuda.is_available():
                total_gb = float(
                    torch.cuda.get_device_properties(int(device_id)).total_memory
                ) / float(1024**3)
        except Exception as _degrade_error:
            _swallowed("roop/vectorized_pipeline.py:120", _degrade_error, "fallback continued")
            total_gb = 0.0
        laptop = 0.0 < total_gb < 7.0
        default_batch = 2 if laptop else 8
        default_ring = 2 if laptop else 4
        preset = env_str("ROOP_NVENC_PRESET", NVENC_PRESET_DEFAULT)
        if preset not in NVENC_PRESETS:
            preset = NVENC_PRESET_DEFAULT
        return cls(
            batch_size=_positive_int(
                os.environ.get("ROOP_PIPELINE_BATCH", default_batch), default_batch
            ),
            ring_capacity=_positive_int(
                os.environ.get("ROOP_PIPELINE_RING", default_ring), default_ring
            ),
            device_id=int(device_id),
            strict_nvdec=env_bool("ROOP_STRICT_NVDEC", STRICT_NVDEC_DEFAULT),
            strict_trt=env_bool("ROOP_OPT_STRICT_TRT", STRICT_TRT_DEFAULT),
            encoder=os.environ.get("ROOP_NVENC_CODEC", "h264_nvenc"),
            encoder_preset=preset,
            quality=_positive_int(os.environ.get("ROOP_NVENC_CQ", 18), 18),
            detection_batch_size=_positive_int(
                os.environ.get("ROOP_DETECTION_BATCH", default_batch), default_batch
            ),
            colorspace=os.environ.get("ROOP_FFMPEG_COLORSPACE", "bt709"),
        )


@dataclass
class PipelineStats:
    """End-to-end counters collected without a sampling profiler."""

    frames: int = 0
    detection_frames: int = 0
    elapsed_seconds: float = 0.0
    decode_seconds: float = 0.0
    prepass_seconds: float = 0.0
    gpu_seconds: float = 0.0
    encode_seconds: float = 0.0
    ring_wait_seconds: float = 0.0

    @property
    def fps(self) -> float:
        """End-to-end output FPS."""

        return self.frames / self.elapsed_seconds if self.elapsed_seconds > 0.0 else 0.0


class NvencRawWriter:
    """One persistent raw BGR pipe to an NVIDIA hardware encoder."""

    def __init__(
        self,
        output_path: str | os.PathLike[str],
        width: int,
        height: int,
        fps: float,
        *,
        codec: str = "h264_nvenc",
        preset: str = "p4",
        quality: int = 18,
        ffmpeg: Optional[str] = None,
        audio_source: Optional[str | os.PathLike[str]] = None,
        colorspace: str = "bt709",
    ) -> None:
        self.output_path = str(output_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.codec = str(codec)
        self.preset = str(preset).strip().lower()
        self.quality = max(0, min(51, int(quality)))
        self.ffmpeg = ffmpeg or _ffmpeg_binary()
        self.audio_source = None if audio_source is None else str(audio_source)
        self.colorspace = str(colorspace).strip().lower()
        self.process: Optional[subprocess.Popen[bytes]] = None
        self.frames_written = 0

    def _command(self) -> List[str]:
        """Build the rawvideo-to-NVENC command."""

        if self.codec not in {"h264_nvenc", "hevc_nvenc"}:
            raise ValueError("NvencRawWriter requires h264_nvenc or hevc_nvenc")
        if self.preset not in {f"p{index}" for index in range(1, 8)}:
            raise ValueError("NVENC preset must be p1 through p7")
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.12g}",
            "-i",
            "pipe:0",
        ]
        if self.audio_source:
            command.extend(["-i", self.audio_source])
        command.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?" if self.audio_source else "0:a:0?",
                "-c:v",
                self.codec,
                "-preset",
                self.preset,
                "-rc",
                "vbr",
                "-cq",
                str(self.quality),
                "-tune",
                "hq",
            ]
        )
        if self.audio_source:
            command.extend(["-c:a", "copy", "-shortest"])
        if self.colorspace not in {"off", "none", "passthrough", "0", "false"}:
            command.extend(
                [
                    "-colorspace",
                    "bt709",
                    "-color_primaries",
                    "bt709",
                    "-color_trc",
                    "bt709",
                    "-color_range",
                    "tv",
                ]
            )
        command.extend(["-pix_fmt", "yuv420p", self.output_path])
        return command

    def start(self) -> None:
        """Start the single persistent encoder process."""

        if self.process is not None:
            return
        self.process = subprocess.Popen(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=self.width * self.height * 3 * 2,
            **_popen_kwargs(),
        )

    def write(self, frame: UInt8Array) -> None:
        """Write one contiguous BGR frame without creating a bytes object."""

        array = np.asarray(frame)
        expected = (self.height, self.width, 3)
        if array.shape != expected or array.dtype != np.uint8:
            raise ValueError(
                f"NVENC expects uint8 {expected}, got {array.dtype} {array.shape}"
            )
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        if self.process is None:
            self.start()
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("NVENC process is not writable")
        if self.process.poll() is not None:
            detail = self._stderr_text()
            raise RuntimeError(f"NVENC exited before frame write: {detail}")
        try:
            self.process.stdin.write(memoryview(array))
            self.frames_written += 1
        except (BrokenPipeError, OSError) as error:
            detail = self._stderr_text()
            raise RuntimeError(f"NVENC write failed: {detail}") from error

    def _stderr_text(self) -> str:
        """Read currently available encoder diagnostics."""

        if self.process is None or self.process.stderr is None:
            return ""
        try:
            return self.process.stderr.read().decode("utf-8", "replace").strip()
        except OSError:
            return ""

    def close(self, discard: bool = False) -> None:
        """Close stdin, wait for the final file, and surface encoder errors."""

        process = self.process
        self.process = None
        if process is None:
            return
        if discard:
            try:
                process.terminate()
                process.wait(timeout=3.0)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                except OSError:
                    pass
            if process.stderr is not None:
                process.stderr.close()
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=120.0)
        except (OSError, subprocess.TimeoutExpired) as error:
            try:
                process.kill()
            except OSError:
                pass
            raise RuntimeError("NVENC process did not shut down") from error
        detail = b""
        if process.stderr is not None:
            try:
                detail = process.stderr.read() or b""
                process.stderr.close()
            except OSError:
                pass
        if process.returncode != 0:
            raise RuntimeError(
                "NVENC process failed: " + detail.decode("utf-8", "replace").strip()
            )

    def __enter__(self) -> "NvencRawWriter":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close(discard=exc_type is not None)


class GpuFrameBatch:
    """Upload packed host BGR frames once and expose normalized CUDA NCHW."""

    def __init__(self, device_id: int = 0, pin_memory: bool = True) -> None:
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("CUDA-enabled PyTorch is required for GpuFrameBatch")
        self.device_id = int(device_id)
        self.device = torch.device(f"cuda:{self.device_id}")
        self.pin_memory = bool(pin_memory)

    def upload(self, frames: Sequence[UInt8Array]) -> Any:
        """Transfer one contiguous host batch and normalize on CUDA."""

        if not frames:
            raise ValueError("cannot upload an empty frame batch")
        host = np.ascontiguousarray(np.stack(frames, axis=0), dtype=np.uint8)
        cpu = torch.from_numpy(host)
        if self.pin_memory:
            cpu = cpu.pin_memory()
        with torch.cuda.device(self.device_id):
            return (
                cpu.to(self.device, non_blocking=self.pin_memory)
                .permute(0, 3, 1, 2)
                .contiguous()
                .to(dtype=torch.float32)
                .div_(255.0)
            )

    def download(self, frames: Any) -> List[UInt8Array]:
        """Download one final batch at the single NVENC host boundary."""

        if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
            raise ValueError("download expects a CUDA NCHW tensor")
        with torch.cuda.device(self.device_id):
            host = (
                frames.detach()
                .clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .to(dtype=torch.uint8)
                .permute(0, 2, 3, 1)
                .contiguous()
                .to("cpu", non_blocking=self.pin_memory)
            )
            torch.cuda.current_stream(self.device_id).synchronize()
        array = np.ascontiguousarray(host.numpy())
        return [array[index] for index in range(int(array.shape[0]))]


class FixedImagePreprocessor:
    """GPU resize/normalize adapter for a detector input contract."""

    def __init__(
        self,
        size: Tuple[int, int] = (640, 640),
        *,
        device_id: int = 0,
        rgb: bool = True,
        mean: Sequence[float] = (0.0, 0.0, 0.0),
        standard_deviation: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA-enabled PyTorch is required for detector preprocessing"
            )
        if len(mean) != 3 or len(standard_deviation) != 3:
            raise ValueError(
                "detector mean and standard deviation require three values"
            )
        self.height, self.width = int(size[0]), int(size[1])
        self.device_id = int(device_id)
        self.device = torch.device(f"cuda:{self.device_id}")
        self.rgb = bool(rgb)
        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device).reshape(
            1, 3, 1, 1
        )
        self.std = torch.tensor(
            standard_deviation, dtype=torch.float32, device=self.device
        ).reshape(1, 3, 1, 1)
        if bool(torch.any(self.std == 0)):
            raise ValueError("detector standard deviation cannot contain zero")

    def __call__(self, frames: Sequence[UInt8Array]) -> Any:
        """Return one fixed-shape CUDA NCHW batch."""

        bridge = GpuFrameBatch(self.device_id)
        batch = bridge.upload(frames)
        if self.rgb:
            batch = batch[:, [2, 1, 0], :, :]
        batch = torch_functional.interpolate(
            batch,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )
        return (batch - self.mean) / self.std


DetectionDecoder = Callable[
    [Mapping[str, Any], Sequence[UInt8Array]], Sequence[Sequence[Any]]
]


class TensorRTDetectionAdapter:
    """Batch a detector engine and decode one result sequence per frame."""

    def __init__(
        self,
        engine: TensorRTEngine,
        input_name: str,
        preprocess: Callable[[Sequence[UInt8Array]], Any],
        postprocess: DetectionDecoder,
        batch_size: int = 8,
    ) -> None:
        self.engine = engine
        self.input_name = str(input_name)
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.batch_size = max(2, int(batch_size))

    def detect_batch(self, frames: Sequence[UInt8Array]) -> List[Sequence[Any]]:
        """Execute one or more governed dynamic batches, never a CPU fallback."""

        if not frames:
            return []
        inputs = self.preprocess(frames)
        if not isinstance(inputs, torch.Tensor) or int(inputs.shape[0]) != len(frames):
            raise ValueError("detector preprocessing returned the wrong batch size")
        outputs = self.engine.run_batched(
            {self.input_name: inputs}, requested_batch=self.batch_size
        )
        decoded = self.postprocess(outputs, frames)
        result = [list(item or []) for item in decoded]
        if len(result) != len(frames):
            raise ValueError("detector decoder returned the wrong frame count")
        return result


def _nms(boxes: Any, scores: Any, threshold: float) -> Any:
    """Small CUDA NMS implementation for RetinaFace candidate boxes."""

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    kept: List[Any] = []
    while int(order.numel()) > 0:
        current = order[0]
        kept.append(current)
        if int(order.numel()) == 1:
            break
        rest = order[1:]
        box = boxes[current]
        xx1 = torch.maximum(box[0], boxes[rest, 0])
        yy1 = torch.maximum(box[1], boxes[rest, 1])
        xx2 = torch.minimum(box[2], boxes[rest, 2])
        yy2 = torch.minimum(box[3], boxes[rest, 3])
        intersection = (xx2 - xx1).clamp_min(0.0) * (yy2 - yy1).clamp_min(0.0)
        area_current = (box[2] - box[0]).clamp_min(0.0) * (box[3] - box[1]).clamp_min(
            0.0
        )
        area_rest = (boxes[rest, 2] - boxes[rest, 0]).clamp_min(0.0) * (
            boxes[rest, 3] - boxes[rest, 1]
        ).clamp_min(0.0)
        iou = intersection / (area_current + area_rest - intersection).clamp_min(1e-7)
        order = rest[iou <= float(threshold)]
    return (
        torch.stack(kept)
        if kept
        else torch.empty((0,), dtype=torch.long, device=boxes.device)
    )


class RetinaFaceDecoder:
    """Decode the standard RetinaFace ``loc/conf/landms`` output layout."""

    def __init__(
        self,
        input_size: Tuple[int, int] = (640, 640),
        *,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        top_k: int = 64,
        variance: Tuple[float, float] = (0.1, 0.2),
        loc_name: str = "loc",
        conf_name: str = "conf",
        landms_name: str = "landms",
    ) -> None:
        self.input_height, self.input_width = int(input_size[0]), int(input_size[1])
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = max(1, int(top_k))
        self.variance = tuple(float(value) for value in variance)
        self.loc_name = loc_name
        self.conf_name = conf_name
        self.landms_name = landms_name
        self._prior_cache: Dict[Tuple[str, int, int], Any] = {}

    def _priors(self, device: Any, height: int, width: int) -> Any:
        """Generate RetinaFace priors once per input geometry/device."""

        key = (str(device), int(height), int(width))
        cached = self._prior_cache.get(key)
        if cached is not None:
            return cached
        min_sizes = ((16, 32), (64, 128), (256, 512))
        strides = (8, 16, 32)
        rows: List[List[float]] = []
        for stride, sizes in zip(strides, min_sizes):
            for y in range(0, height, stride):
                for x in range(0, width, stride):
                    for size in sizes:
                        rows.append(
                            [
                                (x + stride * 0.5) / width,
                                (y + stride * 0.5) / height,
                                size / width,
                                size / height,
                            ]
                        )
        priors = torch.tensor(rows, dtype=torch.float32, device=device)
        self._prior_cache[key] = priors
        return priors

    def __call__(
        self,
        outputs: Mapping[str, Any],
        frames: Sequence[UInt8Array],
    ) -> List[Sequence[Any]]:
        """Decode detections, perform CUDA NMS, then copy only selected faces."""

        if (
            self.loc_name not in outputs
            or self.conf_name not in outputs
            or self.landms_name not in outputs
        ):
            raise ValueError(
                f"RetinaFace outputs must contain {self.loc_name!r}, "
                f"{self.conf_name!r}, and {self.landms_name!r}; got {tuple(outputs)}"
            )
        loc = outputs[self.loc_name]
        conf = outputs[self.conf_name]
        landms = outputs[self.landms_name]
        if loc.ndim != 3 or conf.ndim != 3 or landms.ndim != 3:
            raise ValueError("RetinaFace outputs must be [B,N,C]")
        priors = self._priors(loc.device, self.input_height, self.input_width)
        if int(priors.shape[0]) != int(loc.shape[1]):
            raise ValueError(
                f"RetinaFace prior count {int(priors.shape[0])} does not match "
                f"model output {int(loc.shape[1])}; pass matching input geometry"
            )
        decoded_boxes = torch.cat(
            (
                priors[:, :2] + loc[..., :2] * self.variance[0] * priors[:, 2:],
                priors[:, 2:] * torch.exp(loc[..., 2:] * self.variance[1]),
            ),
            dim=-1,
        )
        decoded_boxes = torch.cat(
            (
                decoded_boxes[..., :2] - decoded_boxes[..., 2:] * 0.5,
                decoded_boxes[..., :2] + decoded_boxes[..., 2:] * 0.5,
            ),
            dim=-1,
        )
        decoded_landmarks = []
        for point in range(5):
            start = point * 2
            decoded_landmarks.append(
                priors[:, :2]
                + landms[..., start : start + 2] * self.variance[0] * priors[:, 2:]
            )
        decoded_landmarks = torch.stack(decoded_landmarks, dim=2)
        if int(conf.shape[-1]) > 1:
            minimum = float(conf.detach().amin().item())
            maximum = float(conf.detach().amax().item())
            scores = (
                torch.softmax(conf, dim=-1)[..., 1]
                if minimum < 0.0 or maximum > 1.0
                else conf[..., 1]
            )
        else:
            scores = conf[..., 0].sigmoid()
        result: List[Sequence[Any]] = []
        scale = torch.tensor(
            [self.input_width, self.input_height, self.input_width, self.input_height],
            dtype=loc.dtype,
            device=loc.device,
        )
        for index, frame in enumerate(frames):
            candidates = torch.nonzero(
                scores[index] >= self.confidence_threshold
            ).reshape(-1)
            if int(candidates.numel()) == 0:
                result.append([])
                continue
            if int(candidates.numel()) > self.top_k * 8:
                top = torch.topk(scores[index, candidates], self.top_k * 8).indices
                candidates = candidates[top]
            boxes = decoded_boxes[index, candidates] * scale
            keep = _nms(boxes, scores[index, candidates], self.nms_threshold)
            keep = keep[: self.top_k]
            boxes = boxes[keep]
            landmarks = decoded_landmarks[index, candidates][keep] * torch.tensor(
                [self.input_width, self.input_height],
                dtype=loc.dtype,
                device=loc.device,
            )
            source_height, source_width = frame.shape[:2]
            box_scale = torch.tensor(
                [
                    source_width / self.input_width,
                    source_height / self.input_height,
                    source_width / self.input_width,
                    source_height / self.input_height,
                ],
                dtype=loc.dtype,
                device=loc.device,
            )
            point_scale = torch.tensor(
                [source_width / self.input_width, source_height / self.input_height],
                dtype=loc.dtype,
                device=loc.device,
            )
            boxes_cpu = (boxes * box_scale).detach().cpu().numpy()
            landmarks_cpu = (landmarks * point_scale).detach().cpu().numpy()
            scores_cpu = scores[index, candidates][keep].detach().cpu().numpy()
            result.append(
                [
                    {
                        "bbox": np.asarray(boxes_cpu[row], dtype=np.float32),
                        "kps": np.asarray(landmarks_cpu[row], dtype=np.float32),
                        "det_score": float(scores_cpu[row]),
                    }
                    for row in range(len(boxes_cpu))
                ]
            )
        return result


class GpuRestorerStage:
    """Batch a face restorer between swap output and reverse compositing."""

    def __init__(
        self,
        engine: TensorRTEngine,
        input_size: int,
        *,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
        mean: Sequence[float] = (0.0, 0.0, 0.0),
        standard_deviation: Sequence[float] = (1.0, 1.0, 1.0),
        denormalize: bool = False,
    ) -> None:
        if torch is None or torch_functional is None:
            raise RuntimeError("PyTorch is required for GpuRestorerStage")
        self.engine = engine
        self.input_size = int(input_size)
        self.input_name = input_name or engine.input_names[0]
        self.output_name = output_name or engine.output_names[0]
        self.mean = torch.tensor(
            mean, dtype=torch.float32, device=engine.device
        ).reshape(1, 3, 1, 1)
        self.std = torch.tensor(
            standard_deviation, dtype=torch.float32, device=engine.device
        ).reshape(1, 3, 1, 1)
        if bool(torch.any(self.std == 0)):
            raise ValueError("restorer standard deviation cannot contain zero")
        self.denormalize = bool(denormalize)

    def __call__(self, faces: Any) -> Any:
        """Restore one CUDA face batch with governed TRT execution."""

        height, width = int(faces.shape[2]), int(faces.shape[3])
        model_input = torch_functional.interpolate(
            faces,
            size=(self.input_size, self.input_size),
            mode="bicubic",
            align_corners=False,
        )
        output = self.engine.run_batched(
            {self.input_name: (model_input - self.mean) / self.std}
        )[self.output_name]
        if output.ndim != 4:
            raise RuntimeError("restorer output must be NCHW")
        if self.denormalize:
            output = (output + 1.0) / 2.0
        if tuple(output.shape[2:]) != (height, width):
            output = torch_functional.interpolate(
                output, size=(height, width), mode="bicubic", align_corners=False
            )
        return output.clamp(0.0, 1.0)


class GpuFaceSwapStage:
    """Align, swap, restore, reverse-warp, and blend all faces on CUDA."""

    def __init__(
        self,
        engine: TensorRTEngine,
        source_embedding: Any,
        input_size: int,
        *,
        target_input: str = "target",
        source_input: str = "source",
        output_name: Optional[str] = None,
        mask_output_name: Optional[str] = None,
        channel_order: str = "rgb",
        mean: Sequence[float] = (0.0, 0.0, 0.0),
        standard_deviation: Sequence[float] = (1.0, 1.0, 1.0),
        denormalize: bool = False,
        blend_ratio: float = 0.85,
        restorer: Optional[GpuRestorerStage] = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for GpuFaceSwapStage")
        self.engine = engine
        self.device_id = int(engine.device_id)
        self.input_size = int(input_size)
        self.target_input = str(target_input)
        self.source_input = str(source_input)
        self.output_name = output_name or engine.output_names[0]
        self.mask_output_name = mask_output_name
        self.channel_order = str(channel_order).strip().lower()
        if self.channel_order not in {"rgb", "bgr"}:
            raise ValueError("channel_order must be rgb or bgr")
        self.mean = torch.tensor(
            mean, dtype=torch.float32, device=engine.device
        ).reshape(1, 3, 1, 1)
        self.std = torch.tensor(
            standard_deviation, dtype=torch.float32, device=engine.device
        ).reshape(1, 3, 1, 1)
        if bool(torch.any(self.std == 0)):
            raise ValueError("swap standard deviation cannot contain zero")
        embedding = source_embedding
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(
                embedding, dtype=torch.float32, device=engine.device
            )
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        if embedding.ndim != 2:
            raise ValueError("source_embedding must be [1, embedding_dim]")
        self.source_embedding = embedding.to(engine.device).contiguous()
        self.denormalize = bool(denormalize)
        self.blend_ratio = max(0.0, min(1.0, float(blend_ratio)))
        self.restorer = restorer
        self.bridge = CudaFrameBridge(device_id=self.device_id)

    def __call__(
        self,
        frames: Sequence[UInt8Array],
        analyses: Sequence[FrameAnalysis],
    ) -> List[UInt8Array]:
        """Process all faces in a frame batch with one or more TRT chunks."""

        if len(frames) != len(analyses):
            raise ValueError("frames and analyses must have equal lengths")
        gpu_frames = self.bridge.upload_bgr(frames)
        frame_ids: List[int] = []
        matrices: List[Float32Array] = []
        for frame_id, analysis in enumerate(analyses):
            for face in analysis.faces:
                frame_ids.append(frame_id)
                matrices.append(face.matrix)
        if not matrices:
            return [np.ascontiguousarray(frame) for frame in frames]
        matrix_tensor = torch.as_tensor(
            np.ascontiguousarray(np.stack(matrices), dtype=np.float32),
            device=gpu_frames.device,
        )
        frame_index_tensor = torch.as_tensor(
            frame_ids, dtype=torch.long, device=gpu_frames.device
        )
        face_frames = gpu_frames.index_select(0, frame_index_tensor)
        aligned = CudaAffineBatch.warp_frames(
            face_frames, matrix_tensor, self.input_size
        )
        if self.channel_order == "rgb":
            aligned = aligned[:, [2, 1, 0], :, :]
        aligned = (aligned - self.mean) / self.std
        source = self.source_embedding.expand(int(aligned.shape[0]), -1).contiguous()
        outputs = self.engine.run_batched(
            {self.target_input: aligned, self.source_input: source}
        )
        swapped = outputs[self.output_name]
        if swapped.ndim != 4:
            raise RuntimeError("swap output must be NCHW")
        if self.denormalize:
            swapped = (swapped + 1.0) / 2.0
        if self.channel_order == "rgb":
            swapped = swapped[:, [2, 1, 0], :, :]
        if self.restorer is not None:
            swapped = self.restorer(swapped)
        masks = outputs.get(self.mask_output_name) if self.mask_output_name else None
        composite = CudaAffineBatch.paste_faces(
            gpu_frames,
            swapped.clamp(0.0, 1.0),
            frame_index_tensor,
            matrix_tensor,
            masks=masks,
            blend_ratio=self.blend_ratio,
        )
        return self.bridge.download_bgr(composite)


class VectorizedPipeline:
    """Consume the shared ring, batch temporal prepass/GPU work, and encode."""

    def __init__(
        self,
        source: Any,
        prepass: Optional[FacePrepass],
        *,
        face_stage: Optional[
            Callable[
                [Sequence[UInt8Array], Sequence[FrameAnalysis]], Sequence[UInt8Array]
            ]
        ] = None,
        config: Optional[PipelineConfig] = None,
        ring: Optional[SharedMemoryFrameRing] = None,
        ffmpeg: Optional[str] = None,
        audio_source: Optional[str | os.PathLike[str]] = None,
    ) -> None:
        self.source = source
        self.prepass = prepass
        self.face_stage = face_stage
        self.config = config or PipelineConfig.from_environment()
        self.ffmpeg = ffmpeg
        self.audio_source = audio_source
        if (
            int(getattr(source, "width", 0)) <= 0
            or int(getattr(source, "height", 0)) <= 0
        ):
            raise ValueError("source must expose positive width and height")
        capacity = max(2, int(self.config.ring_capacity))
        self.ring = ring or SharedMemoryFrameRing.create(
            capacity,
            int(source.height),
            int(source.width),
            3,
        )
        if self.ring.frame_shape != (int(source.height), int(source.width), 3):
            raise ValueError("ring geometry does not match the decoder")
        self._owns_ring = ring is None
        self._host_batch = np.empty(
            (
                max(2, int(self.config.batch_size)),
                int(source.height),
                int(source.width),
                3,
            ),
            dtype=np.uint8,
        )
        self._producer: Optional[RingFrameProducer] = None
        self._failure: Optional[BaseException] = None

    def _collect_batch(self) -> List[FramePacket]:
        """Collect available packets without waiting for a full batch at EOF."""

        first = self.ring.read(destination=self._host_batch[0])
        if first is None:
            return []
        packets = [first]
        while len(packets) < max(2, int(self.config.batch_size)):
            try:
                next_packet = self.ring.read(
                    timeout=0.0,
                    destination=self._host_batch[len(packets)],
                )
            except TimeoutError:
                break
            if next_packet is None:
                break
            packets.append(next_packet)
        return packets

    def run(
        self,
        output_video: str | os.PathLike[str],
        *,
        audio_source: Optional[str | os.PathLike[str]] = None,
    ) -> PipelineStats:
        """Run the bounded pipeline and return measured stage timings."""

        if self.config.strict_trt and self.face_stage is None:
            raise RuntimeError("strict TensorRT pipeline requires a face_stage")
        started = time.perf_counter()
        stats = PipelineStats()
        self._failure = None
        writer = NvencRawWriter(
            output_video,
            int(self.source.width),
            int(self.source.height),
            float(self.source.fps),
            codec=self.config.encoder,
            preset=self.config.encoder_preset,
            quality=self.config.quality,
            ffmpeg=self.ffmpeg,
            audio_source=audio_source or self.audio_source,
            colorspace=self.config.colorspace,
        )
        self._producer = RingFrameProducer(
            self.source,
            self.ring,
            start_index=0,
        )
        discard_writer = False
        try:
            writer.start()
            self._producer.start()
            while True:
                packets = self._collect_batch()
                if not packets:
                    break
                frames = [packet.frame for packet in packets]
                first_index = packets[0].metadata.frame_index
                prepass_started = time.perf_counter()
                if self.prepass is None:
                    analyses = [
                        FrameAnalysis(index, [])
                        for index in range(first_index, first_index + len(frames))
                    ]
                else:
                    analyses = self.prepass.process_batch(
                        frames, start_index=first_index
                    )
                stats.prepass_seconds += time.perf_counter() - prepass_started
                stats.detection_frames += sum(
                    1 for item in analyses if item.detection_run
                )
                gpu_started = time.perf_counter()
                if self.face_stage is None:
                    outputs = frames
                else:
                    outputs = list(self.face_stage(frames, analyses))
                stats.gpu_seconds += time.perf_counter() - gpu_started
                if len(outputs) != len(frames):
                    raise ValueError("face_stage returned the wrong number of frames")
                for output in outputs:
                    encode_started = time.perf_counter()
                    writer.write(np.asarray(output, dtype=np.uint8))
                    stats.encode_seconds += time.perf_counter() - encode_started
                    stats.frames += 1
            self._producer.join(timeout=30.0)
            if self._producer.stats is not None:
                stats.decode_seconds = self._producer.stats.decode_seconds
                stats.ring_wait_seconds = self._producer.stats.ring_wait_seconds
            if self._producer.error is not None:
                raise RuntimeError("ring producer failed") from self._producer.error
        except BaseException as error:
            self._failure = error
            discard_writer = True
            if self._producer is not None:
                self._producer.stop()
            self.ring.abort()
            raise
        finally:
            try:
                writer.close(discard=discard_writer)
            except BaseException as error:
                if self._failure is None:
                    self._failure = error
                    raise
            if self._producer is not None and self._producer.is_alive():
                self._producer.stop()
                self._producer.join(timeout=5.0)
            self.ring.close(unlink=self._owns_ring)
            stats.elapsed_seconds = time.perf_counter() - started
        return stats

    def close(self, unlink_ring: bool = False) -> None:
        """Stop the producer and close ring mappings after a run."""

        if self._producer is not None:
            self._producer.stop()
            self._producer.join(timeout=5.0)
        self.ring.close(unlink=unlink_ring)


def build_nvdec_pipeline(
    input_video: str | os.PathLike[str],
    prepass: Optional[FacePrepass],
    *,
    face_stage: Optional[
        Callable[[Sequence[UInt8Array], Sequence[FrameAnalysis]], Sequence[UInt8Array]]
    ] = None,
    config: Optional[PipelineConfig] = None,
    device_id: int = 0,
    audio_source: Optional[str | os.PathLike[str]] = None,
) -> VectorizedPipeline:
    """Construct the standard strict NVDEC-to-NVENC pipeline."""

    selected = config or PipelineConfig.from_environment(device_id=device_id)
    source = NvdecFrameSource(
        input_video,
        device_id=selected.device_id,
        strict_nvdec=selected.strict_nvdec,
    )
    return VectorizedPipeline(
        source,
        prepass,
        face_stage=face_stage,
        config=selected,
        audio_source=audio_source,
    )


__all__ = [
    "FixedImagePreprocessor",
    "GpuFaceSwapStage",
    "GpuFrameBatch",
    "GpuRestorerStage",
    "NvencRawWriter",
    "PipelineConfig",
    "PipelineStats",
    "RetinaFaceDecoder",
    "TensorRTDetectionAdapter",
    "VectorizedPipeline",
    "build_nvdec_pipeline",
]
