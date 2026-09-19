"""Bounded memory video processing and adaptive ONNX batching.

This module separates transport from the repository's quality-sensitive face
compositor.  Frames are decoded from an FFmpeg raw-video pipe, held only in
bounded queues, analyzed by :class:`FacePrepass`, transformed by an injected
frame or batch callback, and written to one long-lived FFmpeg pipe.  There is
no extracted-frame directory and no per-frame FFmpeg process.

``OnnxBatchRunner`` is intentionally independent of the face-swap model.  The
repository contains several model families with different input contracts and
some exports are permanently batch-one.  The runner therefore tries the
requested batch, halves it after an OOM/shape failure, and converges to batch
one without changing the caller's numerical path.  CUDA I/O binding keeps
inputs and execution on the device until the single output transfer required
by a CPU compositor.
"""

from __future__ import annotations
from roop.degrade import swallowed as _swallowed

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from .optimized_prepass import FacePrepass, FrameAnalysis
from .trt_session_builder import (
    TensorRTSessionConfig,
    assert_strict_tensorrt_session,
    build_tensorrt_session,
    prepare_tensorrt_runtime,
)


Float32Array = NDArray[np.float32]
UInt8Array = NDArray[np.uint8]

try:
    # Import Torch before ONNX Runtime.  On Windows, loading ORT's CUDA DLLs
    # first can make a later Torch import fail with error 127 even though both
    # packages are individually installed and the CUDA provider is usable.
    import torch as _torch_module
except Exception as _degrade_error:  # pragma: no cover - CPU-only/minimal installations
    _swallowed("roop/optimized_processor.py:53", _degrade_error, "fallback continued")
    _torch_module = None

prepare_tensorrt_runtime()

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - exercised on CPU-only minimal installs
    ort = None  # type: ignore[assignment]


def _truthy(value: Any, default: bool = False) -> bool:
    """Parse the environment-style boolean values used by this project."""

    if value is None:
        return default
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")


def _positive_int(value: Any, default: int) -> int:
    """Parse a positive integer without allowing an invalid queue size."""

    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(default))


def _ffmpeg_binary() -> str:
    """Resolve FFmpeg from the repository helper, then from PATH."""

    try:
        from roop.ffmpeg_path import ffmpeg_binary

        return str(ffmpeg_binary())
    except Exception as _degrade_error:
        _swallowed("roop/optimized_processor.py:88", _degrade_error, "fallback continued")
        return shutil.which("ffmpeg") or "ffmpeg"


def _ffprobe_binary(ffmpeg: str) -> str:
    """Find the ffprobe next to a configured FFmpeg executable."""

    candidate = Path(ffmpeg)
    name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    sibling = candidate.with_name(name)
    return str(sibling) if sibling.exists() else (shutil.which(name) or name)


@dataclass(frozen=True)
class VideoSpec:
    """Raw BGR video geometry used by both pipe endpoints."""

    width: int
    height: int
    fps: float
    frame_count: Optional[int] = None

    @property
    def frame_bytes(self) -> int:
        """Bytes in one packed BGR24 frame."""

        return int(self.width) * int(self.height) * 3


def probe_video(path: str | os.PathLike[str], ffmpeg: Optional[str] = None) -> VideoSpec:
    """Read video metadata without decoding frames or creating intermediates."""

    filename = str(path)
    ffmpeg_path = ffmpeg or _ffmpeg_binary()
    ffprobe = _ffprobe_binary(ffmpeg_path)
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames",
        "-of",
        "json",
        filename,
    ]
    try:
        raw = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
        stream = json.loads(raw)["streams"][0]
        rate = Fraction(str(stream.get("r_frame_rate", "0/1")))
        fps = float(rate) if rate.denominator else 0.0
        count_value = stream.get("nb_frames")
        frame_count = int(count_value) if count_value not in (None, "N/A") else None
        return VideoSpec(
            width=int(stream["width"]),
            height=int(stream["height"]),
            fps=fps if fps > 0.0 else 30.0,
            frame_count=frame_count,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        capture = cv2.VideoCapture(filename)
        try:
            if not capture.isOpened():
                raise RuntimeError(f"cannot open video: {filename}")
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            return VideoSpec(width, height, fps if fps > 0 else 30.0, count or None)
        finally:
            capture.release()


class FFmpegRawReader:
    """Sequential BGR24 reader backed by one FFmpeg child process."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        spec: Optional[VideoSpec] = None,
        ffmpeg: Optional[str] = None,
        hwaccel: Optional[str] = None,
        start_frame: int = 0,
    ) -> None:
        self.path = str(path)
        self.spec = spec or probe_video(self.path, ffmpeg=ffmpeg)
        self.ffmpeg = ffmpeg or _ffmpeg_binary()
        self.hwaccel = hwaccel
        self.start_frame = max(0, int(start_frame))
        self.process: Optional[subprocess.Popen[bytes]] = None
        self._closed = False

    def start(self) -> None:
        """Spawn the decoder and configure one rawvideo stdout stream."""

        if self.process is not None:
            return
        command: List[str] = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
        if self.hwaccel:
            command.extend(["-hwaccel", self.hwaccel])
        if self.start_frame and self.spec.fps > 0.0:
            timestamp = max(0.0, (self.start_frame - 0.5) / self.spec.fps)
            command.extend(["-ss", f"{timestamp:.6f}"])
        command.extend(
            [
                "-noautorotate",
                "-i",
                self.path,
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-an",
                "-sn",
                "pipe:1",
            ]
        )
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self.spec.frame_bytes * 2,
            creationflags=0x08000000 if os.name == "nt" else 0,
        )
        self._closed = False

    def _read_exact(self, size: int) -> Optional[bytes]:
        """Read one complete frame, distinguishing EOF from a short frame."""

        if self.process is None or self.process.stdout is None:
            return None
        output = bytearray(size)
        view = memoryview(output)
        offset = 0
        while offset < size:
            chunk = self.process.stdout.read(size - offset)
            if not chunk:
                if offset == 0:
                    return None
                raise RuntimeError(
                    f"FFmpeg ended with a partial raw frame ({offset}/{size} bytes)"
                )
            view[offset : offset + len(chunk)] = chunk
            offset += len(chunk)
        return bytes(output)

    def read(self) -> Optional[UInt8Array]:
        """Read the next BGR frame or return ``None`` at clean EOF."""

        if self.process is None:
            self.start()
        if self._closed:
            return None
        payload = self._read_exact(self.spec.frame_bytes)
        if payload is None:
            return None
        return np.frombuffer(payload, dtype=np.uint8).reshape(
            self.spec.height, self.spec.width, 3
        )

    def __iter__(self) -> Iterable[UInt8Array]:
        """Iterate frames until EOF."""

        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame

    def close(self, ignore_errors: bool = False) -> None:
        """Stop the child process and optionally ignore planned pipe breaks."""

        if self.process is None:
            return
        process = self.process
        self._closed = True
        try:
            if process.stdout is not None:
                process.stdout.close()
        except OSError:
            pass
        try:
            process.terminate()
            process.wait(timeout=3.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass
            process.wait(timeout=3.0)
        error = b""
        if process.stderr is not None:
            try:
                error = process.stderr.read() or b""
                process.stderr.close()
            except OSError:
                pass
        self.process = None
        if not ignore_errors and process.returncode not in (0, -15, 15, None) and error:
            raise RuntimeError(
                "FFmpeg decode failed: " + error.decode("utf-8", "replace").strip()
            )

    def __enter__(self) -> "FFmpegRawReader":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close(ignore_errors=exc_type is not None)


class FFmpegRawWriter:
    """One long-lived rawvideo-to-file FFmpeg encoder."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        spec: VideoSpec,
        codec: Optional[str] = None,
        crf: int = 14,
        preset: Optional[str] = None,
        audio_source: Optional[str | os.PathLike[str]] = None,
        ffmpeg: Optional[str] = None,
        colorspace: str = "bt709",
        fallback_codec: str = "libx264",
    ) -> None:
        self.path = str(path)
        self.spec = spec
        self.ffmpeg = ffmpeg or _ffmpeg_binary()
        self.codec = codec or os.environ.get("ROOP_OPT_ENCODER", "h264_nvenc")
        self.fallback_codec = fallback_codec
        self.crf = int(crf)
        self.preset = preset or os.environ.get("ROOP_OPT_ENCODER_PRESET", "p5")
        self.audio_source = None if audio_source is None else str(audio_source)
        self.colorspace = str(colorspace).strip().lower()
        self.process: Optional[subprocess.Popen[bytes]] = None
        self.frames_written = 0
        self._fallback_used = False

    def _command(self, codec: str) -> List[str]:
        """Build a command that accepts packed BGR frames on stdin."""

        width = self.spec.width - (self.spec.width % 2)
        height = self.spec.height - (self.spec.height % 2)
        command: List[str] = [
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
            f"{self.spec.width}x{self.spec.height}",
            "-r",
            str(self.spec.fps),
            "-i",
            "pipe:0",
        ]
        if self.audio_source:
            command.extend(["-i", self.audio_source, "-map", "0:v:0", "-map", "1:a:0?"])
        else:
            command.extend(["-an"])
        command.extend(["-c:v", codec])
        if codec.endswith("_nvenc"):
            command.extend(
                [
                    "-rc",
                    "vbr",
                    "-cq",
                    str(max(0, min(51, self.crf))),
                    "-preset",
                    self.preset if self.preset in {f"p{i}" for i in range(1, 8)} else "p5",
                    "-tune",
                    "hq",
                ]
            )
        elif codec in ("libx264", "libx265"):
            command.extend(
                [
                    "-crf",
                    str(max(0, min(51, self.crf))),
                    "-preset",
                    self.preset if self.preset in {
                        "ultrafast", "superfast", "veryfast", "faster", "fast",
                        "medium", "slow", "slower", "veryslow", "placebo",
                    } else "faster",
                ]
            )
        if width != self.spec.width or height != self.spec.height:
            filters = [f"scale={width}:{height}"]
        else:
            filters = []
        if self.colorspace not in ("off", "none", "passthrough", "0", "false"):
            filters.append("colorspace=bt709:iall=bt601-6-625:fast=1")
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
        if filters:
            command.extend(["-vf", ",".join(filters)])
        command.extend(["-pix_fmt", "yuv420p"])
        if self.audio_source:
            command.extend(["-c:a", "aac", "-b:a", "192k"])
        if self.path.lower().endswith((".mp4", ".mov", ".m4v")):
            command.extend(["-movflags", "+faststart"])
        command.append(self.path)
        return command

    def start(self, codec: Optional[str] = None) -> None:
        """Start the encoder process once."""

        if self.process is not None:
            return
        selected = codec or self.codec
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            self._command(selected),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=0x08000000 if os.name == "nt" else 0,
        )

    def _failed_before_first_frame(self) -> bool:
        """Allow a hardware encoder to fail over before output is committed."""

        return bool(
            self.process is not None
            and self.process.poll() is not None
            and self.frames_written == 0
            and not self._fallback_used
            and self.fallback_codec
            and self.codec != self.fallback_codec
        )

    def _restart_fallback(self) -> bool:
        """Replace a failed empty output with a software encoder."""

        if not self._failed_before_first_frame():
            return False
        self._fallback_used = True
        self._close_process(discard=True)
        try:
            Path(self.path).unlink(missing_ok=True)
        except OSError:
            pass
        self.codec = self.fallback_codec
        self.start(codec=self.codec)
        return True

    def write(self, frame: UInt8Array) -> None:
        """Write one contiguous BGR frame with bounded pipe backpressure."""

        array = np.asarray(frame)
        expected = (self.spec.height, self.spec.width, 3)
        if array.shape != expected or array.dtype != np.uint8:
            raise ValueError(f"expected uint8 BGR frame {expected}, got {array.shape} {array.dtype}")
        if self.process is None:
            self.start()
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("FFmpeg encoder did not start")
        if self.process.poll() is not None:
            if self._restart_fallback():
                return self.write(array)
            raise RuntimeError(f"FFmpeg encoder exited before frame {self.frames_written}")
        try:
            payload = np.ascontiguousarray(array)
            self.process.stdin.write(memoryview(payload))
            self.frames_written += 1
        except (BrokenPipeError, OSError) as exc:
            if self._restart_fallback():
                return self.write(array)
            raise RuntimeError(f"FFmpeg encoder pipe failed: {exc}") from exc

    def _close_process(self, discard: bool = False) -> None:
        """Close the child and optionally discard a failed partial file."""

        process = self.process
        if process is None:
            return
        error = b""
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
        if process.stderr is not None:
            try:
                error = process.stderr.read() or b""
                process.stderr.close()
            except OSError:
                pass
        self.process = None
        if process.returncode not in (0, None) and not discard:
            detail = error.decode("utf-8", "replace").strip()
            raise RuntimeError(
                f"FFmpeg encoder exited with code {process.returncode}: {detail}"
            )

    def close(self) -> None:
        """Finalize the output container."""

        self._close_process(discard=False)


class AsyncRawVideoWriter:
    """Bounded writer queue with explicit sentinel and failure propagation."""

    def __init__(self, writer: FFmpegRawWriter, queue_depth: int = 3) -> None:
        self.writer = writer
        self.queue: Queue[Optional[UInt8Array]] = Queue(maxsize=max(1, int(queue_depth)))
        self.stop_event = threading.Event()
        self.error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, name="optimized-video-writer", daemon=True)

    def start(self) -> None:
        """Start FFmpeg and the writer thread."""

        self.writer.start()
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                item = self.queue.get()
                if item is None:
                    break
                self.writer.write(item)
        except BaseException as exc:  # propagated to submit/close, never hidden
            _swallowed("roop/optimized_processor.py:532", exc, "fallback continued")
            self.error = exc
            self.stop_event.set()

    def submit(self, frame: UInt8Array) -> None:
        """Submit with backpressure, aborting promptly after a writer failure."""

        while True:
            if self.error is not None:
                raise RuntimeError("video writer failed") from self.error
            if self.stop_event.is_set():
                raise RuntimeError("video writer stopped")
            try:
                self.queue.put(frame, timeout=0.25)
                return
            except Full:
                continue

    def close(self, discard: bool = False) -> None:
        """Drain/finalize the writer without leaving a live queue consumer."""

        if not self._thread.is_alive():
            if self.error is not None:
                self.writer._close_process(discard=discard)
                raise RuntimeError("video writer failed") from self.error
            return
        if discard:
            while True:
                try:
                    self.queue.get_nowait()
                except Empty:
                    break
        while True:
            try:
                self.queue.put(None, timeout=0.25)
                break
            except Full:
                if self.error is not None:
                    break
        self._thread.join(timeout=60.0)
        if self._thread.is_alive():
            self.stop_event.set()
            raise RuntimeError("video writer did not shut down")
        try:
            self.writer.close()
        except BaseException as exc:
            _swallowed("roop/optimized_processor.py:577", exc, "fallback continued")
            if self.error is None:
                self.error = exc
        if self.error is not None:
            raise RuntimeError("video writer failed") from self.error


@dataclass
class VramGovernor:
    """Runtime batch governor for the two supported GPU tiers."""

    total_gb: float
    hard_cap_mb: int
    reserve_mb: int = 1024
    batch_limit: int = 1
    device_id: int = 0
    oom_events: int = 0

    @classmethod
    def from_environment(cls, device_id: int = 0, requested_batch: int = 8) -> "VramGovernor":
        """Select the conservative tier before a model is loaded."""

        total_gb = 0.0
        try:
            if _torch_module is not None and _torch_module.cuda.is_available():
                total_gb = float(
                    _torch_module.cuda.get_device_properties(device_id).total_memory
                ) / float(1024 ** 3)
        except Exception as _degrade_error:
            _swallowed("roop/optimized_processor.py:605", _degrade_error, "fallback continued")
            pass
        hard_cap = 1536 if 0.0 < total_gb < 7.0 else 4096
        configured_cap = os.environ.get("ROOP_OPT_GPU_CAP_MB")
        if configured_cap:
            try:
                hard_cap = max(512, min(hard_cap, int(configured_cap)))
            except ValueError:
                pass
        default_batch = 1 if 0.0 < total_gb < 7.0 else max(1, int(requested_batch))
        return cls(
            total_gb=total_gb,
            hard_cap_mb=hard_cap,
            reserve_mb=_positive_int(os.environ.get("ROOP_OPT_VRAM_RESERVE_MB", 1024), 1024),
            batch_limit=default_batch,
            device_id=int(device_id),
        )

    def free_mb(self) -> Optional[float]:
        """Read whole-device free memory, including ORT/TensorRT allocations."""

        try:
            if _torch_module is not None and _torch_module.cuda.is_available():
                free, _total = _torch_module.cuda.mem_get_info(self.device_id)
                return float(free) / float(1024 ** 2)
        except Exception as _degrade_error:
            _swallowed("roop/optimized_processor.py:630", _degrade_error, "fallback continued")
            pass
        return None

    def safe_batch(self, requested: int, per_item_mb: float = 96.0) -> int:
        """Return a batch ceiling with a reserve and hard-cap guard."""

        requested = max(1, int(requested))
        limit = min(requested, max(1, self.batch_limit))
        free = self.free_mb()
        if free is not None:
            usable = min(float(self.hard_cap_mb), max(0.0, free - self.reserve_mb))
            limit = min(limit, max(1, int(usable / max(1.0, per_item_mb))))
        return max(1, limit)

    def note_oom(self, failed_batch: int) -> int:
        """Halve future work after an OOM or shape-capacity failure."""

        self.oom_events += 1
        self.batch_limit = max(1, min(self.batch_limit, max(1, int(failed_batch) // 2)))
        return self.batch_limit


def _numpy_dtype(ort_type: str) -> np.dtype[Any]:
    """Map ORT tensor type strings to NumPy dtypes."""

    mapping: Dict[str, np.dtype[Any]] = {
        "tensor(float)": np.dtype(np.float32),
        "tensor(float16)": np.dtype(np.float16),
        "tensor(double)": np.dtype(np.float64),
        "tensor(int64)": np.dtype(np.int64),
        "tensor(int32)": np.dtype(np.int32),
        "tensor(int16)": np.dtype(np.int16),
        "tensor(int8)": np.dtype(np.int8),
        "tensor(uint8)": np.dtype(np.uint8),
        "tensor(bool)": np.dtype(np.bool_),
    }
    return mapping.get(str(ort_type), np.dtype(np.float32))


def _is_batch_axis_error(error: BaseException) -> bool:
    """Recognize errors that are safe to retry with a smaller batch."""

    message = str(error).lower()
    return any(
        token in message
        for token in (
            "out of memory",
            "outofmemory",
            "failed to allocate",
            "memory allocation",
            "cudaerror",
            "cudnn_status_alloc_failed",
            "shape",
            "dimension",
            "invalid argument",
            "reshape",
        )
    )


def _torch_dtype_for_ort(ort_type: str) -> Any:
    """Map an ONNX Runtime tensor type to a PyTorch dtype."""

    if _torch_module is None:
        raise RuntimeError("PyTorch is required for GPU TensorRT I/O binding")
    mapping = {
        "tensor(float)": _torch_module.float32,
        "tensor(float16)": _torch_module.float16,
        "tensor(double)": _torch_module.float64,
        "tensor(int64)": _torch_module.int64,
        "tensor(int32)": _torch_module.int32,
        "tensor(int16)": _torch_module.int16,
        "tensor(int8)": _torch_module.int8,
        "tensor(uint8)": _torch_module.uint8,
        "tensor(bool)": _torch_module.bool,
    }
    try:
        return mapping[str(ort_type)]
    except KeyError as error:
        raise TypeError(f"unsupported TensorRT output type {ort_type!r}") from error


def _numpy_dtype_for_torch(dtype: Any) -> np.dtype[Any]:
    """Map a CUDA tensor dtype to the dtype expected by ``bind_input``."""

    if _torch_module is None:
        raise RuntimeError("PyTorch is required for GPU TensorRT I/O binding")
    mapping = {
        _torch_module.float32: np.dtype(np.float32),
        _torch_module.float16: np.dtype(np.float16),
        _torch_module.float64: np.dtype(np.float64),
        _torch_module.int64: np.dtype(np.int64),
        _torch_module.int32: np.dtype(np.int32),
        _torch_module.int16: np.dtype(np.int16),
        _torch_module.int8: np.dtype(np.int8),
        _torch_module.uint8: np.dtype(np.uint8),
        _torch_module.bool: np.dtype(np.bool_),
    }
    try:
        return mapping[dtype]
    except KeyError as error:
        raise TypeError(f"unsupported CUDA input dtype {dtype!r}") from error


def _resolved_output_shape(
    output_shape: Sequence[Any],
    input_metas: Mapping[str, Any],
    feeds: Mapping[str, Any],
    override: Optional[Sequence[int]] = None,
) -> Tuple[int, ...]:
    """Resolve symbolic output dimensions from bound input dimensions."""

    if override is not None:
        resolved = tuple(int(value) for value in override)
        if any(value <= 0 for value in resolved):
            raise ValueError(f"output shape override must be positive: {resolved!r}")
        return resolved
    symbol_values: Dict[str, int] = {}
    first_batch: Optional[int] = None
    for name, meta in input_metas.items():
        tensor = feeds[name]
        if first_batch is None:
            first_batch = int(tensor.shape[0])
        for declared, actual in zip(getattr(meta, "shape", ()), tensor.shape):
            if isinstance(declared, str) and not declared.isdigit():
                symbol_values.setdefault(declared, int(actual))
    if first_batch is None:
        raise ValueError("at least one bound input is required")
    values: List[int] = []
    for dimension in output_shape:
        if isinstance(dimension, int) and dimension > 0:
            values.append(int(dimension))
            continue
        try:
            parsed = int(str(dimension))
        except (TypeError, ValueError):
            parsed = 0
        if parsed > 0:
            values.append(parsed)
            continue
        symbol = str(dimension)
        if symbol.lower() in {"b", "batch", "batch_size"}:
            values.append(first_batch)
        elif symbol in symbol_values:
            values.append(symbol_values[symbol])
        else:
            raise RuntimeError(
                f"cannot preallocate dynamic output dimension {symbol!r}; "
                "provide output_shapes to the strict TensorRT runner"
            )
    return tuple(values)


class CudaIOBinding:
    """Best-effort persistent CUDA input binding for one ORT session."""

    def __init__(self, session: Any, device_id: int = 0) -> None:
        self.session = session
        self.device_id = int(device_id)
        self.enabled = False
        self._buffers: Dict[Tuple[str, Tuple[int, ...], str], Any] = {}
        self._lock = threading.RLock()
        self._reported_failure = False
        self._disabled_reason: Optional[str] = None
        try:
            if _torch_module is None:
                raise ImportError("PyTorch is unavailable; CUDA I/O binding is disabled")
            torch = _torch_module
            providers = list(session.get_providers())
            self.enabled = bool(
                torch.cuda.is_available()
                and any("cuda" in name.lower() or "tensorrt" in name.lower() for name in providers)
            )
            self._torch = torch
        except Exception as error:
            _swallowed("roop/optimized_processor.py:805", error, "fallback continued")
            self._torch = None
            self._disabled_reason = repr(error)

    def run(self, feeds: Mapping[str, NDArray[Any]]) -> Optional[List[NDArray[Any]]]:
        """Run with device inputs and one final output copy to host arrays."""

        if not self.enabled or self._torch is None:
            return None
        try:
            with self._lock, self._torch.cuda.device(self.device_id):
                binding = self.session.io_binding()
                device_refs: List[Any] = []
                batch: Optional[int] = None
                input_meta = {meta.name: meta for meta in self.session.get_inputs()}
                for name, value in feeds.items():
                    array = np.ascontiguousarray(np.asarray(value))
                    if array.ndim == 0:
                        return None
                    if batch is None:
                        batch = int(array.shape[0])
                    if int(array.shape[0]) != batch:
                        return None
                    expected = _numpy_dtype(getattr(input_meta.get(name), "type", "tensor(float)"))
                    if array.dtype != expected:
                        array = np.ascontiguousarray(array.astype(expected, copy=False))
                    key = (name, tuple(int(item) for item in array.shape), array.dtype.str)
                    tensor = self._buffers.get(key)
                    if tensor is None:
                        tensor = self._torch.empty(
                            array.shape,
                            dtype=self._torch.from_numpy(array).dtype,
                            device=f"cuda:{self.device_id}",
                        )
                        self._buffers[key] = tensor
                    tensor.copy_(self._torch.from_numpy(array), non_blocking=True)
                    device_refs.append(tensor)
                    binding.bind_input(
                        name,
                        "cuda",
                        self.device_id,
                        array.dtype,
                        tuple(int(item) for item in array.shape),
                        int(tensor.data_ptr()),
                    )
                for output in self.session.get_outputs():
                    # ORT allocates dynamic outputs on CUDA.  Avoid guessing a
                    # shape for masks or model-family-specific auxiliary heads.
                    binding.bind_output(output.name, "cuda", self.device_id)
                self.session.run_with_iobinding(binding)
                outputs = binding.copy_outputs_to_cpu()
                del device_refs
                return [np.ascontiguousarray(output) for output in outputs]
        except Exception as error:
            if _is_batch_axis_error(error):
                # A model-family export may be permanently batch-one.  Keep
                # the binding alive so the governor can retry the next smaller
                # batch and recover the fast device path instead of falling
                # back to host copies for the remainder of the run.
                self._buffers.clear()
                return None
            self.enabled = False
            self._disabled_reason = repr(error)
            if not self._reported_failure:
                self._reported_failure = True
                print(f"[optimized-io-binding] disabled for session: {error}", flush=True)
            return None

    def run_gpu(
        self,
        feeds: Mapping[str, Any],
        output_shapes: Optional[Mapping[str, Sequence[int]]] = None,
    ) -> List[Any]:
        """Run with CUDA tensor inputs and CUDA tensor outputs only.

        Unlike :meth:`run`, this method never calls ``copy_outputs_to_cpu``
        and never retries through ``session.run``.  Output buffers are
        preallocated from the declared symbolic shapes so PyTorch can consume
        them without an intermediate host array.  Models with runtime-sized
        data-dependent outputs must provide ``output_shapes`` or use a
        detector-specific GPU postprocessor that binds its own output OrtValue.
        """

        if not self.enabled or self._torch is None:
            reason = self._disabled_reason or "CUDA I/O binding is unavailable"
            raise RuntimeError(reason)
        torch = self._torch
        if not feeds:
            raise ValueError("GPU ONNX feed is empty")
        output_overrides = dict(output_shapes or {})
        with self._lock, torch.cuda.device(self.device_id):
            binding = self.session.io_binding()
            input_metas = {meta.name: meta for meta in self.session.get_inputs()}
            missing = sorted(set(input_metas) - set(feeds))
            if missing:
                raise ValueError(f"strict TensorRT feed is missing inputs: {missing!r}")
            device_refs: List[Any] = []
            batch: Optional[int] = None
            for name, value in feeds.items():
                if name not in input_metas:
                    raise ValueError(f"unknown ONNX input {name!r}")
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"strict TensorRT input {name!r} must be a torch.Tensor"
                    )
                tensor = value
                if not tensor.is_cuda or tensor.device.index != self.device_id:
                    raise ValueError(
                        f"input {name!r} must be on cuda:{self.device_id}, got {tensor.device}"
                    )
                if tensor.ndim == 0:
                    raise ValueError(f"input {name!r} has no batch axis")
                if batch is None:
                    batch = int(tensor.shape[0])
                elif int(tensor.shape[0]) != batch:
                    raise ValueError("all strict TensorRT inputs must share batch size")
                expected = _torch_dtype_for_ort(getattr(input_metas[name], "type", "tensor(float)"))
                if tensor.dtype != expected:
                    tensor = tensor.to(dtype=expected)
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                device_refs.append(tensor)
                binding.bind_input(
                    name,
                    "cuda",
                    self.device_id,
                    _numpy_dtype_for_torch(tensor.dtype),
                    tuple(int(item) for item in tensor.shape),
                    int(tensor.data_ptr()),
                )
            if batch is None or batch <= 0:
                raise ValueError("strict TensorRT input batch is empty")

            output_tensors: List[Any] = []
            for output in self.session.get_outputs():
                shape = _resolved_output_shape(
                    getattr(output, "shape", ()),
                    input_metas,
                    feeds,
                    output_overrides.get(output.name),
                )
                tensor = torch.empty(
                    shape,
                    dtype=_torch_dtype_for_ort(getattr(output, "type", "tensor(float)")),
                    device=f"cuda:{self.device_id}",
                )
                output_tensors.append(tensor)
                binding.bind_output(
                    output.name,
                    "cuda",
                    self.device_id,
                    _numpy_dtype_for_torch(tensor.dtype),
                    shape,
                    int(tensor.data_ptr()),
                )
            self.session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            return output_tensors


def create_onnx_session(
    model_path: str | os.PathLike[str],
    providers: Optional[Sequence[Any]] = None,
) -> Any:
    """Create an ORT session without silently losing a usable CUDA provider.

    ORT can report TensorRT as available while a required ``nvinfer`` DLL is
    absent.  In that case constructing the full TensorRT/CUDA/CPU list may
    fall all the way back to CPU.  Retry the same model with TensorRT removed
    before accepting a CPU-only session.
    """

    if ort is None:
        raise ImportError("onnxruntime is required for OnnxBatchRunner")
    if providers is None:
        from roop.backend_manager import canonical_provider_decision
        requested: List[Any] = list(
            canonical_provider_decision("auto").active_chain
        )
    else:
        requested = list(providers)
    if not requested:
        raise RuntimeError("no ONNX Runtime execution provider is available")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = _positive_int(
        os.environ.get("ROOP_OPT_ORT_INTRA_THREADS", 1), 1
    )
    options.inter_op_num_threads = _positive_int(
        os.environ.get("ROOP_OPT_ORT_INTER_THREADS", 1), 1
    )
    candidates: List[List[Any]] = []

    def add_candidate(chain: Sequence[Any]) -> None:
        values = list(chain)
        names = tuple(
            str(item[0] if isinstance(item, (tuple, list)) else item)
            for item in values
        )
        if values and names not in {
            tuple(
                str(item[0] if isinstance(item, (tuple, list)) else item)
                for item in existing
            )
            for existing in candidates
        }:
            candidates.append(values)

    add_candidate(requested)
    without_tensor_rt = [
        item
        for item in requested
        if "tensorrt" not in str(item[0] if isinstance(item, (tuple, list)) else item).lower()
    ]
    add_candidate(without_tensor_rt)
    available_names = {
        str(item[0] if isinstance(item, (tuple, list)) else item)
        for item in requested
    }
    if "CUDAExecutionProvider" in available_names:
        add_candidate(["CUDAExecutionProvider", "CPUExecutionProvider"])
    add_candidate(["CPUExecutionProvider"])

    last_error: Optional[BaseException] = None
    for candidate in candidates:
        try:
            session = ort.InferenceSession(
                str(model_path), options, providers=candidate
            )
            active = [str(name) for name in session.get_providers()]
            requested_gpu = any(
                "cuda" in str(item[0] if isinstance(item, (tuple, list)) else item).lower()
                or "tensorrt"
                in str(item[0] if isinstance(item, (tuple, list)) else item).lower()
                for item in candidate
            )
            active_gpu = any(
                "cuda" in name.lower() or "tensorrt" in name.lower()
                for name in active
            )
            if requested_gpu and not active_gpu:
                last_error = RuntimeError(
                    f"requested GPU providers {candidate} were not active; got {active}"
                )
                del session
                continue
            return session
        except Exception as error:
            _swallowed("roop/optimized_processor.py:1058", error, "fallback continued")
            last_error = error
    raise RuntimeError(
        f"unable to create ONNX Runtime session for {model_path}"
    ) from last_error


def create_strict_tensorrt_session(
    model_path: str | os.PathLike[str],
    config: Optional[TensorRTSessionConfig] = None,
) -> Any:
    """Create a TensorRT-only session and reject every provider fallback."""

    return build_tensorrt_session(model_path, config=config)


class OnnxBatchRunner:
    """Adaptive batched ORT runner with automatic batch-one fallback."""

    def __init__(
        self,
        model_path: Optional[str | os.PathLike[str]] = None,
        session: Any = None,
        providers: Optional[Sequence[Any]] = None,
        requested_batch: int = 8,
        device_id: int = 0,
        per_item_vram_mb: float = 96.0,
    ) -> None:
        self.session = session or create_onnx_session(model_path or "", providers)
        self.governor = VramGovernor.from_environment(
            device_id=device_id, requested_batch=requested_batch
        )
        self.requested_batch = max(1, int(requested_batch))
        self.per_item_vram_mb = max(1.0, float(per_item_vram_mb))
        self.binding = CudaIOBinding(self.session, device_id=device_id)
        self._lock = threading.RLock()

    @property
    def active_providers(self) -> List[str]:
        """Return providers actually registered by ORT."""

        try:
            return [str(name) for name in self.session.get_providers()]
        except Exception as _degrade_error:
            _swallowed("roop/optimized_processor.py:1101", _degrade_error, "fallback continued")
            return []

    @staticmethod
    def _batch_size(feeds: Mapping[str, NDArray[Any]]) -> int:
        """Validate that all model inputs share the same leading batch axis."""

        batch: Optional[int] = None
        for name, value in feeds.items():
            array = np.asarray(value)
            if array.ndim == 0:
                raise ValueError(f"input {name!r} has no batch axis")
            if batch is None:
                batch = int(array.shape[0])
            elif int(array.shape[0]) != batch:
                raise ValueError("all ONNX inputs must have the same batch dimension")
        if batch is None or batch <= 0:
            raise ValueError("ONNX feed is empty")
        return batch

    @staticmethod
    def _slice_feeds(
        feeds: Mapping[str, NDArray[Any]], start: int, end: int
    ) -> Dict[str, NDArray[Any]]:
        """Slice every batched input without copying until ORT needs it."""

        return {
            name: np.ascontiguousarray(np.asarray(value)[start:end])
            for name, value in feeds.items()
        }

    def _run_once(self, feeds: Mapping[str, NDArray[Any]]) -> List[NDArray[Any]]:
        """Use I/O binding when possible, then the canonical ORT path."""

        bound = self.binding.run(feeds)
        if bound is not None:
            return bound
        return [np.ascontiguousarray(value) for value in self.session.run(None, dict(feeds))]

    def run(
        self,
        feeds: Mapping[str, NDArray[Any]],
        batch_size: Optional[int] = None,
    ) -> List[NDArray[Any]]:
        """Run a feed in bounded chunks, shrinking after batch failures."""

        with self._lock:
            total = self._batch_size(feeds)
            requested = batch_size or self.requested_batch
            chunk_size = self.governor.safe_batch(requested, self.per_item_vram_mb)
            chunk_size = min(total, max(1, chunk_size))
            outputs: List[List[NDArray[Any]]] = []
            start = 0
            while start < total:
                end = min(total, start + chunk_size)
                current = self._slice_feeds(feeds, start, end)
                while True:
                    try:
                        outputs.append(self._run_once(current))
                        break
                    except Exception as error:
                        current_size = end - start
                        if current_size <= 1 or not _is_batch_axis_error(error):
                            raise
                        chunk_size = self.governor.note_oom(current_size)
                        end = min(total, start + max(1, chunk_size))
                        current = self._slice_feeds(feeds, start, end)
                start = end

            if not outputs:
                return []
            output_count = len(outputs[0])
            merged: List[NDArray[Any]] = []
            for output_index in range(output_count):
                merged.append(
                    np.ascontiguousarray(
                        np.concatenate(
                            [chunk[output_index] for chunk in outputs], axis=0
                        )
                    )
                )
            return merged


class TrtOnnxBatchRunner:
    """Strict GPU-only ORT runner with dynamic-batch tail padding.

    The runner never reduces a failed request to batch one.  It shrinks a
    dynamic request from 16 to 8 to 4 to 2, and raises if two items still do
    not fit the configured VRAM budget.  A one-item tail is padded with its
    last tensor and trimmed after GPU inference, so TensorRT sees a valid
    batched profile for every enqueue.
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        config: Optional[TensorRTSessionConfig] = None,
        session: Any = None,
        output_shapes: Optional[Mapping[str, Sequence[int]]] = None,
        per_item_vram_mb: float = 96.0,
    ) -> None:
        self.config = config or TensorRTSessionConfig.from_environment()
        _minimum, optimal, maximum = self.config.resolved_batches()
        if maximum < 2:
            raise ValueError(
                "TrtOnnxBatchRunner requires a dynamic profile with max_batch >= 2"
            )
        self.session = session or build_tensorrt_session(model_path, config=self.config)
        assert_strict_tensorrt_session(self.session, model_path)
        self.device_id = int(self.config.device_id)
        self.requested_batch = max(2, int(optimal), min(16, int(maximum)))
        self.max_batch = int(maximum)
        self.output_shapes = dict(output_shapes or {})
        self.per_item_vram_mb = max(1.0, float(per_item_vram_mb))
        self.governor = VramGovernor.from_environment(
            device_id=self.device_id,
            requested_batch=self.max_batch,
        )
        self.binding = CudaIOBinding(self.session, device_id=self.device_id)
        if not self.binding.enabled:
            raise RuntimeError(
                "strict TensorRT runner could not enable CUDA I/O binding: "
                f"{self.binding._disabled_reason or 'unknown reason'}"
            )
        self._lock = threading.RLock()

    @property
    def active_providers(self) -> List[str]:
        """Return the validated active provider chain."""

        return [str(name) for name in self.session.get_providers()]

    @staticmethod
    def _validate_feeds(feeds: Mapping[str, Any]) -> Tuple[int, Any]:
        """Validate a non-empty CUDA tensor feed and return batch/device."""

        if _torch_module is None:
            raise RuntimeError("PyTorch is required for strict TensorRT inference")
        batch: Optional[int] = None
        device: Optional[Any] = None
        for name, value in feeds.items():
            if not isinstance(value, _torch_module.Tensor):
                raise TypeError(f"strict TensorRT input {name!r} must be a torch.Tensor")
            if not value.is_cuda or value.ndim == 0:
                raise ValueError(f"strict TensorRT input {name!r} must be a CUDA batch tensor")
            if batch is None:
                batch = int(value.shape[0])
                device = value.device
            elif int(value.shape[0]) != batch:
                raise ValueError("all strict TensorRT inputs must share the batch dimension")
            elif value.device != device:
                raise ValueError("all strict TensorRT inputs must share the CUDA device")
        if batch is None or batch <= 0 or device is None:
            raise ValueError("strict TensorRT feed is empty")
        return batch, device

    @staticmethod
    def _slice_feeds(feeds: Mapping[str, Any], start: int, end: int) -> Dict[str, Any]:
        """Slice CUDA tensors without creating host arrays."""

        return {name: value[start:end] for name, value in feeds.items()}

    @staticmethod
    def _pad_feeds(feeds: Mapping[str, Any], target_size: int) -> Dict[str, Any]:
        """Pad a CUDA feed by repeating its final item on device."""

        result: Dict[str, Any] = {}
        for name, value in feeds.items():
            current = int(value.shape[0])
            if current >= target_size:
                result[name] = value
                continue
            repeats = target_size - current
            tail = value[-1:].expand((repeats,) + tuple(value.shape[1:]))
            result[name] = _torch_module.cat((value, tail), dim=0)
        return result

    def run_gpu(
        self,
        feeds: Mapping[str, Any],
        batch_size: Optional[int] = None,
        pad_to_batch: bool = True,
    ) -> List[Any]:
        """Run all items on TRT, shrinking only down to dynamic batch two."""

        total, _device = self._validate_feeds(feeds)
        with self._lock:
            requested = min(self.max_batch, max(2, int(batch_size or self.requested_batch)))
            safe = self.governor.safe_batch(
                requested,
                per_item_mb=self.per_item_vram_mb,
            )
            chunk_size = max(2, min(self.max_batch, safe))
            chunks: List[List[Any]] = []
            start = 0
            while start < total:
                item_count = min(chunk_size, total - start)
                run_size = item_count
                if pad_to_batch and run_size < 2:
                    run_size = 2
                current = self._slice_feeds(feeds, start, start + item_count)
                while True:
                    if run_size < item_count:
                        item_count = run_size
                        current = self._slice_feeds(feeds, start, start + item_count)
                    padded = self._pad_feeds(current, run_size) if run_size > item_count else current
                    try:
                        outputs = self.binding.run_gpu(
                            padded,
                            output_shapes=self.output_shapes,
                        )
                        trimmed: List[Any] = []
                        for output in outputs:
                            if output.ndim == 0 or int(output.shape[0]) != run_size:
                                raise RuntimeError(
                                    "strict batched output does not expose the same leading batch "
                                    f"dimension as the input: {tuple(output.shape)} vs {run_size}"
                                )
                            trimmed.append(output[:item_count])
                        chunks.append(trimmed)
                        break
                    except BaseException as error:
                        if not _is_batch_axis_error(error) or run_size <= 2:
                            raise RuntimeError(
                                "TensorRT dynamic batch execution failed at the minimum "
                                f"batch size {run_size}; no CPU/CUDA fallback was attempted"
                            ) from error
                        self.governor.note_oom(run_size)
                        run_size = max(2, run_size // 2)
                        chunk_size = min(chunk_size, run_size)
                start += item_count
            if not chunks:
                return []
            merged: List[Any] = []
            for output_index in range(len(chunks[0])):
                merged.append(_torch_module.cat([chunk[output_index] for chunk in chunks], dim=0))
            return merged


class CudaAffineBatch:
    """GPU affine sampler that preserves the prepass's OpenCV matrix meaning."""

    @staticmethod
    def _matrix3(matrices: Any) -> Any:
        """Convert N x 2 x 3 matrices to homogeneous N x 3 x 3 tensors."""

        if _torch_module is None:
            raise RuntimeError("PyTorch is required for CUDA affine transforms")
        rows = _torch_module.zeros(
            (int(matrices.shape[0]), 1, 3), dtype=matrices.dtype, device=matrices.device
        )
        rows[:, :, 2] = 1.0
        return _torch_module.cat((matrices, rows), dim=1)

    @staticmethod
    def _grid(
        matrices: Any,
        input_height: int,
        input_width: int,
        output_height: int,
        output_width: int,
    ) -> Any:
        """Build an align-corners-false pixel grid entirely on CUDA."""

        torch = _torch_module
        if torch is None:
            raise RuntimeError("PyTorch is required for CUDA affine transforms")
        y, x = torch.meshgrid(
            torch.arange(output_height, device=matrices.device, dtype=matrices.dtype),
            torch.arange(output_width, device=matrices.device, dtype=matrices.dtype),
            indexing="ij",
        )
        homogeneous = torch.stack(
            (x.reshape(-1), y.reshape(-1), torch.ones_like(x).reshape(-1)), dim=0
        )
        points = torch.bmm(
            matrices,
            homogeneous.unsqueeze(0).expand(int(matrices.shape[0]), -1, -1),
        ).transpose(1, 2)
        normalized_x = ((points[..., 0] + 0.5) / float(input_width)) * 2.0 - 1.0
        normalized_y = ((points[..., 1] + 0.5) / float(input_height)) * 2.0 - 1.0
        return torch.stack((normalized_x, normalized_y), dim=-1).reshape(
            int(matrices.shape[0]), output_height, output_width, 2
        )

    @classmethod
    def warp_frames(cls, frames: Any, matrices: Any, output_size: int) -> Any:
        """Apply source-to-aligned matrices to a CUDA NCHW frame batch."""

        if _torch_module is None:
            raise RuntimeError("PyTorch is required for CUDA affine transforms")
        import torch.nn.functional as functional

        if frames.ndim != 4 or matrices.ndim != 3 or matrices.shape[1:] != (2, 3):
            raise ValueError("warp_frames expects NCHW frames and N x 2 x 3 matrices")
        height = int(frames.shape[2])
        width = int(frames.shape[3])
        inverse = torch_linalg_inverse(cls._matrix3(matrices))[:, :2, :]
        grid = cls._grid(
            inverse,
            input_height=height,
            input_width=width,
            output_height=int(output_size),
            output_width=int(output_size),
        )
        return functional.grid_sample(
            frames,
            grid,
            mode="bicubic",
            padding_mode="border",
            align_corners=False,
        )

    @classmethod
    def paste_faces(
        cls,
        frames: Any,
        aligned_faces: Any,
        frame_ids: Any,
        matrices: Any,
        masks: Optional[Any] = None,
        blend_ratio: float = 1.0,
    ) -> Any:
        """Reverse-warp aligned faces and alpha-composite them on CUDA."""

        if _torch_module is None:
            raise RuntimeError("PyTorch is required for CUDA face compositing")
        import torch.nn.functional as functional

        if aligned_faces.ndim != 4:
            raise ValueError("aligned_faces must be NCHW")
        frame_height = int(frames.shape[2])
        frame_width = int(frames.shape[3])
        crop_size = int(aligned_faces.shape[2])
        grid = cls._grid(
            cls._matrix3(matrices)[:, :2, :],
            input_height=crop_size,
            input_width=crop_size,
            output_height=frame_height,
            output_width=frame_width,
        )
        patches = functional.grid_sample(
            aligned_faces,
            grid,
            mode="bicubic",
            padding_mode="zeros",
            align_corners=False,
        )
        if masks is None:
            alpha = _torch_module.ones(
                (int(aligned_faces.shape[0]), 1, crop_size, crop_size),
                dtype=aligned_faces.dtype,
                device=aligned_faces.device,
            )
        else:
            alpha = masks
            if alpha.ndim == 3:
                alpha = alpha.unsqueeze(1)
            alpha = alpha.to(dtype=aligned_faces.dtype)
        alpha = functional.grid_sample(
            alpha,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).clamp(0.0, 1.0) * float(max(0.0, min(1.0, blend_ratio)))
        composite = frames.clone()
        for index in range(int(patches.shape[0])):
            frame_index = int(frame_ids[index])
            weight = alpha[index]
            composite[frame_index] = (
                patches[index] * weight + composite[frame_index] * (1.0 - weight)
            )
        return composite


def torch_linalg_inverse(matrix: Any) -> Any:
    """Keep the affine helper import-safe on installations without Torch."""

    if _torch_module is None:
        raise RuntimeError("PyTorch is required for CUDA affine transforms")
    return _torch_module.linalg.inv(matrix)


class CudaFrameBridge:
    """Upload/download one batched frame boundary around the raw FFmpeg pipe."""

    def __init__(self, device_id: int = 0, pin_memory: bool = True) -> None:
        if _torch_module is None or not _torch_module.cuda.is_available():
            raise RuntimeError("CUDA is required for CudaFrameBridge")
        self.torch = _torch_module
        self.device_id = int(device_id)
        self.pin_memory = bool(pin_memory)
        self.device = self.torch.device(f"cuda:{self.device_id}")

    def upload_bgr(self, frames: Sequence[UInt8Array]) -> Any:
        """Upload packed BGR frames once and return normalized CUDA NCHW tensors."""

        if not frames:
            raise ValueError("cannot upload an empty frame batch")
        host = np.ascontiguousarray(np.stack(frames, axis=0))
        cpu_tensor = self.torch.from_numpy(host).permute(0, 3, 1, 2).contiguous()
        if self.pin_memory:
            cpu_tensor = cpu_tensor.pin_memory()
        return cpu_tensor.to(self.device, dtype=self.torch.float32, non_blocking=self.pin_memory) / 255.0

    def download_bgr(self, frames: Any) -> List[UInt8Array]:
        """Download one packed BGR batch after all GPU work has completed."""

        if not isinstance(frames, self.torch.Tensor) or frames.ndim != 4:
            raise ValueError("download_bgr expects a CUDA NCHW tensor")
        normalized = frames.detach().clamp(0.0, 1.0).mul(255.0).round().to(self.torch.uint8)
        host = normalized.permute(0, 2, 3, 1).contiguous().to(
            "cpu", non_blocking=self.pin_memory
        )
        if self.pin_memory:
            self.torch.cuda.current_stream(self.device_id).synchronize()
        array = np.ascontiguousarray(host.numpy())
        return [array[index] for index in range(int(array.shape[0]))]


class GpuFaceRestorer:
    """Batch face restoration on CUDA between swap and reverse compositing."""

    def __init__(
        self,
        runner: TrtOnnxBatchRunner,
        input_size: int = 512,
        input_name: str = "input",
        output_index: int = 0,
        model_mean: Sequence[float] = (0.0, 0.0, 0.0),
        model_standard_deviation: Sequence[float] = (1.0, 1.0, 1.0),
        model_denormalize: bool = False,
    ) -> None:
        if _torch_module is None:
            raise RuntimeError("PyTorch is required for GpuFaceRestorer")
        if int(input_size) <= 0:
            raise ValueError("restorer input_size must be positive")
        if len(model_mean) != 3 or len(model_standard_deviation) != 3:
            raise ValueError("restorer mean and standard deviation need three channels")
        self.torch = _torch_module
        self.runner = runner
        self.input_size = int(input_size)
        self.input_name = str(input_name)
        self.output_index = int(output_index)
        self.mean = self.torch.tensor(
            tuple(float(value) for value in model_mean),
            dtype=self.torch.float32,
            device=f"cuda:{runner.device_id}",
        ).reshape(1, 3, 1, 1)
        self.standard_deviation = self.torch.tensor(
            tuple(float(value) for value in model_standard_deviation),
            dtype=self.torch.float32,
            device=f"cuda:{runner.device_id}",
        ).reshape(1, 3, 1, 1)
        if bool(self.torch.any(self.standard_deviation == 0)):
            raise ValueError("restorer standard deviation cannot contain zero")
        self.denormalize = bool(model_denormalize)

    def __call__(self, faces: Any) -> Any:
        """Restore every swapped crop with one dynamic-batch TRT enqueue."""

        import torch.nn.functional as functional

        height = int(faces.shape[2])
        width = int(faces.shape[3])
        model_input = functional.interpolate(
            faces,
            size=(self.input_size, self.input_size),
            mode="bicubic",
            align_corners=False,
        )
        model_input = (model_input - self.mean) / self.standard_deviation
        outputs = self.runner.run_gpu(
            {self.input_name: model_input},
            pad_to_batch=True,
        )
        restored = outputs[self.output_index]
        if restored.ndim != 4:
            raise RuntimeError(f"restorer output must be NCHW, got {tuple(restored.shape)}")
        if self.denormalize:
            restored = (restored + 1.0) / 2.0
        if int(restored.shape[2]) != height or int(restored.shape[3]) != width:
            restored = functional.interpolate(
                restored,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )
        return restored.clamp(0.0, 1.0)


class GpuFaceSwapProcessor:
    """Batch all aligned faces across frames and keep swap/composite on CUDA.

    This adapter is deliberately model-contract driven.  It handles the
    common ``target``/``source`` 512-D embedding interface used by the
    dynamic face-swap exports; model-specific color correction and mask
    preparation remain injectable so the existing roop look settings are not
    overwritten by a generic compositor.
    """

    def __init__(
        self,
        runner: TrtOnnxBatchRunner,
        source_embedding: Any,
        input_size: int,
        target_input: str = "target",
        source_input: str = "source",
        output_index: int = 0,
        mask_output_index: Optional[int] = 1,
        model_channel_order: str = "rgb",
        model_mean: Sequence[float] = (0.0, 0.0, 0.0),
        model_standard_deviation: Sequence[float] = (1.0, 1.0, 1.0),
        model_denormalize: bool = False,
        blend_ratio: float = 1.0,
        restorer: Optional[Callable[[Any], Any]] = None,
        bridge: Optional[CudaFrameBridge] = None,
    ) -> None:
        if _torch_module is None:
            raise RuntimeError("PyTorch is required for GpuFaceSwapProcessor")
        self.torch = _torch_module
        self.runner = runner
        self.input_size = int(input_size)
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        embedding = source_embedding
        if not isinstance(embedding, self.torch.Tensor):
            embedding = self.torch.as_tensor(embedding, dtype=self.torch.float32)
        if not embedding.is_cuda or embedding.device.index != runner.device_id:
            embedding = embedding.to(f"cuda:{runner.device_id}")
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        if embedding.ndim != 2 or int(embedding.shape[0]) != 1:
            raise ValueError("source_embedding must have shape [1, embedding_dim]")
        self.source_embedding = embedding.contiguous()
        self.target_input = str(target_input)
        self.source_input = str(source_input)
        self.output_index = int(output_index)
        self.mask_output_index = mask_output_index
        self.model_channel_order = str(model_channel_order).strip().lower()
        if self.model_channel_order not in {"bgr", "rgb"}:
            raise ValueError("model_channel_order must be 'bgr' or 'rgb'")
        if len(model_mean) != 3 or len(model_standard_deviation) != 3:
            raise ValueError("model mean and standard deviation must have three channels")
        self.model_mean = self.torch.tensor(
            tuple(float(value) for value in model_mean),
            dtype=self.torch.float32,
            device=f"cuda:{runner.device_id}",
        ).reshape(1, 3, 1, 1)
        self.model_standard_deviation = self.torch.tensor(
            tuple(float(value) for value in model_standard_deviation),
            dtype=self.torch.float32,
            device=f"cuda:{runner.device_id}",
        ).reshape(1, 3, 1, 1)
        if bool(self.torch.any(self.model_standard_deviation == 0)):
            raise ValueError("model standard deviation cannot contain zero")
        self.model_denormalize = bool(model_denormalize)
        self.blend_ratio = float(max(0.0, min(1.0, blend_ratio)))
        self.restorer = restorer
        self.bridge = bridge or CudaFrameBridge(device_id=runner.device_id)

    def __call__(
        self,
        frames: Sequence[UInt8Array],
        analyses: Sequence[FrameAnalysis],
    ) -> List[UInt8Array]:
        """Process a host frame batch with one or more padded TRT enqueues."""

        if len(frames) != len(analyses):
            raise ValueError("frames and analyses must have equal lengths")
        gpu_frames = self.bridge.upload_bgr(frames)
        frame_ids: List[int] = []
        matrices: List[Float32Array] = []
        for frame_index, analysis in enumerate(analyses):
            for face in analysis.faces:
                frame_ids.append(frame_index)
                matrices.append(face.matrix)
        if not matrices:
            return [np.ascontiguousarray(frame) for frame in frames]
        matrix_tensor = self.torch.as_tensor(
            np.ascontiguousarray(np.stack(matrices), dtype=np.float32),
            device=gpu_frames.device,
        )
        frame_index_tensor = self.torch.as_tensor(
            frame_ids, dtype=self.torch.long, device=gpu_frames.device
        )
        face_frames = gpu_frames.index_select(0, frame_index_tensor)
        aligned = CudaAffineBatch.warp_frames(face_frames, matrix_tensor, self.input_size)
        if self.model_channel_order == "rgb":
            aligned = aligned[:, [2, 1, 0], :, :]
        aligned = (aligned - self.model_mean) / self.model_standard_deviation
        source = self.source_embedding.expand(int(aligned.shape[0]), -1).contiguous()
        outputs = self.runner.run_gpu(
            {self.target_input: aligned, self.source_input: source},
            pad_to_batch=True,
        )
        swapped = outputs[self.output_index]
        if swapped.ndim != 4:
            raise RuntimeError(f"swap output must be NCHW, got {tuple(swapped.shape)}")
        if self.model_denormalize:
            swapped = (swapped + 1.0) / 2.0
        if self.model_channel_order == "rgb":
            swapped = swapped[:, [2, 1, 0], :, :]
        if self.restorer is not None:
            swapped = self.restorer(swapped)
        masks = None
        if self.mask_output_index is not None and self.mask_output_index < len(outputs):
            masks = outputs[self.mask_output_index]
        composite = CudaAffineBatch.paste_faces(
            gpu_frames,
            swapped,
            frame_index_tensor,
            matrix_tensor,
            masks=masks,
            blend_ratio=self.blend_ratio,
        )
        return self.bridge.download_bgr(composite)


@dataclass
class PipelineStats:
    """Counters returned by :class:`MemoryStreamingProcessor`."""

    frames: int = 0
    elapsed_seconds: float = 0.0
    decode_seconds: float = 0.0
    process_seconds: float = 0.0
    write_seconds: float = 0.0
    detection_frames: int = 0
    max_queue_depth: int = 0

    @property
    def fps(self) -> float:
        """End-to-end frames per second."""

        return self.frames / self.elapsed_seconds if self.elapsed_seconds > 0.0 else 0.0


@dataclass
class _FramePacket:
    frame_index: int
    frame: UInt8Array


class MemoryStreamingProcessor:
    """Decode/process/encode pipeline with bounded queues and clean shutdown.

    ``frame_processor`` is the quality-preserving integration point for the
    existing ``ProcessMgr.process_frame`` path.  ``batch_processor`` is an
    optional adapter for a model-aware implementation that collects aligned
    crops across consecutive frames and calls :class:`OnnxBatchRunner` once.
    ``gpu_batch_processor`` is the strict TensorRT path.  It receives host
    frames only at the raw FFmpeg boundary, performs upload, alignment,
    inference, reverse warp, and blend as one CUDA batch, then downloads one
    output batch for NVENC.
    When a batch callback fails on a batch larger than one, the processor falls
    back to the single-frame callback if supplied; this preserves output
    continuity while leaving the model's failure visible to the caller.
    """

    def __init__(
        self,
        prepass: Optional[FacePrepass] = None,
        frame_processor: Optional[Callable[[UInt8Array, FrameAnalysis], UInt8Array]] = None,
        batch_processor: Optional[
            Callable[[Sequence[UInt8Array], Sequence[FrameAnalysis]], Sequence[UInt8Array]]
        ] = None,
        gpu_batch_processor: Optional[
            Callable[[Sequence[UInt8Array], Sequence[FrameAnalysis]], Sequence[UInt8Array]]
        ] = None,
        queue_depth: int = 3,
        batch_size: Optional[int] = None,
        hwaccel: Optional[str] = None,
        strict_trt: bool = False,
    ) -> None:
        self.prepass = prepass
        self.frame_processor = frame_processor
        self.batch_processor = batch_processor
        self.gpu_batch_processor = gpu_batch_processor
        self.strict_trt = bool(strict_trt)
        self.queue_depth = max(1, int(queue_depth))
        requested = batch_size or (
            prepass.config.detector_batch_size if prepass is not None else 1
        )
        self.batch_size = max(2 if self.strict_trt else 1, int(requested))
        if self.strict_trt and self.gpu_batch_processor is None:
            raise ValueError("strict_trt requires gpu_batch_processor")
        self.hwaccel = hwaccel
        self._failure: Optional[BaseException] = None
        self._stop = threading.Event()

    def _put_or_abort(self, queue: Queue[Any], item: Any) -> bool:
        """Put into a bounded queue until a peer fails or accepts the item."""

        while not self._stop.is_set():
            try:
                queue.put(item, timeout=0.25)
                return True
            except Full:
                continue
        return False

    def _record_failure(self, error: BaseException) -> None:
        """Publish the first error and wake every blocked stage."""

        if self._failure is None:
            self._failure = error
        self._stop.set()

    def run(
        self,
        input_video: str | os.PathLike[str],
        output_video: str | os.PathLike[str],
        audio_source: Optional[str | os.PathLike[str]] = None,
        codec: Optional[str] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        spec: Optional[VideoSpec] = None,
    ) -> PipelineStats:
        """Process one video without materializing intermediate frames."""

        self._stop.clear()
        self._failure = None
        video_spec = spec or probe_video(input_video)
        governor = VramGovernor.from_environment(
            requested_batch=self.batch_size
        )
        effective_batch = governor.safe_batch(self.batch_size)
        if self.strict_trt:
            # A dynamic TensorRT profile is mandatory.  If the laptop guard
            # reports one item, retain one TRT context but still enqueue two
            # items by padding inside TrtOnnxBatchRunner; never use a batch-one
            # model call or silently switch execution providers.
            effective_batch = max(2, effective_batch)
        reader = FFmpegRawReader(
            input_video,
            spec=video_spec,
            hwaccel=self.hwaccel,
            start_frame=start_frame,
        )
        writer = AsyncRawVideoWriter(
            FFmpegRawWriter(
                output_video,
                video_spec,
                codec=codec,
                audio_source=audio_source,
            ),
            queue_depth=self.queue_depth,
        )
        input_queue: Queue[Optional[_FramePacket]] = Queue(maxsize=self.queue_depth)
        stats = PipelineStats()
        started = time.perf_counter()
        def reader_loop() -> None:
            """Decode into RAM and stop cleanly on a downstream failure."""

            index = int(start_frame)
            try:
                reader.start()
                while not self._stop.is_set():
                    if end_frame is not None and index >= int(end_frame):
                        break
                    decode_start = time.perf_counter()
                    frame = reader.read()
                    stats.decode_seconds += time.perf_counter() - decode_start
                    if frame is None:
                        break
                    packet = _FramePacket(index, frame)
                    if not self._put_or_abort(input_queue, packet):
                        return
                    stats.max_queue_depth = max(stats.max_queue_depth, input_queue.qsize())
                    index += 1
            except BaseException as error:
                _swallowed("roop/optimized_processor.py:1874", error, "fallback continued")
                self._record_failure(error)
            finally:
                planned_stop = end_frame is not None and index >= int(end_frame)
                try:
                    reader.close(ignore_errors=planned_stop or self._failure is not None)
                except BaseException as error:
                    _swallowed("roop/optimized_processor.py:1880", error, "fallback continued")
                    self._record_failure(error)
                self._put_or_abort(input_queue, None)

        def process_loop() -> None:
            """Run ordered prepass and transformation work."""

            try:
                input_done = False
                while not self._stop.is_set():
                    packets: List[_FramePacket] = []
                    while len(packets) < effective_batch and not self._stop.is_set():
                        try:
                            packet = input_queue.get(timeout=0.25)
                        except Empty:
                            continue
                        if packet is None:
                            input_done = True
                            break
                        packets.append(packet)
                    if not packets:
                        break
                    frames = [packet.frame for packet in packets]
                    process_start = time.perf_counter()
                    if self.prepass is not None:
                        analyses = self.prepass.process_batch(
                            frames, start_index=packets[0].frame_index
                        )
                    else:
                        analyses = [
                            FrameAnalysis(packet.frame_index, []) for packet in packets
                        ]
                    stats.detection_frames += sum(
                        1 for analysis in analyses if analysis.detection_run
                    )
                    if self.gpu_batch_processor is not None:
                        output_frames = list(self.gpu_batch_processor(frames, analyses))
                    elif self.batch_processor is not None and len(frames) > 1:
                        try:
                            output_frames = list(self.batch_processor(frames, analyses))
                        except BaseException:
                            if self.frame_processor is None:
                                raise
                            output_frames = [
                                self.frame_processor(frame, analysis)
                                for frame, analysis in zip(frames, analyses)
                            ]
                    elif self.frame_processor is not None:
                        output_frames = [
                            self.frame_processor(frame, analysis)
                            for frame, analysis in zip(frames, analyses)
                        ]
                    else:
                        output_frames = frames
                    stats.process_seconds += time.perf_counter() - process_start
                    if len(output_frames) != len(frames):
                        raise ValueError("processor returned the wrong number of frames")
                    for output in output_frames:
                        array = np.asarray(output)
                        if array.shape != frames[0].shape or array.dtype != np.uint8:
                            raise ValueError("processor must return uint8 BGR frames of input size")
                        writer_start = time.perf_counter()
                        writer.submit(np.ascontiguousarray(array))
                        stats.write_seconds += time.perf_counter() - writer_start
                        stats.frames += 1
                    if input_done:
                        break
            except BaseException as error:
                _swallowed("roop/optimized_processor.py:1947", error, "fallback continued")
                self._record_failure(error)
            finally:
                self._stop.set()

        reader_thread = threading.Thread(target=reader_loop, name="optimized-video-reader", daemon=True)
        process_thread = threading.Thread(target=process_loop, name="optimized-video-processor", daemon=True)
        writer.start()
        reader_thread.start()
        process_thread.start()
        process_thread.join()
        self._stop.set()
        try:
            reader.close()
        except BaseException as error:
            _swallowed("roop/optimized_processor.py:1961", error, "fallback continued")
            self._record_failure(error)
        reader_thread.join(timeout=10.0)
        try:
            writer.close(discard=self._failure is not None)
        except BaseException as error:
            _swallowed("roop/optimized_processor.py:1966", error, "fallback continued")
            self._record_failure(error)
        stats.elapsed_seconds = time.perf_counter() - started
        if self._failure is not None:
            raise RuntimeError("optimized pipeline failed") from self._failure
        return stats


__all__ = [
    "AsyncRawVideoWriter",
    "CudaAffineBatch",
    "CudaFrameBridge",
    "CudaIOBinding",
    "FFmpegRawReader",
    "FFmpegRawWriter",
    "GpuFaceSwapProcessor",
    "GpuFaceRestorer",
    "MemoryStreamingProcessor",
    "OnnxBatchRunner",
    "PipelineStats",
    "TrtOnnxBatchRunner",
    "VideoSpec",
    "VramGovernor",
    "create_onnx_session",
    "create_strict_tensorrt_session",
    "probe_video",
]
