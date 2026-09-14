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


Float32Array = NDArray[np.float32]
UInt8Array = NDArray[np.uint8]

try:
    # Import Torch before ONNX Runtime.  On Windows, loading ORT's CUDA DLLs
    # first can make a later Torch import fail with error 127 even though both
    # packages are individually installed and the CUDA provider is usable.
    import torch as _torch_module
except Exception:  # pragma: no cover - CPU-only/minimal installations
    _torch_module = None

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
    except Exception:
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
            command.extend(["-c:a", "copy"])
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
        except Exception:
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
        except Exception:
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
            "cudaerror",
            "cudnn_status_alloc_failed",
            "shape",
            "dimension",
            "invalid argument",
            "reshape",
        )
    )


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
        available = set(ort.get_available_providers())
        requested: List[Any] = [
            name
            for name in (
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            )
            if name in available
        ]
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
            last_error = error
    raise RuntimeError(
        f"unable to create ONNX Runtime session for {model_path}"
    ) from last_error


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
        except Exception:
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
        queue_depth: int = 3,
        batch_size: Optional[int] = None,
        hwaccel: Optional[str] = None,
    ) -> None:
        self.prepass = prepass
        self.frame_processor = frame_processor
        self.batch_processor = batch_processor
        self.queue_depth = max(1, int(queue_depth))
        requested = batch_size or (
            prepass.config.detector_batch_size if prepass is not None else 1
        )
        self.batch_size = max(1, int(requested))
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
                self._record_failure(error)
            finally:
                planned_stop = end_frame is not None and index >= int(end_frame)
                try:
                    reader.close(ignore_errors=planned_stop or self._failure is not None)
                except BaseException as error:
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
                    if self.batch_processor is not None and len(frames) > 1:
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
            self._record_failure(error)
        reader_thread.join(timeout=10.0)
        try:
            writer.close(discard=self._failure is not None)
        except BaseException as error:
            self._record_failure(error)
        stats.elapsed_seconds = time.perf_counter() - started
        if self._failure is not None:
            raise RuntimeError("optimized pipeline failed") from self._failure
        return stats


__all__ = [
    "AsyncRawVideoWriter",
    "CudaIOBinding",
    "FFmpegRawReader",
    "FFmpegRawWriter",
    "MemoryStreamingProcessor",
    "OnnxBatchRunner",
    "PipelineStats",
    "VideoSpec",
    "VramGovernor",
    "create_onnx_session",
    "probe_video",
]
