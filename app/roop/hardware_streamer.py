"""NVDEC ingestion and a pickle-free shared-memory frame ring.

The optimized renderer has two different memory boundaries that are easy to
confuse:

* FFmpeg/PyAV can keep a decoded surface in CUDA memory until a consumer asks
  for pixels.  The Python ``multiprocessing.shared_memory`` API, however, is
  backed by system RAM and cannot contain a CUDA device pointer.
* A shared-memory ring is therefore the correct zero-pickle IPC transport for
  a host-frame boundary, but it is not a CUDA IPC mechanism.  This module keeps
  that boundary explicit: NVDEC performs the decode, one packed BGR download
  feeds the ring, and the consumer can upload from pinned memory asynchronously.

``NvdecFrameSource`` refuses to silently switch to software decode when
``strict_nvdec`` is true.  ``PyAvCudaSource`` is available for installations
with a PyAV build that exposes CUDA hardware frames; the FFmpeg backend is the
more portable path on Windows.  Neither backend writes intermediate images.

The ring uses only shared memory plus ``multiprocessing`` synchronization
objects.  Frames and metadata are never pickled through a queue.  A handle can
be passed to a child process before it starts; the child then attaches to the
same shared-memory blocks and synchronization primitives.
"""

from __future__ import annotations
from roop.degrade import swallowed as _swallowed

import json
import math
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from multiprocessing import Condition, Lock, Value
from multiprocessing import shared_memory
from typing import Any, Iterator, Optional, Tuple
import weakref

import numpy as np
from numpy.typing import NDArray

try:
    from roop.ffmpeg_writer import FFMPEG_BINARY
except Exception as _degrade_error:  # pragma: no cover - import-safe outside the app package
    _swallowed("roop/hardware_streamer.py:44", _degrade_error, "fallback continued")
    FFMPEG_BINARY = shutil.which("ffmpeg") or "ffmpeg"


UInt8Array = NDArray[np.uint8]

_METADATA_DTYPE = np.dtype(
    [
        ("frame_index", "<i8"),
        ("timestamp", "<f8"),
        ("scene_cut", "u1"),
        ("width", "<i4"),
        ("height", "<i4"),
        ("sequence", "<i8"),
    ],
    align=True,
)


def _popen_kwargs() -> dict[str, Any]:
    """Prevent a console window from being created for FFmpeg on Windows."""

    return {"creationflags": 0x08000000} if os.name == "nt" else {}


def _positive_int(value: Any, default: int) -> int:
    """Parse an integer while keeping allocation dimensions positive."""

    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(default))


@dataclass(frozen=True)
class FrameMetadata:
    """Metadata stored beside one ring slot."""

    frame_index: int
    timestamp: float
    scene_cut: bool
    width: int
    height: int
    sequence: int


@dataclass(frozen=True)
class FramePacket:
    """A copied consumer view and its shared-memory metadata."""

    frame: UInt8Array
    metadata: FrameMetadata


@dataclass(frozen=True)
class SharedRingHandle:
    """Serializable descriptor for attaching to an existing ring.

    The synchronization wrappers are intentionally part of the handle.  On
    Windows they must be supplied to the spawned child before it starts rather
    than recreated by name, otherwise producers and consumers would wait on
    different locks.
    """

    data_name: str
    metadata_name: str
    capacity: int
    height: int
    width: int
    channels: int
    write_sequence: Any
    read_sequence: Any
    producer_done: Any
    closed: Any
    condition: Any


@dataclass(frozen=True)
class RingStats:
    """Monotonic counters useful for back-pressure diagnostics."""

    written: int
    read: int
    capacity: int
    dropped: int


class SharedMemoryFrameRing:
    """Bounded circular frame storage backed by ``multiprocessing.shared_memory``.

    The producer blocks when all slots are occupied, and the consumer blocks
    when the ring is empty.  ``finish`` is the normal end-of-stream sentinel;
    ``abort`` wakes both sides immediately after an error.  ``read`` copies a
    slot while holding the condition lock, so the producer can safely reuse it
    as soon as the method returns.
    """

    def __init__(
        self,
        capacity: int,
        height: int,
        width: int,
        channels: int = 3,
        *,
        handle: Optional[SharedRingHandle] = None,
    ) -> None:
        if handle is None:
            self.capacity = _positive_int(capacity, 4)
            self.height = _positive_int(height, 1)
            self.width = _positive_int(width, 1)
            self.channels = _positive_int(channels, 3)
            if self.channels not in (1, 3, 4):
                raise ValueError("channels must be 1, 3, or 4")
            frame_bytes = self.capacity * self.height * self.width * self.channels
            self._data_shm = shared_memory.SharedMemory(create=True, size=frame_bytes)
            try:
                metadata_bytes = self.capacity * _METADATA_DTYPE.itemsize
                self._metadata_shm = shared_memory.SharedMemory(
                    create=True, size=metadata_bytes
                )
            except BaseException:
                self._data_shm.close()
                self._data_shm.unlink()
                raise
            self._owner = True
            self._write_sequence = Value("q", 0)
            self._read_sequence = Value("q", 0)
            self._producer_done = Value("b", 0)
            self._closed = Value("b", 0)
            self._condition = Condition(Lock())
        else:
            self.capacity = _positive_int(handle.capacity, 4)
            self.height = _positive_int(handle.height, 1)
            self.width = _positive_int(handle.width, 1)
            self.channels = _positive_int(handle.channels, 3)
            self._data_shm = shared_memory.SharedMemory(name=handle.data_name)
            self._metadata_shm = shared_memory.SharedMemory(name=handle.metadata_name)
            self._owner = False
            self._write_sequence = handle.write_sequence
            self._read_sequence = handle.read_sequence
            self._producer_done = handle.producer_done
            self._closed = handle.closed
            self._condition = handle.condition
        self._frames = np.ndarray(
            (self.capacity, self.height, self.width, self.channels),
            dtype=np.uint8,
            buffer=self._data_shm.buf,
        )
        self._metadata = np.ndarray(
            (self.capacity,), dtype=_METADATA_DTYPE, buffer=self._metadata_shm.buf
        )
        self._closed_local = False
        self._unlinked = False
        if self._owner:
            self._finalizer = weakref.finalize(
                self, self._cleanup_shm, self._data_shm, self._metadata_shm
            )

    @classmethod
    def create(
        cls, capacity: int, height: int, width: int, channels: int = 3
    ) -> "SharedMemoryFrameRing":
        """Create and own a new shared-memory ring."""

        return cls(capacity, height, width, channels)

    @classmethod
    def attach(cls, handle: SharedRingHandle) -> "SharedMemoryFrameRing":
        """Attach to a ring created by another process."""

        return cls(
            handle.capacity,
            handle.height,
            handle.width,
            handle.channels,
            handle=handle,
        )

    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        """Shape expected by :meth:`write` and returned by :meth:`read`."""

        return self.height, self.width, self.channels

    @property
    def handle(self) -> SharedRingHandle:
        """Return the descriptor to pass to a child process before it starts."""

        return SharedRingHandle(
            data_name=self._data_shm.name,
            metadata_name=self._metadata_shm.name,
            capacity=self.capacity,
            height=self.height,
            width=self.width,
            channels=self.channels,
            write_sequence=self._write_sequence,
            read_sequence=self._read_sequence,
            producer_done=self._producer_done,
            closed=self._closed,
            condition=self._condition,
        )

    def _wait_remaining(self, deadline: Optional[float]) -> Optional[float]:
        """Return a condition wait timeout or raise on a passed deadline."""

        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError("shared-memory frame ring operation timed out")
        return remaining

    def write(
        self,
        frame: NDArray[np.uint8],
        metadata: FrameMetadata,
        timeout: Optional[float] = None,
    ) -> None:
        """Copy one packed frame into the next available slot.

        The copy is a fixed-size memcpy into preallocated shared memory.  No
        Python object containing frame data crosses a process boundary.
        """

        source = np.asarray(frame)
        if source.shape != self.frame_shape or source.dtype != np.uint8:
            raise ValueError(
                f"ring expects uint8 {self.frame_shape}, got {source.dtype} "
                f"{source.shape}"
            )
        if not source.flags.c_contiguous:
            source = np.ascontiguousarray(source)
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while (
                self._write_sequence.value - self._read_sequence.value >= self.capacity
            ):
                if self._closed.value:
                    raise RuntimeError("cannot write to a closed frame ring")
                self._condition.wait(self._wait_remaining(deadline))
            if self._closed.value:
                raise RuntimeError("cannot write to a closed frame ring")
            sequence = int(self._write_sequence.value)
            slot = sequence % self.capacity
            np.copyto(self._frames[slot], source, casting="no")
            self._metadata[slot] = (
                int(metadata.frame_index),
                float(metadata.timestamp),
                1 if metadata.scene_cut else 0,
                int(metadata.width),
                int(metadata.height),
                sequence,
            )
            self._write_sequence.value = sequence + 1
            self._condition.notify_all()

    def read(
        self,
        timeout: Optional[float] = None,
        destination: Optional[UInt8Array] = None,
    ) -> Optional[FramePacket]:
        """Read one packet, or ``None`` after a normal producer shutdown.

        Supplying ``destination`` lets a consumer recycle one host buffer and
        avoids a per-frame allocation.  It must not be the ring's internal
        array; the returned packet owns a stable consumer-side copy.
        """

        if destination is None:
            target = np.empty(self.frame_shape, dtype=np.uint8)
        else:
            target = np.asarray(destination)
            if target.shape != self.frame_shape or target.dtype != np.uint8:
                raise ValueError(
                    f"destination must be uint8 {self.frame_shape}, got "
                    f"{target.dtype} {target.shape}"
                )
            if not target.flags.c_contiguous:
                raise ValueError("destination must be C-contiguous")
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while self._read_sequence.value >= self._write_sequence.value:
                if self._closed.value:
                    return None
                if self._producer_done.value:
                    return None
                self._condition.wait(self._wait_remaining(deadline))
            sequence = int(self._read_sequence.value)
            slot = sequence % self.capacity
            np.copyto(target, self._frames[slot], casting="no")
            record = self._metadata[slot]
            metadata = FrameMetadata(
                frame_index=int(record["frame_index"]),
                timestamp=float(record["timestamp"]),
                scene_cut=bool(record["scene_cut"]),
                width=int(record["width"]),
                height=int(record["height"]),
                sequence=int(record["sequence"]),
            )
            if metadata.sequence != sequence:
                raise RuntimeError(
                    "shared-memory ring metadata was overwritten before consumption"
                )
            self._read_sequence.value = sequence + 1
            self._condition.notify_all()
            return FramePacket(target, metadata)

    def finish(self) -> None:
        """Publish the normal end-of-stream sentinel and wake readers."""

        with self._condition:
            self._producer_done.value = 1
            self._condition.notify_all()

    def abort(self) -> None:
        """Abort the ring and wake blocked producers and consumers."""

        with self._condition:
            self._closed.value = 1
            self._producer_done.value = 1
            self._condition.notify_all()

    def stats(self) -> RingStats:
        """Return current sequence counters without copying frame data."""

        with self._condition:
            written = int(self._write_sequence.value)
            read = int(self._read_sequence.value)
            return RingStats(
                written=written,
                read=read,
                capacity=self.capacity,
                dropped=max(0, written - read - self.capacity),
            )

    @staticmethod
    def _cleanup_shm(data_shm: Any, metadata_shm: Any) -> None:
        for shm in (data_shm, metadata_shm):
            try:
                shm.close()
            except Exception as _degrade_error:
                _swallowed("roop/hardware_streamer.py:387", _degrade_error, "shm finalizer close")
            try:
                shm.unlink()
            except Exception as _degrade_error:
                _swallowed("roop/hardware_streamer.py:391", _degrade_error, "shm finalizer unlink")

    def close(self, unlink: bool = False) -> None:
        """Close local mappings and optionally unlink blocks owned by creator."""

        if not self._closed_local:
            self._closed_local = True
            try:
                self._data_shm.close()
            finally:
                self._metadata_shm.close()

        if unlink and self._owner and not self._unlinked:
            self._unlinked = True
            if hasattr(self, "_finalizer"):
                self._finalizer.detach()
            try:
                self._data_shm.unlink()
            finally:
                self._metadata_shm.unlink()

    def __enter__(self) -> "SharedMemoryFrameRing":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close(unlink=self._owner)


@dataclass(frozen=True)
class DecodeCapabilities:
    """Observed decode path; useful for refusing accidental software decode."""

    backend: str
    hardware_requested: bool
    hardware_active: bool
    gpu_surface_to_ring: bool
    host_download_boundary: bool
    detail: str


class FrameMetadataTracker:
    """Low-cost scene-cut and timestamp tracker for decoded frames."""

    def __init__(self, threshold: float = 0.40, sample_width: int = 96) -> None:
        self.threshold = max(0.0, float(threshold))
        self.sample_width = max(16, int(sample_width))
        self._previous: Optional[NDArray[np.float32]] = None

    def reset(self) -> None:
        """Forget the previous frame signature."""

        self._previous = None

    def _signature(self, frame: UInt8Array) -> NDArray[np.float32]:
        """Compute a small, allocation-bounded RGB mean signature."""

        height, width = frame.shape[:2]
        step = max(1, int(max(height, width) / self.sample_width))
        sampled = frame[::step, ::step].astype(np.float32, copy=False)
        means = sampled.reshape(-1, sampled.shape[-1]).mean(axis=0)
        scales = sampled.reshape(-1, sampled.shape[-1]).std(axis=0)
        return np.ascontiguousarray(np.concatenate((means, scales)), dtype=np.float32)

    def observe(self, frame: UInt8Array) -> bool:
        """Return true when the frame differs materially from its predecessor."""

        signature = self._signature(frame)
        previous = self._previous
        self._previous = signature
        if previous is None:
            return False
        scale = np.maximum(previous[3:], 8.0)
        distance = float(np.mean(np.abs(signature[:3] - previous[:3]) / scale))
        return bool(math.isfinite(distance) and distance > self.threshold)


def _probe_video(path: str, ffprobe: Optional[str] = None) -> Tuple[int, int, float]:
    """Read width, height, and frame rate without decoding a frame."""

    probe = ffprobe or shutil.which("ffprobe") or "ffprobe"
    command = [
        probe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "json",
        path,
    ]
    try:
        payload = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
        stream = json.loads(payload)["streams"][0]
        numerator, denominator = str(stream["r_frame_rate"]).split("/", 1)
        fps = float(numerator) / max(1.0, float(denominator))
        return int(stream["width"]), int(stream["height"]), max(1.0, fps)
    except (
        OSError,
        subprocess.SubprocessError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise RuntimeError(
            f"could not probe video geometry for {path}: {error}"
        ) from error


class NvdecFrameSource:
    """Read BGR frames from one FFmpeg process using CUDA/NVDEC.

    FFmpeg's rawvideo stdout is a host-memory API.  The decoder therefore stays
    on NVDEC until the explicit ``hwdownload`` required to populate the shared
    memory ring.  This is the maximal zero-copy design available with the
    requested ``multiprocessing.shared_memory`` transport; a fully GPU-resident
    path needs CUDA IPC or a native PyNvVideoCodec surface, not shared memory.
    """

    def __init__(
        self,
        video_path: str | os.PathLike[str],
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        ffmpeg: Optional[str] = None,
        ffprobe: Optional[str] = None,
        device_id: int = 0,
        strict_nvdec: bool = True,
        start_frame: int = 0,
    ) -> None:
        self.video_path = str(video_path)
        probed_width, probed_height, probed_fps = _probe_video(self.video_path, ffprobe)
        self.width = int(width or probed_width)
        self.height = int(height or probed_height)
        self.fps = float(fps or probed_fps)
        self.ffmpeg = str(ffmpeg or FFMPEG_BINARY)
        self.device_id = int(device_id)
        self.strict_nvdec = bool(strict_nvdec)
        self.start_frame = max(0, int(start_frame))
        self.process: Optional[subprocess.Popen[bytes]] = None
        self._frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self.capabilities = DecodeCapabilities(
            backend="ffmpeg",
            hardware_requested=True,
            hardware_active=False,
            gpu_surface_to_ring=False,
            host_download_boundary=True,
            detail="not started",
        )

    def _nvdec_probe(self) -> None:
        """Fail before the render if CUDA decode cannot initialize."""

        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-hwaccel",
            "cuda",
            "-hwaccel_device",
            str(self.device_id),
            "-hwaccel_output_format",
            "cuda",
            "-i",
            self.video_path,
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ]
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30.0,
            **_popen_kwargs(),
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", "replace").strip()
            raise RuntimeError(f"NVDEC initialization failed: {detail[-1000:]}")

    def _command(self) -> list[str]:
        """Build one raw BGR pipe command with no intermediate files."""

        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-hwaccel",
            "cuda",
            "-hwaccel_device",
            str(self.device_id),
            "-hwaccel_output_format",
            "cuda",
        ]
        if self.start_frame and self.fps > 0.0:
            timestamp = max(0.0, (self.start_frame - 0.5) / self.fps)
            command.extend(["-ss", f"{timestamp:.6f}"])
        command.extend(
            [
                "-noautorotate",
                "-i",
                self.video_path,
                "-fps_mode",
                "passthrough",
                "-vf",
                "hwdownload,format=bgr24",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-an",
                "-sn",
                "pipe:1",
            ]
        )
        return command

    def start(self) -> None:
        """Probe and spawn the one persistent decoder process."""

        if self.process is not None:
            return
        self._nvdec_probe()
        self.process = subprocess.Popen(
            self._command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            bufsize=self._frame.nbytes * 2,
            **_popen_kwargs(),
        )
        self.capabilities = DecodeCapabilities(
            backend="ffmpeg",
            hardware_requested=True,
            hardware_active=True,
            gpu_surface_to_ring=False,
            host_download_boundary=True,
            detail="NVDEC surface downloaded once to packed BGR for shared memory",
        )

    def read(self) -> Optional[UInt8Array]:
        """Read one reusable BGR frame or return ``None`` at clean EOF."""

        if self.process is None:
            self.start()
        if self.process is None or self.process.stdout is None:
            return None
        view = memoryview(self._frame).cast("B")
        offset = 0
        while offset < self._frame.nbytes:
            count = self.process.stdout.readinto(view[offset:])
            if not count:
                if offset == 0:
                    return None
                raise RuntimeError(
                    "NVDEC pipe ended with a partial frame "
                    f"({offset}/{self._frame.nbytes})"
                )
            offset += int(count)
        return self._frame

    def close(self, ignore_errors: bool = False) -> None:
        """Stop FFmpeg and surface decode errors unless explicitly planned."""

        process = self.process
        self.process = None
        if process is None:
            return
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
        if not ignore_errors and process.returncode not in (0, -15, 15, None):
            detail = error.decode("utf-8", "replace").strip()
            raise RuntimeError(f"NVDEC FFmpeg process failed: {detail}")

    def __enter__(self) -> "NvdecFrameSource":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close(ignore_errors=exc_type is not None)


class PyAvCudaSource:
    """Optional PyAV CUDA-frame reader with an explicit host conversion edge."""

    def __init__(
        self,
        video_path: str | os.PathLike[str],
        *,
        device_id: int = 0,
        strict_nvdec: bool = True,
    ) -> None:
        self.video_path = str(video_path)
        self.device_id = int(device_id)
        self.strict_nvdec = bool(strict_nvdec)
        self.container: Any = None
        self.stream: Any = None
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.capabilities = DecodeCapabilities(
            backend="pyav",
            hardware_requested=True,
            hardware_active=False,
            gpu_surface_to_ring=False,
            host_download_boundary=True,
            detail="not started",
        )

    def start(self) -> None:
        """Open PyAV with its CUDA hardware-acceleration option."""

        if self.container is not None:
            return
        try:
            import av

            self.container = av.open(self.video_path, mode="r", hwaccel="cuda")
            self.stream = self.container.streams.video[0]
            self.width = int(self.stream.codec_context.width)
            self.height = int(self.stream.codec_context.height)
            rate = self.stream.average_rate or self.stream.base_rate
            self.fps = float(rate) if rate is not None else 30.0
        except Exception as error:
            self.container = None
            if self.strict_nvdec:
                raise RuntimeError(
                    f"PyAV CUDA decode initialization failed: {error}"
                ) from error
            raise RuntimeError(
                "PyAV CUDA decode is unavailable; construct NvdecFrameSource for "
                "the explicit FFmpeg backend"
            ) from error
        self.capabilities = DecodeCapabilities(
            backend="pyav",
            hardware_requested=True,
            hardware_active=True,
            gpu_surface_to_ring=False,
            host_download_boundary=True,
            detail="PyAV CUDA frame converted to BGR for shared-memory transport",
        )

    def read(self) -> Optional[UInt8Array]:
        """Decode one frame and convert the CUDA frame at the ring boundary."""

        if self.container is None:
            self.start()
        if self.container is None or self.stream is None:
            return None
        try:
            frame = next(self.container.decode(self.stream))
        except StopIteration:
            return None
        array = np.asarray(frame.to_ndarray(format="bgr24"), dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] != 3:
            raise RuntimeError(
                f"PyAV returned an unexpected frame shape: {array.shape}"
            )
        return np.ascontiguousarray(array)

    def close(self, ignore_errors: bool = False) -> None:
        """Close the PyAV container."""

        if self.container is not None:
            self.container.close()
            self.container = None


@dataclass(frozen=True)
class StreamStats:
    """Counters from a ring producer."""

    frames: int
    elapsed_seconds: float
    decode_seconds: float
    ring_wait_seconds: float

    @property
    def fps(self) -> float:
        """Produced frames per second."""

        return self.frames / self.elapsed_seconds if self.elapsed_seconds > 0.0 else 0.0


class RingFrameProducer:
    """Feed a shared-memory ring from a hardware frame source."""

    def __init__(
        self,
        source: Any,
        ring: SharedMemoryFrameRing,
        tracker: Optional[FrameMetadataTracker] = None,
        *,
        start_index: int = 0,
        end_frame: Optional[int] = None,
    ) -> None:
        self.source = source
        self.ring = ring
        self.tracker = tracker or FrameMetadataTracker()
        self.start_index = int(start_index)
        self.end_frame = None if end_frame is None else int(end_frame)
        self.stats: Optional[StreamStats] = None
        self.error: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def run(self) -> StreamStats:
        """Produce until EOF, a requested frame limit, or shutdown."""

        started = time.perf_counter()
        decode_seconds = 0.0
        ring_wait_seconds = 0.0
        frames = 0
        frame_index = self.start_index
        try:
            self.source.start()
            while not self._stop.is_set():
                if self.end_frame is not None and frame_index >= self.end_frame:
                    break
                decode_started = time.perf_counter()
                frame = self.source.read()
                decode_seconds += time.perf_counter() - decode_started
                if frame is None:
                    break
                scene_cut = self.tracker.observe(frame)
                fps = max(1.0, float(getattr(self.source, "fps", 30.0)))
                metadata = FrameMetadata(
                    frame_index=frame_index,
                    timestamp=float(frame_index) / fps,
                    scene_cut=scene_cut,
                    width=int(frame.shape[1]),
                    height=int(frame.shape[0]),
                    sequence=0,
                )
                wait_started = time.perf_counter()
                self.ring.write(frame, metadata)
                ring_wait_seconds += time.perf_counter() - wait_started
                frame_index += 1
                frames += 1
            self.ring.finish()
        except BaseException as error:
            self.error = error
            self.ring.abort()
            raise
        finally:
            try:
                self.source.close(ignore_errors=self.error is not None)
            except BaseException as error:
                if self.error is None:
                    self.error = error
                    self.ring.abort()
                    raise
            self.stats = StreamStats(
                frames=frames,
                elapsed_seconds=time.perf_counter() - started,
                decode_seconds=decode_seconds,
                ring_wait_seconds=ring_wait_seconds,
            )
        return self.stats

    def start(self) -> threading.Thread:
        """Start ``run`` on a daemon thread and return the thread object."""

        if self._thread is not None and self._thread.is_alive():
            return self._thread
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._thread_main, name="nvdec-ring-producer", daemon=True
        )
        self._thread.start()
        return self._thread

    def _thread_main(self) -> None:
        try:
            self.run()
        except BaseException as error:
            _swallowed("roop/hardware_streamer.py:873", error, "fallback continued")
            self.error = error

    def stop(self) -> None:
        """Request shutdown and wake a blocked ring operation."""

        self._stop.set()
        self.ring.abort()
        try:
            self.source.close(ignore_errors=True)
        except Exception as _degrade_error:
            # The worker owns the definitive close/error path.  This call only
            # exists to interrupt a blocking FFmpeg/PyAV read during shutdown.
            _swallowed("roop/hardware_streamer.py:883", _degrade_error, "fallback continued")
            pass

    def join(self, timeout: Optional[float] = None) -> None:
        """Join the producer and surface its worker exception."""

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                raise TimeoutError("NVDEC ring producer did not stop")
        if self.error is not None:
            raise RuntimeError("NVDEC ring producer failed") from self.error

    def is_alive(self) -> bool:
        """Return whether the asynchronous producer thread is still running."""

        return self._thread is not None and self._thread.is_alive()


def iter_ring_packets(
    ring: SharedMemoryFrameRing, timeout: Optional[float] = None
) -> Iterator[FramePacket]:
    """Yield packets until the producer publishes its sentinel."""

    while True:
        packet = ring.read(timeout=timeout)
        if packet is None:
            return
        yield packet


__all__ = [
    "DecodeCapabilities",
    "FrameMetadata",
    "FrameMetadataTracker",
    "FramePacket",
    "NvdecFrameSource",
    "PyAvCudaSource",
    "RingFrameProducer",
    "RingStats",
    "SharedMemoryFrameRing",
    "SharedRingHandle",
    "StreamStats",
    "iter_ring_packets",
]
