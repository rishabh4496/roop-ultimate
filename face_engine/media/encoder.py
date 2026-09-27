"""NVENC encoding of GPU frames through an ffmpeg pipe.

:class:`NVENCVideoWriter` is :class:`~face_engine.media.ffmpeg_pipe.FFmpegWriter`
with ``h264_nvenc`` in place of libx264: same even-size pad, same
source-matrix conversion with ``accurate_rnd`` and full colour tagging, same
audio handling (stream copy bounded by ``-t N/fps``; never a bare
``-shortest``, which was measured dropping video frames), same output
verification. The video part follows the spec::

    -f rawvideo -pix_fmt rgb24 -r FPS -i -
    -c:v h264_nvenc -preset p4 -tune hq -rc vbr -cq 19 -b:v 0 -pix_fmt yuv420p
    -movflags +faststart

:meth:`NVENCVideoWriter.write_tensor` takes CUDA tensors: each frame is
converted to uint8 RGB on the GPU, copied into one of a ring of pinned host
buffers with ``non_blocking=True``, and a writer thread waits for that copy's
CUDA event and pushes the bytes into ffmpeg's stdin, so the caller's
inference thread never blocks on the pipe.

Measured 2026-09-28 (RTX 4070, 1080p H.264 source, 418 frames already on the
GPU, re-decoded and compared with the input):

    writer                                   fps    size     PSNR vs input
    libx264 medium crf 18 (FFmpegWriter)     170.5  10.1 MB  41.61 dB
    h264_nvenc p4 cq 19, rgb24 pipe          339.5  15.1 MB  41.96 dB

NVENC doubles encode throughput at equal fidelity for ~50% more bytes.
Feeding ffmpeg NV12 converted on the GPU was only ~10% faster (373.7 fps)
and came out 6 dB worse through ffmpeg's rawvideo NV12 input, although the
conversion itself round-trips at 52.8 dB in memory, so the RGB pipe is kept.

When NVENC is unavailable (no NVIDIA GPU, driver, or a busy encoder session
limit) :func:`open_video_writer` falls back to libx264 and says so.
"""
from __future__ import annotations

import logging
import queue
import subprocess
import threading
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from face_engine.media.ffmpeg_pipe import FFmpegError, FFmpegWriter
from face_engine.media.tools import find_tool

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


# h264_nvenc refuses smaller frames ("Frame Dimension less than the minimum
# supported value"); probed on an RTX 4070, 2026-09-28: 145x49 opens,
# 144x48 / 256x48 / 128x128 do not.
NVENC_MIN_WIDTH, NVENC_MIN_HEIGHT = 145, 49

# ITU-T H.264 Table E-3/E-4/E-5 codes for the names FFmpegWriter uses.
_H264_CODE = {"bt709": 1, "smpte170m": 6, "bt2020": 9, "bt2020nc": 9, "bt601": 6}


@cache
def nvenc_available(codec: str = "h264_nvenc") -> bool:
    """True when ffmpeg can actually open ``codec`` (a 1-frame test encode)."""
    cmd = [find_tool("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
           "color=black:s=256x256:d=0.04", "-frames:v", "1", "-c:v", codec, "-f", "null", "-"]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=60, check=False).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class TensorWriterMixin:
    """``write_tensor`` for FFmpegWriter subclasses: GPU -> pinned -> pipe, on a thread."""

    #: Pinned staging buffers (frames in flight between GPU and pipe).
    pinned_buffers = 4

    def _start_tensor_path(self) -> None:
        self._queue: queue.Queue[Any] = queue.Queue()
        self._free: queue.Queue[int] = queue.Queue()
        self._pinned: list[Any] = []
        self._thread_error: BaseException | None = None
        self._writer = threading.Thread(target=self._pump, name="nvenc-writer", daemon=True)
        self._writer.start()

    def _pump(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            slot, event = item
            try:
                if self._thread_error is None:
                    event.synchronize()
                    FFmpegWriter.write(self, self._pinned[slot].numpy())  # type: ignore[arg-type]
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
                self._thread_error = exc
            finally:
                self._free.put(slot)

    def _raise_thread_error(self) -> None:
        if self._thread_error is not None:
            raise self._thread_error

    def write_tensor(self, frames: torch.Tensor) -> None:
        """Queue BGR frames living on the GPU.

        Accepts ``(3, H, W)``, ``(H, W, 3)`` or a batch ``(B, 3, H, W)``; uint8,
        or float in ``[0, 255]``. Returns once the frames are queued; the GPU
        -> host copy and the pipe write happen on the writer thread.
        """
        import torch

        self._raise_thread_error()
        batch = frames if frames.ndim == 4 else frames[None]
        if batch.shape[-1] == 3 and batch.shape[1] != 3:
            batch = batch.permute(0, 3, 1, 2)
        h, w = self.height, self.width  # type: ignore[attr-defined]
        if tuple(batch.shape[1:]) != (3, h, w):
            raise ValueError(f"expected (3, {h}, {w}) frames, got {tuple(batch.shape[1:])}")
        order = [2, 1, 0] if self.input_pix_fmt == "rgb24" else [0, 1, 2]  # type: ignore[attr-defined]
        for frame in batch:
            hwc = frame[order].permute(1, 2, 0)
            if hwc.dtype != torch.uint8:
                hwc = hwc.round().clamp(0, 255).to(torch.uint8)
            if len(self._pinned) < self.pinned_buffers and self._free.empty():
                self._pinned.append(torch.empty((h, w, 3), dtype=torch.uint8).pin_memory())
                slot = len(self._pinned) - 1
            else:
                slot = self._free.get()
                self._raise_thread_error()
            self._pinned[slot].copy_(hwc, non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(frame.device))
            self._queue.put((slot, event))

    def _stop_tensor_path(self) -> None:
        self._queue.put(None)
        self._writer.join()


class NVENCVideoWriter(TensorWriterMixin, FFmpegWriter):
    """:class:`FFmpegWriter` with H.264 NVENC (spec settings) and GPU-tensor input.

    Args (beyond :class:`FFmpegWriter`'s):
        cq: NVENC constant-quality target (``-cq``; 19 = spec).
        nvenc_preset: ``p1`` (fastest) .. ``p7`` (best); ``p4`` = spec.
    """

    input_pix_fmt = "rgb24"

    def __init__(self, *args: Any, cq: int = 19, nvenc_preset: str = "p4", **kwargs: Any) -> None:
        self.cq = cq
        self.nvenc_preset = nvenc_preset
        super().__init__(*args, **kwargs)
        self._start_tensor_path()

    def _video_args(self, crf: int, preset: str) -> list[str]:
        return ["-c:v", "h264_nvenc", "-preset", self.nvenc_preset, "-tune", "hq", "-rc", "vbr",
                "-cq", str(self.cq), "-b:v", "0", "-pix_fmt", "yuv420p"]

    def _vui_args(self, space: str, primaries: str, trc: str) -> list[str]:
        # ffmpeg 8.1's h264_nvenc writes the matrix but not primaries/transfer
        # (read back as unknown, 2026-09-28), like libx264; h264_metadata sets
        # all four in the H.264 VUI.
        vui = (f"h264_metadata=colour_primaries={_H264_CODE[primaries]}"
               f":transfer_characteristics={_H264_CODE[trc]}"
               f":matrix_coefficients={_H264_CODE[space]}:video_full_range_flag=0")
        return ["-bsf:v", vui]

    def close(self, timeout: float = 600.0, verify: bool = True) -> Any:
        self._stop_tensor_path()
        self._raise_thread_error()
        return super().close(timeout=timeout, verify=verify)

    def abort(self) -> None:
        self._stop_tensor_path()
        super().abort()


class X264TensorWriter(TensorWriterMixin, FFmpegWriter):
    """The libx264 writer with the same ``write_tensor`` path (NVENC fallback)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._start_tensor_path()

    def close(self, timeout: float = 600.0, verify: bool = True) -> Any:
        self._stop_tensor_path()
        self._raise_thread_error()
        return super().close(timeout=timeout, verify=verify)

    def abort(self) -> None:
        self._stop_tensor_path()
        super().abort()


def open_video_writer(path: str | Path, width: int, height: int, fps: Any, *,
                      encoder: str = "auto", **kwargs: Any) -> FFmpegWriter:
    """An NVENC writer, or libx264 when NVENC is unavailable (``encoder="auto"``).

    ``encoder``: ``"auto"``, ``"nvenc"`` (raise if unavailable) or ``"x264"``.
    Both returned writers accept :meth:`TensorWriterMixin.write_tensor`.
    """
    if encoder not in ("auto", "nvenc", "x264"):
        raise ValueError("encoder must be 'auto', 'nvenc' or 'x264'")
    padded_w, padded_h = width + width % 2, height + height % 2
    fits = padded_w >= NVENC_MIN_WIDTH and padded_h >= NVENC_MIN_HEIGHT
    if encoder != "x264" and fits and nvenc_available():
        return NVENCVideoWriter(path, width, height, fps, **kwargs)
    if encoder == "nvenc":
        reason = ("is not available on this machine" if fits else
                  f"needs at least {NVENC_MIN_WIDTH}x{NVENC_MIN_HEIGHT}, got {width}x{height}")
        raise FFmpegError(f"h264_nvenc {reason}")
    if encoder == "auto":
        logger.warning("h264_nvenc %s; encoding %s with libx264",
                       "unavailable" if fits else f"cannot encode {width}x{height}",
                       Path(path).name)
    kwargs.pop("cq", None)
    kwargs.pop("nvenc_preset", None)
    return X264TensorWriter(path, width, height, fps, **kwargs)
