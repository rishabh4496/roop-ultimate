"""Video decoding into batches of CUDA tensors: software (default) or NVDEC.

:class:`HardwareVideoDecoder` yields ``(B, 3, H, W)`` uint8 BGR CUDA batches.
Two backends:

* ``"software"`` (default): PyAV decodes on a background thread into
  preallocated pinned buffers while the caller works on the previous batch.
* ``"nvdec"``: NVDEC through PyAV (``hwaccel=cuda``) in a CHILD PROCESS,
  pictures downloaded as NV12, passed through a shared-memory ring, and
  converted NV12 -> BGR on the GPU (the upload moves 1.5 bytes/pixel).

End to end the software decoder wins on this machine, so it is the default
(``face_engine/tests/verify_video_pipeline.py``, 1080p H.264, 300 frames, RTX
4070, 2026-09-28, whole-run fps):

    I/O                                          pass-through   face swap
    software decode -> host -> libx264               74.3          25.3
    NVDEC (child) -> GPU -> NVENC                    68.8          29.5
    software decode (thread) -> GPU -> NVENC        161.9          44.1

NVENC is the gain; NVDEC is not. Alone, the NVDEC process delivers 176-180
fps against software's 239-241; with detection running on the GPU it drops
to 56 fps against software's 96, because a second process's CUDA context
time-slices the GPU with the inference process. ``backend="nvdec"`` remains
for decode-only work.

How the NVDEC path is built (decode-only, 1080p, 418 frames, same machine):

    PyAV software decode -> BGR on host                         245 fps
    PyAV NVDEC -> NV12 on host (no colour conversion)            411 fps (H.264) / 496 (HEVC)
    PyAV NVDEC -> BGR on host (PyAV/swscale converts)            55 fps
    PyAV h264_cuvid codec -> BGR on host                         55 fps
    ffmpeg CLI (software or -hwaccel cuda) -> BGR pipe           65-70 fps

NVDEC itself is fast; converting its NV12 on the CPU is what made
"hardware decoding" 4x slower than software, here and in the earlier 720p
measurement in :mod:`face_engine.media.capturer`. PyAV 17 exposes no
DLPack / CUDA-pointer route for hardware frames, so a true device-to-device
path is not available through PyAV; the NV12 download is the cheapest
copy left.

Colour: the NV12 -> BGR matrix and range come from the stream's own tags
(:class:`~face_engine.media.capturer.ColorProfile`; BT.709 for HD, BT.601
for SD, limited range unless tagged ``pc``). Against PyAV's software BGR the
GPU result differs by +1.0 / +1.5 / +1.0 levels mean (B, G, R) and at most
3: swscale's default rounding biases its own output low (the Stage 4 writer
notes measured -1.6 levels for the same reason).

NVDEC runs in its OWN PROCESS. With ffmpeg's NVDEC decoding on a thread of
the process that also runs torch + ONNX Runtime CUDA work, the GPU work
deadlocked after ~100 frames (a stream fence waited forever; reproduced with
SCRFD alone, 2026-09-28; software decoding on the same thread ran clean).
ffmpeg's NVDEC creates its own CUDA context and PyAV's build rejects
``primary_ctx`` (error -129), so the contexts cannot be shared; a separate
process keeps them apart. Frames cross as NV12 through
:class:`~face_engine.media.ipc_pool.SharedMemoryRingBuffer` (zero-copy
slots; one host copy into pinned staging), and the availability probe runs
in a child as well, so the rendering process never creates an ffmpeg CUDA
context. Software decoding (no CUDA) stays on a thread.

NVDEC fallbacks (``"auto"``), each logged and reported by
:attr:`HardwareVideoDecoder.backend`:
no NVDEC (driver, codec, or PyAV without CUDA) -> software; a stream NVDEC
does not hand back as 8-bit NV12 (10-bit, 4:2:2/4:4:4) -> software BGR
through PyAV (tag-honouring); HDR -> refused (8-bit BGR cannot carry it).

Frames are matched to presentation-order indices through
:class:`~face_engine.media.capturer.VideoSource`'s packet index, so
``start``/``end`` decode exactly the frames a sequential decode would.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from face_engine.media.capturer import ColorProfile, VideoInfo, VideoSource
from face_engine.media.ipc_pool import RingAborted, SharedMemoryRingBuffer

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

_KR_KB = {ColorProfile.BT709: (0.2126, 0.0722), ColorProfile.BT601: (0.299, 0.114)}


@dataclass
class FrameBatch:
    """Consecutive decoded frames on the GPU.

    Attributes:
        frames: ``(B, 3, H, W)`` uint8 BGR CUDA tensor.
        indices: Presentation-order frame numbers (length B).
        pts: Presentation timestamps (stream time base).
    """

    frames: Any
    indices: list[int]
    pts: list[int]

    def __len__(self) -> int:
        return len(self.indices)


def ycbcr_to_bgr(y: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
                 profile: ColorProfile = ColorProfile.BT709, full_range: bool = False,
                 dim: int = 1) -> torch.Tensor:
    """Full-resolution float Y/Cb/Cr planes -> float BGR ``[0, 255]`` stacked on ``dim``.

    The matrix and range follow the stream's tags (see the module docstring).
    """
    import torch

    kr, kb = _KR_KB.get(profile, _KR_KB[ColorProfile.BT709])
    kg = 1.0 - kr - kb
    if full_range:
        cb, cr = u - 128.0, v - 128.0
    else:
        y = (y - 16.0) * (255.0 / 219.0)
        cb, cr = (u - 128.0) * (255.0 / 224.0), (v - 128.0) * (255.0 / 224.0)
    red = y + (2.0 * (1.0 - kr)) * cr
    blue = y + (2.0 * (1.0 - kb)) * cb
    green = (y - kr * red - kb * blue) / kg
    return torch.stack([blue, green, red], dim=dim).round_().clamp_(0, 255)


def nv12_to_bgr(nv12: torch.Tensor, height: int, width: int,
                profile: ColorProfile = ColorProfile.BT709, full_range: bool = False,
                out: torch.Tensor | None = None) -> torch.Tensor:
    """``(B, H*3/2, W)`` uint8 NV12 -> ``(B, 3, H, W)`` uint8 BGR, on the tensor's device.

    Chroma is upsampled nearest-neighbour (what swscale does by default, and
    measured closer to PyAV than bilinear: 1.17 vs 1.24 levels mean).
    """
    import torch

    x = nv12.float()
    b = x.shape[0]
    y = x[:, :height]
    uv = x[:, height:height + height // 2].reshape(b, height // 2, width // 2, 2)
    u = uv[..., 0].repeat_interleave(2, 1).repeat_interleave(2, 2)
    v = uv[..., 1].repeat_interleave(2, 1).repeat_interleave(2, 2)
    bgr = ycbcr_to_bgr(y, u, v, profile, full_range, dim=1)
    if out is not None:
        out.copy_(bgr)
        return out
    return bgr.to(torch.uint8)


def _open_nvdec(path: str) -> Any:
    import av
    from av.codec.hwaccel import HWAccel

    return av.open(path, hwaccel=HWAccel(device_type="cuda", allow_software_fallback=False))


def _nvdec_probe(path: str, result: Any) -> None:
    """Child: decode one frame on NVDEC; report its pixel format or the error."""
    try:
        with _open_nvdec(path) as container:
            frame = next(container.decode(video=0))
            result.put(("ok", frame.format.name, frame.width, frame.height))
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        result.put(("error", f"{type(exc).__name__}: {exc}", 0, 0))


def _nvdec_worker(path: str, pts_list: list[int], first_index: int, seek_pts: int,
                  ring: SharedMemoryRingBuffer, errors: Any) -> None:
    """Child: decode frames on NVDEC and commit NV12 into the ring (seq = frame index)."""
    pts_to_index = {p: i for i, p in enumerate(pts_list, first_index)}
    last_pts = pts_list[-1]
    wanted = len(pts_list)
    emitted = 0
    try:
        with _open_nvdec(path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                i = pts_to_index.get(int(frame.pts))
                if i is None:
                    if frame.pts > last_pts:
                        break
                    continue
                data = frame.to_ndarray(format="nv12")
                slot = ring.acquire_write(timeout=None)
                np.copyto(slot.array, data)
                ring.commit(slot, i)
                emitted += 1
                if emitted == wanted:
                    break
        if emitted != wanted:
            raise RuntimeError(f"decoded {emitted} of {wanted} frames")
        ring.close()
    except RingAborted:
        pass
    except BaseException:  # noqa: BLE001 - reported to the parent with the traceback
        errors.put(traceback.format_exc())
        ring.abort()


class HardwareVideoDecoder:
    """Decode a video (or a keyframe-aligned range of it) into CUDA batches.

    Args:
        path: Video file.
        batch_size: Frames per :class:`FrameBatch` (the last may be shorter).
        device: CUDA device for the output tensors.
        backend: ``"software"`` (default; see the module docstring),
            ``"nvdec"`` (raise if unavailable) or ``"auto"`` (NVDEC when
            available, else software).
        start, end: Frame range ``[start, end)`` in presentation order.
        prefetch: Batches decoded ahead by the background thread.

    Iterate for batches; :attr:`backend` says what actually decoded.
    """

    def __init__(self, path: str | Path, *, batch_size: int = 4, device: Any = "cuda",
                 backend: str = "software", start: int = 0, end: int | None = None,
                 prefetch: int = 3) -> None:
        if backend not in ("auto", "nvdec", "software"):
            raise ValueError("backend must be 'auto', 'nvdec' or 'software'")
        if batch_size < 1 or prefetch < 1:
            raise ValueError("batch_size and prefetch must be >= 1")
        self.path = Path(path)
        self.batch_size = batch_size
        self.device = device
        self.requested = backend
        self.prefetch = prefetch
        self.source = VideoSource(self.path)
        info = self.source.info
        if info.color_profile.is_hdr:
            raise ValueError(f"{self.path.name} is {info.color_profile.value}; an 8-bit BGR "
                             "pipeline cannot carry HDR (tone-map first)")
        total = self.source.frame_count
        self.start = max(0, int(start))
        self.end = total if end is None else min(int(end), total)
        self.backend = "software" if backend == "software" else self._probe_nvdec(backend)
        self.frames_decoded = 0

    # ------------------------------------------------------------------ metadata
    @property
    def info(self) -> VideoInfo:
        """Width, height, frame count, exact fps, sample aspect ratio, colour, tracks."""
        return self.source.info

    @property
    def fps(self) -> Fraction:
        return self.info.fps

    @property
    def frame_count(self) -> int:
        return self.end - self.start

    @property
    def sample_aspect_ratio(self) -> Fraction:
        return self.info.sample_aspect_ratio

    @property
    def full_range(self) -> bool:
        return self.info.color_tags.get("color_range") in ("pc", "jpeg")

    # ------------------------------------------------------------------ backend
    def _open(self) -> Any:
        import av

        return av.open(str(self.path))

    def _probe_nvdec(self, backend: str) -> str:
        """Decode one frame on NVDEC (in a child process) and check it is 8-bit NV12."""
        try:
            ctx = mp.get_context("spawn")
            result = ctx.Queue()
            proc = ctx.Process(target=_nvdec_probe, args=(str(self.path), result), daemon=True)
            proc.start()
            try:
                status, fmt, width, height = result.get(timeout=120)
            finally:
                proc.join(timeout=30)
            if status != "ok":
                raise RuntimeError(fmt)
            if fmt != "nv12" or width % 2 or height % 2:
                raise RuntimeError(f"NVDEC returned {fmt} {width}x{height}; "
                                   "only 8-bit NV12 with even sides is converted on the GPU")
            return "nvdec"
        except Exception as exc:
            if backend == "nvdec":
                raise RuntimeError(f"NVDEC unavailable for {self.path.name}: {exc}") from exc
            logger.info("NVDEC unavailable for %s (%s); decoding in software", self.path.name, exc)
            return "software"

    # ------------------------------------------------------------------ decode thread
    def _producer(self, slots: list[np.ndarray], free: queue.Queue[int],
                  ready: queue.Queue[Any], stop: threading.Event) -> None:
        idx = self.source.index
        pts_to_index = {int(p): i for i, p in enumerate(idx.pts[self.start:self.end], self.start)}
        keys = idx.keyframes[idx.keyframes <= self.start]
        seek_pts = int(idx.pts[keys[-1]]) if keys.size else int(idx.pts[0])
        last_pts = int(idx.pts[self.end - 1])
        wanted = self.end - self.start
        emitted = 0
        slot = -1
        fill = 0
        indices: list[int] = []
        pts: list[int] = []
        try:
            with self._open() as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
                for frame in container.decode(stream):
                    if stop.is_set():
                        return
                    if frame.pts is None:
                        continue
                    i = pts_to_index.get(int(frame.pts))
                    if i is None:
                        if frame.pts > last_pts:
                            break
                        continue
                    if slot < 0:
                        slot = free.get()
                        if slot < 0:
                            return
                    slots[slot][fill] = frame.to_ndarray(format="bgr24")
                    indices.append(i)
                    pts.append(int(frame.pts))
                    fill += 1
                    emitted += 1
                    if fill == self.batch_size or emitted == wanted:
                        ready.put((slot, fill, indices, pts))
                        slot, fill, indices, pts = -1, 0, [], []
                    if emitted == wanted:
                        break
            if emitted != wanted:
                raise RuntimeError(f"decoded {emitted} of {wanted} frames "
                                   f"[{self.start}, {self.end}) from {self.path.name}")
            ready.put(None)
        except BaseException as exc:  # noqa: BLE001 - surfaced in the consumer thread
            ready.put(exc)

    # ------------------------------------------------------------------ iterate
    def __iter__(self) -> Iterator[FrameBatch]:
        if self.frame_count <= 0:
            return
        if self.backend == "nvdec":
            yield from self._iter_nvdec()
        else:
            yield from self._iter_software()

    def _iter_nvdec(self) -> Iterator[FrameBatch]:
        import torch

        info = self.info
        h, w = info.height, info.width
        idx = self.source.index
        keys = idx.keyframes[idx.keyframes <= self.start]
        seek_pts = int(idx.pts[keys[-1]]) if keys.size else int(idx.pts[0])
        pts_list = [int(p) for p in idx.pts[self.start:self.end]]
        ctx = mp.get_context("spawn")
        ring = SharedMemoryRingBuffer(self.prefetch * self.batch_size, (h * 3 // 2, w), ctx=ctx)
        errors = ctx.Queue()
        proc = ctx.Process(target=_nvdec_worker, name="nvdec-decode",
                           args=(str(self.path), pts_list, self.start, seek_pts, ring, errors),
                           daemon=True)
        proc.start()
        pinned = torch.empty((self.batch_size, h * 3 // 2, w), dtype=torch.uint8).pin_memory()
        host = pinned.numpy()
        staging = torch.empty(pinned.shape, dtype=torch.uint8, device=self.device)
        stream = torch.cuda.current_stream(torch.device(self.device))
        pts_of = dict(zip(range(self.start, self.end), pts_list))

        def failure() -> RuntimeError:
            try:
                detail = errors.get(timeout=2)
            except queue.Empty:
                detail = f"NVDEC process exited with code {proc.exitcode}"
            return RuntimeError(f"NVDEC decoding of {self.path.name} failed:\n{detail}")

        try:
            fill, indices = 0, []
            while True:
                try:
                    slot = ring.acquire_read(timeout=1.0)
                except TimeoutError:
                    if not proc.is_alive():
                        raise failure() from None
                    continue
                except RingAborted:
                    raise failure() from None
                if slot is not None:
                    np.copyto(host[fill], slot.array)
                    ring.release(slot)
                    indices.append(slot.seq)
                    fill += 1
                if fill and (fill == self.batch_size or slot is None):
                    staging[:fill].copy_(pinned[:fill], non_blocking=True)
                    frames = nv12_to_bgr(staging[:fill], h, w, info.color_profile,
                                         self.full_range)
                    stream.synchronize()  # pinned staging is refilled next
                    self.frames_decoded += fill
                    yield FrameBatch(frames, indices, [pts_of[i] for i in indices])
                    fill, indices = 0, []
                if slot is None:
                    break
            if self.frames_decoded != self.frame_count:
                raise RuntimeError(f"NVDEC delivered {self.frames_decoded} of "
                                   f"{self.frame_count} frames")
        finally:
            if proc.is_alive():
                ring.abort()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
            ring.destroy()

    def _iter_software(self) -> Iterator[FrameBatch]:
        import torch

        info = self.info
        h, w = info.height, info.width
        shape = (self.batch_size, h, w, 3)
        slots = [torch.empty(shape, dtype=torch.uint8).pin_memory() for _ in range(self.prefetch)]
        host = [s.numpy() for s in slots]
        staging = torch.empty(shape, dtype=torch.uint8, device=self.device)
        free: queue.Queue[int] = queue.Queue()
        for i in range(self.prefetch):
            free.put(i)
        ready: queue.Queue[Any] = queue.Queue()
        stop = threading.Event()
        worker = threading.Thread(target=self._producer, args=(host, free, ready, stop),
                                  name="software-decode", daemon=True)
        worker.start()
        stream = torch.cuda.current_stream(torch.device(self.device))
        try:
            while True:
                item = ready.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                slot, count, indices, pts = item
                staging[:count].copy_(slots[slot][:count], non_blocking=True)
                done = torch.cuda.Event()
                done.record(stream)
                frames = staging[:count].permute(0, 3, 1, 2).contiguous()
                done.synchronize()  # the pinned slot may be refilled once the copy landed
                free.put(slot)
                self.frames_decoded += count
                yield FrameBatch(frames, indices, pts)
        finally:
            stop.set()
            free.put(-1)
            worker.join(timeout=10)
