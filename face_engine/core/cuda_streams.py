"""Triple-buffered decode / inference / encode on three CUDA streams.

:class:`CUDAStreamPipeline` renders a video with three stages running at once,
each on its own host thread and its own ``torch.cuda.Stream``:

* **decode** (``stream_decode``): PyAV decodes frame N+1 on the CPU (software;
  NVDEC loses once inference shares the GPU and deadlocks in-process, see
  :mod:`face_engine.media.decoder`). Its YUV 4:2:0 planes are copied into a
  pinned staging buffer, uploaded (1.5 bytes/pixel) and converted to BGR on
  the GPU, into a VRAM slot of the :class:`CUDARingBuffer`.
* **inference** (``stream_inference``): frame N - detection, matching, kornia
  crop warps, the swap / enhancer networks (TensorRT engine or ONNX Runtime
  IOBinding) and masks: :meth:`GpuFrameProcessor.infer`.
* **encode** (``stream_encode``): frame N-1 - inverse warps and compositing
  (:meth:`GpuFrameProcessor.composite`), conversion to the writer's uint8
  RGB/BGR back into the frame's slot, and one device -> pinned copy; a fourth
  thread waits for that copy and writes it into ffmpeg (NVENC, libx264
  fallback).

Hand-offs between streams are ``torch.cuda.Event`` s, never a device-wide
``torch.cuda.synchronize()``::

    event_decoded[k].record(stream_decode)      stream_inference.wait_event(event_decoded[k])
    event_inferred[k].record(stream_inference)  stream_encode.wait_event(event_inferred[k])
    event_released[k].record(stream_encode)     stream_decode.wait_event(event_released[k])

The third pair closes the ring: slot ``k`` is refilled only after the encode
stream's copy out of it. Host threads block only on the event that guards a
specific pinned host buffer before rewriting it (a CPU write cannot wait on a
stream).

Memory: the VRAM slots, the pinned ingress (YUV) and egress (RGB) buffers
and the GPU staging planes are allocated once, before the first frame. The
per-frame loop allocates no frame-sized host memory (``tests/test_cuda_streams.py``
checks it with ``tracemalloc``); PyAV's decoded pictures come from libav's own
frame pool. Pixel formats other than 8-bit YUV 4:2:0 fall back to PyAV's BGR
conversion, which does allocate per frame; :attr:`PipelineStats.host_bgr_frames`
counts them.

Colour: YUV -> BGR uses the stream's matrix and range
(:func:`~face_engine.media.decoder.ycbcr_to_bgr`, the NVDEC path's
conversion). It differs from PyAV/swscale's software BGR by ~1 level
(swscale rounds low; measured for NVDEC, Stage 4).
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

#: Pixel formats whose planes are uploaded as they are (8-bit 4:2:0 planar).
PLANAR_420 = frozenset({"yuv420p", "yuvj420p"})
_POLL = 0.1  # seconds between checks for a failure / cancel while a queue is empty


class PipelineCancelled(RuntimeError):
    """The caller's ``cancel`` event was set."""


class CUDARingBuffer:
    """``capacity`` frame slots in VRAM, their pinned host staging, and hand-off events.

    Args:
        capacity: Frames in flight (3: one being ingested, one processed, one
            encoded).
        frame_shape: ``(H, W, 3)``.
        dtype: Slot element type (``torch.uint8``).
        device: CUDA device.

    Attributes:
        slots: ``(capacity, H, W, 3)`` contiguous VRAM tensor; ``slots[k]`` is
            frame BGR after decode and writer-order RGB/BGR after encode.
        ingress: ``(capacity, H*W + 2*ch*cw)`` pinned uint8: Y, U, V planes.
        egress: ``(capacity, H, W, 3)`` pinned uint8 (ffmpeg's input).
        planes: ``(capacity, H*W + 2*ch*cw)`` VRAM copy of ``ingress``.
        event_decoded / event_inferred / event_released: one per slot.
    """

    def __init__(self, capacity: int = 3, frame_shape: tuple[int, int, int] = (1080, 1920, 3),
                 dtype: Any = None, device: Any = "cuda") -> None:
        import torch

        if capacity < 2:
            raise ValueError("capacity must be >= 2")
        h, w, c = frame_shape
        if c != 3:
            raise ValueError("frame_shape must be (H, W, 3)")
        self.capacity = capacity
        self.frame_shape = (h, w, c)
        self.device = torch.device(device)
        dtype = torch.uint8 if dtype is None else dtype
        self.chroma_shape = ((h + 1) // 2, (w + 1) // 2)
        ch, cw = self.chroma_shape
        self.plane_bytes = h * w + 2 * ch * cw
        self.slots = torch.empty((capacity, h, w, c), dtype=dtype, device=self.device)
        self.planes = torch.empty((capacity, self.plane_bytes), dtype=torch.uint8,
                                  device=self.device)
        self.ingress = torch.empty((capacity, self.plane_bytes), dtype=torch.uint8).pin_memory()
        self.egress = torch.empty((capacity, h, w, c), dtype=torch.uint8).pin_memory()
        self.ingress_np = self.ingress.numpy()
        self.egress_np = self.egress.numpy()
        self.event_decoded = [torch.cuda.Event() for _ in range(capacity)]
        self.event_inferred = [torch.cuda.Event() for _ in range(capacity)]
        self.event_released = [torch.cuda.Event() for _ in range(capacity)]
        self.event_uploaded = [torch.cuda.Event() for _ in range(capacity)]  # ingress[k] reusable
        self.event_downloaded = [torch.cuda.Event() for _ in range(capacity)]  # egress[k] ready

    def host_planes(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Views of ``ingress[k]`` as ``(Y (H, W), U (ch, cw), V (ch, cw))``."""
        h, w, _ = self.frame_shape
        ch, cw = self.chroma_shape
        flat = self.ingress_np[k]
        return (flat[:h * w].reshape(h, w), flat[h * w:h * w + ch * cw].reshape(ch, cw),
                flat[h * w + ch * cw:].reshape(ch, cw))

    @property
    def nbytes(self) -> dict[str, int]:
        """Bytes allocated: VRAM (slots + planes) and pinned host (ingress + egress)."""
        vram = self.slots.numel() * self.slots.element_size() + self.planes.numel()
        return {"vram": int(vram), "pinned": int(self.ingress.numel() + self.egress.numel())}


def i420_to_bgr_hwc(planes: torch.Tensor, height: int, width: int, profile: Any,
                    full_range: bool, out: torch.Tensor) -> torch.Tensor:
    """Flat uint8 Y/U/V planes on the GPU -> ``out`` ``(H, W, 3)`` uint8 BGR."""
    from face_engine.media.decoder import ycbcr_to_bgr

    ch, cw = (height + 1) // 2, (width + 1) // 2
    x = planes.float()
    y = x[:height * width].view(height, width)
    u = x[height * width:height * width + ch * cw].view(ch, cw)
    v = x[height * width + ch * cw:].view(ch, cw)
    u = u.repeat_interleave(2, 0).repeat_interleave(2, 1)[:height, :width]
    v = v.repeat_interleave(2, 0).repeat_interleave(2, 1)[:height, :width]
    out.copy_(ycbcr_to_bgr(y, u, v, profile, full_range, dim=2))
    return out


def write_slot(frame: torch.Tensor, slot: torch.Tensor, rgb: bool) -> None:
    """``(1, 3, H, W)`` float BGR -> ``slot`` ``(H, W, 3)`` uint8 in the writer's channel order."""
    chw = frame[0].flip(0) if rgb else frame[0]
    slot.copy_(chw.permute(1, 2, 0).round().clamp_(0, 255))


@dataclass
class PipelineStats:
    """Progress and per-stage host time (seconds) of one :meth:`CUDAStreamPipeline.run`.

    ``*_s`` is time a stage's thread spent working (decode includes PyAV;
    inference and encode include kernel launches and any host syncs inside
    the models / kornia). ``*_wait_s`` is time it sat idle waiting for the
    previous stage (starved) or a free slot (back-pressure).
    """

    frames_total: int = 0
    frames_done: int = 0
    faces: int = 0
    swapped: int = 0
    fps: float = 0.0
    elapsed: float = 0.0
    host_bgr_frames: int = 0
    passthrough_frames: int = 0
    decode_s: float = 0.0
    decode_wait_s: float = 0.0
    infer_s: float = 0.0
    infer_wait_s: float = 0.0
    encode_s: float = 0.0
    encode_wait_s: float = 0.0
    write_s: float = 0.0
    write_wait_s: float = 0.0
    output: str | None = None
    done: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class _Stop(Exception):
    """Internal: another stage failed or the run was cancelled."""


class CUDAStreamPipeline:
    """Render videos through decode / inference / encode running concurrently.

    Args:
        capacity: Ring slots (frames in flight). 3 = triple buffering.
        device: CUDA device.
        encoder: ``"auto"`` (NVENC, libx264 fallback), ``"nvenc"`` or ``"x264"``.
        remux: Mux the source's audio, subtitles and chapters into the output
            (lossless; :func:`~face_engine.media.demuxer.remux`).
        gpu_color: Convert YUV -> BGR on the GPU (default). False uses PyAV's
            BGR (the sequential render's decode, bit for bit; allocates a host
            frame per frame).
        inference_priority: CUDA priority of the inference stream (lower is
            higher; ``torch.cuda.Stream.priority_range()``). None (default) =
            the highest the device offers; decode and encode stay at 0. It
            reaches torch kernels and AOT TensorRT engines (ONNX Runtime
            computes on its own stream). Without it the decode stream's colour
            conversion delayed inference: Ultra Fast 47.4 vs 51.8 fps
            (2026-09-28, see the README).
    """

    def __init__(self, capacity: int = 3, device: Any = "cuda", encoder: str = "auto",
                 remux: bool = True, gpu_color: bool = True,
                 inference_priority: int | None = None) -> None:
        self.gpu_color = gpu_color
        self.inference_priority = inference_priority
        self.capacity = capacity
        self.device = device
        self.encoder = encoder
        self.remux = remux

    # ------------------------------------------------------------------ public
    def run(self, source_path: str | Path, target_path: str | Path, pipeline_config: Any = None,
            *, cancel: threading.Event | None = None,
            on_progress: Callable[[PipelineStats], None] | None = None,
            progress_interval: float = 0.25, max_frames: int | None = None) -> PipelineStats:
        """Render ``source_path`` into ``target_path``; blocks until done.

        ``pipeline_config``: a :class:`~face_engine.server.processing.ProcessorConfig`
        (a :class:`GpuFrameProcessor` is built and closed here), any object
        with ``infer(frame) -> plan`` / ``composite(plan) -> frame`` (reused,
        not closed), or None to pass frames through unchanged.
        ``max_frames`` renders only the first frames (benchmarks).
        """
        from face_engine.media.capturer import VideoSource

        source_path, target_path = Path(source_path), Path(target_path)
        src = VideoSource(source_path)
        info = src.info
        if info.color_profile.is_hdr:
            raise ValueError(f"{source_path.name} is {info.color_profile.value}; an 8-bit "
                             "pipeline cannot carry HDR (tone-map first)")
        total = src.frame_count if max_frames is None else min(src.frame_count, max_frames)
        stats = PipelineStats(frames_total=total)
        processor, owned = self._processor(pipeline_config)
        try:
            return self._render(src, total, stats, processor, source_path, target_path,
                                cancel, on_progress, progress_interval)
        finally:
            if owned:
                processor.close()

    def _render(self, src: Any, total: int, stats: PipelineStats, processor: Any,
                source_path: Path, target_path: Path, cancel: threading.Event | None,
                on_progress: Callable[[PipelineStats], None] | None,
                progress_interval: float) -> PipelineStats:
        import torch

        from face_engine.media.demuxer import remux
        from face_engine.media.encoder import open_video_writer

        info = src.info
        device = torch.device(self.device)
        if processor is not None and hasattr(processor, "prepare"):
            processor.prepare(info.height, info.width)  # e.g. CUDA graph captures (device syncs)
        ring = CUDARingBuffer(self.capacity, (info.height, info.width, 3), torch.uint8, device)
        top = (torch.cuda.Stream.priority_range()[1] if self.inference_priority is None
               else self.inference_priority)
        streams = {name: torch.cuda.Stream(device=device,
                                           priority=top if name == "inference" else 0)
                   for name in ("decode", "inference", "encode")}
        video = target_path.with_name(f".{target_path.stem}_video.mp4") if self.remux \
            else target_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        writer = open_video_writer(video, info.width, info.height, info.fps,
                                   expected_frames=total, color=info.color_profile,
                                   encoder=self.encoder)
        rgb = writer.input_pix_fmt == "rgb24"
        cancel = cancel or threading.Event()
        failure: list[BaseException] = []
        stop = threading.Event()
        free: queue.Queue[int] = queue.Queue()
        for k in range(self.capacity):
            free.put(k)
        decoded: queue.Queue[Any] = queue.Queue()
        inferred: queue.Queue[Any] = queue.Queue()
        egress_free: queue.Queue[int] = queue.Queue()
        for k in range(self.capacity):
            egress_free.put(k)
        to_pipe: queue.Queue[Any] = queue.Queue()
        lock = threading.Lock()
        t_start = time.perf_counter()
        warm: list[tuple[float, int]] = []

        def get(q: queue.Queue[Any], wait_attr: str) -> Any:
            t0 = time.perf_counter()
            while True:
                if stop.is_set():
                    raise _Stop()
                if cancel.is_set():
                    raise PipelineCancelled("cancelled")
                try:
                    item = q.get(timeout=_POLL)
                    break
                except queue.Empty:
                    continue
            with lock:
                setattr(stats, wait_attr, getattr(stats, wait_attr) + time.perf_counter() - t0)
            return item

        def add(attr: str, seconds: float) -> None:
            with lock:
                setattr(stats, attr, getattr(stats, attr) + seconds)

        def guarded(fn: Callable[[], None]) -> Callable[[], None]:
            def body() -> None:
                try:
                    fn()
                except _Stop:
                    pass
                except BaseException as exc:  # noqa: BLE001 - re-raised by run()
                    failure.append(exc)
                    stop.set()
            return body

        # ------------------------------------------------ stage 1: decode
        def decode() -> None:
            import av

            idx = src.index
            pts_to_index = {int(p): i for i, p in enumerate(idx.pts[:total])}
            last_pts = int(idx.pts[total - 1])
            full_range = info.color_tags.get("color_range") in ("pc", "jpeg")
            emitted = 0
            with torch.cuda.stream(streams["decode"]), av.open(str(source_path)) as container:
                vstream = container.streams.video[0]
                vstream.thread_type = "AUTO"
                for frame in container.decode(vstream):
                    if frame.pts is None:
                        continue
                    i = pts_to_index.get(int(frame.pts))
                    if i is None:
                        if frame.pts > last_pts:
                            break
                        continue
                    k = get(free, "decode_wait_s")
                    t0 = time.perf_counter()
                    ring.event_uploaded[k].synchronize()  # ingress[k]'s last upload landed
                    fmt = frame.format.name
                    if fmt in PLANAR_420 and self.gpu_color:
                        for plane, dst in zip(frame.planes, ring.host_planes(k)):
                            rows, cols = dst.shape
                            view = np.frombuffer(plane, np.uint8, count=rows * plane.line_size)
                            np.copyto(dst, view.reshape(rows, plane.line_size)[:, :cols])
                        streams["decode"].wait_event(ring.event_released[k])
                        ring.planes[k].copy_(ring.ingress[k], non_blocking=True)
                        ring.event_uploaded[k].record(streams["decode"])
                        i420_to_bgr_hwc(ring.planes[k], info.height, info.width,
                                        info.color_profile,
                                        full_range or fmt == "yuvj420p", ring.slots[k])
                    else:
                        bgr = frame.to_ndarray(format="bgr24")  # allocates (see module doc)
                        streams["decode"].wait_event(ring.event_released[k])
                        ring.slots[k].copy_(torch.from_numpy(bgr))  # synchronous upload
                        ring.event_uploaded[k].record(streams["decode"])
                        add("host_bgr_frames", 1)
                    ring.event_decoded[k].record(streams["decode"])
                    decoded.put((k, i))
                    emitted += 1
                    add("decode_s", time.perf_counter() - t0)
                    if emitted == total:
                        break
            if emitted != total:
                raise RuntimeError(f"decoded {emitted} of {total} frames from {source_path.name}")
            decoded.put(None)

        # ------------------------------------------------ stage 2: inference
        def infer() -> None:
            with torch.cuda.stream(streams["inference"]):
                while True:
                    item = get(decoded, "infer_wait_s")
                    if item is None:
                        inferred.put(None)
                        return
                    k, i = item
                    t0 = time.perf_counter()
                    streams["inference"].wait_event(ring.event_decoded[k])
                    frame = ring.slots[k].permute(2, 0, 1)[None]
                    if processor is None:
                        plan = frame.float()
                        faces = swapped = 0
                    else:
                        plan = processor.infer(frame)
                        faces, swapped = plan.stats.faces, plan.stats.swapped
                    # Tensors made on this stream and read on the encode stream
                    # must not be recycled by the caching allocator meanwhile.
                    for t in ([plan] if processor is None else plan.tensors()):
                        t.record_stream(streams["encode"])
                    ring.event_inferred[k].record(streams["inference"])
                    inferred.put((k, i, plan))
                    with lock:
                        stats.faces += faces
                        stats.swapped += swapped
                    add("infer_s", time.perf_counter() - t0)

        # ------------------------------------------------ stage 3: encode
        def encode() -> None:
            with torch.cuda.stream(streams["encode"]):
                while True:
                    item = get(inferred, "encode_wait_s")
                    if item is None:
                        to_pipe.put(None)
                        return
                    k, i, plan = item
                    e = get(egress_free, "encode_wait_s")
                    t0 = time.perf_counter()
                    streams["encode"].wait_event(ring.event_inferred[k])
                    out = plan if processor is None else processor.composite(plan)
                    if out is None:
                        # No face: the decoded slot IS the output; only the
                        # writer's channel order may differ (guardrails, Stage 8).
                        if rgb:
                            ring.slots[k].copy_(ring.slots[k].flip(2))
                        with lock:
                            stats.passthrough_frames += 1
                    else:
                        write_slot(out, ring.slots[k], rgb)
                    ring.egress[e].copy_(ring.slots[k], non_blocking=True)
                    ring.event_downloaded[e].record(streams["encode"])
                    ring.event_released[k].record(streams["encode"])
                    del plan, out
                    free.put(k)
                    to_pipe.put((e, i))
                    add("encode_s", time.perf_counter() - t0)

        # ------------------------------------------------ stage 4: pipe
        def pipe() -> None:
            expected = 0
            last_report = 0.0
            while True:
                item = get(to_pipe, "write_wait_s")
                if item is None:
                    return
                e, i = item
                if i != expected:
                    raise RuntimeError(f"frame {i} reached the writer, expected {expected}")
                expected += 1
                t0 = time.perf_counter()
                ring.event_downloaded[e].synchronize()
                writer.write(ring.egress_np[e])
                egress_free.put(e)
                add("write_s", time.perf_counter() - t0)
                now = time.perf_counter()
                with lock:
                    stats.frames_done = expected
                    stats.elapsed = now - t_start
                    # fps from the end of the first `capacity` frames: model
                    # sessions and engines load on the first frames.
                    if not warm and expected >= self.capacity:
                        warm.append((now, expected))
                    elif warm and now > warm[0][0]:
                        stats.fps = (expected - warm[0][1]) / (now - warm[0][0])
                if on_progress is not None and now - last_report >= progress_interval:
                    last_report = now
                    on_progress(PipelineStats(**stats.as_dict()))

        threads = [threading.Thread(target=guarded(fn), name=f"stream-{fn.__name__}",
                                    daemon=True) for fn in (decode, infer, encode, pipe)]
        try:
            for t in threads:
                t.start()
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(timeout=_POLL)
                if cancel.is_set():
                    stop.set()
            if failure:
                raise failure[0]
            if cancel.is_set():
                raise PipelineCancelled("cancelled")
            writer.close()
        except BaseException:
            stop.set()
            for t in threads:
                t.join(timeout=10)
            writer.abort()
            if self.remux:
                video.unlink(missing_ok=True)
            raise
        if self.remux:
            try:
                remux(video, source_path, target_path)
            finally:
                video.unlink(missing_ok=True)
        from face_engine.core.guardrails import check_av_sync

        sync = check_av_sync(target_path, source_path, total, fps=info.fps, audio=self.remux)
        stats.extra["av_sync"] = {"video_s": sync.video_duration, "audio_s": sync.audio_duration,
                                  "fps": str(sync.fps)}
        if not sync.ok:
            raise RuntimeError(f"{target_path.name} failed the A/V check: "
                               + "; ".join(sync.problems))
        stats.elapsed = time.perf_counter() - t_start
        stats.output = str(target_path)
        stats.done = True
        stats.extra["encoder"] = type(writer).__name__
        stats.extra["ring_bytes"] = ring.nbytes
        if on_progress is not None:
            on_progress(PipelineStats(**stats.as_dict()))
        return stats

    async def process_video(self, source_path: str | Path, target_path: str | Path,
                            pipeline_config: Any = None, *,
                            cancel: threading.Event | None = None,
                            max_frames: int | None = None) -> AsyncIterator[PipelineStats]:
        """Async generator over :meth:`run`: yields :class:`PipelineStats` ~4x a second.

        The render runs in a worker thread; the last item has ``done=True`` and
        ``output``. Leaving the loop early (or cancelling the task) stops the
        render.
        """
        loop = asyncio.get_running_loop()
        updates: asyncio.Queue[PipelineStats] = asyncio.Queue()
        cancel = cancel or threading.Event()

        def report(s: PipelineStats) -> None:
            loop.call_soon_threadsafe(updates.put_nowait, s)

        future = loop.run_in_executor(None, lambda: self.run(
            source_path, target_path, pipeline_config, cancel=cancel, on_progress=report,
            max_frames=max_frames))
        try:
            while True:
                getter = asyncio.ensure_future(updates.get())
                done, _ = await asyncio.wait({getter, future},
                                             return_when=asyncio.FIRST_COMPLETED)
                if getter in done:
                    yield getter.result()
                    continue
                getter.cancel()
                future.result()  # raises the render's exception
                while not updates.empty():
                    yield updates.get_nowait()
                return
        finally:
            if not future.done():
                cancel.set()
                try:
                    await future
                except BaseException as exc:  # noqa: BLE001 - the caller already left
                    logger.debug("render stopped after the consumer left: %s", exc)

    # ------------------------------------------------------------------ helpers
    def _processor(self, config: Any) -> tuple[Any, bool]:
        if config is None:
            return None, False
        if hasattr(config, "infer") and hasattr(config, "composite"):
            return config, False
        from face_engine.server.processing import GpuFrameProcessor, ProcessorConfig

        if isinstance(config, ProcessorConfig):
            return GpuFrameProcessor(config, device=self.device, tracking=True), True
        raise TypeError(f"pipeline_config must be a ProcessorConfig, a processor or None, "
                        f"got {type(config).__name__}")


#: Default pipeline: ``await``-free use is :meth:`CUDAStreamPipeline.run`.
stream_pipeline = CUDAStreamPipeline()


def process_video(source_path: str | Path, target_path: str | Path,
                  pipeline_config: Any = None, **kwargs: Any) -> AsyncIterator[PipelineStats]:
    """``stream_pipeline.process_video(...)``: see :meth:`CUDAStreamPipeline.process_video`."""
    return stream_pipeline.process_video(source_path, target_path, pipeline_config, **kwargs)
