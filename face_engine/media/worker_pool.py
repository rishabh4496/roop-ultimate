"""Segment-parallel rendering: GOP-aligned chunks, one process per chunk, seamless concat.

:class:`SegmentWorkerPool` splits the source at keyframes
(:meth:`~face_engine.media.capturer.VideoSource.plan_segments`), assigns the
segments to GPUs round-robin, and runs each in its own spawned process:

    HardwareVideoDecoder(start, end)  ->  processor(batch)  ->  NVENC part_i.mp4

Every segment starts on a keyframe, so each worker decodes it independently
(byte-identical to a sequential decode; tested in ``test_media``), and each
part starts with its own IDR frame, so the parts join with ffmpeg's concat
demuxer and ``-c copy``: no re-encode, no seam. The joined video is then
remuxed with the source's audio, subtitles and chapters
(:func:`~face_engine.media.demuxer.remux`) and the frame count checked.

Shared memory: workers report progress through one
``multiprocessing.shared_memory`` block (an int64 per segment: frames done,
then a state word), which the parent reads without messages or copies. The
parent creates and owns it; :func:`~face_engine.media.ipc_pool.install_cleanup_handlers`
closes and unlinks it on exit, exceptions and SIGINT/SIGTERM/SIGBREAK, and
workers only attach (they never unlink). Frames themselves never cross a
process boundary in this design: each worker decodes, processes and encodes
its own range on its own GPU, which is what makes it scale across GPUs.
Frame transport between processes on ONE GPU is
:class:`~face_engine.media.ipc_pool.FramePipeline`'s job.

The processor is named as ``"package.module:factory"`` (a spawned process
imports it); ``factory(device, **kwargs)`` returns a callable
``(FrameBatch) -> (B, 3, H, W)`` tensor on the GPU.
"""
from __future__ import annotations

import importlib
import logging
import multiprocessing as mp
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from face_engine.media.capturer import Segment, VideoSource
from face_engine.media.ffmpeg_pipe import FFmpegError, OutputReport, inspect_output
from face_engine.media.ipc_pool import (
    attach_tracked,
    create_owned,
    install_cleanup_handlers,
    release,
)
from face_engine.media.tools import find_tool

logger = logging.getLogger(__name__)

_PENDING, _RUNNING, _DONE, _FAILED = 0, 1, 2, 3


def load_processor(spec: str) -> Callable[..., Any]:
    """``"package.module:attr"`` -> the attribute."""
    module, _, attr = spec.partition(":")
    if not module or not attr:
        raise ValueError(f"processor must be 'module:factory', got {spec!r}")
    return getattr(importlib.import_module(module), attr)


def identity_processor(device: Any = "cuda", **_kwargs: Any) -> Callable[[Any], Any]:
    """Pass-through factory (I/O-only renders and tests)."""
    return lambda batch: batch.frames


@dataclass(frozen=True)
class SegmentJob:
    source: str
    segment: Segment
    part: str
    processor: str
    processor_kwargs: dict[str, Any]
    encoder: str
    batch_size: int
    progress: str  # shared memory name
    slot: int


def run_segment(job: SegmentJob) -> dict[str, Any]:
    """Worker entry point (a spawned process): decode, process, encode one segment."""
    import torch

    from face_engine.media.decoder import HardwareVideoDecoder
    from face_engine.media.encoder import open_video_writer

    install_cleanup_handlers()
    shm = attach_tracked(job.progress)
    progress = np.ndarray((2 * (job.slot + 1),), np.int64, buffer=shm.buf)
    seg = job.segment
    device = torch.device("cuda", seg.device)
    torch.cuda.set_device(device)
    t0 = time.perf_counter()
    try:
        progress[2 * job.slot + 1] = _RUNNING
        decoder = HardwareVideoDecoder(job.source, start=seg.start, end=seg.end,
                                       batch_size=job.batch_size, device=device)
        process = load_processor(job.processor)(device=device, **job.processor_kwargs)
        info = decoder.info
        writer = open_video_writer(job.part, info.width, info.height, info.fps,
                                   encoder=job.encoder, expected_frames=seg.frames,
                                   color=info.color_profile)
        try:
            for batch in decoder:
                writer.write_tensor(process(batch))
                progress[2 * job.slot] += len(batch)
            report = writer.close()
        except BaseException:
            writer.abort()
            raise
        progress[2 * job.slot + 1] = _DONE
        return {"segment": seg.index, "frames": report.frames, "decoder": decoder.backend,
                "writer": type(writer).__name__, "seconds": time.perf_counter() - t0}
    except BaseException:
        progress[2 * job.slot + 1] = _FAILED
        raise
    finally:
        del progress
        release(shm, unlink=False)


@dataclass
class PoolReport:
    output: OutputReport
    frames: int
    seconds: float
    segments: list[dict[str, Any]] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds else 0.0


class SegmentWorkerPool:
    """Render a video as keyframe-aligned segments in parallel processes.

    Args:
        devices: CUDA device ids; segments are assigned round-robin.
        segments_per_device: Segments (and concurrent processes) per GPU.
        encoder: ``"auto"`` / ``"nvenc"`` / ``"x264"`` (see ``open_video_writer``).
        batch_size: Frames per decoded batch.
        workdir: Where part files go (default: next to the output).
    """

    def __init__(self, devices: list[int] | None = None, *, segments_per_device: int = 1,
                 encoder: str = "auto", batch_size: int = 4,
                 workdir: str | Path | None = None) -> None:
        self.devices = devices or [0]
        self.segments_per_device = max(1, segments_per_device)
        self.encoder = encoder
        self.batch_size = batch_size
        self.workdir = Path(workdir) if workdir else None

    def run(self, source: str | Path, output: str | Path, *,
            processor: str = "face_engine.media.worker_pool:identity_processor",
            processor_kwargs: dict[str, Any] | None = None,
            on_progress: Callable[[int, int], None] | None = None) -> PoolReport:
        from face_engine.media.demuxer import remux

        install_cleanup_handlers()
        source, output = Path(source), Path(output)
        workdir = self.workdir or output.parent / f".{output.stem}_parts"
        workdir.mkdir(parents=True, exist_ok=True)
        src = VideoSource(source)
        workers = len(self.devices) * self.segments_per_device
        segments = src.plan_segments(workers, self.devices)
        total = sum(s.frames for s in segments)
        shm = create_owned(max(16, 16 * len(segments)))
        progress = np.ndarray((2 * len(segments),), np.int64, buffer=shm.buf)
        progress[:] = 0
        parts = [workdir / f"part_{s.index:03d}.mp4" for s in segments]
        jobs = [SegmentJob(str(source), s, str(p), processor, dict(processor_kwargs or {}),
                           self.encoder, self.batch_size, shm.name, s.index)
                for s, p in zip(segments, parts)]
        t0 = time.perf_counter()
        results: list[dict[str, Any]] = []
        try:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=len(jobs), mp_context=ctx) as pool:
                futures = [pool.submit(run_segment, job) for job in jobs]
                pending = set(futures)
                while pending:
                    done = {f for f in pending if f.done()}
                    for f in done:
                        results.append(f.result())  # re-raises a worker failure
                    pending -= done
                    if on_progress is not None:
                        on_progress(int(progress[0::2].sum()), total)
                    if pending:
                        time.sleep(0.2)
            joined = workdir / "joined.mp4"
            self._concat(parts, joined)
            report = remux(joined, source, output)
            if report.frames != total:
                raise FFmpegError(f"{output.name}: {report.frames} frames, expected {total}")
        finally:
            del progress
            release(shm, unlink=True)
        seconds = time.perf_counter() - t0
        for p in [*parts, workdir / "joined.mp4", workdir / "concat.txt"]:
            p.unlink(missing_ok=True)
        try:
            workdir.rmdir()
        except OSError:
            pass
        return PoolReport(report, total, seconds, sorted(results, key=lambda r: r["segment"]))

    @staticmethod
    def _concat(parts: list[Path], joined: Path) -> None:
        listing = joined.with_name("concat.txt")
        listing.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in parts),
                           encoding="utf-8")
        proc = subprocess.run([find_tool("ffmpeg"), "-y", "-v", "error", "-f", "concat",
                               "-safe", "0", "-i", str(listing), "-c", "copy",
                               "-movflags", "+faststart", str(joined)],
                              capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise FFmpegError(f"concat failed: {proc.stderr.strip()[-800:]}")
        expected = sum(inspect_output(p).frames for p in parts)
        got = inspect_output(joined).frames
        if got != expected:
            raise FFmpegError(f"concat produced {got} frames from {expected}")
