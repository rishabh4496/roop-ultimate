"""Stage 7: the triple-buffered decode / inference / encode stream pipeline.

The hazard tests stall ONE stream with ``torch.cuda._sleep`` so the other two
run ahead of it; a missing ``wait_event`` (or a tensor recycled across
streams) then shows up as a frame that is not its own input's result. Clips
are synthesised with ffmpeg ``testsrc2``, whose frames all differ, so "the
output of frame i is closer to input i than to its neighbours" is a real
check of both content and order.
"""
from __future__ import annotations

import asyncio
import subprocess
import threading
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from face_engine.core import cuda_streams
from face_engine.core.cuda_streams import (
    CUDARingBuffer,
    CUDAStreamPipeline,
    PipelineCancelled,
    i420_to_bgr_hwc,
)
from face_engine.media.capturer import ColorProfile
from face_engine.media.tools import find_tool
from face_engine.server.processing import FramePlan, FrameStats

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")]

FRAMES = 48
SLEEP_CYCLES = 20_000_000  # ~10 ms of spinning on one stream per frame


def _decode_bgr(path: Path) -> list[np.ndarray]:
    import av

    with av.open(str(path)) as c:
        return [f.to_ndarray(format="bgr24") for f in c.decode(video=0)]


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("streams") / "src.mp4"
    subprocess.run([find_tool("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"testsrc2=size=640x360:rate=25,trim=end_frame={FRAMES}",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-pix_fmt", "yuv420p",
                    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                    "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


@pytest.fixture(scope="module")
def source_frames(clip: Path) -> list[np.ndarray]:
    frames = _decode_bgr(clip)
    assert len(frames) == FRAMES
    return frames


class Invert:
    """Processor that inverts the frame, optionally stalling one stream."""

    def __init__(self, stall: str | None = None) -> None:
        self.stall = stall
        self.fail_at: int | None = None
        self.calls = 0

    def infer(self, frame: torch.Tensor) -> FramePlan:
        self.calls += 1
        if self.fail_at is not None and self.calls > self.fail_at:
            raise ValueError("inference exploded")
        if self.stall == "inference":
            torch.cuda._sleep(SLEEP_CYCLES)
        return FramePlan(255.0 - frame.float(), [], FrameStats(faces=1, swapped=1))

    def composite(self, plan: FramePlan) -> torch.Tensor:
        if self.stall == "encode":
            torch.cuda._sleep(SLEEP_CYCLES)
        return plan.canvas * 1.0  # a new tensor made on the encode stream

    def close(self) -> None:
        pass


def _assert_frames(out: list[np.ndarray], expected: list[np.ndarray], tol: float) -> None:
    assert len(out) == len(expected)
    for i, frame in enumerate(out):
        own = np.abs(frame.astype(np.int16) - expected[i]).mean()
        assert own < tol, f"frame {i}: mean difference {own:.2f} from its own input"
        for j in (i - 1, i + 1):
            if 0 <= j < len(expected):
                other = np.abs(frame.astype(np.int16) - expected[j]).mean()
                assert own < other, f"frame {i} is closer to input {j} ({other:.2f} <= {own:.2f})"


# ------------------------------------------------------------------ ring buffer
def test_ring_buffer_allocates_once_in_vram_and_pinned() -> None:
    ring = CUDARingBuffer(3, (360, 640, 3), torch.uint8, "cuda")
    assert ring.slots.shape == (3, 360, 640, 3) and ring.slots.is_cuda
    assert ring.slots.is_contiguous() and ring.slots.dtype == torch.uint8
    assert ring.planes.is_cuda and ring.planes.shape == (3, 360 * 640 * 3 // 2)
    assert ring.ingress.is_pinned() and ring.egress.is_pinned()
    assert len(ring.event_decoded) == len(ring.event_inferred) == len(ring.event_released) == 3
    y, u, v = ring.host_planes(1)
    assert y.shape == (360, 640) and u.shape == v.shape == (180, 320)
    assert np.shares_memory(y, ring.ingress_np) and np.shares_memory(v, ring.ingress_np)
    assert ring.nbytes["vram"] == 3 * 360 * 640 * 3 + 3 * 360 * 640 * 3 // 2
    with pytest.raises(ValueError):
        CUDARingBuffer(1, (8, 8, 3))


def test_odd_frame_sizes_round_chroma_up() -> None:
    ring = CUDARingBuffer(2, (9, 7, 3))
    assert ring.chroma_shape == (5, 4)
    assert ring.plane_bytes == 9 * 7 + 2 * 5 * 4


def test_gpu_yuv_conversion_matches_pyav(clip: Path, source_frames: list[np.ndarray]) -> None:
    import av

    with av.open(str(clip)) as c:
        frame = next(c.decode(video=0))
        planes = [np.frombuffer(p, np.uint8, count=p.height * p.line_size)
                  .reshape(p.height, p.line_size)[:, :p.width] for p in frame.planes]
    flat = torch.as_tensor(np.concatenate([p.reshape(-1) for p in planes]), device="cuda")
    out = torch.empty((360, 640, 3), dtype=torch.uint8, device="cuda")
    i420_to_bgr_hwc(flat, 360, 640, ColorProfile.BT709, False, out)
    diff = np.abs(out.cpu().numpy().astype(np.int16) - source_frames[0])
    # swscale rounds low (Stage 4 measured ~1 level for the NVDEC path).
    assert diff.mean() < 2.0 and diff.max() <= 6


# ------------------------------------------------------------------ correctness
def test_pass_through_is_frame_exact_and_keeps_audio(clip: Path, tmp_path: Path,
                                                     source_frames: list[np.ndarray]) -> None:
    out = tmp_path / "out.mp4"
    stats = CUDAStreamPipeline(gpu_color=False).run(clip, out, None)
    assert stats.done and stats.frames_done == FRAMES and stats.output == str(out)
    assert stats.host_bgr_frames == FRAMES  # PyAV colour path, by request
    _assert_frames(_decode_bgr(out), source_frames, tol=3.0)
    import av

    with av.open(str(out)) as c:
        assert len(c.streams.audio) == 1
    assert not (tmp_path / ".out_video.mp4").exists()


@pytest.mark.parametrize("stall", ["decode", "inference", "encode", "egress"])
def test_stalled_stream_never_hands_over_a_wrong_frame(
        clip: Path, tmp_path: Path, source_frames: list[np.ndarray], stall: str,
        monkeypatch: pytest.MonkeyPatch) -> None:
    if stall == "decode":
        convert = cuda_streams.i420_to_bgr_hwc

        def slow_convert(*args: object, **kwargs: object) -> torch.Tensor:
            torch.cuda._sleep(SLEEP_CYCLES)
            return convert(*args, **kwargs)

        monkeypatch.setattr(cuda_streams, "i420_to_bgr_hwc", slow_convert)
    if stall == "egress":  # between the slot write and the copy out of the slot
        write = cuda_streams.write_slot

        def slow_write(*args: object) -> None:
            write(*args)
            torch.cuda._sleep(SLEEP_CYCLES)

        monkeypatch.setattr(cuda_streams, "write_slot", slow_write)
    processor = Invert(stall if stall in ("inference", "encode") else None)
    out = tmp_path / "inv.mp4"
    stats = CUDAStreamPipeline().run(clip, out, processor)
    assert stats.frames_done == FRAMES and stats.swapped == FRAMES
    assert stats.host_bgr_frames == 0
    expected = [255 - f for f in source_frames]
    _assert_frames(_decode_bgr(out), expected, tol=4.0)  # GPU colour: ~1-2 levels from PyAV


def test_per_frame_loop_allocates_no_host_frames(clip: Path, tmp_path: Path) -> None:
    """Python/numpy host allocations over a whole render stay below one frame.

    libav's own decode buffers are outside tracemalloc (they come from its
    frame pool); the control arm proves the measurement sees a frame-sized
    numpy allocation made on a stage thread.
    """
    frame_bytes = 640 * 360 * 3

    class Allocating(Invert):
        def infer(self, frame: torch.Tensor) -> FramePlan:
            np.ones(frame_bytes, np.uint8)  # what a host round trip would cost
            return super().infer(frame)

    def peak(processor: Invert) -> int:
        tracemalloc.start()
        try:
            CUDAStreamPipeline(remux=False).run(clip, tmp_path / "a.mp4", processor)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    assert peak(Allocating()) > frame_bytes
    assert peak(Invert()) < frame_bytes // 2


# ------------------------------------------------------------------ control
def test_async_generator_reports_progress_then_the_output(clip: Path, tmp_path: Path) -> None:
    out = tmp_path / "async.mp4"

    async def collect() -> list:
        return [s async for s in CUDAStreamPipeline().process_video(
            clip, out, Invert(), max_frames=30)]

    items = asyncio.run(collect())
    assert items and items[-1].done and items[-1].output == str(out)
    assert items[-1].frames_done == 30 and len(_decode_bgr(out)) == 30
    done = [s.frames_done for s in items]
    assert done == sorted(done)


def test_module_level_process_video_uses_the_default_pipeline(clip: Path, tmp_path: Path) -> None:
    async def last() -> object:
        final = None
        async for s in cuda_streams.process_video(clip, tmp_path / "m.mp4", None, max_frames=5):
            final = s
        return final

    assert asyncio.run(last()).frames_done == 5


def test_cancel_stops_the_render_and_removes_the_partial_file(clip: Path, tmp_path: Path) -> None:
    cancel = threading.Event()
    processor = Invert("inference")

    def progress(s: object) -> None:
        cancel.set()

    with pytest.raises(PipelineCancelled):
        CUDAStreamPipeline().run(clip, tmp_path / "c.mp4", processor, cancel=cancel,
                                 on_progress=progress, progress_interval=0.0)
    assert not (tmp_path / ".c_video.mp4").exists() and not (tmp_path / "c.mp4").exists()
    assert processor.calls < FRAMES


def test_a_failing_stage_raises_its_error_without_hanging(clip: Path, tmp_path: Path) -> None:
    processor = Invert()
    processor.fail_at = 5
    with pytest.raises(ValueError, match="inference exploded"):
        CUDAStreamPipeline().run(clip, tmp_path / "f.mp4", processor)
    assert not (tmp_path / ".f_video.mp4").exists()


def test_rejects_an_unknown_config(clip: Path, tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        CUDAStreamPipeline().run(clip, tmp_path / "x.mp4", object())
