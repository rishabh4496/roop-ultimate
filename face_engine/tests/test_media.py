"""Stage 4: ingestion, shared-memory pipeline, ffmpeg output, HTTP 206 streaming.

All clips are synthesised with ffmpeg (testsrc2 / sine / a click at a known
time), so every property is known exactly.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from fractions import Fraction
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import pytest

from face_engine.media.capturer import (
    ColorProfile,
    VideoSource,
    choose_frame_rate,
    probe,
)
from face_engine.media.ffmpeg_pipe import (
    FFmpegError,
    FFmpegWriter,
    inspect_output,
    verify_http_range_streaming,
)
from face_engine.media.ipc_pool import (
    FramePipeline,
    PipelineError,
    RingAborted,
    SharedMemoryRingBuffer,
    VideoFrames,
)
from face_engine.media.tools import find_tool
from face_engine.tests import media_fixtures as mf
from face_engine.tests import media_workers as mw

REPO = Path(__file__).resolve().parents[2]

# Needs ffmpeg + PyAV (the "full stack" in conftest terms): CI's light profile deselects it.
pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def clips(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("clips")
    click = d / "click.mp4"
    subprocess.run([find_tool("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=320x240:rate=24000/1001:duration=3", "-f", "lavfi", "-i",
                    "aevalsrc='if(between(t,1,1.005),sin(2*PI*1000*t),0)':s=48000:d=3",
                    "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", str(click)], check=True)
    return {"ntsc": mf.ntsc_clip(d / "ntsc.mp4"), "vfr": mf.vfr_clip(d / "vfr.mp4"),
            "opus": mf.ntsc_clip(d / "opus.mkv", seconds=2, audio="opus"),
            "silent": mf.ntsc_clip(d / "silent.mp4", seconds=2, audio="none"),
            "subs": mf.with_subtitles(mf.ntsc_clip(d / "base.mp4", seconds=2), d / "subs.mp4",
                                      "1\n00:00:00,500 --> 00:00:01,500\nhello\n"),
            "click": click, "dir": d}


def _onset(path: Path) -> float:
    pcm = subprocess.run([find_tool("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:a:0",
                          "-f", "f32le", "-ac", "1", "-ar", "48000", "-"],
                         capture_output=True, check=True).stdout
    a = np.abs(np.frombuffer(pcm, np.float32))
    return float(np.argmax(a > 0.2 * a.max()) / 48000)


def _md5(a: np.ndarray) -> str:
    return hashlib.md5(a.tobytes()).hexdigest()


# ------------------------------------------------------------------ capturer
def test_frame_rate_rule() -> None:
    assert choose_frame_rate("24000/1001", "24000/1001") == (Fraction(24000, 1001), "r_frame_rate")
    # VFR: nominal 30 but 180 frames over 7.93 s -> the average wins.
    fps, which = choose_frame_rate("30/1", "90000000/3966667")
    assert which == "avg_frame_rate" and abs(float(fps) - 22.689) < 1e-3
    assert choose_frame_rate("0/0", "25/1") == (Fraction(25), "avg_frame_rate")
    with pytest.raises(ValueError):
        choose_frame_rate("0/0", "0/0")


def test_probe_metadata(clips: dict[str, Path]) -> None:
    ntsc = probe(clips["ntsc"])
    assert (ntsc.width, ntsc.height, ntsc.frame_count) == (320, 240, 119)
    assert ntsc.fps == Fraction(24000, 1001) and not ntsc.is_vfr
    assert ntsc.color_profile is ColorProfile.BT709
    assert ntsc.sample_aspect_ratio == 1 and ntsc.audio[0].codec == "aac"
    vfr = probe(clips["vfr"])
    assert vfr.is_vfr and vfr.frame_count == 180
    # N frames at the chosen rate last as long as the video: A/V stays in sync.
    assert abs(vfr.frame_count / float(vfr.fps) - vfr.duration) < 1e-3
    assert probe(clips["subs"]).subtitles[0].codec == "mov_text"


def test_segments_and_random_access_are_frame_exact(clips: dict[str, Path]) -> None:
    src = VideoSource(clips["ntsc"])
    assert src.keyframes.tolist()[:4] == [0, 12, 24, 36]
    sequential = [_md5(f.image) for f in src.frames()]
    assert len(sequential) == src.frame_count == 119
    segments = src.plan_segments(4, devices=[0, 1])
    assert segments[0].start == 0 and segments[-1].end == 119
    assert all(a.end == b.start for a, b in zip(segments, segments[1:]))
    assert all(s.start in set(src.keyframes.tolist()) for s in segments)
    assert [s.device for s in segments] == [0, 1, 0, 1]
    assert [_md5(f.image) for s in segments for f in src.frames(s.start, s.end)] == sequential
    assert [_md5(f.image) for f in src.frames(37, 41)] == sequential[37:41]
    assert len(src.plan_segments(1000)) == len(src.keyframes)  # never splits inside a GOP


def test_pyav_honours_the_bt709_tag(clips: dict[str, Path]) -> None:
    """Matches ffmpeg's tag-honouring conversion byte for byte (cv2 would use BT.601)."""
    first = next(VideoSource(clips["ntsc"]).frames()).image
    raw = subprocess.run([find_tool("ffmpeg"), "-v", "error", "-i", str(clips["ntsc"]),
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True, check=True).stdout
    assert np.array_equal(first, np.frombuffer(raw, np.uint8).reshape(first.shape))


def test_demux_is_lossless_and_keeps_sync(clips: dict[str, Path], tmp_path: Path) -> None:
    out = VideoSource(clips["click"]).demux_streams(tmp_path / ".temp")
    assert [p.name for p in out["audio"]] == ["audio.m4a"]
    # A raw ADTS .aac copy would play the 1024 priming samples: +21.3 ms.
    assert abs(_onset(out["audio"][0]) - _onset(clips["click"])) < 0.001
    opus = VideoSource(clips["opus"]).demux_streams(tmp_path / "opus")
    assert opus["audio"][0].suffix == ".mka"
    subs = VideoSource(clips["subs"]).demux_streams(tmp_path / "subs")
    assert "hello" in subs["subtitles"][0].read_text(encoding="utf-8")


# ------------------------------------------------------------------ ring buffer
def test_ring_is_zero_copy_fifo_and_closes_cleanly() -> None:
    with SharedMemoryRingBuffer(3, (4, 4, 3)) as ring:
        slot = ring.acquire_write(timeout=1)
        slot.array[...] = 7
        assert np.shares_memory(slot.array, ring.view(slot.index))
        ring.commit(slot, seq=0)
        ring.put(np.full((4, 4, 3), 9, np.uint8), seq=1)
        assert ring.pending() == 2
        first = ring.acquire_read(timeout=1)
        assert first is not None and first.seq == 0 and first.array.max() == 7
        first.array[0, 0, 0] = 42  # writes land in shared memory
        assert ring.view(first.index)[0, 0, 0] == 42
        ring.release(first)
        assert ring.get(timeout=1)[0] == 1
        with pytest.raises(TimeoutError):
            ring.acquire_read(timeout=0.05)
        for i in range(3):
            ring.put(np.zeros((4, 4, 3), np.uint8), seq=i)
        with pytest.raises(TimeoutError):  # all slots full
            ring.acquire_write(timeout=0.05)
        ring.close()
        assert [ring.get(timeout=1)[0] for _ in range(3)] == [0, 1, 2]
        assert ring.acquire_read(timeout=1) is None and ring.acquire_read(timeout=1) is None
        name = ring.name
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)


def test_abort_wakes_waiters() -> None:
    with SharedMemoryRingBuffer(1, (2, 2)) as ring:
        ring.abort()
        with pytest.raises(RingAborted):
            ring.acquire_read(timeout=1)
        with pytest.raises(RingAborted):
            ring.acquire_write(timeout=1)


def test_pipeline_delivers_every_frame_in_order() -> None:
    shape = (48, 64, 3)
    seen: list[int] = []

    def sink(seq: int, frame: np.ndarray) -> None:
        assert frame.shape == shape and int(frame[0, 0, 0]) == 255 - seq % 256
        seen.append(seq)

    n = FramePipeline(mw.Counting(300, shape), mw.slow_invert, shape, workers=3, slots=5).run(sink)
    assert n == 300 and seen == list(range(300))


def test_pipeline_surfaces_a_worker_exception() -> None:
    with pytest.raises(PipelineError, match="simulated inference failure at frame 5"):
        FramePipeline(mw.Counting(50, (8, 8, 3)), mw.fail_at_five, (8, 8, 3),
                      workers=2).run(lambda s, f: None)


def test_pipeline_does_not_hang_when_a_worker_dies() -> None:
    start = time.monotonic()
    with pytest.raises(PipelineError, match="exit code 3"):
        FramePipeline(mw.Counting(50, (8, 8, 3)), mw.die_at_five, (8, 8, 3),
                      workers=2).run(lambda s, f: None)
    assert time.monotonic() - start < 30


def _child(mode: str, **kw: object) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-m", "face_engine.tests.media_workers", mode],
                            cwd=REPO, stdout=subprocess.PIPE, text=True, **kw)  # type: ignore[call-overload]


def test_sigint_handler_closes_and_unlinks() -> None:
    child = _child("sigint")
    out, _ = child.communicate(timeout=60)
    assert "released=True attachable=False" in out, out


def test_segment_is_freed_when_its_owner_is_killed() -> None:
    """No Python code runs on a hard kill; the OS / resource tracker must free it."""
    child = _child("hold")
    try:
        name = child.stdout.readline().strip()  # type: ignore[union-attr]
        shared_memory.SharedMemory(name=name).close()  # alive while the owner lives
    finally:
        child.kill()
        child.wait(timeout=30)
    deadline = time.monotonic() + 10
    while True:
        try:
            shared_memory.SharedMemory(name=name).close()
        except FileNotFoundError:
            break
        assert time.monotonic() < deadline, "segment survived its owner's death"
        time.sleep(0.2)


# ------------------------------------------------------------------ writer
def _render(src: Path, out: Path, audio: Path | None = None, workers: int = 2) -> tuple[list, object]:
    source = VideoSource(src)
    info = source.info
    got: list[np.ndarray] = []
    with FFmpegWriter(out, info.width, info.height, info.fps, audio=audio,
                      color=info.color_profile, expected_frames=info.frame_count) as writer:
        def sink(seq: int, frame: np.ndarray) -> None:
            got.append(frame.copy())
            writer.write(frame)

        FramePipeline(VideoFrames(str(src)), mw.invert, (info.height, info.width, 3),
                      workers=workers).run(sink)
        report = writer.close()
    return got, report


def test_end_to_end_render_is_web_ready(clips: dict[str, Path], tmp_path: Path) -> None:
    src = VideoSource(clips["ntsc"])
    expected = [255 - f.image for f in src.frames()]
    got, report = _render(clips["ntsc"], tmp_path / "out.mp4", audio=clips["ntsc"])
    assert len(got) == 119 and all(np.array_equal(a, b) for a, b in zip(got, expected))
    assert report.frames == 119 and report.faststart
    assert (report.video_codec, report.pix_fmt, report.audio_codec) == ("h264", "yuv420p", "aac")
    assert report.color_tags == {"color_space": "bt709", "color_primaries": "bt709",
                                 "color_transfer": "bt709", "color_range": "tv"}
    assert abs(report.duration - src.info.duration) < 1 / 23.976
    assert abs(report.audio_duration - report.duration) < 0.03  # -shortest: one AAC packet
    # Colour: less drift than the spec's bare command (untagged, default scaler).
    spec = tmp_path / "spec.mp4"
    subprocess.run([find_tool("ffmpeg"), "-y", "-v", "error", "-f", "rawvideo", "-vcodec",
                    "rawvideo", "-s", "320x240", "-pix_fmt", "bgr24", "-r", "24000/1001", "-i",
                    "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", "-preset", "medium", "-crf", "18", str(spec)],
                   input=b"".join(e.tobytes() for e in expected), check=True)

    def drift(path: Path) -> float:
        decoded = [f.image.astype(float) for f in VideoSource(path).frames()]
        return float(np.mean([np.abs(d - e).mean() for d, e in zip(decoded, expected)]))

    assert drift(report.path) < drift(spec)


@pytest.mark.parametrize("audio", ["original", "sidecar"])
def test_audio_stays_in_sync(clips: dict[str, Path], tmp_path: Path, audio: str) -> None:
    source = clips["click"] if audio == "original" else \
        VideoSource(clips["click"]).demux_streams(tmp_path / ".temp")["audio"][0]
    _, report = _render(clips["click"], tmp_path / "sync.mp4", audio=source)
    assert abs(_onset(report.path) - 1.0) < 0.001  # the click is at exactly 1.000 s
    # This clip's audio is 3 ms SHORTER than its video: -shortest would drop frames.
    assert report.frames == 72


def test_vfr_source_keeps_its_duration(clips: dict[str, Path], tmp_path: Path) -> None:
    _, report = _render(clips["vfr"], tmp_path / "vfr.mp4", audio=clips["vfr"])
    assert report.frames == 180
    assert abs(report.duration - probe(clips["vfr"]).duration) < 0.05
    assert abs(report.audio_duration - report.duration) < 0.03


def test_odd_size_is_padded_and_opus_is_encoded(clips: dict[str, Path], tmp_path: Path) -> None:
    # x264 cannot even store 321x241, so odd frames are fed to the writer directly.
    frames = [np.full((241, 321, 3), 40 * i, np.uint8) for i in range(24)]
    for expected in (24, None):  # bounded copy path and padded re-encode path
        out = tmp_path / f"odd_{expected}.mp4"
        with FFmpegWriter(out, 321, 241, Fraction(24000, 1001), audio=clips["opus"],
                          expected_frames=expected) as writer:
            for frame in frames:
                writer.write(frame)
            report = writer.close()
        assert report.audio_codec == "aac" and report.frames == 24
        info = probe(out)
        assert (info.width, info.height) == (322, 242)
    with pytest.raises(ValueError, match="expected_frames"):
        FFmpegWriter(tmp_path / "x.mp4", 64, 64, 24, audio=clips["ntsc"], audio_codec="copy")


def test_silent_render_and_writer_errors(clips: dict[str, Path], tmp_path: Path) -> None:
    _, report = _render(clips["silent"], tmp_path / "silent.mp4")
    assert report.audio_codec is None and report.frames == 48
    with pytest.raises(ValueError, match="tone-map"):
        FFmpegWriter(tmp_path / "hdr.mp4", 64, 64, 24, color=ColorProfile.BT2020_PQ)
    writer = FFmpegWriter(tmp_path / "bad.mp4", 64, 64, 24)
    with pytest.raises(ValueError):
        writer.write(np.zeros((32, 64, 3), np.uint8))
    writer._proc.kill()  # ffmpeg dies mid-stream
    writer._proc.wait()
    with pytest.raises(FFmpegError):
        for _ in range(200):
            writer.write(np.zeros((64, 64, 3), np.uint8))
    writer.abort()
    assert not (tmp_path / "bad.mp4").exists()


# ------------------------------------------------------------------ HTTP 206
def test_http_range_streaming(clips: dict[str, Path], tmp_path: Path) -> None:
    _, report = _render(clips["ntsc"], tmp_path / "web.mp4", audio=clips["ntsc"])
    result = verify_http_range_streaming(report.path)
    assert result["frames_over_http"] == 119 and result["seek_checked"] is False
    # Big enough that ffmpeg's HTTP client must seek with a mid-file range.
    big = tmp_path / "big.mp4"
    subprocess.run([find_tool("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=1280x720:rate=30:duration=8", "-c:v", "libx264",
                    "-preset", "ultrafast", "-crf", "10", "-g", "30", "-movflags", "+faststart",
                    str(big)], check=True)
    assert inspect_output(big).faststart
    result = verify_http_range_streaming(big)
    assert result["seek_checked"] and any(r != "bytes=0-" for r in result["seek_ranges"])
