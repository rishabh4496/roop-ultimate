"""Stage 4 hardware I/O: NVDEC decoder, NVENC writer, demux/remux, segment worker pool.

Clips are synthesised with ffmpeg (``media_fixtures``), so every property is
known exactly. Needs a CUDA device for the decoder/writer; NVENC-specific
checks skip when ffmpeg cannot open ``h264_nvenc``.
"""
from __future__ import annotations

import subprocess
from fractions import Fraction
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from face_engine.media.capturer import VideoSource, probe
from face_engine.media.decoder import HardwareVideoDecoder, nv12_to_bgr
from face_engine.media.demuxer import demux, remux
from face_engine.media.encoder import (
    NVENCVideoWriter,
    X264TensorWriter,
    nvenc_available,
    open_video_writer,
)
from face_engine.media.ffmpeg_pipe import inspect_output
from face_engine.media.tools import ffprobe_json, find_tool
from face_engine.media.worker_pool import SegmentWorkerPool
from face_engine.tests import media_fixtures as mf

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")]
needs_nvenc = pytest.mark.skipif(not nvenc_available(), reason="h264_nvenc unavailable")


def _ffmpeg(*args: str) -> None:
    subprocess.run([find_tool("ffmpeg"), "-y", "-v", "error", *args], check=True)


@pytest.fixture(scope="module")
def clips(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("hwclips")
    click = d / "click.mp4"
    # 72 frames at 24000/1001 (3.003 s) with a click at exactly 1.000 s and
    # 3.000 s of audio: -shortest would drop frames here.
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24000/1001:duration=3",
            "-f", "lavfi", "-i", "aevalsrc='if(between(t,1,1.005),sin(2*PI*1000*t),0)':s=48000:d=3",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-c:a", "aac", "-b:a", "192k", str(click))
    ten = d / "tenbit.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=1", "-c:v", "libx264",
            "-pix_fmt", "yuv420p10le", str(ten))
    base = mf.ntsc_clip(d / "ntsc.mp4")  # 119 frames, B-frames, keyframe every 12
    meta = d / "chapters.txt"
    meta.write_text(";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=2000\ntitle=One\n"
                    "[CHAPTER]\nTIMEBASE=1/1000\nSTART=2000\nEND=5000\ntitle=Two\n",
                    encoding="utf-8")
    chaptered = d / "chapters.mp4"
    _ffmpeg("-i", str(base), "-i", str(meta), "-map_metadata", "1", "-map_chapters", "1",
            "-c", "copy", str(chaptered))
    subs = mf.with_subtitles(mf.ntsc_clip(d / "sbase.mp4", seconds=2), d / "subs.mp4",
                             "1\n00:00:00,500 --> 00:00:01,500\nhello\n")
    return {"click": click, "ten": ten, "ntsc": base, "chapters": chaptered, "subs": subs,
            "dir": d}


# ------------------------------------------------------------------ colour
def test_nv12_to_bgr_known_values() -> None:
    """BT.709 limited range: Y 16 = black, 235 = white; primaries land on their channel."""
    h, w = 2, 2

    def nv12(y: float, u: float, v: float) -> torch.Tensor:
        t = torch.empty((1, h * 3 // 2, w), dtype=torch.uint8, device="cuda")
        t[:, :h] = y
        t[:, h:, 0::2] = u
        t[:, h:, 1::2] = v
        return t

    def bgr(y: float, u: float, v: float) -> list[int]:
        return nv12_to_bgr(nv12(y, u, v), h, w)[0, :, 0, 0].tolist()

    assert bgr(16, 128, 128) == [0, 0, 0]
    assert bgr(235, 128, 128) == [255, 255, 255]
    # BT.709 red (R'=1): Y=63, Cb=102, Cr=240
    b, g, r = bgr(63, 102, 240)
    assert r >= 253 and g <= 2 and b <= 2


# ------------------------------------------------------------------ decoder
@pytest.mark.parametrize("backend", ["nvdec", "software"])
def test_decoder_is_frame_exact(clips: dict[str, Path], backend: str) -> None:
    ref = {f.index: f.image for f in VideoSource(clips["ntsc"]).frames()}
    decoder = HardwareVideoDecoder(clips["ntsc"], backend=backend, batch_size=4)
    assert decoder.backend == backend
    assert decoder.fps == Fraction(24000, 1001) and decoder.frame_count == 119
    assert decoder.sample_aspect_ratio == 1
    got: dict[int, np.ndarray] = {}
    for batch in decoder:
        assert batch.frames.device.type == "cuda" and batch.frames.dtype == torch.uint8
        assert tuple(batch.frames.shape[1:]) == (3, 240, 320)
        for k, i in enumerate(batch.indices):
            got[i] = batch.frames[k].permute(1, 2, 0).cpu().numpy()
    assert sorted(got) == list(range(119))
    for i in range(119):
        diff = np.abs(got[i].astype(int) - ref[i])
        if backend == "software":
            assert diff.max() == 0
        else:  # GPU conversion vs swscale's default rounding
            assert diff.mean() < 2.0 and diff.max() <= 6


def test_decoder_range_starts_mid_gop(clips: dict[str, Path]) -> None:
    ref = {f.index: f.image for f in VideoSource(clips["ntsc"]).frames(13, 40)}
    got = [i for b in HardwareVideoDecoder(clips["ntsc"], start=13, end=40, batch_size=5)
           for i in b.indices]
    assert got == list(range(13, 40)) and sorted(ref) == got


def test_ten_bit_falls_back_to_software(clips: dict[str, Path]) -> None:
    decoder = HardwareVideoDecoder(clips["ten"], backend="auto")  # would pick NVDEC if it could
    assert decoder.backend == "software"
    assert sum(len(b) for b in decoder) == 25
    with pytest.raises(RuntimeError, match="NVDEC"):
        HardwareVideoDecoder(clips["ten"], backend="nvdec")


# ------------------------------------------------------------------ encoder
@needs_nvenc
def test_nvenc_writer_is_web_ready_and_in_sync(clips: dict[str, Path], tmp_path: Path) -> None:
    info = probe(clips["click"])
    out = tmp_path / "nvenc.mp4"
    writer = open_video_writer(out, info.width, info.height, info.fps, encoder="nvenc",
                               audio=clips["click"], expected_frames=info.frame_count,
                               color=info.color_profile)
    assert isinstance(writer, NVENCVideoWriter)
    frames = []
    for batch in HardwareVideoDecoder(clips["click"], batch_size=3):
        writer.write_tensor(batch.frames)  # (B, 3, H, W) uint8
        frames.append(batch.frames.permute(0, 2, 3, 1).cpu().numpy())
    report = writer.close()
    assert report.frames == 72 == info.frame_count  # audio is 3 ms short: nothing dropped
    assert (report.video_codec, report.pix_fmt, report.audio_codec) == ("h264", "yuv420p", "aac")
    assert report.faststart
    assert report.color_tags == {"color_space": "bt709", "color_primaries": "bt709",
                                 "color_transfer": "bt709", "color_range": "tv"}
    encoder_name = ffprobe_json(out, "-show_streams")["streams"][0].get("codec_tag_string")
    assert encoder_name == "avc1"
    pcm = subprocess.run([find_tool("ffmpeg"), "-v", "error", "-i", str(out), "-map", "0:a:0",
                          "-f", "f32le", "-ac", "1", "-ar", "48000", "-"],
                         capture_output=True, check=True).stdout
    a = np.abs(np.frombuffer(pcm, np.float32))
    assert abs(np.argmax(a > 0.2 * a.max()) / 48000 - 1.0) < 0.001
    decoded = [f.image for f in VideoSource(out).frames()]
    src = np.concatenate(frames)
    psnr = [10 * np.log10(255 ** 2 / max(np.mean((d.astype(float) - s) ** 2), 1e-9))
            for d, s in zip(decoded, src)]
    assert min(psnr) > 35.0


@needs_nvenc
def test_write_tensor_layouts_and_validation(tmp_path: Path) -> None:
    with NVENCVideoWriter(tmp_path / "l.mp4", 160, 64, Fraction(25), expected_frames=3) as w:
        w.write_tensor(torch.zeros((3, 64, 160), device="cuda"))            # CHW float
        w.write_tensor(torch.zeros((64, 160, 3), dtype=torch.uint8, device="cuda"))  # HWC
        w.write_tensor(torch.full((1, 3, 64, 160), 300.0, device="cuda"))   # clamped
        with pytest.raises(ValueError):
            w.write_tensor(torch.zeros((3, 40, 160), device="cuda"))
        report = w.close()
    assert report.frames == 3


@needs_nvenc
def test_frames_below_the_nvenc_minimum_use_x264(tmp_path: Path) -> None:
    """h264_nvenc refuses frames under 145x49; auto routes them to libx264."""
    w = open_video_writer(tmp_path / "tiny.mp4", 64, 48, Fraction(25), expected_frames=2)
    assert isinstance(w, X264TensorWriter)
    w.write_tensor(torch.zeros((2, 3, 48, 64), device="cuda"))
    assert w.close().frames == 2
    with pytest.raises(Exception, match="at least 145x49"):
        open_video_writer(tmp_path / "t2.mp4", 64, 48, Fraction(25), encoder="nvenc")


def test_writer_falls_back_to_x264(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import face_engine.media.encoder as enc

    monkeypatch.setattr(enc, "nvenc_available", lambda *a, **k: False)
    w = enc.open_video_writer(tmp_path / "x.mp4", 160, 64, Fraction(25), expected_frames=2)
    assert isinstance(w, X264TensorWriter)
    w.write_tensor(torch.zeros((2, 3, 64, 160), device="cuda"))
    assert w.close().frames == 2
    with pytest.raises(Exception, match="not available"):
        enc.open_video_writer(tmp_path / "y.mp4", 160, 64, Fraction(25), encoder="nvenc")


# ------------------------------------------------------------------ demux / remux
def test_demux_extracts_audio_subtitles_and_chapters(clips: dict[str, Path],
                                                    tmp_path: Path) -> None:
    got = demux(clips["chapters"], tmp_path / "a")
    assert [p.suffix for p in got.audio] == [".m4a"] and got.chapters is not None
    assert got.chapters.read_text(encoding="utf-8").count("[CHAPTER]") == 2
    subs = demux(clips["subs"], tmp_path / "b")
    assert [p.suffix for p in subs.subtitles] == [".srt"]
    assert "hello" in subs.subtitles[0].read_text(encoding="utf-8")


def _silent_copy(src: Path, dst: Path) -> Path:
    _ffmpeg("-i", str(src), "-map", "0:v:0", "-c", "copy", str(dst))
    return dst


def test_remux_keeps_frames_sync_subtitles_and_chapters(clips: dict[str, Path],
                                                        tmp_path: Path) -> None:
    # Click clip: audio 3 ms shorter than video. -shortest would drop frames.
    video = _silent_copy(clips["click"], tmp_path / "v.mp4")
    out = remux(video, clips["click"], tmp_path / "click_out.mp4")
    assert out.frames == 72 and out.audio_codec == "aac"
    pcm = subprocess.run([find_tool("ffmpeg"), "-v", "error", "-i", str(out.path), "-map",
                          "0:a:0", "-f", "f32le", "-ac", "1", "-ar", "48000", "-"],
                         capture_output=True, check=True).stdout
    a = np.abs(np.frombuffer(pcm, np.float32))
    assert abs(np.argmax(a > 0.2 * a.max()) / 48000 - 1.0) < 0.001
    subs = remux(_silent_copy(clips["subs"], tmp_path / "s.mp4"), clips["subs"],
                 tmp_path / "subs_out.mp4")
    kinds = [s["codec_name"] for s in ffprobe_json(subs.path, "-show_streams")["streams"]]
    assert "mov_text" in kinds
    chap = remux(_silent_copy(clips["chapters"], tmp_path / "c.mp4"), clips["chapters"],
                 tmp_path / "chap_out.mp4")
    assert len(ffprobe_json(chap.path, "-show_chapters")["chapters"]) == 2


# ------------------------------------------------------------------ worker pool
@needs_nvenc
def test_segment_pool_is_seamless_and_frees_shared_memory(clips: dict[str, Path],
                                                          tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    import face_engine.media.worker_pool as wp

    created: list[str] = []
    original = wp.create_owned

    def spy(size: int) -> shared_memory.SharedMemory:
        shm = original(size)
        created.append(shm.name)
        return shm

    monkeypatch.setattr(wp, "create_owned", spy)
    seen: list[tuple[int, int]] = []
    report = SegmentWorkerPool([0, 0], encoder="nvenc").run(
        clips["ntsc"], tmp_path / "pool.mp4", on_progress=lambda d, t: seen.append((d, t)))
    assert report.frames == 119 == report.output.frames
    assert len(report.segments) == 2 and sum(r["frames"] for r in report.segments) == 119
    assert seen and seen[-1][1] == 119
    # Seamless: every output frame matches its source frame; timestamps uniform.
    ref = [f.image for f in VideoSource(clips["ntsc"]).frames()]
    out = [f.image for f in VideoSource(tmp_path / "pool.mp4").frames()]
    assert len(out) == 119
    psnr = [10 * np.log10(255 ** 2 / max(np.mean((o.astype(float) - r) ** 2), 1e-9))
            for o, r in zip(out, ref)]
    assert min(psnr) > 30.0
    packets = ffprobe_json(tmp_path / "pool.mp4", "-select_streams", "v:0",
                           "-show_entries", "packet=pts_time", "-show_packets")["packets"]
    steps = np.diff(sorted(float(p["pts_time"]) for p in packets))
    assert np.allclose(steps, 1001 / 24000, atol=1e-3)
    assert abs(report.output.audio_duration - report.output.duration) < 0.05
    # The progress segment was unlinked (POSIX) / closed (Windows frees it with the last handle).
    for name in created:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)
    assert not (tmp_path / ".pool_parts").exists()
    assert inspect_output(tmp_path / "pool.mp4").faststart
