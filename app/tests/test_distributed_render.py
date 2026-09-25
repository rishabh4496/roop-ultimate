import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roop import distributed_render as dist


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())


def test_plan_gop_boundaries_and_stream_copy(tmp_path):
    source = tmp_path / "source.mp4"
    dist._run([dist.ffmpeg_binary(), "-v", "error", "-f", "lavfi", "-i",
               "testsrc2=size=256x256:rate=20:duration=2", "-c:v", "libx264",
               "-g", "10", "-keyint_min", "10", "-sc_threshold", "0",
               "-bf", "2", str(source)])
    chunks, codec, rate = dist.plan_chunks(str(source), 0.49)
    assert codec == "h264"
    assert rate == 20
    assert [(c.start_frame, c.end_frame) for c in chunks] == [
        (0, 10), (10, 20), (20, 30), (30, 40)]
    for chunk in chunks:
        sliced = tmp_path / f"slice{chunk.index}.mp4"
        dist.copy_slice(str(source), chunk, codec, str(sliced))
        assert dist._count(str(sliced)) == chunk.frames
        assert dist._first_nal_is_idr(str(sliced), codec)
    listing = tmp_path / "concat.txt"
    listing.write_text("".join(f"file 'slice{i}.mp4'\n" for i in range(4)),
                       encoding="utf-8")
    joined = tmp_path / "joined.mp4"
    dist._run([dist.ffmpeg_binary(), "-v", "error", "-f", "concat", "-safe", "0",
               "-i", str(listing), "-c", "copy", str(joined)])
    assert dist._count(str(joined)) == 40


def test_non_keyframe_trim_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(dist, "_probe", lambda *a, **kw: {
        "streams": [{"codec_name": "h264", "avg_frame_rate": "20/1"}],
        "packets": [{"pts_time": str(i / 20), "flags": "K_" if i % 10 == 0 else "__"}
                    for i in range(21)]})
    with pytest.raises(ValueError, match="trim start"):
        dist.plan_chunks("unused", frame_start=1)
    with pytest.raises(ValueError, match="trim end"):
        dist.plan_chunks("unused", frame_end=9)


def test_vfr_rejected(monkeypatch):
    monkeypatch.setattr(dist, "_probe", lambda *a, **kw: {
        "streams": [{"codec_name": "hevc", "avg_frame_rate": "20/1"}],
        "packets": [{"pts_time": "0", "flags": "K_"},
                    {"pts_time": ".05", "flags": "__"},
                    {"pts_time": ".20", "flags": "K_"}]})
    with pytest.raises(ValueError, match="variable-frame-rate"):
        dist.plan_chunks("unused")


def test_gpu_mask_respected(monkeypatch):
    class Result:
        stdout = "0, GPU-a\n1, GPU-b\n2, GPU-c\n"
    monkeypatch.setattr(dist, "_run", lambda *a, **kw: Result())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0")
    assert dist.detect_gpus() == ["GPU-c", "GPU-a"]


def test_chunk_project_is_local_and_audio_disabled(tmp_path):
    originals = tmp_path / "originals"
    originals.mkdir()
    asset = originals / "face.png"
    asset.write_bytes(b"test")
    original_project = originals / "job.roop"
    doc = {"format": "roop-session", "project_version": 1,
           "media": {"target": {}, "sources": [{"id": "source-1",
               "relative_path": "face.png", "absolute_path": "C:/stale/face.png"}]},
           "timeline": {"frame_start": 20, "frame_end": 30},
           "settings": {"blend_ratio": .85}, "automation": {}}
    project = tmp_path / "worker.roop"
    chunk = dist.Chunk(0, 20, 30, Fraction(1), Fraction(3, 2))
    dist._project_for_chunk(doc, str(original_project), str(tmp_path / "slice.mp4"),
                            chunk, str(tmp_path / "out.mp4"), str(project))
    saved = json.loads(project.read_text(encoding="utf-8"))
    assert saved["timeline"]["frame_start"] == 0
    assert saved["timeline"]["frame_end"] == 10
    assert saved["settings"]["skip_audio"] is True
    assert saved["settings"]["blend_ratio"] == .85
    assert saved["render"]["filename"] == "out.mp4"
    assert dist.project_io.resolve_media(saved["media"]["sources"][0], str(project)) == str(asset)
