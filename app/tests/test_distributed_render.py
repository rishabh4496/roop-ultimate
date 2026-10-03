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


# --- multi-GPU worker isolation (Stage 5) -----------------------------------
# One GPU is all this rig has, so the orchestration is exercised with stubs; what is
# asserted is the contract that makes the CUDA-stream work safe across workers: each
# worker is its OWN process (torch streams, the caching allocator and cuBLAS workspaces
# are process-private), pinned to exactly one physical GPU and addressing it as device
# 0, and no GPU ever runs two chunks at once.
def _stub_render_distributed(monkeypatch, tmp_path, gpus, n_chunks, render_seconds=0.05):
    import sys
    import threading
    import time
    import types

    source = tmp_path / "target.mp4"
    source.write_bytes(b"x")
    document = {"media": {"target": {}, "sources": []}, "timeline": {}, "settings": {}}
    chunks = [dist.Chunk(i, i * 10, (i + 1) * 10, Fraction(i), Fraction(i + 1))
              for i in range(n_chunks)]
    monkeypatch.setattr(dist.project_io, "load_project", lambda p: document)
    monkeypatch.setattr(dist.project_io, "resolve_media", lambda ref, p: str(source))
    monkeypatch.setattr(dist, "plan_chunks", lambda *a, **k: (chunks, "h264", Fraction(10)))
    monkeypatch.setattr(dist, "detect_gpus", lambda: list(gpus))
    monkeypatch.setattr(dist, "copy_slice",
                        lambda src, chunk, codec, dest: Path(dest).write_bytes(b"s"))

    def project_for_chunk(doc, project_path, src, chunk, output, destination):
        Path(destination).write_text(json.dumps({"id": f"p{chunk.index}"}), encoding="utf-8")
    monkeypatch.setattr(dist, "_project_for_chunk", project_for_chunk)
    monkeypatch.setattr(dist, "_export_elementary", lambda c, o, codec: Path(o).write_bytes(b"e"))
    monkeypatch.setattr(dist, "_signature", lambda path: ("h264", 1920, 1080))
    monkeypatch.setattr(dist, "_duration",
                        lambda path: float(sum(c.frames for c in chunks)) / 10
                        if path.endswith("joined.mp4") or path.endswith("final.mp4")
                        else 1.0)
    total = sum(c.frames for c in chunks)
    monkeypatch.setattr(dist, "_count",
                        lambda path: total if path.endswith(("joined.mp4", "final.mp4")) else 10)

    checkpoint = types.ModuleType("project_checkpoint")
    checkpoint.project_path = lambda cid: str(tmp_path / f"{cid}.ckpt")
    monkeypatch.setitem(sys.modules, "project_checkpoint", checkpoint)
    util = types.ModuleType("roop.util_ffmpeg")
    util.restore_audio = lambda joined, audio, a, b, out: Path(out).write_bytes(b"o") or True
    monkeypatch.setitem(sys.modules, "roop.util_ffmpeg", util)

    class Cfg:
        output_video_codec = "libx264"
        output_video_format = "mp4"
        clear_output = False
        video_swapping_method = "In-Memory processing"
    settings = types.ModuleType("settings")
    settings.Settings = lambda path: Cfg()
    monkeypatch.setitem(sys.modules, "settings", settings)

    calls, active, overlaps, lock = [], {}, [], threading.Lock()

    def fake_run(command, *, timeout=None, env=None, cwd=None):
        if "--render" in command:
            device = env["CUDA_VISIBLE_DEVICES"]
            with lock:
                calls.append({"command": list(command), "env": dict(env), "cwd": cwd,
                              "thread": threading.get_ident()})
                if active.get(device):
                    overlaps.append(device)
                active[device] = active.get(device, 0) + 1
            time.sleep(render_seconds)
            Path(command[command.index("--output") + 1]).write_bytes(b"r")
            with lock:
                active[device] -= 1
        elif command[-1].endswith("joined.mp4"):
            Path(command[-1]).write_bytes(b"j")
        return types.SimpleNamespace(stdout="", stderr="", returncode=0)
    monkeypatch.setattr(dist, "_run", fake_run)
    # the stub final file is named by the caller; give _count/_duration their names
    final = str(tmp_path / "final.mp4")
    dist.render_distributed(str(tmp_path / "job.roop"), final, work_dir=str(tmp_path / "w"))
    return calls, overlaps


def test_each_worker_is_a_separate_process_pinned_to_one_gpu(monkeypatch, tmp_path):
    calls, overlaps = _stub_render_distributed(monkeypatch, tmp_path, ["GPU-a", "GPU-b"], 6)
    assert len(calls) == 6                                   # every chunk rendered once
    for call in calls:
        command = call["command"]
        assert command[0] == sys.executable                  # a child process, not a thread
        assert command[command.index("--cuda_device_id") + 1] == "0"
        assert call["env"]["CUDA_VISIBLE_DEVICES"] in ("GPU-a", "GPU-b")
        assert "," not in call["env"]["CUDA_VISIBLE_DEVICES"]    # exactly one GPU
    assert {c["env"]["CUDA_VISIBLE_DEVICES"] for c in calls} == {"GPU-a", "GPU-b"}
    assert overlaps == []                                    # no GPU ran two chunks at once


def test_workers_never_outnumber_chunks_and_never_share_a_gpu(monkeypatch, tmp_path):
    calls, overlaps = _stub_render_distributed(monkeypatch, tmp_path,
                                               ["GPU-a", "GPU-b", "GPU-c"], 2)
    assert len(calls) == 2
    assert len({c["env"]["CUDA_VISIBLE_DEVICES"] for c in calls}) == 2   # one chunk per GPU
    assert overlaps == []
