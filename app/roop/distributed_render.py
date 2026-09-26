"""GOP-aligned, multi-process project rendering.

The temporary MP4s are the timing authority.  Elementary streams are exported
as deliverables, but cannot be used as concat inputs: Annex B has no PTS/DTS.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

_ROOP_DIR = Path(__file__).resolve().parent
_APP_DIR = _ROOP_DIR.parent
while str(_ROOP_DIR) in sys.path:
    sys.path.remove(str(_ROOP_DIR))
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from roop.ffmpeg_path import ffmpeg_binary, ffprobe_binary
from roop import project_io


@dataclass(frozen=True)
class Chunk:
    index: int
    start_frame: int
    end_frame: int
    start_time: Fraction
    end_time: Fraction
    first_packet_hash: str = ""

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame


def _run(command: list[str], *, timeout: int | None = None,
         env: dict | None = None, cwd: str | None = None) -> subprocess.CompletedProcess:
    kwargs = {"capture_output": True, "text": True, "timeout": timeout,
              "env": env, "cwd": cwd, "check": False}
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000
    result = subprocess.run(command, **kwargs)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n"
                           f"{result.stdout[-1500:]}\n{result.stderr[-3000:]}")
    return result


def _probe(path: str, *, packets: bool = False) -> dict:
    args = [ffprobe_binary(), "-v", "error", "-select_streams", "v:0",
            "-show_streams", "-show_format"]
    if packets:
        args += ["-show_packets", "-show_data_hash", "sha256", "-show_entries",
                 "stream=codec_name,avg_frame_rate,r_frame_rate,nb_frames,duration,time_base:"
                 "format=duration:packet=pts_time,flags,data_hash"]
    result = _run(args + ["-of", "json", path], timeout=600)
    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise ValueError(f"no video stream: {path}")
    return data


def plan_chunks(path: str, target_seconds: float = 10.0,
                frame_start: int = 0, frame_end: int = 0) -> tuple[list[Chunk], str, Fraction]:
    """Scan encoded packets, never decoded frames; fail on non-CFR or unsafe trim.

    FFprobe's K flag is a random-access indication, not a proof of an H.264
    IDR NAL.  A separate first-access-unit NAL check in ``copy_slice`` is
    mandatory before the worker sees a chunk.
    """
    data = _probe(path, packets=True)
    stream = data["streams"][0]
    codec = stream.get("codec_name")
    if codec not in ("h264", "hevc"):
        raise ValueError("GOP slicing supports H.264/H.265 sources only")
    rate = Fraction(stream.get("avg_frame_rate") or "0")
    if rate <= 0:
        raise ValueError("source has no usable frame rate")
    packets = data.get("packets") or []
    if not packets or any(p.get("pts_time") is None for p in packets):
        raise ValueError("source has missing video packet PTS")
    ordered = sorted(packets, key=lambda p: Fraction(p["pts_time"]))
    times = [Fraction(p["pts_time"]) for p in ordered]
    # The render engine is CFR.  Reject VFR rather than manufacturing a
    # frame/time mapping that would slip against audio after concat.
    period = 1 / rate
    if any(abs(float(b - a - period)) > max(0.0002, float(period) * 0.02)
           for a, b in zip(times, times[1:])):
        raise ValueError("variable-frame-rate source is not safe for CFR chunk rendering")
    end = len(ordered) if frame_end <= 0 else min(frame_end, len(ordered))
    if not 0 <= frame_start < end:
        raise ValueError("empty or invalid project trim")
    keys = {i for i, packet in enumerate(ordered) if "K" in packet.get("flags", "")}
    if frame_start not in keys:
        raise ValueError("trim start must be a random-access GOP boundary")
    if end < len(ordered) and end not in keys:
        raise ValueError("trim end must be a random-access GOP boundary")
    edges = [frame_start]
    next_goal = times[frame_start] + Fraction(str(target_seconds))
    for i in sorted(k for k in keys if frame_start < k < end):
        if times[i] >= next_goal:
            edges.append(i)
            next_goal = times[i] + Fraction(str(target_seconds))
    edges.append(end)
    if any(not ordered[index].get("data_hash") for index in edges[:-1]):
        raise ValueError("ffprobe did not provide keyframe packet hashes")
    chunks = [Chunk(n, a, b, times[a], times[b] if b < len(times) else times[-1] + period,
                    ordered[a].get("data_hash", ""))
              for n, (a, b) in enumerate(zip(edges, edges[1:]))]
    return chunks, codec, rate


def detect_gpus() -> list[str]:
    """Return physical indices/UUIDs permitted by the parent's CUDA mask."""
    result = _run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], timeout=15)
    devices = [tuple(part.strip() for part in line.split(",", 1))
               for line in result.stdout.splitlines() if "," in line]
    allowed = os.environ.get("CUDA_VISIBLE_DEVICES")
    if allowed is not None:
        tokens = [s.strip() for s in allowed.split(",") if s.strip()]
        devices = [device for token in tokens for device in devices
                   if token in device or (token.startswith("GPU-") and device[1].startswith(token))]
    return [uuid for _, uuid in devices]


def _first_nal_is_idr(path: str, codec: str) -> bool:
    """Inspect the first copied access unit's Annex-B VCL NAL, not its K flag.

    FFmpeg's bitstream filter converts MP4 length-prefixed NALs to Annex B
    without decoding.  H.264 IDR is type 5; HEVC IDR is type 19/20.  CRA
    (type 21) is deliberately excluded because it can have leading pictures.
    """
    fmt, bsf = (("h264", "h264_mp4toannexb") if codec == "h264" else
                ("hevc", "hevc_mp4toannexb"))
    raw = subprocess.run([ffmpeg_binary(), "-v", "error", "-i", path,
                          "-map", "0:v:0", "-frames:v", "1", "-c:v", "copy",
                          "-bsf:v", bsf, "-f", fmt, "-"],
                         capture_output=True, timeout=30)
    if raw.returncode:
        return False
    units = re.split(rb"\x00\x00\x00\x01|\x00\x00\x01", raw.stdout)
    for nal in units:
        if not nal:
            continue
        kind = (nal[0] & 31) if codec == "h264" else ((nal[0] >> 1) & 63)
        if codec == "h264" and 1 <= kind <= 5:
            return kind == 5
        if codec == "hevc" and kind <= 31:
            return kind in (19, 20)
    return False


def _count(path: str) -> int:
    result = _run([ffprobe_binary(), "-v", "error", "-select_streams", "v:0",
                   "-count_packets", "-show_entries", "stream=nb_read_packets",
                   "-of", "default=nw=1:nk=1", path], timeout=120)
    return int(result.stdout.strip())


def _signature(path: str) -> tuple:
    stream = _probe(path)["streams"][0]
    return tuple(stream.get(key) for key in (
        "codec_name", "width", "height", "pix_fmt", "avg_frame_rate",
        "color_primaries", "color_transfer", "color_space"))


def _duration(path: str) -> float:
    stream = _probe(path)["streams"][0]
    return float(stream.get("duration") or 0)


def copy_slice(source: str, chunk: Chunk, codec: str, destination: str) -> None:
    _run([ffmpeg_binary(), "-hide_banner", "-v", "error", "-y",
          "-ss", str(float(chunk.start_time)), "-i", source,
          "-map", "0:v:0", "-an", "-sn", "-dn", "-frames:v", str(chunk.frames),
          "-c:v", "copy", "-avoid_negative_ts", "make_zero", destination], timeout=300)
    actual = _count(destination)
    if actual != chunk.frames or not _first_nal_is_idr(destination, codec):
        raise RuntimeError(f"unsafe GOP slice {chunk.index}: {actual}/{chunk.frames} packets "
                           "or first access unit is not an IDR")
    if chunk.first_packet_hash:
        first = _run([ffprobe_binary(), "-v", "error", "-select_streams", "v:0",
                      "-show_packets", "-show_data_hash", "sha256", "-show_entries",
                      "packet=data_hash", "-of", "compact=p=0:nk=1", destination],
                     timeout=120).stdout.splitlines()[0].strip()
        if first != chunk.first_packet_hash:
            raise RuntimeError(f"GOP slice {chunk.index} starts on the wrong IDR packet")


def _project_for_chunk(document: dict, project_path: str, source: str,
                       chunk: Chunk, output: str, destination: str) -> None:
    item = copy.deepcopy(document)
    item["id"] = f"distributed_{uuid.uuid4().hex}"
    item["media"]["target"] = project_io.media_reference(source, destination, kind="target")
    item["media"]["sources"] = [
        {**ref, **project_io.media_reference(
            project_io.resolve_media(ref, project_path), destination,
            kind="source", asset_id=ref.get("id"))}
        for ref in item["media"].get("sources", [])]
    item["timeline"]["frame_start"] = 0
    item["timeline"]["frame_end"] = chunk.frames
    item["automation"] = copy.deepcopy(item.get("automation") or {})
    item["automation"]["in_out"] = {"in": 0, "out": chunk.frames}
    # Absolute-frame automation cannot be silently applied to local chunks.
    for name in ("keyframes", "fidelity_ramps", "mask_parameters"):
        if item["automation"].get(name):
            raise ValueError(f"distributed render does not support {name} automation")
    item["checkpoint"] = {"sequence": 0, "safe_frame": 0, "next_frame": 0, "segments": []}
    item["render"] = {"directory": str(Path(output).parent), "filename": Path(output).name}
    item["settings"] = copy.deepcopy(item.get("settings") or {})
    item["settings"].update({"skip_audio": True, "keep_frames": False,
                             "wait_after_extraction": False})
    # A project file is private to the worker; don't reuse an old checkpoint.
    project_io.save_project(destination, item, journal=False)


def _export_elementary(container: str, output: str, codec: str) -> None:
    fmt, bsf = (("h264", "h264_mp4toannexb") if codec == "h264" else
                ("hevc", "hevc_mp4toannexb"))
    _run([ffmpeg_binary(), "-v", "error", "-y", "-i", container,
          "-map", "0:v:0", "-c:v", "copy", "-bsf:v", bsf,
          "-f", fmt, output], timeout=300)


def render_distributed(project_path: str, output: str, *, target_seconds: float = 10.0,
                       work_dir: str | None = None) -> str:
    if target_seconds <= 0:
        raise ValueError("target_seconds must be positive")
    project_path = os.path.abspath(project_path)
    output = os.path.abspath(output)
    if Path(output).suffix.lower() not in (".mp4", ".mov", ".mkv"):
        raise ValueError("output must be MP4, MOV, or MKV")
    if os.path.exists(output):
        raise FileExistsError(output)
    document = project_io.load_project(project_path)
    source = project_io.resolve_media(document["media"]["target"], project_path)
    if not os.path.isfile(source):
        raise FileNotFoundError(source)
    timeline = document.get("timeline") or {}
    chunks, codec, rate = plan_chunks(source, target_seconds,
                                      int(timeline.get("frame_start") or 0),
                                      int(timeline.get("frame_end") or 0))
    gpus = detect_gpus()
    if not gpus:
        raise RuntimeError("no NVIDIA GPU available for distributed render")
    # The existing renderer writes a CFR stream at its configured codec.  It
    # must match the requested elementary type; no implicit transcode here.
    from settings import Settings
    cfg = Settings(str(Path(__file__).resolve().parents[1] / "config.yaml"))
    encoder = cfg.output_video_codec
    output_codec = "h264" if encoder in ("libx264", "h264_nvenc") else (
        "hevc" if encoder in ("libx265", "hevc_nvenc") else "")
    if not output_codec:
        raise ValueError(f"configure an H.264/H.265 encoder first (got {encoder})")
    if cfg.output_video_format != "mp4":
        raise ValueError("distributed workers require MP4 output format in config.yaml")
    if cfg.clear_output:
        raise ValueError("disable clear_output before distributed rendering")
    if cfg.video_swapping_method != "In-Memory processing":
        raise ValueError("distributed render requires in-memory video processing")
    root = Path(work_dir).resolve() if work_dir else Path(output + ".chunks")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"chunk workspace is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    tasks: queue.Queue[Chunk] = queue.Queue()
    for chunk in chunks:
        tasks.put(chunk)
    failures: list[Exception] = []
    lock = threading.Lock()
    app_dir = Path(__file__).resolve().parents[1]
    containers: list[str] = [""] * len(chunks)

    def worker(device: str) -> None:
        while True:
            try:
                chunk = tasks.get_nowait()
            except queue.Empty:
                return
            try:
                name = f"chunk_{chunk.index:05d}"
                sliced = str(root / f"{name}_source.mp4")
                rendered = str(root / f"{name}_render.mp4")
                child_project = str(root / f"{name}.roop")
                copy_slice(source, chunk, codec, sliced)
                _project_for_chunk(document, project_path, sliced, chunk, rendered, child_project)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = device
                command = [sys.executable, str(app_dir / "run.py"), "--project", child_project,
                           "--render", "--output", rendered, "--cuda_device_id", "0"]
                result = _run(command, env=env, cwd=str(app_dir))
                (root / f"{name}.log").write_text(result.stdout + "\n" + result.stderr,
                                                  encoding="utf-8")
                if not os.path.isfile(rendered) or _count(rendered) != chunk.frames:
                    raise RuntimeError(f"worker {device} rendered wrong frame count for {name}")
                _export_elementary(rendered, str(root / f"{name}.{output_codec}"), output_codec)
                containers[chunk.index] = rendered
                print(f"[Distributed] GPU {device} completed {name}: {chunk.frames} frames", flush=True)
                # Headless projects create durable checkpoints in app/projects;
                # this one is an implementation detail, not a user project.
                from project_checkpoint import project_path as checkpoint_path
                with open(child_project, encoding="utf-8") as fh:
                    checkpoint_id = json.load(fh)["id"]
                Path(checkpoint_path(checkpoint_id)).unlink(missing_ok=True)
            except Exception as exc:
                print(f"[Distributed] GPU {device} failed chunk {chunk.index}: {exc}",
                      file=sys.stderr, flush=True)
                with lock:
                    failures.append(exc)
            finally:
                tasks.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=True)
               for gpu in gpus[:len(chunks)]]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"{len(failures)} chunk(s) failed; intermediates kept in {root}: "
                           f"{failures[0]}")
    signatures = [_signature(path) for path in containers]
    if len(set(signatures)) != 1:
        raise RuntimeError(f"worker streams differ; cannot concat-copy: {signatures}")
    if signatures[0][0] != output_codec:
        raise RuntimeError(f"encoder produced {signatures[0][0]}, expected {output_codec}")
    listing = root / "concat.txt"
    with listing.open("w", encoding="utf-8") as fh:
        for container in containers:
            fh.write("file '%s'\n" % container.replace("\\", "/").replace("'", r"'\''"))
    joined = str(root / "joined.mp4")
    _run([ffmpeg_binary(), "-v", "error", "-y", "-f", "concat", "-safe", "0",
          "-i", str(listing), "-map", "0:v:0", "-c:v", "copy", joined], timeout=600)
    expected = sum(chunk.frames for chunk in chunks)
    if _count(joined) != expected:
        raise RuntimeError("concat frame count mismatch; intermediates retained")
    expected_seconds = float(Fraction(expected, 1) / rate)
    if abs(_duration(joined) - expected_seconds) > max(0.002, float(1 / rate)):
        raise RuntimeError("concat timestamp duration mismatch; intermediates retained")
    from roop.util_ffmpeg import restore_audio
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    audio_source = source
    settings = document.get("settings") or {}
    if settings.get("lipsync_enabled") and settings.get("lipsync_audio_source") == "upload":
        audio_source = settings.get("lipsync_audio_path") or source
    if settings.get("skip_audio"):
        shutil.copyfile(joined, output)
    elif not restore_audio(joined, audio_source, chunks[0].start_frame,
                           chunks[-1].end_frame, output):
        raise RuntimeError("audio remux failed; intermediates retained")
    if _count(output) != expected:
        raise RuntimeError("final frame count mismatch; intermediates retained")
    if abs(_duration(output) - expected_seconds) > max(0.002, float(1 / rate)):
        raise RuntimeError("final timestamp duration mismatch; intermediates retained")
    print(f"[Distributed] {len(chunks)} GOP chunks, {expected} frames, "
          f"{len(threads)} GPU worker(s) -> {output}; chunks: {root}", flush=True)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a .roop project across NVIDIA GPUs")
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--work-dir", help="keep stream-copy slices, rendered chunks, and elementary streams here")
    args = parser.parse_args(argv)
    render_distributed(args.project, args.output, target_seconds=args.chunk_seconds,
                       work_dir=args.work_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
