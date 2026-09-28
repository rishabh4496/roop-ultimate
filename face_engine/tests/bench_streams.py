"""A/B: the server's sequential render vs the Stage 7 stream pipeline.

Not collected by pytest. Usage::

    python -m face_engine.tests.bench_streams VIDEO SOURCE_IMAGE [--frames 600]

For each preset in ``face_engine.server.processing.PRESETS``, after one
warm-up render (TensorRT / ORT engines load or build): sequential, streams,
streams, sequential (A-B-B-A, so neither arm always runs first). Each run
builds its own :class:`GpuFrameProcessor`, as a server job does.

* sequential: the Stage 4-6 server render, kept here as the reference:
  ``HardwareVideoDecoder`` thread -> ``process_tensor`` per frame on the
  default stream -> NVENC writer thread -> remux (:func:`render_sequential`).
* streams: :class:`~face_engine.core.cuda_streams.CUDAStreamPipeline` (the
  server's render since Stage 7).

Printed per run: steady fps (after the first frames), whole-job fps (wall
clock incl. model load and remux), and for the last pair the per-frame mean
absolute difference between the two outputs (a missing stream wait shows up
as a spike on some frames, not as a shifted mean).
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from face_engine.core.cuda_streams import CUDAStreamPipeline
from face_engine.media.tools import find_tool
from face_engine.server.processing import (
    PRESETS,
    GpuFrameProcessor,
    ProcessorConfig,
    RenderParams,
)
from face_engine.server.state import AppState, ServerSettings


def trim(video: Path, frames: int, out: Path) -> Path:
    subprocess.run([find_tool("ffmpeg"), "-y", "-v", "error", "-i", str(video), "-frames:v",
                    str(frames), "-c:v", "copy", "-c:a", "copy", str(out)], check=True)
    return out


def frame_diffs(a: Path, b: Path) -> np.ndarray:
    import av

    out = []
    with av.open(str(a)) as ca, av.open(str(b)) as cb:
        for fa, fb in zip(ca.decode(video=0), cb.decode(video=0)):
            x = fa.to_ndarray(format="bgr24").astype(np.int16)
            y = fb.to_ndarray(format="bgr24").astype(np.int16)
            out.append(float(np.abs(x - y).mean()))
    return np.asarray(out)


def render_sequential(source: Path, out: Path, config: ProcessorConfig) -> tuple[float, int]:
    """The pre-Stage 7 ``AppState._render_video`` loop: ``(steady fps, frames)``."""
    import torch

    from face_engine.media.decoder import HardwareVideoDecoder
    from face_engine.media.demuxer import remux
    from face_engine.media.encoder import open_video_writer

    decoder = HardwareVideoDecoder(source, batch_size=4)
    info = decoder.info
    video = out.with_name(f".{out.stem}_video.mp4")
    processor = GpuFrameProcessor(config, tracking=True)
    done, fps, warm = 0, 0.0, None
    try:
        writer = open_video_writer(video, info.width, info.height, info.fps,
                                   expected_frames=info.frame_count, color=info.color_profile)
        try:
            for batch in decoder:
                frames = torch.cat([processor.process_tensor(f[None])[0] for f in batch.frames])
                writer.write_tensor(frames)
                done += len(batch)
                now = time.monotonic()
                if warm is None:
                    warm = (now, done)
                elif now > warm[0]:
                    fps = (done - warm[1]) / (now - warm[0])
            writer.close()
        except BaseException:
            writer.abort()
            raise
    finally:
        processor.close()
    try:
        remux(video, source, out)
    finally:
        video.unlink(missing_ok=True)
    return fps, done


def run_sequential(state: AppState, params: RenderParams,
                   out: Path | None = None) -> tuple[float, float, Path]:
    out = out or state.outputs_dir() / "sequential.mp4"
    config = state.processor_config(params)
    t0 = time.perf_counter()
    fps, frames = render_sequential(state.target.path, out, config)
    return fps, frames / (time.perf_counter() - t0), out


def run_streams(state: AppState, params: RenderParams, out: Path,
                **options: object) -> tuple[float, float, Path]:
    config = state.processor_config(params)
    t0 = time.perf_counter()
    stats = CUDAStreamPipeline(**options).run(state.target.path, out, config)
    wall = time.perf_counter() - t0
    return stats.fps, stats.frames_done / wall, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("source")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--workspace", default=".temp/bench_streams")
    ap.add_argument("--presets", default=",".join(PRESETS))
    args = ap.parse_args()
    ws = Path(args.workspace)
    ws.mkdir(parents=True, exist_ok=True)
    clip = trim(Path(args.video), args.frames, ws / "clip.mp4")
    state = AppState(ServerSettings(workspace=ws / "workspace"))
    state.load_project([("source", Path(args.source))], ("clip.mp4", clip))
    print(f"{Path(args.video).name}: {state.target.width}x{state.target.height}, "
          f"{state.target.frames} frames", flush=True)
    for name in args.presets.split(","):
        params = RenderParams(**PRESETS[name]["params"])
        label = PRESETS[name]["label"]
        run_sequential(state, params)  # warm-up: engine builds / loads
        rows: dict[str, list[tuple[float, float]]] = {"sequential": [], "streams": []}
        outputs: dict[str, Path] = {}
        for i, arm in enumerate(("sequential", "streams", "streams", "sequential")):
            if arm == "sequential":
                steady, whole, path = run_sequential(state, params)
            else:
                steady, whole, path = run_streams(state, params, ws / f"streams_{name}_{i}.mp4")
            rows[arm].append((steady, whole))
            outputs[arm] = path
            print(f"  {label:22s} {arm:10s} {steady:6.1f} fps steady  {whole:6.1f} fps whole job",
                  flush=True)
        for arm, r in rows.items():
            print(f"  {label:22s} {arm:10s} mean {statistics.mean(x[0] for x in r):6.1f} steady "
                  f"/ {statistics.mean(x[1] for x in r):6.1f} whole", flush=True)
        _, _, pyav = run_streams(state, params, ws / f"streams_{name}_pyav.mp4", gpu_color=False)
        for tag, other in (("streams", outputs["streams"]), ("streams, PyAV colour", pyav)):
            d = frame_diffs(outputs["sequential"], other)
            print(f"  {label:22s} output diff sequential vs {tag}: {d.size} frames, mean "
                  f"{d.mean():.2f}, p99 {np.percentile(d, 99):.2f}, max {d.max():.2f} "
                  f"(frame {int(d.argmax())})", flush=True)
    state.shutdown()


if __name__ == "__main__":
    main()
