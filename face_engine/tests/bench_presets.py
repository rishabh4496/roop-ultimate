"""Measure the server's presets: preview latency and full render fps.

Not collected by pytest. Usage::

    python -m face_engine.tests.bench_presets VIDEO SOURCE_IMAGE [--frames 300]

For each preset in ``face_engine.server.processing.PRESETS``:

* preview: the first request after a parameter change (model load), a cache
  miss (seek + decode to the frame), and warm cache hits on 20 frames inside
  already-decoded GOPs; median total and the decode / process / encode split.
* render: the server's own job path (decode -> GpuFrameProcessor -> NVENC ->
  remux) over the first ``--frames`` frames, after a warm-up render so
  TensorRT builds are not counted. Two rates: rendering (the job's own fps,
  from the end of its first batch) and the whole job (plus model loading and the remux,
  which dominate a short clip).
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from face_engine.media.tools import find_tool
from face_engine.server.preview import PreviewService
from face_engine.server.processing import PRESETS, RenderParams
from face_engine.server.state import AppState, ServerSettings


def trim(video: Path, frames: int, out: Path) -> Path:
    subprocess.run([find_tool("ffmpeg"), "-y", "-v", "error", "-i", str(video), "-frames:v",
                    str(frames), "-c:v", "copy", "-c:a", "copy", str(out)], check=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("source")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--workspace", default=".temp/bench_presets")
    ap.add_argument("--presets", default=",".join(PRESETS))
    args = ap.parse_args()
    ws = Path(args.workspace)
    ws.mkdir(parents=True, exist_ok=True)
    clip = trim(Path(args.video), args.frames, ws / "clip.mp4")
    state = AppState(ServerSettings(workspace=ws / "workspace"))
    state.load_project([("source", Path(args.source))], ("clip.mp4", clip))
    preview = PreviewService(state)
    print(f"{Path(args.video).name}: {state.target.width}x{state.target.height}, "
          f"{state.target.frames} frames", flush=True)
    for name in args.presets.split(","):
        preset = PRESETS[name]
        params = RenderParams(**preset["params"])
        t0 = time.perf_counter()
        preview.render(0, params, "swapped")
        first = (time.perf_counter() - t0) * 1000
        _, miss = preview.render(150, params, "swapped")
        time.sleep(3.0)  # let the GOP prefetch finish
        hits = [preview.render(int(i), params, "swapped")[1]
                for i in np.linspace(0, 150, 20).astype(int)]
        hits = [h for h in hits if h["cache"] == "hit"]
        med = {k: statistics.median(h[k] for h in hits)
               for k in ("render_ms", "decode_ms", "process_ms", "encode_ms")}
        print(f"  {preset['label']:22s} preview: first {first:7.0f} ms (model load)  "
              f"miss {miss['render_ms']:6.1f} ms  hit median {med['render_ms']:5.1f} ms "
              f"[decode {med['decode_ms']:.1f} + process {med['process_ms']:.1f} + "
              f"jpeg {med['encode_ms']:.2f}] over {len(hits)} hits, "
              f"faces {hits[0].get('faces')}", flush=True)
        preview.reset()
        fps = []
        for run in range(2):  # the first render builds / warms engines
            job = state.start_job(params)
            job.thread.join()
            if job.state != "completed":
                raise SystemExit(f"{name}: render {job.state}: {job.message}")
            if run:
                fps.append((job.fps, job.frames_done / job.elapsed))
        print(f"  {preset['label']:22s} render: {fps[0][0]:5.1f} fps rendering "
              f"(after the first batch), {fps[0][1]:5.1f} fps whole job "
              f"(incl. model load + remux), {job.frames_done} frames", flush=True)
    state.shutdown()


if __name__ == "__main__":
    main()
