"""Benchmark the boosted pipeline: per-engine TensorRT latency, fps per preset, VRAM, CPU.

Usage (repo root)::

    python -m face_engine.boosted.benchmark_boosted --input face_engine/tests/sample_1080p.mp4 \\
        --frames 500 [--source face.jpg] [--presets boosted,ultra] [--json out.json]

1. **Engine latency**: each TensorRT engine on real inputs cut from the clip
   (R50 on a letterboxed frame; HyperSwap / XSeg on the clip's face crops,
   batch = faces per frame; GPEN-1024 on one 1024 crop), median of CUDA-event
   times on the stream the engine runs on. Also R50's whole ``detect_cuda``
   (letterbox + engine + decode + NMS).
2. **Throughput**: a real render of ``--frames`` frames per preset through
   :func:`~face_engine.boosted.ultra_pipeline.render`, with device VRAM (NVML)
   and host CPU sampled, and device-wide syncs on the render threads counted.

Exit 1 when the peak VRAM this run added exceeds ``--vram-limit-gb`` (6.0),
when faces were seen but not all swapped, when a device-wide synchronize ran
inside a render loop, or below ``--min-fps`` (per preset, optional).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


TARGET_FPS = {"boosted": (50.0, 70.0), "ultra": None}


@dataclass
class PresetResult:
    preset: str
    fps: float
    whole_fps: float
    faces: int
    swapped: int
    peak_vram_mb: float
    device_peak_vram_mb: float
    cpu_process_pct: float
    cpu_system_pct: float
    render_syncs: int


@dataclass
class Report:
    gpu: str
    input: str
    frames: int
    engines_ms: dict[str, float] = field(default_factory=dict)
    engine_backends: dict[str, str] = field(default_factory=dict)
    presets: list[PresetResult] = field(default_factory=list)
    vram_limit_mb: float = 6144.0
    failures: list[str] = field(default_factory=list)


def _event_ms(fn: Any, runs: int = 30, warmup: int = 5) -> float:
    import torch

    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return float(statistics.median(times))


def engine_latencies(video: Path, source: Path | None) -> tuple[dict[str, float], dict[str, str]]:
    import av
    import torch

    from face_engine.benchmark import source_embedding
    from face_engine.boosted.ultra_pipeline import BoostedFrameProcessor, preset_params
    from face_engine.models.zoo import build_default_registry
    from face_engine.pipeline.aligner import similarity_matrices_cuda, warp_face_cuda
    from face_engine.processors.enhancer import TEMPLATE as FFHQ
    from face_engine.server.processing import (
        ProcessorConfig,
        model_paths,
        required_models,
    )

    registry = build_default_registry()
    params = preset_params("ultra")
    config = ProcessorConfig(params, model_paths(required_models(params), registry),
                             {"s": source_embedding(source, video, registry)})
    proc = BoostedFrameProcessor(config)
    try:
        frame = None
        with av.open(str(video)) as c:
            for i, fr in enumerate(c.decode(video=0)):
                f = torch.from_numpy(fr.to_ndarray(format="bgr24")).cuda().permute(2, 0, 1)[None]
                faces = proc.detector.detect_cuda(f)
                if len(faces) >= 2 or (len(faces) and i > 60):
                    frame, kps = f.float(), faces.kps
                    break
        if frame is None:
            raise SystemExit(f"no face in the first frames of {video.name}")
        det = proc.detector
        blob, _ = det.blob_cuda(frame)
        run = det.runner
        n = det._priors_cuda(frame.device).shape[0]
        names = run.output_names
        shapes = {names[0]: (1, n, 4), names[1]: (1, n, 2), names[2]: (1, n, 10)}
        ms = {"RetinaFace R50 engine (640, b1)": _event_ms(
            lambda: run.run_binding({run.input_names[0]: blob}, output_shapes=shapes)),
            "RetinaFace R50 detect_cuda (letterbox+decode+NMS)": _event_ms(
                lambda: det.detect_cuda(frame))}
        faces_n = int(kps.shape[0])
        sw = proc.swapper
        m = similarity_matrices_cuda(kps, sw.spec.size, sw.spec.template)
        crops = warp_face_cuda(frame, m, sw.spec.size, padding_mode="border").clamp(0, 255)
        latents = sw.latent_cuda(proc._sources[:1]).expand(faces_n, -1)
        ms[f"HyperSwap-256 engine (b{faces_n})"] = _event_ms(lambda: sw.run_crops(crops, latents))
        ms[f"DFL XSeg engine + feather (b{faces_n})"] = _event_ms(lambda: proc.masker.xseg(crops))
        enh = proc.enhancer
        em = similarity_matrices_cuda(kps[:1], enh.size, FFHQ)
        ecrop = warp_face_cuda(frame, em, enh.size, padding_mode="border",
                               mode="bicubic").clamp(0, 255)
        ms["GPEN-BFR-1024 engine (b1, per face)"] = _event_ms(lambda: enh.restore(ecrop), runs=15)
        backends = {"retinaface_r50": "TensorRT" if det.uses_tensorrt_engine else "ONNX Runtime",
                    "hyperswap_1a_256": "TensorRT" if sw.aot is not None else "ONNX Runtime",
                    "gpen_bfr_1024": ("TensorRT " + enh.precision) if enh.aot is not None
                    else "ONNX Runtime"}
        return ms, backends
    finally:
        proc.close()


def run_preset(preset: str, video: Path, source: Path | None, frames: int) -> PresetResult:
    import torch

    from face_engine.benchmark import Sampler, SyncCounter
    from face_engine.boosted.ultra_pipeline import render

    src = source or _clip_face(video)
    with tempfile.TemporaryDirectory() as tmp, Sampler() as sampler, SyncCounter() as syncs:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        stats = render(video, Path(tmp) / f"{preset}.mp4", src, preset, max_frames=frames)
        wall = time.perf_counter() - t0
    peak = max(sampler.vram, default=sampler.baseline_mb)
    return PresetResult(preset, stats.fps, stats.frames_done / wall, stats.faces, stats.swapped,
                        peak - sampler.baseline_mb, peak,
                        statistics.mean(sampler.cpu_proc) if sampler.cpu_proc else 0.0,
                        statistics.mean(sampler.cpu_sys) if sampler.cpu_sys else 0.0, syncs.count)


def _clip_face(video: Path) -> Path:
    """A source image: the clip's first frame (the swap then uses its own first face)."""
    import av
    import cv2

    path = Path(tempfile.gettempdir()) / f"{video.stem}_source.png"
    if not path.exists():
        with av.open(str(video)) as c:
            cv2.imwrite(str(path), next(c.decode(video=0)).to_ndarray(format="bgr24"))
    return path


def main(argv: list[str] | None = None) -> int:
    import torch

    from face_engine.core.execution import register_gpu_runtime_dirs

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--source")
    ap.add_argument("--presets", default="boosted,ultra")
    ap.add_argument("--vram-limit-gb", type=float, default=6.0)
    ap.add_argument("--min-fps", type=float, help="fail a preset below this fps")
    ap.add_argument("--json", help="write the structured report here")
    args = ap.parse_args(argv)
    register_gpu_runtime_dirs()
    video, source = Path(args.input), (Path(args.source) if args.source else None)
    if not video.is_file():
        raise SystemExit(f"{video} not found")
    report = Report(torch.cuda.get_device_name(0), str(video), args.frames,
                    vram_limit_mb=args.vram_limit_gb * 1024)
    report.engines_ms, report.engine_backends = engine_latencies(video, source)
    for preset in args.presets.split(","):
        r = run_preset(preset, video, source, args.frames)
        report.presets.append(r)
        if r.peak_vram_mb > report.vram_limit_mb:
            report.failures.append(f"{preset}: peak VRAM {r.peak_vram_mb:.0f} MB > "
                                   f"{report.vram_limit_mb:.0f} MB")
        if r.faces and r.swapped < r.faces:
            report.failures.append(f"{preset}: {r.faces - r.swapped} of {r.faces} faces not swapped")
        if r.render_syncs:
            report.failures.append(f"{preset}: {r.render_syncs} device-wide syncs in the render loop")
        if args.min_fps is not None and r.fps < args.min_fps:
            report.failures.append(f"{preset}: {r.fps:.1f} fps < {args.min_fps:.0f}")
    _print(report)
    if args.json:
        Path(args.json).write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return 1 if report.failures else 0


def _print(report: Report) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    t = Table(title=f"boosted pipeline - {report.gpu} - {Path(report.input).name}, "
                    f"{report.frames} frames")
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    for name, ms in report.engines_ms.items():
        t.add_row(name, f"{ms:.2f} ms")
    t.add_row("Engine backends", ", ".join(f"{k}: {v}" for k, v in report.engine_backends.items()))
    for r in report.presets:
        t.add_section()
        target = TARGET_FPS.get(r.preset)
        goal = f" (target {target[0]:.0f}-{target[1]:.0f})" if target else ""
        t.add_row(f"{r.preset}: throughput", f"{r.fps:.1f} fps{goal}")
        t.add_row(f"{r.preset}: whole job (model load + remux)", f"{r.whole_fps:.1f} fps")
        t.add_row(f"{r.preset}: faces swapped / seen", f"{r.swapped} / {r.faces}")
        t.add_row(f"{r.preset}: peak VRAM added (device total)",
                  f"{r.peak_vram_mb:.0f} MB ({r.device_peak_vram_mb:.0f} MB)")
        t.add_row(f"{r.preset}: host CPU (process / system)",
                  f"{r.cpu_process_pct:.0f}% / {r.cpu_system_pct:.0f}%")
        t.add_row(f"{r.preset}: device-wide syncs in the loop", str(r.render_syncs))
    console.print(t)
    console.print("[bold red]FAIL[/]: " + "; ".join(report.failures) if report.failures
                  else "[bold green]PASS[/]")


if __name__ == "__main__":
    sys.exit(main())
