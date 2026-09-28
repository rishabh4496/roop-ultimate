"""Benchmark the face_engine render: per-stage latency, sustained fps, VRAM, CPU, sync locks.

Usage (from ``face_engine/``, or ``python -m face_engine.benchmark`` from the repo root)::

    python benchmark.py --input tests/sample_1080p.mp4 --frames 500 [--profile balanced]

Two passes over the same clip and the same models:

1. **Stage latencies** (isolated): up to ``--latency-frames`` frames run through
   the render's own components one stage at a time, with a device
   synchronize around each stage, so every row is that stage's own wall time
   per frame. Rows: decode as the render does it (PyAV software decode + GPU
   YUV -> BGR; NVDEC measured slower once inference shares the GPU, Stage 4),
   NVDEC decode for reference, detection, alignment and kornia warps (swap
   crop + paste-back), the swap network, masks, the enhancer network, NVENC.
2. **Sustained throughput**: a real :class:`~face_engine.core.cuda_streams.CUDAStreamPipeline`
   render of ``--frames`` frames (decode / inference / encode overlapped), with
   device VRAM (NVML; Windows reports no per-process figure) and host CPU
   sampled 4x a second. The latency rows do not add up to it: stages overlap.

Exit code 1 when:

* sustained fps < ``--min-fps``. Default: 40 fps for the fast profile on an
  RTX 3080/4080-class GPU (:data:`GATED_GPUS`). The Stage 5 targets are fast
  60+, balanced 30, cinema 12, so a 40 fps floor only fits the fast profile;
  balanced / cinema and other GPUs are reported, not gated, unless
  ``--min-fps`` is given;
* a CPU-GPU sync lock ran inside the render loop: a device-wide
  ``torch.cuda.synchronize()``, which stalls all three streams. Stream-level
  waits are part of the design (ONNX Runtime's fence, NMS, kornia) and are
  not counted;
* faces were seen but none were swapped (a render that swaps nothing reads fast).

The sample clip is not in the repository: make one from any 1080p clip with
faces, e.g. ``ffmpeg -stream_loop 1 -i clip.mp4 -c copy tests/sample_1080p.mp4``.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # run as a script from face_engine/
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

#: GPUs the default fps gate applies to (RTX 3080/4080 class; the 4070
#: family benchmarks with the 3080). Substring match on the device name.
GATED_GPUS = ("3080", "3090", "4070", "4080", "4090", "5070", "5080", "5090")
DEFAULT_MIN_FPS = 40.0
PROFILES = {"fast": "ultra_fast", "balanced": "balanced", "cinema": "cinema"}


@dataclass
class Row:
    name: str
    ms: float | None
    note: str = ""


@dataclass
class Result:
    gpu: str
    profile: str
    frames: int
    rows: list[Row] = field(default_factory=list)
    fps: float = 0.0
    whole_fps: float = 0.0
    peak_vram_mb: float = 0.0
    cpu_process_pct: float = 0.0
    cpu_system_pct: float = 0.0
    device_syncs: int = 0
    setup_syncs: int = 0
    sync_sites: dict[str, int] = field(default_factory=dict)
    faces: int = 0
    swapped: int = 0
    min_fps: float | None = None
    failures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------- helpers
def _timer() -> Any:
    import torch

    class T:
        def __init__(self) -> None:
            self.samples: dict[str, list[float]] = {}

        def __call__(self, name: str) -> Any:
            timer = self

            class Ctx:
                def __enter__(self) -> None:
                    torch.cuda.synchronize()
                    self.t0 = time.perf_counter()

                def __exit__(self, *exc: object) -> None:
                    torch.cuda.synchronize()
                    timer.samples.setdefault(name, []).append(
                        (time.perf_counter() - self.t0) * 1000.0)
            return Ctx()

        def median(self, name: str) -> float | None:
            s = self.samples.get(name)
            return statistics.median(s) if s else None
    return T()


def source_embedding(image_path: Path | None, video: Path, registry: Any) -> np.ndarray:
    """ArcFace embedding of the source image's first face, else of the clip's first face."""
    import cv2

    from face_engine.core.config import EngineConfig, Provider
    from face_engine.core.execution import ExecutionEngine
    from face_engine.pipeline.detector import SCRFDDetector
    from face_engine.processors.swapper import IdentityEncoder

    if image_path is not None:
        images = [cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_COLOR)]
    else:
        import av

        with av.open(str(video)) as c:
            images = [f.to_ndarray(format="bgr24") for _, f in zip(range(60), c.decode(video=0))]
    with ExecutionEngine(EngineConfig(providers=[Provider.CUDA, Provider.CPU])) as engine:
        detector = SCRFDDetector(engine, registry.ensure("scrfd_10g_bnkps", show_progress=False))
        encoder = IdentityEncoder(engine, registry.ensure("arcface_w600k_r50", show_progress=False))
        for image in images[::5]:
            faces = detector.detect(image)
            if faces:
                return encoder.embed(image, faces[0]).embedding
    where = image_path.name if image_path else f"the first 60 frames of {video.name}"
    raise SystemExit(f"no face found in {where}")


class Sampler:
    """Device VRAM (NVML) and host CPU (psutil) sampled on a thread."""

    def __init__(self, interval: float = 0.25) -> None:
        import psutil
        import pynvml

        pynvml.nvmlInit()
        self._nvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._proc = psutil.Process()
        self._psutil = psutil
        self.interval = interval
        self.baseline_mb = self._used_mb()
        self.vram: list[float] = []
        self.cpu_proc: list[float] = []
        self.cpu_sys: list[float] = []
        self._stop = threading.Event()

    def _used_mb(self) -> float:
        return self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / 2 ** 20

    def __enter__(self) -> Sampler:  # noqa: PYI034
        self._proc.cpu_percent(None)
        self._psutil.cpu_percent(None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        cores = self._psutil.cpu_count() or 1
        while not self._stop.wait(self.interval):
            self.vram.append(self._used_mb())
            self.cpu_proc.append(self._proc.cpu_percent(None) / cores)
            self.cpu_sys.append(self._psutil.cpu_percent(None))

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


class SyncCounter:
    """Device-wide ``torch.cuda.synchronize()`` calls while active, by call site.

    ``count`` is the calls made on the stream pipeline's stage threads (names
    ``stream-*``): the render loop. ``setup`` counts the rest (model loading,
    CUDA graph capture in ``prepare``), which run before the loop starts.
    """

    def __enter__(self) -> SyncCounter:  # noqa: PYI034
        import torch

        self.count = 0
        self.setup = 0
        self.sites: dict[str, int] = {}
        self._orig = torch.cuda.synchronize

        def counted(*args: Any, **kwargs: Any) -> Any:
            import traceback

            if threading.current_thread().name.startswith("stream-"):
                self.count += 1
            else:
                self.setup += 1
            frames = [f for f in traceback.extract_stack()[:-1]
                      if "benchmark.py" not in f.filename]
            site = " <- ".join(f"{Path(f.filename).name}:{f.lineno} {f.name}"
                               for f in reversed(frames[-4:]))
            self.sites[site] = self.sites.get(site, 0) + 1
            return self._orig(*args, **kwargs)

        torch.cuda.synchronize = counted
        return self

    def __exit__(self, *exc: object) -> None:
        import torch

        torch.cuda.synchronize = self._orig


# ---------------------------------------------------------------------------- pass 1
def stage_latencies(video: Path, config: Any, frames: int) -> list[Row]:
    import av
    import torch

    from face_engine.core.cuda_streams import CUDARingBuffer, i420_to_bgr_hwc
    from face_engine.media.capturer import VideoSource
    from face_engine.media.decoder import HardwareVideoDecoder
    from face_engine.media.encoder import open_video_writer
    from face_engine.pipeline.aligner import (
        similarity_matrices_cuda,
        warp_face_cuda,
        warp_face_inverse_cuda,
    )
    from face_engine.processors.enhancer import TEMPLATE as ENHANCER_TEMPLATE
    from face_engine.processors.swapper import implode_pixel_boost_cuda
    from face_engine.server.processing import GpuFrameProcessor

    t = _timer()
    info = VideoSource(video).info
    ring = CUDARingBuffer(2, (info.height, info.width, 3))
    full = info.color_tags.get("color_range") in ("pc", "jpeg")
    decoded: list[torch.Tensor] = []
    with av.open(str(video)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        it = c.decode(stream)
        for _ in range(frames):
            with t("decode"):
                frame = next(it)
                for plane, dst in zip(frame.planes, ring.host_planes(0)):
                    rows, cols = dst.shape
                    v = np.frombuffer(plane, np.uint8, count=rows * plane.line_size)
                    np.copyto(dst, v.reshape(rows, plane.line_size)[:, :cols])
                ring.planes[0].copy_(ring.ingress[0], non_blocking=True)
                i420_to_bgr_hwc(ring.planes[0], info.height, info.width, info.color_profile, full,
                                ring.slots[0])
            decoded.append(ring.slots[0].permute(2, 0, 1)[None].clone())

    rows = [Row("Decode (render path: PyAV software + GPU YUV->BGR)", t.median("decode"))]
    try:
        dec = HardwareVideoDecoder(video, batch_size=4, backend="nvdec", end=frames)
        t0 = time.perf_counter()
        n = sum(len(b) for b in dec)
        rows.append(Row("NVDEC decode (reference; not used by the render)",
                        (time.perf_counter() - t0) * 1000.0 / max(n, 1), "child process, NV12"))
    except Exception as exc:  # noqa: BLE001 - reported in the table
        rows.append(Row("NVDEC decode (reference; not used by the render)", None,
                        f"unavailable: {str(exc)[:60]}"))

    processor = GpuFrameProcessor(config)  # detects every frame: per-frame detector cost
    p = config.params
    boost = p.boost_size or processor.swapper.spec.size
    size = processor.swapper.spec.size
    faces_seen = 0
    try:
        for i, frame in enumerate(decoded):
            f = frame.float()
            warm = i < 3  # sessions and engines load on the first frames
            name = (lambda s: f"warm:{s}") if warm else (lambda s: s)
            with t(name("detect")):
                faces = processor.detector.detect_cuda(f)
            if len(faces) == 0:
                continue
            keep, sources = processor._choose_sources(f, faces.kps)
            if keep.shape[0] == 0:
                continue
            kps = faces.kps[keep]
            faces_seen += int(kps.shape[0])
            with t(name("align")):
                m = similarity_matrices_cuda(kps, boost, processor.swapper.spec.template)
                crops = warp_face_cuda(f, m, boost, padding_mode="border", antialias=True,
                                       mode="bicubic").clamp(0, 255)
            latents = processor.swapper.latent_cuda(sources)
            factor = boost // size
            tiles = implode_pixel_boost_cuda(crops, size, factor)
            tile_latents = latents.repeat_interleave(factor * factor, dim=0)
            with t(name("swap")):
                processor.swapper.run_crops(tiles, tile_latents)
            with t(name("mask")):
                mask = processor.masker.generate(f, kps).mask
            with t(name("paste")):
                out = warp_face_inverse_cuda(f, crops, m, mask)
            if processor.enhancer is not None:
                enh = processor.enhancer
                with t(name("enh_align")):
                    em = similarity_matrices_cuda(kps, enh.size, ENHANCER_TEMPLATE)
                    ecrops = warp_face_cuda(out, em, enh.size, padding_mode="border",
                                            antialias=True, mode="bicubic").clamp(0, 255)
                with t(name("enhance")):
                    enh.restore(ecrops)
    finally:
        processor.close()

    def per_frame(*names: str) -> float | None:
        vals = [t.median(n) for n in names]
        return sum(v for v in vals if v is not None) if any(v is not None for v in vals) else None

    backend = "TensorRT AOT engine" if processor.swapper.aot is not None else "ONNX Runtime"
    rows += [
        Row("Detection (SCRFD, every frame)", per_frame("detect"),
            f"stride {p.detection_stride} in the render"),
        Row("Alignment & kornia warps (crop + paste-back)",
            per_frame("align", "paste", "enh_align"), f"{faces_seen} faces"),
        Row(f"Swap network ({p.swapper_model}, {backend})", per_frame("swap"),
            f"pixel boost {p.pixel_boost}"),
        Row("Masks (" + "+".join(p.mask_types) + ")", per_frame("mask")),
    ]
    if processor.enhancer is not None:
        eb = "TensorRT AOT engine" if processor.enhancer.aot is not None else "ONNX Runtime"
        rows.append(Row(f"Enhancer network ({p.enhancer_model}, {eb})", per_frame("enhance")))
    else:
        rows.append(Row("Enhancer network", None, "none in this profile"))

    with tempfile.TemporaryDirectory() as tmp:
        writer = open_video_writer(Path(tmp) / "nvenc.mp4", info.width, info.height, info.fps,
                                   expected_frames=len(decoded), color=info.color_profile)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for frame in decoded:
            writer.write_tensor(frame)
        writer.close()
        encoder = type(writer).__name__.replace("VideoWriter", "").replace("TensorWriter", "")
        rows.append(Row(f"{encoder} output (GPU -> pinned -> ffmpeg, incl. close)",
                        (time.perf_counter() - t0) * 1000.0 / len(decoded)))
    return rows


# ---------------------------------------------------------------------------- pass 2
def sustained(video: Path, config: Any, frames: int, workdir: Path) -> dict[str, Any]:
    from face_engine.core.cuda_streams import CUDAStreamPipeline

    with Sampler() as sampler, SyncCounter() as syncs:
        t0 = time.perf_counter()
        stats = CUDAStreamPipeline().run(video, workdir / "sustained.mp4", config,
                                         max_frames=frames)
        wall = time.perf_counter() - t0
    return {"stats": stats, "wall": wall, "syncs": syncs.count, "setup_syncs": syncs.setup,
            "sync_sites": syncs.sites,
            "peak_vram_mb": max(sampler.vram, default=sampler.baseline_mb) - sampler.baseline_mb,
            "cpu_proc": statistics.mean(sampler.cpu_proc) if sampler.cpu_proc else 0.0,
            "cpu_sys": statistics.mean(sampler.cpu_sys) if sampler.cpu_sys else 0.0}


# ---------------------------------------------------------------------------- main
def build_config(video: Path, source: Path | None, profile: str) -> Any:
    from face_engine.models.zoo import build_default_registry
    from face_engine.server.processing import (
        PRESETS,
        ProcessorConfig,
        RenderParams,
        model_paths,
        required_models,
    )

    params = RenderParams(**PRESETS[PROFILES[profile]]["params"])
    registry = build_default_registry()
    paths = model_paths(required_models(params), registry)
    return ProcessorConfig(params, paths, {"source": source_embedding(source, video, registry)})


def gate_fps(gpu: str, profile: str, min_fps: float | None) -> float | None:
    if min_fps is not None:
        return min_fps
    if profile == "fast" and any(g in gpu for g in GATED_GPUS):
        return DEFAULT_MIN_FPS
    return None


def run(args: argparse.Namespace) -> Result:
    import torch

    from face_engine.core.execution import register_gpu_runtime_dirs

    register_gpu_runtime_dirs()
    video = Path(args.input)
    if not video.is_file():
        raise SystemExit(f"{video} not found. Make a sample from any clip with faces, e.g.\n"
                         f"  ffmpeg -stream_loop 1 -i clip.mp4 -c copy {video}")
    gpu = torch.cuda.get_device_name(0)
    config = build_config(video, Path(args.source) if args.source else None, args.profile)
    result = Result(gpu, args.profile, args.frames, min_fps=gate_fps(gpu, args.profile,
                                                                        args.min_fps))
    result.rows = stage_latencies(video, config, min(args.latency_frames, args.frames))
    with tempfile.TemporaryDirectory() as tmp:
        s = sustained(video, config, args.frames, Path(tmp))
    stats = s["stats"]
    result.fps = stats.fps
    result.whole_fps = stats.frames_done / s["wall"]
    result.peak_vram_mb = s["peak_vram_mb"]
    result.cpu_process_pct, result.cpu_system_pct = s["cpu_proc"], s["cpu_sys"]
    result.device_syncs = s["syncs"]
    result.setup_syncs, result.sync_sites = s["setup_syncs"], s["sync_sites"]
    result.faces, result.swapped = stats.faces, stats.swapped
    result.frames = stats.frames_done
    if result.min_fps is not None and result.fps < result.min_fps:
        result.failures.append(f"sustained {result.fps:.1f} fps < {result.min_fps:.0f} fps")
    if result.device_syncs:
        result.failures.append(f"{result.device_syncs} device-wide torch.cuda.synchronize() "
                               "call(s) inside the render loop")
    if result.faces and not result.swapped:
        result.failures.append(f"{result.faces} faces seen, none swapped")
    return result


def render_table(result: Result) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"face_engine benchmark - {result.gpu} - profile {result.profile} - "
                        f"{result.frames} frames")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Note", style="dim")
    for row in result.rows:
        table.add_row(row.name, "n/a" if row.ms is None else f"{row.ms:.2f} ms", row.note)
    table.add_section()
    gate = ("not gated" if result.min_fps is None else f"gate {result.min_fps:.0f} fps")
    table.add_row("Sustained pipeline throughput", f"{result.fps:.1f} fps",
                  f"{gate}; whole job incl. model load + remux {result.whole_fps:.1f} fps")
    table.add_row("Faces swapped / seen", f"{result.swapped} / {result.faces}")
    table.add_row("Peak VRAM (device, above the pre-run baseline)",
                  f"{result.peak_vram_mb:.0f} MB")
    table.add_row("Host CPU load (this process / whole system)",
                  f"{result.cpu_process_pct:.0f}% / {result.cpu_system_pct:.0f}%",
                  "process % is of all logical cores")
    table.add_row("Device-wide syncs in the render loop", str(result.device_syncs),
                  f"+{result.setup_syncs} before the loop (model load, CUDA graph capture)")
    console.print(table)
    for site, n in result.sync_sites.items():
        console.print(f"  [dim]torch.cuda.synchronize x{n}: {site}[/]")
    if result.failures:
        console.print("[bold red]FAIL[/]: " + "; ".join(result.failures))
    else:
        console.print("[bold green]PASS[/]")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="benchmark.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True, help="video with faces (1080p for the gate)")
    ap.add_argument("--frames", type=int, default=500, help="frames for the sustained render")
    ap.add_argument("--latency-frames", type=int, default=100,
                    help="frames for the isolated per-stage pass")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="fast")
    ap.add_argument("--source", help="source face image (default: the clip's first face)")
    ap.add_argument("--min-fps", type=float, help="fail below this sustained fps (any GPU)")
    ap.add_argument("--json", help="also write the result as JSON here")
    args = ap.parse_args(argv)
    result = run(args)
    render_table(result)
    if args.json:
        Path(args.json).write_text(json.dumps(result, default=lambda o: o.__dict__, indent=2),
                                   encoding="utf-8")
    return 1 if result.failures else 0


if __name__ == "__main__":
    sys.exit(main())
