"""Production audit of the face_engine GPU render: hardware, memory, detection, restore, throughput.

Usage (repo root, the app's Python)::

    app/env/Scripts/python.exe tools/audit_system.py --profile balanced [--json out.json]

Every check drives the REAL components, never mocks of them:

* ``audit_hardware``            torch / CUDA / cuDNN / driver, compute capability,
  the caching-allocator cap (``set_per_process_memory_fraction(0.85)``, proven by
  a refused over-cap allocation), TensorRT 10.x, and every AOT engine the
  profile and this audit need deserialized from ``.cache/trt_engines``.
* ``audit_async_pipeline``      full-clip renders through
  :class:`~face_engine.core.cuda_streams.CUDAStreamPipeline` (the server's
  render): steady-state fps against the profile target, faces swapped / seen,
  device-wide syncs in the loop, per-process VRAM from the Windows
  ``GPU Process Memory`` counters (Dedicated = VRAM, Shared = system RAM mapped
  for the GPU) and NVML, A/V parity. Then the per-stage latencies of
  :func:`face_engine.benchmark.stage_latencies` (each stage bracketed by a
  device synchronize, 100 frames).
* ``audit_memory_invariants``   100- and 500-frame renders on one processor,
  ``memory_allocated`` read with nothing in flight after each (a mid-render
  reading moves by whole in-flight plans, 1-2 faces each); the 500-frame one
  with a tracing processor (frame pointers, devices) under CUPTI
  (``torch.profiler``, process-wide: it sees ONNX Runtime's and TensorRT's
  copies too), counting every H2D / D2H memcpy and its size. Plus the
  per-render retention between the timed renders.
* ``audit_detector_angles``     :class:`AngleResilientSCRFD` on insightface's
  t1.jpg (6 faces) at 0/90/180/270 deg, the upright fast path under a live
  tracker, :class:`RobustByteTracker` through a 0.82 -> 0.23 profile turn, and
  the profile-guarded Umeyama fit up to a fully collapsed profile.
* ``audit_ultra_restore``       :class:`UltraRestorer` on a real face scaled to
  85 / 220 / 450 px box diagonal (route, network calls, latency, luminance),
  the frequency split's decomposition, the inner-mouth weight.
* ``audit_pose_routing``        the APP render's pose-adaptive source routing
  (``app/roop/source_portfolio.route``): per-face overhead through each pose
  path - frame-LUT hit, neighbouring scanned frame, live keypoint solve - against
  a 0.8 ms budget, and that every routed vector keeps the swap input's shape and
  dtype (a changed shape is what would make TensorRT re-bind or re-allocate).

Statuses: PASS, FAIL, UNVERIFIED (a required criterion this rig cannot
measure: counts as not passing), INFO (reported, not gated). Latency and fps
targets are the RTX 4070 12GB's; on another GPU they are reported as INFO.
Exit code 0 only when every gated row passes.

Runs one render at a time and loads models: stop the app first.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------- thresholds
REFERENCE_GPU = "4070"
VRAM_CEILING_GB = 6.0
MEMORY_FRACTION = 0.85
FPS_TARGET = {"fast": 60.0, "balanced": 45.0, "cinema": 12.0}
SWEEP_BUDGET_MS = 4.0
LOW_SCORE_FLOOR = 0.20
UMEYAMA_RATIO = 0.15
BYPASS_DIAG = 85.0
GPEN512_DIAG, GPEN512_BUDGET_MS = 220.0, 3.5
GPEN1024_DIAG, GPEN1024_BUDGET_MS = 450.0, 8.5
LUMA_TOLERANCE = 0.02
INNER_MOUTH_MAX = 0.20
ROUTING_BUDGET_MS = 0.8
MEMORY_FRAMES = 500
DEFAULT_CLIP = ROOT / "face_engine" / "tests" / "sample_1080p.mp4"
PASS, FAIL, UNVERIFIED, INFO = "PASS", "FAIL", "UNVERIFIED", "INFO"


@dataclass
class Check:
    section: str
    component: str
    status: str
    target: str
    measured: str
    latency_ms: float | None = None
    fix: str = ""


@dataclass
class Report:
    gpu: str = ""
    profile: str = ""
    checks: list[Check] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    def add(self, section: str, component: str, ok: bool | None, target: str, measured: str,
            latency_ms: float | None = None, fix: str = "", gated: bool = True) -> Check:
        if ok is None:
            status = UNVERIFIED if gated else INFO
        else:
            status = (PASS if ok else FAIL) if gated else INFO
        c = Check(section, component, status, target, measured, latency_ms, fix)
        self.checks.append(c)
        return c

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status in (FAIL, UNVERIFIED)]


# ---------------------------------------------------------------------------- helpers
def _free_gpu() -> None:
    import torch

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _quiescent_allocated(clear_cublas: bool = False) -> int:
    """``memory_allocated()`` with nothing in flight (after a render returned).

    ``clear_cublas`` first drops PyTorch's per-(handle, stream) cuBLAS
    workspaces: each render creates new CUDA streams and leaves one 8.125 MiB
    workspace per stream that ran a cuBLAS / cuSOLVER call (see the
    per-render retention check), which would otherwise mask a per-frame leak.
    """
    import torch

    gc.collect()
    torch.cuda.synchronize()
    if clear_cublas:
        torch._C._cuda_clearCublasWorkspaces()
    return torch.cuda.memory_allocated()


def _cuda_ms(fn: Callable[[], Any], reps: int = 30, warm: int = 5,
             before: Callable[[], Any] | None = None) -> float:
    """Median ``torch.cuda.Event`` time of ``fn`` (``before`` runs untimed each rep)."""
    import torch

    for _ in range(warm):
        if before:
            before()
        fn()
    times = []
    for _ in range(reps):
        if before:
            before()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def _on_reference(gpu: str) -> bool:
    return REFERENCE_GPU in gpu


class WinGpuMemory:
    """Per-process GPU memory from Windows' ``GPU Process Memory`` counters.

    ``Dedicated Usage`` is this process's VRAM; ``Shared Usage`` is system RAM
    the GPU maps for it (pinned staging buffers, and sysmem fallback when VRAM
    runs out). NVML has no per-process figure under WDDM. Sampled by a
    PowerShell ``Get-Counter`` every ``interval`` seconds (~1 s per sample).
    """

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.pid = os.getpid()
        self.dedicated: list[float] = []
        self.shared: list[float] = []
        self.available = sys.platform == "win32"
        self._stop = threading.Event()

    def sample(self) -> tuple[float, float] | None:
        if not self.available:
            return None
        q = (f"(Get-Counter '\\GPU Process Memory(pid_{self.pid}_*)\\Dedicated Usage',"
             f"'\\GPU Process Memory(pid_{self.pid}_*)\\Shared Usage' -ErrorAction "
             "SilentlyContinue).CounterSamples | % { $_.Path + '=' + $_.CookedValue }")
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", q], capture_output=True,
                                 text=True, timeout=30, check=False,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        except Exception:  # noqa: BLE001 - counters unavailable: reported as such
            self.available = False
            return None
        ded = sum(float(line.rsplit("=", 1)[1]) for line in out.splitlines()
                  if "dedicated usage=" in line.lower())
        sh = sum(float(line.rsplit("=", 1)[1]) for line in out.splitlines()
                 if "shared usage=" in line.lower())
        if not out.strip():
            return None
        return ded / 2 ** 20, sh / 2 ** 20

    def __enter__(self) -> WinGpuMemory:  # noqa: PYI034
        self._thread = threading.Thread(target=self._run, daemon=True, name="audit-wincounters")
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            s = self.sample()
            if s is not None:
                self.dedicated.append(s[0])
                self.shared.append(s[1])
            self._stop.wait(self.interval)

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=35)


# ============================================================================ 1. hardware
def audit_hardware(r: Report, profile: str) -> None:
    import pynvml
    import torch

    from face_engine.core.execution import register_gpu_runtime_dirs

    sec = "Hardware & drivers"
    register_gpu_runtime_dirs()
    if not torch.cuda.is_available():
        r.add(sec, "CUDA device", False, "a CUDA GPU", "torch.cuda.is_available() is False",
              fix="Install the CUDA build of torch into app/env (torch.js).")
        return
    pynvml.nvmlInit()
    drv = pynvml.nvmlSystemGetDriverVersion()
    drv = drv.decode() if isinstance(drv, bytes) else drv
    cuda_drv = pynvml.nvmlSystemGetCudaDriverVersion()
    gpu = torch.cuda.get_device_name(0)
    r.gpu = gpu
    total = torch.cuda.get_device_properties(0).total_memory
    cudnn = torch.backends.cudnn.version()
    r.facts.update(gpu=gpu, driver=drv, cuda_driver=f"{cuda_drv // 1000}.{cuda_drv % 1000 // 10}",
                   cuda_runtime=torch.version.cuda, torch=torch.__version__, cudnn=cudnn,
                   vram_total_gb=round(total / 2 ** 30, 2))
    r.add(sec, "PyTorch CUDA runtime / driver / cuDNN",
          bool(cudnn) and torch.backends.cudnn.enabled, "CUDA + cuDNN usable",
          f"torch {torch.__version__}, CUDA rt {torch.version.cuda}, driver {drv} "
          f"(CUDA {r.facts['cuda_driver']}), cuDNN {cudnn}")

    cc = torch.cuda.get_device_capability(0)
    ada = cc == (8, 9)
    r.add(sec, "GPU architecture", ada if _on_reference(gpu) else None,
          "SM 8.9 (Ada Lovelace)", f"{gpu}, SM {cc[0]}.{cc[1]}, "
          f"{total / 2 ** 30:.1f} GB", gated=_on_reference(gpu),
          fix="The audit's targets are for an RTX 4070 (SM 8.9).")

    # The cap applies to PyTorch's caching allocator only; ONNX Runtime and
    # TensorRT allocate through their own allocators (measured separately below).
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, 0)
    refused = False
    try:
        x = torch.empty(int(total * (MEMORY_FRACTION + 0.05)), dtype=torch.uint8, device="cuda")
        del x
    except torch.cuda.OutOfMemoryError:
        refused = True
    ok_alloc = False
    try:
        y = torch.empty(int(total * 0.05), dtype=torch.uint8, device="cuda")
        del y
        ok_alloc = True
    except torch.cuda.OutOfMemoryError:
        pass
    _free_gpu()
    r.add(sec, "Allocator cap set_per_process_memory_fraction", refused and ok_alloc,
          f"{MEMORY_FRACTION} enforced", f"{(MEMORY_FRACTION + 0.05):.0%} alloc "
          f"{'refused' if refused else 'ALLOWED'}; 5% alloc {'ok' if ok_alloc else 'refused'} "
          "(caching allocator only; ORT/TRT arenas are outside it)",
          fix="torch.cuda.set_per_process_memory_fraction did not take effect.")

    try:
        import tensorrt as trt
    except Exception as exc:  # noqa: BLE001
        r.add(sec, "TensorRT Python bindings", False, "tensorrt 10.x", f"import failed: {exc}",
              fix="Reinstall tensorrt into app/env (ORT 1.23.2 / TRT 10.9 pairing).")
        return
    major = int(trt.__version__.split(".")[0])
    r.facts["tensorrt"] = trt.__version__
    r.add(sec, "TensorRT Python bindings", major == 10, "tensorrt 10.x", trt.__version__,
          fix="Install TensorRT 10.x matching onnxruntime-gpu.")

    from face_engine.benchmark import PROFILES
    from face_engine.core.trt_compiler import ENGINE_DIR, find_engine
    from face_engine.models.zoo import build_default_registry
    from face_engine.server.processing import PRESETS, RenderParams, required_models

    params = RenderParams(**PRESETS[PROFILES[profile]]["params"])
    wanted = list(dict.fromkeys(required_models(params)
                                + ["scrfd_10g_bnkps", "gpen_bfr_512", "gpen_bfr_1024"]))
    reg = build_default_registry()
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    loaded, missing, t0 = [], [], time.perf_counter()
    for name in wanted:
        try:
            src = reg.ensure(name, show_progress=False)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{name} (model: {str(exc)[:40]})")
            continue
        path = find_engine(name, src, "fp16") or find_engine(name, src, "fp32")
        if path is None:
            missing.append(name)
            continue
        eng = runtime.deserialize_cuda_engine(path.read_bytes())
        if eng is None:
            missing.append(f"{path.name} (deserialize failed)")
        else:
            loaded.append(path.stem)
        del eng
    _free_gpu()
    no_aot = {"arcface_w600k_r50"}  # the identity encoder runs once per source, ONNX Runtime
    hard_missing = [m for m in missing if m.split(" ")[0] not in no_aot]
    r.add(sec, "TensorRT AOT engine cache", not hard_missing,
          f"{ENGINE_DIR.relative_to(ROOT).as_posix()}/ verified + deserializable",
          f"{len(loaded)} loaded in {time.perf_counter() - t0:.1f}s"
          + (f"; no engine: {', '.join(missing)}" if missing else ""),
          fix="Build the missing engines: python tools/compile_engines.py --models <name>.")


# ============================================================================ 2. pipeline
def _build_config(video: Path, profile: str) -> Any:
    from face_engine.benchmark import build_config

    return build_config(video, None, profile)


def audit_async_pipeline(r: Report, video: Path, profile: str, config: Any, repeats: int,
                         workdir: Path) -> None:
    import pynvml

    from face_engine.benchmark import Sampler, SyncCounter, stage_latencies
    from face_engine.core.cuda_streams import CUDAStreamPipeline

    sec = "Async CUDA streams"
    ref = _on_reference(r.gpu)
    runs = []
    for i in range(repeats):
        with WinGpuMemory() as win, Sampler() as nv, SyncCounter() as syncs:
            stats = CUDAStreamPipeline().run(video, workdir / f"timed_{i}.mp4", config)
        after = _quiescent_allocated()
        runs.append({"allocated_after": after, "fps": stats.fps, "frames": stats.frames_done, "faces": stats.faces,
                     "swapped": stats.swapped, "syncs": syncs.count, "sync_sites": syncs.sites,
                     "win_dedicated_mb": max(win.dedicated, default=float("nan")),
                     "win_shared_mb": max(win.shared, default=float("nan")),
                     "win_samples": len(win.dedicated),
                     "nvml_peak_mb": max(nv.vram, default=nv.baseline_mb),
                     "nvml_baseline_mb": nv.baseline_mb,
                     "ring_bytes": stats.extra.get("ring_bytes", {}),
                     "av": stats.extra.get("av_sync", {}), "encoder": stats.extra.get("encoder"),
                     "elapsed": stats.elapsed})
        print(f"  timed render {i + 1}/{repeats}: {stats.fps:.1f} fps steady, "
              f"{stats.swapped}/{stats.faces} faces swapped, {stats.frames_done} frames")
        _free_gpu()
    r.facts["timed_runs"] = runs
    if len(runs) >= 2:
        kept = runs[-1]["allocated_after"] - runs[-2]["allocated_after"]
        r.add("Zero-copy & leaks", "No VRAM retained per render (fresh processor)", kept == 0,
              "0 bytes left behind by a finished render",
              f"{kept / 2 ** 20:+.3f} MB per render (memory_allocated after render {len(runs) - 1} "
              f"vs {len(runs)}, nothing in flight)",
              fix="PyTorch caches a cuBLAS workspace (8.125 MiB on SM 8.9) per (handle, stream); "
                  "CUDAStreamPipeline._render creates new streams every render, so the inference "
                  "stream (aligner SVD solve) and the encode stream (kornia inverse) each strand "
                  "one. It plateaus at ~393 MB once the 32-stream pools wrap (measured 2026-09-28, "
                  "40 renders). Fix: torch._C._cuda_clearCublasWorkspaces() when a render ends, "
                  "or create the three streams once per pipeline.")
    else:
        r.add("Zero-copy & leaks", "No VRAM retained per render (fresh processor)", None,
              "0 bytes left behind by a finished render", "needs --repeats >= 2")

    fps = statistics.mean(x["fps"] for x in runs)
    target = FPS_TARGET[profile]
    spread = (max(x["fps"] for x in runs) - min(x["fps"] for x in runs)) / fps if fps else 0
    r.add(sec, f"End-to-end throughput ({profile})", fps >= target if ref else None,
          f">= {target:.0f} fps", f"{fps:.1f} fps mean of {repeats} "
          f"({', '.join(format(x['fps'], '.1f') for x in runs)}; spread {spread:.1%}), "
          f"{runs[0]['frames']} frames 1080p", 1000.0 / fps if fps else None, gated=ref,
          fix=(f"Throughput below the {profile} target. The pipeline is GPU-bound: only removing "
               "per-face GPU work moves it (AGENTS.md). Re-run with nothing else on the GPU; "
               "the rig drifts ~30% with browsers open."))
    faces, swapped = runs[0]["faces"], runs[0]["swapped"]
    r.add(sec, "Faces swapped / seen (render did real work)", faces > 0 and swapped == faces,
          "all seen faces swapped, > 0", f"{swapped} / {faces}",
          fix="A render that swaps nothing reads fast: check the source embedding and matching.")
    syncs = max(x["syncs"] for x in runs)
    r.add(sec, "Device-wide syncs inside the render loop", syncs == 0, "0",
          str(syncs) + ("" if not syncs else " at " + "; ".join(runs[0]["sync_sites"])[:120]),
          fix="A torch.cuda.synchronize() on a stage thread stalls all three streams.")

    # VRAM: Windows' per-process Dedicated Usage is the process's own VRAM.
    ded = max(x["win_dedicated_mb"] for x in runs)
    shared = max(x["win_shared_mb"] for x in runs)
    pynvml.nvmlInit()
    total_mb = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).total / 2 ** 20
    nv_peak = max(x["nvml_peak_mb"] for x in runs)
    if math.isnan(ded):
        ded_gb = (nv_peak - runs[0]["nvml_baseline_mb"]) / 1024
        how = "NVML device delta (no per-process counter)"
    else:
        ded_gb = ded / 1024
        how = "Windows per-process Dedicated Usage"
    r.add(sec, "Peak dedicated VRAM (this process)", ded_gb <= VRAM_CEILING_GB,
          f"<= {VRAM_CEILING_GB:.1f} GB", f"{ded_gb:.2f} GB ({how}); device peak "
          f"{nv_peak / 1024:.2f} / {total_mb / 1024:.1f} GB incl. other apps",
          fix="Lower pools / batch sizes or drop the enhancer; TensorRT workspaces dominate.")
    pinned_mb = runs[0]["ring_bytes"].get("pinned", 0) / 2 ** 20
    # Sysmem fallback only engages when VRAM is exhausted; the pinned staging
    # buffers (by design) and the CUDA context's host mappings show as Shared.
    headroom = nv_peak < 0.95 * total_mb
    if math.isnan(shared):
        r.add(sec, "No sysmem fallback (shared GPU memory)", None,
              "0 fallback allocations", "Windows GPU counters unavailable",
              fix="Run on Windows with the GPU Process Memory counters enabled.")
    else:
        r.add(sec, "No sysmem fallback (shared GPU memory)", headroom and shared < 1024,
              "VRAM never exhausted; shared < 1 GB", f"shared peak {shared:.0f} MB "
              f"(pinned ring {pinned_mb:.0f} MB of it); device VRAM peak "
              f"{nv_peak / total_mb:.0%} of physical",
              fix="Shared usage grew past the pinned buffers or VRAM hit its limit: the driver "
                  "is spilling to system RAM (NVIDIA Control Panel > CUDA Sysmem Fallback "
                  "Policy > Prefer No Sysmem Fallback).")

    av = runs[0]["av"]
    if av and av.get("audio_s") is not None:
        from fractions import Fraction

        rate = float(Fraction(av["fps"]))
        want = av["audio_s"] * rate
        frames = runs[0]["frames"]
        r.add(sec, "A/V duration parity", abs(frames - want) <= 1.0,
              "video frames == audio_s x fps (+-1 frame: AAC packets are 1024 samples)",
              f"{frames} frames vs {av['audio_s']:.4f} s x {rate:g} = {want:.2f}; video "
              f"{av['video_s']:.4f} s", fix="check_av_sync failed: see guardrails.check_av_sync.")
    else:
        r.add(sec, "A/V duration parity", None, "video frames == audio_s x fps",
              "source clip has no audio stream" if av else "no A/V report from the render",
              fix="Use a clip with an audio track (--input).")

    # Isolated per-stage latencies (the benchmark's pass 1, synchronize-bracketed).
    rows = stage_latencies(video, config, 100)
    _free_gpu()
    r.facts["stage_rows"] = [asdict(x) for x in rows]

    def row(prefix: str) -> Any:
        return next((x for x in rows if x.name.startswith(prefix)), None)

    def ms(*prefixes: str) -> float | None:
        vals = [row(p).ms for p in prefixes if row(p) is not None and row(p).ms is not None]
        return sum(vals) if vals else None

    for label, prefixes, note in (
            ("Decode (render: PyAV sw + GPU YUV->BGR)", ("Decode",), "per frame"),
            ("NVDEC decode (reference only)", ("NVDEC",),
             "not used by the render: slower once inference shares the GPU"),
            ("Kornia affine warps (crop + paste-back)", ("Alignment",), "per frame"),
            ("TensorRT inference (HyperSwap + XSeg)", ("Swap network", "Masks"), "per frame"),
            ("Detection (SCRFD)", ("Detection",), "per frame"),
            ("Enhancer / restore network", ("Enhancer",), "per frame"),
            ("Encode (GPU -> pinned -> ffmpeg)", ("ENCODER",), "per frame")):
        if prefixes == ("ENCODER",):  # named after the writer class: "NVENC output (...)"
            srcs = [x for x in rows if " output (" in x.name]
            value = srcs[0].ms if srcs else None
        else:
            value = ms(*prefixes)
            srcs = [row(p) for p in prefixes if row(p) is not None]
        detail = "; ".join(x.name + (f" - {x.note}" if x.note else "") for x in srcs) or note
        r.add("Stage latency", label, None, "informational (stages overlap in the render)",
              "n/a" if value is None else detail, value, gated=False)


# ============================================================================ 3. memory
class TracingProcessor:
    """Wraps :class:`GpuFrameProcessor`: records frame pointers and devices."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.frames = 0
        self.ptrs: set[int] = set()
        self.host_frames = 0
        self.host_tensors = 0
        self.host_outputs = 0

    def infer(self, frame: Any) -> Any:
        self.frames += 1
        self.ptrs.add(frame.data_ptr())
        if not frame.is_cuda:
            self.host_frames += 1
        plan = self.inner.infer(frame)
        self.host_tensors += sum(1 for t in plan.tensors() if not t.is_cuda)
        return plan

    def composite(self, plan: Any) -> Any:
        out = self.inner.composite(plan)
        if out is not None and not out.is_cuda:
            self.host_outputs += 1
        return out


def _memcpy_events(trace: Path) -> list[dict[str, Any]]:
    data = json.loads(trace.read_text(encoding="utf-8"))
    events = data.get("traceEvents", data) if isinstance(data, dict) else data
    return [e for e in events if e.get("cat") == "gpu_memcpy"]


def audit_memory_invariants(r: Report, video: Path, config: Any, frames: int,
                            workdir: Path) -> None:
    import torch
    from torch.profiler import ProfilerActivity, profile

    from face_engine.core.cuda_streams import CUDAStreamPipeline
    from face_engine.media.capturer import VideoSource
    from face_engine.server.processing import GpuFrameProcessor

    sec = "Zero-copy & leaks"
    info = VideoSource(video).info
    frame_bytes = info.width * info.height
    inner = GpuFrameProcessor(config, tracking=True)
    try:
        pipe = CUDAStreamPipeline()
        pipe.run(video, workdir / "warm.mp4", inner, max_frames=24)  # engines load, graphs capture
        pipe.run(video, workdir / "f100.mp4", inner, max_frames=100)
        q100 = _quiescent_allocated(clear_cublas=True)
        free100 = torch.cuda.mem_get_info()[0]
        tracer = TracingProcessor(inner)
        trace = workdir / "cupti_trace.json"
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            stats = pipe.run(video, workdir / "traced.mp4", tracer, max_frames=frames)
        prof.export_chrome_trace(str(trace))
        q500 = _quiescent_allocated(clear_cublas=True)
        free500 = torch.cuda.mem_get_info()[0]
        peak = torch.cuda.max_memory_allocated()
    finally:
        inner.close()
        _free_gpu()
    n = stats.frames_done
    r.facts["traced_frames"] = n

    # --- frame pointers: the frame lives in the ring's VRAM slots throughout
    r.add(sec, "Frame tensor stays in VRAM ring slots (data_ptr trace)",
          tracer.host_frames == 0 and tracer.host_tensors == 0 and tracer.host_outputs == 0
          and len(tracer.ptrs) <= pipe.capacity,
          f"every frame on cuda, <= {pipe.capacity} distinct slot pointers",
          f"{n} frames through {len(tracer.ptrs)} slot pointer(s); host frames "
          f"{tracer.host_frames}, host plan tensors {tracer.host_tensors}, host outputs "
          f"{tracer.host_outputs}",
          fix="A stage produced a CPU tensor or allocated a new frame per frame.")

    # --- leak: what a 500-frame render leaves vs a 100-frame one, nothing in flight
    delta = q500 - q100
    r.add(sec, f"No VRAM growth frame 100 -> {n} (memory_allocated)", delta == 0,
          f"memory_allocated({n}) - memory_allocated(100) == 0",
          f"{delta:+d} B ({q100 / 2 ** 20:.3f} -> {q500 / 2 ** 20:.3f} MB after each render, "
          f"cuBLAS workspaces cleared); device-wide {(free100 - free500) / 2 ** 20:+.1f} MB "
          f"(ORT/TRT arenas + the CUPTI trace buffers of the profiled render: not a leak "
          f"measure); max_memory_allocated {peak / 2 ** 20:.0f} MB",
          fix="A 500-frame render left more live memory than a 100-frame one: a per-frame "
              "tensor is retained (a list, a cache keyed by frame, a captured graph).")

    # --- CUPTI memcpy census (process-wide: ORT and TensorRT copies included)
    copies = _memcpy_events(trace)
    kinds: dict[str, dict[str, float]] = {}
    for e in copies:
        name = e.get("name", "")
        kind = "HtoD" if "HtoD" in name else "DtoH" if "DtoH" in name else "DtoD" \
            if "DtoD" in name else "other"
        size = float(e.get("args", {}).get("bytes", 0))
        big = size >= 0.5 * frame_bytes
        k = kinds.setdefault(kind, {"count": 0, "bytes": 0.0, "frame_sized": 0,
                                    "small": 0, "small_bytes": 0.0, "pageable": 0})
        k["count"] += 1
        k["bytes"] += size
        if big:
            k["frame_sized"] += 1
        else:
            k["small"] += 1
            k["small_bytes"] += size
        if "Pageable" in name:
            k["pageable"] += 1
    r.facts["memcpy"] = kinds
    if not copies:
        r.add(sec, "PCIe copy census (CUPTI)", None, "memcpy events traced",
              "torch.profiler recorded no gpu_memcpy events (CUPTI unavailable?)",
              fix="Kineto/CUPTI tracing is unavailable in this torch build.")
        return
    h2d = kinds.get("HtoD", {})
    d2h = kinds.get("DtoH", {})
    big_h2d, big_d2h = h2d.get("frame_sized", 0), d2h.get("frame_sized", 0)
    # The render decodes on the CPU and encodes through an ffmpeg pipe, so each
    # frame crosses PCIe exactly twice: YUV planes in, RGB out. Anything else
    # frame-sized would be a mid-pipeline round trip.
    r.add(sec, "Frame data crosses PCIe only at the edges", big_h2d == n and big_d2h == n,
          "1 frame-sized H2D (decode upload) + 1 D2H (encoder readback) per frame",
          f"frame-sized H2D {big_h2d} / D2H {big_d2h} over {n} frames "
          f"({h2d.get('bytes', 0) / max(n, 1) / 2 ** 20:.2f} / "
          f"{d2h.get('bytes', 0) / max(n, 1) / 2 ** 20:.2f} MB per frame)",
          fix="A frame-sized copy beyond the two edges means a stage round-trips the frame "
              "through the host.")
    small = h2d.get("small", 0) + d2h.get("small", 0)
    pageable = h2d.get("pageable", 0) + d2h.get("pageable", 0)
    r.add(sec, "Zero H2D/D2H copies in the per-frame loop (spec, literal)", small == 0
          and big_h2d == 0 and big_d2h == 0, "0 copies",
          f"{(h2d.get('count', 0) + d2h.get('count', 0)) / max(n, 1):.1f} per frame: the 2 "
          f"frame edges + {small / max(n, 1):.1f} control-size copies "
          f"({(h2d.get('small_bytes', 0) + d2h.get('small_bytes', 0)) / max(n, 1):.0f} B/frame, "
          f"{pageable / max(n, 1):.1f}/frame pageable = synchronous; not attributed by CUPTI)",
          fix="Unreachable with this decoder/encoder: PyAV decodes on the CPU and ffmpeg "
              "encodes from host memory (in-process NVDEC measured slower and deadlocks, "
              "face_engine/media/decoder.py). Zero would need NVDEC+NVENC device-to-device "
              "and no host control reads (NMS sizes, route decisions).")


# ============================================================================ 4. detection
def _t1(width: int, height: int) -> Any:
    import cv2
    import insightface
    import torch

    image = cv2.imread(str(Path(insightface.__file__).parent / "data" / "images" / "t1.jpg"))
    return torch.from_numpy(cv2.resize(image, (width, height))).cuda().permute(2, 0, 1)[None]


def _rotate_points(points: Any, k: int, height: int, width: int) -> Any:
    """Upright -> rotated-frame coordinates (inverse of ``unrotate_points_cuda``)."""
    import torch

    x, y = points[..., 0], points[..., 1]
    k %= 4
    if k == 0:
        return points.clone()
    if k == 1:
        return torch.stack([y, width - x], -1)
    if k == 2:
        return torch.stack([width - x, height - y], -1)
    return torch.stack([height - y, x], -1)


def _dual(boxes: list[list[float]], scores: list[float], high: float = 0.5) -> Any:
    import torch

    from face_engine.pipeline.detector import DualDetections, GPUDetections

    b = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    rel = torch.tensor([[.3, .4], [.7, .4], [.5, .55], [.35, .75], [.65, .75]])
    k = b[:, None, :2] + rel[None] * (b[:, None, 2:] - b[:, None, :2])
    s = torch.tensor(scores, dtype=torch.float32)
    idx = torch.zeros(len(scores), dtype=torch.int64)
    hi = s >= high

    def mk(m: Any) -> Any:
        return GPUDetections(b[m], k[m], s[m], idx[m], (1080, 1920), 1)
    return DualDetections(mk(hi), mk(~hi))


def audit_detector_angles(r: Report) -> None:
    import torch

    from face_engine.core.config import EngineConfig, Provider
    from face_engine.core.execution import ExecutionEngine
    from face_engine.models.zoo import build_default_registry
    from face_engine.pipeline.aligner import (
        ProfileGuardedAligner,
        profile_guarded_similarity_cuda,
        template_tensor,
    )
    from face_engine.pipeline.detector import AngleResilientSCRFD
    from face_engine.pipeline.tracker import ByteTrackConfig, RobustByteTracker

    sec = "Angle-resilient detection"
    ref = _on_reference(r.gpu)
    path = build_default_registry().ensure("scrfd_10g_bnkps", show_progress=False)
    engine = ExecutionEngine(EngineConfig(providers=[Provider.TENSORRT, Provider.CUDA]))
    frame = _t1(1920, 1080)
    hw = tuple(frame.shape[-2:])
    try:
        det = AngleResilientSCRFD(engine, path)
        det.prepare(*hw)
        truth = det.detect_dual(frame).high
        n_ref = len(truth)
        backend = "TensorRT b3 engine" if det.uses_tensorrt_engine else "ONNX Runtime"

        # --- upright fast path under a live tracker: never sweeps
        det.reset()
        det.stats.sweeps = 0
        tracker = RobustByteTracker(on_lost=det.note_track_lost)
        for _ in range(10):
            tracker.update(det.detect_dual(frame))
        sweeps_before = det.stats.sweeps
        up_ms = _cuda_ms(lambda: tracker.update(det.detect_dual(frame)), reps=60)
        upright_sweeps = det.stats.sweeps - sweeps_before
        r.add(sec, "0 deg fast path skips the sweep (active track)",
              upright_sweeps == 0 and sweeps_before == 0 and n_ref == 6,
              "0 sweeps on upright frames", f"{upright_sweeps + sweeps_before} sweeps over 70 "
              f"upright frames, {n_ref}/6 faces, {backend}", up_ms,
              fix="The primary pass found rolled landmarks or lost a face on upright content.")

        # --- the tracker losing a face asks for a sweep
        det.note_track_lost(1)
        s0 = det.stats.sweeps
        det.detect_dual(frame)
        forced = det.stats.sweeps - s0
        det.detect_dual(frame)
        after = det.stats.sweeps - s0
        r.add(sec, "Sweep triggers when the track is lost", forced == 1 and after == 1,
              "note_track_lost -> exactly one sweep, then none",
              f"sweeps: {forced} on the lost frame, {after - forced} on the next",
              fix="RobustByteTracker.on_lost -> AngleResilientSCRFD.note_track_lost is unwired.")

        # --- 90 / 180 / 270: recall, landmarks, sweep-frame latency
        for k in (1, 2, 3):  # warm the sweep canvases
            det.reset()
            det.detect_dual(torch.rot90(frame, k, dims=[2, 3]).contiguous())
        size = (truth.boxes[:, 2:] - truth.boxes[:, :2]).prod(1).sqrt()
        for k in (0, 1, 2, 3):
            rot = torch.rot90(frame, k, dims=[2, 3]).contiguous()
            det.reset()
            out = det.detect_dual(rot)
            gt = _rotate_points(truth.kps, k, *hw)
            if len(out.high):
                d = (gt[:, None] - out.high.kps[None]).norm(dim=-1).mean(-1).min(1).values / size
                err = float(d.max())
            else:
                err = float("inf")
            found = len(out.high)
            if k == 0:
                lat = up_ms
                r.add(sec, "Recall at 0 deg", found == n_ref and err < 0.01,
                      "6/6 faces, landmarks < 1% of face size",
                      f"{found}/{n_ref}, landmark error {err:.2%}", lat)
                continue
            lat = _cuda_ms(lambda rot=rot: det.detect_dual(rot), reps=20, before=det.reset)
            ok = found == n_ref and err < 0.01 and bool(out.swept)
            r.add(sec, f"Recall at {90 * k} deg roll", ok,
                  "6/6 faces, landmarks < 1% of face size", f"{found}/{n_ref}, landmark error "
                  f"{err:.2%}, swept={out.swept}, best pass {out.angle} deg", lat,
                  fix="The rotation sweep missed faces; see AngleResilientSCRFD.max_pass_roll.")
            r.add(sec, f"Sweep frame latency at {90 * k} deg", lat <= SWEEP_BUDGET_MS if ref else None,
                  f"<= {SWEEP_BUDGET_MS:.1f} ms", f"{lat:.2f} ms (primary pass + batch-3 sweep "
                  "+ NMS, CUDA events, median of 20)", lat, gated=ref,
                  fix="The sweep is a batch of three 640x640 SCRFD passes plus the primary pass; "
                      "meeting 4 ms needs a smaller sweep canvas or fewer angles per frame.")
    finally:
        engine.close()
        _free_gpu()

    # --- ByteTrack: 0.82 -> 0.23 profile turn keeps the id; the control arm loses it
    scores = [0.82, 0.80, 0.62, 0.41, 0.30, 0.23, 0.23, 0.25, 0.33, 0.78]

    def turn(cfg: Any) -> tuple[list[list[int]], Any]:
        t = RobustByteTracker(cfg)
        ids = []
        for i, sc in enumerate(scores):
            x = 800 + 4 * i
            narrow = 12 if 3 <= i <= 8 else 0
            ids.append(t.update(_dual([[x + narrow, 300, x + 200 - narrow, 560]], [sc])).track_ids)
        return ids, t.stats
    ids, st = turn(ByteTrackConfig())
    ctl, _ = turn(ByteTrackConfig(low_association=False, emit_lost=False))
    kept = ids == [[0]] * len(scores)
    r.add(sec, "ByteTrack low-score association (0.82 -> 0.23)",
          kept and st.new_tracks == 1 and st.lost_events == 0 and ByteTrackConfig().low_threshold
          <= LOW_SCORE_FLOOR, f"one track id through the turn, low floor {LOW_SCORE_FLOOR}",
          f"ids {sorted({i for x in ids for i in x})}, {st.low_matches} low-score matches, "
          f"{st.lost_events} lost; control arm (no 2nd association): ids "
          f"{sorted({i for x in ctl for i in x})}, {sum(1 for x in ctl if not x)} empty frames",
          fix="The second (low-score) association is not matching the turning face.")

    r.add(sec, "Detector recall at up to 85 deg yaw", None, "100% recall",
          "not measurable here: yaw cannot be synthesised from a 2D image and no labelled "
          "profile footage is in the repo; the tracker path is covered by the row above",
          fix="Needs a clip with per-frame face ground truth through a profile turn.")

    # --- Umeyama singular-value clamp
    dst = template_tensor(256, device="cpu", dtype=torch.float64)
    tpl = torch.tensor([[38.29, 51.70], [73.53, 51.50], [56.03, 71.74],
                        [41.55, 92.37], [70.73, 92.20]], dtype=torch.float64)
    base = ((tpl - 56.0) * (120.0 / 112.0) + torch.tensor([640.0, 360.0], dtype=torch.float64))[None]
    rows = []
    for yaw in [*range(90), 90]:
        kps = base.clone()
        nose = kps[:, 2:3, 0]
        kps[..., 0] = nose + (base[..., 0] - nose) * math.cos(math.radians(yaw))
        m, ratio, clamped = profile_guarded_similarity_cuda(kps, dst, UMEYAMA_RATIO)
        det_m = float(torch.linalg.det(m[0, :, :2]))
        rows.append((yaw, float(ratio[0]), bool(clamped[0]), bool(torch.isfinite(m).all()), det_m))
    fired = [x for x in rows if x[2]]
    finite = all(x[3] for x in rows)
    clamp_rule = all(x[2] == (x[1] < UMEYAMA_RATIO) for x in rows)
    held = bool(fired) and min(x[4] for x in fired) >= 0.99 * fired[0][4]
    collapsed = base[0].clone()
    collapsed[:, 0] = base[0, 2, 0]              # every landmark on the nose's vertical line
    aligned = ProfileGuardedAligner().align(torch.rand(1, 3, 720, 1280, device="cuda") * 255,
                                            torch.stack([base[0], collapsed]).float().cuda())
    crops_ok = bool(torch.isfinite(aligned.crops).all()) and bool(aligned.valid.all())
    yaw80 = next(x for x in rows if x[0] == 80)
    yaw85 = next(x for x in rows if x[0] == 85)
    r.add(sec, "Umeyama SVD clamp (s2/s1 < 0.15)",
          finite and clamp_rule and held and crops_ok and min(x[4] for x in rows) > 0,
          "no NaN/Inf, det > 0, scale held when clamped",
          f"fires from {fired[0][0] if fired else '-'} deg (ratio {fired[0][1]:.3f}); 80 deg "
          f"ratio {yaw80[1]:.3f} clamped={yaw80[2]}, 85 deg {yaw85[1]:.3f}; det min "
          f"{min(x[4] for x in rows):.4f}; collapsed-profile crop finite={crops_ok}"
          if fired else "never fired",
          fix="profile_guarded_similarity_cuda produced a degenerate matrix.")


# ============================================================================ 5. ultra restore
def audit_ultra_restore(r: Report) -> None:
    import torch
    import torch.nn.functional as F
    from kornia.filters import gaussian_blur2d

    from face_engine.core.config import EngineConfig, Provider
    from face_engine.core.execution import ExecutionEngine
    from face_engine.enhancers import (
        FrequencySplitBlender,
        Route,
        SemanticRegionalRestorer,
        UltraRestoreEngine,
        UltraRestorer,
        gaussian_kernel_size,
    )
    from face_engine.enhancers.semantic_fusion import INNER_MOUTH, RegionWeights
    from face_engine.models.zoo import build_default_registry
    from face_engine.pipeline.aligner import similarity_matrices_cuda, warp_face_cuda
    from face_engine.pipeline.detector import GPUSCRFDDetector
    from face_engine.pipeline.masker import GPUMasker

    sec = "Ultra Restore"
    ref = _on_reference(r.gpu)
    reg = build_default_registry()
    engine = ExecutionEngine(EngineConfig(providers=[Provider.TENSORRT, Provider.CUDA]))
    try:
        base = _t1(1920, 1080).float()
        det = GPUSCRFDDetector(engine, reg.ensure("scrfd_10g_bnkps", show_progress=False))
        d = det.detect_cuda(base)
        diag = (d.boxes[:, 2:] - d.boxes[:, :2]).norm(dim=1)
        j = int(diag.argmax())
        d0 = float(diag[j])
        ure = UltraRestoreEngine(engine, {m: reg.ensure(m, show_progress=False)
                                          for m in ("gpen_bfr_512", "gpen_bfr_1024")})
        calls = {"n": 0}
        real_restore = ure.restore

        def counted(model: str, crops: Any) -> Any:
            calls["n"] += 1
            return real_restore(model, crops)
        ure.restore = counted  # type: ignore[method-assign]
        parser = GPUMasker(engine, bisenet_path=reg.ensure("bisenet_resnet34", show_progress=False))
        ur = UltraRestorer(ure, parser=parser)

        def scaled(target: float) -> tuple[Any, Any, Any, float]:
            s = target / d0
            h, w = round(1080 * s), round(1920 * s)
            f = F.interpolate(base, size=(h, w), mode="bilinear", antialias=True,
                              align_corners=False)
            sc = torch.tensor([w / 1920, h / 1080], device=f.device)
            k = d.kps[j:j + 1] * sc
            b = d.boxes[j:j + 1] * sc.repeat(2)
            return f, k, b, float((b[:, 2:] - b[:, :2]).norm(dim=1))

        def luma(x: Any, box: Any) -> float:
            x0, y0, x1, y1 = (int(v) for v in box[0].round().tolist())
            c = x[0, :, max(y0, 0):y1, max(x0, 0):x1]
            return float((0.114 * c[0] + 0.587 * c[1] + 0.299 * c[2]).mean())

        # --- bypass
        f, k, b, dg = scaled(BYPASS_DIAG)
        calls["n"] = 0
        out = ur.process(f, k, b, reference=f)
        byp_ms = _cuda_ms(lambda: ur.process(f, k, b, reference=f), reps=30)
        identical = torch.equal(out.frames, f)
        r.add(sec, f"Scale gate: d = {dg:.0f} px -> bypass",
              out.plan.counts == {Route.BYPASS: 1} and calls["n"] == 0 and identical,
              "no restoration (0.0 ms network)", f"route {next(iter(out.plan.counts)).name}, "
              f"{calls['n']} network calls, output bit-identical={identical}; routing + "
              f"clone {byp_ms:.2f} ms", 0.0 if calls["n"] == 0 else byp_ms,
              fix="ScaleAwareEnhancerRouter.bypass_below must be > the face diagonal.")

        # --- GPEN-512 / GPEN-1024
        for target, want, budget, model, size in (
                (GPEN512_DIAG, Route.GPEN_512, GPEN512_BUDGET_MS, "gpen_bfr_512", 512),
                (GPEN1024_DIAG, Route.GPEN_1024, GPEN1024_BUDGET_MS, "gpen_bfr_1024", 1024)):
            f, k, b, dg = scaled(target)
            calls["n"] = 0
            out = ur.process(f, k, b, reference=f)
            routed = out.plan.counts == {want: 1} and bool(out.restored.all())
            total = _cuda_ms(lambda f=f, k=k, b=b: ur.process(f, k, b, reference=f), reps=30)
            crop = warp_face_cuda(f, similarity_matrices_cuda(k, size, "ffhq_512"), size)
            net = _cuda_ms(lambda crop=crop, model=model: real_restore(model, crop), reps=30)
            trt = ure.uses_tensorrt(model)
            r.add(sec, f"Scale gate: d = {dg:.0f} px -> {want.name}", routed,
                  f"routed to {want.name}, restored", f"routes {out.plan.to_dict()['routes']}, "
                  f"restored={bool(out.restored.all())}", fix="Router thresholds changed.")
            r.add(sec, f"{want.name} FP16 latency (1 face)", total <= budget if ref else None,
                  f"<= {budget} ms", f"{total:.2f} ms end to end; network alone {net:.2f} ms "
                  f"({'TensorRT FP16 engine' if trt else 'ONNX Runtime'})", total, gated=ref,
                  fix=f"The {model} network alone exceeds the budget on this card (measured "
                      "2026-09-28: 17.8 / 32.3 ms TensorRT FP16, 14 layers pinned FP32 to stop "
                      "the overflow). A budget this low needs a smaller restorer, not tuning.")
            lin, lout = luma(f, b), luma(out.frames, b)
            rel = abs(lout - lin) / max(lin, 1e-6)
            r.add(sec, f"Luminance preserved ({want.name})", rel <= LUMA_TOLERANCE,
                  f"face-box mean luma within {LUMA_TOLERANCE:.0%}", f"{lin:.2f} -> {lout:.2f} "
                  f"({rel:.2%})", fix="The split's low band is not the swap's: check lab_lock.")

        # --- frequency split formulation
        blender = ur.regional.blender
        f, k, _, _ = scaled(GPEN512_DIAG)
        x = warp_face_cuda(f, similarity_matrices_cuda(k, 512, "ffhq_512"), 512).clamp(0, 255)
        lo, hi = blender.split(x)
        recon = float((lo + hi - x).abs().max())
        ksz = gaussian_kernel_size(blender.sigma)
        ref_lo = gaussian_blur2d(x, (ksz, ksz), (2.5, 2.5))
        same = float((lo - ref_lo).abs().max())
        crossfade = FrequencySplitBlender(swap_detail="complement").blend(x, x * 0.9,
                                                                         texture_boost=0.0)
        r.add(sec, "Frequency split: Gaussian sigma 2.5 low + high detail",
              blender.sigma == 2.5 and blender.low_pass == "gaussian" and recon < 1e-3
              and same < 1e-3 and float((crossfade - x).abs().max()) < 1e-3,
              "low = G(sigma 2.5); low + high == crop", f"sigma {blender.sigma}, kernel {ksz}, "
              f"|low+high-x| {recon:.1e}, |low-G2.5(x)| {same:.1e}, boost 0 returns the swap",
              fix="FrequencySplitBlender defaults changed.")

        # --- inner mouth attenuation
        weights = ur.regional.weights
        labels = torch.full((1, 512, 512), INNER_MOUTH, dtype=torch.long, device="cuda")
        wmap = SemanticRegionalRestorer(RegionWeights(), feather_kernel=1).label_weights(labels, 512)
        applied = float(wmap.max())
        r.add(sec, "Oral cavity attenuation (inner mouth)",
              weights.inner_mouth <= INNER_MOUTH_MAX and abs(applied - weights.inner_mouth) < 1e-6,
              f"texture weight <= {INNER_MOUTH_MAX}", f"configured {weights.inner_mouth}, applied "
              f"{applied:.2f} over class {INNER_MOUTH} (BiSeNet inner mouth); eyes "
              f"{weights.eyes}, skin {weights.skin}",
              fix="RegionWeights.inner_mouth was raised.")
    finally:
        engine.close()
        _free_gpu()


# ============================================================================ report
def audit_pose_routing(r: Report) -> None:
    """Pose-adaptive source routing overhead, CPU only (no model is loaded)."""
    import numpy as np

    app = ROOT / "app"
    if str(app) not in sys.path:
        sys.path.insert(0, str(app))
    from roop import face_util as fu
    from roop.angle_portfolio import AngleBin
    from roop.source_portfolio import FrameLUT, LUTEntry, SourcePortfolio, SourceRef, route

    sec = "Pose routing"
    rng = np.random.default_rng(0)

    def unit(v):
        return (v / np.linalg.norm(v)).astype(np.float32)

    dim = 512
    refs = {b: SourceRef(unit(rng.normal(size=dim)), 0.0, 0.0, 0.9, i) for i, b in enumerate(AngleBin)}
    pf = SourcePortfolio(refs=refs, fused=unit(rng.normal(size=dim)), dim=dim)
    # Target keypoints on the reference head across the yaw range, in a 1080p frame.
    faces = []
    for yaw in np.linspace(-75, 75, 31):
        pts = fu._project_reference(float(yaw), float(rng.uniform(-20, 20)))
        pts = (pts - pts.mean(axis=0)) * 180 + (960, 540)
        x0, y0 = pts.min(axis=0) - 60
        x1, y1 = pts.max(axis=0) + 60
        faces.append((pts.astype(np.float32), [float(x0), float(y0), float(x1), float(y1)], float(yaw)))
    entries = {i: [LUTEntry(yaw=f[2], pitch_up=0.0, bin=None, bbox=tuple(f[1]))] for i, f in enumerate(faces) if i % 3 == 0}
    lut = FrameLUT(media_path="audit", step=3, entries=entries)

    paths = {"LUT hit": lambda i, f: route(pf, f[0], f[1], frame_idx=(i // 3) * 3, lut=lut),
             "neighbouring frame": lambda i, f: route(pf, f[0], f[1], frame_idx=(i // 3) * 3 + 1, lut=lut),
             "live keypoint solve": lambda i, f: route(pf, f[0], f[1], frame_idx=None, lut=None)}
    shapes = set()
    for name, call in paths.items():
        for i, f in enumerate(faces):                      # warm
            call(i, f)
        samples = []
        seen_from = set()
        for rep_ in range(60):
            for i, f in enumerate(faces):
                t = time.perf_counter_ns()
                out = call(i, f)
                samples.append((time.perf_counter_ns() - t) / 1e6)
                shapes.add((out.embedding.shape, str(out.embedding.dtype)))
                seen_from.add(out.pose_from)
        samples.sort()
        mean = statistics.fmean(samples)
        p99 = samples[int(len(samples) * 0.99) - 1]
        r.add(sec, f"route() per face, pose from {name}", mean < ROUTING_BUDGET_MS and p99 < ROUTING_BUDGET_MS,
              f"< {ROUTING_BUDGET_MS} ms (mean and p99)",
              f"mean {mean:.4f} ms, p99 {p99:.4f} ms over {len(samples)} faces; path {sorted(seen_from)}",
              latency_ms=mean)
    r.add(sec, "Swap input vector shape across reference switches", shapes == {((dim,), "float32")},
          f"one shape ({dim},) float32 for every route",
          f"{sorted(shapes)}",
          fix="route() must never hand the swapper a vector of another size.")


def render_report(r: Report) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console(width=200)
    f = r.facts
    console.print(f"[bold]face_engine production audit[/] - {r.gpu} - profile [bold]{r.profile}"
                  f"[/] - torch {f.get('torch')} / CUDA {f.get('cuda_runtime')} / TensorRT "
                  f"{f.get('tensorrt')} / driver {f.get('driver')}")
    table = Table(show_lines=False)
    table.add_column("Subsystem Component", overflow="fold", max_width=48)
    table.add_column("Status", justify="center")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Target Metric", overflow="fold", max_width=34)
    table.add_column("Measured", overflow="fold", style="dim", max_width=80)
    colours = {PASS: "green", FAIL: "bold red", UNVERIFIED: "yellow", INFO: "cyan"}
    section = None
    for c in r.checks:
        if c.section != section:
            if section is not None:
                table.add_section()
            table.add_row(f"[bold]{c.section}[/]", "", "", "", "")
            section = c.section
        lat = "-" if c.latency_ms is None else f"{c.latency_ms:.2f}"
        table.add_row("  " + c.component, f"[{colours[c.status]}]{c.status}[/]", lat, c.target,
                      c.measured)
    console.print(table)
    counts = {s: sum(1 for c in r.checks if c.status == s) for s in (PASS, FAIL, UNVERIFIED, INFO)}
    console.print(f"{counts[PASS]} passed, {counts[FAIL]} failed, {counts[UNVERIFIED]} unverified, "
                  f"{counts[INFO]} informational")
    if r.failed:
        console.print("\n[bold red]AUDIT FAILED[/] - diagnostics:")
        for c in r.failed:
            console.print(f"  [bold]{c.status}[/] {c.section} / {c.component}: {c.measured}")
            if c.fix:
                console.print(f"      -> {c.fix}")
    else:
        console.print("[bold green]AUDIT PASSED[/]")


SECTIONS = ("hardware", "pipeline", "memory", "detector", "restore", "routing")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="audit_system.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", choices=sorted(FPS_TARGET), default="balanced")
    ap.add_argument("--input", default=str(DEFAULT_CLIP),
                    help="1080p clip with faces and audio (>= 500 frames)")
    ap.add_argument("--repeats", type=int, default=2, help="timed full-clip renders")
    ap.add_argument("--memory-frames", type=int, default=MEMORY_FRAMES)
    ap.add_argument("--only", nargs="+", choices=SECTIONS, help="run only these sections")
    ap.add_argument("--json", help="also write the report as JSON here")
    args = ap.parse_args(argv)
    sections = set(args.only or SECTIONS)
    video = Path(args.input).resolve()
    if not video.is_file() and sections & {"pipeline", "memory"}:
        raise SystemExit(f"{video} not found (--input)")

    r = Report(profile=args.profile)
    t0 = time.perf_counter()
    # Order: renders first, so their VRAM reading is not inflated by the
    # detector / restorer sessions the later sections load.
    audit_hardware(r, args.profile)
    if not r.gpu:
        render_report(r)
        return 1
    with tempfile.TemporaryDirectory(prefix="audit_") as tmp:
        work = Path(tmp)
        config = _build_config(video, args.profile) if sections & {"pipeline", "memory"} else None
        steps: list[tuple[str, Callable[[], None]]] = [
            ("pipeline", lambda: audit_async_pipeline(r, video, args.profile, config,
                                                      args.repeats, work)),
            ("memory", lambda: audit_memory_invariants(r, video, config, args.memory_frames,
                                                       work)),
            ("detector", lambda: audit_detector_angles(r)),
            ("restore", lambda: audit_ultra_restore(r)),
            ("routing", lambda: audit_pose_routing(r)),
        ]
        for name, fn in steps:
            if name not in sections:
                continue
            print(f"[audit] {name} ... ({time.perf_counter() - t0:.0f}s)", flush=True)
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - a crashed section is a failed section
                import traceback

                traceback.print_exc()
                r.add(name, f"{name} section crashed", False, "runs to completion",
                      f"{type(exc).__name__}: {str(exc)[:160]}", fix="See the traceback above.")
                _free_gpu()
    order = {"Hardware & drivers": 0, "Zero-copy & leaks": 1, "Angle-resilient detection": 2,
             "Ultra Restore": 3, "Async CUDA streams": 4, "Stage latency": 5, "Pose routing": 6}
    r.checks.sort(key=lambda c: order.get(c.section, 9))
    r.facts["elapsed_s"] = round(time.perf_counter() - t0, 1)
    render_report(r)
    if args.json:
        Path(args.json).write_text(json.dumps({"gpu": r.gpu, "profile": r.profile,
                                               "checks": [asdict(c) for c in r.checks],
                                               "facts": r.facts}, indent=2, default=str),
                                   encoding="utf-8")
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
