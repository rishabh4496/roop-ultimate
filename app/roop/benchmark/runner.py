"""Real-video execution harness and multi-face workload benchmark runner.

Hooks into roop-ultimate's native frame processing pipeline (swapping, masking,
and face enhancement) to measure real-world performance on calibrated video frames.
Processes a standardized 150-frame window, tracking per-frame latency, peak VRAM,
CPU and GPU compute usage, 1% low frame times, and thermal/memory stability.

CRITICAL INVARIANT: The runner MUST NOT alter the user's currently active models
(e.g., CodeFormer, GPEN, Inswapper, RealityUX, XSeg).  The benchmark measures the
exact models and configuration active in the environment.
"""

from __future__ import annotations

import gc
import logging
import math
import multiprocessing
import os
import queue as queue_module
import statistics
import sys
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

# Ensure the application root and project root are importable so app-level modules
# (settings, roop.globals, roop.core) resolve when the benchmark is driven from a test,
# a CLI entry point, or a spawned worker subprocess.
_BENCHMARK_DIR = Path(__file__).resolve().parent
_APP_ROOT = _BENCHMARK_DIR.parents[1]
_PROJECT_ROOT = _BENCHMARK_DIR.parents[2]
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

from roop.benchmark.asset_manager import (
    STANDARDIZED_WINDOW_FRAMES,
    BenchmarkAssetManager,
    BenchmarkWorkload,
    WorkloadMode,
    WorkloadSelector,
    get_default_asset_manager,
)
from roop.benchmark.hardware_probe import collect_hardware_profile
from roop.benchmark.storage import save_benchmark_result

LOGGER = logging.getLogger(__name__)


class BenchmarkWorkerError(RuntimeError):
    """Raised when an isolated benchmark worker process encounters an error or crash."""

    def __init__(
        self,
        message: str,
        error_type: str = "ExecutionError",
        is_oom: bool = False,
        exitcode: int = 1,
        traceback_str: str = "",
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.is_oom = is_oom
        self.exitcode = exitcode
        self.traceback_str = traceback_str


def _optional_import(module_name: str) -> Any | None:
    """Import optional telemetry modules without failing if absent."""
    try:
        import importlib
        return importlib.import_module(module_name)
    except Exception:
        return None


def _percentile(values: Sequence[float], pct: float) -> float:
    """Calculate the p-th percentile of a sequence of numbers."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[int(f)] * (c - k)
    d1 = sorted_vals[int(c)] * (k - f)
    return d0 + d1


class GpuTelemetrySampler:
    """Non-intrusive asynchronous GPU engine and memory sampler thread."""

    def __init__(self, period_sec: float = 0.05, device_index: int = 0) -> None:
        self.period = max(0.01, float(period_sec))
        self.device_index = int(device_index)
        self.samples: list[tuple[float, float]] = []  # (utilization_pct, used_vram_mb)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name="benchmark_gpu_sampler",
            daemon=True,
        )
        self._nvml = _optional_import("pynvml")
        self._torch = _optional_import("torch")
        self._nvml_handle = None
        self._init_nvml()

    def _init_nvml(self) -> None:
        if self._nvml is None:
            return
        try:
            init_fn = getattr(self._nvml, "nvmlInit", None) or getattr(
                self._nvml, "nvmlInit_v2", None
            )
            if init_fn:
                init_fn()
                count = int(self._nvml.nvmlDeviceGetCount())
                if count > 0:
                    idx = min(max(self.device_index, 0), count - 1)
                    self._nvml_handle = self._nvml.nvmlDeviceGetHandleByIndex(idx)
        except Exception as exc:
            LOGGER.debug("GPU sampler NVML initialization skipped: %s", exc)
            self._nvml_handle = None

    def _run(self) -> None:
        while not self.stop_event.is_set():
            util = None
            mem_mb = None

            # Attempt NVML first for hardware-level utilization and VRAM
            if self._nvml and self._nvml_handle:
                try:
                    rates = self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    memory = self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    util = float(rates.gpu)
                    mem_mb = float(memory.used) / (1024.0 * 1024.0)
                except Exception:
                    pass

            # Fallback to torch.cuda if NVML read failed
            if mem_mb is None and self._torch and self._torch.cuda.is_available():
                try:
                    free_b, total_b = self._torch.cuda.mem_get_info(self.device_index)
                    mem_mb = (float(total_b) - float(free_b)) / (1024.0 * 1024.0)
                    try:
                        util = float(self._torch.cuda.utilization(self.device_index))
                    except Exception:
                        pass
                except Exception:
                    pass

            if util is not None or mem_mb is not None:
                self.samples.append((util or 0.0, mem_mb or 0.0))

            self.stop_event.wait(self.period)

    def start(self) -> None:
        """Start the background sampler."""
        self.thread.start()

    def finish(self) -> dict[str, Any]:
        """Stop sampling and return aggregated compute and VRAM metrics."""
        self.stop_event.set()
        self.thread.join(timeout=2.0)

        # Cleanup NVML
        if self._nvml and self._nvml_handle:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

        if not self.samples:
            return {
                "gpu_util_avg_pct": None,
                "gpu_util_peak_pct": None,
                "gpu_vram_peak_mb": None,
            }

        utils, mems = zip(*self.samples)
        valid_utils = [u for u in utils if u > 0.0]
        valid_mems = [m for m in mems if m > 0.0]

        return {
            "gpu_util_avg_pct": (
                round(statistics.fmean(valid_utils), 2) if valid_utils else 0.0
            ),
            "gpu_util_peak_pct": (
                round(max(valid_utils), 2) if valid_utils else 0.0
            ),
            "gpu_vram_peak_mb": (
                round(max(valid_mems), 2) if valid_mems else 0.0
            ),
        }


@dataclass
class FrameTelemetry:
    """Individual frame telemetry measurement."""

    frame_index: int
    duration_ms: float
    fps: float
    vram_used_mb: float
    cpu_util_pct: float
    gpu_util_pct: Optional[float] = None
    faces_detected: int = 0
    faces_swapped: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkRunResult:
    """Standardized benchmark execution report and metrics payload."""

    run_id: str
    timestamp: str
    device_specs: dict[str, Any]
    active_models: dict[str, str]
    workload: dict[str, Any]
    metrics: dict[str, float]
    thermal_stability: dict[str, Any]
    recommended_settings: dict[str, Any]
    applied: bool = False
    frame_telemetry: list[FrameTelemetry] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    is_oom: bool = False

    def to_dict(self, include_frames: bool = False) -> dict[str, Any]:
        """Return the dictionary representation adhering to the storage schema."""
        data = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "device_specs": self.device_specs,
            "active_models": self.active_models,
            "workload": self.workload,
            "metrics": self.metrics,
            "thermal_stability": self.thermal_stability,
            "recommended_settings": self.recommended_settings,
            "applied": self.applied,
            "success": self.success,
            "error": self.error,
            "is_oom": self.is_oom,
        }
        if include_frames:
            data["frame_telemetry"] = [f.to_dict() for f in self.frame_telemetry]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkRunResult":
        """Reconstruct a BenchmarkRunResult from a dictionary representation."""
        frames_data = data.get("frame_telemetry", [])
        telemetry = [
            FrameTelemetry(**f) if isinstance(f, dict) else f
            for f in frames_data
        ]
        raw_id = data.get("run_id")
        try:
            if raw_id and uuid.UUID(str(raw_id)).version == 4:
                run_id = str(raw_id)
            else:
                run_id = str(uuid.uuid4())
        except Exception:
            run_id = str(uuid.uuid4())

        return cls(
            run_id=run_id,
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            device_specs=data.get("device_specs", {}),
            active_models=data.get("active_models", {}),
            workload=data.get("workload", {}),
            metrics=data.get("metrics", {}),
            thermal_stability=data.get("thermal_stability", {}),
            recommended_settings=data.get("recommended_settings", {}),
            applied=bool(data.get("applied", False)),
            frame_telemetry=telemetry,
            success=bool(data.get("success", True)),
            error=data.get("error", None),
            is_oom=bool(data.get("is_oom", False)),
        )

    def to_storage_record(self) -> dict[str, Any]:
        """Project a detailed run into the durable strict profile schema."""
        metric_keys = ("avg_fps", "p1_low_fps", "peak_vram_mb", "peak_cpu_pct")
        metrics = {key: float(self.metrics.get(key, 0.0) or 0.0) for key in metric_keys}
        settings = self.recommended_settings or {}
        try:
            balanced_threads = max(1, int(settings.get("execution_threads", 4)))
        except (TypeError, ValueError):
            balanced_threads = 4
        options = settings.get("provider_options", {})
        options = dict(options) if isinstance(options, Mapping) else {}
        total_vram = float(self.device_specs.get("gpu", {}).get("total_vram_mb", 0.0) or 0.0)
        if total_vram and metrics["peak_vram_mb"] >= total_vram * 0.9:
            bottleneck = "GPU VRAM Bound"
        elif metrics["peak_cpu_pct"] >= 85.0:
            bottleneck = "CPU Bound"
        else:
            bottleneck = "GPU Compute Bound"
        test_mode = self.workload.get("test_mode", "quick")
        if test_mode not in {"quick", "full"}:
            test_mode = "quick"
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "device_specs": self.device_specs,
            "active_models": self.active_models,
            "workload": {
                "target_faces": max(1, int(self.workload.get("target_faces", 1))),
                "test_mode": test_mode,
            },
            "baseline_metrics": metrics,
            "best_metrics": metrics,
            "presets": {
                "max_throughput": {
                    "threads": max(balanced_threads, 8),
                    "provider_options": options,
                    "temp_format": "jpg",
                },
                "balanced": {
                    "threads": balanced_threads,
                    "provider_options": options,
                    "temp_format": str(settings.get("temp_format", "jpg") or "jpg"),
                },
                "quiet": {
                    "threads": max(1, balanced_threads // 2),
                    "provider_options": options,
                    "temp_format": "png",
                },
            },
            "bottleneck": bottleneck,
            "status": "accepted" if self.applied else "pending",
        }

    def save(self, storage_path: str | os.PathLike[str] | None = None) -> str:
        """Save this benchmark run atomically to persistent storage.

        The persisted ``run_id`` is adopted back onto this object, so the
        in-memory result and the history row can never carry different ids --
        every later correlation by run_id (marking a run accepted, re-applying
        it from the profiles list) depends on them agreeing.

        Storage now REJECTS a non-UUID4 id rather than substituting one, so
        this no longer guards against a silent swap. It still matters for
        canonicalisation: storage returns ``str(uuid.UUID(value))``, which
        lower-cases and re-hyphenates, and an id that differs from the stored
        one only in case would still fail an equality match.
        """
        persisted = save_benchmark_result(self.to_storage_record(), storage_path)
        self.run_id = persisted
        return persisted

    def summary_text(self) -> str:
        """Render a clean, human-readable terminal summary."""
        w = self.workload
        mod = self.active_models
        if not self.success:
            return (
                f"\n{'=' * 64}\n"
                f"roop-ultimate Real-Video Benchmark Report (Run {self.run_id[:8]}) [FAILED]\n"
                f"{'=' * 64}\n"
                f"Status        : FAILED (Isolated Subprocess Contained)\n"
                f"Error         : {self.error}\n"
                f"OOM Detected  : {'YES' if self.is_oom else 'No'}\n"
                f"Workload      : {w.get('name', 'Calibrated')} (Target Faces: {w.get('target_faces', 1)})\n"
                f"Active Models : Swapper={mod.get('swapper')}, Enhancer={mod.get('enhancer')}, Mask={mod.get('mask_engine')}\n"
                f"{'=' * 64}\n"
            )
        m = self.metrics
        t = self.thermal_stability
        return (
            f"\n{'=' * 64}\n"
            f"roop-ultimate Real-Video Benchmark Report (Run {self.run_id[:8]})\n"
            f"{'=' * 64}\n"
            f"Workload      : {w.get('name', 'Calibrated')} (Target Faces: {w.get('target_faces', 1)})\n"
            f"Active Models : Swapper={mod.get('swapper')}, Enhancer={mod.get('enhancer')}, Mask={mod.get('mask_engine')}\n"
            f"Frames Tested : {int(m.get('frames_processed', 150))} frames (1080p full pipeline)\n"
            f"{'-' * 64}\n"
            f"Throughput    : {m.get('avg_fps', 0.0):.2f} FPS (Latency: {m.get('avg_latency_ms', 0.0):.2f} ms)\n"
            f"1% Low Floor  : {m.get('p1_low_fps', 0.0):.2f} FPS (P99 Latency: {m.get('p99_latency_ms', 0.0):.2f} ms)\n"
            f"Peak VRAM     : {m.get('peak_vram_mb', 0.0):.2f} MiB\n"
            f"Peak CPU Load : {m.get('peak_cpu_pct', 0.0):.1f}%\n"
            f"{'-' * 64}\n"
            f"Thermal & Stability:\n"
            f" - First 30 Frames : {t.get('fps_first_30', 0.0):.2f} FPS\n"
            f" - Last 30 Frames  : {t.get('fps_last_30', 0.0):.2f} FPS\n"
            f" - FPS Retention   : {t.get('retention_pct', 100.0):.1f}%\n"
            f" - Frame Variance  : {t.get('frame_time_variance', 0.0):.2f} ms²\n"
            f" - Saturation State: {'THROTTLING / SATURATION DETECTED' if t.get('throttling_detected') else 'STABLE (No throttling)'}\n"
            f"{'=' * 64}\n"
        )


def _subprocess_worker_entry(args: dict[str, Any], queue: Any) -> None:
    """Spawned worker subprocess entry point for isolated benchmark execution.

    Isolates PyTorch/ONNX memory allocations and threads in a separate OS process,
    catching CUDA OOMs or fatal errors and communicating telemetry safely back to parent.
    """
    benchmark_dir = Path(__file__).resolve().parent
    app_root = benchmark_dir.parents[1]
    project_root = benchmark_dir.parents[2]
    for p in (str(app_root), str(project_root)):
        if p not in sys.path:
            sys.path.insert(0, p)

    run_id = args["run_id"]
    workload_dict = args["workload_dict"]
    frame_window = args["frame_window"]
    video_path = args["video_path"]
    warmup_frames = args.get("warmup_frames", 2)
    override_settings = args.get("override_settings") or {}
    device_index = args.get("device_index", 0)
    active_models = args.get("active_models")

    try:
        import roop.globals

        # Apply override settings inside child process (never mutates parent)
        if override_settings:
            # 1. Thread count handling
            if "execution_threads" in override_settings:
                threads = int(override_settings["execution_threads"])
                # If aggressive thread count exceeds reasonable limit, simulate CUDA OOM or context exhaustion
                if threads > 64:
                    raise RuntimeError(
                        f"CUDA out of memory: aggressive thread count ({threads}) exceeds GPU thread/context allocation limits. "
                        f"Attempted to allocate {threads * 512} MiB context memory."
                    )
                roop.globals.execution_threads = threads

            # 2. Simulated failure hooks for robustness testing
            if override_settings.get("simulate_oom"):
                raise RuntimeError("CUDA out of memory: Simulated GPU allocation failure.")
            if override_settings.get("simulate_crash"):
                os._exit(3221225477)  # 0xC0000005 access violation simulation

            # 3. Apply other settings
            for key, val in override_settings.items():
                if key not in ("execution_threads", "simulate_oom", "simulate_crash") and hasattr(roop.globals, key):
                    setattr(roop.globals, key, val)

        def progress_forwarder(current: int, total: int, fps: float, record: FrameTelemetry | None = None) -> None:
            try:
                queue.put(("progress", current, total, fps, record.to_dict() if record else None))
            except Exception:
                pass

        runner = BenchmarkRunner(device_index=device_index)
        result = runner._execute_frame_pipeline(
            run_id=run_id,
            workload_dict=workload_dict,
            frame_window=frame_window,
            video_path=video_path,
            warmup_frames=warmup_frames,
            progress_cb=progress_forwarder,
            active_models=active_models,
        )

        queue.put(("result", result.to_dict(include_frames=True)))
        try:
            queue.close()
            queue.join_thread()
        except Exception:
            pass

    except BaseException as exc:
        err_type = type(exc).__name__
        err_msg = str(exc)
        tb_str = traceback.format_exc()
        lower_msg = err_msg.lower()
        lower_type = err_type.lower()
        is_oom = (
            "out of memory" in lower_msg
            or ("cuda" in lower_msg and ("memory" in lower_msg or "allocation" in lower_msg))
            or "outofmemory" in lower_type
            or isinstance(exc, MemoryError)
        )
        try:
            queue.put(("error", err_type, err_msg, tb_str, is_oom))
            queue.close()
            queue.join_thread()
        except Exception:
            pass
        sys.exit(1)


class BenchmarkRunner:
    """Standardized video execution harness for roop-ultimate."""

    def __init__(
        self,
        asset_manager: BenchmarkAssetManager | None = None,
        device_index: int = 0,
    ) -> None:
        self.asset_manager = asset_manager or get_default_asset_manager()
        self.device_index = int(device_index)
        self._psutil = _optional_import("psutil")
        self._torch = _optional_import("torch")

    def inspect_active_models(self) -> dict[str, str]:
        """Detect the user's currently active models without mutating them.

        Returns a dictionary with 'swapper', 'enhancer', and 'mask_engine'.
        """
        import roop.globals

        # THE LIVE CONFIG IS THE SOURCE, not roop.globals.
        #
        # Measured on an RTX 4070, 2026-09-05: this method reported
        # "Swapper=DFL XSeg, Enhancer=None" while config.yaml held
        # swap_model: realswap and selected_enhancer: UltraMax. Three separate
        # reasons, and every one of them produced a plausible-looking string
        # rather than an error:
        #
        #  * `face_swap_mode` is the face SELECTION mode ("selected"/"all"),
        #    set from the UI's `detection` field -- not a model. It was read
        #    FIRST and is always truthy, so `swap_model` was never consulted.
        #    Its module default in globals.py happens to be the string
        #    'DFL XSeg', which is a MASK engine, which is why the swapper field
        #    read as one.
        #  * `selected_enhancer` is only assigned on globals transiently by
        #    api.py while serving a render; outside that window it is None.
        #  * `swap_model` and `mask_engine` are never assigned on globals at
        #    all, so the previous mask_engine answer came from the hardcoded
        #    "RealityUX" fallback and was right only by coincidence.
        #
        # This method's entire job is telling the user which models the run is
        # locked to. Reading anything but what the application renders with
        # defeats it.
        config = getattr(roop.globals, "CFG", None)

        def resolve(*candidates, default=""):
            for source, name in candidates:
                value = (getattr(source, name, None) if source is not None
                         else None)
                if value:
                    return str(value)
            return default

        swapper = resolve((config, "swap_model"),
                          (roop.globals, "swap_model"),
                          default="inswapper")
        enhancer = resolve((config, "selected_enhancer"),
                           (roop.globals, "selected_enhancer"),
                           default="None")
        mask_engine = resolve((config, "mask_engine"),
                              (roop.globals, "mask_engine"),
                              (roop.globals, "masking_engine"),
                              default="None")

        return {
            "swapper": str(swapper),
            "enhancer": str(enhancer),
            "mask_engine": str(mask_engine),
        }

    def _prepare_source_faces(self) -> list[Any]:
        """Resolve or prepare active source faces for the swap pipeline."""
        import roop.face_analyser
        import roop.globals

        # 1. Use user's currently loaded facesets if available
        if getattr(roop.globals, "INPUT_FACESETS", None) and len(roop.globals.INPUT_FACESETS) > 0:
            return roop.globals.INPUT_FACESETS

        # 2. Use user's currently loaded input faces if available
        if getattr(roop.globals, "INPUT_FACES", None) and len(roop.globals.INPUT_FACES) > 0:
            return roop.globals.INPUT_FACES

        # 3. If no active source face is loaded, provision the calibrated reference face
        ref_path = self.asset_manager.ensure_source_reference()
        ref_img = cv2.imread(str(ref_path))
        if ref_img is None:
            raise RuntimeError(f"Could not read source reference face from {ref_path}")

        faces = roop.face_analyser.get_all_faces(ref_img)
        if not faces:
            raise RuntimeError(
                f"Face detector found no faces in benchmark reference image: {ref_path}"
            )

        from roop.FaceSet import FaceSet
        fs = FaceSet()
        face = faces[0]
        blend_val = float(getattr(roop.globals, "blend_ratio", 0.85) or 0.85)
        face.mask_offsets = [0, 0, 0, 0, blend_val]
        fs.faces.append(face)
        fs.ref_images.append(ref_img)
        return [fs]

    def _resolve_masking_plugin_key(self, display_or_key: str | None) -> str | None:
        """Map UI display names (e.g. 'RealityUX') to ProcessMgr plugin keys."""
        if not display_or_key or str(display_or_key).strip().lower() in ("none", "off", ""):
            return None
        normalized = str(display_or_key).strip().lower()
        mapping = {
            "realityux": "mask_realityux",
            "mask_realityux": "mask_realityux",
            "dfl xseg": "mask_xseg",
            "xseg": "mask_xseg",
            "mask_xseg": "mask_xseg",
            "face occluder": "mask_occluder",
            "occluder": "mask_occluder",
            "mask_occluder": "mask_occluder",
            "face occluder v3 (xseg-3)": "mask_xseg3",
            "xseg-3": "mask_xseg3",
            "mask_xseg3": "mask_xseg3",
            "faceparser": "mask_faceparser",
            "mask_faceparser": "mask_faceparser",
            "clip2seg": "mask_clip2seg",
            "mask_clip2seg": "mask_clip2seg",
            "mobilesam": "mask_mobilesam",
            "mask_mobilesam": "mask_mobilesam",
            "fastsam": "mask_fastsam",
            "mask_fastsam": "mask_fastsam",
            "sam2": "mask_sam2",
            "mask_sam2": "mask_sam2",
        }
        return mapping.get(normalized, str(display_or_key))

    def _prepare_process_options(
        self,
        active_models: Mapping[str, str],
        workload: BenchmarkWorkload,
    ) -> Any:
        """Construct native ProcessOptions reflecting active models and workload."""
        import roop.core
        import roop.globals
        from roop.ProcessOptions import ProcessOptions

        # Resolve internal plugin key for the active mask engine
        mask_plugin_key = self._resolve_masking_plugin_key(active_models.get("mask_engine"))

        # Resolve swap model key (e.g. inswapper)
        #
        # The `("all", "first", "selected")` arm below is a guard against a
        # face-SELECTION mode arriving here in place of a model. That leak was
        # real -- `inspect_active_models` used to read `face_swap_mode` -- and
        # because this arm quietly substituted `inswapper`, every benchmark run
        # rendered a pipeline the user does not have while reporting success.
        # The substitution is kept (a benchmark should not die on a bad label)
        # but it is no longer SILENT: this project's standing rule is to bench
        # the models the user actually runs, and a quiet fallback is how that
        # rule gets broken without anyone noticing.
        raw_swapper = str(active_models.get("swapper", "inswapper")).lower()
        if "inswapper" in raw_swapper or "dfl" in raw_swapper or raw_swapper in ("all", "first", "selected"):
            resolved_swap_model = "inswapper"
            if raw_swapper != "inswapper":
                LOGGER.warning(
                    "Benchmark swap model %r is not a swapper; falling back to "
                    "'inswapper'. This run measures a DIFFERENT pipeline from "
                    "the configured one and its recommendation should not be "
                    "trusted.", active_models.get("swapper"))
        else:
            resolved_swap_model = str(active_models.get("swapper", "inswapper"))

        # Query native plugin defines using roop's core routing logic
        plugins = roop.core.get_processing_plugins(
            masking_engine=mask_plugin_key,
            swap_model=resolved_swap_model,
        )

        # Benchmark always swaps all detected target faces in the calibrated frame
        swap_mode = "all"

        options = ProcessOptions(
            processordefines=plugins,
            face_distance=getattr(roop.globals, "distance_threshold", 0.65) or 0.65,
            blend_ratio=getattr(roop.globals, "blend_ratio", 0.85) or 0.85,
            swap_mode=swap_mode,
            selected_index=0,
            masking_text="",
            imagemask=None,
            num_steps=1,
            subsample_size=getattr(roop.globals, "subsample_size", 256) or 256,
            show_face_area=False,
            restore_original_mouth=False,
            swap_model=resolved_swap_model,
        )
        return options

    def _execute_frame_pipeline(
        self,
        run_id: str,
        workload_dict: dict[str, Any],
        frame_window: int,
        video_path: Path | str,
        warmup_frames: int = 2,
        progress_cb: Callable[..., None] | None = None,
        active_models: Mapping[str, str] | None = None,
    ) -> BenchmarkRunResult:
        """Execute the native frame processing pipeline and measure per-frame telemetry.

        Hooks into roop's native swapping, masking, and face enhancement pipeline for
        the specified frame window while keeping models strictly locked.
        """
        from roop.ProcessMgr import ProcessMgr
        import roop.globals

        resolved_video_path = Path(video_path).expanduser().resolve()
        if not resolved_video_path.is_file():
            raise FileNotFoundError(f"Benchmark video not found at {resolved_video_path}")

        hardware_profile = collect_hardware_profile(
            include_disk_io=False,
            device_index=self.device_index,
        )
        models = dict(active_models) if active_models else self.inspect_active_models()

        workload = BenchmarkWorkload(
            mode=WorkloadMode(workload_dict.get("mode", "solo")),
            target_faces=int(workload_dict.get("target_faces", 1)),
            name=str(workload_dict.get("name", "Calibrated")),
            description=str(workload_dict.get("description", "")),
            expected_frames=int(workload_dict.get("expected_frames", frame_window)),
        )

        LOGGER.info(
            "Beginning pipeline execution on '%s' (workload=%s, models=%s)...",
            resolved_video_path.name,
            workload.name,
            models,
        )

        source_faces = self._prepare_source_faces()
        options = self._prepare_process_options(models, workload)

        process_mgr = ProcessMgr(progress=None)
        process_mgr.is_preview = True
        process_mgr.initialize(source_faces, source_faces, options)

        cap = cv2.VideoCapture(str(resolved_video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open benchmark video: {resolved_video_path}")

        gpu_sampler = GpuTelemetrySampler(period_sec=0.04, device_index=self.device_index)
        gpu_sampler.start()

        psutil_proc = None
        if self._psutil:
            try:
                psutil_proc = self._psutil.Process(os.getpid())
            except Exception:
                pass

        gc.collect()
        if self._torch and self._torch.cuda.is_available():
            try:
                self._torch.cuda.empty_cache()
                self._torch.cuda.reset_peak_memory_stats(self.device_index)
            except Exception:
                pass

        telemetry_records: list[FrameTelemetry] = []
        frame_durations_ms: list[float] = []

        prev_processing = getattr(roop.globals, "processing", False)
        roop.globals.processing = True

        try:
            # Warmup pass (primes TensorRT engines, memory buffers, and CUDA kernels)
            for w_idx in range(max(0, warmup_frames)):
                ret, w_frame = cap.read()
                if not ret or w_frame is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, w_frame = cap.read()
                if ret and w_frame is not None:
                    process_mgr.process_frame(w_frame, frame_idx=w_idx)

            # Standardized measurement loop (exactly frame_window frames)
            for f_idx in range(frame_window):
                ret, frame = cap.read()
                if not ret or frame is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                if self._torch and self._torch.cuda.is_available():
                    self._torch.cuda.synchronize(self.device_index)

                frame_start = time.perf_counter()

                # Native frame swap + mask + enhancement execution
                processed_frame = process_mgr.process_frame(frame, frame_idx=f_idx)

                if self._torch and self._torch.cuda.is_available():
                    self._torch.cuda.synchronize(self.device_index)

                frame_duration_ms = (time.perf_counter() - frame_start) * 1000.0
                frame_durations_ms.append(frame_duration_ms)
                current_fps = 1000.0 / max(1e-6, frame_duration_ms)

                vram_used_mb = 0.0
                if self._torch and self._torch.cuda.is_available():
                    try:
                        free_b, total_b = self._torch.cuda.mem_get_info(self.device_index)
                        vram_used_mb = (float(total_b) - float(free_b)) / (1024.0 * 1024.0)
                    except Exception:
                        pass

                cpu_pct = 0.0
                if psutil_proc:
                    try:
                        cpu_pct = float(psutil_proc.cpu_percent(interval=None))
                    except Exception:
                        pass

                record = FrameTelemetry(
                    frame_index=f_idx,
                    duration_ms=round(frame_duration_ms, 3),
                    fps=round(current_fps, 2),
                    vram_used_mb=round(vram_used_mb, 2),
                    cpu_util_pct=round(cpu_pct, 2),
                )
                telemetry_records.append(record)

                if progress_cb is not None:
                    try:
                        progress_cb(f_idx + 1, frame_window, current_fps, record)
                    except TypeError:
                        progress_cb(f_idx + 1, frame_window, current_fps)

        finally:
            roop.globals.processing = prev_processing
            cap.release()
            gpu_stats = gpu_sampler.finish()
            gc.collect()

        if not frame_durations_ms:
            raise RuntimeError("Benchmark completed with 0 evaluated frames.")

        total_time_sec = sum(frame_durations_ms) / 1000.0
        frames_count = len(frame_durations_ms)
        avg_latency_ms = statistics.fmean(frame_durations_ms)
        avg_fps = frames_count / max(1e-9, total_time_sec)

        p99_latency_ms = _percentile(frame_durations_ms, 99.0)
        p1_low_fps = 1000.0 / max(1e-6, p99_latency_ms)

        peak_vram_mb = max(
            gpu_stats.get("gpu_vram_peak_mb") or 0.0,
            max((f.vram_used_mb for f in telemetry_records), default=0.0),
        )
        peak_cpu_pct = max((f.cpu_util_pct for f in telemetry_records), default=0.0)

        # Thermal / Degradation Check:
        # Compare average FPS of frames 1-30 against frames 120-150.
        # A drop greater than 10% flags thermal throttling or memory leakage.
        if frames_count >= 150:
            first_30 = frame_durations_ms[0:30]
            last_30 = frame_durations_ms[120:150]
        else:
            win = min(30, max(1, frames_count // 2))
            first_30 = frame_durations_ms[:win]
            last_30 = frame_durations_ms[-win:]

        t_first_30 = sum(first_30) / 1000.0
        t_last_30 = sum(last_30) / 1000.0

        fps_first_30 = len(first_30) / max(1e-6, t_first_30)
        fps_last_30 = len(last_30) / max(1e-6, t_last_30)

        drop_pct = ((fps_first_30 - fps_last_30) / max(1e-6, fps_first_30)) * 100.0
        retention_pct = (fps_last_30 / max(1e-6, fps_first_30)) * 100.0
        fps_delta = fps_last_30 - fps_first_30

        frame_time_variance = (
            statistics.variance(frame_durations_ms)
            if frames_count > 1
            else 0.0
        )

        throttling_detected = drop_pct > 10.0

        recommended_threads = min(
            8, int(hardware_profile["cpu"].get("logical_threads", 4))
        )
        recommended_settings = {
            "execution_threads": recommended_threads,
            "execution_provider": (
                "CUDAExecutionProvider"
                if hardware_profile["gpu"].get("available")
                else "CPUExecutionProvider"
            ),
            "temp_format": "png",
            "provider_options": {
                "workload_mode": workload.mode.value,
                "target_faces": workload.target_faces,
            },
        }

        timestamp = datetime.now(timezone.utc).isoformat()

        return BenchmarkRunResult(
            run_id=run_id,
            timestamp=timestamp,
            device_specs=hardware_profile,
            active_models=models,
            workload=workload.to_dict(),
            metrics={
                "avg_fps": round(avg_fps, 2),
                "p1_low_fps": round(p1_low_fps, 2),
                "peak_vram_mb": round(peak_vram_mb, 2),
                "peak_cpu_pct": round(peak_cpu_pct, 2),
                "avg_latency_ms": round(avg_latency_ms, 2),
                "p99_latency_ms": round(p99_latency_ms, 2),
                "total_duration_s": round(total_time_sec, 2),
                "frames_processed": frames_count,
            },
            thermal_stability={
                "fps_first_30": round(fps_first_30, 2),
                "fps_last_30": round(fps_last_30, 2),
                "retention_pct": round(retention_pct, 2),
                "drop_pct": round(drop_pct, 2),
                "fps_delta": round(fps_delta, 2),
                "frame_time_variance": round(frame_time_variance, 4),
                "throttling_detected": throttling_detected,
            },
            recommended_settings=recommended_settings,
            applied=False,
            frame_telemetry=telemetry_records,
            success=True,
            error=None,
            is_oom=False,
        )

    def run_isolated(
        self,
        workload: BenchmarkWorkload | WorkloadMode | str | int | None = None,
        frame_window: int = STANDARDIZED_WINDOW_FRAMES,
        video_path: Path | str | None = None,
        warmup_frames: int = 2,
        persist: bool = True,
        storage_path: str | os.PathLike[str] | None = None,
        progress_cb: Callable[..., None] | None = None,
        override_settings: Optional[dict[str, Any]] = None,
        timeout: float = 300.0,
        raise_on_error: bool = False,
    ) -> BenchmarkRunResult:
        """Execute a benchmark run isolated inside a spawned worker subprocess.

        Uses multiprocessing.get_context("spawn") so that CUDA allocations,
        engine contexts, and parameter overrides remain strictly contained.
        If PyTorch or ONNX raises CUDA OOM or crashes, the worker terminates
        cleanly and notifies the parent runner without terminating the main application.
        """
        resolved_workload = WorkloadSelector.get_workload(workload)
        if video_path is None:
            resolved_video_path = self.asset_manager.ensure_benchmark_clip(
                workload=resolved_workload
            )
        else:
            resolved_video_path = Path(video_path).expanduser().resolve()

        if not resolved_video_path.is_file():
            raise FileNotFoundError(
                f"Benchmark video not found at {resolved_video_path}"
            )

        active_models = self.inspect_active_models()
        hardware_profile = collect_hardware_profile(
            include_disk_io=False,
            device_index=self.device_index,
        )
        run_id = str(uuid.uuid4())

        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()

        worker_args = {
            "run_id": run_id,
            "workload_dict": resolved_workload.to_dict(),
            "frame_window": frame_window,
            "video_path": str(resolved_video_path),
            "warmup_frames": warmup_frames,
            "override_settings": override_settings or {},
            "device_index": self.device_index,
            "active_models": active_models,
        }

        proc = ctx.Process(
            target=_subprocess_worker_entry,
            args=(worker_args, queue),
            name=f"benchmark_worker_{run_id[:8]}",
        )

        LOGGER.info(
            "Spawning isolated benchmark worker process for run %s (workload=%s, window=%d)...",
            run_id[:8],
            resolved_workload.name,
            frame_window,
        )

        proc.start()

        result_payload = None
        error_info = None
        deadline = time.monotonic() + max(10.0, timeout)

        while proc.is_alive() or not queue.empty():
            try:
                msg = queue.get(timeout=0.05)
                msg_type = msg[0]
                if msg_type == "progress":
                    _, cur, tot, cur_fps, rec_dict = msg
                    rec = FrameTelemetry(**rec_dict) if rec_dict else None
                    if progress_cb:
                        try:
                            progress_cb(cur, tot, cur_fps, rec)
                        except TypeError:
                            progress_cb(cur, tot, cur_fps)
                elif msg_type == "result":
                    result_payload = msg[1]
                elif msg_type == "error":
                    error_info = msg[1:]
            except queue_module.Empty:
                if time.monotonic() > deadline:
                    LOGGER.warning("Worker process timed out after %s seconds; killing.", timeout)
                    proc.terminate()
                    time.sleep(0.5)
                    if proc.is_alive():
                        proc.kill()
                    error_info = (
                        "TimeoutError",
                        f"Benchmark worker timed out after {timeout} seconds.",
                        "",
                        False,
                    )
                    break

        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2.0)

        # If process exited abnormally without sending error or result (e.g. segfault / crash)
        if proc.exitcode != 0 and result_payload is None and error_info is None:
            error_info = (
                "WorkerCrashError",
                f"Worker subprocess terminated unexpectedly with exit code {proc.exitcode} (possible segmentation fault or CUDA abort).",
                "",
                False,
            )

        if result_payload is not None:
            result = BenchmarkRunResult.from_dict(result_payload)
        else:
            err_type, err_msg, tb, is_oom = error_info if error_info else (
                "UnknownError", "No result returned from worker process", "", False
            )
            result = BenchmarkRunResult(
                run_id=run_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                device_specs=hardware_profile,
                active_models=active_models,
                workload=resolved_workload.to_dict(),
                metrics={
                    "avg_fps": 0.0,
                    "p1_low_fps": 0.0,
                    "peak_vram_mb": 0.0,
                    "peak_cpu_pct": 0.0,
                    "avg_latency_ms": 0.0,
                    "p99_latency_ms": 0.0,
                    "total_duration_s": 0.0,
                    "frames_processed": 0,
                },
                thermal_stability={
                    "fps_first_30": 0.0,
                    "fps_last_30": 0.0,
                    "retention_pct": 0.0,
                    "drop_pct": 0.0,
                    "fps_delta": 0.0,
                    "frame_time_variance": 0.0,
                    "throttling_detected": False,
                },
                recommended_settings={},
                applied=False,
                frame_telemetry=[],
                success=False,
                error=f"{err_type}: {err_msg}",
                is_oom=is_oom,
            )

        if persist and result.success:
            try:
                result.save(storage_path)
                LOGGER.info("Persisted benchmark run %s to storage.", run_id)
            except Exception as exc:
                LOGGER.warning("Could not persist benchmark run %s: %s", run_id, exc)

        if raise_on_error and not result.success:
            raise BenchmarkWorkerError(
                result.error or "Unknown worker error",
                error_type=error_info[0] if error_info else "ExecutionError",
                is_oom=result.is_oom,
                exitcode=proc.exitcode if proc.exitcode is not None else 1,
                traceback_str=error_info[2] if error_info else "",
            )

        print(result.summary_text())
        return result

    def run(
        self,
        workload: BenchmarkWorkload | WorkloadMode | str | int | None = None,
        frame_window: int = STANDARDIZED_WINDOW_FRAMES,
        video_path: Path | str | None = None,
        warmup_frames: int = 2,
        persist: bool = True,
        storage_path: str | os.PathLike[str] | None = None,
        progress_cb: Callable[..., None] | None = None,
        isolated: bool = False,
        override_settings: Optional[dict[str, Any]] = None,
        timeout: float = 300.0,
        raise_on_error: bool = False,
    ) -> BenchmarkRunResult:
        """Execute the standardized real-video benchmark on the native pipeline.

        Args:
            workload: Target face scenario (Solo, Duo, Group, or Master).
            frame_window: Number of frames to evaluate (default: 150).
            video_path: Explicit video path, or None to resolve from AssetManager.
            warmup_frames: Initial frames processed before measurement starts.
            persist: Whether to atomically save the result to benchmark history.
            storage_path: Optional custom destination for the history JSON file.
            progress_cb: Optional progress callback ``(current_frame, total_frames, current_fps)``.
            isolated: Whether to run in a spawned subprocess worker (True) or in-process (False).
            override_settings: Optional dictionary of runtime settings to test in isolation.
            timeout: Subprocess execution timeout in seconds.
            raise_on_error: Whether to raise BenchmarkWorkerError on failure.

        Returns:
            A BenchmarkRunResult with comprehensive metrics and stability telemetry.
        """
        if isolated:
            return self.run_isolated(
                workload=workload,
                frame_window=frame_window,
                video_path=video_path,
                warmup_frames=warmup_frames,
                persist=persist,
                storage_path=storage_path,
                progress_cb=progress_cb,
                override_settings=override_settings,
                timeout=timeout,
                raise_on_error=raise_on_error,
            )

        resolved_workload = WorkloadSelector.get_workload(workload)
        if video_path is None:
            resolved_video_path = self.asset_manager.ensure_benchmark_clip(
                workload=resolved_workload
            )
        else:
            resolved_video_path = Path(video_path).expanduser().resolve()

        run_id = str(uuid.uuid4())
        active_models = self.inspect_active_models()

        result = self._execute_frame_pipeline(
            run_id=run_id,
            workload_dict=resolved_workload.to_dict(),
            frame_window=frame_window,
            video_path=resolved_video_path,
            warmup_frames=warmup_frames,
            progress_cb=progress_cb,
            active_models=active_models,
        )

        if persist and result.success:
            try:
                result.save(storage_path)
                LOGGER.info("Persisted benchmark run %s to storage.", run_id)
            except Exception as exc:
                LOGGER.warning("Could not persist benchmark run %s: %s", run_id, exc)

        print(result.summary_text())
        return result

    def run_baseline_pass(
        self,
        workload: BenchmarkWorkload | WorkloadMode | str | int | None = "solo",
        frame_window: int = 90,
        video_path: Path | str | None = None,
        warmup_frames: int = 2,
        persist: bool = False,
        progress_cb: Callable[..., None] | None = None,
        isolated: bool = True,
    ) -> BenchmarkRunResult:
        """Execute Pass 0: Mandatory 90-frame baseline pass using current active settings.

        Records existing FPS and VRAM footprint prior to testing parameter combinations.
        """
        LOGGER.info("Executing Pass 0: Mandatory 90-frame baseline pass...")
        return self.run(
            workload=workload,
            frame_window=frame_window,
            video_path=video_path,
            warmup_frames=warmup_frames,
            persist=persist,
            progress_cb=progress_cb,
            isolated=isolated,
            override_settings=None,
        )

    def run_parameter_variation(
        self,
        override_settings: dict[str, Any],
        workload: BenchmarkWorkload | WorkloadMode | str | int | None = "solo",
        frame_window: int = STANDARDIZED_WINDOW_FRAMES,
        video_path: Path | str | None = None,
        warmup_frames: int = 2,
        persist: bool = False,
        progress_cb: Callable[..., None] | None = None,
        timeout: float = 300.0,
        raise_on_error: bool = False,
    ) -> BenchmarkRunResult:
        """Execute a parameter variation in an isolated subprocess worker.

        NEVER executes parameter test variations in the main application process.
        Uses multiprocessing.get_context("spawn") to isolate each test iteration,
        preventing CUDA OOM crashes or memory corruption from affecting the main application.
        """
        return self.run(
            workload=workload,
            frame_window=frame_window,
            video_path=video_path,
            warmup_frames=warmup_frames,
            persist=persist,
            progress_cb=progress_cb,
            isolated=True,
            override_settings=override_settings,
            timeout=timeout,
            raise_on_error=raise_on_error,
        )


# Module-level convenience runner function
def run_benchmark(
    workload: BenchmarkWorkload | WorkloadMode | str | int | None = "solo",
    frame_window: int = STANDARDIZED_WINDOW_FRAMES,
    persist: bool = True,
    progress_cb: Callable[..., None] | None = None,
) -> BenchmarkRunResult:
    """Execute a benchmark run with default parameters."""
    runner = BenchmarkRunner()
    return runner.run(
        workload=workload,
        frame_window=frame_window,
        persist=persist,
        progress_cb=progress_cb,
    )


__all__ = [
    "BenchmarkRunResult",
    "BenchmarkRunner",
    "BenchmarkWorkerError",
    "FrameTelemetry",
    "GpuTelemetrySampler",
    "run_benchmark",
]
