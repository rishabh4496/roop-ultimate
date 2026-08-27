"""Centralized, conservative runtime optimization infrastructure.

This module is intentionally an optimizer *policy* layer, not a second model
runtime.  TensorRT, ONNX Runtime, the frame pipeline, and the existing session
pool remain the owners of their resources.  The optimizer profiles the machine
and workload, derives bounded recommendations, and applies only settings that
are still automatic.  It never builds an engine, changes precision, or widens
a pool as a side effect of importing the module.

The separation is useful for three reasons:

* hardware and workload facts can be tested without importing the UI or models;
* future benchmark results can be stored against a complete runtime identity;
* explicit settings remain authoritative while ``auto`` values get a single,
  explainable decision instead of scattered VRAM-only rules.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


SCHEMA_VERSION = 1


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _value(settings: Any, name: str, default: Any = None) -> Any:
    if settings is None:
        return default
    if isinstance(settings, Mapping):
        return settings.get(name, default)
    return getattr(settings, name, default)


def _is_auto(settings: Any, name: str) -> bool:
    """Whether a setting is eligible for an automatic recommendation."""
    if name == "max_threads":
        if _value(settings, "auto_thread_selection", True) is False:
            return False
        return bool(_value(settings, "_threads_auto", True))
    value = _value(settings, name, "auto")
    return value is None or str(value).strip().lower() in ("", "auto", "default")


def _short(value: Any) -> str:
    return str(value or "unknown").strip()


@dataclass(frozen=True)
class HardwareProfile:
    gpu_name: str = ""
    gpu_vendor: str = "unknown"
    compute_capability: str = ""
    vram_total_gb: float = 0.0
    vram_available_gb: float = 0.0
    cuda_available: bool = False
    cuda_version: str = ""
    driver_version: str = ""
    tensorrt_available: bool = False
    tensorrt_version: str = ""
    onnxruntime_version: str = ""
    nvdec_available: bool = False
    nvenc_available: bool = False
    cpu_physical_cores: int = 1
    cpu_logical_cores: int = 1
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    platform: str = ""

    @property
    def vram_tier(self) -> str:
        if self.vram_total_gb < 7.0:
            return "small"
        if self.vram_total_gb < 11.5:
            return "medium"
        if self.vram_total_gb < 15.5:
            return "desktop"
        return "large"

    def as_dict(self) -> dict:
        result = asdict(self)
        result["vram_tier"] = self.vram_tier
        return result


@dataclass(frozen=True)
class WorkloadProfile:
    input_width: int = 0
    input_height: int = 0
    output_width: int = 0
    output_height: int = 0
    faces_per_frame: float = 1.0
    face_count: int = 0
    enhancement_enabled: bool = False
    enhancement_model: str = ""
    stabilization_enabled: bool = False
    upscaling_enabled: bool = False
    temporal_detection_enabled: bool = False
    video_length_frames: int = 0
    fps: float = 0.0
    estimated_complexity: float = 0.0

    @property
    def pixels_per_frame(self) -> int:
        return max(0, self.input_width * self.input_height)

    @property
    def resolution_class(self) -> str:
        pixels = self.pixels_per_frame
        if pixels <= 0:
            return "unknown"
        if pixels <= 1280 * 720:
            return "720p_or_less"
        if pixels <= 1920 * 1080:
            return "1080p"
        return "above_1080p"

    def as_dict(self) -> dict:
        result = asdict(self)
        result["pixels_per_frame"] = self.pixels_per_frame
        result["resolution_class"] = self.resolution_class
        return result


@dataclass(frozen=True)
class RuntimeTuning:
    """All derived knobs, with safe bounds applied before exposure."""

    trt_context_count: int = 1
    worker_count: int = 1
    detector_pool_size: int = 0
    detmask_pool_size: int = 0
    swapper_pool_size: int = 0
    enhancer_pool_size: int = 0
    expression_pool_size: int = 0
    batch_size: int = 1
    tile_batch_size: int = 1
    detector_resolution: int = 640
    queue_depth: int = 1
    stabilization_workers: int = 1
    stabilization_chunk_size: int = 64
    ort_intra_threads: int = 1
    ort_inter_threads: int = 1
    opencv_threads: int = 1
    encoder: str = "libx264"
    encoder_preset: str = "medium"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeProfile:
    schema_version: int
    created_at: float
    hardware: HardwareProfile
    workload: WorkloadProfile
    tuning: RuntimeTuning
    precision: str
    provider: str
    explicit_settings: Tuple[str, ...] = field(default_factory=tuple)
    automatic_settings: Tuple[str, ...] = field(default_factory=tuple)
    reasons: Tuple[str, ...] = field(default_factory=tuple)
    cache_key: str = ""

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "hardware": self.hardware.as_dict(),
            "workload": self.workload.as_dict(),
            "tuning": self.tuning.as_dict(),
            "precision": self.precision,
            "provider": self.provider,
            "explicit_settings": list(self.explicit_settings),
            "automatic_settings": list(self.automatic_settings),
            "reasons": list(self.reasons),
            "cache_key": self.cache_key,
        }


class ResourceManager:
    """Central safe-bound and memory-budget policy.

    The bounds are intentionally modest.  They prevent a future tuner or stale
    profile from turning a 6GB laptop into a context/queue thrashing workload.
    They are not claims that the upper bound is optimal.
    """

    BOUNDS = {
        "trt_context_count": (1, 4),
        "worker_count": (1, 16),
        "detector_pool_size": (0, 4),
        "detmask_pool_size": (0, 4),
        "swapper_pool_size": (0, 4),
        "enhancer_pool_size": (0, 3),
        "expression_pool_size": (0, 3),
        "batch_size": (1, 4),
        "tile_batch_size": (1, 4),
        "detector_resolution": (320, 1280),
        "queue_depth": (1, 4),
        "stabilization_workers": (1, 8),
        "stabilization_chunk_size": (32, 288),
        "ort_intra_threads": (1, 4),
        "ort_inter_threads": (1, 2),
        "opencv_threads": (1, 4),
    }

    @classmethod
    def clamp(cls, name: str, value: Any) -> int:
        lo, hi = cls.BOUNDS[name]
        return max(lo, min(hi, _integer(value, lo)))

    @staticmethod
    def frame_budget_mb(hardware: HardwareProfile, workload: WorkloadProfile) -> int:
        """Return a bounded frame-buffer budget, not a model-memory budget."""
        free_ram = hardware.ram_available_gb or hardware.ram_total_gb * 0.25
        free_vram = hardware.vram_available_gb or hardware.vram_total_gb
        if hardware.vram_total_gb < 7.0:
            cap = 1536
        elif hardware.vram_total_gb < 11.5:
            cap = 2048
        else:
            cap = 4096
        if free_ram < 4.0:
            cap = min(cap, 1536)
        if workload.resolution_class == "above_1080p":
            cap = min(cap, 2048)
        if free_vram and free_vram < 1.5:
            cap = min(cap, 1024)
        return max(512, cap)


class HardwareProfiler:
    """Detect hardware using cheap, optional probes only."""

    def __init__(self, device_id: int = 0):
        self.device_id = max(0, _integer(device_id, 0))
        self._profile: Optional[HardwareProfile] = None

    @staticmethod
    def _vendor(name: str) -> str:
        lower = name.lower()
        if "nvidia" in lower or "geforce" in lower or "quadro" in lower:
            return "nvidia"
        if "amd" in lower or "radeon" in lower:
            return "amd"
        if "intel" in lower or "arc" in lower:
            return "intel"
        return "unknown"

    @staticmethod
    def _command(*args: str, timeout: float = 1.5) -> str:
        try:
            result = subprocess.run(args, capture_output=True, text=True,
                                    timeout=timeout, check=False)
            return (result.stdout or "") + (result.stderr or "")
        except (OSError, subprocess.SubprocessError):
            return ""

    def profile(self, refresh: bool = False) -> HardwareProfile:
        if self._profile is not None and not refresh:
            return self._profile

        gpu_name = ""
        compute = ""
        vram_total = vram_free = 0.0
        cuda = False
        cuda_version = ""
        trt = False
        trt_version = ""
        ort_version = ""
        try:
            import torch
            cuda = bool(torch.cuda.is_available())
            cuda_version = str(getattr(torch.version, "cuda", "") or "")
            if cuda and self.device_id < torch.cuda.device_count():
                gpu_name = str(torch.cuda.get_device_name(self.device_id))
                props = torch.cuda.get_device_properties(self.device_id)
                vram_total = _number(props.total_memory) / (1024 ** 3)
                major, minor = torch.cuda.get_device_capability(self.device_id)
                compute = f"{major}.{minor}"
                try:
                    free, total = torch.cuda.mem_get_info(self.device_id)
                    vram_free = _number(free) / (1024 ** 3)
                    vram_total = _number(total) / (1024 ** 3) or vram_total
                except Exception:
                    pass
        except Exception:
            pass

        try:
            import onnxruntime as ort
            ort_version = str(getattr(ort, "__version__", ""))
            available = set(ort.get_available_providers())
            trt = "TensorrtExecutionProvider" in available and cuda
        except Exception:
            available = set()

        try:
            import tensorrt as _trt
            trt = trt or bool(cuda)
            trt_version = str(getattr(_trt, "__version__", ""))
        except Exception:
            pass

        try:
            import psutil
            physical = psutil.cpu_count(logical=False) or 1
            logical = psutil.cpu_count(logical=True) or physical
            memory = psutil.virtual_memory()
            ram_total = _number(memory.total) / (1024 ** 3)
            ram_available = _number(memory.available) / (1024 ** 3)
        except Exception:
            physical = max(1, (os.cpu_count() or 1) // 2)
            logical = max(1, os.cpu_count() or physical)
            ram_total = ram_available = 0.0

        nvidia_smi = self._command(
            "nvidia-smi", f"--id={self.device_id}",
            "--query-gpu=driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits")
        driver = ""
        if nvidia_smi:
            row = next((line.strip() for line in nvidia_smi.splitlines()
                        if line.strip() and not line.lower().startswith("failed")), "")
            parts = [part.strip() for part in row.split(",")]
            if parts:
                driver = parts[0]
            if len(parts) >= 3:
                vram_total = _number(parts[1], vram_total) / (1024 if _number(parts[1]) > 100 else 1)
                vram_free = _number(parts[2], vram_free) / (1024 if _number(parts[2]) > 100 else 1)

        ffmpeg = shutil.which("ffmpeg")
        hwaccels = self._command(ffmpeg, "-hide_banner", "-hwaccels") if ffmpeg else ""
        encoders = self._command(ffmpeg, "-hide_banner", "-encoders") if ffmpeg else ""
        nvdec = "cuda" in hwaccels.lower() and cuda
        nvenc = ("h264_nvenc" in encoders.lower() or "hevc_nvenc" in encoders.lower()) and cuda

        self._profile = HardwareProfile(
            gpu_name=gpu_name,
            gpu_vendor=self._vendor(gpu_name),
            compute_capability=compute,
            vram_total_gb=round(vram_total, 3),
            vram_available_gb=round(vram_free, 3),
            cuda_available=cuda,
            cuda_version=cuda_version,
            driver_version=driver,
            tensorrt_available=trt,
            tensorrt_version=trt_version,
            onnxruntime_version=ort_version,
            nvdec_available=nvdec,
            nvenc_available=nvenc,
            cpu_physical_cores=max(1, physical),
            cpu_logical_cores=max(1, logical),
            ram_total_gb=round(ram_total, 3),
            ram_available_gb=round(ram_available, 3),
            platform=f"{platform.system()}-{platform.release()}",
        )
        return self._profile


class WorkloadProfiler:
    """Collect workload facts without running a detector or model."""

    def profile(self, source_video: Optional[str] = None,
                settings: Any = None, frame_count: int = 0,
                resolution: Optional[Tuple[int, int]] = None,
                output_resolution: Optional[Tuple[int, int]] = None,
                faces_per_frame: Optional[float] = None,
                face_count: int = 0) -> WorkloadProfile:
        width = height = fps = 0.0
        if resolution:
            width, height = resolution
        if source_video and (not width or not height):
            try:
                import cv2
                cap = cv2.VideoCapture(source_video)
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if not frame_count:
                    frame_count = _integer(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            except Exception:
                pass

        out_w, out_h = output_resolution or (width, height)
        enhancer = _short(_value(settings, "selected_enhancer", ""))
        enhancement = bool(enhancer and enhancer.lower() not in ("none", "keep"))
        stabilization = _bool(_value(settings, "stabilize_face", False))
        upscale = _bool(_value(settings, "upscale_after_swap", False))
        temporal = _bool(_value(settings, "temporal_detection", False))
        fpf = max(0.0, _number(faces_per_frame, 1.0))
        pixels = max(0.0, width * height)
        resolution_factor = pixels / (1280.0 * 720.0) if pixels else 1.0
        complexity = 1.0
        complexity += min(3.0, max(0.0, fpf - 1.0) * 0.65)
        complexity += min(1.5, resolution_factor * 0.25)
        complexity += 0.8 if enhancement else 0.0
        complexity += 0.35 if stabilization else 0.0
        complexity += 0.35 if upscale else 0.0
        complexity += 0.35 if temporal else 0.0

        return WorkloadProfile(
            input_width=max(0, _integer(width)),
            input_height=max(0, _integer(height)),
            output_width=max(0, _integer(out_w)),
            output_height=max(0, _integer(out_h)),
            faces_per_frame=round(fpf, 3),
            face_count=max(0, _integer(face_count)),
            enhancement_enabled=enhancement,
            enhancement_model=enhancer,
            stabilization_enabled=stabilization,
            upscaling_enabled=upscale,
            temporal_detection_enabled=temporal,
            video_length_frames=max(0, _integer(frame_count)),
            fps=max(0.0, _number(fps)),
            estimated_complexity=round(complexity, 3),
        )


class PrecisionSelector:
    """Preserve configured precision; automatic mode stays quality-safe."""

    def select(self, settings: Any, hardware: HardwareProfile) -> str:
        configured = str(_value(settings, "trt_precision", "mixed") or "mixed").lower()
        if configured in ("fp16", "fp32", "mixed"):
            return configured
        if hardware.tensorrt_available:
            return "mixed"
        return "fp32"


class TensorRTEngineManager:
    """Engine identity and cache policy; engine construction remains ORT-owned."""

    def cache_key(self, hardware: HardwareProfile, workload: WorkloadProfile,
                  settings: Any, precision: str) -> str:
        identity = {
            "gpu": hardware.gpu_name,
            "compute": hardware.compute_capability,
            "driver": hardware.driver_version,
            "cuda": hardware.cuda_version,
            "tensorrt": hardware.tensorrt_version,
            "ort": hardware.onnxruntime_version,
            "model": {
                "swap": _value(settings, "swap_model", "") or "",
                "detector": _value(settings, "detector_engine", "") or "",
                "mask": _value(settings, "mask_engine", "") or "",
            },
            "enhancer": workload.enhancement_model,
            "precision": precision,
            "input": [workload.input_width, workload.input_height],
            "output": [workload.output_width, workload.output_height],
            "faces": round(workload.faces_per_frame, 2),
            "stabilization": workload.stabilization_enabled,
            "upscaling": workload.upscaling_enabled,
        }
        raw = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()[:24]

    def cache_path(self, profile_dir: Optional[str], cache_key: str) -> Path:
        root = Path(profile_dir or os.environ.get(
            "ROOP_RUNTIME_PROFILE_DIR", "models/runtime_profiles"))
        return root / f"{cache_key}.json"


class CUDAGraphManager:
    """CUDA graph readiness gate; capture is deliberately not performed here."""

    def readiness(self, settings: Any, workload: WorkloadProfile,
                  hardware: HardwareProfile) -> dict:
        requested = _bool(_value(settings, "trt_cuda_graph", False))
        stable_shapes = workload.input_width > 0 and workload.input_height > 0
        safe = requested and hardware.cuda_available and stable_shapes
        reason = "enabled by user" if safe else (
            "disabled by default; stable-shape capture is not yet wired" if not requested
            else "requested but runtime capture is not implemented")
        return {"requested": requested, "safe": safe, "reason": reason}


class AutoTuner:
    """Derive a conservative profile from hardware *and* workload facts."""

    def tune(self, hardware: HardwareProfile, workload: WorkloadProfile,
              settings: Any = None) -> Tuple[RuntimeTuning, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
        small = hardware.vram_total_gb > 0 and hardware.vram_total_gb < 7.0
        gpu_cap = 8 if small else 10
        cpu_cap = max(1, min(hardware.cpu_logical_cores, gpu_cap))
        if workload.estimated_complexity >= 3.5:
            cpu_cap = max(1, min(cpu_cap, 8 if not small else 6))
        worker = max(1, min(cpu_cap, hardware.cpu_physical_cores or cpu_cap))
        contexts = 1 if small else 2
        detector_pool = 0 if small else 2
        detmask_pool = 0 if small else 2
        swapper_pool = 0 if small else 2
        enhancer_pool = 0 if small else (2 if workload.enhancement_enabled else 1)
        expression_pool = 0 if small else 2
        stabilization_workers = max(1, min(worker, 4 if small else 6))
        if not workload.stabilization_enabled:
            stabilization_workers = 1

        pixels = workload.pixels_per_frame or 1280 * 720
        if pixels > 1920 * 1080 or small:
            queue_depth = 1
        elif pixels > 1280 * 720:
            queue_depth = 2
        else:
            queue_depth = 3
        chunk = 64 if small else (96 if pixels > 1920 * 1080 else 144)
        opencv = 1 if worker >= 8 or small else 2
        encoder = "hevc_nvenc" if hardware.nvenc_available else "libx264"
        preset = "p5" if hardware.nvenc_available else "medium"
        values = {
            "trt_context_count": contexts,
            "worker_count": worker,
            "detector_pool_size": detector_pool,
            "detmask_pool_size": detmask_pool,
            "swapper_pool_size": swapper_pool,
            "enhancer_pool_size": enhancer_pool,
            "expression_pool_size": expression_pool,
            "batch_size": 2 if (not small and workload.faces_per_frame >= 2) else 1,
            "tile_batch_size": 2 if (not small and workload.faces_per_frame >= 2) else 1,
            "detector_resolution": 640,
            "queue_depth": queue_depth,
            "stabilization_workers": stabilization_workers,
            "stabilization_chunk_size": chunk,
            "ort_intra_threads": 1,
            "ort_inter_threads": 1,
            "opencv_threads": opencv,
            "encoder": encoder,
            "encoder_preset": preset,
        }

        explicit = []
        automatic = []
        reasons = ["bounded hardware/workload policy; no benchmark result was applied"]
        fields = {
            "worker_count": "max_threads",
            "detector_pool_size": "perf_detector_pool",
            "detmask_pool_size": "perf_detmask_pool",
            "swapper_pool_size": "perf_trt_pool",
            "expression_pool_size": "perf_expr_pool",
            "batch_size": "perf_batch_swap",
            "tile_batch_size": "perf_batch_swap",
            "detector_resolution": "face_detector_size",
            "opencv_threads": "cpu_opencv_threads",
            "encoder_preset": "perf_encoder_preset",
            "encoder": "output_video_codec",
        }
        for field_name, setting_name in fields.items():
            if _is_auto(settings, setting_name):
                automatic.append(setting_name)
            else:
                explicit.append(setting_name)
                reasons.append(f"{setting_name} is explicit and remains authoritative")

        # Make the tuning object an effective profile, not merely a default
        # recommendation.  Values supplied by the user are parsed and bounded
        # at this single boundary; the rest of the application can consume the
        # result without accidentally replacing a pin with an automatic value.
        def _explicit_int(setting_name: str, current: int) -> int:
            if _is_auto(settings, setting_name):
                return current
            raw = _value(settings, setting_name, current)
            if str(raw).strip().lower() in ("off", "none"):
                raw = 0
            return ResourceManager.clamp(
                next(name for name, setting in fields.items() if setting == setting_name),
                raw)

        values["worker_count"] = _explicit_int("max_threads", values["worker_count"])
        values["detector_pool_size"] = _explicit_int("perf_detector_pool", values["detector_pool_size"])
        values["detmask_pool_size"] = _explicit_int("perf_detmask_pool", values["detmask_pool_size"])
        values["swapper_pool_size"] = _explicit_int("perf_trt_pool", values["swapper_pool_size"])
        # Enhancers share the established TRT pool.  The expression restorer
        # has its own pool and is represented separately above.
        values["enhancer_pool_size"] = values["swapper_pool_size"]
        values["expression_pool_size"] = _explicit_int("perf_expr_pool", values["expression_pool_size"])
        values["detector_resolution"] = _explicit_int("face_detector_size", values["detector_resolution"])
        if not _is_auto(settings, "perf_batch_swap"):
            batch_setting = str(_value(settings, "perf_batch_swap", "on")).strip().lower()
            if batch_setting in ("off", "0", "false", "no"):
                values["batch_size"] = values["tile_batch_size"] = 1
        if not _is_auto(settings, "cpu_opencv_threads"):
            values["opencv_threads"] = _explicit_int("cpu_opencv_threads", values["opencv_threads"])
        if not _is_auto(settings, "perf_encoder_preset"):
            values["encoder_preset"] = _short(_value(settings, "perf_encoder_preset", values["encoder_preset"]))
        if not _is_auto(settings, "output_video_codec"):
            values["encoder"] = _short(_value(settings, "output_video_codec", values["encoder"]))

        for name in ResourceManager.BOUNDS:
            values[name] = ResourceManager.clamp(name, values[name])
        tuning = RuntimeTuning(**values)
        return tuning, tuple(sorted(set(explicit))), tuple(sorted(set(automatic))), tuple(reasons)


class ProfileStore:
    """Small atomic JSON store for future measured profiles."""

    def __init__(self, directory: Optional[str] = None):
        self.directory = Path(directory or os.environ.get(
            "ROOP_RUNTIME_PROFILE_DIR", "models/runtime_profiles"))
        self._lock = threading.Lock()

    def load(self, key: str) -> Optional[dict]:
        path = self.directory / f"{key}.json"
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if data.get("schema_version") == SCHEMA_VERSION else None
        except (OSError, ValueError, TypeError):
            return None

    def save(self, profile: RuntimeProfile) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f"{profile.cache_key}.json"
        with self._lock:
            fd, temporary = tempfile.mkstemp(prefix=".runtime-profile-",
                                              suffix=".tmp", dir=str(self.directory))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(profile.as_dict(), handle, indent=2, sort_keys=True)
                    handle.write("\n")
                os.replace(temporary, destination)
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
        return destination


class RuntimeMonitor:
    """Low-overhead run telemetry, ready for integration with stage counters."""

    def __init__(self):
        self.started_at = 0.0
        self.samples = []

    def start(self) -> None:
        self.started_at = time.perf_counter()

    def observe(self, **metrics: Any) -> None:
        sample = {"time": time.time()}
        sample.update(metrics)
        self.samples.append(sample)
        if len(self.samples) > 256:
            del self.samples[:-256]

    def summary(self) -> dict:
        elapsed = time.perf_counter() - self.started_at if self.started_at else 0.0
        return {"elapsed_sec": round(elapsed, 3), "samples": list(self.samples)}


class RuntimeOptimizer:
    """Facade used by startup and future workload-aware pipeline hooks."""

    def __init__(self, settings: Any = None, device_id: int = 0,
                 profile_dir: Optional[str] = None):
        self.settings = settings
        self.hardware_profiler = HardwareProfiler(device_id)
        self.workload_profiler = WorkloadProfiler()
        self.precision_selector = PrecisionSelector()
        self.engine_manager = TensorRTEngineManager()
        self.cuda_graphs = CUDAGraphManager()
        self.auto_tuner = AutoTuner()
        self.store = ProfileStore(profile_dir)
        self.monitor = RuntimeMonitor()

    def build_profile(self, workload: WorkloadProfile,
                      save: bool = False) -> RuntimeProfile:
        hardware = self.hardware_profiler.profile()
        precision = self.precision_selector.select(self.settings, hardware)
        tuning, explicit, automatic, reasons = self.auto_tuner.tune(
            hardware, workload, self.settings)
        key = self.engine_manager.cache_key(hardware, workload, self.settings, precision)
        profile = RuntimeProfile(
            schema_version=SCHEMA_VERSION,
            created_at=time.time(),
            hardware=hardware,
            workload=workload,
            tuning=tuning,
            precision=precision,
            provider=_short(_value(self.settings, "provider", "cuda")),
            explicit_settings=explicit,
            automatic_settings=automatic,
            reasons=reasons,
            cache_key=key,
        )
        if save:
            self.store.save(profile)
        return profile

    def startup_profile(self) -> RuntimeProfile:
        workload = self.workload_profiler.profile(settings=self.settings)
        return self.build_profile(workload, save=False)

    def profile_video(self, source_video: str, frame_count: int = 0,
                      resolution: Optional[Tuple[int, int]] = None,
                      faces_per_frame: Optional[float] = None,
                      output_resolution: Optional[Tuple[int, int]] = None,
                      save: bool = True) -> RuntimeProfile:
        workload = self.workload_profiler.profile(
            source_video=source_video, settings=self.settings,
            frame_count=frame_count, resolution=resolution,
            output_resolution=output_resolution,
            faces_per_frame=faces_per_frame)
        return self.build_profile(workload, save=save)

    @staticmethod
    def apply_environment(profile: RuntimeProfile, settings: Any = None) -> dict:
        """Apply only non-import-time, safe runtime hints for automatic fields.

        Existing explicit environment variables are never replaced.  The
        current pipeline may consume these hints incrementally; unused hints
        are harmless and make the profile visible to future stages.
        """
        tuning = profile.tuning
        env_values = {
            "ROOP_RUNTIME_QUEUE_DEPTH": tuning.queue_depth,
            "ROOP_RUNTIME_STABILIZATION_WORKERS": tuning.stabilization_workers,
            "ROOP_RUNTIME_STAB_CHUNK": tuning.stabilization_chunk_size,
            "ROOP_RUNTIME_ORT_INTRA_THREADS": tuning.ort_intra_threads,
            "ROOP_RUNTIME_ORT_INTER_THREADS": tuning.ort_inter_threads,
            "ROOP_RUNTIME_CV_THREADS": tuning.opencv_threads,
            "ROOP_RUNTIME_BATCH_SIZE": tuning.batch_size,
            "ROOP_RUNTIME_TILE_BATCH_SIZE": tuning.tile_batch_size,
        }
        # These values are hints for the staged rollout.  Do not expose a
        # derived value where an existing user-facing setting already pins the
        # same concern; a later consumer must be able to distinguish "auto"
        # from "the user chose this".
        setting_for_hint = {
            "ROOP_RUNTIME_CV_THREADS": "cpu_opencv_threads",
            "ROOP_RUNTIME_BATCH_SIZE": "perf_batch_swap",
            "ROOP_RUNTIME_TILE_BATCH_SIZE": "perf_batch_swap",
        }
        applied = {}
        for name, value in env_values.items():
            setting_name = setting_for_hint.get(name)
            if setting_name and not _is_auto(settings, setting_name):
                continue
            if name not in os.environ:
                os.environ[name] = str(value)
                applied[name] = value
        os.environ["ROOP_RUNTIME_PROFILE_KEY"] = profile.cache_key
        os.environ["ROOP_RUNTIME_PROFILE_JSON"] = json.dumps(profile.as_dict(), separators=(",", ":"))
        os.environ["ROOP_RUNTIME_PROFILE_ORIGIN"] = "optimizer"
        return applied


__all__ = [
    "AutoTuner", "CUDAGraphManager", "HardwareProfile", "HardwareProfiler",
    "PrecisionSelector", "ProfileStore", "ResourceManager", "RuntimeMonitor",
    "RuntimeOptimizer", "RuntimeProfile", "RuntimeTuning", "TensorRTEngineManager",
    "WorkloadProfile", "WorkloadProfiler",
]
