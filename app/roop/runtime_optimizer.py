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
import re
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


def _file_digest(path: Any) -> str:
    """Return a cheap content identity when a caller supplies a model path."""
    try:
        candidate = Path(str(path)).expanduser()
        if not candidate.is_file():
            return ""
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, TypeError, ValueError):
        return ""


def _model_identity(settings: Any) -> dict:
    """Collect explicit model revision data without guessing model locations."""
    fields = (
        "model_version", "model_hash", "swap_model_hash",
        "detector_model_hash", "enhancer_model_hash", "mask_model_hash",
    )
    identity = {name: _value(settings, name, "") or "" for name in fields}
    identity["environment_version"] = os.environ.get("ROOP_MODEL_VERSION", "")
    identity["environment_hash"] = os.environ.get("ROOP_MODEL_HASH", "")
    for setting_name in (
            "model_path", "swap_model_path", "detector_model_path",
            "enhancer_model_path", "mask_model_path"):
        path = _value(settings, setting_name, "")
        if path:
            identity[setting_name] = str(path)
            digest = _file_digest(path)
            if digest:
                identity[setting_name + "_sha256"] = digest
    return identity


@dataclass(frozen=True)
class HardwareProfile:
    device_id: int = 0
    gpu_name: str = ""
    gpu_vendor: str = "unknown"
    architecture: str = ""
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
    nvdec_codecs: Tuple[str, ...] = field(default_factory=tuple)
    nvenc_codecs: Tuple[str, ...] = field(default_factory=tuple)
    tensor_core_capabilities: Tuple[str, ...] = field(default_factory=tuple)
    fp16_supported: bool = False
    bf16_supported: bool = False
    int8_supported: bool = False
    fp8_supported: bool = False
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
        result["capabilities"] = {
            "tensor_cores": list(self.tensor_core_capabilities),
            "fp16": self.fp16_supported,
            "bf16": self.bf16_supported,
            "int8": self.int8_supported,
            "fp8": self.fp8_supported,
            "nvdec": list(self.nvdec_codecs),
            "nvenc": list(self.nvenc_codecs),
        }
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
    upscale_tile_batch_size: int = 1
    face_concurrency: int = 1
    in_flight_frames: int = 1
    detector_resolution: int = 640
    queue_depth: int = 1
    stabilization_workers: int = 1
    stabilization_chunk_size: int = 64
    ort_intra_threads: int = 1
    ort_inter_threads: int = 1
    opencv_threads: int = 1
    cuda_stream_count: int = 1
    cuda_auxiliary_streams: int = 0
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
        "upscale_tile_batch_size": (1, 4),
        "face_concurrency": (1, 16),
        "in_flight_frames": (1, 8),
        "detector_resolution": (320, 1280),
        "queue_depth": (1, 4),
        "stabilization_workers": (1, 8),
        "stabilization_chunk_size": (16, 288),
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
    def _architecture(compute_capability: Tuple[int, int]) -> str:
        """Map the runtime-reported SM version to a CUDA architecture family.

        This intentionally uses compute capability, not the marketing name.  An
        unknown future device remains usable and is recorded as its SM family
        instead of being silently treated as an RTX 4070 or RTX 3060.
        """
        families = {
            (5, 2): "Maxwell",
            (6, 0): "Pascal",
            (6, 1): "Pascal",
            (6, 2): "Pascal",
            (7, 0): "Volta",
            (7, 2): "Xavier",
            (7, 5): "Turing",
            (8, 0): "Ampere",
            (8, 6): "Ampere",
            (8, 7): "Ampere",
            (8, 9): "Ada Lovelace",
            (9, 0): "Hopper",
        }
        if compute_capability in families:
            return families[compute_capability]
        if compute_capability and all(isinstance(v, int) for v in compute_capability):
            return "SM %d.%d" % compute_capability
        return ""

    @staticmethod
    def _precision_capabilities(torch_module, device_id: int,
                                compute: Tuple[int, int], trt_available: bool):
        """Probe exposed math modes without building an engine.

        ``is_bf16_supported`` and TensorRT's builder feature flags are runtime
        capability checks.  They are deliberately kept separate from GPU name
        matching so the same code handles a future NVIDIA device safely.
        """
        fp16 = bool(compute >= (5, 3) and hasattr(torch_module, "float16"))
        bf16 = False
        try:
            probe = getattr(torch_module.cuda, "is_bf16_supported", None)
            if probe is not None:
                try:
                    bf16 = bool(probe(including_emulation=False))
                except TypeError:
                    bf16 = bool(probe())
        except Exception:
            pass

        int8 = fp8 = False
        trt_flags = set()
        if trt_available:
            try:
                import tensorrt as trt
                builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
                int8 = bool(getattr(builder, "platform_has_fast_int8", False))
                fp8 = bool(getattr(builder, "platform_has_fast_fp8", False))
                if bool(getattr(builder, "platform_has_fast_fp16", False)):
                    trt_flags.add("fp16")
                if int8:
                    trt_flags.add("int8")
                if fp8:
                    trt_flags.add("fp8")
            except Exception:
                pass

        # Tensor Core availability is reported as a capability only when the
        # runtime exposes at least one Tensor Core math mode.  SM version is a
        # hardware fact, while these flags describe what this software stack can
        # actually use.
        tensor_cores = set(trt_flags)
        if fp16 and compute >= (7, 0):
            tensor_cores.add("fp16")
        if bf16 and compute >= (8, 0):
            tensor_cores.add("bf16")
        return fp16, bf16, int8, fp8, tuple(sorted(tensor_cores))

    @staticmethod
    def _ffmpeg_capabilities(ffmpeg: Optional[str], cuda: bool):
        if not ffmpeg or not cuda:
            return False, False, (), ()
        hwaccels = HardwareProfiler._command(ffmpeg, "-hide_banner", "-hwaccels")
        decoders = HardwareProfiler._command(ffmpeg, "-hide_banner", "-decoders")
        encoders = HardwareProfiler._command(ffmpeg, "-hide_banner", "-encoders")
        decoder_names = tuple(sorted(set(re.findall(
            r"\b((?:h264|hevc|av1|vp9)_cuvid)\b", decoders.lower()))))
        encoder_names = tuple(sorted(set(re.findall(
            r"\b((?:h264|hevc|av1)_nvenc)\b", encoders.lower()))))
        nvdec = bool("cuda" in hwaccels.lower() and (decoder_names or "cuda" in decoders.lower()))
        nvenc = bool(encoder_names)
        return nvdec, nvenc, decoder_names, encoder_names

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
        architecture = ""
        compute_tuple = (0, 0)
        compute = ""
        vram_total = vram_free = 0.0
        cuda = False
        cuda_version = ""
        trt = False
        trt_version = ""
        ort_version = ""
        fp16 = bf16 = int8 = fp8 = False
        tensor_cores = ()
        try:
            import torch
            cuda = bool(torch.cuda.is_available())
            cuda_version = str(getattr(torch.version, "cuda", "") or "")
            if cuda and self.device_id < torch.cuda.device_count():
                gpu_name = str(torch.cuda.get_device_name(self.device_id))
                props = torch.cuda.get_device_properties(self.device_id)
                vram_total = _number(props.total_memory) / (1024 ** 3)
                major, minor = torch.cuda.get_device_capability(self.device_id)
                compute_tuple = (int(major), int(minor))
                compute = f"{major}.{minor}"
                architecture = self._architecture(compute_tuple)
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
        # TensorRT's Builder is a heavyweight host allocation.  On the
        # sub-7GB tier the backend admission policy rejects TensorRT before any
        # production engine is built, so constructing a Builder here would
        # consume the very RSS budget this profiler is meant to protect.  The
        # installed/provider capability and version are still recorded above;
        # only the optional Builder feature probe is deferred on that tier.
        trt_builder_probe = trt
        if cuda and vram_total and vram_total < 7.0:
            allow_small_trt = os.environ.get(
                "ROOP_ALLOW_TRT_SMALL_GPU", "").strip().lower() in (
                    "1", "true", "yes", "on")
            trt_builder_probe = bool(allow_small_trt)
            if not trt_builder_probe:
                print("[Hardware] sub-7GB GPU: TensorRT Builder capability probe "
                      "deferred; backend admission remains CUDA/CPU", flush=True)
        if cuda:
            try:
                import torch as _torch
                fp16, bf16, int8, fp8, tensor_cores = self._precision_capabilities(
                    _torch, self.device_id, compute_tuple, trt_builder_probe)
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
        nvdec, nvenc, nvdec_codecs, nvenc_codecs = self._ffmpeg_capabilities(ffmpeg, cuda)

        self._profile = HardwareProfile(
            device_id=self.device_id,
            gpu_name=gpu_name,
            gpu_vendor=self._vendor(gpu_name),
            architecture=architecture,
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
            nvdec_codecs=nvdec_codecs,
            nvenc_codecs=nvenc_codecs,
            tensor_core_capabilities=tensor_cores,
            fp16_supported=fp16,
            bf16_supported=bf16,
            int8_supported=int8,
            fp8_supported=fp8,
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
        # The sub-7GB profile intentionally does not admit TensorRT. Reporting
        # the effective precision as FP32 keeps diagnostics/profile identity
        # honest; the configured UI value remains available for an explicit,
        # separately audited override.
        if 0 < hardware.vram_total_gb < 7.0 and configured != "fp32":
            return "fp32"
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
            # Keep the complete runtime identity in the key.  In particular,
            # total VRAM and architecture are required: the same model graph
            # must not inherit a tuning profile from a different card merely
            # because both cards expose CUDA and TensorRT.
            "hardware": {
                "device_id": hardware.device_id,
                "gpu": hardware.gpu_name,
                "vendor": hardware.gpu_vendor,
                "architecture": hardware.architecture,
                "compute": hardware.compute_capability,
                "vram_total_gb": hardware.vram_total_gb,
                "vram_tier": hardware.vram_tier,
                "driver": hardware.driver_version,
                "cuda": hardware.cuda_version,
                "tensorrt": hardware.tensorrt_version,
                "ort": hardware.onnxruntime_version,
                "tensor_cores": hardware.tensor_core_capabilities,
                "fp16": hardware.fp16_supported,
                "bf16": hardware.bf16_supported,
                "int8": hardware.int8_supported,
                "fp8": hardware.fp8_supported,
                "nvdec": hardware.nvdec_codecs,
                "nvenc": hardware.nvenc_codecs,
            },
            "model": {
                "swap": _value(settings, "swap_model", "") or "",
                "detector": _value(settings, "detector_engine", "") or "",
                "mask": _value(settings, "mask_engine", "") or "",
                # Callers that load a locally revised model can provide a
                # version/hash or an explicit path.  The latter is hashed when
                # available; unknown locations are never guessed.
                **_model_identity(settings),
            },
            "enhancer": workload.enhancement_model,
            "precision": precision,
            "cuda_execution": {
                "stream_count": _value(settings, "cuda_stream_count", "auto"),
                "auxiliary_streams": _value(settings, "trt_auxiliary_streams", "auto"),
                "cuda_graph": _value(settings, "trt_cuda_graph", False),
            },
            # Workload shape and characteristics are part of profile identity;
            # these are not universal settings even on one GPU.
            "workload": workload.as_dict(),
        }
        raw = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()[:24]

    def cache_path(self, profile_dir: Optional[str], cache_key: str) -> Path:
        root = Path(profile_dir or os.environ.get(
            "ROOP_RUNTIME_PROFILE_DIR", "models/runtime_profiles"))
        return root / f"{cache_key}.json"


class CUDAGraphManager:
    """CUDA stream/graph policy and bounded graph-capture gate.

    ORT/TensorRT owns its internal execution streams.  This class therefore
    only recommends a small number of independent streams; it never creates a
    stream for every worker.  CUDA Graph capture is opt-in and is only safe for
    a caller that can provide fixed shapes, fixed addresses, and one immutable
    execution path.  The actual runner below is deliberately independent of
    ORT so an unsupported provider can fall back without changing inference.
    """

    @staticmethod
    def stream_policy(settings: Any, workload: WorkloadProfile,
                      hardware: HardwareProfile,
                      independent_work: int = 1,
                      shared_mutable_buffers: bool = False) -> dict:
        """Return a bounded stream recommendation without creating streams.

        A sub-7GB device gets one stream.  Larger devices may use two only
        when the caller has actually identified independent work and no shared
        mutable buffer.  The limit is intentionally a capability tier, not a
        GPU model name or a worker-count multiplier.
        """
        small = 0 < hardware.vram_total_gb < 7.0
        max_streams = 1 if small else 2
        requested = _value(settings, "cuda_stream_count", "auto")
        if str(requested).strip().lower() in ("", "auto", "default", "none"):
            count = min(max_streams, max(1, _integer(independent_work, 1)))
            source = "hardware/workload policy"
        else:
            count = max(1, min(max_streams, _integer(requested, 1)))
            source = "explicit request bounded by hardware"
        safe = bool(hardware.cuda_available and count > 1 and
                    _integer(independent_work, 1) > 1 and
                    not shared_mutable_buffers)
        if shared_mutable_buffers:
            count = 1
            safe = False
        # TensorRT auxiliary streams are per-context, not application-wide.
        # Keep the small-card setting serial and never request more than one
        # auxiliary stream from the larger-card profile.
        auxiliary = 0 if small else (1 if safe else 0)
        return {
            "stream_count": count,
            "max_streams": max_streams,
            "auxiliary_streams": auxiliary,
            "safe_to_overlap": safe,
            "source": source,
            "reason": ("independent work has no shared mutable buffers"
                        if safe else "serial execution preserves dependency safety"),
        }

    def readiness(self, settings: Any, workload: WorkloadProfile,
                  hardware: HardwareProfile) -> dict:
        requested = _bool(_value(settings, "trt_cuda_graph", False))
        stable_shapes = workload.input_width > 0 and workload.input_height > 0
        small = 0 < hardware.vram_total_gb < 7.0
        safe = (requested and hardware.cuda_available and stable_shapes and
                not small)
        streams = self.stream_policy(settings, workload, hardware,
                                     independent_work=2,
                                     shared_mutable_buffers=False)
        if small and requested:
            reason = ("not admitted on the sub-7GB tier; TensorRT/CUDA graph "
                      "capture requires a separately bounded candidate")
        elif safe:
            reason = ("enabled by user; caller still needs a candidate-specific "
                      "stable-shape/address contract")
        elif not requested:
            reason = "disabled by default; stable-shape capture is not yet wired"
        else:
            reason = "requested but the workload has no stable shape"
        return {"requested": requested, "safe": safe, "reason": reason,
                "stream_policy": streams}


class CUDAGraphInvalidation(RuntimeError):
    """Raised when replay is attempted with a different execution identity."""


class CUDAGraphRunner:
    """A one-owner CUDA Graph runner for fixed-shape PyTorch work.

    The owner must be one worker/thread.  Sharing a runner across workers would
    race its static input buffers, so callers should keep one runner per worker
    (or serialize access themselves).  ``capture`` performs the required warmup
    and one synchronization before capture.  Replay uses stream ordering and
    does not add a device-wide synchronization; copying the output to CPU is
    the caller's natural completion point.
    """

    def __init__(self, key: Tuple[Any, ...], warmup: int = 3):
        self.key = tuple(key)
        self.warmup = max(1, int(warmup))
        self.graph = None
        self.static_inputs = None
        self.static_output = None
        self.captured = False
        self.invalidation_reason = ""

    @staticmethod
    def supported() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available() and
                        hasattr(torch.cuda, "CUDAGraph"))
        except Exception:
            return False

    def capture(self, inputs, function):
        """Warm and capture ``function(*static_inputs)`` for this key."""
        if not self.supported():
            raise RuntimeError("CUDA Graphs are unavailable on this runtime")
        import torch
        values = tuple(inputs)
        if not values or any(not isinstance(value, torch.Tensor)
                             for value in values):
            raise TypeError("CUDA Graph inputs must be torch tensors")
        if any(not value.is_cuda for value in values):
            raise TypeError("CUDA Graph inputs must be on CUDA")
        if any(not value.is_contiguous() for value in values):
            raise TypeError("CUDA Graph inputs must be contiguous")
        self.static_inputs = tuple(torch.empty_like(value) for value in values)
        for static, value in zip(self.static_inputs, values):
            static.copy_(value)
        for _ in range(self.warmup):
            function(*self.static_inputs)
        torch.cuda.synchronize(device=values[0].device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = function(*self.static_inputs)
        if not isinstance(output, torch.Tensor):
            raise TypeError("CUDA Graph runner currently requires one tensor output")
        self.graph = graph
        self.static_output = output
        self.captured = True
        return self

    def replay(self, inputs, key=None):
        """Copy inputs into stable addresses, then enqueue one graph replay."""
        if not self.captured or self.graph is None:
            raise RuntimeError("CUDA Graph has not been captured")
        if key is not None and tuple(key) != self.key:
            raise CUDAGraphInvalidation(
                "CUDA Graph key changed; capture a new graph for this workload")
        values = tuple(inputs)
        if len(values) != len(self.static_inputs):
            raise CUDAGraphInvalidation("CUDA Graph input count changed")
        for static, value in zip(self.static_inputs, values):
            if tuple(static.shape) != tuple(value.shape) or static.dtype != value.dtype:
                raise CUDAGraphInvalidation("CUDA Graph input shape or dtype changed")
            static.copy_(value)
        self.graph.replay()
        return self.static_output

    def invalidate(self, reason: str = "configuration changed") -> None:
        """Drop graph/static-buffer references so the next call recaptures."""
        self.graph = None
        self.static_inputs = None
        self.static_output = None
        self.captured = False
        self.invalidation_reason = str(reason)


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
        # The available 12GB profile's two-face benchmark found the swapper
        # knee at three contexts; the sub-7GB profile remains explicitly
        # single-context.  This is a bounded workload-aware choice, not a
        # universal desktop default.
        swapper_pool = (0 if small else
                        (3 if workload.faces_per_frame >= 2 else 2))
        enhancer_pool = 0 if small else (2 if workload.enhancement_enabled else 1)
        expression_pool = 0 if small else 2
        # The 16GB laptop has only limited RSS headroom after the CUDA model
        # sessions are resident. Keep stabilization's live frame set single
        # threaded there; the frame workers remain independent and the bounded
        # scheduler still owns the same 1536MB hard cap.
        stabilization_workers = 1 if small else max(1, min(worker, 6))
        if not workload.stabilization_enabled:
            stabilization_workers = 1

        pixels = workload.pixels_per_frame or 1280 * 720
        if pixels > 1920 * 1080 or small:
            queue_depth = 1
        elif pixels > 1280 * 720:
            queue_depth = 2
        else:
            queue_depth = 3
        chunk = 16 if small else (96 if pixels > 1920 * 1080 else 144)
        opencv = 1 if worker >= 8 or small else 2
        encoder = "hevc_nvenc" if hardware.nvenc_available else "libx264"
        preset = "p5" if hardware.nvenc_available else "medium"
        stream_policy = CUDAGraphManager.stream_policy(
            settings, workload,
            hardware,
            independent_work=2 if workload.faces_per_frame >= 2 else 1,
            shared_mutable_buffers=False)
        # Batching and independent contexts compete for the same GPU memory.
        # The small-VRAM tier therefore gets one face, one tile, and one
        # in-flight frame.  Larger devices receive only a bounded candidate;
        # the benchmark may later replace it with a model/workload-specific
        # measured value.
        face_concurrency = (1 if small else
                            max(1, min(worker, max(1, swapper_pool))))
        swap_batch = 1 if small else (2 if worker > 1 else 1)
        swap_tile_batch = (2 if (not small and workload.faces_per_frame >= 2)
                           else 1)
        # Frame_Upscale is model-specific and its measured SPAN x4 result on
        # the 4070 was faster at batch 1 than 2/4/8.  Keep auto mode at the
        # safe measured baseline; an explicit benchmark result or environment
        # override may opt a different model/workload into batching.
        upscale_tile_batch = 1
        in_flight_frames = 1 if small else max(1, min(4, queue_depth + 1))
        values = {
            "trt_context_count": contexts,
            "worker_count": worker,
            "detector_pool_size": detector_pool,
            "detmask_pool_size": detmask_pool,
            "swapper_pool_size": swapper_pool,
            "enhancer_pool_size": enhancer_pool,
            "expression_pool_size": expression_pool,
            "batch_size": swap_batch,
            "tile_batch_size": swap_tile_batch,
            "upscale_tile_batch_size": upscale_tile_batch,
            "face_concurrency": face_concurrency,
            "in_flight_frames": in_flight_frames,
            "detector_resolution": 640,
            "queue_depth": queue_depth,
            "stabilization_workers": stabilization_workers,
            "stabilization_chunk_size": chunk,
            "ort_intra_threads": 1,
            "ort_inter_threads": 1,
            "opencv_threads": opencv,
            "cuda_stream_count": stream_policy["stream_count"],
            "cuda_auxiliary_streams": stream_policy["auxiliary_streams"],
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
            "ROOP_RUNTIME_FACE_CONCURRENCY": tuning.face_concurrency,
            "ROOP_RUNTIME_INFLIGHT_FRAMES": tuning.in_flight_frames,
            "ROOP_RUNTIME_CUDA_STREAMS": tuning.cuda_stream_count,
            "ROOP_RUNTIME_TRT_AUX_STREAMS": tuning.cuda_auxiliary_streams,
        }
        # A face-processing profile must not leak its default tile decision
        # into the later post-swap Frame_Upscale pass.  Only an explicitly
        # upscaling workload may publish this separate hint.
        if profile.workload.upscaling_enabled:
            env_values["ROOP_RUNTIME_UPSCALE_TILE_BATCH"] = (
                tuning.upscale_tile_batch_size)
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


def small_card_enhancer_policy(hardware: HardwareProfile,
                               requested: str | None) -> dict:
    """Resolve the host-RSS-safe enhancer policy for a detected small GPU.

    The default ``auto`` behavior is a measured quality/safety tradeoff: the
    6GB end-to-end path exceeds the strict 2.5GB RSS ceiling with an enhancer,
    while the unenhanced adaptive-NVDEC path stays below it. ``keep`` remains
    available for an operator who accepts that measured gate failure for a
    quality experiment. This is based on detected VRAM, never a model name,
    and does not affect larger cards.
    """
    value = str(requested or "None")
    if not (0 < float(hardware.vram_total_gb or 0) < 7.0):
        return {"requested": value, "effective": value, "changed": False,
                "reason": "not a sub-7GB hardware profile"}
    mode = os.environ.get("ROOP_SMALL_CARD_ENHANCER", "auto").strip().lower()
    if mode in ("keep", "on", "force", "1", "true", "yes"):
        return {"requested": value, "effective": value, "changed": False,
                "reason": "explicit small-card enhancer override"}
    if value.strip().lower() in ("", "none", "keep"):
        return {"requested": value, "effective": value, "changed": False,
                "reason": "enhancer was already disabled"}
    return {"requested": value, "effective": "None", "changed": True,
            "reason": "measured enhancer path exceeds the strict 2.5GB RSS gate"}


def small_card_decode_policy(hardware: HardwareProfile) -> dict:
    """Choose the small-card default decode path from the measured A/B.

    On the physical 6GB laptop, adaptive NVDEC added host RSS without
    improving end-to-end throughput on the acceptance fixture. Automatic mode
    therefore selects the lower-RSS CPU reader on the sub-7GB tier. An explicit
    ``ROOP_NVDEC=1`` remains an intentional experiment and is never silently
    overridden; larger cards are unchanged.
    """
    requested = os.environ.get("ROOP_NVDEC", "auto").strip().lower()
    if requested in ("1", "true", "yes", "on"):
        return {"requested": requested, "effective": "1", "changed": False,
                "reason": "explicit NVDEC request"}
    if requested in ("0", "false", "no", "off"):
        return {"requested": requested, "effective": "0", "changed": False,
                "reason": "explicit CPU decode request"}
    if not (0 < float(hardware.vram_total_gb or 0) < 7.0):
        return {"requested": requested or "auto", "effective": requested or "auto",
                "changed": False, "reason": "not a sub-7GB hardware profile"}
    mode = os.environ.get("ROOP_SMALL_CARD_NVDEC", "auto").strip().lower()
    if mode in ("keep", "on", "force", "1", "true", "yes"):
        return {"requested": requested or "auto", "effective": "1", "changed": False,
                "reason": "explicit small-card NVDEC override"}
    return {"requested": requested or "auto", "effective": "0", "changed": True,
            "reason": "measured NVDEC path increases RSS without an end-to-end speed win"}


__all__ = [
    "AutoTuner", "CUDAGraphInvalidation", "CUDAGraphManager",
    "CUDAGraphRunner", "HardwareProfile", "HardwareProfiler",
    "PrecisionSelector", "ProfileStore", "ResourceManager", "RuntimeMonitor",
    "RuntimeOptimizer", "RuntimeProfile", "RuntimeTuning", "TensorRTEngineManager",
    "WorkloadProfile", "WorkloadProfiler", "small_card_enhancer_policy",
    "small_card_decode_policy",
]
