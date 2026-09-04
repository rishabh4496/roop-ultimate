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
from collections import deque
from dataclasses import asdict, dataclass, field, fields, replace
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
    # TensorRT uses -1 to mean "let the runtime choose" for auxiliary
    # streams.  Treat the config-file sentinel as automatic here too; it must
    # not enter the explicit-value clamp path below.
    if name == "trt_auxiliary_streams" and _integer(value, 0) == -1:
        return True
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


def _trt_builder_identity(precision: str) -> dict:
    """Capture builder inputs that can alter an engine or timing cache.

    Keep this as an open mapping of environment-backed settings. New
    TensorRT/ORT options can be added without changing GPU-family logic, and
    an unset value remains distinct from an explicit override.
    """
    names = (
        "ROOP_TRT_WORKSPACE_MB", "ROOP_TRT_WORKSPACE_FRACTION",
        "ROOP_TRT_MAX_WORKSPACE_BYTES", "ROOP_TRT_PARTITION_ITERATIONS",
        "ROOP_TRT_BUILDER_OPT_LEVEL", "ROOP_TRT_AUX_STREAMS",
        "ROOP_TRT_CUDA_GRAPH",
    )
    return {
        "precision": str(precision or "mixed").lower(),
        "options": {name: os.environ.get(name, "<unset>") for name in names},
        "context_memory_sharing": True,
        "layer_norm_fp32_fallback": str(precision).lower() == "mixed",
        "force_sequential_engine_build": str(precision).lower() == "mixed",
        "build_heuristics": str(precision).lower() == "mixed",
    }


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
    cpu_name: str = ""
    cpu_frequency_mhz: float = 0.0
    cpu_max_frequency_mhz: float = 0.0
    cpu_simd_capabilities: Tuple[str, ...] = field(default_factory=tuple)
    os_affinity_supported: bool = False
    cpu_performance_indices: Tuple[int, ...] = field(default_factory=tuple)
    cpu_efficiency_indices: Tuple[int, ...] = field(default_factory=tuple)
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    platform: str = ""
    # Optional heterogeneous-CPU topology.  These stay descriptive in Phase 10;
    # Gate D owns any i9-specific affinity or P/E scheduling policy.
    cpu_performance_cores: int = 0
    cpu_efficiency_cores: int = 0
    cpu_topology_source: str = "unknown"

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
        # One stable identity is shared by diagnostics, benchmark reports, and
        # persisted profile consumers. The helper excludes transient free VRAM.
        try:
            from roop.hardware_validation import hardware_profile_key
            result["hardware_profile_key"] = hardware_profile_key(result)
        except Exception:
            result["hardware_profile_key"] = None
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
    tracking_enabled: bool = False
    mask_enabled: bool = False
    enhancement_resolution: int = 0
    output_codec: str = ""
    video_length_frames: int = 0
    fps: float = 0.0
    estimated_complexity: float = 0.0
    # Optional measured face height in input pixels.  Unknown is deliberately
    # quality-safe: the automatic detector policy keeps the 640 canvas unless
    # workload evidence proves a lower or higher canvas is appropriate.
    estimated_face_size_px: float = 0.0

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
    ffmpeg_threads: int = 1
    cuda_stream_count: int = 1
    cuda_auxiliary_streams: int = 0
    cuda_graph_enabled: bool = False
    cpu_performance_threads: int = 0
    cpu_efficiency_threads: int = 0
    cpu_distribution: str = "auto"
    ram_buffer_mb: int = 512
    backend: str = "cuda"
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
    autotune: dict = field(default_factory=dict)

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
            "autotune": self.autotune,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Optional["RuntimeProfile"]:
        """Load a profile while tolerating fields added by newer phases."""
        if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
            return None

        def build(kind, value):
            value = dict(value or {})
            names = {item.name for item in fields(kind)}
            value = {name: value[name] for name in names if name in value}
            return kind(**value)

        try:
            hardware = build(HardwareProfile, payload.get("hardware"))
            workload = build(WorkloadProfile, payload.get("workload"))
            tuning = build(RuntimeTuning, payload.get("tuning"))
            for name in ("nvdec_codecs", "nvenc_codecs", "tensor_core_capabilities",
                         "cpu_simd_capabilities", "cpu_performance_indices",
                         "cpu_efficiency_indices"):
                value = getattr(hardware, name)
                if isinstance(value, list):
                    hardware = replace(hardware, **{name: tuple(value)})
            return cls(
                schema_version=SCHEMA_VERSION,
                created_at=_number(payload.get("created_at")),
                hardware=hardware,
                workload=workload,
                tuning=tuning,
                precision=str(payload.get("precision", "fp32")),
                provider=str(payload.get("provider", "cuda")),
                explicit_settings=tuple(payload.get("explicit_settings", ()) or ()),
                automatic_settings=tuple(payload.get("automatic_settings", ()) or ()),
                reasons=tuple(payload.get("reasons", ()) or ()),
                cache_key=str(payload.get("cache_key", "")),
                autotune=dict(payload.get("autotune", {}) or {}),
            )
        except (TypeError, ValueError):
            return None


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
        "ffmpeg_threads": (1, 4),
        "cuda_stream_count": (1, 4),
        "cuda_auxiliary_streams": (-1, 3),
        "cpu_performance_threads": (0, 16),
        "cpu_efficiency_threads": (0, 16),
        "ram_buffer_mb": (512, 4096),
    }

    @classmethod
    def clamp(cls, name: str, value: Any,
              hardware: Optional[HardwareProfile] = None) -> int:
        lo, hi = cls.BOUNDS[name]
        # The static bounds are a safety floor for unprofiled callers.  A
        # profiled machine may expose more logical CPUs (for example a hybrid
        # desktop); do not silently turn a P+E candidate into a 16-thread
        # candidate merely because the test harness predates that topology.
        if hardware is not None and name in (
                "worker_count", "cpu_performance_threads",
                "cpu_efficiency_threads"):
            hi = max(hi, int(hardware.cpu_logical_cores or hi))
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


def _positive_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _cpu_topology(physical: int, logical: int) -> Tuple[int, int, str]:
    """Best-effort P/E topology probe without imposing a platform policy.

    Linux exposes per-CPU capacity in sysfs on many hybrid systems.  The
    environment override is useful for Windows telemetry providers and test
    harnesses that know the topology but do not expose it through psutil.
    Phase 10 records the information only; it does not bind threads to cores.
    """
    detected = detect_cpu_topology(physical, logical)
    return (int(detected["p_cores"]), int(detected["e_cores"]),
            str(detected["source"]))


def _parse_cpu_indices(value: Any) -> Tuple[int, ...]:
    """Parse an explicit affinity list without accepting a broad mask."""
    values = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if index >= 0:
            values.append(index)
    return tuple(sorted(set(values)))


def _windows_cpu_set_topology() -> dict:
    """Read Windows hybrid CPU sets and their efficiency classes.

    ``SYSTEM_CPU_SET_INFORMATION`` is a variable-sized Windows structure.
    Its efficiency class is a runtime OS fact: higher values are faster and
    less power-efficient, so the highest class is treated as P-core class.
    Unknown record types are skipped for forward compatibility.
    """
    if os.name != "nt":
        return {"p_indices": (), "e_indices": (), "p_cores": 0,
                "e_cores": 0, "source": "not-windows"}
    try:
        import ctypes
        import struct
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        query = kernel32.GetSystemCpuSetInformation
        query.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
                          ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p,
                          ctypes.c_ulong]
        query.restype = ctypes.c_bool
        required = ctypes.c_ulong()
        query(None, 0, ctypes.byref(required), None, 0)
        if required.value < 32:
            return {"p_indices": (), "e_indices": (), "p_cores": 0,
                    "e_cores": 0, "source": "windows-cpu-set-unavailable"}
        buffer = ctypes.create_string_buffer(required.value)
        if not query(buffer, required.value, ctypes.byref(required), None, 0):
            return {"p_indices": (), "e_indices": (), "p_cores": 0,
                    "e_cores": 0, "source": "windows-cpu-set-unavailable"}

        records = []
        offset = 0
        while offset + 8 <= required.value:
            size, record_type = struct.unpack_from("<II", buffer, offset)
            if size < 8 or offset + size > required.value:
                break
            # Type 0 is CpuSetInformation. Fields after the union begin at
            # offset 8: Id (DWORD), Group (WORD), logical/core indices, then
            # EfficiencyClass (BYTE at offset 18).
            if record_type == 0 and size >= 32:
                ident, group, logical_index, core_index, _llc, _numa, efficiency = struct.unpack_from(
                    "<I H B B B B B", buffer, offset + 8)
                records.append((ident, group, logical_index, core_index,
                                efficiency))
            offset += size
        if not records:
            return {"p_indices": (), "e_indices": (), "p_cores": 0,
                    "e_cores": 0, "source": "windows-cpu-set-empty"}

        classes = {row[4] for row in records}
        if len(classes) < 2:
            return {"p_indices": (), "e_indices": (), "p_cores": 0,
                    "e_cores": 0, "source": "windows-cpu-set-uniform"}
        p_class = max(classes)
        p = [group * 64 + logical_index for _id, group, logical_index,
             _core, efficiency in records if efficiency == p_class]
        e = [group * 64 + logical_index for _id, group, logical_index,
             _core, efficiency in records if efficiency != p_class]
        p_core_set = {(group, core) for _id, group, _logical, core, efficiency
                      in records if efficiency == p_class}
        e_core_set = {(group, core) for _id, group, _logical, core, efficiency
                      in records if efficiency != p_class}
        return {
            "p_indices": tuple(sorted(set(p))),
            "e_indices": tuple(sorted(set(e))),
            "p_cores": len(p_core_set),
            "e_cores": len(e_core_set),
            "source": "windows-cpu-set-efficiency-class",
        }
    except (AttributeError, OSError, TypeError, ValueError, struct.error):
        return {"p_indices": (), "e_indices": (), "p_cores": 0,
                "e_cores": 0, "source": "windows-cpu-set-unavailable"}


def detect_cpu_topology(physical: int, logical: int) -> dict:
    """Return measured P/E logical indices and physical-core counts."""
    explicit_p = _parse_cpu_indices(os.environ.get("ROOP_CPU_P_INDICES"))
    explicit_e = _parse_cpu_indices(os.environ.get("ROOP_CPU_E_INDICES"))
    if explicit_p or explicit_e:
        return {
            "p_indices": explicit_p,
            "e_indices": explicit_e,
            "p_cores": _positive_int(os.environ.get("ROOP_CPU_P_CORES")) or len(explicit_p),
            "e_cores": _positive_int(os.environ.get("ROOP_CPU_E_CORES")) or len(explicit_e),
            "source": "environment-indices",
        }

    windows = _windows_cpu_set_topology()
    if windows["p_indices"] and windows["e_indices"]:
        return windows

    capacities = []
    try:
        if platform.system().lower() == "linux":
            for path in sorted(Path("/sys/devices/system/cpu").glob(
                    "cpu[0-9]*/cpu_capacity")):
                capacities.append(_positive_int(path.read_text().strip()))
    except (OSError, ValueError):
        capacities = []
    if len(capacities) >= 2 and max(capacities) > min(capacities):
        peak = max(capacities)
        p_indices = tuple(index for index, capacity in enumerate(capacities)
                          if capacity >= peak * 0.8)
        e_indices = tuple(index for index, capacity in enumerate(capacities)
                          if capacity < peak * 0.8)
        if p_indices and e_indices:
            # Linux exposes logical CPUs here; retain that fact in the fields
            # while avoiding a guessed physical-core mapping.
            return {"p_indices": p_indices, "e_indices": e_indices,
                    "p_cores": len(p_indices), "e_cores": len(e_indices),
                    "source": "linux-cpu-capacity-logical"}
    return {"p_indices": (), "e_indices": (), "p_cores": 0,
            "e_cores": 0, "source": "unknown"}


def _cpu_simd_capabilities() -> Tuple[str, ...]:
    """Report SIMD features exposed by the active NumPy dispatch runtime."""
    try:
        import numpy as np
        features = getattr(getattr(np, "core", None), "_multiarray_umath", None)
        features = getattr(features, "__cpu_features__", {}) or {}
        return tuple(sorted(str(name) for name, enabled in features.items()
                            if enabled))
    except Exception:
        return ()


def _cpu_frequency() -> Tuple[float, float]:
    try:
        import psutil
        frequency = psutil.cpu_freq()
        if frequency is not None:
            return (_number(getattr(frequency, "current", 0.0)),
                    _number(getattr(frequency, "max", 0.0)))
    except Exception:
        pass
    return 0.0, 0.0


def _cpu_name() -> str:
    """Return the best available CPU brand string without assuming a model.

    On Windows, ``platform.processor()`` is commonly only the generic
    ``Intel64 Family ...`` identifier.  The registry value is the OS-reported
    brand string (for example an i7-12700H), so use it when available and keep
    the portable platform fallbacks for systems where it is not.
    """
    candidates = []
    try:
        if platform.system().lower() == "windows":
            import winreg
            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                candidates.append(winreg.QueryValueEx(key, "ProcessorNameString")[0])
    except (ImportError, OSError, TypeError):
        pass
    try:
        candidates.extend((platform.processor(), platform.uname().processor))
    except (AttributeError, OSError):
        pass
    return next((str(value).strip() for value in candidates
                 if str(value or "").strip()), "")


def _cpu_affinity_info() -> Tuple[bool, Tuple[int, ...]]:
    try:
        import psutil
        process = psutil.Process()
        indices = tuple(int(index) for index in process.cpu_affinity())
        return True, indices
    except Exception:
        return False, ()


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
                # TensorRT exposes no ``platform_has_fast_fp8``.  Asking for
                # one returned False on every GPU -- including Ada and Hopper
                # parts that do have FP8 tensor cores -- so "FP8 unsupported"
                # was indistinguishable from "we asked a question TensorRT
                # cannot answer".  Ask what this builder can answer instead:
                # whether it exposes the FP8 build flag and datatype, gated on
                # the first SM version with FP8 tensor cores.  This records
                # *exposure* only; precision_policy still refuses to select
                # FP8 until a calibrated provider path and measured quality
                # exist, so a future device cannot silently opt into it.
                fp8 = bool(
                    hasattr(getattr(trt, "BuilderFlag", None), "FP8")
                    and hasattr(getattr(trt, "DataType", None), "FP8")
                    and compute >= (8, 9))
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
    def _resolve_ffmpeg() -> Optional[str]:
        """Find ffmpeg without depending on the caller's PATH.

        WHY THIS IS NOT `shutil.which` ALONE. Pinokio's own shell puts ffmpeg on
        PATH, but this app is also entered from venv pythons, benchmark child
        processes and ordinary terminals that do not inherit it. When the lookup
        failed, `_ffmpeg_capabilities` returned all-False and the hardware
        profile silently recorded a machine with NO NVDEC AND NO NVENC.

        That is not a cosmetic field. `tuning()` selects
        ``hevc_nvenc if nvenc_available else libx264``, and the autotuner only
        offers nvenc candidates when the flag is set -- so a PATH accident
        downgraded a working hardware encoder to software encoding. Phase 13
        measured that exact codec choice at +16.97% end-to-end on the 4070.

        MEASURED ON THE PHYSICAL RTX 3060 LAPTOP: ffmpeg exposes av1/h264/
        hevc_nvenc and ten CUVID decoders, and a render encoded with hevc_nvenc
        successfully, while the profile reported both engines as unavailable.
        The capability must be probed from a resolvable binary, per the
        hardware-matrix rule that capabilities are detected rather than assumed.
        """
        # Delegated to the shared resolver so the profiler and the render path
        # can never disagree about which ffmpeg exists.  The search order below
        # is retained as the fallback it always was.
        from roop.ffmpeg_path import ffmpeg_binary
        resolved = ffmpeg_binary()
        if resolved and os.path.isabs(resolved) and os.path.isfile(resolved):
            return resolved
        found = shutil.which("ffmpeg")
        if found:
            return found
        home = os.environ.get("PINOKIO_HOME")
        if not home or not os.path.isdir(home):
            try:
                cfg = os.path.join(os.path.expanduser("~"), ".pinokio",
                                   "config.json")
                with open(cfg, "r", encoding="utf-8") as fh:
                    home = json.load(fh).get("home")
            except Exception:
                home = None
        # <PINOKIO_HOME>/api/<launcher>/app/roop/this_file.py
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not home or not os.path.isdir(home):
            home = os.path.dirname(os.path.dirname(os.path.dirname(app_dir)))
        candidates = (
            os.path.join(home, "bin", "miniforge", "Library", "bin"),
            os.path.join(home, "bin", "miniconda", "Library", "bin"),
            os.path.join(app_dir, "env", "Library", "bin"),
        )
        for root in candidates:
            for name in ("ffmpeg.exe", "ffmpeg"):
                cand = os.path.join(root, name)
                if os.path.isfile(cand):
                    return cand
        return None

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
            trt = bool(cuda and any(
                str(provider).lower() == "tensorrtexecutionprovider"
                for provider in available))
        except Exception:
            available = set()

        try:
            import tensorrt as _trt
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
            # psutil is the normal source.  The fallback is intentionally
            # conservative and is only used when that optional dependency is
            # unavailable; it is never the worker-count policy.
            logical = max(1, _positive_int(os.environ.get("NUMBER_OF_PROCESSORS")) or 1)
            physical = max(1, logical // 2)
            ram_total = ram_available = 0.0

        topology = detect_cpu_topology(physical, logical)
        p_cores = int(topology["p_cores"])
        e_cores = int(topology["e_cores"])
        topology_source = str(topology["source"])
        cpu_frequency_mhz, cpu_max_frequency_mhz = _cpu_frequency()
        cpu_name = _cpu_name()
        affinity_supported, _affinity_indices = _cpu_affinity_info()

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
        if not driver and cuda:
            # Single shared resolver.  The previous fallback here called
            # ``torch._C._cuda_getDriverVersion``, which does not exist in any
            # supported build: it never ran, so the "fallback" was dead code
            # and a machine without nvidia-smi silently lost this dimension of
            # profile identity rather than reporting that it was unavailable.
            from roop.backend_manager import _driver_from_smi
            driver = _driver_from_smi(self.device_id)

        ffmpeg = self._resolve_ffmpeg()
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
            cpu_name=cpu_name,
            cpu_frequency_mhz=round(cpu_frequency_mhz, 3),
            cpu_max_frequency_mhz=round(cpu_max_frequency_mhz, 3),
            cpu_simd_capabilities=_cpu_simd_capabilities(),
            os_affinity_supported=affinity_supported,
            cpu_performance_indices=tuple(topology["p_indices"]),
            cpu_efficiency_indices=tuple(topology["e_indices"]),
            ram_total_gb=round(ram_total, 3),
            ram_available_gb=round(ram_available, 3),
            platform=f"{platform.system()}-{platform.release()}",
            cpu_performance_cores=p_cores,
            cpu_efficiency_cores=e_cores,
            cpu_topology_source=topology_source,
        )
        return self._profile


# ── Process-wide hardware profile cache ──────────────────────────────────────
# HardwareProfiler caches on the INSTANCE, and every hot caller constructed a
# fresh instance -- so the cache never once fired in production. Measured on an
# RTX 4070: `HardwareProfiler().profile()` costs 4.2-6.5 s, of which 4.43 s is
# `_precision_capabilities`' TensorRT Builder probe (the module's own comment
# calls that Builder "a heavyweight host allocation"); every other probe in the
# function totals under 70 ms.
#
# `ProcessMgr.initialize` calls it, and `live_swap` calls `initialize` on EVERY
# `/api/preview`. Measured cost of one warm repeat preview of an unchanged frame
# with unchanged settings, RTX 4070 / CUDA / realswap+RealityUX+UltraMax:
#
#     initialize      5.31 - 5.58 s      <- essentially all of it this probe
#     process_frame   0.63 - 0.66 s      <- the actual swap
#
# i.e. the preview spent ~89% of its wall clock re-deriving facts that cannot
# change while the process is alive. This module-level cache is what makes the
# instance cache's intent true.
#
# WHAT IS AND IS NOT SAFE TO CACHE. Every field this returns describes fixed
# hardware or a fixed toolchain (GPU name, compute capability, precision
# support, driver, CPU topology) EXCEPT `vram_free` and `ram_available`, which
# are instantaneous readings. No consumer treats those as live: the small-card
# policies tier on `vram_total`, and `session_pool` compares
# `hardware_profile_key`. A caller that genuinely needs a fresh free-memory
# reading must pass `refresh=True` and thereby pay the probe knowingly.
_SHARED_PROFILE_LOCK = threading.Lock()
_SHARED_PROFILES: Dict[int, HardwareProfile] = {}


def shared_hardware_profile(device_id: int = 0,
                            refresh: bool = False) -> HardwareProfile:
    """The hardware profile for `device_id`, probed once per process.

    Use this anywhere the profile is read on a per-call, per-frame or per-run
    path. `HardwareProfiler(...).profile()` remains available for the callers
    that deliberately want a fresh probe.
    """
    key = max(0, _integer(device_id, 0))
    if not refresh:
        cached = _SHARED_PROFILES.get(key)
        if cached is not None:
            return cached
    with _SHARED_PROFILE_LOCK:
        # Re-check: another thread may have probed while we queued for the lock,
        # which is exactly what concurrent worker start-up looks like.
        if not refresh:
            cached = _SHARED_PROFILES.get(key)
            if cached is not None:
                return cached
        profile = HardwareProfiler(key).profile(refresh=refresh)
        _SHARED_PROFILES[key] = profile
        return profile


def reset_shared_hardware_profile() -> None:
    """Drop the cache. For tests, and for a deliberate re-probe after a change
    that can actually move the answer (a driver swap needs a restart anyway)."""
    with _SHARED_PROFILE_LOCK:
        _SHARED_PROFILES.clear()


class WorkloadProfiler:
    """Collect workload facts without running a detector or model."""

    def profile(self, source_video: Optional[str] = None,
                settings: Any = None, frame_count: int = 0,
                resolution: Optional[Tuple[int, int]] = None,
                output_resolution: Optional[Tuple[int, int]] = None,
                faces_per_frame: Optional[float] = None,
                face_count: int = 0,
                estimated_face_size_px: float = 0.0) -> WorkloadProfile:
        width = height = fps = 0.0
        if resolution:
            width, height = resolution
        if source_video:
            try:
                import cv2
                cap = cv2.VideoCapture(source_video)
                if not width or not height:
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if not fps:
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
        tracking = _bool(_value(settings, "track_identities", False))
        mask_engine = _short(_value(settings, "mask_engine", ""))
        mask_strength = _number(_value(settings, "swap_model_mask_strength", 0))
        mask = bool(mask_engine and mask_engine.lower() not in ("none", "keep")) or mask_strength > 0
        enhancement_resolution = _integer(
            _value(settings, "enhancement_resolution",
                   _value(settings, "enhancer_resolution", 0)), 0)
        if not enhancement_resolution and enhancement:
            match = re.search(r"(?:^|\s)(\d{3,4})(?:\s|$)", enhancer)
            enhancement_resolution = _integer(match.group(1), 0) if match else 0
        output_codec = _short(_value(settings, "output_video_codec", ""))
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
        complexity += 0.25 if tracking else 0.0
        complexity += 0.25 if mask else 0.0

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
            tracking_enabled=tracking,
            mask_enabled=mask,
            enhancement_resolution=max(0, enhancement_resolution),
            output_codec=output_codec,
            video_length_frames=max(0, _integer(frame_count)),
            fps=max(0.0, _number(fps)),
            estimated_complexity=round(complexity, 3),
            estimated_face_size_px=max(0.0, _number(estimated_face_size_px)),
        )


class PrecisionSelector:
    """Preserve configured precision; automatic mode stays quality-safe."""

    def select(self, settings: Any, hardware: HardwareProfile,
               model_key: str = "") -> str:
        configured = str(_value(settings, "trt_precision", "mixed") or "mixed").lower()
        # The sub-7GB profile intentionally does not admit TensorRT. Reporting
        # the effective precision as FP32 keeps diagnostics/profile identity
        # honest; the configured UI value remains available for an explicit,
        # separately audited override.
        if 0 < hardware.vram_total_gb < 7.0 and configured != "fp32":
            return "fp32"
        if configured == "fp32":
            return configured
        if configured in ("fp16", "mixed"):
            if not (hardware.cuda_available and hardware.tensorrt_available and
                    hardware.fp16_supported):
                return "fp32"
            if model_key:
                try:
                    from roop.precision_policy import get_policy
                    evidence = getattr(get_policy(model_key), configured,
                                       "not-validated")
                    if evidence not in ("safe", "candidate"):
                        return "fp32"
                except Exception:
                    return "fp32"
            return configured
        if configured in ("bf16", "int8", "fp8"):
            # Novel modes are never selected from a capability flag alone.
            # They need a hardware probe, TensorRT, model evidence, and a
            # real provider implementation with quality validation.
            if not (hardware.cuda_available and hardware.tensorrt_available):
                return "fp32"
            if not bool(getattr(hardware, f"{configured}_supported", False)):
                return "fp32"
            try:
                from roop.precision_policy import get_policy
                evidence = getattr(get_policy(model_key), configured,
                                   "not-validated")
                if evidence != "safe" or configured != "bf16":
                    return "fp32"
            except Exception:
                return "fp32"
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
                "cpu_name": hardware.cpu_name,
                "cpu_physical": hardware.cpu_physical_cores,
                "cpu_logical": hardware.cpu_logical_cores,
                "cpu_frequency_max_mhz": hardware.cpu_max_frequency_mhz,
                "cpu_simd": hardware.cpu_simd_capabilities,
                "cpu_p_cores": hardware.cpu_performance_cores,
                "cpu_e_cores": hardware.cpu_efficiency_cores,
                "cpu_topology_source": hardware.cpu_topology_source,
                "os_affinity": hardware.os_affinity_supported,
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
            "tensorrt_builder": _trt_builder_identity(precision),
            # Settings that can change throughput, memory pressure, or output
            # quality are part of the cache identity.  This prevents a tuned
            # UltraMax/two-face profile from being reused for a different
            # enhancer, tracking mode, or explicit codec.
            "runtime_settings": {
                name: _value(settings, name, "auto") for name in (
                    "provider", "trt_precision", "perf_trt_pool",
                    "perf_detector_pool", "perf_detmask_pool", "perf_expr_pool",
                    "perf_batch_swap", "max_threads", "auto_thread_selection",
                    "cpu_ort_intra_threads", "cpu_ort_inter_threads",
                    "cpu_opencv_threads", "cpu_ffmpeg_threads",
                    "cpu_distribution", "cpu_e_limit",
                    "perf_encoder_preset", "output_video_codec",
                    "track_identities", "temporal_detection", "stabilize_face",
                    "stabilize_mask", "stabilize_enhancer", "upscale_after_swap",
                    "enhancement_resolution", "enhancer_resolution" )
            },
            # Workload shape and characteristics are part of profile identity;
            # these are not universal settings even on one GPU.
            "workload": workload.as_dict(),
        }
        # Controlled CPU-policy A/B runs arrive through the environment so a
        # benchmark need not rewrite the user's config. Include those values
        # in the profile namespace; otherwise a cached auto profile could
        # silently erase a requested P-only or P+E candidate.
        identity["runtime_settings"]["cpu_distribution"] = os.environ.get(
            "ROOP_CPU_DISTRIBUTION",
            _value(settings, "cpu_distribution", "auto"))
        identity["runtime_settings"]["cpu_e_limit"] = os.environ.get(
            "ROOP_CPU_E_LIMIT", _value(settings, "cpu_e_limit", "auto"))
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
        self.stream = None
        self.device = None
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
        device = values[0].device
        if any(value.device != device for value in values):
            raise TypeError("CUDA Graph inputs must share one CUDA device")

        # A graph has fixed addresses and must not share the caller's default
        # stream.  The dedicated stream owns static input/output allocation,
        # exactly three warm-up launches by default, capture, and every replay.
        # Stream waits preserve dependency ordering without a device-wide sync
        # on the per-frame fast path.
        stream = torch.cuda.Stream(device=device)
        caller_stream = torch.cuda.current_stream(device=device)
        stream.wait_stream(caller_stream)
        with torch.cuda.stream(stream):
            static_inputs = tuple(torch.empty_like(value) for value in values)
            for static, value in zip(static_inputs, values):
                static.copy_(value)
            for _ in range(self.warmup):
                function(*static_inputs)
        stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = function(*static_inputs)
        if not isinstance(output, torch.Tensor):
            raise TypeError("CUDA Graph runner currently requires one tensor output")
        self.static_inputs = static_inputs
        self.graph = graph
        self.static_output = output
        self.stream = stream
        self.device = device
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
        if any(value.device != self.device for value in values):
            raise CUDAGraphInvalidation("CUDA Graph input device changed")
        import torch
        caller_stream = torch.cuda.current_stream(device=self.device)
        self.stream.wait_stream(caller_stream)
        with torch.cuda.stream(self.stream):
            for static, value in zip(self.static_inputs, values):
                if tuple(static.shape) != tuple(value.shape) or static.dtype != value.dtype:
                    raise CUDAGraphInvalidation("CUDA Graph input shape or dtype changed")
                static.copy_(value)
            self.graph.replay()
        caller_stream.wait_stream(self.stream)
        return self.static_output

    def invalidate(self, reason: str = "configuration changed") -> None:
        """Drop graph/static-buffer references so the next call recaptures."""
        self.graph = None
        self.static_inputs = None
        self.static_output = None
        self.stream = None
        self.device = None
        self.captured = False
        self.invalidation_reason = str(reason)


class AutoTuner:
    """Derive a conservative profile from hardware *and* workload facts."""

    def tune(self, hardware: HardwareProfile, workload: WorkloadProfile,
              settings: Any = None) -> Tuple[RuntimeTuning, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
        small = hardware.vram_total_gb > 0 and hardware.vram_total_gb < 7.0
        gpu_cap = 8 if small else 10
        cpu_cap = max(1, min(hardware.cpu_logical_cores, gpu_cap))
        worker = max(1, min(cpu_cap, hardware.cpu_physical_cores or cpu_cap))
        distribution = str(os.environ.get(
            "ROOP_CPU_DISTRIBUTION",
            _value(settings, "cpu_distribution", "auto")) or "auto").strip().lower()
        if distribution in ("p+e", "p_e", "p-plus-e", "pplus_e"):
            distribution = "p_plus_e"
        elif distribution in ("p-priority-e", "p_priority_e_limited",
                              "p+e-limited", "p_plus_e_limited"):
            distribution = "p_priority_e"
        elif distribution not in ("auto", "p_only", "p_plus_e", "p_priority_e"):
            distribution = "auto"
        p_available = (len(hardware.cpu_performance_indices) or
                       hardware.cpu_performance_cores)
        e_available = (len(hardware.cpu_efficiency_indices) or
                       hardware.cpu_efficiency_cores)
        # MEASURED AND REJECTED (2026-08-30, physical RTX 3060 Laptop,
        # i7-12700H, 6 P + 8 E, `windows-cpu-set-efficiency-class`).
        # DO NOT RE-ADD an automatic `auto` -> `p_plus_e` promotion here without
        # a 600-frame counterbalanced measurement; three attempts' worth of
        # short-window evidence says it works and it does not.
        #
        # `auto` deliberately does NOT select a P/E-aware distribution. The
        # obvious change -- promote it whenever the OS reports a real hybrid
        # topology -- was implemented, verified on hardware, and then reverted.
        #
        #   d4.mp4, 120-frame window, counterbalanced, four experiments:
        #       distribution at fixed workers + pinning        +19.6%
        #       candidate vs shipped production default        +18.9%
        #       distribution at 8 workers, no pinning          +18.8%
        #       shipped default through the fixed harness      +19%
        #   d4.mp4, 600-frame window, counterbalanced:
        #       auto      4.55 / 4.52   mean 4.535
        #       p_plus_e  4.52 / 4.50   mean 4.51      -0.5%  NEUTRAL
        #
        # The gain is entirely a short-window artefact. It is coherent with
        # this pipeline being GPU-bound in steady state (mean GPU utilisation
        # ~57% with peaks at 100% while CPU sits near 31% mean): CPU scheduling
        # only helps the CPU-bound warm-up, which a 120-frame window
        # over-weights and a production-length render amortises away.
        #
        # Worker count was tested separately over the same windows and is also
        # neutral (8 vs 20 is -0.9%), which is what vindicates `max_threads: 8`
        # on real 6 GB silicon.
        #
        # An explicit `ROOP_CPU_DISTRIBUTION` / `cpu_distribution` setting still
        # selects a policy below; only the automatic promotion is refused.
        if distribution != "auto" and p_available and e_available:
            if distribution == "p_only":
                worker = max(1, min(hardware.cpu_logical_cores, p_available))
            elif distribution == "p_plus_e":
                worker = max(1, min(hardware.cpu_logical_cores,
                                    p_available + e_available))
            else:
                try:
                    e_limit = int(os.environ.get(
                        "ROOP_CPU_E_LIMIT", max(1, e_available // 4)))
                except (TypeError, ValueError):
                    e_limit = max(1, e_available // 4)
                e_limit = max(1, min(e_available, e_limit))
                worker = max(1, min(hardware.cpu_logical_cores,
                                    p_available + e_limit))
        if workload.estimated_complexity >= 3.5 and distribution == "auto":
            cpu_cap = max(1, min(cpu_cap, 8 if not small else 6))
            worker = max(1, min(worker, cpu_cap))
        contexts = 1 if small or not hardware.tensorrt_available else 2
        detector_pool = 0 if small or not hardware.cuda_available else 2
        detmask_pool = 0 if small or not hardware.cuda_available else 2
        # The available 12GB profile's two-face benchmark found the swapper
        # knee at three contexts; the sub-7GB profile remains explicitly
        # single-context.  This is a bounded workload-aware choice, not a
        # universal desktop default.
        swapper_pool = (0 if small or not hardware.tensorrt_available else
                        (3 if workload.faces_per_frame >= 2 else 2))
        enhancer_pool = 0 if small or not hardware.cuda_available else (2 if workload.enhancement_enabled else 1)
        expression_pool = 0 if small or not hardware.cuda_available else 2
        # Stabilization is a host-side stateful stage, not a second GPU context.
        # Do not collapse the whole pipeline to one worker on the small-VRAM
        # tier: swap/mask/enhance calls already serialize through their own
        # single-context guards, while CPU compositing can overlap that work.
        # ProcessMgr limits the actual stabilization width again from the
        # measured RAM/frame budget, so this is only the available worker cap.
        stabilization_workers = worker if workload.stabilization_enabled else 1
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
        # Keep the encoder's CPU pool separate from the frame-worker pool.  The
        # encoder is already offloaded to NVENC where available; software
        # encoding gets only a small bounded pool so it cannot multiply the
        # Python/ORT/OpenCV budget.
        ffmpeg = 1 if small or worker >= 8 else 2

        # Detector canvas is a quality/latency trade-off, not a GPU-name
        # switch.  Keep unknown face size at the calibrated 640 baseline.  A
        # measured large face can use 512 on 720p-class input; higher-resolution
        # input gets a larger canvas so small faces are not silently discarded.
        if small or not pixels:
            detector_resolution = 640
        elif pixels > 3840 * 2160:
            detector_resolution = 960
        elif pixels > 1920 * 1080:
            detector_resolution = 768
        elif (pixels <= 1280 * 720 and
              workload.estimated_face_size_px >= 160):
            detector_resolution = 512
        else:
            detector_resolution = 640

        encoder = "hevc_nvenc" if hardware.nvenc_available else "libx264"
        preset = "p5" if hardware.nvenc_available else "medium"
        stream_policy = CUDAGraphManager.stream_policy(
            settings, workload,
            hardware,
            independent_work=2 if workload.faces_per_frame >= 2 else 1,
            shared_mutable_buffers=False)
        graph_readiness = self._graph_readiness(settings, workload, hardware)
        configured_provider = _short(_value(settings, "provider", "auto")).lower()
        if configured_provider in ("cpu", "cpu only", "none"):
            backend = "cpu"
        elif (hardware.tensorrt_available and not small and
              configured_provider in ("auto", "", "cuda", "tensorrt")):
            backend = "tensorrt"
        elif hardware.cuda_available and configured_provider not in ("cpu", "cpu only"):
            backend = "cuda"
        else:
            backend = "cpu"
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
        p_cores = max(0, hardware.cpu_performance_cores)
        e_cores = max(0, hardware.cpu_efficiency_cores)
        if hardware.cpu_performance_indices or hardware.cpu_efficiency_indices:
            p_capacity = len(hardware.cpu_performance_indices)
            e_capacity = len(hardware.cpu_efficiency_indices)
            cpu_p = min(worker, p_capacity)
            cpu_e = min(max(0, worker - cpu_p), e_capacity)
        elif p_cores or e_cores:
            cpu_p = min(worker, p_cores)
            cpu_e = min(max(0, worker - cpu_p), e_cores)
        else:
            cpu_p, cpu_e = worker, 0
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
            "detector_resolution": detector_resolution,
            "queue_depth": queue_depth,
            "stabilization_workers": stabilization_workers,
            "stabilization_chunk_size": chunk,
            "ort_intra_threads": 1,
            "ort_inter_threads": 1,
            "opencv_threads": opencv,
            "ffmpeg_threads": ffmpeg,
            "cuda_stream_count": stream_policy["stream_count"],
            "cuda_auxiliary_streams": stream_policy["auxiliary_streams"],
            "cuda_graph_enabled": graph_readiness["safe"],
            "cpu_performance_threads": cpu_p,
            "cpu_efficiency_threads": cpu_e,
            "cpu_distribution": distribution,
            "ram_buffer_mb": ResourceManager.frame_budget_mb(hardware, workload),
            "backend": backend,
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
            "ort_intra_threads": "cpu_ort_intra_threads",
            "ort_inter_threads": "cpu_ort_inter_threads",
            "encoder_preset": "perf_encoder_preset",
            "encoder": "output_video_codec",
            "trt_context_count": "perf_trt_pool",
            "cuda_stream_count": "cuda_stream_count",
            "cuda_auxiliary_streams": "trt_auxiliary_streams",
            "cuda_graph_enabled": "trt_cuda_graph",
            "ffmpeg_threads": "cpu_ffmpeg_threads",
            "cpu_distribution": "cpu_distribution",
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
                raw, hardware)

        values["worker_count"] = _explicit_int("max_threads", values["worker_count"])
        values["detector_pool_size"] = _explicit_int("perf_detector_pool", values["detector_pool_size"])
        values["detmask_pool_size"] = _explicit_int("perf_detmask_pool", values["detmask_pool_size"])
        values["swapper_pool_size"] = _explicit_int("perf_trt_pool", values["swapper_pool_size"])
        values["trt_context_count"] = max(
            1, min(values["trt_context_count"], values["swapper_pool_size"] or 1))
        # Enhancers share the established TRT pool.  The expression restorer
        # has its own pool and is represented separately above.
        values["enhancer_pool_size"] = values["swapper_pool_size"]
        values["expression_pool_size"] = _explicit_int("perf_expr_pool", values["expression_pool_size"])
        values["detector_resolution"] = _explicit_int("face_detector_size", values["detector_resolution"])
        values["ort_intra_threads"] = _explicit_int(
            "cpu_ort_intra_threads", values["ort_intra_threads"])
        values["ort_inter_threads"] = _explicit_int(
            "cpu_ort_inter_threads", values["ort_inter_threads"])
        if not _is_auto(settings, "perf_batch_swap"):
            batch_setting = str(_value(settings, "perf_batch_swap", "on")).strip().lower()
            if batch_setting in ("off", "0", "false", "no"):
                values["batch_size"] = values["tile_batch_size"] = 1
        if not _is_auto(settings, "cpu_opencv_threads"):
            values["opencv_threads"] = _explicit_int("cpu_opencv_threads", values["opencv_threads"])
        if not _is_auto(settings, "cpu_ffmpeg_threads"):
            values["ffmpeg_threads"] = ResourceManager.clamp(
                "ffmpeg_threads", _value(settings, "cpu_ffmpeg_threads", values["ffmpeg_threads"]))
        if not _is_auto(settings, "cuda_stream_count"):
            values["cuda_stream_count"] = ResourceManager.clamp(
                "cuda_stream_count", _value(settings, "cuda_stream_count", values["cuda_stream_count"]))
        if not _is_auto(settings, "trt_auxiliary_streams"):
            values["cuda_auxiliary_streams"] = ResourceManager.clamp(
                "cuda_auxiliary_streams", _value(settings, "trt_auxiliary_streams", values["cuda_auxiliary_streams"]))
        if not _is_auto(settings, "trt_cuda_graph"):
            values["cuda_graph_enabled"] = _bool(_value(settings, "trt_cuda_graph", False)) and graph_readiness["safe"]
        if not _is_auto(settings, "perf_encoder_preset"):
            values["encoder_preset"] = _short(_value(settings, "perf_encoder_preset", values["encoder_preset"]))
        if not _is_auto(settings, "output_video_codec"):
            values["encoder"] = _short(_value(settings, "output_video_codec", values["encoder"]))

        for name in ResourceManager.BOUNDS:
            values[name] = ResourceManager.clamp(name, values[name], hardware)
        tuning = RuntimeTuning(**values)
        return tuning, tuple(sorted(set(explicit))), tuple(sorted(set(automatic))), tuple(reasons)

    @staticmethod
    def _graph_readiness(settings: Any, workload: WorkloadProfile,
                         hardware: HardwareProfile) -> dict:
        """Keep graph capture opt-in until a fixed-shape runner owns buffers."""
        return CUDAGraphManager().readiness(settings, workload, hardware)


@dataclass(frozen=True)
class TuneMeasurement:
    """Normalized result returned by one short candidate run."""

    end_to_end_fps: float = 0.0
    peak_vram_gb: float = 0.0
    peak_ram_gb: float = 0.0
    startup_seconds: float = 0.0
    cpu_utilization_pct: float = 0.0
    gpu_utilization_pct: float = 0.0
    stable: bool = True
    quality_regression: bool = False
    metrics: dict = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Any) -> "TuneMeasurement":
        if isinstance(value, cls):
            return value
        data = dict(value or {})
        peak_vram = data.get("peak_vram_gb")
        if peak_vram is None:
            peak_vram = _number(data.get("peak_vram_mb")) / 1024.0
        peak_ram = data.get("peak_ram_gb")
        if peak_ram is None:
            peak_ram = _number(data.get("peak_rss_gb"))
        fps = data.get("end_to_end_fps", data.get("fps", 0.0))
        stable = data.get("stable", data.get("stability", True))
        if isinstance(stable, str):
            stable = stable.strip().lower() in ("1", "true", "yes", "on", "pass", "passed")
        return cls(
            end_to_end_fps=max(0.0, _number(fps)),
            peak_vram_gb=max(0.0, _number(peak_vram)),
            peak_ram_gb=max(0.0, _number(peak_ram)),
            startup_seconds=max(0.0, _number(data.get("startup_seconds", 0.0))),
            cpu_utilization_pct=max(0.0, _number(
                data.get("cpu_utilization_pct", data.get("mean_cpu_pct", 0.0)))),
            gpu_utilization_pct=max(0.0, _number(
                data.get("gpu_utilization_pct", data.get("mean_gpu_util_pct", 0.0)))),
            stable=bool(stable),
            quality_regression=_bool(data.get("quality_regression", False)),
            metrics=data,
        )


class RuntimeAutotuner:
    """Bounded staged search scored by real end-to-end throughput.

    The evaluator is deliberately injected.  The application can provide a
    warmup/full-pipeline runner, while tests and offline benchmark tools can
    provide a deterministic runner without loading models.  No combinatorial
    grid is built: each stage changes one concern from the current best and
    stops after two negligible stages or the candidate budget.
    """

    MAX_CANDIDATES = 12
    DEFAULT_WARMUP_FRAMES = 24
    # Floor only. The ACCEPTANCE threshold is measured on the live machine at
    # the start of every search -- see BASELINE_REPLICATES below -- because this
    # constant alone cannot be right on two different targets.
    #
    # WHY, measured 2026-08-31 on the RTX 4070: two runs of this same
    # deterministic search on the same machine four days apart returned
    # "0.0%, promote nothing" and "+3.59%, promote trt_context_count=1". The
    # twelve candidates of the second run spanned 5.45-5.77 fps, so its winner
    # sat inside the run-to-run band and the promotion was noise. Worse, the
    # setting it proposed halves a TensorRT context pool this project measured
    # as worth +46% at 2. A fixed 1% cannot separate signal from noise on a rig
    # whose spread is 1.6% at 60 frames, ~8% at 600, and ~15% on the 3060.
    MIN_IMPROVEMENT = 0.01
    # How many times the unchanged baseline is re-measured before any candidate
    # is tried. The observed spread across these becomes the acceptance
    # threshold, so the search adapts to the noise of the machine it is on
    # rather than to a constant chosen on one GPU.
    BASELINE_REPLICATES = 3

    _SETTING_FOR_STAGE = {
        "backend_precision": ("provider", "trt_precision"),
        "trt_concurrency": ("perf_trt_pool",),
        "batch_size": ("perf_batch_swap",),
        "cpu_threading": ("max_threads", "cpu_ort_intra_threads",
                           "cpu_ort_inter_threads", "cpu_opencv_threads",
                           "cpu_distribution"),
        "queue_buffer": (),
        "encoder": ("output_video_codec", "perf_encoder_preset"),
    }

    def _explicit(self, settings: Any, stage: str) -> bool:
        if any(not _is_auto(settings, name)
               for name in self._SETTING_FOR_STAGE.get(stage, ())):
            return True
        env_pins = {
            "trt_concurrency": ("ROOP_TRT_POOL", "ROOP_DETMASK_POOL"),
            "queue_buffer": ("ROOP_OUTPUT_QUEUE_DEPTH", "ROOP_STAB_CHUNK",
                              "ROOP_STAB_CHUNK_MB"),
            "encoder": ("ROOP_ENCODER_PRESET",),
            "backend_precision": ("ROOP_TRT_AUX_STREAMS", "ROOP_TRT_CUDA_GRAPH",
                                   "ROOP_SWAP_FP32"),
        }
        return any(os.environ.get(name) not in (None, "")
                   for name in env_pins.get(stage, ()))

    @staticmethod
    def _candidate(base: Mapping[str, Any], stage: str, **changes) -> dict:
        result = dict(base)
        result.update(changes)
        result["stage"] = stage
        return result

    def candidates(self, base: RuntimeTuning, hardware: HardwareProfile,
                   workload: WorkloadProfile, settings: Any = None) -> list[dict]:
        """Return a small ordered candidate set, excluding pinned settings."""
        initial = dict(base.as_dict())
        configured_precision = str(
            _value(settings, "trt_precision", "auto") or "auto").lower()
        initial["precision"] = (
            configured_precision if not _is_auto(settings, "trt_precision") else
            ("fp32" if hardware.vram_total_gb < 7.0 else "mixed"))
        initial["stage"] = "baseline"
        result = [initial]

        def add(stage: str, **changes):
            if self._explicit(settings, stage):
                return
            item = self._candidate(initial, stage, **changes)
            # Each stage compares to the same baseline; the evaluator's staged
            # loop applies an accepted candidate before moving to the next one.
            item_config = {key: value for key, value in item.items()
                           if key != "stage"}
            if not any({key: value for key, value in existing.items()
                        if key != "stage"} == item_config for existing in result):
                result.append(item)

        small = hardware.vram_total_gb > 0 and hardware.vram_total_gb < 7.0
        if not self._explicit(settings, "backend_precision"):
            if (hardware.tensorrt_available and not small and
                    hardware.fp16_supported):
                add("backend_precision", backend="tensorrt", precision="fp16")
            if hardware.cuda_available:
                add("backend_precision", backend="cuda", precision="fp32")
        if not small and not self._explicit(settings, "trt_concurrency"):
            for value in sorted(set((1, base.trt_context_count,
                                     min(3, max(1, base.trt_context_count + 1))))):
                add("trt_concurrency", trt_context_count=value,
                    swapper_pool_size=value)
            if hardware.vram_total_gb >= 7.0 and hardware.cuda_available:
                # Detector/mask/enhancer pools have different model footprints;
                # compare their shared-concurrency alternative as one bounded
                # candidate instead of multiplying a full Cartesian grid.
                add("trt_concurrency",
                    detector_pool_size=max(1, min(2, base.detector_pool_size)),
                    detmask_pool_size=max(1, min(2, base.detmask_pool_size)),
                    enhancer_pool_size=max(1, min(2, base.enhancer_pool_size)))
        if not self._explicit(settings, "batch_size"):
            values = (1, 2) if workload.faces_per_frame >= 2 else (1,)
            for value in values:
                add("batch_size", batch_size=value, tile_batch_size=value)
        if not self._explicit(settings, "cpu_threading"):
            worker_values = sorted(set((max(1, base.worker_count // 2),
                                        base.worker_count,
                                        min(hardware.cpu_logical_cores,
                                            base.worker_count + 2))))
            for value in worker_values:
                add("cpu_threading", worker_count=value,
                    cpu_performance_threads=value,
                    cpu_efficiency_threads=0)
            p_capacity = len(hardware.cpu_performance_indices)
            e_capacity = len(hardware.cpu_efficiency_indices)
            if p_capacity and e_capacity:
                e_limit = max(1, e_capacity // 4)
                add("cpu_threading", cpu_distribution="p_only",
                    worker_count=p_capacity,
                    cpu_performance_threads=p_capacity,
                    cpu_efficiency_threads=0)
                add("cpu_threading", cpu_distribution="p_priority_e",
                    worker_count=p_capacity + e_limit,
                    cpu_performance_threads=p_capacity,
                    cpu_efficiency_threads=e_limit)
                add("cpu_threading", cpu_distribution="p_plus_e",
                    worker_count=p_capacity + e_capacity,
                    cpu_performance_threads=p_capacity,
                    cpu_efficiency_threads=e_capacity)
        if not self._explicit(settings, "queue_buffer"):
            for value in sorted(set((1, base.queue_depth, min(4, base.queue_depth + 1)))):
                add("queue_buffer", queue_depth=value,
                    in_flight_frames=min(ResourceManager.BOUNDS["in_flight_frames"][1], value + 1),
                    ram_buffer_mb=max(base.ram_buffer_mb, 512 * value))
            if workload.stabilization_enabled:
                for value in sorted(set((max(16, base.stabilization_chunk_size // 2),
                                         base.stabilization_chunk_size,
                                         min(288, base.stabilization_chunk_size * 2)))):
                    add("queue_buffer", stabilization_chunk_size=value)
        if not self._explicit(settings, "encoder"):
            codecs = []
            if hardware.nvenc_available:
                codecs.extend(("h264_nvenc", "hevc_nvenc"))
            codecs.append("libx264")
            for codec in codecs:
                add("encoder", encoder=codec,
                    encoder_preset="p5" if codec.endswith("_nvenc") else "faster")

        return result[:self.MAX_CANDIDATES]

    @staticmethod
    def score(measurement: TuneMeasurement, hardware: HardwareProfile) -> float:
        """Score throughput after resource, stability, quality, and startup penalties."""
        fps = measurement.end_to_end_fps
        if fps <= 0.0 or not measurement.stable or measurement.quality_regression:
            return 0.0
        penalty = 0.0
        if hardware.vram_total_gb and measurement.peak_vram_gb:
            pressure = measurement.peak_vram_gb / hardware.vram_total_gb
            penalty += min(0.45, max(0.0, pressure - 0.80) * 1.5)
        if hardware.ram_total_gb and measurement.peak_ram_gb:
            pressure = measurement.peak_ram_gb / hardware.ram_total_gb
            penalty += min(0.30, max(0.0, pressure - 0.75) * 1.0)
        # Startup is amortized over a short representative run.  It is a
        # penalty, never a reason to prefer high GPU utilization by itself.
        penalty += min(0.20, measurement.startup_seconds /
                       max(1.0, measurement.startup_seconds + 60.0))
        return max(0.0, fps * (1.0 - min(0.90, penalty)))

    def tune(self, baseline: RuntimeTuning, hardware: HardwareProfile,
              workload: WorkloadProfile, settings: Any = None,
              measure=None, warmup_frames: int = DEFAULT_WARMUP_FRAMES,
              max_candidates: int = MAX_CANDIDATES) -> tuple[RuntimeTuning, dict]:
        if measure is None:
            raise ValueError("RuntimeAutotuner requires an end-to-end measure callback")
        candidates = self.candidates(baseline, hardware, workload, settings)
        candidates = candidates[:max(1, min(self.MAX_CANDIDATES, int(max_candidates)))]
        tested = []
        def safe_measure(candidate):
            try:
                return TuneMeasurement.from_mapping(measure(candidate, warmup_frames))
            except Exception as exc:
                return TuneMeasurement(
                    stable=False, metrics={"error": "%s: %s" %
                                           (type(exc).__name__, exc)})

        current = dict(baseline.as_dict())
        current["precision"] = str(
            _value(settings, "trt_precision", "auto") or "auto").lower()
        if _is_auto(settings, "trt_precision"):
            current["precision"] = "fp32" if hardware.vram_total_gb < 7.0 else "mixed"
        best = current
        best_measurement = safe_measure(current)
        best_score = self.score(best_measurement, hardware)
        tested.append(self._result(current, best_measurement, best_score))

        # Measure THIS machine's run-to-run spread on the unchanged baseline,
        # and require a candidate to beat it. Without this the search promotes
        # whichever arm the noise favoured; see the MIN_IMPROVEMENT comment.
        replicates = [(best_score, best_measurement)] if best_score > 0 else []
        extra_replicates = 0
        for _ in range(max(0, int(self.BASELINE_REPLICATES) - 1)):
            extra_replicates += 1
            replicate_measurement = safe_measure(current)
            replicate_score = self.score(replicate_measurement, hardware)
            tested.append(self._result(current, replicate_measurement,
                                       replicate_score))
            if replicate_score > 0:
                replicates.append((replicate_score, replicate_measurement))
        replicate_scores = [score for score, _ in replicates]
        if len(replicate_scores) >= 2:
            spread = (max(replicate_scores) - min(replicate_scores)) /                 max(1e-9, sum(replicate_scores) / len(replicate_scores))
            # A candidate must beat the baseline's BEST showing, not its median:
            # the baseline is what already ships, so any doubt resolves in its
            # favour. The reported figures below use the median instead, so a
            # search that promotes nothing reports no improvement.
            replicates.sort(key=lambda item: item[0])
            best_score = replicate_scores and max(replicate_scores) or best_score
            baseline_score, baseline_measurement = replicates[len(replicates) // 2]
        else:
            spread = 0.0
            baseline_score, baseline_measurement = best_score, best_measurement
        min_improvement = max(float(self.MIN_IMPROVEMENT), float(spread))
        best_measurement = baseline_measurement
        stage_order = ("backend_precision", "trt_concurrency", "batch_size",
                       "cpu_threading", "queue_buffer", "encoder")
        stagnant = 0
        for stage in stage_order:
            if stagnant >= 2 or len(tested) - extra_replicates >= len(candidates):
                break
            stage_candidates = [item for item in candidates
                                if item.get("stage") == stage]
            if not stage_candidates:
                continue
            stage_improved = False
            for item in stage_candidates:
                if len(tested) - extra_replicates >= len(candidates):
                    break
                trial = dict(best)
                trial["stage"] = stage
                trial.update({key: value for key, value in item.items()
                              if key not in ("stage", "precision")})
                if "precision" in item:
                    trial["precision"] = item["precision"]
                measurement = safe_measure(trial)
                score = self.score(measurement, hardware)
                tested.append(self._result(trial, measurement, score))
                if score > best_score * (1.0 + min_improvement):
                    best, best_measurement, best_score = trial, measurement, score
                    stage_improved = True
            stagnant = 0 if stage_improved else stagnant + 1

        # CONFIRMATION RUN. A three-sample spread under-estimates the real one
        # (measured 2026-08-31 on the RTX 4070: replicates spanned 1.64% while
        # the candidate population spanned 5.11-5.67 fps), so the winner is
        # re-measured and must clear the bar a SECOND time. One extra run per
        # search is the cheapest thing that separates "faster" from "was lucky",
        # and it is what stopped this search promoting trt_context_count=1 --
        # halving a pool this project measured as worth +46% at 2.
        confirmation = None
        if best is not current and best != current:
            confirmation_measurement = safe_measure(best)
            confirmation_score = self.score(confirmation_measurement, hardware)
            tested.append(self._result(best, confirmation_measurement,
                                       confirmation_score))
            confirmed = confirmation_score > baseline_score * (1.0 + min_improvement)
            confirmation = {
                "fps": confirmation_measurement.end_to_end_fps,
                "score": round(confirmation_score, 6),
                "confirmed": bool(confirmed),
            }
            if not confirmed:
                best, best_measurement = current, baseline_measurement
                best_score = baseline_score

        tuning_values = {name: best[name] for name in RuntimeTuning.__dataclass_fields__
                         if name in best}
        tuning = RuntimeTuning(**tuning_values)
        # The median of the baseline replicates, not the first one: with three
        # runs of the same configuration the first is as likely as any other to
        # be the slow one, and dividing by it manufactures an improvement.
        baseline_fps = baseline_measurement.end_to_end_fps
        report = {
            "mode": "bounded_staged_warmup",
            "baseline_replicates": len(replicate_scores),
            "confirmation": confirmation,
            "measured_noise_spread": round(spread, 6),
            "min_improvement_used": round(min_improvement, 6),
            "warmup_frames": max(1, int(warmup_frames)),
            "candidate_budget": len(candidates),
            "candidates_tested": tested,
            "selected": dict(best),
            "baseline_fps": baseline_fps,
            "best_fps": best_measurement.end_to_end_fps,
            "improvement_pct": ((best_measurement.end_to_end_fps - baseline_fps) /
                                 baseline_fps * 100.0 if baseline_fps else 0.0),
            "stopped_after_stagnant_stages": stagnant,
        }
        return tuning, report

    @staticmethod
    def _result(candidate: Mapping[str, Any], measurement: TuneMeasurement,
                score: float) -> dict:
        return {"stage": candidate.get("stage", "trial"),
                "configuration": dict(candidate),
                "measurement": {
                    "end_to_end_fps": measurement.end_to_end_fps,
                    "peak_vram_gb": measurement.peak_vram_gb,
                    "peak_ram_gb": measurement.peak_ram_gb,
                    "startup_seconds": measurement.startup_seconds,
                    "cpu_utilization_pct": measurement.cpu_utilization_pct,
                    "gpu_utilization_pct": measurement.gpu_utilization_pct,
                    "stable": measurement.stable,
                    "quality_regression": measurement.quality_regression,
                    "score": score,
                }}


class ProfileStore:
    """Small atomic JSON store for measured hardware/workload profiles."""

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

    def invalidate(self, key: Optional[str] = None) -> int:
        """Invalidate one profile or all profiles in this store."""
        removed = 0
        paths = [self.directory / f"{key}.json"] if key else list(self.directory.glob("*.json"))
        for path in paths:
            try:
                path.unlink()
                removed += 1
            except FileNotFoundError:
                pass
            except OSError:
                continue
        return removed

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
    """Optional rolling telemetry for the real end-to-end pipeline.

    The normal path performs only the ``enabled`` branch in ``record_frame``.
    Resource queries and aggregation happen at a coarse interval, never per
    frame, and use non-blocking APIs.  Stage timing is supplied by the existing
    ``procmgr_runtime._prof`` hook, so enabling this monitor does not require a
    second pass over frames.
    """

    STAGES = ("decode", "detect", "swap", "enhance", "stabilize", "encode")

    def __init__(self, hardware: Optional[HardwareProfile] = None,
                 tuning: Optional[RuntimeTuning] = None, settings: Any = None,
                 enabled: Optional[bool] = None,
                 diagnostics: Optional[bool] = None):
        self.hardware = hardware
        self.tuning = tuning
        self.settings = settings
        self.enabled = (_bool(os.environ.get("ROOP_RUNTIME_MONITOR", "0"))
                        if enabled is None else bool(enabled))
        self.diagnostics = (_bool(os.environ.get("ROOP_RUNTIME_DIAGNOSTICS", "0"))
                            if diagnostics is None else bool(diagnostics))
        self.sample_interval = max(0.25, _number(
            os.environ.get("ROOP_RUNTIME_SAMPLE_INTERVAL", "1.0"), 1.0))
        self.window_size = max(4, min(64, _integer(
            os.environ.get("ROOP_RUNTIME_WINDOW", "16"), 16)))
        self.started_at = 0.0
        self._last_sample_at = 0.0
        self._last_stage_at = 0.0
        self._frames = 0
        self._stage_seconds = {}
        self._stage_calls = {}
        self._last_stage_seconds = {}
        self._last_stage_calls = {}
        self._samples = deque(maxlen=self.window_size)
        self._lock = threading.Lock()

    @property
    def samples(self) -> list:
        # Retain the old public shape for diagnostics and callers that inspect
        # the monitor, while keeping storage bounded by a deque.
        with self._lock:
            return list(self._samples)

    def start(self) -> None:
        now = time.perf_counter()
        with self._lock:
            self.started_at = now
            self._last_sample_at = now
            self._last_stage_at = now
            self._frames = 0
            self._stage_seconds.clear()
            self._stage_calls.clear()
            self._last_stage_seconds.clear()
            self._last_stage_calls.clear()
            self._samples.clear()
        self._start_sampler()

    def _start_sampler(self) -> None:
        """Sample on our own timer instead of relying on the pipeline.

        Periodic sampling used to happen only inside ``record_frame``, which
        the render path never calls.  A run therefore produced ONE sample,
        taken by ``finish(force=True)``, and that single sample is why:

        * ``cpu_utilization_pct`` read 0.0 -- a process CPU delta needs two
          reads separated in time, and there was only ever one;
        * the rolling window never filled, so ``SafeAdaptiveController``'s
          three-consecutive-window requirement could not be met and the
          adaptive controller was inert on BOTH validation GPUs.

        The thread is a daemon, holds no GPU resource, and only reads
        counters, so it cannot interfere with in-flight inference.
        """
        if not self.enabled:
            return
        if getattr(self, "_sampler_thread", None) is not None:
            return
        self._sampler_stop = threading.Event()

        def _loop():
            # Prime the process CPU delta before the first real reading.
            self._resource_snapshot()
            while not self._sampler_stop.wait(self.sample_interval):
                try:
                    self.sample(force=True)
                except Exception:
                    # Telemetry must never take down a render.
                    pass

        thread = threading.Thread(target=_loop, name="roop-runtime-monitor",
                                  daemon=True)
        self._sampler_thread = thread
        thread.start()

    def _stop_sampler(self) -> None:
        stop = getattr(self, "_sampler_stop", None)
        thread = getattr(self, "_sampler_thread", None)
        if stop is not None:
            stop.set()
        if thread is not None:
            thread.join(timeout=2.0)
        self._sampler_thread = None

    def record_stage(self, stage: str, elapsed: float, calls: int = 1) -> None:
        if not self.enabled:
            return
        key = str(stage)
        with self._lock:
            self._stage_seconds[key] = self._stage_seconds.get(key, 0.0) + max(0.0, float(elapsed))
            self._stage_calls[key] = self._stage_calls.get(key, 0) + max(0, int(calls))

    def observe(self, **metrics: Any) -> None:
        """Add a caller-supplied rolling sample without probing the machine."""
        if not self.enabled:
            return
        sample = {"time": time.time()}
        sample.update(metrics)
        with self._lock:
            self._samples.append(sample)

    @staticmethod
    def _queue_depths(queues: Any) -> dict:
        if queues is None:
            return {}
        if isinstance(queues, Mapping):
            items = queues.items()
        else:
            try:
                items = enumerate(queues)
            except TypeError:
                return {}
        result = {}
        for name, queue in items:
            try:
                result[str(name)] = (int(queue.qsize()) if hasattr(queue, "qsize")
                                     else int(queue))
            except Exception:
                continue
        return result

    @staticmethod
    def _average(values: list) -> Optional[float]:
        numbers = [_number(value, float("nan")) for value in values]
        numbers = [value for value in numbers if value == value]
        return (sum(numbers) / len(numbers)) if numbers else None

    def _resource_snapshot(self) -> dict:
        result = {}
        try:
            import psutil
            # The Process handle MUST persist across snapshots.  psutil
            # computes cpu_percent(None) as a delta against that object's own
            # previous call, so a freshly constructed Process always returns
            # 0.0 -- which is why this field read a measured-looking 0.0% on
            # both validation GPUs while the module-level percpu call (which
            # keeps its own global state) correctly reported P/E utilization.
            process = getattr(self, "_psutil_process", None)
            if process is None or process.pid != os.getpid():
                process = psutil.Process(os.getpid())
                self._psutil_process = process
                process.cpu_percent(None)  # prime the delta baseline
            logical = max(1, int(psutil.cpu_count(logical=True) or 1))
            cpu_pct = float(process.cpu_percent(None)) / logical
            result["cpu_utilization_pct"] = min(100.0, max(0.0, cpu_pct))
            process_memory = process.memory_info()
            result["ram_used_gb"] = process_memory.rss / 2**30
            result["process_rss_gb"] = process_memory.rss / 2**30
            memory = psutil.virtual_memory()
            result["ram_total_gb"] = memory.total / 2**30
            result["ram_available_gb"] = memory.available / 2**30
            result["ram_used_system_gb"] = memory.used / 2**30
            result["ram_utilization_pct"] = float(memory.percent)
            # psutil does not expose Windows' exact committed-private counter
            # portably.  This is the useful cross-platform estimate: resident
            # system use plus committed swap/pagefile use.  Keep the source in
            # the report so a platform-specific provider can replace it later
            # without presenting an inferred value as an exact counter.
            swap = psutil.swap_memory()
            commit_limit = memory.total + swap.total
            committed = memory.used + swap.used
            result["ram_committed_gb"] = committed / 2**30
            result["ram_commit_limit_gb"] = commit_limit / 2**30
            result["ram_commit_source"] = "physical_used_plus_swap_used"
            result["swap_used_gb"] = swap.used / 2**30
            result["swap_total_gb"] = swap.total / 2**30
            result["swap_utilization_pct"] = (
                100.0 * swap.used / swap.total if swap.total else 0.0)
            per_cpu = psutil.cpu_percent(interval=None, percpu=True)
            p_indices = list(getattr(self.hardware,
                                     "cpu_performance_indices", ()) or
                             self._indices_from_env("ROOP_CPU_P_INDICES"))
            e_indices = list(getattr(self.hardware,
                                     "cpu_efficiency_indices", ()) or
                             self._indices_from_env("ROOP_CPU_E_INDICES"))
            if p_indices:
                result["p_core_utilization_pct"] = self._average(
                    [per_cpu[index] for index in p_indices if index < len(per_cpu)])
            if e_indices:
                result["e_core_utilization_pct"] = self._average(
                    [per_cpu[index] for index in e_indices if index < len(per_cpu)])
            frequency = psutil.cpu_freq()
            if frequency is not None:
                result["cpu_frequency_mhz"] = float(getattr(frequency, "current", 0.0) or 0.0)
                result["cpu_max_frequency_mhz"] = float(getattr(frequency, "max", 0.0) or 0.0)
            temperatures = getattr(psutil, "sensors_temperatures", lambda: {})()
            cpu_temps = []
            for entries in (temperatures or {}).values():
                for entry in entries:
                    label = str(getattr(entry, "label", "") or "").lower()
                    if any(token in label for token in ("cpu", "package", "core", "tctl", "tdie")):
                        current = _number(getattr(entry, "current", 0.0))
                        if current > 0:
                            cpu_temps.append(current)
            if cpu_temps:
                result["cpu_temperature_c"] = max(cpu_temps)
        except Exception:
            pass

        try:
            import torch
            device_id = int(getattr(self.hardware, "device_id", 0) or 0)
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info(device_id)
                result["vram_free_gb"] = int(free_b) / 2**30
                result["vram_total_gb"] = int(total_b) / 2**30
        except Exception:
            pass

        # NVML is optional. It gives utilization without synchronizing CUDA or
        # invoking a subprocess. If it is not installed, memory telemetry still
        # works and GPU utilization is reported as unavailable.
        try:
            import pynvml
            pynvml.nvmlInit()
            device_id = int(getattr(self.hardware, "device_id", 0) or 0)
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            result["gpu_utilization_pct"] = float(
                pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            result.setdefault("vram_free_gb", memory.free / 2**30)
            result.setdefault("vram_total_gb", memory.total / 2**30)
        except Exception:
            pass
        if result.get("gpu_utilization_pct") is None:
            # pynvml is genuinely optional and is absent on this stack, which
            # left GPU utilization permanently None -- and a None GPU reading
            # silently disabled the GPU-bound and synchronization-bound
            # branches of the bottleneck classifier.  nvidia-smi answers the
            # same question without a new dependency.  It is a subprocess, so
            # it is rate-limited well below the sampling interval and never
            # runs per frame.
            value = self._gpu_utilization_from_smi()
            if value is not None:
                result["gpu_utilization_pct"] = value
        if result.get("vram_total_gb"):
            result["vram_pressure_pct"] = max(0.0, min(100.0,
                100.0 * (1.0 - result["vram_free_gb"] / result["vram_total_gb"])))
        return result

    _SMI_MIN_INTERVAL = 2.0

    def _gpu_utilization_from_smi(self) -> Optional[float]:
        """GPU utilization via nvidia-smi, rate-limited and never fatal."""
        now = time.perf_counter()
        last = getattr(self, "_smi_last_time", 0.0)
        if last and (now - last) < self._SMI_MIN_INTERVAL:
            return getattr(self, "_smi_last_value", None)
        self._smi_last_time = now
        try:
            import subprocess
            device_id = int(getattr(self.hardware, "device_id", 0) or 0)
            out = subprocess.run(
                ["nvidia-smi", f"--id={device_id}",
                 "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=False).stdout
            for line in (out or "").splitlines():
                line = line.strip()
                if line and line[0].isdigit():
                    self._smi_last_value = float(line)
                    return self._smi_last_value
        except Exception:
            pass
        self._smi_last_value = None
        return None

    @staticmethod
    def _indices_from_env(name: str) -> list:
        values = []
        for item in os.environ.get(name, "").replace(";", ",").split(","):
            try:
                if item.strip():
                    values.append(max(0, int(item.strip())))
            except ValueError:
                continue
        return values

    def _stage_delta(self) -> tuple:
        with self._lock:
            seconds = {}
            calls = {}
            for name, value in self._stage_seconds.items():
                seconds[name] = max(0.0, value - self._last_stage_seconds.get(name, 0.0))
            for name, value in self._stage_calls.items():
                calls[name] = max(0, value - self._last_stage_calls.get(name, 0))
            self._last_stage_seconds = dict(self._stage_seconds)
            self._last_stage_calls = dict(self._stage_calls)
        return seconds, calls

    def sample(self, queue_depths: Any = None,
               worker_utilization_pct: Optional[float] = None,
               force: bool = False) -> Optional[dict]:
        if not self.enabled:
            return None
        now = time.perf_counter()
        with self._lock:
            if not force and now - self._last_sample_at < self.sample_interval:
                return None
            previous = self._last_sample_at or now
            self._last_sample_at = now
            frames = self._frames
        stage_seconds, stage_calls = self._stage_delta()
        sample = {
            "time": time.time(),
            "interval_sec": max(0.0, now - previous),
            "frames": frames,
            "queue_depths": self._queue_depths(queue_depths),
            "worker_utilization_pct": worker_utilization_pct,
            "stage_seconds": stage_seconds,
            "stage_calls": stage_calls,
        }
        sample.update(self._resource_snapshot())
        with self._lock:
            self._samples.append(sample)
        return sample

    def record_frame(self, queue_depths: Any = None,
                     worker_utilization_pct: Optional[float] = None) -> Optional[dict]:
        if not self.enabled:
            return None
        with self._lock:
            self._frames += 1
        return self.sample(queue_depths, worker_utilization_pct)

    def finish(self, queue_depths: Any = None,
               worker_utilization_pct: Optional[float] = None) -> dict:
        if self.enabled:
            self.sample(queue_depths, worker_utilization_pct, force=True)
        self._stop_sampler()
        return self.summary()

    def _rolling_metric(self, name: str) -> Optional[float]:
        with self._lock:
            return self._average([sample.get(name) for sample in self._samples])

    def summary(self) -> dict:
        with self._lock:
            samples = list(self._samples)
            started = self.started_at
            frames = self._frames
            stage_seconds = dict(self._stage_seconds)
            stage_calls = dict(self._stage_calls)
        # ``samples`` is intentionally a bounded rolling diagnostics window,
        # not the source of truth for totals. Summing it made long renders
        # report only the last N sampling intervals, producing misleading
        # stage FPS/latency and occasionally zero totals after sparse sampling.
        # The cumulative counters cover the complete run; the samples remain
        # available for recent resource/bottleneck inspection.
        elapsed = time.perf_counter() - started if started else 0.0
        stage_fps = {name: calls / max(elapsed, 1e-9)
                     for name, calls in stage_calls.items()}
        stage_latency = {name: 1000.0 * stage_seconds[name] / max(stage_calls.get(name, 1), 1)
                         for name in stage_seconds}
        queue_averages = {}
        for sample in samples:
            for name, value in (sample.get("queue_depths") or {}).items():
                queue_averages.setdefault(name, []).append(value)
        queue_averages = {name: self._average(values)
                          for name, values in queue_averages.items()}
        commit_sources = [sample.get("ram_commit_source") for sample in samples
                          if sample.get("ram_commit_source")]
        result = {
            "elapsed_sec": round(elapsed, 3),
            "frames": frames,
            # ``frames`` is only non-zero when the pipeline calls
            # ``record_frame``. When it does not, ``frames/elapsed`` is 0.0 --
            # a number that reads as a measured throughput of zero. Fall back
            # to the frame_total stage, which counts the same frames and is
            # populated, and report None rather than 0.0 when neither exists.
            "end_to_end_fps": (frames / max(elapsed, 1e-9) if (elapsed and frames)
                               else (stage_fps.get("frame_total") or None)),
            "stage_fps": stage_fps,
            "decode_fps": stage_fps.get("decode", 0.0),
            "detection_fps": stage_fps.get("detect", 0.0),
            "swap_fps": stage_fps.get("swap", 0.0),
            "enhancement_fps": stage_fps.get("enhance", 0.0),
            "stabilization_fps": stage_fps.get("stabilize", 0.0),
            "encode_fps": stage_fps.get("encode", 0.0),
            "stage_seconds": stage_seconds,
            "stage_latency_ms": stage_latency,
            "queue_depths": queue_averages,
            "samples": samples,
            "cpu_utilization_pct": self._rolling_metric("cpu_utilization_pct"),
            "p_core_utilization_pct": self._rolling_metric("p_core_utilization_pct"),
            "e_core_utilization_pct": self._rolling_metric("e_core_utilization_pct"),
            "gpu_utilization_pct": self._rolling_metric("gpu_utilization_pct"),
            "vram_free_gb": self._rolling_metric("vram_free_gb"),
            "vram_total_gb": self._rolling_metric("vram_total_gb"),
            "vram_pressure_pct": self._rolling_metric("vram_pressure_pct"),
            "ram_used_gb": self._rolling_metric("ram_used_gb"),
            "process_rss_gb": self._rolling_metric("process_rss_gb"),
            "ram_total_gb": self._rolling_metric("ram_total_gb"),
            "ram_available_gb": self._rolling_metric("ram_available_gb"),
            "ram_used_system_gb": self._rolling_metric("ram_used_system_gb"),
            "ram_utilization_pct": self._rolling_metric("ram_utilization_pct"),
            "ram_committed_gb": self._rolling_metric("ram_committed_gb"),
            "ram_commit_limit_gb": self._rolling_metric("ram_commit_limit_gb"),
            "ram_commit_source": commit_sources[-1] if commit_sources else None,
            "swap_used_gb": self._rolling_metric("swap_used_gb"),
            "swap_total_gb": self._rolling_metric("swap_total_gb"),
            "swap_utilization_pct": self._rolling_metric("swap_utilization_pct"),
            "worker_utilization_pct": self._rolling_metric("worker_utilization_pct"),
        }
        result["bottleneck"] = self.classify_bottleneck(result)
        return result

    @staticmethod
    def _has_telemetry(summary: Mapping[str, Any]) -> bool:
        """Whether the queue/utilization signals were actually reported.

        The pipeline supplies queue depths, worker utilization and CPU/GPU
        utilization only when it is instrumented to.  When it is not, those
        fields arrive as 0.0/None -- which is indistinguishable from a real
        measurement of "idle" unless it is checked explicitly.
        """
        if summary.get("gpu_utilization_pct") is not None:
            return True
        if summary.get("cpu_utilization_pct") is not None:
            return True
        if summary.get("worker_utilization_pct") is not None:
            return True
        return bool(summary.get("queue_depths"))

    @staticmethod
    def _dominant_stage(summary: Mapping[str, Any]) -> Optional[str]:
        """Return the costliest non-aggregate stage, or None."""
        stage_seconds = summary.get("stage_seconds") or {}
        # ``frame_total`` is the sum of the others; ranking against it would
        # always name the aggregate.
        aggregates = {"frame_total"}
        ranked = sorted(((_number(value), name)
                         for name, value in stage_seconds.items()
                         if name not in aggregates),
                        reverse=True)
        return ranked[0][1] if ranked else None

    @staticmethod
    def classify_bottleneck(summary: Mapping[str, Any]) -> str:
        """Classify from the rolling aggregate, never from one frame.

        Returns an explicit ``unknown``/``stage-bound`` answer when the
        signals a verdict would need were never reported.  The previous
        implementation fell through to ``"I/O-bound"`` whenever both queue
        depths were 0.0, which is also what an UNINSTRUMENTED run looks like:
        measured on both validation GPUs, every run reported "I/O-bound" while
        decode cost 3.3 ms of a 244.8 ms frame.  Absence of evidence must not
        be rendered as a confident diagnosis.
        """
        vram = _number(summary.get("vram_pressure_pct"), 0.0)
        ram = _number(summary.get("ram_utilization_pct"), 0.0)
        if vram >= 88.0:
            return "VRAM-bound"
        if ram >= 90.0:
            return "RAM-bound"
        stage_seconds = summary.get("stage_seconds") or {}
        shares = {name: _number(value) for name, value in stage_seconds.items()}
        # ``0.0 >= 0.0 * 0.45`` is True, so without this guard an empty or
        # all-zero stage table reported a confident "encode-bound".
        peak = max(shares.values()) if shares else 0.0
        if peak > 0.0:
            if shares.get("encode", 0.0) >= peak * 0.45:
                return "encode-bound"
            if shares.get("decode", 0.0) >= peak * 0.45:
                return "decode-bound"
        gpu = summary.get("gpu_utilization_pct")
        cpu = _number(summary.get("cpu_utilization_pct"), 0.0)
        queues = summary.get("queue_depths") or {}
        input_q = _number(queues.get("input", queues.get("0")), 0.0)
        output_q = _number(queues.get("output", queues.get("1")), 0.0)
        if input_q <= 0.0 and output_q <= 0.0 and gpu is not None and _number(gpu) < 45.0:
            return "synchronization-bound"
        if gpu is not None and _number(gpu) >= 80.0:
            return "GPU-bound"
        if cpu >= 80.0:
            return "CPU-bound"
        if not RuntimeMonitor._has_telemetry(summary):
            # No queue, worker or CPU/GPU utilization was ever reported. The
            # per-stage timings are still trustworthy, so name the stage that
            # dominates instead of inventing a system-level verdict.
            stage = RuntimeMonitor._dominant_stage(summary)
            if stage:
                return "stage-bound:%s (no queue/utilization telemetry)" % stage
            return "unknown (insufficient telemetry)"
        if input_q <= 0.0 and output_q <= 0.0:
            return "I/O-bound"
        return "synchronization-bound"


class SafeAdaptiveController:
    """Hysteretic, boundary-only controller for future work.

    It returns a new immutable ``RuntimeTuning``. It never edits a live pool,
    destroys a TensorRT context, or asks CUDA to synchronize. Queue geometry,
    encoder changes, and stabilization chunk changes are marked for the next
    safe boundary/run; batch and in-flight limits can be consumed by the next
    work item without touching in-flight inference.
    """

    REQUIRED_WINDOWS = 3
    COOLDOWN_WINDOWS = 8

    def __init__(self, hardware: HardwareProfile, tuning: RuntimeTuning,
                 settings: Any = None, enabled: Optional[bool] = None,
                 log=None):
        self.hardware = hardware
        self.tuning = tuning
        self.settings = settings
        self.enabled = (_bool(os.environ.get("ROOP_RUNTIME_ADAPTIVE", "0"))
                        if enabled is None else bool(enabled))
        self.log = log
        self._streak = {}
        self._cooldown = 0
        self.actions = []

    def _eligible(self, name: str) -> bool:
        return _is_auto(self.settings, name)

    def _record(self, action: dict) -> dict:
        self.actions.append(action)
        self._cooldown = self.COOLDOWN_WINDOWS
        if self.log:
            self.log(action)
        return action

    def update(self, summary: Mapping[str, Any], safe_boundary: bool = False) -> Optional[dict]:
        if not self.enabled or not safe_boundary:
            return None
        if self._cooldown:
            self._cooldown -= 1
            return None
        bottleneck = str(summary.get("bottleneck", ""))
        vram = _number(summary.get("vram_pressure_pct"), 0.0)
        ram = _number(summary.get("ram_utilization_pct"), 0.0)
        gpu = _number(summary.get("gpu_utilization_pct"), -1.0)
        starvation = bottleneck in ("synchronization-bound", "I/O-bound") and gpu >= 0 and gpu < 45.0
        condition = {
            "vram": vram >= 88.0,
            "ram": ram >= 90.0,
            "gpu_starvation": starvation,
            "encode": bottleneck == "encode-bound",
        }
        key = next((name for name, active in condition.items() if active), "")
        for name in tuple(self._streak):
            if name != key:
                self._streak[name] = 0
        if not key:
            return None
        self._streak[key] = self._streak.get(key, 0) + 1
        if self._streak[key] < self.REQUIRED_WINDOWS:
            return None

        changes = {}
        scope = "next_work"
        reason = key
        if key == "vram":
            if self._eligible("perf_batch_swap"):
                changes["batch_size"] = max(
                    ResourceManager.BOUNDS["batch_size"][0], self.tuning.batch_size - 1)
            if self._eligible("perf_tile_batch"):
                changes["tile_batch_size"] = max(
                    ResourceManager.BOUNDS["tile_batch_size"][0], self.tuning.tile_batch_size - 1)
            changes["in_flight_frames"] = max(
                ResourceManager.BOUNDS["in_flight_frames"][0], self.tuning.in_flight_frames - 1)
        elif key == "ram":
            changes["in_flight_frames"] = max(
                ResourceManager.BOUNDS["in_flight_frames"][0], self.tuning.in_flight_frames - 1)
            if self._eligible("queue_depth"):
                changes["queue_depth"] = max(
                    ResourceManager.BOUNDS["queue_depth"][0], self.tuning.queue_depth - 1)
            changes["ram_buffer_mb"] = max(
                ResourceManager.BOUNDS["ram_buffer_mb"][0], self.tuning.ram_buffer_mb - 256)
            scope = "next_boundary"
        elif key == "gpu_starvation":
            changes["in_flight_frames"] = min(
                ResourceManager.BOUNDS["in_flight_frames"][1], self.tuning.in_flight_frames + 1)
            if self._eligible("perf_batch_swap"):
                changes["batch_size"] = min(
                    ResourceManager.BOUNDS["batch_size"][1], self.tuning.batch_size + 1)
            if self._eligible("queue_depth"):
                changes["queue_depth"] = min(
                    ResourceManager.BOUNDS["queue_depth"][1], self.tuning.queue_depth + 1)
            scope = "next_work"
        elif key == "encode":
            if not self._eligible("output_video_codec"):
                return None
            if self.hardware.nvenc_available and self.tuning.encoder == "libx264":
                changes["encoder"] = "h264_nvenc"
            elif self._eligible("perf_encoder_preset"):
                changes["encoder_preset"] = "fast"
            scope = "next_segment"
        if not changes:
            return None
        bounded = {}
        for name, value in changes.items():
            if name in ResourceManager.BOUNDS:
                bounded[name] = ResourceManager.clamp(name, value)
            else:
                bounded[name] = value
        self.tuning = replace(self.tuning, **bounded)
        return self._record({
            "reason": reason,
            "bottleneck": bottleneck,
            "changes": bounded,
            "scope": scope,
            "safe_boundary": True,
            "cooldown_windows": self.COOLDOWN_WINDOWS,
        })

    def snapshot(self) -> dict:
        return {"enabled": self.enabled, "tuning": self.tuning.as_dict(),
                "actions": list(self.actions), "cooldown_windows": self._cooldown}


# Descriptive alias for integrations that refer to the controller by its role.
RuntimeAdaptiveController = SafeAdaptiveController


class RuntimeOptimizer:
    """Facade for cached profiles and explicit, bounded runtime retunes."""

    def __init__(self, settings: Any = None, device_id: int = 0,
                 profile_dir: Optional[str] = None):
        self.settings = settings
        self.hardware_profiler = HardwareProfiler(device_id)
        self.workload_profiler = WorkloadProfiler()
        self.precision_selector = PrecisionSelector()
        self.engine_manager = TensorRTEngineManager()
        self.cuda_graphs = CUDAGraphManager()
        self.auto_tuner = AutoTuner()
        self.runtime_autotuner = RuntimeAutotuner()
        self.store = ProfileStore(profile_dir)
        self.monitor = RuntimeMonitor()

    def build_profile(self, workload: WorkloadProfile,
                      save: bool = False, use_cache: bool = True,
                      force: bool = False) -> RuntimeProfile:
        hardware = self.hardware_profiler.profile()
        model_key = (workload.enhancement_model or
                     _value(self.settings, "swap_model", "") or "")
        precision = self.precision_selector.select(self.settings, hardware,
                                                   model_key=model_key)
        key = self.engine_manager.cache_key(hardware, workload, self.settings, precision)
        invalidate = os.environ.get("ROOP_RUNTIME_PROFILE_INVALIDATE", "").strip().lower()
        force = force or invalidate in ("1", "true", "yes", "on", "all")
        if force and invalidate in ("1", "true", "yes", "on"):
            self.store.invalidate(key)
        if use_cache and not force:
            cached = RuntimeProfile.from_dict(self.store.load(key) or {})
            if cached is not None and cached.cache_key == key:
                return cached
        tuning, explicit, automatic, reasons = self.auto_tuner.tune(
            hardware, workload, self.settings)
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
                      face_count: int = 0,
                      estimated_face_size_px: float = 0.0,
                      save: bool = True, force: bool = False) -> RuntimeProfile:
        workload = self.workload_profiler.profile(
            source_video=source_video, settings=self.settings,
            frame_count=frame_count, resolution=resolution,
            output_resolution=output_resolution,
            faces_per_frame=faces_per_frame, face_count=face_count,
            estimated_face_size_px=estimated_face_size_px)
        return self.build_profile(workload, save=save, force=force)

    def autotune_profile(self, workload: WorkloadProfile, measure,
                         warmup_frames: int = RuntimeAutotuner.DEFAULT_WARMUP_FRAMES,
                         max_candidates: int = RuntimeAutotuner.MAX_CANDIDATES,
                         save: bool = True, force: bool = False) -> RuntimeProfile:
        """Run one bounded staged retune and persist its measured winner.

        ``measure`` must execute the real workload (or a representative
        warmup) and return end-to-end FPS plus resource/stability/quality
        metrics.  It is intentionally required so a caller cannot mistake an
        isolated GPU-utilization probe for a pipeline optimization.
        """
        base = self.build_profile(workload, save=False, use_cache=False, force=True)
        if not force:
            cached = RuntimeProfile.from_dict(self.store.load(base.cache_key) or {})
            if cached is not None and cached.autotune:
                return cached
        tuning, report = self.runtime_autotuner.tune(
            base.tuning, base.hardware, workload, self.settings, measure,
            warmup_frames=warmup_frames, max_candidates=max_candidates)
        selected = report.get("selected", {})
        profile = replace(
            base,
            tuning=tuning,
            precision=str(selected.get("precision", base.precision)),
            provider=str(selected.get("backend", base.provider)),
            reasons=tuple(base.reasons) + ("bounded end-to-end autotune measured and cached",),
            autotune=report,
        )
        if save:
            self.store.save(profile)
        return profile

    def autotune_profile_video(self, source_video: str, measure,
                               frame_count: int = 0,
                               resolution: Optional[Tuple[int, int]] = None,
                               faces_per_frame: Optional[float] = None,
                               output_resolution: Optional[Tuple[int, int]] = None,
                               face_count: int = 0,
                               estimated_face_size_px: float = 0.0,
                               warmup_frames: int = RuntimeAutotuner.DEFAULT_WARMUP_FRAMES,
                               max_candidates: int = RuntimeAutotuner.MAX_CANDIDATES,
                               save: bool = True, force: bool = False) -> RuntimeProfile:
        workload = self.workload_profiler.profile(
            source_video=source_video, settings=self.settings,
            frame_count=frame_count, resolution=resolution,
            output_resolution=output_resolution,
            faces_per_frame=faces_per_frame, face_count=face_count,
            estimated_face_size_px=estimated_face_size_px)
        return self.autotune_profile(
            workload, measure, warmup_frames=warmup_frames,
            max_candidates=max_candidates, save=save, force=force)

    def invalidate_profiles(self, cache_key: Optional[str] = None) -> int:
        """Support manual profile invalidation without touching user settings."""
        return self.store.invalidate(cache_key)


    @staticmethod
    def apply_environment(profile: RuntimeProfile, settings: Any = None) -> dict:
        """Apply only non-import-time, safe runtime hints for automatic fields.

        Existing explicit environment variables are never replaced.  The
        current pipeline may consume these hints incrementally; unused hints
        are harmless and make the profile visible to future stages.
        """
        tuning = profile.tuning
        env_values = {
            # These are consumed by ProcessMgr/session_pool/face_util during
            # this run.  They are runtime hints, not user-setting rewrites.
            "ROOP_RUNTIME_WORKER_COUNT": tuning.worker_count,
            "ROOP_RUNTIME_DETECTOR_POOL": tuning.detector_pool_size,
            "ROOP_RUNTIME_DETMASK_POOL": tuning.detmask_pool_size,
            "ROOP_RUNTIME_QUEUE_DEPTH": tuning.queue_depth,
            "ROOP_RUNTIME_STABILIZATION_WORKERS": tuning.stabilization_workers,
            "ROOP_RUNTIME_STAB_CHUNK": tuning.stabilization_chunk_size,
            "ROOP_RUNTIME_ORT_INTRA_THREADS": tuning.ort_intra_threads,
            "ROOP_RUNTIME_ORT_INTER_THREADS": tuning.ort_inter_threads,
            "ROOP_RUNTIME_CV_THREADS": tuning.opencv_threads,
            "ROOP_RUNTIME_FFMPEG_THREADS": tuning.ffmpeg_threads,
            "ROOP_RUNTIME_DETECTOR_RESOLUTION": tuning.detector_resolution,
            "ROOP_RUNTIME_BATCH_SIZE": tuning.batch_size,
            "ROOP_RUNTIME_TILE_BATCH_SIZE": tuning.tile_batch_size,
            "ROOP_RUNTIME_FACE_CONCURRENCY": tuning.face_concurrency,
            "ROOP_RUNTIME_INFLIGHT_FRAMES": tuning.in_flight_frames,
            "ROOP_RUNTIME_CUDA_STREAMS": tuning.cuda_stream_count,
            "ROOP_RUNTIME_TRT_AUX_STREAMS": tuning.cuda_auxiliary_streams,
            "ROOP_RUNTIME_CUDA_GRAPH": int(tuning.cuda_graph_enabled),
            "ROOP_RUNTIME_CPU_P_THREADS": tuning.cpu_performance_threads,
            "ROOP_RUNTIME_CPU_E_THREADS": tuning.cpu_efficiency_threads,
            "ROOP_CPU_DISTRIBUTION": tuning.cpu_distribution,
            "ROOP_RUNTIME_RAM_BUFFER_MB": tuning.ram_buffer_mb,
            "ROOP_RUNTIME_ENCODER": tuning.encoder,
            "ROOP_RUNTIME_ENCODER_PRESET": tuning.encoder_preset,
            "ROOP_RUNTIME_BACKEND": tuning.backend,
            # Existing session-pool consumers use these public names. They are
            # only published when the matching user setting is automatic.
            "ROOP_TRT_POOL": tuning.swapper_pool_size,
            "ROOP_DETMASK_POOL": tuning.detmask_pool_size,
            "ROOP_EXPR_POOL": tuning.expression_pool_size,
            "ROOP_TRT_AUX_STREAMS": tuning.cuda_auxiliary_streams,
            "ROOP_TRT_CUDA_GRAPH": int(tuning.cuda_graph_enabled),
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
            "ROOP_RUNTIME_WORKER_COUNT": "max_threads",
            "ROOP_RUNTIME_DETECTOR_POOL": "perf_detector_pool",
            "ROOP_RUNTIME_DETMASK_POOL": "perf_detmask_pool",
            "ROOP_RUNTIME_CV_THREADS": "cpu_opencv_threads",
            "ROOP_RUNTIME_FFMPEG_THREADS": "cpu_ffmpeg_threads",
            "ROOP_RUNTIME_ORT_INTRA_THREADS": "cpu_ort_intra_threads",
            "ROOP_RUNTIME_ORT_INTER_THREADS": "cpu_ort_inter_threads",
            "ROOP_RUNTIME_DETECTOR_RESOLUTION": "face_detector_size",
            "ROOP_RUNTIME_BATCH_SIZE": "perf_batch_swap",
            "ROOP_RUNTIME_TILE_BATCH_SIZE": "perf_batch_swap",
            "ROOP_RUNTIME_ENCODER": "output_video_codec",
            "ROOP_RUNTIME_ENCODER_PRESET": "perf_encoder_preset",
            "ROOP_TRT_POOL": "perf_trt_pool",
            "ROOP_DETMASK_POOL": "perf_detmask_pool",
            "ROOP_EXPR_POOL": "perf_expr_pool",
            "ROOP_TRT_AUX_STREAMS": "trt_auxiliary_streams",
            "ROOP_TRT_CUDA_GRAPH": "trt_cuda_graph",
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
        affinity = apply_cpu_affinity(profile.hardware, tuning.cpu_distribution)
        if affinity.get("applied"):
            applied["ROOP_CPU_AFFINITY"] = affinity["indices"]
        return applied


def _normalise_cpu_distribution(value: Any) -> str:
    mode = str(value or "auto").strip().lower()
    if mode in ("p+e", "p_e", "p-plus-e", "pplus_e"):
        return "p_plus_e"
    if mode in ("p-priority-e", "p_priority_e_limited",
                "p+e-limited", "p_plus_e_limited"):
        return "p_priority_e"
    if mode not in ("auto", "p_only", "p_plus_e", "p_priority_e"):
        return "auto"
    return mode


def apply_cpu_affinity(hardware: HardwareProfile, distribution: Any = "auto") -> dict:
    """Apply an explicit, measured P/E policy to this process.

    Automatic mode deliberately leaves the OS scheduler free.  The explicit
    modes are for controlled A/B runs and an operator who has chosen a
    validated policy; affinity is never inferred from a GPU model or guessed
    from the CPU brand string.
    """
    mode = _normalise_cpu_distribution(
        os.environ.get("ROOP_CPU_DISTRIBUTION", distribution))
    p_indices = tuple(int(index) for index in hardware.cpu_performance_indices)
    e_indices = tuple(int(index) for index in hardware.cpu_efficiency_indices)
    if mode == "auto" or not hardware.os_affinity_supported:
        return {"applied": False, "mode": mode, "indices": (),
                "reason": ("automatic scheduling" if mode == "auto"
                           else "OS affinity unavailable")}
    if mode == "p_only":
        selected = p_indices
    elif mode == "p_plus_e":
        selected = p_indices + e_indices
    else:
        try:
            limit = int(os.environ.get("ROOP_CPU_E_LIMIT", ""))
        except (TypeError, ValueError):
            limit = 0
        if limit <= 0:
            limit = max(1, len(e_indices) // 4)
        selected = p_indices + e_indices[:max(1, min(len(e_indices), limit))]
    selected = tuple(sorted(set(selected)))
    if not selected:
        return {"applied": False, "mode": mode, "indices": (),
                "reason": "measured P/E indices unavailable"}
    try:
        import psutil
        psutil.Process(os.getpid()).cpu_affinity(list(selected))
    except Exception as exc:
        return {"applied": False, "mode": mode, "indices": selected,
                "reason": "%s: %s" % (type(exc).__name__, exc)}
    print("[CPU] affinity distribution=%s logical=%d source=%s" %
          (mode, len(selected), hardware.cpu_topology_source), flush=True)
    return {"applied": True, "mode": mode, "indices": selected,
            "reason": "measured OS CPU-set affinity"}


def small_card_enhancer_policy(hardware: HardwareProfile,
                               requested: str | None) -> dict:
    """Resolve the host-RSS-safe enhancer policy for a detected small GPU.

    The default ``auto`` drops the enhancer on a sub-7GB card. ``keep``
    overrides that. This is based on detected VRAM, never a model name, and
    does not affect larger cards.

    WHAT THIS ACTUALLY BUYS, measured 2026-08-31 on the RTX 3060 Laptop 6GB
    over the 600-frame locked fixture (double/d4.mp4, 1280x720), four
    counterbalanced arms, enhancer execution confirmed at stage level:

        stripped   4.46 fps   3.475 GB host RSS   5002 MB VRAM
        keep       3.69 fps   3.902 GB host RSS   5677 MB VRAM
        delta     -17.3%          +428 MB            +675 MB

    Read the VRAM column first: the keep arms peak at 91-94% of this card's
    6144 MB, so the headroom this policy protects is mostly VRAM, and only
    then host RSS -- where 428 MB is real cover against the
    execution_threads=1 collapse of 2026-08-25 Part 3, which fired when
    available RAM reached 1.68 GB.

    CORRECTED: this docstring used to say the enhanced path "exceeds the
    strict 2.5GB RSS ceiling ... while the unenhanced adaptive-NVDEC path
    stays below it". The second half is false on this workload. The unenhanced
    path measures 3.475 GB and the gate FAILS in BOTH configurations; the
    earlier claim came from the shorter, smaller clips the validation matrix
    already flags at 2.62-2.79 GB. Stripping the enhancer does not achieve the
    2.5 GB gate and never did -- it buys VRAM and RSS headroom, which is a
    good reason, just not the one that was written here.

    The enhancer is not unusable on this tier: with ``keep`` it enhanced every
    swapped face (936 calls / 935 swaps), zero wrong-faceset, both arms within
    0.3% of each other, no thrashing. See
    docs/HARDWARE_VALIDATION_MATRIX.md.
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
    if value.strip().lower() == "adaptive":
        return {"requested": value, "effective": value, "changed": False,
                "reason": "adaptive wrapper enforces its own small-card null-path safety"}
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
    "SafeAdaptiveController", "RuntimeAdaptiveController",
    "RuntimeOptimizer", "RuntimeProfile", "RuntimeTuning", "TensorRTEngineManager",
    "WorkloadProfile", "WorkloadProfiler", "apply_cpu_affinity",
    "detect_cpu_topology", "small_card_enhancer_policy",
    "small_card_decode_policy",
]
