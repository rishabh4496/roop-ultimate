"""Cross-platform hardware telemetry for the roop benchmark engine.

The probes in this module are intentionally best-effort.  Benchmarking must
remain usable on a CPU-only machine, or when an optional accelerator runtime is
not installed.  Every public probe therefore returns a JSON-serialisable
dictionary instead of raising for an unavailable hardware integration.
"""

from __future__ import annotations

import importlib
import logging
import os
import platform
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable


LOGGER = logging.getLogger(__name__)
_MEBIBYTE = 1024 * 1024
DEFAULT_DISK_PROBE_SIZE_MB = 100
DEFAULT_DISK_PROBE_CHUNK_MB = 4


def _optional_import(module_name: str) -> Any | None:
    """Import an optional dependency without making telemetry unavailable."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # Optional native modules can fail during import.
        LOGGER.debug(
            "Optional telemetry module %s is unavailable: %s", module_name, exc
        )
        return None


def _as_text(value: Any, default: str = "Unknown") -> str:
    """Return a safe human-readable value from native telemetry bindings."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip() or default
    text = str(value).strip()
    return text or default


def _mb(byte_count: int | float | None) -> float:
    """Convert bytes to MiB, retaining a stable precision for JSON reports."""
    if byte_count is None:
        return 0.0
    try:
        return round(float(byte_count) / _MEBIBYTE, 2)
    except (TypeError, ValueError):
        return 0.0


def _empty_gpu_info() -> dict[str, Any]:
    """Build the stable GPU schema used when no hardware runtime is available."""
    return {
        "available": False,
        "vendor": "cpu",
        "backend": "cpu",
        "name": platform.processor() or platform.machine() or "CPU fallback",
        "driver_version": "",
        "device_index": None,
        "total_vram_mb": 0.0,
        "used_vram_mb": 0.0,
        "allocated_vram_mb": 0.0,
        "reserved_vram_mb": 0.0,
        "cuda_capability": None,
        "utilization_pct": None,
        "telemetry_available": False,
        "system_memory_limit_mb": _system_memory_limit_mb(),
    }


def _system_memory_limit_mb() -> float:
    """Return the RAM currently available to the CPU execution provider."""
    psutil = _optional_import("psutil")
    if psutil is None:
        return 0.0
    try:
        return _mb(psutil.virtual_memory().available)
    except Exception:
        return 0.0


def get_cpu_info() -> dict[str, Any]:
    """Return CPU topology, reported clock frequencies, and architecture.

    ``psutil`` is preferred because it can distinguish physical and logical
    cores.  The standard-library fallback preserves a useful report when it is
    not installed.
    """
    psutil = _optional_import("psutil")
    physical_cores: int | None = None
    logical_threads: int | None = None
    current_frequency_mhz: float | None = None
    base_frequency_mhz: float | None = None
    max_frequency_mhz: float | None = None

    if psutil is not None:
        try:
            physical_cores = psutil.cpu_count(logical=False)
            logical_threads = psutil.cpu_count(logical=True)
            frequency = psutil.cpu_freq()
            if frequency is not None:
                current_frequency_mhz = round(float(frequency.current), 2)
                # psutil has no cross-platform "base" clock.  ``min`` is the
                # closest firmware-reported baseline; retain max separately.
                base_frequency_mhz = round(float(frequency.min), 2) or None
                max_frequency_mhz = round(float(frequency.max), 2)
        except Exception as exc:
            LOGGER.debug("psutil CPU probe failed: %s", exc)

    if logical_threads is None:
        logical_threads = os.cpu_count() or 1
    if physical_cores is None:
        # There is no portable standard-library physical-core API.  Returning
        # the logical count is a clearer fallback than returning no capacity.
        physical_cores = logical_threads
    if base_frequency_mhz is None:
        # Several Windows and Linux drivers expose only a current and maximum
        # clock.  The maximum reported clock is a more useful stable baseline
        # than leaving a required telemetry field blank.
        base_frequency_mhz = max_frequency_mhz or current_frequency_mhz

    return {
        "physical_cores": int(physical_cores),
        "logical_threads": int(logical_threads),
        "base_frequency_mhz": base_frequency_mhz,
        "current_frequency_mhz": current_frequency_mhz,
        "max_frequency_mhz": max_frequency_mhz,
        "architecture": platform.machine() or "unknown",
        "processor": platform.processor() or "unknown",
        "system": platform.system() or "unknown",
        "release": platform.release() or "unknown",
    }


def get_memory_info() -> dict[str, Any]:
    """Return total/available RAM and pagefile or swap use in MiB."""
    psutil = _optional_import("psutil")
    if psutil is None:
        return {
            "total_memory_mb": 0.0,
            "available_memory_mb": 0.0,
            "used_memory_mb": 0.0,
            "memory_percent": None,
            "swap_total_mb": 0.0,
            "swap_used_mb": 0.0,
            "swap_free_mb": 0.0,
            "swap_percent": None,
        }

    try:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "total_memory_mb": _mb(memory.total),
            "available_memory_mb": _mb(memory.available),
            "used_memory_mb": _mb(memory.used),
            "memory_percent": round(float(memory.percent), 2),
            "swap_total_mb": _mb(swap.total),
            "swap_used_mb": _mb(swap.used),
            "swap_free_mb": _mb(swap.free),
            "swap_percent": round(float(swap.percent), 2),
        }
    except Exception as exc:
        LOGGER.warning("System-memory telemetry is unavailable: %s", exc)
        return {
            "total_memory_mb": 0.0,
            "available_memory_mb": 0.0,
            "used_memory_mb": 0.0,
            "memory_percent": None,
            "swap_total_mb": 0.0,
            "swap_used_mb": 0.0,
            "swap_free_mb": 0.0,
            "swap_percent": None,
        }


def _probe_nvml(device_index: int) -> dict[str, Any] | None:
    """Return NVIDIA metrics through NVML, including real-time GPU load."""
    nvml = _optional_import("pynvml")
    if nvml is None:
        return None

    initialized = False
    try:
        initialize: Callable[[], Any] | None = getattr(nvml, "nvmlInit", None)
        if initialize is None:
            initialize = getattr(nvml, "nvmlInit_v2", None)
        if initialize is None:
            return None
        initialize()
        initialized = True
        count = int(nvml.nvmlDeviceGetCount())
        if count <= 0:
            return None
        index = min(max(int(device_index), 0), count - 1)
        handle = nvml.nvmlDeviceGetHandleByIndex(index)
        memory = nvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
        capability: str | None = None
        try:
            major, minor = nvml.nvmlDeviceGetCudaComputeCapability(handle)
            capability = "%d.%d" % (int(major), int(minor))
        except Exception:
            pass
        allocated, reserved = _torch_memory_usage(index)
        return {
            "available": True,
            "vendor": "nvidia",
            "backend": "pynvml",
            "name": _as_text(nvml.nvmlDeviceGetName(handle)),
            "driver_version": _as_text(nvml.nvmlSystemGetDriverVersion(), ""),
            "device_index": index,
            "total_vram_mb": _mb(memory.total),
            "used_vram_mb": _mb(memory.used),
            "allocated_vram_mb": allocated,
            "reserved_vram_mb": reserved,
            "cuda_capability": capability,
            "utilization_pct": round(float(utilization.gpu), 2),
            "telemetry_available": True,
        }
    except Exception as exc:
        LOGGER.debug("NVML GPU probe failed; falling back to torch: %s", exc)
        return None
    finally:
        if initialized:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass


def _probe_torch_cuda(device_index: int) -> dict[str, Any] | None:
    """Return CUDA or ROCm information from PyTorch when NVML is absent."""
    torch = _optional_import("torch")
    if torch is None:
        return None
    try:
        if not bool(torch.cuda.is_available()):
            return None
        count = int(torch.cuda.device_count())
        if count <= 0:
            return None
        index = min(max(int(device_index), 0), count - 1)
        properties = torch.cuda.get_device_properties(index)
        total = int(getattr(properties, "total_memory", 0))
        used = 0
        try:
            free, total_from_runtime = torch.cuda.mem_get_info(index)
            total = int(total_from_runtime)
            used = total - int(free)
        except Exception:
            # These figures are process-local but are still better than a
            # fabricated global VRAM reading on older torch builds.
            used = int(torch.cuda.memory_reserved(index))

        hip_version = getattr(getattr(torch, "version", None), "hip", None)
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        vendor = "amd" if hip_version else "nvidia"
        utilization: float | None = None
        try:
            value = torch.cuda.utilization(index)
            utilization = round(float(value), 2)
        except Exception:
            pass
        capability: str | None = None
        if not hip_version:
            try:
                major, minor = torch.cuda.get_device_capability(index)
                capability = "%d.%d" % (int(major), int(minor))
            except Exception:
                pass
        smi = _probe_nvidia_smi(index) if vendor == "nvidia" else {}
        return {
            "available": True,
            "vendor": vendor,
            "backend": "rocm" if hip_version else "torch_cuda",
            "name": _as_text(torch.cuda.get_device_name(index)),
            "driver_version": _as_text(smi.get("driver_version") or hip_version or cuda_version, ""),
            "device_index": index,
            "total_vram_mb": _mb(total),
            "used_vram_mb": _mb(used),
            "allocated_vram_mb": _mb(torch.cuda.memory_allocated(index)),
            "reserved_vram_mb": _mb(torch.cuda.memory_reserved(index)),
            "cuda_capability": capability,
            "utilization_pct": smi.get("utilization_pct", utilization),
            "telemetry_available": smi.get("utilization_pct", utilization) is not None,
        }
    except Exception as exc:
        LOGGER.debug("PyTorch CUDA/ROCm GPU probe failed: %s", exc)
        return None


def _probe_directml() -> dict[str, Any] | None:
    """Identify a DirectML accelerator when torch-directml is installed.

    DirectML exposes no portable API for adapter VRAM or engine utilisation, so
    the unsupported live metrics remain explicit ``None``/zero values.
    """
    directml = _optional_import("torch_directml")
    if directml is None:
        return None
    try:
        device = directml.device()
        name_getter = getattr(directml, "device_name", None)
        name = name_getter(0) if callable(name_getter) else str(device)
        return {
            "available": True,
            "vendor": "amd" if "amd" in str(name).lower() else "directml",
            "backend": "directml",
            "name": _as_text(name, "DirectML device"),
            "driver_version": "",
            "device_index": 0,
            "total_vram_mb": 0.0,
            "used_vram_mb": 0.0,
            "allocated_vram_mb": 0.0,
            "reserved_vram_mb": 0.0,
            "cuda_capability": None,
            "utilization_pct": None,
            "telemetry_available": False,
        }
    except Exception as exc:
        LOGGER.debug("DirectML GPU probe failed: %s", exc)
        return None


def _probe_mps() -> dict[str, Any] | None:
    """Identify Apple Silicon MPS without claiming unavailable VRAM telemetry."""
    torch = _optional_import("torch")
    if torch is None:
        return None
    try:
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is None or not bool(mps.is_available()):
            return None
        return {
            "available": True,
            "vendor": "apple",
            "backend": "mps",
            "name": "Apple Silicon GPU",
            "driver_version": platform.mac_ver()[0],
            "device_index": 0,
            "total_vram_mb": 0.0,
            "used_vram_mb": 0.0,
            "allocated_vram_mb": 0.0,
            "reserved_vram_mb": 0.0,
            "cuda_capability": None,
            "utilization_pct": None,
            "telemetry_available": False,
        }
    except Exception as exc:
        LOGGER.debug("MPS GPU probe failed: %s", exc)
        return None


def _probe_openvino() -> dict[str, Any] | None:
    """Identify an Intel OpenVINO accelerator or retain its CPU fallback."""
    openvino = _optional_import("openvino")
    if openvino is None:
        return None
    try:
        core = openvino.Core()
        devices = tuple(str(item) for item in core.available_devices)
        selected = next(
            (item for item in devices if item.upper().startswith("GPU")), None
        )
        selected = selected or next(
            (item for item in devices if item.upper().startswith("NPU")), None
        )
        selected = selected or next(
            (item for item in devices if item.upper().startswith("CPU")), None
        )
        if selected is None:
            return None
        full_name = selected
        try:
            full_name = _as_text(
                core.get_property(selected, "FULL_DEVICE_NAME"), selected
            )
        except Exception:
            pass
        return {
            "available": True,
            "vendor": "intel",
            "backend": "openvino",
            "name": full_name,
            "driver_version": "",
            "device_index": 0,
            "total_vram_mb": 0.0,
            "used_vram_mb": 0.0,
            "allocated_vram_mb": 0.0,
            "reserved_vram_mb": 0.0,
            "cuda_capability": None,
            "utilization_pct": None,
            "telemetry_available": False,
            "openvino_devices": list(devices),
        }
    except Exception as exc:
        LOGGER.debug("OpenVINO device probe failed: %s", exc)
        return None


def get_gpu_info(device_index: int = 0) -> dict[str, Any]:
    """Return the preferred accelerator and a current utilisation snapshot.

    NVIDIA uses NVML first because it supplies actual card-wide memory and GPU
    engine utilisation.  CUDA/ROCm, DirectML, MPS, OpenVINO, and finally the
    CPU each provide a graceful fallback in that order.
    """
    for probe in (
        lambda: _probe_nvml(device_index),
        lambda: _probe_torch_cuda(device_index),
        _probe_directml,
        _probe_windows_adapter,
        _probe_mps,
        _probe_openvino,
    ):
        result = probe()
        if result is not None:
            return result
    return _empty_gpu_info()


def _torch_memory_usage(device_index: int) -> tuple[float, float]:
    """Return process-local allocated and reserved CUDA memory in MiB."""
    torch = _optional_import("torch")
    if torch is None:
        return 0.0, 0.0
    try:
        if not torch.cuda.is_available():
            return 0.0, 0.0
        return (
            _mb(torch.cuda.memory_allocated(device_index)),
            _mb(torch.cuda.memory_reserved(device_index)),
        )
    except Exception:
        return 0.0, 0.0


def _probe_nvidia_smi(device_index: int) -> dict[str, Any]:
    """Supplement torch fallback with card-wide NVIDIA engine utilisation."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,utilization.gpu",
                "--format=csv,noheader,nounits",
                "--id=%d" % device_index,
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
        values = [value.strip() for value in completed.stdout.strip().split(",")]
        if completed.returncode or len(values) != 2:
            return {}
        return {"driver_version": values[0], "utilization_pct": float(values[1])}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def _probe_windows_adapter() -> dict[str, Any] | None:
    """Use WMI/DXGI-backed Win32 data when no vendor runtime is available."""
    if platform.system().lower() != "windows":
        return None
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
        if completed.returncode or not completed.stdout.strip():
            return None
        import json

        adapters = json.loads(completed.stdout)
        if isinstance(adapters, dict):
            adapters = [adapters]
        if not isinstance(adapters, list) or not adapters:
            return None
        adapter = next(
            (item for item in adapters if "microsoft basic" not in str(item.get("Name", "")).lower()),
            adapters[0],
        )
        name = _as_text(adapter.get("Name"), "Windows display adapter")
        lowered = name.lower()
        vendor = "amd" if "amd" in lowered or "radeon" in lowered else "intel" if "intel" in lowered else "directml"
        return {
            "available": True,
            "vendor": vendor,
            "backend": "wmi",
            "name": name,
            "driver_version": _as_text(adapter.get("DriverVersion"), ""),
            "device_index": 0,
            "total_vram_mb": _mb(adapter.get("AdapterRAM")),
            "used_vram_mb": 0.0,
            "allocated_vram_mb": 0.0,
            "reserved_vram_mb": 0.0,
            "cuda_capability": None,
            "utilization_pct": None,
            "telemetry_available": False,
        }
    except Exception as exc:
        LOGGER.debug("Windows adapter probe failed: %s", exc)
        return None


def _configured_temp_directory(temp_directory: str | os.PathLike[str] | None) -> Path:
    """Resolve an explicit roop temp directory before platform defaults."""
    candidate = temp_directory
    if candidate is None:
        candidate = (
            os.environ.get("ROOP_TEMP_DIR")
            or os.environ.get("GRADIO_TEMP_DIR")
            or os.environ.get("TMPDIR")
            or os.environ.get("TEMP")
            or os.environ.get("TMP")
            or tempfile.gettempdir()
        )
    return Path(candidate).expanduser().resolve()


def probe_disk_io(
    temp_directory: str | os.PathLike[str] | None = None,
    *,
    size_mb: int = DEFAULT_DISK_PROBE_SIZE_MB,
    chunk_mb: int = DEFAULT_DISK_PROBE_CHUNK_MB,
) -> dict[str, Any]:
    """Measure sequential temp-frame write and read throughput in MiB/s.

    A 100 MiB file is written in reusable binary chunks, read back in full, and
    removed in a ``finally`` block.  Callers may lower ``size_mb`` only for a
    test environment; normal benchmark runs should retain the default.
    """
    result: dict[str, Any] = {
        "temp_directory": "",
        "bytes_tested": 0,
        "write_mb_per_sec": 0.0,
        "read_mb_per_sec": 0.0,
        "sequential_write_mb_s": 0.0,
        "sequential_read_mb_s": 0.0,
        "access_latency_ms": 0.0,
        "write_seconds": 0.0,
        "read_seconds": 0.0,
        "success": False,
        "error": None,
    }
    if size_mb <= 0 or chunk_mb <= 0:
        result["error"] = "size_mb and chunk_mb must be positive integers"
        return result

    path: Path | None = None
    try:
        directory = _configured_temp_directory(temp_directory)
        directory.mkdir(parents=True, exist_ok=True)
        result["temp_directory"] = str(directory)
        total_bytes = int(size_mb) * _MEBIBYTE
        chunk_size = min(int(chunk_mb) * _MEBIBYTE, total_bytes)
        path = directory / ("roop-benchmark-" + uuid.uuid4().hex + ".bin")
        # A pre-generated binary block models image-frame writes while keeping
        # random-data generation outside the write-throughput measurement.
        chunk = os.urandom(chunk_size)

        latency_started = time.perf_counter()
        with path.open("xb", buffering=0) as handle:
            handle.write(chunk[:4096])
            handle.flush()
            os.fsync(handle.fileno())
        with path.open("rb", buffering=0) as handle:
            handle.read(4096)
        access_latency_ms = (time.perf_counter() - latency_started) * 1000.0
        path.unlink()

        write_started = time.perf_counter()
        remaining = total_bytes
        with path.open("wb", buffering=chunk_size) as handle:
            while remaining:
                written = min(remaining, chunk_size)
                handle.write(chunk if written == chunk_size else chunk[:written])
                remaining -= written
            handle.flush()
            os.fsync(handle.fileno())
        write_seconds = time.perf_counter() - write_started

        read_started = time.perf_counter()
        with path.open("rb", buffering=chunk_size) as handle:
            while handle.read(chunk_size):
                pass
        read_seconds = time.perf_counter() - read_started

        result.update(
            {
                "bytes_tested": total_bytes,
                "write_seconds": round(write_seconds, 4),
                "read_seconds": round(read_seconds, 4),
                "write_mb_per_sec": round(
                    (total_bytes / _MEBIBYTE) / max(write_seconds, 1e-9), 2
                ),
                "read_mb_per_sec": round(
                    (total_bytes / _MEBIBYTE) / max(read_seconds, 1e-9), 2
                ),
                "sequential_write_mb_s": round(
                    (total_bytes / _MEBIBYTE) / max(write_seconds, 1e-9), 2
                ),
                "sequential_read_mb_s": round(
                    (total_bytes / _MEBIBYTE) / max(read_seconds, 1e-9), 2
                ),
                "access_latency_ms": round(access_latency_ms, 3),
                "success": True,
            }
        )
    except (OSError, ValueError) as exc:
        result["error"] = str(exc)
        LOGGER.warning("Disk I/O benchmark probe failed: %s", exc)
    finally:
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                result["success"] = False
                result["error"] = (
                    "Temporary benchmark file could not be removed: %s" % exc
                )
                LOGGER.warning("Disk I/O benchmark cleanup failed: %s", exc)
    return result


def collect_hardware_profile(
    temp_directory: str | os.PathLike[str] | None = None,
    *,
    include_disk_io: bool = True,
    device_index: int = 0,
) -> dict[str, Any]:
    """Collect the JSON-ready device specification block for one benchmark run."""
    profile = {
        "cpu": get_cpu_info(),
        "gpu": get_gpu_info(device_index),
        "memory": get_memory_info(),
    }
    if include_disk_io:
        profile["disk_io"] = probe_disk_io(temp_directory)
    return profile


# Semantic aliases make the module easy to integrate from early benchmark UI
# code without coupling callers to a single naming convention.
inspect_cpu = get_cpu_info
inspect_gpu = get_gpu_info
inspect_ram = get_memory_info
inspect_hardware = collect_hardware_profile


def measure_disk_io_throughput(target_dir: str) -> dict[str, Any]:
    """Run the standard 100 MiB temporary-volume throughput and latency test."""
    return probe_disk_io(target_dir)


__all__ = [
    "DEFAULT_DISK_PROBE_CHUNK_MB",
    "DEFAULT_DISK_PROBE_SIZE_MB",
    "collect_hardware_profile",
    "get_cpu_info",
    "get_gpu_info",
    "get_memory_info",
    "inspect_cpu",
    "inspect_gpu",
    "inspect_hardware",
    "inspect_ram",
    "measure_disk_io_throughput",
    "probe_disk_io",
]
