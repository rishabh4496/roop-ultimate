"""Standardized roop-ultimate benchmark and telemetry suite."""

from __future__ import annotations

from roop.benchmark.asset_manager import (
    DEFAULT_ASSET_DIR,
    DEFAULT_CLIP_DURATION_SEC,
    DEFAULT_CLIP_FRAMES,
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    STANDARDIZED_WINDOW_FRAMES,
    BenchmarkAssetManager,
    BenchmarkWorkload,
    WorkloadMode,
    WorkloadSelector,
    ensure_benchmark_asset,
    get_default_asset_manager,
    get_workload_preset,
)
from roop.benchmark.hardware_probe import (
    DEFAULT_DISK_PROBE_CHUNK_MB,
    DEFAULT_DISK_PROBE_SIZE_MB,
    collect_hardware_profile,
    get_cpu_info,
    get_gpu_info,
    get_memory_info,
    inspect_cpu,
    inspect_gpu,
    inspect_hardware,
    inspect_ram,
    probe_disk_io,
)
from roop.benchmark.runner import (
    BenchmarkRunResult,
    BenchmarkRunner,
    FrameTelemetry,
    GpuTelemetrySampler,
    run_benchmark,
)
from roop.benchmark.storage import (
    BENCHMARK_HISTORY_PATH,
    BenchmarkStorageError,
    get_latest_optimal_settings,
    load_benchmark_history,
    save_benchmark_result,
    update_setting_status,
)

__all__ = [
    # Telemetry and Hardware Probe
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
    "probe_disk_io",
    # History and Storage
    "BENCHMARK_HISTORY_PATH",
    "BenchmarkStorageError",
    "get_latest_optimal_settings",
    "load_benchmark_history",
    "save_benchmark_result",
    "update_setting_status",
    # Asset Management and Workloads
    "DEFAULT_ASSET_DIR",
    "DEFAULT_CLIP_DURATION_SEC",
    "DEFAULT_CLIP_FRAMES",
    "DEFAULT_FPS",
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "STANDARDIZED_WINDOW_FRAMES",
    "BenchmarkAssetManager",
    "BenchmarkWorkload",
    "WorkloadMode",
    "WorkloadSelector",
    "ensure_benchmark_asset",
    "get_default_asset_manager",
    "get_workload_preset",
    # Runner and Execution
    "BenchmarkRunResult",
    "BenchmarkRunner",
    "FrameTelemetry",
    "GpuTelemetrySampler",
    "run_benchmark",
]
