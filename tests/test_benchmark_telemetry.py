"""Standalone Session 1 verification for benchmark telemetry and persistence.

Run directly from the repository root:

    app\\env\\Scripts\\python.exe tests\\test_benchmark_telemetry.py

The runner writes its mock benchmark history to a temporary directory and
removes it on exit.  It does execute the production 100 MiB disk probe so the
reported throughput reflects the configured roop temporary storage.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roop.benchmark.hardware_probe import collect_hardware_profile  # noqa: E402
from roop.benchmark.storage import (  # noqa: E402
    load_benchmark_history,
    save_benchmark_result,
)


def _display(value: Any, fallback: str = "N/A") -> str:
    """Render an optional telemetry value without cluttering the summary."""
    if value is None or value == "":
        return fallback
    return str(value)


def _number(value: Any) -> float:
    """Return a numeric report value, defaulting only for display safety."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _print_summary(profile: Mapping[str, Any]) -> None:
    """Print a compact, human-readable hardware telemetry report."""
    cpu = profile["cpu"]
    gpu = profile["gpu"]
    memory = profile["memory"]
    disk = profile["disk_io"]

    print("Session 1 - Benchmark Telemetry Verification")
    print("=" * 48)
    print(
        "CPU : {processor} | {physical} physical cores / {logical} threads | "
        "{base} MHz current / {maximum} MHz max | {architecture}".format(
            processor=_display(cpu.get("processor")),
            physical=_display(cpu.get("physical_cores")),
            logical=_display(cpu.get("logical_threads")),
            base=_display(cpu.get("base_frequency_mhz")),
            maximum=_display(cpu.get("max_frequency_mhz")),
            architecture=_display(cpu.get("architecture")),
        )
    )
    print(
        "GPU : {name} | {vendor}/{backend} | driver {driver} | load {load}".format(
            name=_display(gpu.get("name")),
            vendor=_display(gpu.get("vendor")),
            backend=_display(gpu.get("backend")),
            driver=_display(gpu.get("driver_version")),
            load=(
                "{:.1f}%".format(_number(gpu.get("utilization_pct")))
                if gpu.get("utilization_pct") is not None
                else "N/A"
            ),
        )
    )
    print(
        "VRAM: {used:.2f} / {total:.2f} MiB".format(
            used=_number(gpu.get("used_vram_mb")),
            total=_number(gpu.get("total_vram_mb")),
        )
    )
    print(
        "RAM : {used:.2f} / {total:.2f} MiB used | {available:.2f} MiB available | "
        "swap {swap_used:.2f} / {swap_total:.2f} MiB".format(
            used=_number(memory.get("used_memory_mb")),
            total=_number(memory.get("total_memory_mb")),
            available=_number(memory.get("available_memory_mb")),
            swap_used=_number(memory.get("swap_used_mb")),
            swap_total=_number(memory.get("swap_total_mb")),
        )
    )
    if disk.get("success"):
        print(
            "Disk: write {write:.2f} MB/s | read {read:.2f} MB/s | "
            "{size:.0f} MiB probe".format(
                write=_number(disk.get("write_mb_per_sec")),
                read=_number(disk.get("read_mb_per_sec")),
                size=_number(disk.get("bytes_tested")) / (1024 * 1024),
            )
        )
    else:
        print("Disk: probe failed - %s" % _display(disk.get("error")))


def _assert_benchmark_schema(record: Mapping[str, Any]) -> None:
    """Assert the complete public storage schema expected by Session 1."""
    required_top_level = {
        "run_id",
        "timestamp",
        "device_specs",
        "active_models",
        "workload",
        "metrics",
        "recommended_settings",
        "applied",
    }
    assert required_top_level == set(record), "unexpected benchmark schema"
    assert isinstance(record["run_id"], str) and record["run_id"]
    assert isinstance(record["timestamp"], str) and record["timestamp"]
    assert isinstance(record["device_specs"], Mapping)
    assert set(record["active_models"]) == {"swapper", "enhancer", "mask_engine"}
    assert record["workload"]["target_faces"] == 1
    assert set(record["metrics"]) == {
        "avg_fps",
        "p1_low_fps",
        "peak_vram_mb",
        "peak_cpu_pct",
    }
    assert set(record["recommended_settings"]) == {
        "execution_threads",
        "execution_provider",
        "temp_format",
        "provider_options",
    }
    assert record["applied"] is False


def main() -> int:
    """Execute telemetry collection and a save/reload schema verification."""
    with tempfile.TemporaryDirectory(prefix="roop_benchmark_telemetry_") as temp_dir:
        temp_path = Path(temp_dir)
        profile = collect_hardware_profile(temp_path, include_disk_io=True)
        _print_summary(profile)
        assert profile["disk_io"]["success"], profile["disk_io"].get("error")

        mock_entry = {
            "device_specs": profile,
            "active_models": {
                "swapper": "mock-inswapper",
                "enhancer": "mock-enhancer",
                "mask_engine": "mock-mask-engine",
            },
            "workload": {"target_faces": 1},
            "metrics": {
                "avg_fps": 30.0,
                "p1_low_fps": 24.0,
                "peak_vram_mb": _number(profile["gpu"].get("used_vram_mb")),
                "peak_cpu_pct": 50.0,
            },
            "recommended_settings": {
                "execution_threads": min(8, int(profile["cpu"]["logical_threads"])),
                "execution_provider": "mock-provider",
                "temp_format": "png",
                "provider_options": {"mock": True},
            },
        }
        history_path = temp_path / "benchmark_history.json"
        saved = save_benchmark_result(mock_entry, history_path)
        reloaded = load_benchmark_history(history_path)

        assert len(reloaded) == 1, "expected exactly one reloaded benchmark entry"
        assert reloaded[0] == saved, "reloaded entry differs from saved entry"
        _assert_benchmark_schema(reloaded[0])
        print(
            "Persistence: mock run %s saved and reloaded without schema errors."
            % saved["run_id"]
        )
        print("Result: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("Result: FAIL - %s" % exc, file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
