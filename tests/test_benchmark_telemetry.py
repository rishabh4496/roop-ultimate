"""Standalone verification for benchmark telemetry and profile persistence.

Run from the repository root:

    app\\env\\Scripts\\python.exe tests\\test_benchmark_telemetry.py

The disk test writes, reads, and removes 100 MiB below the user's temporary
directory. Benchmark history itself is isolated in a temporary test folder.
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

from roop.benchmark.hardware_probe import (  # noqa: E402
    collect_hardware_profile,
    measure_disk_io_throughput,
)
from roop.benchmark.storage import (  # noqa: E402
    get_latest_profile,
    load_benchmark_history,
    save_benchmark_result,
    update_profile_status,
)


METRIC_KEYS = {"avg_fps", "p1_low_fps", "peak_vram_mb", "peak_cpu_pct"}
RECORD_KEYS = {
    "run_id",
    "timestamp",
    "device_specs",
    "active_models",
    "workload",
    "baseline_metrics",
    "best_metrics",
    "presets",
    "bottleneck",
    "status",
}


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _print_specs(profile: Mapping[str, Any], disk: Mapping[str, Any]) -> None:
    cpu = profile["cpu"]
    gpu = profile["gpu"]
    memory = profile["memory"]
    print("Benchmark Telemetry Verification")
    print("=" * 48)
    print(
        "CPU : {physical} physical cores / {logical} threads | "
        "base {base} MHz | current {current} MHz | {architecture}".format(
            physical=cpu["physical_cores"],
            logical=cpu["logical_threads"],
            base=cpu["base_frequency_mhz"],
            current=cpu["current_frequency_mhz"],
            architecture=cpu["architecture"],
        )
    )
    print(
        "RAM : {total:.1f} MiB total | {available:.1f} MiB available | "
        "{swap:.1f} MiB swap".format(
            total=_number(memory["total_memory_mb"]),
            available=_number(memory["available_memory_mb"]),
            swap=_number(memory["swap_total_mb"]),
        )
    )
    print(
        "GPU : {name} | {vendor}/{backend} | VRAM {used:.1f}/{total:.1f} MiB | "
        "engine {util}%".format(
            name=gpu["name"],
            vendor=gpu["vendor"],
            backend=gpu["backend"],
            used=_number(gpu["used_vram_mb"]),
            total=_number(gpu["total_vram_mb"]),
            util=gpu["utilization_pct"] if gpu["utilization_pct"] is not None else "N/A",
        )
    )
    print(
        "Disk: write {write:.2f} MB/s | read {read:.2f} MB/s | latency {latency:.3f} ms".format(
            write=_number(disk["sequential_write_mb_s"]),
            read=_number(disk["sequential_read_mb_s"]),
            latency=_number(disk["access_latency_ms"]),
        )
    )


def _mock_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    metrics = {
        "avg_fps": 30.0,
        "p1_low_fps": 24.0,
        "peak_vram_mb": _number(profile["gpu"]["used_vram_mb"]),
        "peak_cpu_pct": 50.0,
    }
    return {
        "device_specs": dict(profile),
        "active_models": {
            "swapper": "mock-inswapper",
            "enhancer": "mock-enhancer",
            "mask_engine": "mock-mask-engine",
        },
        "workload": {"target_faces": 1, "test_mode": "quick"},
        "baseline_metrics": metrics,
        "best_metrics": metrics,
        "presets": {
            "max_throughput": {"threads": 8, "provider_options": {}, "temp_format": "jpg"},
            "balanced": {"threads": 6, "provider_options": {}, "temp_format": "jpg"},
            "quiet": {"threads": 4, "provider_options": {}, "temp_format": "png"},
        },
        "bottleneck": "GPU Compute Bound",
        "status": "pending",
    }


def _assert_strict_schema(record: Mapping[str, Any]) -> None:
    assert set(record) == RECORD_KEYS
    assert set(record["active_models"]) == {"swapper", "enhancer", "mask_engine"}
    assert record["workload"] == {"target_faces": 1, "test_mode": "quick"}
    assert set(record["baseline_metrics"]) == METRIC_KEYS
    assert set(record["best_metrics"]) == METRIC_KEYS
    assert set(record["presets"]) == {"max_throughput", "balanced", "quiet"}
    assert record["status"] in {"declined", "accepted", "pending"}


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="roop_benchmark_telemetry_") as directory:
        test_directory = Path(directory)
        profile = collect_hardware_profile(include_disk_io=False)
        disk = measure_disk_io_throughput(str(test_directory))
        assert disk["success"], disk.get("error")
        profile["disk_io"] = disk
        _print_specs(profile, disk)

        history_path = test_directory / "benchmark_history.json"
        run_id = save_benchmark_result(_mock_profile(profile), history_path)
        assert isinstance(run_id, str) and run_id
        history = load_benchmark_history(history_path)
        assert len(history) == 1
        _assert_strict_schema(history[0])
        assert history[0]["run_id"] == run_id
        assert update_profile_status(run_id, "accepted", history_path)
        latest = get_latest_profile(history_path)
        assert latest is not None and latest["status"] == "accepted"
        print("Persistence: mock run saved, reloaded, and accepted without schema errors.")
        print("Result: PASS")
    return 0


def test_benchmark_telemetry() -> None:
    """Make the standalone verification discoverable by pytest as well."""
    assert main() == 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("Result: FAIL - %s" % exc, file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
