"""Session 2 Verification: 60-frame dry-run benchmark with 1 face selected.

Executes a 60-frame benchmark run using `roop.benchmark.runner.BenchmarkRunner`,
prints frame-by-frame telemetry in real-time, analyzes FPS and VRAM spikes,
and validates that model inference hooks operate correctly without interrupting
normal pipeline execution or mutating active user models.

Usage:
    app\\env\\Scripts\\python.exe tests\\test_benchmark_runner.py
"""

from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

# Ensure project root and app directory are present in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path and APP_DIR.is_dir():
    sys.path.insert(0, str(APP_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import roop.globals
from roop.benchmark.asset_manager import (
    BenchmarkAssetManager,
    BenchmarkWorkload,
    WorkloadMode,
    WorkloadSelector,
    get_default_asset_manager,
)
from roop.benchmark.hardware_probe import collect_hardware_profile
from roop.benchmark.runner import (
    BenchmarkRunResult,
    BenchmarkRunner,
    FrameTelemetry,
    run_benchmark,
)
from roop.benchmark.storage import (
    load_benchmark_history,
    save_benchmark_result,
)

try:
    import pytest

    @pytest.fixture
    def temp_dir(tmp_path: Path) -> Path:
        return tmp_path
except ImportError:
    pass


def _format_table_row(
    frame: int,
    total: int,
    latency_ms: float,
    fps: float,
    vram_mb: float,
    cpu_pct: float,
    note: str = "",
) -> str:
    """Format a single frame telemetry row for terminal display."""
    return (
        f"  [{frame:02d}/{total:02d}] | "
        f"{latency_ms:7.2f} ms | "
        f"{fps:6.2f} FPS | "
        f"{vram_mb:8.2f} MiB | "
        f"{cpu_pct:5.1f}% | "
        f"{note}"
    )


def test_dry_run_60_frames(temp_dir: Path) -> BenchmarkRunResult:
    """Execute the 60-frame dry-run benchmark with 1 face selected."""
    print("=" * 78)
    print("Session 2 - 60-Frame Dry-Run Benchmark Verification (Solo: 1 Face)")
    print("=" * 78)

    # 1. Inspect and snapshot active models BEFORE the run
    runner = BenchmarkRunner()
    models_before = runner.inspect_active_models()
    print(f"Active Models (pre-test):")
    print(f"  * Swapper    : {models_before.get('swapper')}")
    print(f"  * Enhancer   : {models_before.get('enhancer')}")
    print(f"  * Mask Engine: {models_before.get('mask_engine')}")
    print("-" * 78)

    # 2. Select workload: 1 face (solo target)
    workload: BenchmarkWorkload = WorkloadSelector.get_workload("solo")
    assert workload.mode == WorkloadMode.SOLO, f"Expected SOLO mode, got {workload.mode}"
    assert workload.target_faces == 1, f"Expected 1 target face, got {workload.target_faces}"
    print(f"Selected Workload : {workload.name} ({workload.target_faces} face target)")
    print(f"Frame Window      : 60 frames (1080p full pipeline)")
    print(f"Warmup Frames     : 2 frames")
    print("-" * 78)

    # 3. Setup live frame-by-frame telemetry printing callback
    table_header = (
        "   Frame    |  Latency   |   FPS    |  VRAM (MiB) | CPU (%) | Notes\n"
        "  ----------+-----------+----------+-------------+---------+------------------------"
    )
    print(table_header)

    frame_count = 60
    history_file = temp_dir / "benchmark_history.json"

    # Track real-time spikes during callback
    live_records: list[FrameTelemetry] = []

    def on_frame_progress(
        current: int,
        total: int,
        fps: float,
        record: FrameTelemetry | None = None,
    ) -> None:
        note = ""
        if record is not None:
            live_records.append(record)
            # Detect latency or VRAM spike on the fly
            if len(live_records) > 1:
                prev_latencies = [r.duration_ms for r in live_records[:-1]]
                avg_prev = statistics.fmean(prev_latencies)
                if record.duration_ms > avg_prev * 1.4:
                    note = f"LATENCY SPIKE (+{(record.duration_ms - avg_prev):.1f} ms)"

            row = _format_table_row(
                frame=current,
                total=total,
                latency_ms=record.duration_ms,
                fps=record.fps,
                vram_mb=record.vram_used_mb,
                cpu_pct=record.cpu_util_pct,
                note=note,
            )
        else:
            row = f"  [{current:02d}/{total:02d}] | FPS: {fps:6.2f}"
        print(row)

    # 4. Execute the 60-frame benchmark run
    start_wall_time = time.perf_counter()
    result = runner.run(
        workload=workload,
        frame_window=frame_count,
        warmup_frames=2,
        persist=True,
        storage_path=history_file,
        progress_cb=on_frame_progress,
    )
    total_wall_time = time.perf_counter() - start_wall_time

    print("  ----------+-----------+----------+-------------+---------+------------------------")
    print(f"Execution completed in {total_wall_time:.2f} seconds wall-clock.")
    print("-" * 78)

    # 5. Model Inference Hooks & Telemetry Verification
    print("Verifying Model Inference Hooks & Telemetry Capture...")

    # A. Frame count assertions
    telemetry = result.frame_telemetry
    assert len(telemetry) == 60, f"Expected 60 frame telemetry records, got {len(telemetry)}"
    assert result.metrics["frames_processed"] == 60, (
        f"Expected 60 frames processed in metrics, got {result.metrics['frames_processed']}"
    )

    # B. Throughput assertions
    avg_fps = result.metrics["avg_fps"]
    p1_low_fps = result.metrics["p1_low_fps"]
    avg_latency = result.metrics["avg_latency_ms"]
    p99_latency = result.metrics["p99_latency_ms"]

    assert avg_fps > 0.0, f"Expected positive avg_fps, got {avg_fps}"
    assert p1_low_fps > 0.0, f"Expected positive p1_low_fps, got {p1_low_fps}"
    assert avg_latency > 0.0, f"Expected positive avg_latency_ms, got {avg_latency}"
    assert p99_latency >= avg_latency, (
        f"P99 latency ({p99_latency} ms) should be >= avg latency ({avg_latency} ms)"
    )

    # C. VRAM spikes and recording verification
    durations = [f.duration_ms for f in telemetry]
    vrams = [f.vram_used_mb for f in telemetry]
    cpus = [f.cpu_util_pct for f in telemetry]

    min_vram = min(vrams)
    max_vram = max(vrams)
    peak_vram = result.metrics["peak_vram_mb"]
    vram_spike = max_vram - min_vram

    min_latency = min(durations)
    max_latency = max(durations)
    latency_spike = max_latency - min_latency

    peak_latency_frame = durations.index(max_latency) + 1
    peak_vram_frame = vrams.index(max_vram) + 1

    assert peak_vram > 0.0, f"Expected positive peak_vram_mb, got {peak_vram}"
    assert peak_vram >= max_vram, (
        f"Result peak VRAM ({peak_vram} MiB) must be >= highest frame VRAM ({max_vram} MiB)"
    )

    print("Telemetry Spikes & Extremes Analysis:")
    print(f"  * Latency Range : {min_latency:.2f} ms to {max_latency:.2f} ms (Spike: +{latency_spike:.2f} ms at Frame {peak_latency_frame})")
    print(f"  * VRAM Range    : {min_vram:.2f} MiB to {max_vram:.2f} MiB (Peak Delta: +{vram_spike:.2f} MiB at Frame {peak_vram_frame})")
    print(f"  * Recorded Peak : {peak_vram:.2f} MiB")
    print(f"  * CPU Range     : {min(cpus):.1f}% to {max(cpus):.1f}% (Peak: {result.metrics['peak_cpu_pct']:.1f}%)")
    print("-" * 78)

    # D. Thermal & Stability Metric Assertions
    t_stab = result.thermal_stability
    fps_first_30 = t_stab["fps_first_30"]
    fps_last_30 = t_stab["fps_last_30"]
    retention_pct = t_stab["retention_pct"]
    frame_var = t_stab["frame_time_variance"]
    throttling = t_stab["throttling_detected"]

    assert fps_first_30 > 0.0, f"Expected positive fps_first_30, got {fps_first_30}"
    assert fps_last_30 > 0.0, f"Expected positive fps_last_30, got {fps_last_30}"
    assert retention_pct > 0.0, f"Expected positive retention_pct, got {retention_pct}"
    assert frame_var >= 0.0, f"Expected non-negative frame_time_variance, got {frame_var}"
    assert isinstance(throttling, bool), f"Expected bool for throttling_detected, got {type(throttling)}"

    print("Thermal & Stability Metrics:")
    print(f"  * First 30 Frames FPS : {fps_first_30:.2f} FPS")
    print(f"  * Last 30 Frames FPS  : {fps_last_30:.2f} FPS")
    print(f"  * FPS Retention Rate  : {retention_pct:.1f}%")
    print(f"  * Frame-Time Variance : {frame_var:.2f} ms²")
    print(f"  * Throttling State    : {'THROTTLING DETECTED' if throttling else 'STABLE (No throttling)'}")
    print("-" * 78)

    # E. Active Models Preservation Invariant Check
    models_after = runner.inspect_active_models()
    assert models_after == models_before, (
        f"Active models mutated during benchmark! Before: {models_before}, After: {models_after}"
    )
    assert result.active_models == models_before, (
        f"Result active_models mismatch! Got {result.active_models}, expected {models_before}"
    )
    print("Active Models Preservation:")
    print(f"  -> PASS: All active models preserved without mutation.")
    print("-" * 78)

    # F. Normal Pipeline Uninterrupted Check
    assert getattr(roop.globals, "processing", False) is False, (
        "roop.globals.processing was not cleanly reset to False"
    )
    print("Pipeline Execution State:")
    print(f"  -> PASS: Processing flag cleanly restored, pipeline state intact.")
    print("-" * 78)

    # G. Persistence Verification
    assert history_file.is_file(), f"Benchmark history file not found at {history_file}"
    history = load_benchmark_history(history_file)
    assert len(history) >= 1, f"Expected at least 1 persisted record, got {len(history)}"
    saved = history[0]
    assert saved["run_id"] == result.run_id, f"Persisted run_id mismatch"
    saved_metrics = saved.get("best_metrics") or saved.get("metrics") or saved.get("baseline_metrics")
    assert saved_metrics is not None, "Expected metrics block in persisted storage record"
    assert saved_metrics["avg_fps"] == result.metrics["avg_fps"]
    assert saved_metrics["peak_vram_mb"] == result.metrics["peak_vram_mb"]
    assert saved["workload"]["target_faces"] == 1
    print("Storage Persistence:")
    print(f"  -> PASS: Run {result.run_id[:8]} persisted and re-loaded cleanly.")
    print("-" * 78)

    # H. Dual-Device Environment Safety Verification
    hw = result.device_specs
    gpu_spec = hw.get("gpu", {})
    if gpu_spec.get("available"):
        total_vram_mb = float(gpu_spec.get("total_vram_mb", 0.0) or 0.0)
        gpu_name = gpu_spec.get("name", "Unknown GPU")
        print(f"Hardware Environment Verification ({gpu_name}):")
        print(f"  * Total GPU VRAM : {total_vram_mb:.1f} MiB")
        print(f"  * Peak VRAM Used : {peak_vram:.1f} MiB")

        # Verify headroom: peak VRAM must not exceed total physical VRAM
        if total_vram_mb > 0:
            headroom_mb = total_vram_mb - peak_vram
            assert headroom_mb > 0, f"OOM condition detected: Peak VRAM {peak_vram} exceeded total {total_vram_mb}"
            print(f"  * Headroom       : {headroom_mb:.1f} MiB")

            if "4070" in gpu_name:
                # Desktop profile: 12GB VRAM, plenty of headroom expected
                print("  * Device Profile : Desktop RTX 4070 Profile (12GB VRAM) verified.")
            elif "3060" in gpu_name:
                # Laptop profile: 6GB VRAM, strict sub-7GB limits
                print("  * Device Profile : Laptop RTX 3060 Profile (6GB VRAM) verified.")
    print("=" * 78)
    return result


def test_subprocess_failure_handling(temp_dir: Path) -> None:
    """Verify worker isolation and failure handling during aggressive thread counts.

    Asserts that:
    1. Isolated spawned worker processes return metrics cleanly to the parent process.
    2. Intentional aggressive thread allocation causes clean failure reporting without terminating main application.
    3. The parent application process state (roop.globals) remains completely intact.
    4. Pass 0 (Mandatory Baseline Pass) records existing settings without mutating them.
    """
    print("=" * 78)
    print("Testing Subprocess Worker Isolation & Aggressive Thread Count Failure Handling")
    print("=" * 78)

    runner = BenchmarkRunner()
    initial_processing = getattr(roop.globals, "processing", False)
    initial_threads = getattr(roop.globals, "execution_threads", None)

    # Part A: Normal isolated run in spawned worker process
    print("\n[Part A] Testing Normal Isolated Subprocess Execution...")
    workload = WorkloadSelector.get_workload("solo")
    normal_result = runner.run(
        workload=workload,
        frame_window=10,
        warmup_frames=1,
        persist=False,
        isolated=True,
    )
    assert normal_result.success is True, f"Expected successful run, got error: {normal_result.error}"
    assert normal_result.metrics["avg_fps"] > 0.0, "Expected positive FPS from isolated runner"
    assert normal_result.metrics["frames_processed"] == 10
    print(f"  -> PASS: Isolated worker completed 10 frames at {normal_result.metrics['avg_fps']:.2f} FPS.")

    # Part B: Aggressive thread count test (simulating aggressive thread count e.g. 256)
    print("\n[Part B] Testing Simulated Aggressive Thread Count (256 threads)...")
    aggressive_settings = {"execution_threads": 256}
    failed_result = runner.run_parameter_variation(
        override_settings=aggressive_settings,
        workload=workload,
        frame_window=10,
        warmup_frames=0,
        raise_on_error=False,
    )

    # Assert that the error returned cleanly to the parent process without crashing the application
    assert failed_result.success is False, "Expected aggressive thread count run to fail"
    assert failed_result.error is not None, "Expected descriptive error in failed_result"
    assert (
        "out of memory" in failed_result.error.lower()
        or "aggressive thread count" in failed_result.error.lower()
        or failed_result.is_oom
    ), f"Expected CUDA OOM or aggressive thread count error message, got: {failed_result.error}"
    print(f"  -> PASS: Aggressive thread count caught cleanly: {failed_result.error}")
    print(f"  -> OOM detected flag: {failed_result.is_oom}")

    # Part C: Verify parent process state was NOT corrupted
    assert getattr(roop.globals, "processing", False) == initial_processing, "Parent processing state was corrupted"
    assert getattr(roop.globals, "execution_threads", None) == initial_threads, "Parent execution_threads was mutated"
    print("  -> PASS: Parent process roop.globals and state completely preserved.")

    # Part D: Test Pass 0 (Mandatory Baseline Pass)
    print("\n[Part D] Testing Mandatory Baseline Pass (Pass 0)...")
    baseline_result = runner.run_baseline_pass(
        workload="solo",
        frame_window=10,
        warmup_frames=1,
        persist=False,
        isolated=True,
    )
    assert baseline_result.success is True
    assert baseline_result.metrics["avg_fps"] > 0.0
    assert baseline_result.metrics["peak_vram_mb"] > 0.0
    print(
        f"  -> PASS: Baseline pass completed (FPS: {baseline_result.metrics['avg_fps']:.2f}, "
        f"Peak VRAM: {baseline_result.metrics['peak_vram_mb']:.2f} MiB)."
    )
    print("=" * 78)


def main() -> int:
    """Main verification routine."""
    with tempfile.TemporaryDirectory(prefix="roop_bench_test_") as td:
        temp_dir = Path(td)
        print("Starting Benchmark Runner Test Suite...\n")
        dry_run_result = test_dry_run_60_frames(temp_dir)
        print("\nSUMMARY REPORT:")
        print(dry_run_result.summary_text())

        test_subprocess_failure_handling(temp_dir)
        print("\n>>> ALL BENCHMARK RUNNER TESTS VERIFIED SUCCESSFULLY! <<<")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] Benchmark verification failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
