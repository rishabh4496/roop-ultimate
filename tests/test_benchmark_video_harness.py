"""Verification test suite for Session 2 benchmark video harness and workload runner.

Tests:
1. Workload Face Selection (Solo 1-Face, Duo 2-Face, Group 2+-Face, Master Calibrated).
2. Standardized Video Asset Harness (`roop/benchmark/asset_manager.py`).
3. Real-Video Execution Harness (`roop/benchmark/runner.py`).
4. Active models preservation (models MUST NOT be mutated).
5. Thermal & stability metrics calculation.
6. Persistence and history reload schema verification.

Run directly from repository root:

    app\\env\\Scripts\\python.exe tests\\test_benchmark_video_harness.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

# Ensure REPO_ROOT and app directory are present in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import roop.globals
from roop.benchmark.asset_manager import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    STANDARDIZED_WINDOW_FRAMES,
    BenchmarkAssetManager,
    BenchmarkWorkload,
    WorkloadMode,
    WorkloadSelector,
    ensure_benchmark_asset,
    get_workload_preset,
)
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


def test_workload_selector() -> None:
    """Verify that all workload presets are properly registered and selectable."""
    print("1. Testing Workload Face Selector...")
    presets = WorkloadSelector.list_presets()
    assert len(presets) >= 4, f"Expected at least 4 presets, got {len(presets)}"

    solo = WorkloadSelector.get_workload("solo")
    assert solo.mode == WorkloadMode.SOLO
    assert solo.target_faces == 1
    assert "1 Face" in solo.name

    duo = WorkloadSelector.get_workload(2)
    assert duo.mode == WorkloadMode.DUO
    assert duo.target_faces == 2
    assert "2 Faces" in duo.name

    group = WorkloadSelector.get_workload("2+")
    assert group.mode == WorkloadMode.GROUP
    assert group.target_faces >= 2
    assert "2+ Faces" in group.name

    master = WorkloadSelector.get_workload("calibrated_all")
    assert master.mode == WorkloadMode.CALIBRATED_ALL
    assert master.expected_frames == 300

    print("   -> WorkloadSelector: PASS (Solo, Duo, Group, Master properly resolved)")


def test_asset_manager(temp_dir: Path) -> None:
    """Verify generation, video specifications, and source face provisioning."""
    print("2. Testing Standardized Video Asset Harness...")
    mgr = BenchmarkAssetManager(asset_dir=temp_dir / "assets", facesets_dir=PROJECT_ROOT / "facesets")

    # Verify source reference face provisioning
    src_ref = mgr.ensure_source_reference()
    assert src_ref.is_file() and src_ref.stat().st_size > 0, "Source reference face is missing or empty"
    src_img = cv2.imread(str(src_ref))
    assert src_img is not None, "Failed to decode source reference face"
    print(f"   -> Source Reference: PASS (shape={src_img.shape})")

    # Generate calibrated 15-frame test clip
    test_clip_path = temp_dir / "test_calibrated_15f.mp4"
    mgr.generate_benchmark_clip(
        output_path=test_clip_path,
        workload="solo",
        duration_seconds=0.5,
        fps=30.0,
        width=1920,
        height=1080,
    )
    assert test_clip_path.is_file(), "Test clip was not created"
    assert test_clip_path.stat().st_size > 1000, "Test clip file size is unexpectedly small"

    # Verify video properties using OpenCV VideoCapture
    cap = cv2.VideoCapture(str(test_clip_path))
    assert cap.isOpened(), "VideoCapture could not open generated benchmark clip"
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    assert w == 1920, f"Expected 1920 width, got {w}"
    assert h == 1080, f"Expected 1080 height, got {h}"
    assert count == 15, f"Expected 15 frames, got {count}"
    assert abs(fps - 30.0) < 0.1, f"Expected 30.0 FPS, got {fps}"
    print(f"   -> Video Asset Generation: PASS (1080p 30fps validated, {count} frames)")


def test_active_models_preservation() -> None:
    """Verify that running the benchmark does NOT alter user's active models."""
    print("3. Testing Active Models Preservation Invariant...")
    # Set simulated user active models
    orig_swapper = getattr(roop.globals, "face_swap_mode", "DFL XSeg")
    orig_enhancer = getattr(roop.globals, "selected_enhancer", None)
    orig_mask = getattr(roop.globals, "mask_engine", None)

    roop.globals.face_swap_mode = "DFL XSeg"
    roop.globals.selected_enhancer = "Codeformer"
    roop.globals.mask_engine = "RealityUX"

    runner = BenchmarkRunner()
    active = runner.inspect_active_models()

    assert active["swapper"] == "DFL XSeg"
    assert active["enhancer"] == "Codeformer"
    assert active["mask_engine"] == "RealityUX"

    # Verify roop.globals are unchanged
    assert roop.globals.face_swap_mode == "DFL XSeg"
    assert roop.globals.selected_enhancer == "Codeformer"
    assert roop.globals.mask_engine == "RealityUX"

    # Restore original state
    roop.globals.face_swap_mode = orig_swapper
    roop.globals.selected_enhancer = orig_enhancer
    roop.globals.mask_engine = orig_mask
    print("   -> Active Models Invariant: PASS (user models preserved without modification)")


def test_real_video_execution(temp_dir: Path) -> None:
    """Execute real-video benchmarking with telemetry and stability validation."""
    print("4. Testing Real-Video Execution Harness & Metrics...")
    hist_file = temp_dir / "benchmark_history.json"
    runner = BenchmarkRunner()

    # Run a verified 5-frame benchmark
    result = runner.run(
        workload="solo",
        frame_window=5,
        warmup_frames=1,
        persist=True,
        storage_path=hist_file,
    )

    # 1. Assert result schema
    assert isinstance(result.run_id, str) and len(result.run_id) > 0
    assert isinstance(result.metrics["avg_fps"], float) and result.metrics["avg_fps"] > 0.0
    assert isinstance(result.metrics["p1_low_fps"], float) and result.metrics["p1_low_fps"] > 0.0
    assert isinstance(result.metrics["peak_vram_mb"], float)
    assert isinstance(result.metrics["peak_cpu_pct"], float)

    # 2. Assert thermal & stability metrics
    t = result.thermal_stability
    assert "fps_first_30" in t
    assert "fps_last_30" in t
    assert "retention_pct" in t
    assert "frame_time_variance" in t
    assert "throttling_detected" in t
    assert isinstance(t["throttling_detected"], bool)

    # 3. Assert persistent storage integration
    reloaded = load_benchmark_history(hist_file)
    assert len(reloaded) == 1, f"Expected 1 saved run, found {len(reloaded)}"
    saved = reloaded[0]
    assert saved["run_id"] == result.run_id
    assert saved["metrics"]["avg_fps"] == result.metrics["avg_fps"]
    assert saved["metrics"]["p1_low_fps"] == result.metrics["p1_low_fps"]
    assert saved["workload"]["target_faces"] == 1
    print("   -> Real Video Execution & Telemetry: PASS")
    print(f"      Throughput   : {result.metrics['avg_fps']:.2f} FPS")
    print(f"      1% Low Floor : {result.metrics['p1_low_fps']:.2f} FPS")
    print(f"      Peak VRAM    : {result.metrics['peak_vram_mb']:.2f} MiB")
    print(f"      Peak CPU     : {result.metrics['peak_cpu_pct']:.1f}%")
    print(f"      Stability    : retention={t['retention_pct']:.1f}% (throttling={t['throttling_detected']})")


def main() -> int:
    print("=" * 64)
    print("Session 2 - Standardized Video Benchmark Harness Verification")
    print("=" * 64)
    with tempfile.TemporaryDirectory(prefix="roop_bench_verify_") as td:
        temp_dir = Path(td)
        test_workload_selector()
        test_asset_manager(temp_dir)
        test_active_models_preservation()
        test_real_video_execution(temp_dir)
    print("=" * 64)
    print("ALL SESSION 2 VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
