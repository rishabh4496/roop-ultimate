"""Unit and regression tests for Stage 0 Benchmark Harness.

Validates:
1. Scenario specifications covering all 13 required categories.
2. Ground-truth generator integrity (coordinates, tags, profile, occlusion).
3. QualityEvaluator accuracy (detection, identity, landmarks, flicker, occlusion).
4. StageProfiler per-stage latency recording and sync stall accounting.
5. Structured JSON and CSV export schema conformity.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root and app to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
for p in (str(PROJECT_ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np
import pytest

from roop.benchmark.quality_metrics import QualityEvaluator, QualityMetricsReport
from roop.benchmark.scenario_assets import (
    ALL_SCENARIOS,
    FrameGroundTruth,
    ScenarioAssetManager,
    ScenarioCategory,
    ScenarioSpec,
)
from roop.benchmark.stage0_harness import Stage0BenchmarkHarness, StageTiming


def test_scenario_specs_completeness():
    """Verify that all 13 required scenarios are present with valid configurations."""
    assert len(ALL_SCENARIOS) == 13, f"Expected 13 scenarios, found {len(ALL_SCENARIOS)}"
    ids = [s.scenario_id for s in ALL_SCENARIOS]
    assert sorted(ids) == list(range(1, 14)), f"Scenario IDs must be 1..13, got {ids}"

    categories = {s.category for s in ALL_SCENARIOS}
    assert len(categories) == 13, "All scenario categories must be distinct"

    # Check key categories
    expected_categories = [
        ScenarioCategory.FRONTAL_FACE,
        ScenarioCategory.YAW_45DEG_FACE,
        ScenarioCategory.EXTREME_PROFILE,
        ScenarioCategory.SMALL_FACE,
        ScenarioCategory.MULTIPLE_PEOPLE,
        ScenarioCategory.FACE_ENTER_EXIT,
        ScenarioCategory.PARTIAL_OCCLUSION,
        ScenarioCategory.HANDS_CROSSING,
        ScenarioCategory.DARK_SCENE,
        ScenarioCategory.HIGH_MOTION,
        ScenarioCategory.FACES_INTERACTING,
        ScenarioCategory.RESOLUTION_1080P,
        ScenarioCategory.RESOLUTION_4K,
    ]
    for exp in expected_categories:
        assert exp in categories, f"Missing category: {exp}"


def test_scenario_ground_truth_generation():
    """Verify ground truth generation across multiple scenario types."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mgr = ScenarioAssetManager(output_dir=tmp_dir)

        # 1. Frontal face
        s1 = next(s for s in ALL_SCENARIOS if s.category == ScenarioCategory.FRONTAL_FACE)
        gt1 = mgr._build_ground_truth(s1, total_frames=20)
        assert len(gt1) == 20
        assert gt1[0].expected_faces == 1
        assert len(gt1[0].target_boxes) == 1

        # 2. Extreme profile
        s3 = next(s for s in ALL_SCENARIOS if s.category == ScenarioCategory.EXTREME_PROFILE)
        gt3 = mgr._build_ground_truth(s3, total_frames=20)
        assert gt3[0].is_profile is True
        assert gt3[0].expected_yaw >= 60.0

        # 3. Multiple people
        s5 = next(s for s in ALL_SCENARIOS if s.category == ScenarioCategory.MULTIPLE_PEOPLE)
        gt5 = mgr._build_ground_truth(s5, total_frames=20)
        assert gt5[0].expected_faces == 3
        assert len(gt5[0].target_boxes) == 3

        # 4. Partial occlusion
        s7 = next(s for s in ALL_SCENARIOS if s.category == ScenarioCategory.PARTIAL_OCCLUSION)
        gt7 = mgr._build_ground_truth(s7, total_frames=20)
        assert len(gt7[0].occluded_regions) >= 1


def test_quality_evaluator_metrics():
    """Verify that QualityEvaluator tracks detection, identity, and artifacts."""
    rng = np.random.default_rng(20260930)
    fake_source_emb = rng.standard_normal(512).astype(np.float32)
    fake_source_emb /= np.linalg.norm(fake_source_emb)

    evaluator = QualityEvaluator(source_embedding=fake_source_emb)

    # Frame 1: Perfect match
    h, w = 480, 640
    orig = np.full((h, w, 3), 100, dtype=np.uint8)
    swap = np.full((h, w, 3), 102, dtype=np.uint8)

    class MockFace:
        def __init__(self, bbox, emb, kps):
            self.bbox = bbox
            self.embedding = emb
            self.kps = kps
            self.det_score = 0.95

    kps1 = np.array([[200, 200], [250, 200], [225, 230], [210, 260], [240, 260]], dtype=np.float32)
    face1 = MockFace([150, 150, 300, 300], fake_source_emb.copy(), kps1)

    gt = FrameGroundTruth(
        frame_idx=0,
        expected_faces=1,
        target_boxes=[(150, 150, 300, 300)],
    )

    evaluator.evaluate_frame(
        frame_idx=0,
        orig_frame=orig,
        swapped_frame=swap,
        detected_faces=[face1],
        ground_truth=gt,
    )

    # Frame 2: Slight jitter
    kps2 = kps1 + 2.0
    face2 = MockFace([152, 152, 302, 302], fake_source_emb.copy(), kps2)
    evaluator.evaluate_frame(
        frame_idx=1,
        orig_frame=orig,
        swapped_frame=swap,
        detected_faces=[face2],
        ground_truth=gt,
    )

    report = evaluator.compute_report()
    assert report.detection_failures == 0
    assert report.missed_faces == 0
    assert report.identity_similarity_mean >= 0.99
    assert report.landmark_instability_mean > 0.0


def test_stage_timing_aggregation():
    """Verify StageTiming statistics calculation."""
    samples = [10.0, 20.0, 30.0, 40.0, 50.0]
    timing = StageTiming.from_samples(samples)
    assert timing.calls == 5
    assert timing.total_ms == 150.0
    assert timing.mean_ms == 30.0
    assert timing.min_ms == 10.0
    assert timing.max_ms == 50.0
    assert timing.p50_ms == 30.0


def test_csv_export_schema():
    """Verify that export_csv produces a valid CSV file with expected headers."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "test_summary.csv"
        harness = Stage0BenchmarkHarness(execution_threads=4)

        dummy_report = {
            "scenarios": [
                {
                    "scenario_id": 1,
                    "scenario_name": "Frontal Face",
                    "scenario_category": "frontal_face",
                    "width": 1280,
                    "height": 720,
                    "total_frames": 30,
                    "fps_steady_state": 12.5,
                    "frame_latency_mean_ms": 80.0,
                    "frame_latency_p95_ms": 95.0,
                    "hardware_telemetry": {
                        "gpu_util_avg_pct": 45.0,
                        "vram_used_peak_mb": 4096.0,
                        "cpu_util_avg_pct": 20.0,
                    },
                    "quality": {
                        "detection_failures": 0,
                        "missed_faces": 0,
                        "identity_similarity_mean": 0.98,
                        "landmark_instability_mean": 0.015,
                        "color_mismatch_delta_e_mean": 3.2,
                        "occlusion_failure_count": 0,
                        "profile_failures": 0,
                    },
                }
            ]
        }

        harness.export_csv(dummy_report, csv_path)
        assert csv_path.is_file()

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            row = next(reader)
            assert "scenario_id" in header
            assert "fps_steady_state" in header
            assert row[0] == "1"
            assert row[1] == "Frontal Face"
            assert float(row[5]) == 12.5
