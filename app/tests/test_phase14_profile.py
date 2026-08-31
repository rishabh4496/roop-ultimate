"""Regression tests for Phase 14 profile parsing and bottleneck reporting."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for path in (APP, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from baseline_controlled import parse_detailed_stage_profile
from phase14_bottleneck_report import compare


def _profile():
    return {
        "schema": "roop-phase14-stage-profile-v1",
        "mode": "event-only",
        "canonical_stages": {
            "detection": {
                "status": "measured", "calls": 2, "cpu_ms_total": 20.0,
                "gpu_event_ms_total": 4.0, "sync_ms_total": 0.0,
                "gpu_sync_window_ms_total": 0.0, "alloc_peak_mb": 10.0,
                "steady_state_allocated_mb": 8.0,
                "steady_state_reserved_mb": 12.0,
                "transfer_h2d_bytes": 0, "transfer_d2h_bytes": 0,
                "transfer_attribution": "opaque",
            }
        },
    }


def test_detailed_profile_parser_reads_marked_json():
    text = ("==== DETAILED STAGE PROFILE (ROOP_PROFILE_DETAIL) ====\n"
            '{"schema": "roop-phase14-stage-profile-v1"}\n'
            "=======================================================\n")
    assert parse_detailed_stage_profile(text)["schema"] == \
        "roop-phase14-stage-profile-v1"


def test_bottleneck_report_keeps_unobserved_stage_gaps_explicit():
    record = {"workload": {"clip_id": "fixture"}, "detailed_stage_profile": _profile()}
    report = compare(record)
    assert report["bottlenecks_by_cpu_time"][0]["stage"] == "detection"
    assert "swap" in report["measurement"]["required_stage_gaps"]
    assert report["optimization_decision"].startswith("INCOMPLETE")


def test_blend_roi_warp_matches_legacy_full_frame(monkeypatch):
    MaskingMixin = pytest.importorskip("roop.procmgr_masking").MaskingMixin

    class Bench(MaskingMixin):
        def __init__(self):
            self.options = SimpleNamespace(
                show_face_area_overlay=False, blend_ratio=1.0)

    rng = np.random.default_rng(14)
    fake = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    source = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    matrix = np.array([[1.0, 0.0, 23.0], [0.0, 1.0, 19.0]], dtype=np.float32)
    bench = Bench()

    monkeypatch.setenv("ROOP_BLEND_ROI_WARP", "0")
    legacy = bench.paste_upscale(
        fake, fake, matrix, source.copy(), 1, [0, 0, 0, 0, 0, 0],
        inplace=True)
    monkeypatch.setenv("ROOP_BLEND_ROI_WARP", "1")
    optimized = bench.paste_upscale(
        fake, fake, matrix, source.copy(), 1, [0, 0, 0, 0, 0, 0],
        inplace=True)
    assert np.array_equal(optimized, legacy)
