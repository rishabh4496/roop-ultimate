"""Contracts for the Phase 16 standardized final-quality gate."""

import json
import zipfile
from pathlib import Path

from roop.final_quality_gate import (
    COMPONENT_ARMS,
    PERFORMANCE_METRICS,
    QUALITY_METRICS,
    PREVIOUS_PHASES,
    REQUIRED_METRICS,
    STANDARD_CLIPS,
    audit_faceset,
    build_report,
)
from roop.regression_audit import ENHANCERS, QUALITY_MODES


def test_manifest_has_all_requested_clip_categories_and_arms():
    assert len(STANDARD_CLIPS) == 17
    assert {item[0] for item in STANDARD_CLIPS} >= {"frontal", "night_scene", "two_crossing_faces", "motion_blur"}
    assert [item["label"] for item in COMPONENT_ARMS] == ["RealityUX", "RealSwap", "GPEN 256 Pro", "GPEN Realistic", "UltraMax"]
    assert QUALITY_MODES == ("FAST", "BALANCED", "REALISTIC", "MAX QUALITY")


def test_report_has_every_clip_mode_and_component_run_and_no_false_winners():
    report = build_report()
    assert report["summary"]["run_count"] == 17 * (4 + 5 + len(ENHANCERS))
    assert report["summary"]["measured_complete_runs"] == 0
    assert report["winners"]["fastest_configuration"] is None
    assert report["program_gate"] == "OPEN_INCOMPLETE"
    assert set(report["performance_metrics"]) == set(PERFORMANCE_METRICS)
    assert set(report["quality_metrics"]) == set(QUALITY_METRICS)


def test_incomplete_evidence_cannot_select_a_winner(tmp_path: Path):
    report = build_report(evidence=[{"id": "frontal:FAST", "status": "pass", "metrics": {"fps_equivalent": 100.0}}])
    assert report["summary"]["measured_complete_runs"] == 0
    assert report["winners"]["fastest_configuration"] is None
    assert len(REQUIRED_METRICS) == len(PERFORMANCE_METRICS) + len(QUALITY_METRICS)


def test_evidence_for_missing_clip_cannot_be_called_complete():
    metrics = {name: 1.0 for name in REQUIRED_METRICS}
    report = build_report(evidence=[{"id": "frontal:FAST", "status": "pass", "metrics": metrics}])
    assert report["summary"]["measured_complete_runs"] == 0
    assert report["winners"]["fastest_configuration"] is None


def test_legacy_and_v2_facesets_are_audited_separately(tmp_path: Path):
    old = tmp_path / "old.fsz"
    new = tmp_path / "new.fsz"
    with zipfile.ZipFile(old, "w") as archive:
        archive.writestr("0.png", b"png")
    metadata = {"schema": "roop.fsz", "version": 2, "sources": [{"member": "0.png"}], "identity": {}, "identity_details": {}, "pose_bank": {}, "integrity": {}}
    with zipfile.ZipFile(new, "w") as archive:
        archive.writestr("0.png", b"png")
        archive.writestr("metadata.json", json.dumps(metadata))
    assert audit_faceset(old)["format"] == "legacy_v1"
    assert audit_faceset(old)["status"] == "pass"
    assert audit_faceset(new)["format"] == "v2"
    assert audit_faceset(new)["status"] == "pass"


def test_source_enhancer_set_is_not_reduced():
    assert {"RealityUX", "RealSwap"} <= {item["label"] for item in COMPONENT_ARMS}
    assert {"GPEN 256 Pro", "GPEN Realistic", "UltraMax"} <= set(ENHANCERS)
    assert {"9", "10", "11", "12", "13", "14", "15"} <= {row["phase"] for row in PREVIOUS_PHASES}


def load_tests(loader, tests, pattern):
    """Expose this module's bare `test_*` functions to `unittest discover`.

    Without this, `unittest` collects nothing here and reports OK; see
    tests/unittest_shim.py. pytest never calls load_tests, so it is unaffected.
    """
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:  # discovery started from inside tests/
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
