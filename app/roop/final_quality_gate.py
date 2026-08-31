"""Standardized Phase 16 production-quality benchmark contract.

This is a report/validation layer around the existing render harnesses. It
does not create synthetic clips, run a second swap pipeline, or infer quality
from process success. Every benchmark row remains ``not_run`` until an
evidence record supplies all required measurements.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from roop.regression_audit import ENHANCERS, QUALITY_MODES


STANDARD_CLIPS = (
    ("frontal", "frontal face"),
    ("mild_angle", "mild angle"),
    ("extreme_lateral", "extreme lateral angle"),
    ("inverted_steep", "inverted/steep pose"),
    ("fast_movement", "fast movement"),
    ("blinking", "blinking"),
    ("speaking", "speaking"),
    ("dark_scene", "dark scene"),
    ("night_scene", "night scene"),
    ("foreign_object_occlusion", "foreign-object occlusion"),
    ("hand_occlusion", "hand occlusion"),
    ("glasses_hair", "glasses/hair interaction"),
    ("two_interacting_faces", "two interacting faces"),
    ("two_crossing_faces", "two crossing faces"),
    ("mixed_lighting", "mixed lighting"),
    ("low_resolution", "low resolution"),
    ("motion_blur", "motion blur"),
)

COMPONENT_ARMS = (
    {"id": "realityux", "label": "RealityUX", "kind": "mask_engine"},
    {"id": "realswap", "label": "RealSwap", "kind": "swap_model"},
    {"id": "gpen_256_pro", "label": "GPEN 256 Pro", "kind": "enhancer"},
    {"id": "gpen_realistic", "label": "GPEN Realistic", "kind": "enhancer"},
    {"id": "ultramax", "label": "UltraMax", "kind": "enhancer"},
)

PERFORMANCE_METRICS = (
    "total_processing_time_s", "fps_equivalent", "average_frame_time_ms",
    "peak_vram_mb", "cpu_utilization_pct", "gpu_utilization_pct",
    "detection_time_ms", "swap_time_ms", "enhancer_time_ms",
    "blending_time_ms", "dropped_frames", "fallback_frames",
)

QUALITY_METRICS = (
    "identity_consistency", "temporal_flicker", "expression_consistency",
    "eye_state_consistency", "pose_consistency", "occlusion_correctness",
    "boundary_quality", "color_consistency", "low_light_realism",
    "identity_detail_retention",
)

REQUIRED_METRICS = PERFORMANCE_METRICS + QUALITY_METRICS

PREVIOUS_PHASES = (
    {"phase": "1-8", "name": "foundational detection/tracking/faceset/pose/occlusion/expression", "code_status": "regression_suite", "quality_status": "evidence_pending"},
    {"phase": "9", "name": "identity-specific detail preservation", "code_status": "regression_suite", "quality_status": "evidence_pending"},
    {"phase": "10", "name": "target-conditioned lighting and color realism", "code_status": "regression_suite", "quality_status": "evidence_pending"},
    {"phase": "11", "name": "adaptive enhancer orchestration", "code_status": "regression_suite", "quality_status": "evidence_pending"},
    {"phase": "12", "name": "temporal compositing and natural blending", "code_status": "regression_suite", "quality_status": "evidence_pending"},
    {"phase": "13", "name": "temporal artifact detection and selective correction", "code_status": "regression_suite", "quality_status": "evidence_pending"},
    {"phase": "14", "name": "end-to-end GPU performance optimization", "code_status": "regression_suite", "quality_status": "evidence_pending"},
    {"phase": "15", "name": "cross-hardware and regression validation", "code_status": "regression_suite", "quality_status": "evidence_pending"},
)


def _row_id(clip_id: str, configuration: str) -> str:
    return "%s:%s" % (clip_id, configuration)


def _empty_run(clip_id: str, configuration: str, kind: str, clip_ready: bool = False) -> dict:
    return {
        "id": _row_id(clip_id, configuration),
        "clip_id": clip_id,
        "configuration": configuration,
        "kind": kind,
        "clip_ready": bool(clip_ready),
        "status": "not_run",
        "metrics": {name: None for name in REQUIRED_METRICS},
        "evidence": None,
    }


def _clip_rows(clip_paths: Mapping[str, str | Path] | None = None) -> list[dict]:
    clip_paths = clip_paths or {}
    rows = []
    for clip_id, label in STANDARD_CLIPS:
        raw_path = clip_paths.get(clip_id)
        path = str(raw_path) if raw_path else None
        exists = bool(path and Path(path).is_file())
        rows.append({"id": clip_id, "label": label, "path": path,
                     "status": "ready" if exists else "missing",
                     "required": True})
    return rows


def _apply_evidence(rows: list[dict], evidence: Iterable[Mapping[str, Any]] | None) -> None:
    by_id = {str(row.get("id")): row for row in rows}
    for item in evidence or ():
        if not isinstance(item, Mapping) or str(item.get("id")) not in by_id:
            continue
        target = by_id[str(item["id"])]
        status = str(item.get("status", "not_run")).lower()
        metrics = dict(item.get("metrics") or {})
        target["status"] = status
        target["metrics"].update({key: metrics.get(key) for key in REQUIRED_METRICS if key in metrics})
        target["evidence"] = {key: value for key, value in item.items() if key not in ("metrics",)}


def _valid_evidence(row: Mapping[str, Any]) -> bool:
    if not row.get("clip_ready") or str(row.get("status", "")).lower() not in {"pass", "measured", "validated", "complete"}:
        return False
    metrics = row.get("metrics") or {}
    return all(metrics.get(name) is not None for name in REQUIRED_METRICS)


def _winner(rows: Iterable[Mapping[str, Any]], *, key: str, reverse: bool = False, clip_id: str | None = None) -> dict | None:
    candidates = [row for row in rows if _valid_evidence(row) and (clip_id is None or row.get("clip_id") == clip_id)]
    if not candidates:
        return None
    candidates.sort(key=lambda row: float((row.get("metrics") or {}).get(key)), reverse=reverse)
    row = candidates[0]
    return {"configuration": row["configuration"], "clip_id": row["clip_id"], "metric": key, "value": row["metrics"][key]}


def audit_faceset(path: str | Path) -> dict:
    """Validate old root-PNG archives and new metadata-bearing V2 archives."""
    path = Path(path)
    result = {"path": str(path), "status": "fail", "format": "unknown", "reason": ""}
    try:
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                result["reason"] = "corrupt ZIP member: %s" % bad
                return result
            names = archive.namelist()
            root_pngs = sorted(name for name in names if name.count("/") == 0 and name.lower().endswith(".png"))
            if not root_pngs:
                result["reason"] = "archive has no root-level PNG references"
                return result
            if "metadata.json" not in names:
                result.update({"status": "pass", "format": "legacy_v1", "root_pngs": len(root_pngs), "reason": "legacy root-PNG contract remains readable"})
                return result
            metadata = json.loads(archive.read("metadata.json").decode("utf-8"))
            required = ("schema", "version", "sources", "identity", "identity_details", "pose_bank", "integrity")
            missing = [key for key in required if key not in metadata]
            if missing or metadata.get("schema") != "roop.fsz" or int(metadata.get("version", -1)) != 2:
                result["reason"] = "V2 metadata missing/invalid: %s" % (", ".join(missing) or "schema/version")
                return result
            result.update({"status": "pass", "format": "v2", "root_pngs": len(root_pngs), "metadata_sources": len(metadata.get("sources") or []), "reason": "required V2 metadata present"})
            return result
    except (OSError, ValueError, TypeError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        result["reason"] = str(exc)
        return result


def build_report(*, clip_paths: Mapping[str, str | Path] | None = None,
                 evidence: Iterable[Mapping[str, Any]] | None = None,
                 facesets: Iterable[str | Path] = ()) -> dict:
    clips = _clip_rows(clip_paths)
    runs = [_empty_run(clip["id"], mode, "quality_mode", clip["status"] == "ready")
            for clip in clips for mode in QUALITY_MODES]
    runs.extend(_empty_run(clip["id"], arm["label"], arm["kind"], clip["status"] == "ready")
                for clip in clips for arm in COMPONENT_ARMS)
    # Every registered manual enhancer gets its own row as well. The five
    # named component arms above are the requested headline comparisons; this
    # additional matrix prevents a quieter legacy/restoration path from being
    # mistaken for covered merely because the headline arms passed.
    runs.extend(_empty_run(clip["id"], enhancer, "enhancer", clip["status"] == "ready")
                for clip in clips for enhancer in ENHANCERS)
    _apply_evidence(runs, evidence)
    measured = sum(_valid_evidence(row) for row in runs)
    report = {
        "schema": "roop-phase16-final-quality-gate-v1",
        "clips": clips,
        "quality_modes": list(QUALITY_MODES),
        "component_arms": list(COMPONENT_ARMS),
        "enhancers": list(ENHANCERS),
        "previous_phase_audit": list(PREVIOUS_PHASES),
        "performance_metrics": list(PERFORMANCE_METRICS),
        "quality_metrics": list(QUALITY_METRICS),
        "runs": runs,
        "facesets": [audit_faceset(path) for path in facesets],
        "winners": {
            "fastest_configuration": _winner(runs, key="total_processing_time_s"),
            "best_balanced_configuration": _winner(runs, key="identity_consistency", reverse=True),
            "best_quality_configuration": _winner(runs, key="identity_detail_retention", reverse=True),
            "best_night_configuration": _winner(runs, key="low_light_realism", reverse=True, clip_id="night_scene"),
            "best_difficult_angle_configuration": _winner(runs, key="pose_consistency", reverse=True, clip_id="extreme_lateral"),
            "best_multi_face_configuration": _winner(runs, key="identity_consistency", reverse=True, clip_id="two_interacting_faces"),
        },
        "summary": {
            "clip_count": len(clips),
            "ready_clips": sum(clip["status"] == "ready" for clip in clips),
            "run_count": len(runs),
            "measured_complete_runs": measured,
            "all_runs_complete": measured == len(runs),
            "facesets_pass": bool(facesets) and all(row["status"] == "pass" for row in [audit_faceset(path) for path in facesets]),
        },
        "program_gate": "OPEN_INCOMPLETE",
        "gate_reason": "real clip/configuration evidence is required for every run; availability or process success is not a quality pass",
    }
    return report


__all__ = [
    "COMPONENT_ARMS", "ENHANCERS", "PERFORMANCE_METRICS", "QUALITY_METRICS",
    "QUALITY_MODES", "PREVIOUS_PHASES", "REQUIRED_METRICS", "STANDARD_CLIPS", "audit_faceset",
    "build_report",
]
