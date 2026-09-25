"""Headless execution for portable ``.roop`` sessions."""

from __future__ import annotations

import copy
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import project_checkpoint
from roop import nle_interchange, project_io


def _safe_id(document: dict, project_path: str) -> str:
    raw = str(document.get("id") or Path(project_path).stem)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    return safe or "roop_project"


def _checkpoint_from_project(document: dict, project_path: str, cfg: Any) -> dict:
    """Build the existing runtime checkpoint shape from a standalone .roop file."""
    inputs = project_io.to_checkpoint_payload(document, project_path)
    source_items = []
    for item in inputs["sources"]:
        path = item.get("path") or ""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"source media is missing: {path}")
        identity = project_checkpoint.file_identity(path)
        identity.update({k: item[k] for k in ("id", "source_id") if k in item})
        source_items.append(identity)
    target_path = inputs["target"].get("path") or ""
    if not os.path.isfile(target_path):
        raise FileNotFoundError(f"target media is missing: {target_path}")
    target = project_checkpoint.file_identity(target_path)
    target.update({k: inputs["target"][k] for k in ("id", "target_media_id") if k in inputs["target"]})
    face_bank = document.get("face_bank") or {}
    context = {
        "target_media_id": target.get("target_media_id"),
        "face_mapping": copy.deepcopy(face_bank.get("target_to_source") or {}),
        "target_person_source_mapping": copy.deepcopy(face_bank.get("target_to_source") or {}),
        "target_person_names": copy.deepcopy(face_bank.get("target_person_names") or {}),
        "clusters": copy.deepcopy(face_bank.get("clusters") or []),
        "target_person_ids": copy.deepcopy(face_bank.get("target_person_ids") or []),
        "target_reference_face_ids": copy.deepcopy(face_bank.get("target_reference_face_ids") or []),
    }
    output = copy.deepcopy(document.get("render") or {})
    output.setdefault("directory", os.path.dirname(os.path.abspath(project_path)))
    payload = copy.deepcopy(document.get("settings") or {})
    project_id = _safe_id(document, project_path)
    record = {
        "schema_version": project_checkpoint.PROJECT_SCHEMA_VERSION,
        "id": project_id,
        "job_id": None,
        "name": document.get("name") or Path(project_path).stem,
        "state": "PROCESSING",
        "created_at": document.get("created_at", time.time()),
        "updated_at": time.time(),
        "application": {"compatibility": dict(project_checkpoint.COMPATIBILITY), "version": "headless"},
        "inputs": {
            "sources": source_items,
            "target": target,
            "frame_start": inputs["frame_start"],
            "frame_end": inputs["frame_end"],
            "target_faces": copy.deepcopy(face_bank.get("embeddings") or []),
            "target_context": context,
        },
        "settings": {"payload": payload, "fingerprint": project_checkpoint.fingerprint(payload)},
        "runtime": project_checkpoint.runtime_identity(payload, cfg),
        "output": output,
        "checkpoint": copy.deepcopy(document.get("checkpoint") or {"sequence": 0, "safe_frame": inputs["frame_start"], "next_frame": inputs["frame_start"], "segments": []}),
        "partial_output": {"files": [], "integrity": "unknown"},
        "error": "",
    }
    record["timeline"] = copy.deepcopy(document.get("timeline") or {})
    record["automation"] = copy.deepcopy(document.get("automation") or {})
    return record


def _ensure_checkpoint(document: dict, project_path: str, cfg: Any) -> dict:
    project_id = _safe_id(document, project_path)
    try:
        record = project_checkpoint.load(project_id)
        return record
    except (OSError, ValueError, KeyError):
        record = _checkpoint_from_project(document, project_path, cfg)
        project_checkpoint.save(record)
        return record


def _apply_output_override(record: dict, output: str | None) -> None:
    if not output:
        return
    requested = os.path.abspath(os.path.expanduser(output))
    suffix = os.path.splitext(requested)[1]
    is_file = bool(suffix) and not os.path.isdir(requested)
    directory = os.path.dirname(requested) if is_file else requested
    os.makedirs(directory, exist_ok=True)
    saved_output = record.setdefault("output", {})
    saved_output["directory"] = directory
    if is_file:
        saved_output["filename"] = os.path.basename(requested)
    else:
        saved_output.pop("filename", None)
    project_checkpoint.save(record)


def export_project(project_path: str, *, fcpxml: str | None = None, edl: str | None = None,
                   detect_cuts: bool = False) -> list[str]:
    document = project_io.load_project(project_path)
    target = project_io.resolve_media((document.get("media") or {}).get("target") or {}, project_path)
    cuts = None
    if detect_cuts:
        cuts = nle_interchange.detect_scene_cuts(target)
        document.setdefault("timeline", {})["scene_cuts"] = cuts
        project_io.save_project(project_path, document)
    generated = []
    if fcpxml:
        generated.append(nle_interchange.export_fcpxml(document, project_path, fcpxml, scene_cuts=cuts))
    if edl:
        generated.extend(nle_interchange.export_resolve_edl(document, project_path, edl, scene_cuts=cuts))
    return generated


def render_project(project_path: str, *, output: str | None = None, cfg: Any = None) -> int:
    """Load a project into the normal API renderer and wait for completion."""
    document = project_io.load_project(project_path)
    if cfg is None:
        from settings import Settings
        cfg = Settings("config.yaml")
    record = _ensure_checkpoint(document, project_path, cfg)
    _apply_output_override(record, output)

    # Core has completed provider admission before this function is called.
    # Reuse the exact project loader and worker used by the React Resume button.
    import api
    import routes_projects
    import roop.globals as globals_

    routes_projects._load_into_runtime(record)
    globals.output_path = record.get("output", {}).get("directory") or globals.output_path
    globals._project_output_file = (record.get("output") or {}).get("filename") or None
    response = api._start_existing_project(record["id"], (record.get("settings") or {}).get("payload") or {})
    if not isinstance(response, dict) or response.get("status") != "started":
        print(f"[Project] could not start render: {response}", file=sys.stderr)
        return 2

    while api._progress.get("processing"):
        progress = float(api._progress.get("progress", 0.0) or 0.0) * 100.0
        desc = str(api._progress.get("desc") or "")
        print(f"[Project] {progress:6.2f}% {desc}", flush=True)
        time.sleep(2.0)
    if api._progress.get("error"):
        print(f"[Project] render failed: {api._progress['error']}", file=sys.stderr)
        return 1
    return 0


def run_project_cli(args: Any, cfg: Any) -> int:
    project_path = os.path.abspath(os.path.expanduser(str(args.project)))
    if not os.path.isfile(project_path):
        print(f"project file not found: {project_path}", file=sys.stderr)
        return 2
    if getattr(args, "export_fcpxml", None) or getattr(args, "export_edl", None) or getattr(args, "scene_detect", False):
        generated = export_project(project_path, fcpxml=args.export_fcpxml, edl=args.export_edl, detect_cuts=args.scene_detect)
        for path in generated:
            print(path)
        if not getattr(args, "render", False):
            return 0
    if getattr(args, "render", False):
        return render_project(project_path, output=args.output, cfg=cfg)
    return 0


__all__ = ["export_project", "render_project", "run_project_cli"]
