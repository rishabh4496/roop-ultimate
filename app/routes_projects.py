"""HTTP boundary for persistent processing projects."""

from __future__ import annotations

import os
import json

import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse

import project_checkpoint as checkpoints


router = APIRouter()


def _api():
    # api.py includes this router before wiring its shared objects. Importing at
    # request time avoids a module cycle while keeping one state owner.
    import api
    return api


def _failure(record, reasons):
    reasons = list(reasons)
    checkpoints.update_state(record["id"], "RECOVERABLE", "; ".join(reasons))
    return JSONResponse(status_code=409, content={
        "message": "cannot safely resume this project",
        "recoverability_error": True,
        "project": checkpoints.summarize(record, reasons),
        "reasons": reasons,
    })


def _restore_face(data):
    from insightface.app.common import Face
    restored = {}
    for key, value in (data or {}).items():
        if key in ("bbox", "kps", "embedding", "landmark_2d_106", "landmark_3d_68"):
            restored[key] = np.asarray(value, dtype=np.float32)
        else:
            restored[key] = value
    return Face(**restored)


def _recover_stale(record):
    api = _api()
    if (record.get("state") == "PROCESSING" and
            not getattr(api, "_progress", {}).get("processing", False)):
        record = checkpoints.update_state(
            record["id"], "RECOVERABLE",
            "application stopped before the project completed") or record
    return record


def _load_into_runtime(record):
    api = _api()
    from source_gallery import _ingest_faceset, _sources_clear, _sources_append
    from roop.face_util import extract_face_images
    from roop.FaceSet import FaceSet
    import roop.globals as globals_
    import ui.globals as ui_globals
    from roop.ProcessEntry import ProcessEntry

    _sources_clear()
    globals_.source_path = None
    for source in (record.get("inputs") or {}).get("sources") or []:
        path = source["path"]
        if path.lower().endswith(".fsz"):
            _ingest_faceset(path)
            continue
        faces = extract_face_images(path, (False, 0))
        if not faces:
            raise ValueError(f"no source face could be restored from {path}")
        globals_.source_path = path
        for fd in faces:
            faceset = FaceSet()
            faceset._source_path = os.path.abspath(path)
            face = fd[0]
            faceset.faces.append(face)
            _sources_append(faceset, api.util.convert_to_gradio(fd[1]))

    target = (record.get("inputs") or {}).get("target") or {}
    target_path = target.get("path")
    api.list_files_process.clear()
    globals_.TARGET_FACES.clear()
    globals_.TARGET_FACE_GROUP.clear()
    if getattr(globals_, "TARGET_FACE_NAMES", None):
        globals_.TARGET_FACE_NAMES.clear()
    ui_globals.ui_target_thumbs.clear()
    entry = ProcessEntry(target_path, 0, 0, 0)
    api.list_files_process.append(entry)
    api._refresh_target_frames(0)
    entry.startframe = int((record.get("inputs") or {}).get("frame_start", 0) or 0)
    saved_end = int((record.get("inputs") or {}).get("frame_end", 0) or 0)
    if saved_end:
        entry.endframe = min(saved_end, int(getattr(entry, "total_frames", saved_end) or saved_end))
    api.state.selected_target_index = 0
    globals_.target_path = target_path
    output_directory = (record.get("output") or {}).get("directory")
    if output_directory:
        globals_.output_path = os.path.abspath(output_directory)

    for item in (record.get("inputs") or {}).get("target_faces") or []:
        face = _restore_face(item.get("data"))
        globals_.TARGET_FACES.append(face)
        globals_.TARGET_FACE_GROUP.append(int(item.get("group", len(globals_.TARGET_FACE_GROUP))))
        bbox = np.asarray(getattr(face, "bbox", face.get("bbox")), dtype=np.int32)
        thumbnail = item.get("thumbnail") or ""
        if thumbnail:
            try:
                ui_globals.ui_target_thumbs.append(
                    api.util.convert_to_gradio(api._dataurl_to_bgr(thumbnail)))
                continue
            except Exception:
                pass
        # Older records have no thumbnail. It is display-only; processing uses
        # the restored detector object above, so do not run a detector here.
        frame = np.zeros((max(1, int(bbox[3] + 2)), max(1, int(bbox[2] + 2)), 3), dtype=np.uint8)
        ui_globals.ui_target_thumbs.append(frame)
    return api.get_state()


def _validated(record):
    api = _api()
    reasons = checkpoints.validate(record, api.roop_globals.CFG)
    return reasons


@router.get("/api/projects")
def projects_list():
    api = _api()
    result = []
    for record in checkpoints.list_projects():
        record = _recover_stale(record)
        reasons = checkpoints.validate(record, api.roop_globals.CFG)
        result.append(checkpoints.summarize(record, reasons))
    return {"projects": result}


@router.get("/api/projects/{project_id}")
def project_detail(project_id: str):
    try:
        record = checkpoints.load(project_id)
    except (OSError, ValueError, json.JSONDecodeError):
        return JSONResponse(status_code=404, content={"message": "project not found"})
    record = _recover_stale(record)
    return {"project": checkpoints.summarize(record, _validated(record)), "record": record}


@router.post("/api/projects/{project_id}/validate")
def project_validate(project_id: str):
    try:
        record = checkpoints.load(project_id)
    except (OSError, ValueError, json.JSONDecodeError):
        return JSONResponse(status_code=404, content={"message": "project not found"})
    record = _recover_stale(record)
    reasons = _validated(record)
    if reasons:
        return _failure(record, reasons)
    return {"project": checkpoints.summarize(record), "valid": True}


@router.post("/api/projects/{project_id}/load")
def project_load(project_id: str):
    try:
        record = checkpoints.load(project_id)
    except (OSError, ValueError, json.JSONDecodeError):
        return JSONResponse(status_code=404, content={"message": "project not found"})
    record = _recover_stale(record)
    reasons = _validated(record)
    if reasons:
        return _failure(record, reasons)
    try:
        api = _api()
        if (getattr(api, "_active_project_id", "") == project_id and
                getattr(api, "_progress", {}).get("processing")):
            return JSONResponse(status_code=409, content={
                "message": "project is already active; use Resume on the live job"})
        state = _load_into_runtime(record)
    except Exception as exc:
        return _failure(record, [f"project inputs could not be reloaded: {exc}"])
    return {"project": checkpoints.summarize(record), "state": state}


@router.post("/api/projects/{project_id}/resume")
def project_resume(project_id: str):
    try:
        record = checkpoints.load(project_id)
    except (OSError, ValueError, json.JSONDecodeError):
        return JSONResponse(status_code=404, content={"message": "project not found"})
    record = _recover_stale(record)
    reasons = _validated(record)
    if reasons:
        return _failure(record, reasons)
    try:
        # A paused in-process project already owns the live worker and model
        # sessions. Wake that worker instead of reloading globals underneath it.
        api = _api()
        if (record.get("state") == "PAUSED" and
                getattr(api, "_active_project_id", "") == project_id and
                getattr(api, "_progress", {}).get("processing")):
            return api.resume_swap()
        _load_into_runtime(record)
        import routes_queue
        if record.get("job_id"):
            return routes_queue.resume_project_job(project_id)
        payload = dict((record.get("settings") or {}).get("payload") or {})
        return _api()._start_existing_project(project_id, payload)
    except Exception as exc:
        return _failure(record, [f"project could not be loaded: {exc}"])
