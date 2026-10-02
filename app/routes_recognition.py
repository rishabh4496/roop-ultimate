"""Face recognition backend: selection, hardware advice and apply.

Transport only; the logic is in roop/ui_recognition.py. The selection is persisted through
Settings (config.yaml) like every other setting, and applying it hot-swaps the engine in
roop.face_analyser. See ui_recognition.SCOPE_NOTE for what a selection does and does not
change in a render.
"""

import os

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

import roop.globals as roop_globals
from roop import ui_recognition
from roop.degrade import swallowed as _swallowed

router = APIRouter(prefix="/api/recognition")

_progress = {"processing": False}


def bind_progress(progress) -> None:
    global _progress
    _progress = progress


def models_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


@router.get("")
def recognition_status():
    return ui_recognition.status(roop_globals.CFG, models_dir(),
                                 int(getattr(roop_globals, "cuda_device_id", 0)))


@router.post("/apply")
def recognition_apply(payload: dict = Body(default=None)):
    payload = payload or {}
    try:
        model, provider = ui_recognition.normalise_selection(payload.get("model"), payload.get("provider"))
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"message": str(exc)})
    if _progress.get("processing"):
        # A render holds most of the card; loading a second model set mid-run is how renders die.
        return JSONResponse(status_code=409, content={
            "message": "A render is running; apply the recognition model after it finishes."})
    try:
        result = ui_recognition.apply_selection(model, provider, models_dir())
    except Exception as exc:                                  # download / hash / provider failure
        # Nothing was saved and the previous engine (if any) is still serving.
        _swallowed("routes_recognition.py:recognition_apply", exc, "apply failed; previous engine kept")
        return JSONResponse(status_code=502, content={
            "message": "Could not load %s: %s" % (model, exc)})
    cfg = roop_globals.CFG
    if cfg is not None:
        cfg.recognition_model = model
        cfg.recognition_provider = provider
        cfg.save()
    print("[Recognition] applied %s via %s (persisted)" % (model, provider), flush=True)
    return {"status": "success", **result, "after": recognition_status()}
