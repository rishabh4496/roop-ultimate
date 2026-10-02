"""Face recognition backend: model list, current state, and live hot-swap.

    GET  /api/recognition/models    every registered backbone (key, display_name, dim, description, ...)
    GET  /api/recognition/current   saved selection, what is loaded, hardware advice, provider options
    POST /api/recognition/set       {"model_name": str, "provider": str}: build, then persist, then report

Transport only; the logic is in roop/ui_recognition.py. The choice is persisted through Settings
(config.yaml) like every other setting, and `set` hot-swaps the engine in roop.face_analyser. See
ui_recognition.SCOPE_NOTE for what a selection does and does not change in a render.

Errors use the app's {"message": ...} shape (the React client reads `message`):
  400 unknown model / provider / missing model_name   409 a render is running
  500 the model could not be loaded (download, hash, provider); the exception text is returned and
      the traceback goes to the server log only -- the API can be exposed on a network, and a
      traceback in a response is file paths and internals handed to whoever asks.
"""

import os
import traceback

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


@router.get("/models")
def recognition_models():
    return {"models": ui_recognition.model_catalog(models_dir())}


@router.get("/current")
def recognition_current():
    return ui_recognition.current(roop_globals.CFG, models_dir(),
                                  int(getattr(roop_globals, "cuda_device_id", 0)))


@router.post("/set")
def recognition_set(payload: dict = Body(default=None)):
    payload = payload or {}
    if not payload.get("model_name"):
        return JSONResponse(status_code=400, content={"message": "model_name is required"})
    try:
        model, provider = ui_recognition.normalise_selection(payload.get("model_name"), payload.get("provider"))
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
        _swallowed("routes_recognition.py:recognition_set", exc, "apply failed; previous engine kept")
        print("[Recognition] apply of %s via %s failed:\n%s" % (model, provider, traceback.format_exc()), flush=True)
        return JSONResponse(status_code=500, content={
            "message": "Could not load %s: %s: %s" % (model, type(exc).__name__, exc)})
    cfg = roop_globals.CFG
    if cfg is not None:
        cfg.recognition_model = model
        cfg.recognition_provider = provider
        cfg.save()
    print("[Recognition] applied %s via %s (persisted)" % (model, provider), flush=True)
    return {"status": "success", **result, "after": recognition_current()}
