"""One-click export presets: crop an already-produced output to a social-media
aspect ratio and re-encode it as a new file, leaving the original untouched.

This operates entirely on finished output files — it is not a swap-time option
and does not touch ProcessMgr. The natural entry point is Run History (a
finished job's output), so the trust boundary mirrors /api/output/delete: the
requested name is basename-only and must resolve to a real file directly under
roop.globals.output_path.
"""

import os

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

import roop.globals as roop_globals
from roop.util_ffmpeg import export_preset

router = APIRouter()

PRESETS = {
    "reel_9x16":      {"label": "Reels / TikTok / Shorts (9:16)", "w": 1080, "h": 1920},
    "portrait_4x5":   {"label": "Instagram Portrait (4:5)",       "w": 1080, "h": 1350},
    "square_1x1":     {"label": "Square (1:1)",                   "w": 1080, "h": 1080},
    "landscape_16x9": {"label": "YouTube / Landscape (16:9)",     "w": 1920, "h": 1080},
}


def _resolve_output_path(name: str):
    """basename -> full path under output_path, or None. Never escapes the
    output directory — mirrors delete_output's trust boundary (api.py)."""
    out = getattr(roop_globals, "output_path", None)
    if not name or not out or not os.path.isdir(out):
        return None
    full = os.path.join(out, os.path.basename(name))
    return full if os.path.isfile(full) else None


@router.get("/api/export/presets")
def list_presets():
    return {"presets": [{"key": k, **v} for k, v in PRESETS.items()]}


@router.post("/api/export/apply")
def apply_export(payload: dict = Body(...)):
    source = _resolve_output_path(payload.get("source"))
    if source is None:
        return JSONResponse(status_code=404, content={"message": "output file not found"})
    preset_key = payload.get("preset")
    preset = PRESETS.get(preset_key)
    if preset is None:
        return JSONResponse(status_code=400, content={"message": f"unknown preset '{preset_key}'"})

    out_dir = os.path.dirname(source)
    stem, ext = os.path.splitext(os.path.basename(source))
    dest = os.path.join(out_dir, f"{stem}_{preset_key}{ext}")
    n = 1
    while os.path.exists(dest):                         # never overwrite a prior export
        dest = os.path.join(out_dir, f"{stem}_{preset_key}_{n}{ext}")
        n += 1

    ok = export_preset(source, dest, preset["w"], preset["h"])
    if not ok or not os.path.exists(dest):
        return JSONResponse(status_code=500, content={
            "message": "export failed — see the terminal log"})
    return {"path": dest, "name": os.path.basename(dest)}
