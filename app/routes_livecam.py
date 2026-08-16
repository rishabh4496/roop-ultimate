"""Live camera preview endpoints.

Split out of api.py; handler bodies are unchanged, only the decorator moved
from @app to @router. Registered via app.include_router() in api.py, which is
safe here because every /api route is a literal path with no path parameters,
so declaration order cannot change which handler matches.
"""

from fastapi import APIRouter, Body
import traceback

import cv2
from fastapi.responses import JSONResponse, Response

import ui.globals as ui_globals


router = APIRouter()

# ── Injected by api.py at import time ────────────────────────────────────
# Verified to be mutated in place and never rebound in api.py, so binding the
# same object here keeps one shared value rather than two that drift.
_progress = None


@router.get("/api/livecam/status")
def livecam_status():
    try:
        from roop import virtualcam
        return {"active": bool(virtualcam.cam_active)}
    except Exception:
        return {"active": False}

@router.post("/api/livecam/start")
def livecam_start(payload: dict = Body(...)):
    if _progress["processing"]:
        return JSONResponse(status_code=409, content={"message": "busy processing"})
    try:
        from roop import virtualcam
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"message": f"virtual camera support unavailable: {e}"})
    if virtualcam.cam_active:
        return {"active": True}
    try:
        cam_number = int(payload.get("cam_number", 0))
        resolution = str(payload.get("resolution", "1280x720"))
        if "x" not in resolution:
            resolution = "1280x720"
        virtualcam.start_virtual_cam(
            bool(payload.get("stream_obs", False)),
            bool(payload.get("use_xseg", False)),
            bool(payload.get("restore_mouth", False)),
            cam_number, resolution)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"could not start camera: {e}"})
    # The capture thread opens the device asynchronously — the UI re-polls
    # /status shortly after to confirm it actually came up.
    return {"starting": True}

@router.post("/api/livecam/stop")
def livecam_stop():
    try:
        from roop import virtualcam
        virtualcam.stop_virtual_cam()
    except Exception:
        traceback.print_exc()
    return {"active": False}

@router.get("/api/livecam/frame")
def livecam_frame():
    frame = getattr(ui_globals, "ui_camera_frame", None)
    if frame is None:
        return JSONResponse(status_code=404, content={"message": "no frame yet"})
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return JSONResponse(status_code=500, content={"message": "encode failed"})
    return Response(content=buf.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})
