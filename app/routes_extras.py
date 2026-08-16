"""Standalone frame operations: enhance, upscale, colorize, filters.

Split out of api.py; handler bodies are unchanged, only the decorator moved
from @app to @router. Registered via app.include_router() in api.py, which is
safe here because every /api route is a literal path with no path parameters,
so declaration order cannot change which handler matches.
"""

from fastapi import APIRouter
import os
import traceback

import cv2
from fastapi import File, Form, UploadFile
from fastapi.responses import JSONResponse

import roop.globals as roop_globals
from roop import utilities as util
from roop.capturer import (get_image_frame, get_video_frame,
                           get_video_frame_total)
from api_media import _save_upload


router = APIRouter()

# ── Injected by api.py at import time ────────────────────────────────────
# Verified to be mutated in place and never rebound in api.py, so binding the
# same object here keeps one shared value rather than two that drift.
_FRAME_COLORIZERS = None
_FRAME_FILTERS = None
_FRAME_UPSCALERS = None


# ── Extras: media editor (resize / rotate / fps / crop) ──────────────────────
@router.post("/api/extras/apply")
def extras_apply(file: UploadFile = File(...),
                       resolution: str = Form("Original"),
                       rotation: str = Form("None"),
                       fps: float = Form(30.0),
                       crop_left: int = Form(0), crop_right: int = Form(0),
                       crop_top: int = Form(0), crop_bottom: int = Form(0)):
    from ui.main import prepare_environment
    prepare_environment()
    path = _save_upload(file)
    is_video = util.is_video(path) or path.lower().endswith("gif")

    def _process_frame(img):
        h, w = img.shape[:2]
        x0 = int(w * crop_left / 100)
        x1 = w - int(w * crop_right / 100)
        y0 = int(h * crop_top / 100)
        y1 = h - int(h * crop_bottom / 100)
        if x1 > x0 and y1 > y0:
            img = img[y0:y1, x0:x1]
        if rotation == "90° Clockwise":
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == "90° Counter-Clockwise":
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation == "180°":
            img = cv2.rotate(img, cv2.ROTATE_180)
        if resolution and resolution != "Original":
            try:
                target_w = int(str(resolution).split("x")[0])
                scale = target_w / img.shape[1]
                img = cv2.resize(img, (target_w, int(img.shape[0] * scale)))
            except Exception:
                pass
        return img

    out_dir = roop_globals.output_path
    if not is_video:
        img = get_image_frame(path)
        out = _process_frame(img)
        outpath = os.path.join(out_dir, "edited_" + os.path.splitext(os.path.basename(path))[0] + ".png")
        cv2.imwrite(outpath, out)
        return {"path": outpath, "kind": "image"}

    total = get_video_frame_total(path) or 1
    first_frame = get_video_frame(path, 1)
    if first_frame is None:
        return JSONResponse(status_code=400, content={"message": "could not read the video's first frame"})
    first = _process_frame(first_frame)
    oh, ow = first.shape[:2]
    outpath = os.path.join(out_dir, "edited_" + os.path.splitext(os.path.basename(path))[0] + ".mp4")
    writer = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    for i in range(1, total + 1):
        fr = get_video_frame(path, i)
        if fr is None:
            continue
        writer.write(_process_frame(fr))
    writer.release()
    return {"path": outpath, "kind": "video"}

def _make_frame_processor(operation: str, subtype: str):
    """Instantiate + initialize the requested frame processor. Returns a class
    whose Run(frame)->frame does the work."""
    from roop.utilities import get_device
    devicename = get_device()
    if operation == "upscale":
        from roop.processors.Frame_Upscale import Frame_Upscale
        proc = Frame_Upscale()
    elif operation == "colorize":
        from roop.processors.Frame_Colorizer import Frame_Colorizer
        proc = Frame_Colorizer()
    elif operation == "filter":
        from roop.processors.Frame_Filter import Frame_Filter
        proc = Frame_Filter()
    else:
        raise ValueError(f"unknown operation {operation}")
    proc.Initialize({"devicename": devicename, "subtype": subtype})
    return proc

@router.get("/api/extras/frame_ops")
def get_frame_ops():
    return {"upscale": _FRAME_UPSCALERS, "colorize": _FRAME_COLORIZERS, "filter": _FRAME_FILTERS}

@router.post("/api/extras/enhance")
def extras_enhance(file: UploadFile = File(...),
                         operation: str = Form("upscale"),
                         subtype: str = Form("esrganx2")):
    """Run a frame post-processor (AI upscale / colorize / stylize) over an
    uploaded image or video and write the result to the output folder."""
    from ui.main import prepare_environment
    prepare_environment()   # ensures the Frame/* models are downloaded

    valid = {"upscale": _FRAME_UPSCALERS, "colorize": _FRAME_COLORIZERS, "filter": _FRAME_FILTERS}
    if operation not in valid or subtype not in valid[operation]:
        return JSONResponse(status_code=400, content={"message": "invalid operation/subtype"})

    path = _save_upload(file)
    is_video = util.is_video(path) or path.lower().endswith("gif")
    out_dir = roop_globals.output_path
    stem = os.path.splitext(os.path.basename(path))[0]

    try:
        proc = _make_frame_processor(operation, subtype)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"failed to load model: {e}"})

    try:
        if not is_video:
            img = get_image_frame(path)
            if img is None:
                return JSONResponse(status_code=400, content={"message": "could not read image"})
            out = proc.Run(img)
            outpath = os.path.join(out_dir, f"{operation}_{subtype}_{stem}.png")
            cv2.imwrite(outpath, out)
            return {"path": outpath, "kind": "image"}

        total = get_video_frame_total(path) or 1
        first = get_video_frame(path, 1)
        if first is None:
            return JSONResponse(status_code=400, content={"message": "could not read the video's first frame"})
        out_first = proc.Run(first)
        oh, ow = out_first.shape[:2]
        outpath = os.path.join(out_dir, f"{operation}_{subtype}_{stem}.mp4")
        try:
            fps = util.detect_fps(path)
        except Exception:
            fps = 30
        writer = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
        writer.write(out_first)
        for i in range(2, total + 1):
            fr = get_video_frame(path, i)
            if fr is None:
                continue
            res = proc.Run(fr)
            if res.shape[:2] != (oh, ow):
                res = cv2.resize(res, (ow, oh))
            writer.write(res)
        writer.release()
        return {"path": outpath, "kind": "video"}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": f"processing failed: {e}"})
    finally:
        try:
            proc.Release()
        except Exception:
            pass
