"""Face Manager: build a faceset from a video or images, score and prune it.

Split out of api.py; handler bodies are unchanged, only the decorator moved
from @app to @router. Registered via app.include_router() in api.py, safe
because every /api route is a literal path with no path parameters, so
declaration order cannot change which handler matches.
"""

from fastapi import APIRouter, Body
import contextlib
import io
import traceback
import os
import shutil

import cv2

import roop.globals as roop_globals
from roop import utilities as util
from roop.face_util import extract_face_images
from fastapi import File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from roop.capturer import get_video_frame, get_video_frame_total
import api_state as state
from api_media import _rgb_to_dataurl, _save_upload


router = APIRouter()

# ── Injected by api.py at import time ────────────────────────────────────
# Never rebound in api.py, only read or mutated in place, so binding the same
# object here keeps one shared value instead of two that drift apart.
API_TEMP = None
_FACEMGR_DETECTORS = None


# Face Manager state (separate faceset builder)
fm_thumbs: list = []                   # RGB numpy thumbnails

fm_images: list = []                   # BGR numpy full face images

fm_scores: list = []                   # FIQA quality score 0..1 per face (parallel to fm_images)

fm_meta: list = []                     # FIQA breakdown dict per face (parallel to fm_images)

fm_files: list = []                    # uploaded source paths for the facemgr tab

# Lazily-built face restorer (GFPGAN v1.4, already shipped in models/). Cleaning
# a soft/compressed reference before it is baked into the faceset gives the swap
# a sharper identity to blend from. Kept as a module singleton so repeated adds
# don't reload the ONNX session.
_fm_restorer = None

def _get_fm_restorer():
    global _fm_restorer
    if _fm_restorer is None:
        from roop.processors.Enhance_GFPGAN import Enhance_GFPGAN
        from roop.utilities import get_device
        r = Enhance_GFPGAN()
        r.Initialize({"devicename": get_device()})
        _fm_restorer = r
    return _fm_restorer

def _restore_crop(crop_bgr):
    """Return a face-restored copy of the crop, or the original on any failure."""
    try:
        out, _scale = _get_fm_restorer().Run(None, None, crop_bgr)
        return out
    except Exception:
        traceback.print_exc()
        return crop_bgr

def _facemgr_ingest(face, crop_bgr, restore):
    """Score one detected face, optionally restore its crop, and append it to the
    Face Manager state (image + thumbnail + FIQA score + breakdown)."""
    from roop import face_quality
    # Score the ORIGINAL crop — quality is a property of the captured face, not of
    # what the restorer can invent; a low score should still gate a blurry source.
    score, breakdown = face_quality.score_face(face, crop_bgr)
    stored = _restore_crop(crop_bgr) if restore else crop_bgr
    fm_images.append(stored)
    fm_thumbs.append(util.convert_to_gradio(stored))
    fm_scores.append(round(float(score), 3))
    fm_meta.append(breakdown)

@contextlib.contextmanager
def _facemgr_detector(detector):
    """Temporarily point the shared detector at `detector` for one extraction,
    then restore whatever the pipeline had, so a Face Manager choice never leaks
    into a later swap run."""
    prev = getattr(roop_globals, "detector_engine", "scrfd")
    use = detector if detector in _FACEMGR_DETECTORS else prev
    roop_globals.detector_engine = use
    try:
        yield
    finally:
        roop_globals.detector_engine = prev

def _facemgr_faces_payload():
    """Uniform response body: thumbnails + parallel scores + breakdowns."""
    return {
        "faces": [_rgb_to_dataurl(t) for t in fm_thumbs],
        "scores": list(fm_scores),
        "meta": list(fm_meta),
    }

@router.post("/api/facemgr/add")
def facemgr_add(files: list[UploadFile] = File(...),
                      detector: str = Form("scrfd"),
                      restore: bool = Form(False)):
    video_path = None
    for f in files:
        path = _save_upload(f)
        if util.has_image_extension(path):
            with _facemgr_detector(detector):
                for fd in extract_face_images(path, (False, 0), 0.5):
                    _facemgr_ingest(fd[0], fd[1], restore)
        elif util.is_video(path) or path.lower().endswith("gif"):
            fm_files.append(path)
            video_path = path
            try:
                state.current_video_fps = util.detect_fps(path)
            except Exception:
                state.current_video_fps = 30
    resp = _facemgr_faces_payload()
    if video_path:
        resp["video"] = os.path.basename(video_path)
        resp["frames"] = get_video_frame_total(video_path) or 1
    return resp

@router.post("/api/facemgr/faceset")
def facemgr_faceset(file: UploadFile = File(...)):
    path = _save_upload(file)
    fm_thumbs.clear()
    fm_images.clear()
    fm_scores.clear()
    fm_meta.clear()
    if path.lower().endswith("fsz"):
        unzipfolder = os.path.join(os.environ.get("TEMP", API_TEMP), "faceset")
        if os.path.isdir(unzipfolder):
            shutil.rmtree(unzipfolder, ignore_errors=True)
        os.makedirs(unzipfolder, exist_ok=True)
        util.unzip(path, unzipfolder)
        for file_ in os.listdir(unzipfolder):
            if file_.endswith(".png"):
                for fd in extract_face_images(os.path.join(unzipfolder, file_), (False, 0), 0.5):
                    _facemgr_ingest(fd[0], fd[1], False)
    return _facemgr_faces_payload()

@router.get("/api/facemgr/frame")
def facemgr_frame(frame: int = 1):
    if not fm_files:
        return JSONResponse(status_code=404, content={"message": "no video"})
    img = get_video_frame(fm_files[-1], frame)
    if img is None:
        return JSONResponse(status_code=404, content={"message": "no frame"})
    ok, buf = cv2.imencode(".jpg", img)
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")

@router.post("/api/facemgr/cut")
def facemgr_cut(payload: dict = Body(...)):
    frame = int(payload.get("frame", 1))
    detector = payload.get("detector", "scrfd")
    restore = bool(payload.get("restore", False))
    if not fm_files:
        return _facemgr_faces_payload()
    with _facemgr_detector(detector):
        for fd in extract_face_images(fm_files[-1], (True, frame), 0.5):
            _facemgr_ingest(fd[0], fd[1], restore)
    return _facemgr_faces_payload()

@router.post("/api/facemgr/remove")
def facemgr_remove(payload: dict = Body(...)):
    idx = int(payload.get("index", -1))
    if 0 <= idx < len(fm_thumbs):
        fm_thumbs.pop(idx)
        fm_images.pop(idx)
        if idx < len(fm_scores):
            fm_scores.pop(idx)
        if idx < len(fm_meta):
            fm_meta.pop(idx)
    return _facemgr_faces_payload()

@router.post("/api/facemgr/prune")
def facemgr_prune(payload: dict = Body(...)):
    """Drop every face whose FIQA score is below `threshold` (0..1) — the quality
    gate. Returns how many were removed alongside the surviving set."""
    try:
        threshold = float(payload.get("threshold", 0.0))
    except (TypeError, ValueError):
        threshold = 0.0
    before = len(fm_images)
    keep = [i for i, s in enumerate(fm_scores) if s >= threshold]
    # Rebuild the three parallel lists in place so any held references stay valid.
    kept_imgs = [fm_images[i] for i in keep]
    kept_thumbs = [fm_thumbs[i] for i in keep]
    kept_scores = [fm_scores[i] for i in keep]
    kept_meta = [fm_meta[i] for i in keep]
    fm_images.clear(); fm_images.extend(kept_imgs)
    fm_thumbs.clear(); fm_thumbs.extend(kept_thumbs)
    fm_scores.clear(); fm_scores.extend(kept_scores)
    fm_meta.clear(); fm_meta.extend(kept_meta)
    resp = _facemgr_faces_payload()
    resp["removed"] = before - len(keep)
    return resp

@router.post("/api/facemgr/clear")
def facemgr_clear():
    fm_thumbs.clear()
    fm_images.clear()
    fm_scores.clear()
    fm_meta.clear()
    fm_files.clear()
    return _facemgr_faces_payload()

@router.post("/api/facemgr/build")
def facemgr_build():
    if len(fm_images) < 1:
        return JSONResponse(status_code=400, content={"message": "no faces"})
    from ui.main import prepare_environment
    prepare_environment()
    imgnames = []
    for index, img in enumerate(fm_images):
        filename = os.path.join(roop_globals.output_path, f"{index}.png")
        cv2.imwrite(filename, img)
        imgnames.append(filename)
    finalzip = os.path.join(roop_globals.output_path, "faceset.fsz")
    util.zip(imgnames, finalzip)
    return {"path": finalzip, "name": "faceset.fsz"}
