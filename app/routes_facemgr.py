"""Face Manager: build a faceset from a video or images, score and prune it.

Split out of api.py; handler bodies are unchanged, only the decorator moved
from @app to @router. Registered via app.include_router() in api.py, safe
because every /api route is a literal path with no path parameters, so
declaration order cannot change which handler matches.
"""
from roop.degrade import swallowed as _swallowed

from fastapi import APIRouter, Body
import contextlib
import io
import traceback
import os
import shutil

import cv2
import numpy as np

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

fm_faces: list = []                    # Detected Face objects / dicts with 512-D embedding and 3D pose

fm_mean_embedding = None               # Global mean embedding for backward compatibility

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

def _update_fm_mean_embedding():
    global fm_mean_embedding
    vectors = []
    for f in fm_faces:
        e = getattr(f, "normed_embedding", None)
        if e is None:
            e = getattr(f, "embedding", None)
        if e is None and isinstance(f, dict):
            e = f.get("normed_embedding", f.get("embedding"))
        if e is not None:
            arr = np.asarray(e, dtype=np.float32).ravel()
            norm = float(np.linalg.norm(arr))
            if norm > 1e-6 and np.isfinite(arr).all():
                vectors.append(arr / norm)
    if vectors:
        mean_v = np.mean(vectors, axis=0)
        norm_m = float(np.linalg.norm(mean_v))
        if norm_m > 1e-6:
            fm_mean_embedding = (mean_v / norm_m).astype(np.float32).tolist()
        else:
            fm_mean_embedding = mean_v.tolist()
    else:
        fm_mean_embedding = None

def _facemgr_ingest(face, crop_bgr, restore):
    """Score one detected face, optionally restore its crop, and append it to the
    Face Manager state (image + thumbnail + FIQA score + breakdown + face object with 512-D embedding and 3D pose)."""
    from roop import face_quality
    from roop.face_clustering import extract_face_embedding, normalize_embedding
    from roop.face_util import solve_pose_5pt

    # Score the ORIGINAL crop — quality is a property of the captured face, not of
    # what the restorer can invent; a low score should still gate a blurry source.
    score, breakdown = face_quality.score_face(face, crop_bgr)
    stored = _restore_crop(crop_bgr) if restore else crop_bgr

    # Ensure 512-D ArcFace embedding is available and normalized
    emb = extract_face_embedding(face, crop_bgr)
    if emb is None:
        raw_emb = getattr(face, "embedding", None)
        if raw_emb is None and isinstance(face, dict):
            raw_emb = face.get("embedding")
        emb = normalize_embedding(raw_emb)

    # Ensure 3D head pose (yaw, pitch, roll) is estimated
    kps = getattr(face, "kps", None)
    if kps is None and isinstance(face, dict):
        kps = face.get("kps")
    pose = getattr(face, "pose", None)
    if pose is None and isinstance(face, dict):
        pose = face.get("pose")

    if pose is not None and len(pose) >= 3:
        yaw, pitch, roll = float(pose[0]), float(pose[1]), float(pose[2])
    elif kps is not None:
        try:
            solved = solve_pose_5pt(kps)
            yaw, pitch, roll = (float(solved[0]), float(solved[1]), float(solved[2])) if solved else (0.0, 0.0, 0.0)
        except Exception:
            yaw, pitch, roll = 0.0, 0.0, 0.0
    else:
        yaw, pitch, roll = 0.0, 0.0, 0.0

    if isinstance(face, dict):
        face["embedding"] = emb
        face["normed_embedding"] = emb
        face["pose"] = [yaw, pitch, roll]
        face["yaw"] = yaw
        face["pitch"] = pitch
        face["roll"] = roll
    else:
        try:
            setattr(face, "embedding", emb)
            setattr(face, "normed_embedding", emb)
            setattr(face, "pose", [yaw, pitch, roll])
            setattr(face, "yaw", yaw)
            setattr(face, "pitch", pitch)
            setattr(face, "roll", roll)
        except Exception:
            pass

    if isinstance(breakdown, dict):
        breakdown["yaw"] = round(yaw, 2)
        breakdown["pitch"] = round(pitch, 2)
        breakdown["roll"] = round(roll, 2)

    fm_images.append(stored)
    fm_thumbs.append(util.convert_to_gradio(stored))
    fm_scores.append(round(float(score), 3))
    fm_meta.append(breakdown)
    fm_faces.append(face)
    _update_fm_mean_embedding()

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
    resp = {
        "faces": [_rgb_to_dataurl(t) for t in fm_thumbs],
        "scores": list(fm_scores),
        "meta": list(fm_meta),
    }
    if fm_mean_embedding is not None:
        resp["mean_embedding"] = list(fm_mean_embedding)
    return resp

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
            except Exception as _degrade_error:
                _swallowed("routes_facemgr.py:123", _degrade_error, "fallback continued")
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
    fm_faces.clear()
    global fm_mean_embedding
    fm_mean_embedding = None
    if path.lower().endswith("fsz"):
        unzipfolder = os.path.join(os.environ.get("TEMP", API_TEMP), "faceset")
        if os.path.isdir(unzipfolder):
            shutil.rmtree(unzipfolder, ignore_errors=True)
        os.makedirs(unzipfolder, exist_ok=True)
        util.unzip(path, unzipfolder)

        from roop.faceset_v2 import read_faceset_archive
        metadata = None
        try:
            metadata = read_faceset_archive(path)
        except Exception as exc:
            _swallowed("routes_facemgr.py:facemgr_faceset:read_archive", exc, "metadata reading fallback")

        if metadata and "sources" in metadata:
            for entry in metadata.get("sources", []):
                member = entry.get("reference_member")
                if not member:
                    continue
                member_path = os.path.join(unzipfolder, member)
                if not os.path.isfile(member_path):
                    continue
                img = cv2.imread(member_path)
                if img is None:
                    continue
                ident = entry.get("identity") or {}
                emb = ident.get("normalized_embedding") or ident.get("arcface_embedding") or ident.get("embedding")
                geo = entry.get("geometry") or {}
                yaw = geo.get("yaw", 0.0)
                pitch = geo.get("pitch", 0.0)
                roll = geo.get("roll", 0.0)
                face_obj = {
                    "embedding": np.asarray(emb, dtype=np.float32) if emb else None,
                    "normed_embedding": np.asarray(emb, dtype=np.float32) if emb else None,
                    "pose": [float(yaw or 0.0), float(pitch or 0.0), float(roll or 0.0)],
                    "yaw": float(yaw or 0.0),
                    "pitch": float(pitch or 0.0),
                    "roll": float(roll or 0.0),
                    "bbox": geo.get("bbox"),
                    "kps": geo.get("landmarks_2d"),
                    "det_score": (entry.get("quality") or {}).get("detector_confidence", 0.99),
                }
                _facemgr_ingest(face_obj, img, False)

            mean_emb = metadata.get("mean_embedding") or (metadata.get("identity") or {}).get("mean_embedding") or metadata.get("default_embedding")
            if mean_emb:
                fm_mean_embedding = mean_emb
        else:
            for file_ in sorted(os.listdir(unzipfolder)):
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
        if idx < len(fm_faces):
            fm_faces.pop(idx)
        _update_fm_mean_embedding()
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
    kept_faces = [fm_faces[i] for i in keep if i < len(fm_faces)]
    fm_images.clear(); fm_images.extend(kept_imgs)
    fm_thumbs.clear(); fm_thumbs.extend(kept_thumbs)
    fm_scores.clear(); fm_scores.extend(kept_scores)
    fm_meta.clear(); fm_meta.extend(kept_meta)
    fm_faces.clear(); fm_faces.extend(kept_faces)
    _update_fm_mean_embedding()
    resp = _facemgr_faces_payload()
    resp["removed"] = before - len(keep)
    return resp

@router.post("/api/facemgr/clear")
def facemgr_clear():
    global fm_mean_embedding
    fm_thumbs.clear()
    fm_images.clear()
    fm_scores.clear()
    fm_meta.clear()
    fm_files.clear()
    fm_faces.clear()
    fm_mean_embedding = None
    return _facemgr_faces_payload()

@router.post("/api/facemgr/build")
def facemgr_build():
    if len(fm_images) < 1:
        return JSONResponse(status_code=400, content={"message": "no faces"})
    try:
        from ui.main import prepare_environment
        prepare_environment()
    except Exception:
        pass
    if not roop_globals.output_path:
        roop_globals.output_path = os.path.abspath(os.path.join(os.getcwd(), "output"))
    os.makedirs(roop_globals.output_path, exist_ok=True)
    from roop.FaceSet import FaceSet
    from roop.faceset_v2 import write_faceset_v2

    fs = FaceSet()
    fs.faces = list(fm_faces)
    fs.ref_images = list(fm_images)

    finalzip = os.path.join(roop_globals.output_path, "faceset.fsz")
    try:
        write_faceset_v2(finalzip, fs, fm_images, source_name="faceset")
    except Exception as exc:
        _swallowed("routes_facemgr.py:facemgr_build:v2", exc, "fallback to legacy zip")
        imgnames = []
        for index, img in enumerate(fm_images):
            filename = os.path.join(roop_globals.output_path, f"{index}.png")
            cv2.imwrite(filename, img)
            imgnames.append(filename)
        util.zip(imgnames, finalzip)

    return {"path": finalzip, "name": "faceset.fsz"}
