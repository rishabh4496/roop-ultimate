"""Named faceset library: save, load, rename, delete, import and thumbnails.

Split out of api.py; handler bodies are unchanged, only the decorator moved
from @app to @router. Registered via app.include_router() in api.py, safe
because every /api route is a literal path with no path parameters, so
declaration order cannot change which handler matches.
"""

from fastapi import APIRouter, Body
import subprocess
import sys
import threading
import traceback
import os
import shutil

import cv2
import numpy as np

import roop.globals as roop_globals
import ui.globals as ui_globals
from roop import utilities as util
from roop.face_util import extract_face_images
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse
import api_state as state
from api_media import _bgr_to_jpg_dataurl
from source_gallery import (_frontality, _ingest_faceset,
                            _shrink_for_thumb, _source_faces_payload)


router = APIRouter()

# ── Injected by api.py at import time ────────────────────────────────────
# Never rebound in api.py, only read or mutated in place, so binding the same
# object here keeps one shared value instead of two that drift apart.
API_TEMP = None


def _faceset_library_dir() -> str:
    p = ""
    if roop_globals.CFG:
        p = getattr(roop_globals.CFG, "faceset_library_path", "") or ""
    if not p:
        p = os.path.join(os.getcwd(), "facesets")
    p = os.path.abspath(os.path.expanduser(p))
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        traceback.print_exc()
    return p

def _imread_unicode(path):
    # cv2.imread mangles non-ASCII paths on Windows (the library may live under a
    # unicode OneDrive/Dropbox path), so decode from raw bytes instead.
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def _imwrite_unicode(path, img) -> bool:
    ext = os.path.splitext(path)[1] or ".png"
    try:
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        traceback.print_exc()
        return False

def _slugify_faceset_name(name: str) -> str:
    name = (name or "").strip()
    # Keep it a safe, cross-platform filename stem.
    keep = "".join(c for c in name if c.isalnum() or c in " ._-()").strip(" .")
    return keep or "faceset"

def _unique_fsz_path(name: str) -> str:
    lib = _faceset_library_dir()
    stem = _slugify_faceset_name(name)
    path = os.path.join(lib, f"{stem}.fsz")
    n = 1
    while os.path.exists(path):
        path = os.path.join(lib, f"{stem} ({n}).fsz")
        n += 1
    return path

def _library_thumb_dataurl(fsz_path: str) -> str:
    thumb_path = os.path.splitext(fsz_path)[0] + ".png"
    if os.path.exists(thumb_path):
        return _bgr_to_jpg_dataurl(_imread_unicode(thumb_path))
    return ""

def _faceset_face_count(fsz_path: str) -> int:
    try:
        import zipfile
        with zipfile.ZipFile(fsz_path, "r") as zf:
            return sum(1 for n in zf.namelist() if n.lower().endswith(".png"))
    except Exception:
        return 0

def _library_entries() -> list:
    lib = _faceset_library_dir()
    entries = []
    try:
        names = os.listdir(lib)
    except Exception:
        names = []
    for fn in names:
        if not fn.lower().endswith(".fsz"):
            continue
        fsz_path = os.path.join(lib, fn)
        try:
            st = os.stat(fsz_path)
        except Exception:
            continue
        entries.append({
            "filename": fn,
            "name": os.path.splitext(fn)[0],
            "path": fsz_path,
            "size": st.st_size,
            "mtime": st.st_mtime,
            "faces": _faceset_face_count(fsz_path),
            "thumb": _library_thumb_dataurl(fsz_path),
        })
    entries.sort(key=lambda e: e["name"].lower())
    return entries

def _faceset_library_payload(extra=None) -> dict:
    payload = {"entries": _library_entries(), "dir": _faceset_library_dir()}
    if extra:
        payload.update(extra)
    return payload

def _frontal_crop_from_images(images):
    """Return the most frontal face crop (BGR) across a list of BGR images, so
    a multi-angle faceset shows a front-facing thumbnail instead of a profile."""
    best = None
    best_score = None
    # Unique temp name: concurrent calls (e.g. rebuild while a save runs) must
    # not clobber each other's scratch file.
    tmp = os.path.join(API_TEMP, f"_libthumb_{os.getpid()}_{threading.get_ident()}.png")
    os.makedirs(API_TEMP, exist_ok=True)
    # Cap the scan — a frontal face is found quickly and we only need a preview.
    for img in list(images)[:15]:
        if img is None:
            continue
        if best_score is not None and best_score < 0.05:
            break  # already have a near-perfect frontal crop
        _imwrite_unicode(tmp, img)
        try:
            for fd in extract_face_images(tmp, (False, 0), 0.5):
                face, crop = fd[0], fd[1]
                kps = getattr(face, "kps", None)
                if kps is None and isinstance(face, dict):
                    kps = face.get("kps")
                score = _frontality(kps) if kps is not None else 999.0
                if best_score is None or score < best_score:
                    best_score, best = score, crop
        except Exception:
            pass
    try:
        os.remove(tmp)
    except Exception:
        pass
    return best

def _write_library_thumb(fsz_path: str, images=None):
    """Write the `<name>.png` thumbnail sidecar, choosing the most frontal face.
    `images` (BGR) is used when known; otherwise the .fsz PNGs are read back."""
    try:
        if images is None:
            import zipfile
            images = []
            with zipfile.ZipFile(fsz_path, "r") as zf:
                for n in sorted(n for n in zf.namelist() if n.lower().endswith(".png")):
                    img = cv2.imdecode(np.frombuffer(zf.read(n), dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        images.append(img)
        if not images:
            return
        thumb = _frontal_crop_from_images(images)
        if thumb is None:
            thumb = images[0]  # fall back to the raw first image
        _imwrite_unicode(os.path.splitext(fsz_path)[0] + ".png", _shrink_for_thumb(thumb))
    except Exception:
        traceback.print_exc()

def _faceset_member_images(idx: int):
    """BGR images to store in the .fsz for source faceset `idx`.

    Prefer the full reference frames (facesets loaded from .fsz keep these), so
    embeddings re-average identically on reload; fall back to the cropped thumb
    for facesets that came from a single image upload.
    """
    imgs = []
    if 0 <= idx < len(roop_globals.INPUT_FACESETS):
        fs = roop_globals.INPUT_FACESETS[idx]
        for rimg in getattr(fs, "ref_images", None) or []:
            if rimg is not None:
                imgs.append(rimg)
    if not imgs and 0 <= idx < len(ui_globals.ui_input_thumbs):
        rgb = ui_globals.ui_input_thumbs[idx]
        if rgb is not None:
            imgs.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return imgs

@router.get("/api/faceset/library")
def faceset_library_list():
    return _faceset_library_payload()

@router.post("/api/faceset/library/save")
def faceset_library_save(payload: dict = Body(...)):
    idx = payload.get("index", None)
    idx = state.selected_input_face_index if idx is None else int(idx)
    if not (0 <= idx < len(roop_globals.INPUT_FACESETS)):
        return JSONResponse(status_code=400, content={"message": "no source faceset selected"})

    imgs = _faceset_member_images(idx)
    if not imgs:
        return JSONResponse(status_code=400, content={"message": "nothing to save for this faceset"})

    # Write member PNGs into an ASCII temp dir, then zip into the (possibly
    # unicode) library path — zipfile handles the unicode zipname fine.
    tmpdir = os.path.join(API_TEMP, "_libsave")
    shutil.rmtree(tmpdir, ignore_errors=True)
    os.makedirs(tmpdir, exist_ok=True)
    imgnames = []
    for j, img in enumerate(imgs):
        fn = os.path.join(tmpdir, f"{j}.png")
        _imwrite_unicode(fn, img)
        imgnames.append(fn)

    fsz_path = _unique_fsz_path(payload.get("name", ""))
    util.zip(imgnames, fsz_path)

    # Thumbnail sidecar: pick the most frontal face in the set (fall back to the
    # source panel's crop) so the preview is never a side profile.
    _write_library_thumb(fsz_path, images=imgs)
    thumb_sidecar = os.path.splitext(fsz_path)[0] + ".png"
    if not os.path.exists(thumb_sidecar) and 0 <= idx < len(ui_globals.ui_input_thumbs) \
            and ui_globals.ui_input_thumbs[idx] is not None:
        _imwrite_unicode(thumb_sidecar,
                         _shrink_for_thumb(cv2.cvtColor(ui_globals.ui_input_thumbs[idx], cv2.COLOR_RGB2BGR)))

    shutil.rmtree(tmpdir, ignore_errors=True)
    return _faceset_library_payload({"saved": os.path.splitext(os.path.basename(fsz_path))[0]})

@router.post("/api/faceset/library/load")
def faceset_library_load(payload: dict = Body(...)):
    fn = os.path.basename(str(payload.get("filename", "")))
    path = os.path.join(_faceset_library_dir(), fn)
    if not (fn.lower().endswith(".fsz") and os.path.exists(path)):
        return JSONResponse(status_code=404, content={"message": "faceset not found"})
    try:
        _ingest_faceset(path)
    except Exception:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": "failed to load faceset"})
    return _source_faces_payload()

@router.post("/api/faceset/library/rename")
def faceset_library_rename(payload: dict = Body(...)):
    fn = os.path.basename(str(payload.get("filename", "")))
    lib = _faceset_library_dir()
    old = os.path.join(lib, fn)
    if not (fn.lower().endswith(".fsz") and os.path.exists(old)):
        return JSONResponse(status_code=404, content={"message": "faceset not found"})
    # Renaming to the current name must be a no-op — _unique_fsz_path would see
    # the file itself and "dedupe" it into "<name> (1).fsz".
    if _slugify_faceset_name(payload.get("name", "")) == os.path.splitext(fn)[0]:
        return _faceset_library_payload()
    new = _unique_fsz_path(payload.get("name", ""))
    try:
        os.replace(old, new)
        old_thumb = os.path.splitext(old)[0] + ".png"
        if os.path.exists(old_thumb):
            os.replace(old_thumb, os.path.splitext(new)[0] + ".png")
    except Exception:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": "rename failed"})
    return _faceset_library_payload()

@router.post("/api/faceset/library/delete")
def faceset_library_delete(payload: dict = Body(...)):
    fn = os.path.basename(str(payload.get("filename", "")))
    lib = _faceset_library_dir()
    path = os.path.join(lib, fn)
    if fn.lower().endswith(".fsz") and os.path.exists(path):
        try:
            os.remove(path)
            thumb = os.path.splitext(path)[0] + ".png"
            if os.path.exists(thumb):
                os.remove(thumb)
        except Exception:
            traceback.print_exc()
    return _faceset_library_payload()

@router.post("/api/faceset/library/import")
def faceset_library_import(file: UploadFile = File(...)):
    base = os.path.basename(file.filename or "")
    if not base.lower().endswith(".fsz"):
        return JSONResponse(status_code=400, content={"message": "expected a .fsz file"})
    dest = _unique_fsz_path(os.path.splitext(base)[0])
    with open(dest, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    _write_library_thumb(dest)
    return _faceset_library_payload({"imported": os.path.splitext(os.path.basename(dest))[0]})

@router.post("/api/faceset/library/rebuild_thumbs")
def faceset_library_rebuild_thumbs(payload: dict = Body(default=None)):
    """Regenerate the frontal-face thumbnail sidecars from each .fsz's contents.
    Use this to refresh facesets saved before frontal-thumbnail selection existed,
    or any whose preview looks like a side profile. Optionally pass {"filename"}
    to rebuild just one; otherwise rebuilds every faceset in the library."""
    lib = _faceset_library_dir()
    only = os.path.basename(str((payload or {}).get("filename", ""))) if payload else ""
    rebuilt = 0
    for fn in os.listdir(lib) if os.path.isdir(lib) else []:
        if not fn.lower().endswith(".fsz"):
            continue
        if only and fn != only:
            continue
        _write_library_thumb(os.path.join(lib, fn))
        rebuilt += 1
    return _faceset_library_payload({"rebuilt": rebuilt})

@router.post("/api/faceset/library/open")
def faceset_library_open():
    d = _faceset_library_dir()
    try:
        if sys.platform.startswith("win"):
            os.startfile(d)  # Windows-only API; branch is platform-guarded
        elif sys.platform == "darwin":
            subprocess.Popen(["open", d])
        else:
            subprocess.Popen(["xdg-open", d])
    except Exception:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": "could not open folder"})
    return {"dir": d}
