"""Face Manager extras: the Pose Radar, Auto-Prune, and the Embedding Mixer & Morph Lab.

The rules live in ``roop/identity_lab.py`` (pure, tested against hand-built sets). This module is the
thin layer that reads the live Face Manager lists (``routes_facemgr``) and the source gallery, and
writes results back. Registered in api.py beside the other routers; every route is a literal path.

The Morph Lab blends 2-5 identities on the ArcFace hypersphere (``identity_algebra.blend_vectors``: Slerp or
linear) and can push the result away from reference identities (``identity_algebra.deidentify``).
The blend can be loaded as a Face Swap SOURCE to preview on the active frame, or saved as a V2 .fsz
whose cached identity IS the blend: ``FaceSet.attach_v2_metadata`` reads ``default_embedding`` and
``pose_bins`` from the archive, not the embedding of the detected photo, so a saved blend survives
reload. (Loading re-detects a face in the stored reference image, which is why the anchor image has to
be a real photo of the dominant identity rather than a bare vector.)
"""
from __future__ import annotations

import os
import shutil
import time
import traceback
import zipfile

import cv2
import numpy as np
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

import api_state as state
import roop.globals as roop_globals
import routes_facemgr as fm
from api_media import _rgb_to_dataurl
from roop import identity_lab as lab
from roop import utilities as util
from roop.degrade import swallowed as _swallowed
from roop.identity_algebra import BLEND_MODES, DEFAULT_BLEND_MODE, DEFAULT_PRIVACY_COSINE, MAX_BLEND_SOURCES

router = APIRouter()


# ── Pose radar + Auto-Prune ─────────────────────────────────────────────────

def _keep_only(keep) -> None:
    """Rebuild the Face Manager's parallel lists keeping the given indices, in place so any held
    reference stays valid."""
    keep = sorted({int(i) for i in keep})
    kept = {
        "imgs": [fm.fm_images[i] for i in keep],
        "thumbs": [fm.fm_thumbs[i] for i in keep],
        "scores": [fm.fm_scores[i] for i in keep],
        "meta": [fm.fm_meta[i] for i in keep],
        "faces": [fm.fm_faces[i] for i in keep if i < len(fm.fm_faces)],
    }
    for target, key in ((fm.fm_images, "imgs"), (fm.fm_thumbs, "thumbs"), (fm.fm_scores, "scores"),
                        (fm.fm_meta, "meta"), (fm.fm_faces, "faces")):
        target.clear()
        target.extend(kept[key])
    fm._update_fm_mean_embedding()


def _embedding_of(face):
    e = getattr(face, "normed_embedding", None)
    if e is None:
        e = getattr(face, "embedding", None)
    if e is None and isinstance(face, dict):
        e = face.get("normed_embedding", face.get("embedding"))
    return e


def _state():
    rows = lab.face_rows(fm.fm_meta, fm.fm_scores)
    return rows, [_embedding_of(f) for f in fm.fm_faces]


@router.get("/api/facemgr/radar")
def facemgr_radar():
    """Pose coverage of the current set (yaw/pitch buckets, sharpness) and what Auto-Prune would do."""
    rows, embeddings = _state()
    out = lab.radar(rows)
    out["faces"] = rows
    out["prune_preview"] = lab.plan_prune(rows, embeddings) if rows else None
    out["bucket_order"] = list(lab.BUCKETS)
    return out


@router.post("/api/facemgr/auto_prune")
def facemgr_auto_prune(payload: dict = Body(default=None)):
    """Drop blurry and near-duplicate faces. ``dry_run: true`` returns the plan and changes nothing."""
    payload = payload or {}
    rows, embeddings = _state()
    if not rows:
        return JSONResponse(status_code=400, content={"message": "no faces to prune"})
    plan = lab.plan_prune(rows, embeddings,
                          duplicate_cosine=float(payload.get("duplicate_cosine", lab.DUPLICATE_COSINE)))
    dry = bool(payload.get("dry_run"))
    if not dry:
        _keep_only(plan["keep"])
    resp = fm._facemgr_faces_payload()
    resp["removed"] = 0 if dry else len(plan["remove"])
    resp["plan"] = plan
    resp["dry_run"] = dry
    return resp


# ── Embedding Mixer & Morph Lab ─────────────────────────────────────────────

_embedding_cache = {}          # (abspath, mtime_ns, size) -> unit embedding


def _is_v2(path):
    """Does the archive carry V2 metadata (cached embeddings)? Cheap: reads the member list only."""
    from roop.faceset_v2 import METADATA_MEMBER
    try:
        with zipfile.ZipFile(path) as zf:
            return METADATA_MEMBER in zf.namelist()
    except zipfile.BadZipFile as exc:
        raise ValueError("invalid or corrupt .fsz ZIP archive") from exc


def _legacy_detect(path):
    """[(face, full BGR image)] for every face the detector finds in a legacy (PNG-only) archive.

    A legacy .fsz stores no embeddings; the loader re-detects its images (source_gallery._ingest_faceset)
    and so does this, which costs a detector pass per image - which is why it runs on first USE of an
    identity, never when the candidate list is built."""
    import tempfile
    from roop.face_util import extract_face_images
    folder = tempfile.mkdtemp(prefix="lab_legacy_")
    out = []
    try:
        util.unzip(path, folder)
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith(".png"):
                continue
            full = os.path.join(folder, fn)
            img = cv2.imdecode(np.fromfile(full, np.uint8), cv2.IMREAD_COLOR)
            for found in extract_face_images(full, (False, 0)):
                out.append((found[0], img))
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return out


def _face_vector(face):
    e = face.get("embedding") if isinstance(face, dict) else getattr(face, "embedding", None)
    return e


def _library_embedding(path):
    """The identity of a library .fsz as a unit vector.

    V2: the centroid of the embeddings cached in its metadata. Legacy: the mean of the embeddings the
    detector finds in its images (what V1 loading does), computed once per file version and cached."""
    from roop.faceset_v2 import read_faceset_archive
    from roop.identity_algebra import blend_embeddings
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    if key in _embedding_cache:
        return _embedding_cache[key]
    if _is_v2(path):
        md = read_faceset_archive(path)
        vectors = [v for v in ((md.get("index") or {}).get("normalized_embeddings") or []) if v]
        if not vectors:
            vectors = [((s.get("identity") or {}).get("normalized_embedding")) for s in md.get("sources", [])]
            vectors = [v for v in vectors if v]
    else:
        vectors = [v for v in (_face_vector(f) for f, _ in _legacy_detect(path)) if v is not None]
    if not vectors:
        raise ValueError(f"no face found in '{os.path.basename(path)}'")
    vec = blend_embeddings(vectors, [1.0] * len(vectors))
    _embedding_cache[key] = vec
    return vec


def _library_anchor(path):
    """(reference BGR image, reference face) a blend of this identity can be anchored to."""
    from roop.faceset_v2 import read_faceset_archive
    if _is_v2(path):
        img = None
        with zipfile.ZipFile(path) as zf:
            members = sorted(n for n in zf.namelist() if n.lower().endswith(".png") and "/" not in n)
            if members:
                img = cv2.imdecode(np.frombuffer(zf.read(members[0]), np.uint8), cv2.IMREAD_COLOR)
        src0 = ((read_faceset_archive(path) or {}).get("sources") or [{}])[0]
        bbox = (src0.get("geometry") or {}).get("bbox")
        face = ({"bbox": np.asarray(bbox, np.float32), "det_score": 1.0, "pose": [0.0, 0.0, 0.0]}
                if bbox is not None else None)
        return img, face
    found = _legacy_detect(path)
    if not found:
        return None, None
    face, img = max(found, key=lambda t: float((t[0].get("det_score") if isinstance(t[0], dict)
                                                 else getattr(t[0], "det_score", 0.0)) or 0.0))
    return img, face


def _candidates():
    """Every identity the lab can mix, with an ``available`` flag and the reason when it is not."""
    import ui.globals as ui_globals
    from routes_faceset import _library_entries
    from roop.identity_algebra import faceset_identity_vector
    out = []
    for i, fs in enumerate(roop_globals.INPUT_FACESETS):
        vec = faceset_identity_vector(fs)
        thumb = ui_globals.ui_input_thumbs[i] if i < len(ui_globals.ui_input_thumbs) else None
        out.append({"id": f"src:{i}", "kind": "source", "label": f"Source face {i + 1}",
                    "thumb": _rgb_to_dataurl(thumb) if thumb is not None else "",
                    "available": vec is not None, "reason": None if vec is not None else "no embedding"})
    for entry in _library_entries():
        try:
            legacy = not _is_v2(entry["path"])
            reason = None
        except ValueError as exc:
            legacy, reason = False, str(exc)
        # A legacy archive is usable: its identity is computed from its images the first time it is
        # blended (see _legacy_detect), not here, so listing a library of fifty never runs the detector.
        out.append({"id": f"lib:{entry['filename']}", "kind": "library", "label": entry["name"],
                    "thumb": entry.get("thumb", ""), "faces": entry.get("faces"), "legacy": legacy,
                    "available": reason is None, "reason": reason})
    if fm.fm_faces:
        ready = fm.fm_mean_embedding is not None
        out.append({"id": "fm:set", "kind": "facemgr", "label": "Face Manager set",
                    "thumb": _rgb_to_dataurl(fm.fm_thumbs[0]) if fm.fm_thumbs else "",
                    "faces": len(fm.fm_faces), "available": ready,
                    "reason": None if ready else "no embedding"})
    return out


def _library_path(ref):
    from routes_faceset import _faceset_library_dir
    fn = os.path.basename(ref)
    path = os.path.join(_faceset_library_dir(), fn)
    if not (fn.lower().endswith(".fsz") and os.path.exists(path)):
        raise ValueError(f"saved identity '{fn}' not found")
    return fn, path


def _identity(cid):
    """(unit embedding, label) for a candidate id. Cheap after the first use of a legacy archive; touches
    no images, so the live blend readout (one request per slider pause) never re-runs a detector."""
    from roop.identity_algebra import faceset_identity_vector
    kind, _, ref = str(cid).partition(":")
    if kind == "src":
        idx = int(ref)
        if not 0 <= idx < len(roop_globals.INPUT_FACESETS):
            raise ValueError(f"no source face {ref}")
        return faceset_identity_vector(roop_globals.INPUT_FACESETS[idx]), f"Source face {idx + 1}"
    if kind == "lib":
        fn, path = _library_path(ref)
        return _library_embedding(path), os.path.splitext(fn)[0]
    if kind == "fm" and fm.fm_faces:
        mean = fm.fm_mean_embedding
        return (np.asarray(mean, np.float32) if mean is not None else None), "Face Manager set"
    raise ValueError(f"unknown identity '{cid}'")


def _anchor(cid):
    """(reference BGR image, reference face) to anchor a SAVED blend to. Only Save / Preview need it,
    and only for the identity with the largest share."""
    from routes_faceset import _faceset_member_images
    kind, _, ref = str(cid).partition(":")
    if kind == "src":
        idx = int(ref)
        fs = roop_globals.INPUT_FACESETS[idx]
        imgs = _faceset_member_images(idx)
        return (imgs[0] if imgs else None), (fs.faces[0] if getattr(fs, "faces", None) else None)
    if kind == "lib":
        return _library_anchor(_library_path(ref)[1])
    if kind == "fm" and fm.fm_faces:
        best = int(np.argmax(fm.fm_scores)) if fm.fm_scores else 0
        return (fm.fm_images[best] if best < len(fm.fm_images) else None), fm.fm_faces[best]
    raise ValueError(f"unknown identity '{cid}'")


def _components(payload):
    comps = payload.get("components") or []
    if not 2 <= len(comps) <= MAX_BLEND_SOURCES:
        raise ValueError(f"pick 2 to {MAX_BLEND_SOURCES} identities")
    ids = [str(c.get("id", "")) for c in comps]
    if len(set(ids)) != len(ids):
        raise ValueError("each identity can be used once")
    try:
        weights = [float(c.get("weight", 1.0)) for c in comps]
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must be numbers") from exc
    return ids, [_identity(i) for i in ids], weights


def _options(payload):
    """(mode, deid) from a morph request: the blend mode and the optional de-identification
    ``{"references": [candidate ids], "max_cosine", "min_primary_cosine"}`` with the references
    resolved to unit vectors. ``deid`` is None unless ``deidentify.enabled`` and it names references."""
    mode = str(payload.get("mode") or DEFAULT_BLEND_MODE).strip().lower()
    if mode not in BLEND_MODES:
        raise ValueError(f"mode must be one of {', '.join(BLEND_MODES)}")
    raw = payload.get("deidentify")
    if not isinstance(raw, dict) or not raw.get("enabled"):
        return mode, None
    refs = [str(r) for r in (raw.get("references") or [])]
    if not refs:
        return mode, None
    if len(refs) > MAX_BLEND_SOURCES or len(set(refs)) != len(refs):
        raise ValueError(f"name 1 to {MAX_BLEND_SOURCES} different de-identification references")
    try:
        max_cosine = float(raw.get("max_cosine", DEFAULT_PRIVACY_COSINE))
        floor = float(raw.get("min_primary_cosine", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("de-identification thresholds must be numbers") from exc
    vectors = []
    for ref in refs:
        vec = _identity(ref)[0]
        if vec is None:
            raise ValueError(f"de-identification reference '{ref}' has no usable embedding")
        vectors.append(vec)
    return mode, {"references": vectors, "max_cosine": max_cosine, "min_primary_cosine": floor}


def _summarise(payload):
    """(ids, resolved, summary) for a morph request, validated."""
    ids, resolved, weights = _components(payload)
    mode, deid = _options(payload)
    return ids, resolved, lab.morph_summary([r[0] for r in resolved], weights, mode=mode, deid=deid)


@router.get("/api/facemgr/morph/candidates")
def morph_candidates():
    return {"candidates": _candidates()}


@router.post("/api/facemgr/morph/blend")
def morph_blend(payload: dict = Body(...)):
    """Normalised weights, the blend's cosine to each input, and a warning if it is far from all."""
    try:
        ids, resolved, summary = _summarise(payload)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"message": str(exc)})
    summary.pop("embedding")
    summary["components"] = [{"id": i, "label": r[1]} for i, r in zip(ids, resolved)]
    return summary


def write_blended_fsz(path, embedding, base_image, base_face, name):
    """Write a V2 .fsz whose cached identity is ``embedding``, keeping ``base_face``'s geometry."""
    from roop.FaceSet import FaceSet
    from roop.faceset_v2 import write_faceset_v2
    if base_image is None:
        raise ValueError("the dominant identity has no reference image to anchor the blend to")
    h, w = base_image.shape[:2]
    if base_face is None:
        face = {"bbox": np.array([0, 0, w, h], np.float32), "det_score": 1.0, "pose": [0.0, 0.0, 0.0]}
    else:
        # insightface's Face is a dict subclass that copy.copy cannot copy (its attribute fallback answers
        # None for the copy protocol); faceset_v2 rebuilds it as type(face)(face), and so does this.
        try:
            face = type(base_face)(base_face)
        except Exception as exc:
            _swallowed("routes_identity_lab:morph_faceset", exc, "face copy fallback to dict")
            face = dict(base_face)
    vec = np.asarray(embedding, np.float32)
    # Item assignment, never setattr: insightface's Face derives `normed_embedding` from `embedding` as a
    # read-only property (so the unit vector below is its normed form too), and a plain dict stands in for
    # a Face in tests, where both keys are just keys.
    face["embedding"] = vec
    if type(face) is dict:
        face["normed_embedding"] = vec
    fs = FaceSet()
    fs.faces = [face]
    fs.ref_images = [base_image]
    # One entry, so there is no median to reject it against and no reason to gate it on photo quality:
    # the blend is a vector, the photo is only its anchor.
    write_faceset_v2(path, fs, [base_image], source_name=name, min_quality=0.0, max_entries=1,
                     laplacian_floor=0.0, min_identity_cosine=0.0)
    return path


def save_blended_to_library(name, embedding, image, face, extra):
    """Write ``embedding`` into the faceset library as a new V2 .fsz (plus a thumbnail) and return the library
    payload with ``saved`` and ``extra``. Raises ValueError when there is no anchor image. Shared by the Morph Lab
    (``/api/facemgr/morph/save``) and the Face Swap blender's export (``/api/identity/export_fsz``)."""
    from routes_faceset import _faceset_library_payload, _imwrite_unicode, _unique_fsz_path
    from source_gallery import _shrink_for_thumb
    path = _unique_fsz_path(name)
    write_blended_fsz(path, embedding, image, face, name)
    if image is not None:
        _imwrite_unicode(os.path.splitext(path)[0] + ".png", _shrink_for_thumb(image))
    return _faceset_library_payload({"saved": os.path.splitext(os.path.basename(path))[0], **extra})


def _prepared(payload):
    """(summary, anchor image, anchor face): the blend and what to anchor it to (the dominant identity's)."""
    ids, resolved, summary = _summarise(payload)
    image, face = _anchor(ids[summary["dominant"]])
    return summary, image, face


@router.post("/api/facemgr/morph/save")
def morph_save(payload: dict = Body(...)):
    """Save the blend to the faceset library as a new .fsz."""
    try:
        summary, image, face = _prepared(payload)
        name = str(payload.get("name") or "Blended identity").strip() or "Blended identity"
        return save_blended_to_library(name, summary["embedding"], image, face,
                                       {"weights": summary["weights"], "mode": summary["mode"],
                                        "deidentify": summary["deidentify"],
                                        "cosine_to_each": summary["cosine_to_each"]})
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"message": str(exc)})
    except Exception:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": "failed to write the blended faceset"})


@router.post("/api/facemgr/morph/preview")
def morph_preview(payload: dict = Body(...)):
    """Load the blend as a Face Swap source (not added to the library) and select it, so the Face Swap
    preview on the active frame renders it."""
    from routes_faceset import _faceset_library_dir
    from source_gallery import _ingest_faceset, _source_faces_payload
    try:
        summary, image, face = _prepared(payload)
        folder = os.path.join(_faceset_library_dir(), ".morph_preview")   # hidden: not a library entry
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "blend_%d.fsz" % int(time.time() * 1000))
        write_blended_fsz(path, summary["embedding"], image, face, "morph preview")
        _ingest_faceset(path)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"message": str(exc)})
    except Exception:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": "failed to load the blend as a source"})
    index = len(roop_globals.INPUT_FACESETS) - 1
    state.selected_input_face_index = index
    out = _source_faces_payload()
    out["index"] = index
    out["weights"] = summary["weights"]
    out["mode"] = summary["mode"]
    out["deidentify"] = summary["deidentify"]
    return out
