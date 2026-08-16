"""Output quality analysis endpoint.

Split out of api.py; handler bodies are unchanged, only the decorator moved
from @app to @router. Registered via app.include_router() in api.py, which is
safe here because every /api route is a literal path with no path parameters,
so declaration order cannot change which handler matches.
"""

from fastapi import APIRouter, Body
import os

import cv2
from fastapi.responses import JSONResponse

import roop.globals as roop_globals
from roop import utilities as util
from roop.face_util import get_all_faces
from roop.capturer import (get_image_frame, get_video_frame,
                           get_video_frame_total)


router = APIRouter()

# ── Injected by api.py at import time ────────────────────────────────────
# Verified to be mutated in place and never rebound in api.py, so binding the
# same object here keeps one shared value rather than two that drift.
_last_output = None


# ── Quality report card ──────────────────────────────────────────────────────
def _clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, v))

def _analyze_output_frame(img, src_embs):
    """Detect the dominant face in a rendered frame and return its embedding,
    best identity similarity vs the loaded source(s), and a sharpness score."""
    faces = get_all_faces(img) or []
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = getattr(face, 'embedding', None)
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    x1, y1 = max(0, x1), max(0, y1)
    crop = img[y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)]
    sharp = 0.0
    try:
        if crop.size:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        pass
    sim = None
    if emb is not None and src_embs:
        try:
            sim = max(1.0 - util.compute_cosine_distance(emb, se) for se in src_embs)
        except Exception:
            sim = None
    return {"emb": emb, "sharp": sharp, "sim": sim}

@router.post("/api/quality/analyze")
def quality_analyze(payload: dict = Body(...)):
    """Post-run quality report: identity likeness vs the source, sharpness,
    temporal stability (video), and detection coverage. Samples up to 24 frames
    from the output and re-detects faces to measure the actual result."""
    path = payload.get("path") or _last_output.get("path")
    if not path or not os.path.exists(path):
        return JSONResponse(status_code=404, content={"message": "No output available to analyze"})

    src_embs = []
    for fs in roop_globals.INPUT_FACESETS:
        for f in fs.faces:
            e = getattr(f, 'embedding', None)
            if e is not None:
                src_embs.append(e)

    is_vid = util.is_video(path) or path.lower().endswith(("gif", "webp"))
    sims, sharps, embs_seq = [], [], []
    sampled = 0
    detected = 0

    try:
        if is_vid:
            total = int(get_video_frame_total(path) or 0)
            n = min(24, max(1, total))
            idxs = [int(i * (total - 1) / (n - 1)) + 1 for i in range(n)] if total > 1 else [1]
            for fi in idxs:
                frame = get_video_frame(path, fi)
                if frame is None:
                    continue
                sampled += 1
                r = _analyze_output_frame(frame, src_embs)
                if r is None:
                    continue
                detected += 1
                if r["sim"] is not None:
                    sims.append(r["sim"])
                sharps.append(r["sharp"])
                if r["emb"] is not None:
                    embs_seq.append(r["emb"])
        else:
            img = get_image_frame(path)
            if img is not None:
                sampled = 1
                r = _analyze_output_frame(img, src_embs)
                if r is not None:
                    detected = 1
                    if r["sim"] is not None:
                        sims.append(r["sim"])
                    sharps.append(r["sharp"])
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Analysis failed: {e}"})

    if detected == 0:
        return {"ok": False, "message": "No face detected in the output", "is_video": is_vid,
                "sampled": sampled, "detection_rate": 0}

    metrics = {}
    identity_sim = (sum(sims) / len(sims)) if sims else None
    if identity_sim is not None:
        metrics["identity_similarity"] = round(float(identity_sim), 3)
        metrics["identity_score"] = round(_clamp((identity_sim - 0.15) / (0.55 - 0.15) * 100), 1)

    mean_sharp = (sum(sharps) / len(sharps)) if sharps else 0.0
    metrics["sharpness_raw"] = round(mean_sharp, 1)
    metrics["sharpness_score"] = round(_clamp(mean_sharp / 800.0 * 100), 1)

    detection_rate = detected / max(1, sampled)
    metrics["detection_rate"] = round(detection_rate * 100, 1)

    if is_vid and len(embs_seq) >= 2:
        dists = [util.compute_cosine_distance(embs_seq[i], embs_seq[i + 1]) for i in range(len(embs_seq) - 1)]
        stability = 1.0 - min(1.0, sum(dists) / len(dists))
        metrics["temporal_stability"] = round(_clamp(stability * 100), 1)

    # Weighted overall score across whatever metrics we have.
    weights, total_w, acc = {}, 0.0, 0.0
    if "identity_score" in metrics:
        weights["identity_score"] = 0.45 if is_vid else 0.7
    if "temporal_stability" in metrics:
        weights["temporal_stability"] = 0.3
    weights["sharpness_score"] = 0.15 if is_vid else 0.3
    if is_vid:
        weights["detection_rate"] = 0.1
    for k, w in weights.items():
        acc += metrics[k] * w
        total_w += w
    overall = acc / total_w if total_w else 0
    metrics["overall_score"] = round(overall, 1)
    metrics["grade"] = ("A" if overall >= 85 else "B" if overall >= 70 else
                        "C" if overall >= 55 else "D" if overall >= 40 else "E")

    return {"ok": True, "is_video": is_vid, "sampled": sampled,
            "has_source": bool(src_embs), "metrics": metrics}
