"""Head pose (perspective solvePnP) and per-frame quality for Stage-1 candidates.

Takes the tracklet detections ``roop/scanner.py`` harvests and turns each into
pose + photometric telemetry, so a caller can keep the frames worth capturing
and drop blurred, dark, clipped or too-small ones.

Pose
----
``estimate_head_pose`` solves a full perspective pose with ``cv2.solvePnP``
(EPnP, then one ``solvePnPRefineLM`` pass; ``ROOP_POSE_Q_REFINE=0`` disables
it) against the project's reference head: the same five points
``face_util._reference_5pt`` derives from ``face_3d_recon._REF3D_68``, so there
is one head shape in the codebase, not two. The rotation is turned into angles
by ``face_util._decompose_projection``, the single place this project converts a
fitted rotation into yaw/pitch/roll: two pose sources that disagreed about which
way a head faces is a bug this project has already had (face_3d_recon's
180-degree offset; face_frontalize's mirrored rvec=0 reference). The tests pin
this solver to ``face_util.solve_pose_5pt`` on synthetic heads.

Five keypoints carry six degrees of freedom (the jaw moves the mouth corners),
so an open mouth reads partly as pitch here, exactly as it does in the
weak-perspective solve. ``face_util.solve_pose_jaw_5pt`` exists for callers who
need that separated.

Quality
-------
The size gate reads a FRONTAL-EQUIVALENT inter-ocular distance: the pixel eye
distance this face would have at its solved depth if it faced the camera. Raw
eye distance collapses toward zero as a head turns, so a raw "IOD < 45 px"
gate rejects every profile at any resolution -- the hard poses a multi-angle
bank exists to collect (see ``face_quality.image_quality``). Raw IOD is still
reported.

Sharpness is Laplacian variance on a size-normalised crop, as in
``face_quality``. It measures edge energy, not detail (a sharpened or noisy
crop scores high), and its absolute level slides with grain and compression, so
the batch evaluator also applies ``face_quality.blur_outlier``: a frame is
refused for blur only relative to the median of the candidates it came with.

Eye openness
------------
Before any sharpness or illumination work, the batch evaluator measures the Eye
Aspect Ratio on the 68-point landmarks (Soukupova & Cech; the same formula as
``eyelid_preserver.calculate_ear``) and discards the frame when EAR < 0.20, so a
closed-eye frame never becomes a reference. Both eyes are measured and the gate
reads the MORE OPEN one: a blink closes both, while the far eye of a turned head
is foreshortened and its landmarks are guessed, so gating on either eye alone
would discard profiles as "closed". A frame whose 68 points cannot be obtained
is NOT treated as open: ``ear`` stays None and the batch logs how many.
Uncorrected for pose: looking down shortens the lid opening in the picture, so
a strongly pitched-down open eye can read below 0.20 (unmeasured on footage).
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field

from roop.degrade import swallowed as _swallowed
from roop.face_util import _decompose_projection, _reference_5pt
from roop.scanner import open_capture

logger = logging.getLogger("roop.pose_quality")


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


# ── Tunables (env-overridable; none of these were fitted to footage) ─────────
MIN_IOD_PX = _envf("ROOP_POSE_Q_MIN_IOD", 45.0)
IOD_FULL_PX = _envf("ROOP_POSE_Q_IOD_FULL", 112.0)       # norm_iod saturates here
LUMA_LO = _envf("ROOP_POSE_Q_LUMA_LO", 40.0)
LUMA_HI = _envf("ROOP_POSE_Q_LUMA_HI", 215.0)
CLIP_LO = 5          # Y <= this counts as crushed shadow
CLIP_HI = 250        # Y >= this counts as blown highlight
MAX_CLIPPED = _envf("ROOP_POSE_Q_MAX_CLIPPED", 0.25)     # fraction of the crop
SHARP_FULL = _envf("ROOP_POSE_Q_SHARP_FULL", 350.0)      # same scale as face_quality
SHARP_SIDE = 160     # crop is resized to this before the Laplacian
EAR_MIN = _envf("ROOP_POSE_Q_EAR_MIN", 0.20)             # below: eyes closed
_REFINE = os.environ.get("ROOP_POSE_Q_REFINE", "1").strip().lower() not in ("0", "false", "off", "no")

W_SHARP, W_IOD, W_ILLUM, W_ID, W_ROLL = 0.35, 0.20, 0.15, 0.30, 0.01

# ── Reference head in OpenCV camera axes ─────────────────────────────────────
# _REF3D_68 is y-up with the nose standing out toward +z (toward the viewer).
# OpenCV's camera frame is x right, y DOWN, z forward (away from the viewer), so
# a face looking at the camera is the reference turned 180 deg about x. With
# this model a frontal face solves to R = I.
_F = np.diag([1.0, -1.0, -1.0])
_REF5 = np.asarray(_reference_5pt(), dtype=np.float64)
MODEL_POINTS_5 = ((_REF5 - _REF5.mean(axis=0)) @ _F).astype(np.float64)
_REF_IOD = float(np.linalg.norm(_REF5[1] - _REF5[0]))
# A = diag(1,1,-1) @ R[:2].T maps an OpenCV rotation into the 3x2 projection
# _decompose_projection reads (its reference is the y-flipped, z-toward-viewer
# head). Derivation: image rows are X @ F @ R[:2].T; its basis is X @ D with
# D = diag(1,-1,1), and D @ F = diag(1,1,-1).
_TO_DECOMPOSE = np.diag([1.0, 1.0, -1.0])


def _frame_hw(frame_shape: Sequence[int]) -> Tuple[int, int]:
    """(height, width) from a numpy frame shape (h, w[, c])."""
    return int(frame_shape[0]), int(frame_shape[1])


def camera_matrix(frame_shape: Sequence[int]) -> np.ndarray:
    h, w = _frame_hw(frame_shape)
    f = float(max(w, h))
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _solve(landmarks_2d: Any, frame_shape: Sequence[int]):
    """(yaw, pitch, roll, frontal_iod_px) or None."""
    pts = np.asarray(landmarks_2d, dtype=np.float64).reshape(-1, 2)
    if pts.shape != (5, 2) or not np.isfinite(pts).all():
        return None
    if float(np.ptp(pts, axis=0).max()) < 1e-3:
        return None
    K = camera_matrix(frame_shape)
    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS_5, pts, K, None, flags=cv2.SOLVEPNP_EPNP)
    if not ok or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
        return None
    # EPnP on five points has a heavy tail under landmark noise. One LM
    # refinement from its answer, measured on 400 random poses at ~64 px IOD
    # with 1 px noise: worst error 11.2 -> 6.4 deg, p95 3.9 -> 3.4, for ~0.1 ms.
    if _REFINE:
        rvec, tvec = cv2.solvePnPRefineLM(MODEL_POINTS_5, pts, K, None, rvec, tvec)
        if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            return None
    R, _ = cv2.Rodrigues(rvec)
    out = _decompose_projection(_TO_DECOMPOSE @ R[:2].T)
    if out is None:
        return None
    yaw, pitch, roll, _scale = out
    tz = float(tvec[2, 0])
    # EPnP can return the mirror solution behind the camera; a head there is
    # not a head in the picture.
    if not tz > 1e-6:
        return None
    frontal_iod = float(K[0, 0]) * _REF_IOD / tz
    return float(yaw), float(pitch), float(roll), frontal_iod


def estimate_head_pose(landmarks_2d: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees from InsightFace's 5 keypoints (left eye,
    right eye, nose tip, left mouth, right mouth) in full-frame pixels.

    ``frame_shape`` is numpy order, (height, width[, channels]). Angles follow
    ``face_util.solve_pose_5pt``'s signs. Returns (nan, nan, nan) when the
    points are degenerate or the solve fails.
    """
    try:
        out = _solve(landmarks_2d, frame_shape)
    except cv2.error as exc:
        _swallowed("roop/pose_quality.py:estimate_head_pose", exc, "pose read as unknown")
        out = None
    if out is None:
        return (math.nan, math.nan, math.nan)
    return out[0], out[1], out[2]


# ── Quality ──────────────────────────────────────────────────────────────────
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _weak_perspective_iod_and_roll(kps: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Frontal-equivalent IOD and roll from the keypoints alone (no frame
    geometry needed): the fitted weak-perspective scale times the reference
    head's eye distance."""
    from roop.face_util import _REF5_PINV
    x = np.asarray(kps, dtype=np.float64).reshape(5, 2)
    x = x - x.mean(axis=0)
    out = _decompose_projection(_REF5_PINV @ x)
    if out is None:
        return None, None
    return float(out[3]) * _REF_IOD, float(out[2])


def sharpness(face_crop: np.ndarray) -> float:
    gray = face_crop if face_crop.ndim == 2 else cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (SHARP_SIDE, SHARP_SIDE), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def illumination(face_crop: np.ndarray) -> Dict[str, float]:
    if face_crop.ndim == 2:
        y = face_crop
    else:
        y = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)[..., 0]
    n = float(y.size)
    mean = float(y.mean())
    lo = float(np.count_nonzero(y <= CLIP_LO)) / n
    hi = float(np.count_nonzero(y >= CLIP_HI)) / n
    # 1 inside [LUMA_LO, LUMA_HI], linear to 0 at black/white.
    if mean < LUMA_LO:
        band = mean / LUMA_LO
    elif mean > LUMA_HI:
        band = (255.0 - mean) / (255.0 - LUMA_HI)
    else:
        band = 1.0
    norm = _clamp01(band) * _clamp01(1.0 - (lo + hi) / MAX_CLIPPED) if MAX_CLIPPED > 0 else _clamp01(band)
    return {"luma_mean": mean, "shadow_clipped": lo, "highlight_clipped": hi, "norm_illum": norm}


def compute_quality_score(face_crop: np.ndarray, kps: np.ndarray, det_score: float,
                          id_similarity: Optional[float], *, roll: Optional[float] = None,
                          frontal_iod: Optional[float] = None) -> Tuple[float, Dict[str, float]]:
    """(Q in [0, 1], breakdown) for one face crop (BGR) and its 5 keypoints.

    Q = 0.35 sharp + 0.20 iod + 0.15 illum + 0.30 id - 0.01 |roll|, clamped.
    ``id_similarity`` of None drops the identity term and renormalises the rest
    (a missing embedding is not evidence of the wrong person). ``roll`` and
    ``frontal_iod`` default to the weak-perspective solve of ``kps``; the batch
    evaluator passes its perspective solve instead. ``det_score`` is reported,
    not weighted (the brief's formula has no term for it).
    """
    kps = np.asarray(kps, dtype=np.float64).reshape(5, 2)
    if roll is None or frontal_iod is None:
        wp_iod, wp_roll = _weak_perspective_iod_and_roll(kps)
        frontal_iod = wp_iod if frontal_iod is None else frontal_iod
        roll = wp_roll if roll is None else roll

    raw_iod = float(np.linalg.norm(kps[1] - kps[0]))
    sharp = sharpness(face_crop)
    illum = illumination(face_crop)
    iod_for_gate = frontal_iod if frontal_iod is not None else raw_iod
    norm_sharp = _clamp01(sharp / SHARP_FULL)
    norm_iod = _clamp01((iod_for_gate - MIN_IOD_PX) / max(1e-6, IOD_FULL_PX - MIN_IOD_PX))

    terms = W_SHARP * norm_sharp + W_IOD * norm_iod + W_ILLUM * illum["norm_illum"]
    weight = W_SHARP + W_IOD + W_ILLUM
    norm_id = None
    if id_similarity is not None and math.isfinite(float(id_similarity)):
        norm_id = _clamp01(float(id_similarity))
        terms += W_ID * norm_id
        weight += W_ID
    # Rescale to the full formula's range so Q is comparable with or without the id term.
    q = terms * (W_SHARP + W_IOD + W_ILLUM + W_ID) / weight
    roll_abs = abs(float(roll)) if roll is not None and math.isfinite(float(roll)) else 0.0
    q = _clamp01(q - W_ROLL * roll_abs)

    breakdown = {
        "sharpness": sharp,
        "norm_sharpness": norm_sharp,
        "iod_px": raw_iod,
        "frontal_iod_px": float(iod_for_gate),
        "norm_iod": norm_iod,
        "luma_mean": illum["luma_mean"],
        "shadow_clipped": illum["shadow_clipped"],
        "highlight_clipped": illum["highlight_clipped"],
        "norm_illum": illum["norm_illum"],
        "id_similarity": float("nan") if norm_id is None else norm_id,
        "roll_abs": roll_abs,
        "det_score": float(det_score),
    }
    return q, breakdown


def reject_reasons(breakdown: Dict[str, float]) -> List[str]:
    """Hard gates, independent of the composite score."""
    reasons = []
    if breakdown["frontal_iod_px"] < MIN_IOD_PX:
        reasons.append("too_small")
    if not (LUMA_LO <= breakdown["luma_mean"] <= LUMA_HI):
        reasons.append("too_dark" if breakdown["luma_mean"] < LUMA_LO else "too_bright")
    if breakdown["shadow_clipped"] + breakdown["highlight_clipped"] > MAX_CLIPPED:
        reasons.append("clipped")
    return reasons


# ── Eye openness ─────────────────────────────────────────────────────────────
def _ear(eye: np.ndarray) -> Optional[float]:
    """(|p1-p5| + |p2-p4|) / (2 |p0-p3|) over one eye's six points, or None."""
    width = float(np.linalg.norm(eye[0] - eye[3]))
    if not width > 1e-6:
        return None
    return (float(np.linalg.norm(eye[1] - eye[5])) + float(np.linalg.norm(eye[2] - eye[4]))) / (2.0 * width)


def eye_aspect_ratios(landmarks_68: Any) -> Optional[Tuple[float, float]]:
    """(EAR of points 36-41, EAR of points 42-47) from 68-point landmarks
    (x, y[, z]; z ignored), or None when they are missing or degenerate."""
    if landmarks_68 is None:
        return None
    try:
        pts = np.asarray(landmarks_68, dtype=np.float64)
        pts = pts.reshape(pts.shape[0], -1)[:, :2]
    except (TypeError, ValueError, IndexError):
        return None
    if pts.shape[0] < 68 or not np.isfinite(pts[36:48]).all():
        return None
    first, second = _ear(pts[36:42]), _ear(pts[42:48])
    if first is None or second is None:
        return None
    return first, second


def eyes_open(landmarks_68: Any, threshold: Optional[float] = None) -> Optional[bool]:
    """True / False on the more open eye, None when it cannot be measured."""
    ears = eye_aspect_ratios(landmarks_68)
    if ears is None:
        return None
    return max(ears) >= (EAR_MIN if threshold is None else float(threshold))


def app_landmarks_68(frame: np.ndarray, bbox: Sequence[float], kps: Any = None) -> Optional[np.ndarray]:
    """68 points from the app's own ``landmark_3d_68`` model (loaded with the
    face analyser; run on demand when it is held lazily), or None."""
    from insightface.app.common import Face
    from roop.face_util import lease_face_analyser
    with lease_face_analyser() as fa:
        model = getattr(fa, "lm68_model", None) or getattr(fa, "models", {}).get("landmark_3d_68")
        if model is None:
            return None
        face = Face(bbox=np.asarray(bbox, dtype=np.float32),
                    kps=None if kps is None else np.asarray(kps, dtype=np.float32))
        pred = model.get(frame, face)
    return None if pred is None else np.asarray(pred, dtype=np.float64)[:, :2]


# ── Batch API ────────────────────────────────────────────────────────────────
class CandidateFaceMetric(BaseModel):
    frame_idx: int
    track_id: Optional[int] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    composite_score: float = 0.0
    sharpness: float = 0.0
    iod_px: float = 0.0
    frontal_iod_px: float = 0.0
    luma_mean: float = 0.0
    clipped_fraction: float = 0.0
    det_score: float = 0.0
    id_similarity: Optional[float] = None
    # The detection's geometry, carried so later stages (angle_portfolio) can
    # crop / embed the chosen frame without re-joining to the scanner output.
    bbox: Optional[List[float]] = None
    kps: Optional[List[List[float]]] = None
    ear: Optional[float] = None            # the more open eye; None = unmeasured
    ear_left: Optional[float] = None       # points 36-41
    ear_right: Optional[float] = None      # points 42-47
    is_valid: bool = False
    reject_reasons: List[str] = Field(default_factory=list)


def crop_face(frame: np.ndarray, bbox: Sequence[float], pad: float = 0.15) -> Optional[np.ndarray]:
    x0, y0, x1, y1 = (float(v) for v in bbox)
    w, h = x1 - x0, y1 - y0
    if w <= 1 or h <= 1:
        return None
    H, W = frame.shape[:2]
    ax0, ay0 = max(0, int(x0 - pad * w)), max(0, int(y0 - pad * h))
    ax1, ay1 = min(W, int(math.ceil(x1 + pad * w))), min(H, int(math.ceil(y1 + pad * h)))
    if ax1 - ax0 < 2 or ay1 - ay0 < 2:
        return None
    return frame[ay0:ay1, ax0:ax1].copy()


def _geometry(cand: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, shape in (("bbox", (4,)), ("kps", (5, 2))):
        value = cand.get(key)
        if value is None:
            continue
        try:
            out[key] = np.asarray(value, dtype=np.float64).reshape(shape).tolist()
        except (TypeError, ValueError):
            pass
    return out


def _base(cand: Dict[str, Any]) -> Dict[str, Any]:
    return {"frame_idx": int(cand["frame_idx"]), "track_id": cand.get("track_id"),
            "det_score": float(cand.get("det_score") or 0.0),
            "id_similarity": cand.get("similarity"), **_geometry(cand)}


def _metric_for(frame: np.ndarray, cand: Dict[str, Any],
                landmarks_fn: Optional[Callable] = None) -> CandidateFaceMetric:
    base = _base(cand)
    kps = cand.get("kps")
    bbox = cand.get("bbox")
    if kps is None or bbox is None:
        return CandidateFaceMetric(**base, reject_reasons=["no_landmarks"])
    crop = crop_face(frame, bbox)
    if crop is None:
        return CandidateFaceMetric(**base, reject_reasons=["bad_bbox"])
    try:
        solved = _solve(kps, frame.shape)
    except cv2.error as exc:
        _swallowed("roop/pose_quality.py:_metric_for", exc, "pose read as unknown")
        solved = None
    yaw = pitch = roll = frontal = None
    if solved is not None:
        yaw, pitch, roll, frontal = solved

    # Eye openness FIRST: a closed-eye frame is discarded before any sharpness
    # or illumination work is spent on it.
    lm68 = cand.get("landmarks_68")
    if lm68 is None and landmarks_fn is not None:
        try:
            lm68 = landmarks_fn(frame, bbox, kps)
        except Exception as exc:  # one failed landmark call must not end the batch
            _swallowed("roop/pose_quality.py:landmarks_fn", exc, "EAR read as unmeasured")
            lm68 = None
    ears = eye_aspect_ratios(lm68)
    if ears is not None:
        base.update(ear_left=round(ears[0], 4), ear_right=round(ears[1], 4), ear=round(max(ears), 4))
        if max(ears) < EAR_MIN:
            return CandidateFaceMetric(**base, yaw=yaw, pitch=pitch, roll=roll,
                                       reject_reasons=["eyes_closed"])

    q, bd = compute_quality_score(crop, np.asarray(kps), base["det_score"], base["id_similarity"],
                                  roll=roll, frontal_iod=frontal)
    reasons = reject_reasons(bd)
    if solved is None:
        reasons.append("pose_unsolved")
    return CandidateFaceMetric(
        **base, yaw=yaw, pitch=pitch, roll=roll, composite_score=round(q, 4),
        sharpness=round(bd["sharpness"], 2), iod_px=round(bd["iod_px"], 2),
        frontal_iod_px=round(bd["frontal_iod_px"], 2), luma_mean=round(bd["luma_mean"], 2),
        clipped_fraction=round(bd["shadow_clipped"] + bd["highlight_clipped"], 4),
        is_valid=not reasons, reject_reasons=reasons)


def evaluate_candidate_frames(media_path: str, candidates: List[Dict],
                              landmarks_fn: Optional[Callable] = app_landmarks_68) -> List[CandidateFaceMetric]:
    """Pose + eye openness + quality for Stage-1 detections (dicts with
    frame_idx, bbox, kps, det_score, similarity; track_id and landmarks_68
    optional). Output order matches input.

    ``landmarks_fn(frame, bbox, kps) -> (68, 2) | None`` supplies the eye points
    for candidates without ``landmarks_68``; the default runs the app's
    landmark_3d_68 model. Pass None to skip it (EAR is then unmeasured).

    Decodes the video once, sequentially (no seeks), up to the last requested
    frame; each frame is released as soon as its candidates are scored. A frame
    that cannot be decoded marks its candidates ``unreadable_frame``.
    """
    if not candidates:
        return []
    by_frame: Dict[int, List[int]] = {}
    for i, c in enumerate(candidates):
        by_frame.setdefault(int(c["frame_idx"]), []).append(i)
    results: List[Optional[CandidateFaceMetric]] = [None] * len(candidates)
    last = max(by_frame)

    with open_capture(media_path) as capture:
        idx = -1
        while idx < last:
            idx += 1
            if not capture.grab():
                break
            wanted = by_frame.get(idx)
            if not wanted:
                continue
            ok, frame = capture.retrieve()
            if not ok or frame is None:
                continue
            for i in wanted:
                results[i] = _metric_for(frame, dict(candidates[i]), landmarks_fn)
            del frame

    # Blur is judged against what the clip offered, not a fixed line.
    from roop.face_quality import blur_outlier
    samples = [m.sharpness for m in results if m is not None and m.sharpness > 0]
    for m in results:
        if m is not None and m.sharpness > 0 and blur_outlier(m.sharpness, samples):
            m.reject_reasons.append("blurred")
            m.is_valid = False

    scored = [m for m in results if m is not None]
    unmeasured = sum(1 for m in scored if m.ear is None)
    if unmeasured:
        logger.warning("pose_quality: eye openness unmeasured on %d of %d candidate(s) - "
                       "no 68-point landmarks; closed-eye frames were NOT filtered there",
                       unmeasured, len(scored))

    for i, m in enumerate(results):
        if m is None:
            results[i] = CandidateFaceMetric(**_base(candidates[i]), reject_reasons=["unreadable_frame"])
    return results  # type: ignore[return-value]


__all__ = ["CandidateFaceMetric", "estimate_head_pose", "compute_quality_score",
           "evaluate_candidate_frames", "camera_matrix", "crop_face", "MODEL_POINTS_5",
           "eye_aspect_ratios", "eyes_open", "app_landmarks_68", "EAR_MIN"]
