"""Face Image Quality Assessment (FIQA) for faceset building.

A faceset is only as good as the crops that go into it: a blurry, tiny, or
sharply-profile reference drags the blended identity toward mush. This scores
each detected face 0..1 so the Face Manager can rank and gate crops before they
are baked into a .fsz.

Rather than ship a separate FIQA network (CR-FIQA / MagFace and friends need
their own model download, aligned-input preprocessing and provider wiring), this
composes the signals we already have for free from the detector + recogniser
that produced the Face — the same axes those papers optimise for:

  - detector confidence   (det_score)          — is it even a clean face?
  - sharpness             (Laplacian variance)  — blur / focus / compression
  - resolution            (bbox size)           — enough pixels to matter
  - frontality            (pose or kps symmetry) — extreme profiles blend poorly
  - embedding magnitude   (‖ArcFace emb‖)        — MagFace's insight: norm tracks
                                                   quality even on a non-MagFace net

Weights are env-overridable (ROOP_FIQA_W_*) and any missing component (e.g. no
embedding when recognition is disabled) drops out and the rest renormalise, so
the score stays meaningful in every module configuration.
"""

import os
import statistics

import cv2
import numpy as np


def _envf(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


W_DET = _envf("ROOP_FIQA_W_DET", 0.25)
W_SHARP = _envf("ROOP_FIQA_W_SHARP", 0.30)
W_RES = _envf("ROOP_FIQA_W_RES", 0.20)
W_POSE = _envf("ROOP_FIQA_W_POSE", 0.15)
W_NORM = _envf("ROOP_FIQA_W_NORM", 0.10)


def _clamp01(x):
    return max(0.0, min(1.0, float(x)))


def _sharpness_score(crop_bgr):
    """Laplacian variance on a size-normalised grayscale crop, so blur is judged
    consistently regardless of how large the original cutout was."""
    try:
        g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (160, 160), interpolation=cv2.INTER_AREA)
        var = float(cv2.Laplacian(g, cv2.CV_64F).var())
        # ~<50 reads blurry, ~>350 crisp after the fixed-size resize.
        return _clamp01(var / 350.0), var
    except Exception:
        return 0.5, 0.0


def _resolution_score(face):
    try:
        x1, y1, x2, y2 = [float(v) for v in face.bbox]
        side = min(x2 - x1, y2 - y1)
    except Exception:
        return 0.5, 0.0
    # <64px is too small to contribute detail; >=192px is plenty.
    return _clamp01((side - 64.0) / (192.0 - 64.0)), side


def _pose_score(face):
    """Frontality in [0,1]. Uses the real 3D pose when the landmark_3d_68 model
    ran, otherwise a 5-point-keypoint symmetry proxy (always available)."""
    pose = getattr(face, "pose", None)
    if pose is not None and len(pose) >= 3:
        pitch, yaw = abs(float(pose[0])), abs(float(pose[1]))
        # frontal (0°) -> 1, at/beyond 45° off-axis -> 0.
        return _clamp01(1.0 - max(yaw, pitch) / 45.0)
    kps = getattr(face, "kps", None)
    if kps is not None and len(kps) >= 3:
        try:
            le, re, no = np.asarray(kps[0]), np.asarray(kps[1]), np.asarray(kps[2])
            eye_mid = (le + re) / 2.0
            iod = float(np.linalg.norm(re - le)) + 1e-6
            off = abs(float(no[0]) - float(eye_mid[0])) / iod  # 0 frontal, grows toward profile
            return _clamp01(1.0 - off / 0.5)
        except Exception:
            pass
    return 0.6  # unknown — neutral-ish, never a hard reject on its own


def _norm_score(face):
    emb = getattr(face, "embedding", None)
    if emb is None:
        return None
    try:
        n = float(np.linalg.norm(np.asarray(emb, dtype=np.float32)))
    except Exception:
        return None
    # buffalo_l ArcFace norms cluster ~14 (poor) .. ~26 (strong); soft-map.
    return _clamp01((n - 14.0) / (26.0 - 14.0))


def image_quality(face, crop_bgr):
    """Return (score in [0,1], breakdown) judging the PICTURE, not the pose.

    `score_face` weights frontality, which is right when ranking crops to blend
    into a faceset — a profile does blend badly. It is exactly wrong as an
    intake gate for the multi-angle target bank, whose entire purpose is to
    collect profiles: gating on the composite score would reject the hard poses
    the feature exists to find, while happily admitting a crisp frontal frame of
    the wrong person.

    What that gate actually needs to exclude is a DEGRADED image — blurred,
    tiny, or barely detected — because a degraded embedding collapses toward the
    middle of the space, and that is how a face that is nobody in particular
    lands close enough to a captured target to be banked as them. So this is the
    same composition with frontality dropped and the rest renormalised.
    """
    score, breakdown = score_face(face, crop_bgr)
    comps = {k: breakdown[k] for k in ('sharpness', 'confidence', 'resolution')
             if k in breakdown}
    weights = {'sharpness': W_SHARP, 'confidence': W_DET, 'resolution': W_RES}
    if 'embedding' in breakdown:
        comps['embedding'] = breakdown['embedding']
        weights['embedding'] = W_NORM
    tw = sum(weights[k] for k in comps) or 1.0
    out = dict(breakdown)
    out.pop('frontality', None)
    return _clamp01(sum(comps[k] * weights[k] for k in comps) / tw), out


# Owned here rather than in api.py so there is one source of truth: a second
# env read at the call site is how a gate ends up on in one place and off in
# another, and how a test can pass while the default is disabled.
BLUR_FRAC = _envf("ROOP_ANGLE_BLUR_FRAC", 0.5)
BLUR_WARMUP = int(_envf("ROOP_ANGLE_BLUR_WARMUP", 8))


def blur_outlier(sharpness, samples, frac=None, warmup=None):
    """True when this candidate is far blurrier than what the clip is offering.

    Blur has to be judged relatively. Measured on face-like crops the sharpness
    axis reads 0.96-1.00 for a clean face and 0.03 for a mildly blurred one — a
    wide gap, but one that slides wholesale with grain, compression and focus,
    so any fixed line either passes everything on a crisp clip or rejects
    everything on a soft one. The median of the faces actually scanned tracks
    that automatically: a soft clip simply lowers the bar for itself.

    `samples` should hold every candidate seen, blurred ones included, or the
    median describes what was accepted rather than what was available. Below
    `warmup` samples there is no median worth trusting and nothing is refused.
    """
    frac = BLUR_FRAC if frac is None else frac
    warmup = BLUR_WARMUP if warmup is None else warmup
    if frac <= 0 or len(samples) < max(1, warmup):
        return False
    return float(sharpness) < frac * statistics.median(samples)


def score_face(face, crop_bgr):
    """Return (score in [0,1], breakdown dict) for one detected face + its crop."""
    comps, weights = {}, {}

    s, var = _sharpness_score(crop_bgr)
    comps["sharpness"] = s
    weights["sharpness"] = W_SHARP

    comps["confidence"] = _clamp01(getattr(face, "det_score", 0.0) or 0.0)
    weights["confidence"] = W_DET

    r, side = _resolution_score(face)
    comps["resolution"] = r
    weights["resolution"] = W_RES

    comps["frontality"] = _pose_score(face)
    weights["frontality"] = W_POSE

    ns = _norm_score(face)
    if ns is not None:
        comps["embedding"] = ns
        weights["embedding"] = W_NORM

    tw = sum(weights.values()) or 1.0
    score = sum(comps[k] * weights[k] for k in comps) / tw

    breakdown = {k: round(v, 3) for k, v in comps.items()}
    breakdown["sharp_var"] = round(var, 1)
    breakdown["face_px"] = int(round(side))
    return _clamp01(score), breakdown
