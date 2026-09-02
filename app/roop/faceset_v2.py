"""Versioned, deterministic FaceSet V2 metadata and identity-bank helpers.

The original ``.fsz`` contract is a ZIP containing root-level PNG reference
images.  V2 keeps those exact members and adds ``metadata.json``.  The image
members remain the source of truth for the legacy loader; the metadata is an
index and cached analysis layer, so loading an old archive never requires a
format rewrite and the video path never needs to recompute these measurements.

This module intentionally uses only classical image operations during FaceSet
creation.  It does not create an inference session or add per-video neural
work.  Detector-produced face objects are accepted as dicts or InsightFace
Face instances, keeping the existing extraction and provider paths intact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import copy
import tempfile
import zipfile

import cv2
import numpy as np

from roop.identity_detail import (aggregate_detail_representations,
                                   build_detail_representation)


FORMAT_NAME = "roop.fsz"
FORMAT_VERSION = 2
METADATA_MEMBER = "metadata.json"
POSE_BINS = (
    "frontal", "mild_left", "mild_right", "medium_left", "medium_right",
    "strong_left", "strong_right", "profile_left", "profile_right",
)
DEFAULT_MIN_QUALITY = 0.35
DEFAULT_MAX_ENTRIES = 32
DEFAULT_MAX_PER_BIN = 6

# Absolute motion-blur floor, applied to the native-resolution face crop.
#
# Measured before it was chosen, on the 190 reference images across the 38
# facesets in this install: native-resolution crop variance runs p05 143.6 /
# p50 242.0 / p95 432.3, and a floor of 100 rejects 0.5% of them (1 image).
# On the two existing V2 archives, whose crops are the population this gate
# actually reads, native crop variance is 211-361 and the floor rejects 0 of
# 10. It is a catastrophic-blur floor, not a selectivity knob -- the composite
# `quality.score` gate above already does the grading. Raising it towards 200
# would start rejecting 28% of real material.
DEFAULT_LAPLACIAN_FLOOR = 100.0

# Cosine-similarity floor against the median identity centroid.
#
# RE-MEASURED on the whole 38-archive library (224 detected faces), because the
# first calibration used 10 crops from 2 clean, frontal-ish archives and that
# population was not representative. Full distribution: p01 0.488 / p05 0.723 /
# p50 0.906. Two separable populations sit at the bottom:
#
#   < 0.60   n=4,  face 119-146 px  -- background bystanders in group photos,
#                                      all BELOW the library's p05 face size
#                                      (161 px)
#   0.67-0.71 n=6, face 333-452 px  -- the SUBJECT turned to profile
#
# The widest gap anywhere in the 0.55-0.75 band is 0.591 -> 0.668, exactly
# where those two populations separate, so the floor sits at 0.60.
#
# 0.70 -- the value this gate was first written with -- rejects 10 of 224, and
# 6 of those 10 are large-face profile references of the subject themselves.
# That is actively counterproductive: profile coverage is what the pose bank
# exists to provide, so discarding it degrades exactly the lateral-angle case
# the V2 format was built to improve. Do not raise this back to 0.70 without
# re-measuring; the small-n calibration that produced it is recorded here so
# the mistake is not repeated.
DEFAULT_MIN_IDENTITY_COSINE = 0.60

# A median is only a cluster statistic once there is a cluster. With two
# references the median is their midpoint and both sit equidistant from it, so
# "outlier" is undefined and the gate is skipped rather than guessed.
MIN_ENTRIES_FOR_OUTLIER_REJECTION = 3

# Spec'd 3x3 pose matrix. This is a coarse, additive index built alongside the
# existing nine-way `POSE_BINS` yaw ladder, which stays the runtime selector's
# population -- see `pose_matrix_cell`.
POSE_MATRIX_YAW_EDGES = (-25.0, 25.0)
POSE_MATRIX_PITCH_EDGES = (-15.0, 15.0)
POSE_MATRIX_YAW_NAMES = ("left", "center", "right")
POSE_MATRIX_PITCH_NAMES = ("down", "center", "up")
POSE_MATRIX_CELLS = tuple((y, p) for y in POSE_MATRIX_YAW_NAMES
                          for p in POSE_MATRIX_PITCH_NAMES)


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _array(value, shape=None):
    if value is None:
        return None
    try:
        out = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if out.size == 0 or not np.isfinite(out).all():
        return None
    if shape is not None and out.shape != shape:
        return None
    return out.copy()


def _normalised(value):
    out = _array(value)
    if out is None:
        return None
    out = out.reshape(-1)
    norm = float(np.linalg.norm(out))
    if norm <= 1e-8:
        return None
    return (out / norm).astype(np.float32)


def _json_array(value, decimals=None):
    arr = _array(value)
    if arr is None:
        return None
    if decimals is not None:
        arr = np.round(arr, decimals=decimals)
    return arr.tolist()


def _json_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _clamp01(value, default=0.0):
    value = _json_float(value, default)
    return max(0.0, min(1.0, value))


def _bbox(face):
    out = _array(_get(face, "bbox"))
    if out is None or out.size != 4:
        return None
    out = out.reshape(4)
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def _crop_for_face(image, bbox):
    if image is None or bbox is None:
        return None
    try:
        h, w = image.shape[:2]
        x0 = max(0, min(w - 1, int(math.floor(float(bbox[0])))))
        y0 = max(0, min(h - 1, int(math.floor(float(bbox[1])))))
        x1 = max(x0 + 1, min(w, int(math.ceil(float(bbox[2])))))
        y1 = max(y0 + 1, min(h, int(math.ceil(float(bbox[3])))))
        crop = image[y0:y1, x0:x1]
        return crop.copy() if crop.size else None
    except Exception:
        return None


def _landmarks_2d(face):
    for name in ("landmark_2d_106", "landmark_2d_68", "landmarks_2d", "kps"):
        value = _array(_get(face, name))
        if value is not None and value.ndim == 2 and value.shape[1] >= 2:
            return value[:, :2].astype(np.float32)
    return None


def _landmarks_68(face):
    for name in ("landmark_2d_68", "landmarks_68"):
        value = _array(_get(face, name))
        if value is not None and value.ndim == 2 and value.shape[0] >= 68 and value.shape[1] >= 2:
            return value[:68, :2].astype(np.float32)
    value = _array(_get(face, "landmark_3d_68"))
    if value is not None and value.ndim == 2 and value.shape[0] >= 68 and value.shape[1] >= 2:
        return value[:68, :2].astype(np.float32)
    return None


def _landmarks_3d(face):
    value = _array(_get(face, "landmark_3d_68"))
    if value is not None and value.ndim == 2 and value.shape[0] >= 68 and value.shape[1] >= 3:
        return value[:68, :3].astype(np.float32)
    return None


def _pose(face, landmarks_2d):
    """Return canonical (yaw, pitch, roll) degrees when available."""
    if landmarks_2d is not None and landmarks_2d.shape == (5, 2):
        try:
            # This is the project's canonical solver. Import lazily so metadata
            # validation and legacy archives work without importing detector code.
            from roop.face_util import solve_pose_5pt
            value = solve_pose_5pt(landmarks_2d)
            if value is not None:
                return tuple(float(v) for v in value)
        except Exception:
            pass
    value = _array(_get(face, "pose"))
    if value is not None and value.size >= 3:
        # InsightFace exposes pose as pitch, yaw, roll; normalize to this
        # module's public yaw, pitch, roll convention.
        return (float(value.reshape(-1)[1]), float(value.reshape(-1)[0]),
                float(value.reshape(-1)[2]))
    return None


def _pose_bin(pose):
    if pose is None or len(pose) < 3:
        return "frontal"
    yaw = float(pose[0])
    ay = abs(yaw)
    side = "left" if yaw < -1e-5 else "right"
    if ay <= 10.0:
        return "frontal"
    if ay <= 25.0:
        return f"mild_{side}"
    if ay <= 45.0:
        return f"medium_{side}"
    if ay <= 70.0:
        return f"strong_{side}"
    return f"profile_{side}"


def _sharpness(crop):
    if crop is None or crop.size == 0:
        return 0.0, 0.0
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA)
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return _clamp01(variance / 350.0), variance
    except Exception:
        return 0.0, 0.0


def _native_laplacian_variance(crop):
    """Laplacian variance on the crop at its own resolution.

    Deliberately separate from `_sharpness`, which resizes to 160x160 first so
    its 0-1 score is comparable across differently sized references. That
    resize also rescales the variance (the same crops read 148-745 resized
    against 99-513 native), so a threshold calibrated on one scale is wrong on
    the other. This is the value the documented floor is calibrated against.
    """
    if crop is None or getattr(crop, "size", 0) == 0:
        return 0.0
    try:
        gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        value = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return value if math.isfinite(value) else 0.0
    except Exception:
        return 0.0


def pose_matrix_cell(pose):
    """Return the spec'd ``(yaw_bin, pitch_bin)`` 3x3 cell for a pose.

    An unknown pose is treated as ``("center", "center")`` to match
    `_pose_bin`, which resolves a missing pose to "frontal": a reference whose
    pose could not be solved is still usable frontal-ish material, and dropping
    it would silently shrink the bank.
    """
    if pose is None or len(pose) < 2:
        return ("center", "center")
    yaw = _json_float(pose[0], 0.0) or 0.0
    pitch = _json_float(pose[1], 0.0) or 0.0
    if yaw < POSE_MATRIX_YAW_EDGES[0]:
        yaw_bin = "left"
    elif yaw > POSE_MATRIX_YAW_EDGES[1]:
        yaw_bin = "right"
    else:
        yaw_bin = "center"
    if pitch < POSE_MATRIX_PITCH_EDGES[0]:
        pitch_bin = "down"
    elif pitch > POSE_MATRIX_PITCH_EDGES[1]:
        pitch_bin = "up"
    else:
        pitch_bin = "center"
    return (yaw_bin, pitch_bin)


def pose_matrix_key(cell):
    """Serialize a ``(yaw, pitch)`` cell to its JSON object key."""
    return f"{cell[0]}_{cell[1]}"


def parse_pose_matrix_key(key):
    """Inverse of `pose_matrix_key`; returns ``None`` for an unknown key."""
    try:
        yaw_bin, pitch_bin = str(key).split("_", 1)
    except ValueError:
        return None
    if yaw_bin in POSE_MATRIX_YAW_NAMES and pitch_bin in POSE_MATRIX_PITCH_NAMES:
        return (yaw_bin, pitch_bin)
    return None


def _exposure_saturation(crop):
    if crop is None or crop.size == 0:
        return 0.5, 0.5, {}
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        mean_l = float(gray.mean())
        std_l = float(gray.std())
        exposure = _clamp01(1.0 - abs(mean_l - 0.5) / 0.5)
        saturation_mean = float(hsv[..., 1].mean()) / 255.0
        saturation = _clamp01(1.0 - max(0.0, saturation_mean - 0.72) / 0.28)
        return exposure, saturation, {
            "mean": mean_l, "std": std_l,
            "p10": float(np.percentile(gray, 10)),
            "p50": float(np.percentile(gray, 50)),
            "p90": float(np.percentile(gray, 90)),
            "saturation_mean": saturation_mean,
        }
    except Exception:
        return 0.5, 0.5, {}


def _appearance(crop):
    if crop is None or crop.size == 0:
        return {
            "luminance": {}, "skin_color_bgr": {}, "local_contrast": 0.0,
            "color_temperature": 1.0, "shadow_fraction": 0.0,
            "highlight_fraction": 0.0,
        }
    try:
        small = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[..., 0].astype(np.float32) / 255.0
        # Conservative skin region. If the crop is monochrome, back off to an
        # elliptical central face region rather than inventing color statistics.
        skin = ((hsv[..., 0] <= 35) & (hsv[..., 1] >= 18)
                & (hsv[..., 2] >= 25) & (ycrcb[..., 1] >= 125)
                & (ycrcb[..., 2] <= 190))
        yy, xx = np.ogrid[:128, :128]
        ellipse = (((xx - 64.0) / 52.0) ** 2 + ((yy - 64.0) / 60.0) ** 2) <= 1.0
        mask = skin & ellipse
        if int(mask.sum()) < 128:
            mask = ellipse
        pixels = small[mask].astype(np.float32)
        luminance = {
            "mean": float(y.mean()), "std": float(y.std()),
            "p10": float(np.percentile(y, 10)),
            "p50": float(np.percentile(y, 50)),
            "p90": float(np.percentile(y, 90)),
        }
        bgr_mean = pixels.mean(axis=0)
        bgr_std = pixels.std(axis=0)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        local_contrast = _clamp01(float(cv2.Laplacian(gray, cv2.CV_32F).std()) / 48.0)
        b, g, r = [float(v) for v in bgr_mean]
        return {
            "luminance": luminance,
            "skin_color_bgr": {"mean": bgr_mean.tolist(), "std": bgr_std.tolist()},
            "local_contrast": local_contrast,
            "color_temperature": r / max(1.0, b),
            "shadow_fraction": float((y < 0.16).mean()),
            "highlight_fraction": float((y > 0.88).mean()),
        }
    except Exception:
        return {
            "luminance": {}, "skin_color_bgr": {}, "local_contrast": 0.0,
            "color_temperature": 1.0, "shadow_fraction": 0.0,
            "highlight_fraction": 0.0,
        }


def _proportions(lm68, bbox):
    if lm68 is None or lm68.shape[0] < 68 or bbox is None:
        return {}
    try:
        eye_l = lm68[36:42].mean(axis=0)
        eye_r = lm68[42:48].mean(axis=0)
        mouth_l, mouth_r = lm68[48], lm68[54]
        eye_dist = float(np.linalg.norm(eye_r - eye_l))
        face_w = max(1.0, float(bbox[2] - bbox[0]))
        face_h = max(1.0, float(bbox[3] - bbox[1]))
        eye_mouth = float(np.linalg.norm((eye_l + eye_r) * 0.5 - (mouth_l + mouth_r) * 0.5))
        return {
            "face_width_height": face_w / face_h,
            "eye_distance_face_width": eye_dist / face_w,
            "eye_mouth_distance_face_height": eye_mouth / face_h,
            "mouth_width_face_width": float(np.linalg.norm(mouth_r - mouth_l)) / face_w,
        }
    except Exception:
        return {}


def _expression(lm68, bbox):
    if lm68 is None or lm68.shape[0] < 68 or bbox is None:
        return {}
    try:
        face_w = max(1.0, float(bbox[2] - bbox[0]))
        eye_l = lm68[36:42]
        eye_r = lm68[42:48]
        eye_width = max(1.0, float(np.linalg.norm(eye_l[3] - eye_l[0])))
        eye_width_r = max(1.0, float(np.linalg.norm(eye_r[3] - eye_r[0])))
        eye_open = (float(np.linalg.norm(eye_l[1] - eye_l[5]))
                    + float(np.linalg.norm(eye_l[2] - eye_l[4]))) / (2.0 * eye_width)
        eye_open_r = (float(np.linalg.norm(eye_r[1] - eye_r[5]))
                      + float(np.linalg.norm(eye_r[2] - eye_r[4]))) / (2.0 * eye_width_r)
        mouth_width = max(1.0, float(np.linalg.norm(lm68[54] - lm68[48])))
        mouth_open = float(np.linalg.norm(lm68[51] - lm68[57])) / mouth_width
        smile = mouth_width / face_w
        return {
            "eye_open_score": _clamp01((eye_open + eye_open_r) / 0.55),
            "mouth_open_score": _clamp01(mouth_open / 0.75),
            "smile_width_score": _clamp01(smile / 0.42),
            "descriptor": [float(eye_open), float(eye_open_r), float(mouth_open), float(smile)],
        }
    except Exception:
        return {}


def _identity_detail_crop(image, face, fallback):
    """Return a canonical source crop for V2 detail extraction when possible."""
    kps = _array(_get(face, "kps"))
    if image is not None and kps is not None and kps.shape == (5, 2):
        try:
            from roop.face_util import align_crop
            aligned, _ = align_crop(image, kps, 128, mode="arcface")
            if aligned is not None and aligned.size:
                return aligned
        except Exception:
            pass
    return fallback


def _identity_details(crop, sharp_score, image=None, face=None):
    """Cache a compact high-frequency map and candidate detail masks.

    These are deliberately candidates, never definitive mole/scar labels. A
    later consumer can compare them across selected references without adding a
    neural call to the video path.
    """
    if crop is None or crop.size == 0:
        return {"descriptor_shape": [16, 16], "descriptor": [], "mask": [], "candidates": []}
    try:
        detail_crop = _identity_detail_crop(image, face, crop)
        gray = cv2.cvtColor(cv2.resize(detail_crop, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        base = cv2.GaussianBlur(gray, (0, 0), 2.0)
        residual = gray - base
        magnitude = np.abs(residual)
        descriptor = cv2.resize(magnitude, (16, 16), interpolation=cv2.INTER_AREA)
        descriptor = np.clip(descriptor / 32.0, 0.0, 1.0)
        threshold = float(np.percentile(magnitude[4:-4, 4:-4], 92.0)) if magnitude.size else 0.0
        mask = (magnitude >= max(5.0, threshold)).astype(np.uint8)
        candidates = []
        work = magnitude.copy()
        for _ in range(8):
            y, x = np.unravel_index(int(np.argmax(work)), work.shape)
            value = float(work[y, x])
            if value < max(5.0, threshold):
                break
            if x < 5 or y < 5 or x >= 59 or y >= 59:
                work[max(0, y - 5):min(64, y + 6), max(0, x - 5):min(64, x + 6)] = 0
                continue
            patch = residual[y - 3:y + 4, x - 3:x + 4]
            candidates.append({
                "position": [round(float(x) / 63.0, 5), round(float(y) / 63.0, 5)],
                "confidence": round(_clamp01(value / 32.0) * _clamp01(sharp_score + 0.25), 5),
                "polarity": "dark" if float(patch.mean()) < 0.0 else "light",
                "mask": (np.abs(patch) >= max(3.0, value * 0.35)).astype(np.uint8).tolist(),
            })
            work[max(0, y - 5):min(64, y + 6), max(0, x - 5):min(64, x + 6)] = 0
        representation = build_detail_representation(detail_crop, sharp_score)
        return {
            "descriptor_shape": [16, 16],
            "descriptor": np.round(descriptor, 5).tolist(),
            "mask_shape": [64, 64],
            "mask": mask.tolist(),
            "candidates": candidates,
            "high_frequency": representation,
        }
    except Exception:
        return {"descriptor_shape": [16, 16], "descriptor": [], "mask": [], "candidates": []}


def _entry(face, image, source_index):
    bbox = _bbox(face)
    crop = _crop_for_face(image, bbox)
    # A freshly uploaded single-face source may retain only the cropped gallery
    # thumbnail in `ui_input_thumbs`, while its detector bbox is expressed in
    # the original frame. The image is still valid reference material; use the
    # whole stored crop for appearance/detail scoring instead of rejecting it
    # because the old coordinate space is unavailable.
    if crop is None and image is not None and getattr(image, "size", 0):
        crop = image
    lm2d = _landmarks_2d(face)
    lm68 = _landmarks_68(face)
    lm3d = _landmarks_3d(face)
    pose = _pose(face, _array(_get(face, "kps")))
    raw_embedding = _array(_get(face, "embedding"))
    normalized = _normalised(_get(face, "normed_embedding"))
    if normalized is None:
        normalized = _normalised(raw_embedding)
    sharp_score, sharp_var = _sharpness(crop)
    laplacian_variance = _native_laplacian_variance(crop)
    exposure_score, saturation_score, luminance = _exposure_saturation(crop)
    appearance = _appearance(crop)
    face_px = 0.0 if bbox is None else min(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))
    detector_conf = _clamp01(_get(face, "det_score", 0.0))
    explicit_lm_conf = _get(face, "landmark_confidence", None)
    landmark_conf = (_clamp01(explicit_lm_conf) if explicit_lm_conf is not None
                     else (1.0 if lm68 is not None or (lm2d is not None and lm2d.shape[0] >= 5) else 0.0))
    explicit_occ = _get(face, "occlusion", _get(face, "occlusion_score", None))
    occlusion = _clamp01(explicit_occ, 0.75) if explicit_occ is not None else 0.75
    face_size_score = _clamp01((face_px - 48.0) / 176.0)
    pose_suitability = 1.0 if pose is not None else 0.5
    quality_parts = {
        "face_size": face_size_score,
        "sharpness": sharp_score,
        "blur": 1.0 - sharp_score,
        "detector_confidence": detector_conf,
        "landmark_confidence": landmark_conf,
        "exposure": exposure_score,
        "saturation": saturation_score,
        "occlusion": occlusion,
        "pose_suitability": pose_suitability,
    }
    weights = {"face_size": 0.18, "sharpness": 0.20, "detector_confidence": 0.16,
               "landmark_confidence": 0.12, "exposure": 0.10, "saturation": 0.05,
               "occlusion": 0.10, "pose_suitability": 0.09}
    quality = sum(quality_parts[key] * weight for key, weight in weights.items())
    return {
        "source_index": int(source_index),
        "reference_member": "",
        "identity": {
            "arcface_embedding": _json_array(raw_embedding),
            "embedding": _json_array(raw_embedding),
            "normalized_embedding": _json_array(normalized),
            "quality_confidence": round(float(quality), 6),
        },
        "geometry": {
            "bbox": _json_array(bbox),
            "landmarks_68": _json_array(lm68),
            "landmarks_68_3d": _json_array(lm3d),
            "landmarks_2d": _json_array(lm2d),
            "yaw": _json_float(pose[0]) if pose else None,
            "pitch": _json_float(pose[1]) if pose else None,
            "roll": _json_float(pose[2]) if pose else None,
            "face_scale": {"pixels": round(face_px, 5),
                            "relative_height": round(face_px / max(1.0, float(image.shape[0])) if image is not None else 0.0, 6),
                            "relative_width": round(face_px / max(1.0, float(image.shape[1])) if image is not None else 0.0, 6)},
            "facial_proportions": _proportions(lm68, bbox),
        },
        "quality": {**{key: round(float(value), 6) for key, value in quality_parts.items()},
                    "sharpness_variance": round(float(sharp_var), 5),
                    "laplacian_variance": round(float(laplacian_variance), 5),
                    "face_pixels": round(face_px, 5),
                    "score": round(float(quality), 6)},
        "appearance": appearance,
        "expression": _expression(lm68, bbox),
        "identity_details": _identity_details(crop, sharp_score, image=image, face=face),
        "pose_bin": _pose_bin(pose),
        "pose_cell": pose_matrix_key(pose_matrix_cell(pose)),
    }


def _embedding(entry):
    return _normalised((entry.get("identity") or {}).get("normalized_embedding"))


def _cosine_distance(a, b):
    a, b = _normalised(a), _normalised(b)
    if a is None or b is None or a.shape != b.shape:
        return None
    return float(1.0 - np.dot(a, b))


def _image_bytes(image):
    if image is None or not hasattr(image, "size") or image.size == 0:
        raise ValueError("reference image is empty")
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError("could not encode reference image")
    return bytes(encoded)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _reject_blurred(entries, laplacian_floor):
    """Drop references below the absolute motion-blur floor.

    Returns ``(kept_indices, rejections)``. Runs before identity clustering so
    a smeared frame cannot drag the median centroid it would then be measured
    against.
    """
    kept, rejected = [], []
    for i, entry in enumerate(entries):
        variance = float((entry.get("quality") or {}).get("laplacian_variance", 0.0))
        if variance < laplacian_floor:
            rejected.append({"index": int(i), "reason": "motion_blur",
                             "laplacian_variance": round(variance, 5),
                             "threshold": round(float(laplacian_floor), 5)})
        else:
            kept.append(i)
    return kept, rejected


def _reject_identity_outliers(entries, candidates, min_cosine):
    """Drop references whose identity disagrees with the primary cluster.

    The reference is the per-component MEDIAN of the candidate embeddings, not
    their mean: the mean is dragged by the very outlier being looked for, so a
    single wrong-person frame in a small bank can pull the centroid far enough
    towards itself to survive its own test. The median is not.
    """
    kept, rejected = [], []
    vectors = {i: _embedding(entries[i]) for i in candidates}
    usable = [i for i in candidates if vectors[i] is not None]
    if len(usable) < MIN_ENTRIES_FOR_OUTLIER_REJECTION:
        return list(candidates), []
    sizes = {vectors[i].shape for i in usable}
    if len(sizes) != 1:
        # Mixed embedding widths cannot be clustered; leave selection to the
        # quality gate rather than rejecting on an incomparable measurement.
        return list(candidates), []
    centroid = _normalised(np.median(np.stack([vectors[i] for i in usable]), axis=0))
    if centroid is None:
        return list(candidates), []
    for i in candidates:
        vector = vectors[i]
        if vector is None:
            # No embedding to disagree with; the quality gate still governs it.
            kept.append(i)
            continue
        similarity = float(np.dot(centroid, vector))
        if similarity < min_cosine:
            rejected.append({"index": int(i), "reason": "identity_outlier",
                             "cosine_similarity": round(similarity, 6),
                             "threshold": round(float(min_cosine), 6)})
        else:
            kept.append(i)
    if not kept:
        # Every reference disagreed with the median, which means the median is
        # not describing a cluster. Refusing the whole set on that basis would
        # be a worse answer than declining to reject.
        return list(candidates), []
    return kept, rejected


def _pose_matrix(entries):
    """Normalized mean centroid embedding per non-empty 3x3 pose cell."""
    grouped = {}
    for index, entry in enumerate(entries):
        cell = entry.get("pose_cell") or pose_matrix_key(("center", "center"))
        grouped.setdefault(cell, []).append(index)
    matrix = {}
    for cell in POSE_MATRIX_CELLS:
        key = pose_matrix_key(cell)
        members = grouped.get(key)
        if not members:
            continue
        vectors, weights = [], []
        for index in members:
            vector = _embedding(entries[index])
            if vector is None:
                continue
            vectors.append(vector)
            weights.append(max(0.01, float((entries[index].get("quality") or {}).get("score", 0.0))))
        centroid = None
        if vectors and len({v.shape for v in vectors}) == 1:
            centroid = _normalised(np.average(np.asarray(vectors), axis=0,
                                              weights=np.asarray(weights)))
        matrix[key] = {
            "yaw_bin": cell[0],
            "pitch_bin": cell[1],
            "members": [int(i) for i in members],
            "support": len(members),
            "embedding": _json_array(centroid),
        }
    return matrix


def _select_candidates(entries, min_quality, max_entries, max_per_bin, allowed=None):
    valid = []
    rejected = []
    allowed = range(len(entries)) if allowed is None else allowed
    for i in allowed:
        entry = entries[i]
        q = float((entry.get("quality") or {}).get("score", 0.0))
        face_px = float((entry.get("quality") or {}).get("face_pixels", 0.0))
        if q < min_quality or face_px < 32.0:
            rejected.append({"index": int(i), "reason": "quality", "score": round(q, 6)})
        else:
            valid.append(i)
    if not valid:
        raise ValueError("no reference image met the FaceSet V2 quality threshold")

    # Cover each represented pose before filling remaining slots by quality.
    chosen = []
    for pose_bin in POSE_BINS:
        options = [i for i in valid if entries[i].get("pose_bin") == pose_bin]
        options.sort(key=lambda i: (-float(entries[i]["quality"]["score"]), i))
        if options:
            chosen.append(options[0])
    for i in sorted(valid, key=lambda j: (-float(entries[j]["quality"]["score"]), j)):
        if i in chosen or len(chosen) >= max_entries:
            continue
        same_bin = [j for j in chosen if entries[j].get("pose_bin") == entries[i].get("pose_bin")]
        if len(same_bin) >= max_per_bin:
            continue
        emb = _embedding(entries[i])
        if emb is not None and any(_cosine_distance(emb, _embedding(entries[j])) is not None
                                   and _cosine_distance(emb, _embedding(entries[j])) < 0.08
                                   and entries[j].get("pose_bin") == entries[i].get("pose_bin")
                                   for j in same_bin):
            rejected.append({"index": int(i), "reason": "near_duplicate", "score": round(float(entries[i]["quality"]["score"]), 6)})
            continue
        chosen.append(i)
    return chosen[:max_entries], rejected


def _dermal_patch(entries):
    """Pick the sharpest, highest-resolution frontal reference's detail map.

    "Frontal" widens in two steps before giving up, because the spec'd +-15
    degree pitch edge is tight enough to exclude genuinely frontal material: on
    the real `harjot` bank the head-on reference solves to yaw -4.3 / pitch
    -16.4, which lands in `center_down` and, under a strict `center_center`
    test, handed the dermal patch to a 38-degree profile instead. Yaw is what
    decides whether a face is presented to camera; a mild nod is not a profile.

    Within the chosen pool, ranked on native Laplacian variance then face
    pixels -- sharpest first, highest-resolution to break ties.
    """
    by_cell = {}
    for i, entry in enumerate(entries):
        by_cell.setdefault(entry.get("pose_cell"), []).append(i)
    strict = list(by_cell.get(pose_matrix_key(("center", "center"))) or [])
    yaw_centered = [i for pitch_bin in POSE_MATRIX_PITCH_NAMES
                    for i in (by_cell.get(pose_matrix_key(("center", pitch_bin))) or [])]
    frontal = strict or yaw_centered
    pool = frontal or list(range(len(entries)))

    def rank(index):
        quality = entries[index].get("quality") or {}
        return (float(quality.get("laplacian_variance", 0.0)),
                float(quality.get("face_pixels", 0.0)),
                -index)

    for index in sorted(pool, key=rank, reverse=True):
        detail = ((entries[index].get("identity_details") or {}).get("high_frequency"))
        if not detail or not detail.get("residual_q"):
            continue
        quality = entries[index].get("quality") or {}
        return {
            "source_index": int(index),
            "is_frontal": bool(index in frontal),
            "frontal_basis": ("pose_cell_center" if index in strict else
                              ("yaw_center" if index in yaw_centered else "none")),
            "laplacian_variance": round(float(quality.get("laplacian_variance", 0.0)), 5),
            "face_pixels": round(float(quality.get("face_pixels", 0.0)), 5),
            "texture": detail,
            "uv_anchors": _detail_uv_anchors(),
        }
    return {}


def _detail_uv_anchors():
    """Normalized 0-1 UV positions of the 5 landmarks the patch is anchored to.

    `_identity_detail_crop` aligns with ``align_crop(..., 128, mode="arcface")``,
    so the anchors are asked of the project's own `swap_template_points` for
    that exact size. Reconstructing them as ``arcface_dst * 128/112`` is the
    documented trap: 128 takes the ``% 128`` branch, which scales by size/128
    and shifts x by 8, and the naive guess is wrong by 13px at 128 while still
    looking plausible.
    """
    try:
        from roop.face_util import swap_template_points
        points = np.asarray(swap_template_points(128, "arcface"), dtype=np.float32)
        if points.shape != (5, 2) or not np.isfinite(points).all():
            return None
        return {
            "space": "arcface_128",
            "order": ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"],
            "uv": np.round(points / 128.0, 6).tolist(),
        }
    except Exception:
        return None


def prepare_faceset_v2(faces, images, source_name="", min_quality=None,
                       max_entries=None, max_per_bin=None,
                       laplacian_floor=None, min_identity_cosine=None):
    """Analyze and select source images, returning ``(metadata, indices)``."""
    min_quality = float(os.environ.get("ROOP_FACESET_V2_MIN_QUALITY", DEFAULT_MIN_QUALITY)
                        if min_quality is None else min_quality)
    max_entries = int(os.environ.get("ROOP_FACESET_V2_MAX_ENTRIES", DEFAULT_MAX_ENTRIES)
                      if max_entries is None else max_entries)
    max_per_bin = int(os.environ.get("ROOP_FACESET_V2_MAX_PER_BIN", DEFAULT_MAX_PER_BIN)
                      if max_per_bin is None else max_per_bin)
    laplacian_floor = float(os.environ.get("ROOP_FACESET_V2_BLUR_FLOOR", DEFAULT_LAPLACIAN_FLOOR)
                            if laplacian_floor is None else laplacian_floor)
    min_identity_cosine = float(
        os.environ.get("ROOP_FACESET_V2_MIN_IDENTITY_COSINE", DEFAULT_MIN_IDENTITY_COSINE)
        if min_identity_cosine is None else min_identity_cosine)
    faces = list(faces or [])
    images = list(images or [])
    entries = []
    for i, image in enumerate(images):
        face = faces[i] if i < len(faces) else (faces[0] if faces else {})
        entries.append(_entry(face, image, i))
    if not entries:
        raise ValueError("FaceSet has no reference images")

    # Pre-screen: motion blur first, then identity outliers against the median
    # of what survives. Both are absolute floors and run before the relative
    # quality/pose selection so a smeared or wrong-person frame can neither win
    # a pose slot nor shift the centroid it is measured against.
    screened, blur_rejected = _reject_blurred(entries, laplacian_floor)
    if not screened:
        raise ValueError(
            "every reference image was below the FaceSet V2 motion-blur floor "
            f"(laplacian variance < {laplacian_floor:g})")
    screened, identity_rejected = _reject_identity_outliers(
        entries, screened, min_identity_cosine)

    chosen, rejected = _select_candidates(
        entries, min_quality, max_entries, max_per_bin, allowed=screened)
    rejected = blur_rejected + identity_rejected + rejected
    selected_entries = [entries[i] for i in chosen]
    for new_index, entry in enumerate(selected_entries):
        entry["source_index"] = int(chosen[new_index])
        entry["reference_member"] = f"{new_index}.png"

    vectors = []
    weights = []
    for entry in selected_entries:
        vector = _embedding(entry)
        if vector is not None:
            vectors.append(vector)
            weights.append(max(0.01, float(entry["quality"]["score"])))
    global_embedding = None
    if vectors:
        global_embedding = np.average(np.asarray(vectors), axis=0, weights=np.asarray(weights))
        global_embedding = _normalised(global_embedding)

    pose_bank = {name: [] for name in POSE_BINS}
    for i, entry in enumerate(selected_entries):
        pose_bank.setdefault(entry.get("pose_bin", "frontal"), []).append(i)
    pose_bins = _pose_matrix(selected_entries)
    dermal_patch = _dermal_patch(selected_entries)
    descriptor_vectors = []
    detail_vectors = []
    entry_weights = [max(0.01, float((entry.get("quality") or {}).get("score", 0.0)))
                     for entry in selected_entries]
    for entry, weight in zip(selected_entries, entry_weights):
        desc = (entry.get("identity_details") or {}).get("descriptor")
        if desc:
            descriptor_vectors.append((np.asarray(desc, dtype=np.float32), weight))
        detail = (entry.get("identity_details") or {}).get("high_frequency")
        if detail:
            detail_vectors.append((detail, weight))
    global_details = {}
    if descriptor_vectors:
        total_weight = sum(w for _, w in descriptor_vectors)
        global_details = {
            "descriptor_shape": [16, 16],
            "descriptor": np.average(np.asarray([d for d, _ in descriptor_vectors]), axis=0,
                                      weights=np.asarray([w for _, w in descriptor_vectors])).round(5).tolist(),
            "support": len(descriptor_vectors),
            "confidence": round(min(1.0, total_weight / max(1.0, len(selected_entries))), 6),
        }

    persistent_detail = aggregate_detail_representations(
        [item for item, _ in detail_vectors],
        [weight for _, weight in detail_vectors]) if detail_vectors else {}
    global_details["high_frequency"] = persistent_detail

    metadata = {
        "schema": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "source_name": str(source_name or ""),
        "compatibility": {
            "legacy_root_png_members": True,
            "legacy_fields_preserved": ["faces", "ref_images", "embedding_average", "embeddings_backup", "face_3d", "face_3d_bank", "face_poses"],
        },
        "identity": {
            "arcface_embedding": _json_array(global_embedding),
            "embedding": _json_array(global_embedding),
            "normalized_embedding": _json_array(global_embedding),
            "aggregation": "quality_weighted_mean_of_pose_specific_embeddings",
        },
        "identity_details": global_details,
        # Spec'd V2 surface. `default_embedding` is the same vector as
        # `identity.normalized_embedding`, hoisted to the top level so a reader
        # needs no knowledge of the nested identity block; `pose_bins` is the
        # 3x3 yaw/pitch matrix; `dermal_patch` is the single sharpest frontal
        # texture. `pose_bank` (the nine-way yaw ladder) is retained unchanged
        # beside them because `final_quality_gate` requires the key and
        # `pose_source_selector` drives runtime selection from `sources`.
        "default_embedding": _json_array(global_embedding),
        "pose_bins": pose_bins,
        "dermal_patch": dermal_patch,
        "pose_bank": pose_bank,
        "sources": selected_entries,
        "rejected": rejected,
        "gates": {
            "laplacian_floor": round(float(laplacian_floor), 5),
            "min_identity_cosine": round(float(min_identity_cosine), 6),
            "min_quality": round(float(min_quality), 6),
            "rejected_motion_blur": len(blur_rejected),
            "rejected_identity_outlier": len(identity_rejected),
        },
        "index": {
            "embedding_metric": "cosine_distance",
            "normalized_embeddings": [_json_array(_embedding(entry)) for entry in selected_entries],
            "reference_members": [entry["reference_member"] for entry in selected_entries],
        },
    }
    return metadata, chosen


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _zip_write(zf, name, data):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 0
    info.external_attr = 0o600 << 16
    zf.writestr(info, data, compresslevel=6)


def write_faceset_v2(path, faceset, images, source_name="", min_quality=None,
                     max_entries=None, max_per_bin=None,
                     laplacian_floor=None, min_identity_cosine=None):
    """Write a V2 archive atomically while retaining root-level PNG members."""
    faces = getattr(faceset, "faces", None) or []
    # V1 loading may have placed the original first embedding in
    # `embeddings_backup` before replacing face[0].embedding with its legacy
    # average. Recover that pose-specific value for V2 serialization without
    # mutating the live FaceSet or changing the old runtime path.
    if faces and getattr(faceset, "embeddings_backup", None) is not None:
        if isinstance(faces[0], dict):
            first = dict(faces[0])
        else:
            try:
                first = type(faces[0])(faces[0])
            except Exception:
                first = copy.copy(faces[0])
        try:
            first["embedding"] = np.asarray(faceset.embeddings_backup).copy()
            faces = [first] + list(faces[1:])
        except Exception:
            pass
    metadata, selected = prepare_faceset_v2(
        faces, images, source_name=source_name, min_quality=min_quality,
        max_entries=max_entries, max_per_bin=max_per_bin,
        laplacian_floor=laplacian_floor, min_identity_cosine=min_identity_cosine)
    encoded_images = [_image_bytes(images[i]) for i in selected]
    metadata["integrity"] = {
        "sha256": {f"{j}.png": _sha256(data) for j, data in enumerate(encoded_images)},
    }
    metadata_bytes = _canonical_json(metadata)
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".faceset-v2-", suffix=".fsz", dir=parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            _zip_write(zf, METADATA_MEMBER, metadata_bytes)
            for i, data in enumerate(encoded_images):
                _zip_write(zf, f"{i}.png", data)
        os.replace(temp_path, path)
    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise
    return metadata


def validate_metadata(metadata):
    """Validate a decoded metadata object and return a compact report."""
    if not isinstance(metadata, dict):
        raise ValueError("FaceSet metadata is not an object")
    if metadata.get("schema") != FORMAT_NAME:
        raise ValueError("unsupported FaceSet schema")
    if int(metadata.get("version", -1)) != FORMAT_VERSION:
        raise ValueError(f"unsupported FaceSet version: {metadata.get('version')}")
    sources = metadata.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("FaceSet metadata has no sources")
    members = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("FaceSet source entry is not an object")
        member = source.get("reference_member")
        if not isinstance(member, str) or os.path.basename(member) != member or not member.endswith(".png"):
            raise ValueError("FaceSet source member is unsafe")
        members.append(member)
    if len(set(members)) != len(members):
        raise ValueError("FaceSet source members are duplicated")
    for source in sources:
        geometry = source.get("geometry") or {}
        bbox = geometry.get("bbox")
        if bbox is not None:
            try:
                values = [float(v) for v in bbox]
                if len(values) != 4 or not np.isfinite(values).all() or values[2] <= values[0] or values[3] <= values[1]:
                    raise ValueError("FaceSet source bbox is invalid")
            except (TypeError, ValueError) as exc:
                raise ValueError("FaceSet source bbox is invalid") from exc
        for key in ("yaw", "pitch", "roll"):
            if geometry.get(key) is not None and _json_float(geometry.get(key)) is None:
                raise ValueError(f"FaceSet source {key} is invalid")
        detail = (source.get("identity_details") or {}).get("high_frequency")
        if detail:
            shape = detail.get("shape")
            if detail.get("schema") != "roop.identity_detail.v1" or shape != [64, 64]:
                raise ValueError("FaceSet identity detail representation is invalid")
            expected = 64 * 64
            for key in ("residual_q", "confidence_q", "mask_q"):
                values = detail.get(key)
                if not isinstance(values, list) or len(values) != expected:
                    raise ValueError("FaceSet identity detail channel is invalid")
    global_detail = (metadata.get("identity_details") or {}).get("high_frequency")
    if global_detail:
        if global_detail.get("schema") != "roop.identity_detail.v1" or global_detail.get("shape") != [64, 64]:
            raise ValueError("FaceSet global identity detail representation is invalid")
        for key in ("residual_q", "confidence_q", "mask_q"):
            values = global_detail.get(key)
            if not isinstance(values, list) or len(values) != 64 * 64:
                raise ValueError("FaceSet global identity detail channel is invalid")
    pose_bank = metadata.get("pose_bank") or {}
    if not isinstance(pose_bank, dict):
        raise ValueError("FaceSet pose bank is not an object")
    for indexes in pose_bank.values():
        if not isinstance(indexes, list) or any(not isinstance(i, int) or i < 0 or i >= len(sources) for i in indexes):
            raise ValueError("FaceSet pose bank index is invalid")

    # The three keys below are optional: V2 archives written before they
    # existed (and `migrate_legacy_fsz` output from those builds) stay valid,
    # so they are checked for shape only when present.
    default_embedding = metadata.get("default_embedding")
    if default_embedding is not None and _normalised(default_embedding) is None:
        raise ValueError("FaceSet default embedding is invalid")

    pose_bins = metadata.get("pose_bins")
    if pose_bins is not None:
        if not isinstance(pose_bins, dict):
            raise ValueError("FaceSet pose bins is not an object")
        for key, cell in pose_bins.items():
            if parse_pose_matrix_key(key) is None:
                raise ValueError(f"FaceSet pose bin key is invalid: {key}")
            if not isinstance(cell, dict):
                raise ValueError("FaceSet pose bin entry is not an object")
            # NOT `members`: that name already holds the reference-member
            # list the checksum check below reads, and rebinding it here made
            # every V2 archive fail as "checksum metadata is invalid".
            cell_members = cell.get("members")
            if not isinstance(cell_members, list) or any(
                    not isinstance(i, int) or i < 0 or i >= len(sources) for i in cell_members):
                raise ValueError("FaceSet pose bin member index is invalid")
            embedding = cell.get("embedding")
            if embedding is not None and _normalised(embedding) is None:
                raise ValueError("FaceSet pose bin embedding is invalid")

    dermal = metadata.get("dermal_patch")
    if dermal:
        if not isinstance(dermal, dict):
            raise ValueError("FaceSet dermal patch is not an object")
        index = dermal.get("source_index")
        if not isinstance(index, int) or index < 0 or index >= len(sources):
            raise ValueError("FaceSet dermal patch source index is invalid")
        texture = dermal.get("texture") or {}
        if texture.get("schema") != "roop.identity_detail.v1" or texture.get("shape") != [64, 64]:
            raise ValueError("FaceSet dermal patch texture is invalid")
        for key in ("residual_q", "confidence_q", "mask_q"):
            values = texture.get(key)
            if not isinstance(values, list) or len(values) != 64 * 64:
                raise ValueError("FaceSet dermal patch channel is invalid")
        anchors = dermal.get("uv_anchors")
        if anchors is not None:
            uv = _array((anchors or {}).get("uv"))
            if uv is None or uv.shape != (5, 2):
                raise ValueError("FaceSet dermal patch UV anchors are invalid")

    hashes = ((metadata.get("integrity") or {}).get("sha256") or {})
    if not isinstance(hashes, dict):
        raise ValueError("FaceSet integrity metadata is invalid")
    for member, digest in hashes.items():
        if member not in members or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("FaceSet checksum metadata is invalid")
    return {"version": FORMAT_VERSION, "sources": len(sources), "members": tuple(members)}


def read_faceset_archive(path):
    """Read and validate a V2 archive, or return ``None`` for a legacy archive."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise ValueError(f"corrupt FaceSet member: {bad}")
            names = set(zf.namelist())
            if METADATA_MEMBER not in names:
                return None
            metadata = json.loads(zf.read(METADATA_MEMBER).decode("utf-8"))
            report = validate_metadata(metadata)
            for member in report["members"]:
                if member not in names:
                    raise ValueError(f"FaceSet metadata references missing member: {member}")
            hashes = ((metadata.get("integrity") or {}).get("sha256") or {})
            for member in report["members"]:
                expected = hashes.get(member)
                if expected and _sha256(zf.read(member)) != expected:
                    raise ValueError(f"FaceSet checksum mismatch: {member}")
            return metadata
    except zipfile.BadZipFile as exc:
        raise ValueError("invalid or corrupt .fsz ZIP archive") from exc
    except UnicodeDecodeError as exc:
        raise ValueError("FaceSet metadata is not UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("FaceSet metadata is invalid JSON") from exc


def migrate_legacy_fsz(path, output_path=None, faceset=None, images=None,
                       source_name=""):
    """Create a V2 archive from a legacy archive or an already-loaded FaceSet.

    Passing ``faceset`` and ``images`` preserves detector-derived embeddings and
    geometry. Without them, migration remains lossless for the reference PNGs
    but records empty analysis fields; the normal source-gallery loader can then
    enrich the in-memory FaceSet when it detects those references.
    """
    output_path = output_path or path
    if faceset is not None:
        return write_faceset_v2(output_path, faceset, images or getattr(faceset, "ref_images", None) or [], source_name=source_name)
    with zipfile.ZipFile(path, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f"corrupt FaceSet member: {bad}")
        names = sorted(n for n in zf.namelist() if os.path.basename(n) == n and n.lower().endswith(".png"))
        if not names:
            raise ValueError("legacy FaceSet contains no root PNG references")
        images_bytes = [zf.read(name) for name in names]
    entries = []
    for i, data in enumerate(images_bytes):
        entries.append({
            "source_index": i, "reference_member": f"{i}.png", "pose_bin": "frontal",
            "identity": {"embedding": None, "normalized_embedding": None, "quality_confidence": 0.0},
            "geometry": {}, "quality": {"score": 0.0, "face_pixels": 0.0},
            "appearance": {}, "expression": {}, "identity_details": {},
        })
    metadata = {
        "schema": FORMAT_NAME, "version": FORMAT_VERSION, "source_name": source_name,
        "compatibility": {"legacy_root_png_members": True, "migrated_without_detection": True},
        "identity": {"embedding": None, "normalized_embedding": None, "aggregation": "none"},
        "identity_details": {}, "pose_bank": {name: [] for name in POSE_BINS},
        "sources": entries, "rejected": [],
        "index": {"embedding_metric": "cosine_distance", "normalized_embeddings": [None] * len(entries),
                  "reference_members": [e["reference_member"] for e in entries]},
        "integrity": {"sha256": {f"{i}.png": _sha256(data) for i, data in enumerate(images_bytes)}},
    }
    metadata_bytes = _canonical_json(metadata)
    parent = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(parent, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".faceset-migrate-", suffix=".fsz", dir=parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            _zip_write(zf, METADATA_MEMBER, metadata_bytes)
            for i, data in enumerate(images_bytes):
                _zip_write(zf, f"{i}.png", data)
        os.replace(temp_path, output_path)
    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise
    return metadata


def measure_lighting(image, bbox=None):
    """Cheap frame lighting vector for optional runtime source selection."""
    crop = _crop_for_face(image, _array(bbox).reshape(4) if bbox is not None else None)
    return _appearance(crop)


def select_reference_index(metadata, pose=None, appearance=None, embedding=None):
    """Select a cached source index using pose, identity, quality and lighting."""
    sources = list((metadata or {}).get("sources") or [])
    if not sources:
        return 0
    target_embedding = _normalised(embedding)
    target_yaw = _json_float(pose[0], 0.0) if pose is not None and len(pose) > 0 else None
    target_pitch = _json_float(pose[1], 0.0) if pose is not None and len(pose) > 1 else None
    target_lum = ((appearance or {}).get("luminance") or {}).get("mean")
    best_i, best_score = 0, float("inf")
    for i, source in enumerate(sources):
        geo, quality, app = source.get("geometry") or {}, source.get("quality") or {}, source.get("appearance") or {}
        dist = 0.0
        if target_embedding is not None:
            d = _cosine_distance(target_embedding, _embedding(source))
            dist += 0.60 * (d if d is not None else 0.5)
        if target_yaw is not None and geo.get("yaw") is not None:
            dist += 0.25 * min(2.0, abs(target_yaw - float(geo["yaw"])) / 45.0)
        if target_pitch is not None and geo.get("pitch") is not None:
            dist += 0.10 * min(2.0, abs(target_pitch - float(geo["pitch"])) / 35.0)
        if target_lum is not None:
            src_lum = ((app.get("luminance") or {}).get("mean"))
            if src_lum is not None:
                dist += 0.05 * min(2.0, abs(float(target_lum) - float(src_lum)) / 0.25)
        dist -= 0.04 * _clamp01(quality.get("score", 0.0))
        if dist < best_score:
            best_i, best_score = i, dist
    return int(best_i)


__all__ = [
    "FORMAT_NAME", "FORMAT_VERSION", "METADATA_MEMBER", "POSE_BINS",
    "prepare_faceset_v2", "write_faceset_v2", "read_faceset_archive",
    "validate_metadata", "migrate_legacy_fsz", "measure_lighting",
    "select_reference_index",
]
