"""Classical pose geometry and temporal FaceSet V2 source selection.

This module consumes detector landmarks and cached FaceSet V2 metadata.  It
does not create inference sessions or add neural work to the video path.  V1
source-bank selection remains in ``FaceSet.select_reference_index``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os

import numpy as np


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _finite(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _clamp(value, low=0.0, high=1.0):
    value = _finite(value, low)
    return max(low, min(high, value))


def _array(value, ndim=None):
    if value is None:
        return None
    try:
        out = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if out.size == 0 or not np.isfinite(out).all():
        return None
    if ndim is not None and out.ndim != ndim:
        return None
    return out


def _kps(face):
    value = _array(_get(face, "kps"), ndim=2)
    if value is None or value.shape[0] < 5 or value.shape[1] < 2:
        return None
    return value[:5, :2].astype(np.float32)


def _bbox(face):
    value = _array(_get(face, "bbox"))
    if value is None or value.size != 4:
        return None
    value = value.reshape(4).astype(np.float32)
    return value if value[2] > value[0] and value[3] > value[1] else None


def _lm68(face):
    for name in ("landmark_2d_68", "landmarks_68", "landmark_3d_68"):
        value = _array(_get(face, name), ndim=2)
        if value is not None and value.shape[0] >= 68 and value.shape[1] >= 2:
            return value[:68, :2].astype(np.float32)
    return None


def _wrap180(value):
    return float((float(value) + 180.0) % 360.0 - 180.0)


def _angle_distance(a, b):
    return None if a is None or b is None else abs(_wrap180(float(a) - float(b)))


def _off_axis(yaw, pitch):
    if yaw is None or pitch is None:
        return 90.0
    value = math.cos(math.radians(float(yaw))) * math.cos(math.radians(float(pitch)))
    return math.degrees(math.acos(max(-1.0, min(1.0, value))))


def _proportions(lm, bbox):
    if lm is None or bbox is None:
        return {}
    try:
        eye_l, eye_r = lm[36:42].mean(axis=0), lm[42:48].mean(axis=0)
        width = max(1.0, float(bbox[2] - bbox[0]))
        height = max(1.0, float(bbox[3] - bbox[1]))
        return {
            "face_width_height": width / height,
            "eye_distance_face_width": float(np.linalg.norm(eye_r - eye_l)) / width,
            "eye_mouth_distance_face_height": float(np.linalg.norm(
                (eye_l + eye_r) * 0.5 - (lm[48] + lm[54]) * 0.5)) / height,
            "mouth_width_face_width": float(np.linalg.norm(lm[54] - lm[48])) / width,
        }
    except Exception:
        return {}


def expression_from_landmarks(lm, bbox=None):
    """Return the compact expression descriptor used by FaceSet V2."""
    if lm is None or lm.shape[0] < 68:
        return {}
    try:
        if bbox is None:
            lo, hi = lm.min(axis=0), lm.max(axis=0)
            bbox = np.asarray([lo[0], lo[1], hi[0], hi[1]], dtype=np.float32)
        left, right = lm[36:42], lm[42:48]
        lw = max(1.0, float(np.linalg.norm(left[3] - left[0])))
        rw = max(1.0, float(np.linalg.norm(right[3] - right[0])))
        eye_l = (float(np.linalg.norm(left[1] - left[5]))
                 + float(np.linalg.norm(left[2] - left[4]))) / (2.0 * lw)
        eye_r = (float(np.linalg.norm(right[1] - right[5]))
                 + float(np.linalg.norm(right[2] - right[4]))) / (2.0 * rw)
        mouth_w = max(1.0, float(np.linalg.norm(lm[54] - lm[48])))
        mouth = float(np.linalg.norm(lm[51] - lm[57])) / mouth_w
        smile = mouth_w / max(1.0, float(bbox[2] - bbox[0]))
        return {
            "eye_open_score": _clamp((eye_l + eye_r) / 0.55),
            "mouth_open_score": _clamp(mouth / 0.75),
            "smile_width_score": _clamp(smile / 0.42),
            "descriptor": [eye_l, eye_r, mouth, smile],
        }
    except Exception:
        return {}


@dataclass(frozen=True)
class PoseEstimate:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    face_scale: float = 0.0
    relative_scale: float = 0.0
    proportions: dict = field(default_factory=dict)
    expression: dict = field(default_factory=dict)
    confidence: float = 0.0
    off_axis: float = 0.0
    perspective_risk: float = 0.0
    inverted: bool = False
    available: bool = False

    def as_dict(self):
        return {"yaw": float(self.yaw), "pitch": float(self.pitch),
                "roll": float(self.roll), "face_scale": float(self.face_scale),
                "relative_scale": float(self.relative_scale),
                "proportions": dict(self.proportions),
                "expression": dict(self.expression),
                "confidence": float(self.confidence),
                "off_axis": float(self.off_axis),
                "perspective_risk": float(self.perspective_risk),
                "inverted": bool(self.inverted), "available": bool(self.available)}


def _pose_from_kps(kps):
    if kps is None:
        return None
    try:
        from roop.face_util import solve_pose_jaw_5pt, solve_pose_5pt
        value = solve_pose_jaw_5pt(kps)
        if value is None:
            value = solve_pose_5pt(kps)
        if value is not None and len(value) >= 3:
            return tuple(float(value[i]) for i in range(3))
    except Exception:
        pass
    return None


def estimate_target_pose(face, frame_shape=None):
    """Estimate target geometry, reusing an ordered temporal annotation when present."""
    cached = _get(face, "_pose_v5")
    if isinstance(cached, dict) and cached.get("available"):
        try:
            return PoseEstimate(
                yaw=float(cached.get("yaw", 0.0)), pitch=float(cached.get("pitch", 0.0)),
                roll=float(cached.get("roll", 0.0)), face_scale=float(cached.get("face_scale", 0.0)),
                relative_scale=float(cached.get("relative_scale", 0.0)),
                proportions=dict(cached.get("proportions") or {}),
                expression=dict(cached.get("expression") or {}),
                confidence=_clamp(cached.get("confidence", 0.0)),
                off_axis=float(cached.get("off_axis", 0.0)),
                perspective_risk=_clamp(cached.get("perspective_risk", 0.0)),
                inverted=bool(cached.get("inverted", False)), available=True)
        except (TypeError, ValueError):
            pass
    kps, bbox, lm = _kps(face), _bbox(face), _lm68(face)
    pose = _pose_from_kps(kps)
    if pose is None:
        value = _array(_get(face, "pose"))
        if value is not None and value.size >= 3:
            value = value.reshape(-1)
            pose = (float(value[1]), float(value[0]), float(value[2]))
    available = pose is not None
    yaw, pitch, roll = pose if pose is not None else (0.0, 0.0, 0.0)
    resolved_roll = _finite(_get(face, "roll_deg"), None)
    roll = _wrap180(resolved_roll if resolved_roll is not None else roll)
    if bbox is not None:
        width, height = float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])
        face_scale = min(width, height)
    elif lm is not None:
        extent = lm.max(axis=0) - lm.min(axis=0)
        width, height, face_scale = float(extent[0]), float(extent[1]), float(min(extent))
    else:
        width = height = face_scale = 0.0
    det = _clamp(_get(face, "det_score", 0.0))
    lm_conf_value = _get(face, "landmark_confidence", None)
    lm_conf = (_clamp(lm_conf_value) if lm_conf_value is not None
               else (1.0 if lm is not None or kps is not None else 0.0))
    size_conf = _clamp((face_scale - 24.0) / 128.0) if face_scale else 0.0
    confidence = 0.52 * det + 0.28 * lm_conf + 0.12 * available + 0.08 * size_conf
    if frame_shape is not None and len(frame_shape) >= 2:
        frame_min = max(1.0, float(min(frame_shape[:2])))
        confidence = 0.90 * confidence + 0.10 * _clamp(face_scale / (0.08 * frame_min))
    relative_scale = 0.0
    if frame_shape is not None and len(frame_shape) >= 2:
        relative_scale = face_scale / max(1.0, float(frame_shape[0]))
    asymmetry = 0.0
    if kps is not None:
        left, right = np.linalg.norm(kps[2] - kps[0]), np.linalg.norm(kps[1] - kps[2])
        asymmetry = abs(float(left - right)) / max(1.0, float(left + right))
    aspect = width / max(1.0, height)
    perspective = _clamp(0.55 * asymmetry / 0.45 + 0.45 * abs(aspect - 0.78) / 0.55)
    return PoseEstimate(float(yaw), float(pitch), float(roll), float(face_scale),
                        float(relative_scale),
                        _proportions(lm, bbox), expression_from_landmarks(lm, bbox),
                        _clamp(confidence), _off_axis(yaw, pitch), perspective,
                        abs(roll) >= 135.0, bool(available))


def _source_pose(entry):
    geo = entry.get("geometry") or {}
    scale = geo.get("face_scale") or {}
    return {"yaw": _finite(geo.get("yaw"), None),
            "pitch": _finite(geo.get("pitch"), None),
            "roll": _finite(geo.get("roll"), 0.0),
            "face_scale": _finite(scale.get("pixels"), None),
            "relative_scale": _finite(scale.get("relative_height"), None),
            "proportions": geo.get("facial_proportions") or {}}


def _target(target, name, default=None):
    if isinstance(target, PoseEstimate):
        return getattr(target, name, default)
    return (target or {}).get(name, default)


def _prop_distance(a, b):
    vals = []
    for key in ("face_width_height", "eye_distance_face_width",
                "eye_mouth_distance_face_height", "mouth_width_face_width"):
        if a.get(key) is not None and b.get(key) is not None:
            vals.append(min(2.0, abs(float(a[key]) - float(b[key]))
                           / max(0.05, abs(float(b[key])))))
    return float(np.mean(vals)) if vals else 0.0


def _expr_distance(a, b):
    a, b = (a or {}).get("descriptor") or [], (b or {}).get("descriptor") or []
    if len(a) != len(b) or not a:
        return 0.0
    scales = (0.55, 0.55, 0.75, 0.42)
    return float(np.mean([min(2.0, abs(float(x) - float(y)) / scale)
                          for x, y, scale in zip(a, b, scales)]))


def _light_distance(target, source):
    if not target or not source:
        return 0.0
    vals = []
    tl, sl = (target.get("luminance") or {}).get("mean"), (source.get("luminance") or {}).get("mean")
    if tl is not None and sl is not None:
        vals.append(min(2.0, abs(float(tl) - float(sl)) / 0.25))
    tt, st = _finite(target.get("color_temperature"), None), _finite(source.get("color_temperature"), None)
    if tt is not None and st is not None:
        vals.append(min(2.0, abs(tt - st) / 0.22))
    tc, sc = ((target.get("skin_color_bgr") or {}).get("mean") or [],
              (source.get("skin_color_bgr") or {}).get("mean") or [])
    if tc and len(tc) == len(sc):
        vals.append(min(2.0, float(np.mean(np.abs(np.asarray(tc) - np.asarray(sc))) / 32.0)))
    return float(np.mean(vals)) if vals else 0.0


def _quality(entry):
    q = entry.get("quality") or {}
    iq = entry.get("identity") or {}
    quality, identity = _finite(q.get("score"), None), _finite(iq.get("quality_confidence"), None)
    quality = identity if quality is None else quality
    return _clamp(0.65 * quality + 0.35 * identity) if identity is not None else _clamp(quality)


def score_source_entry(entry, target, appearance=None, expression=None):
    """Return ``(cost, diagnostics)``; lower cost is better."""
    source = _source_pose(entry)
    ty, tp, tr = (_finite(_target(target, "yaw"), 0.0),
                  _finite(_target(target, "pitch"), 0.0),
                  _finite(_target(target, "roll"), 0.0))
    if source["yaw"] is None or source["pitch"] is None:
        pose_cost, pose_bits = 1.25, {"yaw": 2.0, "pitch": 2.0, "roll": 2.0, "missing": True}
    else:
        dy = min(2.0, abs(ty - source["yaw"]) / 55.0)
        dp = min(2.0, abs(tp - source["pitch"]) / 42.0)
        dr = min(2.0, (_angle_distance(tr, source["roll"]) or 0.0) / 55.0)
        pose_cost, pose_bits = 0.52 * dy + 0.28 * dp + 0.20 * dr, {"yaw": dy, "pitch": dp, "roll": dr, "missing": False}
    pose_weight = 0.50 + 0.18 * _clamp(_target(target, "confidence", 0.0))
    quality_cost = 1.0 - _quality(entry)
    expression_cost = _expr_distance(expression, entry.get("expression"))
    lighting_cost = _light_distance(appearance, entry.get("appearance"))
    geometry_cost = _prop_distance(_target(target, "proportions", {}) or source["proportions"], source["proportions"])
    scale_cost = 0.0
    ts, ss = _finite(_target(target, "relative_scale"), None), source["relative_scale"]
    if ts and ss:
        scale_cost = min(1.0, abs(math.log(ts / ss)) / math.log(2.0))
    total = (pose_weight * pose_cost + 0.18 * quality_cost
             + 0.10 * min(1.0, expression_cost) + 0.08 * min(1.0, lighting_cost)
             + 0.09 * min(1.0, geometry_cost) + 0.05 * scale_cost)
    return float(total), {"pose_cost": float(pose_cost), "pose": pose_bits,
                          "quality": _quality(entry), "expression_cost": expression_cost,
                          "lighting_cost": lighting_cost, "geometry_cost": geometry_cost,
                          "scale_cost": scale_cost, "source_pose": source, "total": float(total)}


@dataclass(frozen=True)
class SourceSelection:
    index: int
    score: float
    switched: bool
    needs_3d: bool
    reason: str
    diagnostics: dict = field(default_factory=dict)


def select_pose_aware_source(metadata, target, appearance=None, expression=None,
                             previous_index=None, switch_margin=None):
    """Select a V2 source, applying hysteresis around pose-bank boundaries."""
    sources = list((metadata or {}).get("sources") or [])
    if not sources:
        return SourceSelection(0, float("inf"), False, True, "no_sources", {})
    scored = [score_source_entry(entry, target, appearance, expression) for entry in sources]
    best, (best_score, _) = min(enumerate(scored), key=lambda item: (item[1][0], item[0]))
    chosen, switched = best, False
    margin = _finite(switch_margin, _finite(os.environ.get("ROOP_POSE_SOURCE_SWITCH_MARGIN"), 0.035))
    if previous_index is not None:
        try:
            previous = int(previous_index)
        except (TypeError, ValueError):
            previous = -1
        if 0 <= previous < len(scored):
            if scored[previous][0] <= best_score + max(0.0, margin):
                chosen = previous
            else:
                switched = previous != best
    score, details = scored[chosen]
    yaw = _finite(_target(target, "yaw"), 0.0)
    lower = [i for i, e in enumerate(sources) if (_source_pose(e)["yaw"] is not None and _source_pose(e)["yaw"] < yaw)]
    upper = [i for i, e in enumerate(sources) if (_source_pose(e)["yaw"] is not None and _source_pose(e)["yaw"] > yaw)]
    between = bool(lower and upper and abs(_source_pose(sources[chosen])["yaw"] - yaw) > 10.0)
    confidence, off_axis = _clamp(_target(target, "confidence", 0.0)), _finite(_target(target, "off_axis"), 90.0)
    inverted = bool(_target(target, "inverted", False))
    pose_gap, geometry_gap = details["pose_cost"], details["geometry_cost"]
    if inverted:
        reason = "inverted"
    elif not bool(_target(target, "available", False)) or confidence < 0.45:
        reason = "low_pose_confidence"
    elif pose_gap > 0.42:
        reason = "source_pose_gap"
    elif between and pose_gap > 0.12:
        reason = "between_source_poses"
    elif off_axis >= 72.0 and pose_gap > 0.16:
        reason = "unusually_rotated"
    elif geometry_gap > 0.42:
        reason = "proportion_mismatch"
    elif _clamp(_target(target, "perspective_risk", 0.0)) > 0.72:
        reason = "perspective_risk"
    else:
        reason = "source_pose_sufficient"
    details = dict(details)
    details.update({"best_index": int(best), "best_score": float(best_score),
                    "chosen_index": int(chosen), "switched": bool(switched),
                    "between_source_poses": between})
    return SourceSelection(int(chosen), float(score), bool(switched),
                           reason != "source_pose_sufficient", reason, details)


def annotate_face_pose(face, frame_shape=None):
    """Stamp an ordered, detached pose record on a replayed face."""
    estimate = estimate_target_pose(face, frame_shape=frame_shape)
    try:
        face["_pose_v5"] = estimate.as_dict()
    except Exception:
        try:
            setattr(face, "_pose_v5", estimate.as_dict())
        except Exception:
            pass
    return estimate


__all__ = ["PoseEstimate", "SourceSelection", "annotate_face_pose",
           "estimate_target_pose", "expression_from_landmarks",
           "score_source_entry", "select_pose_aware_source"]
