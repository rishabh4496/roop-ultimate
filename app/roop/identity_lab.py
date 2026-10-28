"""Pure logic behind the Face Manager's Pose Radar, Auto-Prune and Morph Lab.

No globals, no detector, no I/O: lists of numbers in, plain dicts out, so every rule is testable
against a hand-built set. ``routes_facemgr`` owns the live state and calls these.

Pose convention (the project's, see ``face_util.solve_pose_5pt``): ``yaw`` > 0 is the head turned
toward the VIEWER'S right, ``pitch`` > 0 is the head tilted DOWN. So a "left profile" is yaw <= -40
(the face is turned to the viewer's left) and a "high angle" shot - camera above the subject, who
therefore appears looking down - is pitch >= +20. A set is covered when it has examples of each
bucket the swap will meet, which is the whole point of the radar.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from roop.identity_algebra import (BLEND_MODES, DEFAULT_BLEND_MODE, DEFAULT_PRIVACY_COSINE, MAX_BLEND_SOURCES,
                                   blend_vectors, deidentify, l2_normalize, normalize_weights)

BUCKETS = ("frontal", "left_profile", "right_profile", "high_angle", "low_angle")
BUCKET_LABELS = {
    "frontal": "Frontal",
    "left_profile": "Left profile",
    "right_profile": "Right profile",
    "high_angle": "High angle",
    "low_angle": "Low angle",
}

PROFILE_YAW = 40.0          # |yaw| at or beyond this is a profile
ANGLE_PITCH = 20.0          # |pitch| at or beyond this (and not a profile) is a high / low angle

# Auto-prune. A face is blurry when its sharpness component is under BOTH a fraction of the set's own
# median (so a uniformly soft phone-video set is not emptied) and an absolute ceiling; duplicates are
# near-identical embeddings in the same pose bucket.
BLUR_RELATIVE = 0.45
BLUR_ABS_FLOOR = 0.10       # the threshold never drops below this (a uniformly soft set is not emptied)
BLUR_ABS_CEILING = 0.35     # ...and never rises above this (a sharp set does not condemn merely-OK faces)
DUPLICATE_COSINE = 0.96


def pose_bucket(yaw: Optional[float], pitch: Optional[float]) -> str:
    """Which bucket a head pose belongs to. Profile wins over angle; unknown pose is frontal."""
    y = 0.0 if yaw is None or not np.isfinite(yaw) else float(yaw)
    p = 0.0 if pitch is None or not np.isfinite(pitch) else float(pitch)
    if abs(y) >= PROFILE_YAW:
        return "left_profile" if y < 0 else "right_profile"
    if abs(p) >= ANGLE_PITCH:
        return "high_angle" if p > 0 else "low_angle"
    return "frontal"


def _num(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def face_rows(meta: Sequence[Dict[str, Any]], scores: Sequence[float]) -> List[Dict[str, Any]]:
    """One row per face from the Face Manager's parallel ``meta`` / ``scores`` lists."""
    rows = []
    for i, m in enumerate(meta):
        m = m or {}
        yaw, pitch = _num(m.get("yaw")), _num(m.get("pitch"))
        rows.append({
            "index": i,
            "yaw": round(yaw, 1),
            "pitch": round(pitch, 1),
            "bucket": pose_bucket(yaw, pitch),
            "sharpness": round(_num(m.get("sharpness")), 3),
            "sharp_var": round(_num(m.get("sharp_var")), 1),
            "score": round(_num(scores[i] if i < len(scores) else None), 3),
        })
    return rows


def radar(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-bucket coverage of a set: counts, mean sharpness, member indices and what is missing."""
    buckets = {b: {"label": BUCKET_LABELS[b], "count": 0, "indices": [], "mean_sharpness": 0.0,
                   "best_score": 0.0} for b in BUCKETS}
    for r in rows:
        b = buckets[r["bucket"]]
        b["count"] += 1
        b["indices"].append(r["index"])
        b["mean_sharpness"] += r["sharpness"]
        b["best_score"] = max(b["best_score"], r["score"])
    for b in buckets.values():
        if b["count"]:
            b["mean_sharpness"] = round(b["mean_sharpness"] / b["count"], 3)
    covered = [k for k, v in buckets.items() if v["count"]]
    return {
        "buckets": buckets,
        "total": len(rows),
        "covered": covered,
        "missing": [k for k in BUCKETS if k not in covered],
        # 0..1: the share of buckets that have at least one face. Not a quality claim.
        "coverage": round(len(covered) / len(BUCKETS), 3) if rows else 0.0,
    }


def _unit(embedding: Any) -> Optional[np.ndarray]:
    return l2_normalize(embedding)


def plan_prune(rows: Sequence[Dict[str, Any]], embeddings: Sequence[Any],
               duplicate_cosine: float = DUPLICATE_COSINE,
               blur_relative: float = BLUR_RELATIVE) -> Dict[str, Any]:
    """Decide which faces an Auto-Prune would drop, and why. Mutates nothing.

    Rules, in order:
      1. BLURRY - sharpness under ``max(BLUR_ABS_FLOOR, blur_relative x median)``, capped at
         ``BLUR_ABS_CEILING``. A blurry face is KEPT when it is the only example of its pose bucket
         (a soft profile is worth more than no profile): it is reported as ``kept_for_coverage``.
      2. DUPLICATE - among the faces that survive, two in the same bucket whose embeddings are within
         ``duplicate_cosine`` of each other are the same shot twice; the better ``score`` is kept.
    Never removes every face.
    """
    n = len(rows)
    if n == 0:
        return {"remove": [], "kept_for_coverage": [], "keep": [], "thresholds": {}}
    sharp = np.array([r["sharpness"] for r in rows], dtype=np.float64)
    median = float(np.median(sharp))
    blur_below = min(BLUR_ABS_CEILING, max(BLUR_ABS_FLOOR, blur_relative * median))

    per_bucket: Dict[str, List[int]] = {}
    for r in rows:
        per_bucket.setdefault(r["bucket"], []).append(r["index"])

    remove: Dict[int, Dict[str, Any]] = {}
    coverage_kept: List[Dict[str, Any]] = []
    for r in rows:
        if r["sharpness"] < blur_below:
            if len(per_bucket[r["bucket"]]) == 1:
                coverage_kept.append({"index": r["index"], "reason": "kept_for_coverage",
                                      "bucket": r["bucket"], "sharpness": r["sharpness"]})
            else:
                remove[r["index"]] = {"index": r["index"], "reason": "blurry", "bucket": r["bucket"],
                                      "sharpness": r["sharpness"], "below": round(blur_below, 3)}

    # Duplicates among the survivors, best-scoring first so the better shot is the one that stays.
    vectors = {r["index"]: _unit(embeddings[r["index"]] if r["index"] < len(embeddings) else None) for r in rows}
    survivors = [r for r in rows if r["index"] not in remove]
    survivors.sort(key=lambda r: (-r["score"], -r["sharpness"], r["index"]))
    kept: List[Dict[str, Any]] = []
    for r in survivors:
        v = vectors[r["index"]]
        twin = None
        if v is not None:
            for k in kept:
                if k["bucket"] != r["bucket"]:
                    continue
                kv = vectors[k["index"]]
                if kv is not None and kv.shape == v.shape and float(np.dot(kv, v)) >= duplicate_cosine:
                    twin = (k, float(np.dot(kv, v)))
                    break
        if twin is None:
            kept.append(r)
        else:
            remove[r["index"]] = {"index": r["index"], "reason": "duplicate", "bucket": r["bucket"],
                                  "duplicate_of": twin[0]["index"], "cosine": round(twin[1], 4)}

    if len(remove) >= n:                     # never empty the set
        best = max(rows, key=lambda r: (r["score"], r["sharpness"]))
        remove.pop(best["index"], None)
    removed = sorted(remove.values(), key=lambda d: d["index"])
    return {
        "remove": removed,
        "kept_for_coverage": coverage_kept,
        "keep": sorted(r["index"] for r in rows if r["index"] not in remove),
        "thresholds": {"blur_below": round(blur_below, 3), "median_sharpness": round(median, 3),
                       "duplicate_cosine": duplicate_cosine},
    }


# ── Morph Lab ───────────────────────────────────────────────────────────────

def morph_summary(embeddings: Sequence[Any], weights: Sequence[float], mode: str = DEFAULT_BLEND_MODE,
                  deid: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The blend of 2-5 identities and how far it sits from each.

    ``weights`` may be free-weighted (50, 30, 20) or already normalised (0.5, 0.3, 0.2): the blend is
    always unit length, so only the RATIOS matter, and the normalised weights are reported back.
    ``mode`` is "slerp" (great-circle / spherical mean) or "linear" (normalised weighted sum).
    ``deid`` = ``{"references": [vectors], "max_cosine": c, "min_primary_cosine": m}`` pushes the blend
    away from the references until its cosine to each is <= c (``identity_algebra.deidentify``); the
    returned ``embedding`` and ``cosine_to_each`` are then the de-identified vector's.
    """
    if mode not in BLEND_MODES:
        raise ValueError(f"mode must be one of {', '.join(BLEND_MODES)}")
    if not 2 <= len(embeddings) <= MAX_BLEND_SOURCES:
        raise ValueError(f"a morph needs 2 to {MAX_BLEND_SOURCES} identities, got {len(embeddings)}")
    if len(weights) != len(embeddings):
        raise ValueError("one weight per identity")
    raw = np.array([max(0.0, _num(w)) for w in weights], dtype=np.float64)
    units = [_unit(e) for e in embeddings]
    if any(u is None for u in units):
        raise ValueError("an identity has no usable embedding")
    if raw.sum() <= 1e-12:
        raise ValueError("at least one weight must be above zero")
    used = normalize_weights(raw.tolist())
    blended = blend_vectors(units, used.tolist(), mode)
    if blended is None:
        raise ValueError("the chosen identities cancel out; pick closer ones or change the weights")
    deid_info = None
    if deid is not None:
        refs = [_unit(r) for r in (deid.get("references") or [])]
        if any(r is None for r in refs):
            raise ValueError("a de-identification reference has no usable embedding")
        res = deidentify(blended, refs, _num(deid.get("max_cosine"), DEFAULT_PRIVACY_COSINE),
                         _num(deid.get("min_primary_cosine"), 0.0))
        if res is not None and refs:
            deid_info = {"applied": res.applied, "satisfied": res.satisfied, "reason": res.reason,
                         "max_cosine": float(min(max(_num(deid.get("max_cosine"), DEFAULT_PRIVACY_COSINE), 0.0), 0.99)),
                         "cosine_before": [round(x, 4) for x in res.cosine_before],
                         "cosine_after": [round(x, 4) for x in res.cosine_after],
                         "cosine_to_blend": round(res.cosine_to_primary, 4)}
            blended = res.embedding
    cos = [round(float(np.dot(blended, u)), 4) for u in units]
    pair = [[round(float(np.dot(a, b)), 4) for b in units] for a in units]
    dominant = int(np.argmax(used))
    return {
        "mode": mode,
        "weights": [round(float(w), 4) for w in used],
        "weights_as_given": [round(float(w), 4) for w in raw],
        "cosine_to_each": cos,
        "pairwise_cosine": pair,
        "dominant": dominant,
        "deidentify": deid_info,
        "embedding": blended.astype(np.float32),
        # Below ~0.5 to the dominant source the result reads as a different person rather than a
        # blend of the inputs; surfaced as a warning, not enforced.  A de-identified result is far from
        # its references on purpose, so the warning is only about distance from the blended inputs.
        "warning": ("the blend sits far from every source (cosine to the dominant one is "
                    f"{cos[dominant]:.2f}); the swap may not resemble any of them")
        if cos[dominant] < 0.5 and deid_info is None else None,
    }
