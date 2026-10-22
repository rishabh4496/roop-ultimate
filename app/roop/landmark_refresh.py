"""Re-measure a flow-placed face's landmarks with the landmark net, on the box flow produced.

WHY. ``FlowCoaster.step`` fits a SIMILARITY transform to the tracked point cloud and applies it to the KEY FRAME's
landmarks, so a face the pre-pass places by optical flow carries the key frame's expression: a blink or a jaw opening
between two detections is not followed. Measured 2026-10-22 (``tests/probe_flow_vs_neural_landmarks.py``, Weeds.mp4, K = 4,
219 frames, error in % of eye distance against the detector's own landmarks on every frame):

                              median    p90     p95     p99     max
    flow-placed               1.21      4.39    5.58    6.91    7.46
    net on the flow box       0.58      1.99    2.87    4.64    5.54       better on 92% of frames

(Love.mp4 agrees at the median, 1.32 -> 0.78; its tail is a cut the probe does not handle and both arms fail on.)

WHAT. For each flow step the pre-pass stores, run ``landmark_2d_106`` on the flow result's bbox / kps and replace the
step's landmarks with the answer. The chain itself is untouched (it keeps its own point cloud), the closing step is NOT
refreshed (it is pinned by the real detection), and the gap bridge still requires flow to arrive at the closing detection
(``flow_coast.bridge_gap``'s tolerance is the guard against a path that followed an occluder). A refreshed step is an
independent measurement, so ``procmgr_tracking._flow_bridge_geometry`` uses it as measured and does not add the linear
drift correction that exists for un-refreshed flow.

A net that is given a bad box can answer confidently and wrongly, so an answer more than ``MAX_DEVIATION_IOD`` x the
inter-ocular distance away from what flow believed is refused and the flow landmarks stay.

OPT-IN: ``ROOP_PREPASS_LANDMARK_REFRESH=1``. Nothing here has been through a swap-rate-guarded end-to-end A/B (those renders
were killed by the system for low memory on 2026-10-22), and the pre-pass bridge only runs where a detection was missed
or ``prepass_detect_interval`` > 1. It does not address why K > 1 loses swaps (identity binding, Phase 14).
"""
from __future__ import annotations

import threading
from typing import Any, List, Optional

import numpy as np

from roop.degrade import swallowed as _swallowed
from roop.env import env_bool

MAX_DEVIATION_IOD = 0.25        # measured flow-vs-truth worst case on Weeds was 0.075; the net's was 0.055

_lock = threading.Lock()
_stats = {"tried": 0, "applied": 0, "no_frame": 0, "no_model": 0, "rejected": 0, "failed": 0}


def enabled() -> bool:
    return env_bool("ROOP_PREPASS_LANDMARK_REFRESH", False)


def reset() -> None:
    with _lock:
        for k in _stats:
            _stats[k] = 0


def _count(key: str) -> None:
    with _lock:
        _stats[key] += 1


def summary_lines() -> List[str]:
    with _lock:
        s = dict(_stats)
    if not s["tried"]:
        return []
    return [f"[LandmarkRefresh] {s['applied']}/{s['tried']} flow-placed face(s) had their landmarks re-measured by the "
            f"landmark net ({s['rejected']} answers refused as too far from the flow geometry, {s['no_frame']} without "
            f"a frame, {s['no_model']} without the model, {s['failed']} errors). ROOP_PREPASS_LANDMARK_REFRESH=0 disables."]


def _iod(kps: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(kps, np.float32)[1] - np.asarray(kps, np.float32)[0]))


def refresh_landmarks(frame: Optional[np.ndarray], result: Any, lm_model: Any,
                      max_deviation_iod: float = MAX_DEVIATION_IOD) -> Optional[np.ndarray]:
    """The net's (106, 2) landmarks for ``result``'s face in ``frame``, or None (and a counter says why).

    ``lm_model`` is the analyser's ``landmark_2d_106`` model (anything with ``get(img, face)`` that sets
    ``face['landmark_2d_106']``). Pure apart from the counters, so it is testable with a fake model.
    """
    _count("tried")
    if frame is None:
        _count("no_frame")
        return None
    if lm_model is None:
        _count("no_model")
        return None
    if result is None or result.bbox is None or result.kps is None or result.landmarks is None:
        _count("failed")
        return None
    try:
        from insightface.app.common import Face
        face = Face(bbox=np.asarray(result.bbox, np.float32).reshape(4),
                    kps=np.asarray(result.kps, np.float32), det_score=1.0)
        lm_model.get(frame, face)
        lm = face.get("landmark_2d_106")
        if lm is None:
            _count("failed")
            return None
        lm = np.asarray(lm, np.float32)[:, :2]
        flow_lm = np.asarray(result.landmarks, np.float32)[:, :2]
        if lm.shape != flow_lm.shape or not np.isfinite(lm).all():
            _count("failed")
            return None
        deviation = float(np.mean(np.linalg.norm(lm - flow_lm, axis=1))) / max(_iod(result.kps), 1e-6)
        if deviation > max_deviation_iod:
            _count("rejected")
            return None
        _count("applied")
        return lm
    except Exception as exc:
        _swallowed("roop/landmark_refresh.py:refresh_landmarks", exc, "flow landmarks kept")
        _count("failed")
        return None


def refresh_with_leased_analyser(frame: Optional[np.ndarray], result: Any) -> Optional[np.ndarray]:
    """``refresh_landmarks`` on an analyser leased from the pool (the same lease the detector takes)."""
    from roop import face_util
    with face_util.lease_face_analyser() as fa:
        model = (getattr(fa, "models", None) or {}).get("landmark_2d_106")
        return refresh_landmarks(frame, result, model)
