"""Detection-to-track association cost for the whole-clip pre-pass.

`procmgr_tracking._precompute_tracks._consume` solves a Hungarian assignment
between this frame's detections and the live tracks. The cost it has always used
is `legacy_cost` below, lifted out unchanged so that it and the candidate share
one definition and one test:

    clean face :  (1 - IoU) + cosine_distance        (gated: IoU >= IOU_MIN and
                                                       cosine_distance <= EMB_MAX)
    dirty face :  (1 - IoU)                          (gated: IoU >= IOU_MIN)

A "dirty" face is one whose recognition crop is mostly its neighbour
(`roop/face_contact.py`). Its embedding measures the pair, so the legacy rule
ignores appearance for it entirely and leaves position to decide. That is the
right call when the reading is HALF the neighbour; it throws away evidence when
the reading is a fifth of it -- the contamination census (Stage 16,
`app/tests/probe_bystander_stage16.py`) found that at 0.2-0.4 crop overlap the
embedding still prefers the right person by a median 0.45 cosine, and only past
~0.4 does it stop being usable.

`composite_cost` is the Stage 16 candidate:

    C = l_id * r(c) * (1 - S)  +  l_iou * (1 - GIoU)  +  l_kpt * KPT  +  l_pose * POSE

where S is the cosine similarity to the track's identity, r(c) tapers the
identity term as contamination c rises (full strength below `contam_full`,
`contam_floor` above `contam_zero`), GIoU is generalised IoU against the
track's propagated box, KPT is the mean 5-keypoint distance to the propagated
keypoints in units of the detection's own size, and POSE is the mean absolute
yaw/pitch/roll difference in units of 90 degrees.

Nothing here touches a model or a frame; the caller supplies numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np


INF_COST = 1e5          # what `_consume` writes for "not a candidate"


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = (float(v) for v in a[:4])
    bx0, by0, bx1, by1 = (float(v) for v in b[:4])
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = (max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
             + max(0.0, bx1 - bx0) * max(0.0, by1 - by0) - inter)
    return inter / union if union > 1e-9 else 0.0


def giou(a: Sequence[float], b: Sequence[float]) -> float:
    """Generalised IoU in [-1, 1]. Unlike IoU it still ranks two DISJOINT boxes
    (by how far apart they are), which is what lets a track whose box the
    detector missed by a few pixels prefer the nearer of two candidates."""
    ax0, ay0, ax1, ay1 = (float(v) for v in a[:4])
    bx0, by0, bx1, by1 = (float(v) for v in b[:4])
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 1e-9:
        return -1.0
    hull = (max(ax1, bx1) - min(ax0, bx0)) * (max(ay1, by1) - min(ay0, by0))
    if hull <= 1e-9:
        return -1.0
    return inter / union - (hull - union) / hull


def cosine_distance(a: Any, b: Any) -> float:
    a = np.asarray(a, np.float32).reshape(-1)
    b = np.asarray(b, np.float32).reshape(-1)
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 if d <= 1e-9 else float(1.0 - np.dot(a, b) / d)


def legacy_cost(box_iou: float, cos_dist: float, dirty: bool,
                iou_min: float, emb_max: float) -> Optional[float]:
    """The cost `_consume` has always used. ``None`` = not a candidate.

    Kept bit-for-bit: `tests/test_association_cost.py` pins the four branches.
    """
    if box_iou < iou_min:
        return None
    if cos_dist > emb_max and not dirty:
        return None
    return (1.0 - box_iou) if dirty else ((1.0 - box_iou) + cos_dist)


@dataclass(frozen=True)
class CostWeights:
    identity: float = 1.0
    iou: float = 1.0
    kpt: float = 0.5
    pose: float = 0.25
    # identity reliability r(c): 1 below contam_full, linearly down to
    # contam_floor at contam_zero, flat beyond.
    contam_full: float = 0.30
    contam_zero: float = 0.60
    contam_floor: float = 0.25


def identity_reliability(contam: float, w: CostWeights) -> float:
    c = float(contam or 0.0)
    if c <= w.contam_full:
        return 1.0
    if c >= w.contam_zero:
        return w.contam_floor
    t = (c - w.contam_full) / max(w.contam_zero - w.contam_full, 1e-9)
    return 1.0 - t * (1.0 - w.contam_floor)


def propagate_kps(kps: Optional[np.ndarray], last_bbox: Optional[np.ndarray],
                  pred_bbox: np.ndarray) -> Optional[np.ndarray]:
    """The track's last keypoints carried onto its predicted box (translate by the
    centre displacement, scale by the size ratio). The cheap stand-in for optical
    flow when no flow state exists; a flow-propagated set can be passed instead."""
    if kps is None or last_bbox is None:
        return None
    k = np.asarray(kps, np.float32).reshape(-1, 2)
    lb = np.asarray(last_bbox, np.float32)
    pb = np.asarray(pred_bbox, np.float32)
    lw, lh = max(lb[2] - lb[0], 1e-3), max(lb[3] - lb[1], 1e-3)
    pw, ph = max(pb[2] - pb[0], 1e-3), max(pb[3] - pb[1], 1e-3)
    lc = np.array([(lb[0] + lb[2]) * 0.5, (lb[1] + lb[3]) * 0.5], np.float32)
    pc = np.array([(pb[0] + pb[2]) * 0.5, (pb[1] + pb[3]) * 0.5], np.float32)
    return ((k - lc) * np.array([pw / lw, ph / lh], np.float32) + pc).astype(np.float32)


def composite_cost(det_bbox: Sequence[float], det_kps: Optional[np.ndarray],
                   det_pose: Optional[np.ndarray], cos_dist: float,
                   contam: float, dirty: bool,
                   prior_bbox: Sequence[float], prior_kps: Optional[np.ndarray],
                   prior_pose: Optional[np.ndarray],
                   iou_min: float, emb_max: float,
                   w: CostWeights = CostWeights()) -> Optional[float]:
    """The Stage 16 cost. ``None`` = not a candidate.

    Gates are the legacy ones with ONE change: a dirty face is no longer
    appearance-blind, it is appearance-DISCOUNTED (`identity_reliability`).
    """
    b_iou = iou(det_bbox, prior_bbox)
    if b_iou < iou_min:
        return None
    if cos_dist > emb_max and not dirty:
        return None
    sim = 1.0 - cos_dist
    cost = w.identity * identity_reliability(contam, w) * (1.0 - sim)
    cost += w.iou * (1.0 - giou(det_bbox, prior_bbox))
    if det_kps is not None and prior_kps is not None and w.kpt > 0:
        dk = np.asarray(det_kps, np.float32).reshape(-1, 2)
        pk = np.asarray(prior_kps, np.float32).reshape(-1, 2)
        if dk.shape == pk.shape:
            db = np.asarray(det_bbox, np.float32)
            size = max(float(np.sqrt(max((db[2] - db[0]) * (db[3] - db[1]), 1e-6))), 1.0)
            cost += w.kpt * float(np.mean(np.linalg.norm(dk - pk, axis=1))) / size
    if det_pose is not None and prior_pose is not None and w.pose > 0:
        dp = np.asarray(det_pose, np.float32).reshape(-1)[:3]
        pp = np.asarray(prior_pose, np.float32).reshape(-1)[:3]
        if dp.shape == pp.shape and np.isfinite(dp).all() and np.isfinite(pp).all():
            cost += w.pose * float(np.mean(np.abs(dp - pp))) / 90.0
    return float(cost)
