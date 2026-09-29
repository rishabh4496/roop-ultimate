"""Pose-adaptive SOURCE routing: which source identity vector a target face is
swapped with, chosen by the target face's head pose.

Two halves, deliberately on different people:

* ``SourcePortfolio`` is the SOURCE person (the identity pasted in), built
  from their FaceSet's faces with the Stage 2/3 machinery: a per-face pose and
  quality, the 9-bin lattice (``angle_portfolio.AngleBin``), and a fused
  L2-normalised embedding of the frontal and quarter views. It lives on the
  FaceSet (``faceset.angle_portfolio``), so it travels with that source into
  the render and into queued jobs.
* ``FrameLUT`` is the TARGET person's pose per frame, read off the Stage 1
  scan of the target clip: frame index -> (yaw, pitch_up, bin, bbox). It lets
  the render route a frame by a dictionary read instead of a pose solve.

The target portfolio (Stages 1-4) must NOT be fed to the swapper: it is the
person being replaced, and conditioning the swap on it re-renders them onto
themselves. It reaches the render the right way already - as angles in the
target person's recognition bank ("Add to angle bank").

Routing (``route``), per target face:
  |yaw| > 35 and the source has the matching profile bin (5 left / 6 right):
      e = normalise(0.7 * e_profile + 0.3 * e_fused)
  otherwise:
      e = e_fused
The swap's input tensor keeps its shape (one d-vector), so switching the
vector between frames re-binds nothing in the TensorRT context; ``route``
refuses a vector of any other dimension than the source's own.

Frame indices: the LUT is keyed by ABSOLUTE 0-based decoder index. The render
counts from 0 at the trim start, so ProcessMgr adds its trim offset before the
lookup (``lookup(frame_start + frame_idx, ...)``).
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from roop.angle_portfolio import AngleBin, angle_bin, neutral_pitch, pitch_up
from roop.degrade import swallowed as _swallowed

PROFILE_YAW_DEG = 35.0
PROFILE_WEIGHT = 0.7          # e_render = 0.7 e_profile + 0.3 e_fused
FAR_EYE_DAMPING = 0.25        # enhancement on the foreshortened eye is cut by this
LUT_MIN_IOU = 0.3             # a LUT row applies only to the face it was measured on
NEAR_MIN_IOU = 0.15           # ...looser for a neighbouring scanned frame (motion)
FUSION_BINS = (AngleBin.BIN_0_FRONTAL, AngleBin.BIN_1_QUARTER_LEFT, AngleBin.BIN_2_QUARTER_RIGHT)


def _unit(v: Any) -> Optional[np.ndarray]:
    if v is None:
        return None
    try:
        a = np.asarray(v, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    n = float(np.linalg.norm(a))
    if not (n > 1e-9 and np.isfinite(a).all()):
        return None
    return a / n


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _field(face: Any, name: str) -> Any:
    if isinstance(face, dict):
        return face.get(name)
    try:
        return getattr(face, name, None)
    except Exception as exc:  # insightface Face.__getattr__ can raise on odd keys
        _swallowed("roop/source_portfolio.py:_field", exc, "field read as missing")
        return None


# ── Source side ──────────────────────────────────────────────────────────────
@dataclass
class SourceRef:
    embedding: np.ndarray          # unit vector
    yaw: float
    pitch_up: float
    score: float
    face_index: int                # index into the FaceSet's faces


@dataclass
class SourcePortfolio:
    refs: Dict[AngleBin, SourceRef]
    fused: np.ndarray              # unit vector
    dim: int
    rejected: Dict[str, int] = field(default_factory=dict)
    pitch_neutral: float = 0.0     # the source person's neutral pitch (see neutral_pitch)

    def summary(self) -> Dict[str, Any]:
        return {
            "bins": {b.name: {"face_index": r.face_index, "yaw": round(r.yaw, 1),
                              "pitch_up": round(r.pitch_up, 1), "score": round(r.score, 3)}
                     for b, r in sorted(self.refs.items())},
            "profile_left": AngleBin.BIN_5_PROFILE_LEFT in self.refs,
            "profile_right": AngleBin.BIN_6_PROFILE_RIGHT in self.refs,
            "fused_sources": [b.name for b in FUSION_BINS if b in self.refs],
            "dim": self.dim,
            "pitch_neutral": round(self.pitch_neutral, 2),
            "rejected": dict(self.rejected),
        }


def _face_score(face: Any) -> float:
    """Detector confidence x embedding-norm quality (face_quality's MagFace-style
    term). Both come with the Face; no crop or model is needed."""
    det = _field(face, "det_score")
    det = float(det) if det is not None and math.isfinite(float(det)) else 0.5
    emb = _field(face, "embedding")
    norm_q = 0.5
    if emb is not None:
        n = float(np.linalg.norm(np.asarray(emb, dtype=np.float32)))
        norm_q = max(0.0, min(1.0, (n - 14.0) / (26.0 - 14.0)))
    return max(0.0, min(1.0, det)) * (0.5 + 0.5 * norm_q)


def build_source_portfolio(faceset: Any, ear_min: Optional[float] = None) -> Optional[SourcePortfolio]:
    """Stages 2-3 over a SOURCE FaceSet's faces.

    Pose is the weak-perspective ``face_util.solve_pose_5pt`` on each face's own
    keypoints: a source still has no video frame, and that solver reads the
    face's appearance, which is what the bins mean. A face whose 68-point
    landmarks show closed eyes (EAR < 0.20, both eyes) is left out. Returns None
    when no face has both an embedding and a solvable pose, or when no frontal /
    quarter face exists to fuse (the routing needs e_fused)."""
    from roop.face_util import solve_pose_5pt
    from roop.pose_quality import EAR_MIN, eye_aspect_ratios

    ear_min = EAR_MIN if ear_min is None else float(ear_min)
    faces = list(getattr(faceset, "faces", None) or [])
    rejected: Dict[str, int] = {}
    best: Dict[AngleBin, SourceRef] = {}
    dim = None
    poses = [solve_pose_5pt(_field(face, "kps")) for face in faces]
    neutral = neutral_pitch((p[0], p[1]) for p in poses if p is not None)
    # FaceSet.AverageEmbeddings (V1 sets, on load) OVERWRITES faces[0].embedding
    # with the mean of every face and keeps face 0's own vector in
    # embeddings_backup. Binning that mean under face 0's pose would mix every
    # angle into one bin, so face 0 is read from the backup.
    backup = getattr(faceset, "embeddings_backup", None)
    for i, face in enumerate(faces):
        raw = backup if (i == 0 and backup is not None) else _field(face, "embedding")
        emb = _unit(raw)
        if emb is None:
            rejected["no_embedding"] = rejected.get("no_embedding", 0) + 1
            continue
        if dim is None:
            dim = emb.size
        elif emb.size != dim:
            rejected["dimension"] = rejected.get("dimension", 0) + 1
            continue
        pose = poses[i]
        if pose is None:
            rejected["no_pose"] = rejected.get("no_pose", 0) + 1
            continue
        ears = eye_aspect_ratios(_field(face, "landmark_3d_68"))
        if ears is not None and max(ears) < ear_min:
            rejected["eyes_closed"] = rejected.get("eyes_closed", 0) + 1
            continue
        yaw, pitch, _roll = pose
        b = angle_bin(yaw, pitch - neutral)
        if b is None:
            rejected["between_bins"] = rejected.get("between_bins", 0) + 1
            continue
        ref = SourceRef(embedding=emb, yaw=float(yaw), pitch_up=pitch_up(pitch - neutral),
                        score=_face_score(face), face_index=i)
        if b not in best or ref.score > best[b].score:
            best[b] = ref
    if not best or dim is None:
        return None
    vecs = [best[b].embedding for b in FUSION_BINS if b in best]
    weights = [max(best[b].score, 1e-3) for b in FUSION_BINS if b in best]
    if not vecs:
        rejected["no_frontal_or_quarter"] = 1
        return None
    fused = _unit(np.average(np.stack(vecs), axis=0, weights=weights))
    if fused is None:
        return None
    return SourcePortfolio(refs=best, fused=fused, dim=dim, rejected=rejected, pitch_neutral=neutral)


# ── Target side ──────────────────────────────────────────────────────────────
@dataclass
class LUTEntry:
    yaw: float
    pitch_up: float
    bin: Optional[AngleBin]
    bbox: Tuple[float, float, float, float]


@dataclass
class FrameLUT:
    """Target pose per scanned frame (absolute 0-based decoder indices)."""
    media_path: str
    step: int
    entries: Dict[int, List[LUTEntry]]

    def lookup(self, frame_idx: Optional[int], bbox: Sequence[float]) -> Tuple[Optional[LUTEntry], str]:
        """(entry, how) for a face at ``bbox``. how: "lut" (this frame),
        "near" (the closest scanned frame within one stride), or "miss". A row
        is only used for the face it was measured on (IoU with ``bbox``), so a
        second person in the frame never takes the tracked person's pose."""
        if frame_idx is None or not self.entries:
            return None, "miss"
        rows = self.entries.get(int(frame_idx))
        if rows:
            hit = max(rows, key=lambda e: _iou(e.bbox, bbox))
            if _iou(hit.bbox, bbox) >= LUT_MIN_IOU:
                return hit, "lut"
        for k in range(1, max(1, self.step) + 1):
            for idx in (int(frame_idx) - k, int(frame_idx) + k):
                rows = self.entries.get(idx)
                if not rows:
                    continue
                hit = max(rows, key=lambda e: _iou(e.bbox, bbox))
                if _iou(hit.bbox, bbox) >= NEAR_MIN_IOU:
                    return hit, "near"
        return None, "miss"

    def __len__(self) -> int:
        return len(self.entries)


def build_frame_lut(scan: Dict[str, Any]) -> Optional[FrameLUT]:
    """From a Stage 1 scan result (``scanner`` output): every detection of the
    matching tracklets, posed with Stage 2's line-of-sight solver."""
    from roop.pose_quality import estimate_head_pose
    shape = scan.get("frame_shape") or [0, 0]
    if not (shape and shape[0] and shape[1]):
        return None
    entries: Dict[int, List[LUTEntry]] = {}
    for track in scan.get("tracks", []):
        posed = []
        for det in track.get("detections", []):
            kps, bbox = det.get("kps"), det.get("bbox")
            if kps is None or bbox is None:
                continue
            yaw, pitch, _roll = estimate_head_pose(np.asarray(kps, dtype=np.float64), tuple(shape))
            if math.isfinite(yaw) and math.isfinite(pitch):
                posed.append((det, yaw, pitch, bbox))
        neutral = neutral_pitch((y, p) for _, y, p, _ in posed)     # per person (track)
        for det, yaw, pitch, bbox in posed:
            rel = pitch - neutral
            entries.setdefault(int(det["frame_idx"]), []).append(
                LUTEntry(yaw=float(yaw), pitch_up=pitch_up(rel), bin=angle_bin(yaw, rel),
                         bbox=tuple(float(v) for v in bbox)))
    if not entries:
        return None
    return FrameLUT(media_path=str(scan.get("video_path") or ""), step=int(scan.get("step_frames") or 1),
                    entries=entries)


# ── Routing ──────────────────────────────────────────────────────────────────
@dataclass
class Route:
    embedding: np.ndarray          # unit vector to condition the swap on
    kind: str                      # "profile" | "fused"
    bin: Optional[AngleBin]
    yaw: float
    pitch_up: float
    pose_from: str                 # "lut" | "near" | "live"
    far_eye: Optional[int]         # 0 / 1 = which 5-pt eye to damp; None = frontal


def far_eye_index(kps: Any) -> Optional[int]:
    """The foreshortened eye (0 or 1 in 5-pt order): the one nearer the nose
    tip horizontally. Convention-free; verified on the reference head at
    yaw +/-40 and +/-60."""
    try:
        p = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if p.shape[0] < 3:
        return None
    return 0 if abs(p[0, 0] - p[2, 0]) < abs(p[1, 0] - p[2, 0]) else 1


class RouteStats:
    """Per-render counters, so a run can prove the routing ran (and how)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counts: Dict[str, int] = {}
        self.route_ns = 0

    def add(self, key: str, ns: int = 0) -> None:
        with self._lock:
            self.counts[key] = self.counts.get(key, 0) + 1
            self.route_ns += ns

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            n = self.counts.get("routed", 0)
            return {**self.counts, "mean_route_ms": round(self.route_ns / n / 1e6, 4) if n else None}


def route(portfolio: SourcePortfolio, target_kps: Any, target_bbox: Sequence[float],
          frame_idx: Optional[int] = None, lut: Optional[FrameLUT] = None) -> Optional[Route]:
    """The source vector for one target face. Pose comes from the LUT when it
    has this face (O(1)); otherwise from ``solve_pose_5pt`` on the keypoints
    (~11 us), the render's own appearance-pose solver."""
    entry, how = (lut.lookup(frame_idx, target_bbox) if lut is not None else (None, "miss"))
    if entry is not None:
        yaw, p_up, b = entry.yaw, entry.pitch_up, entry.bin
    else:
        from roop.face_util import solve_pose_5pt
        pose = solve_pose_5pt(target_kps)
        if pose is None:
            return Route(embedding=portfolio.fused, kind="fused", bin=None, yaw=float("nan"),
                         pitch_up=float("nan"), pose_from="live", far_eye=None)
        yaw, pitch, _roll = pose
        p_up, b, how = pitch_up(pitch), angle_bin(yaw, pitch), "live"
    far = far_eye_index(target_kps) if abs(yaw) > PROFILE_YAW_DEG else None
    if abs(yaw) > PROFILE_YAW_DEG:
        want = AngleBin.BIN_6_PROFILE_RIGHT if yaw > 0 else AngleBin.BIN_5_PROFILE_LEFT
        ref = portfolio.refs.get(want)
        if ref is not None and ref.embedding.size == portfolio.dim:
            e = _unit(PROFILE_WEIGHT * ref.embedding + (1.0 - PROFILE_WEIGHT) * portfolio.fused)
            if e is not None:
                return Route(embedding=e, kind="profile", bin=b, yaw=float(yaw), pitch_up=float(p_up),
                             pose_from=how, far_eye=far)
    return Route(embedding=portfolio.fused, kind="fused", bin=b, yaw=float(yaw), pitch_up=float(p_up),
                 pose_from=how, far_eye=far)


def damp_far_eye(enhanced: np.ndarray, pre: np.ndarray, kps_in_enh: Any, eye: int,
                 amount: float = FAR_EYE_DAMPING) -> np.ndarray:
    """Pull the enhanced crop ``amount`` of the way back to the pre-enhance crop
    inside a feathered ellipse on the foreshortened eye. ``pre`` is resized to
    ``enhanced`` if the enhancer upscaled; ``kps_in_enh`` are the 5 points in
    ``enhanced``'s pixel space."""
    import cv2
    if enhanced is None or pre is None or amount <= 0:
        return enhanced
    h, w = enhanced.shape[:2]
    if pre.shape[:2] != (h, w):
        pre = cv2.resize(pre, (w, h), interpolation=cv2.INTER_LINEAR)
    p = np.asarray(kps_in_enh, dtype=np.float32).reshape(-1, 2)
    iod = float(np.linalg.norm(p[1] - p[0])) or w * 0.2
    cx, cy = float(p[eye, 0]), float(p[eye, 1])
    rx = max(4.0, 0.45 * max(iod, 0.18 * w))
    ry = rx * 0.6
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (int(round(cx)), int(round(cy))), (int(round(rx)), int(round(ry))), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, rx * 0.35))
    peak = float(mask.max())
    if peak > 0:
        mask /= peak                   # the feather must not dilute the eye's own 25%
    a = (mask * float(amount))[:, :, None]
    out = enhanced.astype(np.float32) * (1.0 - a) + pre.astype(np.float32) * a
    return np.clip(out + 0.5, 0, 255).astype(enhanced.dtype)


__all__ = ["SourcePortfolio", "SourceRef", "FrameLUT", "LUTEntry", "Route", "RouteStats",
           "build_source_portfolio", "build_frame_lut", "route", "far_eye_index", "damp_far_eye",
           "PROFILE_YAW_DEG", "PROFILE_WEIGHT", "FAR_EYE_DAMPING"]
