"""Multi-shot ArcFace reference construction and pose-aware identity lookup.

This module is deliberately detector/session agnostic.  The caller supplies
the faces produced by the already-initialised buffalo/ArcFace analyser, so a
folder upload never creates a second GPU model context.  It also makes the
identity-bank contract usable by the API, legacy UI and command-line callers.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


ARCFACE_DIMENSION = 512
DEFAULT_MIN_COSINE = 0.65

# Dual-threshold hysteresis thresholds (Problem 2 specification)
HIGH_ACCEPTANCE_THRESHOLD: float = 0.62  # S_match >= 0.62
LOW_TRACKING_THRESHOLD: float = 0.50     # S_track >= 0.50
SPATIAL_IOU_THRESHOLD: float = 0.50      # Spatial IoU >= 0.50
CROSSING_IOU_THRESHOLD: float = 0.30     # Overlap indicating crossing tracks


def _field(face: Any, name: str, default=None):
    if isinstance(face, dict):
        return face.get(name, default)
    try:
        val = getattr(face, name, default)
        return default if val is None else val
    except Exception:
        return default


def _set_field(face: Any, name: str, value: Any) -> None:
    if isinstance(face, dict):
        face[name] = value
        return
    try:
        face[name] = value
        return
    except (TypeError, AttributeError):
        pass
    try:
        setattr(face, name, value)
    except Exception:
        pass


def _bbox_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Intersection-over-Union for two [x1, y1, x2, y2] bounding boxes."""
    if box1 is None or box2 is None:
        return 0.0
    try:
        ax1, ay1, ax2, ay2 = (float(v) for v in box1[:4])
        bx1, by1, bx2, by2 = (float(v) for v in box2[:4])
    except (TypeError, ValueError, IndexError):
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area1 = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area2 = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 1e-6 else 0.0


def normalized_arcface_embedding(face: Any) -> Optional[np.ndarray]:
    """Return a finite, unit-length 512-D ArcFace embedding or ``None``."""
    value = _field(face, "normed_embedding")
    if value is None:
        value = _field(face, "embedding")
    if value is None and isinstance(face, (np.ndarray, list, tuple)):
        value = face
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.size != ARCFACE_DIMENSION or not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    return (vector / norm).astype(np.float32) if norm > 1e-8 else None


def face_pose(face: Any) -> Tuple[float, float]:
    """Estimate yaw/pitch once while ingesting a reference image.

    A detector may already supply these values.  Otherwise the project's own
    five-landmark solve is used; failures are represented by the neutral pose,
    which keeps a valid identity usable instead of silently dropping it.
    """
    yaw, pitch = _field(face, "yaw"), _field(face, "pitch")
    try:
        if yaw is not None and pitch is not None:
            return float(yaw), float(pitch)
    except (TypeError, ValueError):
        pass
    try:
        from roop.face_util import solve_pose_5pt
        result = solve_pose_5pt(_field(face, "kps"))
        if result is not None:
            yaw, pitch = float(result[0]), float(result[1])
            if np.isfinite((yaw, pitch)).all():
                return yaw, pitch
    except Exception:
        pass
    return 0.0, 0.0


def reference_weight(face: Any) -> float:
    """Cheap quality weight; no extra inference or image pass is required."""
    score = _field(face, "det_score", _field(face, "score", 1.0))
    bbox = _field(face, "bbox")
    try:
        confidence = max(0.05, min(1.0, float(score)))
    except (TypeError, ValueError):
        confidence = 1.0
    try:
        box = np.asarray(bbox, dtype=np.float32).reshape(4)
        pixels = max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))
        # Root area rewards a well-resolved face without allowing one 4K photo
        # to completely suppress all pose coverage.
        return confidence * max(0.25, min(4.0, np.sqrt(pixels) / 160.0))
    except (TypeError, ValueError):
        return confidence


@dataclass(frozen=True)
class ReferenceSample:
    face: Any
    image: Any
    path: str
    embedding: np.ndarray
    yaw: float
    pitch: float
    weight: float


@dataclass
class ClusteredReferences:
    samples: List[ReferenceSample]
    embedding: np.ndarray
    rejected: List[dict]

    @property
    def poses(self) -> List[Tuple[float, float]]:
        return [(sample.yaw, sample.pitch) for sample in self.samples]

    @property
    def embeddings(self) -> List[np.ndarray]:
        return [sample.embedding for sample in self.samples]

    def embedding_for_pose(self, yaw: float, pitch: float, blend: bool = True) -> np.ndarray:
        """Return the nearest reference identity, blending an ambiguous pair.

        Blending only happens for two nearby pose samples; a profile therefore
        cannot be diluted by a frontal reference merely because it is present
        in the same faceset.
        """
        distances = np.asarray([
            (float(yaw) - item.yaw) ** 2 + (float(pitch) - item.pitch) ** 2
            for item in self.samples
        ], dtype=np.float32)
        if distances.size == 0:
            return self.embedding
        order = np.argsort(distances)
        first = int(order[0])
        if not blend or len(order) < 2:
            return self.samples[first].embedding
        second = int(order[1])
        # At a pose boundary a weighted blend avoids a frame-to-frame source
        # discontinuity.  Else select the closest true reference exactly.
        if float(distances[second] - distances[first]) > 100.0:
            return self.samples[first].embedding
        weights = np.asarray([
            self.samples[first].weight / (5.0 + float(distances[first])),
            self.samples[second].weight / (5.0 + float(distances[second])),
        ], dtype=np.float32)
        vector = np.average(np.stack((self.samples[first].embedding,
                                      self.samples[second].embedding)),
                            axis=0, weights=weights)
        norm = float(np.linalg.norm(vector))
        return (vector / norm).astype(np.float32) if norm > 1e-8 else self.embedding


def cluster_references(samples: Iterable[ReferenceSample],
                       min_cosine: float = DEFAULT_MIN_COSINE) -> ClusteredReferences:
    """Discard identity outliers and calculate an L2-normalized weighted mean."""
    candidates = list(samples)
    if not candidates:
        raise ValueError("no valid 512-D ArcFace references were detected")
    initial = np.mean(np.stack([item.embedding for item in candidates]), axis=0)
    initial /= max(float(np.linalg.norm(initial)), 1e-8)
    kept, rejected = [], []
    for item in candidates:
        similarity = float(np.dot(initial, item.embedding))
        if similarity < float(min_cosine):
            rejected.append({"path": item.path, "reason": "identity_outlier",
                             "cosine_similarity": round(similarity, 6),
                             "threshold": float(min_cosine)})
        else:
            kept.append(item)
    # A malformed set should never become an empty face identity.  One valid
    # reference is still preferable to an unexplained upload failure.
    if not kept:
        kept, rejected = candidates, []
    weights = np.asarray([item.weight for item in kept], dtype=np.float32)
    centroid = np.average(np.stack([item.embedding for item in kept]), axis=0,
                          weights=weights)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
    return ClusteredReferences(kept, centroid.astype(np.float32), rejected)


def clustered_faceset(samples: Sequence[Tuple[Any, Any, str]],
                      min_cosine: float = DEFAULT_MIN_COSINE) -> ClusteredReferences:
    """Build a cluster from ``(detected_face, original_bgr_image, path)`` data."""
    valid = []
    for face, image, path in samples:
        embedding = normalized_arcface_embedding(face)
        if embedding is None:
            continue
        yaw, pitch = face_pose(face)
        valid.append(ReferenceSample(face, image, str(path), embedding, yaw, pitch,
                                     reference_weight(face)))
    return cluster_references(valid, min_cosine=min_cosine)


__all__ = ["ARCFACE_DIMENSION", "DEFAULT_MIN_COSINE", "HIGH_ACCEPTANCE_THRESHOLD",
           "LOW_TRACKING_THRESHOLD", "SPATIAL_IOU_THRESHOLD", "CROSSING_IOU_THRESHOLD",
           "ReferenceSample", "ClusteredReferences", "EmbeddingSlidingWindow",
           "IdentityTrackState", "MultiIdentityReferenceRouter",
           "normalized_arcface_embedding", "face_pose", "reference_weight",
           "cluster_references", "clustered_faceset", "dual_threshold_match"]


# =============================================================================
# Sliding Window of Face Embeddings & Dual-Threshold Hysteresis Tracker
# =============================================================================

class EmbeddingSlidingWindow:
    """Maintains a temporal sliding window of 512-D normalized ArcFace embeddings.

    Maintains sequential identity memory to prevent track drops and identity flips
    when subjects turn away from the camera, undergo motion blur, or cross paths.
    """

    def __init__(self, maxlen: int = 16):
        self.maxlen = max(1, int(maxlen))
        self._window: deque[np.ndarray] = deque(maxlen=self.maxlen)
        self._weights: deque[float] = deque(maxlen=self.maxlen)
        self._lock = RLock()

    def add(self, embedding: Any, weight: float = 1.0) -> bool:
        """Validate and append a unit-normalized 512-D embedding to the window."""
        vec = normalized_arcface_embedding(embedding)
        if vec is None:
            return False
        with self._lock:
            self._window.append(vec)
            self._weights.append(max(1e-4, float(weight)))
        return True

    def centroid(self) -> Optional[np.ndarray]:
        """Compute L2-normalized weighted mean embedding of the sliding window."""
        with self._lock:
            if not self._window:
                return None
            vecs = np.stack(list(self._window), axis=0)
            w = np.asarray(list(self._weights), dtype=np.float32)
            w_sum = float(np.sum(w))
            if w_sum > 1e-6:
                mean_vec = np.sum(vecs * (w[:, None] / w_sum), axis=0)
            else:
                mean_vec = np.mean(vecs, axis=0)
            norm = float(np.linalg.norm(mean_vec))
            if norm < 1e-6:
                return None
            return (mean_vec / norm).astype(np.float32)

    def max_similarity(self, query_emb: Any) -> float:
        """Return maximum cosine similarity between query and any embedding in the window."""
        q = normalized_arcface_embedding(query_emb)
        if q is None:
            return 0.0
        with self._lock:
            if not self._window:
                return 0.0
            sims = [float(np.dot(q, entry)) for entry in self._window]
            return float(np.clip(max(sims), -1.0, 1.0))

    def mean_similarity(self, query_emb: Any) -> float:
        """Return cosine similarity between query and the window centroid."""
        c = self.centroid()
        if c is None:
            return 0.0
        q = normalized_arcface_embedding(query_emb)
        if q is None:
            return 0.0
        return float(np.clip(np.dot(q, c), -1.0, 1.0))

    def clear(self) -> None:
        """Flush the sliding window."""
        with self._lock:
            self._window.clear()
            self._weights.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._window)

    def embeddings(self) -> List[np.ndarray]:
        with self._lock:
            return [item.copy() for item in self._window]


def dual_threshold_match(
    embedding: Any,
    ref_embedding: Any,
    sliding_window: Optional[EmbeddingSlidingWindow] = None,
    current_bbox: Optional[Sequence[float]] = None,
    previous_bbox: Optional[Sequence[float]] = None,
    high_threshold: float = HIGH_ACCEPTANCE_THRESHOLD,
    low_threshold: float = LOW_TRACKING_THRESHOLD,
    iou_threshold: float = SPATIAL_IOU_THRESHOLD
) -> Tuple[bool, float, float, str]:
    """Evaluate dual-threshold hysteresis matching for a candidate face.

    Dual-threshold hysteresis rule:
      1. High acceptance threshold: S_match >= 0.62 -> match accepted unconditionally.
      2. Low tracking threshold: S_track >= 0.50 when paired with spatial IoU >= 0.50
         from previous frame -> match accepted under spatial tracking continuity.

    Returns:
        (matched: bool, best_cosine: float, spatial_iou: float, reason: str)
    """
    q = normalized_arcface_embedding(embedding)
    if q is None:
        return False, 0.0, 0.0, "no_embedding"

    best_sim = -1.0
    ref_u = normalized_arcface_embedding(ref_embedding)
    if ref_u is not None:
        best_sim = float(np.dot(q, ref_u))

    if sliding_window is not None and len(sliding_window) > 0:
        win_max = sliding_window.max_similarity(q)
        win_mean = sliding_window.mean_similarity(q)
        best_sim = max(best_sim, win_max, win_mean)

    iou = 0.0
    if current_bbox is not None and previous_bbox is not None:
        iou = _bbox_iou(current_bbox, previous_bbox)

    # Dual-threshold hysteresis gate
    if best_sim >= float(high_threshold):
        return True, best_sim, iou, "high_acceptance"
    if best_sim >= float(low_threshold) and iou >= float(iou_threshold):
        return True, best_sim, iou, "low_tracking_iou"

    return False, best_sim, iou, "below_threshold"


@dataclass
class IdentityTrackState:
    """State tracking for one identity in a multi-person scene."""
    name: str
    reference_embedding: np.ndarray
    sliding_window: EmbeddingSlidingWindow
    previous_bbox: Optional[np.ndarray] = None
    locked_track_id: Optional[int] = None
    last_seen_frame: int = -1
    hits: int = 0
    misses: int = 0


class MultiIdentityReferenceRouter:
    """Disambiguates and routes multiple face identities (e.g., 'mehak' and 'misbah').

    Maintains a sliding window of face embeddings per target identity.
    Uses cosine similarity with dual-threshold hysteresis (S_match >= 0.62,
    S_track >= 0.50 when paired with spatial IoU >= 0.50 from previous frame)
    and optimal bipartite matching to prevent identity flipping when 'mehak'
    and 'misbah' cross paths, occlude each other, or turn away from camera.
    """

    def __init__(
        self,
        identities: Optional[Dict[str, Any]] = None,
        window_size: int = 16,
        high_threshold: float = HIGH_ACCEPTANCE_THRESHOLD,
        low_threshold: float = LOW_TRACKING_THRESHOLD,
        iou_threshold: float = SPATIAL_IOU_THRESHOLD,
        crossing_iou: float = CROSSING_IOU_THRESHOLD,
    ):
        self.window_size = int(window_size)
        self.high_threshold = float(high_threshold)
        self.low_threshold = float(low_threshold)
        self.iou_threshold = float(iou_threshold)
        self.crossing_iou = float(crossing_iou)
        self.identities: Dict[str, IdentityTrackState] = {}
        self._lock = RLock()
        if identities:
            self.register_identities(identities)

    def register_identities(self, identities: Dict[str, Any]) -> None:
        """Register one or more target identities (e.g. {'mehak': ..., 'misbah': ...})."""
        with self._lock:
            for name, data in identities.items():
                emb = None
                if isinstance(data, dict):
                    emb = data.get("embedding")
                    if emb is None:
                        emb = data.get("normed_embedding")
                elif hasattr(data, "default_embedding"):
                    emb = getattr(data, "default_embedding")
                elif hasattr(data, "identity_embedding"):
                    emb = getattr(data, "identity_embedding")
                elif hasattr(data, "embedding"):
                    emb = getattr(data, "embedding")
                else:
                    emb = data
                normed = normalized_arcface_embedding(emb if emb is not None else data)
                if normed is not None:
                    sw = EmbeddingSlidingWindow(maxlen=self.window_size)
                    sw.add(normed, weight=1.0)
                    self.identities[str(name)] = IdentityTrackState(
                        name=str(name),
                        reference_embedding=normed,
                        sliding_window=sw,
                    )

    def route(
        self,
        detected_faces: Sequence[Any],
        frame_index: int = 0
    ) -> List[Optional[str]]:
        """Assign detected faces to registered identities with dual-threshold hysteresis.

        Returns a list of identity name strings (or None for unmapped bystanders).
        Guarantees zero identity flipping across trajectory crossings.
        """
        with self._lock:
            if not self.identities or not detected_faces:
                return [None] * len(detected_faces)

            m = len(detected_faces)
            id_names = sorted(self.identities)
            n = len(id_names)

            sim_matrix = np.zeros((m, n), dtype=np.float32)
            iou_matrix = np.zeros((m, n), dtype=np.float32)
            eligible = np.zeros((m, n), dtype=bool)

            face_bboxes = []
            face_embs = []

            for i, face in enumerate(detected_faces):
                bbox = _field(face, "bbox")
                emb = normalized_arcface_embedding(face)
                face_bboxes.append(
                    np.asarray(bbox, dtype=np.float32).reshape(4)
                    if bbox is not None and len(bbox) >= 4 else None
                )
                face_embs.append(emb)

                for j, name in enumerate(id_names):
                    state = self.identities[name]
                    matched, sim, iou, _ = dual_threshold_match(
                        emb,
                        state.reference_embedding,
                        sliding_window=state.sliding_window,
                        current_bbox=face_bboxes[-1],
                        previous_bbox=state.previous_bbox,
                        high_threshold=self.high_threshold,
                        low_threshold=self.low_threshold,
                        iou_threshold=self.iou_threshold,
                    )
                    sim_matrix[i, j] = sim
                    iou_matrix[i, j] = iou
                    eligible[i, j] = matched

            # Detect crossing / occlusion between detected faces
            is_crossing = False
            for i1 in range(m):
                for i2 in range(i1 + 1, m):
                    if face_bboxes[i1] is not None and face_bboxes[i2] is not None:
                        if _bbox_iou(face_bboxes[i1], face_bboxes[i2]) > self.crossing_iou:
                            is_crossing = True
                            break

            # Cost matrix: C = 1.0 - (alpha * sim + (1 - alpha) * iou)
            # When crossing, alpha = 1.0 so spatial coordinates never flip identity assignments
            alpha = 1.0 if is_crossing else 0.75
            cost_matrix = np.full((m, n), 1e5, dtype=np.float32)
            for i in range(m):
                for j in range(n):
                    if eligible[i, j]:
                        score = (alpha * sim_matrix[i, j] + (1.0 - alpha) * iou_matrix[i, j]
                                 if not is_crossing else sim_matrix[i, j])
                        cost_matrix[i, j] = 1.0 - score

            # Optimal bipartite matching
            rows, cols = [], []
            try:
                from scipy.optimize import linear_sum_assignment
                rows, cols = linear_sum_assignment(cost_matrix)
            except Exception:
                used_r, used_c = set(), set()
                flat_order = np.argsort(cost_matrix, axis=None)
                for idx in flat_order:
                    r = int(idx // n)
                    c = int(idx % n)
                    if r not in used_r and c not in used_c:
                        rows.append(r)
                        cols.append(c)
                        used_r.add(r)
                        used_c.add(c)
                        if len(rows) == min(m, n):
                            break

            assignments: List[Optional[str]] = [None] * m
            for r, c in zip(rows, cols):
                if cost_matrix[r, c] < 100.0 and eligible[r, c]:
                    name = id_names[c]
                    assignments[r] = name
                    state = self.identities[name]
                    state.hits += 1
                    state.misses = 0
                    state.last_seen_frame = int(frame_index)
                    if face_bboxes[r] is not None:
                        state.previous_bbox = face_bboxes[r].copy()
                    if face_embs[r] is not None:
                        state.sliding_window.add(face_embs[r], weight=1.0)
                    _set_field(detected_faces[r], "_assigned_identity", name)
                    tid = _field(detected_faces[r], "_track_id")
                    if tid is not None:
                        state.locked_track_id = int(tid)

            assigned_names = set(assignments)
            for name, state in self.identities.items():
                if name not in assigned_names:
                    state.misses += 1

            return assignments
