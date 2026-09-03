"""Hungarian Bipartite Identity Matching & Hysteresis State Machine.

Prevents identity flipping and track swapping in multi-person videos (e.g.,
'double' dataset containing distinct identities: 'mehak' and 'misbah').

MATHEMATICAL SPECIFICATION:
1. Global Cost Matrix Formulation:
   - For M target faces in current frame and N tracked identities:
     Cost matrix C of shape (M, N), where each entry is:
     C_{i,j} = alpha * (1.0 - CosineSimilarity(Emb_i, RefEmb_j)) +
               (1.0 - alpha) * (1.0 - GeneralizedIoU(Box_i, PredictedBox_j))
   - alpha = 0.75 by default to prioritize deep ArcFace identity embeddings
     over pure spatial position.

2. Optimal Bipartite Matching:
   - Solved via the Jonker-Volgenant algorithm (scipy.optimize.linear_sum_assignment).
   - Gating threshold: if C_{i,j} > 0.45 (low similarity and poor overlap),
     assignment is rejected as an unmapped background face rather than forcing
     a false positive swap.

3. Hysteresis State Machine:
   - When tracks cross (IoU > 0.3 between two tracked faces), lock identity
     assignments and disable spatial weight (alpha = 1.0) until bounding boxes
     separate by at least 1.5x average face width.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math
from threading import RLock
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


def _linear_sum_assignment_fallback(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pure-Python fallback for scipy.optimize.linear_sum_assignment."""
    cost = np.asarray(cost_matrix, dtype=np.float64)
    m, n = cost.shape
    if m == 0 or n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if m <= 8 and n <= 8:
        import itertools
        if m <= n:
            best_cost = float('inf')
            best_cols = tuple(range(m))
            for cols in itertools.permutations(range(n), m):
                c = sum(cost[r, c_idx] for r, c_idx in enumerate(cols))
                if c < best_cost:
                    best_cost = c
                    best_cols = cols
            return np.arange(m, dtype=np.int64), np.asarray(best_cols, dtype=np.int64)
        else:
            best_cost = float('inf')
            best_rows = tuple(range(n))
            for rows in itertools.permutations(range(m), n):
                c = sum(cost[r_idx, c] for c, r_idx in enumerate(rows))
                if c < best_cost:
                    best_cost = c
                    best_rows = rows
            return np.asarray(best_rows, dtype=np.int64), np.arange(n, dtype=np.int64)
    rows, cols = [], []
    used_rows, used_cols = set(), set()
    flat_indices = np.argsort(cost, axis=None)
    for idx in flat_indices:
        r = int(idx // n)
        c = int(idx % n)
        if r not in used_rows and c not in used_cols:
            rows.append(r)
            cols.append(c)
            used_rows.add(r)
            used_cols.add(c)
            if len(rows) == min(m, n):
                break
    order = np.argsort(rows)
    return np.asarray(rows, dtype=np.int64)[order], np.asarray(cols, dtype=np.int64)[order]


if linear_sum_assignment is None:
    linear_sum_assignment = _linear_sum_assignment_fallback


# Default hyper-parameters from mathematical specification
DEFAULT_ALPHA: float = 0.75
CROSSING_ALPHA: float = 1.0
DEFAULT_GATING_THRESHOLD: float = 0.45
CROSSING_IOU_THRESHOLD: float = 0.3
SEPARATION_MULTIPLIER: float = 1.5


# =============================================================================
# Mathematical Primitives & Geometry
# =============================================================================

def cosine_similarity(emb1: Optional[Sequence[float]],
                      emb2: Optional[Sequence[float]]) -> float:
    """Compute bounded cosine similarity in [-1.0, 1.0].
    
    Missing, empty, or zero-norm embeddings return 0.0.
    """
    if emb1 is None or emb2 is None:
        return 0.0
    a = np.asarray(emb1, dtype=np.float32).reshape(-1)
    b = np.asarray(emb2, dtype=np.float32).reshape(-1)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 1e-6 or norm_b <= 1e-6:
        return 0.0
    cos_val = float(np.dot(a, b) / (norm_a * norm_b))
    return float(np.clip(cos_val, -1.0, 1.0))


def cosine_distance(emb1: Optional[Sequence[float]],
                    emb2: Optional[Sequence[float]]) -> float:
    """Compute cosine distance in [0.0, 2.0]: 1.0 - CosineSimilarity."""
    return 1.0 - cosine_similarity(emb1, emb2)


def bbox_iou(box1: Optional[Sequence[float]],
             box2: Optional[Sequence[float]]) -> float:
    """Compute standard Intersection-over-Union (IoU) for [x1, y1, x2, y2] boxes."""
    if box1 is None or box2 is None or len(box1) < 4 or len(box2) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in box1[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box2[:4]]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 1e-6 else 0.0


def generalized_iou(box1: Optional[Sequence[float]],
                    box2: Optional[Sequence[float]]) -> float:
    """Compute Generalized Intersection-over-Union (GIoU) in [-1.0, 1.0].
    
    GIoU = IoU - (Area(C) - Area(Union)) / Area(C)
    where C is the smallest enclosing convex box.
    """
    if box1 is None or box2 is None or len(box1) < 4 or len(box2) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in box1[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box2[:4]]

    # Ensure min < max
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    by1, by2 = min(by1, by2), max(by1, by2)

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if area_a <= 1e-6 or area_b <= 1e-6:
        return 0.0

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih

    union = area_a + area_b - intersection
    iou = intersection / union if union > 1e-6 else 0.0

    # Smallest enclosing box C
    cx1 = min(ax1, bx1)
    cy1 = min(ay1, by1)
    cx2 = max(ax2, bx2)
    cy2 = max(ay2, by2)
    area_c = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)

    if area_c <= 1e-6:
        return float(iou)

    giou = iou - (area_c - union) / area_c
    return float(np.clip(giou, -1.0, 1.0))


def giou_distance(box1: Optional[Sequence[float]],
                  box2: Optional[Sequence[float]]) -> float:
    """Compute GIoU distance in [0.0, 2.0]: 1.0 - GeneralizedIoU."""
    return 1.0 - generalized_iou(box1, box2)


def average_face_width(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Compute the average width of two face bounding boxes."""
    w1 = max(1e-3, abs(float(box1[2]) - float(box1[0])))
    w2 = max(1e-3, abs(float(box2[2]) - float(box2[0])))
    return (w1 + w2) * 0.5


def bbox_separation_distance(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Compute Euclidean boundary-to-boundary distance between two bounding boxes.
    
    Returns 0.0 if the boxes intersect or touch.
    """
    ax1, ay1, ax2, ay2 = [float(v) for v in box1[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box2[:4]]
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    by1, by2 = min(by1, by2), max(by1, by2)

    dx = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    dy = max(0.0, max(ay1, by1) - min(ay2, by2))
    return float(math.hypot(dx, dy))


def bbox_center_distance(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Compute Euclidean center-to-center distance between two bounding boxes."""
    cax = (float(box1[0]) + float(box1[2])) * 0.5
    cay = (float(box1[1]) + float(box1[3])) * 0.5
    cbx = (float(box2[0]) + float(box2[2])) * 0.5
    cby = (float(box2[1]) + float(box2[3])) * 0.5
    return float(math.hypot(cax - cbx, cay - cby))


def are_boxes_separated(box1: Sequence[float],
                       box2: Sequence[float],
                       multiplier: float = SEPARATION_MULTIPLIER) -> bool:
    """Check if two bounding boxes have separated by at least multiplier * average face width.
    
    Condition is satisfied if:
    1. Edge-to-edge separation distance >= multiplier * average_face_width, OR
    2. Center-to-center distance >= multiplier * average_face_width with IoU == 0.
    """
    avg_w = average_face_width(box1, box2)
    threshold = multiplier * avg_w
    edge_sep = bbox_separation_distance(box1, box2)
    if edge_sep >= threshold:
        return True
    center_dist = bbox_center_distance(box1, box2)
    if center_dist >= threshold and bbox_iou(box1, box2) <= 1e-4:
        return True
    return False


# =============================================================================
# Helper Utilities for Face Objects & Embeddings
# =============================================================================

def extract_face_bbox(face: Any) -> Optional[np.ndarray]:
    """Uniformly read bounding box [x1, y1, x2, y2] from InsightFace, dict, or object."""
    if face is None:
        return None
    if isinstance(face, dict):
        box = face.get('bbox')
    else:
        box = getattr(face, 'bbox', None)
        if box is None and hasattr(face, 'get'):
            box = face.get('bbox')
    if box is None:
        return None
    try:
        arr = np.asarray(box, dtype=np.float32).reshape(4)
        return arr if np.all(np.isfinite(arr)) else None
    except (TypeError, ValueError):
        return None


def extract_face_embedding(face: Any) -> Optional[np.ndarray]:
    """Uniformly extract normalized ArcFace embedding (512-D) from any Face, FaceSet, or array."""
    if face is None:
        return None
    if isinstance(face, np.ndarray):
        arr = face.reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(arr))
        return (arr / norm) if norm > 1e-6 and np.all(np.isfinite(arr)) else None

    # Check FaceSet attributes (V2 and legacy)
    for attr in ('identity_embedding', 'default_embedding', 'normalized_embedding'):
        emb = getattr(face, attr, None)
        if emb is not None:
            arr = np.asarray(emb, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(arr))
            if norm > 1e-6 and np.all(np.isfinite(arr)):
                return (arr / norm).astype(np.float32)

    # Check Face / dict attributes
    for attr in ('normed_embedding', 'embedding'):
        emb = getattr(face, attr, None)
        if emb is None and isinstance(face, dict):
            emb = face.get(attr)
        if emb is not None:
            arr = np.asarray(emb, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(arr))
            if norm > 1e-6 and np.all(np.isfinite(arr)):
                return (arr / norm).astype(np.float32)

    # Check FaceSet.faces list
    faces_list = getattr(face, 'faces', None)
    if faces_list and len(faces_list) > 0:
        return extract_face_embedding(faces_list[0])

    return None


def set_face_meta(face: Any, key: str, value: Any) -> None:
    """Safely set metadata on dict or object."""
    if face is None:
        return
    if isinstance(face, dict):
        face[key] = value
        return
    try:
        face[key] = value
        return
    except (TypeError, AttributeError):
        pass
    try:
        setattr(face, key, value)
    except Exception:
        pass


# =============================================================================
# Tracked Identity Data Structures
# =============================================================================

@dataclass
class TrackedIdentity:
    """One reference identity being tracked across frames."""
    identity_id: Any                                      # e.g., 'mehak', 'misbah', 0, 1
    name: str                                            # String display name
    reference_embedding: np.ndarray                      # 512-D unit ArcFace embedding
    source_face: Any = None                              # Target source face/FaceSet for swapping
    current_bbox: Optional[np.ndarray] = None            # Last observed [x1, y1, x2, y2]
    predicted_bbox: Optional[np.ndarray] = None          # Predicted [x1, y1, x2, y2]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    last_seen_frame: int = -1
    hits: int = 0
    misses: int = 0
    embedding_history: deque = field(default_factory=lambda: deque(maxlen=16))
    locked_track_id: Optional[int] = None

    def update_observation(self, bbox: np.ndarray,
                           embedding: Optional[np.ndarray],
                           frame_index: int,
                           track_id: Optional[int] = None) -> None:
        """Update identity state from a confirmed assignment."""
        bbox = np.asarray(bbox, dtype=np.float32).reshape(4)
        if self.current_bbox is not None:
            # Estimate simple box velocity: [dx1, dy1, dx2, dy2]
            self.velocity = 0.7 * self.velocity + 0.3 * (bbox - self.current_bbox)
        self.current_bbox = bbox.copy()
        # Linear velocity prediction for next frame
        self.predicted_bbox = (self.current_bbox + self.velocity).astype(np.float32)
        self.last_seen_frame = int(frame_index)
        self.hits += 1
        self.misses = 0
        if track_id is not None:
            self.locked_track_id = int(track_id)
        if embedding is not None:
            emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(emb))
            if norm > 1e-6:
                self.embedding_history.append(emb / norm)


# =============================================================================
# Global Cost Matrix & Bipartite Matching
# =============================================================================

def compute_cost_matrix(target_faces: Sequence[Any],
                        reference_identities: Sequence[TrackedIdentity],
                        alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    """Construct global cost matrix C of shape (M, N).
    
    C_{i,j} = alpha * (1.0 - CosineSimilarity(Emb_i, RefEmb_j)) +
              (1.0 - alpha) * (1.0 - GeneralizedIoU(Box_i, PredictedBox_j))
    
    When alpha == 1.0, the spatial term is completely disabled.
    """
    m = len(target_faces)
    n = len(reference_identities)
    cost = np.empty((m, n), dtype=np.float32)

    for i, target in enumerate(target_faces):
        emb_i = extract_face_embedding(target)
        box_i = extract_face_bbox(target)

        for j, ref in enumerate(reference_identities):
            # Identity term: 1 - CosineSimilarity
            sim = cosine_similarity(emb_i, ref.reference_embedding)
            id_cost = 1.0 - sim

            # Spatial term: 1 - GeneralizedIoU
            ref_box = ref.predicted_bbox if ref.predicted_bbox is not None else ref.current_bbox
            if box_i is not None and ref_box is not None:
                spatial_cost = giou_distance(box_i, ref_box)
            else:
                spatial_cost = 1.0

            if alpha >= 1.0:
                cost[i, j] = id_cost
            elif alpha <= 0.0:
                cost[i, j] = spatial_cost
            else:
                cost[i, j] = alpha * id_cost + (1.0 - alpha) * spatial_cost

    return cost


def match_bipartite(cost_matrix: np.ndarray,
                    gating_threshold: float = DEFAULT_GATING_THRESHOLD
                    ) -> List[Tuple[int, int, float]]:
    """Solve optimal bipartite matching with gating.
    
    Uses Jonker-Volgenant algorithm (scipy.optimize.linear_sum_assignment).
    Rejects any assignment where C_{i,j} > gating_threshold.
    
    Returns list of (target_index, reference_index, cost).
    """
    if cost_matrix.size == 0 or linear_sum_assignment is None:
        return []

    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matches: List[Tuple[int, int, float]] = []

    for r, c in zip(row_indices, col_indices):
        val = float(cost_matrix[r, c])
        if np.isfinite(val) and val <= gating_threshold:
            matches.append((int(r), int(c), val))

    return matches


# =============================================================================
# Hysteresis State Machine & Identity Manager
# =============================================================================

class HysteresisState(Enum):
    NORMAL = "normal"
    CROSSING_LOCKED = "crossing_locked"


class IdentityManager:
    """Thread-safe Hungarian Bipartite Identity Manager with Hysteresis State Machine."""

    def __init__(self,
                 alpha: float = DEFAULT_ALPHA,
                 crossing_alpha: float = CROSSING_ALPHA,
                 gating_threshold: float = DEFAULT_GATING_THRESHOLD,
                 crossing_iou_threshold: float = CROSSING_IOU_THRESHOLD,
                 separation_multiplier: float = SEPARATION_MULTIPLIER):
        self.alpha = float(alpha)
        self.crossing_alpha = float(crossing_alpha)
        self.gating_threshold = float(gating_threshold)
        self.crossing_iou_threshold = float(crossing_iou_threshold)
        self.separation_multiplier = float(separation_multiplier)

        self._lock = RLock()
        self.identities: List[TrackedIdentity] = []
        self.state = HysteresisState.NORMAL
        self.active_crossing_pairs: Set[Tuple[int, int]] = set()
        self._last_frame_index: Optional[int] = None
        self.stats = {
            'frames': 0,
            'crossings_entered': 0,
            'crossings_cleared': 0,
            'assignments_made': 0,
            'gating_rejections': 0,
        }

    def reset(self) -> None:
        """Reset temporal state between video clips or harness runs."""
        with self._lock:
            self.identities.clear()
            self.state = HysteresisState.NORMAL
            self.active_crossing_pairs.clear()
            self._last_frame_index = None
            for k in self.stats:
                self.stats[k] = 0

    def bind_facesets(self,
                      source_faces: Union[Any, Sequence[Any], Dict[str, Any]]
                      ) -> List[TrackedIdentity]:
        """Configure tracked reference identities from single or double facesets.
        
        Accepts:
        - Single Face / FaceSet (e.g., 'mehak')
        - Sequence of Face / FaceSet (e.g., ['mehak', 'misbah'])
        - Dictionary mapping identity names to Face / FaceSet
        """
        with self._lock:
            self.identities.clear()
            self.state = HysteresisState.NORMAL
            self.active_crossing_pairs.clear()

            if source_faces is None:
                return []

            def _get_name(src_item: Any, default_idx: int) -> str:
                if isinstance(src_item, dict):
                    n = src_item.get('name')
                else:
                    n = getattr(src_item, 'name', None)
                return str(n) if n is not None else f"identity_{default_idx}"

            if isinstance(source_faces, dict):
                if ('embedding' in source_faces or 'normed_embedding' in source_faces
                        or 'bbox' in source_faces or 'faces' in source_faces):
                    items = [(_get_name(source_faces, 0), source_faces)]
                else:
                    items = list(source_faces.items())
            elif isinstance(source_faces, (list, tuple)):
                items = [(_get_name(item, idx), item) for idx, item in enumerate(source_faces)]
            else:
                items = [(_get_name(source_faces, 0), source_faces)]

            for idx, (name, src) in enumerate(items):
                emb = extract_face_embedding(src)
                if emb is None:
                    # Synthetic fallback embedding if None (e.g., in headless test double)
                    emb = np.zeros(512, dtype=np.float32)
                    emb[idx % 512] = 1.0

                identity = TrackedIdentity(
                    identity_id=name if isinstance(name, (str, int)) else idx,
                    name=str(name),
                    reference_embedding=emb,
                    source_face=src,
                )
                self.identities.append(identity)

            return list(self.identities)

    def _check_and_update_hysteresis(self, current_boxes: Dict[int, np.ndarray]) -> float:
        """Run hysteresis state machine over active tracked identities.
        
        Returns effective alpha (DEFAULT_ALPHA or CROSSING_ALPHA).
        """
        n_id = len(self.identities)
        if n_id < 2:
            self.state = HysteresisState.NORMAL
            self.active_crossing_pairs.clear()
            return self.alpha

        # Check existing crossing pairs for separation
        if self.state == HysteresisState.CROSSING_LOCKED:
            cleared_pairs = set()
            for pair in self.active_crossing_pairs:
                i, j = pair
                box_i = current_boxes.get(i)
                box_j = current_boxes.get(j)
                if box_i is None or box_j is None:
                    # If one face is lost, use last known
                    box_i = self.identities[i].current_bbox if box_i is None else box_i
                    box_j = self.identities[j].current_bbox if box_j is None else box_j

                if box_i is not None and box_j is not None:
                    if are_boxes_separated(box_i, box_j, self.separation_multiplier):
                        cleared_pairs.add(pair)

            self.active_crossing_pairs -= cleared_pairs
            if not self.active_crossing_pairs:
                self.state = HysteresisState.NORMAL
                self.stats['crossings_cleared'] += 1

        # Check for new crossing events: IoU > CROSSING_IOU_THRESHOLD
        new_crossings = set()
        for i in range(n_id):
            for j in range(i + 1, n_id):
                box_i = current_boxes.get(i, self.identities[i].current_bbox)
                box_j = current_boxes.get(j, self.identities[j].current_bbox)
                if box_i is not None and box_j is not None:
                    iou = bbox_iou(box_i, box_j)
                    if iou > self.crossing_iou_threshold:
                        new_crossings.add((i, j))

        if new_crossings:
            if self.state != HysteresisState.CROSSING_LOCKED:
                self.stats['crossings_entered'] += 1
            self.state = HysteresisState.CROSSING_LOCKED
            self.active_crossing_pairs.update(new_crossings)

        return self.crossing_alpha if self.state == HysteresisState.CROSSING_LOCKED else self.alpha

    def assign(self,
               target_faces: Sequence[Any],
               frame_index: Optional[int] = None
               ) -> List[Optional[Tuple[TrackedIdentity, float]]]:
        """Perform optimal bipartite matching between detected target faces and tracked identities.
        
        Returns a list of length len(target_faces):
        - Tuple[TrackedIdentity, cost] for matched faces
        - None for rejected/unmapped background faces
        """
        with self._lock:
            self.stats['frames'] += 1
            if frame_index is None:
                frame_index = 0 if self._last_frame_index is None else self._last_frame_index + 1
            self._last_frame_index = int(frame_index)

            m = len(target_faces)
            n = len(self.identities)
            assignments: List[Optional[Tuple[TrackedIdentity, float]]] = [None] * m

            if m == 0 or n == 0:
                return assignments

            # Collect active bounding boxes for hysteresis evaluation
            active_boxes: Dict[int, np.ndarray] = {}
            for i, target in enumerate(target_faces):
                t_box = extract_face_bbox(target)
                t_id = None
                if isinstance(target, dict):
                    t_id = target.get('_track_id')
                else:
                    t_id = getattr(target, '_track_id', None)
                if t_box is not None and t_id is not None:
                    # Map to identity if track already locked
                    for j, ref in enumerate(self.identities):
                        if ref.locked_track_id == t_id:
                            active_boxes[j] = t_box

            # Evaluate hysteresis state and determine alpha
            effective_alpha = self._check_and_update_hysteresis(active_boxes)

            # Construct global cost matrix
            cost_matrix = compute_cost_matrix(target_faces, self.identities, alpha=effective_alpha)

            # Solve optimal bipartite matching with gating
            matches = match_bipartite(cost_matrix, gating_threshold=self.gating_threshold)

            matched_targets = set()
            matched_identities = set()

            for target_idx, ref_idx, cost in matches:
                target_face = target_faces[target_idx]
                ref_identity = self.identities[ref_idx]

                # If in crossing locked state and tracklet is already bound to another identity, enforce lock
                target_track_id = None
                if isinstance(target_face, dict):
                    target_track_id = target_face.get('_track_id')
                else:
                    target_track_id = getattr(target_face, '_track_id', None)

                if (self.state == HysteresisState.CROSSING_LOCKED
                        and ref_identity.locked_track_id is not None
                        and target_track_id is not None
                        and ref_identity.locked_track_id != target_track_id):
                    # Check if another identity owns this track
                    owner = next((id_obj for id_obj in self.identities
                                  if id_obj.locked_track_id == target_track_id), None)
                    if owner is not None and owner.identity_id != ref_identity.identity_id:
                        # Lock prevents flipping
                        continue

                assignments[target_idx] = (ref_identity, cost)
                matched_targets.add(target_idx)
                matched_identities.add(ref_idx)
                self.stats['assignments_made'] += 1

                # Update identity observation
                t_box = extract_face_bbox(target_face)
                t_emb = extract_face_embedding(target_face)
                if t_box is not None:
                    ref_identity.update_observation(t_box, t_emb, frame_index, track_id=target_track_id)

                # Stamp metadata on face
                set_face_meta(target_face, '_assigned_identity', ref_identity.identity_id)
                set_face_meta(target_face, '_assignment_cost', cost)
                set_face_meta(target_face, '_assigned_source', ref_identity.source_face)

            # Count unmapped background faces
            rejections = m - len(matched_targets)
            self.stats['gating_rejections'] += rejections

            # Decay misses on unmatched identities
            for j, ref in enumerate(self.identities):
                if j not in matched_identities:
                    ref.misses += 1

            return assignments
