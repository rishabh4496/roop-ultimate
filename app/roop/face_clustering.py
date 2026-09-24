"""Face identity clustering, 512-dimensional normalized embedding extraction,
and Hungarian matching (Linear Sum Assignment) for multi-target tracking and Re-ID.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, AgglomerativeClustering

from roop import recognizer_adaface as _ada
from roop.degrade import swallowed as _swallowed


def normalize_embedding(emb: Any) -> Optional[np.ndarray]:
    """Return L2-normalized float32 vector, or None if invalid."""
    if emb is None:
        return None
    try:
        arr = np.asarray(emb, dtype=np.float32).ravel()
        if arr.size == 0 or not np.isfinite(arr).all():
            return None
        norm = float(np.linalg.norm(arr))
        if norm < 1e-9 or not np.isfinite(norm):
            return np.zeros_like(arr, dtype=np.float32)
        return arr / norm
    except (ValueError, TypeError):
        return None


def extract_face_embedding(
    face: Any,
    frame: Optional[np.ndarray] = None,
    prefer_adaface: bool = False
) -> Optional[np.ndarray]:
    """Extract normalized 512-d embedding using AdaFace or ArcFace.

    Caches the normalized embedding on the face object under '_normed_embedding'
    and updates 'embedding' if missing.
    """
    if face is None:
        return None

    # Check cached normalized vector
    cached = getattr(face, '_normed_embedding', None)
    if cached is None and isinstance(face, dict):
        cached = face.get('_normed_embedding')
    if cached is not None:
        return cached

    raw_emb = None
    # 1. Try AdaFace if preferred or ready
    if prefer_adaface or _ada.ready():
        try:
            raw_emb = _ada.face_embedding(face, frame)
        except Exception as err:
            _swallowed("face_clustering.py:extract_face_embedding:ada", err, "fallback to arcface")

    # 2. Try ArcFace embedding from FaceAnalysis
    if raw_emb is None:
        raw_emb = getattr(face, 'normed_embedding', None)
        if raw_emb is None:
            raw_emb = getattr(face, 'embedding', None)
        if raw_emb is None and isinstance(face, dict):
            raw_emb = face.get('normed_embedding', face.get('embedding'))

    normed = normalize_embedding(raw_emb)
    if normed is not None:
        try:
            if isinstance(face, dict):
                face['_normed_embedding'] = normed
            setattr(face, '_normed_embedding', normed)
        except (AttributeError, TypeError, KeyError):
            pass

    return normed


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity in [-1.0, 1.0] between two vectors."""
    na = normalize_embedding(a)
    nb = normalize_embedding(b)
    if na is None or nb is None:
        return 0.0
    sim = float(np.dot(na, nb))
    return max(-1.0, min(1.0, sim))


def compute_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance in [0.0, 2.0] where 0.0 is identical."""
    return max(0.0, 1.0 - compute_cosine_similarity(a, b))


def compute_distance_matrix(embeddings: List[np.ndarray]) -> np.ndarray:
    """Compute pairwise cosine distance matrix between normalized embeddings."""
    if not embeddings:
        return np.zeros((0, 0), dtype=np.float32)

    # Determine common dimension D from first non-empty embedding
    dim = 512
    for e in embeddings:
        if e is not None and hasattr(e, '__len__') and len(e) > 0:
            dim = len(np.asarray(e).ravel())
            break

    cleaned = []
    for e in embeddings:
        normed = normalize_embedding(e)
        if normed is None or len(normed) != dim:
            cleaned.append(np.zeros(dim, dtype=np.float32))
        else:
            cleaned.append(normed)

    embs = np.asarray(cleaned, dtype=np.float32)
    # Matrix multiplication for cosine similarities: E @ E.T
    sims = np.dot(embs, embs.T)
    # Clip to [-1.0, 1.0] and compute distance = 1 - sim
    sims = np.clip(sims, -1.0, 1.0)
    dists = 1.0 - sims
    np.fill_diagonal(dists, 0.0)
    return np.maximum(0.0, dists).astype(np.float32)


def cluster_face_embeddings(
    embeddings: List[np.ndarray],
    method: str = 'dbscan',
    eps: float = 0.45,
    min_samples: int = 1,
    distance_threshold: float = 0.45,
) -> np.ndarray:
    """Cluster face embeddings using DBSCAN or Agglomerative Clustering.

    Returns an array of integer cluster IDs of length len(embeddings).
    """
    n = len(embeddings)
    if n == 0:
        return np.array([], dtype=np.int32)
    if n == 1:
        return np.zeros(1, dtype=np.int32)

    dist_matrix = compute_distance_matrix(embeddings)

    method_key = (method or 'dbscan').lower()
    if method_key == 'dbscan':
        # DBSCAN with precomputed distance matrix
        clustering = DBSCAN(
            eps=float(eps),
            min_samples=max(1, int(min_samples)),
            metric='precomputed'
        )
        labels = clustering.fit_predict(dist_matrix)

        # Post-process noise points (-1): assign each to its own unique cluster
        next_label = int(np.max(labels)) + 1 if np.any(labels >= 0) else 0
        for i in range(len(labels)):
            if labels[i] == -1:
                # Check if it is close enough to any formed cluster
                assigned = False
                for c in range(next_label):
                    members = np.where(labels == c)[0]
                    if len(members) > 0:
                        min_d = np.min(dist_matrix[i, members])
                        if min_d <= eps * 1.15:
                            labels[i] = c
                            assigned = True
                            break
                if not assigned:
                    labels[i] = next_label
                    next_label += 1

    elif method_key in ('agglomerative', 'agg'):
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='average',
            distance_threshold=float(distance_threshold),
        )
        labels = clustering.fit_predict(dist_matrix)
    else:
        raise ValueError(f"Unsupported clustering method: {method}. Choose 'dbscan' or 'agglomerative'.")

    # Re-index labels contiguously starting at 0
    unique_labels = sorted(set(labels))
    mapping = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(unique_labels)}
    contiguous_labels = np.array([mapping[lbl] for lbl in labels], dtype=np.int32)
    return contiguous_labels


def solve_hungarian_matching(
    cost_matrix: np.ndarray,
    cost_limit: float = 1.0
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Solve global optimal bipartite assignment using the Hungarian algorithm.

    Args:
        cost_matrix: (N, M) matrix of assignment costs.
        cost_limit: Maximum permissible cost for an accepted assignment.

    Returns:
        matches: List of (row_idx, col_idx, cost) for accepted pairs.
        unmatched_rows: List of row indices with no accepted assignment.
        unmatched_cols: List of col indices with no accepted assignment.
    """
    if cost_matrix.size == 0:
        n_rows, n_cols = cost_matrix.shape
        return [], list(range(n_rows)), list(range(n_cols))

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    matched_rows = set()
    matched_cols = set()

    for r, c in zip(row_indices, col_indices):
        cost = float(cost_matrix[r, c])
        if cost <= cost_limit and np.isfinite(cost):
            matches.append((int(r), int(c), cost))
            matched_rows.add(int(r))
            matched_cols.add(int(c))

    n_rows, n_cols = cost_matrix.shape
    unmatched_rows = [r for r in range(n_rows) if r not in matched_rows]
    unmatched_cols = [c for c in range(n_cols) if c not in matched_cols]

    return matches, unmatched_rows, unmatched_cols
