"""Face analysis interface and early fast-path bypass.

Provides detection, recognition, and early fast-path bypass routing for non-target
and empty frames (0 faces or similarity < threshold), routing raw frames directly
to encoder queues and completely bypassing face enhancement, restorer modules,
stabilizer matrices, and GPU tensor allocations.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import numpy as np

import roop.globals
from roop.typing import Frame, Face
from roop.face_util import (
    FACE_ANALYSER,
    FACE_ANALYSER_POOL,
    get_face_analyser,
    lease_face_analyser,
    release_face_analyser,
    release_face_analyser_aux,
    get_all_faces,
    get_first_face,
    get_all_faces_in_roi,
    detect_boxes_in_roi,
    get_all_faces_hires,
    align_crop,
    analysis_pooled,
    estimate_norm,
)


def has_face(frame: Frame) -> bool:
    """Fast check whether any face is present in the frame."""
    if frame is None:
        return False
    face = get_first_face(frame)
    return face is not None


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine similarity in [0, 1] range (1.0 = identical)."""
    if emb1 is None or emb2 is None:
        return 0.0
    v1 = np.asarray(emb1, dtype=np.float32).flatten()
    v2 = np.asarray(emb2, dtype=np.float32).flatten()
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 1e-6 or n2 <= 1e-6:
        return 0.0
    dot = float(np.dot(v1, v2))
    sim = dot / (n1 * n2)
    # Clamp to [-1.0, 1.0] then map to [0, 1] or return direct cosine similarity
    return max(-1.0, min(1.0, sim))


def check_face_matches_target(
    face: Any,
    target_faces: List[Any],
    threshold: Optional[float] = None
) -> Tuple[bool, float]:
    """Check if face matches any target face above similarity threshold (or below distance threshold).
    
    Returns (matches: bool, best_score: float).
    """
    if not target_faces:
        return True, 1.0
    emb = getattr(face, "embedding", None)
    if emb is None and isinstance(face, dict):
        emb = face.get("embedding")
    if emb is None:
        return False, 0.0

    # Default distance threshold from globals/options (lower distance = higher similarity)
    dist_thresh = float(threshold if threshold is not None
                        else getattr(roop.globals, "face_distance_threshold", 0.65) or 0.65)

    best_sim = -1.0
    min_dist = float("inf")

    for target in target_faces:
        t_emb = getattr(target, "embedding", None)
        if t_emb is None and isinstance(target, dict):
            t_emb = target.get("embedding")
        if t_emb is None:
            continue
        
        sim = compute_cosine_similarity(emb, t_emb)
        dist = 1.0 - sim  # Cosine distance
        if dist < min_dist:
            min_dist = dist
            best_sim = sim

    # Match passes if distance <= threshold
    matches = bool(min_dist <= dist_thresh)
    return matches, best_sim


def evaluate_fast_path(
    frame: Frame,
    target_faces: Optional[List[Any]] = None,
    threshold: Optional[float] = None,
    precomputed_faces: Optional[List[Any]] = None,
) -> Tuple[bool, List[Any]]:
    """Determine whether the frame should take the early fast-path bypass.
    
    Returns (should_bypass: bool, detected_faces: List[Face]).
    If should_bypass is True:
        The frame has 0 faces or similarity score < threshold.
        The caller must route the raw input frame directly to the encoder output queue,
        completely bypassing face enhancement, restorer modules, stabilizer matrices,
        and GPU tensor allocations.
    """
    if frame is None:
        return True, []

    # 1. Check precomputed faces if available
    if precomputed_faces is not None:
        faces = list(precomputed_faces)
    else:
        faces = get_all_faces(frame)

    # If 0 faces found -> immediate bypass
    if not faces:
        return True, []

    # If target faces are specified, check if ANY detected face matches
    if target_faces:
        any_matched = False
        for face in faces:
            matched, _ = check_face_matches_target(face, target_faces, threshold=threshold)
            if matched:
                any_matched = True
                break
        if not any_matched:
            # All tracks/faces failed to match target source -> bypass!
            return True, faces

    return False, faces
