"""Face analysis interface and early fast-path bypass.

Provides detection, recognition, and early fast-path bypass routing for non-target
and empty frames (0 faces or similarity < threshold), routing raw frames directly
to encoder queues and completely bypassing face enhancement, restorer modules,
stabilizer matrices, and GPU tensor allocations.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

import roop.globals
from roop.typing import Frame, Face
from roop.utilities import (compose_affines, pre_rotate_face_crop,
                            profile_alignment_matrix, transform_points)
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


def face_roll_degrees(face: Any) -> float:
    """Return the resolved roll when present, otherwise derive it from the eyes."""
    try:
        value = face.get('roll_deg') if isinstance(face, dict) else getattr(face, 'roll_deg', None)
        if value is not None and np.isfinite(float(value)):
            return float(value)
    except (TypeError, ValueError):
        pass
    try:
        kps = face.get('kps') if isinstance(face, dict) else getattr(face, 'kps', None)
        left_eye, right_eye = np.asarray(kps, dtype=np.float32).reshape(5, 2)[:2]
        return float(np.degrees(np.arctan2(right_eye[1] - left_eye[1],
                                           right_eye[0] - left_eye[0])))
    except (TypeError, ValueError, AttributeError):
        return 0.0


def face_yaw_pitch(face: Any) -> Tuple[float, float]:
    """Read target pose stamped by the tracker, with a 5-point solver fallback."""
    def _value(name):
        return face.get(name) if isinstance(face, dict) else getattr(face, name, None)
    try:
        yaw, pitch = _value('_adaptive_yaw'), _value('_adaptive_pitch')
        if yaw is not None and pitch is not None:
            return float(yaw), float(pitch)
    except (TypeError, ValueError):
        pass
    try:
        from roop.face_util import solve_pose_5pt
        pose = solve_pose_5pt(_value('kps'))
        if pose is not None:
            return float(pose[0]), float(pose[1])
    except Exception:
        pass
    return 0.0, 0.0


def profile_anchors(face: Any, yaw_degrees: float) -> Optional[np.ndarray]:
    """Find visible tragus/ear, nose tip and chin centre from refined landmarks.

    A detector may optionally stamp ``profile_anchors`` explicitly.  Otherwise
    the lateral contour and lowest refined point supply the visible ear/tragus
    and chin.  Without 106-point refinement this returns ``None`` so callers
    retain the well-defined five-point path rather than inventing an ear.
    """
    def _value(name):
        return face.get(name) if isinstance(face, dict) else getattr(face, name, None)
    explicit = _value('profile_anchors')
    if explicit is not None:
        try:
            anchors = np.asarray(explicit, dtype=np.float32).reshape(3, 2)
            return anchors if np.isfinite(anchors).all() else None
        except (TypeError, ValueError):
            return None
    try:
        refined = np.asarray(_value('landmark_2d_106'), dtype=np.float32).reshape(-1, 2)
        kps = np.asarray(_value('kps'), dtype=np.float32).reshape(5, 2)
        if refined.shape[0] < 20 or not np.isfinite(refined).all():
            return None
        # Keep the lower face from being chosen as the lateral ear contour.
        upper = refined[refined[:, 1] <= np.percentile(refined[:, 1], 80)]
        if len(upper) == 0:
            return None
        ear = upper[np.argmax(upper[:, 0]) if yaw_degrees >= 0.0 else np.argmin(upper[:, 0])]
        chin = refined[np.argmax(refined[:, 1])]
        return np.asarray((ear, kps[2], chin), dtype=np.float32)
    except (TypeError, ValueError, AttributeError):
        return None


def adaptive_alignment_matrix(kps: np.ndarray, image_size: int, mode: str,
                              yaw_degrees: float = 0.0,
                              anchors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, str]:
    """Choose standard five-point or non-squashing profile alignment."""
    if abs(float(yaw_degrees)) > 45.0 and anchors is not None:
        matrix = profile_alignment_matrix(anchors, image_size, yaw_degrees)
        if matrix is not None:
            return matrix, 'profile_3pt'
    return estimate_norm(np.asarray(kps, dtype=np.float32).reshape(5, 2), image_size, mode), 'five_point'


def canonicalize_face_alignment(image: Frame, face: Any, image_size: int, mode: str,
                                dst: Optional[np.ndarray] = None):
    """Pre-upright a rolled face and return its canonical crop plus paste affine.

    ``paste_matrix`` maps original target coordinates directly to canonical crop
    coordinates.  Its inverse is therefore the exact un-rotation/paste mapping
    used before alpha blending, with no second resample of the swapped result.
    """
    def _value(name):
        return face.get(name) if isinstance(face, dict) else getattr(face, name, None)
    kps = np.asarray(_value('kps'), dtype=np.float32).reshape(5, 2)
    bbox = np.asarray(_value('bbox'), dtype=np.float32).reshape(4)
    yaw, pitch = face_yaw_pitch(face)
    roll = face_roll_degrees(face)
    upright, pre_rotation, inverse_pre_rotation, applied = pre_rotate_face_crop(image, bbox, roll)
    upright_kps = transform_points(kps, pre_rotation)
    anchors = profile_anchors(face, yaw)
    if anchors is not None:
        anchors = transform_points(anchors, pre_rotation)
    local_matrix, alignment_kind = adaptive_alignment_matrix(
        upright_kps, image_size, mode, yaw_degrees=yaw, anchors=anchors)
    paste_matrix = compose_affines(local_matrix, pre_rotation)
    if dst is not None:
        cv2.warpAffine(upright, local_matrix, (image_size, image_size), dst=dst,
                       borderMode=cv2.BORDER_REPLICATE)
        aligned = dst
    else:
        aligned = cv2.warpAffine(upright, local_matrix, (image_size, image_size),
                                 borderMode=cv2.BORDER_REPLICATE)
    return aligned, paste_matrix, {
        'yaw': yaw, 'pitch': pitch, 'roll': roll, 'pre_rotation': pre_rotation,
        'inverse_pre_rotation': inverse_pre_rotation, 'applied_roll_prerotation': applied,
        'alignment_kind': alignment_kind,
    }


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
