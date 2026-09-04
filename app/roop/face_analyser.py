"""Face analysis interface and early fast-path bypass.

Provides detection, recognition, and early fast-path bypass routing for non-target
and empty frames (0 faces or similarity < threshold), routing raw frames directly
to encoder queues and completely bypassing face enhancement, restorer modules,
stabilizer matrices, and GPU tensor allocations.
"""

from __future__ import annotations

import os
from threading import RLock
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


# The Kalman/Hungarian tracker and its face-field helpers now live in
# `roop.tracker`, which also owns coasting through occlusion and landmark
# symmetry inpainting. They are re-exported here unchanged so every existing
# `from roop.face_analyser import FaceTracker` keeps working -- ProcessMgr and
# tests/test_face_tracker.py both import them from this module.
from roop.tracker import (                                  # noqa: F401
    FaceTrack,
    FaceTracker,
    _bbox_iou,
    _cosine_similarity,
    _face_field,
    _set_face_field,
)


# ---------------------------------------------------------------------------
# Canonical 3D Facial Mesh Template (OpenCV camera convention: +X right, +Y down, +Z forward)
# ---------------------------------------------------------------------------

CANONICAL_FACE_3D_5 = np.array([
    [-0.342, -0.412,  0.169],  # Left eye center
    [ 0.342, -0.412,  0.169],  # Right eye center
    [ 0.000, -0.095,  0.487],  # Nose tip
    [-0.368,  0.547,  0.299],  # Left mouth corner
    [ 0.368,  0.547,  0.299],  # Right mouth corner
], dtype=np.float64)


def compute_canonical_roll_angle(landmarks: Any) -> Tuple[float, float]:
    """Compute the exact 2D roll angle from eye/nose landmarks with continuous angle math.

    Formula:
        theta = atan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x)

    Returns:
        (theta_rad, theta_deg) continuously wrapped in (-pi, pi] and (-180, 180].
    """
    if landmarks is None:
        return 0.0, 0.0
    try:
        pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 2 or not np.isfinite(pts[:2]).all():
            return 0.0, 0.0

        if pts.shape[0] >= 68:
            left_eye = np.mean(pts[36:42], axis=0)
            right_eye = np.mean(pts[42:48], axis=0)
        else:
            left_eye = pts[0]
            right_eye = pts[1]

        dx = float(right_eye[0] - left_eye[0])
        dy = float(right_eye[1] - left_eye[1])

        if abs(dx) < 1e-7 and abs(dy) < 1e-7:
            return 0.0, 0.0

        theta_rad = float(np.arctan2(dy, dx))
        # Continuous wrap-around without discontinuities around +-pi
        theta_rad = float((theta_rad + np.pi) % (2.0 * np.pi) - np.pi)
        theta_deg = float(np.degrees(theta_rad))
        return theta_rad, theta_deg
    except Exception:
        return 0.0, 0.0


def build_canonical_rotation_matrix(
    center: Tuple[float, float],
    theta_deg: float,
    threshold_deg: float = 45.0
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Dynamically construct an affine rotation matrix R(theta, center) that maps
    the cropped face to an upright canonical orientation, along with its inverse inv(R).

    If abs(theta) > 45 degrees, active rotation R is computed.
    Otherwise, identity matrices are returned to preserve hot paths.

    Returns:
        (R, inv_R, applied) where R and inv_R are 2x3 float32 affine matrices.
    """
    norm_deg = float((theta_deg + 180.0) % 360.0 - 180.0)
    if abs(norm_deg) <= float(threshold_deg):
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        return identity, identity.copy(), False

    cx, cy = float(center[0]), float(center[1])
    # In OpenCV (+y down), getRotationMatrix2D rotates with positive angle counter-clockwise.
    # To upright a face with in-plane roll of norm_deg (atan2(dy, dx)), we rotate by norm_deg around center.
    R = cv2.getRotationMatrix2D((cx, cy), norm_deg, 1.0).astype(np.float32)
    inv_R = cv2.getRotationMatrix2D((cx, cy), -norm_deg, 1.0).astype(np.float32)
    return R, inv_R, True


def estimate_head_pose_pnp(
    landmarks: Any,
    image_shape: Optional[Tuple[int, int]] = None
) -> Tuple[float, float, float]:
    """Estimate 3D head pose (Yaw, Pitch, Roll in degrees) using 2D-to-3D PnP
    with a canonical 3D facial mesh template.

    Returns:
        (yaw_deg, pitch_deg, roll_deg) in degrees.
    """
    if landmarks is None:
        return 0.0, 0.0, 0.0

    try:
        pts = np.asarray(landmarks, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 5 or not np.isfinite(pts).all():
            return 0.0, 0.0, 0.0

        if image_shape is not None and len(image_shape) >= 2:
            h, w = int(image_shape[0]), int(image_shape[1])
        else:
            span = float(max(np.std(pts) * 4.0, 128.0))
            h, w = int(span), int(span)

        focal = max(h, w) * 1.2
        cx = float(w) * 0.5
        cy = float(h) * 0.5
        cam_matrix = np.array([
            [focal, 0.0,   cx],
            [0.0,   focal, cy],
            [0.0,   0.0,   1.0]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        if pts.shape[0] >= 68:
            from roop.face_3d_recon import _REF3D_68
            ref3d = (_REF3D_68 * np.array([1.0, -1.0, 1.0])).astype(np.float64)
            target_pts = pts[:68]
        else:
            ref3d = CANONICAL_FACE_3D_5
            target_pts = pts[:5]

        lm_std = float(np.std(target_pts))
        ref_std = float(np.std(ref3d[:, :2]))
        scale = lm_std / max(ref_std, 1e-6)
        pts3d = (ref3d * scale).copy()
        pts3d[:, 2] += focal

        # Use SQPNP for 5 points, or EPNP as fallback
        ok, rvec, tvec = False, None, None
        for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP):
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    pts3d, target_pts.astype(np.float64),
                    cam_matrix, dist_coeffs,
                    flags=flag
                )
                if ok and np.isfinite(rvec).all():
                    break
            except Exception:
                pass

        if not ok or rvec is None or not np.isfinite(rvec).all():
            from roop.face_util import solve_pose_5pt
            pose = solve_pose_5pt(pts[:5])
            if pose is not None:
                return float(pose[0]), float(pose[1]), float(pose[2])
            return 0.0, 0.0, 0.0

        R, _ = cv2.Rodrigues(rvec)
        r02 = float(R[0, 2])
        r22 = float(R[2, 2])
        r12 = float(np.clip(-R[1, 2], -1.0, 1.0))
        r10 = float(R[1, 0])
        r11 = float(R[1, 1])

        yaw = float(np.degrees(np.arctan2(r02, r22)))
        pitch = float(np.degrees(np.arcsin(r12)))
        roll = float(np.degrees(np.arctan2(r10, r11)))

        yaw = float((yaw + 180.0) % 360.0 - 180.0)
        pitch = float((pitch + 180.0) % 360.0 - 180.0)
        roll = float((roll + 180.0) % 360.0 - 180.0)
        return yaw, pitch, roll
    except Exception:
        try:
            from roop.face_util import solve_pose_5pt
            pose = solve_pose_5pt(np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)[:5])
            if pose is not None:
                return float(pose[0]), float(pose[1]), float(pose[2])
        except Exception:
            pass
        return 0.0, 0.0, 0.0


def weighted_umeyama_alignment(
    src: np.ndarray,
    dst: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """Closed-form Weighted Umeyama similarity transform estimation.

    Finds similarity transform M = [c*R | t] minimizing:
        sum_i w_i || dst_i - (c*R*src_i + t) ||^2

    src: (N, 2) source points (observed landmarks)
    dst: (N, 2) destination points (canonical template)
    weights: (N,) positive weights for each landmark

    Returns:
        (2, 3) float32 affine similarity matrix.
    """
    X = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    Y = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    n = X.shape[0]
    if n < 2:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    if weights is None:
        w = np.full(n, 1.0 / float(n), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        w_sum = float(np.sum(w))
        if w_sum <= 1e-9 or not np.isfinite(w_sum):
            w = np.full(n, 1.0 / float(n), dtype=np.float64)
        else:
            w = np.clip(w / w_sum, 1e-6, None)
            w = w / float(np.sum(w))

    # 1. Weighted centroids
    mu_X = np.sum(X * w[:, None], axis=0)
    mu_Y = np.sum(Y * w[:, None], axis=0)

    # 2. Shifted centered coordinates
    X_c = X - mu_X
    Y_c = Y - mu_Y

    # 3. Weighted variance of source points
    var_X = float(np.sum(w * np.sum(X_c ** 2, axis=1)))
    if var_X < 1e-9:
        t = mu_Y - mu_X
        return np.array([[1.0, 0.0, float(t[0])], [0.0, 1.0, float(t[1])]], dtype=np.float32)

    # 4. Weighted cross-covariance matrix: Sigma_{YX} = Y_c^T W X_c
    cov_YX = (Y_c * w[:, None]).T @ X_c

    # 5. SVD
    U, S, Vt = np.linalg.svd(cov_YX)

    # 6. Reflection correction
    d = float(np.linalg.det(U) * np.linalg.det(Vt))
    D = np.diag([1.0, 1.0 if d >= 0.0 else -1.0])

    # 7. Rotation
    R = U @ D @ Vt

    # 8. Scale
    scale = float(S[0] + D[1, 1] * S[1]) / var_X
    if scale <= 1e-7 or not np.isfinite(scale):
        scale = 1.0

    # 9. Translation
    t = mu_Y - scale * (R @ mu_X)

    # 10. Composite affine
    M = np.zeros((2, 3), dtype=np.float32)
    M[:, :2] = (scale * R).astype(np.float32)
    M[:, 2] = t.astype(np.float32)
    return M


class AffineEMAFilter:
    """Exponential Moving Average (EMA) / Kalman filter for 2x3 affine transformation matrices.

    Eliminates micro-jitter across sequential video frames without adding temporal lag (alpha = 0.85).
    """

    def __init__(self, alpha: float = 0.85, max_displacement: float = 50.0, max_gap: int = 5):
        self.alpha = float(alpha)
        self.max_displacement = float(max_displacement)
        self.max_gap = int(max_gap)
        self.tracks: Dict[int, Tuple[np.ndarray, int]] = {}
        self._lock = RLock()

    def filter(self, matrix: np.ndarray, track_id: Optional[int] = None, frame_index: Optional[int] = None) -> np.ndarray:
        if matrix is None or track_id is None:
            return matrix
        tid = int(track_id)
        f_idx = int(frame_index) if frame_index is not None else 0
        with self._lock:
            if tid in self.tracks:
                prev_m, prev_f = self.tracks[tid]
                dt = abs(f_idx - prev_f) if frame_index is not None else 1
                # Translation displacement check: ignore filter if scene cut / large teleport occurred
                disp = float(np.linalg.norm(matrix[:, 2] - prev_m[:, 2]))
                if dt <= self.max_gap and disp <= self.max_displacement:
                    filtered = (self.alpha * matrix + (1.0 - self.alpha) * prev_m).astype(np.float32)
                    self.tracks[tid] = (filtered, f_idx)
                    return filtered
            self.tracks[tid] = (matrix.copy().astype(np.float32), f_idx)
            return matrix

    def reset(self, track_id: Optional[int] = None) -> None:
        with self._lock:
            if track_id is None:
                self.tracks.clear()
            elif track_id in self.tracks:
                del self.tracks[track_id]


DEFAULT_AFFINE_EMA = AffineEMAFilter(alpha=0.85)


def profile_stable_anchor_alignment(
    kps: np.ndarray,
    image_size: int = 128,
    mode: str = "arcface",
    yaw_degrees: float = 0.0,
    landmarks_68: Optional[np.ndarray] = None,
    face: Any = None
) -> Tuple[np.ndarray, np.ndarray]:
    """3-Point Stable Anchor Alignment & Estimated Virtual Landmark Synthesis.

    When |yaw| >= 45 (profile), standard 2D Umeyama similarity transform collapses
    because far-side eye/mouth landmarks overlap or occlude, causing sheared or crushed
    face crops.
    This solver fits a rigid similarity transform strictly from the 3 stable visible anchors:
        1. Visible eye corner (or eye center)
        2. Nose tip
        3. Chin centre
    against canonical 112x112 (RealSwap) or 256x256 (GPEN) template coordinates,
    and synthesizes virtual 5-point landmarks via inverse projection.
    Guarantees zero aspect-ratio shearing (sigma_1 == sigma_2) and preserves faithful
    facial proportions and scale across extreme profiles (|yaw| up to 85°).

    Returns:
        (affine_matrix_2x3: np.ndarray, virtual_kps_5: np.ndarray)
    """
    from roop.face_util import swap_template_points
    kps_5 = np.asarray(kps, dtype=np.float32).reshape(5, 2)
    dst_5 = swap_template_points(image_size, mode).astype(np.float32)

    yaw = float(yaw_degrees)
    is_right_turn = (yaw >= 0.0)  # nose pointing to image right

    # 1. Extract Visible Eye Corner / Center
    lm68 = None
    if landmarks_68 is not None:
        try:
            arr = np.asarray(landmarks_68, dtype=np.float32).reshape(-1, 2)
            if arr.shape[0] >= 68 and np.all(np.isfinite(arr[:68])):
                lm68 = arr
        except Exception:
            lm68 = None

    if lm68 is not None:
        # Outer eye corner: index 36 for left eye, 45 for right eye
        vis_eye = lm68[36] if is_right_turn else lm68[45]
        nose_tip = lm68[30]
        chin = lm68[8]
    else:
        # 5-point fallback
        vis_eye = kps_5[0] if is_right_turn else kps_5[1]
        nose_tip = kps_5[2]
        refined_chin = None
        if face is not None:
            try:
                refined = np.asarray(_face_field(face, 'landmark_2d_106'), dtype=np.float32).reshape(-1, 2)
                if refined.shape[0] >= 20 and np.all(np.isfinite(refined)):
                    refined_chin = refined[np.argmax(refined[:, 1])]
            except Exception:
                refined_chin = None

        if refined_chin is not None:
            chin = refined_chin
        else:
            # Estimate chin along facial symmetry midline
            vis_mouth = kps_5[3] if is_right_turn else kps_5[4]
            v_mouth = vis_mouth - nose_tip
            v_eye = nose_tip - vis_eye
            norm_mouth = float(np.linalg.norm(v_mouth))
            norm_eye = float(np.linalg.norm(v_eye))
            if norm_mouth > 1e-4:
                dir_vert = v_mouth / norm_mouth
                chin = nose_tip + dir_vert * (norm_mouth * 1.65)
            elif norm_eye > 1e-4:
                dir_vert = v_eye / norm_eye
                chin = nose_tip + dir_vert * (norm_eye * 1.25)
            else:
                chin = nose_tip + np.array([0.0, 50.0], dtype=np.float32)

    # 2. Canonical Destination 3-Point Anchor
    dst_vis_eye = dst_5[0] if is_right_turn else dst_5[1]
    dst_nose = dst_5[2]
    dst_chin_y = float(dst_5[2, 1] + 1.65 * (dst_5[3, 1] - dst_5[2, 1]))
    dst_chin = np.array([float(dst_5[2, 0]), dst_chin_y], dtype=np.float32)

    src_3 = np.array([vis_eye, nose_tip, chin], dtype=np.float32)
    dst_3 = np.array([dst_vis_eye, dst_nose, dst_chin], dtype=np.float32)

    # 3. Solve Optimal Similarity Transform (Uniform scale, Rotation, Translation)
    M = weighted_umeyama_alignment(src_3, dst_3)

    # Enforce strict similarity constraint (sigma_1 == sigma_2)
    scales = np.linalg.norm(M[:, :2], axis=1)
    if abs(scales[0] - scales[1]) > 1e-3 or not np.isfinite(scales).all():
        s = float((scales[0] + scales[1]) * 0.5) if np.isfinite(scales).all() and min(scales) > 1e-6 else 1.0
        u, _, vt = np.linalg.svd(M[:, :2])
        r = u @ vt
        M[:, :2] = (s * r).astype(np.float32)

    # 4. Virtual Landmark Synthesis via inverse transform
    inv_M = cv2.invertAffineTransform(M).astype(np.float32)
    virtual_kps = transform_points(dst_5, inv_M)

    return M, virtual_kps


def profile_aware_umeyama_alignment(
    kps: np.ndarray,
    image_size: int = 128,
    yaw_degrees: float = 0.0,
    pitch_degrees: float = 0.0,
    landmarks_68: Optional[np.ndarray] = None,
    face: Any = None
) -> Tuple[np.ndarray, str]:
    """Adaptive Profile-Aware Alignment Solver.

    When |yaw| < 45 and |pitch| < 30:
        Utilizes standard 5-point ArcFace/Umeyama affine transform.
    When |yaw| >= 45 (profile):
        Switches to 3-point stable anchor (visible eye corner, nose tip, chin)
        with estimated virtual landmark synthesis to maintain strict 112x112
        (RealSwap) and 256x256 (GPEN) crop geometry without horizontal squashing.
    """
    from roop.face_util import swap_template_points
    base_dst_5 = swap_template_points(image_size, "arcface")
    kps_5 = np.asarray(kps, dtype=np.float32).reshape(5, 2)

    yaw = float(yaw_degrees)
    pitch = float(pitch_degrees)
    abs_yaw = abs(yaw)
    abs_pitch = abs(pitch)

    if abs_yaw < 45.0 and abs_pitch < 30.0:
        M = weighted_umeyama_alignment(kps_5, base_dst_5, weights=None)
        return M, "five_point"

    if abs_yaw >= 45.0:
        # Check if explicit profile anchors exist on face or arguments
        anchors = profile_anchors(face, yaw) if face is not None else None
        if anchors is not None:
            M = profile_alignment_matrix(anchors, image_size, yaw)
            if M is not None:
                return M, "profile_3pt"

        M, _ = profile_stable_anchor_alignment(
            kps_5, image_size=image_size, mode="arcface",
            yaw_degrees=yaw, landmarks_68=landmarks_68, face=face
        )
        return M, "profile_3pt"

    # Pitch foreshortening (|pitch| >= 30, |yaw| < 45)
    gamma = float(np.clip((abs_pitch - 30.0) / 30.0, 0.0, 1.0))
    weights = np.ones(5, dtype=np.float64)
    weights[2] = 2.0 + 1.0 * gamma  # nose tip
    M = weighted_umeyama_alignment(kps_5, base_dst_5, weights=weights)
    return M, "five_point"


def compute_composite_inverse(
    inv_R: np.ndarray,
    inv_M_warp: np.ndarray
) -> np.ndarray:
    """Compute the composite inverse transformation T_final = inv(R) @ inv(M_warp).

    Maps canonical crop coordinates directly back to original full frame coordinates.
    """
    a = np.vstack([np.asarray(inv_R, dtype=np.float64).reshape(2, 3), [0.0, 0.0, 1.0]])
    b = np.vstack([np.asarray(inv_M_warp, dtype=np.float64).reshape(2, 3), [0.0, 0.0, 1.0]])
    composite = a @ b
    return composite[:2].astype(np.float32)


def compute_composite_forward(
    M_warp: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    """Compute composite forward transform M_composite = M_warp @ R.

    Maps original full frame coordinates to canonical crop coordinates.
    """
    a = np.vstack([np.asarray(M_warp, dtype=np.float64).reshape(2, 3), [0.0, 0.0, 1.0]])
    b = np.vstack([np.asarray(R, dtype=np.float64).reshape(2, 3), [0.0, 0.0, 1.0]])
    composite = a @ b
    return composite[:2].astype(np.float32)


def face_roll_degrees(face: Any) -> float:
    """Return the resolved roll when present, otherwise derive it from the eyes."""
    try:
        value = face.get('roll_deg') if isinstance(face, dict) else getattr(face, 'roll_deg', None)
        if value is not None and np.isfinite(float(value)):
            return float(value)
    except (TypeError, ValueError):
        pass
    kps = face.get('kps') if isinstance(face, dict) else getattr(face, 'kps', None)
    if kps is not None:
        _, deg = compute_canonical_roll_angle(kps)
        return deg
    return 0.0


def face_yaw_pitch(face: Any) -> Tuple[float, float]:
    """Read target pose stamped by tracker, with PnP solver fallback."""
    def _value(name):
        return face.get(name) if isinstance(face, dict) else getattr(face, name, None)
    try:
        yaw, pitch = _value('_adaptive_yaw'), _value('_adaptive_pitch')
        if yaw is not None and pitch is not None:
            return float(yaw), float(pitch)
    except (TypeError, ValueError):
        pass
    kps = _value('kps')
    lm68 = _value('landmark_3d_68')
    if lm68 is None:
        lm68 = _value('landmarks_68')
    if lm68 is None:
        lm68 = _value('landmarks')
    yaw, pitch, _ = estimate_head_pose_pnp(lm68 if lm68 is not None else kps)
    return yaw, pitch


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


def adaptive_alignment_matrix(kps: np.ndarray, image_size: int, mode: str = "arcface",
                              yaw_degrees: float = 0.0,
                              pitch_degrees: float = 0.0,
                              anchors: Optional[np.ndarray] = None,
                              landmarks_68: Optional[np.ndarray] = None,
                              face: Any = None) -> Tuple[np.ndarray, str]:
    """Adaptive alignment solver for facial angle alignment and S5 profile defect mitigation.

    When |yaw| < 45 and |pitch| < 30:
        Standard 5-point ArcFace/Umeyama affine transform.
    When |yaw| >= 45 (profile):
        Switch to 3D pose-invariant landmark projection or 3-point stable anchor
        (visible eye corner, nose tip, chin) with estimated virtual landmark synthesis
        to maintain strict 112x112 (RealSwap) and 256x256 (GPEN) crop geometry.
    """
    yaw = float(yaw_degrees)
    pitch = float(pitch_degrees)
    abs_yaw = abs(yaw)
    abs_pitch = abs(pitch)

    if abs_yaw < 45.0 and abs_pitch < 30.0:
        from roop.face_util import swap_template_points
        return weighted_umeyama_alignment(
            np.asarray(kps, dtype=np.float32).reshape(5, 2),
            swap_template_points(image_size, mode)
        ), 'five_point'

    if abs_yaw >= 45.0:
        # 1. Check if explicit profile anchors are provided or stamped on face
        if anchors is not None:
            M = profile_alignment_matrix(anchors, image_size, yaw)
            if M is not None:
                return M, 'profile_3pt'
        if face is not None:
            p_anchors = profile_anchors(face, yaw)
            if p_anchors is not None:
                M = profile_alignment_matrix(p_anchors, image_size, yaw)
                if M is not None:
                    return M, 'profile_3pt'

        # 2. 3-point stable anchor (visible eye corner, nose tip, chin) with virtual landmark synthesis
        M, _ = profile_stable_anchor_alignment(
            kps, image_size=image_size, mode=mode,
            yaw_degrees=yaw, landmarks_68=landmarks_68, face=face
        )
        return M, 'profile_3pt'

    # Pitch foreshortening
    return profile_aware_umeyama_alignment(
        kps, image_size=image_size,
        yaw_degrees=yaw, pitch_degrees=pitch,
        landmarks_68=landmarks_68, face=face
    )


def canonicalize_face_alignment(image: Frame, face: Any, image_size: int, mode: str,
                                dst: Optional[np.ndarray] = None):
    """Pre-upright a rolled face and return its canonical crop plus paste affine.

    Applies:
    1. Roll pre-rotation when abs(roll) > 45°.
    2. Adaptive alignment solver:
       - |yaw| < 45 and |pitch| < 30: standard 5-point ArcFace/Umeyama affine transform
       - |yaw| >= 45 (profile): 3-point stable anchor with virtual landmark synthesis
         maintaining strict 112x112 (RealSwap) and 256x256 (GPEN) crop geometry.
    3. Exponential Moving Average (EMA) / Kalman filter (alpha = 0.85) across sequential
       video frames to eliminate micro-jitter without adding temporal lag.
    """
    def _value(name):
        return face.get(name) if isinstance(face, dict) else getattr(face, name, None)
    kps = np.asarray(_value('kps'), dtype=np.float32).reshape(5, 2)
    bbox = np.asarray(_value('bbox'), dtype=np.float32).reshape(4)
    lm68 = _value('landmark_3d_68')
    if lm68 is None:
        lm68 = _value('landmarks_68')
    if lm68 is None:
        lm68 = _value('landmarks')
    if lm68 is not None:
        lm68_arr = np.asarray(lm68, dtype=np.float32)
        lm68 = lm68_arr[:, :2] if lm68_arr.ndim == 2 and lm68_arr.shape[1] >= 2 else None
    _, roll_deg = compute_canonical_roll_angle(kps)
    center = ((float(bbox[0]) + float(bbox[2])) * 0.5,
              (float(bbox[1]) + float(bbox[3])) * 0.5)

    R, inv_R, applied = build_canonical_rotation_matrix(center, roll_deg, threshold_deg=45.0)

    h, w = image.shape[:2]
    if applied:
        upright = cv2.warpAffine(image, R, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        upright_kps = transform_points(kps, R)
        # On a rolled face (>45 deg), the 2D/3D-68 model running on the unrotated crop hallucinates an upright
        # orientation, making its landmarks inverted. Do not use unrotated 68-pt landmarks for upright alignment.
        upright_68 = None
    else:
        upright = image
        upright_kps = kps
        upright_68 = lm68

    # Read target pose stamped by tracker or estimate via PnP
    yaw, pitch = face_yaw_pitch(face)
    if yaw == 0.0 and pitch == 0.0 and _value('_adaptive_yaw') is None and _value('yaw') is None:
        yaw, pitch, _ = estimate_head_pose_pnp(upright_68 if upright_68 is not None else upright_kps, (h, w))

    # Check for explicit profile anchors on face
    raw_anchors = _value('profile_anchors')
    if raw_anchors is None:
        raw_anchors = profile_anchors(face, yaw)
    anchors = None
    if raw_anchors is not None:
        try:
            anch_arr = np.asarray(raw_anchors, dtype=np.float32).reshape(3, 2)
            if np.all(np.isfinite(anch_arr)):
                anchors = transform_points(anch_arr, R) if applied else anch_arr
        except Exception:
            anchors = None

    local_matrix, alignment_kind = adaptive_alignment_matrix(
        upright_kps, image_size=image_size, mode=mode,
        yaw_degrees=yaw, pitch_degrees=pitch, anchors=anchors,
        landmarks_68=upright_68, face=face)

    # Apply Exponential Moving Average (EMA) filter (alpha = 0.85) to eliminate micro-jitter
    track_id = _face_field(face, '_track_id', _face_field(face, 'track_id'))
    frame_idx = _face_field(face, 'frame_idx', _face_field(face, 'last_seen_frame'))
    if track_id is not None:
        local_matrix = DEFAULT_AFFINE_EMA.filter(local_matrix, track_id=track_id, frame_index=frame_idx)

    paste_matrix = compute_composite_forward(local_matrix, R)
    inv_local = cv2.invertAffineTransform(local_matrix).astype(np.float32)
    inv_paste = compute_composite_inverse(inv_R, inv_local)

    if dst is not None:
        cv2.warpAffine(upright, local_matrix, (image_size, image_size), dst=dst,
                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        aligned = dst
    else:
        aligned = cv2.warpAffine(upright, local_matrix, (image_size, image_size),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return aligned, paste_matrix, {
        'yaw': yaw, 'pitch': pitch, 'roll': roll_deg, 'pre_rotation': R,
        'inverse_pre_rotation': inv_R, 'applied_roll_prerotation': applied,
        'alignment_kind': alignment_kind, 'inv_paste_matrix': inv_paste,
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
    threshold: Optional[float] = None,
    sliding_window: Optional[Any] = None,
    previous_bbox: Optional[Sequence[float]] = None,
    high_threshold: Optional[float] = None,
    low_threshold: Optional[float] = None,
    iou_threshold: Optional[float] = None
) -> Tuple[bool, float]:
    """Check if face matches any target face above similarity threshold (or below distance threshold).
    
    Supports dual-threshold hysteresis:
      - High acceptance threshold: S_match >= high_threshold (default 0.62).
      - Low tracking threshold: S_track >= low_threshold (default 0.50) when paired with
        spatial IoU >= iou_threshold (default 0.50) with previous_bbox.

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

    if sliding_window is not None and len(sliding_window) > 0:
        win_max = sliding_window.max_similarity(emb)
        win_mean = sliding_window.mean_similarity(emb)
        best_sim = max(best_sim, win_max, win_mean)
        min_dist = min(min_dist, 1.0 - best_sim)

    # Dual-threshold hysteresis evaluation
    if previous_bbox is not None or high_threshold is not None or low_threshold is not None:
        h_thresh = 0.62 if high_threshold is None else float(high_threshold)
        l_thresh = 0.50 if low_threshold is None else float(low_threshold)
        i_thresh = 0.50 if iou_threshold is None else float(iou_threshold)

        cur_bbox = getattr(face, "bbox", None) if not isinstance(face, dict) else face.get("bbox")
        iou = 0.0
        if cur_bbox is not None and previous_bbox is not None:
            from roop.face_reference import _bbox_iou
            iou = _bbox_iou(cur_bbox, previous_bbox)

        if best_sim >= h_thresh:
            matches = True
        elif best_sim >= l_thresh and iou >= i_thresh:
            matches = True
        else:
            matches = False
        return matches, best_sim

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
            # If identity router already locked or assigned this face
            assigned = getattr(face, "_assigned_identity", None) if not isinstance(face, dict) else face.get("_assigned_identity")
            if assigned is not None:
                any_matched = True
                break
            matched, _ = check_face_matches_target(face, target_faces, threshold=threshold)
            if matched:
                any_matched = True
                break
        if not any_matched:
            # All tracks/faces failed to match target source -> bypass!
            return True, faces

    return False, faces


# ── Heterogeneous detector offload (OpenVINO NPU / integrated GPU) ─────────
#
# The detector, its landmark models and the occlusion mask are small graphs
# that run once per face, while the swapper and the restorer are the large
# ones.  On an Intel machine with an NPU or an integrated GPU there is idle
# silicon that could take the small graphs and leave the discrete card to the
# swap and restoration models, which is where this pipeline's GPU time
# actually goes.
#
# THIS IS OFF BY DEFAULT, and the reason is worth stating rather than leaving
# as an accident:
#
#   1. Neither validated machine can execute it.  The RTX 4070 desktop and the
#      RTX 3060 laptop both run an ORT build whose providers are exactly
#      ('TensorrtExecutionProvider', 'CUDAExecutionProvider',
#      'CPUExecutionProvider') -- no OpenVINOExecutionProvider, and no
#      `openvino` package.  Every number in this repo's records comes from one
#      of those two boxes, so there is no measurement to support switching it
#      on and this project's standing rule is that a default change must prove
#      no regression.
#
#   2. The blast radius is the whole pipeline, not one stage.  Detection
#      output feeds every identity gate, the track binding and the alignment
#      matrices.  A different kernel's rounding here does not show up as a
#      slower render, it shows up as a different face being swapped -- and the
#      swap audit counts intent, not outcome, so it would read 100% either way.
#
# So this ships as a real, wired, testable path that a user on Intel hardware
# can turn on explicitly, and as an honest no-op everywhere else.  When it is
# requested but cannot be satisfied, it says so once instead of pretending.

_OPENVINO_ENV = "ROOP_DETECTOR_OPENVINO"

# Only the small per-face graphs are ever offloaded.  The swapper and the
# restorers stay on the discrete GPU: moving them is the opposite of the point.
_OFFLOADABLE = frozenset(("face_detection", "masking", "recognition"))

_openvino_state: Dict[str, Any] = {}
_openvino_lock = RLock()


def openvino_available() -> bool:
    """Whether the installed ONNX Runtime build LISTS the OpenVINO EP.

    Listed is not usable, and for this EP the gap is unusually dangerous --
    see openvino_device_usable().
    """
    try:
        from roop.backend_manager import provider_available
        return provider_available("OpenVINOExecutionProvider")
    except Exception:
        return False


def _probe_model() -> bytes:
    """A one-node ONNX graph, built in memory, for the usability probe."""
    from onnx import TensorProto, helper
    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph(
        [node], "probe",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    return model.SerializeToString()


def openvino_device_usable(device: str) -> bool:
    """Whether the OpenVINO EP genuinely ACTIVATES for *device* on this box.

    THIS EP FAILS OPEN, which is why a probe is needed rather than a listing.
    Measured 2026-09-03 against onnxruntime-openvino 1.23.0 on this machine,
    asking for a device the system does not have:

        device_type=CPU    build 0.4s  EP active   -> ran on OpenVINO
        device_type=GPU    build 0.3s  EP ABSENT   -> ran on CPUExecutionProvider
        device_type=NPU    build 0.3s  EP ABSENT   -> ran on CPUExecutionProvider
        device_type=GPU.1  build 0.4s  EP ABSENT   -> ran on CPUExecutionProvider

    In every failing row `InferenceSession` returned a working session and
    raised NOTHING; ORT logged to stderr and quietly dropped the provider.  A
    mismatched OpenVINO runtime version does the same thing.  So neither
    `build_session_with_fallback` nor any try/except can see this: the only
    reliable signal is asking the constructed session which providers it ended
    up with, which is what this does -- once per device, on a one-node graph.
    """
    key = f"usable::{device}"
    with _openvino_lock:
        if key in _openvino_state:
            return _openvino_state[key]
    usable = False
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(
            _probe_model(),
            providers=[("OpenVINOExecutionProvider", {"device_type": device}),
                       "CPUExecutionProvider"])
        usable = "OpenVINOExecutionProvider" in session.get_providers()
    except Exception:
        usable = False
    with _openvino_lock:
        _openvino_state[key] = usable
    return usable


def openvino_devices() -> Tuple[str, ...]:
    """Device strings OpenVINO reports on this machine, e.g. NPU, GPU.1, CPU.

    Queried through the ``openvino`` runtime when it is importable.  The ORT
    provider can be present without it, so an empty tuple means "cannot
    enumerate", not "no devices" -- callers treat an explicitly requested
    device as usable in that case and let the session build decide.
    """
    with _openvino_lock:
        if "devices" in _openvino_state:
            return _openvino_state["devices"]
    devices: Tuple[str, ...] = ()
    try:
        import openvino as ov
        devices = tuple(str(d) for d in ov.Core().available_devices)
    except Exception:
        devices = ()
    with _openvino_lock:
        _openvino_state["devices"] = devices
    return devices


def detector_offload_target(requested: Optional[str] = None) -> Optional[str]:
    """Resolve the configured offload device, or None when it is not usable.

    ``requested`` accepts an explicit device string ("NPU", "GPU.1"), "auto"
    to take the best available accelerator, or an off value.  Unset is off.
    """
    raw = requested if requested is not None else os.environ.get(_OPENVINO_ENV, "")
    value = str(raw or "").strip()
    if value.lower() in ("", "0", "off", "false", "no", "none"):
        return None
    if not openvino_available():
        _warn_once("openvino_missing",
                   f"[Detector] {_OPENVINO_ENV}={value} but this ONNX Runtime "
                   f"build has no OpenVINOExecutionProvider; detection stays on "
                   f"the primary provider.")
        return None
    devices = openvino_devices()
    if value.lower() in ("auto", "1", "on", "true", "yes"):
        # Prefer the NPU, then a secondary GPU.  GPU.0 is skipped deliberately:
        # OpenVINO 2026.3 on this machine enumerates GPU as
        # "NVIDIA GeForce RTX 4070 (dGPU)", i.e. the first GPU slot really can
        # be the discrete card this offload exists to keep free.  Taking it
        # would move detection onto the very device we are trying to unload,
        # through a slower runtime than the TensorRT path it already has.
        for candidate in ("NPU", "GPU.1"):
            if candidate in devices and openvino_device_usable(candidate):
                return candidate
        _warn_once("openvino_auto_none",
                   f"[Detector] {_OPENVINO_ENV}=auto found no usable NPU or "
                   f"secondary GPU (devices: {', '.join(devices) or 'none reported'}); "
                   f"detection stays on the primary provider.")
        return None
    if devices and value not in devices:
        _warn_once(f"openvino_absent_{value}",
                   f"[Detector] {_OPENVINO_ENV}={value} is not among the "
                   f"OpenVINO devices on this machine "
                   f"({', '.join(devices) or 'none reported'}); detection stays "
                   f"on the primary provider.")
        return None
    # The listing can be absent or stale, and the EP fails OPEN on a device it
    # cannot serve -- a session that silently runs on CPU while every log line
    # says the detector was offloaded. Confirm the EP actually activates for
    # this device before routing anything to it.
    if not openvino_device_usable(value):
        _warn_once(f"openvino_inactive_{value}",
                   f"[Detector] {_OPENVINO_ENV}={value}: the OpenVINO provider "
                   f"is listed but does not activate for that device (ORT would "
                   f"fall back to CPU without raising); detection stays on the "
                   f"primary provider.")
        return None
    return value


def _warn_once(key: str, message: str) -> None:
    with _openvino_lock:
        if key in _openvino_state:
            return
        _openvino_state[key] = True
    print(message)


def detector_providers(providers: Sequence, model_key: str = "face_detection",
                       requested: Optional[str] = None) -> List:
    """Prepend the OpenVINO EP for a small per-face model, when enabled.

    The primary chain is kept behind it rather than replaced, so any operator
    OpenVINO cannot take still falls back to CUDA/CPU inside the same session.
    Models outside ``_OFFLOADABLE`` are returned untouched.
    """
    chain = list(providers or ())
    family = str(model_key or "").split(":", 1)[0].strip().lower()
    if family not in _OFFLOADABLE:
        return chain
    device = detector_offload_target(requested)
    if device is None:
        return chain
    if any("openvino" in str(p[0] if isinstance(p, (tuple, list)) else p).lower()
           for p in chain):
        return chain
    options = {"device_type": device}
    precision = os.environ.get("ROOP_DETECTOR_OPENVINO_PRECISION", "").strip()
    if precision:
        options["precision"] = precision
    _warn_once(f"openvino_active_{device}_{family}",
               f"[Detector] routing {family} models to OpenVINO {device}; the "
               f"primary provider is retained as in-session fallback.")
    return [("OpenVINOExecutionProvider", options)] + chain


def offload_report(requested: Optional[str] = None) -> dict:
    """JSON-safe diagnostics for the panel and the self-verification test."""
    target = detector_offload_target(requested)
    return {
        "env": _OPENVINO_ENV,
        "configured": (requested if requested is not None
                       else os.environ.get(_OPENVINO_ENV, "") or "off"),
        "provider_available": openvino_available(),
        "devices": list(openvino_devices()),
        "target": target,
        "offloadable_models": sorted(_OFFLOADABLE),
        "active": target is not None,
    }


# =============================================================================
# Temporal Tracking & Dynamic Detection Intervals
# =============================================================================

def compute_histogram_signature(frame: Frame) -> Optional[np.ndarray]:
    """Compute normalized 3D color histogram for fast scene cut detection."""
    if frame is None:
        return None
    try:
        h, w = frame.shape[:2]
        if max(h, w) > 160:
            small = cv2.resize(frame, (160, int(round(160 * h / w))), interpolation=cv2.INTER_AREA)
        else:
            small = frame
        hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)
        return hist.astype(np.float32)
    except Exception:
        return None


def compare_histogram_difference(hist1: Optional[np.ndarray], hist2: Optional[np.ndarray]) -> float:
    """Bhattacharyya histogram difference in [0, 1] range (0 = identical, 1 = disjoint)."""
    if hist1 is None or hist2 is None:
        return 1.0
    try:
        return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA))
    except Exception:
        return 1.0


def scale_face_coordinates(face: Any, inv_scale: float) -> None:
    """Scale spatial fields (bbox, kps, landmarks) uniformly by inv_scale."""
    for attr in ('bbox', 'kps', 'landmark_2d_106'):
        v = _face_field(face, attr)
        if v is not None:
            try:
                scaled = (np.asarray(v, dtype=np.float32) * inv_scale).copy()
                _set_face_field(face, attr, scaled)
            except Exception:
                pass
    lm68 = _face_field(face, 'landmark_3d_68')
    if lm68 is not None:
        try:
            lm68_s = np.asarray(lm68, dtype=np.float32).copy()
            lm68_s[:, :2] *= inv_scale
            _set_face_field(face, 'landmark_3d_68', lm68_s)
        except Exception:
            pass


def detect_faces_half_res(frame: Frame, max_dim: int = 640) -> List[Face]:
    """Downsample 1080p/4K input frames to max dimension 640px solely for the
    detector forward pass, then scale landmark coordinates back up.
    Reduces detection inference latency by ~4x without compromising landmark precision.
    """
    if frame is None:
        return []
    h, w = frame.shape[:2]
    max_side = max(h, w)
    if max_side <= max_dim:
        return get_all_faces(frame) or []

    scale = float(max_dim) / float(max_side)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    downscaled = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    faces = get_all_faces(downscaled)
    if not faces:
        # Fallback to native resolution if downscaled detect missed small face
        return get_all_faces(frame) or []

    inv_scale = 1.0 / scale
    for f in faces:
        scale_face_coordinates(f, inv_scale)

    return faces or []


class TemporalFaceDetector:
    """Temporal tracking with dynamic detection intervals and scene cut reset.

    Features:
    1. Half-resolution detection: Downsamples 1080p/4K frames to max dimension 640px for
       full detector passes, reducing inference latency by ~4x while preserving full-resolution
       landmark precision.
    2. Dynamic detection intervals:
       - Frame 0: Run full face detection and populate tracklet embeddings.
       - Frames 1..4: Predict face bounding boxes from previous frame with 15% safety margin crop;
         skips the full-frame detector pass entirely.
       - Frame 5 (or if tracking confidence / IoU drops below 0.65): Re-trigger full-frame detection
         to correct drift.
    3. Scene cut detection:
       - Immediately resets tracklets and triggers full detection whenever color histogram
         difference > 0.4.
    """

    def __init__(
        self,
        interval: int = 5,
        max_det_dim: int = 640,
        scene_cut_thresh: float = 0.4,
        confidence_thresh: float = 0.65,
        iou_thresh: float = 0.65,
        safety_margin: float = 0.15,
        max_age: int = 15,
    ):
        self.interval = int(interval)
        self.max_det_dim = int(max_det_dim)
        self.scene_cut_thresh = float(scene_cut_thresh)
        self.confidence_thresh = float(confidence_thresh)
        self.iou_thresh = float(iou_thresh)
        self.safety_margin = float(safety_margin)

        self.tracker = FaceTracker(max_age=max_age, max_cost=0.65)
        self.frame_count: int = 0
        self.frames_since_full_detect: int = 0
        self.last_frame_index: Optional[int] = None
        self.prev_hist: Optional[np.ndarray] = None
        self.last_faces: List[Face] = []
        self._lock = RLock()

    def reset(self) -> None:
        """Reset temporal tracking state and scene history."""
        with self._lock:
            self.tracker.reset()
            self.frame_count = 0
            self.frames_since_full_detect = 0
            self.last_frame_index = None
            self.prev_hist = None
            self.last_faces = []

    def detect(
        self,
        frame: Frame,
        frame_index: Optional[int] = None,
        force_detect: bool = False
    ) -> List[Face]:
        """Detect and track faces across sequential video frames."""
        if frame is None:
            return []

        with self._lock:
            if frame_index is not None:
                current_idx = int(frame_index)
                if self.last_frame_index is not None and abs(current_idx - self.last_frame_index) > 1:
                    # Non-sequential seek or jump -> force full detection
                    force_detect = True
                self.last_frame_index = current_idx
            else:
                current_idx = self.frame_count
                self.last_frame_index = current_idx

            # Scene cut detection
            curr_hist = compute_histogram_signature(frame)
            is_cut = False
            if self.prev_hist is not None and curr_hist is not None:
                diff = compare_histogram_difference(self.prev_hist, curr_hist)
                if diff > self.scene_cut_thresh:
                    is_cut = True
                    self.reset()
            self.prev_hist = curr_hist

            # Determine whether full detection is required
            needs_full_detect = (
                force_detect or
                is_cut or
                (current_idx % self.interval == 0) or
                (self.frames_since_full_detect >= self.interval - 1 and self.frame_count > 0) or
                not self.last_faces or
                len(self.tracker.tracks) == 0
            )

            if not needs_full_detect:
                # Frames 1 through 4: Predict face bounding boxes with 15% safety margin crop
                verified = self._predict_and_verify(frame, current_idx)
                if verified is not None:
                    self.frame_count += 1
                    self.frames_since_full_detect += 1
                    self.last_faces = sorted(verified, key=lambda x: _face_field(x, 'bbox', [0])[0])
                    return self.last_faces
                # If tracking confidence / IoU drops below 0.65, fall through to re-trigger full detection

            # Full-frame detection (Half-Resolution optimized)
            raw_faces = detect_faces_half_res(frame, max_dim=self.max_det_dim)
            tracked_faces = self.tracker.update(raw_faces, frame_index=current_idx)
            self.frame_count += 1
            self.frames_since_full_detect = 0
            self.last_faces = sorted(tracked_faces, key=lambda x: _face_field(x, 'bbox', [0])[0])
            return self.last_faces

    def _predict_and_verify(self, frame: Frame, current_idx: int) -> Optional[List[Face]]:
        """Predict face bounding boxes with 15% safety margin crop.
        Returns predicted faces populated with tracklet embeddings from Frame 0.
        Returns None if tracking confidence / IoU drops below 0.65, re-triggering full-frame detection.
        """
        if not self.last_faces or not self.tracker.tracks:
            return None

        h, w = frame.shape[:2]
        predicted_faces = []

        for face in self.last_faces:
            tid = _face_field(face, '_track_id')
            if tid is None or tid not in self.tracker.tracks:
                return None

            track = self.tracker.tracks[tid]
            # Predict Kalman bounding box in center/aspect/height space
            pred_measurement = track.state[:4].copy() + track.velocity
            pred_box = FaceTracker.measurement_to_bbox(pred_measurement)

            # Validate box dimensions
            bw = float(pred_box[2] - pred_box[0])
            bh = float(pred_box[3] - pred_box[1])
            if bw <= 10.0 or bh <= 10.0:
                return None

            # Calculate visible ratio / IoU with frame canvas to detect rapid pans off-screen
            ix1 = max(0.0, float(pred_box[0]))
            iy1 = max(0.0, float(pred_box[1]))
            ix2 = min(float(w), float(pred_box[2]))
            iy2 = min(float(h), float(pred_box[3]))
            inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            box_area = bw * bh
            visible_ratio = inter_area / max(box_area, 1e-6)

            if visible_ratio < self.iou_thresh:
                # Face panned out of frame or off-screen -> re-trigger full detection
                return None

            # 15% safety margin crop bounds
            pad_w = bw * self.safety_margin
            pad_h = bh * self.safety_margin
            cx1 = max(0, int(round(pred_box[0] - pad_w)))
            cy1 = max(0, int(round(pred_box[1] - pad_h)))
            cx2 = min(w, int(round(pred_box[2] + pad_w)))
            cy2 = min(h, int(round(pred_box[3] + pad_h)))
            if cx2 <= cx1 or cy2 <= cy1:
                return None

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0 or float(np.std(crop)) < 6.0:
                return None  # Blank or completely uniform crop -> re-trigger full detection

            # Calculate frame-to-frame displacement
            last_box = _face_field(face, 'bbox')
            dx, dy = 0.0, 0.0
            if last_box is not None:
                dx = float((pred_box[0] + pred_box[2] - last_box[0] - last_box[2]) * 0.5)
                dy = float((pred_box[1] + pred_box[3] - last_box[1] - last_box[3]) * 0.5)

            # Propagate 5-point keypoints with displacement
            pred_kps = None
            old_kps = _face_field(face, 'kps')
            if old_kps is not None:
                pred_kps = np.asarray(old_kps, dtype=np.float32).copy()
                pred_kps[:, 0] += dx
                pred_kps[:, 1] += dy

            # Confidence check with mild temporal decay
            conf = float(_face_field(face, 'det_score', _face_field(face, 'score', 0.95)))
            conf *= 0.96
            if conf < self.confidence_thresh:
                # Confidence dropped below 0.65 -> re-trigger full detection
                return None

            # Construct predicted Face object
            pred_face = dict(face) if isinstance(face, dict) else {
                k: getattr(face, k) for k in dir(face) if not k.startswith('__')
            }
            pred_face['bbox'] = np.asarray(pred_box, dtype=np.float32)
            if pred_kps is not None:
                pred_face['kps'] = pred_kps
            pred_face['det_score'] = conf
            pred_face['_track_id'] = tid
            # Retain tracklet embedding populated at Frame 0
            if track.embedding is not None:
                pred_face['embedding'] = track.embedding

            # Shift 68/106 landmarks if present
            lm106 = _face_field(face, 'landmark_2d_106')
            if lm106 is not None:
                pred_lm106 = np.asarray(lm106, dtype=np.float32).copy()
                pred_lm106[:, 0] += dx
                pred_lm106[:, 1] += dy
                pred_face['landmark_2d_106'] = pred_lm106

            lm68 = _face_field(face, 'landmark_3d_68')
            if lm68 is not None:
                pred_lm68 = np.asarray(lm68, dtype=np.float32).copy()
                pred_lm68[:, 0] += dx
                pred_lm68[:, 1] += dy
                pred_face['landmark_3d_68'] = pred_lm68

            # Update track state with predicted measurement
            track.state[:4] = pred_measurement
            track.confidence = conf
            predicted_faces.append(pred_face)

        return predicted_faces


DEFAULT_TEMPORAL_DETECTOR = TemporalFaceDetector()


def reset_temporal_detector() -> None:
    """Reset the global temporal face detector state."""
    DEFAULT_TEMPORAL_DETECTOR.reset()


def get_many_faces(
    frame: Frame,
    frame_index: Optional[int] = None,
    force_detect: bool = False,
    detector: Optional[TemporalFaceDetector] = None
) -> List[Face]:
    """Detect and track multiple faces across sequential video frames with dynamic intervals.

    Drop-in replacement for get_all_faces / get_many_faces.
    Optimizations:
    - Half-resolution detection on 1080p/4K frames (downsampled to 640px, ~4x speedup).
    - Dynamic intervals: full detection on Frame 0, then 15% crop prediction on Frames 1..4.
    - Automatic drift correction on Frame 5 or when IoU/confidence < 0.65.
    - Scene cut detection: resets on histogram difference > 0.4.
    """
    det = detector if detector is not None else DEFAULT_TEMPORAL_DETECTOR
    return det.detect(frame, frame_index=frame_index, force_detect=force_detect)


def find_similar_faces(
    frame: Frame,
    reference_faces: List[Any],
    threshold: Optional[float] = None,
    frame_index: Optional[int] = None,
    force_detect: bool = False,
    detector: Optional[TemporalFaceDetector] = None
) -> List[Face]:
    """Detect faces in frame and return those matching reference target faces.

    Drop-in replacement for find_similar_faces.
    Leverages temporal tracking and dynamic detection intervals for high throughput,
    with dual-threshold hysteresis matching against reference faces.
    """
    if frame is None:
        return []
    faces = get_many_faces(frame, frame_index=frame_index, force_detect=force_detect, detector=detector)
    if not faces:
        return []
    if not reference_faces:
        return faces

    matching_faces = []
    for face in faces:
        matched, _ = check_face_matches_target(face, reference_faces, threshold=threshold)
        if matched:
            matching_faces.append(face)

    return matching_faces
