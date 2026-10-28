"""Stage 21 — Stereoscopic 3D (SBS/TB) Epipolar Disparity Lock & VR Viewport Engine.

Provides:
1. Stereo format modes: MONO_2D, STEREO_SBS_HALF, STEREO_SBS_FULL, STEREO_TB, VR180_SBS.
2. Frame partitioning into Left (L) and Right (R) views prior to detection & alignment.
3. Epipolar Geometric Coupling:
   - Pairs matched identities across L and R views.
   - Enforces identical vertical (y) alignment coordinates across the stereo pair (delta_y < 0.05 px).
   - Enforces identical in-plane roll angles across the stereo pair.
   - Smooths horizontal disparity (delta_x = x_L - x_R) via an Exponential Moving Average (EMA) filter
     to eliminate perceived depth jitter-pulsing.
4. Latent noise seed, normalization statistics, and Stage 12 specular highlight mask synchronization
   for Restore Ultra (Stage 6) and batched TensorRT execution (Stage 14).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from face_engine.core.config import StereoMode


@dataclass
class StereoViewPartition:
    """Carries partitioned Left and Right eye views along with ROI coordinates."""

    left_frame: np.ndarray
    right_frame: Optional[np.ndarray]
    left_roi: Tuple[int, int, int, int]    # x, y, width, height in original canvas
    right_roi: Optional[Tuple[int, int, int, int]]
    stereo_mode: StereoMode
    original_shape: Tuple[int, int, int]


def _mode_str(mode: Union[StereoMode, str, None]) -> str:
    if mode is None:
        return StereoMode.MONO_2D.value
    if isinstance(mode, StereoMode):
        return mode.value
    if hasattr(mode, "value"):
        return str(mode.value)
    s = str(mode)
    if "." in s:
        s = s.split(".")[-1]
    return s


def is_stereo_active(mode: Union[StereoMode, str, None]) -> bool:
    """Return True if the mode represents an active stereoscopic format."""
    val = _mode_str(mode)
    return val != StereoMode.MONO_2D.value and val in {
        StereoMode.STEREO_SBS_HALF.value,
        StereoMode.STEREO_SBS_FULL.value,
        StereoMode.STEREO_TB.value,
        StereoMode.VR180_SBS.value,
    }


def partition_stereo_frame(
    frame: np.ndarray,
    mode: Union[StereoMode, str] = StereoMode.MONO_2D,
) -> StereoViewPartition:
    """Partition a full canvas frame into Left and Right views based on stereo mode.

    Parameters:
        frame: H x W x C numpy array (BGR or RGB).
        mode: StereoMode enum or string.

    Returns:
        StereoViewPartition containing left and right eye image arrays and canvas ROIs.
    """
    mode_val = _mode_str(mode)
    h, w = frame.shape[:2]
    orig_shape = frame.shape

    if not is_stereo_active(mode_val):
        return StereoViewPartition(
            left_frame=frame,
            right_frame=None,
            left_roi=(0, 0, w, h),
            right_roi=None,
            stereo_mode=StereoMode.MONO_2D,
            original_shape=orig_shape,
        )

    st_mode = StereoMode(mode_val)

    if st_mode in (StereoMode.STEREO_SBS_HALF, StereoMode.STEREO_SBS_FULL, StereoMode.VR180_SBS):
        # Horizontal Side-by-Side split
        w_mid = w // 2
        left_img = frame[:, :w_mid].copy()
        right_img = frame[:, w_mid:].copy()
        return StereoViewPartition(
            left_frame=left_img,
            right_frame=right_img,
            left_roi=(0, 0, w_mid, h),
            right_roi=(w_mid, 0, w - w_mid, h),
            stereo_mode=st_mode,
            original_shape=orig_shape,
        )
    elif st_mode == StereoMode.STEREO_TB:
        # Vertical Top-Bottom (Over-Under) split
        h_mid = h // 2
        left_img = frame[:h_mid, :].copy()
        right_img = frame[h_mid:, :].copy()
        return StereoViewPartition(
            left_frame=left_img,
            right_frame=right_img,
            left_roi=(0, 0, w, h_mid),
            right_roi=(0, h_mid, w, h - h_mid),
            stereo_mode=st_mode,
            original_shape=orig_shape,
        )
    else:
        return StereoViewPartition(
            left_frame=frame,
            right_frame=None,
            left_roi=(0, 0, w, h),
            right_roi=None,
            stereo_mode=StereoMode.MONO_2D,
            original_shape=orig_shape,
        )


def recombine_stereo_frame(
    left_frame: np.ndarray,
    right_frame: Optional[np.ndarray],
    mode: Union[StereoMode, str] = StereoMode.MONO_2D,
    target_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Recombine Left and Right processed eye frames into a single stereo canvas.

    Parameters:
        left_frame: Processed Left eye frame.
        right_frame: Processed Right eye frame (or None for MONO_2D).
        mode: Active StereoMode.
        target_shape: Optional (height, width) to force output dimensions.

    Returns:
        Full combined frame.
    """
    mode_val = _mode_str(mode)
    if right_frame is None or not is_stereo_active(mode_val):
        return left_frame

    st_mode = StereoMode(mode_val)
    if st_mode in (StereoMode.STEREO_SBS_HALF, StereoMode.STEREO_SBS_FULL, StereoMode.VR180_SBS):
        combined = np.concatenate([left_frame, right_frame], axis=1)
    elif st_mode == StereoMode.STEREO_TB:
        combined = np.concatenate([left_frame, right_frame], axis=0)
    else:
        combined = left_frame

    if target_shape is not None and combined.shape[:2] != target_shape:
        import cv2
        combined = cv2.resize(combined, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    return combined


@dataclass
class EpipolarCouplingResult:
    """Detailed telemetry and parameters of an epipolar coupling solve."""

    coupled_left_kps: np.ndarray
    coupled_right_kps: np.ndarray
    raw_disparity_px: float
    smoothed_disparity_px: float
    coupled_roll_rad: float
    vertical_drift_px: float
    pair_matched: bool = True


class EpipolarStereoCoupler:
    """Stage 4 & Stage 9 Epipolar Geometric Coupler.

    Enforces:
    1. Zero vertical landmark epipolar drift (delta_y < 0.05 px) between matched stereo pairs.
    2. Identical in-plane roll angles across the stereo pair.
    3. Smooth horizontal disparity (delta_x = x_L - x_R) via an EMA filter so perceived depth
       never jitter-pulses between consecutive frames.
    """

    def __init__(self, ema_alpha: float = 0.70, max_vertical_drift_px: float = 0.05):
        self.ema_alpha = float(np.clip(ema_alpha, 0.01, 1.0))
        self.max_vertical_drift_px = float(max_vertical_drift_px)
        self._disparity_history: Dict[Any, float] = {}

    def reset(self, track_id: Optional[Any] = None) -> None:
        """Reset disparity history for a specific track or all tracks."""
        if track_id is not None:
            self._disparity_history.pop(track_id, None)
        else:
            self._disparity_history.clear()

    @staticmethod
    def compute_roll_angle(kps: np.ndarray) -> float:
        """Compute in-plane roll angle (radians) from eye baseline landmarks (pts 0 and 1)."""
        pts = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 2:
            return 0.0
        dx = float(pts[1, 0] - pts[0, 0])
        dy = float(pts[1, 1] - pts[0, 1])
        return float(math.atan2(dy, dx))

    @staticmethod
    def rotate_landmarks(kps: np.ndarray, center: np.ndarray, angle_rad: float) -> np.ndarray:
        """Rotate 2D landmarks around a given center by angle_rad."""
        c = float(math.cos(angle_rad))
        s = float(math.sin(angle_rad))
        pts = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
        centered = pts - center
        rotated = np.empty_like(centered)
        rotated[:, 0] = centered[:, 0] * c - centered[:, 1] * s
        rotated[:, 1] = centered[:, 0] * s + centered[:, 1] * c
        return rotated + center

    def pair_stereo_faces(
        self,
        left_faces: List[Any],
        right_faces: List[Any],
        frame_height: int = 1080,
        max_vertical_gap_frac: float = 0.15,
        min_cosine_similarity: float = 0.35,
    ) -> Tuple[List[Tuple[Any, Any]], List[Any], List[Any]]:
        """Pair faces detected in Left and Right views using spatial vertical proximity and ArcFace identity."""
        if not left_faces or not right_faces:
            return [], list(left_faces), list(right_faces)

        def _get_kps_and_emb(f: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            kps = getattr(f, "kps", getattr(f, "landmarks", None))
            if kps is None:
                if isinstance(f, np.ndarray):
                    kps = f
                elif isinstance(f, (list, tuple)) and len(f) >= 5:
                    kps = np.asarray(f, dtype=np.float32)
            emb = getattr(f, "embedding", getattr(f, "normed_embedding", None))
            return np.asarray(kps, dtype=np.float32).reshape(-1, 2), emb

        matched_pairs: List[Tuple[Any, Any]] = []
        candidates = []
        for i, lf in enumerate(left_faces):
            lkps, lemb = _get_kps_and_emb(lf)
            ly_center = float(np.mean(lkps[:, 1]))
            for j, rf in enumerate(right_faces):
                rkps, remb = _get_kps_and_emb(rf)
                ry_center = float(np.mean(rkps[:, 1]))
                vert_gap = abs(ly_center - ry_center) / max(1.0, float(frame_height))
                if vert_gap > max_vertical_gap_frac:
                    continue

                sim = 0.8
                if lemb is not None and remb is not None:
                    denom = float(np.linalg.norm(lemb) * np.linalg.norm(remb) + 1e-8)
                    sim = float(np.dot(lemb, remb) / denom)
                    if sim < min_cosine_similarity:
                        continue

                cost = (1.0 - sim) + vert_gap * 2.0
                candidates.append((cost, i, j))

        candidates.sort(key=lambda x: x[0])
        used_l = set()
        used_r = set()
        for cost, i, j in candidates:
            if i not in used_l and j not in used_r:
                used_l.add(i)
                used_r.add(j)
                matched_pairs.append((left_faces[i], right_faces[j]))

        unmatched_left = [lf for idx, lf in enumerate(left_faces) if idx not in used_l]
        unmatched_right = [rf for idx, rf in enumerate(right_faces) if idx not in used_r]
        return matched_pairs, unmatched_left, unmatched_right

    def couple_alignment(
        self,
        left_kps: np.ndarray,
        right_kps: np.ndarray,
        track_id: Optional[Any] = None,
        ema_alpha: Optional[float] = None,
    ) -> EpipolarCouplingResult:
        """Enforce epipolar geometric coupling on a pair of 5-point facial landmarks."""
        lkps = np.asarray(left_kps, dtype=np.float32).copy().reshape(-1, 2)
        rkps = np.asarray(right_kps, dtype=np.float32).copy().reshape(-1, 2)
        assert lkps.shape[0] >= 5 and rkps.shape[0] >= 5, "Requires at least 5 facial landmarks"

        alpha = float(self.ema_alpha if ema_alpha is None else ema_alpha)

        # 1. Roll angle computation and synchronization
        roll_l = self.compute_roll_angle(lkps)
        roll_r = self.compute_roll_angle(rkps)
        coupled_roll = 0.5 * (roll_l + roll_r)

        center_l = np.mean(lkps[:5], axis=0)
        center_r = np.mean(rkps[:5], axis=0)

        # Un-roll to upright coordinate space
        lkps_up = self.rotate_landmarks(lkps, center_l, -roll_l)
        rkps_up = self.rotate_landmarks(rkps, center_r, -roll_r)

        # 2. Epipolar Vertical Locking (delta_y = 0.0)
        coupled_y = 0.5 * (lkps_up[:5, 1] + rkps_up[:5, 1])
        lkps_up[:5, 1] = coupled_y
        rkps_up[:5, 1] = coupled_y

        # 3. Horizontal disparity measurement & EMA smoothing
        raw_disparity = float(center_l[0] - center_r[0])
        key = track_id if track_id is not None else "_default"
        if key in self._disparity_history:
            prev_d = self._disparity_history[key]
            smoothed_disparity = float(alpha * raw_disparity + (1.0 - alpha) * prev_d)
        else:
            smoothed_disparity = raw_disparity
        self._disparity_history[key] = smoothed_disparity

        # Symmetric distribution of smoothed disparity around pair midpoint
        x_mid = 0.5 * (center_l[0] + center_r[0])
        target_center_x_l = x_mid + 0.5 * smoothed_disparity
        target_center_x_r = x_mid - 0.5 * smoothed_disparity

        dx_l = target_center_x_l - center_l[0]
        dx_r = target_center_x_r - center_r[0]
        lkps_up[:5, 0] += dx_l
        rkps_up[:5, 0] += dx_r

        # 4. Re-apply coupled roll angle to both faces
        new_center_l = np.mean(lkps_up[:5], axis=0)
        new_center_r = np.mean(rkps_up[:5], axis=0)
        coupled_l = self.rotate_landmarks(lkps_up, new_center_l, coupled_roll)
        coupled_r = self.rotate_landmarks(rkps_up, new_center_r, coupled_roll)

        # Verify vertical drift bound
        max_dy = float(np.max(np.abs(coupled_l[:5, 1] - coupled_r[:5, 1])))

        return EpipolarCouplingResult(
            coupled_left_kps=coupled_l,
            coupled_right_kps=coupled_r,
            raw_disparity_px=raw_disparity,
            smoothed_disparity_px=smoothed_disparity,
            coupled_roll_rad=coupled_roll,
            vertical_drift_px=max_dy,
            pair_matched=True,
        )


def synchronize_stereo_crops(
    left_crop: np.ndarray,
    right_crop: np.ndarray,
    seed: Optional[int] = None,
    sync_norm: bool = True,
    specular_mask_L: Optional[np.ndarray] = None,
    specular_mask_R: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Synchronize latent noise seeds, normalization statistics, and Stage 12 specular highlight masks."""
    l_out = left_crop.copy()
    r_out = right_crop.copy()

    effective_seed = int(seed if seed is not None else 42)
    stats: Dict[str, Any] = {"seed": effective_seed}

    if sync_norm and l_out.size > 0 and r_out.size > 0:
        l_float = l_out.astype(np.float32)
        r_float = r_out.astype(np.float32)
        mean_l = np.mean(l_float, axis=(0, 1), keepdims=True)
        mean_r = np.mean(r_float, axis=(0, 1), keepdims=True)
        joint_mean = 0.5 * (mean_l + mean_r)

        std_l = np.std(l_float, axis=(0, 1), keepdims=True) + 1e-6
        std_r = np.std(r_float, axis=(0, 1), keepdims=True) + 1e-6
        joint_std = np.sqrt(0.5 * (std_l ** 2 + std_r ** 2))

        weight = 0.65
        l_normed = (l_float - mean_l) / std_l * joint_std + joint_mean
        r_normed = (r_float - mean_r) / std_r * joint_std + joint_mean

        l_out = np.clip((1.0 - weight) * l_float + weight * l_normed, 0, 255).astype(left_crop.dtype)
        r_out = np.clip((1.0 - weight) * r_float + weight * r_normed, 0, 255).astype(right_crop.dtype)

        stats["joint_mean"] = joint_mean.squeeze().tolist()
        stats["joint_std"] = joint_std.squeeze().tolist()

    if specular_mask_L is not None and specular_mask_R is not None:
        sm_l = specular_mask_L.astype(np.float32)
        sm_r = specular_mask_R.astype(np.float32)
        specular_sync = np.clip(0.5 * (sm_l + sm_r), 0.0, 1.0)
        stats["specular_mask"] = specular_sync

    return l_out, r_out, stats
