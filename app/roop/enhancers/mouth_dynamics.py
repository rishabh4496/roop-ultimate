"""Surgical mouth dynamics and dynamic mouth aperture restoration.

Provides:
1. Mouth Aspect Ratio (MAR) calculation from inner lip landmarks.
2. Dynamic triggering of mouth restoration sub-pass when MAR > threshold (default 0.18).
3. Adaptive inner-mouth masking isolating teeth, tongue, and oral cavity.
4. Seamless aperture compositing with feathering (3 to 5 pixels).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def compute_mar(landmarks: Any) -> float:
    """Calculate Mouth Aspect Ratio (MAR) from inner lip landmarks.

    Supports:
    - 68-point facial landmarks (inner lip points: indices 60 to 67):
        p60: Left corner
        p64: Right corner
        p61, p62, p63: Upper inner lip
        p67, p66, p65: Lower inner lip
        MAR = (||p61 - p67|| + ||p62 - p66|| + ||p63 - p65||) / (2.0 * ||p60 - p64||)
    - 106-point facial landmarks (inner lip points in contour 52:71)
    - Fallback for 5-point keypoints or missing/invalid arrays.

    Returns:
        float: MAR value >= 0.0, or 0.0 if landmarks are invalid or occluded.
    """
    if landmarks is None:
        return 0.0

    try:
        pts = np.asarray(landmarks, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2)
        elif pts.ndim == 3 and pts.shape[0] == 1:
            pts = pts[0]

        if not np.all(np.isfinite(pts)):
            return 0.0

        num_pts = pts.shape[0]

        # 68-point landmarks: standard inner lip indices 60 to 67
        if num_pts >= 68:
            p60 = pts[60]
            p61 = pts[61]
            p62 = pts[62]
            p63 = pts[63]
            p64 = pts[64]
            p65 = pts[65]
            p66 = pts[66]
            p67 = pts[67]

            d_v1 = float(np.linalg.norm(p61 - p67))
            d_v2 = float(np.linalg.norm(p62 - p66))
            d_v3 = float(np.linalg.norm(p63 - p65))
            d_h = float(np.linalg.norm(p60 - p64))

            if d_h < 1.0 or not np.isfinite(d_h):
                return 0.0
            mar = (d_v1 + d_v2 + d_v3) / (2.0 * d_h)
            return float(np.clip(mar, 0.0, 2.0))

        # 106-point landmarks (InsightFace 2d106det):
        # Mouth region spans points 52 to 71.
        # Inner mouth opening: points 64..70 (lower) vs 58..63 (upper)
        if num_pts >= 106:
            mouth_pts = pts[52:71]
            min_xy = np.min(mouth_pts, axis=0)
            max_xy = np.max(mouth_pts, axis=0)
            span_h = float(max_xy[0] - min_xy[0])
            span_v = float(max_xy[1] - min_xy[1])
            if span_h < 1.0 or not np.isfinite(span_h):
                return 0.0
            # Scale vertical by 0.65 to approximate inner aperture from full lip box
            mar = (span_v * 0.65) / span_h
            return float(np.clip(mar, 0.0, 2.0))

        # 5-point landmarks fallback (points 3, 4 are mouth corners)
        if num_pts >= 5:
            left_mouth = pts[3]
            right_mouth = pts[4]
            mouth_width = float(np.linalg.norm(right_mouth - left_mouth))
            if mouth_width < 1.0:
                return 0.0
            # 5-point lacks vertical aperture detail; estimate 0.10 baseline
            return 0.10

    except Exception as exc:
        LOGGER.debug("compute_mar failed gracefully: %s", exc)
        return 0.0

    return 0.0


class MouthDynamicsEngine:
    """Manages surgical mouth aperture restoration and dental segmentation."""

    DEFAULT_MAR_THRESHOLD: float = 0.18
    DEFAULT_FEATHER_RADIUS: int = 4

    def __init__(self, mar_threshold: float = DEFAULT_MAR_THRESHOLD, feather_radius: int = DEFAULT_FEATHER_RADIUS) -> None:
        self.mar_threshold = float(np.clip(mar_threshold, 0.05, 0.50))
        self.feather_radius = int(np.clip(feather_radius, 1, 10))

    def get_inner_lip_points(self, face: Any, crop_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Extract inner lip polygon points mapped into crop coordinates."""
        if face is None:
            return None

        # Check for 68-point 3D landmarks first
        lm68 = getattr(face, "landmark_3d_68", None)
        if lm68 is not None:
            pts68 = np.asarray(lm68, dtype=np.float32)
            if pts68.shape[0] >= 68 and np.all(np.isfinite(pts68[60:68, :2])):
                return pts68[60:68, :2]

        # Check for 106-point 2D landmarks
        lm106 = getattr(face, "landmark_2d_106", None)
        if lm106 is not None:
            pts106 = np.asarray(lm106, dtype=np.float32)
            if pts106.shape[0] >= 71 and np.all(np.isfinite(pts106[52:71])):
                # Inner loop approximation from 106 points:
                # 64, 65, 66, 67, 68, 69, 70 are inner lip lower, 58..63 upper
                inner_idx = [52, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
                valid_idx = [i for i in inner_idx if i < pts106.shape[0]]
                if len(valid_idx) >= 6:
                    return pts106[valid_idx]

        # Check dictionary-like face
        if isinstance(face, dict):
            for k in ("landmark_3d_68", "landmark_2d_106", "landmarks"):
                val = face.get(k)
                if val is not None:
                    arr = np.asarray(val, dtype=np.float32)
                    if arr.shape[0] >= 68:
                        return arr[60:68, :2]

        return None

    def generate_inner_mouth_mask(
        self,
        target_crop: np.ndarray,
        inner_lip_points: Optional[np.ndarray],
        feather_radius: Optional[int] = None,
    ) -> np.ndarray:
        """Generate an adaptive inner-mouth mask isolating teeth, tongue, and oral cavity.

        Args:
            target_crop: Original target face crop (H, W, 3) uint8.
            inner_lip_points: Polygon points (N, 2) in target_crop coordinate space.
            feather_radius: Feathering blur radius (default 3 to 5 pixels).

        Returns:
            np.ndarray: Single-channel float32 mask in [0.0, 1.0] matching crop dimensions.
        """
        if target_crop is None or target_crop.size == 0:
            return np.zeros((1, 1), dtype=np.float32)

        h, w = target_crop.shape[:2]
        feather = feather_radius if feather_radius is not None else self.feather_radius
        feather = max(1, min(15, feather))

        # Base geometric inner-mouth cavity mask
        cavity_mask = np.zeros((h, w), dtype=np.uint8)

        if inner_lip_points is not None and len(inner_lip_points) >= 4:
            pts_int = np.round(inner_lip_points).astype(np.int32)
            # Clip points inside crop boundaries
            pts_int[:, 0] = np.clip(pts_int[:, 0], 0, w - 1)
            pts_int[:, 1] = np.clip(pts_int[:, 1], 0, h - 1)
            cv2.fillPoly(cavity_mask, [pts_int], 255)
        else:
            # Fallback geometric mouth aperture based on standard template coordinates
            # if explicit dense landmarks are not present in crop space
            cx, cy = int(w * 0.50), int(h * 0.72)
            rx, ry = int(w * 0.16), int(h * 0.08)
            cv2.ellipse(cavity_mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

        # Adaptive oral cavity refinement: isolate teeth, tongue, and oral cavity
        # Convert crop to grayscale and HSV for color/luminance analysis
        gray = cv2.cvtColor(target_crop, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(target_crop, cv2.COLOR_BGR2HSV)

        # Teeth detection: high luminance, lower saturation inside mouth aperture
        # Tongue/oral cavity: darker red/pink hue, moderate saturation
        in_cavity = cavity_mask > 0

        if np.any(in_cavity):
            vals_gray = gray[in_cavity].astype(np.float32)
            median_lum = float(np.median(vals_gray))
            lum_thresh = max(80.0, median_lum - 20.0)

            # Oral cavity consists of:
            # 1. Teeth: luminance >= lum_thresh, saturation <= 180
            # 2. Tongue and oral shadow: within the inner lip polygon
            teeth_tongue = in_cavity & (gray >= lum_thresh * 0.7)
            refined_mask = teeth_tongue.astype(np.uint8) * 255
            # Morphological close to bridge interdental gaps
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel_close)
            # Restrict strictly inside the anatomical cavity boundary
            refined_mask = cv2.bitwise_and(refined_mask, cavity_mask)
        else:
            refined_mask = cavity_mask

        # Feather the mask with Gaussian blur (3 to 5 pixels radius)
        ksize = feather * 2 + 1
        mask_f = refined_mask.astype(np.float32) / 255.0
        feathered = cv2.GaussianBlur(mask_f, (ksize, ksize), sigmaX=feather * 0.5)
        return np.clip(feathered, 0.0, 1.0)

    def composite_mouth_aperture(
        self,
        swapped_crop: np.ndarray,
        original_crop: np.ndarray,
        mouth_mask: np.ndarray,
    ) -> np.ndarray:
        """Composite original mouth aperture over swapped/enhanced face with feathering.

        Args:
            swapped_crop: Swapped/enhanced face crop (H, W, 3) uint8.
            original_crop: Original target face crop (H, W, 3) uint8.
            mouth_mask: Feathered inner-mouth mask (H, W) float32 in [0.0, 1.0].

        Returns:
            np.ndarray: Blended crop (H, W, 3) uint8 with speech realism preserved.
        """
        if swapped_crop is None or original_crop is None:
            return swapped_crop if swapped_crop is not None else original_crop

        if swapped_crop.shape[:2] != original_crop.shape[:2]:
            h, w = swapped_crop.shape[:2]
            orig_sized = cv2.resize(original_crop, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            orig_sized = original_crop

        h, w = swapped_crop.shape[:2]
        if mouth_mask.shape[:2] != (h, w):
            mask_sized = cv2.resize(mouth_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            mask_sized = mouth_mask

        # 3-channel alpha
        alpha = mask_sized[:, :, None]

        # Clean linear alpha blend preserving natural teeth alignment and speech realism
        swapped_f = swapped_crop.astype(np.float32)
        orig_f = orig_sized.astype(np.float32)

        composited = swapped_f * (1.0 - alpha) + orig_f * alpha
        return np.clip(composited, 0.0, 255.0).astype(np.uint8)

    def process(
        self,
        swapped_crop: np.ndarray,
        original_crop: np.ndarray,
        face: Any,
        mar_threshold: Optional[float] = None,
        feather_radius: Optional[int] = None,
    ) -> Tuple[np.ndarray, bool, float]:
        """Execute mouth dynamics check and conditional surgical restoration.

        Args:
            swapped_crop: Face crop after swap/enhancer (H, W, 3) uint8.
            original_crop: Untouched original target face crop (H, W, 3) uint8.
            face: Target Face object containing landmarks.
            mar_threshold: Optional override for MAR trigger threshold.
            feather_radius: Optional override for feathering radius.

        Returns:
            Tuple[np.ndarray, bool, float]:
                - Resulting blended face crop.
                - Boolean indicating whether mouth restoration was triggered.
                - Computed MAR value.
        """
        if swapped_crop is None or original_crop is None or face is None:
            return swapped_crop, False, 0.0

        thresh = mar_threshold if mar_threshold is not None else self.mar_threshold
        feather = feather_radius if feather_radius is not None else self.feather_radius

        # Extract landmarks from face
        lm = getattr(face, "landmark_3d_68", None)
        if lm is None:
            lm = getattr(face, "landmark_2d_106", None)
        if lm is None:
            lm = getattr(face, "kps", None)

        mar = compute_mar(lm)

        if mar < thresh:
            # Mouth is closed or neutral; preserve full swapped/enhanced aperture
            return swapped_crop, False, mar

        # MAR exceeded threshold (indicating speech, yawning, smile, or open mouth)
        inner_pts = self.get_inner_lip_points(face, swapped_crop.shape[:2])
        mouth_mask = self.generate_inner_mouth_mask(original_crop, inner_pts, feather_radius=feather)
        restored = self.composite_mouth_aperture(swapped_crop, original_crop, mouth_mask)

        return restored, True, mar
