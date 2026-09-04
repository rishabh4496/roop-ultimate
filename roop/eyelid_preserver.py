"""Eyelid preservation and Eye Aspect Ratio (EAR) blink gating module.

Eliminates "dead-eye" artifacts by preserving natural eyelid closure during blinks.

Mathematical Specifications:
1. Eye Aspect Ratio (EAR) Blink Gating:
   Calculate EAR for both eyes from 68-point landmarks:
     EAR = (||p2 - p6|| + ||p3 - p5||) / (2.0 * ||p1 - p4||)
   where p1..p6 represent the 6 landmark points surrounding each eye.

2. Authentic Eyelid Restoration:
   When EAR < 0.18 (full eyelid closure / active blink):
     - Generate an eye-region elliptical mask around landmarks p1-p6.
     - Blend the original target eyelids back over the swapped face at 95% opacity
       to preserve authentic blinking, natural eyelashes, and eye-crease folding.
"""

import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

DEFAULT_EAR_BLINK_THRESHOLD: float = float(os.environ.get('ROOP_EAR_BLINK_THRESHOLD', '0.18'))
DEFAULT_EYELID_BLEND_OPACITY: float = float(os.environ.get('ROOP_EYELID_BLEND_OPACITY', '0.95'))
EAR_BLINK_THRESHOLD = DEFAULT_EAR_BLINK_THRESHOLD
EYELID_BLEND_OPACITY = DEFAULT_EYELID_BLEND_OPACITY


def calculate_ear(eye_points: np.ndarray) -> float:
    """Calculate Eye Aspect Ratio (EAR) from 6 landmark points.

    Formula (Soukupová and Čech):
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2.0 * ||p1 - p4||)

    Parameters:
        eye_points: Array of shape (6, 2) or (>=6, 2) representing:
            p1: lateral or medial corner (start)
            p2, p3: upper eyelid points
            p4: opposite corner (end)
            p5, p6: lower eyelid points

    Returns:
        float: EAR scalar value. Open eyes are typically > 0.25; closed eyes < 0.18.
    """
    pts = np.asarray(eye_points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 6:
        return 0.30

    p1, p2, p3, p4, p5, p6 = pts[:6]
    v1 = float(np.linalg.norm(p2 - p6))
    v2 = float(np.linalg.norm(p3 - p5))
    h = float(np.linalg.norm(p1 - p4))

    if h < 1e-6:
        return 0.30

    return float((v1 + v2) / (2.0 * h))


def compute_eye_aspect_ratios(landmarks_68: np.ndarray) -> Tuple[float, float, float]:
    """Compute (left_ear, right_ear, mean_ear) from standard 68-point landmarks.

    0-indexed landmark numbering:
        Left eye (viewer's left): points 36..41 (indices 36:42)
        Right eye (viewer's right): points 42..47 (indices 42:48)
    """
    pts = np.asarray(landmarks_68, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 68:
        return 0.30, 0.30, 0.30

    left_ear = calculate_ear(pts[36:42])
    right_ear = calculate_ear(pts[42:48])
    mean_ear = float(0.5 * (left_ear + right_ear))
    return left_ear, right_ear, mean_ear


def is_eye_blinking(ear: float, threshold: float = DEFAULT_EAR_BLINK_THRESHOLD) -> bool:
    """Return True if EAR is strictly below the blink gating threshold."""
    return float(ear) < float(threshold)


def generate_eye_elliptical_mask(
    eye_points: np.ndarray,
    shape: Tuple[int, int],
    vertical_scale: float = 1.6,
    horizontal_scale: float = 0.75
) -> np.ndarray:
    """Generate an eye-region elliptical mask around landmarks p1-p6.

    Encompasses the upper and lower eyelid creases, eyelashes, and palpebral fold.
    """
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), dtype=np.float32)
    pts = np.asarray(eye_points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 6:
        return mask

    # Compute eye geometric center
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))

    # Width across corners (p1 to p4)
    ew = float(np.linalg.norm(pts[0] - pts[3]))
    # Height between lid pairs
    eh = float(max(np.linalg.norm(pts[1] - pts[5]), np.linalg.norm(pts[2] - pts[4])))

    axis_x = max(6, int(round(ew * horizontal_scale)))
    axis_y = max(4, int(round(max(eh * vertical_scale, ew * 0.35))))

    # Orientation angle of the eye fissure
    dx = pts[3, 0] - pts[0, 0]
    dy = pts[3, 1] - pts[0, 1]
    angle = float(np.degrees(np.arctan2(dy, dx)))

    # Shift center slightly upwards toward the superior palpebral fold (upper eyelid)
    shift_y = -axis_y * 0.12
    rad = np.radians(angle)
    center_x = int(round(cx - shift_y * np.sin(rad)))
    center_y = int(round(cy + shift_y * np.cos(rad)))

    cv2.ellipse(
        mask,
        (center_x, center_y),
        (axis_x, axis_y),
        angle,
        0, 360,
        1.0,
        -1
    )

    # Soft Gaussian boundary feathering
    k = max(3, int(round(axis_x * 0.35)) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    max_val = float(np.max(mask))
    if max_val > 1e-6:
        mask /= max_val

    return np.clip(mask, 0.0, 1.0)


def build_blink_eyelid_mask(
    landmarks_68: np.ndarray,
    shape: Tuple[int, int],
    ear_threshold: float = DEFAULT_EAR_BLINK_THRESHOLD
) -> np.ndarray:
    """Generate combined eye-region elliptical mask for eyes below the blink threshold (< 0.18).

    If both eyes are open (EAR >= threshold), returns an empty mask (zeros).
    """
    h, w = int(shape[0]), int(shape[1])
    combined_mask = np.zeros((h, w), dtype=np.float32)
    pts = np.asarray(landmarks_68, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 68:
        return combined_mask

    left_ear, right_ear, _ = compute_eye_aspect_ratios(pts)

    if left_ear < ear_threshold:
        left_mask = generate_eye_elliptical_mask(pts[36:42], (h, w))
        combined_mask = np.maximum(combined_mask, left_mask)

    if right_ear < ear_threshold:
        right_mask = generate_eye_elliptical_mask(pts[42:48], (h, w))
        combined_mask = np.maximum(combined_mask, right_mask)

    return np.clip(combined_mask, 0.0, 1.0)


def blend_eyelid_preservation(
    target_crop: np.ndarray,
    swapped_crop: np.ndarray,
    eyelid_mask: np.ndarray,
    opacity: float = DEFAULT_EYELID_BLEND_OPACITY
) -> np.ndarray:
    """Blend original target eyelids back over swapped face at specified opacity (default 95%).

    Formula:
        W = clip(eyelid_mask * opacity, 0.0, 1.0)
        I_out = (1.0 - W) * I_swapped + W * I_target

    Preserves authentic blinking, natural eyelashes, and eye-crease folding.
    """
    if eyelid_mask is None or not np.any(eyelid_mask > 1e-4):
        return swapped_crop.copy()

    t_f = target_crop.astype(np.float32)
    s_f = swapped_crop.astype(np.float32)

    weight = np.clip(eyelid_mask * opacity, 0.0, 1.0)[..., np.newaxis]
    blended = s_f * (1.0 - weight) + t_f * weight
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def blend_eyelid_multiscale(
    target_crop: np.ndarray,
    swapped_crop: np.ndarray,
    eyelid_mask: np.ndarray,
    opacity: float = DEFAULT_EYELID_BLEND_OPACITY
) -> np.ndarray:
    """Multi-scale frequency decomposition blend scaling target eyelid contribution by opacity."""
    if eyelid_mask is None or not np.any(eyelid_mask > 1e-4):
        return swapped_crop.copy()

    t_f = target_crop.astype(np.float32)
    s_f = swapped_crop.astype(np.float32)

    # Multi-scale mask decomposition
    m_fine = (cv2.GaussianBlur(eyelid_mask, (5, 5), 1.5) * opacity)[..., np.newaxis]
    m_coarse = (cv2.GaussianBlur(eyelid_mask, (15, 15), 5.0) * opacity)[..., np.newaxis]

    # Frequency split
    t_low = cv2.GaussianBlur(t_f, (15, 15), 5.0)
    s_low = cv2.GaussianBlur(s_f, (15, 15), 5.0)
    t_high = t_f - t_low
    s_high = s_f - s_low

    blended_low = s_low * (1.0 - m_coarse) + t_low * m_coarse
    blended_high = s_high * (1.0 - m_fine) + t_high * m_fine

    blended = blended_low + blended_high
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def get_closed_eyes_attenuation(
    landmarks_68: np.ndarray,
    shape: Tuple[int, int],
    ear_threshold: float = DEFAULT_EAR_BLINK_THRESHOLD
) -> Tuple[bool, np.ndarray]:
    """Generate attenuation mask over closed eye bounding boxes to bypass restorer hallucination."""
    h, w = int(shape[0]), int(shape[1])
    attenuation_mask = np.ones((h, w), dtype=np.float32)
    pts = np.asarray(landmarks_68, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 68:
        return False, attenuation_mask

    left_ear, right_ear, _ = compute_eye_aspect_ratios(pts)
    is_blinking = (left_ear < ear_threshold or right_ear < ear_threshold)

    for start_idx, end_idx, ear in ((36, 42, left_ear), (42, 48, right_ear)):
        if ear < ear_threshold:
            eye_pts = pts[start_idx:end_idx]
            x1 = max(0, int(np.min(eye_pts[:, 0]) - 8))
            y1 = max(0, int(np.min(eye_pts[:, 1]) - 12))
            x2 = min(w, int(np.max(eye_pts[:, 0]) + 8))
            y2 = min(h, int(np.max(eye_pts[:, 1]) + 8))
            attenuation_mask[y1:y2, x1:x2] = 0.0

    if is_blinking:
        attenuation_mask = cv2.GaussianBlur(attenuation_mask, (7, 7), 2.0)
    return is_blinking, attenuation_mask


def preserve_eyelids(
    target_crop: np.ndarray,
    swapped_crop: np.ndarray,
    landmarks_68: Optional[np.ndarray],
    ear_threshold: float = DEFAULT_EAR_BLINK_THRESHOLD,
    opacity: float = DEFAULT_EYELID_BLEND_OPACITY
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Execute complete EAR blink gating and eyelid preservation pipeline."""
    meta: Dict[str, Any] = {
        'is_blinking': False,
        'left_ear': 0.30,
        'right_ear': 0.30,
        'ear_threshold': ear_threshold,
        'opacity': opacity,
        'eyelid_mask': None,
        'attenuation_mask': None,
    }

    if landmarks_68 is None:
        return swapped_crop.copy(), meta

    pts = np.asarray(landmarks_68, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 68:
        return swapped_crop.copy(), meta

    shape = target_crop.shape[:2]
    left_ear, right_ear, _ = compute_eye_aspect_ratios(pts)
    meta['left_ear'] = left_ear
    meta['right_ear'] = right_ear

    is_blink = (left_ear < ear_threshold or right_ear < ear_threshold)
    meta['is_blinking'] = is_blink

    if is_blink:
        _, att_mask = get_closed_eyes_attenuation(pts, shape, ear_threshold=ear_threshold)
        meta['attenuation_mask'] = att_mask

        eyelid_mask = build_blink_eyelid_mask(pts, shape, ear_threshold=ear_threshold)
        meta['eyelid_mask'] = eyelid_mask

        blended = blend_eyelid_preservation(target_crop, swapped_crop, eyelid_mask, opacity=opacity)
        return blended, meta

    return swapped_crop.copy(), meta
