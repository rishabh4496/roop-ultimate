"""Visual quality, temporal stability, flicker quantification, and angle-bucket
stability metrics for the Roop Ultimate pipeline benchmark.

This module provides:
1. Quantitative Flicker Metrics:
   - Frame-to-frame temporal instability inside the detected target face ROI across
     consecutive frames:
     - Mean L1 luminance delta: pixel-wise |Y_t - Y_{t-1}| in face ROI.
     - SSIM drop: Structural similarity drop between consecutive aligned face ROIs
       (1.0 - SSIM(ROI_t, ROI_{t-1})).
     - Landmark jitter in pixels: Mean Euclidean displacement sqrt(dx^2 + dy^2) of
       5-point / 68-point landmarks between consecutive frames after global translation
       compensation.
     - Mask-area variance: Relative change and normalized variance in occluder/face mask
       pixel area across consecutive frames.
     - Scene cut detector: Detect shot transitions (e.g. large global histogram delta or
       SSIM < 0.35) and exclude cut boundaries from flicker calculations.
2. Angle-Bucket Stability Report:
   - Head yaw estimation using existing `solve_pose_5pt` from `roop.face_util` with
     robust 5-point landmark geometry fallback.
   - Bucketing per frame by absolute estimated head yaw: 0-20, 20-40, 40-60, 60+.
   - Metrics per bucket:
     - Detection recall: fraction of frames where target face was successfully detected.
     - Swap coverage rate: fraction of detected frames where face swap was executed / covered.
     - Seam-error score: edge/boundary discontinuity metric around the composite face mask border.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# Optional import of solve_pose_5pt from roop.face_util with fallback
try:
    from roop.face_util import solve_pose_5pt
except ImportError:
    try:
        from app.roop.face_util import solve_pose_5pt
    except ImportError:
        solve_pose_5pt = None


YAW_BUCKETS: Tuple[str, ...] = ("0-20", "20-40", "40-60", "60+")


@dataclass
class FrameMetricRecord:
    """Per-frame input data container for metric analysis."""

    frame_index: int
    image: Optional[np.ndarray] = None  # Full output composite frame (HxWxC, BGR or RGB)
    target_image: Optional[np.ndarray] = None  # Original target frame before swap (optional)
    face_roi: Optional[np.ndarray] = None  # Aligned/cropped face ROI (HxWxC)
    landmarks: Optional[np.ndarray] = None  # 5-pt or 68-pt landmarks array (Nx2)
    mask: Optional[np.ndarray] = None  # Composite face/occlusion mask (HxW, float or uint8)
    is_detected: bool = True  # True if target face detected in this frame
    is_swapped: bool = True  # True if face swap was applied/covered
    yaw: Optional[float] = None  # Optional precomputed head yaw in degrees
    bbox: Optional[Sequence[float]] = None  # [x0, y0, x1, y1] bounding box in frame


@dataclass
class FlickerMetricResult:
    """Aggregated quantitative flicker metric results across a video sequence."""

    mean_l1_luminance_delta: float = 0.0
    mean_ssim_drop: float = 0.0
    mean_landmark_jitter_px: float = 0.0
    mask_area_relative_change: float = 0.0
    mask_area_variance: float = 0.0
    valid_transitions: int = 0
    scene_cuts_excluded: int = 0
    per_frame_l1_delta: List[float] = field(default_factory=list)
    per_frame_ssim_drop: List[float] = field(default_factory=list)
    per_frame_landmark_jitter: List[float] = field(default_factory=list)
    per_frame_mask_relative_change: List[float] = field(default_factory=list)
    scene_cut_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_l1_luminance_delta": round(float(self.mean_l1_luminance_delta), 4),
            "mean_ssim_drop": round(float(self.mean_ssim_drop), 4),
            "mean_landmark_jitter_px": round(float(self.mean_landmark_jitter_px), 4),
            "mask_area_relative_change": round(float(self.mask_area_relative_change), 4),
            "mask_area_variance": round(float(self.mask_area_variance), 6),
            "valid_transitions": int(self.valid_transitions),
            "scene_cuts_excluded": int(self.scene_cuts_excluded),
            "scene_cut_indices": list(self.scene_cut_indices),
        }


@dataclass
class AngleBucketStats:
    """Statistics for one head yaw bucket."""

    bucket_name: str
    total_frames: int = 0
    detected_frames: int = 0
    swapped_frames: int = 0
    detection_recall: float = 0.0
    swap_coverage_rate: float = 0.0
    seam_error_score: float = 0.0
    mean_abs_yaw: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bucket": self.bucket_name,
            "total_frames": int(self.total_frames),
            "detected_frames": int(self.detected_frames),
            "swapped_frames": int(self.swapped_frames),
            "detection_recall": round(float(self.detection_recall), 4),
            "swap_coverage_rate": round(float(self.swap_coverage_rate), 4),
            "seam_error_score": round(float(self.seam_error_score), 4),
            "mean_abs_yaw": round(float(self.mean_abs_yaw), 2),
        }


@dataclass
class AngleBucketReportResult:
    """Overall angle-bucket stability report across yaw categories."""

    buckets: Dict[str, AngleBucketStats] = field(default_factory=dict)
    overall_detection_recall: float = 0.0
    overall_swap_coverage: float = 0.0
    overall_seam_error: float = 0.0
    total_frames_analyzed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "buckets": {k: v.to_dict() for k, v in self.buckets.items()},
            "overall_detection_recall": round(float(self.overall_detection_recall), 4),
            "overall_swap_coverage": round(float(self.overall_swap_coverage), 4),
            "overall_seam_error": round(float(self.overall_seam_error), 4),
            "total_frames_analyzed": int(self.total_frames_analyzed),
        }


@dataclass
class BenchmarkMetricsResult:
    """Comprehensive metric summary object combining flicker and angle reports."""

    flicker: FlickerMetricResult = field(default_factory=FlickerMetricResult)
    angle_report: AngleBucketReportResult = field(default_factory=AngleBucketReportResult)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flicker": self.flicker.to_dict(),
            "angle_buckets": {k: v.to_dict() for k, v in self.angle_report.buckets.items()},
            "overall_stability": {
                "detection_recall": round(float(self.angle_report.overall_detection_recall), 4),
                "swap_coverage_rate": round(float(self.angle_report.overall_swap_coverage), 4),
                "mean_seam_error_score": round(float(self.angle_report.overall_seam_error), 4),
                "total_frames_analyzed": int(self.angle_report.total_frames_analyzed),
            },
        }


# ==============================================================================
# 1. Structural Similarity & Scene Cut Detection
# ==============================================================================

def compute_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    data_range: float = 255.0,
    win_size: int = 7,
) -> float:
    """Compute mean Structural Similarity Index (SSIM) between two images.

    Uses uniform box filtering consistent with skimage and roop canary standards.
    Supports 1-channel or 3-channel images of matching or mismatched shapes (auto-resized).
    """
    if img1 is None or img2 is None or img1.size == 0 or img2.size == 0:
        return 0.0

    a = np.asarray(img1, dtype=np.float64)
    b = np.asarray(img2, dtype=np.float64)

    # Standardize dimensions
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)

    if a.ndim == 2:
        a = a[:, :, np.newaxis]
    if b.ndim == 2:
        b = b[:, :, np.newaxis]

    channels = min(a.shape[2], b.shape[2])
    a = a[:, :, :channels]
    b = b[:, :, :channels]

    h, w = a.shape[:2]
    win = min(win_size, min(h, w))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        # Image too small for spatial SSIM window; fall back to normalized correlation
        denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
        return float(np.clip(float(np.sum(a * b)) / denom, -1.0, 1.0))

    npx = win * win
    cov_norm = npx / (npx - 1.0)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    total_ssim = 0.0
    for ch in range(channels):
        x = a[:, :, ch]
        y = b[:, :, ch]

        blur = lambda im: cv2.blur(im, (win, win), borderType=cv2.BORDER_REFLECT)
        ux = blur(x)
        uy = blur(y)
        uxx = blur(x * x)
        uyy = blur(y * y)
        uxy = blur(x * y)

        vx = cov_norm * np.maximum(0.0, uxx - ux * ux)
        vy = cov_norm * np.maximum(0.0, uyy - uy * uy)
        vxy = cov_norm * (uxy - ux * uy)

        num = (2.0 * ux * uy + c1) * (2.0 * vxy + c2)
        den = (ux * ux + uy * uy + c1) * (vx + vy + c2)
        s = num / (den + 1e-12)

        pad = (win - 1) // 2
        valid = s[pad:-pad, pad:-pad] if (h > win and w > win) else s
        total_ssim += float(np.mean(valid))

    return float(np.clip(total_ssim / channels, -1.0, 1.0))


def compute_ssim_drop(roi_prev: np.ndarray, roi_curr: np.ndarray) -> float:
    """Compute SSIM drop between consecutive aligned face ROIs: 1.0 - SSIM(ROI_t, ROI_{t-1})."""
    if roi_prev is None or roi_curr is None or roi_prev.size == 0 or roi_curr.size == 0:
        return 0.0
    sim = compute_ssim(roi_prev, roi_curr, data_range=255.0)
    drop = 1.0 - sim
    return float(np.clip(drop, 0.0, 2.0))


def detect_scene_cut(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    ssim_threshold: float = 0.35,
    hist_threshold: float = 0.50,
    l1_threshold: float = 65.0,
) -> Tuple[bool, Dict[str, float]]:
    """Detect whether a transition between two consecutive frames is a scene cut/shot change.

    Uses a fusion of downsampled full-frame SSIM, color histogram correlation, and mean L1 delta.
    Returns (is_scene_cut, diagnostics_dict).
    """
    if frame_prev is None or frame_curr is None or frame_prev.size == 0 or frame_curr.size == 0:
        return False, {"ssim": 1.0, "hist_correlation": 1.0, "l1_diff": 0.0}

    # Downsample frames to max 160px for ultra-fast scene cut evaluation
    max_dim = 160
    h, w = frame_prev.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    target_size = (max(16, int(w * scale)), max(16, int(h * scale)))

    p_small = cv2.resize(frame_prev, target_size, interpolation=cv2.INTER_AREA)
    c_small = cv2.resize(frame_curr, target_size, interpolation=cv2.INTER_AREA)

    # 1. Full-frame SSIM
    ssim_val = compute_ssim(p_small, c_small, data_range=255.0, win_size=7)

    # 2. Histogram correlation
    if p_small.ndim == 3 and p_small.shape[2] == 3:
        hsv_p = cv2.cvtColor(p_small, cv2.COLOR_BGR2HSV)
        hsv_c = cv2.cvtColor(c_small, cv2.COLOR_BGR2HSV)
        # 16 bins for Hue, 16 for Saturation
        hist_p = cv2.calcHist([hsv_p], [0, 1], None, [16, 16], [0, 180, 0, 256])
        hist_c = cv2.calcHist([hsv_c], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist_p, hist_p, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_c, hist_c, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        hist_corr = float(cv2.compareHist(hist_p, hist_c, cv2.HISTCMP_CORREL))
    else:
        hist_p = cv2.calcHist([p_small], [0], None, [32], [0, 256])
        hist_c = cv2.calcHist([c_small], [0], None, [32], [0, 256])
        cv2.normalize(hist_p, hist_p, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_c, hist_c, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        hist_corr = float(cv2.compareHist(hist_p, hist_c, cv2.HISTCMP_CORREL))

    # 3. Mean L1 pixel difference
    diff = np.abs(c_small.astype(np.float32) - p_small.astype(np.float32))
    l1_diff = float(np.mean(diff))

    diag = {
        "ssim": round(ssim_val, 4),
        "hist_correlation": round(hist_corr, 4),
        "l1_diff": round(l1_diff, 2),
    }

    # Cut decision: low SSIM, or low histogram correlation with significant L1 jump
    is_cut = (ssim_val < ssim_threshold) or (hist_corr < hist_threshold and l1_diff > 35.0) or (l1_diff > l1_threshold)

    return bool(is_cut), diag


# ==============================================================================
# 2. Flicker Metric Implementations
# ==============================================================================

def compute_luminance_l1_delta(roi_prev: np.ndarray, roi_curr: np.ndarray) -> float:
    """Compute mean L1 luminance delta: pixel-wise |Y_t - Y_{t-1}| in face ROI.

    Luminance is calculated via standard Rec.601 weights (0.299 R + 0.587 G + 0.114 B)
    on a 0..255 scale.
    """
    if roi_prev is None or roi_curr is None or roi_prev.size == 0 or roi_curr.size == 0:
        return 0.0

    p = roi_prev
    c = roi_curr

    # Align resolution if shapes differ
    if p.shape[:2] != c.shape[:2]:
        c = cv2.resize(c, (p.shape[1], p.shape[0]), interpolation=cv2.INTER_LINEAR)

    if p.ndim == 3 and p.shape[2] >= 3:
        y_prev = cv2.cvtColor(p[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        y_prev = p.squeeze().astype(np.float32)

    if c.ndim == 3 and c.shape[2] >= 3:
        y_curr = cv2.cvtColor(c[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        y_curr = c.squeeze().astype(np.float32)

    delta = np.abs(y_curr - y_prev)
    return float(np.mean(delta))


def compute_landmark_jitter(
    kps_prev: np.ndarray,
    kps_curr: np.ndarray,
    compensate_translation: bool = True,
) -> float:
    """Compute mean Euclidean displacement of landmarks between consecutive frames

    Mean sqrt(dx^2 + dy^2) across landmark points after global translation compensation.
    Supports both 5-point and 68-point landmark structures (Nx2 or Nx3 coordinates).
    """
    if kps_prev is None or kps_curr is None:
        return 0.0

    p = np.asarray(kps_prev, dtype=np.float64)[:, :2]
    c = np.asarray(kps_curr, dtype=np.float64)[:, :2]

    if p.shape != c.shape or p.shape[0] == 0:
        return 0.0

    if compensate_translation:
        centroid_prev = p.mean(axis=0)
        centroid_curr = c.mean(axis=0)
        translation = centroid_curr - centroid_prev
        c_aligned = c - translation
    else:
        c_aligned = c

    diff = c_aligned - p
    dist = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
    return float(np.mean(dist))


def compute_mask_area(mask: np.ndarray) -> float:
    """Compute total active pixel area of an occluder or face mask."""
    if mask is None or mask.size == 0:
        return 0.0

    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]

    if m.dtype == np.uint8:
        return float(np.sum(m > 127))
    # Soft mask in 0..1 range
    return float(np.sum(m > 0.5)) if np.max(m) <= 1.0 else float(np.sum(m > 127.0))


def compute_mask_relative_change(mask_prev: np.ndarray, mask_curr: np.ndarray) -> float:
    """Compute relative change in mask pixel area across consecutive frames:

    |Area_t - Area_{t-1}| / max(Area_t, Area_{t-1}, 1.0).
    """
    if mask_prev is None or mask_curr is None:
        return 0.0

    area_p = compute_mask_area(mask_prev)
    area_c = compute_mask_area(mask_curr)
    denom = max(area_p, area_c, 1.0)
    return float(abs(area_c - area_p) / denom)


def compute_mask_area_variance(masks: Sequence[np.ndarray]) -> Tuple[float, float]:
    """Compute (mean_relative_change, normalized_variance) across a sequence of masks."""
    valid_masks = [m for m in masks if m is not None and m.size > 0]
    if len(valid_masks) < 2:
        return 0.0, 0.0

    areas = [compute_mask_area(m) for m in valid_masks]
    arr = np.asarray(areas, dtype=np.float64)
    mean_area = float(np.mean(arr))
    if mean_area < 1e-6:
        return 0.0, 0.0

    # Normalized variance Var(A / mean(A))
    norm_var = float(np.var(arr / mean_area))

    # Mean consecutive relative change
    diffs = np.abs(arr[1:] - arr[:-1])
    denoms = np.maximum(np.maximum(arr[1:], arr[:-1]), 1.0)
    rel_change = float(np.mean(diffs / denoms))

    return rel_change, norm_var


# ==============================================================================
# 3. Head Yaw Estimation & Angle Bucketing
# ==============================================================================

def estimate_head_yaw(landmarks: np.ndarray) -> float:
    """Estimate head yaw angle in degrees from facial landmarks.

    Uses `solve_pose_5pt` from `roop.face_util` if available.
    Supports both 5-point and 68-point landmarks with robust geometric fallback.
    Returns yaw in degrees (negative = turned left, positive = turned right).
    """
    if landmarks is None:
        return 0.0

    pts = np.asarray(landmarks, dtype=np.float64)[:, :2]
    n_pts = pts.shape[0]

    # Convert 68-point landmarks to 5-point standard (eye centers, nose tip, mouth corners)
    if n_pts >= 68:
        kps_5 = np.array(
            [
                pts[36:42].mean(axis=0),  # left eye
                pts[42:48].mean(axis=0),  # right eye
                pts[30],  # nose tip
                pts[48],  # left mouth corner
                pts[54],  # right mouth corner
            ],
            dtype=np.float64,
        )
    elif n_pts == 5:
        kps_5 = pts
    else:
        return 0.0

    # Attempt solve_pose_5pt first
    if solve_pose_5pt is not None:
        try:
            pose = solve_pose_5pt(kps_5)
            if pose is not None and len(pose) >= 3 and math.isfinite(pose[0]):
                return float(pose[0])
        except Exception:
            pass

    # Landmark geometric yaw estimation fallback
    le, re, nose = kps_5[0], kps_5[1], kps_5[2]
    eye_vec = re - le
    eye_dist = float(np.linalg.norm(eye_vec))
    if eye_dist < 1e-6:
        return 0.0

    u_eye = eye_vec / eye_dist
    eye_mid = 0.5 * (le + re)
    nose_disp = nose - eye_mid
    proj = float(np.dot(nose_disp, u_eye) / (0.5 * eye_dist))
    proj_clamped = max(-1.0, min(1.0, proj))
    yaw_deg = float(math.asin(proj_clamped) * (180.0 / math.pi))
    return yaw_deg


def get_yaw_bucket(yaw: float) -> str:
    """Map a head yaw angle (degrees) to its absolute yaw bucket: 0-20, 20-40, 40-60, 60+."""
    abs_yaw = abs(float(yaw))
    if abs_yaw < 20.0:
        return "0-20"
    if abs_yaw < 40.0:
        return "20-40"
    if abs_yaw < 60.0:
        return "40-60"
    return "60+"


# ==============================================================================
# 4. Seam Error Score (Border Discontinuity)
# ==============================================================================

def compute_seam_error(
    composite_frame: np.ndarray,
    mask: np.ndarray,
    target_frame: Optional[np.ndarray] = None,
    band_width: int = 3,
) -> float:
    """Calculate edge/boundary discontinuity metric around the composite face mask border.

    Quantifies hard mask seams and blending artifacts by evaluating the directional derivative
    of pixel intensities normal to the face mask border across a morphological ribbon band.
    Higher values indicate sharp, unfeathered seams or hard boundary transitions.
    Seamless or well-blended transitions yield low scores.
    """
    if composite_frame is None or mask is None or composite_frame.size == 0 or mask.size == 0:
        return 0.0

    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]

    # Normalize mask to float [0, 1] and binary uint8
    if m.dtype == np.uint8:
        m_float = m.astype(np.float32) / 255.0
        m_bin = (m > 127).astype(np.uint8)
    else:
        m_float = np.clip(m.astype(np.float32), 0.0, 1.0)
        m_bin = (m_float > 0.5).astype(np.uint8)

    # If mask is completely empty or covers entire frame, no seam exists
    total_active = int(np.sum(m_bin))
    if total_active == 0 or total_active == m_bin.size:
        return 0.0

    # Resize mask if it does not match composite frame dimensions
    h, w = composite_frame.shape[:2]
    if m_bin.shape[:2] != (h, w):
        m_bin = cv2.resize(m_bin, (w, h), interpolation=cv2.INTER_NEAREST)
        m_float = cv2.resize(m_float, (w, h), interpolation=cv2.INTER_LINEAR)

    # Morphological ribbon band around the mask perimeter
    k_size = 2 * band_width + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv2.dilate(m_bin, kernel)
    eroded = cv2.erode(m_bin, kernel)
    ribbon = (dilated > 0) & (eroded == 0)

    if not np.any(ribbon):
        return 0.0

    # Image intensity gradient
    if composite_frame.ndim == 3 and composite_frame.shape[2] >= 3:
        gray = cv2.cvtColor(composite_frame[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = composite_frame.squeeze().astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Mask boundary normal vectors
    mx = cv2.Sobel(m_float, cv2.CV_32F, 1, 0, ksize=3)
    my = cv2.Sobel(m_float, cv2.CV_32F, 0, 1, ksize=3)
    m_norm = np.sqrt(mx * mx + my * my) + 1e-6
    nx = mx / m_norm
    ny = my / m_norm

    # Directional derivative along the boundary normal (measures step discontinuity across border)
    directional_deriv = np.abs(gx * nx + gy * ny)
    seam_score = float(np.mean(directional_deriv[ribbon]))

    # If original target frame is provided, measure outer boundary color divergence
    if target_frame is not None and target_frame.shape[:2] == (h, w):
        outer_rim = (dilated > 0) & (m_bin == 0)
        if np.any(outer_rim):
            if target_frame.ndim == 3 and target_frame.shape[2] >= 3:
                t_gray = cv2.cvtColor(target_frame[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                t_gray = target_frame.squeeze().astype(np.float32)
            outer_step = float(np.mean(np.abs(gray[outer_rim] - t_gray[outer_rim])))
            seam_score = float(seam_score + outer_step * 0.4)

    return float(seam_score)


# ==============================================================================
# 5. Face ROI Extraction & Alignment Utilities
# ==============================================================================

def extract_face_roi(
    image: np.ndarray,
    landmarks: Optional[np.ndarray] = None,
    bbox: Optional[Sequence[float]] = None,
    crop_size: int = 256,
) -> Optional[np.ndarray]:
    """Extract or align a face ROI from an image using landmarks or bounding box.

    If landmarks (5-point) are given, computes an affine canonical alignment crop.
    If bbox is given, crops the face with standard margin and resizes to crop_size.
    If image itself is already cropped, resizes directly to crop_size.
    """
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]

    # 1. 5-point alignment crop if landmarks exist
    if landmarks is not None:
        pts = np.asarray(landmarks, dtype=np.float32)[:, :2]
        if pts.shape[0] >= 5:
            kps_5 = pts[:5]
            # Arcface standard canonical 112x112 destination scaled to crop_size
            dst_112 = np.array(
                [
                    [38.2946, 51.6963],
                    [73.5318, 51.5014],
                    [56.0252, 71.7366],
                    [41.5493, 92.3655],
                    [70.7299, 92.2041],
                ],
                dtype=np.float32,
            )
            dst_scaled = dst_112 * (float(crop_size) / 112.0)
            M, _ = cv2.estimateAffinePartial2D(kps_5, dst_scaled)
            if M is not None:
                warped = cv2.warpAffine(
                    image,
                    M,
                    (crop_size, crop_size),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                return warped

    # 2. Bounding box crop if bbox exists
    if bbox is not None and len(bbox) >= 4:
        x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)
        # Margin 15%
        pad_x = int(0.15 * bw)
        pad_y = int(0.15 * bh)
        cx0 = max(0, x0 - pad_x)
        cy0 = max(0, y0 - pad_y)
        cx1 = min(w, x1 + pad_x)
        cy1 = min(h, y1 + pad_y)
        if cx1 > cx0 and cy1 > cy0:
            crop = image[cy0:cy1, cx0:cx1]
            return cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

    # 3. Direct resize of image
    return cv2.resize(image, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)


# ==============================================================================
# 6. Benchmark Metrics Tracker & Aggregator
# ==============================================================================

class BenchmarkMetricsTracker:
    """Accumulator and evaluator for visual quality and temporal stability metrics."""

    def __init__(self, face_crop_size: int = 256):
        self.face_crop_size = face_crop_size
        self.frames: List[FrameMetricRecord] = []

    def add_frame(
        self,
        frame_index: int,
        image: Optional[np.ndarray] = None,
        target_image: Optional[np.ndarray] = None,
        face_roi: Optional[np.ndarray] = None,
        landmarks: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        is_detected: bool = True,
        is_swapped: bool = True,
        yaw: Optional[float] = None,
        bbox: Optional[Sequence[float]] = None,
    ) -> FrameMetricRecord:
        """Add a frame's evaluation data into the tracker."""
        # Auto-extract face ROI if not provided but image + (landmarks or bbox) exist
        if face_roi is None and image is not None:
            face_roi = extract_face_roi(
                image,
                landmarks=landmarks,
                bbox=bbox,
                crop_size=self.face_crop_size,
            )

        # Estimate head yaw if not explicitly provided
        if yaw is None and landmarks is not None:
            yaw = estimate_head_yaw(landmarks)

        record = FrameMetricRecord(
            frame_index=frame_index,
            image=image,
            target_image=target_image,
            face_roi=face_roi,
            landmarks=landmarks,
            mask=mask,
            is_detected=is_detected,
            is_swapped=is_swapped,
            yaw=yaw,
            bbox=bbox,
        )
        self.frames.append(record)
        return record

    def add_frame_record(self, record: FrameMetricRecord) -> None:
        """Add a pre-built FrameMetricRecord."""
        if record.face_roi is None and record.image is not None:
            record.face_roi = extract_face_roi(
                record.image,
                landmarks=record.landmarks,
                bbox=record.bbox,
                crop_size=self.face_crop_size,
            )
        if record.yaw is None and record.landmarks is not None:
            record.yaw = estimate_head_yaw(record.landmarks)
        self.frames.append(record)

    def compute_flicker_metrics(self) -> FlickerMetricResult:
        """Calculate quantitative temporal flicker metrics across consecutive frames.

        Excludes shot transitions detected by the scene cut detector.
        """
        result = FlickerMetricResult()
        n = len(self.frames)
        if n < 2:
            return result

        masks_for_var: List[np.ndarray] = []

        for i in range(1, n):
            prev = self.frames[i - 1]
            curr = self.frames[i]

            # Scene cut detection
            is_cut = False
            if prev.image is not None and curr.image is not None:
                is_cut, _ = detect_scene_cut(prev.image, curr.image)

            if is_cut:
                result.scene_cuts_excluded += 1
                result.scene_cut_indices.append(curr.frame_index)
                continue

            # 1. Luminance delta & SSIM drop
            if prev.face_roi is not None and curr.face_roi is not None:
                l1_d = compute_luminance_l1_delta(prev.face_roi, curr.face_roi)
                s_drop = compute_ssim_drop(prev.face_roi, curr.face_roi)
                result.per_frame_l1_delta.append(l1_d)
                result.per_frame_ssim_drop.append(s_drop)

            # 2. Landmark jitter
            if prev.landmarks is not None and curr.landmarks is not None:
                jitter = compute_landmark_jitter(prev.landmarks, curr.landmarks, compensate_translation=True)
                result.per_frame_landmark_jitter.append(jitter)

            # 3. Mask relative area change
            if prev.mask is not None and curr.mask is not None:
                rel_change = compute_mask_relative_change(prev.mask, curr.mask)
                result.per_frame_mask_relative_change.append(rel_change)

            result.valid_transitions += 1

        for f in self.frames:
            if f.mask is not None:
                masks_for_var.append(f.mask)

        # Aggregate averages
        if result.per_frame_l1_delta:
            result.mean_l1_luminance_delta = float(np.mean(result.per_frame_l1_delta))
        if result.per_frame_ssim_drop:
            result.mean_ssim_drop = float(np.mean(result.per_frame_ssim_drop))
        if result.per_frame_landmark_jitter:
            result.mean_landmark_jitter_px = float(np.mean(result.per_frame_landmark_jitter))
        if result.per_frame_mask_relative_change:
            result.mask_area_relative_change = float(np.mean(result.per_frame_mask_relative_change))

        if len(masks_for_var) >= 2:
            _, norm_var = compute_mask_area_variance(masks_for_var)
            result.mask_area_variance = float(norm_var)

        return result

    def compute_angle_bucket_report(self) -> AngleBucketReportResult:
        """Generate stability and coverage metrics grouped by head yaw buckets."""
        report = AngleBucketReportResult()
        buckets: Dict[str, List[FrameMetricRecord]] = {b: [] for b in YAW_BUCKETS}

        for f in self.frames:
            yaw_val = f.yaw if f.yaw is not None else 0.0
            bucket_key = get_yaw_bucket(yaw_val)
            buckets[bucket_key].append(f)

        total_detected_all = 0
        total_swapped_all = 0
        all_seam_errors: List[float] = []

        for b_name in YAW_BUCKETS:
            frames_in_b = buckets[b_name]
            total_b = len(frames_in_b)
            detected_b = sum(1 for f in frames_in_b if f.is_detected)
            swapped_b = sum(1 for f in frames_in_b if f.is_detected and f.is_swapped)

            det_recall = float(detected_b / total_b) if total_b > 0 else 0.0
            swap_cov = float(swapped_b / detected_b) if detected_b > 0 else 0.0

            seam_scores: List[float] = []
            yaws_b: List[float] = []

            for f in frames_in_b:
                if f.yaw is not None:
                    yaws_b.append(abs(float(f.yaw)))
                if f.is_swapped and f.image is not None and f.mask is not None:
                    score = compute_seam_error(f.image, f.mask, target_frame=f.target_image)
                    seam_scores.append(score)
                    all_seam_errors.append(score)

            mean_seam = float(np.mean(seam_scores)) if seam_scores else 0.0
            mean_yaw = float(np.mean(yaws_b)) if yaws_b else 0.0

            stats = AngleBucketStats(
                bucket_name=b_name,
                total_frames=total_b,
                detected_frames=detected_b,
                swapped_frames=swapped_b,
                detection_recall=det_recall,
                swap_coverage_rate=swap_cov,
                seam_error_score=mean_seam,
                mean_abs_yaw=mean_yaw,
            )
            report.buckets[b_name] = stats
            total_detected_all += detected_b
            total_swapped_all += swapped_b

        total_all = len(self.frames)
        report.total_frames_analyzed = total_all
        report.overall_detection_recall = float(total_detected_all / total_all) if total_all > 0 else 0.0
        report.overall_swap_coverage = float(total_swapped_all / total_detected_all) if total_detected_all > 0 else 0.0
        report.overall_seam_error = float(np.mean(all_seam_errors)) if all_seam_errors else 0.0

        return report

    def compute_summary(self) -> Dict[str, Any]:
        """Compute full combined metrics dictionary ready for JSON serialization."""
        flicker_res = self.compute_flicker_metrics()
        angle_res = self.compute_angle_bucket_report()
        combined = BenchmarkMetricsResult(flicker=flicker_res, angle_report=angle_res)
        return combined.to_dict()


def evaluate_benchmark_metrics(
    frames_data: Sequence[Union[FrameMetricRecord, Dict[str, Any]]]
) -> Dict[str, Any]:
    """Convenience function to evaluate metrics from a list of FrameMetricRecord or dicts."""
    tracker = BenchmarkMetricsTracker()
    for item in frames_data:
        if isinstance(item, FrameMetricRecord):
            tracker.add_frame_record(item)
        elif isinstance(item, dict):
            tracker.add_frame(**item)
    return tracker.compute_summary()


__all__ = [
    "AngleBucketReportResult",
    "AngleBucketStats",
    "BenchmarkMetricsResult",
    "BenchmarkMetricsTracker",
    "FlickerMetricResult",
    "FrameMetricRecord",
    "YAW_BUCKETS",
    "compute_landmark_jitter",
    "compute_luminance_l1_delta",
    "compute_mask_area",
    "compute_mask_area_variance",
    "compute_mask_relative_change",
    "compute_seam_error",
    "compute_ssim",
    "compute_ssim_drop",
    "detect_scene_cut",
    "estimate_head_yaw",
    "evaluate_benchmark_metrics",
    "extract_face_roi",
    "get_yaw_bucket",
]


if __name__ == "__main__":
    print("Testing bench_metrics with synthetic sequence...")
    rs = np.random.RandomState(42)
    tracker = BenchmarkMetricsTracker()

    base_frame = np.full((120, 120, 3), 100, dtype=np.uint8)
    base_mask = np.zeros((120, 120), dtype=np.uint8)
    base_mask[30:90, 30:90] = 255
    kps_base = np.array([
        [40.0, 50.0],
        [80.0, 50.0],
        [60.0, 70.0],
        [45.0, 90.0],
        [75.0, 90.0],
    ])

    # Generate 10 test frames
    for idx in range(10):
        f = base_frame.copy()
        # Add slight intensity noise
        f = np.clip(f.astype(int) + rs.randint(-3, 4, size=f.shape), 0, 255).astype(np.uint8)
        # Jitter landmarks slightly
        kps = kps_base + rs.randn(*kps_base.shape) * 0.5
        yaw_val = float(idx * 8.0)  # spans 0 to 72 deg (crosses all buckets)
        swapped = (idx % 2 == 0)

        tracker.add_frame(
            frame_index=idx,
            image=f,
            landmarks=kps,
            mask=base_mask,
            is_detected=True,
            is_swapped=swapped,
            yaw=yaw_val,
        )

    summary = tracker.compute_summary()
    import json
    print(json.dumps(summary, indent=2))
    print("bench_metrics self-test completed successfully.")

