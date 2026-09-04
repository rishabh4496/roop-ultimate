"""Motion blur harmonization module for video face synthesis.

Fixes the "sharp sticker" artifact on moving subjects by harmonizing swapped
face sharpness with background and target actor motion blur prior to alpha compositing.

Mathematical Specifications:
1. Target Motion-Blur Estimation:
   Compute the blur metric of the target face crop prior to swapping using
   the variance of the Laplacian:
     B_target = Var(Laplacian(I_target_gray))

2. Adaptive Directional Convolution:
   If B_target < BLUR_THRESHOLD (indicating rapid movement or out-of-focus capture):
     - Calculate optical flow magnitude vector (u, v) between frames t-1 and t.
     - Construct an anisotropic directional motion-blur kernel K of
       length L = sqrt(u^2 + v^2) at angle phi = atan2(v, u).
     - Convolve the swapped and enhanced face patch with K before alpha compositing:
       I_swapped_harmonized = cv2.filter2D(I_swapped, -1, K)
"""

import os
import threading
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

DEFAULT_BLUR_THRESHOLD: float = float(os.environ.get('ROOP_MOTION_BLUR_THRESHOLD', '100.0'))
DEFAULT_MAX_KERNEL_SIZE: int = int(os.environ.get('ROOP_MOTION_BLUR_MAX_KERNEL', '31'))
BLUR_THRESHOLD = DEFAULT_BLUR_THRESHOLD


def compute_blur_metric(image: np.ndarray) -> float:
    """Compute blur metric of an image using the variance of the Laplacian.

    Formula:
        B_target = Var(Laplacian(I_target_gray))

    Higher values indicate crisp, high-frequency details.
    Lower values (< BLUR_THRESHOLD) indicate motion blur or out-of-focus capture.
    """
    if image is None or image.size == 0:
        return 0.0
    img = np.asarray(image)
    if img.ndim == 3:
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        gray = img
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def calculate_optical_flow_vector(
    prev_image: Optional[np.ndarray],
    curr_image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    flow_size: int = 128
) -> Tuple[float, float]:
    """Calculate optical flow motion vector (u, v) between frames t-1 and t.

    Parameters:
        prev_image: Frame or face crop at t-1.
        curr_image: Frame or face crop at t.
        mask: Optional binary or float mask defining face/foreground region.
        flow_size: Downsampled dimension for fast, robust motion field evaluation.

    Returns:
        (u, v): Horizontal and vertical displacement in current image pixel units.
    """
    if prev_image is None or curr_image is None:
        return 0.0, 0.0

    def _to_gray_small(img: np.ndarray, size: int) -> np.ndarray:
        arr = np.asarray(img)
        if arr.ndim == 3:
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)

    h, w = curr_image.shape[:2]
    if h == 0 or w == 0:
        return 0.0, 0.0

    prev_gray = _to_gray_small(prev_image, flow_size)
    curr_gray = _to_gray_small(curr_image, flow_size)

    try:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
    except Exception:
        return 0.0, 0.0

    if not np.isfinite(flow).all():
        return 0.0, 0.0

    scale_x = float(w) / float(flow_size)
    scale_y = float(h) / float(flow_size)
    flow[..., 0] *= scale_x
    flow[..., 1] *= scale_y

    if mask is not None and mask.size > 0:
        mask_res = cv2.resize(mask.astype(np.float32), (flow_size, flow_size), interpolation=cv2.INTER_NEAREST)
        valid = mask_res > 0.5
        if np.any(valid):
            u = float(np.median(flow[valid, 0]))
            v = float(np.median(flow[valid, 1]))
            return u, v

    # Default: evaluate central 60% region to reject crop boundary remap artifacts
    pad_y = max(1, int(flow_size * 0.2))
    pad_x = max(1, int(flow_size * 0.2))
    center_region = flow[pad_y:flow_size - pad_y, pad_x:flow_size - pad_x]
    u = float(np.median(center_region[..., 0]))
    v = float(np.median(center_region[..., 1]))
    return u, v


def construct_motion_blur_kernel(
    length: float,
    angle: float,
    max_kernel_size: int = DEFAULT_MAX_KERNEL_SIZE
) -> np.ndarray:
    """Construct an anisotropic directional motion-blur kernel K.

    Parameters:
        length: Motion length L = sqrt(u^2 + v^2) in pixels.
        angle: Motion orientation phi = atan2(v, u) in radians.
        max_kernel_size: Upper bound on odd kernel dimension to preserve bounded execution.

    Returns:
        K: Normalized 2D convolution kernel (sum == 1.0) of shape (k, k).
    """
    length_val = float(max(0.0, length))
    if length_val < 1.0:
        return np.array([[1.0]], dtype=np.float32)

    # Bound kernel size and ensure odd dimension
    effective_length = min(length_val, float(max_kernel_size))
    k = max(3, int(np.ceil(effective_length)))
    if k % 2 == 0:
        k += 1
    if k > max_kernel_size:
        k = max_kernel_size if max_kernel_size % 2 == 1 else max_kernel_size - 1

    c = (k - 1) / 2.0
    u = float(effective_length * np.cos(angle))
    v = float(effective_length * np.sin(angle))

    p0 = (int(round(c - u / 2.0)), int(round(c - v / 2.0)))
    p1 = (int(round(c + u / 2.0)), int(round(c + v / 2.0)))

    # Clamp endpoints inside the kernel grid
    p0 = (max(0, min(k - 1, p0[0])), max(0, min(k - 1, p0[1])))
    p1 = (max(0, min(k - 1, p1[0])), max(0, min(k - 1, p1[1])))

    kernel = np.zeros((k, k), dtype=np.float32)
    cv2.line(kernel, p0, p1, 1.0, thickness=1, lineType=cv2.LINE_AA)

    total_sum = float(kernel.sum())
    if total_sum > 1e-6:
        kernel /= total_sum
    else:
        kernel[int(c), int(c)] = 1.0

    return kernel


def construct_motion_blur_kernel_from_vector(
    u: float,
    v: float,
    max_kernel_size: int = DEFAULT_MAX_KERNEL_SIZE
) -> np.ndarray:
    """Construct motion blur kernel K directly from optical flow displacement vector (u, v).

    Formula:
        L = sqrt(u^2 + v^2)
        phi = atan2(v, u)
    """
    length = float(np.hypot(u, v))
    angle = float(np.arctan2(v, u))
    return construct_motion_blur_kernel(length, angle, max_kernel_size=max_kernel_size)


def apply_motion_blur(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve image patch with motion-blur kernel K.

    Formula:
        I_swapped_harmonized = cv2.filter2D(I_swapped, -1, K)
    """
    if kernel is None or kernel.size <= 1:
        return image.copy()
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def harmonize_motion_blur(
    swapped_face: np.ndarray,
    target_crop: np.ndarray,
    flow_vector: Optional[Tuple[float, float]] = None,
    prev_crop: Optional[np.ndarray] = None,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    max_kernel_size: int = DEFAULT_MAX_KERNEL_SIZE
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Harmonize swapped face sharpness with target actor / background motion blur.

    Pipeline:
    1. B_target = Var(Laplacian(I_target_gray))
    2. If B_target < blur_threshold:
       - Determine optical flow vector (u, v) from flow_vector or (prev_crop, target_crop).
       - L = sqrt(u^2 + v^2), phi = atan2(v, u).
       - If L >= 1.0:
           K = construct_motion_blur_kernel(L, phi)
           I_swapped_harmonized = cv2.filter2D(I_swapped, -1, K)
    """
    b_target = compute_blur_metric(target_crop)
    meta: Dict[str, Any] = {
        'b_target': b_target,
        'blur_threshold': blur_threshold,
        'is_motion_blurred': False,
        'flow_vector': (0.0, 0.0),
        'flow_length': 0.0,
        'flow_angle': 0.0,
        'kernel': None,
    }

    if b_target >= blur_threshold:
        return swapped_face.copy(), meta

    # B_target < blur_threshold: target is blurred (movement or out-of-focus)
    u, v = 0.0, 0.0
    if flow_vector is not None:
        u, v = float(flow_vector[0]), float(flow_vector[1])
    elif prev_crop is not None:
        u, v = calculate_optical_flow_vector(prev_crop, target_crop)

    length = float(np.hypot(u, v))
    angle = float(np.arctan2(v, u))
    meta['flow_vector'] = (u, v)
    meta['flow_length'] = length
    meta['flow_angle'] = angle

    if length >= 1.0:
        kernel = construct_motion_blur_kernel(length, angle, max_kernel_size=max_kernel_size)
        meta['kernel'] = kernel
        meta['is_motion_blurred'] = True
        harmonized = apply_motion_blur(swapped_face, kernel)
        return harmonized, meta

    return swapped_face.copy(), meta


class MotionBlurHarmonizer:
    """Thread-safe stateful motion-blur harmonizer maintaining per-track frame history."""

    def __init__(
        self,
        blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
        max_kernel_size: int = DEFAULT_MAX_KERNEL_SIZE
    ) -> None:
        self.blur_threshold = blur_threshold
        self.max_kernel_size = max_kernel_size
        self._states: Dict[Any, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def harmonize(
        self,
        swapped_face: np.ndarray,
        target_crop: np.ndarray,
        track_id: Any = 0,
        flow_vector: Optional[Tuple[float, float]] = None,
        blur_threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Harmonize swapped face with motion blur from previous crop in the tracklet."""
        eff_threshold = self.blur_threshold if blur_threshold is None else float(blur_threshold)

        with self._lock:
            state = self._states.get(track_id)
            prev_crop = state['prev_crop'] if state is not None else None

            harmonized, meta = harmonize_motion_blur(
                swapped_face=swapped_face,
                target_crop=target_crop,
                flow_vector=flow_vector,
                prev_crop=prev_crop,
                blur_threshold=eff_threshold,
                max_kernel_size=self.max_kernel_size
            )

            self._states[track_id] = {
                'prev_crop': target_crop.copy(),
                'last_flow': meta.get('flow_vector', (0.0, 0.0)),
            }
            return harmonized, meta

    def reset(self, track_id: Optional[Any] = None) -> None:
        """Reset motion blur history for a tracklet or all tracks."""
        with self._lock:
            if track_id is None:
                self._states.clear()
            else:
                self._states.pop(track_id, None)
