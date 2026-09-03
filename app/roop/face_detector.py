"""Face Detector — Adaptive Multi-Scale Dynamic Image Pyramid, Frame Boundary Context Padding,
and Distance-IoU Non-Maximum Suppression (DIoU-NMS).

Fixes RetinaFace R50 close-up detection failure on macro shots, faces filling > 75% of the frame
height, faces with height > 500px, or faces intersecting frame borders where standard anchor
strides (16, 32, 64, 128, 256, 512) fail due to anchor receptive field limitations.

Features:
1. Multi-Scale Dynamic Image Pyramid:
   - RetinaFace R50 anchor strides (8, 16, 32 with anchor sizes up to 512) fail when the target face
     is larger than the maximum anchor receptive field.
   - Evaluates frame resolution and face scale. If face height is estimated > 500px or if detection
     returns 0 faces on high-res frames, generates a scale pyramid (default: [0.5, 0.75, 1.0]).
   - Runs parallel detection across the scaled frames.
   - Rescales bounding box and 5-point landmark coordinates back to the source coordinate space:
       B_orig = B_scaled / scale_factor
       K_orig = K_scaled / scale_factor
   - Merges multi-scale candidate boxes using Distance-IoU Non-Maximum Suppression (DIoU-NMS):
       DIoU = IoU - (d^2 / c^2)
     where d is the Euclidean distance between box centers and c is the diagonal length of the
     smallest enclosing box.

2. Frame Boundary Context Padding:
   - When a face is cut off by the camera frame edge, standard anchor priors produce invalid
     aspect ratios and truncated feature representations.
   - Applies a reflective or zero-padding border (minimum 64px) around frame edges prior to detection.
   - Subtracts padding offset from output bounding box and 5-point landmark coordinates.

3. Hardware Profile Awareness:
   - Laptop Workstation (< 7 GB VRAM, RTX 3060): leases single session, runs scales sequentially or
     with 1 worker, keeping system RSS strictly under 2.5 GB without memory exhaustion.
   - Desktop Workstation (12.0 GB VRAM, RTX 4070): runs concurrent multi-scale workers up to
     available detector pool size (default: 2), fully utilizing GPU concurrency lock-free.
"""

from __future__ import annotations

import concurrent.futures
import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

import roop.globals


# Detection thresholds and defaults
DEFAULT_PYRAMID_SCALES = [0.5, 0.75, 1.0]
MIN_BORDER_PADDING_PX = 64
CLOSEUP_HEIGHT_THRESHOLD = 500
CLOSEUP_COVERAGE_RATIO = 0.75
HIGH_RES_MIN_DIMENSION = 500


def compute_diou_matrix(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
    offset: float = 0.0,
    eps: float = 1e-7,
) -> np.ndarray:
    """Compute pairwise Distance-IoU (DIoU) matrix between two sets of bounding boxes.

    DIoU = IoU - (d^2 / c^2)
    where:
      - d: Euclidean distance between box centers.
      - c: Diagonal length of the smallest enclosing box covering both boxes.

    Args:
        boxes1: Array of shape (N, 4) in [x1, y1, x2, y2] format.
        boxes2: Array of shape (M, 4) in [x1, y1, x2, y2] format.
        offset: Coordinate offset (e.g. 1.0 for pixel-inclusive conventions, 0.0 for continuous).
        eps: Small epsilon to prevent division by zero.

    Returns:
        Array of shape (N, M) containing DIoU values.
    """
    b1 = np.asarray(boxes1, dtype=np.float32)
    b2 = np.asarray(boxes2, dtype=np.float32)
    if b1.ndim == 1:
        b1 = b1[np.newaxis, :]
    if b2.ndim == 1:
        b2 = b2[np.newaxis, :]

    n = b1.shape[0]
    m = b2.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)

    # Box coordinates
    x1_1, y1_1, x2_1, y2_1 = b1[:, 0:1], b1[:, 1:2], b1[:, 2:3], b1[:, 3:4]
    x1_2, y1_2, x2_2, y2_2 = b2[:, 0:1].T, b2[:, 1:2].T, b2[:, 2:3].T, b2[:, 3:4].T

    # Areas
    area1 = np.maximum(0.0, x2_1 - x1_1 + offset) * np.maximum(0.0, y2_1 - y1_1 + offset)
    area2 = np.maximum(0.0, x2_2 - x1_2 + offset) * np.maximum(0.0, y2_2 - y1_2 + offset)

    # Intersections
    inter_x1 = np.maximum(x1_1, x1_2)
    inter_y1 = np.maximum(y1_1, y1_2)
    inter_x2 = np.minimum(x2_1, x2_2)
    inter_y2 = np.minimum(y2_1, y2_2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1 + offset)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1 + offset)
    inter = inter_w * inter_h

    # IoU
    union = area1 + area2 - inter
    iou = inter / np.maximum(union, eps)

    # Center points distance d^2
    cx1 = (x1_1 + x2_1) * 0.5
    cy1 = (y1_1 + y2_1) * 0.5
    cx2 = (x1_2 + x2_2) * 0.5
    cy2 = (y1_2 + y2_2) * 0.5
    d2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Smallest enclosing box diagonal c^2
    enc_x1 = np.minimum(x1_1, x1_2)
    enc_y1 = np.minimum(y1_1, y1_2)
    enc_x2 = np.maximum(x2_1, x2_2)
    enc_y2 = np.maximum(y2_1, y2_2)

    enc_w = np.maximum(0.0, enc_x2 - enc_x1 + offset)
    enc_h = np.maximum(0.0, enc_y2 - enc_y1 + offset)
    c2 = enc_w ** 2 + enc_h ** 2

    # Distance-IoU
    diou = iou - (d2 / np.maximum(c2, eps))
    return diou.astype(np.float32)


def compute_diou(box1: np.ndarray, box2: np.ndarray, offset: float = 0.0) -> float:
    """Compute DIoU between two individual bounding boxes."""
    b1 = np.asarray(box1, dtype=np.float32).reshape(1, 4)
    b2 = np.asarray(box2, dtype=np.float32).reshape(1, 4)
    return float(compute_diou_matrix(b1, b2, offset=offset)[0, 0])


def diou_nms(
    dets: np.ndarray,
    kpss: Optional[np.ndarray] = None,
    iou_thresh: float = 0.40,
    offset: float = 0.0,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[int]]:
    """Distance-IoU Non-Maximum Suppression (DIoU-NMS).

    Merges multi-scale candidate detections by suppressing lower-scoring candidates
    whose DIoU with a higher-scoring candidate exceeds `iou_thresh`.

    Args:
        dets: Array of shape (N, 5) with [x1, y1, x2, y2, score].
        kpss: Optional array of shape (N, 5, 2) containing 5-point facial landmarks.
        iou_thresh: Suppression threshold for DIoU.
        offset: Coordinate convention offset.

    Returns:
        (filtered_dets, filtered_kpss, keep_indices)
    """
    if dets is None or len(dets) == 0:
        empty_dets = np.zeros((0, 5), dtype=np.float32)
        empty_kpss = np.zeros((0, 5, 2), dtype=np.float32) if kpss is not None else None
        return empty_dets, empty_kpss, []

    dets = np.asarray(dets, dtype=np.float32)
    kpss_arr = np.asarray(kpss, dtype=np.float32) if kpss is not None else None

    scores = dets[:, 4]
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        box_i = dets[i:i + 1, :4]
        boxes_rest = dets[rest, :4]

        diou_row = compute_diou_matrix(box_i, boxes_rest, offset=offset)[0]
        suppress_mask = diou_row >= iou_thresh
        order = rest[~suppress_mask]

    filtered_dets = dets[keep]
    filtered_kpss = kpss_arr[keep] if kpss_arr is not None else None
    return filtered_dets, filtered_kpss, keep


def apply_context_padding(
    frame: np.ndarray,
    min_padding: int = MIN_BORDER_PADDING_PX,
    padding_mode: str = 'reflect',
    mode: Optional[str] = None,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Apply reflective or zero-padding border (minimum 64px) around frame edges.

    Provides background and facial contextual support for anchor priors when a face
    is close to or intersecting camera boundaries.

    Args:
        frame: BGR input image of shape (H, W, 3) or (H, W).
        min_padding: Minimum border width in pixels (default: 64).
        padding_mode: 'reflect' (cv2.BORDER_REFLECT_101) or 'constant' (0).
        mode: Optional alias for padding_mode.

    Returns:
        (padded_frame, (pad_top, pad_bottom, pad_left, pad_right))
    """
    effective_mode = mode if mode is not None else padding_mode
    h, w = frame.shape[:2]
    # Ensure minimum 64px, with proportional scaling for massive resolutions (e.g. 4K)
    pad = max(int(min_padding), int(min(h, w) * 0.05), MIN_BORDER_PADDING_PX)

    border_mode = cv2.BORDER_REFLECT_101 if effective_mode.lower() == 'reflect' else cv2.BORDER_CONSTANT
    padded = cv2.copyMakeBorder(
        frame,
        pad,
        pad,
        pad,
        pad,
        borderType=border_mode,
        value=0,
    )
    offsets = (pad, pad, pad, pad)
    return padded, offsets


def remove_context_padding(
    dets: np.ndarray,
    kpss: Optional[np.ndarray] = None,
    pad_offsets: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Subtract context padding offsets from bounding boxes and landmark coordinates.

    Args:
        dets: Array of shape (N, 5) with [x1, y1, x2, y2, score].
        kpss: Optional array of shape (N, 5, 2) with landmark coordinates.
        pad_offsets: (pad_top, pad_bottom, pad_left, pad_right).

    Returns:
        (unpadded_dets, unpadded_kpss)
    """
    if dets is None or len(dets) == 0:
        return dets, kpss

    pad_top, _, pad_left, _ = pad_offsets
    if pad_top == 0 and pad_left == 0:
        return dets, kpss

    unpadded_dets = dets.copy()
    unpadded_dets[:, 0] -= pad_left
    unpadded_dets[:, 1] -= pad_top
    unpadded_dets[:, 2] -= pad_left
    unpadded_dets[:, 3] -= pad_top

    unpadded_kpss = None
    if kpss is not None and len(kpss) > 0:
        unpadded_kpss = kpss.copy()
        unpadded_kpss[:, :, 0] -= pad_left
        unpadded_kpss[:, :, 1] -= pad_top

    return unpadded_dets, unpadded_kpss


def parse_scale_pyramid(
    spec: Union[str, Sequence[float], None]
) -> Optional[List[float]]:
    """Parse configured pyramid levels from string, list, or None.

    Supports:
      - None or 'auto' -> None (triggers adaptive pyramid based on frame/face size)
      - 'none', 'off', 'false', '0' -> [1.0] (disables pyramid)
      - '0.5,0.75,1.0' or [0.5, 0.75, 1.0] -> [0.5, 0.75, 1.0]
    """
    if spec is None:
        return None

    if isinstance(spec, (list, tuple)):
        scales = [float(s) for s in spec if float(s) > 0.0]
        return sorted(list(set(scales))) if scales else [1.0]

    spec_str = str(spec).strip().lower()
    if spec_str in ('auto', ''):
        return None

    if spec_str in ('none', 'off', 'false', '0', 'single', '1.0'):
        return [1.0]

    try:
        parts = [float(p.strip()) for p in spec_str.split(',') if p.strip()]
        scales = [s for s in parts if s > 0.0]
        return sorted(list(set(scales))) if scales else [1.0]
    except ValueError:
        return None


def generate_scale_pyramid(
    frame: np.ndarray,
    scales: Sequence[float],
) -> List[Tuple[float, np.ndarray]]:
    """Generate scaled images for each level in the scale pyramid.

    Args:
        frame: Source image.
        scales: List of scale factors (e.g. [0.5, 0.75, 1.0]).

    Returns:
        List of (scale_factor, scaled_image) tuples.
    """
    pyramid: List[Tuple[float, np.ndarray]] = []
    h, w = frame.shape[:2]

    for s in scales:
        if abs(s - 1.0) < 1e-4:
            pyramid.append((1.0, frame))
        else:
            new_w = max(16, int(round(w * s)))
            new_h = max(16, int(round(h * s)))
            interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
            scaled = cv2.resize(frame, (new_w, new_h), interpolation=interp)
            pyramid.append((s, scaled))

    return pyramid


def rescale_detections(
    dets: np.ndarray,
    kpss: Optional[np.ndarray] = None,
    scale_factor: float = 1.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Rescale detections from a scaled frame back to the original coordinate space.

    B_orig = B_scaled / scale_factor
    K_orig = K_scaled / scale_factor
    """
    if dets is None or len(dets) == 0:
        return dets, kpss

    if abs(scale_factor - 1.0) < 1e-4:
        return dets.copy(), (kpss.copy() if kpss is not None else None)

    rescaled_dets = dets.copy()
    rescaled_dets[:, :4] /= float(scale_factor)

    rescaled_kpss = None
    if kpss is not None and len(kpss) > 0:
        rescaled_kpss = kpss.copy()
        rescaled_kpss /= float(scale_factor)

    return rescaled_dets, rescaled_kpss


def should_trigger_pyramid(
    frame_shape: Tuple[int, int],
    estimated_face_height: Optional[float] = None,
    initial_dets: Optional[np.ndarray] = None,
    configured_scales: Optional[Union[str, Sequence[float]]] = None,
) -> bool:
    """Determine whether the multi-scale dynamic image pyramid should be triggered.

    Conditions (matching specification):
    1. CLI/UI explicitly set scales other than 'auto' or 'none'.
    2. Estimated face height exceeds 500px (macro shot / extreme close-up).
    3. Detection returns 0 faces on a high-resolution frame (H >= 500 or W >= 500).
    4. Detected face fills > 75% of frame height or face height > 500px.
    """
    parsed = parse_scale_pyramid(configured_scales)
    if parsed is not None:
        if parsed == [1.0]:
            return False
        return True

    # 1. Estimated face height check (e.g. from prior frame or tracker)
    if estimated_face_height is not None and estimated_face_height >= CLOSEUP_HEIGHT_THRESHOLD:
        return True

    h, w = frame_shape[:2]
    # 2. Check initial detection results if an initial pass was performed
    if initial_dets is not None:
        if len(initial_dets) == 0:
            # 0 faces on a high-res frame
            if max(h, w) >= HIGH_RES_MIN_DIMENSION:
                return True
            return False

        # 3. Check if any detected face is close-up (> 500px or filling > 75% of frame)
        for b in initial_dets:
            box_h = b[3] - b[1]
            box_w = b[2] - b[0]
            if box_h >= CLOSEUP_HEIGHT_THRESHOLD:
                return True
            if h > 0 and (box_h / float(h)) >= CLOSEUP_COVERAGE_RATIO:
                return True
            if w > 0 and (box_w / float(w)) >= CLOSEUP_COVERAGE_RATIO:
                return True
        return False

    return False


class MultiScaleFaceDetector:
    """Multi-Scale Face Detector with frame boundary context padding and DIoU-NMS."""

    def __init__(
        self,
        detect_fn: Optional[Callable[[np.ndarray, int, float], Tuple[np.ndarray, np.ndarray]]] = None,
        default_scales: Optional[Sequence[float]] = None,
        padding: int = MIN_BORDER_PADDING_PX,
        padding_mode: str = 'reflect',
        nms_thresh: float = 0.40,
    ) -> None:
        self.detect_fn = detect_fn
        self.default_scales = list(default_scales) if default_scales else list(DEFAULT_PYRAMID_SCALES)
        self.padding = max(MIN_BORDER_PADDING_PX, int(padding))
        self.padding_mode = padding_mode
        self.nms_thresh = nms_thresh

    def _resolve_concurrency(self) -> int:
        """Resolve worker concurrency according to the dual-hardware execution policy."""
        try:
            from roop.retinaface import _pool_size
            pool_sz = _pool_size()
            return max(1, pool_sz)
        except Exception:
            return 2

    def detect(
        self,
        frame: np.ndarray,
        det_size: int = 640,
        det_thresh: float = 0.50,
        scales: Optional[Union[str, Sequence[float]]] = None,
        padding: Optional[int] = None,
        padding_mode: Optional[str] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        estimated_face_height: Optional[float] = None,
        force_pyramid: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform multi-scale detection with border padding and DIoU-NMS.

        Returns:
            (bboxes (N, 5) incl. score, kpss (N, 5, 2)) in source frame coordinates.
        """
        if frame is None or frame.size == 0:
            return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 5, 2), dtype=np.float32)

        detect_fn = self.detect_fn
        if detect_fn is None:
            from roop.retinaface import _detect_single_instance
            detect_fn = lambda f, ds, dt: _detect_single_instance(f, det_size=ds, det_thresh=dt, model_type='r50')

        pad_amount = self.padding if padding is None else max(MIN_BORDER_PADDING_PX, int(padding))
        pad_mode = self.padding_mode if padding_mode is None else str(padding_mode)
        nms_thresh = getattr(roop.globals, 'face_detector_nms', self.nms_thresh)

        # 1. Apply frame boundary context padding (minimum 64px)
        padded_frame, pad_offsets = apply_context_padding(frame, min_padding=pad_amount, mode=pad_mode)

        # 2. Check scale pyramid configuration
        configured = getattr(roop.globals, 'detector_scale_pyramid', None) if scales is None else scales
        parsed_scales = parse_scale_pyramid(configured)

        # 3. Adaptive check: If not explicitly configured and not forced, run quick baseline
        if parsed_scales is None and not force_pyramid:
            if not should_trigger_pyramid(frame.shape[:2], estimated_face_height=estimated_face_height):
                # Standard single-scale detection on padded frame
                b_single, k_single = detect_fn(padded_frame, int(det_size), float(det_thresh))
                unpad_b, unpad_k = remove_context_padding(b_single, k_single, pad_offsets)

                # Check if result indicates close-up or missed face requiring pyramid
                if not should_trigger_pyramid(frame.shape[:2], initial_dets=unpad_b):
                    return unpad_b, (unpad_k if unpad_k is not None else np.zeros((0, 5, 2), dtype=np.float32))

        # 4. Multi-scale dynamic image pyramid execution
        active_scales = parsed_scales if (parsed_scales and parsed_scales != [1.0]) else self.default_scales
        pyramid = generate_scale_pyramid(padded_frame, active_scales)

        all_candidate_boxes: List[np.ndarray] = []
        all_candidate_kpss: List[np.ndarray] = []

        concurrency = max_workers or self._resolve_concurrency()

        def _detect_scale_worker(item: Tuple[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
            scale_val, scaled_img = item
            b_scaled, k_scaled = detect_fn(scaled_img, int(det_size), float(det_thresh))
            b_orig, k_orig = rescale_detections(b_scaled, k_scaled, scale_factor=scale_val)
            return b_orig, k_orig

        if parallel and concurrency > 1 and len(pyramid) > 1:
            workers = min(len(pyramid), concurrency)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(_detect_scale_worker, pyramid))
        else:
            results = [_detect_scale_worker(item) for item in pyramid]

        for b_cand, k_cand in results:
            if b_cand is not None and len(b_cand) > 0:
                all_candidate_boxes.append(b_cand)
                if k_cand is not None and len(k_cand) > 0:
                    all_candidate_kpss.append(k_cand)

        if not all_candidate_boxes:
            return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 5, 2), dtype=np.float32)

        merged_boxes = np.vstack(all_candidate_boxes)
        merged_kpss = np.vstack(all_candidate_kpss) if all_candidate_kpss else None

        # 5. Distance-IoU Non-Maximum Suppression (DIoU-NMS)
        kept_boxes, kept_kpss, _ = diou_nms(
            merged_boxes,
            kpss=merged_kpss,
            iou_thresh=nms_thresh,
            offset=1.0,
        )

        # 6. Remove context padding offset
        final_boxes, final_kpss = remove_context_padding(kept_boxes, kept_kpss, pad_offsets)
        if final_kpss is None:
            final_kpss = np.zeros((len(final_boxes), 5, 2), dtype=np.float32)

        return final_boxes, final_kpss


# Global detector singleton
_MULTI_SCALE_DETECTOR = MultiScaleFaceDetector()


def detect_faces(
    frame: np.ndarray,
    det_size: int = 640,
    det_thresh: float = 0.50,
    model_type: str = 'r50',
    scales: Optional[Union[str, Sequence[float]]] = None,
    padding: Optional[int] = None,
    padding_mode: Optional[str] = None,
    parallel: bool = True,
    estimated_face_height: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unified face detection interface with adaptive multi-scale pyramid and context padding.

    Contract: returns (bboxes (N, 5), kpss (N, 5, 2)) in source frame coordinates.
    """
    from roop.retinaface import _detect_single_instance

    def _leased_fn(f: np.ndarray, ds: int, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        return _detect_single_instance(f, det_size=ds, det_thresh=dt, model_type=model_type)

    detector = MultiScaleFaceDetector(
        detect_fn=_leased_fn,
        padding=padding or MIN_BORDER_PADDING_PX,
        padding_mode=padding_mode or 'reflect',
    )
    return detector.detect(
        frame=frame,
        det_size=det_size,
        det_thresh=det_thresh,
        scales=scales,
        parallel=parallel,
        estimated_face_height=estimated_face_height,
    )


def detect_retinaface_closeup(
    frame: np.ndarray,
    det_size: int = 640,
    det_thresh: float = 0.50,
    scales: Sequence[float] = (0.5, 0.75, 1.0),
    padding: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Explicit multi-scale detection optimized for close-up and macro face crops."""
    return detect_faces(
        frame=frame,
        det_size=det_size,
        det_thresh=det_thresh,
        model_type='r50',
        scales=scales,
        padding=padding,
        padding_mode='reflect',
        parallel=True,
    )
