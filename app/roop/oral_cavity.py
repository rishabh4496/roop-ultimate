"""Oral cavity and natural teeth restoration module using BiSeNet semantic segmentation.

Reconstructs natural inner-mouth geometry (teeth boundaries, tongue placement,
pharyngeal depth) conditioned on active speech phonemes, and eliminates
the "clamped lip" or "blurry teeth" artifact produced when InSwapper attempts
to swap closed lips onto an open-mouthed target.
"""

import math
import cv2
import numpy as np

import roop.globals
from roop.typing import Frame
from roop.degrade import swallowed as _swallowed


def _get_landmarks_106(face):
    """Retrieve 106-point 2D landmarks from Face object or dict."""
    if face is None:
        return None
    lm = getattr(face, 'landmark_2d_106', None)
    if lm is None and isinstance(face, dict):
        lm = face.get('landmark_2d_106')
    if lm is None:
        return None
    pts = np.asarray(lm, dtype=np.float32).reshape(-1, 2)
    return pts if pts.shape[0] >= 71 else None


def _get_inner_mouth_landmarks(face):
    """Extract inner lip / oral cavity landmark points."""
    pts106 = _get_landmarks_106(face)
    if pts106 is not None:
        # In 106-point landmark schema:
        # 52..71: mouth contour. Points 66..71 define the inner oral cavity aperture.
        # Points: 66 (upper center), 67 (upper left), 68 (lower left),
        # 69 (lower center), 70 (lower right), 71 (upper right).
        inner_pts = pts106[66:72] if pts106.shape[0] >= 72 else pts106[52:71]
        return inner_pts

    # Fallback to 68-point landmarks if present
    lm68 = getattr(face, 'landmarks_68', None)
    if lm68 is None and isinstance(face, dict):
        lm68 = face.get('landmarks_68')
    if lm68 is not None:
        pts68 = np.asarray(lm68, dtype=np.float32).reshape(-1, 2)
        if pts68.shape[0] >= 68:
            return pts68[60:68]  # inner lip contour in 68-point

    return None


def detect_oral_cavity_mask(frame: Frame, face, parser=None) -> tuple[np.ndarray, dict]:
    """Detect oral cavity region using BiSeNet semantic segmentation (Class 11)
    or landmark-guided inner lip contour.
    
    Returns:
      (cavity_mask, metrics) where cavity_mask is a uint8 (H, W) mask [0, 255]
      and metrics contains 'is_open', 'area', 'aperture_h', 'aperture_w'.
    """
    h, w = frame.shape[:2]
    empty_mask = np.zeros((h, w), dtype=np.uint8)
    empty_metrics = {
        'is_open': False,
        'area': 0,
        'aperture_h': 0,
        'aperture_w': 0,
        'aspect_ratio': 0.0,
        'bbox': None
    }

    if frame is None or face is None:
        return empty_mask, empty_metrics

    # 1. Use BiSeNet semantic parser if available
    if parser is not None and hasattr(parser, 'RunLabels'):
        try:
            from roop.processors.Lipsync_MuseTalk import face_bbox_crop
            bbox = getattr(face, 'bbox', None)
            if bbox is None and isinstance(face, dict):
                bbox = face.get('bbox')
            if bbox is not None:
                crop, crop_box = face_bbox_crop(frame, bbox)
                if crop is not None and crop_box is not None:
                    # RunLabels returns (512, 512) class IDs:
                    # 11: mouth (oral cavity / teeth / tongue)
                    labels = parser.RunLabels(crop)
                    mouth_cavity_crop = (labels == 11).astype(np.uint8) * 255
                    if mouth_cavity_crop.any():
                        x1, y1, x2, y2 = crop_box
                        bw, bh = x2 - x1, y2 - y1
                        resized_cavity = cv2.resize(mouth_cavity_crop, (bw, bh), interpolation=cv2.INTER_NEAREST)
                        mask = np.zeros((h, w), dtype=np.uint8)
                        mask[y1:y2, x1:x2] = resized_cavity

                        area = int((mask > 0).sum())
                        if area > 12:
                            ys, xs = np.where(mask > 0)
                            ah = int(ys.max() - ys.min() + 1)
                            aw = int(xs.max() - xs.min() + 1)
                            metrics = {
                                'is_open': ah >= 4 and area >= 16,
                                'area': area,
                                'aperture_h': ah,
                                'aperture_w': aw,
                                'aspect_ratio': float(ah) / max(1.0, float(aw)),
                                'bbox': (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                            }
                            return mask, metrics
        except Exception as _degrade_error:
            _swallowed("roop/oral_cavity.py:110", _degrade_error,
                       "landmark fallback continued")

    # 2. Geometric landmark fallback
    inner_pts = _get_inner_mouth_landmarks(face)
    if inner_pts is not None and inner_pts.shape[0] >= 4 and np.isfinite(inner_pts).all():
        hull = cv2.convexHull(np.round(inner_pts).astype(np.int32))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        ys, xs = inner_pts[:, 1], inner_pts[:, 0]
        ah = int(ys.max() - ys.min())
        aw = int(xs.max() - xs.min())
        area = int((mask > 0).sum())
        is_open = ah >= 5 and area >= 20
        metrics = {
            'is_open': is_open,
            'area': area,
            'aperture_h': ah,
            'aperture_w': aw,
            'aspect_ratio': float(ah) / max(1.0, float(aw)),
            'bbox': (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        }
        return mask, metrics

    return empty_mask, empty_metrics


def detect_clamped_lip_artifact(swapped_frame: Frame, plate_frame: Frame, face, cavity_metrics: dict) -> dict:
    """Analyze if InSwapper produced a clamped lip (closed lips over open target mouth)
    or blurry teeth (loss of dental contrast and incisal boundaries).
    """
    result = {
        'clamped_lip': False,
        'blurry_teeth': False,
        'severity': 0.0,
    }
    if not cavity_metrics.get('is_open', False) or cavity_metrics.get('bbox') is None:
        return result

    x1, y1, x2, y2 = cavity_metrics['bbox']
    h, w = swapped_frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return result

    swap_roi = swapped_frame[y1:y2, x1:x2]
    plate_roi = plate_frame[y1:y2, x1:x2]
    if swap_roi.size == 0 or plate_roi.size == 0:
        return result

    # Check clamped lip: plate has open cavity contrast, but swap has collapsed variance
    gray_swap = cv2.cvtColor(swap_roi, cv2.COLOR_BGR2GRAY)
    gray_plate = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)

    var_swap = float(np.var(gray_swap))
    var_plate = float(np.var(gray_plate))

    # Blurry teeth: check high-frequency Laplacian variance in mouth center
    lap_swap = cv2.Laplacian(gray_swap, cv2.CV_32F).var()
    lap_plate = cv2.Laplacian(gray_plate, cv2.CV_32F).var()

    clamped = (var_swap < var_plate * 0.35 and cavity_metrics['aperture_h'] >= 6)
    blurry = (lap_swap < lap_plate * 0.40 and var_plate > 80.0)

    severity = 0.0
    if clamped:
        severity = max(severity, 0.9)
    if blurry:
        severity = max(severity, 0.7)

    result['clamped_lip'] = clamped
    result['blurry_teeth'] = blurry
    result['severity'] = severity
    return result


def reconstruct_inner_mouth_geometry(swapped_frame: Frame, plate_frame: Frame, face,
                                     phoneme_energy: float = 0.0, viseme_openness: float = 0.0,
                                     teeth_sharpness: float = 0.85, parser=None, region=None) -> Frame:
    """Reconstruct natural inner-mouth geometry (teeth boundaries, tongue placement, oral depth)
    conditioned on active speech phonemes. Prevents clamped-lip and blurry-teeth artifacts.
    """
    if swapped_frame is None or plate_frame is None or face is None:
        return swapped_frame

    # Detect oral cavity region
    cavity_mask, metrics = detect_oral_cavity_mask(plate_frame, face, parser=parser)
    if not metrics.get('is_open', False) and viseme_openness < 0.20:
        # Mouth is closed in both video target and driving speech
        return swapped_frame

    h, w = swapped_frame.shape[:2]
    bbox = metrics.get('bbox')
    if bbox is None:
        return swapped_frame

    x1, y1, x2, y2 = bbox
    # Add modest padding around oral cavity
    pad_x = max(2, int((x2 - x1) * 0.15))
    pad_y = max(2, int((y2 - y1) * 0.15))
    rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    rx2, ry2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    if rx2 <= rx1 or ry2 <= ry1:
        return swapped_frame

    cavity_sub = cavity_mask[ry1:ry2, rx1:rx2]
    if not cavity_sub.any():
        return swapped_frame

    swap_roi = swapped_frame[ry1:ry2, rx1:rx2].astype(np.float32)
    plate_roi = plate_frame[ry1:ry2, rx1:rx2].astype(np.float32)

    # ── 1. Teeth Boundary Delineation & Sharpness Reconstruction ──────────
    # Isolate tooth candidate pixels: high brightness, low saturation
    hsv_plate = cv2.cvtColor(plate_roi.astype(np.uint8), cv2.COLOR_BGR2HSV)
    gray_plate = cv2.cvtColor(plate_roi.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    v_chan = hsv_plate[:, :, 2]
    s_chan = hsv_plate[:, :, 1]

    # Dental mask: bright enamel, low chroma
    p50_v = float(np.percentile(v_chan[cavity_sub > 0], 50)) if (cavity_sub > 0).any() else 100.0
    thresh_v = max(100.0, p50_v + 15.0)
    teeth_mask = (cavity_sub > 0) & (v_chan >= thresh_v) & (s_chan <= 155)

    # Reconstruct crisp dental edges using multi-scale unsharp masking
    teeth_enhanced = plate_roi.copy()
    if teeth_mask.any():
        blurred = cv2.GaussianBlur(plate_roi, (0, 0), sigmaX=1.5)
        high_freq = plate_roi - blurred
        # Boost dental sharpness and interdental crevice delineation
        sharpness_boost = 1.0 + float(np.clip(teeth_sharpness, 0.0, 1.5)) * 0.75
        teeth_enhanced = np.clip(plate_roi + high_freq * sharpness_boost, 0.0, 255.0)

    # ── 2. Natural Oral Depth Gradient (Tongue and Pharynx) ───────────────
    # Oral cavity darkness increases with depth away from front teeth
    dist_map = cv2.distanceTransform(cavity_sub, cv2.DIST_L2, 3)
    max_d = float(dist_map.max()) if dist_map.max() > 0 else 1.0
    depth_gradient = (dist_map / max_d)[:, :, None]

    # Non-teeth oral cavity: tongue and deep pharyngeal cavity
    cavity_darkening = 1.0 - 0.35 * depth_gradient * (1.0 - teeth_mask[:, :, None].astype(np.float32))
    restored_interior = teeth_enhanced * cavity_darkening

    # ── 3. Phoneme Conditioning ───────────────────────────────────────────
    # Condition the reconstruction weight on speech phoneme energy/openness:
    # High openness (open vowels) -> full aperture restoration
    # Closed bilabials -> fade restoration to lip boundary
    speech_condition = 1.0
    if viseme_openness > 0.0:
        speech_condition = 0.5 + 0.5 * float(np.clip(viseme_openness, 0.0, 1.0))
    elif phoneme_energy > 0.0:
        speech_condition = 0.6 + 0.4 * float(np.clip(phoneme_energy, 0.0, 1.0))

    # ── 4. Seamless Feathered Composite ───────────────────────────────────
    # Erode outer boundary of cavity mask slightly to stay within inner lip vermilion
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inner_cavity = cv2.erode(cavity_sub, k, iterations=1)
    alpha = cv2.GaussianBlur(inner_cavity.astype(np.float32) / 255.0, (0, 0), sigmaX=1.2)
    alpha = np.clip(alpha * speech_condition, 0.0, 1.0)[:, :, None]

    if region is not None:
        own = region.crop(rx1, ry1, rx2 - rx1, ry2 - ry1)
        if own is not None:
            own = np.asarray(own, dtype=np.float32)
            if own.shape[:2] != alpha.shape[:2]:
                own = cv2.resize(own, (rx2 - rx1, ry2 - ry1), interpolation=cv2.INTER_LINEAR)
            alpha *= np.clip(own[:, :, None], 0.0, 1.0)

    # Blend restored inner mouth into swapped ROI
    blended_roi = swap_roi * (1.0 - alpha) + restored_interior * alpha
    out = swapped_frame.copy()
    out[ry1:ry2, rx1:rx2] = np.clip(blended_roi, 0, 255).astype(np.uint8)
    return out


def coarticulate_jawline_frame(frame: Frame, target_face, phoneme_energy: float = 0.0,
                              viseme_openness: float = 0.0, strength: float = 0.6) -> Frame:
    """Dynamically alter mouth and jawline geometry conditioned on speech phonemes.
    In 'Re-dub Sync' mode, this depresses and flexes the mandible / chin contour
    in sync with speech vowels and acoustic energy, producing authentic audio-driven
    co-articulation.
    """
    if frame is None or target_face is None or strength <= 0.0:
        return frame

    # Co-articulation displacement factor from acoustic features
    acoustic_factor = float(np.clip(0.6 * viseme_openness + 0.4 * phoneme_energy, 0.0, 1.0))
    if acoustic_factor < 0.08:
        # Neutral / resting jaw
        return frame

    pts106 = _get_landmarks_106(target_face)
    h, w = frame.shape[:2]

    if pts106 is not None and pts106.shape[0] >= 33:
        # Chin tip in 106-point is landmark 16. Mandible contour: 6..26
        chin_tip = pts106[16]
        nose_tip = pts106[46] if pts106.shape[0] >= 47 else pts106[49]
        face_h = float(np.linalg.norm(chin_tip - nose_tip))
        if face_h < 10.0:
            return frame

        # Maximum jaw depression: proportional to face height and acoustic factor
        max_disp_y = min(22.0, face_h * 0.16 * float(strength) * acoustic_factor)
        if max_disp_y < 1.0:
            return frame

        # Define source and destination control points for lower face warp
        # Central features (eyes, nose, upper cheeks) remain strictly pinned
        pinned_indices = [33, 38, 46, 72, 73, 75, 76] if pts106.shape[0] >= 77 else [33, 46]
        src_pts = []
        dst_pts = []

        for idx in pinned_indices:
            if idx < pts106.shape[0]:
                pt = pts106[idx]
                src_pts.append(pt)
                dst_pts.append(pt)

        # Jawline contour points (mandible)
        jaw_indices = list(range(8, 25))  # lower jaw around chin
        for idx in jaw_indices:
            if idx < pts106.shape[0]:
                pt = pts106[idx]
                src_pts.append(pt)
                # Weight displacement by closeness to chin tip (index 16)
                dist_to_center = abs(idx - 16) / 8.0
                weight = max(0.0, 1.0 - dist_to_center)
                dy = max_disp_y * weight
                # Subtle lateral widening on wide vowels
                dx = (pt[0] - chin_tip[0]) * 0.05 * weight
                dst_pts.append([pt[0] + dx, pt[1] + dy])

        src_pts = np.asarray(src_pts, dtype=np.float32)
        dst_pts = np.asarray(dst_pts, dtype=np.float32)

        # Bounding box around lower face
        x0, y0 = src_pts.min(axis=0)
        x1, y1 = src_pts.max(axis=0)
        pad = int(face_h * 0.35)
        rx0, ry0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
        rx1, ry1 = min(w, int(x1 + pad)), min(h, int(y1 + pad))
        if rx1 - rx0 < 16 or ry1 - ry0 < 16:
            return frame

        roi = frame[ry0:ry1, rx0:rx1].copy()
        rh, rw = roi.shape[:2]

        # Shift to ROI local coordinates
        src_local = src_pts - np.array([rx0, ry0], dtype=np.float32)
        dst_local = dst_pts - np.array([rx0, ry0], dtype=np.float32)

        # Piecewise affine or thin-plate spline local warp
        try:
            # Estimate similarity/affine transform on control points
            M, inliers = cv2.estimateAffinePartial2D(dst_local, src_local)
            if M is not None:
                warped_roi = cv2.warpAffine(roi, M, (rw, rh), flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_REFLECT101)
                # Soft blend mask around lower jaw
                mask = np.zeros((rh, rw), dtype=np.uint8)
                chin_local = chin_tip - np.array([rx0, ry0], dtype=np.float32)
                cv2.ellipse(mask, (int(chin_local[0]), int(chin_local[1])),
                            (int(face_h * 0.45), int(face_h * 0.35)),
                            0, 0, 180, 255, -1)
                mask = cv2.GaussianBlur(mask, (15, 15), 0)
                alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
                out_roi = (roi.astype(np.float32) * (1.0 - alpha) + warped_roi.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
                out = frame.copy()
                out[ry0:ry1, rx0:rx1] = out_roi
                return out
        except Exception as _degrade_error:
            _swallowed("roop/oral_cavity.py:384", _degrade_error,
                       "jaw warp fallback continued")

    return frame
