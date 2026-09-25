"""Neural environment re-lighting and normal harmonization for face swapping.

Implements intrinsic face decomposition, surface normal estimation, target
lighting transfer (diffuse shading + specular catchlights), and dynamic shadow
occlusion masking.

Pipeline Overview:
------------------
1. Intrinsic Face Decomposition:
   - Extract continuous 3D surface normals N(x, y) with ||N|| = 1.0 using an
     anthropometric 3D face geometry prior anchored by 3D/2D facial landmarks,
     refined with Shape-from-Shading (SfS) micro-geometry.
   - Decompose face into diffuse albedo A(x, y) (pigmentation without illumination)
     and diffuse shading S(x, y).
   - Recover specular environment map: key light direction L_key = (Lx, Ly, Lz),
     intensity, warmth/chrominance C_key, ambient fill level L_amb, and specular
     hardness.

2. Lighting Transfer & Specular Matching:
   - Recompute diffuse shading on the swapped identity using the target scene's
     light vector: Shading_new = L_amb + key_intensity * max(0, N_swap . L_key) * C_key.
   - Project target illumination onto swapped diffuse albedo.
   - Match specular highlights: Blinn-Phong T-zone specular shine (nose tip,
     forehead) and precise corneal eye catchlight restoration.

3. Shadow Occlusion Masking:
   - When foreground objects (microphones, hair strands, hats, hands) intercept
     the target scene's light path, project dynamic soft shadows across the
     swapped face along the inverted light path vector s = -(Lx/Lz, Ly/Lz) * depth,
     with distance-adaptive penumbra blur.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, Any

import roop.globals
from roop.degrade import swallowed as _swallowed


# ---------------------------------------------------------------------------
# Anthropometric 3D canonical face depth prior & normal field
# ---------------------------------------------------------------------------

def _build_face_depth_prior(
    h: int,
    w: int,
    landmarks: Optional[np.ndarray] = None,
    kps: Optional[np.ndarray] = None
) -> np.ndarray:
    """Build a continuous 3D depth surface Z(x, y) for an aligned face crop.

    The depth surface integrates:
    1. Base ellipsoidal skull geometry centered on the facial midpoint.
    2. Anatomical depth features: nose bridge & tip elevation, ocular orbit
       concavities, brow ridge prominence, zygomatic cheek curves, and chin protrusion.
    3. If 68-point 3D landmarks or 5 keypoints are supplied, adaptively adjusts
       feature centers, pitch tilt, and yaw roll.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Default canonical face anchors in standard aligned crop space
    cx = float(w) * 0.50
    cy = float(h) * 0.52
    rx = float(w) * 0.44
    ry = float(h) * 0.48
    rz = float(min(h, w)) * 0.40

    left_eye_center = np.array([w * 0.36, h * 0.42], dtype=np.float32)
    right_eye_center = np.array([w * 0.64, h * 0.42], dtype=np.float32)
    nose_tip = np.array([w * 0.50, h * 0.58], dtype=np.float32)
    nose_bridge = np.array([w * 0.50, h * 0.46], dtype=np.float32)
    mouth_center = np.array([w * 0.50, h * 0.74], dtype=np.float32)
    chin_center = np.array([w * 0.50, h * 0.88], dtype=np.float32)

    # Refine anchors from landmarks if available
    if landmarks is not None and len(landmarks) >= 68:
        pts = np.asarray(landmarks, dtype=np.float32)
        # 68-point landmark indices:
        # 36-41 left eye, 42-47 right eye, 27-30 nose ridge, 30 nose tip,
        # 48-67 mouth, 8 chin
        left_eye_center = pts[36:42].mean(axis=0)[:2]
        right_eye_center = pts[42:48].mean(axis=0)[:2]
        nose_bridge = pts[27][:2]
        nose_tip = pts[30][:2]
        mouth_center = pts[48:68].mean(axis=0)[:2]
        chin_center = pts[8][:2]
        cx = float((left_eye_center[0] + right_eye_center[0]) * 0.5)
        cy = float((nose_bridge[1] + mouth_center[1]) * 0.5)
    elif kps is not None and len(kps) >= 5:
        pts = np.asarray(kps, dtype=np.float32)
        left_eye_center = pts[0][:2]
        right_eye_center = pts[1][:2]
        nose_tip = pts[2][:2]
        mouth_center = (pts[3][:2] + pts[4][:2]) * 0.5
        chin_center = np.array([mouth_center[0], min(h - 1.0, mouth_center[1] + (mouth_center[1] - nose_tip[1]))], dtype=np.float32)
        cx = float((left_eye_center[0] + right_eye_center[0]) * 0.5)
        cy = float(nose_tip[1])

    # 1. Base ellipsoidal skull depth
    norm_x = (xx - cx) / rx
    norm_y = (yy - cy) / ry
    rad_sq = norm_x ** 2 + norm_y ** 2
    z_base = np.where(rad_sq < 1.0, rz * np.sqrt(np.maximum(0.0, 1.0 - rad_sq)), 0.0)

    # 2. Nose bridge and tip elevation
    # Distance to the line segment between nose_bridge and nose_tip
    v = nose_tip - nose_bridge
    v_len_sq = float(np.sum(v ** 2)) + 1e-6
    u = ((xx - nose_bridge[0]) * v[0] + (yy - nose_bridge[1]) * v[1]) / v_len_sq
    u_clamped = np.clip(u, 0.0, 1.0)
    proj_x = nose_bridge[0] + u_clamped * v[0]
    proj_y = nose_bridge[1] + u_clamped * v[1]
    dist_ridge_sq = (xx - proj_x) ** 2 + (yy - proj_y) ** 2
    sigma_ridge = max(4.0, float(w) * 0.045)
    z_ridge = (rz * 0.38) * np.exp(-dist_ridge_sq / (2.0 * sigma_ridge ** 2))

    # Prominent nose tip sphere
    dist_tip_sq = (xx - nose_tip[0]) ** 2 + (yy - nose_tip[1]) ** 2
    sigma_tip = max(5.0, float(w) * 0.055)
    z_tip = (rz * 0.48) * np.exp(-dist_tip_sq / (2.0 * sigma_tip ** 2))

    # 3. Eye socket concavities (orbital depressions)
    sigma_eye = max(6.0, float(w) * 0.075)
    dist_leye_sq = (xx - left_eye_center[0]) ** 2 + (yy - left_eye_center[1]) ** 2
    dist_reye_sq = (xx - right_eye_center[0]) ** 2 + (yy - right_eye_center[1]) ** 2
    z_eye_socket = -(rz * 0.28) * (
        np.exp(-dist_leye_sq / (2.0 * sigma_eye ** 2)) +
        np.exp(-dist_reye_sq / (2.0 * sigma_eye ** 2))
    )

    # 4. Brow ridge (supraorbital arch)
    brow_y = (left_eye_center[1] + right_eye_center[1]) * 0.5 - float(h) * 0.04
    brow_dist_sq = (yy - brow_y) ** 2
    brow_x_weight = np.exp(-((xx - cx) / (rx * 0.75)) ** 2)
    z_brow = (rz * 0.15) * np.exp(-brow_dist_sq / (2.0 * (float(h) * 0.04) ** 2)) * brow_x_weight

    # 5. Lips and chin elevation
    sigma_mouth = max(6.0, float(w) * 0.08)
    dist_mouth_sq = (xx - mouth_center[0]) ** 2 + (yy - mouth_center[1]) ** 2
    z_mouth = (rz * 0.16) * np.exp(-dist_mouth_sq / (2.0 * sigma_mouth ** 2))

    sigma_chin = max(7.0, float(w) * 0.09)
    dist_chin_sq = (xx - chin_center[0]) ** 2 + (yy - chin_center[1]) ** 2
    z_chin = (rz * 0.22) * np.exp(-dist_chin_sq / (2.0 * sigma_chin ** 2))

    z_total = z_base + z_ridge + z_tip + z_eye_socket + z_brow + z_mouth + z_chin
    return cv2.GaussianBlur(z_total, (0, 0), sigmaX=max(1.0, float(w) * 0.02))


def estimate_surface_normals(
    crop: np.ndarray,
    landmarks: Optional[np.ndarray] = None,
    kps: Optional[np.ndarray] = None,
    sfs_weight: float = 0.35
) -> np.ndarray:
    """Extract continuous 3D surface normals N(x, y) from an aligned face crop.

    Combines:
    - Macro-geometric depth gradient from anthropometric face shape.
    - Micro-geometric surface tilt from photometric Shape-from-Shading (SfS)
      on the luminance channel.

    Returns:
    --------
    normals : np.ndarray (H, W, 3), float32 in [-1, 1], with ||N(x, y)|| = 1.0.
              Coordinates: X points right, Y points down, Z points towards camera.
    """
    h, w = crop.shape[:2]
    z = _build_face_depth_prior(h, w, landmarks=landmarks, kps=kps)

    # Macro-depth gradients using central difference Sobel
    dz_dx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    dz_dy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3) / 8.0

    # Photometric micro-geometry (Shape-from-Shading refinement)
    # Convert crop to luminance and apply bilateral filter to decouple pigmentation
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth_lum = cv2.bilateralFilter(gray, d=7, sigmaColor=25.0, sigmaSpace=5.0)
    high_freq_lum = gray - smooth_lum

    # Edge-stopping gate to avoid steep creases creating normal spikes
    dl_dx = cv2.Sobel(high_freq_lum, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    dl_dy = cv2.Sobel(high_freq_lum, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    edge_mag = np.hypot(dl_dx, dl_dy)
    gate = 1.0 / (1.0 + (edge_mag / 12.0) ** 2)

    grad_x = dz_dx + sfs_weight * dl_dx * gate
    grad_y = dz_dy + sfs_weight * dl_dy * gate

    # Normal vector: n = (-dz/dx, -dz/dy, 1.0)
    nx = -grad_x
    ny = -grad_y
    nz = np.ones((h, w), dtype=np.float32) * max(1.0, float(w) * 0.15)

    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-8
    normals = np.stack([nx / norm, ny / norm, nz / norm], axis=-1).astype(np.float32)
    return normals


# ---------------------------------------------------------------------------
# Intrinsic image decomposition: Albedo, Shading & Light Vector Estimation
# ---------------------------------------------------------------------------

def _extract_skin_mask(crop: np.ndarray) -> np.ndarray:
    """Robust facial skin region mask in aligned crop space."""
    h, w = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)

    yy, xx = np.ogrid[:h, :w]
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.52
    rx, ry = max(1.0, w * 0.42), max(1.0, h * 0.44)
    prior = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0

    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    val = hsv[:, :, 2]
    evidence = (prior & (cr >= 120) & (cr <= 195) & (cb >= 70) & (cb <= 150) & (val >= 25))

    if int(evidence.sum()) < max(64, int(0.015 * h * w)):
        soft = prior.astype(np.float32)
    else:
        soft = cv2.GaussianBlur(evidence.astype(np.float32), (0, 0), sigmaX=max(1.0, min(h, w) / 48.0))
        soft = np.maximum(soft, prior.astype(np.float32) * 0.15)
    return np.clip(soft, 0.0, 1.0).astype(np.float32)


def decompose_intrinsic_components(
    crop: np.ndarray,
    normals: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Decompose an aligned face crop into intrinsic components:
    - Diffuse Albedo A(x, y)
    - Diffuse Shading S(x, y)
    - Specular Environment Lighting Parameters

    Returns:
    --------
    albedo : np.ndarray (H, W, 3) float32 in [0, 1]
    shading : np.ndarray (H, W, 1) float32 in [0, 1]
    light_params : dict containing:
        - 'key_dir': (3,) unit vector (Lx, Ly, Lz) pointing toward key light
        - 'key_intensity': float
        - 'key_color': (3,) float array normalized key light RGB tint
        - 'ambient_level': float
        - 'specular_hardness': float
        - 'specular_map': (H, W) float32 highlight residual
    """
    h, w = crop.shape[:2]
    crop_f = crop.astype(np.float32) / 255.0
    skin_mask = _extract_skin_mask(crop)

    # 1. Luminance channel (Rec. 709)
    lum = (0.0722 * crop_f[:, :, 0] + 0.7152 * crop_f[:, :, 1] + 0.2126 * crop_f[:, :, 2])
    lum_u8 = np.clip(lum * 255.0, 0, 255).astype(np.uint8)

    # 2. Retinex bilateral filter to isolate low-frequency shading
    shading = cv2.bilateralFilter(lum_u8, d=11, sigmaColor=35.0, sigmaSpace=max(5.0, float(w) * 0.08))
    shading_f = (shading.astype(np.float32) / 255.0)[:, :, None]
    shading_f = np.clip(shading_f, 0.04, 1.0)

    # 3. Diffuse albedo extraction
    albedo = np.clip(crop_f / shading_f, 0.0, 1.0)

    # 4. Fit directional key light vector L_key and ambient bias via linear least squares
    # Shading(x, y) ~ c0 * Nx + c1 * Ny + c2 * Nz + c3
    sample_mask = skin_mask > 0.4
    if int(sample_mask.sum()) < 64:
        sample_mask = np.ones((h, w), dtype=bool)

    N_sampled = normals[sample_mask]  # (K, 3)
    S_sampled = shading_f[:, :, 0][sample_mask]  # (K,)

    # Design matrix M = [Nx, Ny, Nz, 1.0]
    K = N_sampled.shape[0]
    M = np.hstack([N_sampled, np.ones((K, 1), dtype=np.float32)])
    # Ridge regression: (M^T M + lambda I)^(-1) M^T S
    lambda_reg = 0.01 * K
    reg_mat = lambda_reg * np.eye(4, dtype=np.float32)
    reg_mat[3, 3] = 0.0  # Do not regularize bias
    try:
        sol = np.linalg.solve(M.T @ M + reg_mat, M.T @ S_sampled)
        lx, ly, lz, amb = sol[0], sol[1], sol[2], sol[3]
    except Exception as _degrade_error:
        _swallowed("roop/light_harmonizer.py:270", _degrade_error, "least squares fallback")
        lx, ly, lz, amb = 0.2, -0.3, 0.9, 0.25

    # Light must illuminate from front of camera (Lz > 0)
    lz = max(0.18, float(lz))
    l_vec = np.array([lx, ly, lz], dtype=np.float32)
    l_len = float(np.linalg.norm(l_vec))
    if l_len < 1e-4:
        key_dir = np.array([0.0, -0.2, 0.98], dtype=np.float32)
        key_intensity = 0.8
    else:
        key_dir = l_vec / l_len
        key_intensity = min(2.0, max(0.2, l_len * 1.5))

    ambient_level = float(np.clip(amb, 0.08, 0.90))

    # 5. Measure key light chromaticity (warmth/coolness)
    # Compare color of brightly lit skin regions vs ambient/shadowed skin regions
    key_shading = np.maximum(0.0, np.sum(normals * key_dir[None, None, :], axis=-1))
    lit_sample = sample_mask & (key_shading > 0.65)
    if int(lit_sample.sum()) >= 32:
        lit_bgr = crop_f[lit_sample].mean(axis=0)
        # Normalize so max channel is 1.0 to preserve brightness scale
        key_color = lit_bgr / (float(np.max(lit_bgr)) + 1e-6)
    else:
        key_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # 6. Specular highlight detection
    # Diffuse model prediction:
    diffuse_pred = np.clip(ambient_level + key_intensity * key_shading[:, :, None] * key_color[None, None, :], 0.0, 1.0)
    specular_residual = np.maximum(0.0, lum - (diffuse_pred[:, :, 0] * 0.114 + diffuse_pred[:, :, 1] * 0.587 + diffuse_pred[:, :, 2] * 0.299))
    specular_map = np.clip(specular_residual * 2.5, 0.0, 1.0) * skin_mask

    light_params = {
        'key_dir': key_dir,
        'key_intensity': key_intensity,
        'key_color': key_color,
        'ambient_level': ambient_level,
        'specular_hardness': 24.0,
        'specular_map': specular_map,
    }
    return albedo, shading_f, light_params


# ---------------------------------------------------------------------------
# Eye corneal catchlight isolation & restoration
# ---------------------------------------------------------------------------

def restore_eye_catchlights(
    swapped_crop: np.ndarray,
    target_crop: np.ndarray,
    landmarks: Optional[np.ndarray] = None,
    kps: Optional[np.ndarray] = None,
    strength: float = 0.8
) -> np.ndarray:
    """Isolate corneal catchlights from target eyes and restore them on the swap.

    High-luminance pinpoint reflections in the target cornea preserve eye vitality,
    specular curvature, and lifelike gaze direction that neural swappers blur.
    """
    if strength <= 1e-4:
        return swapped_crop

    h, w = swapped_crop.shape[:2]
    out = swapped_crop.copy()

    # Determine eye bounding boxes
    eye_boxes = []
    if landmarks is not None and len(landmarks) >= 68:
        pts = np.asarray(landmarks, dtype=np.float32)
        leye = pts[36:42]
        reye = pts[42:48]
        for e in (leye, reye):
            pad = float(w) * 0.02
            x0 = int(max(0, np.min(e[:, 0]) - pad))
            x1 = int(min(w - 1, np.max(e[:, 0]) + pad))
            y0 = int(max(0, np.min(e[:, 1]) - pad))
            y1 = int(min(h - 1, np.max(e[:, 1]) + pad))
            if x1 > x0 + 4 and y1 > y0 + 4:
                eye_boxes.append((x0, y0, x1, y1))
    elif kps is not None and len(kps) >= 5:
        pts = np.asarray(kps, dtype=np.float32)
        radius = int(max(6, float(w) * 0.075))
        for kp in (pts[0], pts[1]):
            x0 = max(0, int(kp[0] - radius))
            x1 = min(w - 1, int(kp[0] + radius))
            y0 = max(0, int(kp[1] - radius))
            y1 = min(h - 1, int(kp[1] + radius))
            if x1 > x0 + 4 and y1 > y0 + 4:
                eye_boxes.append((x0, y0, x1, y1))
    else:
        # Canonical eye boxes
        ew = int(w * 0.16)
        eh = int(h * 0.12)
        eye_boxes.append((int(w * 0.28), int(h * 0.36), int(w * 0.28) + ew, int(h * 0.36) + eh))
        eye_boxes.append((int(w * 0.56), int(h * 0.36), int(w * 0.56) + ew, int(h * 0.36) + eh))

    for x0, y0, x1, y1 in eye_boxes:
        tgt_roi = target_crop[y0:y1, x0:x1]
        swp_roi = out[y0:y1, x0:x1]

        # Isolate corneal highlight glints in target ROI
        tgt_gray = cv2.cvtColor(tgt_roi, cv2.COLOR_BGR2GRAY)
        # Glint: high-pass peak above local median
        ksize = max(3, (min(y1 - y0, x1 - x0) // 4) * 2 + 1)
        base = cv2.medianBlur(tgt_gray, ksize)
        glint_raw = np.maximum(0, tgt_gray.astype(np.float32) - base.astype(np.float32))

        # Threshold to high-intensity glints and bright pupils
        glint_mask = (tgt_gray > 165) & (glint_raw > 18.0)
        if not np.any(glint_mask):
            continue

        glint_float = cv2.GaussianBlur(glint_mask.astype(np.float32), (3, 3), 0.8)
        glint_float = np.clip(glint_float * 1.5, 0.0, 1.0)[:, :, None]

        # Transfer glint color and peak intensity onto swapped eye
        glint_pixels = tgt_roi.astype(np.float32)
        swp_float = swp_roi.astype(np.float32)
        blended_roi = np.maximum(swp_float, swp_float * (1.0 - strength * glint_float) + glint_pixels * (strength * glint_float))
        out[y0:y1, x0:x1] = np.clip(blended_roi, 0, 255).astype(np.uint8)

    return out


# ---------------------------------------------------------------------------
# Dynamic shadow occlusion masking
# ---------------------------------------------------------------------------

def cast_shadow_occlusion(
    crop: np.ndarray,
    occluder_mask: np.ndarray,
    light_dir: np.ndarray,
    shadow_strength: float = 0.6,
    skin_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Project dynamic soft shadows across the swapped face if foreign objects
    (microphones, hair strands, hats, hands) intercept the key light path.

    Parameters:
    -----------
    crop : np.ndarray (H, W, 3) BGR uint8
    occluder_mask : np.ndarray (H, W) or (H, W, 1), float in [0, 1] where 1.0 = occluder
    light_dir : np.ndarray (3,) unit vector (Lx, Ly, Lz) pointing toward key light
    shadow_strength : float in [0, 1]
    """
    if shadow_strength <= 1e-4 or occluder_mask is None:
        return crop

    occ = np.asarray(occluder_mask, dtype=np.float32)
    if occ.ndim == 3:
        occ = occ[:, :, 0]

    # If no significant occlusion detected, skip
    if float(occ.mean()) < 0.005:
        return crop

    h, w = crop.shape[:2]
    if occ.shape[:2] != (h, w):
        occ = cv2.resize(occ, (w, h), interpolation=cv2.INTER_LINEAR)

    lx, ly, lz = float(light_dir[0]), float(light_dir[1]), max(0.15, float(light_dir[2]))

    # Projected shadow displacement along inverted light vector:
    # Shadow casts away from light source
    standoff_pixels = float(min(h, w)) * 0.065
    dx = -float(lx / lz) * standoff_pixels
    dy = -float(ly / lz) * standoff_pixels

    # Clamp maximum shadow travel to prevent runaway projections
    max_travel = float(min(h, w)) * 0.18
    dx = float(np.clip(dx, -max_travel, max_travel))
    dy = float(np.clip(dy, -max_travel, max_travel))

    # Shift occluder mask along (dx, dy)
    M_shift = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    cast_raw = cv2.warpAffine(occ, M_shift, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    # Physical penumbra softening (Gaussian blur proportional to displacement)
    travel_dist = np.hypot(dx, dy)
    penumbra_sigma = max(1.5, travel_dist * 0.35)
    cast_soft = cv2.GaussianBlur(cast_raw, (0, 0), sigmaX=penumbra_sigma)

    # Face skin restriction: shadow casts on the face skin, not into thin air or under occluder itself
    if skin_mask is None:
        skin_mask = _extract_skin_mask(crop)
    elif skin_mask.shape[:2] != (h, w):
        skin_mask = cv2.resize(skin_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    # Shadow exists on skin where the occluder itself is NOT currently covering
    shadow_effective = np.clip(cast_soft, 0.0, 1.0) * skin_mask * np.clip(1.0 - occ, 0.0, 1.0)

    # Attenuate illumination in shadowed regions
    attenuation = 1.0 - (shadow_strength * 0.65) * shadow_effective[:, :, None]
    shadowed_crop = crop.astype(np.float32) * attenuation
    return np.clip(shadowed_crop, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# High-level environment re-lighting orchestrator
# ---------------------------------------------------------------------------

def apply_light_harmonization(
    swapped_crop: np.ndarray,
    target_crop: np.ndarray,
    target_face: Optional[Any] = None,
    occluder_mask: Optional[np.ndarray] = None,
    key_intensity: Optional[float] = None,
    ambient_bias: Optional[float] = None,
    eye_specular: Optional[float] = None,
    shadow_occlusion: Optional[float] = None
) -> np.ndarray:
    """Environment re-lighting and normal harmonization controller.

    1. Decomposes the aligned target frame into intrinsic components (normals,
       albedo, specular environment map).
    2. Re-shades the swapped source identity using the target scene's light vector.
    3. Matches specular highlights (forehead, nose tip, and cornea catchlights).
    4. Casts dynamic soft shadows from foreground occluders.
    """
    # Bit-identical no-op guard if disabled
    enabled = bool(getattr(roop.globals, 'light_harmonizer', False))
    if not enabled and key_intensity is None:
        return swapped_crop

    try:
        # Read parameters from globals with graceful defaults
        k_int = float(getattr(roop.globals, 'light_harmonizer_key_intensity', 1.0) if key_intensity is None else key_intensity)
        amb_bias = float(getattr(roop.globals, 'light_harmonizer_ambient_bias', 0.0) if ambient_bias is None else ambient_bias)
        eye_spec = float(getattr(roop.globals, 'light_harmonizer_eye_specular', 0.8) if eye_specular is None else eye_specular)
        shd_occ = float(getattr(roop.globals, 'light_harmonizer_shadow_occlusion', 0.6) if shadow_occlusion is None else shadow_occlusion)

        h, w = swapped_crop.shape[:2]
        tgt = target_crop
        if tgt.shape[:2] != (h, w):
            tgt = cv2.resize(tgt, (w, h), interpolation=cv2.INTER_AREA)

        # Extract landmarks if available
        lm68 = None
        kps = None
        if target_face is not None:
            if hasattr(target_face, 'landmark_3d_68'):
                lm68 = getattr(target_face, 'landmark_3d_68', None)
            elif isinstance(target_face, dict):
                lm68 = target_face.get('landmark_3d_68')
            if hasattr(target_face, 'kps'):
                kps = getattr(target_face, 'kps', None)
            elif isinstance(target_face, dict):
                kps = target_face.get('kps')

        # 1. Target intrinsic decomposition
        tgt_normals = estimate_surface_normals(tgt, landmarks=lm68, kps=kps)
        _, _, tgt_light = decompose_intrinsic_components(tgt, tgt_normals)

        # 2. Swapped face intrinsic decomposition
        swp_normals = estimate_surface_normals(swapped_crop, landmarks=lm68, kps=kps)
        swp_albedo, _, _ = decompose_intrinsic_components(swapped_crop, swp_normals)

        # 3. Target lighting projection
        key_dir = tgt_light['key_dir']
        key_color = tgt_light['key_color']
        amb_level = np.clip(tgt_light['ambient_level'] + amb_bias, 0.05, 0.95)

        # Recomputed diffuse shading on swapped face normals
        n_dot_l = np.maximum(0.0, np.sum(swp_normals * key_dir[None, None, :], axis=-1))
        diffuse_shading = amb_level + (k_int * tgt_light['key_intensity']) * n_dot_l[:, :, None] * key_color[None, None, :]
        diffuse_shading = np.clip(diffuse_shading, 0.05, 1.25)

        # Project diffuse lighting onto swapped albedo
        relit_diffuse = np.clip(swp_albedo * diffuse_shading * 255.0, 0, 255).astype(np.uint8)

        # 4. T-Zone Specular highlight shine (forehead, nose tip)
        view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        half_vec = key_dir + view_dir
        half_norm = float(np.linalg.norm(half_vec))
        if half_norm > 1e-4:
            half_vec /= half_norm
            n_dot_h = np.maximum(0.0, np.sum(swp_normals * half_vec[None, None, :], axis=-1))
            specular_lobe = (n_dot_h ** 20.0) * (k_int * 0.35)

            # T-zone mask (forehead center + nose ridge)
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            tzone = np.exp(-((xx - w * 0.50) / (w * 0.14)) ** 2) * np.exp(-((yy - h * 0.50) / (h * 0.28)) ** 2)
            specular_term = np.clip(specular_lobe * tzone * 255.0, 0, 255)[:, :, None]
            relit_diffuse = np.clip(relit_diffuse.astype(np.float32) + specular_term * key_color[None, None, :], 0, 255).astype(np.uint8)

        # 5. Eye corneal catchlight restoration
        harmonized = restore_eye_catchlights(
            relit_diffuse, tgt, landmarks=lm68, kps=kps, strength=eye_spec
        )

        # 6. Dynamic shadow occlusion casting
        if occluder_mask is not None and shd_occ > 1e-4:
            harmonized = cast_shadow_occlusion(
                harmonized, occluder_mask, key_dir, shadow_strength=shd_occ
            )

        return harmonized

    except Exception as e:
        _swallowed("roop/light_harmonizer.py:apply_light_harmonization", e, "light harmonizer failed, keeping crop")
        return swapped_crop
