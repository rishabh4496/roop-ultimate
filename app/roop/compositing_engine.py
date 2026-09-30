"""Stage 8 — Compositing Quality Engine.

Comprehensive, deterministic compositing and paste-back engine providing:
1. Linear-Light Alpha Blending:
   - Eliminates the 15-32 level non-linear sRGB gamma dip (dark boundary seam/ring).
   - Physically correct photon flux superposition in Linear RGB.
   - Invertible highlight soft-knee roll-off preventing blown-out specular highlights.

2. Skin-Tone & White Balance Matching in Perceptually Uniform Space (OKLab):
   - Computes illumination and chromatic statistics strictly on skin-masked regions
     (excluding hair, background, and dark clothing bias).
   - Aligns perceived lightness (L) and opponent chrominance (a: green-red, b: blue-yellow).
   - Bounded white-balance shift preventing neon skin tone blowout.

3. Dark Scene & Low-Light Preservation:
   - Exposure-aware tone mapping for NORMAL, DARK, and VERY_DARK scenes.
   - Shadow floor anchoring preventing floating milky shadows or crushed black cutouts.
   - Chrominance gain compression preventing sensor noise amplification in low light.

4. Multi-Band Seam Blending (Hairline & Jaw/Neck Transitions):
   - Frequency-split multi-scale Laplacian decomposition.
   - Low-frequency band: spatial illumination and ambient color bridge across wide feather zone.
   - High-frequency band: sharp facial detail (pores, jaw contour) preserved with tight alpha gating.

5. Edge-Preserving Micro-Sharpening:
   - Compensates for bilinear warp interpolation softening via cored unsharp masking.
   - Edge-stop gating prevents boundary haloing.
"""

from __future__ import annotations

import math
import os
import threading
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

from roop.degrade import swallowed as _swallowed

# ==============================================================================
# 1. Color Space Science: Linear sRGB and OKLab
# ==============================================================================

# Fast precomputed LUT: 8-bit sRGB [0, 255] -> Linear float32 [0.0, 1.0]
_LUT_SRGB_TO_LINEAR = np.array([
    (i / 255.0) / 12.92 if (i / 255.0) <= 0.04045
    else (((i / 255.0) + 0.055) / 1.055) ** 2.4
    for i in range(256)
], dtype=np.float32)

# OKLab transformation matrices (Björn Ottosson, 2020)
# Linear sRGB -> LMS
_M1_RGB_TO_LMS = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005]
], dtype=np.float32)

# LMS' -> OKLab
_M2_LMS_TO_OKLAB = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660]
], dtype=np.float32)

# Inverse matrices: OKLab -> LMS' and LMS -> Linear sRGB
_M2_INV_OKLAB_TO_LMS = np.linalg.inv(_M2_LMS_TO_OKLAB).astype(np.float32)
_M1_INV_LMS_TO_RGB = np.linalg.inv(_M1_RGB_TO_LMS).astype(np.float32)


class LinearColorSpace:
    """Accurate, SIMD-accelerated conversions between sRGB and Linear RGB."""

    @staticmethod
    def srgb_to_linear(img_bgr: np.ndarray) -> np.ndarray:
        """Convert uint8 [0, 255] or float32 sRGB BGR to Linear float32 BGR [0.0, 1.0]."""
        if img_bgr.dtype == np.uint8:
            return np.take(_LUT_SRGB_TO_LINEAR, img_bgr)
        # float32 input: apply exact IEC 61966-2-1 piecewise EOTF
        x = np.clip(img_bgr, 0.0, 1.0)
        return np.where(x <= 0.04045, x * (1.0 / 12.92), np.power((x + 0.055) * (1.0 / 1.055), 2.4))

    @staticmethod
    def linear_to_srgb(img_lin: np.ndarray, soft_knee: bool = True) -> np.ndarray:
        """Convert Linear float32 BGR back to uint8 [0, 255] sRGB.
        
        soft_knee: Compresses super-white HDR specularities (>1.0) smoothly
        to avoid harsh clipping artifacts.
        """
        x = np.maximum(img_lin, 0.0)
        if soft_knee:
            # Soft-knee highlight rolloff for values above 1.0
            x = np.where(x > 1.0, 1.0 + np.tanh(x - 1.0) * 0.45, x)

        # Exact sRGB inverse EOTF
        srgb = np.where(
            x <= 0.0031308,
            12.92 * x,
            1.055 * np.power(np.maximum(x, 1e-12), 1.0 / 2.4) - 0.055
        )
        return np.clip(np.round(srgb * 255.0), 0.0, 255.0).astype(np.uint8)


class OKLabColorSpace:
    """Perceptually uniform color space separating lightness from chromaticity."""

    @staticmethod
    def linear_bgr_to_oklab(bgr_lin: np.ndarray) -> np.ndarray:
        """Linear BGR float32 -> OKLab float32 (L in [0, 1], a, b in [-0.4, 0.4])."""
        # BGR -> RGB
        rgb = bgr_lin[..., ::-1]
        lms = rgb @ _M1_RGB_TO_LMS.T
        lms_p = np.cbrt(np.maximum(lms, 1e-12))
        return lms_p @ _M2_LMS_TO_OKLAB.T

    @staticmethod
    def oklab_to_linear_bgr(oklab: np.ndarray) -> np.ndarray:
        """OKLab float32 -> Linear BGR float32."""
        lms_p = oklab @ _M2_INV_OKLAB_TO_LMS.T
        lms = lms_p ** 3
        rgb = lms @ _M1_INV_LMS_TO_RGB.T
        return rgb[..., ::-1]


# ==============================================================================
# 2. Skin-Tone, Exposure & White-Balance Matching
# ==============================================================================

class SkinPhotometricMatcher:
    """Matches exposure, white balance, and skin tone strictly on skin regions."""

    @staticmethod
    def extract_skin_mask(bgr_u8: np.ndarray) -> np.ndarray:
        """Extract soft dermal reflectance mask [0.0, 1.0] excluding hair and background."""
        h, w = bgr_u8.shape[:2]
        ycrcb = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)

        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        val = hsv[:, :, 2]

        # Biological melanin/hemoglobin chromatic cluster
        skin_chroma = (cr >= 122) & (cr <= 185) & (cb >= 75) & (cb <= 145) & (val >= 18)

        # Central face prior ellipse (smooth falloff to boundary)
        cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
        rx, ry = max(1.0, w * 0.45), max(1.0, h * 0.45)
        yy, xx = np.ogrid[:h, :w]
        prior = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0

        evidence = (skin_chroma & prior).astype(np.float32)
        if evidence.sum() < max(64.0, 0.05 * float(prior.sum())):
            evidence = prior.astype(np.float32)

        k = max(3, int(min(h, w) * 0.04) | 1)
        return cv2.GaussianBlur(evidence, (k, k), 0)

    @classmethod
    def match_photometrics(
        cls,
        paste_bgr: np.ndarray,
        target_bgr: np.ndarray,
        strength: float = 0.85,
        exposure_weight: float = 0.90,
        chroma_weight: float = 0.65,
        dark_tier: str = "NORMAL"
    ) -> np.ndarray:
        """Match swapped face skin tone, exposure, and white balance to target plate.
        
        Operates in OKLab color space strictly on skin pixels to avoid hair/clothing bias.
        """
        if strength <= 1e-4:
            return paste_bgr

        # Resize target to paste size if needed for sampling
        h, w = paste_bgr.shape[:2]
        if target_bgr.shape[:2] != (h, w):
            ref_target = cv2.resize(target_bgr, (w, h), interpolation=cv2.INTER_AREA)
        else:
            ref_target = target_bgr

        # Extract skin regions
        skin_paste = cls.extract_skin_mask(paste_bgr)
        skin_target = cls.extract_skin_mask(ref_target)
        joint_skin = skin_paste * skin_target
        sample_mask = joint_skin > 0.25

        if int(sample_mask.sum()) < 48:
            sample_mask = skin_target > 0.20
            if int(sample_mask.sum()) < 48:
                return paste_bgr

        # Convert both to Linear Light -> OKLab
        paste_lin = LinearColorSpace.srgb_to_linear(paste_bgr)
        target_lin = LinearColorSpace.srgb_to_linear(ref_target)

        paste_oklab = OKLabColorSpace.linear_bgr_to_oklab(paste_lin)
        target_oklab = OKLabColorSpace.linear_bgr_to_oklab(target_lin)

        # 1. Lightness (L) Exposure & Contrast Alignment
        L_p = paste_oklab[:, :, 0]
        L_t = target_oklab[:, :, 0]

        med_L_p = float(np.median(L_p[sample_mask]))
        med_L_t = float(np.median(L_t[sample_mask]))

        # Target exposure delta
        delta_L = (med_L_t - med_L_p) * exposure_weight * strength

        # Bounded contrast scaling on skin
        std_L_p = float(np.std(L_p[sample_mask]))
        std_L_t = float(np.std(L_t[sample_mask]))
        contrast_ratio = float(np.clip(std_L_t / max(std_L_p, 0.02), 0.75, 1.25))

        # Modulate L channel
        adjusted_L = (L_p - med_L_p) * contrast_ratio + med_L_p + delta_L
        # Smoothly blend adjusted L onto skin
        paste_oklab[:, :, 0] = L_p + (adjusted_L - L_p) * skin_paste

        # 2. White Balance & Chrominance (a, b) Alignment
        # In dark scenes, damp chroma correction to prevent color noise explosion
        chroma_damp = 0.50 if dark_tier in ("DARK", "VERY_DARK") else 1.0
        eff_chroma_weight = chroma_weight * strength * chroma_damp

        a_p = paste_oklab[:, :, 1]
        b_p = paste_oklab[:, :, 2]
        a_t = target_oklab[:, :, 1]
        b_t = target_oklab[:, :, 2]

        med_a_p = float(np.median(a_p[sample_mask]))
        med_a_t = float(np.median(a_t[sample_mask]))
        med_b_p = float(np.median(b_p[sample_mask]))
        med_b_t = float(np.median(b_t[sample_mask]))

        # Bounded chromatic shifts (prevents neon orange/red or severe blue casts)
        delta_a = float(np.clip(med_a_t - med_a_p, -0.06, 0.06)) * eff_chroma_weight
        delta_b = float(np.clip(med_b_t - med_b_p, -0.06, 0.06)) * eff_chroma_weight

        paste_oklab[:, :, 1] = a_p + delta_a * skin_paste
        paste_oklab[:, :, 2] = b_p + delta_b * skin_paste

        # Convert back to Linear BGR -> sRGB
        matched_lin = OKLabColorSpace.oklab_to_linear_bgr(paste_oklab)
        return LinearColorSpace.linear_to_srgb(matched_lin, soft_knee=True)


# ==============================================================================
# 3. Dark Scene & Low-Light Tone Mapping
# ==============================================================================

class DarkSceneToneMapper:
    """Ensures shadow continuity, black level anchoring, and noise suppression."""

    @staticmethod
    def classify_scene_luminance(target_bgr: np.ndarray) -> str:
        """Classify scene luminance tier: NORMAL, DARK, or VERY_DARK."""
        gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
        med_val = float(np.median(gray))
        if med_val < 32.0:
            return "VERY_DARK"
        elif med_val < 72.0:
            return "DARK"
        return "NORMAL"

    @staticmethod
    def anchor_shadows(
        paste_lin: np.ndarray,
        target_lin: np.ndarray,
        dark_tier: str
    ) -> np.ndarray:
        """Anchor black point and dark shadow floor to target plate ambient level."""
        if dark_tier == "NORMAL":
            return paste_lin

        # Measure 1st percentile of target luminance (true black floor)
        t_luma = 0.114 * target_lin[:, :, 0] + 0.587 * target_lin[:, :, 1] + 0.299 * target_lin[:, :, 2]
        black_floor = float(np.percentile(t_luma, 1.0))

        # Clamp paste black floor so it never drops below ambient darkness
        # or floats as a milky washed-out grey patch
        p_luma = 0.114 * paste_lin[:, :, 0] + 0.587 * paste_lin[:, :, 1] + 0.299 * paste_lin[:, :, 2]
        lift = np.maximum(0.0, black_floor - p_luma)
        if dark_tier == "VERY_DARK":
            # Strongly anchor low-end shadows in night / very dark footage
            paste_lin = paste_lin + lift[:, :, None] * 0.75
        else:
            paste_lin = paste_lin + lift[:, :, None] * 0.45

        return paste_lin


# ==============================================================================
# 4. Multi-Band Seam Blender (Hairline & Jaw/Neck Transitions)
# ==============================================================================

class MultiBandSeamBlender:
    """Laplacian multi-band frequency decomposition for seamless boundary transitions.
    
    Low frequencies (illumination, skin tone) blend across an expanded feather band
    to eliminate jawline/neck and hairline steps.
    High frequencies (pores, sharp contours) blend with tight alpha gating.
    """

    @staticmethod
    def blend(
        paste_lin: np.ndarray,
        target_lin: np.ndarray,
        alpha: np.ndarray,
        feather_px: float = 6.0
    ) -> np.ndarray:
        """Blend paste and target in Linear Light using two-band frequency split."""
        h, w = paste_lin.shape[:2]
        a = np.clip(alpha, 0.0, 1.0)
        if a.ndim == 2:
            a = a[:, :, None]

        sigma = max(1.5, float(feather_px))

        # 1. Low-frequency band (spatial illumination & skin tone)
        low_p = cv2.GaussianBlur(paste_lin, (0, 0), sigmaX=sigma, sigmaY=sigma)
        low_t = cv2.GaussianBlur(target_lin, (0, 0), sigmaX=sigma, sigmaY=sigma)

        # 2. High-frequency band (micro-pores, hair wisps, sharp edges)
        high_p = paste_lin - low_p
        high_t = target_lin - low_t

        # 3. Expanded low-frequency alpha: bridges neck/jaw and hairline illumination steps
        # Dilate alpha slightly and blur to create a wider, seamless ambient transition
        k_sz = max(3, int(sigma * 2.0) | 1)
        k_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
        expanded_a = cv2.dilate(a[:, :, 0], k_el, iterations=1)
        expanded_a = cv2.GaussianBlur(expanded_a, (0, 0), sigmaX=sigma * 1.5, sigmaY=sigma * 1.5)[:, :, None]

        # Low band blended with wide ambient transition
        blended_low = expanded_a * low_p + (1.0 - expanded_a) * low_t

        # High band blended with tight interior mask (prevents double edges/ghosting)
        blended_high = a * high_p + (1.0 - a) * high_t

        return blended_low + blended_high


# ==============================================================================
# 5. Edge-Preserving Micro-Sharpening & Detail Recovery
# ==============================================================================

class EdgePreservingSharpener:
    """Restores micro-contrast lost during bilinear affine warp interpolation."""

    @staticmethod
    def sharpen(
        img_lin: np.ndarray,
        alpha: np.ndarray,
        strength: float = 0.20
    ) -> np.ndarray:
        """Subtle cored unsharp masking applied strictly in face interior."""
        if strength <= 1e-4:
            return img_lin

        # Blur to isolate high-pass micro-contrast
        blurred = cv2.GaussianBlur(img_lin, (0, 0), sigmaX=1.0, sigmaY=1.0)
        high_pass = img_lin - blurred

        # Soft-knee coring: suppress sensor noise (|D| < 0.004) and huge specular steps (|D| > 0.15)
        mag = np.abs(high_pass)
        coring = np.clip((mag - 0.004) / 0.03, 0.0, 1.0) * np.exp(-((mag / 0.15) ** 2))

        # Confined strictly to deep face interior (alpha > 0.7) to prevent perimeter halos
        a_interior = np.clip((alpha - 0.6) / 0.35, 0.0, 1.0)
        if a_interior.ndim == 2:
            a_interior = a_interior[:, :, None]

        sharpened = img_lin + high_pass * coring * (strength * a_interior)
        return np.maximum(sharpened, 0.0)


# ==============================================================================
# 6. Unified Compositing Quality Engine
# ==============================================================================

class CompositingQualityEngine:
    """Thread-safe, deterministic master compositor."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def composite_roi(
        self,
        roi_paste: np.ndarray,
        roi_target: np.ndarray,
        roi_matte: np.ndarray,
        enable_photometric: bool = True,
        enable_linear_blend: bool = True,
        enable_multiband: bool = True,
        enable_sharpening: bool = True,
        photometric_strength: float = 0.85,
        sharpen_strength: float = 0.22
    ) -> np.ndarray:
        """Composite swapped face ROI into target frame ROI with photographic continuity.
        
        Args:
            roi_paste: BGR uint8 or float32 swapped face crop in ROI space.
            roi_target: BGR uint8 or float32 untouched target frame in ROI space.
            roi_matte: float32 alpha matte [0.0, 1.0] (H, W) or (H, W, 1).
            
        Returns:
            uint8 BGR composite.
        """
        # Ensure uint8 inputs for initial color analysis
        paste_u8 = np.clip(roi_paste, 0.0, 255.0).astype(np.uint8)
        target_u8 = np.clip(roi_target, 0.0, 255.0).astype(np.uint8)

        # 1. Scene Luminance Classification (Dark scenes vs Normal)
        dark_tier = DarkSceneToneMapper.classify_scene_luminance(target_u8)

        # 2. Skin-Tone, Exposure & White Balance Matching in OKLab
        if enable_photometric:
            try:
                paste_matched = SkinPhotometricMatcher.match_photometrics(
                    paste_u8, target_u8,
                    strength=photometric_strength,
                    dark_tier=dark_tier
                )
            except Exception as _e_photo:
                _swallowed("roop/compositing_engine.py:photometric", _e_photo, "photometric fallback")
                paste_matched = paste_u8
        else:
            paste_matched = paste_u8

        # 3. Linear-Light Conversion
        if enable_linear_blend:
            paste_lin = LinearColorSpace.srgb_to_linear(paste_matched)
            target_lin = LinearColorSpace.srgb_to_linear(target_u8)

            # 4. Shadow Floor Anchoring for Dark Scenes
            paste_lin = DarkSceneToneMapper.anchor_shadows(paste_lin, target_lin, dark_tier)

            # 5. Seam Blending (Multi-Band Laplacian Frequency Split)
            a = roi_matte if roi_matte.ndim == 2 else roi_matte[:, :, 0]
            if enable_multiband:
                try:
                    blended_lin = MultiBandSeamBlender.blend(paste_lin, target_lin, a, feather_px=6.0)
                except Exception as _e_mb:
                    _swallowed("roop/compositing_engine.py:multiband", _e_mb, "linear alpha fallback")
                    a3 = a[:, :, None]
                    blended_lin = a3 * paste_lin + (1.0 - a3) * target_lin
            else:
                a3 = a[:, :, None]
                blended_lin = a3 * paste_lin + (1.0 - a3) * target_lin

            # 6. Edge-Preserving Micro-Sharpening
            if enable_sharpening:
                try:
                    blended_lin = EdgePreservingSharpener.sharpen(blended_lin, a, strength=sharpen_strength)
                except Exception as _e_shp:
                    _swallowed("roop/compositing_engine.py:sharpen", _e_shp, "sharpen fallback")

            # 7. Inverse Gamma sRGB Conversion with Soft-Knee Highlights
            return LinearColorSpace.linear_to_srgb(blended_lin, soft_knee=True)

        else:
            # Fallback legacy linear sRGB blend
            a = roi_matte if roi_matte.ndim == 3 else roi_matte[:, :, None]
            out = a * paste_matched.astype(np.float32) + (1.0 - a) * target_u8.astype(np.float32)
            return np.clip(out, 0.0, 255.0).astype(np.uint8)


# Global module singleton instance
COMPOSITING_ENGINE = CompositingQualityEngine()
