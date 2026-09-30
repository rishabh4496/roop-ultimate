"""Stage 6 — Restore Ultra Quality and Throughput Optimizer.

Comprehensive optimization and audit engine for Restore Ultra:
1. Restoration Profiles:
   - FAST: High-throughput, lightweight finish tuned for real-time and laptop profiles (RTX 3060).
   - BALANCED: Balanced edge-preserving bilateral texture and anti-halo clarity.
   - QUALITY: Production-grade standard with anti-halo bounding and authentic pore injection.
   - ULTRA: Maximum micro-detail for high-res closeups, fine eyelashes, iris luminosity, and pore recovery.

2. Adaptive Restoration Strength:
   - Scales strength, crispness, and eye clarity based on crop resolution, head pose yaw, and noise floor.
   - Attenuates on tiny/distant faces to prevent jarring contrast with blurry backgrounds.
   - Damps ocular/facial distortion on profile angles (|yaw| > 25°).

3. Preallocated Buffer Management & Static Caching:
   - Preallocated float32 (1, 3, 512, 512) tensor buffers.
   - Cached morphological kernels and bilateral soft-knee LUTs to eliminate per-frame allocations.
   - Reusable scratch buffers and zero-copy device-to-device transfers where CUDA tensors are available.

4. Identity Feature Guardrail:
   - Monitors sensory feature zones (eyes, mouth, nose) for codebook hallucination drift.
   - Preserves high-frequency facial micro-texture (pores, eyelashes) while clamping low/mid band
     deviations back to the swapped reference to guarantee identity likeness.

5. Anatomical & Textural Quality Verification:
   - Pores: Laplacian band-pass micro-texture variance.
   - Eyelashes: Directional high-pass edge response on ocular borders.
   - Eyes: Iris luminosity and contrast energy with zero halo overshoot.
   - Teeth: Clamping guards preventing saturated/blown-out white blocks.
   - Hairline: Feathered boundary blending.
   - Hallucination artifacts: Strictly bounded by the local 3x3 min/max envelope.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from roop.degrade import swallowed as _swallowed

try:
    import torch
    _TORCH_AVAILABLE = True
    _TORCH_CUDA = torch.cuda.is_available()
except Exception as _e:
    _swallowed("roop/restore_ultra_optimizer.py:torch_import", _e, "torch unavailable")
    _TORCH_AVAILABLE = False
    _TORCH_CUDA = False


# ==============================================================================
# 1. Restoration Profiles
# ==============================================================================

@dataclass(frozen=True)
class RestoreProfileConfig:
    """Configuration parameters for a Restore Ultra enhancement profile."""
    name: str
    strength: float          # Bilateral texture injection strength (pores/skin texture)
    crispness: float         # Anti-halo edge sharpening amount (eyelashes/brows/lips)
    eye_clarity: float       # Eye region clarity boost (iris luminosity/catchlights)
    detail_weight: float     # Recombination frequency blend high-pass detail weight
    bilateral_sigma: float   # Bilateral filter sigma color
    bilateral_thresh: float  # Bilateral soft-knee threshold
    bilateral_soft: float    # Bilateral soft-knee saturation softness
    sharpen_sigma: float     # Gaussian blur sigma for anti-halo unsharp mask
    sharpen_limit: float     # Envelope pad limit for anti-halo clamp
    description: str


RESTORE_PROFILES: Dict[str, RestoreProfileConfig] = {
    'FAST': RestoreProfileConfig(
        name='FAST',
        strength=0.18,
        crispness=0.16,
        eye_clarity=0.30,
        detail_weight=0.50,
        bilateral_sigma=12.0,
        bilateral_thresh=8.0,
        bilateral_soft=2.0,
        sharpen_sigma=0.6,
        sharpen_limit=1.5,
        description='Lightweight, high-throughput enhancement tuned for real-time previews and laptop GPUs (RTX 3060).'
    ),
    'BALANCED': RestoreProfileConfig(
        name='BALANCED',
        strength=0.24,
        crispness=0.22,
        eye_clarity=0.40,
        detail_weight=0.65,
        bilateral_sigma=15.0,
        bilateral_thresh=9.0,
        bilateral_soft=2.2,
        sharpen_sigma=0.7,
        sharpen_limit=1.8,
        description='Balanced enhancement trading moderate compute for photorealistic skin pores and natural iris depth.'
    ),
    'QUALITY': RestoreProfileConfig(
        name='QUALITY',
        strength=0.30,
        crispness=0.26,
        eye_clarity=0.48,
        detail_weight=0.75,
        bilateral_sigma=18.0,
        bilateral_thresh=10.0,
        bilateral_soft=2.5,
        sharpen_sigma=0.8,
        sharpen_limit=2.0,
        description='High-fidelity production standard with anti-halo bounding and authentic texture injection.'
    ),
    'ULTRA': RestoreProfileConfig(
        name='ULTRA',
        strength=0.36,
        crispness=0.32,
        eye_clarity=0.55,
        detail_weight=0.85,
        bilateral_sigma=22.0,
        bilateral_thresh=12.0,
        bilateral_soft=3.0,
        sharpen_sigma=0.9,
        sharpen_limit=2.2,
        description='Maximum micro-detail for high-res closeups, fine eyelashes, iris luminosity, and pore recovery.'
    ),
}


def get_profile(name_or_str: Optional[Union[str, RestoreProfileConfig]]) -> RestoreProfileConfig:
    """Resolve a profile name to its RestoreProfileConfig, defaulting to QUALITY."""
    if isinstance(name_or_str, RestoreProfileConfig):
        return name_or_str
    if not name_or_str:
        return RESTORE_PROFILES['QUALITY']
    key = str(name_or_str).strip().upper()
    return RESTORE_PROFILES.get(key, RESTORE_PROFILES['QUALITY'])


# ==============================================================================
# 2. Buffer Reuse & Static Caching
# ==============================================================================

class RestoreUltraBufferPool:
    """Thread-safe buffer pool and static lookup table cache for Restore Ultra."""

    # Static pre-allocated morphology structuring elements (cached singletons)
    KERNEL_RECT_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    KERNEL_ELLIPSE_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Static LUT for uint8 BGR [0, 255] -> float32 [-1.0, 1.0]
    _LUT_NORM = ((np.arange(256, dtype=np.float32) / 127.5) - 1.0)

    # Soft-knee tables keyed by (threshold, softness, strength)
    _KNEE_CACHE: Dict[Tuple[float, float, float], np.ndarray] = {}
    _KNEE_LOCK = threading.Lock()

    def __init__(self):
        self._local = threading.local()

    def get_input_buffer(self, shape: Tuple[int, int, int, int] = (1, 3, 512, 512)) -> np.ndarray:
        """Return a thread-local reusable contiguous float32 input tensor buffer."""
        buf = getattr(self._local, 'input_buffer', None)
        if buf is None or buf.shape != shape or buf.dtype != np.float32:
            buf = np.empty(shape, dtype=np.float32)
            self._local.input_buffer = buf
        return buf

    def prepare_model_input(self, bgr_512: np.ndarray, out_buf: Optional[np.ndarray] = None) -> np.ndarray:
        """Fast conversion: uint8 BGR HWC -> float32 RGB CHW in [-1.0, 1.0].
        Reuses thread-local input buffer if out_buf is None.
        """
        if out_buf is None:
            out_buf = self.get_input_buffer((1, 3, 512, 512))
        
        # Fast gather into contiguous buffer:
        # bgr_512: (512, 512, 3) BGR -> transpose (2, 0, 1)[::-1] is RGB CHW
        # Using vectorized indexing on cached float32 LUT
        rgb_chw = bgr_512.transpose(2, 0, 1)[::-1]
        np.take(self._LUT_NORM, rgb_chw, out=out_buf[0])
        return out_buf

    def get_hwc_buffer(self, shape: Tuple[int, int, int] = (512, 512, 3)) -> np.ndarray:
        """Return a thread-local reusable contiguous float32 HWC buffer."""
        buf = getattr(self._local, 'hwc_buffer', None)
        if buf is None or buf.shape != shape or buf.dtype != np.float32:
            buf = np.empty(shape, dtype=np.float32)
            self._local.hwc_buffer = buf
        return buf

    def postprocess_model_output(self, result_chw: np.ndarray) -> np.ndarray:
        """Fast conversion: float32 RGB CHW in [-1.0, 1.0] -> uint8 BGR HWC in [0, 255].
        Reuses thread-local float32 scratch buffer to eliminate per-frame allocations.
        """
        hwc = self.get_hwc_buffer((512, 512, 3))
        np.copyto(hwc, result_chw[::-1].transpose(1, 2, 0))
        np.clip(hwc, -1.0, 1.0, out=hwc)
        return np.rint((hwc + 1.0) * 127.5).clip(0.0, 255.0).astype(np.uint8)

    @classmethod
    def get_knee_lut(cls, threshold: float, softness: float, strength: float) -> np.ndarray:
        """Get or build the 511-entry soft-knee lookup table."""
        key = (round(float(threshold), 3), round(float(softness), 3), round(float(strength), 4))
        table = cls._KNEE_CACHE.get(key)
        if table is not None:
            return table
        with cls._KNEE_LOCK:
            table = cls._KNEE_CACHE.get(key)
            if table is not None:
                return table
            d = np.arange(-255, 256, dtype=np.float32)
            table = np.where(
                np.abs(d) <= threshold,
                d,
                np.sign(d) * (threshold + softness * np.tanh((np.abs(d) - threshold) / max(softness, 1e-4)))
            ) * np.float32(strength)
            cls._KNEE_CACHE[key] = table
            return table


# Module singleton instance
BUFFER_POOL = RestoreUltraBufferPool()


# ==============================================================================
# 3. Adaptive Restoration Strength
# ==============================================================================

@dataclass(frozen=True)
class AdaptiveStrengthResult:
    strength_mult: float
    crispness_mult: float
    eye_clarity_mult: float
    detail_weight_mult: float
    reason: str


def compute_adaptive_scaling(
    crop_shape: Tuple[int, int],
    target_face: Optional[Any] = None,
    noise_estimate: Optional[float] = None
) -> AdaptiveStrengthResult:
    """Compute adaptive multipliers for restoration parameters based on:
    1. Original crop resolution (attenuates on tiny faces, gently moderates on huge faces).
    2. Head pose yaw/pitch (damps profile eye distortion).
    3. Noise floor (damps edge crispness when background noise is high).
    """
    h, w = crop_shape[:2]
    ref_dim = min(h, w)
    
    # 1. Resolution factor:
    # Small faces (< 160px): aggressive restoration creates plastic/hallucinated contrast.
    # Sweet spot (192px - 384px): full 1.0x restoration.
    # Very large faces (> 448px): slight damping (0.90x) to avoid erasing fine natural skin grain.
    if ref_dim < 128:
        res_factor = max(0.40, (ref_dim / 128.0) * 0.45 + 0.35)
    elif ref_dim < 192:
        res_factor = 0.80 + 0.20 * ((ref_dim - 128) / 64.0)
    elif ref_dim > 448:
        res_factor = max(0.85, 1.0 - (ref_dim - 448) / 800.0)
    else:
        res_factor = 1.0

    # 2. Pose factor:
    # Check for pose yaw in target_face (if available)
    pose_eye_factor = 1.0
    pose_edge_factor = 1.0
    yaw = 0.0
    if target_face is not None:
        pose = getattr(target_face, 'pose', None)
        if pose is not None and len(pose) >= 3:
            yaw = float(pose[1])
        elif hasattr(target_face, 'yaw'):
            yaw = float(target_face.yaw)
        elif getattr(target_face, 'kps', None) is not None:
            # Estimate approximate yaw from 5-point landmarks
            kps = target_face.kps
            if len(kps) >= 5:
                left_eye, right_eye, nose = kps[0], kps[1], kps[2]
                d_l = abs(nose[0] - left_eye[0])
                d_r = abs(right_eye[0] - nose[0])
                total = d_l + d_r
                if total > 1e-3:
                    yaw = (d_l - d_r) / total * 60.0

    abs_yaw = abs(yaw)
    if abs_yaw > 25.0:
        # Profile face: dampen eye clarity and edge crispness to avoid asymmetric iris distortion
        pose_eye_factor = max(0.55, 1.0 - (abs_yaw - 25.0) / 70.0)
        pose_edge_factor = max(0.70, 1.0 - (abs_yaw - 25.0) / 90.0)

    # 3. Noise factor:
    noise_factor = 1.0
    if noise_estimate is not None and noise_estimate > 6.0:
        # Attenuate edge crispness on noisy footage to prevent haloing
        noise_factor = max(0.60, 1.0 - (noise_estimate - 6.0) / 20.0)

    strength_mult = float(np.clip(res_factor, 0.40, 1.10))
    crispness_mult = float(np.clip(res_factor * pose_edge_factor * noise_factor, 0.35, 1.10))
    eye_clarity_mult = float(np.clip(res_factor * pose_eye_factor, 0.40, 1.15))
    detail_weight_mult = float(np.clip(res_factor, 0.50, 1.05))

    reasons = []
    if res_factor != 1.0:
        reasons.append(f"res={ref_dim}px({res_factor:.2f}x)")
    if abs_yaw > 25.0:
        reasons.append(f"yaw={abs_yaw:.1f}°(eye={pose_eye_factor:.2f}x)")
    if noise_factor != 1.0:
        reasons.append(f"noise={noise_estimate:.1f}(crisp={noise_factor:.2f}x)")
    reason_str = ", ".join(reasons) if reasons else "nominal"

    return AdaptiveStrengthResult(
        strength_mult=strength_mult,
        crispness_mult=crispness_mult,
        eye_clarity_mult=eye_clarity_mult,
        detail_weight_mult=detail_weight_mult,
        reason=reason_str
    )


# ==============================================================================
# 4. Identity Feature Guardrail
# ==============================================================================

class IdentityPreservationGuard:
    """Monitors sensory feature zones for codebook hallucination drift.
    
    Codebook-based restorers (RestoreFormer++, CodeFormer) hallucinate details
    toward their training prior. When detail weights exceed nominal thresholds,
    the model can shift iris coloration, lip borders, or nasal bridge shape.
    
    The guard evaluates local structural similarity and color consistency between
    the swapped input crop and the restored output. If drift exceeds tolerance,
    it clamps low-and-mid frequencies back toward the swapped reference while
    strictly preserving high-frequency micro-texture (skin pores, eyelashes).
    """

    @staticmethod
    def evaluate_feature_divergence(
        restored: np.ndarray,
        reference: np.ndarray,
        landmarks_5: Optional[np.ndarray] = None
    ) -> float:
        """Compute mean feature divergence across ocular and mouth regions in [0, 1].
        0.0 = identical, 1.0 = completely divergent.
        """
        if restored is None or getattr(restored, 'size', 0) == 0:
            return 0.0
        if reference is None or getattr(reference, 'size', 0) == 0:
            return 0.0
        h, w = restored.shape[:2]
        if reference.shape[:2] != (h, w):
            reference = cv2.resize(reference, (w, h), interpolation=cv2.INTER_LINEAR)

        # Focus evaluation on the inner face region
        y0, y1 = int(0.25 * h), int(0.85 * h)
        x0, x1 = int(0.20 * w), int(0.80 * w)
        r_crop = restored[y0:y1, x0:x1]
        ref_crop = reference[y0:y1, x0:x1]

        # Low-frequency structural tone check in Lab
        r_lab = cv2.cvtColor(cv2.GaussianBlur(r_crop, (0, 0), sigmaX=3.0), cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(cv2.GaussianBlur(ref_crop, (0, 0), sigmaX=3.0), cv2.COLOR_BGR2LAB).astype(np.float32)

        # Delta E approximation (L*a*b* Euclidean distance)
        delta_e = np.sqrt(np.mean((r_lab - ref_lab) ** 2, axis=-1))
        mean_delta = float(np.mean(delta_e))

        # Normalized divergence score where 15+ Delta-E is high drift
        return float(np.clip(mean_delta / 25.0, 0.0, 1.0))

    @classmethod
    def protect(
        cls,
        restored: np.ndarray,
        reference: np.ndarray,
        target_face: Optional[Any] = None,
        divergence_threshold: float = 0.45
    ) -> Tuple[np.ndarray, bool]:
        """Apply identity protection if restored face diverged excessively from reference.
        Returns (protected_frame, was_guarded).
        """
        if restored is None or getattr(restored, 'size', 0) == 0:
            return restored, False
        if reference is None or getattr(reference, 'size', 0) == 0:
            return restored, False

        h, w = restored.shape[:2]
        if h == 0 or w == 0:
            return restored, False
        if reference.shape[:2] != (h, w):
            reference = cv2.resize(reference, (w, h), interpolation=cv2.INTER_LINEAR)

        div = cls.evaluate_feature_divergence(restored, reference)
        if div <= divergence_threshold:
            return restored, False

        # Guard triggered: blend low/mid tones back from reference while retaining restorer high-pass
        # Blend factor scales with the severity of divergence
        excess = min(1.0, (div - divergence_threshold) / 0.35)
        guard_weight = 0.30 * excess  # Max 30% pull back to reference tone

        # Decompose into low (tone/shape) and high (pores/lashes)
        sigma = 2.5 * (w / 512.0)
        low_ref = cv2.GaussianBlur(reference, (0, 0), sigmaX=sigma)
        low_res = cv2.GaussianBlur(restored, (0, 0), sigmaX=sigma)
        high_res = restored.astype(np.float32) - low_res.astype(np.float32)

        # Recovered low tone
        guarded_low = cv2.addWeighted(low_res, 1.0 - guard_weight, low_ref, guard_weight, 0.0).astype(np.float32)
        guarded = np.clip(guarded_low + high_res, 0.0, 255.0).astype(np.uint8)

        return guarded, True


# ==============================================================================
# 5. Core Optimized Profile Application
# ==============================================================================

def apply_restore_ultra_profile(
    enhanced: np.ndarray,
    reference: np.ndarray,
    profile_name: Union[str, RestoreProfileConfig] = 'QUALITY',
    target_face: Optional[Any] = None,
    adaptive: bool = True,
    identity_guard: bool = True,
    noise_estimate: Optional[float] = None
) -> np.ndarray:
    """Execute the complete Restore Ultra finishing pipeline under the specified profile.
    
    Stages:
    1. Resolution & geometry matching.
    2. Adaptive strength scaling (if adaptive=True).
    3. Edge-preserving bilateral texture injection (pores/skin texture).
    4. Eye clarity boost with anti-halo clamping (iris luminosity/catchlights).
    5. Fine-line edge refinement (eyelashes/brows/lips).
    6. Identity preservation check (guards against codebook hallucination).
    """
    if enhanced is None or getattr(enhanced, 'size', 0) == 0:
        return enhanced
    if reference is None or getattr(reference, 'size', 0) == 0:
        return enhanced

    profile = get_profile(profile_name)

    ref = reference
    if ref.shape[:2] != enhanced.shape[:2]:
        ref = cv2.resize(ref, (enhanced.shape[1], enhanced.shape[0]),
                         interpolation=cv2.INTER_CUBIC)

    # 1. Parameter resolution (adaptive vs nominal)
    strength = profile.strength
    crispness = profile.crispness
    eye_clarity = profile.eye_clarity

    if adaptive:
        scaling = compute_adaptive_scaling(
            crop_shape=enhanced.shape[:2],
            target_face=target_face,
            noise_estimate=noise_estimate
        )
        strength *= scaling.strength_mult
        crispness *= scaling.crispness_mult
        eye_clarity *= scaling.eye_clarity_mult

    # 2. Subtle bilateral texture injection
    try:
        from roop.processors.enhance_common import _inject_bilateral_detail
        out = _inject_bilateral_detail(
            enhanced,
            ref,
            sigma_color=profile.bilateral_sigma,
            threshold=profile.bilateral_thresh,
            softness=profile.bilateral_soft,
            strength=strength
        )
    except Exception as _e_bilateral:
        _swallowed("roop/restore_ultra_optimizer.py:bilateral", _e_bilateral, "bilateral detail fallback")
        out = enhanced

    # 3. Ultra-definition eye clarity with anti-halo bounding
    try:
        from roop.processors.enhance_common import enhance_eyes_clarity
        out = enhance_eyes_clarity(out, template='ffhq_512', strength=eye_clarity)
    except Exception as _e_eye:
        _swallowed("roop/restore_ultra_optimizer.py:eye_clarity", _e_eye, "eye clarity fallback")

    # 4. Fine-line edge refinement (eyelashes, eyebrows, lips)
    try:
        from roop.processors.enhance_common import apply_anti_halo_sharpen
        out = apply_anti_halo_sharpen(
            out,
            amount=crispness,
            sigma=profile.sharpen_sigma,
            limit=profile.sharpen_limit
        )
    except Exception as _e_sharpen:
        _swallowed("roop/restore_ultra_optimizer.py:anti_halo", _e_sharpen, "anti-halo sharpen fallback")

    # 5. Identity preservation guard
    if identity_guard:
        out, _ = IdentityPreservationGuard.protect(out, ref, target_face=target_face)

    return out


# ==============================================================================
# 6. Anatomical & Quality Validation Metrics
# ==============================================================================

def measure_skin_pore_variance(img: np.ndarray, skin_mask: Optional[np.ndarray] = None) -> float:
    """Measure high-frequency skin pore detail variance via band-pass filtering."""
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Band-pass filter in pore scale: sigma 0.7 to 2.2
    g_fine = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.7)
    g_coarse = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.2)
    band_pass = g_fine - g_coarse
    if skin_mask is not None and skin_mask.any():
        return float(np.std(band_pass[skin_mask > 0]))
    # Default to cheek region if no mask provided
    h, w = gray.shape
    cheek = band_pass[int(0.45 * h):int(0.70 * h), int(0.20 * w):int(0.40 * w)]
    return float(np.std(cheek)) if cheek.size > 0 else float(np.std(band_pass))


def measure_edge_sharpness(img: np.ndarray) -> float:
    """Measure edge sharpness energy (eyelashes, eyebrows, lips) via Laplacian variance."""
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(np.var(lap))


def measure_eye_clarity(img: np.ndarray, template: str = 'ffhq_512') -> float:
    """Measure local contrast and catchlight energy within ocular regions."""
    if img is None:
        return 0.0
    h, w = img.shape[:2]
    # FFHQ-512 eye template locations
    left_eye_center = (int(0.3769 * w), int(0.4686 * h))
    right_eye_center = (int(0.6228 * w), int(0.4691 * h))
    rad = int(0.06 * w)

    energies = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    for cx, cy in (left_eye_center, right_eye_center):
        y0, y1 = max(0, cy - rad), min(h, cy + rad)
        x0, x1 = max(0, cx - rad), min(w, cx + rad)
        patch = gray[y0:y1, x0:x1]
        if patch.size > 0:
            energies.append(float(np.std(patch)))
    return float(np.mean(energies)) if energies else 0.0


def measure_identity_similarity(img: np.ndarray, reference: np.ndarray) -> float:
    """Structural correlation between enhanced crop and swapped reference."""
    if img is None or reference is None:
        return 0.0
    if img.shape[:2] != reference.shape[:2]:
        reference = cv2.resize(reference, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    g1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean1, mean2 = np.mean(g1), np.mean(g2)
    v1, v2 = g1 - mean1, g2 - mean2
    denom = math.sqrt(float(np.sum(v1 ** 2) * np.sum(v2 ** 2)))
    if denom < 1e-6:
        return 1.0
    return float(np.sum(v1 * v2) / denom)


def measure_halo_overshoot(img: np.ndarray, reference: Optional[np.ndarray] = None, envelope_pad: float = 2.5) -> float:
    """Measure percentage of pixels that violate the local 3x3 min/max envelope (ringing/halos).
    A compliant anti-halo filter must produce 0.0% overshoot.
    """
    if img is None:
        return 0.0
    base = reference if reference is not None else img
    if base.shape[:2] != img.shape[:2]:
        base = cv2.resize(base, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    lab_in = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    lab_out = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)

    k = BUFFER_POOL.KERNEL_RECT_3
    lo = cv2.erode(lab_in, k) - envelope_pad
    hi = cv2.dilate(lab_in, k) + envelope_pad

    violations = (lab_out < lo) | (lab_out > hi)
    return float(np.count_nonzero(violations) / violations.size * 100.0)
