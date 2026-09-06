"""Target-conditioned lighting analysis and conservative appearance control.

The swapper supplies identity.  This module measures only the aligned TARGET
crop, so exposure, colour cast, shadows, highlights, and local contrast remain
properties of the shot.  It intentionally contains no face-restoration model
and never copies target high-frequency texture.
"""

import copy
import math
from threading import RLock

import cv2
import numpy as np


NORMAL = "NORMAL"
DARK = "DARK"
VERY_DARK = "VERY_DARK"


def _safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _face_ellipse(shape):
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    # The aligned face occupies the centre.  A soft central region avoids
    # letting hair/background pixels decide the skin-region colour statistics.
    cv2.ellipse(mask, (w // 2, int(h * 0.51)),
                (max(1, int(w * 0.36)), max(1, int(h * 0.43))),
                0, 0, 360, 255, -1)
    return mask


def _soft_mask(shape):
    mask = _face_ellipse(shape).astype(np.float32) / 255.0
    sigma = max(1.0, min(shape[:2]) / 32.0)
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)


def _percentiles(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0,
                "mean": 0.0, "std": 0.0}
    return {
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def classify_low_light(luminance):
    """Classify target exposure without trying to repair it.

    Percentiles are used together so a bright specular point cannot make a
    dark face look normal.  The thresholds are in normalized luminance.
    """
    lum = luminance or {}
    p50 = _safe_float(lum.get("p50"))
    p90 = _safe_float(lum.get("p90"))
    mean = _safe_float(lum.get("mean"))
    if p50 <= 0.16 or (mean <= 0.19 and p90 <= 0.48):
        return VERY_DARK
    if p50 <= 0.33 or (mean <= 0.35 and p90 <= 0.70):
        return DARK
    return NORMAL


def restoration_factor(tier):
    return {NORMAL: 1.0, DARK: 0.58, VERY_DARK: 0.20}.get(tier, 1.0)


def sharpen_factor(tier):
    return {NORMAL: 1.0, DARK: 0.50, VERY_DARK: 0.20}.get(tier, 1.0)


def analyze_target_appearance(image):
    """Return robust target illumination/color statistics and a low-res field.

    ``illumination`` is a 16x16 low-pass luminance field.  It carries spatial
    shadow/highlight structure, but deliberately excludes high-frequency
    source texture, sensor noise, and JPEG detail.
    """
    if image is None or getattr(image, "size", 0) == 0:
        return {"tier": VERY_DARK, "luminance": {}, "skin_chroma": {},
                "local_contrast": 0.0, "shadow_fraction": 1.0,
                "highlight_fraction": 0.0, "color_temperature": 1.0,
                "illumination": np.zeros((16, 16), np.float32)}
    try:
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("target appearance expects a BGR image")
        small = cv2.resize(image[:, :, :3], (128, 128), interpolation=cv2.INTER_AREA)
        bgr = small.astype(np.float32) / 255.0
        y = (0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1]
             + 0.299 * bgr[:, :, 2])
        region = _face_ellipse(y.shape)
        pixels = region > 0
        if int(pixels.sum()) < 64:
            pixels = np.ones_like(pixels, dtype=bool)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        chroma = lab[pixels, 1:3]
        lum = _percentiles(y[pixels])
        low = cv2.GaussianBlur(y, (0, 0), sigmaX=6.0, sigmaY=6.0)
        field = cv2.resize(low, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
        residual = y - low
        local_contrast = float(np.std(residual[pixels]))
        b, g, r = bgr[pixels].mean(axis=0)
        # Red/blue ratio is retained as a diagnostic; unlike a white-balance
        # normalizer it never forces the result toward neutral white.
        temp = float(r / max(1e-4, b))
        result = {
            "tier": classify_low_light(lum),
            "luminance": lum,
            "skin_chroma": {
                "mean": [float(v) for v in chroma.mean(axis=0)],
                "std": [float(v) for v in chroma.std(axis=0)],
            },
            "local_contrast": local_contrast,
            "shadow_fraction": float(np.mean(y[pixels] < 0.16)),
            "highlight_fraction": float(np.mean(y[pixels] > 0.88)),
            "color_temperature": temp,
            "illumination": field,
        }
        return result
    except (cv2.error, TypeError, ValueError, FloatingPointError):
        return {"tier": VERY_DARK, "luminance": {}, "skin_chroma": {},
                "local_contrast": 0.0, "shadow_fraction": 1.0,
                "highlight_fraction": 0.0, "color_temperature": 1.0,
                "illumination": np.zeros((16, 16), np.float32)}


def _blend_value(old, new, alpha):
    if isinstance(old, dict) and isinstance(new, dict):
        out = dict(old)
        for key, value in new.items():
            out[key] = _blend_value(old.get(key), value, alpha)
        return out
    if isinstance(old, (list, tuple)) and isinstance(new, (list, tuple)):
        if len(old) != len(new):
            return copy.deepcopy(new)
        return [_blend_value(a, b, alpha) for a, b in zip(old, new)]
    if isinstance(old, np.ndarray) and isinstance(new, np.ndarray):
        if old.shape != new.shape:
            return new.copy()
        return ((1.0 - alpha) * old + alpha * new).astype(np.float32)
    if isinstance(old, (int, float, np.number)) and isinstance(new, (int, float, np.number)):
        return float((1.0 - alpha) * float(old) + alpha * float(new))
    return copy.deepcopy(new)


class TargetAppearanceStabilizer:
    """Bounded per-track EMA for target lighting, suitable for block cloning."""

    def __init__(self, enabled=False, alpha=0.30, cache_size=256):
        self.enabled = bool(enabled)
        self.alpha = min(1.0, max(0.05, float(alpha)))
        self.cache_size = max(16, int(cache_size))
        self.states = {}
        self.ordered = True
        self._lock = RLock()

    def warmup_frames(self, eps=0.01):
        numerator = math.log(max(float(eps), 1e-4))
        denominator = math.log(max(1e-6, 1.0 - self.alpha))
        return int(max(0, math.ceil(numerator / denominator)))

    def set_ordered(self, ordered):
        self.ordered = bool(ordered)

    def clone_for_block(self):
        clone = copy.copy(self)
        clone._lock = RLock()
        clone.states = {}
        return clone

    def reset(self):
        with self._lock:
            self.states.clear()

    def update(self, track_id, appearance, confidence=1.0, motion=0.0):
        if not self.enabled or appearance is None or track_id is None:
            return appearance
        try:
            key = int(track_id)
        except (TypeError, ValueError):
            return appearance
        with self._lock:
            current = copy.deepcopy(appearance)
            previous = self.states.get(key)
            if previous is None:
                self.states[key] = current
                return current
            alpha = self.alpha + 0.35 * min(1.0, max(0.0, float(motion)))
            alpha *= 0.65 + 0.35 * min(1.0, max(0.0, float(confidence)))
            # Scene/exposure changes must be allowed through rather than being
            # smeared into a warm/cool ghost.
            old_tier, new_tier = previous.get("tier"), current.get("tier")
            if old_tier != new_tier:
                alpha = max(alpha, 0.75)
            merged = _blend_value(previous, current, min(1.0, max(0.05, alpha)))
            merged["tier"] = classify_low_light(merged.get("luminance"))
            self.states[key] = merged
            if len(self.states) > self.cache_size:
                self.states.pop(next(iter(self.states)))
            return copy.deepcopy(merged)


def protect_restorer_output(enhanced, pre_restore, tier):
    """Keep GPEN/UltraMax/etc. from lifting a dark target into a clean bright face."""
    if enhanced is None or pre_restore is None or tier == NORMAL:
        return enhanced
    reference = pre_restore
    if reference.shape[:2] != enhanced.shape[:2]:
        reference = cv2.resize(reference, (enhanced.shape[1], enhanced.shape[0]),
                                interpolation=cv2.INTER_AREA)
    factor = restoration_factor(tier)
    return cv2.addWeighted(enhanced, factor, reference, 1.0 - factor, 0.0)


__all__ = [
    "NORMAL", "DARK", "VERY_DARK", "analyze_target_appearance",
    "classify_low_light", "restoration_factor", "sharpen_factor",
    "TargetAppearanceStabilizer", "protect_restorer_output", "_soft_mask",
]
