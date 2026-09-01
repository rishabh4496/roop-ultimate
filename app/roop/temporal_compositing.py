"""Adaptive, temporally stable face paste-back compositing.

The swap crop owns identity and the untouched frame owns illumination.  This
module only combines those two already-produced images; it does not copy the
target's high-frequency texture.  The production method is a small two-band
blend: low frequencies are locally adapted to the target and high frequencies
are admitted strongly in the face interior, but conservatively at semantic
boundaries.  A Poisson/gradient-domain pass is deliberately not used in the
hot path: OpenCV's ``seamlessClone`` is CPU-only, allocates a full patch, and
does not improve the synthetic boundary metric enough to justify its cost.
"""

from dataclasses import dataclass
import math
from threading import RLock

import cv2
import numpy as np


# Compositing failures reported once per distinct cause.
#
# Both call sites in `procmgr_masking.paste_upscale` fall back to the legacy
# linear paste on any exception, which is the right RUNTIME behaviour -- this is
# a quality layer and a malformed optional track field must leave the
# established matte usable. Falling back SILENTLY is not: a user who enables
# `temporal_compositing` and hits an exception on every face sees a run that
# returns 0, reports 100% swapped, and produces exactly the legacy output, which
# is indistinguishable from "the feature had no effect". That confusion is the
# single most expensive pattern in this project's history.
#
# Bounded like face_util's detector reporter and the adaptive enhancer's
# fallback: these sit on the per-face path, so an unbounded print is one line
# per face per frame.
_COMPOSITE_FAIL_SEEN = {}
_COMPOSITE_FAIL_LOCK = RLock()


def warn_compositing_fallback(stage, exc):
    """Announce a fall back to the legacy paste -- once per distinct cause."""
    sig = (str(stage), type(exc).__name__, str(exc)[:200])
    with _COMPOSITE_FAIL_LOCK:
        seen = sig in _COMPOSITE_FAIL_SEEN
        _COMPOSITE_FAIL_SEEN[sig] = _COMPOSITE_FAIL_SEEN.get(sig, 0) + 1
    if seen:
        return
    # Each stage falls back to a DIFFERENT established path, and naming the
    # wrong one sends a reader to the wrong code.
    fallback = {'temporal occlusion': 'the established mask'}.get(
        str(stage), 'the legacy linear paste')
    print("[TemporalCompositing] %s FAILED; this face fell back to %s: %s: %s\n"
          "[TemporalCompositing] The render continues and reports success, so "
          "output that looks identical to having the feature off is expected "
          "while this persists. Further occurrences of this cause are counted, "
          "not printed."
          % (stage, fallback, type(exc).__name__, str(exc)[:200]), flush=True)


def compositing_fallback_counts():
    """Per-cause fallback totals, for harness and test reporting."""
    with _COMPOSITE_FAIL_LOCK:
        return dict(_COMPOSITE_FAIL_SEEN)


def reset_compositing_fallbacks():
    """Clear the once-per-process record. Used by tests."""
    with _COMPOSITE_FAIL_LOCK:
        _COMPOSITE_FAIL_SEEN.clear()


NORMAL = "NORMAL"
DARK = "DARK"
VERY_DARK = "VERY_DARK"


def _clip(value, low=0.0, high=1.0, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return float(np.clip(value, low, high))


def _field(face, key, default=None):
    if face is None:
        return default
    try:
        if isinstance(face, dict):
            return face.get(key, default)
        return getattr(face, key, default)
    except Exception:
        return default


def _tier(appearance):
    value = str((appearance or {}).get("tier", NORMAL)).upper()
    return value if value in (NORMAL, DARK, VERY_DARK) else NORMAL


def _canonical_small(mask, size=64):
    value = np.asarray(mask, dtype=np.float32)
    if value.ndim == 3:
        value = value[..., 0]
    value = np.clip(value, 0.0, 1.0)
    if value.shape != (size, size):
        value = cv2.resize(value, (size, size), interpolation=cv2.INTER_AREA)
    return value.astype(np.float32, copy=False)


@dataclass
class CompositeState:
    track_id: int
    mask: object = None
    last_frame_index: int = -1


class TemporalCompositeController:
    """Per-track EMA for matte geometry plus compositor planning.

    The state is intentionally a 64x64 mask, not a full-resolution frame.  It
    is therefore safe for the 3060 profile and can be cloned for contiguous
    parallel blocks just like the other ordered temporal engines.
    """

    def __init__(self, enabled=False, strength=0.65, alpha=0.30,
                 cache_size=256, detail_weight=0.86, color_strength=0.55,
                 max_feather=8):
        self.enabled = bool(enabled)
        self.strength = _clip(strength)
        self.alpha = _clip(alpha, 0.05, 0.8, 0.30)
        self.cache_size = max(16, int(cache_size))
        self.detail_weight = _clip(detail_weight, 0.45, 1.0, 0.86)
        self.color_strength = _clip(color_strength, 0.0, 1.0, 0.55)
        self.max_feather = max(1, min(16, int(max_feather)))
        self.states = {}
        self.ordered = True
        self._lock = RLock()

    @classmethod
    def from_config(cls, globals_module):
        return cls(
            enabled=bool(getattr(globals_module, "temporal_compositing", False)),
            strength=getattr(globals_module, "temporal_compositing_strength", 0.65),
            alpha=getattr(globals_module, "temporal_compositing_mask_alpha", 0.30),
            cache_size=getattr(globals_module, "temporal_compositing_cache_size", 256),
            detail_weight=getattr(globals_module, "temporal_compositing_detail_weight", 0.86),
            color_strength=getattr(globals_module, "temporal_compositing_color_strength", 0.55),
            max_feather=getattr(globals_module, "temporal_compositing_max_feather", 8),
        )

    def clone_for_block(self):
        clone = TemporalCompositeController(
            enabled=self.enabled, strength=self.strength, alpha=self.alpha,
            cache_size=self.cache_size, detail_weight=self.detail_weight,
            color_strength=self.color_strength, max_feather=self.max_feather)
        clone.ordered = self.ordered
        return clone

    def warmup_frames(self, eps=0.01):
        return int(max(0, math.ceil(math.log(max(float(eps), 1e-4)) /
                                  math.log(max(1e-6, 1.0 - self.alpha)))) )

    def set_ordered(self, ordered):
        self.ordered = bool(ordered)

    def reset(self):
        with self._lock:
            self.states.clear()

    def _state(self, track_id):
        try:
            key = int(track_id)
        except (TypeError, ValueError):
            return None
        state = self.states.get(key)
        if state is None:
            if len(self.states) >= self.cache_size:
                self.states.pop(next(iter(self.states)))
            state = self.states[key] = CompositeState(key)
        return state

    def stabilize_mask(self, track_id, mask, frame_index=None, confidence=1.0,
                       motion=0.0, occlusion=0.0):
        """Return a canonical mask whose boundary cannot chatter frame to frame."""
        if not self.enabled or track_id is None:
            return mask
        current = _canonical_small(mask)
        with self._lock:
            state = self._state(track_id)
            if state is None:
                return mask
            try:
                frame_index = int(frame_index)
            except (TypeError, ValueError):
                frame_index = state.last_frame_index + 1
            if state.mask is None or frame_index <= state.last_frame_index:
                state.mask = current.copy()
                state.last_frame_index = frame_index
                return mask
            change = float(np.mean(np.abs(current - state.mask)))
            # Small detector/mask noise gets a slow EMA.  A real pose or
            # occlusion change gets through faster, while still remaining
            # continuous rather than hard/soft/hard.
            update = self.alpha + 0.55 * min(1.0, change / 0.12)
            update += 0.20 * _clip(motion)
            update += 0.20 * _clip(occlusion)
            update *= 0.65 + 0.35 * _clip(confidence)
            update = _clip(update, self.alpha, 0.92)
            stable = (1.0 - update) * state.mask + update * current
            state.mask = np.clip(stable, 0.0, 1.0).astype(np.float32)
            state.last_frame_index = frame_index
            if np.asarray(mask).ndim == 3:
                shape = np.asarray(mask).shape[:2]
            else:
                shape = np.asarray(mask).shape
            if tuple(shape) == state.mask.shape:
                return state.mask.copy()
            return cv2.resize(state.mask, (int(shape[1]), int(shape[0])),
                              interpolation=cv2.INTER_LINEAR).astype(np.float32)

    def plan(self, target_face=None, appearance=None, local_contrast=0.0,
             occlusion=0.0):
        """Compute bounded, explainable blending controls for one face."""
        yaw = abs(_clip(_field(target_face, "_adaptive_yaw", 0.0), 0.0, 180.0))
        pitch = abs(_clip(_field(target_face, "_adaptive_pitch", 0.0), 0.0, 90.0))
        angle = _clip(max(yaw / 90.0, pitch / 60.0))
        confidence = _clip(_field(target_face, "_temporal_confidence",
                                  _field(target_face, "det_score", 1.0)), default=1.0)
        tier = _tier(appearance)
        tier_factor = {NORMAL: 1.0, DARK: 0.72, VERY_DARK: 0.42}[tier]
        # Boundary contrast calls for more low-band adaptation and a little
        # more feather, never for more sharpening or more copied texture.
        contrast = _clip(local_contrast / 48.0)
        feather = 1.0 + 2.0 * angle + 2.0 * contrast + 1.2 * _clip(occlusion)
        feather = max(1.0, min(float(self.max_feather), feather))
        strength = self.strength * tier_factor
        strength *= 0.72 + 0.28 * confidence
        strength *= 1.0 - 0.25 * angle
        strength *= 1.0 - 0.28 * _clip(occlusion)
        return {
            "strength": _clip(strength),
            "angle": float(angle),
            "confidence": float(confidence),
            "occlusion": _clip(occlusion),
            "local_contrast": float(contrast),
            "tier": tier,
            "feather_px": float(feather),
            "color_strength": float(self.color_strength * (0.85 + 0.15 * tier_factor)),
            "detail_weight": float(self.detail_weight * (1.0 - 0.25 * contrast)),
            "method": "multiband",
        }


def boundary_contrast(image, alpha):
    """Measure target edge energy only in the current matte boundary band."""
    if image is None or alpha is None:
        return 0.0
    try:
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
        mask = np.asarray(alpha, dtype=np.float32)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if gray.shape != mask.shape:
            return 0.0
        hard = (mask > 0.04).astype(np.uint8)
        band = cv2.morphologyEx(hard, cv2.MORPH_GRADIENT,
                                np.ones((5, 5), np.uint8)) > 0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        values = np.hypot(gx, gy)[band]
        return float(np.percentile(values, 75)) if values.size else 0.0
    except (cv2.error, TypeError, ValueError):
        return 0.0


def _semantic_boundary_weight(alpha, landmarks=None):
    """Lower only the outer semantic contour (jaw/cheek/forehead/hair edge)."""
    value = np.asarray(alpha, dtype=np.float32)
    binary = (value > 0.08).astype(np.uint8)
    if not np.any(binary):
        return np.ones_like(value, dtype=np.float32)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    scale = max(2.0, float(np.percentile(dist[dist > 0], 80)) if np.any(dist > 0) else 2.0)
    interior = np.clip(dist / scale, 0.0, 1.0)
    weight = 0.82 + 0.18 * interior
    # Landmark availability confirms that the contour is a face boundary. The
    # weight remains geometry-safe when a detector has no dense landmarks.
    if landmarks is not None:
        try:
            points = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
            if points.shape[0] >= 5 and np.isfinite(points).all():
                hull = np.zeros(value.shape, np.uint8)
                cv2.fillConvexPoly(hull, cv2.convexHull(points.astype(np.int32)), 1)
                # The established paste mask may intentionally extend beyond
                # the raw 106-point hull (notably the projected forehead). Do
                # not clip that geometry here; semantic weighting only softens
                # the contour where landmarks actually identify it.
                weight = np.where(hull > 0, weight, 1.0)
        except (cv2.error, TypeError, ValueError):
            pass
    return weight.astype(np.float32)


def refine_alpha(alpha, plan, target=None, landmarks=None):
    """Apply a small adaptive feather, semantic edge falloff and strength gate."""
    value = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    radius = float((plan or {}).get("feather_px", 1.0))
    if radius > 1.05:
        soft = cv2.GaussianBlur(value, (0, 0), sigmaX=radius, sigmaY=radius)
        # Preserve the already-good matte; only 35% of the adaptive softening
        # is admitted, preventing an overly broad blur at high resolution.
        value = 0.65 * value + 0.35 * soft
    value *= _semantic_boundary_weight(value, landmarks)
    value *= float((plan or {}).get("strength", 1.0))
    return np.clip(value, 0.0, 1.0).astype(np.float32)


def composite_linear(paste, target, alpha):
    """Reference linear compositor used by the benchmark and regression tests."""
    a = np.asarray(alpha, dtype=np.float32)[..., None]
    return np.clip(np.asarray(target, dtype=np.float32) * (1.0 - a) +
                   np.asarray(paste, dtype=np.float32) * a, 0, 255).astype(np.uint8)


def composite_multiband(paste, target, alpha, plan=None):
    """Cheap two-band blend with target-conditioned low-frequency adaptation."""
    paste = np.asarray(paste, dtype=np.uint8)
    target = np.asarray(target, dtype=np.uint8)
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    if a.ndim == 3:
        a = a[..., 0]
    if paste.shape != target.shape or paste.ndim != 3 or a.shape != paste.shape[:2]:
        raise ValueError("paste, target and alpha must share an image shape")
    sigma = max(1.0, min(8.0, float((plan or {}).get("feather_px", 1.0)) * 1.35))
    p = paste.astype(np.float32)
    t = target.astype(np.float32)
    low_p = cv2.GaussianBlur(p, (0, 0), sigmaX=sigma, sigmaY=sigma)
    low_t = cv2.GaussianBlur(t, (0, 0), sigmaX=sigma, sigmaY=sigma)
    color_strength = _clip((plan or {}).get("color_strength", 0.0))
    # Low-band adaptation follows target shadows/highlights but is clipped so a
    # bright background cannot bleach the identity crop.
    delta = np.clip(low_t - low_p, -42.0, 42.0) * color_strength
    adapted_low = low_p + delta
    high = p - low_p
    interior = cv2.distanceTransform((a > 0.5).astype(np.uint8),
                                     cv2.DIST_L2, 3)
    scale = max(1.0, float(np.percentile(interior[interior > 0], 75))
                if np.any(interior > 0) else 1.0)
    interior = np.clip(interior / scale, 0.0, 1.0)
    detail_weight = _clip((plan or {}).get("detail_weight", 0.86), 0.45, 1.0)
    high_alpha = a * (detail_weight + (1.0 - detail_weight) * interior)
    out = t + a[..., None] * (adapted_low - t) + high_alpha[..., None] * high
    return np.clip(out, 0, 255).astype(np.uint8)


def technique_report():
    """Describe the measured production choice for benchmark/documentation."""
    return {
        "production": "multiband_local_color",
        "gradient_domain_available": bool(hasattr(cv2, "seamlessClone")),
        "gradient_domain_production": False,
        "reason": "CPU full-patch Poisson/gradient-domain cost is not justified by the boundary metric",
    }
