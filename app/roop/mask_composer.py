"""Yaw-adaptive semantic paste matte (Bug #2, Stage 2).

Builds the single crop-space alpha that ``paste_upscale`` warps into the frame:

    alpha = base_alpha                       (ellipse / user offsets / occluder keeps)
          * target semantic interior          (skin, brows, eyes, nose, lips, mouth)
          * swapped-crop valid interior       (nothing the swap net did not draw)
          * landmark hull                     (a clip, never a second feather)
          * asymmetric inward feather         (yaw-aware, see below)

It replaces the static ellipse/hull silhouette on the contour only; the engine
keeps (``crop_alpha``), the user's mask offsets and the premultiplied paste are
untouched and stay in ``roop.unified_matte`` / ``paste_upscale``.

Sign convention
---------------
``yaw`` uses ``roop.unified_matte.yaw_from_kps``: positive yaw = the nose is
displaced toward the crop's +x (the viewer's right), i.e. the cheek on +x is the
FAR one -- it turns away from the camera and is foreshortened behind the nose
bridge. The near cheek (-x) is the broad, well-seen one.

Asymmetric boundary (the Bug #2 half of this module)
----------------------------------------------------
For ``s = min(|yaw| / 60, 1)`` and a lateral ramp ``f`` that is 0 at the face's
centre and 1 at its far extreme (``n`` the same toward the near extreme):

    feather(x) = feather * (1 + 0.6 s f(x) - 0.4 s n(x))     softer far, tighter near
    inset(x)   = 0.04 * span * s * f(x)^2                    far edge pulled inward

so the far silhouette -- where the swapped face's cheek meets whatever is behind
the head -- is eroded harder and fades slowly, and the near jawline hugs the
parsed contour. At yaw 0 both collapse to one symmetric inward feather.

These constants are analytic priors, NOT calibrated against footage. The audit
(`.audit/03_halo_audit.md` section 5) calls for a counterbalanced 600-frame A/B
before they are treated as tuned.

Deliberate deviation from the brief: "strictly excluding neck and background".
BiSeNet's frontal priors label the outer part of an angled face as background and
a tucked chin as neck (measured on s4.mp4, see Mask_RealityUX._NONFACE_OPAQUE).
With ``core_protect`` (default) those two classes cannot remove a pixel inside a
landmark-derived core; hair, ears, hat and cloth always can. ``core_protect=False``
is the literal brief and is what the zero-bleed tests assert.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from roop import face_parser as fp
from roop.unified_matte import mask2d

_YAW_FULL = 60.0


@dataclass
class ComposeResult:
    alpha: np.ndarray                     # (h, w) float32, 1 = swap
    interior: np.ndarray                  # (h, w) bool, semantic interior before the feather
    source: str                           # 'bisenet' | 'hull' | 'hull-fallback'
    parse_cover: float = 1.0              # |parsed interior ∩ hull| / |hull|
    swap_used: bool = False               # a swapped-crop parse contributed
    stats: dict = field(default_factory=dict)


class MaskComposer:
    def __init__(self, *, core_protect: bool = True, core_erode_frac: float = 0.12,
                 min_cover: float = 0.35, far_inset_frac: float = 0.04,
                 far_soften: float = 0.6, near_tighten: float = 0.4,
                 feather_cap: float = 0.18, keep_largest: bool = True):
        self.core_protect = bool(core_protect)
        self.core_erode_frac = float(core_erode_frac)
        self.min_cover = float(min_cover)
        self.far_inset_frac = float(far_inset_frac)
        self.far_soften = float(far_soften)
        self.near_tighten = float(near_tighten)
        self.feather_cap = float(feather_cap)
        self.keep_largest = bool(keep_largest)

    # ------------------------------------------------------------------ pieces
    def _core(self, hull_b: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Hull eroded inward: the part of the face neck/background labels may not remove."""
        if hull_b is None or not self.core_protect:
            return None
        area = float(hull_b.sum())
        if area < 64:
            return None
        r = max(1, int(round(self.core_erode_frac * np.sqrt(area))))     # erosion RADIUS in px
        # One distance transform instead of eroding with an 80+ px structuring element (5.7 ms vs ~1 ms
        # at 512 px). Padding by 1 keeps "outside the crop" as background, like BORDER_CONSTANT 0.
        dist = cv2.distanceTransform(np.pad(hull_b.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
        return dist > r

    def _interior(self, labels: np.ndarray, core: Optional[np.ndarray]) -> np.ndarray:
        keep = fp.class_mask(labels, fp.INTERIOR_CLASSES + ('glasses',))
        if core is None:
            return keep
        return keep | (fp.class_mask(labels, fp.AMBIGUOUS_EXCLUDED) & core)

    @staticmethod
    def _cover(interior: np.ndarray, hull_b: Optional[np.ndarray]) -> float:
        if hull_b is None:
            return 1.0
        denom = int(hull_b.sum())
        return 1.0 if denom == 0 else float((interior & hull_b).sum()) / denom

    def _largest(self, m: np.ndarray) -> np.ndarray:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        if n <= 2:
            return m
        areas = stats[1:, cv2.CC_STAT_AREA]
        big = int(np.argmax(areas)) + 1
        # Several comparable components = an ambiguous parse; do not guess which is the face.
        if areas.max() < 0.6 * areas.sum():
            return m
        return lab == big

    def asymmetric_inward(self, alpha, yaw: float, feather: float) -> np.ndarray:
        """Inward-only feather with a far-soft / near-tight lateral profile."""
        a = mask2d(alpha, alpha.shape[:2])
        support = (a > 0.01).astype(np.uint8)
        x0, y0, sw, sh = cv2.boundingRect(support)
        out = np.zeros_like(a)
        if sw == 0 or sh == 0:
            return out
        local = a[y0:y0 + sh, x0:x0 + sw]
        dist = cv2.distanceTransform(np.pad(support[y0:y0 + sh, x0:x0 + sw], 1),
                                     cv2.DIST_L2, 3)[1:-1, 1:-1]
        span = max(1, min(sw, sh))
        s = min(abs(float(yaw)) / _YAW_FULL, 1.0)
        if s > 0.0:
            rel = np.arange(sw, dtype=np.float32) / max(sw - 1, 1)
            far = rel if yaw > 0 else 1.0 - rel
            f = np.clip(2.0 * far - 1.0, 0.0, 1.0)[None, :]
            n = np.clip(1.0 - 2.0 * far, 0.0, 1.0)[None, :]
            dist = np.maximum(dist - self.far_inset_frac * span * s * f * f, 0.0)
            width = feather * (1.0 + self.far_soften * s * f - self.near_tighten * s * n)
        else:
            width = feather
        width = np.maximum(np.minimum(width, self.feather_cap * span), 1.0)
        t = np.clip(dist / width, 0.0, 1.0)
        out[y0:y0 + sh, x0:x0 + sw] = local * t * t * (3.0 - 2.0 * t)
        return out

    # --------------------------------------------------------------- main entry
    def compose(self, shape, target: fp.ParseResult, *,
                swap: Optional[fp.ParseResult] = None,
                hull=None, base_alpha=None,
                yaw: float = 0.0, feather: float = 8.0) -> ComposeResult:
        h, w = int(shape[0]), int(shape[1])
        hull_b = None if hull is None else (mask2d(hull, (h, w)) > 0.5)
        core = self._core(hull_b)

        t_labels = fp.resize_labels(target.labels, (h, w))
        source = target.source
        if target.is_semantic:
            inter = self._interior(t_labels, core)
            # Reliability is the PARSER's own coverage of the hull, measured before the
            # core rescue: counting rescued pixels would make an all-background parse
            # look trustworthy.
            cover = self._cover(self._interior(t_labels, None), hull_b)
            if hull_b is not None and cover < self.min_cover:
                # The parser found (almost) no face where the landmarks say one is:
                # fall back to the hull rather than swap on a garbage parse.
                inter, source = hull_b.copy(), 'hull-fallback'
        else:
            inter = t_labels > 0
            cover = self._cover(inter, hull_b)

        swap_used = False
        if swap is not None and swap.is_semantic:
            s_labels = fp.resize_labels(swap.labels, (h, w))
            if self._cover(self._interior(s_labels, None), hull_b) >= self.min_cover:
                s_inter = self._interior(s_labels, core)
                inter = inter & s_inter
                swap_used = True

        m = inter if hull_b is None else (inter & hull_b)
        if self.keep_largest and m.any():
            m = self._largest(m)
        interior = m.copy()

        alpha0 = m.astype(np.float32)
        if base_alpha is not None:
            alpha0 = alpha0 * mask2d(base_alpha, (h, w))
        alpha = self.asymmetric_inward(alpha0, yaw, feather)
        return ComposeResult(alpha=alpha, interior=interior, source=source,
                             parse_cover=cover, swap_used=swap_used,
                             stats={'yaw': float(yaw), 'feather': float(feather)})


# --------------------------------------------------------------------------- #
# Debug channels for the UI preview API
# --------------------------------------------------------------------------- #
_debug_lock = threading.Lock()
_debug_latest: dict = {}


def publish_debug(channels: dict) -> None:
    """Store the latest face's debug channels (uint8 images). Last writer wins."""
    with _debug_lock:
        _debug_latest.clear()
        _debug_latest.update(channels)


def take_debug() -> dict:
    """Pop the stored channels ({name: uint8 image}); empty when nothing was published."""
    with _debug_lock:
        out = dict(_debug_latest)
        _debug_latest.clear()
    return out


def debug_channels(crop_bgr, parse: fp.ParseResult, result: ComposeResult) -> dict:
    """``parsing_overlay`` (class colours on the crop) and ``final_alpha_mask`` (0..255)."""
    return {
        'parsing_overlay': fp.overlay(crop_bgr, parse.labels),
        'final_alpha_mask': np.clip(result.alpha * 255.0 + 0.5, 0, 255).astype(np.uint8),
    }
