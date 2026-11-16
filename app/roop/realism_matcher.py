"""Two realism ops that sit between the enhancer and the occlusion mask / multi-band paste:

    Swapper -> Enhancer -> temporal blend -> [velocity motion blur] -> merger post-ops -> [skin grain transfer]
            -> BiSeNet + occlusion mask -> multi-band paste

``velocity_blur``  A pin-sharp face on a head that is travelling is the loudest tell in a moving shot. The existing
                   ``merger_motion_blur`` infers the smear axis from the plate crop's structure tensor; this one reads
                   the head's own VELOCITY from the ``TrajectoryPlan`` (the smoothed Stage 2 landmarks), so the blur
                   follows the motion the pre-pass actually measured, not a texture statistic. When the head moves
                   faster than ``SPEED_THRESHOLD_PX`` (3 px/frame) the swapped crop is convolved with a linear kernel
                   whose length is the displacement during the exposure (``SHUTTER`` = 0.5 of a frame, a 180 degree
                   shutter) and whose direction is the velocity, both carried into crop space through the alignment
                   matrix. The kernel fades in over the next half-threshold so the sharpness does not pop at 3.0.
                   It stands down when ``merger_motion_blur`` is on, so the two never stack.

``grain_transfer`` The swap comes back cleaner than the footage under it on some plates. The high-frequency luminance
                   of the plate's own ERODED SKIN (``L - GaussianBlur(L)``, the sensor grain plus pore-scale texture) is
                   lifted off the aligned plate crop and added to the swapped skin interior at ``grain_match_strength``
                   (default 0.35). It is the plate's own noise field, so the added grain keeps the plate's spectrum
                   (PSD) rather than a white-noise approximation; where the eroded skin does not cover the interior
                   (eyes, lips) the field is filled from rolled copies of itself so every interior pixel carries the
                   same spectrum.

                   It adds only the DEFICIT. A swap that already carries as much noise as the plate (variances add:
                   ``sqrt(max(s_plate^2 - s_swap^2, 0))``) gets nothing, exactly like ``roop.grain_match`` at the paste.
                   Adding the plate's full grain on top of a restorer's own would make the face NOISIER than the
                   footage. Measured on every clip so far the swap sits above a delivered plate's floor (AGENTS.md /
                   ``roop/grain_match.py``), so on those this op is a counted no-op.

Both are bit-identical no-ops at strength 0 and say what they did (``info`` / ``summary_lines``).
"""
from __future__ import annotations

import math
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from roop.degrade import swallowed as _swallowed
from roop.grain_match import MIN_PIXELS, _luma, deficit_sigma, noise_sigma

SPEED_THRESHOLD_PX = 3.0          # frame-space px/frame above which the swapped crop is blurred
RAMP = 0.5                        # the kernel fades in over [T, T * (1 + RAMP)]
SHUTTER = 0.5                     # fraction of a frame the shutter is open (180 degrees)
MAX_LEN_FRAC = 0.06               # longest kernel as a share of the crop width
MAX_TRAVEL_FACE = 0.6             # a per-frame jump bigger than this many face widths is a cut, not a move
GRAIN_SIGMA = 1.0                 # Gaussian of the high-pass: pore scale and finer
_FILL_SHIFTS = ((1, 3), (-2, 5), (3, -4), (-5, -2))     # (rows/den, cols/den) offsets for the fill copies

# --- per-render counters (a stage that returns its input without saying so looks identical to one that never ran) ---
_lock = threading.Lock()
_stats: Dict[str, float] = {}


def reset() -> None:
    with _lock:
        _stats.clear()


def _bump(name: str, n: float = 1) -> None:
    with _lock:
        _stats[name] = _stats.get(name, 0) + n


def summary_lines() -> List[str]:
    with _lock:
        s = dict(_stats)
    out: List[str] = []
    vb = s.get("blur_faces", 0)
    if vb:
        line = ("[Realism] velocity blur: %d/%d face(s) blurred, %d below %.1f px/frame, %d no plan/velocity, "
                "%d deferred to merger_motion_blur" % (s.get("blur_applied", 0), vb, s.get("blur_slow", 0),
                                                       SPEED_THRESHOLD_PX, s.get("blur_no_velocity", 0),
                                                       s.get("blur_deferred", 0)))
        if s.get("blur_applied", 0):
            line += "; mean kernel %.1f px, max %.1f px" % (s.get("blur_len_sum", 0.0) / s["blur_applied"],
                                                          s.get("blur_len_max", 0.0))
        out.append(line)
    gf = s.get("grain_faces", 0)
    if gf:
        line = ("[Realism] skin grain transfer: %d/%d face(s) got grain, %d already at least as noisy as the plate, "
                "%d without a usable skin region" % (s.get("grain_applied", 0), gf, s.get("grain_no_deficit", 0),
                                                     s.get("grain_no_skin", 0)))
        if s.get("grain_applied", 0):
            line += "; added sigma mean %.2f, max %.2f (8-bit levels)" % (
                s.get("grain_add_sum", 0.0) / s["grain_applied"], s.get("grain_add_max", 0.0))
        out.append(line)
    return out


# ═══════════════════════════════ velocity from the trajectory plan ═══════════════════════════════
def _row_for(rows: np.ndarray, track_id: int) -> Optional[np.void]:
    if rows is None or not len(rows):
        return None
    hit = np.flatnonzero(rows["track"] == int(track_id))
    return rows[int(hit[0])] if len(hit) else None


def landmark_velocity(prev_kps, next_kps, frames: float = 1.0) -> Tuple[float, float]:
    """Centroid displacement of two 5-point landmark sets per frame, ``(vx, vy)`` in frame pixels."""
    a = np.asarray(prev_kps, np.float32).reshape(-1, 2).mean(axis=0)
    b = np.asarray(next_kps, np.float32).reshape(-1, 2).mean(axis=0)
    d = (b - a) / max(float(frames), 1e-6)
    return float(d[0]), float(d[1])


def plan_velocity(plan: Any, track_id: Any, frame: int, absolute: bool = False) -> Optional[Tuple[float, float]]:
    """Head velocity ``(vx, vy)`` in frame px/frame for ``track_id`` at ``frame``, from the plan's smoothed landmarks.

    A central difference when both neighbours hold the track (the exposure is centred on the frame), else the
    one-sided difference. ``None`` when the plan holds no neighbour, or when the step is a jump (more than
    ``MAX_TRAVEL_FACE`` face widths: a shot cut, a re-association), which is not a velocity.
    """
    if plan is None or track_id is None or frame is None:
        return None
    try:
        cur = _row_for(plan.at(int(frame), absolute=absolute), track_id)
        if cur is None:
            return None
        prev = _row_for(plan.at(int(frame) - 1, absolute=absolute), track_id) if int(frame) > 0 else None
        nxt = _row_for(plan.at(int(frame) + 1, absolute=absolute), track_id)
        if prev is not None and nxt is not None:
            v = landmark_velocity(prev["kps"], nxt["kps"], 2.0)
            step_from, step_to = prev["kps"], nxt["kps"]
        elif nxt is not None:
            v = landmark_velocity(cur["kps"], nxt["kps"], 1.0)
            step_from, step_to = cur["kps"], nxt["kps"]
        elif prev is not None:
            v = landmark_velocity(prev["kps"], cur["kps"], 1.0)
            step_from, step_to = prev["kps"], cur["kps"]
        else:
            return None
        k = np.asarray(cur["kps"], np.float32)
        face = max(float(np.ptp(k[:, 0])), float(np.ptp(k[:, 1])), 1.0)
        jump = float(np.linalg.norm(np.asarray(step_to, np.float32).mean(0) - np.asarray(step_from, np.float32).mean(0)))
        if jump > MAX_TRAVEL_FACE * face * (2.0 if (prev is not None and nxt is not None) else 1.0):
            return None
        if not (math.isfinite(v[0]) and math.isfinite(v[1])):
            return None
        return v
    except Exception as exc:                                   # a velocity is a hint; never cost the frame
        _swallowed("roop/realism_matcher.py:plan_velocity", exc, "no velocity for this face; no motion blur applied")
        return None


# ═══════════════════════════════════════ velocity motion blur ═══════════════════════════════════════
def _line_kernel(length: float, angle: float, mix: float) -> np.ndarray:
    """Normalised line kernel of ``length`` px along ``angle``, blended with the identity by ``mix`` (0..1)."""
    half = max(1, int(math.ceil(length / 2.0)))
    size = 2 * half + 1
    ss = 8                                                    # supersample the line, then box it down
    big = np.zeros((size * ss, size * ss), np.float32)
    c = (size * ss) // 2
    dx, dy = math.cos(angle) * length / 2.0 * ss, math.sin(angle) * length / 2.0 * ss
    cv2.line(big, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0,
             max(1, ss // 2))
    k = cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA)
    s = float(k.sum())
    if s <= 0.0:
        k = np.zeros((size, size), np.float32)
        k[half, half] = 1.0
    else:
        k /= s
    ident = np.zeros_like(k)
    ident[half, half] = 1.0
    return ((1.0 - mix) * ident + mix * k).astype(np.float32)


def velocity_blur(crop: np.ndarray, velocity: Optional[Tuple[float, float]], M: Any, *, strength: float = 1.0,
                  plate_width: Optional[int] = None, defer_to_plate_blur: bool = False
                  ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Blur ``crop`` along the head's measured velocity. Returns ``(crop, info)``; the input object comes back
    untouched whenever nothing was applied (``info['reason']`` says why).

    ``velocity`` is ``(vx, vy)`` in FRAME px/frame; ``M`` the 2x3 frame -> aligned-crop matrix whose linear part
    carries it into crop space; ``plate_width`` the width of the aligned plate crop ``M`` maps to (the swapped crop
    may be larger, e.g. a 512 enhancement of a 256 swap crop).
    """
    info: Dict[str, Any] = {"applied": False, "reason": None, "speed": None, "length": 0.0}
    _bump("blur_faces")
    if strength <= 0.0:
        info["reason"] = "strength 0"
        return crop, info
    if defer_to_plate_blur:
        _bump("blur_deferred")
        info["reason"] = "merger_motion_blur is on"
        return crop, info
    if velocity is None or M is None:
        _bump("blur_no_velocity")
        info["reason"] = "no velocity (no plan, no neighbour frame, or a jump)"
        return crop, info
    speed = math.hypot(velocity[0], velocity[1])
    info["speed"] = speed
    if speed <= SPEED_THRESHOLD_PX:
        _bump("blur_slow")
        info["reason"] = "below threshold"
        return crop, info

    A = np.asarray(M, np.float64).reshape(2, 3)[:, :2]
    scale = (crop.shape[1] / float(plate_width)) if plate_width else 1.0
    v_crop = A @ np.asarray(velocity, np.float64) * scale
    speed_crop = float(np.hypot(v_crop[0], v_crop[1]))
    length = min(speed_crop * SHUTTER * float(strength), MAX_LEN_FRAC * crop.shape[1])
    if length < 1.0:
        _bump("blur_slow")
        info["reason"] = "kernel shorter than a pixel"
        return crop, info
    mix = float(np.clip((speed - SPEED_THRESHOLD_PX) / (SPEED_THRESHOLD_PX * RAMP), 0.0, 1.0))
    kernel = _line_kernel(length, math.atan2(v_crop[1], v_crop[0]), mix)
    out = cv2.filter2D(crop, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    info.update(applied=True, length=float(length), reason=None)
    _bump("blur_applied")
    _bump("blur_len_sum", length)
    with _lock:
        _stats["blur_len_max"] = max(_stats.get("blur_len_max", 0.0), float(length))
    return out, info


# ═══════════════════════════════════════ skin grain transfer ═══════════════════════════════════════
def _fill_field(field: np.ndarray, support: np.ndarray, want: np.ndarray) -> np.ndarray:
    """``field`` where ``support``; elsewhere inside ``want`` the field rolled in from other places, so the fill has
    the same spectrum. Deterministic. Pixels no roll reaches stay 0."""
    out = field * support
    have = support.astype(bool).copy()
    h, w = field.shape
    for ry, rx in _FILL_SHIFTS:
        missing = want & ~have
        if not missing.any():
            break
        dy, dx = (h * ry) // 7, (w * rx) // 7
        f2 = np.roll(field, (dy, dx), axis=(0, 1))
        s2 = np.roll(support.astype(bool), (dy, dx), axis=(0, 1))
        take = missing & s2
        out[take] = f2[take]
        have |= take
    return out


def skin_grain_field(plate: np.ndarray, skin: np.ndarray, *, sigma: float = GRAIN_SIGMA
                     ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """``(field, core, interior)``: the plate's high-pass luminance over its eroded skin (zero-mean, filled over the
    interior), the eroded skin the noise was read from, and the soft interior the grain is added through. All
    ``None`` when the skin region is too small to read."""
    sk = skin if skin.ndim == 2 else skin[:, :, 0]
    skin_b = sk > 0.5
    if int(skin_b.sum()) < MIN_PIXELS:
        return None, None, None
    h, w = skin_b.shape
    r = max(2, int(round(0.012 * max(h, w))))
    core = cv2.erode(skin_b.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    core_b = core.astype(bool)
    if int(core_b.sum()) < MIN_PIXELS:
        return None, None, None
    luma = _luma(plate)
    hp = luma - cv2.GaussianBlur(luma, (0, 0), float(sigma))
    hp = hp - float(hp[core_b].mean())                        # zero-mean over the skin it came from
    interior_b = cv2.erode(skin_b.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    field = _fill_field(hp.astype(np.float32), core_b, interior_b)
    interior = cv2.GaussianBlur(interior_b.astype(np.float32), (0, 0), max(1.0, 0.006 * max(h, w)))
    return field, core_b, interior


def grain_transfer(swapped: np.ndarray, plate: np.ndarray, skin: Optional[np.ndarray], *, strength: float = 0.35,
                   max_sigma: float = 12.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Add the plate's own skin grain to the swapped crop's skin interior. See the module docstring.

    ``swapped`` and ``plate`` are BGR crops in the same face-template space (``plate`` is resized if its shape
    differs); ``skin`` is the 0..1 skin mask of the PLATE crop. Returns ``(out, info)``; ``out is swapped`` whenever
    nothing was added.

    When the enhancer returns a larger crop than the plate (a 512 restoration of a 256 aligned crop) the plate is
    upsampled to match, so its grain is smoother than a native 512 capture would be: the measured plate noise, and
    with it the deficit, comes out LOWER. That errs toward adding less, never more.
    """
    info: Dict[str, Any] = {"applied": False, "reason": None, "sigma_plate": None, "sigma_swap": None, "added": 0.0}
    _bump("grain_faces")
    if strength <= 0.0:
        info["reason"] = "strength 0"
        return swapped, info
    if skin is None or plate is None:
        _bump("grain_no_skin")
        info["reason"] = "no skin mask"
        return swapped, info
    h, w = swapped.shape[:2]
    if plate.shape[:2] != (h, w):
        plate = cv2.resize(plate, (w, h), interpolation=cv2.INTER_AREA)
    if skin.shape[:2] != (h, w):
        skin = cv2.resize(skin.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    field, core, interior = skin_grain_field(plate, skin)
    if field is None:
        _bump("grain_no_skin")
        info["reason"] = "skin region too small"
        return swapped, info

    s_plate = noise_sigma(plate, core)
    inside = interior > 0.6
    s_swap = noise_sigma(swapped, inside)
    info["sigma_plate"], info["sigma_swap"] = s_plate, s_swap
    if s_plate is None or s_swap is None:
        _bump("grain_no_skin")
        info["reason"] = "too few pixels to measure"
        return swapped, info
    add = min(deficit_sigma(s_plate, s_swap, strength), max_sigma)
    if add < 0.15:
        _bump("grain_no_deficit")
        info["reason"] = "swap is already at least as noisy as the plate"
        return swapped, info
    covered = field[inside]
    std = float(covered.std())
    if std < 1e-6:
        _bump("grain_no_skin")
        info["reason"] = "plate skin carries no high-frequency signal"
        return swapped, info
    # Which way the rolled fill goes is fixed by the crop size; nothing here is random, so a rerun adds the same
    # grain (the pixel-noise-floor discipline in AGENTS.md needs a render to be comparable with itself).
    grain = (field * (add / std) * interior).astype(np.float32)
    out = swapped.astype(np.float32) + grain[:, :, None]
    out = np.clip(out + 0.5, 0, 255).astype(np.uint8) if swapped.dtype == np.uint8 else out.astype(swapped.dtype)
    info.update(applied=True, added=float(add))
    _bump("grain_applied")
    _bump("grain_add_sum", add)
    with _lock:
        _stats["grain_add_max"] = max(_stats.get("grain_add_max", 0.0), float(add))
    return out, info
