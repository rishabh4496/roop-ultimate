"""Expression & gaze fidelity: what a 128 px identity swap flattens, put back from the plate.

Two passes, both operating on the finished full-frame composite (after paste-back, in the slot
`apply_eyes_area` and the mouth restores use), both pure functions of ``(result, plate, kps)``:

``retain_iris_pupil``      keeps WHERE the target is looking.
    `restore_original_eyes` pastes the target's whole eye ellipse back, which returns the gaze but
    also the target's lids, lashes and iris COLOUR - half of what makes the swapped eye the
    source's. This one moves only the iris: it finds the iris in the plate and in the swap, slides
    the swapped iris (colour intact) to where the plate's is, erases it from where it was, and then
    matches the iris contrast and takes the sclera's low-frequency shading on top. Lids, lashes and eye colour stay the swap's. A blink is respected for free:
    with no iris visible in the plate there is nothing to move and the pass does nothing.

``restore_micro_expression``  keeps the creases the swapper averaged away.
    Nasolabial folds, inner brows and mouth corners carry the tension of an expression and are a
    few pixels wide, so they vanish first. Inside soft regions built from the five keypoints the
    pass transplants the plate's BAND-PASS luminance (a difference of Gaussians scaled to the
    face), leaving the swap's colour, tone and fine skin texture alone. It restores shading, not
    geometry: it does not warp the swapped face, because nothing here detects landmarks on the
    swapped output.

Neither is a general-purpose restore and neither is on by default: both default to 0, which is a
bit-exact no-op (the functions return the input array untouched).

Geometry is in units of the interocular distance, the same convention as ``apply_eyes_area``, so
a region tracks the face's size on screen without a per-clip retune. The eye opening ellipse here
is that function's (0.21 x 0.13).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

EYE_RX, EYE_RY = 0.21, 0.13          # eye opening ellipse, in interocular distances (apply_eyes_area's)
IRIS_R = 0.09                        # iris radius / interocular distance (11.7 mm iris, 63 mm IPD)
WIDE_GAP_RATIO = 0.22                # inner-lip gap / mouth width above which the mouth is "wide open"
# A difference-of-Gaussians transplant is not idempotent: swapping the plate's band into the swap
# brings back only ~53% of a 3 px crease's depth at d = 80 px (measured on the synthetic crease in
# tests/test_expression_fidelity.py), because the band overlaps its own complement. The operation is
# linear, so a fixed gain makes strength 1.0 mean "the crease is back". CALIBRATED ON A SYNTHETIC
# CREASE ONLY - on real footage the right gain is unmeasured, which is why the control defaults to 0.
MICRO_GAIN = 1.6


def _head_fade(yaw: float, pitch: float) -> float:
    """1 up to 25 degrees, 0 from 38 - the fade the eye and mouth restores use, for the same
    reason: past it the far eye is an edge and the plate's features land on the nose bridge."""
    a = max(abs(float(yaw or 0.0)), abs(float(pitch or 0.0)))
    return 1.0 if a <= 25.0 else max(0.0, min(1.0, (38.0 - a) / 13.0))


def _own(region, x0, y0, x1, y1, shape):
    """The ownership field of this face over the ROI (None when it has the frame to itself)."""
    if region is None:
        return None
    own = region.crop(x0, y0, x1, y1)
    if own is None:
        return None
    own = np.asarray(own, dtype=np.float32)
    if own.shape[:2] != shape:
        own = cv2.resize(own, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(own, 0.0, 1.0)


def _disk(shape, cx, cy, r, soft):
    """A soft disk: 1 inside r - soft, 0 outside r + soft, linear between."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return np.clip((r + soft - dist) / max(2.0 * soft, 1e-3), 0.0, 1.0)


def _ellipse(shape, cx, cy, rx, ry, soft):
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt(((xs - cx) / max(rx, 1e-3)) ** 2 + ((ys - cy) / max(ry, 1e-3)) ** 2)
    edge = soft / max(min(rx, ry), 1e-3)
    return np.clip((1.0 + edge - dist) / max(2.0 * edge, 1e-3), 0.0, 1.0)


# ── gaze ─────────────────────────────────────────────────────────────────────

def locate_iris(y: np.ndarray, search: np.ndarray, r_iris: float) -> Optional[Tuple[float, float, float]]:
    """(cx, cy, mass) of the iris in a luma ROI, or None when no iris is visible.

    The iris and pupil are the darkest compact thing inside the eye opening, so this is a
    darkness-weighted centroid over `search` (a mask a little smaller than the opening, so the
    upper lid's lash line is not mistaken for iris). Returns None when the opening has no contrast
    to speak of (a closed or blinking eye) or too little dark mass to be an iris at this size.
    """
    inside = search > 0.5
    if int(inside.sum()) < 12:
        return None
    yb = cv2.GaussianBlur(y, (0, 0), max(0.7, r_iris * 0.25))
    vals = yb[inside]
    ref = float(np.percentile(vals, 85))
    span = ref - float(np.percentile(vals, 3))
    if span < 18.0:
        return None
    # Squared darkness weights the dark core, which is what pins the centre to a pixel. The cut is 0.10, not
    # 0.25: a pale iris (luma ~150 against a sclera of ~235) has weight 0.38, squared 0.15, and under the old
    # cut only the pupil survived, the mass test failed, and the pass read a light-eyed face as a blink and
    # did nothing (found by a pale-iris regression test; linear weights admitted it but cost centroid accuracy).
    w = np.clip((ref - yb) / span, 0.0, 1.0) ** 2
    w = np.where((w > 0.10) & inside, w, 0.0)
    mass = float(w.sum())
    if mass < 0.30 * math.pi * r_iris * r_iris:
        return None
    ys, xs = np.mgrid[0:w.shape[0], 0:w.shape[1]].astype(np.float32)
    return float((w * xs).sum() / mass), float((w * ys).sum() / mass), mass


def retain_iris_pupil(result: np.ndarray, plate: np.ndarray, kps, strength: float,
                      yaw: float = 0.0, pitch: float = 0.0, region=None) -> np.ndarray:
    """Move each swapped iris to where the plate's is and take on its pupil and sclera shading.

    Writes into `result` and returns it. `strength` 0 returns the very same array, unread.
    """
    strength = max(0.0, min(1.0, float(strength or 0.0)))
    if strength <= 0.0 or result is None or plate is None or kps is None:
        return result
    if result.shape != plate.shape or result.ndim != 3 or result.shape[2] < 3:
        return result
    try:
        eyes = np.asarray(kps, dtype=np.float32).reshape(-1, 2)[:2]
        d = float(np.linalg.norm(eyes[1] - eyes[0]))
    except (TypeError, ValueError):
        return result
    fade = _head_fade(yaw, pitch)
    if d < 12.0 or fade <= 0.0:
        return result

    h, w = result.shape[:2]
    rx, ry, ri = d * EYE_RX, d * EYE_RY, d * IRIS_R
    pad = int(rx * 1.7) + 3
    for ex, ey in eyes:
        x0, x1 = max(0, int(ex) - pad), min(w, int(ex) + pad + 1)
        y0, y1 = max(0, int(ey) - pad), min(h, int(ey) + pad + 1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        cx, cy = float(ex - x0), float(ey - y0)
        res = result[y0:y1, x0:x1]
        pla = plate[y0:y1, x0:x1]
        shape = res.shape[:2]
        soft = max(1.0, ri * 0.35)

        opening = _ellipse(shape, cx, cy, rx, ry, soft)
        inner = _ellipse(shape, cx, cy, rx * 0.78, ry * 0.62, soft)      # where an iris can actually be
        ycc_p = cv2.cvtColor(pla, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        ycc_r = cv2.cvtColor(res, cv2.COLOR_BGR2YCrCb).astype(np.float32)

        c_p = locate_iris(ycc_p[:, :, 0], inner, ri)
        if c_p is None:
            continue                                    # no iris in the plate: a blink, or too small
        c_r = locate_iris(ycc_r[:, :, 0], inner, ri)
        px, py = c_p[0], c_p[1]
        rxx, ryy = (c_r[0], c_r[1]) if c_r is not None else (px, py)
        disp = math.hypot(px - rxx, py - ryy)
        ramp = float(np.clip((disp - 0.2 * ri) / (0.6 * ri), 0.0, 1.0))  # 0 when already together

        own = _own(region, x0, y0, x1, y1, shape)
        gate = strength * fade * (own if own is not None else 1.0)

        out = res.copy()
        if ramp > 0.0 and c_r is not None:
            # 1. erase the swapped iris from where it was (inpaint from the surrounding sclera)
            erase = (_disk(shape, rxx, ryy, ri * 1.25, soft) * inner)
            hard = (erase > 0.2).astype(np.uint8) * 255
            if hard.any():
                filled = cv2.inpaint(res, hard, 3, cv2.INPAINT_TELEA)
                a = (erase * ramp * gate)[:, :, None]
                out = np.clip(res.astype(np.float32) * (1 - a) + filled.astype(np.float32) * a + 0.5, 0, 255).astype(np.uint8)
            # 2. slide the swapped iris - colour and all - onto the plate's position
            shift = np.array([[1, 0, px - rxx], [0, 1, py - ryy]], np.float32)
            moved = cv2.warpAffine(res, shift, (shape[1], shape[0]), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
            land = (_disk(shape, px, py, ri * 1.3, soft) * inner * ramp * gate)[:, :, None]
            out = np.clip(out.astype(np.float32) * (1 - land) + moved.astype(np.float32) * land + 0.5,
                          0, 255).astype(np.uint8)

        yo = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
        # sclera brightness the two images disagree on (enhancer, grade): match the plate to it
        sclera = (opening > 0.5) & (_disk(shape, px, py, ri * 1.5, soft) < 0.05)
        if int(sclera.sum()) >= 8:
            gain = float(np.clip(np.median(yo[sclera]) / max(np.median(ycc_p[:, :, 0][sclera]), 1.0), 0.8, 1.25))
        else:
            gain = 1.0
        yp = ycc_p[:, :, 0] * gain
        # 3. the plate's pupil / catchlight structure inside the iris, luminance only
        # CONTRAST only. Blending the plate's luma into the iris re-brightened a dark swapped iris toward a
        # pale plate iris (real footage, 2026-10-28: washed-out irises, max change 255), and even importing
        # only the plate's deviation from its iris mean brought its pale ring in. So the swap keeps its own
        # iris and its own brightness: this only rescales the swap's deviation from ITS mean so the pupil /
        # limbal contrast matches the plate's, a mean-preserving edit (the weighted mean of dy is zero).
        core = _disk(shape, px, py, ri * 0.9, soft) * inner
        wsum = max(float(core.sum()), 1e-6)
        mean_p = float((yp * core).sum() / wsum)
        mean_o = float((yo * core).sum() / wsum)
        std_p = float(np.sqrt(((yp - mean_p) ** 2 * core).sum() / wsum))
        std_o = float(np.sqrt(((yo - mean_o) ** 2 * core).sum() / wsum))
        k = float(np.clip(std_p / max(std_o, 1.0), 0.6, 1.6))
        dy = 0.6 * gate * core * ((k - 1.0) * (yo - mean_o))
        # 4. the sclera's low-frequency shading (lid shadow, curvature), DC removed so brightness holds
        sig = max(1.0, ry * 0.55)
        shade_zone = opening * (1.0 - _disk(shape, px, py, ri * 1.3, soft))
        zsum = float(shade_zone.sum())
        if zsum > 1.0:
            delta = cv2.GaussianBlur(yp, (0, 0), sig) - cv2.GaussianBlur(yo + dy, (0, 0), sig)
            delta = delta - float((delta * shade_zone).sum() / zsum)
            dy = dy + 0.5 * gate * shade_zone * delta
        # A luma edit is the same delta on B, G and R: Y moves by exactly dy and Cr/Cb not at all,
        # and a pixel with dy == 0 is untouched (no colour-space round trip to perturb it).
        final = np.clip(out.astype(np.float32) + dy[:, :, None] + 0.5, 0, 255).astype(np.uint8)
        result[y0:y1, x0:x1] = final
    return result


# ── micro-expression ─────────────────────────────────────────────────────────

def micro_expression_regions(kps, shape_hw, roi) -> Optional[np.ndarray]:
    """Soft [0,1] mask over `roi` = (x0, y0, x1, y1) for the nasolabial folds, mouth corners and
    inner brows, built from the five keypoints in the face's own (u, v) frame so it follows roll."""
    k = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
    if k.shape[0] < 5 or not np.isfinite(k[:5]).all():
        return None
    el, er, nose, ml, mr = k[0], k[1], k[2], k[3], k[4]
    d = float(np.linalg.norm(er - el))
    if d < 12.0:
        return None
    u = (er - el) / d
    v = np.array([-u[1], u[0]], np.float32)             # +v is "down the face"
    x0, y0, x1, y1 = roi
    shape = (y1 - y0, x1 - x0)
    off = np.array([x0, y0], np.float32)
    m = np.zeros(shape, np.float32)

    def stamp(centre, rx, ry):
        c = centre - off
        m_i = _ellipse(shape, float(c[0]), float(c[1]), rx * d, ry * d, max(1.0, 0.03 * d))
        np.maximum(m, m_i, out=m)

    # The window must be WIDER than the crease it frames: a capsule as thin as the crease, once
    # blurred, never reaches 1 and only about half the crease comes back at full strength. It also has
    # to hold the DoG's bright halos either side of a dark crease: with a window of 0.11 d the halos
    # fell outside it and the region darkened by 6 levels (3.0 at 0.16 d, 1.8 at 0.20 d, measured on
    # the synthetic crease). Where the plate and the swap agree the transplanted delta is ~0, so a
    # generous window costs nothing.
    thick = max(1, int(round(0.18 * d)))
    for wing_sign, corner in ((-1.0, ml), (1.0, mr)):
        wing = nose + wing_sign * 0.20 * d * u + 0.05 * d * v
        bow = (wing + corner) * 0.5 + wing_sign * 0.045 * d * u
        line = np.zeros(shape, np.uint8)
        pts = np.round(np.array([wing - off, bow - off, corner - off], np.float32)).astype(np.int32)
        cv2.polylines(line, [pts.reshape(-1, 1, 2)], False, 255, thick, cv2.LINE_AA)
        np.maximum(m, cv2.GaussianBlur(line.astype(np.float32) / 255.0, (0, 0), max(0.8, 0.02 * d)), out=m)
        stamp(corner + wing_sign * 0.05 * d * u, 0.13, 0.10)                    # the corner itself
    for eye, inward in ((el, 1.0), (er, -1.0)):
        stamp(eye + inward * 0.22 * d * u - 0.30 * d * v, 0.13, 0.07)           # inner brow
    return m


def restore_micro_expression(result: np.ndarray, plate: np.ndarray, kps, strength: float,
                             yaw: float = 0.0, pitch: float = 0.0, region=None) -> np.ndarray:
    """Transplant the plate's band-pass luminance into the fold / brow / corner regions."""
    strength = max(0.0, min(1.0, float(strength or 0.0)))
    if strength <= 0.0 or result is None or plate is None or kps is None:
        return result
    if result.shape != plate.shape or result.ndim != 3 or result.shape[2] < 3:
        return result
    fade = _head_fade(yaw, pitch)
    if fade <= 0.0:
        return result
    try:
        k = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
        d = float(np.linalg.norm(k[1] - k[0]))
    except (TypeError, ValueError, IndexError):
        return result
    if d < 12.0 or not np.isfinite(k[:5]).all():
        return result
    h, w = result.shape[:2]
    pad = int(d * 0.9) + 4
    x0, x1 = max(0, int(k[:5, 0].min()) - pad), min(w, int(k[:5, 0].max()) + pad)
    y0, y1 = max(0, int(k[:5, 1].min()) - pad), min(h, int(k[:5, 1].max()) + pad)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return result
    mask = micro_expression_regions(k, result.shape[:2], (x0, y0, x1, y1))
    if mask is None or float(mask.max()) <= 1e-3:
        return result
    own = _own(region, x0, y0, x1, y1, mask.shape)
    if own is not None:
        mask = mask * own
    res = result[y0:y1, x0:x1]
    pla = plate[y0:y1, x0:x1]
    yr = cv2.cvtColor(res, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    yp = cv2.cvtColor(pla, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    fine, coarse = max(0.7, 0.020 * d), max(1.6, 0.060 * d)

    def band(img):
        return cv2.GaussianBlur(img, (0, 0), fine) - cv2.GaussianBlur(img, (0, 0), coarse)

    # Capped: a plate crease is a few levels deep, a plate specular or hair strand is not a fold.
    delta = np.clip(band(yp) - band(yr), -40.0 / MICRO_GAIN, 40.0 / MICRO_GAIN)
    dy = strength * fade * mask * (MICRO_GAIN * delta)
    # Same delta on B, G, R: luma moves by dy, chroma not at all, untouched pixels stay bit-exact.
    result[y0:y1, x0:x1] = np.clip(res.astype(np.float32) + dy[:, :, None] + 0.5, 0, 255).astype(np.uint8)
    return result
