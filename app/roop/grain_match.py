"""Sensor-noise matching for a pasted face.

A restorer / swapper output is cleaner at the finest scale than the footage it lands in, and the eye reads a
noise-free patch inside noisy footage as a sticker. This module measures the noise floor of the plate under the face
and adds ONLY the missing variance to the pasted face.

Relation to ``merger_grain_match`` (``procmgr_merger.apply_grain_match``, ON in the live config at 0.2): that op
measures the plate floor on the aligned crop BEFORE the paste and adds the full measured sigma, so (a) the inverse warp
then resamples its noise (a 512 crop lands on a ~230-300 px face: roughly halved) and (b) it ignores whatever noise the
swap already carries. This stage runs AFTER the warp, in frame space, where the viewer sees the noise, and measures the
paste as it now is, INCLUDING whatever ``merger_grain_match`` already put there, so the two compose instead of
stacking. It is a closer for the residual gap, not a second copy of the slider.

Why it is not ``face_swapper.inject_film_grain`` (the older reference implementation, not on the render path):
  * that one estimates sigma as std(I - Gaussian5x5(I)) over the whole background, which counts every hair strand and
    edge as "noise" (an overestimate on any real plate);
  * it ADDS the full sigma on top of whatever the swap already carries, so a face that already has sigma_p of its own
    ends up at sqrt(sigma_p^2 + sigma_t^2). Here the added standard deviation is sqrt(max(sigma_t^2 - sigma_p^2, 0)).

Estimator: Immerkaer's 3x3 Laplacian (noise gain exactly 6) with the median absolute deviation instead of the mean,
so edges and features are outliers rather than signal: sigma = 1.4826 * MAD(L) / 6. White noise of standard deviation s
measures s (tested), so the same estimator is used to size the grain and to grade the result.

The noise is luminance-only (the same value added to B, G and R) and independent per frame, as sensor noise is.
It is seeded from ``(frame_index, track_id, face centre)`` so a rerun of the same render adds the same grain: the
repo's pixel-noise-floor discipline (AGENTS.md) needs a render to be comparable with itself.
"""
from __future__ import annotations

import threading
import zlib
from typing import List, Optional, Tuple

import cv2
import numpy as np

from roop.env import env_float

# Immerkaer's operator: sum of squared taps = 36, so the response to white noise of std s has std 6 s.
_LAPLACIAN = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
_NOISE_GAIN = 6.0
_MAD_TO_STD = 1.4826
MIN_PIXELS = 400            # fewer than this under the matte and the estimate is not a statistic
_MAX_SAMPLES = 16384


def _luma(img: np.ndarray) -> np.ndarray:
    """BT.601 luma in float32. Not ``cvtColor`` on a uint8 cast: that quantises a float32 paste (+0.29 levels of
    rounding noise the integer-valued plate does not have), which is the quantity being measured."""
    if img.ndim == 2:
        return img.astype(np.float32)
    x = img.astype(np.float32, copy=False)
    return 0.114 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.299 * x[:, :, 2]


def noise_sigma(img: np.ndarray, where: Optional[np.ndarray] = None) -> Optional[float]:
    """White-noise-equivalent standard deviation (8-bit levels) of the finest-scale noise in ``img``.

    ``where`` is an optional boolean mask of the pixels to read (a 1-pixel inset is applied by the caller's mask, the
    filter itself reflects at the array border). None if fewer than ``MIN_PIXELS`` pixels qualify.
    """
    resp = cv2.filter2D(_luma(img), -1, _LAPLACIAN, borderType=cv2.BORDER_REFLECT)
    vals = resp[where] if where is not None else resp.reshape(-1)
    if vals.size < MIN_PIXELS:
        return None
    if vals.size > _MAX_SAMPLES:
        # A median converges long before 16k samples (relative error ~0.7%); np.median over the ~60k pixels of a
        # face was most of this function's cost (production's own grain op subsamples the same way).
        vals = vals[::vals.size // _MAX_SAMPLES]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return _MAD_TO_STD * mad / _NOISE_GAIN


def deficit_sigma(sigma_target: float, sigma_paste: float, strength: float = 1.0) -> float:
    """Standard deviation to ADD so that the pasted face's noise reaches the plate's (variances add)."""
    gap = max(float(sigma_target) ** 2 - float(sigma_paste) ** 2, 0.0)
    return float(np.sqrt(gap)) * float(np.clip(strength, 0.0, 1.0))


def strength() -> float:
    """``ROOP_GRAIN_MATCH``: 0 disables, 1 (default) closes the whole measured gap. Read per call: it is cheap."""
    return env_float("ROOP_GRAIN_MATCH", 1.0, 0.0, 1.0)


# --- per-render counters: "a feature that returns its input untouched without saying so is indistinguishable from one
# that never ran" (AGENTS.md). Printed in the swap audit next to the other stage counters.
_lock = threading.Lock()
_stats = {"faces": 0, "applied": 0, "no_deficit": 0, "too_small": 0, "sum_target": 0.0, "sum_paste": 0.0,
          "sum_added": 0.0, "max_added": 0.0}


def reset() -> None:
    with _lock:
        for k in _stats:
            _stats[k] = 0 if isinstance(_stats[k], int) else 0.0


def _record(info: dict) -> None:
    with _lock:
        _stats["faces"] += 1
        if info["applied"]:
            _stats["applied"] += 1
            _stats["sum_added"] += info["added"]
            _stats["max_added"] = max(_stats["max_added"], info["added"])
        elif info["sigma_target"] is None or info["sigma_paste"] is None:
            _stats["too_small"] += 1
        elif info["reason"] != "strength 0":
            _stats["no_deficit"] += 1
        if info["sigma_target"] is not None and info["sigma_paste"] is not None:
            _stats["sum_target"] += info["sigma_target"]
            _stats["sum_paste"] += info["sigma_paste"]


def summary_lines() -> List[str]:
    """Audit lines; [] when the stage never saw a face (disabled, or no swap reached the paste)."""
    with _lock:
        s = dict(_stats)
    if not s["faces"]:
        return []
    measured = s["faces"] - s["too_small"]
    line = (f"grain match: {s['applied']}/{s['faces']} face(s) got grain, {s['no_deficit']} already at least as "
            f"noisy as the plate, {s['too_small']} too small to measure")
    if measured:
        line += (f"; mean noise floor plate {s['sum_target'] / measured:.2f} vs swap {s['sum_paste'] / measured:.2f}"
                 f" (8-bit levels)")
    if s["applied"]:
        line += f"; added sigma mean {s['sum_added'] / s['applied']:.2f}, max {s['max_added']:.2f}"
    return [line]


def _seed(frame_index: Optional[int], track_id: Optional[int], centre: Tuple[int, int]) -> int:
    return zlib.crc32(repr((frame_index, track_id, int(centre[0]) // 4, int(centre[1]) // 4)).encode()) & 0xFFFFFFFF


def match(paste: np.ndarray, target: np.ndarray, matte: np.ndarray, *, strength: float = 1.0,
          frame_index: Optional[int] = None, track_id: Optional[int] = None,
          origin: Tuple[int, int] = (0, 0), max_sigma: float = 12.0) -> Tuple[np.ndarray, dict]:
    """``_match`` plus the per-render counters (``summary_lines``)."""
    out, info = _match(paste, target, matte, strength, frame_index, track_id, origin, max_sigma)
    _record(info)
    return out, info


def _match(paste, target, matte, strength, frame_index, track_id, origin, max_sigma):
    """Return ``(paste_with_grain, info)``.

    ``paste`` / ``target`` are the warped swap and the untouched plate over the same ROI (float32 or uint8 BGR),
    ``matte`` the 0..1 alpha over that ROI, ``origin`` the ROI's top-left in the frame (seed only). The grain is
    weighted by the matte, so it fades out through the feather instead of stopping at the ellipse edge.
    ``info`` always names what happened (``applied`` False carries a ``reason``): a feature that returns the input
    untouched without saying so is indistinguishable from one that never ran.
    """
    info = {"applied": False, "reason": None, "sigma_target": None, "sigma_paste": None, "added": 0.0}
    if strength <= 0.0:
        info["reason"] = "strength 0"
        return paste, info
    m = matte if matte.ndim == 2 else matte[:, :, 0]
    inside = m > 0.6
    # Both read inside the 3x3 filter's own support: erode the mask by 1 so edge responses of the matte are not read.
    inside = cv2.erode(inside.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    # Read the central 70% of the matte's box: that is skin, and the luma conversion over the whole ROI was most of
    # this stage's cost (it runs on every face of every frame, and does nothing at all on a clean plate).
    bx, by, bw, bh = cv2.boundingRect(inside.view(np.uint8))
    if bw > 0 and bh > 0:
        cx0, cy0 = bx + int(bw * 0.15), by + int(bh * 0.15)
        sl = (slice(cy0, by + bh - int(bh * 0.15)), slice(cx0, bx + bw - int(bw * 0.15)))
    else:
        sl = (slice(None), slice(None))
    s_t = noise_sigma(target[sl], inside[sl])
    s_p = noise_sigma(paste[sl], inside[sl])
    info["sigma_target"], info["sigma_paste"] = s_t, s_p
    if s_t is None or s_p is None:
        info["reason"] = f"fewer than {MIN_PIXELS} pixels under the matte"
        return paste, info
    add = min(deficit_sigma(s_t, s_p, strength), max_sigma)
    if add < 0.15:
        info["reason"] = "paste is already at least as noisy as the plate"
        return paste, info
    h, w = m.shape[:2]
    rng = np.random.default_rng(_seed(frame_index, track_id, (origin[0] + w // 2, origin[1] + h // 2)))
    grain = rng.standard_normal((h, w)).astype(np.float32) * np.float32(add)
    grain *= m.astype(np.float32)
    out = paste.astype(np.float32) + grain[:, :, None]
    info.update(applied=True, added=float(add))
    return out, info
