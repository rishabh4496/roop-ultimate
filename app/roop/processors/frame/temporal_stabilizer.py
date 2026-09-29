"""EMA filter on the 5-point alignment matrix M, per face track.

    pred  = level + trend
    level <- s * pred + (1 - s) * M                     s = smoothing_factor
    trend <- b * (level - level_prev) + (1 - b) * trend b = trend_factor

An exponential moving average with Holt's trend term (double exponential
smoothing). trend_factor=0 is the plain EMA the brief names; it is not the
default because a plain EMA LAGS a moving head, measured below.

Entrywise EMA is safe for these matrices: estimate_norm fits a SIMILARITY, whose
2x2 part is [[a, -b], [b, a]], and any weighted mean of such matrices has the
same form -- the smoothed M is still a rotation+scale+shift, never a shear.

Where it must be applied: to the M that ALIGNS the crop, and the same M then
pastes it back (OnnxSpecSwapper.align -> post_process). Smoothing only the
paste matrix would put a face rendered from the jittery crop at the smoothed
place, i.e. off the head by exactly the jitter it was meant to hide.

Two properties a plain EMA lacks, both needed in this pipeline:

1. Bounded lag. At s = 0.9 a steadily moving head is followed ~9 frames late,
   which reads as the face sliding off the head ("pose smearing"). Two parts:
   the trend term predicts constant motion (zero steady-state lag), and the
   update weight rises with how far M sits from that prediction -- the largest
   frame-space displacement of the crop's corners, in face widths: below
   `snap_threshold` it ramps quadratically from (1 - s) toward 1, at or above
   it the state jumps to M and the trend is dropped.

   MEASURED 2026-09-29 (synthetic, 175 px face, 1.5 px landmark noise, mean
   corner distance from the noise-free placement, frames 20-150):

                          raw    EMA only (b=0)   default (b=0.2, thr 0.06)
       still head         3.31        1.35              1.80
       pan 4 px/frame     3.31        4.87              1.80
       pan 8 px/frame     3.31        4.57              1.82
       accelerating       3.31        4.46              2.56

   The plain EMA is worse than no filter at all on every moving head; the
   default is better than raw on all four. Frame-to-frame wobble on the still
   head: 4.44 -> 0.64 px (-86%). Larger thresholds smooth a still head a little
   more and lag an accelerating one much more (thr 0.12: 1.69 / 3.54).

2. Order. ProcessMgr deals frames to workers round-robin (frame i -> worker
   i % N; memory: frame-dispatch-is-round-robin), so "the previous update for
   this track" is often NOT the previous frame. Pass `frame_index`: a track
   whose last update is not 1..max_gap frames earlier is reset, never blended
   with a frame from half a second away. Without `frame_index`, calls are
   assumed sequential.

The repo's existing kps smoothing (roop/one_euro.py KpsStabilizer) filters the
landmarks before alignment instead; the two should not both be on.

A torch M is accepted and returned on its device; the arithmetic on six
numbers is done on the host (the snap decision is a host branch regardless).
"""
from __future__ import annotations

import threading
from typing import Dict, Hashable, Optional

import numpy as np


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


class TemporalMatrixStabilizer:
    def __init__(self, smoothing_factor: float = 0.9, snap_threshold: float = 0.06,
                 crop_size: int = 256, max_gap: int = 1, trend_factor: float = 0.2) -> None:
        if not 0.0 <= smoothing_factor < 1.0:
            raise ValueError(f"smoothing_factor must be in [0, 1), got {smoothing_factor}")
        if not 0.0 <= trend_factor <= 1.0:
            raise ValueError(f"trend_factor must be in [0, 1], got {trend_factor}")
        self.trend_factor = float(trend_factor)
        if snap_threshold <= 0:
            raise ValueError("snap_threshold must be > 0")
        self.smoothing_factor = float(smoothing_factor)
        self.snap_threshold = float(snap_threshold)
        self.crop_size = int(crop_size)
        self.max_gap = int(max_gap)
        self._state: Dict[Hashable, np.ndarray] = {}
        self._trend: Dict[Hashable, np.ndarray] = {}
        self._last_index: Dict[Hashable, Optional[int]] = {}
        self._lock = threading.Lock()
        # counters, so a caller can prove the filter ran and how
        self.stats = {"updates": 0, "resets": 0, "snaps": 0}

    # -- geometry ----------------------------------------------------------------

    def _corners_in_frame(self, M: np.ndarray) -> np.ndarray:
        """Frame-space positions of the crop's four corners under M (frame->crop)."""
        s = float(self.crop_size)
        c = np.array([[0, 0], [s, 0], [0, s], [s, s]], dtype=np.float64)
        A, t = M[:, :2], M[:, 2]
        return (c - t) @ np.linalg.inv(A).T

    def deviation(self, M_a: np.ndarray, M_b: np.ndarray) -> float:
        """Max corner displacement between two alignments, in face widths."""
        pa, pb = self._corners_in_frame(M_a), self._corners_in_frame(M_b)
        face = np.linalg.norm(pa[1] - pa[0]) or 1.0
        return float(np.max(np.linalg.norm(pa - pb, axis=1)) / face)

    # -- filter ------------------------------------------------------------------

    def update(self, M, track_id: Hashable = 0, frame_index: Optional[int] = None):
        """Feed this frame's raw M for `track_id`; return the smoothed M."""
        device = None
        if _is_torch(M):
            device, dtype = M.device, M.dtype
            M = M.detach().cpu().numpy()
        raw = np.asarray(M, dtype=np.float64).reshape(2, 3)
        if not np.isfinite(raw).all() or abs(np.linalg.det(raw[:, :2])) < 1e-12:
            raise ValueError("M must be a finite, invertible 2x3 affine")

        with self._lock:
            self.stats["updates"] += 1
            state = self._state.get(track_id)
            last = self._last_index.get(track_id)
            if state is not None and frame_index is not None and last is not None:
                if not 1 <= frame_index - last <= self.max_gap:
                    state = None                      # gap or out of order
            trend = self._trend.get(track_id)
            if state is None:
                if track_id in self._state:
                    self.stats["resets"] += 1
                out, trend = raw, np.zeros_like(raw)
            else:
                pred = state + trend
                dev = self.deviation(pred, raw)
                if dev >= self.snap_threshold:
                    self.stats["snaps"] += 1
                    out, trend = raw, np.zeros_like(raw)
                else:
                    s, b = self.smoothing_factor, self.trend_factor
                    alpha = (1.0 - s) + s * (dev / self.snap_threshold) ** 2
                    out = (1.0 - alpha) * pred + alpha * raw
                    trend = b * (out - state) + (1.0 - b) * trend
            self._state[track_id] = out
            self._trend[track_id] = trend
            self._last_index[track_id] = frame_index

        result = out.astype(np.float32)
        if device is not None:
            import torch
            return torch.as_tensor(result, device=device, dtype=dtype)
        return result

    def reset(self, track_id: Optional[Hashable] = None) -> None:
        with self._lock:
            if track_id is None:
                self._state.clear()
                self._trend.clear()
                self._last_index.clear()
            else:
                self._state.pop(track_id, None)
                self._trend.pop(track_id, None)
                self._last_index.pop(track_id, None)
