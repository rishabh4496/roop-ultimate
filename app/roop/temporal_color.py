"""Temporal smoothing of the LAB colour-transfer transform, one recurrence per tracked face.

NOT WIRED INTO THE RENDER, ON PURPOSE. Measured 2026-10-22 (``tests/probe_color_ema.py``, real bank swapped by the
production pipeline, 4 clips, 402 usable frames, alpha 0.15, skin-mean dE vs the plate):

    arm              pumping (dE/frame)   |error| vs plate (dE)
    none                  0.163                 1.247
    lct   (shipped)       0.141                 0.920
    lct + stats-EMA       0.154  (+9%)          0.930
    lct + transform-EMA   0.157  (+11%)         0.923
    rct   (default)       0.138                 0.992
    rct + stats-EMA       0.165  (+20%)         1.002
    rct + transform-EMA   0.165  (+20%)         0.996

The EMA makes the frame-to-frame wobble of the face's tone relative to the plate WORSE, and buys no accuracy. The
per-frame transform is not noise to be averaged: it is tracking the plate, which cancels the swap's own frame-to-frame
fluctuation; freezing it with a recurrence removes that compensation. A 1.12x plate brightness step did not separate
the arms either (the error never left the 2.3 dE just-noticeable difference, even with the transfer off), so the lag
cost is UNMEASURED, not zero. Earlier result with mask-isolated statistics, same conclusion: the 2026-10-19 note in
``tests/probe_skin_tone_transfer.py``. Kept so the next brief that asks for this finds the numbers, and the exact
split of ``_color_transfer_lct`` (bit-identical, tested) is there if a different statistic is ever worth smoothing.

``ColorTransferMixin`` recomputes its transform from scratch for every frame: the source (swapped crop) and target
(plate crop) statistics both move with the face, so the transform moves too. This module splits the transform into
its two parts so a per-track EMA can sit between them:

    stats_of(crop)  -> (mean[3], cov[3,3])       first / second order LAB statistics, sampled like production
    lct_affine / rct_affine(src stats, tgt stats) -> (A[3,3], b[3])      LAB' = A @ LAB + b
    apply_affine(crop, A, b) -> BGR uint8        the production clamp-and-truncate

Two smoothing modes, because they fail differently (measured in ``tests/probe_color_ema.py``):

    ``stats``      EMA of the source and target statistics, transform recomputed from the smoothed pair (the
                   literal "EMA of the LAB mean / std" reading). The SOURCE statistics lag the swap's own output,
                   so a sudden change in the swap's brightness is corrected with stale numbers.
    ``transform``  EMA of ``(A, b)`` itself, applied to the CURRENT pixels. The output still follows the swap
                   frame by frame; only the correction toward the plate is slow.

``lct_affine`` replicates ``procmgr_color._color_transfer_lct`` and ``tests/test_temporal_color.py`` compares the two
bit for bit, so a track's first frame is the shipped behaviour and not a near copy of it. ``rct_affine`` replicates
``_color_transfer_rct`` to within one level: production evaluates ``(x - mean) * scale + mean'`` in float64 and this
evaluates the same line as ``x * scale + (mean' - mean * scale)`` in float32, which the EMA needs (it blends the pair).

The recurrence must see a track's frames IN ORDER. It checks: a repeated or backwards frame index, or a gap larger
than ``MAX_GAP``, restarts the track instead of blending across it, and a jump larger than the cut thresholds
(a scene cut, or the plate's lighting really changing) restarts it too, rather than smearing the old grade over the
new one. ``clone_for_block`` / ``reset`` / ``warmup_frames`` / ``set_ordered`` follow the other temporal engines so
the parallel-stabilization blocks can own one each.
"""
from __future__ import annotations

import math
from threading import RLock
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

MAX_GAP = 3                     # frames; coasted faces skip a few indices
CUT_MEAN = 18.0                 # LAB levels (0..255 scale) a mean may jump before the track restarts
CUT_GAIN = 0.35                 # relative change of the transform's matrix that restarts the track
_STRIDE = 16                    # production's LCT sampling: every 16th pixel


def lab_f32(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def lct_stats(source_bgr: np.ndarray, target_bgr: np.ndarray):
    """The statistics ``_color_transfer_lct`` builds its transform from, plus the source LAB image (reused)."""
    s_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB)
    s_sub = s_lab.reshape(-1, 3)[::_STRIDE].astype(np.float32)
    t_sub_bgr = np.ascontiguousarray(target_bgr.reshape(-1, 3)[::_STRIDE]).reshape(-1, 1, 3)
    t_sub = cv2.cvtColor(t_sub_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    s_mean, t_mean = s_sub.mean(0), t_sub.mean(0)
    eps = np.eye(3, dtype=np.float32) * 1e-4
    s_c, t_c = s_sub - s_mean, t_sub - t_mean
    Cs = (s_c.T @ s_c) / max(1, s_c.shape[0] - 1) + eps
    Ct = (t_c.T @ t_c) / max(1, t_c.shape[0] - 1) + eps
    return s_lab, (s_mean, Cs), (t_mean, Ct)


def lct_affine(src, tgt) -> Tuple[np.ndarray, np.ndarray]:
    (s_mean, Cs), (t_mean, Ct) = src, tgt
    w_s, V_s = np.linalg.eigh(Cs)
    minv_s = (V_s * (1.0 / np.sqrt(np.clip(w_s, 1e-6, None)))) @ V_s.T
    w_t, V_t = np.linalg.eigh(Ct)
    msqrt_t = (V_t * np.sqrt(np.clip(w_t, 0, None))) @ V_t.T
    A = msqrt_t @ minv_s
    return A, t_mean - s_mean @ A.T


def rct_stats(source_bgr: np.ndarray, target_bgr: np.ndarray):
    s = lab_f32(source_bgr)
    t = lab_f32(target_bgr)
    s_mean, s_std = cv2.meanStdDev(s)
    t_mean, t_std = cv2.meanStdDev(t)
    return s, (s_mean.reshape(3), np.maximum(s_std.reshape(3), 1.0)), (t_mean.reshape(3), t_std.reshape(3))


def rct_affine(src, tgt) -> Tuple[np.ndarray, np.ndarray]:
    (s_mean, s_std), (t_mean, t_std) = src, tgt
    scale = t_std / s_std
    scale[1] = np.clip(scale[1], 0.80, 1.20)           # chroma std ratio bounded, exactly as production
    scale[2] = np.clip(scale[2], 0.80, 1.20)
    return np.diag(scale).astype(np.float32), (t_mean - s_mean * scale).astype(np.float32)


def apply_affine(source_bgr: np.ndarray, A: np.ndarray, b: np.ndarray, lab_image: Optional[np.ndarray] = None):
    """``LAB' = A @ LAB + b`` with production's clamp-then-truncate, back to BGR uint8."""
    s_f = (cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB) if lab_image is None else lab_image).astype(np.float32)
    out = cv2.transform(s_f, np.hstack([A, np.asarray(b, np.float32).reshape(3, 1)]).astype(np.float32))
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def _ema(old, new, alpha):
    return (1.0 - alpha) * old + alpha * new


class TemporalColorEMA:
    """Per-track recurrence for the colour-transfer transform. ``alpha`` is the weight of the NEW frame."""

    MODES = ("stats", "transform")

    def __init__(self, enabled: bool = False, alpha: float = 0.15, mode: str = "transform", cache_size: int = 256):
        self.enabled = bool(enabled)
        self.alpha = min(1.0, max(0.02, float(alpha)))
        self.mode = mode if mode in self.MODES else "transform"
        self.cache_size = max(16, int(cache_size))
        self.states: Dict[int, dict] = {}
        self.ordered = True
        self.restarts = 0
        self.blended = 0
        self._lock = RLock()

    # -- the engine contract the other temporal engines share ---------------------------------------------------
    def set_ordered(self, ordered) -> None:
        self.ordered = bool(ordered)

    def reset(self) -> None:
        with self._lock:
            self.states.clear()

    def clone_for_block(self) -> "TemporalColorEMA":
        clone = TemporalColorEMA(self.enabled, self.alpha, self.mode, self.cache_size)
        clone.ordered = self.ordered
        return clone

    def warmup_frames(self, eps: float = 0.01) -> int:
        return int(max(0, math.ceil(math.log(max(float(eps), 1e-4)) / math.log(max(1e-6, 1.0 - self.alpha)))))

    # -- the recurrence ------------------------------------------------------------------------------------------
    def _restart(self, key, entry, frame_index):
        entry["frame"] = frame_index
        self.states[key] = entry
        self.restarts += 1
        return entry

    def smooth(self, track_id, frame_index, kind: str, src, tgt, cuts: Optional[set[int]] = None):
        """Return the ``(A, b)`` to apply for this frame.

        ``kind`` ('lct' / 'rct') and the statistics ``src`` / ``tgt`` come from the ``*_stats`` helpers. With no
        track id, with the engine off or unordered, the answer is just this frame's own transform.
        Keyed strictly by persistent (segment_id, track_id, kind) - never raw box index.
        """
        affine = lct_affine if kind == "lct" else rct_affine
        if not self.enabled or track_id is None or not self.ordered:
            return affine(src, tgt)
        if isinstance(track_id, tuple) and len(track_id) == 2:
            try:
                key = (int(track_id[0]), int(track_id[1]), str(kind))
            except (TypeError, ValueError):
                return affine(src, tgt)
        else:
            try:
                import roop.globals
                seg = int(getattr(getattr(roop.globals, '_segment_spec', None), 'index', 0) or 0)
                key = (seg, int(track_id), str(kind))
            except (TypeError, ValueError):
                return affine(src, tgt)
        with self._lock:
            prev = self.states.get(key)
            A_now, b_now = affine(src, tgt)
            fresh = {"src": src, "tgt": tgt, "A": A_now, "b": b_now, "frame": frame_index}
            if cuts is not None and frame_index is not None and frame_index in cuts:
                self._restart(key, fresh, frame_index)
                return A_now, b_now
            if prev is None or frame_index is None or prev["frame"] is None:
                self._restart(key, fresh, frame_index)
                return A_now, b_now
            gap = frame_index - prev["frame"]
            if gap < 1 or gap > MAX_GAP:                     # repeated, backwards (out of order) or too far apart
                self._restart(key, fresh, frame_index)
                return A_now, b_now
            if self._is_cut(prev, src, tgt, A_now):
                self._restart(key, fresh, frame_index)
                return A_now, b_now
            a = 1.0 - (1.0 - self.alpha) ** gap              # a skipped index counts as the frames it skipped
            if self.mode == "stats":
                s_m, s_c = (_ema(p, n, a) for p, n in zip(prev["src"], src))
                t_m, t_c = (_ema(p, n, a) for p, n in zip(prev["tgt"], tgt))
                src_e, tgt_e = (s_m, s_c), (t_m, t_c)
                A, b = affine(src_e, tgt_e)
                prev.update(src=src_e, tgt=tgt_e, A=A, b=b)
            else:
                A, b = _ema(prev["A"], A_now, a), _ema(prev["b"], b_now, a)
                prev.update(A=A, b=b)
            prev["frame"] = frame_index
            self.blended += 1
            if len(self.states) > self.cache_size:
                self.states.pop(next(iter(self.states)))
            return A, b

    @staticmethod
    def _is_cut(prev, src, tgt, A_now) -> bool:
        """A change the EMA must not blend across: the plate's own mean jumped, or the transform changed shape."""
        if float(np.max(np.abs(np.asarray(tgt[0]) - np.asarray(prev["tgt"][0])))) > CUT_MEAN:
            return True
        ref = prev["A"]
        denom = max(float(np.linalg.norm(ref)), 1e-6)
        return float(np.linalg.norm(A_now - ref)) / denom > CUT_GAIN
