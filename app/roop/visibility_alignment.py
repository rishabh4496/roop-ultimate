"""Occlusion-visibility-weighted similarity alignment.

THE DEFECT THIS TARGETS
-----------------------
`face_analyser.canonicalize_face_alignment` builds the swap crop's matrix with
`face_util.estimate_norm`: an unweighted similarity fit of the detector's five
keypoints to the model template. When a hand, a microphone or another person's
cheek covers an eye or a mouth corner, the detector does not report "hidden": it
puts that keypoint SOMEWHERE (on the occluder, or on a plausible guess), and an
unweighted fit lets one wrong point out of five tilt, rescale and shift the
whole crop. The swap then wobbles behind the occluder.

WHAT THIS DOES
--------------
1. `visibility_weights`: each landmark gets ``w = clamp(1 - gain * P_occ, floor, 1)``
   from an occlusion probability map sampled AT the landmark.
2. `reconstruct_keypoints`: a primary keypoint with ``w < threshold`` is replaced by
   where the face's own last good observation puts it, after fitting that
   observation to the CURRENT frame with only the visible dense landmarks
   (`consensus_transform`: weighted similarity, a few IRLS rounds so a point that
   is still on the occluder is voted out). The head's rigid motion is read from
   what can be seen and applied to what cannot.
3. `solve_alignment`: weighted Umeyama of the (repaired) keypoints to the template.
4. `KinematicClamp`: when most of the face is covered, the new matrix may differ
   from the previous one only by a bounded scale / rotation / translation step, so
   a bad frame cannot jerk the crop.

Pure numpy, no model, no frame. The caller supplies the occlusion probability map
(`sample_probability`) and the previous observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

GAIN = 1.5
FLOOR = 0.05
OCCLUDED_BELOW = 0.50        # a primary keypoint below this is rebuilt
CLAMP_ABOVE_OCCLUSION = 0.40  # kinematic clamp engages past this occluded fraction


def sample_probability(prob: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear sample of a HxW probability map at (N,2) frame points; off-map = 0."""
    p = np.asarray(prob, np.float32)
    h, w = p.shape[:2]
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    x = np.clip(pts[:, 0], 0, w - 1.001)
    y = np.clip(pts[:, 1], 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    v = (p[y0, x0] * (1 - fx) * (1 - fy) + p[y0, x0 + 1] * fx * (1 - fy)
         + p[y0 + 1, x0] * (1 - fx) * fy + p[y0 + 1, x0 + 1] * fx * fy)
    off = (pts[:, 0] < 0) | (pts[:, 0] > w - 1) | (pts[:, 1] < 0) | (pts[:, 1] > h - 1)
    v = np.where(off, 0.0, v)
    return np.clip(v, 0.0, 1.0).astype(np.float32)


def visibility_weights(p_occ: np.ndarray, gain: float = GAIN, floor: float = FLOOR) -> np.ndarray:
    """``w = clamp(1 - gain * P_occ, floor, 1)``."""
    return np.clip(1.0 - gain * np.asarray(p_occ, np.float32), floor, 1.0).astype(np.float32)


def weighted_similarity(src: np.ndarray, dst: np.ndarray,
                        w: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Closed-form weighted Umeyama: the 2x3 similarity minimising
    ``sum_k w_k ||M src_k - dst_k||^2`` (reflection excluded). None if degenerate."""
    X = np.asarray(src, np.float64).reshape(-1, 2)
    Y = np.asarray(dst, np.float64).reshape(-1, 2)
    n = len(X)
    if n < 2 or len(Y) != n:
        return None
    wt = np.ones(n) if w is None else np.asarray(w, np.float64).reshape(-1)
    s = wt.sum()
    if not np.isfinite(s) or s <= 1e-9:
        return None
    wt = wt / s
    mx, my = (X * wt[:, None]).sum(0), (Y * wt[:, None]).sum(0)
    Xc, Yc = X - mx, Y - my
    var = float((wt * (Xc ** 2).sum(1)).sum())
    if var < 1e-9:
        return None
    cov = (Yc * wt[:, None]).T @ Xc
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1.0
    R = U @ S @ Vt
    c = float((D * np.diag(S)).sum() / var)
    t = my - c * R @ mx
    return np.hstack([c * R, t[:, None]]).astype(np.float32)


def apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    return (pts @ np.asarray(M, np.float32)[:, :2].T + np.asarray(M, np.float32)[:, 2]).astype(np.float32)


def decompose(M: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """(scale, rotation degrees, translation) of a similarity 2x3."""
    M = np.asarray(M, np.float64)
    return (float(np.hypot(M[0, 0], M[1, 0])),
            float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), M[:, 2].copy())


def compose_similarity(scale: float, deg: float, t: np.ndarray) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[scale * np.cos(a), -scale * np.sin(a), t[0]],
                     [scale * np.sin(a), scale * np.cos(a), t[1]]], np.float32)


def _lmeds_similarity(ref: np.ndarray, cur: np.ndarray, w: np.ndarray,
                      samples: int = 160) -> Tuple[Optional[np.ndarray], float]:
    """Least-median-of-squares similarity from two-point samples (deterministic).

    A starting fit from ALL points is already pulled toward whatever is stuck to
    the occluder, and IRLS started there stays pulled. A two-point sample drawn
    from the visible majority fits the head exactly; the sample whose WEIGHTED
    MEDIAN residual is smallest is that one. Returns ``(M, median residual)``.
    """
    n = len(ref)
    rng = np.random.default_rng(12345)
    ok = np.where(w > 0.2)[0]
    pool = ok if len(ok) >= 4 else np.arange(n)
    best, best_med = None, float('inf')
    for _ in range(samples):
        i, j = rng.choice(pool, 2, replace=False)
        if float(np.linalg.norm(ref[i] - ref[j])) < 1e-3:
            continue
        M = weighted_similarity(ref[[i, j]], cur[[i, j]])
        if M is None:
            continue
        r = np.linalg.norm(apply(M, ref[pool]) - cur[pool], axis=1)
        med = float(np.median(r))
        if med < best_med:
            best, best_med = M, med
    return best, best_med


def consensus_transform(ref_pts: np.ndarray, cur_pts: np.ndarray, w: np.ndarray,
                        iters: int = 4, keep: float = 2.5) -> Optional[np.ndarray]:
    """Weighted similarity ref -> cur that the VISIBLE majority agrees on.

    A robust (least-median) start, then IRLS: points whose residual is large
    against that median are down-weighted, so a landmark that sits on the occluder
    (visible by probability, wrong in position) is voted out of the fit.
    """
    ref = np.asarray(ref_pts, np.float32).reshape(-1, 2)
    cur = np.asarray(cur_pts, np.float32).reshape(-1, 2)
    wt = np.asarray(w, np.float32).reshape(-1).copy()
    M, med = _lmeds_similarity(ref, cur, wt)
    if M is None:
        return weighted_similarity(ref, cur, wt)
    scale = max(1.4826 * med, 0.5) * keep
    for _ in range(max(0, iters)):
        r = np.linalg.norm(apply(M, ref) - cur, axis=1)
        irls = np.where(r <= scale, 1.0, (scale / np.maximum(r, 1e-6)) ** 2)
        M2 = weighted_similarity(ref, cur, wt * irls)
        if M2 is None:
            break
        M = M2
    return M


@dataclass
class Observation:
    """A face's last good (unoccluded) observation, in frame pixels."""
    kps: np.ndarray                 # (5, 2)
    landmarks: np.ndarray           # (N, 2) dense (106 or 68), same layout as the current frame's


def reconstruct_keypoints(kps: np.ndarray, w_kps: np.ndarray,
                          landmarks: np.ndarray, w_landmarks: np.ndarray,
                          prior: Optional[Observation],
                          threshold: float = OCCLUDED_BELOW
                          ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Replace each primary keypoint with ``w < threshold`` by the prior
    observation carried onto the current frame. Returns ``(kps, weights, n_rebuilt)``.

    Without a usable prior (none, or a dense layout that does not match, or too
    few visible points to fit) nothing is rebuilt and the inputs come back as-is.
    """
    kps = np.asarray(kps, np.float32).reshape(5, 2).copy()
    w5 = np.asarray(w_kps, np.float32).reshape(5).copy()
    bad = np.where(w5 < threshold)[0]
    if len(bad) == 0 or prior is None:
        return kps, w5, 0
    lm = np.asarray(landmarks, np.float32).reshape(-1, 2)
    ref = np.asarray(prior.landmarks, np.float32).reshape(-1, 2)
    wl = np.asarray(w_landmarks, np.float32).reshape(-1)
    if lm.shape != ref.shape or wl.shape[0] != lm.shape[0]:
        return kps, w5, 0
    if int((wl >= threshold).sum()) < 8:
        return kps, w5, 0                    # not enough visible face to read the motion from
    T = consensus_transform(ref, lm, wl)
    if T is None:
        return kps, w5, 0
    carried = apply(T, np.asarray(prior.kps, np.float32).reshape(5, 2))
    kps[bad] = carried[bad]
    # A rebuilt point is a good estimate but not a measurement: it counts, at a
    # moderate weight, rather than at 1.0 or at its (near-zero) occluded weight.
    w5[bad] = 0.6
    return kps, w5, int(len(bad))


def solve_alignment(kps: np.ndarray, w_kps: np.ndarray, template: np.ndarray) -> Optional[np.ndarray]:
    """Weighted Umeyama of the five keypoints to the model template."""
    return weighted_similarity(np.asarray(kps, np.float32).reshape(5, 2),
                               np.asarray(template, np.float32).reshape(5, 2), w_kps)


class KinematicClamp:
    """Bound how far the alignment may move between frames while occluded.

    State is the previous accepted matrix and its per-frame velocity in
    (log-scale, rotation, translation). When the occluded fraction exceeds
    `engage_above`, the new matrix is pulled back to within `max_*` of the
    constant-velocity prediction; otherwise it passes through unchanged and just
    refreshes the state. The bounds are per frame and deliberately small: a head
    does not rescale 2% and turn 2 degrees between consecutive frames behind a
    hand unless the detector is wrong.
    """

    def __init__(self, engage_above: float = CLAMP_ABOVE_OCCLUSION,
                 max_log_scale: float = 0.02, max_deg: float = 1.5,
                 max_trans_frac: float = 0.01, crop: float = 256.0):
        self.engage_above = engage_above
        self.max_log_scale = max_log_scale
        self.max_deg = max_deg
        self.max_trans = max_trans_frac * crop
        self._prev: Optional[Tuple[float, float, np.ndarray]] = None
        self._vel = (0.0, 0.0, np.zeros(2))

    def reset(self) -> None:
        self._prev = None
        self._vel = (0.0, 0.0, np.zeros(2))

    def update(self, M: np.ndarray, occluded_fraction: float) -> np.ndarray:
        s, deg, t = decompose(M)
        ls = float(np.log(max(s, 1e-6)))
        if self._prev is None:
            self._prev = (ls, deg, t)
            return np.asarray(M, np.float32)
        pls, pdeg, pt = self._prev
        vls, vdeg, vt = self._vel
        if occluded_fraction > self.engage_above:
            pred = (pls + vls, pdeg + vdeg, pt + vt)
            ls = pred[0] + float(np.clip(ls - pred[0], -self.max_log_scale, self.max_log_scale))
            deg = pred[1] + float(np.clip(deg - pred[1], -self.max_deg, self.max_deg))
            d = t - pred[2]
            n = float(np.linalg.norm(d))
            t = pred[2] + (d * (self.max_trans / n) if n > self.max_trans else d)
        self._vel = (ls - pls, deg - pdeg, t - pt)
        self._prev = (ls, deg, t)
        return compose_similarity(float(np.exp(ls)), deg, t)
