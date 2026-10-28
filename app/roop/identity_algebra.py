"""Latent identity arithmetic on the ArcFace hypersphere.

Every embedding-driven swapper here consumes the buffalo_l / w600k_r50 identity
vector, L2-normalised onto S^511.  This module does two things in that space:

1. **Blending** several source identities::

       z_blend = normalize(sum_i w_i * z_i),   w_i >= 0, sum w_i = 1

   The normalisation is not cosmetic.  Two different people sit at cosine
   ~0.0-0.2 in this space, so a 50/50 mix of two unit vectors has length ~0.7;
   fed un-normalised, the swapper sees an identity 30% "quieter" than any face
   it was trained on.

2. **Directional attribute offsets** (age, gender, jawline/feature weight,
   expression)::

       z_final = normalize(z + t),   t = P_z(sum_k alpha_k * v_k)

   ``P_z`` projects onto the tangent plane at ``z``.  Without it, the part of
   the offset parallel to ``z`` is simply removed again by the normalisation,
   so the same dial moves different identities by different angles.  With it,
   ``cos(z_final, z) = 1 / sqrt(1 + |t|^2)`` exactly, which is what makes the
   identity guard below a closed form rather than a search.

The direction vectors are FITTED, not invented: ``fit_directions`` regresses
labels (genderage age / sex, 68-landmark geometry) on real embeddings, and
``tools/fit_identity_directions.py`` writes the result to
``roop/assets/identity_directions.npz`` together with the corpus statistics and
held-out scores.  Read those scores before trusting a direction -- ArcFace was
trained to be invariant to expression and largely to age, so some of these
directions are weak by construction.  A dial whose direction is not shipped is
reported as unavailable, never silently ignored.

Pure numpy; no model is loaded here.
"""
from __future__ import annotations

import bisect
import json
import math
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from roop.degrade import swallowed as _swallowed

EMBEDDING_DIM = 512
DIRECTION_NAMES = ("age", "gender", "jawline", "expression")
MAX_BLEND_SOURCES = 5
# How N weighted identities are combined.  "slerp" walks the great circle (exact
# Slerp for two sources, the weighted spherical mean for 3-5); "linear" is the
# older normalize(sum w_i z_i), same geodesic but a different speed along it.
BLEND_MODES = ("slerp", "linear")
DEFAULT_BLEND_MODE = "slerp"
# Interpolation of the identity between two keyframes, set on the keyframe the
# segment STARTS at: hold the vector until the next keyframe, walk the great
# circle at constant angular speed, or walk it through a CSS-style cubic bezier.
CURVE_HOLD, CURVE_LINEAR, CURVE_CUBIC_BEZIER = "hold", "linear", "cubic_bezier"
CURVES = (CURVE_HOLD, CURVE_LINEAR, CURVE_CUBIC_BEZIER)
DEFAULT_BEZIER = (0.42, 0.0, 0.58, 1.0)          # ease-in-out
MAX_KEYFRAMES = 64
# De-identification: the cosine to every reference identity stays at or below
# this.  Same-person w600k pairs sit well above it, strangers around 0.0-0.2.
DEFAULT_PRIVACY_COSINE = 0.30
AGE_LIMIT_YEARS = 30.0
# Unit dials (gender, jawline, expression) are in multiples of the corpus's own
# spread along that direction: +/-1 moves the identity by DIAL_SIGMAS standard
# deviations of how real faces vary along it -- i.e. still inside the population.
DIAL_SIGMAS = 2.0
# cos(z_final, z_source) floor.  The swapper's own output already lands at
# ~0.62-0.67 cosine to its source (RECODE_STATUS id_src tables); 0.80 on the
# INPUT vector keeps the requested edit well inside "the same person" while
# leaving room for a +/-30-year age move on the shipped directions.
DEFAULT_MIN_IDENTITY_COSINE = 0.80
DIRECTIONS_PATH = os.path.join(os.path.dirname(__file__), "assets",
                               "identity_directions.npz")
# Render validation (tests/identity_algebra_bench.py --sweep): a direction can
# PREDICT a label and still not be WRITABLE through the swapper. Measured
# 2026-09-26 on hyperswap: age and gender are non-monotone / null in the
# rendered face, jawline is monotone but weak, expression is null. A dial whose
# direction is marked unwritable is reported unavailable with the verdict.
RENDER_VALIDATION_PATH = os.path.join(os.path.dirname(__file__), "assets",
                                      "identity_render_validation.json")


# ── hypersphere primitives ────────────────────────────────────────────────────

def l2_normalize(vector: Any) -> Optional[np.ndarray]:
    """Unit float32 vector, or None for None / empty / non-finite / zero."""
    if vector is None:
        return None
    try:
        arr = np.asarray(vector, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return None
    return (arr / norm).astype(np.float32)


def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    """Clip negatives to 0 and rescale to sum 1.  All-zero -> uniform."""
    w = np.asarray([float(x) if np.isfinite(float(x)) else 0.0 for x in weights],
                   dtype=np.float64)
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if w.size == 0:
        return w
    if total <= 1e-12:
        return np.full(w.size, 1.0 / w.size)
    return w / total


def blend_embeddings(embeddings: Sequence[Any], weights: Sequence[float]) -> Optional[np.ndarray]:
    """``normalize(sum w_i z_i)`` over unit-normalised inputs.

    Inputs are normalised first, so a raw (norm ~20) ArcFace vector and a unit
    one contribute by weight alone.  Components that are None / invalid are
    dropped and the remaining weights renormalised.  Returns None when nothing
    valid remains, or when the weighted vectors cancel (antipodal inputs).
    """
    pairs = [(l2_normalize(e), float(w)) for e, w in zip(embeddings, weights)]
    pairs = [(e, w) for e, w in pairs if e is not None]
    if not pairs:
        return None
    w = normalize_weights([p[1] for p in pairs])
    stack = np.stack([p[0].astype(np.float64) for p in pairs])
    return l2_normalize((w[:, None] * stack).sum(axis=0))


def tangent_project(offset: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Component of ``offset`` orthogonal to unit ``anchor``."""
    offset = np.asarray(offset, dtype=np.float64).reshape(-1)
    anchor = np.asarray(anchor, dtype=np.float64).reshape(-1)
    return offset - float(offset @ anchor) * anchor


def max_tangent_norm(min_cosine: float) -> float:
    """Largest |t| with cos(normalize(z + t), z) >= min_cosine (t ⟂ z)."""
    c = float(min(max(min_cosine, 1e-6), 1.0))
    if c >= 1.0:
        return 0.0
    return math.tan(math.acos(c))


def gram_schmidt(vectors: Sequence[np.ndarray]) -> List[Optional[np.ndarray]]:
    """Orthonormalise in the given priority order.  A vector that collapses
    (already spanned by the earlier ones) comes back as None."""
    basis: List[np.ndarray] = []
    out: List[Optional[np.ndarray]] = []
    for v in vectors:
        if v is None:
            out.append(None)
            continue
        u = np.asarray(v, dtype=np.float64).reshape(-1).copy()
        for b in basis:
            u -= float(u @ b) * b
        n = float(np.linalg.norm(u))
        if n <= 1e-8:
            out.append(None)
            continue
        u /= n
        basis.append(u)
        out.append(u)
    return out


# ── spherical algebra: Slerp, spherical mean, de-identification ───────────────

def _perpendicular(a: np.ndarray) -> np.ndarray:
    """A deterministic unit vector orthogonal to unit ``a`` (the axis it is
    least aligned with, Gram-Schmidt'd).  Only used where the geodesic is
    otherwise undefined (antipodal endpoints)."""
    k = int(np.argmin(np.abs(a)))
    e = np.zeros_like(a)
    e[k] = 1.0
    p = e - float(e @ a) * a
    return p / float(np.linalg.norm(p))


def slerp(a: Any, b: Any, t: float, antipodal: str = "perpendicular") -> Optional[np.ndarray]:
    """Spherical linear interpolation ``sin((1-t)θ)/sinθ·a + sin(tθ)/sinθ·b``.

    Inputs are unit-normalised first; ``t`` is clipped to [0, 1].  Constant
    angular speed in ``t``: ``angle(slerp(t0), slerp(t1)) = |t1-t0|·θ``.  For
    antipodal endpoints the great circle is not unique: ``antipodal="perpendicular"``
    picks one deterministically (a keyframe track needs a path), ``"refuse"``
    returns None for any interior ``t`` (a blend has no meaningful midpoint).
    """
    za, zb = l2_normalize(a), l2_normalize(b)
    if za is None or zb is None:
        return None
    t = float(min(max(t, 0.0), 1.0))
    if t <= 0.0:
        return za
    if t >= 1.0:
        return zb
    a64, b64 = za.astype(np.float64), zb.astype(np.float64)
    theta = math.acos(float(min(max(a64 @ b64, -1.0), 1.0)))
    if theta < 1e-6:                                   # same point: nlerp is exact enough
        return l2_normalize((1.0 - t) * a64 + t * b64)
    if math.pi - theta < 1e-6:
        if antipodal == "refuse":
            return None
        return l2_normalize(math.cos(t * math.pi) * a64 + math.sin(t * math.pi) * _perpendicular(a64))
    s = math.sin(theta)
    return l2_normalize((math.sin((1.0 - t) * theta) / s) * a64 + (math.sin(t * theta) / s) * b64)


def spherical_mean(embeddings: Sequence[Any], weights: Sequence[float],
                   max_iter: int = 100, tol: float = 1e-10) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Weighted spherical (Karcher / Fréchet) mean of unit vectors.

    The point ``m`` minimising ``Σ w_i·angle(m, z_i)²``.  For one source it is
    that source, for two it is EXACTLY ``slerp(a, b, w_b)``, for 3+ it is found
    by the fixed-point iteration ``m ← exp_m(Σ w_i·log_m(z_i))`` from a fixed
    start (the normalised linear blend), so the result is deterministic.
    Invalid inputs are dropped and the rest renormalised.  Returns
    ``(vector_or_None, info)``; ``info["converged"]`` is False when the iteration
    did not settle (inputs spread over more than a hemisphere), in which case the
    normalised linear blend is returned instead and ``info["fallback"]`` says so.
    """
    pairs = [(l2_normalize(e), float(w)) for e, w in zip(embeddings, weights)]
    pairs = [(e, w) for e, w in pairs if e is not None]
    info: Dict[str, Any] = {"iterations": 0, "residual": 0.0, "converged": True, "fallback": None}
    if not pairs:
        return None, info
    w = normalize_weights([p[1] for p in pairs])
    stack = np.stack([p[0].astype(np.float64) for p in pairs])
    keep = w > 1e-12                                   # a zero-weight identity does not take part
    stack, w = stack[keep], w[keep]
    if len(w) == 1:
        return l2_normalize(stack[0]), info
    if len(w) == 2:
        out = slerp(stack[0], stack[1], float(w[1]), antipodal="refuse")
        if out is None:
            info.update(converged=False, fallback="antipodal")
        return out, info
    m = l2_normalize((w[:, None] * stack).sum(axis=0))
    if m is None:                                      # the linear start cancelled
        m = l2_normalize(stack[int(np.argmax(w))])
    m = m.astype(np.float64)
    residual = float("inf")
    for it in range(1, max_iter + 1):
        dots = np.clip(stack @ m, -1.0, 1.0)
        theta = np.arccos(dots)
        # log_m(z_i) = θ_i / sinθ_i · (z_i - cosθ_i·m); θ/sinθ → 1 as θ → 0
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(theta > 1e-9, theta / np.sin(theta), 1.0)
        tangent = (w[:, None] * scale[:, None] * (stack - dots[:, None] * m)).sum(axis=0)
        residual = float(np.linalg.norm(tangent))
        info["iterations"] = it
        if residual < tol:
            break
        m = math.cos(residual) * m + math.sin(residual) * tangent / residual
        m /= float(np.linalg.norm(m))
    info["residual"] = residual
    # `dots` is from before the last step; recompute the gate on the final point.
    info["converged"] = residual < 1e-6 and bool(np.all(stack @ m > 0.0))
    if not info["converged"]:
        info["fallback"] = "linear"
        return blend_embeddings(list(stack), w.tolist()), info
    return l2_normalize(m), info


def blend_vectors(embeddings: Sequence[Any], weights: Sequence[float],
                  mode: str = DEFAULT_BLEND_MODE) -> Optional[np.ndarray]:
    """``blend_embeddings`` (linear) or ``spherical_mean`` (slerp), by ``mode``."""
    if mode == "linear":
        return blend_embeddings(embeddings, weights)
    return spherical_mean(embeddings, weights)[0]


@dataclass
class DeidResult:
    embedding: np.ndarray
    applied: bool                   # the vector moved
    satisfied: bool                 # every reference is at or below max_cosine afterwards
    cosine_before: List[float]      # to each reference, as given
    cosine_after: List[float]
    cosine_to_primary: float        # how far the synthesised identity sits from the input
    progress: float                 # s along the geodesic primary -> orthogonal point
    reason: str                     # "" | "already_below" | "degenerate" | "floor" | "no_references"


def deidentify(primary: Any, references: Sequence[Any], max_cosine: float = DEFAULT_PRIVACY_COSINE,
               min_primary_cosine: float = 0.0) -> Optional[DeidResult]:
    """Orthogonal de-identification of ``primary`` against ``references``.

    Let ``B`` be an orthonormal basis of the span of the references and
    ``r = p - B Bᵀp`` the part of ``p`` orthogonal to all of them; ``r̂ = r/|r|``
    has cosine exactly 0 to every reference.  The output moves ``p`` along the
    great circle toward ``r̂``::

        z(s) = sin((1-s)φ)/sinφ · p + sin(sφ)/sinφ · r̂,   φ = acos(|r|)

    which is the tangent direction at ``p`` that is orthogonal to the references
    (it leaves the span of {p, r̂}), so ``z`` stays a unit ArcFace vector -- a
    face the swapper can render -- and moves no further from ``p`` than needed.
    Because ``r̂ ⟂ refs``, ``cos(z(s), ref_i) = [sin((1-s)φ)/sinφ]·cos(p, ref_i)``
    EXACTLY, so the smallest ``s`` meeting ``max_i cos <= max_cosine`` is closed
    form: ``(1-s)φ = asin(max_cosine/c_max · sinφ)``.

    ``min_primary_cosine`` (default 0 = off) bounds how far ``z`` may leave ``p``
    (``cos(z, p) = cos(sφ)``); when the threshold would need more, ``s`` stops at
    the bound and ``satisfied`` is False with ``reason == "floor"``.  A primary
    that lies inside the span of the references has no orthogonal direction
    (``reason == "degenerate"``): it is returned unchanged, ``satisfied`` False.

    The guarantee is on the INPUT vector.  What the swapper renders from it is
    measured by tests/identity_algebra_bench.py, not assumed.
    """
    p = l2_normalize(primary)
    if p is None:
        return None
    refs = [r for r in (l2_normalize(x) for x in references) if r is not None]
    p64 = p.astype(np.float64)
    tau = float(min(max(max_cosine, 0.0), 0.99))
    before = [float(p64 @ r.astype(np.float64)) for r in refs]
    if not refs:
        return DeidResult(p, False, True, [], [], 1.0, 0.0, "no_references")
    cmax = max(before)
    if cmax <= tau:
        return DeidResult(p, False, True, before, list(before), 1.0, 0.0, "already_below")
    basis = [b for b in gram_schmidt([r.astype(np.float64) for r in refs]) if b is not None]
    resid = p64 - sum(float(p64 @ b) * b for b in basis)
    rn = float(np.linalg.norm(resid))
    if rn < 1e-6:
        return DeidResult(p, False, False, before, list(before), 1.0, 0.0, "degenerate")
    ortho = resid / rn
    phi = math.acos(min(max(rn, -1.0), 1.0))          # p·r̂ = |r|
    target = tau / cmax                                # in [0, 1)
    s = 1.0 - math.asin(min(target * math.sin(phi), 1.0)) / phi
    reason = ""
    floor = float(min(max(min_primary_cosine, 0.0), 1.0))
    if floor > 0.0:
        s_max = math.acos(floor) / phi
        if s > s_max:
            s, reason = s_max, "floor"
    s = float(min(max(s, 0.0), 1.0))
    z = slerp(p64, ortho, s)
    z64 = z.astype(np.float64)
    after = [float(z64 @ r.astype(np.float64)) for r in refs]
    return DeidResult(z, s > 0.0, max(after) <= tau + 1e-6, before, after,
                      float(z64 @ p64), s, reason)


# ── keyframes: per-frame conditioning vectors ─────────────────────────────────

def bezier_ease(u: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """CSS ``cubic-bezier(x1, y1, x2, y2)`` easing: the y at which the curve
    from (0,0) to (1,1) reaches x = ``u``.  ``x1, x2`` are clamped to [0, 1]
    (so x(t) is monotone and the inverse exists) and ``y1, y2`` to [0, 1] (so the
    identity never overshoots its keyframes).  Bisection, 60 steps: exact to
    double precision and deterministic."""
    u = float(min(max(u, 0.0), 1.0))
    x1, x2 = float(min(max(x1, 0.0), 1.0)), float(min(max(x2, 0.0), 1.0))
    y1, y2 = float(min(max(y1, 0.0), 1.0)), float(min(max(y2, 0.0), 1.0))
    if u <= 0.0 or u >= 1.0:
        return u
    lo, hi = 0.0, 1.0
    for _ in range(60):
        t = 0.5 * (lo + hi)
        x = 3 * (1 - t) ** 2 * t * x1 + 3 * (1 - t) * t ** 2 * x2 + t ** 3
        if x < u:
            lo = t
        else:
            hi = t
    t = 0.5 * (lo + hi)
    return float(3 * (1 - t) ** 2 * t * y1 + 3 * (1 - t) * t ** 2 * y2 + t ** 3)


def ease(curve: str, u: float, bezier: Sequence[float] = DEFAULT_BEZIER) -> float:
    """Progress along a segment (0..1) for ``u`` = elapsed fraction of it."""
    if curve == CURVE_HOLD:
        return 0.0 if u < 1.0 else 1.0
    if curve == CURVE_CUBIC_BEZIER:
        return bezier_ease(u, *tuple(bezier)[:4])
    return float(min(max(u, 0.0), 1.0))


@dataclass(frozen=True)
class IdentityKeyframe:
    """``(frame_index, vector, curve)``.  ``curve`` shapes the segment that
    STARTS at this keyframe (the convention of every NLE), so the last
    keyframe's curve is unused.  ``frame_index`` is 0-based and absolute in the
    clip; ``vector`` is a unit ArcFace embedding."""
    frame_index: int
    vector: np.ndarray
    curve: str = CURVE_LINEAR
    bezier: Tuple[float, float, float, float] = DEFAULT_BEZIER


class KeyframeTrack:
    """Exact per-frame conditioning vectors from identity keyframes.

    * before the first keyframe: the first vector; after the last: the last;
    * between keyframes ``k`` and ``k+1``: ``slerp(v_k, v_{k+1}, ease(curve_k, u))``
      with ``u = (f - f_k)/(f_{k+1} - f_k)`` -- always a unit vector, and under
      LINEAR constant angular speed;
    * HOLD keeps ``v_k`` for ``f_k <= f < f_{k+1}`` and jumps to ``v_{k+1}`` at
      ``f_{k+1}`` (a deliberate cut);
    * two keyframes on one frame: the later one in the input wins.
    """

    def __init__(self, keyframes: Iterable[IdentityKeyframe]):
        by_frame: Dict[int, IdentityKeyframe] = {}
        for kf in keyframes:
            vec = l2_normalize(kf.vector)
            if vec is None:
                continue
            curve = kf.curve if kf.curve in CURVES else CURVE_LINEAR
            by_frame[int(kf.frame_index)] = IdentityKeyframe(int(kf.frame_index), vec, curve,
                                                             tuple(float(x) for x in kf.bezier))
        self.keyframes: List[IdentityKeyframe] = [by_frame[f] for f in sorted(by_frame)]
        self._frames = [k.frame_index for k in self.keyframes]

    def __len__(self) -> int:
        return len(self.keyframes)

    def segment_at(self, frame_index: int) -> Tuple[int, float]:
        """(index of the keyframe the segment starts at, eased progress 0..1)."""
        if not self.keyframes:
            raise ValueError("empty keyframe track")
        f = int(frame_index)
        if f <= self._frames[0]:
            return 0, 0.0
        if f >= self._frames[-1]:
            return len(self.keyframes) - 1, 0.0
        i = bisect.bisect_right(self._frames, f) - 1
        a, b = self.keyframes[i], self.keyframes[i + 1]
        u = (f - a.frame_index) / float(b.frame_index - a.frame_index)
        return i, ease(a.curve, u, a.bezier)

    def vector_at(self, frame_index: int) -> Optional[np.ndarray]:
        if not self.keyframes:
            return None
        i, s = self.segment_at(frame_index)
        a = self.keyframes[i]
        if s <= 0.0 or i + 1 >= len(self.keyframes):
            return a.vector
        return slerp(a.vector, self.keyframes[i + 1].vector, s)

    def vectors(self, frames: Iterable[int]) -> np.ndarray:
        """(N, 512) float32, one row per requested frame."""
        rows = [self.vector_at(f) for f in frames]
        if not rows or any(r is None for r in rows):
            return np.zeros((0, EMBEDDING_DIM), np.float32)
        return np.stack(rows).astype(np.float32)

    def smoothness(self, frames: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        """Angular step (radians) between consecutive frames over ``frames``
        (default: every frame from the first to the last keyframe).  A HOLD
        segment contributes exactly one step (the cut at its end)."""
        if len(self.keyframes) < 2 and frames is None:
            return {"frames": 0, "max_step": 0.0, "mean_step": 0.0, "cuts": 0, "min_norm": 1.0, "max_norm": 1.0}
        if frames is None:
            frames = range(self._frames[0], self._frames[-1] + 1)
        mat = self.vectors(frames).astype(np.float64)
        if len(mat) < 2:
            return {"frames": len(mat), "max_step": 0.0, "mean_step": 0.0, "cuts": 0,
                    "min_norm": 1.0, "max_norm": 1.0}
        steps = np.arccos(np.clip((mat[:-1] * mat[1:]).sum(axis=1), -1.0, 1.0))
        norms = np.linalg.norm(mat, axis=1)
        hold_ends = {self.keyframes[i + 1].frame_index for i in range(len(self.keyframes) - 1)
                     if self.keyframes[i].curve == CURVE_HOLD}
        flist = list(frames)
        cuts = [steps[j] for j in range(len(steps)) if flist[j + 1] in hold_ends and steps[j] > 1e-9]
        smooth = [steps[j] for j in range(len(steps)) if flist[j + 1] not in hold_ends]
        return {"frames": len(mat), "max_step": float(max(smooth) if smooth else 0.0),
                "mean_step": float(np.mean(smooth) if smooth else 0.0), "cuts": len(cuts),
                "max_cut": float(max(cuts) if cuts else 0.0),
                "min_norm": float(norms.min()), "max_norm": float(norms.max())}


# ── attribute directions ──────────────────────────────────────────────────────

@dataclass
class AttributeDirections:
    """Orthonormal attribute directions in normalised ArcFace space.

    ``units_per_step[k]`` is the label change per unit tangent step along
    ``vectors[k]`` (years for age; label units otherwise).  ``spread[k]`` is the
    standard deviation of real embeddings projected on the direction.
    ``heldout[k]`` is the identity-grouped held-out score (Pearson r, or AUC
    for gender) -- the number that says whether the direction is real.
    """
    names: Tuple[str, ...]
    vectors: np.ndarray                     # (K, 512) float32, orthonormal rows
    units_per_step: np.ndarray              # (K,)
    spread: np.ndarray                      # (K,)
    heldout: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    # {name: {"writable": bool, "verdict": str, "step_per_dial": float?}};
    # empty = not render-validated, every fitted direction usable.
    render: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def writable(self, name: str) -> bool:
        entry = self.render.get(name)
        return True if entry is None else bool(entry.get("writable", False))

    def index(self, name: str) -> Optional[int]:
        try:
            return self.names.index(name)
        except ValueError:
            return None

    def vector(self, name: str) -> Optional[np.ndarray]:
        i = self.index(name)
        return None if i is None else self.vectors[i]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path,
                 names=np.asarray(self.names),
                 vectors=self.vectors.astype(np.float32),
                 units_per_step=self.units_per_step.astype(np.float64),
                 spread=self.spread.astype(np.float64),
                 heldout=np.asarray(json.dumps(self.heldout)),
                 meta=np.asarray(json.dumps(self.meta, default=str)))

    @classmethod
    def load(cls, path: str) -> "AttributeDirections":
        with np.load(path, allow_pickle=False) as data:
            names = tuple(str(n) for n in data["names"].tolist())
            vectors = np.asarray(data["vectors"], dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape != (len(names), EMBEDDING_DIM):
                raise ValueError(f"identity directions: bad shape {vectors.shape}")
            gram = vectors.astype(np.float64) @ vectors.astype(np.float64).T
            if not np.allclose(gram, np.eye(len(names)), atol=1e-4):
                raise ValueError("identity directions are not orthonormal")
            return cls(names=names, vectors=vectors,
                       units_per_step=np.asarray(data["units_per_step"], dtype=np.float64),
                       spread=np.asarray(data["spread"], dtype=np.float64),
                       heldout=json.loads(str(data["heldout"])),
                       meta=json.loads(str(data["meta"])))


_directions_lock = threading.Lock()
_directions_cache: Dict[str, Any] = {}


def load_directions(path: Optional[str] = None) -> Optional[AttributeDirections]:
    """Cached load of the shipped direction file; None (and a printed reason)
    when absent or invalid -- callers then report the dials as unavailable."""
    path = os.path.abspath(path or DIRECTIONS_PATH)
    with _directions_lock:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            _directions_cache.pop(path, None)
            return None
        hit = _directions_cache.get(path)
        if hit is not None and hit[0] == mtime:
            return hit[1]
        try:
            value = AttributeDirections.load(path)
        except (OSError, ValueError, KeyError) as exc:
            print(f"[IdentityAlgebra] direction file rejected: {exc}")
            value = None
        if value is not None and path == os.path.abspath(DIRECTIONS_PATH):
            try:
                with open(RENDER_VALIDATION_PATH, encoding="utf-8") as fh:
                    value.render = dict(json.load(fh).get("directions") or {})
            except (OSError, ValueError):
                value.render = {}
        _directions_cache[path] = (mtime, value)
        return value


def _weighted_ridge(Z: np.ndarray, y: np.ndarray, sample_w: np.ndarray, lam: float) -> Tuple[np.ndarray, float]:
    """Weighted ridge regression y ~ Z w + b (intercept unpenalised)."""
    sw = sample_w / sample_w.sum()
    zm = (sw[:, None] * Z).sum(axis=0)
    ym = float((sw * y).sum())
    Zc = Z - zm
    yc = y - ym
    A = (Zc * sw[:, None]).T @ Zc + lam * np.eye(Z.shape[1])
    w = np.linalg.solve(A, (Zc * sw[:, None]).T @ yc)
    return w, ym - float(zm @ w)


def _balanced_weights(groups: np.ndarray) -> np.ndarray:
    """Every identity contributes equal total weight, however many frames."""
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse].astype(np.float64)


def _center_within(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.array(values, dtype=np.float64, copy=True)
    for g in np.unique(groups):
        m = groups == g
        out[m] -= out[m].mean(axis=0)
    return out


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels > 0.5], scores[labels <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size)
    ranks[order] = np.arange(1, order.size + 1)
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def _corr(pred: np.ndarray, y: np.ndarray) -> float:
    """Pearson r.  Scale-free on purpose: ridge shrinks the PREDICTION's
    scale (so held-out R^2 drops as the penalty rises) while the direction
    it points in gets MORE accurate -- measured on planted data, R^2 0.51 at
    cos 0.87 vs R^2 0.70 at cos 0.80.  The dial's slope is re-fitted along
    the final vector anyway, so only the direction needs scoring."""
    if pred.size < 3 or float(np.std(pred)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(pred, y)[0, 1])


# Which label is a property of the PERSON (fit across identities, balanced per
# identity) and which varies within one person (fit on identity-centred data,
# so the direction cannot simply point from one person to another).
_WITHIN_IDENTITY = {"expression"}


def fit_directions(embeddings: np.ndarray, labels: Mapping[str, np.ndarray],
                   groups: Optional[np.ndarray] = None,
                   order: Sequence[str] = DIRECTION_NAMES,
                   ridge: float = 1e-2, folds: int = 5,
                   seed: int = 0) -> AttributeDirections:
    """Fit orthonormal attribute directions from labelled embeddings.

    ``labels[name]`` is (N,) with NaN where unlabelled.  ``groups`` is an (N,)
    identity id; without it every sample is its own identity (and the held-out
    split is no longer identity-disjoint, so the score is optimistic).

    Each raw direction is the normalised ridge coefficient.  Gram-Schmidt then
    runs in ``order`` priority, and the label slope is re-read along the
    orthogonalised vector -- orthogonalisation changes it, and the dial's
    calibration must describe the vector that is actually applied.
    """
    Z = np.stack([l2_normalize(z) for z in embeddings]).astype(np.float64)
    n = Z.shape[0]
    groups = np.arange(n) if groups is None else np.asarray(groups)
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    fold_of = {g: int(i) % folds for i, g in enumerate(rng.permutation(uniq))}
    fold = np.asarray([fold_of[g] for g in groups])

    raw: List[Optional[np.ndarray]] = []
    fit_names: List[str] = []
    fits: Dict[str, Dict[str, Any]] = {}
    for name in order:
        y_all = np.asarray(labels.get(name, np.full(n, np.nan)), dtype=np.float64)
        mask = np.isfinite(y_all)
        if mask.sum() < 8 or np.unique(y_all[mask]).size < 2:
            continue
        Zs, ys, gs = Z[mask], y_all[mask], groups[mask]
        if name in _WITHIN_IDENTITY:
            Zs, ys = _center_within(Zs, gs), _center_within(ys, gs)
            sw = np.ones(ys.size)
        else:
            sw = _balanced_weights(gs)
        w, _ = _weighted_ridge(Zs, ys, sw, ridge)
        # Held-out: identity-disjoint folds.
        preds = np.full(ys.size, np.nan)
        fs = fold[mask]
        for k in range(folds):
            tr, te = fs != k, fs == k
            if te.sum() == 0 or tr.sum() < 4 or np.unique(ys[tr]).size < 2:
                continue
            wk, bk = _weighted_ridge(Zs[tr], ys[tr], sw[tr], ridge)
            preds[te] = Zs[te] @ wk + bk
        ok = np.isfinite(preds)
        binary = set(np.unique(ys).tolist()) <= {0.0, 1.0}
        score = (_auc(preds[ok], ys[ok]) if binary else _corr(preds[ok], ys[ok])) if ok.sum() > 4 else float("nan")
        fits[name] = {"Z": Zs, "sw": sw, "y": ys, "score": score, "binary": binary,
                      "n": int(mask.sum()), "identities": int(np.unique(gs).size)}
        raw.append(w)
        fit_names.append(name)

    ortho = gram_schmidt(raw)
    names, vecs, units, spread, heldout, detail = [], [], [], [], {}, {}
    for name, w, v in zip(fit_names, raw, ortho):
        if v is None:
            continue
        f = fits[name]
        # Orient so +dial means +label (older / male / squarer / more expressive).
        if float(w @ v) < 0:
            v = -v
        proj = f["Z"] @ v
        sw = f["sw"] / f["sw"].sum()
        mu = float((sw * proj).sum())
        sd = float(math.sqrt(max((sw * (proj - mu) ** 2).sum(), 1e-12)))
        # Slope of the label along the applied (orthogonalised) direction.
        slope = float(np.polyfit(proj, f["y"], 1, w=np.sqrt(f["sw"]))[0])
        names.append(name)
        vecs.append(v.astype(np.float32))
        units.append(slope)
        spread.append(sd)
        heldout[name] = f["score"]
        detail[name] = {"n": f["n"], "identities": f["identities"],
                        "metric": "auc" if f["binary"] else "r",
                        "cos_raw_vs_orthogonalised": float(w @ v / (np.linalg.norm(w) or 1.0))}
    return AttributeDirections(
        names=tuple(names),
        vectors=np.stack(vecs) if vecs else np.zeros((0, EMBEDDING_DIM), np.float32),
        units_per_step=np.asarray(units, dtype=np.float64),
        spread=np.asarray(spread, dtype=np.float64),
        heldout=heldout,
        meta={"fit": detail, "ridge": ridge, "folds": folds, "samples": int(n),
              "identities": int(uniq.size), "order": list(order)})


# ── offsets ───────────────────────────────────────────────────────────────────

@dataclass
class OffsetResult:
    embedding: np.ndarray
    requested: Dict[str, float]
    applied: Dict[str, float]          # in the same units as requested
    cosine_to_anchor: float
    clamped: bool
    unavailable: Tuple[str, ...]


def dial_steps(dials: Mapping[str, float], directions: AttributeDirections) -> Tuple[Dict[str, float], Tuple[str, ...]]:
    """Dial values -> tangent step per direction.

    ``age`` is in YEARS (clipped to +/-AGE_LIMIT_YEARS) and converted through
    the direction's own years-per-step slope.  Other dials are in [-1, 1] and
    scaled by DIAL_SIGMAS * the corpus spread along that direction.
    """
    steps: Dict[str, float] = {}
    missing: List[str] = []
    for name, value in dials.items():
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value) or abs(value) < 1e-9:
            continue
        i = directions.index(name)
        if i is None or not directions.writable(name):
            missing.append(name)
            continue
        render_step = (directions.render.get(name) or {}).get("step_per_dial")
        if render_step and name != "age":
            steps[name] = float(np.clip(value, -1.0, 1.0)) * float(render_step)
            continue
        if name == "age":
            years = float(np.clip(value, -AGE_LIMIT_YEARS, AGE_LIMIT_YEARS))
            slope = float(directions.units_per_step[i])
            if abs(slope) < 1e-6:
                missing.append(name)
                continue
            steps[name] = years / slope
        else:
            steps[name] = float(np.clip(value, -1.0, 1.0)) * DIAL_SIGMAS * float(directions.spread[i])
    return steps, tuple(missing)


def apply_offsets(anchor: Any, dials: Mapping[str, float],
                  directions: Optional[AttributeDirections],
                  min_cosine: float = DEFAULT_MIN_IDENTITY_COSINE) -> Optional[OffsetResult]:
    """``normalize(z + P_z(sum alpha_k v_k))`` with the identity guard.

    When the combined tangent step would take the vector below ``min_cosine``
    to the anchor, the WHOLE step is scaled back uniformly -- the direction of
    the edit is kept, only its length is bounded -- and ``clamped`` is set.
    """
    z = l2_normalize(anchor)
    if z is None:
        return None
    requested = {k: float(v) for k, v in dials.items()
                 if isinstance(v, (int, float)) and np.isfinite(v) and abs(float(v)) > 1e-9}
    if directions is None:
        return OffsetResult(z, requested, {}, 1.0, False, tuple(requested))
    steps, missing = dial_steps(requested, directions)
    if not steps:
        return OffsetResult(z, requested, {}, 1.0, False, missing)
    z64 = z.astype(np.float64)
    offset = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    for name, alpha in steps.items():
        offset += alpha * directions.vector(name).astype(np.float64)
    t = tangent_project(offset, z64)
    norm = float(np.linalg.norm(t))
    limit = max_tangent_norm(min_cosine)
    scale = 1.0
    if norm > limit and norm > 1e-12:
        scale = limit / norm
        t *= scale
    out = l2_normalize(z64 + t)
    applied = {}
    for name in steps:
        v = requested[name]
        if name == "age":
            v = float(np.clip(v, -AGE_LIMIT_YEARS, AGE_LIMIT_YEARS))
        else:
            v = float(np.clip(v, -1.0, 1.0))
        applied[name] = v * scale
    return OffsetResult(out, requested, applied, float(out.astype(np.float64) @ z64),
                        scale < 1.0, missing)


# ── recipes ───────────────────────────────────────────────────────────────────

def _finite(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def _bezier_from(value: Any) -> Tuple[float, float, float, float]:
    try:
        raw = [float(x) for x in list(value)[:4]]
    except (TypeError, ValueError):
        return DEFAULT_BEZIER
    if len(raw) != 4 or not all(np.isfinite(raw)):
        return DEFAULT_BEZIER
    x1, y1, x2, y2 = (float(np.clip(x, 0.0, 1.0)) for x in raw)
    return (x1, y1, x2, y2)


@dataclass
class BlendRecipe:
    """What the React Identity Blender sends; plain data, JSON-safe.

    ``keyframes`` make the blend a function of the frame: each keyframe is a
    weight set over the recipe's own ``components`` (``{source_id: weight}``,
    absent = 0) at an absolute 0-based ``frame_index``, with the ``curve`` of the
    segment that starts there.  ``deidentify`` pushes the final vector away from
    ``references`` (source ids) until its cosine to each is <= ``max_cosine``.
    """
    enabled: bool = False
    components: List[Dict[str, Any]] = field(default_factory=list)   # [{source_id, weight}]
    dials: Dict[str, float] = field(default_factory=dict)            # age (years), gender/jawline/expression in [-1,1]
    min_cosine: float = DEFAULT_MIN_IDENTITY_COSINE
    mode: str = DEFAULT_BLEND_MODE
    keyframes: List[Dict[str, Any]] = field(default_factory=list)
    deidentify: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False, "references": [], "max_cosine": DEFAULT_PRIVACY_COSINE,
        "min_primary_cosine": 0.0})

    @classmethod
    def from_payload(cls, payload: Optional[Mapping[str, Any]]) -> "BlendRecipe":
        payload = payload or {}
        comps = []
        seen = set()
        for item in list(payload.get("components") or []):
            if len(comps) >= MAX_BLEND_SOURCES:
                break
            if not isinstance(item, Mapping):
                continue
            sid = str(item.get("source_id") or "").strip()
            if not sid or sid in seen:
                continue
            weight = _finite(item.get("weight", 0.0))
            seen.add(sid)
            comps.append({"source_id": sid, "weight": float(np.clip(weight, 0.0, 100.0))})
        dials = {}
        for name in DIRECTION_NAMES:
            value = _finite((payload.get("dials") or {}).get(name, 0.0))
            limit = AGE_LIMIT_YEARS if name == "age" else 1.0
            dials[name] = float(np.clip(value, -limit, limit))
        min_cos = _finite(payload.get("min_cosine", DEFAULT_MIN_IDENTITY_COSINE), DEFAULT_MIN_IDENTITY_COSINE)
        min_cos = float(np.clip(min_cos, 0.5, 0.99))
        mode = str(payload.get("mode") or DEFAULT_BLEND_MODE).strip().lower()
        if mode not in BLEND_MODES:
            mode = DEFAULT_BLEND_MODE

        raw_deid = payload.get("deidentify")
        raw_deid = raw_deid if isinstance(raw_deid, Mapping) else {}
        refs: List[str] = []
        for r in list(raw_deid.get("references") or []):
            sid = str(r or "").strip()
            if sid and sid not in refs and len(refs) < MAX_BLEND_SOURCES:
                refs.append(sid)
        deid = {"enabled": bool(raw_deid.get("enabled", False)), "references": refs,
                "max_cosine": float(np.clip(_finite(raw_deid.get("max_cosine", DEFAULT_PRIVACY_COSINE),
                                                    DEFAULT_PRIVACY_COSINE), 0.0, 0.95)),
                "min_primary_cosine": float(np.clip(_finite(raw_deid.get("min_primary_cosine", 0.0)), 0.0, 0.99))}

        by_frame: Dict[int, Dict[str, Any]] = {}
        for item in list(payload.get("keyframes") or [])[:MAX_KEYFRAMES]:
            if not isinstance(item, Mapping):
                continue
            raw_frame = _finite(item.get("frame_index", -1), -1.0)
            if raw_frame < 0:
                continue
            weights = {}
            for sid, w in dict(item.get("weights") or {}).items():
                sid = str(sid)
                if sid in seen:
                    weights[sid] = float(np.clip(_finite(w), 0.0, 100.0))
            if sum(weights.values()) <= 1e-12:
                continue                        # a keyframe that names no live source says nothing
            curve = str(item.get("curve") or CURVE_LINEAR).strip().lower()
            frame = int(round(raw_frame))
            by_frame[frame] = {"frame_index": frame,
                               "curve": curve if curve in CURVES else CURVE_LINEAR,
                               "bezier": list(_bezier_from(item.get("bezier") or DEFAULT_BEZIER)),
                               "weights": weights}
        return cls(enabled=bool(payload.get("enabled", False)), components=comps,
                   dials=dials, min_cosine=min_cos, mode=mode,
                   keyframes=[by_frame[f] for f in sorted(by_frame)], deidentify=deid)

    def to_payload(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, "components": [dict(c) for c in self.components],
                "dials": dict(self.dials), "min_cosine": self.min_cosine, "mode": self.mode,
                "keyframes": [{**k, "bezier": list(k["bezier"]), "weights": dict(k["weights"])}
                              for k in self.keyframes],
                "deidentify": {**self.deidentify, "references": list(self.deidentify["references"])}}

    @property
    def has_dials(self) -> bool:
        return any(abs(v) > 1e-9 for v in self.dials.values())

    @property
    def has_deid(self) -> bool:
        return bool(self.deidentify.get("enabled")) and bool(self.deidentify.get("references"))

    @property
    def has_keyframes(self) -> bool:
        return bool(self.keyframes) and len(self.components) >= 2

    @property
    def active(self) -> bool:
        return self.enabled and (len(self.components) >= 2 or self.has_dials or self.has_deid)


def faceset_identity_vector(faceset: Any, target_face: Any = None) -> Optional[np.ndarray]:
    """The unit ArcFace vector a faceset contributes, pose-matched to the
    target when the faceset can do that (V2 cells / folder banks)."""
    if faceset is None:
        return None
    if target_face is not None:
        try:
            from roop.processors.FaceSwapInsightFace import pose_embedding_for_target
            vec = l2_normalize(pose_embedding_for_target(faceset, target_face))
            if vec is not None:
                return vec
        except Exception as _degrade_error:  # fall through to the centroid
            _swallowed("roop/identity_algebra.py:faceset_identity_vector",
                       _degrade_error, "pose embedding skipped; centroid used")
    vec = l2_normalize(getattr(faceset, "default_embedding", None))
    if vec is not None:
        return vec
    faces = getattr(faceset, "faces", None) or []
    if faces:
        face = faces[0]
        value = face.get("embedding") if isinstance(face, dict) else getattr(face, "embedding", None)
        return l2_normalize(value)
    return None


def _sid(faceset: Any) -> str:
    return str(getattr(faceset, "_source_id", "") or "").strip()


@dataclass
class _ResolvedKeyframe:
    frame_index: int
    curve: str
    bezier: Tuple[float, float, float, float]
    weights: np.ndarray                    # aligned with ResolvedBlend.facesets


class ResolvedBlend:
    """A recipe bound to concrete FaceSet objects for one run.

    Built once per ProcessMgr.initialize; ``transform`` is then called per
    face on the swap path and costs a handful of 512-d numpy ops.
    """

    def __init__(self, recipe: BlendRecipe, facesets: Sequence[Any], weights: Sequence[float],
                 directions: Optional[AttributeDirections], missing_ids: Sequence[str] = (),
                 deid_refs: Sequence[Any] = (), keyframes: Sequence[_ResolvedKeyframe] = ()):
        self.recipe = recipe
        self.facesets = list(facesets)
        self.weights = normalize_weights(weights) if len(weights) else np.zeros(0)
        self.directions = directions
        self.missing_ids = tuple(missing_ids)
        self.mode = recipe.mode if recipe.mode in BLEND_MODES else DEFAULT_BLEND_MODE
        self.deid_refs = list(deid_refs)
        self._kf = sorted(keyframes, key=lambda k: k.frame_index) if len(self.facesets) >= 2 else []
        self._kf_frames = [k.frame_index for k in self._kf]
        self._member_ids = {id(fs) for fs in self.facesets}
        self._member_sids = {_sid(fs) for fs in self.facesets} - {""}
        self.calls = 0
        self.clamped = 0
        self.keyframed = 0           # transforms that read the keyframe track
        self.frameless = 0           # ... that had keyframes but no frame number (base weights used)
        self.deid_applied = 0
        self.deid_unsatisfied = 0
        self._lock = threading.Lock()

    @property
    def time_varying(self) -> bool:
        """The vector depends on the frame: a pre-baked table must not answer for it."""
        return len(self._kf) >= 2

    def applies_to(self, source_faceset: Any) -> bool:
        """A blend replaces the identity of faces assigned to ANY of its
        components (whichever one the mapping picked, the result is the blend).
        With fewer than two components it is an attribute edit on whatever
        source the face was assigned."""
        if len(self.facesets) < 2:
            return True
        return (id(source_faceset) in self._member_ids
                or (_sid(source_faceset) or None) in self._member_sids)

    def _component_vectors(self, z: Optional[np.ndarray], target_face: Any,
                           source_faceset: Any) -> List[Optional[np.ndarray]]:
        vecs: List[Optional[np.ndarray]] = []
        for fs in self.facesets:
            # The component the face was assigned already carries the
            # pose-selected vector; reuse it rather than re-deriving.
            if z is not None and (fs is source_faceset
                                  or (_sid(fs) and _sid(fs) == _sid(source_faceset))):
                vecs.append(z)
            else:
                vecs.append(faceset_identity_vector(fs, target_face))
        return vecs

    def _blended(self, vecs: Sequence[Optional[np.ndarray]], frame_index: Optional[int]) -> Optional[np.ndarray]:
        """The blend at ``frame_index``: base weights without keyframes, else the
        slerp between the two keyframes' blends that bracket the frame."""
        if not self._kf:
            return blend_vectors(vecs, self.weights, self.mode)
        if len(self._kf) == 1:
            with self._lock:
                self.keyframed += 1
            return blend_vectors(vecs, self._kf[0].weights, self.mode)
        if frame_index is None:
            with self._lock:
                self.frameless += 1
            return blend_vectors(vecs, self.weights, self.mode)
        with self._lock:
            self.keyframed += 1
        f = int(frame_index)
        if f <= self._kf_frames[0]:
            return blend_vectors(vecs, self._kf[0].weights, self.mode)
        if f >= self._kf_frames[-1]:
            return blend_vectors(vecs, self._kf[-1].weights, self.mode)
        i = bisect.bisect_right(self._kf_frames, f) - 1
        a, b = self._kf[i], self._kf[i + 1]
        s = ease(a.curve, (f - a.frame_index) / float(b.frame_index - a.frame_index), a.bezier)
        va = blend_vectors(vecs, a.weights, self.mode)
        if s <= 0.0 or va is None:
            return va
        vb = blend_vectors(vecs, b.weights, self.mode)
        return va if vb is None else slerp(va, vb, s)

    def compose(self, anchor: Optional[np.ndarray], target_face: Any = None, source_faceset: Any = None,
                frame_index: Optional[int] = None) -> Optional[np.ndarray]:
        """blend (at the frame) -> attribute dials -> de-identification.  No
        ``applies_to`` gate: ``transform`` is that gate plus this.  De-identification
        runs LAST so its cosine bound holds on the vector the swapper receives."""
        z = anchor
        if len(self.facesets) >= 2:
            z = self._blended(self._component_vectors(z, target_face, source_faceset), frame_index)
        if z is None:
            return None
        if self.recipe.has_dials:
            res = apply_offsets(z, self.recipe.dials, self.directions, self.recipe.min_cosine)
            if res is None:
                return None
            if res.clamped:
                with self._lock:
                    self.clamped += 1
            z = res.embedding
        if self.deid_refs:
            refs = [faceset_identity_vector(fs, target_face) for fs in self.deid_refs]
            cfg = self.recipe.deidentify
            out = deidentify(z, refs, cfg["max_cosine"], cfg["min_primary_cosine"])
            if out is not None:
                z = out.embedding
                with self._lock:
                    self.deid_applied += int(out.applied)
                    self.deid_unsatisfied += int(not out.satisfied)
        return z

    def transform(self, anchor_embedding: Any, target_face: Any = None,
                  source_faceset: Any = None, frame_index: Optional[int] = None) -> Optional[np.ndarray]:
        """Blended + offset unit vector, or None to leave the face untouched.
        ``frame_index`` is the 0-based absolute clip frame (keyframes read it)."""
        if not self.applies_to(source_faceset):
            return None
        z = self.compose(l2_normalize(anchor_embedding), target_face, source_faceset, frame_index)
        if z is None:
            return None
        with self._lock:
            self.calls += 1
        return z


def resolve_recipe(recipe: BlendRecipe, facesets: Iterable[Any],
                   directions: Optional[AttributeDirections] = None) -> Optional[ResolvedBlend]:
    """Bind recipe source ids to FaceSet objects (by ``_source_id``).

    Returns None when the recipe is inactive or nothing it names is loaded.
    """
    if not recipe.active:
        return None
    by_id = {}
    for fs in facesets:
        sid = _sid(fs)
        if sid and sid not in by_id:
            by_id[sid] = fs
    bound, bound_ids, weights, missing = [], [], [], []
    for comp in recipe.components:
        fs = by_id.get(comp["source_id"])
        if fs is None:
            missing.append(comp["source_id"])
            continue
        bound.append(fs)
        bound_ids.append(comp["source_id"])
        weights.append(comp["weight"])
    deid_refs = []
    if recipe.has_deid:
        for sid in recipe.deidentify["references"]:
            fs = by_id.get(sid)
            if fs is None:
                missing.append(sid)
            else:
                deid_refs.append(fs)
    if len(bound) < 2 and not recipe.has_dials and not deid_refs:
        return None
    if len(bound) >= 2 and float(np.sum(weights)) <= 1e-12:
        weights = [1.0] * len(bound)
    if recipe.has_dials and directions is None:
        directions = load_directions()
    keyframes: List[_ResolvedKeyframe] = []
    if len(bound) >= 2:
        for kf in recipe.keyframes:
            kw = np.asarray([float(kf["weights"].get(sid, 0.0)) for sid in bound_ids], dtype=np.float64)
            if float(kw.sum()) <= 1e-12:
                continue                # every source this keyframe weighted is missing from the gallery
            keyframes.append(_ResolvedKeyframe(int(kf["frame_index"]), kf["curve"],
                                               tuple(kf["bezier"]), kw))
    return ResolvedBlend(recipe, bound if len(bound) >= 2 else [],
                         weights if len(bound) >= 2 else [], directions, missing,
                         deid_refs=deid_refs, keyframes=keyframes)


def _resolved_vectors(recipe: BlendRecipe, facesets: Sequence[Any]):
    by_id = {str(getattr(fs, "_source_id", "") or ""): fs for fs in facesets}
    comps = [(c, by_id.get(c["source_id"])) for c in recipe.components]
    vecs = [faceset_identity_vector(fs) if fs is not None else None for _, fs in comps]
    return by_id, comps, vecs


def describe_keyframes(recipe: BlendRecipe, facesets: Iterable[Any],
                       frames: Optional[Sequence[int]] = None, samples: int = 96) -> Dict[str, Any]:
    """What the keyframe track does, for the UI's curve readout and for tests:
    each keyframe's blended vector (pose-agnostic centroids), the angle between
    consecutive keyframes, the eased progress and cosine-to-keyframes at sampled
    frames, and the step-to-step smoothness over the whole range."""
    facesets = list(facesets)
    _, comps, vecs = _resolved_vectors(recipe, facesets)
    keyed = []
    for kf in recipe.keyframes:
        weights = [float(kf["weights"].get(c["source_id"], 0.0)) for c, _ in comps]
        vec = blend_vectors(vecs, weights, recipe.mode)
        if vec is not None:
            keyed.append((kf, vec))
    out: Dict[str, Any] = {"mode": recipe.mode, "keyframes": [], "samples": [],
                           "smoothness": None, "skipped": len(recipe.keyframes) - len(keyed)}
    if not keyed:
        return out
    track = KeyframeTrack(IdentityKeyframe(kf["frame_index"], vec, kf["curve"], tuple(kf["bezier"]))
                          for kf, vec in keyed)
    for i, k in enumerate(track.keyframes):
        entry = {"frame_index": k.frame_index, "curve": k.curve, "bezier": list(k.bezier)}
        if i + 1 < len(track.keyframes):
            nxt = track.keyframes[i + 1].vector.astype(np.float64)
            entry["angle_to_next_deg"] = float(np.degrees(np.arccos(
                np.clip(float(k.vector.astype(np.float64) @ nxt), -1.0, 1.0))))
        out["keyframes"].append(entry)
    first, last = track.keyframes[0].frame_index, track.keyframes[-1].frame_index
    if frames is None:
        n = max(2, min(int(samples), last - first + 1))
        picked = sorted({int(round(x)) for x in np.linspace(first, last, n)}) if last > first else [first]
    else:
        picked = [int(f) for f in list(frames)[:2000]]
    kmat = np.stack([k.vector for k in track.keyframes]).astype(np.float64)
    for f in picked:
        v = track.vector_at(f)
        i, s = track.segment_at(f)
        out["samples"].append({"frame_index": f, "segment": i, "progress": float(s),
                               "cosine_to_keyframes": [round(float(x), 4) for x in kmat @ v.astype(np.float64)]})
    out["smoothness"] = track.smoothness()
    return out


def describe_blend(recipe: BlendRecipe, facesets: Iterable[Any],
                   directions: Optional[AttributeDirections] = None) -> Dict[str, Any]:
    """Diagnostics for the UI: where the blended vector sits relative to each
    source (cosine), the effective offset after the identity guard, which
    dials have no shipped direction, and what de-identification did."""
    facesets = list(facesets)
    directions = directions if directions is not None else load_directions()
    by_id, comps, vecs = _resolved_vectors(recipe, facesets)
    valid = [(c, v) for (c, _), v in zip(comps, vecs) if v is not None]
    weights = normalize_weights([c["weight"] for c, _ in valid]) if valid else np.zeros(0)
    blended = blend_vectors([v for _, v in valid], weights, recipe.mode) if len(valid) >= 2 else (
        valid[0][1] if valid else None)
    out: Dict[str, Any] = {"directions_available": list(directions.names) if directions else [],
                           "heldout": dict(directions.heldout) if directions else {},
                           "mode": recipe.mode, "components": []}
    for (c, v), w in zip(valid, weights):
        entry = {"source_id": c["source_id"], "weight": float(w)}
        if blended is not None:
            entry["cosine_to_blend"] = float(np.dot(blended, v))
        out["components"].append(entry)
    if len(valid) >= 2:
        pair = np.stack([v for _, v in valid]).astype(np.float64)
        out["pairwise_cosine"] = (pair @ pair.T).round(4).tolist()
        out["pairwise_ids"] = [c["source_id"] for c, _ in valid]
    out["missing"] = [c["source_id"] for c, fs in comps if fs is None]
    final = blended
    if blended is not None and recipe.has_dials:
        res = apply_offsets(blended, recipe.dials, directions, recipe.min_cosine)
        if res is not None:
            final = res.embedding
            out.update({"applied": res.applied, "clamped": res.clamped,
                        "cosine_to_anchor": res.cosine_to_anchor,
                        "unavailable": list(res.unavailable)})
    if final is not None and recipe.has_deid:
        cfg = recipe.deidentify
        ref_ids = [sid for sid in cfg["references"] if sid in by_id]
        refs = [faceset_identity_vector(by_id[sid]) for sid in ref_ids]
        res = deidentify(final, refs, cfg["max_cosine"], cfg["min_primary_cosine"])
        if res is not None:
            out["deidentify"] = {"references": ref_ids, "max_cosine": cfg["max_cosine"],
                                 "applied": res.applied, "satisfied": res.satisfied, "reason": res.reason,
                                 "cosine_before": [round(x, 4) for x in res.cosine_before],
                                 "cosine_after": [round(x, 4) for x in res.cosine_after],
                                 "cosine_to_primary": round(res.cosine_to_primary, 4),
                                 "progress": round(res.progress, 4)}
        out["missing"] += [sid for sid in cfg["references"] if sid not in by_id]
    if recipe.has_keyframes:
        out["keyframes"] = describe_keyframes(recipe, facesets)
    return out
