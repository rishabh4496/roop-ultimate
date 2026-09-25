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
MAX_BLEND_SOURCES = 4
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

@dataclass
class BlendRecipe:
    """What the React Identity Blender sends; plain data, JSON-safe."""
    enabled: bool = False
    components: List[Dict[str, Any]] = field(default_factory=list)   # [{source_id, weight}]
    dials: Dict[str, float] = field(default_factory=dict)            # age (years), gender/jawline/expression in [-1,1]
    min_cosine: float = DEFAULT_MIN_IDENTITY_COSINE

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
            try:
                weight = float(item.get("weight", 0.0))
            except (TypeError, ValueError):
                weight = 0.0
            if not np.isfinite(weight):
                weight = 0.0
            seen.add(sid)
            comps.append({"source_id": sid, "weight": float(np.clip(weight, 0.0, 100.0))})
        dials = {}
        for name in DIRECTION_NAMES:
            try:
                value = float((payload.get("dials") or {}).get(name, 0.0))
            except (TypeError, ValueError):
                value = 0.0
            if not np.isfinite(value):
                value = 0.0
            limit = AGE_LIMIT_YEARS if name == "age" else 1.0
            dials[name] = float(np.clip(value, -limit, limit))
        try:
            min_cos = float(payload.get("min_cosine", DEFAULT_MIN_IDENTITY_COSINE))
        except (TypeError, ValueError):
            min_cos = DEFAULT_MIN_IDENTITY_COSINE
        min_cos = float(np.clip(min_cos if np.isfinite(min_cos) else DEFAULT_MIN_IDENTITY_COSINE, 0.5, 0.99))
        return cls(enabled=bool(payload.get("enabled", False)), components=comps,
                   dials=dials, min_cosine=min_cos)

    def to_payload(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, "components": [dict(c) for c in self.components],
                "dials": dict(self.dials), "min_cosine": self.min_cosine}

    @property
    def has_dials(self) -> bool:
        return any(abs(v) > 1e-9 for v in self.dials.values())

    @property
    def active(self) -> bool:
        return self.enabled and (len(self.components) >= 2 or self.has_dials)


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


class ResolvedBlend:
    """A recipe bound to concrete FaceSet objects for one run.

    Built once per ProcessMgr.initialize; ``transform`` is then called per
    face on the swap path and costs a handful of 512-d numpy ops.
    """

    def __init__(self, recipe: BlendRecipe, facesets: Sequence[Any], weights: Sequence[float],
                 directions: Optional[AttributeDirections], missing_ids: Sequence[str] = ()):
        self.recipe = recipe
        self.facesets = list(facesets)
        self.weights = normalize_weights(weights) if len(weights) else np.zeros(0)
        self.directions = directions
        self.missing_ids = tuple(missing_ids)
        self._member_ids = {id(fs) for fs in self.facesets}
        self._member_sids = {_sid(fs) for fs in self.facesets} - {""}
        self.calls = 0
        self.clamped = 0
        self._lock = threading.Lock()

    def applies_to(self, source_faceset: Any) -> bool:
        """A blend replaces the identity of faces assigned to ANY of its
        components (whichever one the mapping picked, the result is the blend).
        With fewer than two components it is an attribute edit on whatever
        source the face was assigned."""
        if len(self.facesets) < 2:
            return True
        return (id(source_faceset) in self._member_ids
                or (_sid(source_faceset) or None) in self._member_sids)

    def transform(self, anchor_embedding: Any, target_face: Any = None,
                  source_faceset: Any = None) -> Optional[np.ndarray]:
        """Blended + offset unit vector, or None to leave the face untouched."""
        if not self.applies_to(source_faceset):
            return None
        z = l2_normalize(anchor_embedding)
        if len(self.facesets) >= 2:
            vecs = []
            for fs in self.facesets:
                # The component the face was assigned already carries the
                # pose-selected vector; reuse it rather than re-deriving.
                if z is not None and (fs is source_faceset
                                      or (_sid(fs) and _sid(fs) == _sid(source_faceset))):
                    vecs.append(z)
                else:
                    vecs.append(faceset_identity_vector(fs, target_face))
            z = blend_embeddings(vecs, self.weights)
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
    bound, weights, missing = [], [], []
    for comp in recipe.components:
        fs = by_id.get(comp["source_id"])
        if fs is None:
            missing.append(comp["source_id"])
            continue
        bound.append(fs)
        weights.append(comp["weight"])
    if len(bound) < 2 and not recipe.has_dials:
        return None
    if len(bound) >= 2 and float(np.sum(weights)) <= 1e-12:
        weights = [1.0] * len(bound)
    if recipe.has_dials and directions is None:
        directions = load_directions()
    return ResolvedBlend(recipe, bound if len(bound) >= 2 else [],
                         weights if len(bound) >= 2 else [], directions, missing)


def describe_blend(recipe: BlendRecipe, facesets: Iterable[Any],
                   directions: Optional[AttributeDirections] = None) -> Dict[str, Any]:
    """Diagnostics for the UI: where the blended vector sits relative to each
    source (cosine), the effective offset after the identity guard, and which
    dials have no shipped direction."""
    facesets = list(facesets)
    directions = directions if directions is not None else load_directions()
    by_id = {str(getattr(fs, "_source_id", "") or ""): fs for fs in facesets}
    comps = [(c, by_id.get(c["source_id"])) for c in recipe.components]
    vecs = [faceset_identity_vector(fs) if fs is not None else None for _, fs in comps]
    valid = [(c, v) for (c, _), v in zip(comps, vecs) if v is not None]
    weights = normalize_weights([c["weight"] for c, _ in valid]) if valid else np.zeros(0)
    blended = blend_embeddings([v for _, v in valid], weights) if len(valid) >= 2 else (
        valid[0][1] if valid else None)
    out: Dict[str, Any] = {"directions_available": list(directions.names) if directions else [],
                           "heldout": dict(directions.heldout) if directions else {},
                           "components": []}
    for (c, v), w in zip(valid, weights):
        entry = {"source_id": c["source_id"], "weight": float(w)}
        if blended is not None:
            entry["cosine_to_blend"] = float(np.dot(blended, v))
        out["components"].append(entry)
    if len(valid) >= 2:
        pair = np.stack([v for _, v in valid]).astype(np.float64)
        out["pairwise_cosine"] = (pair @ pair.T).round(4).tolist()
    out["missing"] = [c["source_id"] for c, fs in comps if fs is None]
    if blended is not None and recipe.has_dials:
        res = apply_offsets(blended, recipe.dials, directions, recipe.min_cosine)
        if res is not None:
            out.update({"applied": res.applied, "clamped": res.clamped,
                        "cosine_to_anchor": res.cosine_to_anchor,
                        "unavailable": list(res.unavailable)})
    return out
