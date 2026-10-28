"""Stage 0: the source identity, frozen before the first frame is decoded.

WHAT THE SWAP PASS USED TO DO, PER FACE, PER FRAME. For an embedding-driven swapper `process_face`
chose the identity vector from scratch every time it swapped a face:

    1. `pose_embedding_for_target`   the V2 yaw/pitch cell of the target, normalised, into a COPY of the
                                     source Face with its cached `_latent_*` derivations deleted
    2. `ResolvedBlend.transform`     (only with an Identity Blender recipe) the per-component cell vectors
                                     again, the weighted blend, the tangent-plane dial offsets
                                     `z' = normalize(z + P_z(sum alpha_k v_k))` and the 0.80 identity guard
    3. the swapper's own latent      `normed_embedding @ emap` / the converter MLP / the LRU lookup, on the
                                     copy, because step 1 had thrown the cached one away

All of it is a pure function of (source, target head-pose CELL): `pose_matrix_cell` only ever returns one
of 9 (yaw, pitch) cells. So the answer for every cell can be computed once, here, before any frame is
read, and the swap pass reduces to a three-comparison lookup.

THE TABLE. `(yaw_bin, pitch_bin, roll_bin)` -> a ready-to-swap source Face, a dense 3 x 3 x 3 grid built
per source. The yaw/pitch edges are the .fsz pose matrix's own (`faceset_v2.POSE_MATRIX_*_EDGES`). Roll
(`solve_pose_5pt`'s third angle, + = clockwise) is part of the KEY so a source gallery that learns roll
can fill it; today every roll bin addresses the same Face because nothing that picks a source reads roll
(the in-plane rotation is removed by the similarity alignment before the swapper sees the face).

EQUIVALENCE BY CONSTRUCTION. Each cell is built by calling the SAME two functions the swap pass calls
(`pose_conditioned_input`, `blended_input` below - `process_face` now calls them too) on a stand-in target
whose pose lies inside the cell. Nothing is re-implemented, so the table cannot drift from the live path;
`tests/test_stage0_preflight.py` additionally compares table and live path over a dense pose grid.

WHEN THE TABLE DOES NOT APPLY (it then declines, says why, and the live path runs unchanged):
  * the swapper takes an image or CSCS crops, not the ArcFace vector
  * `use_source_bank` is on and the set holds several faces (a per-frame bank selector)
  * the source carries an angle portfolio (its route reads the target's keypoints and the frame LUT)
  * the temporal quality engine is on (it can re-select the source mid-render)
  * a folder-upload bank (`reference_embeddings`): its nearest-pose blend is CONTINUOUS in (yaw, pitch),
    so a 9-cell table would change the vector - declined, not approximated
  * nothing would be pre-applied (a V1 set with no recipe: the live path already returns the canonical
    face untouched)
A table that answered "nearly" would swap a different identity than the live path; it declines instead.
"""
from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from roop.degrade import swallowed as _swallowed
from roop.env import env_bool
from roop.faceset_v2 import (POSE_MATRIX_PITCH_NAMES, POSE_MATRIX_YAW_NAMES,
                              pose_matrix_cell)

#: `ROOP_STAGE0=0` turns the table off and runs the live per-face selection (the A/B arm of the verifier).
STAGE0_ENV = "ROOP_STAGE0"


def enabled() -> bool:
    return env_bool(STAGE0_ENV, True)


ROLL_EDGES = (-15.0, 15.0)
ROLL_NAMES = ("ccw", "level", "cw")
_YAW_INDEX = {n: i for i, n in enumerate(POSE_MATRIX_YAW_NAMES)}
_PITCH_INDEX = {n: i for i, n in enumerate(POSE_MATRIX_PITCH_NAMES)}

# A pose that lies inside each bin, used to ask the live selectors what they answer for that cell.
# Chosen away from the edges (yaw +-25, pitch +-15, roll +-15) so rounding cannot move them.
_YAW_CENTRE = (-45.0, 0.0, 45.0)
_PITCH_CENTRE = (-30.0, 0.0, 30.0)
_ROLL_CENTRE = (-30.0, 0.0, 30.0)


def _number(face: Any, name: str, default: float = 0.0) -> float:
    """Same reading rule as `pose_embedding_for_target.field`: a missing / None value is the default."""
    value = face.get(name, default) if isinstance(face, dict) else getattr(face, name, default)
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)


def roll_bin(roll: float) -> int:
    """0 (counter-clockwise) / 1 (level) / 2 (clockwise). NaN reads as level, like an unknown pose elsewhere."""
    if roll is None or not math.isfinite(roll):
        return 1
    return 0 if roll < ROLL_EDGES[0] else (2 if roll > ROLL_EDGES[1] else 1)


def cell_index(yaw: float, pitch: float, roll: float = 0.0) -> Tuple[int, int, int]:
    """The (yaw, pitch, roll) bin of a head pose. Yaw/pitch go through `pose_matrix_cell` itself, so the
    table can never disagree with the .fsz pose matrix about which cell a pose is in."""
    yaw_name, pitch_name = pose_matrix_cell((yaw, pitch))
    return _YAW_INDEX[yaw_name], _PITCH_INDEX[pitch_name], roll_bin(roll)


def stand_in_target(index: Tuple[int, int, int]) -> Dict[str, float]:
    """A minimal target face whose pose lies in `index`: what the selectors read, and nothing else."""
    return {"_adaptive_yaw": _YAW_CENTRE[index[0]], "_adaptive_pitch": _PITCH_CENTRE[index[1]],
            "_adaptive_roll": _ROLL_CENTRE[index[2]]}


# ── the two selection steps the swap pass shares with the builder ─────────────────────────────────────

def pose_conditioned_input(inputface: Any, source_faceset: Any, target_face: Any) -> Any:
    """Step 1: the source Face conditioned on the target's V2 pose cell, or `inputface` itself.

    Copy-then-override, because source Face objects are shared by workers: the selected unit vector goes
    on a COPY, and every cached `_latent_*` derivation of the old vector is dropped from it. This is the
    block `process_face` ran inline; it lives here so the swap pass and the Stage 0 builder run one body."""
    from roop.processors.FaceSwapInsightFace import pose_embedding_for_target
    pose_embedding = pose_embedding_for_target(source_faceset, target_face)
    if pose_embedding is None:
        return inputface
    pose_input = type(inputface)(inputface)
    pose_input["embedding"] = pose_embedding
    for key in list(pose_input.keys()):
        if str(key).startswith("_latent_"):
            del pose_input[key]
    return pose_input


def blended_input(inputface: Any, blend: Any, source_faceset: Any, target_face: Any,
                  frame_index: Optional[int] = None) -> Any:
    """Step 2: the Identity Blender (latent blend + tangent-plane dial offsets, 0.80 guard, then
    de-identification) applied to the already pose-selected vector, or `inputface` unchanged when the recipe
    leaves this face alone. `frame_index` (0-based, absolute in the clip) is what a keyframed recipe reads; the
    Stage 0 builder passes none and never sees a keyframed recipe (`build` declines it).
    The block `process_face` ran inline (see `pose_conditioned_input`)."""
    from roop.ProcessMgr import _with_unit_embedding
    raw = inputface.get("embedding") if hasattr(inputface, "get") else None
    new = blend.transform(raw, target_face, source_faceset, frame_index=frame_index)
    if new is None:
        return inputface
    return _with_unit_embedding(inputface, new)


# ── the table ────────────────────────────────────────────────────────────────────────────────────────

@dataclass
class SourceTable:
    """One source's 3 x 3 x 3 grid of ready-to-swap Faces, plus how it was built."""
    source_index: int
    base: Any                                   # the canonical Face the live path starts from
    cells: List[List[List[Any]]]                # [yaw][pitch][roll] -> Face
    distinct: int = 0                           # how many different Faces the 27 cells address
    differs_from_base: int = 0                  # cells whose vector is not the canonical one
    blended: bool = False
    clamped_cells: int = 0                      # of the 9 pose cells, those where the identity guard scaled a dial back
    min_cosine: Optional[float] = None          # the guard's floor for this recipe (cos to the un-dialled vector)
    dials_applied: Tuple[str, ...] = ()         # dials with a render-validated writable direction, pre-applied
    dials_unavailable: Tuple[str, ...] = ()     # dials the render validation says do not write through the swapper
    notes: List[str] = field(default_factory=list)


@dataclass
class Stage0Stats:
    hits: int = 0
    misses: int = 0                             # armed source, but the live path had started from a different Face
    declined: Dict[int, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def hit(self) -> None:
        with self._lock:
            self.hits += 1

    def miss(self) -> None:
        with self._lock:
            self.misses += 1


class SourceIdentityTable:
    """The frozen identity of every source of a run. Read-only after `build`; safe to share across workers."""

    def __init__(self) -> None:
        self.sources: Dict[int, SourceTable] = {}
        self.stats = Stage0Stats()

    # -- lookup (the swap pass) ----------------------------------------------------------------------
    def lookup(self, face_index: int, started_from: Any, target_face: Any) -> Optional[Any]:
        """The source Face for this target, or None to run the live path.

        `started_from` is the Face the live path began with; the table answers only if it is the very
        canonical Face it was built from, so any per-frame re-selection upstream (source bank, quality
        engine) falls through to the live path instead of being overwritten."""
        entry = self.sources.get(face_index)
        if entry is None:
            return None
        if started_from is not entry.base:
            self.stats.miss()
            return None
        yi, pi, ri = cell_index(_number(target_face, "_adaptive_yaw"), _number(target_face, "_adaptive_pitch"),
                                _number(target_face, "_adaptive_roll"))
        self.stats.hit()
        return entry.cells[yi][pi][ri]

    # -- audit ---------------------------------------------------------------------------------------
    def digest(self) -> str:
        """SHA-256 over every armed source's 27 cell vectors in a fixed order: two runs that froze the same
        identities print the same string."""
        h = hashlib.sha256()
        for idx in sorted(self.sources):
            entry = self.sources[idx]
            h.update(("src%d" % idx).encode())
            for yi in range(3):
                for pi in range(3):
                    for ri in range(3):
                        emb = entry.cells[yi][pi][ri].get("embedding")
                        h.update(np.ascontiguousarray(np.asarray(emb, dtype=np.float32)).tobytes())
        return h.hexdigest()

    def summary_lines(self) -> List[str]:
        lines = []
        for idx in sorted(self.sources):
            e = self.sources[idx]
            lines.append("[Stage0] source %d: %d of 9 pose cells hold a vector other than the centroid%s; "
                         "%d distinct Face(s) behind the 27-cell yaw/pitch/roll grid%s"
                         % (idx, e.differs_from_base // 3, ", identity blend + dials pre-applied" if e.blended else "",
                            e.distinct, ("  [" + "; ".join(e.notes) + "]") if e.notes else ""))
            if e.blended and (e.dials_applied or e.dials_unavailable):
                lines.append("[Stage0] source %d: dials pre-applied %s; unavailable (not writable through the swapper, "
                             "identity_render_validation.json) %s; identity guard cos >= %.2f bit in %d of 9 cells"
                             % (idx, list(e.dials_applied) or "none", list(e.dials_unavailable) or "none",
                                e.min_cosine if e.min_cosine is not None else float("nan"), e.clamped_cells))
        for idx in sorted(self.stats.declined):
            if idx < 0:                                   # the whole table is switched off
                lines.append("[Stage0] disabled (%s): the swap pass selects the source per face" % self.stats.declined[idx])
            else:
                lines.append("[Stage0] source %d: live path (%s)" % (idx, self.stats.declined[idx]))
        return lines

    def armed(self) -> bool:
        return bool(self.sources)


def _dial_report(blend: Any) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Which of the recipe's non-zero dials have a render-validated writable direction, and which do not."""
    from roop.identity_algebra import dial_steps
    dials = {k: v for k, v in blend.recipe.dials.items() if abs(float(v)) > 1e-9}
    if not dials:
        return (), ()
    if blend.directions is None:
        return (), tuple(sorted(dials))
    steps, missing = dial_steps(dials, blend.directions)
    return tuple(sorted(steps)), tuple(sorted(missing))


def freeze_for(pm: Any) -> Optional["SourceIdentityTable"]:
    """Stage 0 for one run: the table for `pm`'s sources (a ProcessMgr mid-`run_batch_inmem`), or None.

    Runs before the first frame is read. Never raises and never changes what the swap pass produces: a source
    the table cannot answer EXACTLY is declined with a printed reason and runs the live path."""
    try:
        swap_p = next((p for p in pm.processors if getattr(p, "type", None) == "swap"), None)
        mode = str(getattr(swap_p, "embedding_mode", "normed_emap") or "normed_emap")
        quality = getattr(pm, "_temporal_quality", None)
        prewarm = getattr(swap_p, "_compute_source_input", None)
        table = build(
            list(pm.input_face_datas or []),
            swap_mode=mode,
            use_source_bank=bool(getattr(pm.options, "use_source_bank", False)),
            quality_engine=bool(quality is not None and getattr(quality, "enabled", False)),
            blend=getattr(pm, "_identity_blend", None),
            prewarm=prewarm if callable(prewarm) else None,
            log=lambda line: print(line, flush=True))
        return table if table.armed() else None
    except Exception as exc:
        _swallowed("roop/stage0_preflight.py:freeze_for", exc, "the swap pass selects the source per face")
        print("[Stage0] source identity pre-flight skipped (%s: %s); the swap pass selects the source per face "
              "as before." % (type(exc).__name__, exc), flush=True)
        return None


def decline_reason(source_faceset: Any, *, swap_mode: str, use_source_bank: bool, quality_engine: bool,
                   has_recipe: bool) -> Optional[str]:
    """Why a table must NOT answer for this source (None = it may)."""
    if swap_mode in ("image", "cscs_dual"):
        return "the swapper takes %s input, not the ArcFace vector" % ("an image crop" if swap_mode == "image"
                                                                       else "CSCS crops")
    if quality_engine:
        return "the temporal quality engine can re-select the source per frame"
    if use_source_bank and len(getattr(source_faceset, "faces", ()) or ()) > 1:
        return "use_source_bank selects among %d faces per frame" % len(source_faceset.faces)
    if getattr(source_faceset, "angle_portfolio", None) is not None:
        return "the source carries an angle portfolio (routed from the target's keypoints)"
    if list(getattr(source_faceset, "reference_embeddings", None) or []):
        return "a folder-upload bank blends nearest poses continuously; nine cells would change the vector"
    if not getattr(source_faceset, "faces", None):
        return "the source holds no face"
    if int(getattr(source_faceset, "format_version", 1) or 1) < 2 and not has_recipe:
        return "nothing to pre-apply (V1 set, no identity recipe)"
    return None


def build(input_face_datas: List[Any], *, swap_mode: str, use_source_bank: bool, quality_engine: bool,
          blend: Any, prewarm: Optional[Callable[[Any], None]] = None,
          log: Callable[[str], None] = print) -> SourceIdentityTable:
    """Freeze every eligible source. Never raises: a source that cannot be frozen runs the live path."""
    table = SourceIdentityTable()
    if not enabled():
        table.stats.declined[-1] = "%s=0" % STAGE0_ENV
        for line in table.summary_lines():
            log(line)
        return table
    for idx, fs in enumerate(input_face_datas or []):
        try:
            # The live path calls `blend.transform` for every face but the recipe answers None for a source it
            # does not apply to, so a recipe that skips this source is the same as no recipe for it.
            effective = blend if (blend is not None and blend.applies_to(fs)) else None
            if effective is not None and getattr(effective, "time_varying", False):
                # A keyframed identity is a function of the frame; nine pose cells cannot hold it.
                table.stats.declined[idx] = "the identity recipe is keyframed: the vector changes with the frame"
                continue
            reason = decline_reason(fs, swap_mode=swap_mode, use_source_bank=use_source_bank,
                                    quality_engine=quality_engine, has_recipe=effective is not None)
            if reason is not None:
                table.stats.declined[idx] = reason
                continue
            canonical = fs.canonical_face_index() if hasattr(fs, "canonical_face_index") else 0
            base = fs.faces[canonical]
            cells: List[List[List[Any]]] = [[[None] * 3 for _ in range(3)] for _ in range(3)]
            differs = clamped = 0
            for yi in range(3):
                for pi in range(3):
                    stand_in = stand_in_target((yi, pi, 1))
                    face = pose_conditioned_input(base, fs, stand_in)
                    if effective is not None:
                        guard_before = effective.clamped
                        face = blended_input(face, effective, fs, stand_in)
                        clamped += int(effective.clamped > guard_before)
                    if prewarm is not None and face is not base:
                        try:
                            prewarm(face)
                        except Exception as exc:                 # a latent that cannot be pre-built is built live
                            _swallowed("roop/stage0_preflight.py:prewarm", exc, "latent left for the swap pass")
                    if face is not base:
                        differs += 3
                    for ri in range(3):
                        cells[yi][pi][ri] = face
            distinct = len({id(f) for row in cells for col in row for f in col})
            entry = SourceTable(source_index=idx, base=base, cells=cells, distinct=distinct,
                                differs_from_base=differs, blended=effective is not None, clamped_cells=clamped)
            if effective is not None:
                entry.min_cosine = float(effective.recipe.min_cosine)
                entry.dials_applied, entry.dials_unavailable = _dial_report(effective)
            if differs == 0:
                table.stats.declined[idx] = "nothing to pre-apply (every cell is the canonical face)"
                continue
            table.sources[idx] = entry
        except Exception as exc:
            _swallowed("roop/stage0_preflight.py:build", exc, "this source runs the live path")
            table.stats.declined[idx] = "the table could not be built (%s: %s)" % (type(exc).__name__, exc)
    for line in table.summary_lines():
        log(line)
    return table
