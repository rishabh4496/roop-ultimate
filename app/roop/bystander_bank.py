"""Target / bystander identity banks and the differential-margin lock gate.

WHAT THIS OWNS
--------------
A face may be bound to a target swap slot only on RELATIVE evidence: it must
look like the target AND look clearly more like the target than like anybody
the scene is known to contain who is NOT a target. This module is that
decision as pure numpy -- no model sessions, no frames, no globals -- so the
rule can be exercised with synthetic vectors and driven by any caller.

    bound  <=>  S_t >= tau   and   S_t - max_b S_b >= margin

`tau` is `tau_lock` for a face with no lock yet and the lower `tau_hold` for a
track that is already a confirmed target, provided the caller says the track's
landmark motion is still consistent (optical flow, `roop/flow_coast.py`). A
face that is closer to a registered bystander than to the target is
`NON_TARGET_BYSTANDER`: sticky for the track, so it cannot be re-promoted by
one lucky frame.

WHY A RELATIVE GATE AND NOT A LOWER CAP
---------------------------------------
`roop/selected_routing.py` records the absolute-cap version of this idea being
built and REJECTED (2026-09-22): in contact the neighbour is inside the face's
own recognition crop, the embedding measures the pair, and "no absolute cap
separates them". A margin against a *named other person* is a different
question -- both readings are contaminated by the same pair, so the contamination
mostly cancels in the difference. Whether it cancels ENOUGH is an empirical
question, answered by `app/tests/probe_bystander_stage16.py` and recorded in
`docs/CHANGELOG.md` (Stage 16); the default thresholds here are the brief's and
the caller decides whether to apply them.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not decide which faces are bystanders. `BystanderBank.add` takes an
embedding and a quality verdict from the caller; `bystanders_from_tracks`
is the one place the pre-pass's track dicts become bank entries, and it admits
only CLEAN, SUSTAINED tracks that no target explains. A bystander learned from
a contaminated frame is a chimera of the target and the neighbour and would
veto the target itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


NON_TARGET_BYSTANDER = 'non_target_bystander'
TARGET = 'target'
UNKNOWN = 'unknown'

# The brief's numbers. Similarities are w600k cosine (1 - distance).
TAU_LOCK = 0.42
TAU_HOLD = 0.22
MARGIN = 0.12


def _unit(vector: Any) -> Optional[np.ndarray]:
    if vector is None:
        return None
    try:
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if v.size == 0 or not np.isfinite(v).all():
        return None
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-9 else None


class IdentityBank:
    """L2-normalised centroids of one population, merged by cosine.

    `add` folds an embedding into the nearest centroid when it is at least
    `merge_sim` similar (running weighted mean, re-normalised), else opens a new
    centroid. Capacity-bounded: past `max_entries` the two most similar
    centroids are merged, so a long clip cannot grow it without limit.
    """

    def __init__(self, merge_sim: float = 0.55, max_entries: int = 32):
        self.merge_sim = float(merge_sim)
        self.max_entries = max(1, int(max_entries))
        self._sums: List[np.ndarray] = []      # weighted sums, un-normalised
        self._weights: List[float] = []
        self._lock = RLock()

    def __len__(self) -> int:
        return len(self._sums)

    @property
    def centroids(self) -> np.ndarray:
        with self._lock:
            if not self._sums:
                return np.zeros((0, 0), np.float32)
            return np.stack([_unit(s) for s in self._sums]).astype(np.float32)

    def add(self, embedding: Any, weight: float = 1.0) -> int:
        """Fold `embedding` in; returns the centroid index it landed in (-1 = unusable)."""
        v = _unit(embedding)
        if v is None or weight <= 0:
            return -1
        with self._lock:
            if self._sums:
                sims = np.stack([_unit(s) for s in self._sums]) @ v
                j = int(np.argmax(sims))
                if float(sims[j]) >= self.merge_sim:
                    self._sums[j] = self._sums[j] + v * float(weight)
                    self._weights[j] += float(weight)
                    return j
            self._sums.append(v * float(weight))
            self._weights.append(float(weight))
            if len(self._sums) > self.max_entries:
                self._merge_closest()
            return len(self._sums) - 1

    def _merge_closest(self) -> None:
        c = np.stack([_unit(s) for s in self._sums])
        sims = c @ c.T
        np.fill_diagonal(sims, -2.0)
        i, j = np.unravel_index(int(np.argmax(sims)), sims.shape)
        i, j = int(min(i, j)), int(max(i, j))
        self._sums[i] = self._sums[i] + self._sums[j]
        self._weights[i] += self._weights[j]
        del self._sums[j], self._weights[j]

    def similarities(self, embedding: Any) -> np.ndarray:
        """Cosine similarity to every centroid (empty when the bank is empty)."""
        v = _unit(embedding)
        with self._lock:
            if v is None or not self._sums:
                return np.zeros((0,), np.float32)
            return (np.stack([_unit(s) for s in self._sums]) @ v).astype(np.float32)

    def max_similarity(self, embedding: Any) -> Tuple[float, int]:
        """``(best similarity, centroid index)``; ``(-1.0, -1)`` when empty."""
        s = self.similarities(embedding)
        if s.size == 0:
            return -1.0, -1
        j = int(np.argmax(s))
        return float(s[j]), j

    def clear(self) -> None:
        with self._lock:
            self._sums.clear()
            self._weights.clear()


class TargetBank:
    """The target's gallery: every captured angle, NOT collapsed to one mean.

    A multi-angle `.fsz` gallery is the point of the bank -- a profile and a
    frontal view of one person are far apart, and a single centroid fits neither.
    Similarity is therefore the best match over the gallery.
    """

    def __init__(self, embeddings: Iterable[Any] = ()):
        self._gallery: List[np.ndarray] = []
        for e in embeddings:
            v = _unit(e)
            if v is not None:
                self._gallery.append(v)

    def __len__(self) -> int:
        return len(self._gallery)

    def similarity(self, embedding: Any) -> float:
        v = _unit(embedding)
        if v is None or not self._gallery:
            return -1.0
        return float(np.max(np.stack(self._gallery) @ v))


@dataclass(frozen=True)
class GateDecision:
    bound: bool
    label: str                  # TARGET / NON_TARGET_BYSTANDER / UNKNOWN
    reason: str
    s_target: float
    s_bystander: float          # -1.0 when no bystander is registered
    margin: float               # s_target - s_bystander (s_target when none)
    bystander_index: int = -1


def differential_margin_gate(s_target: float, s_bystander: float, *,
                             locked: bool = False,
                             flow_consistent: bool = True,
                             tau_lock: float = TAU_LOCK,
                             tau_hold: float = TAU_HOLD,
                             margin: float = MARGIN,
                             hold_margin: float = 0.0,
                             bystander_index: int = -1) -> GateDecision:
    """Decide whether one face may bind to the target slot.

    `locked` is "this track is already a confirmed target". Hysteresis:

      * not locked  -> needs ``s_target >= tau_lock`` AND ``margin`` over every bystander;
      * locked      -> holds down to ``tau_hold`` AND ``hold_margin`` (default 0:
                       still closer to the target than to any bystander), but ONLY
                       while `flow_consistent` -- a low similarity is believed
                       only when the landmark motion says it is the same head.

    A face closer to a bystander than to the target is NON_TARGET_BYSTANDER
    whatever the thresholds say.
    """
    s_t = float(s_target)
    s_b = float(s_bystander)
    diff = s_t - s_b if s_b > -1.0 else s_t
    if s_b > -1.0 and s_b > s_t:
        return GateDecision(False, NON_TARGET_BYSTANDER,
                            'closer to a registered bystander than to the target',
                            s_t, s_b, diff, bystander_index)
    if locked:
        if not flow_consistent:
            return GateDecision(False, UNKNOWN,
                                'locked track, but landmark flow is not consistent',
                                s_t, s_b, diff, bystander_index)
        if s_t < tau_hold:
            return GateDecision(False, UNKNOWN, 'below the hold floor',
                                s_t, s_b, diff, bystander_index)
        if s_b > -1.0 and diff < hold_margin:
            return GateDecision(False, UNKNOWN, 'inside the hold margin',
                                s_t, s_b, diff, bystander_index)
        return GateDecision(True, TARGET, 'held', s_t, s_b, diff, bystander_index)
    if s_t < tau_lock:
        return GateDecision(False, UNKNOWN, 'below the lock threshold',
                            s_t, s_b, diff, bystander_index)
    if s_b > -1.0 and diff < margin:
        return GateDecision(False, UNKNOWN, 'inside the differential margin',
                            s_t, s_b, diff, bystander_index)
    return GateDecision(True, TARGET, 'locked', s_t, s_b, diff, bystander_index)


class TrackLedger:
    """Per-track lock state, with a sticky bystander verdict.

    `observe` feeds one frame's gate inputs for one track and returns that
    frame's decision. A track becomes TARGET after one passing frame at the lock
    threshold; it stays TARGET through weak frames as long as `differential_margin_gate`
    holds. A track whose frames read NON_TARGET_BYSTANDER `bystander_frames`
    times in a row is labelled bystander and refused until `release_frames`
    consecutive frames clear the full lock gate -- one lucky frame cannot
    promote it, and one bad frame cannot demote a confirmed target.
    """

    def __init__(self, tau_lock: float = TAU_LOCK, tau_hold: float = TAU_HOLD,
                 margin: float = MARGIN, hold_margin: float = 0.0,
                 bystander_frames: int = 3, release_frames: int = 5):
        self.tau_lock = tau_lock
        self.tau_hold = tau_hold
        self.margin = margin
        self.hold_margin = hold_margin
        self.bystander_frames = max(1, int(bystander_frames))
        self.release_frames = max(1, int(release_frames))
        self._state: Dict[int, str] = {}
        self._bystander_run: Dict[int, int] = {}
        self._release_run: Dict[int, int] = {}

    def label(self, track_id: int) -> str:
        return self._state.get(int(track_id), UNKNOWN)

    def observe(self, track_id: int, s_target: float, s_bystander: float, *,
                flow_consistent: bool = True,
                bystander_index: int = -1) -> GateDecision:
        tid = int(track_id)
        state = self._state.get(tid, UNKNOWN)
        full = differential_margin_gate(
            s_target, s_bystander, locked=False, flow_consistent=flow_consistent,
            tau_lock=self.tau_lock, tau_hold=self.tau_hold, margin=self.margin,
            hold_margin=self.hold_margin, bystander_index=bystander_index)
        if state == NON_TARGET_BYSTANDER:
            if full.bound:
                self._release_run[tid] = self._release_run.get(tid, 0) + 1
                if self._release_run[tid] >= self.release_frames:
                    self._state[tid] = TARGET
                    self._bystander_run[tid] = 0
                    self._release_run[tid] = 0
                    return full
            else:
                self._release_run[tid] = 0
            return GateDecision(False, NON_TARGET_BYSTANDER,
                                'track is a registered bystander',
                                full.s_target, full.s_bystander, full.margin,
                                bystander_index)
        decision = differential_margin_gate(
            s_target, s_bystander, locked=(state == TARGET),
            flow_consistent=flow_consistent, tau_lock=self.tau_lock,
            tau_hold=self.tau_hold, margin=self.margin,
            hold_margin=self.hold_margin, bystander_index=bystander_index)
        if decision.label == NON_TARGET_BYSTANDER:
            run = self._bystander_run.get(tid, 0) + 1
            self._bystander_run[tid] = run
            if run >= self.bystander_frames:
                self._state[tid] = NON_TARGET_BYSTANDER
                self._release_run[tid] = 0
        else:
            self._bystander_run[tid] = 0
        if decision.bound:
            self._state[tid] = TARGET
        return decision

    def reset(self) -> None:
        self._state.clear()
        self._bystander_run.clear()
        self._release_run.clear()


def bystanders_from_tracks(tracks: Sequence[Dict[str, Any]],
                           target: TargetBank,
                           *,
                           exclude_below: float = 0.30,
                           min_obs: int = 5,
                           merge_sim: float = 0.55,
                           embedding_key: str = 'emb_mean',
                           bank: Optional[IdentityBank] = None) -> IdentityBank:
    """Register the scene's non-targets from the pre-pass's track dicts.

    A track qualifies only if it is SUSTAINED (`min_obs` clean observations),
    its identity is CLEAN (`emb_dirty` false: a chimera seeded in a contact
    would veto the target) and NO target explains it (best similarity to the
    target gallery under `exclude_below`). The last test is why this is safe to
    run before source assignment: a track the target gallery cannot rule out is
    never a bystander, so it can only ever lose a bank entry, not gain a wrong
    veto. Tracks are folded in longest-first, so a long clean track anchors the
    centroid and a short fragment of the same person merges into it.
    """
    out = bank if bank is not None else IdentityBank(merge_sim=merge_sim)
    ranked = sorted(tracks, key=lambda t: -int(t.get('emb_n') or 0))
    for t in ranked:
        if t.get('emb_dirty'):
            continue
        if int(t.get('emb_n') or 0) < min_obs:
            continue
        emb = t.get(embedding_key)
        if emb is None:
            continue
        if len(target) and target.similarity(emb) >= exclude_below:
            continue
        out.add(emb, weight=float(int(t.get('emb_n') or 1)))
    return out
