"""The Selected-Face routing decision, as one pure, testable function.

This is the exact per-frame assignment that ``ProcessMgr.swap_faces`` runs in
``selected`` / ``selected_multi`` mode, lifted out of the 400-line method so the
one thing the "Selected Face" bug is about -- *which detected face receives
which source, and which faces are left alone* -- can be exercised with synthetic
embeddings, deterministically, with no GPU and no detector.

It is a decision only. It reads no globals and touches no model: identity is
supplied as two callables so a test can hand it cosine-on-synthetic-vectors and
production hands it the AdaFace/w600k recogniser. ``ProcessMgr`` keeps ownership
of the audit counters and the compositing; it drives both from this result, so
the numbers a render reports are unchanged.

Contract, matching the original block byte-for-byte in behaviour:

* only the captured angles of the SELECTED person(s) are considered
  (``selected_target_groups``);
* a face whose recognition crop is shared with its neighbour (``unreliable``)
  is offered to nobody -- there is no track to fall back on in this mode;
* every remaining (face, selected-person) pair within ``id_threshold`` is a
  candidate; candidates are claimed closest-first, 1:1 -- a person swaps at most
  one face, a face is swapped by at most one person;
* the source is ``selected_index`` when a single person is selected, else the
  person's rank; a face whose person has no source faceset is refused, not
  redirected to person zero.

One addition to that contract: ``track_binding``. Identity is a property of a
PERSON, not of a frame, and a single frame's embedding is a noisy reading of it
-- a turned head, motion blur or a shadow moves it well past the gate on a face
the run swapped a frame earlier and swaps again a frame later. Answering that
noise with "do not paint this face" is what the user sees as the swap blinking
on and off. So a face whose TRACK the whole-clip pre-pass already bound to this
person is held through those frames, as a second tier that claims only after
every confirmed match has: the per-frame gate still decides who is who, and the
track only keeps a decision already made from being dropped by one bad frame.

The hold is not unconditional. It needs the track binding (evidence from every
frame of that track, not this one), the face must still be within
``hold_threshold`` -- a deliberately looser gate, the same relationship the
track veto has to the match threshold -- and no other selected person may fit
it better. A track whose identity really did change therefore stops being held
as soon as the face stops resembling the person at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# reason strings -- identical to the audit-bucket names ProcessMgr prints, so a
# caller can map them straight onto _audit_hit without a translation table.
SWAPPED = "swapped (identity match)"
# A face the per-frame gate refused and the track binding held. Counted as a
# swap (ProcessMgr's audit sums every bucket starting with "swapped"), and named
# apart from the per-frame match so the two populations stay separable in the
# audit -- the whole point of the change is that this number is not zero.
SWAPPED_TRACK_HOLD = "swapped (track continuity)"
REFUSED_NO_SOURCE = "refused: no source faceset for that person"
REFUSED_CONTAMINATED = "refused: crop shared with the face beside it"
REFUSED_OVER_THRESHOLD = "refused: over the identity threshold"
REFUSED_CLOSER_FACE = "refused: that person matched a closer face"


@dataclass
class CandidateDistance:
    """One (face, selected-person) identity comparison -- the assertion-8 log."""
    face_index: int
    group: int
    rank: int
    best_reference_angle: Optional[int]
    distance: Optional[float]
    eligible: bool
    contaminated: bool


@dataclass
class SelectedAssignment:
    # (source_index, face_index), in the order they were claimed.
    pending: List[Tuple[int, int]] = field(default_factory=list)
    # (distance, group, face_index), sorted closest-first -- the eligible pairs.
    candidates: List[Tuple[float, int, int]] = field(default_factory=list)
    # the same shape for pairs admitted by track continuity alone: over the
    # per-frame gate, inside the hold gate, on a track already bound to that
    # person. They claim only after every entry in `candidates` has.
    held: List[Tuple[float, int, int]] = field(default_factory=list)
    # one row per (face, selected-person) comparison actually evaluated, plus a
    # contaminated marker row per contaminated face. This is what assertion 8
    # ("log the identity distance for every detected candidate") reads.
    distances: List[CandidateDistance] = field(default_factory=list)
    # face_index -> reason it was or was not swapped. Every face appears once.
    reasons: Dict[int, str] = field(default_factory=dict)
    single_person: bool = False
    persons: Dict[int, List[int]] = field(default_factory=dict)

    def swapped_face_indices(self) -> List[int]:
        return [fidx for _src, fidx in self.pending]

    def source_for_face(self, face_index: int) -> Optional[int]:
        for src, fidx in self.pending:
            if fidx == face_index:
                return src
        return None


def compute_selected_assignment(
    faces: List[Any],
    target_face_datas: List[Any],
    target_face_groups: List[int],
    selected_target_groups,
    selected_index: int,
    num_sources: int,
    id_threshold: float,
    *,
    identity_match: Callable[[List[Any], Any], Tuple[Optional[int], Optional[float]]],
    unreliable: Callable[[Any], bool],
    track_binding: Optional[Callable[[Any], Optional[int]]] = None,
    hold_threshold: Optional[float] = None,
) -> SelectedAssignment:
    """Pure Selected-Face routing.

    ``identity_match(reference_faces, probe_face) -> (best_angle_index, distance)``
    is ``recognizer_adaface.best_identity_match`` with the frame bound; a test
    supplies a cosine over synthetic embeddings. ``unreliable(face) -> bool`` is
    ``face_contact.unreliable``; a test supplies a lambda. Neither is called with
    a frame here -- production binds the frame before passing them in -- so this
    function is trivially deterministic.

    ``track_binding(face) -> source index | None`` reports which source the
    whole-clip pre-pass bound this face's TRACK to, and ``hold_threshold`` is
    the looser gate such a face is still required to pass. Both absent (or a
    binding that answers None, which is every face on a run with no pre-pass)
    leaves the decision exactly as it was.
    """
    selected = set(selected_target_groups or ())

    # person group id -> its captured target-face (angle) indices, restricted to
    # the captured angles that actually have a datum, then to the selected
    # person(s). An empty selection yields no persons and therefore no swaps.
    persons: Dict[int, List[int]] = {}
    for i, g in enumerate(target_face_groups[:len(target_face_datas)]):
        persons.setdefault(g, []).append(i)
    persons = {g: tis for g, tis in persons.items() if g in selected}

    uniq = sorted(set(target_face_groups)) if target_face_groups else []
    rank = {g: r for r, g in enumerate(uniq)}
    single_person = len(persons) <= 1

    result = SelectedAssignment(single_person=single_person, persons=dict(persons))

    source_of = {g: (selected_index if single_person else rank[g])
                 for g in persons}
    hold_enabled = (track_binding is not None and hold_threshold is not None
                    and hold_threshold > id_threshold)

    candidates: List[Tuple[float, int, int]] = []
    held: List[Tuple[float, int, int]] = []
    contaminated: set = set()
    for fidx, face in enumerate(faces):
        if unreliable(face):
            contaminated.add(fidx)
            result.distances.append(CandidateDistance(
                face_index=fidx, group=-1, rank=-1, best_reference_angle=None,
                distance=None, eligible=False, contaminated=True))
            continue
        measured: List[Tuple[int, Optional[float]]] = []
        for g, tis in persons.items():
            best_angle, d = identity_match(
                [target_face_datas[ti] for ti in tis], face)
            actual_angle = tis[best_angle] if best_angle is not None else None
            eligible = d is not None and d <= id_threshold
            result.distances.append(CandidateDistance(
                face_index=fidx, group=g, rank=rank[g],
                best_reference_angle=actual_angle,
                distance=(None if d is None else float(d)),
                eligible=bool(eligible), contaminated=False))
            measured.append((g, None if d is None else float(d)))
            if eligible:
                candidates.append((float(d), g, fidx))

        if not hold_enabled:
            continue
        # Track continuity. Only for a person this face's own gate refused:
        # everything the gate accepted is already a candidate above.
        bound = track_binding(face)
        if bound is None:
            continue
        scored = [(d, g) for g, d in measured if d is not None]
        if not scored:
            continue
        nearest = min(scored)[1]        # the selected person this face fits best
        for g, d in measured:
            if d is None or d <= id_threshold:
                continue                # not refused, or nothing to compare
            if source_of.get(g) != bound:
                continue                # the pre-pass bound the track elsewhere
            if d > hold_threshold:
                continue                # not this person on this frame at all
            if g != nearest:
                continue                # another selected person fits better
            held.append((float(d), g, fidx))

    candidates.sort(key=lambda c: c[0])   # greedily assign closest pairs first
    held.sort(key=lambda c: c[0])
    result.candidates = list(candidates)
    result.held = list(held)

    claimed_faces: set = set()
    claimed_persons: set = set()
    # Tier order, not one merged sort: a face the gate CONFIRMED must never lose
    # a contested person to one that is only being held, however the two
    # distances compare -- a held face is by construction the further away.
    for tier, reason in ((candidates, SWAPPED), (held, SWAPPED_TRACK_HOLD)):
        for d, g, fidx in tier:
            if fidx in claimed_faces or g in claimed_persons:
                continue
            claimed_faces.add(fidx)
            claimed_persons.add(g)
            src_index = selected_index if single_person else rank[g]
            if 0 <= src_index < num_sources:
                result.pending.append((src_index, fidx))
                result.reasons[fidx] = reason
            else:
                # A selected person with no source faceset is refused here,
                # never silently redirected to person zero -- that redirect was
                # the bug.
                result.reasons[fidx] = REFUSED_NO_SOURCE

    paired = {fidx for _d, _g, fidx in candidates}
    paired |= {fidx for _d, _g, fidx in held}
    for fidx in range(len(faces)):
        if fidx in result.reasons:
            continue
        if fidx in contaminated:
            result.reasons[fidx] = REFUSED_CONTAMINATED
        elif fidx not in paired:
            result.reasons[fidx] = REFUSED_OVER_THRESHOLD
        else:
            result.reasons[fidx] = REFUSED_CLOSER_FACE

    return result


def format_distance_log(result: SelectedAssignment, frame_idx=None) -> str:
    """The assertion-8 line: identity distance for every detected candidate.

    One compact, stable line per frame. Emitted by ProcessMgr when
    ``ROOP_LOG_SELECTED_ROUTE`` is set (and always available to the harness),
    so a routing decision can be audited from a log without a screenshot.
    """
    head = "[SelectedRoute]"
    if frame_idx is not None:
        head += f" frame={frame_idx}"
    head += (f" persons={sorted(result.persons)}"
             f" single_person={result.single_person}"
             f" swapped={result.swapped_face_indices()}"
             f" held={[f for _d, _g, f in result.held]}")
    rows = []
    for cd in result.distances:
        if cd.contaminated:
            rows.append(f"face={cd.face_index}:contaminated")
            continue
        dist = "none" if cd.distance is None else f"{cd.distance:.3f}"
        rows.append(
            f"face={cd.face_index}->person{cd.rank}(g{cd.group})"
            f" angle={cd.best_reference_angle} d={dist}"
            f" eligible={str(cd.eligible).lower()}")
    return head + " | " + " ; ".join(rows) if rows else head + " | (no faces)"
