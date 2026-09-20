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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# reason strings -- identical to the audit-bucket names ProcessMgr prints, so a
# caller can map them straight onto _audit_hit without a translation table.
SWAPPED = "swapped (identity match)"
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
) -> SelectedAssignment:
    """Pure Selected-Face routing.

    ``identity_match(reference_faces, probe_face) -> (best_angle_index, distance)``
    is ``recognizer_adaface.best_identity_match`` with the frame bound; a test
    supplies a cosine over synthetic embeddings. ``unreliable(face) -> bool`` is
    ``face_contact.unreliable``; a test supplies a lambda. Neither is called with
    a frame here -- production binds the frame before passing them in -- so this
    function is trivially deterministic.
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

    candidates: List[Tuple[float, int, int]] = []
    contaminated: set = set()
    for fidx, face in enumerate(faces):
        if unreliable(face):
            contaminated.add(fidx)
            result.distances.append(CandidateDistance(
                face_index=fidx, group=-1, rank=-1, best_reference_angle=None,
                distance=None, eligible=False, contaminated=True))
            continue
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
            if eligible:
                candidates.append((float(d), g, fidx))

    candidates.sort(key=lambda c: c[0])   # greedily assign closest pairs first
    result.candidates = list(candidates)

    claimed_faces: set = set()
    claimed_persons: set = set()
    for d, g, fidx in candidates:
        if fidx in claimed_faces or g in claimed_persons:
            continue
        claimed_faces.add(fidx)
        claimed_persons.add(g)
        src_index = selected_index if single_person else rank[g]
        if 0 <= src_index < num_sources:
            result.pending.append((src_index, fidx))
            result.reasons[fidx] = SWAPPED
        else:
            # A selected person with no source faceset is refused here, never
            # silently redirected to person zero -- that redirect was the bug.
            result.reasons[fidx] = REFUSED_NO_SOURCE

    paired = {fidx for _d, _g, fidx in candidates}
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
             f" swapped={result.swapped_face_indices()}")
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
