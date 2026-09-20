"""Identity-bearing state for captured target people.

The face arrays used by the legacy processing core are ordered collections.  A
collection position (or the integer group written beside it) is therefore not
an identity.  This module contains the small, dependency-free identity layer
used by the API and the React contract: opaque person ids, opaque reference
angle ids, and deterministic display projections.
"""

from __future__ import annotations

from collections.abc import Mapping
from uuid import uuid4


def new_target_person_id() -> str:
    return f"tp_{uuid4().hex}"


def new_target_reference_face_id() -> str:
    return f"tr_{uuid4().hex}"


def _valid_id(value, prefix):
    text = str(value or "").strip()
    return text if text.startswith(prefix) and len(text) > len(prefix) else None


def normalize_parallel_ids(person_ids, reference_ids, face_count, groups=None):
    """Return stable parallel person/reference ids, migrating old group state.

    Existing valid ids are preserved byte-for-byte.  Legacy group values are
    used only once to migrate a loaded context; they are never returned as the
    new identity.  First appearance is the display order, so deleting or
    reordering an angle cannot rename another person.
    """
    count = max(0, int(face_count or 0))
    old_people = list(person_ids or []) if isinstance(person_ids, (list, tuple)) else []
    old_refs = list(reference_ids or []) if isinstance(reference_ids, (list, tuple)) else []
    old_groups = list(groups or []) if isinstance(groups, (list, tuple)) else []
    by_legacy_group = {}
    people = []
    refs = []
    for index in range(count):
        person = _valid_id(old_people[index], "tp_") if index < len(old_people) else None
        if person is None:
            legacy = old_groups[index] if index < len(old_groups) else index
            key = str(legacy)
            person = by_legacy_group.get(key)
            if person is None:
                person = new_target_person_id()
                by_legacy_group[key] = person
        reference = (_valid_id(old_refs[index], "tr_")
                     if index < len(old_refs) else None)
        refs.append(reference or new_target_reference_face_id())
        people.append(person)
    return people, refs


def person_records(person_ids, reference_ids=None, names=None, mapping=None):
    """Build the UI projection in first-appearance order.

    ``display_rank`` is deliberately derived and disposable.  Callers must
    persist and send ``target_person_id`` instead.
    """
    people = list(person_ids or [])
    refs = list(reference_ids or [])
    names = names if isinstance(names, Mapping) else {}
    mapping = mapping if isinstance(mapping, Mapping) else {}
    order = []
    for person in people:
        if person not in order:
            order.append(person)
    result = []
    for rank, person in enumerate(order):
        indices = [i for i, value in enumerate(people) if value == person]
        result.append({
            "target_person_id": person,
            "display_rank": rank,
            "face_indices": indices,
            "reference_face_ids": [refs[i] for i in indices if i < len(refs)],
            "name": str(names.get(person, "") or ""),
            "source_identity_id": mapping.get(person),
        })
    return result


def person_id_for_rank(records, rank):
    try:
        index = int(rank)
    except (TypeError, ValueError):
        return None
    return records[index]["target_person_id"] if 0 <= index < len(records) else None


def rank_for_person_id(records, person_id):
    for record in records:
        if str(record.get("target_person_id")) == str(person_id):
            return int(record["display_rank"])
    return None


def person_id_for_face_index(person_ids, index):
    try:
        index = int(index)
    except (TypeError, ValueError):
        return None
    return person_ids[index] if 0 <= index < len(person_ids) else None
