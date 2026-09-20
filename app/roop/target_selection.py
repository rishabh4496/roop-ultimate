"""Canonical target-person selection state shared by API and ProcessMgr.

The UI has several different indices in flight: source-gallery entries, target
reference angles, target media, current-frame detections, and tracked faces.
This module gives the swap path one explicit, serializable contract so none of
those integers can be silently substituted for a target person id.
"""

from collections.abc import Mapping


SELECTION_NONE = "none"
SELECTION_SELECTED = "selected"
SELECTION_MULTI_PERSON = "multi_person"
SELECTION_MODES = frozenset({SELECTION_NONE, SELECTION_SELECTED, SELECTION_MULTI_PERSON})


def _optional_int(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
        return int(number) if number.is_integer() else None
    except (TypeError, ValueError):
        return None


def normalize_target_selection(selection=None, person_count=None):
    """Return the canonical target-selection state.

    ``person_id`` and ``person_ids`` address normalized target-person ranks.
    They never address source facesets, target reference angles, detections, or
    media entries.  When ``person_count`` is supplied, an out-of-range person
    is invalid rather than being clamped or redirected to person zero.
    """
    raw = selection if isinstance(selection, Mapping) else {}
    mode = raw.get("selection_mode")
    if mode not in (SELECTION_SELECTED, SELECTION_MULTI_PERSON):
        mode = SELECTION_NONE

    person_id = _optional_int(raw.get("person_id"))
    raw_person_ids = raw.get("person_ids")
    person_ids = []
    if isinstance(raw_person_ids, list):
        for value in raw_person_ids:
            parsed = _optional_int(value)
            if parsed is not None and parsed not in person_ids:
                person_ids.append(parsed)

    diagnostic = None
    if mode == SELECTION_SELECTED:
        if person_id is None:
            diagnostic = "selection_required"
            person_ids = []
        elif person_id < 0:
            diagnostic = "invalid_person_id"
            person_ids = []
        else:
            person_ids = [person_id]
    elif mode == SELECTION_MULTI_PERSON:
        if not person_ids:
            diagnostic = "selection_required"
        elif any(value < 0 for value in person_ids):
            diagnostic = "invalid_person_id"
    else:
        person_id = None
        person_ids = []

    if diagnostic is None and person_count is not None and person_ids:
        if any(value < 0 or value >= int(person_count) for value in person_ids):
            diagnostic = "invalid_person_id"

    if diagnostic is not None:
        person_id = None
        person_ids = []

    return {
        "selection_mode": mode,
        "person_id": person_id,
        "person_ids": person_ids,
        "target_reference_index": _optional_int(raw.get("target_reference_index")),
        "target_detection_index": _optional_int(raw.get("target_detection_index")),
        "track_id": _optional_int(raw.get("track_id")),
        "target_media_index": _optional_int(raw.get("target_media_index")),
        "valid": diagnostic is None,
        "diagnostic": diagnostic,
    }


def selection_diagnostic_for_mode(mode, selection, target_count):
    """Return the admission diagnostic for a selected-mode request."""
    selection = normalize_target_selection(selection)
    if mode not in (SELECTION_SELECTED, "selected_multi"):
        return None
    if int(target_count or 0) < 1:
        return "target_required"
    expected_mode = (SELECTION_SELECTED
                     if mode == SELECTION_SELECTED else SELECTION_MULTI_PERSON)
    if selection.get("selection_mode") != expected_mode:
        return "selection_required"
    if not selection.get("valid"):
        return selection.get("diagnostic") or "selection_required"
    return None


def selection_group_ids(target_groups, selection):
    """Map canonical person ranks to the raw target-group ids used by ProcessMgr."""
    groups = list(target_groups or [])
    unique = sorted(set(groups))
    rank_by_group = {group: rank for rank, group in enumerate(unique)}
    selection = normalize_target_selection(selection)
    wanted = set(selection.get("person_ids", [])) if selection.get("valid") else set()
    return {group for group, rank in rank_by_group.items() if rank in wanted}


def selection_face_indices(target_groups, selection):
    """Return every captured angle index belonging to the selected person(s)."""
    allowed = selection_group_ids(target_groups, selection)
    return [index for index, group in enumerate(target_groups or []) if group in allowed]
