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


def normalize_target_selection(selection=None, person_count=None,
                               target_person_ids=None):
    """Return the canonical target-selection state.

    ``person_id`` and ``person_ids`` address stable target-person ids when the
    current context supplies them, and legacy display ranks otherwise.  They
    never address source facesets, target reference angles, detections, or
    media entries.  When ``person_count`` is supplied, an out-of-range person
    is invalid rather than being clamped or redirected to person zero.
    """
    raw = selection if isinstance(selection, Mapping) else {}
    mode = raw.get("selection_mode")
    if mode not in (SELECTION_SELECTED, SELECTION_MULTI_PERSON):
        mode = SELECTION_NONE

    stable_ids = [str(value) for value in (target_person_ids or [])]
    if not stable_ids and target_person_ids is None:
        # No universe supplied.  A selection that already addresses people by
        # stable id must be judged on that contract, not parsed as integer
        # ranks: every re-normalization below the API boundary (ProcessOptions,
        # ProcessMgr) used to turn "tp_..." into a missing rank, report
        # selection_required, and swap nothing -- in preview AND render.
        carried = [raw.get("person_id")] + list(
            raw.get("person_ids") if isinstance(raw.get("person_ids"), list) else [])
        stable_ids = [str(value) for value in carried
                      if value not in (None, "") and not isinstance(value, bool)
                      and _optional_int(value) is None]
        stable_ids = list(dict.fromkeys(stable_ids))
    use_stable_ids = bool(stable_ids)
    raw_person_id = raw.get("person_id")
    if use_stable_ids:
        person_id = (str(raw_person_id).strip()
                     if raw_person_id not in (None, "") else None)
    else:
        person_id = _optional_int(raw_person_id)
    raw_person_ids = raw.get("person_ids")
    person_ids = []
    if isinstance(raw_person_ids, list):
        for value in raw_person_ids:
            parsed = (str(value).strip() if use_stable_ids and value not in (None, "")
                      else _optional_int(value))
            if parsed is not None and parsed not in person_ids:
                person_ids.append(parsed)

    diagnostic = None
    if mode == SELECTION_SELECTED:
        if person_id is None:
            diagnostic = "selection_required"
            person_ids = []
        elif ((not use_stable_ids and person_id < 0)
              or (use_stable_ids and person_id not in stable_ids)):
            diagnostic = "invalid_person_id"
            person_ids = []
        else:
            person_ids = [person_id]
    elif mode == SELECTION_MULTI_PERSON:
        if not person_ids:
            diagnostic = "selection_required"
        elif any((value not in stable_ids if use_stable_ids else value < 0)
                 for value in person_ids):
            diagnostic = "invalid_person_id"
    else:
        person_id = None
        person_ids = []

    if diagnostic is None and person_count is not None and person_ids:
        if use_stable_ids:
            invalid = any(value not in stable_ids for value in person_ids)
        else:
            invalid = any(value < 0 or value >= int(person_count)
                          for value in person_ids)
        if invalid:
            diagnostic = "invalid_person_id"

    if diagnostic is not None:
        person_id = None
        person_ids = []

    # Preserve a structured validation failure created by an API boundary
    # (for example an invalid reference-face id) when the normalized contract
    # is passed through the common normalizer a second time.
    if raw.get("valid") is False and raw.get("diagnostic"):
        diagnostic = str(raw["diagnostic"])
        person_id = None
        person_ids = []

    return {
        "selection_mode": mode,
        "person_id": person_id,
        "person_ids": person_ids,
        "target_reference_face_id": (
            str(raw.get("target_reference_face_id")).strip()
            if raw.get("target_reference_face_id") not in (None, "") else None),
        "target_reference_index": _optional_int(raw.get("target_reference_index")),
        "target_detection_index": _optional_int(raw.get("target_detection_index")),
        "track_id": _optional_int(raw.get("track_id")),
        "target_media_index": _optional_int(raw.get("target_media_index")),
        "valid": diagnostic is None,
        "diagnostic": diagnostic,
    }


def selection_diagnostic_for_mode(mode, selection, target_count,
                                  target_person_ids=None):
    """Return the admission diagnostic for a selected-mode request.

    ``target_person_ids`` must be the active context's stable ids whenever the
    selection addresses people by stable id.  Re-normalizing a stable-id
    selection without them parsed ``"tp_..."`` as a missing integer rank and
    reported ``selection_required`` for every valid Stage 13 selection, so no
    "Selected face" preview or render could ever be admitted.
    """
    stable_ids = [str(value) for value in (target_person_ids or [])]
    if not stable_ids and isinstance(selection, Mapping):
        # No context supplied: accept the ids the selection itself carries so
        # a stable-id selection is judged on its own contract, not as ranks.
        carried = [selection.get("person_id")] + list(selection.get("person_ids") or [])
        stable_ids = [str(value) for value in carried
                      if value not in (None, "") and _optional_int(value) is None]
    selection = normalize_target_selection(
        selection, target_person_ids=stable_ids or None)
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


def selection_group_ids(target_groups, selection, target_person_ids=None):
    """Map stable persons (or legacy ranks) to runtime group ids.

    The stable id list is normally one entry per display person.  Accept a
    parallel per-angle list as a compatibility input too, but always derive
    the group owner from the angle's id; never use an angle index as a person
    rank when a person has multiple references.
    """
    groups = list(target_groups or [])
    unique = sorted(set(groups))
    rank_by_group = {group: rank for rank, group in enumerate(unique)}
    stable_ids = [str(value) for value in (target_person_ids or [])]
    selection = normalize_target_selection(
        selection, person_count=len(unique), target_person_ids=stable_ids)
    wanted = set(selection.get("person_ids", [])) if selection.get("valid") else set()
    if stable_ids:
        if len(stable_ids) == len(groups):
            group_person = {}
            for group, stable_id in zip(groups, stable_ids):
                group_person.setdefault(group, stable_id)
            return {group for group, person_id in group_person.items()
                    if person_id in wanted}
        wanted = {rank for rank, stable_id in enumerate(stable_ids)
                  if stable_id in wanted}
    return {group for group, rank in rank_by_group.items() if rank in wanted}


def resolve_processing_selection(options, target_face_count):
    """ProcessMgr's selection setup, as one pure function.

    Returns ``(target_face_groups, target_selection, selected_group_ids)``
    from the options' canonical request (or the legacy globals fallback the
    caller supplies through ``options``).  Lifted out of
    ``ProcessMgr.initialize`` so the Stage 15 regression -- a stable-id
    selection that normalized to ``selection_required`` and selected nobody --
    is tested against the code that actually runs, not a copy of it.
    """
    request = getattr(options, "processing_request", None)
    request = request if isinstance(request, dict) else None
    request_groups = request.get("target_groups") if request else None
    stable_ids = request.get("target_person_ids") if request else None
    if isinstance(request_groups, list) and len(request_groups) == target_face_count:
        groups = list(request_groups)
    else:
        groups = list(getattr(options, "legacy_target_face_groups", None) or [])
    if len(groups) != target_face_count:
        groups = list(range(target_face_count))
    selection = normalize_target_selection(
        getattr(options, "selection_state", None),
        person_count=len(set(groups)),
        target_person_ids=stable_ids or None,
    )
    selected = selection_group_ids(groups, selection, stable_ids or None)
    return groups, selection, selected


def selection_face_indices(target_groups, selection, target_person_ids=None):
    """Return every captured angle index belonging to the selected person(s)."""
    allowed = selection_group_ids(target_groups, selection, target_person_ids)
    return [index for index, group in enumerate(target_groups or []) if group in allowed]
