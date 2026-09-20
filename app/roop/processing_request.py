"""Canonical, serializable processing request shared by preview and render.

This module deliberately contains no FaceSet or model objects.  It resolves
the identities and indices that decide eligibility once, at the API boundary;
the preview and batch paths then consume the same snapshot.
"""

from uuid import uuid4

from roop.target_selection import normalize_target_selection


SWAP_MODE_LABELS = {
    "Selected face": "selected",
    "Selected people": "selected_multi",
    "First found": "first",
    "All input faces": "all_input",
    "All female": "all_female",
    "All male": "all_male",
}


def normalize_swap_mode(label):
    return SWAP_MODE_LABELS.get(label, "all")


def _optional_int(value, default=None):
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
        return int(number) if number.is_integer() else default
    except (TypeError, ValueError):
        return default


def normalize_source_index_mapping(mapping, source_count, swap_mode):
    """Normalize person->source gallery indices once.

    ``None`` means the mode intentionally uses the gallery position directly
    (currently All input faces), matching the legacy contract.
    """
    if swap_mode == "all_input" or not isinstance(mapping, list) or not mapping:
        return None
    limit = max(0, int(source_count or 0))
    result = []
    for value in mapping:
        index = _optional_int(value, -1)
        result.append(index if 0 <= index < limit else -1)
    return result


def resolve_selected_source_index(source_index_mapping, selected_source_gallery_index):
    """Translate the gallery-selected source into mapped-list coordinates."""
    selected = _optional_int(selected_source_gallery_index, 0)
    if source_index_mapping is None:
        return selected
    for mapped_index, gallery_index in enumerate(source_index_mapping):
        if gallery_index == selected:
            return mapped_index
    return 0


def normalize_processing_request(payload=None, *, target_groups=None,
                                 source_count=0, selected_source_gallery_index=0,
                                 target_media_index=None, request_id=None):
    """Build the one request representation consumed by preview and render."""
    payload = payload if isinstance(payload, dict) else {}
    groups = list(target_groups or [])
    swap_mode = normalize_swap_mode(payload.get("detection", "All faces"))
    selection = normalize_target_selection(
        payload.get("selection_state"),
        person_count=len(set(groups)),
    )
    face_mapping = payload.get("face_mapping")
    source_index_mapping = normalize_source_index_mapping(
        face_mapping, source_count, swap_mode)
    # `face_mapping` is the effective source mapping exposed in the canonical
    # request. All-input mode intentionally has no person-ordered mapping;
    # every other mode uses the validated source-index list.
    normalized_mapping = (
        list(source_index_mapping)
        if source_index_mapping is not None else []
    )
    target_media = _optional_int(target_media_index)
    if target_media is None:
        target_media = _optional_int(payload.get("target_index"))
    if target_media is None:
        target_media = selection.get("target_media_index")
    if target_media is not None:
        # There is one canonical media index. The copy inside selection_state
        # is kept only because that state is the serializable UI contract.
        selection["target_media_index"] = target_media

    return {
        "request_id": str(request_id or payload.get("request_id") or uuid4().hex[:12]),
        "swap_mode": swap_mode,
        "selection_state": selection,
        "target_groups": groups,
        "target_face_count": len(groups),
        "target_person_count": len(set(groups)),
        "target_media_index": target_media,
        "face_mapping": normalized_mapping,
        "source_index_mapping": source_index_mapping,
        "source_face_count": max(0, int(source_count or 0)),
        "selected_source_gallery_index": _optional_int(
            selected_source_gallery_index, 0),
        "source_index": resolve_selected_source_index(
            source_index_mapping, selected_source_gallery_index),
    }


def selection_log_line(request, phase):
    """Return a compact, stable diagnostic line for either route."""
    selection = request.get("selection_state") or {}
    if selection.get("selection_mode") == "multi_person":
        person = selection.get("person_ids", [])
    else:
        person = selection.get("person_id")
    return (
        f"[Selection] phase={phase} request={request.get('request_id')} "
        f"mode={request.get('swap_mode')} person={person} "
        f"target_faces={request.get('target_face_count')} "
        f"source_facesets={request.get('source_face_count')} "
        f"mapping={request.get('face_mapping')} "
        f"source_index={request.get('source_index')} "
        f"target_media={request.get('target_media_index')}"
    )
