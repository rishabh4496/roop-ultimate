"""Canonical, serializable processing request shared by preview and render.

This module deliberately contains no FaceSet or model objects.  It resolves
the identities and indices that decide eligibility once, at the API boundary;
the preview and batch paths then consume the same snapshot.
"""

from uuid import uuid4

from roop.processing_selection import (
    apply_processing_selection,
    build_processing_selection,
    selection_diagnostics,
)
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


def source_index_mapping_errors(mapping, source_count, swap_mode):
    """Return positions containing malformed source mappings.

    ``-1`` is the explicit, serializable skip value and is therefore not an
    error.  Everything else must be an integer source-gallery index that is
    present in the current gallery.  Keeping this diagnostic separate from
    normalization lets callers skip safely without turning bad data into
    source 0.
    """
    if swap_mode == "all_input" or not isinstance(mapping, list) or not mapping:
        return []
    limit = max(0, int(source_count or 0))
    errors = []
    for position, value in enumerate(mapping):
        parsed = _optional_int(value, None)
        if parsed is None or parsed < -1 or parsed >= limit:
            errors.append(position)
    return errors


def resolve_source_mapping_names(mapping, mapping_names, current_names,
                                 source_count, swap_mode):
    """Resolve queued source names against the current gallery order.

    Names are optional compatibility metadata.  When present they are
    authoritative, so a source move cannot make a target person inherit the
    source that slid into its old numeric slot.  ``None`` remains the explicit
    skip value.  Payloads from older clients continue to use numeric indices.
    """
    if (swap_mode == "all_input" or not isinstance(mapping, list) or not mapping
            or not isinstance(mapping_names, list)
            or len(mapping_names) != len(mapping)):
        return normalize_source_index_mapping(mapping, source_count, swap_mode)
    names = {
        str(name).strip().casefold(): index
        for index, name in enumerate(current_names or [])
        if str(name or "").strip()
    }
    result = []
    for raw, name in zip(mapping, mapping_names):
        if name is None or str(name).strip() == "":
            parsed = _optional_int(raw, -1)
            result.append(parsed if parsed == -1 else -1)
            continue
        result.append(names.get(str(name).strip().casefold(), -1))
    return result


def resolve_source_mapping_ids(mapping, mapping_ids, current_ids,
                               source_count, swap_mode):
    """Resolve source bindings by stable faceset identity.

    ``source_mapping_ids`` is preferred over display names because two uploaded
    faces can legitimately come from the same filename.  Numeric indices remain
    the compatibility fallback when older payloads have neither identity field.
    """
    if (swap_mode == "all_input" or not isinstance(mapping, list) or not mapping
            or not isinstance(mapping_ids, list)
            or len(mapping_ids) != len(mapping)):
        return None
    ids = {
        str(source_id).strip().casefold(): index
        for index, source_id in enumerate(current_ids or [])
        if str(source_id or "").strip()
    }
    result = []
    for raw, source_id in zip(mapping, mapping_ids):
        if source_id is None or str(source_id).strip() == "":
            parsed = _optional_int(raw, -1)
            result.append(parsed if parsed == -1 else -1)
            continue
        result.append(ids.get(str(source_id).strip().casefold(), -1))
    return result


def resolve_selected_source_name(selected_index, selected_name, current_names):
    """Resolve the selected source identity after a gallery reorder."""
    name = str(selected_name or "").strip().casefold()
    if name:
        for index, current in enumerate(current_names or []):
            if name == str(current or "").strip().casefold():
                return index
        return -1
    return _optional_int(selected_index, -1)


def resolve_selected_source_identity(selected_index, selected_id, selected_name,
                                     current_ids, current_names):
    """Resolve the selected gallery source by stable id, then display name."""
    identity = str(selected_id or "").strip().casefold()
    if identity:
        for index, current in enumerate(current_ids or []):
            if identity == str(current or "").strip().casefold():
                return index
        return -1
    return resolve_selected_source_name(selected_index, selected_name, current_names)


def remap_source_index_mapping_after_removal(mapping, removed_index):
    """Keep person->source bindings stable after a gallery removal.

    The removed source becomes an explicit skip.  Later source indices shift
    down because the gallery list compacts, but a person never silently starts
    using the source that slid into the deleted slot.
    """
    if not isinstance(mapping, list):
        return mapping
    removed = _optional_int(removed_index, None)
    if removed is None or removed < 0:
        return list(mapping)
    result = []
    for value in mapping:
        source = _optional_int(value, None)
        if source is None or source < 0:
            result.append(-1)
        elif source == removed:
            result.append(-1)
        elif source > removed:
            result.append(source - 1)
        else:
            result.append(source)
    return result


def remap_source_index_mapping_after_move(mapping, from_index, to_index):
    """Keep person->source bindings stable after moving a source gallery item."""
    if not isinstance(mapping, list):
        return mapping
    source_from = _optional_int(from_index, None)
    source_to = _optional_int(to_index, None)
    if (source_from is None or source_to is None or source_from < 0
            or source_to < 0 or source_from == source_to):
        return list(mapping)

    def moved_index(value):
        source = _optional_int(value, None)
        if source is None or source < 0:
            return -1
        if source == source_from:
            return source_to
        if source_from < source_to and source_from < source <= source_to:
            return source - 1
        if source_to < source_from and source_to <= source < source_from:
            return source + 1
        return source

    return [moved_index(value) for value in mapping]


def resolve_selected_source_index(source_index_mapping, selected_source_gallery_index):
    """Translate the gallery-selected source into mapped-list coordinates."""
    if source_index_mapping is None:
        return _optional_int(selected_source_gallery_index, 0)
    selected = _optional_int(selected_source_gallery_index, -1)
    if selected < 0:
        return -1
    for mapped_index, gallery_index in enumerate(source_index_mapping):
        if gallery_index == selected:
            return mapped_index
    # No mapped person owns this source.  Returning zero here used to silently
    # redirect an invalid/removed source to the first target person.
    return -1


def normalize_processing_request(payload=None, *, target_groups=None,
                                 source_count=0, selected_source_gallery_index=0,
                                 target_media_index=None, request_id=None,
                                 target_media_id=None,
                                 current_source_names=None,
                                 current_source_ids=None,
                                 target_person_ids=None,
                                 processing_selection=None):
    """Build the one request representation consumed by preview and render."""
    payload = payload if isinstance(payload, dict) else {}
    # Stage 14: the canonical selection is authoritative over the flat fields.
    # Building it here (when the API boundary has not already) means a direct
    # caller and the API produce the same request from the same payload.
    if not isinstance(processing_selection, dict):
        processing_selection = build_processing_selection(
            payload, target_media_id=target_media_id, request_id=request_id)
    payload = apply_processing_selection(payload, processing_selection)
    request_id = processing_selection["request_id"]
    if target_media_id is None:
        target_media_id = processing_selection.get("target_media_id")
    groups = list(target_groups or [])
    # The UI may send the parallel per-angle id array while the API normally
    # supplies the already-projected per-person list. Canonical requests use
    # first-appearance person order in either case; no angle index becomes an
    # identity by accident.
    stable_person_ids = []
    for value in (target_person_ids or []):
        value = str(value)
        if value not in stable_person_ids:
            stable_person_ids.append(value)
    swap_mode = normalize_swap_mode(payload.get("detection", "All faces"))
    selection = normalize_target_selection(
        payload.get("selection_state"),
        person_count=len(set(groups)),
        target_person_ids=stable_person_ids,
    )
    stable_mapping = payload.get("target_person_source_mapping")
    if (not isinstance(stable_mapping, dict)
            and isinstance(payload.get("face_mapping"), dict)
            and stable_person_ids):
        stable_mapping = payload.get("face_mapping")
    if isinstance(stable_mapping, dict) and stable_person_ids:
        face_mapping = [stable_mapping.get(person_id, -1)
                        for person_id in stable_person_ids]
        source_mapping_ids = [stable_mapping.get(person_id)
                              for person_id in stable_person_ids]
    else:
        face_mapping = payload.get("face_mapping")
        source_mapping_ids = payload.get("source_mapping_ids")
    source_mapping_names = payload.get("source_mapping_names")
    source_index_mapping = resolve_source_mapping_ids(
        face_mapping, source_mapping_ids, current_source_ids,
        source_count, swap_mode)
    if source_index_mapping is None:
        source_index_mapping = resolve_source_mapping_names(
            face_mapping, source_mapping_names, current_source_names,
            source_count, swap_mode)
    mapping_errors = source_index_mapping_errors(
        source_index_mapping, source_count, swap_mode)
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

    selected_source_gallery_index = resolve_selected_source_identity(
        selected_source_gallery_index,
        payload.get("selected_source_id"),
        payload.get("selected_source_name"),
        current_source_ids,
        current_source_names,
    )

    source_index = resolve_selected_source_index(
        source_index_mapping, selected_source_gallery_index)
    # Single-person "Selected face": the source is the one MAPPED to the
    # selected person.  `source_index` used to be re-resolved from the
    # gallery-highlighted source, so highlighting source A while person P was
    # mapped to source B swapped P with A when another person happened to map
    # there, or refused P outright (-1) when nobody did.  ProcessMgr's
    # single-person path reads `selected_index` as the mapped-list slot, so
    # that slot must be the selected person's own rank.
    #
    # The SAME slot rule applies to "Selected people" when exactly ONE person
    # is mapped: ProcessMgr decides `single_person = len(persons) <= 1` from
    # the selected set alone, not from the mode, and then reads
    # `selected_index` instead of the person's rank. With two captured people
    # and only the SECOND one mapped, the gallery-resolved index was -1 (source
    # 0 is mapped to nobody), so that person was refused with "no source
    # faceset" while the first person -- the one the user had NOT mapped --
    # was the only one that could ever swap. Measured on d2.mp4: A-only swapped
    # A, B-only swapped nothing.
    single_person = None
    if selection.get("valid"):
        if swap_mode == "selected" and selection.get("person_id") is not None:
            single_person = selection["person_id"]
        elif swap_mode == "selected_multi":
            people = [p for p in (selection.get("person_ids") or []) if p is not None]
            if len(people) == 1:
                single_person = people[0]
    if single_person is not None and source_index_mapping is not None:
        person = single_person
        if stable_person_ids and str(person) in stable_person_ids:
            rank = stable_person_ids.index(str(person))
        else:
            rank = _optional_int(person, -1)
        if 0 <= rank < len(source_index_mapping):
            source_index = rank if source_index_mapping[rank] >= 0 else -1
        else:
            source_index = -1

    resolved_mapping_ids = [
        (current_source_ids[index]
         if isinstance(current_source_ids, list) and 0 <= index < len(current_source_ids)
         else None)
        for index in (source_index_mapping or [])
    ] if source_index_mapping is not None else None

    return {
        "request_id": str(request_id or payload.get("request_id") or uuid4().hex[:12]),
        "processing_selection": dict(processing_selection),
        "selection_version": processing_selection.get("selection_version"),
        "swap_mode": swap_mode,
        "selection_state": selection,
        "target_groups": groups,
        "target_face_count": len(groups),
        "target_person_count": len(set(groups)),
        "target_person_ids": stable_person_ids,
        "target_person_source_mapping": (
            dict(stable_mapping) if isinstance(stable_mapping, dict) else {}),
        "target_media_index": target_media,
        "target_media_id": (str(target_media_id)
                            if target_media_id is not None else
                            (str(payload.get("target_media_id"))
                             if payload.get("target_media_id") else None)),
        "face_mapping": normalized_mapping,
        "source_index_mapping": source_index_mapping,
        "source_mapping_ids": resolved_mapping_ids,
        "source_mapping_errors": mapping_errors,
        "source_face_count": max(0, int(source_count or 0)),
        "selected_source_gallery_index": _optional_int(
            selected_source_gallery_index, -1),
        "source_index": source_index,
    }


def selection_log_line(request, phase, preview_signature=None):
    """Return a compact, stable diagnostic line for either route.

    Carries the Stage 14 diagnostic fields (request id, target media id,
    target person id, source identity id, selection version, preview
    signature) and never an embedding or face payload.
    """
    selection = request.get("selection_state") or {}
    if selection.get("selection_mode") == "multi_person":
        person = selection.get("person_ids", [])
    else:
        person = selection.get("person_id")
    diag = selection_diagnostics(request.get("processing_selection"),
                                 preview_signature=preview_signature)
    line = (
        f"[Selection] phase={phase} request={request.get('request_id')} "
        f"mode={request.get('swap_mode')} person={person} "
        f"target_faces={request.get('target_face_count')} "
        f"source_facesets={request.get('source_face_count')} "
        f"mapping={request.get('face_mapping')} "
        f"source_index={request.get('source_index')} "
        f"target_media={request.get('target_media_index')} "
        f"target_media_id={request.get('target_media_id')} "
        f"target_person_id={diag['target_person_id']} "
        f"source_identity_id={diag['source_identity_id']} "
        f"selection_version={diag['selection_version']} "
        f"preview_signature={diag['preview_signature']}"
    )
    errors = request.get("source_mapping_errors") or []
    return f"{line} mapping_errors={errors}" if errors else line
