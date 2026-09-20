"""Canonical processing selection shared by preview, queue and final render.

Stage 14.  Every processing request (a live preview, a direct render, or a
queued job) carries ONE immutable, serializable description of *what* is
being swapped: which target media, which target person(s), which source
identity, which person->source mapping, and which detection mode.  The
selection is built by the client at request time, snapshotted into a queue
job at queue-creation time, and consumed unchanged by the worker.  Nothing
downstream re-derives target identity from mutable UI or global state.

The module is dependency-free on purpose: the queue runner, the API boundary
and the tests all import it without touching ProcessMgr or FaceSet objects.
"""

from collections.abc import Mapping
import hashlib
import json
from uuid import uuid4


SELECTION_SCHEMA = 1

# The flat legacy request fields each canonical field projects onto.  Older
# clients send only the flat fields; the canonical object is derived from them
# so both generations of client share one normalizer.
_FLAT_DETECTION = "detection"
_FLAT_SELECTION_STATE = "selection_state"
_FLAT_MAPPING = "target_person_source_mapping"
_FLAT_SOURCE_ID = "selected_source_id"
_FLAT_MEDIA_ID = "target_media_id"


def _text(value):
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None


def _version(value):
    """Selection versions are monotonic integers; anything else is unversioned."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number < 0:  # NaN / negative
        return None
    return int(number)


def _mapping(value):
    if not isinstance(value, Mapping):
        return {}
    result = {}
    for key, raw in value.items():
        person = _text(key)
        source = _text(raw)
        if person and source and source not in ("-1",):
            result[person] = source
    return result


def new_request_id():
    return uuid4().hex[:12]


def build_processing_selection(payload=None, *, target_media_id=None,
                               request_id=None):
    """Return the canonical selection for one processing request.

    ``payload["processing_selection"]`` is authoritative when present.  The
    flat legacy fields are the fallback so a client that predates this
    contract still produces the same canonical object.  ``target_media_id``
    and ``request_id`` passed by the API boundary win over both, because the
    boundary has just resolved/activated the media and allocated the id.
    """
    payload = payload if isinstance(payload, Mapping) else {}
    raw = payload.get("processing_selection")
    raw = raw if isinstance(raw, Mapping) else {}

    selection_state = raw.get("selection_state")
    if not isinstance(selection_state, Mapping):
        selection_state = payload.get(_FLAT_SELECTION_STATE)
    selection_state = dict(selection_state) if isinstance(selection_state, Mapping) else {}

    mapping = raw.get("target_person_source_mapping")
    if not isinstance(mapping, Mapping):
        mapping = payload.get(_FLAT_MAPPING)
    mapping = _mapping(mapping)

    detection = _text(raw.get("detection_mode")) or _text(payload.get(_FLAT_DETECTION))

    media_id = (_text(target_media_id) or _text(raw.get("target_media_id"))
                or _text(payload.get(_FLAT_MEDIA_ID)))

    source_identity = (_text(raw.get("source_identity_id"))
                       or _text(payload.get(_FLAT_SOURCE_ID)))

    mode = selection_state.get("selection_mode")
    person_id = _text(raw.get("target_person_id"))
    if person_id is None and mode == "selected":
        person_id = _text(selection_state.get("person_id"))
    person_ids = raw.get("target_person_ids")
    if not isinstance(person_ids, list):
        person_ids = selection_state.get("person_ids")
    person_ids = [value for value in (_text(v) for v in (person_ids or [])) if value]
    if person_id and person_id not in person_ids:
        person_ids = [person_id] + person_ids
    if mode == "selected" and person_id:
        person_ids = [person_id]

    reference_id = (_text(raw.get("target_reference_face_id"))
                    or _text(selection_state.get("target_reference_face_id")))

    version = _version(raw.get("selection_version"))
    if version is None:
        version = _version(payload.get("selection_version"))
    mapping_version = _version(raw.get("mapping_version"))

    rid = (_text(request_id) or _text(raw.get("request_id"))
           or _text(payload.get("request_id")) or new_request_id())

    return {
        "schema": SELECTION_SCHEMA,
        "request_id": rid,
        "selection_version": version,
        "mapping_version": mapping_version,
        "target_media_id": media_id,
        "target_person_id": person_id,
        "target_person_ids": person_ids,
        "target_reference_face_id": reference_id,
        "source_identity_id": source_identity,
        "detection_mode": detection,
        "target_person_source_mapping": mapping,
        "selection_state": selection_state,
    }


def apply_processing_selection(payload, selection):
    """Project the canonical selection onto the flat request fields.

    Returns a NEW dict.  After this, every legacy reader of ``detection``,
    ``selection_state``, ``target_person_source_mapping``,
    ``selected_source_id`` and ``target_media_id`` sees the canonical values,
    so a stale flat field cannot disagree with the selection that was sent.
    """
    result = dict(payload or {})
    result["processing_selection"] = dict(selection)
    result["request_id"] = selection["request_id"]
    if selection.get("detection_mode"):
        result[_FLAT_DETECTION] = selection["detection_mode"]
    if selection.get("selection_state"):
        result[_FLAT_SELECTION_STATE] = dict(selection["selection_state"])
    if selection.get("target_person_source_mapping") or _FLAT_MAPPING in result:
        result[_FLAT_MAPPING] = dict(selection["target_person_source_mapping"])
    if selection.get("source_identity_id"):
        result[_FLAT_SOURCE_ID] = selection["source_identity_id"]
    if selection.get("target_media_id"):
        result[_FLAT_MEDIA_ID] = selection["target_media_id"]
    if selection.get("selection_version") is not None:
        result["selection_version"] = selection["selection_version"]
    return result


def selection_identity_fields(selection):
    """The fields that decide WHAT is processed (no request id / version)."""
    selection = selection if isinstance(selection, Mapping) else {}
    state = selection.get("selection_state") or {}
    return {
        "target_media_id": selection.get("target_media_id"),
        "target_person_id": selection.get("target_person_id"),
        "target_person_ids": list(selection.get("target_person_ids") or []),
        "target_reference_face_id": selection.get("target_reference_face_id"),
        "source_identity_id": selection.get("source_identity_id"),
        "detection_mode": selection.get("detection_mode"),
        "target_person_source_mapping": dict(
            selection.get("target_person_source_mapping") or {}),
        "selection_mode": state.get("selection_mode"),
    }


def selection_signature(selection, *, frame=None, fake=None, context=None,
                        extra=None):
    """Short stable hash of the identity fields plus the frame coordinates.

    Two requests with the same signature ask for the same picture from the
    same target context.  The client stores it beside each cached preview and
    compares it with the signature of the selection it currently wants; a
    response whose signature is not the wanted one is stale by definition.
    """
    body = {
        "identity": selection_identity_fields(selection),
        "frame": frame,
        "fake": bool(fake) if fake is not None else None,
        "context": context,
        "extra": extra,
    }
    encoded = json.dumps(body, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]


def selection_invalidation_reasons(selection, *, target_media_ids=None,
                                   target_person_ids=None,
                                   reference_face_ids=None,
                                   source_identity_ids=None):
    """Explain why a stored selection can no longer be processed as written.

    Used at queue dispatch.  An empty list means every identity the job was
    created with still exists.  Anything else means the job must be failed
    explicitly rather than rendered with a silently different mapping.  A
    ``None`` universe skips that check (the caller has no such state).
    """
    selection = selection if isinstance(selection, Mapping) else {}
    reasons = []

    media = selection.get("target_media_id")
    if target_media_ids is not None and media and media not in set(map(str, target_media_ids)):
        reasons.append(f"target media {media} is no longer loaded")

    people = set(map(str, target_person_ids)) if target_person_ids is not None else None
    if people is not None:
        for person in selection.get("target_person_ids") or []:
            if person not in people:
                reasons.append(f"target person {person} was removed")
        for person in (selection.get("target_person_source_mapping") or {}):
            if person not in people and person not in (selection.get("target_person_ids") or []):
                reasons.append(f"mapped target person {person} was removed")

    references = (set(map(str, reference_face_ids))
                  if reference_face_ids is not None else None)
    reference = selection.get("target_reference_face_id")
    if references is not None and reference and reference not in references:
        reasons.append(f"target reference face {reference} was removed")

    sources = (set(map(str, source_identity_ids))
               if source_identity_ids is not None else None)
    if sources is not None:
        selected = selection.get("source_identity_id")
        if selected and selected not in sources:
            reasons.append(f"source {selected} was removed")
        for person, source in (selection.get("target_person_source_mapping") or {}).items():
            if source not in sources and f"source {source} was removed" not in reasons:
                reasons.append(f"source {source} mapped to {person} was removed")
    return reasons


def selection_diagnostics(selection, *, preview_signature=None):
    """Compact, embedding-free fields for logs and API responses."""
    selection = selection if isinstance(selection, Mapping) else {}
    return {
        "request_id": selection.get("request_id"),
        "target_media_id": selection.get("target_media_id"),
        "target_person_id": selection.get("target_person_id"),
        "source_identity_id": selection.get("source_identity_id"),
        "selection_version": selection.get("selection_version"),
        "preview_signature": preview_signature,
    }


def is_stale_version(incoming, current):
    """True when ``incoming`` is an older selection version than ``current``.

    Unversioned writes (legacy clients) are never considered stale.
    """
    incoming = _version(incoming)
    current = _version(current)
    if incoming is None or current is None:
        return False
    return incoming < current
