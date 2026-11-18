# SPDX-License-Identifier: MIT
"""Per-target-person render overrides.

A *person* is one entry of the Face Routing dock: a target face the user mapped to a source. Until now every face of a
render shared one set of paste settings; the only per-anything channels were a frame RANGE (scene overrides, timeline
keyframes) and a track-to-faceset binding. This is the per-PERSON one.

WIRE FORMAT (request ``person_overrides``)::

    {"<target person id>": {"mask_feather": 30, "color_match_strength": 0.4, ...}, ...}

Each value is validated against `PARAMETERS` and a bad key or a value out of range REJECTS the whole request
(`PersonOverrideError`) - nothing is clamped or dropped silently, the same rule `timeline_state` follows.

WHICH FACES THEY REACH. ``ProcessMgr.process_face(face_index, ...)`` receives the person's RANK as ``face_index`` in the
two modes that route by person - ``selected`` and ``selected_multi`` (the mapped source list is person-ordered, and
`processing_request` makes a lone selected person's slot its own rank; see the comments there). Every other mode ("All
faces", "First found", "All input faces", gender) has no per-person identity at the paste, so overrides are NOT applied
there and `resolve` says so (``applied: False`` plus the reason) instead of accepting values that nothing reads.

WHAT THEY CHANGE. Each name is one the render already reads, per face, at the paste:

    mask_feather            the paste feather, in `face_mask_blend` units       procmgr_masking.paste_upscale (_tl)
    mask_erode_dilate_radius  signed crop-pixel contour move of the landmark hull  procmgr_masking.create_landmark_mask
    color_match_strength    skin-class Lab match at the seam (replaces skin_lab_match)  paste_upscale (_tl)
    blend_ratio             how much of the restorer's face is mixed over the swap     paste_upscale (options.blend_ratio)
    restore_identity_weight CodeFormer / UltraMax fidelity weight (alias codeformer_fidelity)  the restorers, via value()

``restore_identity_weight`` only means something to a restorer that reads `codeformer_fidelity` (CodeFormer, UltraMax);
the others ignore it, exactly as they ignore the global setting.

PRECEDENCE: run settings < person override < a keyframe at this frame (`studio_parameters.person_scope`).
"""
from __future__ import annotations

import math
from functools import wraps
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: name -> (lo, hi, integer). The ranges are the UI sliders' own (`face_mask_blend` 0..200, erode/dilate +-32) where the
#: control exists, and the setting's meaningful range elsewhere.
PARAMETERS: Dict[str, Tuple[float, float, bool]] = {
    "mask_feather": (0.0, 200.0, False),
    "mask_erode_dilate_radius": (-64.0, 64.0, True),
    "color_match_strength": (0.0, 1.0, False),
    "blend_ratio": (0.0, 1.0, False),
    "restore_identity_weight": (0.0, 1.0, False),
}

#: The swap modes whose ``process_face(face_index=...)`` is the person's rank.
PERSON_ROUTED_MODES = frozenset({"selected", "selected_multi"})


class PersonOverrideError(ValueError):
    """The request carried an override this module will not apply."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def validate_values(values: Any, label: str = "person") -> Dict[str, float]:
    """One person's ``{name: number}`` checked against `PARAMETERS`; raises `PersonOverrideError`."""
    if not isinstance(values, Mapping):
        raise PersonOverrideError([f"{label}: overrides must be an object"])
    errors: List[str] = []
    clean: Dict[str, float] = {}
    for name, raw in values.items():
        spec = PARAMETERS.get(name)
        if spec is None:
            errors.append(f"{label}: unknown override {name!r} (known: {', '.join(sorted(PARAMETERS))})")
            continue
        lo, hi, integer = spec
        number = _number(raw)
        if number is None:
            errors.append(f"{label}: {name} must be a finite number, got {raw!r}")
        elif not lo <= number <= hi:
            errors.append(f"{label}: {name} must be within [{lo:g}, {hi:g}], got {number:g}")
        elif integer and not float(number).is_integer():
            errors.append(f"{label}: {name} must be a whole number, got {number:g}")
        else:
            clean[name] = int(number) if integer else float(number)
    if errors:
        raise PersonOverrideError(errors)
    return clean


def resolve(payload_overrides: Any, person_ids: Sequence[Any], swap_mode: str) -> Dict[str, Any]:
    """Turn the wire ``person_overrides`` into what a render consumes.

    Returns ``{"by_rank": {rank: {name: value}}, "applied": bool, "reason": str|None, "unknown_people": [...]}``.
    ``by_rank`` is empty - and ``applied`` False - when the mode cannot route by person, but the VALUES are still
    validated, so a bad payload is a 400 in every mode rather than only in the ones that would have used it.
    A person id the current target bank does not have is reported in ``unknown_people`` and skipped (a clip change can
    leave stale ids in a saved project; they must neither crash the render nor land on somebody else).
    """
    out: Dict[str, Any] = {"by_rank": {}, "applied": False, "reason": None, "unknown_people": []}
    if payload_overrides in (None, {}, []):
        out["reason"] = None
        return out
    if not isinstance(payload_overrides, Mapping):
        raise PersonOverrideError(["person_overrides must be an object keyed by target person id"])
    ids = [str(p) for p in (person_ids or [])]
    errors: List[str] = []
    by_rank: Dict[int, Dict[str, float]] = {}
    for person, values in payload_overrides.items():
        try:
            clean = validate_values(values, label=f"person {person}")
        except PersonOverrideError as error:
            errors.extend(error.errors)
            continue
        if not clean:
            continue
        key = str(person)
        if key not in ids:
            out["unknown_people"].append(key)
            continue
        by_rank[ids.index(key)] = clean
    if errors:
        raise PersonOverrideError(errors)
    if swap_mode not in PERSON_ROUTED_MODES:
        out["reason"] = (f"swap mode {swap_mode!r} does not route by person, so per-person overrides are not applied; "
                         "use 'Selected face' or 'Selected people'")
        return out
    out["by_rank"] = by_rank
    out["applied"] = bool(by_rank)
    return out


def for_face(options: Any, face_index: Any) -> Dict[str, float]:
    """The overrides for the person whose rank is ``face_index`` in this run (``{}`` almost always: a free lookup)."""
    request = getattr(options, "processing_request", None)
    if not isinstance(request, dict):
        return {}
    table = request.get("person_overrides")
    if not table:
        return {}
    try:
        rank = int(face_index)
    except (TypeError, ValueError):
        return {}
    # A request that went through JSON (a queued job, a saved project) has string keys.
    return table.get(rank) or table.get(str(rank)) or {}


def scoped_face(method):
    """Decorator for ``ProcessMgr.process_face``: run it inside its person's `studio_parameters.person_scope`."""
    @wraps(method)
    def run(self, face_index, *args, **kwargs):
        values = for_face(getattr(self, "options", None), face_index)
        if not values:
            return method(self, face_index, *args, **kwargs)
        from roop import studio_parameters
        with studio_parameters.person_scope(values):
            # Counted so a render report / test can show the scope RAN, not just that it was defined.
            self.person_override_faces = getattr(self, "person_override_faces", 0) + 1
            return method(self, face_index, *args, **kwargs)
    return run
