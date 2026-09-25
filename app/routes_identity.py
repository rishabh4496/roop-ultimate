"""Identity Blender endpoints: latent blend recipe + attribute dials.

The recipe is runtime state (``roop.globals.identity_blend``), like the source
gallery it names.  It reaches a render two ways:

* ``POST /api/identity/blend`` sets it live (the React dock does this on every
  change, debounced), and
* ``identity_blend`` inside a ``/api/preview`` or ``/api/swap`` payload, via
  ``apply_identity_blend_from_payload`` -- which is what freezes it into a
  queued job, because the queue stores the swap payload verbatim.

The maths is in ``roop.identity_algebra``; the render hook is in
``ProcessMgr`` right after the V2 pose-embedding selection.
"""
from fastapi import APIRouter, Body

import roop.globals as roop_globals
from roop.identity_algebra import (AGE_LIMIT_YEARS, DEFAULT_MIN_IDENTITY_COSINE,
                                   DIAL_SIGMAS, DIRECTION_NAMES, MAX_BLEND_SOURCES,
                                   BlendRecipe, describe_blend, load_directions,
                                   max_tangent_norm)

router = APIRouter()


def current_recipe() -> BlendRecipe:
    return BlendRecipe.from_payload(getattr(roop_globals, "identity_blend", None))


def apply_identity_blend_from_payload(payload: dict) -> None:
    """Adopt ``payload['identity_blend']`` when the key is present.  Absent
    leaves the live recipe alone (older clients, other callers)."""
    if isinstance(payload, dict) and "identity_blend" in payload:
        value = payload.get("identity_blend")
        roop_globals.identity_blend = (BlendRecipe.from_payload(value).to_payload()
                                       if isinstance(value, dict) else None)


def _directions_payload():
    dirs = load_directions()
    if dirs is None:
        return {"available": [], "reason": "roop/assets/identity_directions.npz missing "
                                          "(run tools/fit_identity_directions.py)"}
    reach = max_tangent_norm(DEFAULT_MIN_IDENTITY_COSINE)
    out = {"available": [n for n in dirs.names if dirs.writable(n)],
           "fitted_names": list(dirs.names), "heldout": dirs.heldout,
           "corpus": dirs.meta.get("corpus", {}), "fitted": dirs.meta.get("fitted"),
           "dials": {}}
    for i, name in enumerate(dirs.names):
        render = dirs.render.get(name) or {}
        entry = {"metric": dirs.meta.get("fit", {}).get(name, {}).get("metric"),
                 "score": dirs.heldout.get(name),
                 "writable": dirs.writable(name),
                 "render_verdict": render.get("verdict")}
        if name == "age":
            slope = abs(float(dirs.units_per_step[i]))
            entry["years_per_step"] = float(dirs.units_per_step[i])
            # Largest age move the default identity guard admits on its own.
            entry["guard_reach_years"] = min(AGE_LIMIT_YEARS, reach * slope)
        else:
            step = float(render.get("step_per_dial") or DIAL_SIGMAS * float(dirs.spread[i]))
            entry["guard_reach_dial"] = min(1.0, reach / step) if step > 1e-9 else 1.0
        out["dials"][name] = entry
    return out


def _response(recipe: BlendRecipe):
    return {
        "recipe": recipe.to_payload(),
        "active": recipe.active,
        "limits": {"max_sources": MAX_BLEND_SOURCES, "age_years": AGE_LIMIT_YEARS,
                   "dial_sigmas": DIAL_SIGMAS, "dials": list(DIRECTION_NAMES),
                   "default_min_cosine": DEFAULT_MIN_IDENTITY_COSINE},
        "directions": _directions_payload(),
        "diagnostics": describe_blend(recipe, list(roop_globals.INPUT_FACESETS)),
    }


@router.get("/api/identity/blend")
def identity_blend_get():
    return _response(current_recipe())


@router.post("/api/identity/blend")
def identity_blend_set(payload: dict = Body(...)):
    recipe = BlendRecipe.from_payload(payload)
    roop_globals.identity_blend = recipe.to_payload()
    return _response(recipe)
