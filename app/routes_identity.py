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
import traceback

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

import roop.globals as roop_globals
from roop.identity_algebra import (AGE_LIMIT_YEARS, BLEND_MODES, CURVES, DEFAULT_BEZIER,
                                   DEFAULT_MIN_IDENTITY_COSINE, DEFAULT_PRIVACY_COSINE,
                                   DIAL_SIGMAS, DIRECTION_NAMES, MAX_BLEND_SOURCES, MAX_KEYFRAMES,
                                   BlendRecipe, describe_blend, describe_keyframes,
                                   faceset_identity_vector, l2_normalize, load_directions,
                                   max_tangent_norm, resolve_recipe)

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
                   "default_min_cosine": DEFAULT_MIN_IDENTITY_COSINE,
                   "modes": list(BLEND_MODES), "curves": list(CURVES),
                   "max_keyframes": MAX_KEYFRAMES, "default_bezier": list(DEFAULT_BEZIER),
                   "default_privacy_cosine": DEFAULT_PRIVACY_COSINE},
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


def _body_recipe(payload: dict) -> BlendRecipe:
    """The recipe a request names (``payload["recipe"]``), else the live one."""
    raw = payload.get("recipe") if isinstance(payload, dict) else None
    return BlendRecipe.from_payload(raw) if isinstance(raw, dict) else current_recipe()


@router.post("/api/identity/keyframes/evaluate")
def identity_keyframes_evaluate(payload: dict = Body(default=None)):
    """What the keyframe track does, from the loaded sources' centroids: each keyframe's angle to the next, the
    eased progress and cosine-to-keyframes at sampled frames (``frames`` or ``samples`` evenly spaced ones), and the
    frame-to-frame smoothness. Nothing is rendered and nothing is stored."""
    payload = payload if isinstance(payload, dict) else {}
    recipe = _body_recipe(payload)
    if not recipe.has_keyframes:
        return JSONResponse(status_code=400, content={"message": "the recipe has no keyframes over 2 or more sources"})
    frames = payload.get("frames")
    try:
        frames = [int(f) for f in frames] if isinstance(frames, list) else None
        samples = int(payload.get("samples", 96))
    except (TypeError, ValueError):
        return JSONResponse(status_code=400, content={"message": "frames and samples must be integers"})
    return describe_keyframes(recipe, list(roop_globals.INPUT_FACESETS), frames=frames,
                              samples=max(2, min(samples, 512)))


@router.post("/api/identity/export_fsz")
def identity_export_fsz(payload: dict = Body(...)):
    """Save the recipe's composite vector as a V2 .fsz preset in the faceset library.

    The vector is blend (at ``frame_index`` when the recipe is keyframed; the base weights otherwise) -> dials ->
    de-identification, over the sources' centroids (a preset is pose-agnostic). It is anchored to a photo of the
    heaviest source, like the Morph Lab's save. ``frame_index`` is 0-based and absolute."""
    from routes_identity_lab import _anchor, save_blended_to_library
    recipe = _body_recipe(payload)
    recipe.enabled = True
    name = str(payload.get("name") or "Blended identity").strip() or "Blended identity"
    try:
        frame_index = payload.get("frame_index")
        frame_index = None if frame_index is None else int(frame_index)
        facesets = list(roop_globals.INPUT_FACESETS)
        resolved = resolve_recipe(recipe, facesets)
        if resolved is None:
            raise ValueError("nothing to export: add a loaded source to the blender")
        by_sid = {str(getattr(fs, "_source_id", "") or ""): fs for fs in facesets}
        bound = [(c, by_sid.get(c["source_id"])) for c in recipe.components]
        bound = [(c, fs) for c, fs in bound if fs is not None]
        if not bound:
            raise ValueError("none of the blended sources is loaded any more")
        heaviest = max(bound, key=lambda cf: cf[0]["weight"])[1]
        anchor = faceset_identity_vector(heaviest) if len(resolved.facesets) < 2 else None
        vector = resolved.compose(anchor, None, None, frame_index)
        if vector is None or l2_normalize(vector) is None:
            raise ValueError("the recipe has no vector at that frame")
        index = next(i for i, fs in enumerate(facesets) if fs is heaviest)
        image, face = _anchor(f"src:{index}")
        return save_blended_to_library(
            name, vector, image, face,
            {"mode": recipe.mode, "frame_index": frame_index, "keyframed": resolved.time_varying,
             "deidentified": bool(resolved.deid_refs)})
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"message": str(exc)})
    except Exception:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": "failed to write the blended faceset"})
