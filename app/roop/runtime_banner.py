"""One line that says what the runtime is doing AND what it was asked to swap.

Two layers meet at the swap stage and must stay independent:

* the RUNTIME layer -- execution provider, precision, batch path, session
  fallback (``predictor``, ``backend_manager``, ``FaceSwapInsightFace._infer``,
  ``swap_batcher``);
* the SELECTION layer -- which target person is eligible, resolved once in
  ``ProcessMgr.initialize`` from the canonical request (``target_selection``).

A provider failure is allowed to change the first (TensorRT -> CUDA -> CPU,
batch -> sequential, or a controlled render error).  It is never allowed to
change the second.  This module prints both side by side, from the LIVE
objects rather than the configuration, so a log can show in one line that a
fallback happened and that the selected person did not move with it.

``snapshot_selection`` / ``selection_invariant`` back that claim with a check
instead of a promise: the selection is frozen at initialize and compared again
at every later banner, so a later phase that finds it changed says so loudly.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from roop.degrade import swallowed as _swallowed

_SELECTION_KEYS = ("selection_mode", "person_id", "person_ids", "valid", "diagnostic")


def _short(provider) -> str:
    if isinstance(provider, (tuple, list)) and provider:
        provider = provider[0]
    return str(provider).replace("ExecutionProvider", "").lower() or "none"


def _swap_processor(mgr):
    for p in getattr(mgr, "processors", None) or ():
        if getattr(p, "type", None) == "swap":
            return p
    return None


def swap_active_provider(swap_p) -> str:
    """The provider the swap session actually registered, not the one asked for.

    ``get_providers()`` is the only reliable tell (see ``roop/predictor.py``);
    ``_trt_disabled`` marks the run-time rebuild that ``_infer`` performs after
    a genuine batch-1 TensorRT failure, which is a different event from a
    session that never came up on TensorRT at all.
    """
    if swap_p is None:
        return "none"
    session = getattr(swap_p, "model_swap_insightface", None)
    active = "none"
    try:
        providers = list(session.get_providers()) if session is not None else []
        if providers:
            active = _short(providers[0])
    except Exception as _e:
        _swallowed("roop/runtime_banner.py:swap_active_provider", _e,
                   "provider unknown")
        active = "unknown"
    if getattr(swap_p, "_trt_disabled", False):
        active += "(trt-rebuilt)"
    return active


def swap_requested_chain(swap_p) -> str:
    chain = getattr(swap_p, "_swap_providers", None) if swap_p is not None else None
    if not chain:
        import roop.globals
        chain = getattr(roop.globals, "execution_providers", None) or ()
    return ">".join(_short(p) for p in chain) or "none"


def swap_precision(swap_p) -> str:
    """Precision the session was BUILT with, read off its provider options."""
    chain = getattr(swap_p, "_swap_providers", None) if swap_p is not None else None
    for p in chain or ():
        if isinstance(p, (tuple, list)) and len(p) == 2 and "tensorrt" in str(p[0]).lower():
            opts = p[1] if isinstance(p[1], dict) else {}
            if opts.get("trt_bf16_enable"):
                return "bf16"
            return "mixed" if opts.get("trt_fp16_enable", False) else "fp32"
    return "fp32"


def swap_batch_mode(mgr, swap_p, batch_swap_flag: Optional[bool] = None) -> str:
    """Which of the three swap dispatch paths in ``ProcessMgr.process_face`` runs.

    ``xframe``     -- the cross-frame ``SwapBatcher`` (only built for video with
                      >1 worker; ``mgr._swap_batcher`` is the truth).
    ``tile``       -- pixel-boost tiles batched through ``RunBatch`` (only when
                      subsample_size yields more than one tile).
    ``sequential`` -- one ``Run`` per crop, B=1.  A model that declined batching
                      (``_batch_unsupported``) is named as such: that is a
                      runtime decision and must be visible, not silent.
    """
    if swap_p is None:
        return "none"
    if getattr(swap_p, "_batch_unsupported", False):
        return "sequential(model-declined-batch)"
    if getattr(mgr, "_swap_batcher", None) is not None:
        return "xframe"
    if batch_swap_flag is None:
        try:
            from roop.ProcessMgr import _BATCH_SWAP as batch_swap_flag
        except Exception as _e:
            _swallowed("roop/runtime_banner.py:swap_batch_mode", _e,
                       "batch flag unknown")
            batch_swap_flag = False
    if batch_swap_flag and hasattr(swap_p, "RunBatch") and _tiles(mgr, swap_p) > 1:
        return "tile"
    return "sequential"


def _tiles(mgr, swap_p) -> int:
    """Pixel-boost tiles per face, the same arithmetic ProcessMgr.process_face
    uses; one tile means the RunBatch branch is skipped and each face is a
    single B=1 ``Run`` call, whatever the batch flags say."""
    try:
        out = int(getattr(swap_p, "model_output_size", 128) or 128)
        sub = int(getattr(getattr(mgr, "options", None), "subsample_size", out) or out)
        return max(1, max(sub, out) // out)
    except (TypeError, ValueError):
        return 1


def _selection_state(mgr) -> Dict[str, Any]:
    sel = getattr(mgr, "target_selection", None)
    return dict(sel) if isinstance(sel, dict) else {}


def _selected_person(sel: Dict[str, Any]):
    if sel.get("selection_mode") == "multi_person":
        return list(sel.get("person_ids") or [])
    return sel.get("person_id")


def snapshot_selection(mgr) -> Dict[str, Any]:
    """Freeze the selection layer's outputs so a later phase can compare."""
    sel = _selection_state(mgr)
    options = getattr(mgr, "options", None)
    return {
        "swap_mode": getattr(options, "swap_mode", None),
        "selection": {k: sel.get(k) for k in _SELECTION_KEYS},
        "selected_groups": sorted(getattr(mgr, "selected_target_groups", None) or ()),
        "target_groups": list(getattr(mgr, "target_face_groups", None) or ()),
    }


def selection_invariant(mgr) -> Optional[str]:
    """``None`` when the live selection matches the initialize-time snapshot,
    else a description of every field that moved."""
    frozen = getattr(mgr, "_selection_snapshot", None)
    if not frozen:
        return None
    live = snapshot_selection(mgr)
    moved = [f"{key}: {frozen.get(key)!r} -> {live.get(key)!r}"
             for key in frozen if frozen.get(key) != live.get(key)]
    return "; ".join(moved) if moved else None


def runtime_selection_line(mgr, phase: str, batch_mode: Optional[str] = None) -> str:
    """The banner. Every value is read from live objects at call time."""
    swap_p = _swap_processor(mgr)
    sel = _selection_state(mgr)
    options = getattr(mgr, "options", None)
    groups = list(getattr(mgr, "target_face_groups", None) or ())
    fields = [
        f"phase={phase}",
        f"provider_active={swap_active_provider(swap_p)}",
        f"requested={swap_requested_chain(swap_p)}",
        f"precision={swap_precision(swap_p)}",
        f"swap_model={getattr(swap_p, 'loaded_model_key', None) or getattr(options, 'swap_model', None)}",
        f"batch_mode={batch_mode or swap_batch_mode(mgr, swap_p)}",
        f"target_selection={getattr(options, 'swap_mode', None)}/{sel.get('selection_mode', 'none')}",
        f"selected_person={_selected_person(sel)}",
        f"selected_groups={sorted(getattr(mgr, 'selected_target_groups', None) or ())}",
        f"persons={len(set(groups))}",
    ]
    if sel.get("diagnostic"):
        fields.append(f"selection_diagnostic={sel['diagnostic']}")
    violation = selection_invariant(mgr)
    if phase != "init":
        fields.append("selection_invariant=" + ("OK" if violation is None
                                                  else f"VIOLATED({violation})"))
    return "[Runtime] " + " ".join(fields)
