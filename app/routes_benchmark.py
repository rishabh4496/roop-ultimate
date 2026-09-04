"""Benchmark and optimization endpoints.

Transport only. Every number, badge, score and comparison row is built in
``roop.benchmark.ui_dashboard`` so the React panel, the CLI and this API render
the same answer; a value computed here would be a second implementation with
its own bugs.

These sit under ``/api/benchmark/*``. The older ``/api/settings/benchmark_*``
endpoints in api.py drive ``roop.bench``, a different (stage-cost / pool-curve)
benchmark, and are deliberately left alone.
"""

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from roop.benchmark.ui_dashboard import (
    BENCHMARK_OWNED_SETTINGS,
    PreBenchmarkPrompt,
    apply_recommended_settings,
    decline_recommended_settings,
    get_session,
    list_saved_profiles,
    revert_to_default_settings,
    stock_defaults,
)

router = APIRouter(prefix="/api/benchmark")

# ── Injected by api.py at import time ────────────────────────────────────
# The render-progress dict. Bound to the same object api.py mutates in place so
# the guard below reads live state rather than a copy taken at import.
_progress = {"processing": False}


def bind_progress(progress) -> None:
    global _progress
    _progress = progress


@router.get("/prompt")
def benchmark_prompt():
    """What the pre-benchmark modal renders.

    The active models are read back from the live pipeline rather than echoed
    from the UI: the run measures the user's configuration, and a modal that
    merely repeats what it was told cannot notice that a model failed to load.
    """
    return PreBenchmarkPrompt.build().as_dict()


@router.post("/start")
def benchmark_start(payload: dict = Body(default=None)):
    payload = payload or {}
    # Refuse to share the GPU with a render. Both answers would be wrong: the
    # render slows down, and the benchmark measures a card that is already busy
    # and then recommends settings for a machine nobody has.
    if _progress.get("processing"):
        return JSONResponse(status_code=409, content={
            "message": "A render is in progress — benchmarking now would "
                       "measure a busy GPU and slow the render down."})
    session = get_session()
    if session.running:
        return JSONResponse(status_code=409, content={
            "message": "A benchmark is already running."})
    result = session.start(faces=payload.get("faces", "1"),
                           mode=payload.get("mode", "quick"),
                           persist=bool(payload.get("persist", True)))
    if result.get("status") != "started":
        return JSONResponse(status_code=409, content=result)
    return result


@router.get("/progress")
def benchmark_progress():
    """Polled once a second by the live progress screen."""
    return get_session().snapshot().as_dict()


@router.post("/cancel")
def benchmark_cancel():
    return get_session().cancel()


@router.get("/result")
def benchmark_result():
    """The finished dashboard, or an explicit not-ready answer.

    Returns 204-style emptiness as a body rather than an error: the panel polls
    this while a run is in flight and a 404 there would surface to the user as
    a broken backend.
    """
    report = get_session().report()
    if report is None:
        snapshot = get_session().snapshot()
        return {"ready": False, "running": snapshot.running,
                "error": snapshot.error}
    payload = report.as_dict()
    payload["ready"] = True
    return payload


@router.post("/apply")
def benchmark_apply(payload: dict = Body(default=None)):
    """Accept: write the recommendation into the live configuration."""
    payload = payload or {}
    report = get_session().report()
    recommended = payload.get("recommended_settings")
    if recommended is None and report is not None:
        recommended = report.recommended_settings
    run_id = str(payload.get("run_id") or (report.run_id if report else ""))
    result = apply_recommended_settings(
        recommended=recommended, run_id=run_id,
        allow_lossy_temp_frames=bool(payload.get("allow_lossy_temp_frames", False)))
    if result.get("status") == "error":
        return JSONResponse(status_code=400, content=result)
    return result


@router.post("/decline")
def benchmark_decline(payload: dict = Body(default=None)):
    """Decline: touch nothing, but keep the run so it can be applied later."""
    payload = payload or {}
    session = get_session()
    report = session.report()
    run_id = str(payload.get("run_id") or (report.run_id if report else ""))
    # The run was already persisted by the runner (persist=True), so this
    # records the decision rather than writing a duplicate record.
    return decline_recommended_settings(result=None, run_id=run_id)


@router.post("/revert")
def benchmark_revert():
    """Clear benchmark overrides, restoring the shipped defaults.

    Scoped to the settings the benchmark is allowed to write. "Revert to
    default" on a results screen means "undo what this benchmark changed", not
    "reset the application" -- the second reading would be a destructive
    surprise for someone clearing a thread count.
    """
    result = revert_to_default_settings()
    if result.get("status") == "error":
        return JSONResponse(status_code=400, content=result)
    return result


@router.get("/defaults")
def benchmark_defaults():
    """What Revert would restore, so the UI can show it before acting."""
    return {"owned_settings": list(BENCHMARK_OWNED_SETTINGS),
            "defaults": stock_defaults()}


@router.get("/profiles")
def benchmark_profiles(limit: int = 20):
    """Settings > Optimization Profiles."""
    return {"profiles": list_saved_profiles(limit=limit)}


@router.post("/profiles/apply")
def benchmark_profile_apply(payload: dict = Body(default=None)):
    """Apply a stored profile from the history list."""
    payload = payload or {}
    run_id = str(payload.get("run_id") or "")
    recommended = payload.get("recommended_settings")
    if recommended is None:
        for row in list_saved_profiles(limit=200):
            if row.get("run_id") == run_id:
                recommended = row.get("recommended_settings")
                break
    if not recommended:
        return JSONResponse(status_code=404, content={
            "message": "No stored recommendation for that run."})
    result = apply_recommended_settings(
        recommended=recommended, run_id=run_id,
        allow_lossy_temp_frames=bool(payload.get("allow_lossy_temp_frames", False)))
    if result.get("status") == "error":
        return JSONResponse(status_code=400, content=result)
    return result
