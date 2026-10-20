"""Per-stage progress of ONE render, for `/ws/telemetry` (the 3-stage pipeline).

WHY. A sparse-range render runs three stages whose frame counters mean different things: the
pre-scan counts keyframe SAMPLES, the pre-pass counts frames of the ACTIVE ranges only, and the
main swap counts every frame of the trim (passing the skipped ones through). The one progress bar
the client sees is a single number that restarts at each stage, so "62%" cannot tell a user
whether the render is nearly done or about to start its slowest stage. This registry gives each
stage a name and the numbers the client needs to say so.

    STAGE_1_SPARSE_PRESCAN   keyframe-sampled presence scan (only when `sparse_auto_range` is on)
    STAGE_2_ACTIVE_PREPASS   tracking / temporal detection over the active ranges
    STAGE_3_MAIN_SWAP        the swap itself, over the whole trim

Per stage: `state` (PENDING | RUNNING | DONE | SKIPPED | FAILED), `stage_progress_pct`,
`eta_seconds`, and the run-level facts the stages establish: `active_ranges_count` (Stage 1),
`verified_target_frames` (frames of the Stage 2 plan, i.e. holding a face matched to a source),
`skipped_frames_pct` (share of the trim the scan proved target-free and the swap passes through).

DESIGN (same constraint as `routes_telemetry` and `pipeline_status`): the pipeline runs on plain
threads and cannot await, so it only WRITES here - a lock and a dict update every ~10 frames; the
web side SAMPLES (`api._telemetry_snapshot` folds `snapshot()` into each frame). Nothing here
imports the pipeline, so it is testable without a GPU. Stage 3's counter is NOT written from the
frame loop: the web side already reads it from the progress line, and `api._telemetry_snapshot`
overlays it with `set_main_progress`, so the hot loop is untouched.

A registry nobody reads costs one lock acquisition per ~10 frames.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

STAGE_1_SPARSE_PRESCAN = "STAGE_1_SPARSE_PRESCAN"
STAGE_2_ACTIVE_PREPASS = "STAGE_2_ACTIVE_PREPASS"
STAGE_3_MAIN_SWAP = "STAGE_3_MAIN_SWAP"
STAGES = (STAGE_1_SPARSE_PRESCAN, STAGE_2_ACTIVE_PREPASS, STAGE_3_MAIN_SWAP)

PENDING, RUNNING, DONE, SKIPPED, FAILED = "PENDING", "RUNNING", "DONE", "SKIPPED", "FAILED"
# The stage did not run THIS time because a resumed job loaded the answer its interrupted run had already
# computed (roop/stage_cache.py). Distinct from SKIPPED (the stage does not apply) and from DONE (it ran).
RESTORED = "RESTORED"

_lock = threading.RLock()
_run: Dict[str, Any] = {}
_stages: Dict[str, Dict[str, Any]] = {}


def _blank() -> Dict[str, Any]:
    return {"state": PENDING, "done": 0, "total": 0, "since": None, "ended": None}


def _pct(done: float, total: float) -> Optional[float]:
    if not total or total <= 0:
        return None
    return round(max(0.0, min(100.0, 100.0 * float(done) / float(total))), 1)


def _eta(stage: Dict[str, Any], now: float) -> Optional[float]:
    """Seconds left at this stage's own average rate; None until it has measured one."""
    since, done, total = stage.get("since"), stage.get("done") or 0, stage.get("total") or 0
    if stage.get("state") in (DONE, RESTORED):
        return 0.0
    if stage.get("state") != RUNNING or since is None or done <= 0 or total <= 0:
        return None
    elapsed = now - since
    if elapsed <= 0:
        return None
    return round(max(0.0, (total - done) * elapsed / done), 1)


def begin_run(total_frames: int = 0) -> None:
    """A render is starting: every stage PENDING, the run facts unknown."""
    with _lock:
        _run.clear()
        _run.update({"total_frames": int(total_frames or 0), "active_ranges_count": None,
                     "active_frames": None, "verified_target_frames": None,
                     "started": time.time()})
        _stages.clear()
        for name in STAGES:
            _stages[name] = _blank()


def enter(stage: str, total: int = 0) -> None:
    with _lock:
        s = _stages.setdefault(stage, _blank())
        s.update(state=RUNNING, done=0, total=int(total or 0), since=time.time(), ended=None)


def progress(stage: str, done: int, total: Optional[int] = None) -> None:
    """Stage-local counter (samples for Stage 1, active-range frames for Stage 2)."""
    with _lock:
        s = _stages.get(stage)
        if s is None or s["state"] != RUNNING:
            return
        s["done"] = int(done)
        if total:
            s["total"] = int(total)


def leave(stage: str, ok: bool = True) -> None:
    with _lock:
        s = _stages.get(stage)
        if s is None:
            return
        s["state"] = DONE if ok else FAILED
        if ok and s["total"]:
            s["done"] = s["total"]
        s["ended"] = time.time()


def end_run(ok: bool = True) -> None:
    """The render is over (called from the job's `finally`): nothing is left RUNNING.

    A stage still RUNNING here was cut short (a stop, an exception between stages), so it reads
    FAILED rather than being left claiming a burst that ended; the final state stays readable."""
    with _lock:
        for s in _stages.values():
            if s["state"] == RUNNING:
                s["state"] = DONE if ok else FAILED
                s["ended"] = time.time()


def skip(stage: str) -> None:
    """The stage will not run for this render (Stage 1 with `sparse_auto_range` off)."""
    with _lock:
        s = _stages.setdefault(stage, _blank())
        s["state"] = SKIPPED


def restored(stage: str) -> None:
    """The stage's result was loaded from the project checkpoint instead of being recomputed."""
    with _lock:
        s = _stages.setdefault(stage, _blank())
        s.update(state=RESTORED, ended=time.time())
        if s.get("total"):
            s["done"] = s["total"]


def set_ranges(count: Optional[int], active_frames: Optional[int], total_frames: Optional[int] = None) -> None:
    """Stage 1's verdict: how many ranges, and how many frames they cover."""
    with _lock:
        _run["active_ranges_count"] = None if count is None else int(count)
        _run["active_frames"] = None if active_frames is None else int(active_frames)
        if total_frames:
            _run["total_frames"] = int(total_frames)


def set_verified(frames: Optional[int]) -> None:
    """Stage 2's verdict: frames of `stage2_frame_plan` (a face matched to a source)."""
    with _lock:
        _run["verified_target_frames"] = None if frames is None else int(frames)


def set_main_progress(done: int, total: int, eta_seconds: Optional[float] = None) -> None:
    """Stage 3's counter, overlaid by the web side from the progress line (not the frame loop)."""
    with _lock:
        s = _stages.get(STAGE_3_MAIN_SWAP)
        if s is None or s["state"] != RUNNING:
            return
        s["done"], s["total"] = int(done), int(total)
        s["eta_override"] = eta_seconds


def _skipped_pct() -> Optional[float]:
    total, active = _run.get("total_frames") or 0, _run.get("active_frames")
    if active is None or not total:
        return None
    return round(max(0.0, min(100.0, 100.0 * (total - active) / total)), 1)


def snapshot() -> Dict[str, Any]:
    """JSON-safe view for a telemetry frame. Constant while idle (the sampler sends on change)."""
    with _lock:
        if not _stages:
            return {"stage": None, "stages": {}}
        now = time.time()
        skipped = _skipped_pct()
        out: Dict[str, Any] = {}
        current = None
        for name in STAGES:
            s = _stages.get(name) or _blank()
            if s["state"] == RUNNING:
                current = name
            eta = s.get("eta_override") if s["state"] == RUNNING and s.get("eta_override") is not None \
                else _eta(s, now)
            out[name] = {
                "state": s["state"],
                "stage_progress_pct": 100.0 if s["state"] in (DONE, RESTORED) else (_pct(s["done"], s["total"])
                                                                       if s["state"] == RUNNING else 0.0
                                                                       if s["state"] == PENDING else None),
                "active_ranges_count": _run.get("active_ranges_count"),
                "verified_target_frames": _run.get("verified_target_frames"),
                "skipped_frames_pct": skipped,
                "eta_seconds": eta,
            }
        return {"stage": current, "stages": out}


def busy_stage() -> Optional[str]:
    """The pre-render stage whose GPU burst a frame preview must not share, or None.

    Stages 1 and 2 run detectors flat out on the shared card; Stage 3 is the render itself and is
    handled by the existing render-active guard."""
    with _lock:
        for name in (STAGE_1_SPARSE_PRESCAN, STAGE_2_ACTIVE_PREPASS):
            s = _stages.get(name)
            if s is not None and s["state"] == RUNNING:
                return name
    return None


def reset() -> None:
    with _lock:
        _run.clear()
        _stages.clear()
