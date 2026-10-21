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

PHASE 9 (the per-job report). Besides progress the registry now keeps, per stage, the facts the final report
needs, and builds ONE JSON-safe `summary` when the run ends (`end_run`), which `/ws/telemetry` carries, the
job history stores (`api._record_run_history`) and the UI shows:

    stage 1   wall_s, sampled_frames, workers + worker_utilization_pct, active_ranges_count
    stage 2   wall_s, active_ranges_count, grab_calls (cap.grab / read frames walked), set_calls (cap.set
              seeks), seek_walks, verified_target_frames, and what the pre-pass scheduled (None when Stage 2 was
              restored from a checkpoint): detect_interval, detector_calls, scan_frames, coast_frames,
              flow_backend, landmark_filter, trajectory_faces
    stage 3   wall_s, passthrough_frames (zero GPU work), gpu_frames (past the active-range gate),
              swapped_frames (>= 1 face swapped), faces_swapped, peak_vram_gb (device-wide, sampled)
    stage 4   the `finalize_output` result (verification, colour tags, A/V alignment, orphans)
    run       total_wall_s, effective_fps, and `speedup`

`speedup` is an ESTIMATE and says so (`estimated: true`, `method`): a dense 1-stage render is not run
during a job, so the baseline is this run's own per-active-frame cost (Stage 2 + Stage 3 wall, scaled from
the active frames to every frame) divided into what the 3-stage run actually took. A measured dense arm is
`tests/verify_three_stage_pipeline.py`'s benchmark; `tests/verify_stage4_report.py` compares the two.
Counters are only written while their stage is RUNNING, so a live-preview swap between renders cannot move them.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Any, Dict, Optional

from roop.degrade import swallowed as _swallowed
from roop.vram_peak import PeakSampler

STAGE_1_SPARSE_PRESCAN = "STAGE_1_SPARSE_PRESCAN"
STAGE_2_ACTIVE_PREPASS = "STAGE_2_ACTIVE_PREPASS"
STAGE_3_MAIN_SWAP = "STAGE_3_MAIN_SWAP"
STAGES = (STAGE_1_SPARSE_PRESCAN, STAGE_2_ACTIVE_PREPASS, STAGE_3_MAIN_SWAP)
# Not a progress stage (it has no counter or ETA); the report's fourth section (roop/finalize_output.py).
STAGE_4_FINALIZE = "STAGE_4_FINALIZE"

PENDING, RUNNING, DONE, SKIPPED, FAILED = "PENDING", "RUNNING", "DONE", "SKIPPED", "FAILED"
# The stage did not run THIS time because a resumed job loaded the answer its interrupted run had already
# computed (roop/stage_cache.py). Distinct from SKIPPED (the stage does not apply) and from DONE (it ran).
RESTORED = "RESTORED"

_lock = threading.RLock()
_run: Dict[str, Any] = {}
_stages: Dict[str, Dict[str, Any]] = {}
_counters: Dict[str, Dict[str, float]] = {}      # stage -> metric -> value (see `add` / `put`)
_finalize: list = []                              # Stage 4 results of this job, one per finished video
_summary: Dict[str, Any] = {}                     # built by end_run; survives until the next begin_run
_videos = 0                                       # begin_run calls since the job's reset()


def _current_stage() -> Optional[str]:
    # Read by the VRAM sampler thread without taking `_lock`: a stale read costs one sample's attribution,
    # never a deadlock.
    for name in STAGES:
        s = _stages.get(name)
        if s is not None and s.get("state") == RUNNING:
            return name
    return None


_vram = PeakSampler(stage_of=_current_stage)


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
    global _videos
    with _lock:
        _run.clear()
        _run.update({"total_frames": int(total_frames or 0), "active_ranges_count": None,
                     "active_frames": None, "verified_target_frames": None,
                     "started": time.time()})
        _stages.clear()
        _counters.clear()
        _summary.clear()
        _videos += 1
        for name in STAGES:
            _stages[name] = _blank()
    _vram.start()


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


def end_run(ok: bool = True) -> Dict[str, Any]:
    """The render is over (called from the job's `finally`): nothing is left RUNNING.

    A stage still RUNNING here was cut short (a stop, an exception between stages), so it reads
    FAILED rather than being left claiming a burst that ended; the final state stays readable."""
    with _lock:
        for s in _stages.values():
            if s["state"] == RUNNING:
                s["state"] = DONE if ok else FAILED
                s["ended"] = time.time()
        had_run = bool(_stages)
    _vram.stop()
    if had_run:
        report = build_summary(ok=ok)
        with _lock:
            _summary.clear()
            _summary.update(report)
    return summary()


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


def add(stage: str, key: str, n: float = 1) -> None:
    """Count `n` more `key` events of `stage`. Ignored unless the stage is RUNNING (a preview swap between
    renders, or a late worker after the stage ended, must not move a finished report)."""
    with _lock:
        s = _stages.get(stage)
        if s is None or s["state"] != RUNNING:
            return
        c = _counters.setdefault(stage, {})
        c[key] = c.get(key, 0) + n


def put(stage: str, **values: Any) -> None:
    """Record measured facts of `stage` (any state: Stage 1's scan reports after it has left)."""
    with _lock:
        _counters.setdefault(stage, {}).update(values)


def set_finalize(result: Optional[Dict[str, Any]]) -> None:
    """A Stage 4 result (`finalize_output.finalize_render`) of the current job."""
    if result is None:
        return
    with _lock:
        _finalize.append(result)


def _wall(stage: Dict[str, Any], now: float) -> Optional[float]:
    since, ended = stage.get("since"), stage.get("ended")
    if since is None:
        return None
    return round(max(0.0, (ended if ended is not None else now) - since), 3)


def _num(value: Any) -> Optional[float]:
    try:
        out = float(value)
        return out if out == out else None
    except (TypeError, ValueError):
        return None


def estimate_speedup(s1: Optional[float], s2: Optional[float], s3: Optional[float],
                     total_frames: Optional[int], active_frames: Optional[int],
                     restored: bool = False) -> Dict[str, Any]:
    """The 3-stage run's speedup over a dense 1-stage baseline, ESTIMATED from this run alone.

    dense ~ (Stage 2 wall + Stage 3 wall) x total / active: the per-active-frame cost of detecting + swapping,
    scaled to every frame (a dense render has no Stage 1 and swaps every frame). It overstates the dense
    time slightly because Stage 3's wall includes the near-free encode of passthrough frames, so the
    multiplier is, if anything, a little generous; the harness comparison measures by how much."""
    out: Dict[str, Any] = {"multiplier": None, "estimated": True, "baseline_s": None, "actual_s": None,
                           "method": "this run's per-active-frame cost (Stage 2 + Stage 3 wall) scaled to every "
                                     "frame, divided by the 3-stage wall clock; a dense render is not run"}
    if restored:
        out["reason"] = "a stage was restored from the project checkpoint, so its cost is unknown"
        return out
    s3 = _num(s3)
    if not s3 or s3 <= 0 or not total_frames:
        out["reason"] = "Stage 3 did not complete"
        return out
    s1, s2 = _num(s1) or 0.0, _num(s2) or 0.0
    active = int(active_frames) if active_frames else int(total_frames)
    active = max(1, min(active, int(total_frames)))
    dense = (s2 + s3) * float(total_frames) / float(active)
    actual = s1 + s2 + s3
    out.update(baseline_s=round(dense, 2), actual_s=round(actual, 2), multiplier=round(dense / actual, 2))
    return out


def _video_io_snapshot() -> Optional[Dict[str, Any]]:
    """Decode/encode backend, PCIe bytes saved (an estimate) and decode queue saturation for this job, or None
    when roop/video_io_hw.py was never loaded (never a made-up zero)."""
    mod = sys.modules.get("roop.video_io_hw")
    if mod is None:
        return None
    try:
        return mod.snapshot()
    except Exception as exc:
        _swallowed("roop/stage_telemetry.py:video_io", exc, "video I/O block left out of the report")
        return None


def build_summary(ok: bool = True, now: Optional[float] = None) -> Dict[str, Any]:
    """The per-job report. JSON-safe; every metric that was not measured is None, never a fabricated 0."""
    with _lock:
        now = time.time() if now is None else now
        vram = _vram.report()
        st = {n: dict(_stages.get(n) or _blank()) for n in STAGES}
        cnt = {n: dict(_counters.get(n) or {}) for n in STAGES}
        run = dict(_run)
        fins = list(_finalize)
        videos = _videos
        total = run.get("total_frames") or None
        s1, s2, s3 = (_wall(st[n], now) for n in STAGES)
        active = run.get("active_frames")
        c1, c2, c3 = (cnt[n] for n in STAGES)
        restored = any(st[n]["state"] == RESTORED for n in (STAGE_1_SPARSE_PRESCAN, STAGE_2_ACTIVE_PREPASS))

        def _peak(stage: str) -> Optional[float]:
            return (vram.get("by_stage_gb") or {}).get(stage)

        stage1 = {"state": st[STAGE_1_SPARSE_PRESCAN]["state"], "wall_s": s1,
                  "sampled_frames": c1.get("sampled_frames"), "samples_total": c1.get("samples_total"),
                  "stride": c1.get("stride"), "workers": c1.get("workers"),
                  "worker_utilization_pct": c1.get("worker_utilization_pct"),
                  "active_ranges_count": run.get("active_ranges_count"),
                  "skipped_frames_pct": _skipped_pct()}
        stage2 = {"state": st[STAGE_2_ACTIVE_PREPASS]["state"], "wall_s": s2,
                  "active_ranges_count": run.get("active_ranges_count"),
                  "grab_calls": c2.get("grab_calls"), "set_calls": c2.get("set_calls"),
                  "seek_walks": c2.get("seek_walks"),
                  "verified_target_frames": run.get("verified_target_frames"),
                  "detect_interval": c2.get("detect_interval"), "detector_calls": c2.get("detector_calls"),
                  "scan_frames": c2.get("scan_frames"), "coast_frames": c2.get("coast_frames"),
                  "flow_backend": c2.get("flow_backend"), "landmark_filter": c2.get("landmark_filter"),
                  "trajectory_faces": c2.get("trajectory_faces")}
        stage3 = {"state": st[STAGE_3_MAIN_SWAP]["state"], "wall_s": s3,
                  "frames": run.get("total_frames"),
                  "passthrough_frames": c3.get("passthrough_frames"), "gpu_frames": c3.get("gpu_frames"),
                  "swapped_frames": c3.get("swapped_frames"), "faces_swapped": c3.get("faces_swapped"),
                  "peak_vram_gb": _peak(STAGE_3_MAIN_SWAP),
                  "processing_fps": round(total / s3, 2) if total and s3 else None}
        # Stage 4 is the LAST finalize of the job (a multi-video job reports the last video's pipeline).
        last = fins[-1] if fins else None
        stage4 = None
        if last is not None:
            v = last.get("verify") or {}
            stage4 = {"state": "DONE" if last.get("ok") else "FAILED", "ok": bool(last.get("ok")),
                      "wall_s": last.get("wall_s"), "mux_s": last.get("mux_s"), "frames": v.get("frames"),
                      "expected_frames": last.get("expected_frames"), "audio_present": v.get("audio_present"),
                      "audio_start_offset_ms": v.get("audio_start_offset_ms"),
                      "color": v.get("color"), "color_preserved": v.get("color_preserved"),
                      "problems": last.get("problems") or [], "orphans": last.get("orphans") or [],
                      "temp_deleted": last.get("temp_deleted"), "output": last.get("output"),
                      "bytes": v.get("bytes")}
        s4 = (last or {}).get("wall_s") or 0.0
        started = run.get("started")
        total_wall = round(now - started, 3) if started else None
        pipeline_s = sum(x for x in (s1, s2, s3) if x)
        eff_fps = round(total / (pipeline_s + s4), 2) if total and (pipeline_s + s4) > 0 else None
        speedup = estimate_speedup(s1, s2, s3, total, active, restored=restored)
        return {"schema": 1, "ok": bool(ok) and (stage4 is None or stage4["ok"]), "finished_at": now,
                "video_io": _video_io_snapshot(),
                "total_frames": total, "videos_in_job": videos, "total_wall_s": total_wall,
                "effective_fps": eff_fps, "speedup": speedup,
                "vram": {"peak_gb": vram.get("peak_gb"), "base_gb": vram.get("base_gb"),
                         "total_gb": vram.get("total_gb"), "samples": vram.get("samples")},
                "stages": {STAGE_1_SPARSE_PRESCAN: stage1, STAGE_2_ACTIVE_PREPASS: stage2,
                           STAGE_3_MAIN_SWAP: stage3, STAGE_4_FINALIZE: stage4}}


def summary() -> Dict[str, Any]:
    """The last finished run's report ({} while a run is in progress or none has finished)."""
    with _lock:
        return dict(_summary)


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
                "wall_s": _wall(s, now),
            }
        return {"stage": current, "stages": out, "summary": dict(_summary)}


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
    global _videos
    with _lock:
        _run.clear()
        _stages.clear()
        _counters.clear()
        _finalize.clear()
        _summary.clear()
        _videos = 0
    _vram.stop()
