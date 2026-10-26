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

RETIRED FRAMES (Stage 3). Every fps in this report divides frames the OUTPUT WRITER HAS ACCEPTED (`retire()`, called
at the three `write_frame` sites in ProcessMgr, right after the write returns), never the trim's planned frame
count and never a worker's "I finished this frame" tick (the parallel-stabilization path ticks the progress bar when
a frame lands in its result dict, up to a chunk before it is written). Smart-Skip splits the picture in two:
`effective_output_fps` is frames written per second; `neural_inference_fps` is swap-model inferences per second;
`smart_skip_ratio_pct` is the share of faces that reused a flow-warped patch instead of running the model. Frames
that were reused cost no inference but are still written, so the first number can exceed the second.

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
from collections import deque
from typing import Any, Dict, Optional

from roop import gpu_event_timer as _gpu_events
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

# Stage 3 operation timings (decode / align-warp / swap model / blend / encode), fed by procmgr_runtime._prof.
# Written from ~20 worker threads, so it has its own small lock; `_ops_on` is a plain bool read on the hot path
# (a stale read costs one sample at a stage boundary, never correctness) and is True only while Stage 3 RUNS.
_ops_on = False
_ops_lock = threading.Lock()
_ops: Dict[str, list] = {}                       # _prof stage name -> [seconds, calls]
# What `_prof` measures: the host wall clock around a call that blocks until its result is on the host (an ORT run,
# a cv2 call, a pipe write). That IS the operation's duration, so no CUDA event is needed or used; a GPU-resident
# stage that enqueues asynchronously reports through roop/gpu_event_timer.py instead and is labelled "cuda-event".
OPS_TIMING = "host wall (blocking call)"

# Frames the output writer has accepted this Stage 3 (see the module docstring). Written from the writer thread(s),
# so it has its own lock; `retire` is a no-op unless Stage 3 is RUNNING (`_ops_on`).
RETIRE_WINDOW_S = 10.0
_retire_lock = threading.Lock()
_retired = 0
_retire_win: deque = deque(maxlen=4096)           # (monotonic seconds, cumulative frames retired)
# Which _prof stage is which column of the brief's Stage 3 line. `swap` is the swap model's inference call,
# `alignment` the face crop/warp (cv2 on the host unless a GPU warp is wired), `blend` the paste-back.
OPS_COLUMNS = (("decode", "decode"), ("warp", "alignment"), ("model", "swap"), ("restore", "enhance"),
               ("blend", "blend"), ("encode", "encode"))

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


def record_op(stage: str, seconds: float) -> None:
    """One timed Stage 3 operation (called by `_prof` while `_ops_on`)."""
    with _ops_lock:
        slot = _ops.get(stage)
        if slot is None:
            _ops[stage] = [seconds, 1]
        else:
            slot[0] += seconds
            slot[1] += 1


def retire(n: int = 1) -> None:
    """`n` more frames have been accepted by the output writer (call AFTER `write_frame` returns).

    Ignored unless Stage 3 is RUNNING, so a live-preview swap between renders cannot move a finished report."""
    global _retired
    if not _ops_on:
        return
    now = time.monotonic()
    with _retire_lock:
        _retired += int(n)
        _retire_win.append((now, _retired))


def _reset_retired() -> None:
    global _retired
    with _retire_lock:
        _retired = 0
        _retire_win.clear()


def _retired_total() -> int:
    with _retire_lock:
        return _retired


def _retire_rate(running: bool, now: Optional[float] = None) -> Optional[float]:
    """Frames written per second over the last RETIRE_WINDOW_S while running (decays if writes stop).

    The window ends at `now`, not at the last write: a stalled writer must read as a falling rate, not freeze at
    the last good one. None until two writes exist."""
    now = time.monotonic() if now is None else now
    with _retire_lock:
        pts = list(_retire_win)
    if len(pts) < 2:
        return None
    cutoff = now - RETIRE_WINDOW_S
    base = pts[0]
    for p in pts:
        if p[0] <= cutoff:
            base = p
        else:
            break
    end_t = now if running else pts[-1][0]
    dt = end_t - base[0]
    if dt <= 0:
        return None
    return round((pts[-1][1] - base[1]) / dt, 2)


def _smart_skip_enabled() -> Optional[bool]:
    try:
        from roop.temporal_smart_skip import is_smart_skip_enabled
        return bool(is_smart_skip_enabled())
    except Exception as exc:                          # noqa: BLE001 - the label is optional
        _swallowed("roop/stage_telemetry.py:smart_skip_enabled", exc, "smart-skip flag left out of the report")
        return None


def _stage3_rates(wall_s: Optional[float], counters: Dict[str, float], running: bool) -> Dict[str, Any]:
    """The retired-frame metrics of Stage 3. A metric nothing fed is None, never a made-up 0.

    `effective_output_fps`  frames written per second: windowed while RUNNING, retired / wall once it ended.
    `neural_inference_fps`  swap-model inferences per second over the stage's wall clock.
    `smart_skip_ratio_pct`  faces that reused a flow-warped patch / (those + faces the model ran on)."""
    retired = _retired_total()
    neural, skipped = counters.get("neural_inferences"), counters.get("smart_skip_faces")
    if running:
        eff = _retire_rate(True)
    else:
        eff = round(retired / wall_s, 2) if retired and wall_s and wall_s > 0 else None
    return {
        "frames_retired": retired if (retired or wall_s is not None) else None,
        "effective_output_fps": eff,
        "neural_inferences": int(neural) if neural is not None else None,
        "neural_inference_fps": _rate(neural, wall_s) if neural is not None else None,
        "smart_skip_faces": int(skipped) if skipped is not None else None,
        "smart_skip_ratio_pct": (_ratio_pct(skipped or 0, neural or 0)
                                 if (skipped is not None or neural is not None) else None),
        "smart_skip_enabled": _smart_skip_enabled(),
        "fps_basis": "retired (frames accepted by the output writer)",
    }


def retired_view() -> Optional[Dict[str, Any]]:
    """The live retired-frame counter for a headline fps: None unless Stage 3 is RUNNING and the writer has
    accepted at least one frame. A None tells the caller to fall back to (and LABEL) the progress bar's processed
    count, so a write path that never calls `retire` shows its old numbers marked as such instead of zero."""
    with _lock:
        s = _stages.get(STAGE_3_MAIN_SWAP)
        if not s or s["state"] != RUNNING or s.get("since") is None:
            return None
        since = s["since"]
    n = _retired_total()
    if n <= 0:
        return None
    elapsed = time.time() - since
    return {"frames": n, "avg_fps": round(n / elapsed, 2) if elapsed > 0 else None, "now_fps": _retire_rate(True)}


def stage3_rates() -> Optional[Dict[str, Any]]:
    """Stage 3's retired-frame metrics, live while it runs and final once it ended; None before it started."""
    with _lock:
        s = _stages.get(STAGE_3_MAIN_SWAP)
        if not s or s["state"] in (PENDING, SKIPPED):
            return None
        return _stage3_rates(_wall(s, time.time()), dict(_counters.get(STAGE_3_MAIN_SWAP) or {}),
                             running=s["state"] == RUNNING)


def stage3_line(expected_frames: Optional[int] = None) -> Optional[str]:
    """The terminal's Stage 3 closing line, from the same numbers the HUD and the report show; None before Stage 3.

    "[Stage 3] retired 600/600 frames | 21.40 Effective Output FPS | 812 neural inferences = 9.1/s |
    Smart-Skip 12.3% of 926 faces". A frame short of the expectation is printed as such, not rounded away."""
    r = stage3_rates()
    if not r or r["frames_retired"] is None:
        return None
    n = r["frames_retired"]
    parts = [f"retired {n}" + (f"/{int(expected_frames)}" if expected_frames else "") + " frames"]
    if r["effective_output_fps"] is not None:
        parts.append(f"{r['effective_output_fps']:.2f} Effective Output FPS")
    if r["neural_inferences"] is not None and r["neural_inference_fps"] is not None:
        parts.append(f"{r['neural_inferences']} neural inferences = {r['neural_inference_fps']:.1f}/s")
    if r["smart_skip_enabled"] is False and not r["smart_skip_faces"]:
        parts.append("Smart-Skip off")                     # off beats "0.0%": a setting that is off skipped nothing
    elif r["smart_skip_ratio_pct"] is not None:
        faces = (r["smart_skip_faces"] or 0) + (r["neural_inferences"] or 0)
        parts.append(f"Smart-Skip {r['smart_skip_ratio_pct']:.1f}% of {faces} faces")
    return "[Stage 3] " + " | ".join(parts)


def _ops_ms() -> Dict[str, Optional[Dict[str, Any]]]:
    """ms per call and call count for each Stage 3 column; None for one that never ran (never a made-up 0)."""
    with _ops_lock:
        raw = {k: (v[0], v[1]) for k, v in _ops.items()}
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for column, name in OPS_COLUMNS:
        sec, calls = raw.get(name, (0.0, 0))
        out[column] = {"ms": round(1000.0 * sec / calls, 3), "calls": calls} if calls else None
    return out


def _backends() -> Dict[str, Optional[str]]:
    """What each Stage 3 column ran ON, from the live configuration - labels, not measurements.

    `warp` is the host cv2 path: the GPU warp/paste (`CudaAffineBatch`) and CV-CUDA have no production caller, so
    this never says "cv-cuda" unless that changes. `model` is the primary execution provider; `encode` is the
    ffmpeg codec in use (an *_nvenc codec means the hardware encoder)."""
    out: Dict[str, Optional[str]] = {"decode": None, "warp": "cv2 (CPU)", "model": None, "restore": None,
                                     "blend": "cv2 (CPU)", "encode": None}
    try:
        import roop.globals as g
        provs = list(getattr(g, "execution_providers", None) or [])
        if provs:
            first = provs[0]                          # a bare name or a (name, options) pair
            name = str(first[0] if isinstance(first, (tuple, list)) else first).replace("ExecutionProvider", "")
            out["model"] = "TensorRT" if name.lower().startswith("tensorrt") else name
        out["encode"] = getattr(g, "video_encoder", None) or None
    except Exception as exc:                          # noqa: BLE001 - labels are optional
        _swallowed("roop/stage_telemetry.py:backends", exc, "backend labels left out")
    io = _video_io_snapshot() or {}
    out["decode"] = io.get("decode_backend") or "ffmpeg / cv2"
    out["encode"] = io.get("encode_backend") or out["encode"]
    return out


def enter(stage: str, total: int = 0) -> None:
    global _ops_on
    with _lock:
        s = _stages.setdefault(stage, _blank())
        s.update(state=RUNNING, done=0, total=int(total or 0), since=time.time(), ended=None)
    if stage == STAGE_3_MAIN_SWAP:
        with _ops_lock:
            _ops.clear()
        _reset_retired()
        _ops_on = True


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
    global _ops_on
    if stage == STAGE_3_MAIN_SWAP:
        _ops_on = False
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
    global _ops_on
    _ops_on = False
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


def _rate(count: Any, seconds: Any) -> Optional[float]:
    c, t = _num(count), _num(seconds)
    return round(c / t, 2) if c is not None and t and t > 0 else None


def _ratio_pct(part: Any, other: Any) -> Optional[float]:
    a, b = _num(part), _num(other)
    if a is None or b is None or (a + b) <= 0:
        return None
    return round(100.0 * a / (a + b), 1)


def _ops_report() -> Dict[str, Optional[Dict[str, Any]]]:
    """Stage 3's per-operation ms/call with the backend each ran on; a column that never ran is None."""
    ops, labels = _ops_ms(), _backends()
    return {col: (dict(v, backend=labels.get(col), timing=OPS_TIMING) if v else None) for col, v in ops.items()}


def _aux_offload() -> Optional[Dict[str, Any]]:
    """The native OpenVINO runner's live sessions, or None when it was never loaded / has no session."""
    mod = sys.modules.get("roop.openvino_runner")
    if mod is None:
        return None
    try:
        rep = mod.report()
    except Exception as exc:                          # noqa: BLE001
        _swallowed("roop/stage_telemetry.py:aux_offload", exc, "OpenVINO block left out of the report")
        return None
    return rep if rep.get("sessions") else None


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

        rates3 = _stage3_rates(s3, c3, running=st[STAGE_3_MAIN_SWAP]["state"] == RUNNING)

        stage1 = {"state": st[STAGE_1_SPARSE_PRESCAN]["state"], "wall_s": s1,
                  "sampled_frames": c1.get("sampled_frames"), "samples_total": c1.get("samples_total"),
                  "stride": c1.get("stride"), "workers": c1.get("workers"),
                  "worker_utilization_pct": c1.get("worker_utilization_pct"),
                  "active_ranges_count": run.get("active_ranges_count"),
                  "skipped_frames_pct": _skipped_pct(),
                  # Keyframe samples per second of the scan's own wall clock, and the cuts it found. None when
                  # the scan was restored from a checkpoint (it did not run) or the scanner did not report them.
                  "keyframes_per_sec": _rate(c1.get("sampled_frames"), s1),
                  "scene_cuts": c1.get("scene_cuts"), "scan_mode": c1.get("scan_mode")}
        stage2 = {"state": st[STAGE_2_ACTIVE_PREPASS]["state"], "wall_s": s2,
                  "active_ranges_count": run.get("active_ranges_count"),
                  "grab_calls": c2.get("grab_calls"), "set_calls": c2.get("set_calls"),
                  "seek_walks": c2.get("seek_walks"),
                  "verified_target_frames": run.get("verified_target_frames"),
                  "detect_interval": c2.get("detect_interval"), "detector_calls": c2.get("detector_calls"),
                  "scan_frames": c2.get("scan_frames"), "coast_frames": c2.get("coast_frames"),
                  "flow_backend": c2.get("flow_backend"), "landmark_filter": c2.get("landmark_filter"),
                  "trajectory_faces": c2.get("trajectory_faces"),
                  # Neural = frames the detector actually ran on; flow = frames it was skipped on and the previous
                  # result was carried by the tracker (`coast`). The ratio is flow / (flow + neural).
                  "neural_detected_frames": c2.get("detector_calls"),
                  "flow_tracked_frames": c2.get("coast_frames"),
                  "flow_ratio_pct": _ratio_pct(c2.get("coast_frames"), c2.get("detector_calls")),
                  "fps": _rate(c2.get("scan_frames"), s2)}
        stage3 = {"state": st[STAGE_3_MAIN_SWAP]["state"], "wall_s": s3,
                  "frames": run.get("total_frames"),
                  "passthrough_frames": c3.get("passthrough_frames"), "gpu_frames": c3.get("gpu_frames"),
                  "swapped_frames": c3.get("swapped_frames"), "faces_swapped": c3.get("faces_swapped"),
                  "peak_vram_gb": _peak(STAGE_3_MAIN_SWAP),
                  # Retired frames / wall, not the trim's planned count / wall: a stage that died at frame 40 of
                  # 600 used to report 600 / wall.
                  "processing_fps": rates3["effective_output_fps"],
                  **rates3,
                  "ops": _ops_report()}
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
        # Frames the writer accepted over the whole pipeline's wall clock (Stages 1-3 plus finalize). Dividing the
        # planned `total` here reported the trim's length as delivered whether or not Stage 3 got there.
        retired_n = rates3["frames_retired"]
        eff_fps = round(retired_n / (pipeline_s + s4), 2) if retired_n and (pipeline_s + s4) > 0 else None
        speedup = estimate_speedup(s1, s2, s3, total, active, restored=restored)
        return {"schema": 1, "ok": bool(ok) and (stage4 is None or stage4["ok"]), "finished_at": now,
                "video_io": _video_io_snapshot(), "aux_offload": _aux_offload(),
                "total_frames": total, "videos_in_job": videos, "total_wall_s": total_wall,
                "effective_fps": eff_fps, "effective_fps_basis": rates3["fps_basis"],
                "gpu_event_timing": _gpu_events.resolve_registered(force=True),
                "speedup": speedup,
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


def _live_rate(stage: Dict[str, Any], now: float) -> Optional[float]:
    """done / elapsed for a RUNNING stage, done / wall once it ended; None before it has a measurement."""
    since, done = stage.get("since"), stage.get("done") or 0
    if since is None or done <= 0 or stage.get("state") in (PENDING, SKIPPED, RESTORED):
        return None
    end = stage.get("ended") if stage.get("ended") is not None else now
    elapsed = end - since
    return round(done / elapsed, 2) if elapsed > 0 else None


def snapshot() -> Dict[str, Any]:
    """JSON-safe view for a telemetry frame. Constant while idle (the sampler sends on change)."""
    # The one place CUDA event pairs are read: this runs on the telemetry sampler (4 Hz, routes_telemetry), never in
    # the frame loop, and `resolve` self-throttles to 250 ms and reads only pairs the GPU has already finished.
    gpu_events = _gpu_events.resolve_registered()
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
                # Stage-local items per second: samples (Stage 1), active-range frames (Stage 2), frames (Stage 3).
                "fps": _live_rate(s, now),
            }
        s3_state = _stages.get(STAGE_3_MAIN_SWAP, {}).get("state")
        if s3_state in (RUNNING, DONE):
            out[STAGE_3_MAIN_SWAP]["ops"] = _ops_report()
            # Stage 3's `fps` is the retired-frame rate (frames the writer accepted), not the progress bar's
            # processed-frame rate that `_live_rate` derived from `done`.
            rates3 = _stage3_rates(out[STAGE_3_MAIN_SWAP]["wall_s"], _counters.get(STAGE_3_MAIN_SWAP) or {},
                                   running=s3_state == RUNNING)
            out[STAGE_3_MAIN_SWAP].update(rates3)
            out[STAGE_3_MAIN_SWAP]["fps"] = rates3["effective_output_fps"]
            if gpu_events is not None:
                out[STAGE_3_MAIN_SWAP]["gpu_events"] = gpu_events
        c1 = _counters.get(STAGE_1_SPARSE_PRESCAN) or {}
        c2 = _counters.get(STAGE_2_ACTIVE_PREPASS) or {}
        out[STAGE_1_SPARSE_PRESCAN]["scene_cuts"] = c1.get("scene_cuts")
        out[STAGE_2_ACTIVE_PREPASS].update(
            neural_detected_frames=c2.get("detector_calls"), flow_tracked_frames=c2.get("coast_frames"),
            flow_ratio_pct=_ratio_pct(c2.get("coast_frames"), c2.get("detector_calls")),
            # Which optical-flow engine carried the frames between detections ('nvof' | 'cv' | 'torch'); the UI names
            # the engine from this and says plain "flow" when it is absent, so it never claims NVOFA unreported.
            flow_backend=c2.get("flow_backend"))
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
    global _videos, _ops_on
    _ops_on = False
    with _ops_lock:
        _ops.clear()
    _reset_retired()
    with _lock:
        _run.clear()
        _stages.clear()
        _counters.clear()
        _finalize.clear()
        _summary.clear()
        _videos = 0
    _vram.stop()
