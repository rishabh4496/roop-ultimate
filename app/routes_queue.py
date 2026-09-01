"""Server-side batch queue.

The queue used to live entirely in the browser: a `useState([])` in FaceSwap.jsx
whose runner effect POSTed /api/swap job by job. That meant the queue was only
as durable as the tab. A reload, a Pinokio tab switch, or a closed window lost
every remaining job — silently, mid-batch, after possibly hours of rendering.
Every other piece of long-lived UI state (timeline markers, profiles, grid
selections) already survived a reload; the queue, the longest-lived state in the
app, was the one thing that did not.

So the queue lives here instead:

  * the job list is persisted to app/queue.json after every mutation, so it
    survives a browser reload AND an app restart;
  * the runner is a daemon thread in this process, so a batch keeps going with
    no browser attached at all.

The browser is now a view onto this: it polls /api/queue and renders what it
finds. It no longer decides anything.

WHAT A JOB CARRIES
------------------
`payload` is the exact body the client would have POSTed to /api/swap. Keeping
it verbatim rather than re-deriving it here means the queue path and the single
-run path send byte-identical requests, so a new setting cannot reach one and
miss the other — the drift that hand-maintained payload copies always produce.

The runner supplies only what it alone knows: `target_index`, resolved from
`target_name` at dispatch time. Resolving by NAME is deliberate — a stored index
goes stale as soon as a target is removed after queueing, and the job then
silently swaps the wrong file.
"""

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

import json
import os
import threading
import time
import uuid

import roop.globals as roop_globals
import api_state as state


router = APIRouter()

# ── Injected by api.py at import time ────────────────────────────────────────
# All mutated in place and never rebound there, so binding the same objects here
# keeps one shared value rather than two that drift.
_progress = None            # the live run's progress dict
list_files_process = None   # loaded target ProcessEntry list
_run_swap = None            # api.py's blocking single-run entry point
_stop_current = None        # api.py's stop_swap(), to abort the in-flight job
_snapshot_outputs = None    # post_swap._snapshot_output_mtimes
_outputs_since = None       # post_swap._outputs_since
# True while the hardware benchmark holds the GPU. /api/swap refuses to start a
# render then and the benchmark refuses to start during one, but the queue is a
# THIRD way into `_run_swap` that neither guard covers: it calls it directly, so
# a batch dispatching its next job would run alongside a benchmark that is
# holding several pools of TensorRT contexts — an OOM on a 12GB card, and a
# benchmark measuring a card that is busy rendering. Injected as a predicate
# rather than imported to keep the one-way api.py -> routes_queue dependency.
_benchmark_running = lambda: False           # noqa: E731

QUEUE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queue.json")

# Canonical job lifecycle.  `status` below is retained as a compatibility
# projection for the existing V1 queue consumer; new clients should use state.
JOB_STATES = (
    "QUEUED", "PREPARING", "PROCESSING", "PAUSED", "COMPLETED",
    "FAILED", "CANCELLED", "INTERRUPTED", "RECOVERABLE",
)
SCHEMA_VERSION = 2
_LEGACY_STATUS = {
    "QUEUED": "pending",
    "PREPARING": "running",
    "PROCESSING": "running",
    "PAUSED": "running",
    "COMPLETED": "finished",
    "FAILED": "failed",
    "CANCELLED": "stopped",
    "INTERRUPTED": "stopped",
    "RECOVERABLE": "pending",
}
_LEGACY_STATE = {
    "pending": "QUEUED",
    "running": "PROCESSING",
    "finished": "COMPLETED",
    "failed": "FAILED",
    "stopped": "CANCELLED",
}
RUNNABLE_STATES = ("QUEUED", "RECOVERABLE")
DONE = ("finished", "failed", "stopped")

_lock = threading.RLock()
_queue = {"jobs": [], "running": False, "paused": False, "current": None}
_runner = None
# Bumped by every queue_start. A runner only keeps going while its own token is
# still the current one, so a thread that outlives the batch it was started for
# retires instead of dispatching into the next one — see _loop.
_generation = 0
_cancel_requested = set()


def _empty_progress():
    return {
        "fraction": 0.0,
        "frames_done": None,
        "frames_total": None,
        "fps": None,
        "eta_s": None,
        "phase": "QUEUED",
        "updated_at": time.time(),
    }


def _state(job):
    """Read canonical state, accepting old tests/clients that edit status."""
    current = job.get("state")
    legacy = _LEGACY_STATE.get(job.get("status"))
    if current not in JOB_STATES:
        return legacy or "QUEUED"
    # This compatibility case matters for persisted V1 records and for V1's
    # edit/retry calls, which only know about status.
    if current == "QUEUED" and legacy and legacy != "QUEUED":
        return legacy
    return current


def _set_state(job, new_state):
    if new_state not in JOB_STATES:
        raise ValueError(f"unknown queue state: {new_state}")
    job["state"] = new_state
    job["status"] = _LEGACY_STATUS[new_state]
    job["recoverable"] = new_state == "RECOVERABLE"
    progress = job.setdefault("progress", _empty_progress())
    progress["phase"] = new_state
    progress["updated_at"] = time.time()


def _progress_snapshot():
    live = _progress or {}
    fraction = live.get("progress", 0.0)
    try:
        fraction = max(0.0, min(1.0, float(fraction)))
    except (TypeError, ValueError):
        fraction = 0.0
    return {
        "fraction": fraction,
        "frames_done": live.get("frames_done", live.get("frame_done")),
        "frames_total": live.get("frames_total", live.get("frame_total")),
        "fps": live.get("fps", live.get("current_fps")),
        "eta_s": live.get("eta_s"),
        "phase": live.get("desc") or "UNKNOWN",
        "updated_at": time.time(),
    }


def _normalize_loaded_job(job):
    """Migrate V1 records without making an interrupted render look complete."""
    state_value = job.get("state")
    legacy = job.get("status")
    if state_value not in JOB_STATES:
        state_value = _LEGACY_STATE.get(legacy, "QUEUED")
    if state_value in ("PREPARING", "PROCESSING", "PAUSED") or legacy == "running":
        state_value = "RECOVERABLE"
        job["error"] = "interrupted by an app restart; start the queue to retry"
        job["interrupted_at"] = time.time()
    job.setdefault("schema_version", SCHEMA_VERSION)
    job.setdefault("progress", _empty_progress())
    job.setdefault("outputs", [])
    job.setdefault("cancel_requested", False)
    job.setdefault("recoverable", state_value == "RECOVERABLE")
    job.setdefault("error", "")
    job.setdefault("started", 0.0)
    job.setdefault("finished", 0.0)
    _set_state(job, state_value)
    return job


# ── Persistence ──────────────────────────────────────────────────────────────
def _save():
    """Write the job list out. Runner flags are deliberately NOT persisted: a
    queue that was running when the app died must come back idle, or the runner
    would start swapping the moment the process comes up, with no one watching."""
    try:
        tmp = QUEUE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"jobs": _queue["jobs"]}, fh, indent=1)
        os.replace(tmp, QUEUE_FILE)      # atomic: a crash mid-write keeps the old file
    except Exception as e:
        _say(f"[Queue] could not save {QUEUE_FILE}: {e}")


def load():
    """Restore the persisted queue at startup. Any job left marked 'running' was
    interrupted by whatever ended the last process, so it is put back to pending
    — it never finished, and the alternative is a job that can never run again."""
    try:
        if not os.path.exists(QUEUE_FILE):
            return
        with open(QUEUE_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        jobs = [_normalize_loaded_job(job) for job in (data.get("jobs") or [])]
        for job in jobs:
            if job.get("status") == "running":
                job["status"] = "pending"
                job["error"] = "interrupted by an app restart — re-queued"
        with _lock:
            _queue["jobs"] = jobs
        if jobs:
            pending = sum(1 for j in jobs if _state(j) in RUNNABLE_STATES)
            _say(f"[Queue] restored {len(jobs)} job(s) from queue.json "
                 f"({pending} still to run)")
    except Exception as e:
        _say(f"[Queue] could not load {QUEUE_FILE}: {e}")


def _snapshot():
    with _lock:
        jobs = []
        for original in _queue["jobs"]:
            job = dict(original)
            job["state"] = _state(original)
            job["status"] = _LEGACY_STATUS[job["state"]]
            job["position"] = len(jobs) + 1
            if job["id"] == _queue["current"]:
                job["progress"] = _progress_snapshot()
            else:
                job["progress"] = dict(original.get("progress") or _empty_progress())
            jobs.append(job)
        return {
            "schema_version": SCHEMA_VERSION,
            "job_states": list(JOB_STATES),
            "jobs": jobs,
            "running": _queue["running"],
            "paused": _queue["paused"],
            "current": _queue["current"],
        }


def _find(job_id):
    for job in _queue["jobs"]:
        if job["id"] == job_id:
            return job
    return None


# ── Mutations ────────────────────────────────────────────────────────────────
@router.get("/api/queue")
def queue_list():
    return _snapshot()


def _normalize_job(payload: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": uuid.uuid4().hex[:12],
        "target_name": str(payload.get("target_name") or ""),
        "source_index": int(payload.get("source_index") or 0),
        "source_name": str(payload.get("source_name") or ""),
        "payload": payload.get("payload") or {},
        "frame_start": payload.get("frame_start"),
        "frame_end": payload.get("frame_end"),
        "label": str(payload.get("label") or ""),
        "status": "pending",
        "error": "",
        "added": time.time(),
        "started": 0.0,
        "finished": 0.0,
        "state": "QUEUED",
        "progress": _empty_progress(),
        "outputs": [],
        "cancel_requested": False,
        "recoverable": False,
    }


@router.post("/api/queue/add")
def queue_add(payload: dict = Body(...)):
    """Append a job. `payload.payload` is the /api/swap body to replay."""
    job = _normalize_job(payload)
    with _lock:
        _queue["jobs"].append(job)
        _save()
    return _snapshot()


@router.post("/api/queue/add_batch")
def queue_add_batch(payload: dict = Body(...)):
    """Append multiple jobs in one atomic transaction."""
    raw_jobs = payload.get("jobs") or []
    new_jobs = [_normalize_job(j) for j in raw_jobs]
    with _lock:
        _queue["jobs"].extend(new_jobs)
        _save()
    return _snapshot()


@router.post("/api/queue/remove")
def queue_remove(payload: dict = Body(...)):
    job_id = str(payload.get("id") or "")
    with _lock:
        if _queue["current"] == job_id:
            return JSONResponse(status_code=409,
                                content={"message": "that job is running — stop it first"})
        _queue["jobs"] = [j for j in _queue["jobs"] if j["id"] != job_id]
        _save()
    return _snapshot()


@router.post("/api/queue/clear")
def queue_clear(payload: dict = Body(default={})):
    """Empty the queue. `keep_running` leaves the in-flight job in place so
    clearing during a batch does not orphan the job that is mid-render."""
    with _lock:
        cur = _queue["current"]
        if cur and payload.get("keep_running", True):
            _queue["jobs"] = [j for j in _queue["jobs"] if j["id"] == cur]
        else:
            _queue["jobs"] = []
            _queue["running"] = False
        _save()
    return _snapshot()


@router.post("/api/queue/reorder")
def queue_reorder(payload: dict = Body(...)):
    """Reorder to match `ids`. Any job not named is appended in its existing
    order, so a stale client list can only mis-order — never drop a job."""
    ids = [str(i) for i in (payload.get("ids") or [])]
    with _lock:
        by_id = {j["id"]: j for j in _queue["jobs"]}
        ordered = [by_id.pop(i) for i in ids if i in by_id]
        ordered.extend(j for j in _queue["jobs"] if j["id"] in by_id)
        _queue["jobs"] = ordered
        _save()
    return _snapshot()


@router.post("/api/queue/update")
def queue_update(payload: dict = Body(...)):
    """Edit a job that has not started. Running/finished jobs are refused rather
    than silently ignored: editing the settings of a job that already rendered
    would leave the queue describing something the output does not match."""
    job_id = str(payload.get("id") or "")
    with _lock:
        job = _find(job_id)
        if job is None:
            return JSONResponse(status_code=404, content={"message": "no such job"})
        if _state(job) in ("PREPARING", "PROCESSING", "PAUSED"):
            return JSONResponse(status_code=409, content={"message": "job is running"})
        for key in ("payload", "target_name", "source_index", "source_name",
                    "label", "frame_start", "frame_end"):
            if key in payload:
                job[key] = payload[key]
        # An edited job is worth running again even if it already ran.
        if payload.get("requeue") or _state(job) in ("COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"):
            _set_state(job, "QUEUED")
            job["error"] = ""
            job["started"] = job["finished"] = 0.0
            job["progress"] = _empty_progress()
            job["cancel_requested"] = False
        _save()
    return _snapshot()


@router.post("/api/queue/duplicate")
def queue_duplicate(payload: dict = Body(...)):
    """Copy a job in as pending, directly after the original."""
    job_id = str(payload.get("id") or "")
    with _lock:
        job = _find(job_id)
        if job is None:
            return JSONResponse(status_code=404, content={"message": "no such job"})
        clone = dict(job)
        clone.update({"id": uuid.uuid4().hex[:12], "state": "QUEUED", "status": "pending", "error": "",
                      "added": time.time(), "started": 0.0, "finished": 0.0})
        clone["payload"] = dict(job.get("payload") or {})
        clone["progress"] = _empty_progress()
        clone["outputs"] = []
        clone["cancel_requested"] = False
        clone["recoverable"] = False
        _queue["jobs"].insert(_queue["jobs"].index(job) + 1, clone)
        _save()
    return _snapshot()


@router.post("/api/queue/retry")
def queue_retry(payload: dict = Body(default={})):
    """Put failed/stopped jobs back to pending (one by id, or all of them)."""
    job_id = payload.get("id")
    with _lock:
        targets = ([_find(str(job_id))] if job_id
                   else [j for j in _queue["jobs"]
                         if _state(j) in ("FAILED", "CANCELLED", "INTERRUPTED")])
        for job in targets:
            if job is not None and _state(job) not in ("PREPARING", "PROCESSING", "PAUSED"):
                _set_state(job, "QUEUED")
                job.update({"error": "", "started": 0.0, "finished": 0.0,
                            "progress": _empty_progress(), "cancel_requested": False})
        _save()
    return _snapshot()


# ── Runner ───────────────────────────────────────────────────────────────────
def _apply_segment(entry, job):
    """Point the target's trim range at this job's segment, when it has one.

    The range lives on the ProcessEntry rather than in the swap payload (see
    /api/target/set_frame), which is exactly why a multi-segment render has to
    be several jobs: each one sets the range it wants, immediately before it
    runs, instead of every job inheriting whatever the UI last left there."""
    fs, fe = job.get("frame_start"), job.get("frame_end")
    if fs is None and fe is None:
        return
    total = getattr(entry, "total_frames", 0) or entry.endframe or 0
    if fs is not None:
        entry.startframe = max(0, int(fs))
    if fe is not None:
        entry.endframe = min(int(fe), total) if total else int(fe)
    if entry.endframe and entry.startframe > entry.endframe:
        entry.startframe, entry.endframe = entry.endframe, entry.startframe


def _next_pending():
    for job in _queue["jobs"]:
        if _state(job) in RUNNABLE_STATES:
            return job
    return None


def _run_one(job):
    """Dispatch one job and block until it finishes. Returns (state, error)."""
    job["processing_started"] = False
    if job.get("cancel_requested") or job["id"] in _cancel_requested:
        return "CANCELLED", "cancelled before processing"
    names = [os.path.basename(getattr(e, "filename", "") or "") for e in list_files_process]
    try:
        idx = names.index(os.path.basename(job["target_name"]))
    except ValueError:
        return "FAILED", f'target "{job["target_name"]}" is no longer loaded'

    state.selected_target_index = idx

    # Dynamically re-resolve source_index by source_name if faceset list shifted
    src_idx = int(job.get("source_index") or 0)
    target_src_name = str(job.get("source_name") or "").strip()
    if target_src_name and getattr(state, "source_faces_info", None):
        for i, info in enumerate(state.source_faces_info):
            info_name = str(info.get("name") or "").strip()
            if info_name and info_name.lower() == target_src_name.lower():
                src_idx = i
                break
    state.selected_input_face_index = src_idx

    _apply_segment(list_files_process[idx], job)

    payload = dict(job.get("payload") or {})
    payload["target_index"] = idx

    # Sanitize face_mapping payload so null/None entries don't crash ProcessMgr
    fm = payload.get("face_mapping")
    if isinstance(fm, list):
        payload["face_mapping"] = [0 if (x is None or x == "") else int(x) for x in fm]

    _set_state(job, "PROCESSING")
    job["processing_started"] = True
    job["progress"] = _empty_progress()
    _progress.update({"processing": True, "paused": False, "progress": 0.0,
                      "desc": "Starting queued job…", "error": ""})
    before = _snapshot_outputs() if _snapshot_outputs else None
    _run_swap(payload)          # blocking; clears processing in its own finally

    # Auto-Fallback Enhancer Retry on Error
    if _progress.get("error") and (payload.get("auto_fallback") or job.get("auto_fallback")):
        err_msg = str(_progress["error"])
        if payload.get("enhancer") != "None":
            _progress.update({"error": "", "desc": f"Auto-retrying with enhancer=None due to error: {err_msg}"})
            payload["enhancer"] = "None"
            _run_swap(payload)

    if before is not None and _outputs_since:
        try:
            job["outputs"] = list(_outputs_since(before))
        except Exception:
            job["outputs"] = []

    if job.get("cancel_requested") or job["id"] in _cancel_requested:
        return "CANCELLED", "cancelled by user"
    if _progress.get("error"):
        return "FAILED", str(_progress["error"])
    # A deliberate Stop leaves no error but did not finish the job. Report it as
    # its own state so "I stopped this one" is distinguishable from "it crashed"
    # — they want different things from the Retry button.
    if not roop_globals.processing and _progress.get("progress", 0.0) < 1.0:
        return "INTERRUPTED", "processing stopped before completion"
    return "COMPLETED", ""


def _say(msg):
    """Console line for the queue.

    Deliberately ASCII and deliberately swallowing its own errors. This runs on
    the runner thread, and stdout here is a cp1252 console: a single non-ASCII
    character (a '->' arrow was enough) raises UnicodeEncodeError out of print,
    out of the loop, and kills the batch — after the job that just finished had
    already been saved, so the only symptom is a queue that silently stops."""
    try:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)
    except Exception:
        pass


def _loop(gen):
    """Run jobs until the queue empties or is stopped.

    `gen` is this runner's generation token. Stop → Start opens a window where a
    second runner is legitimately started while this one is still alive: Stop
    only sets flags and returns, and `_run_swap` clears `_progress[processing]`
    in its own `finally` — so by the time this thread reaches the bookkeeping
    below, both of queue_start's guards ("not running", "nothing else
    processing") already pass. Without the token this thread would then come
    round the loop, see running=True again, and dispatch the NEXT pending job
    alongside the new runner: two renders into one output directory, through one
    set of ProcessMgr globals.
    """
    while True:
        with _lock:
            if not _queue["running"] or gen != _generation:
                break
            if _queue["paused"] or _benchmark_running():
                # Held, not failed. The benchmark refuses to start while a
                # render is processing, but BETWEEN two jobs nothing is
                # processing, so it can legitimately start there and this thread
                # would otherwise dispatch the next job straight into it. The
                # batch resumes on its own when the benchmark finishes.
                job = None
            else:
                job = _next_pending()
                if job is None:
                    _queue["running"] = False
                    _queue["current"] = None
                    _save()
                    _say("[Queue] all jobs finished")
                    break
                _set_state(job, "PREPARING")
                job["started"] = time.time()
                job["error"] = ""
                job["cancel_requested"] = False
                _queue["current"] = job["id"]
                _save()
        if job is None:
            time.sleep(0.25)     # paused — hold without burning the CPU
            continue

        _say(f'[Queue] running "{job["target_name"]}"')
        try:
            status, error = _run_one(job)
        except Exception as e:          # a crashed job must not kill the batch
            import traceback
            traceback.print_exc()
            status, error = "FAILED", str(e)

        try:
            with _lock:
                live = _find(job["id"])   # it may have been edited or removed meanwhile
                if live is not None:
                    _set_state(live, status)
                    live["error"] = error
                    live["finished"] = time.time()
                    # A target-resolution or pre-dispatch failure must not
                    # inherit the previous job's shared progress snapshot.
                    if status in ("FAILED", "CANCELLED") and not job.get("processing_started"):
                        live["progress"] = dict(job.get("progress") or _empty_progress())
                    else:
                        live["progress"] = _progress_snapshot()
                    if status == "COMPLETED":
                        live["progress"]["fraction"] = 1.0
                    live["cancel_requested"] = False
                    live.pop("processing_started", None)
                _cancel_requested.discard(job["id"])
                _queue["current"] = None
                _save()
            _say(f'[Queue] "{job["target_name"]}" -> {status}' + (f" ({error})" if error else ""))
        except Exception:
            # Bookkeeping must never end the batch either — the job itself is
            # over, and the remaining jobs are the whole point of a queue.
            import traceback
            traceback.print_exc()


@router.post("/api/queue/start")
def queue_start():
    global _runner, _generation
    with _lock:
        if _queue["running"]:
            return _snapshot()
        if not any(_state(j) in RUNNABLE_STATES for j in _queue["jobs"]):
            return JSONResponse(status_code=400,
                                content={"message": "nothing left to run — retry or add a job"})
        if _progress["processing"]:
            return JSONResponse(status_code=409,
                                content={"message": "a single run is already processing"})
        if _benchmark_running():
            return JSONResponse(status_code=409,
                                content={"message": "the hardware benchmark is running — "
                                                    "cancel it first, or wait for it to finish"})
        _queue["running"] = True
        _queue["paused"] = False
        _generation += 1
        gen = _generation
    _runner = threading.Thread(target=_loop, args=(gen,), name="queue-runner", daemon=True)
    _runner.start()
    return _snapshot()


@router.post("/api/queue/pause")
def queue_pause():
    """Pause the batch: hold the in-flight job (the swap loop honours
    roop_globals.pause) and hold dispatch of the next one."""
    with _lock:
        _queue["paused"] = True
        current = _find(_queue["current"]) if _queue["current"] else None
        if current is not None and _state(current) == "PROCESSING":
            _set_state(current, "PAUSED")
    if _progress["processing"]:
        roop_globals.pause = True
        _progress["paused"] = True
    return _snapshot()


@router.post("/api/queue/resume")
def queue_resume():
    with _lock:
        _queue["paused"] = False
        current = _find(_queue["current"]) if _queue["current"] else None
        if current is not None and _state(current) == "PAUSED":
            _set_state(current, "PROCESSING")
    if _progress["processing"]:
        roop_globals.pause = False
        _progress["paused"] = False
    return _snapshot()


@router.post("/api/queue/cancel")
def queue_cancel(payload: dict = Body(...)):
    """Cancel one queued job, or request cooperative cancellation of the
    current job without changing the state of unrelated jobs."""
    job_id = str(payload.get("id") or "")
    if not job_id:
        return JSONResponse(status_code=400, content={"message": "job id is required"})
    stop_current = False
    with _lock:
        job = _find(job_id)
        if job is None:
            return JSONResponse(status_code=404, content={"message": "no such job"})
        current_state = _state(job)
        if current_state in ("COMPLETED", "FAILED", "CANCELLED"):
            return JSONResponse(status_code=409,
                                content={"message": f"job is already {current_state.lower()}"})
        job["cancel_requested"] = True
        if _queue["current"] == job_id:
            _cancel_requested.add(job_id)
            stop_current = True
        else:
            _set_state(job, "CANCELLED")
            job["error"] = "cancelled before processing"
            job["finished"] = time.time()
        _save()
    if stop_current and _stop_current is not None:
        _stop_current()
    return _snapshot()


@router.post("/api/queue/join")
def queue_join(payload: dict = Body(...)):
    """Concatenate the outputs of several finished jobs into one file.

    This is what makes multi-segment rendering a single deliverable rather than
    N clips: each segment is its own run (the trim range lives on the target,
    so it has to be), and this puts them back together in the order given.

    Stream copy, via the concat demuxer. That is safe precisely BECAUSE the
    segments came from one source through one settings set — same codec, same
    resolution, same rate — which is the one case concat-copy is reliable in. A
    copy that fails is reported rather than silently re-encoded: re-encoding
    would quietly halve the quality of a finished render.
    """
    import subprocess
    from roop.ffmpeg_writer import FFMPEG_BINARY

    ids = [str(i) for i in (payload.get("ids") or [])]
    with _lock:
        jobs = [j for j in (_find(i) for i in ids) if j is not None]
    files = [f for job in jobs for f in (job.get("outputs") or []) if os.path.exists(f)]
    if len(files) < 2:
        return JSONResponse(status_code=400, content={
            "message": "need at least two finished segment outputs to join"})

    exts = {os.path.splitext(f)[1].lower() for f in files}
    if len(exts) > 1:
        return JSONResponse(status_code=400, content={
            "message": f"segments have different formats ({', '.join(sorted(exts))}) "
                       f"— re-render them with one output format"})

    out_dir = os.path.dirname(files[0])
    ext = exts.pop()
    stem = str(payload.get("name") or "").strip() or (
        os.path.splitext(os.path.basename(files[0]))[0] + "_joined")
    stem = os.path.basename(stem)                       # never escape the output dir
    dest = os.path.join(out_dir, stem + ext)
    n = 1
    while os.path.exists(dest):                         # never overwrite a render
        dest = os.path.join(out_dir, f"{stem}_{n}{ext}")
        n += 1

    listing = os.path.join(out_dir, f".concat_{uuid.uuid4().hex[:8]}.txt")
    try:
        with open(listing, "w", encoding="utf-8") as fh:
            for f in files:
                # The concat demuxer takes a quoted path with ' escaped.
                fh.write("file '%s'\n" % f.replace("'", r"'\''"))
        cmd = [FFMPEG_BINARY, "-hide_banner", "-y", "-f", "concat", "-safe", "0",
               "-i", listing, "-c", "copy", dest]
        kwargs = {"creationflags": 0x08000000} if os.name == "nt" else {}
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, **kwargs)
        if proc.returncode != 0 or not os.path.exists(dest):
            return JSONResponse(status_code=500, content={
                "message": f"ffmpeg concat failed (exit {proc.returncode}) — see the terminal log"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
    finally:
        try:
            os.remove(listing)
        except OSError:
            pass

    _say(f"[Queue] joined {len(files)} segment(s) -> {os.path.basename(dest)}")
    return {"path": dest, "name": os.path.basename(dest), "segments": len(files)}


@router.post("/api/queue/stop")
def queue_stop():
    """Stop the batch AND whatever it is rendering. Without the second half the
    runner would sit blocked in _run_swap until the current job finished on its
    own, so 'Stop queue' would appear to do nothing for the next hour."""
    current_id = None
    with _lock:
        _queue["running"] = False
        _queue["paused"] = False
        current_id = _queue["current"]
        if current_id:
            current = _find(current_id)
            if current is not None:
                current["cancel_requested"] = True
                _cancel_requested.add(current_id)
        _save()
    if _progress["processing"] and _stop_current is not None:
        _stop_current()
    return _snapshot()
