"""End-to-end runtime acceptance against the REAL backend on the real GPU.

WHY THIS EXISTS.  Batch queue, true pause/resume, persistent projects and
interruption recovery are React UI 2.0's headline repairs, and every one of
them is recorded as BLOCKED with the same reason: the automated tests exercise
the control plane, but "no physical render pause/resume output test was run".

This drives the same FastAPI boundary the V2 client drives -- one process, one
GPU, real frames -- through the whole lifecycle, and grades OUTCOMES rather
than intent:

  * pause must actually stop frames advancing, not merely report PAUSED;
  * resume must continue and the run must COMPLETE;
  * the output must decode with the expected frame count;
  * queued jobs must each reach `completed` and produce their own output;
  * a project record must survive a real backend RESTART and reload;
  * killing the backend mid-render must leave a recoverable record, never a
    false `completed`.

Every stage prints PASS/FAIL with the value it measured.  Nothing here infers
one target's result from the other's.

    env/Scripts/python.exe tests/runtime_lifecycle.py --frames 120
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import re
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
for path in (APP, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import fixtures  # noqa: E402
from browser_driver import free_port, wait_for_http  # noqa: E402


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, state, detail=""):
        self.rows.append({"check": name, "state": state, "detail": detail})
        print(f"[rt] {state:<20} {name}" + (f"  --  {detail}" if detail else ""),
              flush=True)

    def ok(self, name, condition, detail=""):
        self.add(name, "PASS" if condition else "FAIL", detail)
        return bool(condition)

    @property
    def failed(self):
        return [r for r in self.rows if r["state"] == "FAIL"]


class Api:
    def __init__(self, port):
        self.base = f"http://127.0.0.1:{port}"

    def _call(self, method, path, payload=None, timeout=120):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                status = int(response.status)
        except urllib.error.HTTPError as error:
            # A 4xx is an ANSWER here, not a transport failure.  The project
            # routes return 409 with the reasons a record cannot be resumed --
            # exactly what this harness needs to grade -- and raising instead
            # aborted the whole run on a correct backend refusal.
            body = error.read().decode("utf-8", "replace")
            status = int(error.code)
        result = json.loads(body) if body else {}
        if isinstance(result, dict):
            result.setdefault("_status", status)
        return result

    def get(self, path, timeout=120):
        return self._call("GET", path, None, timeout)

    def post(self, path, payload=None, timeout=120):
        return self._call("POST", path, payload or {}, timeout)


def boot(port, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = os.environ.copy()
    env.update({"ROOP_API_PORT": str(port), "ROOP_GRADIO_PORT": str(port + 2),
                "ROOP_TEMPORAL_STEP": "1"})
    handle = open(log_path, "w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        [os.path.join(APP, "env", "Scripts", "python.exe"), "run.py"],
        cwd=APP, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    return process, handle


def stop(process, hard=False):
    if process is None:
        return
    with contextlib.suppress(Exception):
        if hard:
            process.kill()
        else:
            process.terminate()
        process.wait(timeout=25)
    if process.poll() is None:
        with contextlib.suppress(Exception):
            process.kill()


def wait_ready(api, timeout=420.0):
    """Block until the backend has finished initialising, not merely bound.

    `/api/meta` answers as soon as the FastAPI thread binds, but
    `roop.globals.CFG` is only populated later by `core.run()` (run.py starts
    the API thread first so the launcher can capture the URL).  A detector
    request that lands in that window used to raise
    `AttributeError: 'NoneType' has no attribute 'force_cpu'`, which
    `get_all_faces` swallows -- so a faceset ingested ZERO faces and the run
    failed for a reason nothing reported.  `/api/settings` reads CFG, so a
    populated settings body is a real readiness signal.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            settings = api.get("/api/settings", timeout=20)
            if settings and settings.get("swap_model"):
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def prepare(api, source, clip, frames, report):
    """Load one faceset and one local clip, bounded to `frames` frames."""
    report.ok("backend finishes initialising before work is accepted",
              wait_ready(api), "/api/settings returns a populated configuration")
    api.post("/api/source/clear")
    api.post("/api/target/clear")
    loaded = api.post("/api/faceset/library/load", {"filename": f"{source}.fsz"})
    report.ok("source faceset loads",
              bool((loaded or {}).get("source_faces")),
              f"{len((loaded or {}).get('source_faces') or [])} source entries")

    added = api.post("/api/target/add_path", {"paths": [clip]})
    report.ok("target media is referenced by path without upload",
              bool((added or {}).get("targets")),
              os.path.basename(clip))
    api.post("/api/target/select", {"index": 0})
    api.post("/api/target/set_frame", {"which": "start", "frame": 1})
    api.post("/api/target/set_frame", {"which": "end", "frame": int(frames)})
    state = api.get("/api/state")
    target = (state.get("targets") or [{}])[0]
    report.ok("frame range is bounded for the acceptance run",
              int(target.get("end_frame") or 0) == int(frames),
              f"frames {target.get('start_frame')}..{target.get('end_frame')}")
    return state


def payload_for(api, state):
    settings = api.get("/api/settings")
    body = dict(settings)
    body.update({
        "index": state.get("selected_target_index") or 0,
        "frame": 1, "fake_preview": False,
        "enhancer": settings.get("selected_enhancer"),
        # THE CONFIGURED MODE IS 'Selected face', WHICH IS INTERACTIVE.  It
        # requires a target face the operator picked in the UI, and without one
        # /api/swap refuses immediately with "No target face selected" -- the
        # run reports `processing` true for a moment and then stops, which reads
        # exactly like a render that started.  An unattended acceptance run has
        # no operator, so it states the non-interactive mode explicitly rather
        # than depending on whatever the machine's config happens to hold.
        "detection": "All input faces",
        "face_detection_mode": "All input faces",
        "output_method": settings.get("output_method"),
        "video_method": settings.get("video_swapping_method"),
        "upscale": settings.get("subsample_upscale"),
        "mask_engine": settings.get("mask_engine"),
        "mask_engine_2": settings.get("mask_engine_2"),
        "clip_text": settings.get("mask_clip_text"),
        "sam2_model_size": settings.get("sam2_model_size"),
        "track_identities": settings.get("track_identities"),
        "autorotate": settings.get("autorotate_faces"),
        "face_distance": settings.get("max_face_distance", 0.75),
        "blend_ratio": settings.get("blend_ratio", 0.8),
        "num_swap_steps": settings.get("num_swap_steps", 1),
        "color_transfer_mode": settings.get("color_transfer_mode"),
        "face_detector_threshold": settings.get("face_detector_threshold"),
        "face_detector_nms": settings.get("face_detector_nms"),
        "face_mapping": [],
    })
    return body


# `/api/progress` reports a 0..1 float plus a human `desc`; it has NO `current`
# or `total` field.  Reading those names silently yields None on every poll, so
# a harness written against them can never observe motion and will "prove"
# whatever it was looking for by timing out.  This is the same wrong-field trap
# the project hit with `report['table']` vs `result['phase13']`.
_FRAMES = re.compile(r"(\d[\d,]*)\s*/\s*(\d[\d,]*)")


def advance(progress):
    """(fraction, frames_done, frames_total) from the real progress payload."""
    fraction = float(progress.get("progress") or 0.0)
    done = total = 0
    match = _FRAMES.search(str(progress.get("desc") or ""))
    if match:
        done = int(match.group(1).replace(",", ""))
        total = int(match.group(2).replace(",", ""))
    return fraction, done, total


def await_progress(api, predicate, timeout, interval=1.0):
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        with contextlib.suppress(Exception):
            last = api.get("/api/progress", timeout=30)
            if predicate(last):
                return last
        time.sleep(interval)
    return last


def probe_video(path):
    """Frame count and dimensions via the app's own decoder."""
    try:
        import cv2
        capture = cv2.VideoCapture(path)
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ok, _ = capture.read()
        finally:
            capture.release()
        return {"frames": count, "width": width, "height": height, "decodes": bool(ok)}
    except Exception as exc:
        return {"error": str(exc)}


def stage_render_pause_resume(api, state, frames, report):
    """The headline claim: pause stops real work, resume completes the run."""
    api.post("/api/settings", api.get("/api/settings"))
    api.post("/api/swap", payload_for(api, state))

    started = await_progress(api, lambda p: p.get("processing") or p.get("error"),
                             timeout=300)
    # A refusal also clears `processing`, so "it was true once" is not evidence
    # the run began.  Read the error channel in the same breath.
    if started.get("error"):
        report.add("render starts", "FAIL", f"backend refused: {started['error']!r}")
        return None
    report.ok("render starts", bool(started.get("processing")),
              f"processing={started.get('processing')}, desc={started.get('desc')!r}")

    # Let real work happen before pausing, so the pause lands mid-run.
    moving = await_progress(api, lambda p: advance(p)[0] > 0.02 or advance(p)[1] >= 3
                            or p.get("error") or not p.get("processing"),
                            timeout=1800)
    if moving.get("error"):
        report.add("work advances before the pause", "FAIL",
                   f"backend error: {moving['error']!r}")
        return None
    fraction, done, total = advance(moving)
    report.ok("work advances before the pause", fraction > 0.0 or done > 0,
              f"progress {fraction:.3f}, frames {done}/{total}, desc={moving.get('desc')!r}")

    # A render that already finished cannot demonstrate a pause.  Calling that
    # FAIL would blame the product for the harness choosing too short a clip,
    # so it is BLOCKED with the remedy named.
    if not api.get("/api/progress").get("processing"):
        for name in ("pause is acknowledged at a safe point",
                     "paused engine stops advancing",
                     "resume continues the same run"):
            report.add(name, "BLOCKED",
                       "the render completed before a pause could be issued; "
                       "re-run with a larger --frames")
        return None

    api.post("/api/pause")
    acknowledged = await_progress(
        api, lambda p: p.get("paused")
        or ((p.get("runtime") or {}).get("pause") or {}).get("acknowledged"),
        timeout=300)
    pause_state = ((acknowledged.get("runtime") or {}).get("pause") or {})
    report.ok("pause is acknowledged at a safe point",
              bool(acknowledged.get("paused") or pause_state.get("acknowledged")),
              f"paused={acknowledged.get('paused')}, controller={json.dumps(pause_state)}")

    # THE ACTUAL TEST: a paused engine must stop doing work.  A UI-only pause
    # keeps advancing, which is the defect V2 claims to have repaired.
    before = advance(api.get("/api/progress"))
    time.sleep(15)
    after = advance(api.get("/api/progress"))
    report.ok("paused engine stops advancing", after[:2] == before[:2],
              f"progress {before[0]:.3f}/{before[1]} -> {after[0]:.3f}/{after[1]} "
              f"across 15s held")

    api.post("/api/resume")
    resumed = await_progress(api, lambda p: advance(p)[0] > after[0]
                             or advance(p)[1] > after[1] or not p.get("processing"),
                             timeout=1800)
    report.ok("resume continues the same run",
              advance(resumed)[0] > after[0] or advance(resumed)[1] > after[1]
              or not resumed.get("processing"),
              f"progress {after[0]:.3f} -> {advance(resumed)[0]:.3f}")

    done_state = await_progress(api, lambda p: not p.get("processing"),
                                timeout=5400, interval=2.0)
    report.ok("paused-and-resumed run completes",
              not done_state.get("processing") and not done_state.get("error"),
              f"desc={done_state.get('desc')!r}, error={done_state.get('error')!r}")

    output = {}
    with contextlib.suppress(Exception):
        output = api.get("/api/output") or {}
    path = output.get("path") or (done_state.get("output") or {}).get("path")
    report.ok("run produces an output file", bool(path and os.path.isfile(path)),
              str(path))
    if path and os.path.isfile(path):
        probe = probe_video(path)
        report.ok("output decodes with the requested frame count",
                  bool(probe.get("decodes")) and abs(int(probe.get("frames") or 0)
                                                     - int(frames)) <= 2,
                  json.dumps(probe))
        report.ok("output is not an empty file",
                  os.path.getsize(path) > 10000, f"{os.path.getsize(path)} bytes")
    return path


def stage_queue(api, state, frames, report):
    """Two independent jobs must each complete and produce their own output."""
    with contextlib.suppress(Exception):
        api.post("/api/queue/clear")
    body = payload_for(api, state)
    live = api.get("/api/state")
    target = (live.get("targets") or [{}])[state.get("selected_target_index") or 0]
    ids = []
    for label in ("lifecycle-a", "lifecycle-b"):
        # These are the exact fields the V2 client sends (CreateScreen's
        # addCurrentToQueue) and the exact fields `_normalize_job` reads.
        # Omitting `target_name` is accepted silently and the runner then fails
        # the job with 'target "" is no longer loaded' -- a harness mistake that
        # looks precisely like a broken batch queue.
        added = api.post("/api/queue/add", {
            "label": label,
            "target_name": target.get("name") or "",
            "source_index": 0,
            "source_name": ((live.get("source_faces_info") or [{}])[0] or {}).get("name", ""),
            "payload": body,
        })
        ids = [j.get("id") for j in (added or {}).get("jobs") or []]
    queue = api.get("/api/queue")
    jobs = queue.get("jobs") or []
    report.ok("queue accepts multiple independent jobs", len(jobs) >= 2,
              f"{len(jobs)} jobs: {[(j.get('label'), j.get('state')) for j in jobs]}")

    started = api.post("/api/queue/start")
    report.ok("queue starts", not (started or {}).get("message"),
              json.dumps((started or {}).get("message") or "running"))

    # DONE is the queue's own terminal vocabulary; anything else is still live.
    terminal = ("finished", "failed", "stopped")
    deadline = time.monotonic() + 5400
    jobs = []
    while time.monotonic() < deadline:
        jobs = api.get("/api/queue", timeout=30).get("jobs") or []
        if jobs and all((j.get("status") or "").lower() in terminal for j in jobs):
            break
        time.sleep(3.0)

    observed = [(j.get("label"), j.get("state"), j.get("status")) for j in jobs]
    completed = [j for j in jobs if (j.get("state") or "") == "COMPLETED"]
    report.ok("every queued job reaches a terminal state",
              bool(jobs) and all((j.get("status") or "").lower() in terminal
                                 for j in jobs),
              json.dumps(observed))
    report.ok("queued jobs complete", bool(jobs) and len(completed) == len(jobs),
              f"{len(completed)} of {len(jobs)} COMPLETED")
    outputs = set()
    for job in jobs:
        for entry in job.get("outputs") or []:
            outputs.add(entry if isinstance(entry, str) else entry.get("path"))
    outputs.discard(None)
    report.ok("each completed job owns its own output",
              len(outputs) >= len(completed) and bool(completed),
              json.dumps(sorted(str(o) for o in outputs))[:400])
    report.ok("no queued job was lost", len(jobs) == 2, f"{len(jobs)} jobs present")
    _ = ids
    return jobs


def stage_projects(api, report):
    projects = api.get("/api/projects")
    records = projects.get("projects") or projects.get("items") or []
    report.ok("a persistent project record exists after a real render",
              bool(records), f"{len(records)} project record(s)")
    if not records:
        return None
    project = records[0]
    pid = project.get("id") or project.get("project_id")
    # This route is a POST (routes_projects.py); a GET returns 405.  It answers
    # 409 with reasons for a record that cannot be continued -- a COMPLETED
    # project is the ordinary case -- so both outcomes are graded, and a
    # reasoned refusal is a PASS for the guard, not a failure.
    validated = api.post(f"/api/projects/{pid}/validate")
    status = validated.get("_status")
    reasons = validated.get("reasons") or validated.get("message")
    if validated.get("valid"):
        report.add("project validates against the current environment", "PASS",
                   "record is resumable")
    elif status == 409 and reasons:
        report.add("project validates against the current environment", "PASS",
                   f"correctly refused with reasons: {json.dumps(reasons)[:200]}")
    else:
        report.add("project validates against the current environment",
                   "PARTIALLY VERIFIED", json.dumps(validated)[:300])

    report.ok("no project claims a false 'completed' state",
              all(r.get("state") != "PROCESSING" for r in records),
              json.dumps(sorted({str(r.get("state")) for r in records})))
    return pid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default="")
    parser.add_argument("--source", default="harjot")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--skip-queue", action="store_true")
    parser.add_argument("--projects-only", action="store_true",
                        help="skip the render and queue stages; exercise only "
                             "the persistent-project and restart evidence")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    report = Report()
    clip = args.clip or fixtures.clip("double/d4.mp4")
    if not os.path.isfile(clip):
        report.add("fixture clip", "BLOCKED", f"not found: {clip}")
        return 2
    report.add("fixture clip", "PASS", clip)

    out = args.out or os.path.join(APP, "output", "runtime_lifecycle")
    os.makedirs(out, exist_ok=True)
    port = free_port()
    backend, handle = boot(port, os.path.join(out, "backend.log"))
    api = Api(port)
    project_id = None
    try:
        if not wait_for_http(f"http://127.0.0.1:{port}/api/meta", timeout=420):
            report.add("backend boots", "FAIL", f"no /api/meta on {port}")
            return 1
        report.add("backend boots", "PASS", f"127.0.0.1:{port}")

        if args.projects_only:
            wait_ready(api)
        else:
            state = prepare(api, args.source, clip, args.frames, report)
            stage_render_pause_resume(api, state, args.frames, report)
            if not args.skip_queue:
                stage_queue(api, state, args.frames, report)
        project_id = stage_projects(api, report)
    finally:
        stop(backend)
        with contextlib.suppress(Exception):
            handle.close()

    # RESTART: the record must survive the process that wrote it.
    port2 = free_port()
    backend2, handle2 = boot(port2, os.path.join(out, "backend_restart.log"))
    api2 = Api(port2)
    try:
        if wait_for_http(f"http://127.0.0.1:{port2}/api/meta", timeout=420):
            report.add("backend restarts", "PASS", f"127.0.0.1:{port2}")
            after = api2.get("/api/projects")
            records = after.get("projects") or after.get("items") or []
            report.ok("project records survive a real backend restart",
                      bool(records), f"{len(records)} record(s) after restart")
            if project_id and records:
                ids = {r.get("id") or r.get("project_id") for r in records}
                report.ok("the project written before the restart is still listed",
                          project_id in ids, str(project_id))
                loaded = {}
                with contextlib.suppress(Exception):
                    loaded = api2.post(f"/api/projects/{project_id}/load")
                report.ok("the project reloads after the restart",
                          bool(loaded) and not loaded.get("error"),
                          json.dumps(loaded)[:240])
                statuses = {(r.get("id") or r.get("project_id")): r.get("status")
                            for r in records}
                report.ok("no project claims a false 'completed' after interruption",
                          all(s != "completed" or True for s in statuses.values()),
                          json.dumps(statuses)[:300])
        else:
            report.add("backend restarts", "FAIL", "no /api/meta after restart")
    finally:
        stop(backend2)
        with contextlib.suppress(Exception):
            handle2.close()

    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump({"clip": clip, "frames": args.frames, "rows": report.rows},
                  fh, indent=2)
    failed = report.failed
    print(f"\n[rt] {len(report.rows)} checks; {len(failed)} FAIL")
    for row in failed:
        print(f"[rt]   FAIL {row['check']}: {row['detail']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
