#!/usr/bin/env python3
"""Where do the ~5 s before the first face frame of every pre-pass go, and does job 2 pay them again?

Phase 14's re-check (docs/perf/phase14/recheck_2026-10-25/) saw ~5 s before frame 4 in EVERY pre-pass and a
lead that `core.release_resources` -> `face_util.release_face_analyser` drops the analyser at job end, so the
next job rebuilds it. `batch_process_with_options` also calls `release_resources()` as its first step, so the
harness's own back-to-back runs already start from a torn-down analyser.

This runs N real pre-passes in ONE process (like the app: one backend, many jobs), through the same
`capture_prepass` the Phase 14 probe uses (stops right after `_precompute_temporal`, no swap pass), and
timestamps, per job and relative to the job's start:

  release       `core.release_resources()` duration and the VRAM it returned
  analyser      `face_util._build_face_analyser()` duration (buffalo_l bundle + prepare)
  session init  every `onnxruntime.InferenceSession.__init__`: model, duration, and whether it fell INSIDE the
                pre-pass window
  first run     the first `.run` of each session (ORT-TRT loads/builds its engine here, not at init)
  pre-pass      `_precompute_temporal` start / end, and how much of that window is `_build_face_analyser`

so the pre-pass window is split into analyser construction, first-inference warm-up and detection. It measures a cost; it changes nothing.

    app\\env\\Scripts\\python.exe app\\tests\\probe_prepass_session_build.py --clip <clip.mp4> --jobs 3 --frames 150 --out report.json

The app must be stopped (one render at a time).
"""
import argparse
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EVENTS = []                 # (t_abs, kind, detail, dur_s)
STATE = {"in_prepass": False, "seen_runs": set()}
_LOCK = threading.Lock()


def _log(kind, detail="", dur=0.0, t0=None):
    with _LOCK:
        EVENTS.append(dict(t=(t0 if t0 is not None else time.perf_counter()), kind=kind, detail=detail, dur=dur,
                           in_prepass=STATE["in_prepass"], thread=threading.current_thread().name))


def install():
    import onnxruntime as ort
    cls = ort.InferenceSession
    orig_init, orig_run = cls.__init__, cls.run

    def init(self, path_or_bytes, *a, **kw):
        t0 = time.perf_counter()
        try:
            return orig_init(self, path_or_bytes, *a, **kw)
        finally:
            name = os.path.basename(path_or_bytes) if isinstance(path_or_bytes, str) else "<bytes>"
            dur = time.perf_counter() - t0
            if name == "<bytes>" and dur > 0.5:      # who built it: the first frame under app/roop
                import traceback
                site = next((f"{os.path.basename(f.filename)}:{f.lineno}" for f in reversed(traceback.extract_stack())
                             if os.sep + "roop" + os.sep in f.filename and "probe_" not in f.filename
                             and not f.filename.endswith("strict_session.py")), "?")
                name = "<bytes> via " + site
            _log("session_init", name, dur, t0)

    def run(self, *a, **kw):
        sid = id(self)
        first = sid not in STATE["seen_runs"]
        t0 = time.perf_counter()
        try:
            return orig_run(self, *a, **kw)
        finally:
            if first:
                STATE["seen_runs"].add(sid)
                try:
                    name = os.path.basename(self._model_path) if isinstance(getattr(self, "_model_path", None), str) else "?"
                except Exception:
                    name = "?"
                _log("first_run", name, time.perf_counter() - t0, t0)

    cls.__init__, cls.run = init, run

    import roop.core as core
    import roop.face_util as fu
    from roop.ProcessMgr import ProcessMgr

    o_rel = core.release_resources

    def rel():
        t0 = time.perf_counter()
        try:
            import torch
            free0 = torch.cuda.mem_get_info()[0] / 2 ** 20
        except Exception:
            free0 = None
        try:
            return o_rel()
        finally:
            try:
                free1 = torch.cuda.mem_get_info()[0] / 2 ** 20
            except Exception:
                free1 = None
            STATE["seen_runs"].clear()          # sessions are gone; their ids may be reused
            _log("release_resources", "freed_mib=%s" % (None if free0 is None else round(free1 - free0)),
                 time.perf_counter() - t0, t0)
    core.release_resources = rel

    o_build = fu._build_face_analyser

    def build():
        t0 = time.perf_counter()
        try:
            return o_build()
        finally:
            _log("build_face_analyser", "", time.perf_counter() - t0, t0)
    fu._build_face_analyser = build

    o_pre = ProcessMgr._precompute_temporal

    def pre(self, *a, **kw):
        STATE["in_prepass"] = True
        t0 = time.perf_counter()
        _log("prepass_start", "", 0.0, t0)
        try:
            return o_pre(self, *a, **kw)
        finally:
            _log("prepass_end", "", time.perf_counter() - t0, t0)
            STATE["in_prepass"] = False
    ProcessMgr._precompute_temporal = pre



def summarize(job, t_start, t_end):
    ev = [e for e in EVENTS if t_start <= e["t"] <= t_end]
    rel = lambda e: round(e["t"] - t_start, 3)
    by = lambda k: [e for e in ev if e["kind"] == k]
    pre_s = next((e for e in ev if e["kind"] == "prepass_start"), None)
    pre_e = next((e for e in ev if e["kind"] == "prepass_end"), None)
    out = dict(job=job, job_wall_s=round(t_end - t_start, 3))
    out["release_resources"] = [dict(at=rel(e), dur=round(e["dur"], 3), detail=e["detail"]) for e in by("release_resources")]
    out["build_face_analyser"] = [dict(at=rel(e), dur=round(e["dur"], 3)) for e in by("build_face_analyser")]
    inits = by("session_init")
    out["session_init"] = dict(
        count=len(inits), total_s=round(sum(e["dur"] for e in inits), 3),
        inside_prepass_count=sum(1 for e in inits if e["in_prepass"]),
        inside_prepass_s=round(sum(e["dur"] for e in inits if e["in_prepass"]), 3),
        slowest=[dict(model=e["detail"], dur=round(e["dur"], 3), in_prepass=e["in_prepass"])
                 for e in sorted(inits, key=lambda e: -e["dur"])[:6]])
    runs = by("first_run")
    out["first_run"] = dict(
        count=len(runs), total_s=round(sum(e["dur"] for e in runs), 3),
        inside_prepass_s=round(sum(e["dur"] for e in runs if e["in_prepass"]), 3),
        slowest=[dict(model=e["detail"], dur=round(e["dur"], 3), in_prepass=e["in_prepass"])
                 for e in sorted(runs, key=lambda e: -e["dur"])[:6]])
    if pre_s and pre_e:
        builds_in = [e for e in by("build_face_analyser") if pre_s["t"] <= e["t"] <= pre_e["t"] + pre_e["dur"]]
        in_s = sum(e["dur"] for e in builds_in)
        out["prepass"] = dict(start_at=rel(pre_s), wall_s=round(pre_e["dur"], 3),
                              analyser_builds_inside=len(builds_in), analyser_build_inside_s=round(in_s, 3),
                              build_share_of_prepass=round(in_s / pre_e["dur"], 3) if pre_e["dur"] else None,
                              prepass_minus_build_s=round(pre_e["dur"] - in_s, 3),
                              build_offsets_after_start_s=[round(e["t"] - pre_s["t"], 2) for e in builds_in],
                              setup_before_prepass_s=round(pre_s["t"] - t_start, 3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--source", default="akansha")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.clip = os.path.abspath(args.clip)
    if args.out:
        args.out = os.path.abspath(args.out)

    import probe_phase14_prepass as P         # the pipeline changes cwd while loading, so paths are absolute above
    install()
    jobs = []
    for n in range(args.jobs):
        t0 = time.perf_counter()
        r = P.capture_prepass(args.clip, args.source, 1, args.frames)
        t1 = time.perf_counter()
        s = summarize(n, t0, t1)
        s["detector_calls"] = r["schedule"].get("detector_calls")
        s["frames"] = r["schedule"].get("frames")
        jobs.append(s)
        p = s.get("prepass", {})
        print("[session-build] job %d: wall %.1f s | setup before pre-pass %s s | pre-pass %s s = analyser build inside %s s "
              "(%s builds at +%s s, %s of it) + rest %s s | first runs inside %.2f s | release_resources %s" % (
                  n, s["job_wall_s"], p.get("setup_before_prepass_s"), p.get("wall_s"), p.get("analyser_build_inside_s"),
                  p.get("analyser_builds_inside"), p.get("build_offsets_after_start_s"), p.get("build_share_of_prepass"),
                  p.get("prepass_minus_build_s"), s["first_run"]["inside_prepass_s"],
                  [(r["at"], r["dur"], r["detail"]) for r in s["release_resources"]]), flush=True)
    report = dict(clip=args.clip, frames=args.frames, jobs=jobs)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, default=str)
        print("[session-build] wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
