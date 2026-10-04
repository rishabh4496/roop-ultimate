"""The performance baseline snapshot: one run per clip, everything a later stage must match.

    env/Scripts/python.exe tests/baseline_snapshot.py --date 2026-10-04

For each clip it drives `two_face_video.py` -- the end-to-end harness, with the
swap model / enhancer / mask engine / provider / codec / stabilizers read from
`app/config.yaml` LIVE (never `baseline_controlled`'s own defaults, which are
GPEN 256 Pro + RealityUX and are not what this user runs) -- with ROOP_PROFILE=1,
and writes `docs/perf/baseline_<date>.md` (+ `.json`) holding:

  * STAGE TIMING, BASELINE COUNTERS, SWAP AUDIT and the per-person verdicts;
  * the `[Session]` lines (model, bound provider, TRT fp16, input shape), the
    `[VRAM]` samples, the `[Threads]` dump and the `[StabGeometry]` line;
  * fps beside faces/s, free RAM and the effective ROOP_STAB_CHUNK_MB;
  * output hashes (file sha256, decoded-video md5, rows.csv sha256).

It never renders two clips at once (a render holds 12-15 GB) and never edits the
code under test. Per AGENTS.md: the first arm pays the TensorRT engine build, so a
throw-away WARM-UP render goes first; d4 is rendered a second time at the end as
the NULL CONTROL, which is what lets a later stage say "not measurable" instead of
reporting what the noise produced.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import baseline_controlled as bc       # noqa: E402  parsers shared with the Phase-2 baseline
import fixtures                        # noqa: E402
import telemetry as tel                # noqa: E402

# name -> workload. `window` is (start, end) in source frames; end 0 = to the end.
#
# d4 and Love keep the capture pins earlier sessions settled on (d4: the locked
# fixture's frame 4930; Love: --capture 3024 --capture-face 0, the woman -- the
# auto-capture picks the man on this clip). d1 and d6 use the app's own
# auto-capture with a budget large enough that the scan finishes rather than being
# cut off by the wall clock (a time-bounded scan is a different fixture on a
# slower minute); the seed frame it chose is printed in the log and recorded.
CLIPS = {
    "d4":   {"rel": "double/d4.mp4", "sources": "harjot,ashna", "window": (0, 600),
             "capture": 4930, "capture_face": None},
    "d1":   {"rel": "double/d1.mp4", "sources": "harjot,shambhavi", "window": (0, 0),
             "capture": None, "capture_face": None},
    "d6":   {"rel": "double/d6.mp4", "sources": "harjot,mahek", "window": (0, 0),
             "capture": None, "capture_face": None},
    "Love": {"rel": "Love.mp4", "sources": "harjot", "window": (1100, 1700),
             "capture": 3024, "capture_face": 0},
}
ORDER = ["d4", "d1", "d6", "Love"]
CAPTURE_BUDGET_S = 900.0


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def decoded_md5(path, env):
    """md5 of the DECODED video stream: stable across container/mux differences."""
    try:
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0",
                              "-f", "md5", "-"], capture_output=True, text=True,
                             timeout=1800, env=env)
        m = re.search(r"MD5=([0-9a-f]{32})", out.stdout)
        return m.group(1) if m else None
    except Exception:
        return None


def find_output(workdir):
    best = None
    for name in os.listdir(workdir):
        if name.lower().endswith(".mp4") and name != "clip.mp4" and not name.startswith("."):
            full = os.path.join(workdir, name)
            if best is None or os.path.getmtime(full) > os.path.getmtime(best):
                best = full
    return best


def block_after(text, header_re, stop_re):
    m = re.search(header_re, text)
    if not m:
        return ""
    rest = text[m.start():]
    end = re.search(stop_re, rest[len(m.group(0)):], re.S)
    return rest if not end else rest[:len(m.group(0)) + end.start()]


def parse_log(text):
    """Everything the baseline records, from the render's own output."""
    out = {}
    out["sessions"] = re.findall(r"^\[Session\] .*$", text, re.M)
    out["vram"] = re.findall(r"^\[VRAM\] .*$", text, re.M)
    out["thread_lines"] = re.findall(r"^\[Threads\].*$", text, re.M)
    out["stab_geometry"] = re.findall(r"^\[StabGeometry\] .*$", text, re.M)
    out["stab_lines"] = [l for l in re.findall(r"^\[Stabilize\].*$", text, re.M)]
    out["runtime_lines"] = re.findall(r"^\[(?:Runtime|RuntimeScheduler|BatchSwap|CPU)\].*$", text, re.M)
    out["memory_stages"] = re.findall(r"^\[Memory\] .*$", text, re.M)
    out["pipeline_lines"] = re.findall(r"^\[Pipeline\] .*$", text, re.M)
    out["pipeline_done"] = next((l for l in out["pipeline_lines"] if "done:" in l), None)
    mfps = re.search(r"= ([\d.]+) fps", out["pipeline_done"] or "")
    out["frame_loop_fps"] = float(mfps.group(1)) if mfps else None
    mfs = re.search(r"([\d.]+) faces/frame, ([\d.]+) faces/s", out["pipeline_done"] or "")
    out["faces_per_frame"] = float(mfs.group(1)) if mfs else None
    out["faces_per_s"] = float(mfs.group(2)) if mfs else None
    out["capture_lines"] = re.findall(r"^\[bench\] (?:auto-capture|  person|target|plus|  note).*$",
                                      text, re.M)
    out["bench_config"] = next(iter(re.findall(r"^\[bench\] swap_model=.*$", text, re.M)), None)
    out["person_verdicts"] = re.findall(
        r"^  box \d+ from the left .*$\n(?:      .*\n?)+", text, re.M)
    out["swap_audit"] = block_after(
        text, r"==== SWAP AUDIT[^\n]*\n", r"\n\n|\n\[|\n====")
    out["stage_timing_block"] = block_after(
        text, r"==== STAGE TIMING[^\n]*\n", r"\n=====")
    m = re.search(r"\[BaselineCounters\.json\] (\{.*\})", text)
    out["counters"] = json.loads(m.group(1)) if m else {}
    out["failed_frames"] = len(re.findall(r"processing failed|RAISED during processing", text))
    out["detector_failed"] = len(re.findall(r"DETECTOR FAILED", text))
    out["swapped_lines"] = re.findall(r"^\s+(?:swapped[^\n]*|faces seen[^\n]*)$", out["swap_audit"], re.M)
    mm = re.search(r"\[Stabilize\] ([\d.]+) GB RAM free of ([\d.]+) GB: chunk budget (\d+) MB", text)
    if mm:
        out["stab_ram_line"] = {"free_gb": float(mm.group(1)), "total_gb": float(mm.group(2)),
                                "derived_chunk_budget_mb": int(mm.group(3))}
    out["stabilizer_path"] = (
        "parallel-blocks" if (out["stab_geometry"] or re.search(r"\[Stabilize\] parallel:", text)) else
        "unified-scheduler" if "unified frame pipeline ON" in text else
        "2-pass" if "[Stabilize] 2-pass" in text else "sequential/none")
    return out


def counters_total(counters, key):
    return sum((counters.get(key) or {}).values())


def snapshot_machine():
    import psutil
    vm = psutil.virtual_memory()
    info = {"available_ram_mb": round(vm.available / 2 ** 20, 1),
            "total_ram_mb": round(vm.total / 2 ** 20, 1)}
    try:
        q = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits", "-i", "0"], text=True, timeout=10).strip()
        mem, util, sm, temp, pw = [x.strip() for x in q.split(",")]
        info.update({"gpu_mem_used_mib": float(mem), "gpu_util_pct": float(util),
                     "gpu_sm_mhz": float(sm), "gpu_temp_c": float(temp), "gpu_power_w": pw})
    except Exception:
        pass
    return info


def wait_for_quiet_gpu(limit_s=120):
    """A render that starts on a busy card is not the baseline. Wait, then record."""
    t0 = time.time()
    while time.time() - t0 < limit_s:
        s = snapshot_machine()
        # 30, not 10: this desktop idles at ~18% from the compositor with no
        # python process alive; the value is recorded either way.
        if s.get("gpu_util_pct", 0) < 30:
            return s
        time.sleep(5)
    return snapshot_machine()


def build_cmd(clip_name, spec, cfg, args, out_root, tag, threads, window=None, capture=None):
    start, end = window if window is not None else spec["window"]
    video = fixtures.clip(spec["rel"], required=True)
    cap = spec["capture"] if capture is None else capture
    cmd = [sys.executable, os.path.join(HERE, "two_face_video.py"),
           "--tag", tag, "--video", video, "--sources", spec["sources"],
           "--start", str(start), "--end", str(end),
           "--provider", str(cfg.provider),
           "--swap-model", str(cfg.swap_model),
           "--enhancer", str(cfg.selected_enhancer),
           "--mask-engine", str(cfg.mask_engine),
           "--codec", str(cfg.output_video_codec),
           "--tracking", "1" if bool(cfg.track_identities) else "0",
           "--stabilize-face", "1" if bool(cfg.stabilize_face) else "0",
           "--stabilize-mask", "1" if bool(cfg.stabilize_mask) else "0",
           "--stabilize-mask-strength", str(cfg.stabilize_mask_strength),
           "--stabilize-enhancer", "1" if bool(cfg.stabilize_enhancer) else "0",
           "--threads", str(threads),
           "--cuda-device-id", "0",
           "--swap-model-mask-strength", str(cfg.swap_model_mask_strength),
           "--merger-clarity", str(getattr(cfg, "merger_clarity", 0.0)),
           "--identity-detail-strength", str(getattr(cfg, "identity_detail_strength", 0.0)),
           "--temporal-compositing-strength", str(getattr(cfg, "temporal_compositing_strength", 0.65)),
           "--capture-budget", str(CAPTURE_BUDGET_S),
           "--out", out_root]
    if cap is not None:
        cmd += ["--capture", str(cap)]
        if spec["capture_face"] is not None:
            cmd += ["--capture-face", str(spec["capture_face"])]
    return cmd, video


def run_one(label, clip_name, spec, cfg, args, env, out_root, threads, window=None, capture=None):
    tag = label
    cmd, video = build_cmd(clip_name, spec, cfg, args, out_root, tag, threads, window, capture)
    before = wait_for_quiet_gpu()
    fp = fixtures.fingerprint(video)
    print("\n=== %s: %s frames %s..%s, sources %s, threads %d ==="
          % (label, os.path.basename(video), (window or spec["window"])[0],
             (window or spec["window"])[1] or "end", spec["sources"], threads), flush=True)
    print("    machine before: %s" % before, flush=True)
    started = time.perf_counter()
    last = [started]
    latest = [""]

    def on_line(line):
        if line.startswith("[Pipeline] frames"):
            latest[0] = line
        now = time.perf_counter()
        # AGENTS.md: surface processing fps about every three minutes during a run.
        if now - last[0] >= 180:
            last[0] = now
            print("  [%5.1f min] %s" % ((now - started) / 60, (latest[0] or line)[:170]), flush=True)
        elif re.search(r"Traceback|Error|processing failed|RAISED|DETECTOR FAILED", line):
            print("  ! %s" % line.strip()[:170], flush=True)

    rc, text, telem = tel.run_sampled(cmd, env=env, cwd=APP, on_line=on_line, device_id=0)
    elapsed = time.perf_counter() - started
    log_path = os.path.join(out_root, label + ".log")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(text)

    rec = {"label": label, "clip": clip_name, "returncode": rc, "wall_seconds": round(elapsed, 1),
           "video": video, "fixture": fp,
           "window": list(window if window is not None else spec["window"]),
           "sources": spec["sources"], "threads": threads,
           "machine_before": before, "cmd": [os.path.basename(c) if i == 1 else c
                                             for i, c in enumerate(cmd)],
           "log": log_path}
    rec.update(parse_log(text))
    rec["stages"] = bc.parse_stage_timing(text)
    try:
        rec["run"] = bc.parse_run(text)
    except SystemExit as exc:
        rec["run"] = {"parse_refused": str(exc)}
    rec["adaptive_downgrades"] = bc.parse_adaptive_downgrades(text)
    rec["telemetry"] = {k: v for k, v in telem.items() if k != "cpu_topology"}

    outdir = os.path.join(out_root, tag)
    work = os.path.join(outdir, "work")
    out_file = find_output(work) if os.path.isdir(work) else None
    rows = os.path.join(outdir, "rows.csv")
    rec["output"] = {"file": out_file,
                     "sha256": sha256_file(out_file) if out_file else None,
                     "decoded_video_md5": decoded_md5(out_file, env) if out_file else None,
                     "bytes": os.path.getsize(out_file) if out_file else None,
                     "rows_csv_sha256": sha256_file(rows) if os.path.exists(rows) else None}
    return rec


def fmt_counters(counters, keys):
    rows = []
    for k in keys:
        if k in counters:
            row = counters[k]
            rows.append("| `%s` | %d | %d | %d | %d | %d |" % (
                k, row.get("setup", 0), row.get("prepass", 0), row.get("main", 0),
                row.get("main+warmup", 0), sum(row.values())))
    return rows


def md_for(rec):
    run = rec.get("run", {})
    L = []
    L.append("### %s  (`%s`)" % (rec["label"], os.path.basename(rec["video"])))
    fx = rec.get("fixture") or {}
    L.append("")
    L.append("- clip %sx%s, %s frames in file; rendered window **%s..%s**; sources `%s`; threads %s; "
             "returncode **%s**; wall %.0f s"
             % (fx.get("width"), fx.get("height"), fx.get("frames"), rec["window"][0],
                rec["window"][1] or "end", rec["sources"], rec["threads"], rec["returncode"],
                rec["wall_seconds"]))
    L.append("- **frame-loop fps %s, %s faces/frame, %s faces/s** (`[Pipeline] done`, excludes setup and the "
             "pre-pass); end-to-end %s fps (`took N secs`, includes model init + pre-pass)"
             % (rec.get("frame_loop_fps"), rec.get("faces_per_frame"), rec.get("faces_per_s"), run.get("fps")))
    L.append("- `%s`" % rec.get("pipeline_done"))
    if run.get("faces_seen") is not None:
        L.append("- faces_seen %s, faces_swapped (identity lock) %s"
                 % (run.get("faces_seen"), run.get("faces_swapped")))
    L.append("- failed frames in log: **%d**; swallowed detector failures: **%d**; adaptive downgrades: %s"
             % (rec["failed_frames"], rec["detector_failed"], rec["adaptive_downgrades"] or "none"))
    mb = rec["machine_before"]
    L.append("- before the run: %.1f GB RAM available of %.1f GB; GPU %s MiB used, %s%% util, "
             "%s MHz SM, %s C"
             % (mb["available_ram_mb"] / 1024, mb["total_ram_mb"] / 1024, mb.get("gpu_mem_used_mib"),
                mb.get("gpu_util_pct"), mb.get("gpu_sm_mhz"), mb.get("gpu_temp_c")))
    t = rec.get("telemetry", {})
    L.append("- telemetry: peak RSS %s GB, peak GPU mem %s MB, GPU util mean/peak %s/%s %%, "
             "CPU mean/peak %s/%s %%"
             % (t.get("peak_rss_gb"), t.get("peak_gpu_memory_mb"), t.get("mean_gpu_util_pct"),
                t.get("peak_gpu_util_pct"), t.get("mean_cpu_pct"), t.get("peak_cpu_pct")))
    L.append("- stabilizer path: **%s**; %s" % (rec["stabilizer_path"], rec.get("stab_ram_line") or ""))
    for g in rec["stab_geometry"]:
        L.append("  - `%s`" % g)
    L.append("")
    L.append("**Output hashes** (the reference a later stage must match; the pixel noise floor "
             "is 0.7142/255 mean between two renders of one config, so a hash match is lucky "
             "and a mismatch is not by itself a regression)")
    o = rec["output"]
    L.append("")
    L.append("| what | value |")
    L.append("|---|---|")
    L.append("| output file sha256 | `%s` |" % o["sha256"])
    L.append("| decoded video md5 | `%s` |" % o["decoded_video_md5"])
    L.append("| rows.csv sha256 | `%s` |" % o["rows_csv_sha256"])
    L.append("| output bytes | %s |" % o["bytes"])
    L.append("")
    if rec["stages"]:
        L.append("**STAGE TIMING** (thread time summed across workers; NOT a speedup budget)")
        L.append("")
        L.append("| stage | total s | share % | calls | ms/call |")
        L.append("|---|---:|---:|---:|---:|")
        for name, st in sorted(rec["stages"].items(), key=lambda kv: -kv[1]["total_s"]):
            L.append("| %s | %.2f | %.1f | %d | %.2f |"
                     % (name, st["total_s"], st["share_pct"], st["calls"], st["ms_per_call"]))
        L.append("")
    c = rec["counters"]
    if c:
        L.append("**Counters** (setup / prepass / main / main+warmup = discarded stabilizer warm-up)")
        L.append("")
        L.append("| counter | setup | prepass | main | warmup | total |")
        L.append("|---|---:|---:|---:|---:|---:|")
        L.extend(fmt_counters(c, sorted(c)))
        wu = counters_total(c, "stab.warmup_frames")
        outf = counters_total(c, "stab.output_frames")
        if wu or outf:
            L.append("")
            L.append("Measured stabilizer warm-up: **%d discarded vs %d output frames = %.1f%% of processed "
                     "frames** (%.3f per useful frame)." % (wu, outf, 100.0 * wu / max(1, wu + outf),
                                                           wu / max(1, outf)))
        L.append("")
    if rec["swap_audit"]:
        L.append("**SWAP AUDIT** (counts INTENT over the faces it was handed, not outcome)")
        L.append("")
        L.append("```")
        L.append(rec["swap_audit"].rstrip())
        L.append("```")
        L.append("")
    if rec["person_verdicts"]:
        L.append("**Per-person verdicts** (the harness's own decision-log grading)")
        L.append("")
        L.append("```")
        L.extend(v.rstrip() for v in rec["person_verdicts"])
        L.append("```")
        L.append("")
    L.append("<details><summary>Sessions, VRAM samples, threads, runtime lines</summary>")
    L.append("")
    L.append("```")
    for key in ("capture_lines",):
        L.extend(rec[key])
    L.append("")
    L.extend(rec["sessions"])
    L.append("")
    L.extend(rec["vram"])
    L.append("")
    L.extend(rec["thread_lines"])
    L.append("")
    L.extend(rec["stab_lines"])
    L.append("")
    L.extend(rec["runtime_lines"])
    L.append("")
    L.extend(rec["pipeline_lines"][-6:])
    L.append("```")
    L.append("</details>")
    L.append("")
    return "\n".join(L)


def null_control_md(records):
    a = next((r for r in records if r["label"] == "d4"), None)
    b = next((r for r in records if r["label"] == "d4_null_repeat"), None)
    if not (a and b):
        return ""

    def fps(r):
        return r.get("frame_loop_fps")

    ca, cb = a["counters"], b["counters"]
    diff = [k for k in sorted(set(ca) | set(cb))
            if counters_total(ca, k) != counters_total(cb, k)]
    fa, fb = fps(a), fps(b)
    spread = (100.0 * abs(fa - fb) / ((fa + fb) / 2)) if fa and fb else None
    pins = [r for r in records if r["label"].startswith("d4_pinned_")]
    L = []
    if len(pins) >= 2:
        x, y = pins[0], pins[1]
        sp = 100.0 * abs(x["frame_loop_fps"] - y["frame_loop_fps"]) / ((x["frame_loop_fps"] + y["frame_loop_fps"]) / 2)
        L += ["## Null control, pinned: d4 twice with ROOP_STAB_CHUNK_MB=%s" % _geo(x).get("effective_chunk_mb"), "",
              "Frame-loop fps **%s vs %s (spread %.1f %%)**; decoded md5 identical to each other AND to run A: **%s**; "
              "rows.csv identical to run A: **%s**; counters identical: **%s**."
              % (x["frame_loop_fps"], y["frame_loop_fps"], sp,
                 x["output"]["decoded_video_md5"] == y["output"]["decoded_video_md5"] == a["output"]["decoded_video_md5"],
                 x["output"]["rows_csv_sha256"] == y["output"]["rows_csv_sha256"] == a["output"]["rows_csv_sha256"],
                 all(counters_total(x["counters"], k) == counters_total(a["counters"], k) for k in a["counters"])), ""]
    L += ["## Null control, unpinned: d4 rendered twice, free RAM left to decide the geometry", "",
         "| | run A (`d4`) | run B (`d4_null_repeat`) | same? |", "|---|---|---|---|",
         "| frame-loop fps | %s | %s | spread %s |"
         % (fa, fb, "n/a" if spread is None else "%.1f %%" % spread),
         "| end-to-end fps | %s | %s | |" % ((a.get("run") or {}).get("fps"), (b.get("run") or {}).get("fps")),
         "| available RAM before the run (GB) | %.1f | %.1f | |"
         % (a["machine_before"]["available_ram_mb"] / 1024, b["machine_before"]["available_ram_mb"] / 1024),
         "| decoded video md5 | `%s` | `%s` | %s |" % (a["output"]["decoded_video_md5"],
                                                      b["output"]["decoded_video_md5"],
                                                      a["output"]["decoded_video_md5"] == b["output"]["decoded_video_md5"]),
         "| output file sha256 | `%s` | `%s` | %s |" % (a["output"]["sha256"], b["output"]["sha256"],
                                                       a["output"]["sha256"] == b["output"]["sha256"]),
         "| rows.csv sha256 | `%s` | `%s` | %s |" % (a["output"]["rows_csv_sha256"],
                                                    b["output"]["rows_csv_sha256"],
                                                    a["output"]["rows_csv_sha256"] == b["output"]["rows_csv_sha256"]),
         "| faces_seen | %s | %s | %s |" % ((a.get("run") or {}).get("faces_seen"),
                                          (b.get("run") or {}).get("faces_seen"),
                                          (a.get("run") or {}).get("faces_seen") == (b.get("run") or {}).get("faces_seen")),
         "| stabilizer geometry | `%s` | `%s` | %s |" % (
             (a["stab_geometry"] or ["-"])[0].split(" psutil")[0][:90],
             (b["stab_geometry"] or ["-"])[0].split(" psutil")[0][:90],
             [x.split(" psutil")[0] for x in a["stab_geometry"]] == [x.split(" psutil")[0] for x in b["stab_geometry"]]),
         "", "Counters whose totals differ between A and B: %s" % (
             ", ".join("`%s` (%d vs %d)" % (k, counters_total(ca, k), counters_total(cb, k)) for k in diff)
             or "**none**"), ""]
    return "\n".join(L)


def _geo(rec):
    g = (rec.get("stab_geometry") or [""])[0]
    return dict(x.split("=", 1) for x in g.replace("[StabGeometry] ", "").split(" ") if "=" in x)


def summary_md(recs):
    L = ["## Summary", "",
         "Frame-loop fps is `[Pipeline] done` (excludes model init and the pre-pass). `discard %` is "
         "MEASURED stabilizer warm-up frames over all frames the stabilized loop processed. Rows "
         "marked *pinned* ran with `ROOP_STAB_CHUNK_MB` exported at the value the first run derived.", "",
         "| run | clip | window | stab workers x block (blocks/chunk) | chunk MB (source) | loop fps | faces/frame | faces/s | discard % | swapped / faces seen | decoded md5 |",
         "|---|---|---|---|---|---:|---:|---:|---:|---|---|"]
    for r in recs:
        kv = _geo(r)
        c = r["counters"]
        wu, outf = counters_total(c, "stab.warmup_frames"), counters_total(c, "stab.output_frames")
        disc = "%.1f" % (100.0 * wu / (wu + outf)) if (wu + outf) else "-"
        run = r.get("run") or {}
        L.append("| %s | %s | %s..%s | %s x %s (%s) | %s (%s) | **%s** | %s | %s | %s | %s / %s | `%s` |" % (
            r["label"], r["clip"], r["window"][0], r["window"][1] or "end",
            kv.get("workers"), kv.get("block"), kv.get("blocks_per_chunk"),
            kv.get("effective_chunk_mb"), "pinned" if kv.get("chunk_mb_source") == "ROOP_STAB_CHUNK_MB" else "derived",
            r.get("frame_loop_fps"), r.get("faces_per_frame"), r.get("faces_per_s"), disc,
            run.get("faces_swapped"), run.get("faces_seen"),
            (r["output"]["decoded_video_md5"] or "")[:12]))
    L.append("")
    by = {}
    for r in recs:
        by.setdefault(r["clip"], []).append(r)
    L += ["### Repeat pairs (same clip, same window, same config)", "",
          "| clip | first run | repeat | same stabilizer geometry | same decoded md5 | same rows.csv | loop fps first -> repeat |",
          "|---|---|---|---|---|---|---|"]
    for clip, rs in by.items():
        for other in rs[1:]:
            first = rs[0]
            L.append("| %s | %s | %s | %s | %s | %s | %s -> %s |" % (
                clip, first["label"], other["label"],
                {k: v for k, v in _geo(first).items() if k in ("workers", "block", "blocks_per_chunk")}
                == {k: v for k, v in _geo(other).items() if k in ("workers", "block", "blocks_per_chunk")},
                first["output"]["decoded_video_md5"] == other["output"]["decoded_video_md5"],
                first["output"]["rows_csv_sha256"] == other["output"]["rows_csv_sha256"],
                first.get("frame_loop_fps"), other.get("frame_loop_fps")))
    L.append("")
    return "\n".join(L)


def write_markdown(json_path, findings_path=None, merge=()):
    with open(json_path, encoding="utf-8") as fh:
        res = json.load(fh)
    recs = res["records"]
    for extra in merge:
        with open(extra, encoding="utf-8") as fh:
            more = json.load(fh)["records"]
        for r in more:
            if r["label"] == r["clip"]:
                r["label"] = r["clip"] + "_pinned"
        recs = recs + more
    for r in recs:
        # JSON written before the key collision fix kept the [Threads] lines under
        # "threads" (overwriting the integer); recover both and the frame-loop fps.
        if isinstance(r.get("threads"), list):
            r["thread_lines"] = r["threads"]
            r["threads"] = res["threads"]
        if "frame_loop_fps" not in r:
            mf = re.search(r"= ([\d.]+) fps", r.get("pipeline_done") or "")
            r["frame_loop_fps"] = float(mf.group(1)) if mf else None
            mm = re.search(r"([\d.]+) faces/frame, ([\d.]+) faces/s", r.get("pipeline_done") or "")
            r["faces_per_frame"] = float(mm.group(1)) if mm else None
            r["faces_per_s"] = float(mm.group(2)) if mm else None
        if r.get("stab_geometry"):
            r["stabilizer_path"] = "parallel-blocks"
    L = ["# Performance baseline %s" % res["date"], "",
         "Reference every later stage must match. Measurement only: produced by "
         "`app/tests/baseline_snapshot.py` at `%s` (uncommitted python files at run time: %s)."
         % (res["head"]["short"], res["head"]["dirty_files"] or "none"), ""]
    if findings_path and os.path.exists(findings_path):
        with open(findings_path, encoding="utf-8") as fh:
            L += [fh.read().rstrip(), ""]
    L += ["## Stack", "", "```"]
    L += ["%-12s %s" % kv for kv in res["software"].items()]
    L += ["```", "", "## config.yaml values that shaped the run (read live)", "", "```"]
    L += ["%-26s %s" % kv for kv in res["config"].items()]
    L += ["```", ""]
    L += [summary_md([r for r in recs if r["label"] != "d4_null_repeat" or True])]
    nc = null_control_md(recs)
    if nc:
        L += [nc]
    L += ["## Per-clip results", ""]
    for r in recs:
        L.append(md_for(r))
    out = os.path.join(os.path.dirname(json_path), "baseline_%s.md" % res["date"])
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return out


def head_info():
    def git(*a):
        try:
            return subprocess.run(("git",) + a, cwd=REPO, capture_output=True, text=True, timeout=15).stdout.strip()
        except Exception:
            return None
    return {"head": git("rev-parse", "HEAD"), "short": git("rev-parse", "--short", "HEAD"),
            "dirty_files": [l[3:] for l in (git("status", "--porcelain") or "").splitlines()
                            if l[3:].endswith(".py")]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--only", default="", help="comma-separated subset of: " + ",".join(ORDER))
    ap.add_argument("--smoke", type=int, default=0,
                    help="render only the first N frames of d4 (instrumentation check, nothing is written to docs/)")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--no-null", action="store_true")
    ap.add_argument("--threads", type=int, default=None, help="default: config.yaml max_threads")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pin-chunk-mb", default=None, metavar="MB",
                    help="export ROOP_STAB_CHUNK_MB for every render (AGENTS.md: pin it for any "
                         "pixel comparison; free RAM otherwise decides the stabilizer geometry)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="render each selected clip N times (labels <clip>_pinned_<i>)")
    ap.add_argument("--render-md", default=None, metavar="JSON",
                    help="only (re)write the markdown from an existing baseline JSON")
    ap.add_argument("--merge", nargs="*", default=[], metavar="JSON",
                    help="extra baseline JSONs whose records are appended (the pinned repeats)")
    ap.add_argument("--findings", default=None, metavar="FILE",
                    help="hand-written findings inserted under the title by --render-md")
    args = ap.parse_args()
    if args.render_md:
        print(write_markdown(args.render_md, args.findings, args.merge))
        return 0

    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    threads = args.threads if args.threads is not None else int(cfg.max_threads)
    out_root = args.out or os.path.join(APP, "output", "baseline_" + args.date)
    os.makedirs(out_root, exist_ok=True)

    env = dict(os.environ)
    env["ROOP_PROFILE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if args.pin_chunk_mb:
        env["ROOP_STAB_CHUNK_MB"] = str(args.pin_chunk_mb)
    env.setdefault("ROOP_PIPELINE_LOG_EVERY", "100")
    env = bc.ensure_ffmpeg(env)
    os.environ["PATH"] = env["PATH"]

    stack = bc.software_stack(0)
    rev = head_info()
    print("baseline snapshot %s | HEAD %s dirty=%s" % (args.date, rev["short"], rev["dirty_files"]), flush=True)
    for k, v in stack.items():
        print("  %-12s %s" % (k, v), flush=True)
    print("  config: swap=%s enhancer=%s mask=%s provider=%s codec=%s threads=%d trt_precision=%s"
          % (cfg.swap_model, cfg.selected_enhancer, cfg.mask_engine, cfg.provider,
             cfg.output_video_codec, threads, cfg.trt_precision), flush=True)

    if args.smoke:
        rec = run_one("smoke_d4", "d4", CLIPS["d4"], cfg, args, env, out_root, threads,
                      window=(0, args.smoke))
        print(json.dumps({k: rec[k] for k in ("returncode", "failed_frames", "stabilizer_path")}, indent=1))
        print("\n".join(rec["sessions"]))
        print("\n".join(rec["vram"][:6]))
        print("\n".join(rec["threads"][:8]))
        print("\n".join(rec["stab_geometry"]))
        print(json.dumps({k: sum(v.values()) for k, v in rec["counters"].items()}, indent=1, sort_keys=True))
        return 0 if rec["returncode"] == 0 else 1

    wanted = [c for c in ORDER if not args.only or c in args.only.split(",")]
    records = []
    if not args.no_warmup:
        # Discarded: pays any cold TensorRT engine build so the first measured arm doesn't.
        w = run_one("warmup_d4", "d4", CLIPS["d4"], cfg, args, env, out_root, threads, window=(0, 120))
        print("  warm-up done (rc %s, fps %s) -- discarded" % (w["returncode"], w.get("run", {}).get("fps")),
              flush=True)
    for name in wanted:
        for i in range(max(1, args.repeat)):
            label = name if args.repeat <= 1 else "%s_pinned_%d" % (name, i + 1)
            records.append(run_one(label, name, CLIPS[name], cfg, args, env, out_root, threads))
    if not args.no_null and "d4" in wanted:
        records.append(run_one("d4_null_repeat", "d4", CLIPS["d4"], cfg, args, env, out_root, threads))

    result = {"date": args.date, "head": rev, "software": stack, "threads": threads,
              "config": {k: getattr(cfg, k, None) for k in (
                  "swap_model", "selected_enhancer", "mask_engine", "mask_engine_2", "provider",
                  "trt_precision", "output_video_codec", "video_quality", "max_threads",
                  "detector_engine", "face_detector_size", "face_detector_threshold",
                  "detector_scale_pyramid", "rescue_small_faces", "temporal_detection",
                  "track_identities", "stabilize_face", "stabilize_mask", "stabilize_enhancer",
                  "stabilize_landmarks", "stabilize_method", "recognizer", "refine_landmarks",
                  "autorotate_faces", "perf_trt_pool", "perf_detmask_pool", "perf_detector_pool",
                  "perf_batch_swap", "perf_nvdec", "restore_ultra_profile")},
              "records": records}
    docs = os.path.join(REPO, "docs", "perf")
    os.makedirs(docs, exist_ok=True)
    js = os.path.join(docs, "baseline_%s.json" % args.date)
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=str)
    print("\nwrote %s" % js, flush=True)
    print("wrote %s" % write_markdown(js), flush=True)
    return 0 if all(r["returncode"] == 0 for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
