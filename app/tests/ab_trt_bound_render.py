"""ABBA end-to-end render: shipped call path vs `ROOP_TRT_BOUND` (opt-in bound / CUDA-graph XSeg sessions).

    env/Scripts/python.exe tests/ab_trt_bound_render.py --clips d4,s7 --b "xseg=graph" [--date 2026-10-09]

Drives `baseline_snapshot.run_one` (the live app/config.yaml, ROOP_PROFILE, swap audit, fps beside faces/s,
output hashes) so the arms are exactly what a baseline render is. Per clip: one discarded warm-up (pays the
TensorRT engine build and fixes the stabilizer chunk budget), then A B B A, with ROOP_STAB_CHUNK_MB PINNED to
the warm-up's derived budget so every arm takes one stabilizer path (a render's pixels and speed depend on
free RAM). Reported: frame-loop fps, faces/s, faces_seen, swap rate, stabilizer path, output hashes, and the
cross-arm pixel check (A vs A is the noise floor, A vs B the effect). B is only meaningful if the log proves
the bound session ran: the `[Session] mask:xseg[...]` line is counted per arm.
"""
import argparse
import datetime
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import baseline_snapshot as bs                                   # noqa: E402
import baseline_controlled as bc                                 # noqa: E402

# s7 is not in baseline_snapshot.CLIPS (d4/d1/d6/Love only): a single-person clip, first 600 frames, the
# app's own auto-capture picks the seed frame (printed in the arm log and recorded).
EXTRA = {"s7": {"rel": "single/s7.mp4", "sources": "harjot", "window": (0, 600),
                "capture": None, "capture_face": None}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="d4,s7")
    ap.add_argument("--b", default="xseg=graph", help="value of ROOP_TRT_BOUND in the B arm")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--window", nargs=2, type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.detector_engine = None
    args.governor = False

    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    threads = args.threads if args.threads is not None else int(cfg.max_threads)
    out_root = args.out or os.path.join(APP, "output", "ab_trt_bound_" + args.date)
    os.makedirs(out_root, exist_ok=True)
    base_env = dict(os.environ)
    base_env.pop("ROOP_TRT_BOUND", None)
    base_env["ROOP_PROFILE"] = "1"
    base_env["PYTHONIOENCODING"] = "utf-8"
    base_env.setdefault("ROOP_PIPELINE_LOG_EVERY", "100")
    base_env = bc.ensure_ffmpeg(base_env)
    os.environ["PATH"] = base_env["PATH"]

    specs = dict(bs.CLIPS)
    specs.update(EXTRA)
    summary = {"date": args.date, "head": bs.head_info(), "b_value": args.b, "threads": threads,
               "config": {k: str(getattr(cfg, k, None)) for k in (
                   "swap_model", "selected_enhancer", "mask_engine", "provider", "trt_precision",
                   "max_threads", "detector_engine", "face_detector_size", "perf_trt_pool",
                   "perf_detmask_pool", "perf_detector_pool")},
               "clips": {}}
    for clip in args.clips.split(","):
        spec = specs[clip]
        window = tuple(args.window) if args.window else None
        w = bs.run_one("warmup_" + clip, clip, spec, cfg, args, dict(base_env), out_root, threads,
                       window=(0, 120))
        chunk = (w.get("stab_ram_line") or {}).get("derived_chunk_budget_mb")
        print("  warm-up %s done (rc %s); derived ROOP_STAB_CHUNK_MB = %s -> pinned" % (clip, w["returncode"], chunk),
              flush=True)
        recs = []
        for i, arm in enumerate("ABBA"):
            env = dict(base_env)
            if chunk:
                env["ROOP_STAB_CHUNK_MB"] = str(chunk)
            if arm == "B":
                env["ROOP_TRT_BOUND"] = args.b
            rec = bs.run_one("%s_%d%s" % (clip, i + 1, arm), clip, spec, cfg, args, env, out_root, threads,
                             window=window)
            text = open(rec["log"], encoding="utf-8", errors="ignore").read()
            rec["arm"] = arm
            rec["bound_session_lines"] = len(re.findall(r"\[Session\] mask:xseg\[(?:bound|graph)\]", text))
            rec["bound_fallback_lines"] = len(re.findall(r"\[XSeg\] ROOP_TRT_BOUND", text))
            print("  %s: frame-loop %s fps | %s faces/s | faces_seen %s | swapped %s | stab %s | bound-session lines %d"
                  % (rec["label"], rec.get("frame_loop_fps"), rec.get("faces_per_s"),
                     (rec.get("run") or {}).get("faces_seen"), (rec.get("swapped_lines") or ["?"])[0].strip()[:60],
                     rec.get("stabilizer_path"), rec["bound_session_lines"]), flush=True)
            recs.append(rec)
        a = [r["frame_loop_fps"] for r in recs if r["arm"] == "A" and r["frame_loop_fps"]]
        b = [r["frame_loop_fps"] for r in recs if r["arm"] == "B" and r["frame_loop_fps"]]
        row = {"A_fps": a, "B_fps": b, "chunk_mb": chunk,
               "A_mean": round(st.mean(a), 3) if a else None, "B_mean": round(st.mean(b), 3) if b else None,
               "ratio_B_over_A": round(st.mean(b) / st.mean(a), 4) if a and b else None,
               "A_vs_A_spread_pct": round(100 * abs(a[0] - a[1]) / st.mean(a), 2) if len(a) == 2 else None,
               "arms": [{k: r.get(k) for k in ("label", "arm", "returncode", "frame_loop_fps", "faces_per_s",
                                               "faces_per_frame", "stabilizer_path", "bound_session_lines",
                                               "bound_fallback_lines", "failed_frames", "swapped_lines",
                                               "wall_seconds", "stages", "output")} for r in recs]}
        summary["clips"][clip] = row
        print("\n== %s: A %s | B %s | B/A %s | A-vs-A spread %s%%\n" % (
            clip, row["A_fps"], row["B_fps"], row["ratio_B_over_A"], row["A_vs_A_spread_pct"]), flush=True)
    path = os.path.join(REPO, "docs", "perf", "trt_bound_render_%s.json" % args.date)
    json.dump(summary, open(path, "w"), indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
