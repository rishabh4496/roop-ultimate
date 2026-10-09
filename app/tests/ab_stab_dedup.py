"""ABBA: parallel-stabilization warm-up dedup (ROOP_STAB_DEDUP) off vs on, end to end, on the live config.

    env/Scripts/python.exe tests/ab_stab_dedup.py --clips d4,s7 [--window 0 600] [--date 2026-10-09]

A = ROOP_STAB_DEDUP=0 (today's code path: every block re-runs its warm-up frames in full).
B = default (the cache on).
Both arms: ROOP_STAB_CHUNK_MB PINNED to the budget a discarded warm-up render derived, so they take one stabilizer
geometry (a render's pixels and speed depend on free RAM). Driven through `baseline_snapshot.run_one`, i.e. the live
app/config.yaml, ROOP_PROFILE, swap audit, fps beside faces/s, telemetry.

ACCEPTANCE (docs/perf/stab_warmup.md): B's output must be BIT-IDENTICAL to A's (file sha256 and decoded-video md5);
anything else is a defect, not a trade. Reported beside it: how much warm-up inference was skipped (swap-net calls
A vs B against the warm-up face visits), peak RSS, frame-loop fps and faces/s, and the cache's own counters
(`[StabDedup]` line, `stab.dedup.*`). B is only meaningful if its log proves the cache ran: ON + served > 0.
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

EXTRA = {"s7": {"rel": "single/s7.mp4", "sources": "harjot", "window": (0, 600),
                "capture": None, "capture_face": None}}


def swap_calls(rec):
    return int(((rec.get("stages") or {}).get("swap") or {}).get("calls") or 0)


def counter(rec, key, phase=None):
    row = (rec.get("counters") or {}).get(key) or {}
    return int(sum(row.values()) if phase is None else row.get(phase, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="d4,s7")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--window", nargs=2, type=int, default=None)
    ap.add_argument("--env", action="append", default=[], metavar="K=V",
                    help="extra env for BOTH arms (e.g. ROOP_STAB_BLOCK_MULT=8)")
    ap.add_argument("--chunk-mb", default=None, help="pin ROOP_STAB_CHUNK_MB instead of deriving it")
    ap.add_argument("--a-env", action="append", default=[], metavar="K=V",
                    help="env for arm A only (default: ROOP_STAB_DEDUP=0). Set both --a-env and --b-env to A/B "
                         "something else with the cache ON in both, e.g. ROOP_STAB_BLOCK_MULT=4 vs =8")
    ap.add_argument("--b-env", action="append", default=[], metavar="K=V", help="env for arm B only (default: none)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    args.detector_engine = None
    args.governor = False

    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    threads = args.threads if args.threads is not None else int(cfg.max_threads)
    out_root = args.out or os.path.join(APP, "output", "ab_stab_dedup_" + args.date + args.tag)
    os.makedirs(out_root, exist_ok=True)
    base_env = dict(os.environ)
    for k in ("ROOP_STAB_DEDUP", "ROOP_TRT_BOUND"):
        base_env.pop(k, None)
    base_env["ROOP_PROFILE"] = "1"
    base_env["PYTHONIOENCODING"] = "utf-8"
    base_env.setdefault("ROOP_PIPELINE_LOG_EVERY", "100")
    for kv in args.env:
        k, _, v = kv.partition("=")
        base_env[k] = v
    base_env = bc.ensure_ffmpeg(base_env)
    os.environ["PATH"] = base_env["PATH"]

    specs = dict(bs.CLIPS)
    specs.update(EXTRA)
    summary = {"date": args.date, "head": bs.head_info(), "threads": threads, "extra_env": args.env,
               "a_env": args.a_env or ["ROOP_STAB_DEDUP=0"], "b_env": args.b_env,
               "config": {k: str(getattr(cfg, k, None)) for k in (
                   "swap_model", "selected_enhancer", "mask_engine", "provider", "trt_precision", "max_threads",
                   "detector_engine", "face_detector_size", "temporal_detection", "stabilize_face",
                   "stabilize_mask", "stabilize_enhancer", "stabilize_hf_texture", "perf_trt_pool",
                   "perf_detmask_pool")},
               "clips": {}}
    for clip in args.clips.split(","):
        spec = specs[clip]
        window = tuple(args.window) if args.window else None
        chunk = args.chunk_mb
        if not chunk:
            w = bs.run_one("warmup_" + clip, clip, spec, cfg, args, dict(base_env), out_root, threads, window=(0, 120))
            chunk = (w.get("stab_ram_line") or {}).get("derived_chunk_budget_mb")
            print("  warm-up %s done (rc %s); derived ROOP_STAB_CHUNK_MB = %s -> pinned" % (clip, w["returncode"], chunk),
                  flush=True)
        recs = []
        for i, arm in enumerate("ABBA"):
            env = dict(base_env)
            if chunk:
                env["ROOP_STAB_CHUNK_MB"] = str(chunk)
            if arm == "A":
                env.update(dict(kv.partition("=")[::2] for kv in args.a_env) if args.a_env else {"ROOP_STAB_DEDUP": "0"})
            else:
                env.update(dict(kv.partition("=")[::2] for kv in args.b_env))
            rec = bs.run_one("%s_%d%s" % (clip, i + 1, arm), clip, spec, cfg, args, env, out_root, threads, window=window)
            text = open(rec["log"], encoding="utf-8", errors="ignore").read()
            rec["arm"] = arm
            m = re.search(r"^\[StabDedup\] (ON|off)(.*)$", text, re.M)
            rec["dedup_state"] = (m.group(1) + m.group(2)) if m else None
            rec["dedup_summary"] = next(iter(re.findall(r"^\[StabDedup\] produced.*$", text, re.M)), None)
            rec["swap_calls"] = swap_calls(rec)
            tel = rec.get("telemetry") or {}
            print("  %s: frame-loop %s fps | %s faces/s | swap calls %d | peak RSS %s GB | dedup %s | sha %s"
                  % (rec["label"], rec.get("frame_loop_fps"), rec.get("faces_per_s"), rec["swap_calls"],
                     tel.get("peak_rss_gb"), rec["dedup_state"], ((rec.get("output") or {}).get("sha256") or "")[:12]),
                  flush=True)
            recs.append(rec)
        a = [r for r in recs if r["arm"] == "A"]
        b = [r for r in recs if r["arm"] == "B"]
        a_fps = [r["frame_loop_fps"] for r in a if r["frame_loop_fps"]]
        b_fps = [r["frame_loop_fps"] for r in b if r["frame_loop_fps"]]
        shas = {r["label"]: (r.get("output") or {}).get("sha256") for r in recs}
        md5s = {r["label"]: (r.get("output") or {}).get("decoded_video_md5") for r in recs}
        a_calls = st.mean(r["swap_calls"] for r in a) if a else 0
        b_calls = st.mean(r["swap_calls"] for r in b) if b else 0
        wu_frames = counter(a[0], "stab.warmup_frames") if a else 0
        # The cache's own phase-split counters. Overlap frames are visited twice (a block's tail, then the next
        # block's warm-up). Without the cache both visits run swap + restore; with it the first visitor produces and
        # the second is served. Which visitor is which is a race: in practice the warm-up (block start) arrives first.
        produced_wu = counter(b[0], "stab.dedup.produced", "main+warmup") if b else 0
        produced_main = counter(b[0], "stab.dedup.produced", "main") if b else 0
        served_wu = counter(b[0], "stab.dedup.served", "main+warmup") if b else 0
        served_main = counter(b[0], "stab.dedup.served", "main") if b else 0
        bypass = counter(b[0], "stab.dedup.bypass") if b else 0
        served = served_wu + served_main
        produced = produced_wu + produced_main
        row = {"chunk_mb": chunk, "A_fps": a_fps, "B_fps": b_fps,
               "A_mean": round(st.mean(a_fps), 3) if a_fps else None, "B_mean": round(st.mean(b_fps), 3) if b_fps else None,
               "ratio_B_over_A": round(st.mean(b_fps) / st.mean(a_fps), 4) if a_fps and b_fps else None,
               "A_vs_A_spread_pct": round(100 * abs(a_fps[0] - a_fps[1]) / st.mean(a_fps), 2) if len(a_fps) == 2 else None,
               "B_vs_B_spread_pct": round(100 * abs(b_fps[0] - b_fps[1]) / st.mean(b_fps), 2) if len(b_fps) == 2 else None,
               "all_output_sha256_equal": len({v for v in shas.values()}) == 1,
               "all_decoded_md5_equal": len({v for v in md5s.values()}) == 1,
               "sha256": shas, "decoded_md5": md5s,
               "swap_calls_A": a_calls, "swap_calls_B": b_calls,
               "swap_calls_skipped": a_calls - b_calls,
               "warmup_frames": wu_frames,
               "overlap_visits_computed": {"warm-up visit": produced_wu, "output visit": produced_main},
               "overlap_visits_served": {"warm-up visit": served_wu, "output visit": served_main},
               "overlap_visits_bypassed": bypass,
               "every_overlap_frame_computed_once": bool(produced and produced == served and not bypass
                                                          and (a_calls - b_calls) == served),
               "duplicate_inference_removed_pct_of_all": round(100.0 * served / a_calls, 1) if a_calls else None,
               "duplicate_inference_removed_pct_of_overlap": round(100.0 * served / (served + produced), 1)
               if (served + produced) else None,
               "peak_rss_gb_A": [(r.get("telemetry") or {}).get("peak_rss_gb") for r in a],
               "peak_rss_gb_B": [(r.get("telemetry") or {}).get("peak_rss_gb") for r in b],
               "dedup_state_B": [r["dedup_state"] for r in b], "dedup_summary_B": [r["dedup_summary"] for r in b],
               "arms": [{k: r.get(k) for k in ("label", "arm", "returncode", "frame_loop_fps", "faces_per_s",
                                               "faces_per_frame", "stabilizer_path", "failed_frames", "swapped_lines",
                                               "wall_seconds", "stages", "counters", "telemetry", "output",
                                               "dedup_state", "dedup_summary", "swap_calls")} for r in recs]}
        summary["clips"][clip] = row
        # Written BEFORE any formatting, so a slip in the report line below can never lose a 30-minute run.
        path = os.path.join(REPO, "docs", "perf", "stab_dedup_%s%s.json" % (args.date, args.tag))
        json.dump(summary, open(path, "w"), indent=1, default=str)
        print("\n== %s: A %s | B %s | B/A %s | A-A %s%% B-B %s%% | outputs identical: sha %s md5 %s"
              % (clip, row["A_fps"], row["B_fps"], row["ratio_B_over_A"], row["A_vs_A_spread_pct"],
                 row["B_vs_B_spread_pct"], row["all_output_sha256_equal"], row["all_decoded_md5_equal"]), flush=True)
        print("   swap calls A %.0f B %.0f (-%d) | overlap visits computed %d served %d bypassed %d | duplicate runs "
              "removed: %s%% of all inference, %s%% of overlap inference | each overlap frame computed once: %s"
              % (a_calls, b_calls, a_calls - b_calls, produced, served, bypass,
                 row["duplicate_inference_removed_pct_of_all"], row["duplicate_inference_removed_pct_of_overlap"],
                 row["every_overlap_frame_computed_once"]), flush=True)
        print("   peak RSS A %s B %s GB\n" % (row["peak_rss_gb_A"], row["peak_rss_gb_B"]), flush=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
