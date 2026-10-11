"""ABBA end-to-end render: cross-frame swap batching ON (shipped default) vs OFF, for a chosen swap model.

    env/Scripts/python.exe tests/ab_swap_batch_render.py --swap-model realswap --clips d4,s7

Question (RECODE_STATUS 2026-10-11, Stage 5 option 2): does batching the `realswap` pair (hyperswap + the hififace eyelid
band, both coalesced by `FaceSwapInsightFace.RunBatchMulti`) move the render clock? The pair batching is already built and
tested; what was never measured is the end-to-end effect with the pair active.

A = default (batcher at its derived max_batch). B = `ROOP_BATCH_SWAP_MAX=1`, which `ProcessMgr._make_swap_batcher` documents as
"cross-frame batching OFF" while leaving the same batch-dynamic session built, so the arms differ in the batcher alone.
Drives `baseline_snapshot.run_one` (live config.yaml except swap_model, set in memory only; ROOP_PROFILE; swap audit; fps beside
faces/s), one discarded warm-up per clip (pays the TensorRT engine builds), then A B B A with ROOP_STAB_CHUNK_MB pinned.
The A-vs-A spread is the null control. The log must PROVE each arm: A has "[BatchSwap] cross-frame batching ON" and a
batcher report with avg batch size; B has "cross-frame batching OFF" and no report. An arm that did not run as labelled is
flagged, not averaged.
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

ON_RE = re.compile(r"\[BatchSwap\] cross-frame batching ON \(max_batch=(\d+)")
OFF_RE = re.compile(r"\[BatchSwap\] cross-frame batching OFF")
REPORT_RE = re.compile(r"batches: \d+\s+items: \d+\s+avg batch: [\d.]+\s+max: \d+|swap inference: [^\n]*ms/item\)")


def prove(text):
    """What the log says about the batcher, so an arm cannot be believed on its label alone."""
    on = ON_RE.search(text)
    return {"on_line": bool(on), "max_batch": int(on.group(1)) if on else None,
            "off_line": bool(OFF_RE.search(text)),
            "report": [m.group(0).strip()[:220] for m in REPORT_RE.finditer(text)][:6],
            "pair_mix_line": [l.strip()[:200] for l in text.splitlines() if "realswap" in l and "mix" in l.lower()][:2],
            "batch_unsupported": "does not support batched inference" in text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap-model", default="realswap")
    ap.add_argument("--clips", default="d4,s7")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--window", nargs=2, type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.detector_engine = None
    args.governor = False

    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    live_model = str(cfg.swap_model)
    cfg.swap_model = args.swap_model            # in memory only: config.yaml is never written
    threads = args.threads if args.threads is not None else int(cfg.max_threads)
    out_root = args.out or os.path.join(APP, "output", "ab_swap_batch_%s_%s" % (args.swap_model, args.date))
    os.makedirs(out_root, exist_ok=True)
    base_env = dict(os.environ)
    for k in ("ROOP_BATCH_SWAP_MAX",):
        base_env.pop(k, None)
    base_env["ROOP_PROFILE"] = "1"
    base_env["PYTHONIOENCODING"] = "utf-8"
    base_env.setdefault("ROOP_PIPELINE_LOG_EVERY", "100")
    base_env = bc.ensure_ffmpeg(base_env)
    os.environ["PATH"] = base_env["PATH"]

    specs = dict(bs.CLIPS)
    specs.update(EXTRA)
    summary = {"date": args.date, "head": bs.head_info(), "swap_model": args.swap_model, "live_config_swap_model": live_model,
               "threads": threads,
               "config": {k: str(getattr(cfg, k, None)) for k in (
                   "swap_model", "selected_enhancer", "mask_engine", "provider", "trt_precision", "max_threads",
                   "detector_engine", "face_detector_size", "perf_batch_swap", "perf_batch_max", "perf_trt_pool",
                   "perf_detmask_pool", "perf_detector_pool")},
               "clips": {}}
    for clip in args.clips.split(","):
        spec = specs[clip]
        window = tuple(args.window) if args.window else None
        w = bs.run_one("warmup_" + clip, clip, spec, cfg, args, dict(base_env), out_root, threads, window=(0, 120))
        chunk = (w.get("stab_ram_line") or {}).get("derived_chunk_budget_mb")
        print("  warm-up %s done (rc %s); derived ROOP_STAB_CHUNK_MB = %s -> pinned" % (clip, w["returncode"], chunk), flush=True)
        recs = []
        for i, arm in enumerate("ABBA"):
            env = dict(base_env)
            if chunk:
                env["ROOP_STAB_CHUNK_MB"] = str(chunk)
            if arm == "B":
                env["ROOP_BATCH_SWAP_MAX"] = "1"
            rec = bs.run_one("%s_%d%s" % (clip, i + 1, arm), clip, spec, cfg, args, env, out_root, threads, window=window)
            text = open(rec["log"], encoding="utf-8", errors="ignore").read()
            rec["arm"] = arm
            rec["proof"] = prove(text)
            ok = rec["proof"]["on_line"] and not rec["proof"]["batch_unsupported"] if arm == "A" else rec["proof"]["off_line"]
            rec["ran_as_labelled"] = bool(ok)
            print("  %s: frame-loop %s fps | %s faces/s | faces_seen %s | swapped %s | stab %s | ran-as-labelled %s %s"
                  % (rec["label"], rec.get("frame_loop_fps"), rec.get("faces_per_s"),
                     (rec.get("run") or {}).get("faces_seen"), (rec.get("swapped_lines") or ["?"])[0].strip()[:60],
                     rec.get("stabilizer_path"), rec["ran_as_labelled"], rec["proof"]["report"][:1]), flush=True)
            recs.append(rec)
        good = [r for r in recs if r["ran_as_labelled"]]
        a = [r["frame_loop_fps"] for r in good if r["arm"] == "A" and r["frame_loop_fps"]]
        b = [r["frame_loop_fps"] for r in good if r["arm"] == "B" and r["frame_loop_fps"]]
        row = {"A_fps": a, "B_fps": b, "chunk_mb": chunk,
               "A_on_mean": round(st.mean(a), 3) if a else None, "B_off_mean": round(st.mean(b), 3) if b else None,
               "ratio_on_over_off": round(st.mean(a) / st.mean(b), 4) if a and b else None,
               "A_vs_A_spread_pct": round(100 * abs(a[0] - a[1]) / st.mean(a), 2) if len(a) == 2 else None,
               "B_vs_B_spread_pct": round(100 * abs(b[0] - b[1]) / st.mean(b), 2) if len(b) == 2 else None,
               "arms_not_as_labelled": [r["label"] for r in recs if not r["ran_as_labelled"]],
               "arms": [{k: r.get(k) for k in ("label", "arm", "returncode", "frame_loop_fps", "faces_per_s",
                                               "faces_per_frame", "stabilizer_path", "proof", "ran_as_labelled",
                                               "failed_frames", "swapped_lines", "wall_seconds", "stages", "output")}
                        for r in recs]}
        summary["clips"][clip] = row
        print("\n== %s (%s): ON %s | OFF %s | on/off %s | A-vs-A %s%% | B-vs-B %s%% | unlabelled %s\n" % (
            clip, args.swap_model, row["A_fps"], row["B_fps"], row["ratio_on_over_off"], row["A_vs_A_spread_pct"],
            row["B_vs_B_spread_pct"], row["arms_not_as_labelled"]), flush=True)
        path = os.path.join(REPO, "docs", "perf", "swap_batch_render_%s_%s.json" % (args.swap_model, args.date))
        json.dump(summary, open(path, "w"), indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
