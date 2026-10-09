"""Does a LARGER stabilization block land closer to a SEQUENTIAL render? (ground truth for block sizing)

    env/Scripts/python.exe tests/ab_stab_blocks_vs_sequential.py --clip d4 --window 0 240 --mults 2,4,8,16

A sequential render (ROOP_STAB_PARALLEL=0: one thread, every filter advanced through the clip in order) is what the
parallel path is trying to reproduce; its only departure is the seed residual (<= 1%) each block carries into its
first output frames, so the question for block sizing is whether bigger blocks - fewer seams - are closer to it. The
parallel arms use the dedup cache (verified bit-identical to no cache by tests/ab_stab_dedup.py), ROOP_STAB_CHUNK_MB
pinned, and an explicit ROOP_STAB_BLOCK_MULT per arm.

Reported per arm, against the sequential output decoded frame by frame: mean / p99 / max absolute pixel difference, how
many frames differ at all, the same split into HEAD-of-block frames (the first `warm-up` frames after a block start,
where a seed residual would live) and the rest, plus the arm's wall fps and peak RSS. "Closer" means the mean and the
head-of-block difference fall as the block grows; "never further" is checked monotonically across the mults.
"""
import argparse
import datetime
import json
import os
import re
import sys

import cv2
import numpy as np

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


def frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    cap.release()
    return out


def compare(ref, cand, block, wu):
    n = min(len(ref), len(cand))
    per = []
    for i in range(n):
        d = np.abs(ref[i].astype(np.int16) - cand[i].astype(np.int16))
        per.append((float(d.mean()), float(d.max())))
    mean = np.array([p[0] for p in per])
    mx = np.array([p[1] for p in per])
    head = np.array([(i % block) < wu and i >= block for i in range(n)]) if block else np.zeros(n, bool)
    body = (~head) & (np.arange(n) >= block)
    return {"frames_compared": n, "frames_differing": int((mean > 0).sum()),
            "mean_abs_diff": round(float(mean.mean()), 5), "p99_frame_mean": round(float(np.percentile(mean, 99)), 5),
            "max_abs_diff": float(mx.max()),
            "head_of_block_mean": round(float(mean[head].mean()), 5) if head.any() else None,
            "body_mean": round(float(mean[body].mean()), 5) if body.any() else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="d4")
    ap.add_argument("--window", nargs=2, type=int, default=[0, 240])
    ap.add_argument("--mults", default="2,4,8,16", help="comma list; empty = no parallel arms")
    ap.add_argument("--seq-repeat", action="store_true",
                    help="render the sequential arm TWICE and report SEQ2 vs SEQ: is the ground truth itself reproducible?")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--chunk-mb", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    args.detector_engine = None
    args.governor = False

    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    threads = args.threads if args.threads is not None else int(cfg.max_threads)
    out_root = args.out or os.path.join(APP, "output", "ab_stab_blocks_" + args.date + args.tag)
    os.makedirs(out_root, exist_ok=True)
    base_env = dict(os.environ)
    for k in ("ROOP_STAB_DEDUP", "ROOP_STAB_BLOCK_MULT", "ROOP_STAB_PARALLEL", "ROOP_TRT_BOUND"):
        base_env.pop(k, None)
    base_env.update({"ROOP_PROFILE": "1", "PYTHONIOENCODING": "utf-8"})
    base_env.setdefault("ROOP_PIPELINE_LOG_EVERY", "100")
    base_env = bc.ensure_ffmpeg(base_env)
    os.environ["PATH"] = base_env["PATH"]

    specs = dict(bs.CLIPS)
    specs.update(EXTRA)
    spec = specs[args.clip]
    window = tuple(args.window)
    chunk = args.chunk_mb
    if not chunk:
        w = bs.run_one("warmup_" + args.clip, args.clip, spec, cfg, args, dict(base_env), out_root, threads,
                       window=(0, 120))
        chunk = (w.get("stab_ram_line") or {}).get("derived_chunk_budget_mb")
        print("  warm-up done (rc %s); ROOP_STAB_CHUNK_MB = %s pinned" % (w["returncode"], chunk), flush=True)

    arms = [("SEQ", {"ROOP_STAB_PARALLEL": "0"})]
    if args.seq_repeat:
        arms.append(("SEQ2", {"ROOP_STAB_PARALLEL": "0"}))
    arms += [("P%s" % m, {"ROOP_STAB_BLOCK_MULT": str(m)}) for m in args.mults.split(",") if m.strip()]
    recs = {}
    for label, extra in arms:
        env = dict(base_env)
        env["ROOP_STAB_CHUNK_MB"] = str(chunk)
        env.update(extra)
        rec = bs.run_one("%s_%s" % (args.clip, label), args.clip, spec, cfg, args, env, out_root, threads, window=window)
        geo = (rec.get("stab_geometry") or [""])[0]
        m = re.search(r"wu=(\d+) block=(\d+) workers=(\d+).*?blocks_per_chunk=(\d+)", geo)
        rec["geometry"] = {"wu": int(m.group(1)), "block": int(m.group(2)), "workers": int(m.group(3)),
                           "blocks_per_chunk": int(m.group(4))} if m else None
        recs[label] = rec
        print("  %s: %s fps | geometry %s | stabilizer path %s | peak RSS %s GB | sha %s" % (
            label, rec.get("frame_loop_fps"), rec["geometry"], rec.get("stabilizer_path"),
            (rec.get("telemetry") or {}).get("peak_rss_gb"), ((rec.get("output") or {}).get("sha256") or "")[:12]),
            flush=True)

    ref = frames(recs["SEQ"]["output"]["file"])
    rows = {}
    for label, rec in recs.items():
        if label == "SEQ":
            continue
        g = rec["geometry"] or {}
        cmp_ = compare(ref, frames(rec["output"]["file"]), g.get("block", 0), g.get("wu", 0))
        rows[label] = {"geometry": g, "fps": rec.get("frame_loop_fps"), "peak_rss_gb": (rec.get("telemetry") or {}).get("peak_rss_gb"),
                       "faces_per_s": rec.get("faces_per_s"), "stabilizer_path": rec.get("stabilizer_path"), **cmp_}
    means = [rows[k]["mean_abs_diff"] for k in rows if k.startswith("P")]
    summary = {"date": args.date, "head": bs.head_info(), "clip": args.clip, "window": list(window), "threads": threads,
               "chunk_mb": chunk, "sequential_fps": recs["SEQ"].get("frame_loop_fps"),
               "arms": rows, "mean_diff_monotone_non_increasing_with_block": all(a >= b for a, b in zip(means, means[1:])),
               "sequential_vs_sequential": rows.get("SEQ2"),
               "sequential_sha256": {k: (v.get("output") or {}).get("sha256") for k, v in recs.items() if k.startswith("SEQ")}}
    for k, v in rows.items():
        print("  %-4s block %-3s workers %-2s | mean|d| %.5f head-of-block %s body %s p99 %.5f max %.0f | differing frames %d/%d | %s fps"
              % (k, v["geometry"].get("block"), v["geometry"].get("workers"), v["mean_abs_diff"], v["head_of_block_mean"],
                 v["body_mean"], v["p99_frame_mean"], v["max_abs_diff"], v["frames_differing"], v["frames_compared"], v["fps"]))
    if "SEQ2" in rows:
        print("  SEQ2 vs SEQ (noise floor of the ground truth itself): mean|d| %.5f, differing frames %d/%d, max %.0f; sha equal: %s"
              % (rows["SEQ2"]["mean_abs_diff"], rows["SEQ2"]["frames_differing"], rows["SEQ2"]["frames_compared"],
                 rows["SEQ2"]["max_abs_diff"], len(set(summary["sequential_sha256"].values())) == 1))
    print("  mean difference non-increasing as blocks grow:", summary["mean_diff_monotone_non_increasing_with_block"])
    path = os.path.join(REPO, "docs", "perf", "stab_blocks_vs_sequential_%s%s.json" % (args.date, args.tag))
    json.dump(summary, open(path, "w"), indent=1, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
