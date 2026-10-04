"""Whole-render check of the pyramid change on retinaface_r50: no hang, same output.

    env/Scripts/python.exe tests/ab_pyramid_renders.py --base c4bfff9

The function-level A/B (ab_pyramid.py) cannot show the one thing that could go badly
wrong: pool workers that HOLD a FaceAnalysis lease and then call the multi-scale
detector (change 2 keys off that lease). A real render has all of that at once -- the
pre-pass pool, the stabilizer blocks, the detector pool -- so this runs d6 (4K, every
frame triggers the pyramid) through the whole pipeline with detector_engine forced to
retinaface_r50 (a harness flag; config.yaml is not touched), old vs new, ABBA.

OLD = face_detector.py AND face_util.py from --base (the new face_util needs the new
face_detector's lease marker, so the pair is swapped together); NEW = HEAD. Each render
has a hard timeout, because the failure being hunted is a hang. ROOP_STAB_CHUNK_MB is
pinned to the baseline's d6 value so pixels are comparable.
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
FILES = ["app/roop/face_detector.py", "app/roop/face_util.py"]
PIN = "1122.5"


def sh(*a):
    return subprocess.run(a, cwd=REPO, check=True, capture_output=True, text=True).stdout


def set_arm(arm, base):
    for f in FILES:
        if arm == "old":
            src = subprocess.run(["git", "show", "%s:%s" % (base, f)], cwd=REPO, check=True,
                                 capture_output=True).stdout
            with open(os.path.join(REPO, f), "wb") as fh:
                fh.write(src)
        else:
            sh("git", "checkout", "HEAD", "--", f)


def norm(t):
    return re.sub(r"\s+", " ", t or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="c4bfff9")
    ap.add_argument("--date", default="2026-10-04")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--timeout-min", type=int, default=45)
    ap.add_argument("--arms", default="old,new,new,old")
    args = ap.parse_args()

    if sh("git", "status", "--porcelain", "--", *FILES).strip():
        raise SystemExit("commit the NEW code first: %s" % FILES)
    out_root = os.path.join(APP, "output", "pyramid_ab_" + args.date)
    os.makedirs(out_root, exist_ok=True)
    py = os.path.join(APP, "env", "Scripts", "python.exe")
    recs = []
    try:
        for i, arm in enumerate(args.arms.split(",")):
            set_arm(arm, args.base)
            label = "d6r50_%s_%d" % (arm, i + 1)
            cmd = [py, os.path.join(HERE, "baseline_snapshot.py"), "--date", label, "--only", "d6",
                   "--no-null", "--no-warmup", "--pin-chunk-mb", PIN,
                   "--detector-engine", "retinaface_r50", "--window", "0", str(args.frames),
                   "--docs-dir", out_root, "--out", os.path.join(out_root, label)]
            print("=== %s (arm %s) ===" % (label, arm), flush=True)
            try:
                r = subprocess.run(cmd, cwd=APP, timeout=args.timeout_min * 60)
            except subprocess.TimeoutExpired:
                print("!! %s TIMED OUT after %d min -- a hang" % (label, args.timeout_min), flush=True)
                recs.append({"label": label, "arm": arm, "hung": True})
                continue
            js = os.path.join(out_root, "baseline_%s.json" % label)
            if not os.path.exists(js):
                recs.append({"label": label, "arm": arm, "no_result": True, "rc": r.returncode})
                continue
            rec = json.load(open(js, encoding="utf-8"))["records"][0]
            rec["arm"], rec["order"] = arm, i + 1
            recs.append(rec)
    finally:
        for f in FILES:
            sh("git", "checkout", "HEAD", "--", f)
        print("[ab] restored %s from HEAD" % FILES, flush=True)

    rows = []
    for r in recs:
        if r.get("hung") or r.get("no_result"):
            rows.append({k: r.get(k) for k in ("label", "arm", "hung", "no_result")})
            continue
        c = r["counters"]
        tot = lambda k: sum((c.get(k) or {}).values())
        rows.append({"label": r["label"], "arm": r["arm"], "order": r["order"], "rc": r["returncode"],
                     "md5": r["output"]["decoded_video_md5"], "rows_csv": r["output"]["rows_csv_sha256"],
                     "swap_audit": norm(r["swap_audit"]), "verdicts": [norm(v) for v in r["person_verdicts"]],
                     "failed_frames": r["failed_frames"], "detector_failed": r["detector_failed"],
                     "faces_seen": (r.get("run") or {}).get("faces_seen"),
                     "pyramid_detect_calls": tot("pyramid.detect_calls"),
                     "pyramid_executed": tot("pyramid.path.pyramid_executed"),
                     "pyramid_scale_passes": tot("pyramid.scale_passes"),
                     "single_pass_reused": tot("pyramid.single_pass_reused"),
                     "detect_fn_calls": tot("pyramid.detect_fn_calls"),
                     "mode_sequential_in_pool_worker": tot("pyramid.mode.sequential_in_pool_worker"),
                     "mode_shared_executor": tot("pyramid.mode.shared_executor"),
                     "prepass_seconds": r.get("prepass_seconds"), "prepass_fps": r.get("prepass_fps"),
                     "frame_loop_fps": r.get("frame_loop_fps"), "wall_seconds": r["wall_seconds"],
                     "raw_total": tot("raw.total")})
    ok = [r for r in rows if "md5" in r]
    summary = {"base": args.base, "frames": args.frames, "engine": "retinaface_r50", "rows": rows,
               "any_hang": any(r.get("hung") for r in rows),
               "all_returned_zero": all(r.get("rc") == 0 for r in ok),
               "md5_all_equal": len({r["md5"] for r in ok}) == 1,
               "swap_audit_all_equal": len({r["swap_audit"] for r in ok}) == 1,
               "verdicts_all_equal": len({tuple(r["verdicts"]) for r in ok}) == 1,
               "rows_csv_all_equal": len({r["rows_csv"] for r in ok}) == 1}
    path = os.path.join(REPO, "docs", "perf", "pyramid_renders_%s.json" % args.date)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    for r in rows:
        print(r)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
