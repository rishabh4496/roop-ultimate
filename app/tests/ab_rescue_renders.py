"""Whole-render A/B for the rotated-rescue aux change: same output, fewer aux calls, faster pre-pass?

    env/Scripts/python.exe tests/ab_rescue_renders.py --base f9aaf96

OLD = `app/roop/face_util.py` as of --base, put in the working tree for the old arms and
restored from HEAD afterwards (the file is self-contained: nothing else changed in the
render path). NEW = HEAD. Everything else is `baseline_snapshot.py`: config.yaml live,
ROOP_PROFILE=1, ROOP_STAB_CHUNK_MB PINNED to the value the baseline derived (without it
free RAM changes the stabilizer geometry and the pixels, which would make a pixel
comparison meaningless), one render at a time.

Order per clip is ABBA (old,new,new,old), after one discarded warm-up: a clip's first
render runs 28-40% slower than a repeat on this machine, so an unbalanced pair would
measure arm order. Pre-pass time is the driver's timestamps on the two
`[Memory] stage=phase3:temporal-prepass-*` lines.

Acceptance read off the result:
  * decoded-video md5 identical across all arms AND equal to the recorded baseline;
  * swap audit text and per-person verdicts (incl. WRONG FACESET) identical;
  * aux model calls fewer, detector executions unchanged.
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

PIN = {"d4": "990.8", "Love": "1494.9", "d1": "859.8"}
PLAN = [("d4", ["old", "new", "new", "old"]), ("Love", ["old", "new", "new", "old"]),
        ("d1", ["new"])]
BASELINE_MD5 = {"d4": "b37f2bebe6759f20506c0aa90a13c262", "Love": None, "d1": None}
FACE_UTIL = "app/roop/face_util.py"


def sh(*a, **k):
    return subprocess.run(a, cwd=REPO, check=True, capture_output=True, text=True, **k).stdout


def set_arm(arm, base):
    if arm == "old":
        src = subprocess.run(["git", "show", "%s:%s" % (base, FACE_UTIL)], cwd=REPO,
                             check=True, capture_output=True).stdout
        with open(os.path.join(REPO, FACE_UTIL), "wb") as fh:
            fh.write(src)
    else:
        sh("git", "checkout", "HEAD", "--", FACE_UTIL)


def norm(t):
    return re.sub(r"\s+", " ", t or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="f9aaf96")
    ap.add_argument("--date", default="2026-10-04")
    args = ap.parse_args()

    # The baseline's recorded hashes (pinned runs), so "same as baseline" is checked
    # against the stored reference and not just against this session's old arm.
    for js in ("baseline_2026-10-04_pinned_d1.json", "baseline_2026-10-04_pinned_Love.json"):
        d = json.load(open(os.path.join(REPO, "docs", "perf", js), encoding="utf-8"))["records"][0]
        BASELINE_MD5[d["clip"]] = d["output"]["decoded_video_md5"]

    if sh("git", "status", "--porcelain", "--", FACE_UTIL).strip():
        raise SystemExit("%s has uncommitted changes: commit the NEW code first (the old arm "
                         "is swapped in over it and HEAD is restored afterwards)." % FACE_UTIL)

    out_root = os.path.join(APP, "output", "rescue_ab_" + args.date)
    os.makedirs(out_root, exist_ok=True)
    py = os.path.join(APP, "env", "Scripts", "python.exe")
    recs = []
    first = True
    try:
        for clip, arms in PLAN:
            for i, arm in enumerate(arms):
                set_arm(arm, args.base)
                label = "%s_%s_%d" % (clip, arm, i + 1)
                cmd = [py, os.path.join(HERE, "baseline_snapshot.py"), "--date", label,
                       "--only", clip, "--no-null", "--pin-chunk-mb", PIN[clip],
                       "--docs-dir", out_root, "--out", os.path.join(out_root, label)]
                if not first:
                    cmd.append("--no-warmup")
                first = False
                print("=== %s (arm %s) ===" % (label, arm), flush=True)
                r = subprocess.run(cmd, cwd=APP)
                js = os.path.join(out_root, "baseline_%s.json" % label)
                if not os.path.exists(js):
                    raise SystemExit("%s produced no result (rc %s)" % (label, r.returncode))
                rec = json.load(open(js, encoding="utf-8"))["records"][0]
                rec["arm"], rec["order"] = arm, i + 1
                recs.append(rec)
    finally:
        set_arm("new", args.base)
        print("[ab] restored %s from HEAD" % FACE_UTIL, flush=True)

    def aux(r):
        return sum(sum((r["counters"].get(k) or {}).values()) for k in (
            "aux.buffalo_l.recognition.get", "aux.buffalo_l.landmark_2d_106.get",
            "aux.buffalo_l.landmark_3d_68.get"))

    def det(r):
        return sum((r["counters"].get("raw.total") or {}).values())

    rows = []
    for r in recs:
        rows.append({"label": r["label"], "clip": r["clip"], "arm": r["arm"], "order": r["order"],
                     "md5": r["output"]["decoded_video_md5"],
                     "rows_csv": r["output"]["rows_csv_sha256"],
                     "swap_audit": norm(r["swap_audit"]), "verdicts": [norm(v) for v in r["person_verdicts"]],
                     "aux_model_calls": aux(r), "detector_executions": det(r),
                     "prepass_seconds": r.get("prepass_seconds"), "prepass_fps": r.get("prepass_fps"),
                     "frame_loop_fps": r.get("frame_loop_fps"), "faces_seen": (r.get("run") or {}).get("faces_seen"),
                     "failed_frames": r["failed_frames"], "returncode": r["returncode"]})
    summary = {"base": args.base, "rows": rows, "per_clip": {}}
    for clip in {r["clip"] for r in rows}:
        rs = [r for r in rows if r["clip"] == clip]
        olds, news = [r for r in rs if r["arm"] == "old"], [r for r in rs if r["arm"] == "new"]
        def mean(xs):
            xs = [x for x in xs if x is not None]
            return round(sum(xs) / len(xs), 3) if xs else None
        summary["per_clip"][clip] = {
            "md5_all_arms_equal": len({r["md5"] for r in rs}) == 1,
            "md5_equals_recorded_baseline": all(r["md5"] == BASELINE_MD5.get(clip) for r in rs),
            "swap_audit_all_equal": len({r["swap_audit"] for r in rs}) == 1,
            "verdicts_all_equal": len({tuple(r["verdicts"]) for r in rs}) == 1,
            "rows_csv_all_equal": len({r["rows_csv"] for r in rs}) == 1,
            "aux_calls_old": mean([r["aux_model_calls"] for r in olds]),
            "aux_calls_new": mean([r["aux_model_calls"] for r in news]),
            "detector_executions_old": mean([r["detector_executions"] for r in olds]),
            "detector_executions_new": mean([r["detector_executions"] for r in news]),
            "prepass_fps_old": mean([r["prepass_fps"] for r in olds]),
            "prepass_fps_new": mean([r["prepass_fps"] for r in news]),
            "prepass_seconds_old": mean([r["prepass_seconds"] for r in olds]),
            "prepass_seconds_new": mean([r["prepass_seconds"] for r in news]),
            "frame_loop_fps_old": mean([r["frame_loop_fps"] for r in olds]),
            "frame_loop_fps_new": mean([r["frame_loop_fps"] for r in news])}
    path = os.path.join(REPO, "docs", "perf", "rescue_aux_renders_%s.json" % args.date)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary["per_clip"], indent=1))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
