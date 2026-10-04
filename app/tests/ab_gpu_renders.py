"""Whole-render acceptance for detector_engine 'retinaface_r50_gpu' against the baseline engine.

    env/Scripts/python.exe tests/ab_gpu_renders.py

Per clip, real renders through `baseline_snapshot.py` (config.yaml live, ROOP_PROFILE=1,
ROOP_STAB_CHUNK_MB pinned to the baseline's value), arms in a counterbalanced order:

    scrfd   the live config's detector (the baseline)
    gpu_sq  retinaface_r50_gpu, squash preprocessing (the module default)
    gpu_lb  retinaface_r50_gpu, letterbox preprocessing (face_engine's own geometry)

For each arm it reads off what the brief's acceptance names:
  * swapped faces  (every `swapped ...` line of the swap audit, summed) and faces seen,
  * WRONG FACESET APPLIED (per-person verdicts),
  * track counts (`[Track] N tracks over M frames ... K matched to a source`),
  * pre-pass fps (the driver's timestamps on the two `[Memory] ...temporal-prepass-*` lines),
  * detector sessions logged, failed frames, return code.
d9.mp4 does not exist any more (the roster was retired; d6 replaced it), so it is not run.
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

ENGINE = "retinaface_r50_gpu"
ARMS = {"scrfd": (None, {}),
        "gpu_sq": (ENGINE, {"ROOP_R50_GPU_PREPROCESS": "squash"}),
        "gpu_lb": (ENGINE, {"ROOP_R50_GPU_PREPROCESS": "letterbox"}),
        # the CONTROL: the existing engine on the same network. If it swaps what gpu_sq swaps,
        # a gap to scrfd is the network's, not the port's.
        "r50": ("retinaface_r50", {})}

# clip -> (pin MB, [(arm, window or None), ...])
PLAN = {
    "d4": ("990.8", [(a, None) for a in ("scrfd", "gpu_sq", "gpu_lb", "gpu_lb", "gpu_sq", "scrfd")]),
    "Love": ("1494.9", [(a, None) for a in ("scrfd", "gpu_sq", "gpu_lb", "gpu_lb", "gpu_sq", "scrfd")]),
    "d1": ("859.8", [(a, None) for a in ("scrfd", "gpu_sq", "gpu_lb")]),
    # d6 is 4K at ~0.3-0.5 fps: a 100-frame window for the pre-pass timing, the whole clip
    # (as the baseline) for the swap counts
    "d6": ("1122.5", [("scrfd", (0, 100)), ("gpu_sq", (0, 100)), ("gpu_sq", None)]),
}


PLAN_CONTROL = {"d4": ("990.8", [("r50", None)]), "Love": ("1494.9", [("r50", None)]),
                "d1": ("859.8", [("r50", None)])}


def norm(t):
    return re.sub(r"\s+", " ", t or "").strip()


def swapped_total(audit):
    return sum(int(m.group(1)) for m in re.finditer(r"^\s*swapped \([^)]*\)\s+(\d+)\s", audit or "", re.M))


def read_arm(rec, label, arm, clip, window):
    text = open(rec["log"], encoding="utf-8", errors="replace").read()
    tr = re.search(r"\[Track\] (\d+) tracks over (\d+) frames(?: \(stitched down from (\d+)\))?, (\d+) matched to a source", text)
    wrong = [(int(a), int(b)) for a, b in re.findall(r"WRONG FACESET APPLIED on (\d+) of (\d+)", "\n".join(rec["person_verdicts"]))]
    c = rec["counters"]
    tot = lambda k: sum((c.get(k) or {}).values())
    return {
        "label": label, "clip": clip, "arm": arm, "window": list(window) if window else None,
        "rc": rec["returncode"], "failed_frames": rec["failed_frames"], "detector_failed": rec["detector_failed"],
        "faces_seen": (rec.get("run") or {}).get("faces_seen"),
        "swapped": swapped_total(rec["swap_audit"]),
        "wrong_faceset": sum(a for a, _ in wrong), "wrong_faceset_of": sum(b for _, b in wrong),
        "tracks": int(tr.group(1)) if tr else None, "tracks_stitched_from": int(tr.group(3)) if tr and tr.group(3) else None,
        "tracks_matched_to_a_source": int(tr.group(4)) if tr else None, "track_frames": int(tr.group(2)) if tr else None,
        "swap_audit": norm(rec["swap_audit"]), "verdicts": [norm(v) for v in rec["person_verdicts"]],
        "md5": rec["output"]["decoded_video_md5"],
        "prepass_seconds": rec.get("prepass_seconds"), "prepass_fps": rec.get("prepass_fps"),
        "frame_loop_fps": rec.get("frame_loop_fps"), "wall_seconds": rec["wall_seconds"],
        "detector_sessions": [l for l in rec["sessions"] if "detector:" in l or "retinaface" in l],
        "gpudet": {k: tot(k) for k in c if k.startswith("gpudet.")},
        "raw_total": tot("raw.total"), "first_pass_empty": tot("detect.first_pass_empty"),
        "peak_gpu_mb": (rec.get("telemetry") or {}).get("peak_gpu_memory_mb"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-10-04")
    ap.add_argument("--only", default="")
    ap.add_argument("--plan", choices=("main", "control"), default="main")
    ap.add_argument("--timeout-min", type=int, default=75)
    args = ap.parse_args()

    out_root = os.path.join(APP, "output", "gpu_ab_" + args.date)
    os.makedirs(out_root, exist_ok=True)
    py = os.path.join(APP, "env", "Scripts", "python.exe")
    rows = []
    for clip, (pin, arms) in (PLAN if args.plan == "main" else PLAN_CONTROL).items():
        if args.only and clip not in args.only.split(","):
            continue
        for i, (arm, window) in enumerate(arms):
            engine, extra_env = ARMS[arm]
            label = "%s_%s_%d%s" % (clip, arm, i + 1, "_w%d" % window[1] if window else "")
            js = os.path.join(out_root, "baseline_%s.json" % label)
            if os.path.exists(js):                       # resumable
                rec = json.load(open(js, encoding="utf-8"))["records"][0]
            else:
                cmd = [py, os.path.join(HERE, "baseline_snapshot.py"), "--date", label, "--only", clip,
                       "--no-null", "--no-warmup", "--pin-chunk-mb", pin,
                       "--docs-dir", out_root, "--out", os.path.join(out_root, label)]
                if engine:
                    cmd += ["--detector-engine", engine]
                if window:
                    cmd += ["--window", str(window[0]), str(window[1])]
                env = dict(os.environ, **extra_env)
                env.pop("ROOP_R50_GPU_PREPROCESS", None) if not extra_env else None
                print("=== %s (arm %s%s) ===" % (label, arm, ", window %s" % (window,) if window else ""), flush=True)
                try:
                    subprocess.run(cmd, cwd=APP, env=env, timeout=args.timeout_min * 60)
                except subprocess.TimeoutExpired:
                    print("!! %s TIMED OUT after %d min -- a hang" % (label, args.timeout_min), flush=True)
                    rows.append({"label": label, "clip": clip, "arm": arm, "hung": True})
                    continue
                if not os.path.exists(js):
                    rows.append({"label": label, "clip": clip, "arm": arm, "no_result": True})
                    continue
                rec = json.load(open(js, encoding="utf-8"))["records"][0]
            rows.append(read_arm(rec, label, arm, clip, window))
            r = rows[-1]
            print("   %-18s swapped %s/%s wrong %s tracks %s(%s) prepass %ss=%s fps loop %s rc %s" % (
                label, r["swapped"], r["faces_seen"], r["wrong_faceset"], r["tracks"],
                r["tracks_matched_to_a_source"], r["prepass_seconds"], r["prepass_fps"],
                r["frame_loop_fps"], r["rc"]), flush=True)
            with open(os.path.join(REPO, "docs", "perf", "gpu_engine_renders_%s%s.json" % (args.date, "" if args.plan == "main" else "_control")), "w",
                      encoding="utf-8") as fh:
                json.dump({"rows": rows}, fh, indent=1)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
