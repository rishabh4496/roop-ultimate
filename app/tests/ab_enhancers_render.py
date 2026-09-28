"""Counterbalanced end-to-end render A/B across ENHANCERS: fps and swap rate per arm.

    env/Scripts/python.exe tests/ab_enhancers_render.py --source <faceset>
    env/Scripts/python.exe tests/ab_enhancers_render.py --source <faceset> \\
        --arms "GPEN 256 Pro,GPEN 256 Pro|ROOP_GPEN256PRO_FILTER=cpu"

Every arm renders the same clip (default single/s7.mp4, exactly 600 frames:
the acceptance minimum) with everything from config.yaml except the enhancer,
through compare_enhancers_video.render. Arms run A..Z then Z..A in ONE process,
because the first arm pays the TensorRT engine builds (read uncounterbalanced
that has produced false +10-22% here). An arm may carry environment
overrides after a ``|`` (``;``-separated), restored after the arm.

The mask engine comes from config.yaml as the UI label ("DFL XSeg") and is
mapped to its plugin key the way the app does (api.map_mask_engine); passing
the label straight through raised KeyError in ProcessMgr.

WHAT IT FOUND (2026-09-28, RTX 4070, s7.mp4, hyperswap + DFL XSeg, threads 20):

    arm              fps (two arms)     extra wall ms/frame vs None (14.68)
    None             11.77 / 14.68      -
    GPEN 256         12.52 / 13.43      +9.0    network 3.5 ms
    GPEN 256 Pro     11.88 / 11.64      +16.9   network 3.5 ms
    UltraMax         10.71 / 10.62      +25.7   network 22.4 ms
    Restore Ultra    10.09 / 10.12      +30.8   network 20.4 ms

On this clip each enhancer costs the render about its NETWORK's GPU time, and
nothing else it does shows up (same run, counterbalanced):

    Restoreformer++ vs Restore Ultra (+ its 12 ms CPU finish)   7.99 vs 7.98 fps
    GPEN 256 Pro GPU filter vs CPU filter                       7.10 vs 7.19 fps

(that second run sat ~40% slower overall -- the rig drift AGENTS.md
describes; its arms are comparable with each other, not with the first table).
The networks already run TensorRT FP16 ("mixed") and a native TensorRT engine
built from the same ONNX is not faster (GPEN-256 3.47 ORT vs 6.81 native,
RestoreFormer++ 20.44 vs 21.11 ms). So the only lever left for these
enhancers' render cost is running the network LESS (e.g. skipping faces too
small on screen to show it; s7 / d1 faces are 480-610 px diagonal, so that
cannot move this clip).
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys
import time
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compare_enhancers_video as C
import fixtures
from ab_temporal_detection import SEEN_RE, SWAPPED_RE

DEFAULT_ARMS = "None,GPEN 256,GPEN 256 Pro,UltraMax,Restore Ultra"


def _frames(clip):
    import cv2

    cap = cv2.VideoCapture(clip)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def run_arm(arm, i, args, swapper, mask):
    name, _, env = arm.partition("|")
    pairs = dict(kv.split("=", 1) for kv in env.split(";") if kv)
    saved = {k: os.environ.get(k) for k in pairs}
    os.environ.update(pairs)
    out_dir = os.path.join(APP, "output", "ab_enhancers",
                           f"{args.tag}_{i:02d}_" + "".join(c if c.isalnum() else "_"
                                                            for c in arm))
    buf = io.StringIO()
    t0 = time.time()
    try:
        with redirect_stdout(buf):
            _, elapsed = C.render(args.clip, args.source, name, out_dir, swapper, mask,
                                  args.threads)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    text = re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "arm.log"), "w", encoding="utf-8", errors="ignore") as f:
        f.write(text)
    sw, seen = SWAPPED_RE.search(text), SEEN_RE.search(text)
    return {"i": i, "arm": arm, "elapsed": elapsed, "wall": time.time() - t0,
            "fps": args.n_frames / elapsed if elapsed else None,
            "swapped": int(sw.group(1)) if sw else None,
            "swap_pct": float(sw.group(2)) if sw else None,
            "seen": int(seen.group(1)) if seen else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clip", default=fixtures.clip("single/s7.mp4"))
    ap.add_argument("--source", required=True, help="faceset name in app/facesets")
    ap.add_argument("--arms", default=DEFAULT_ARMS,
                    help="comma-separated enhancer names; NAME|ENV=VAL;ENV2=VAL")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    os.chdir(APP)

    from api import map_mask_engine
    from settings import Settings

    cfg = Settings("config.yaml")
    swapper = cfg.swap_model
    mask = map_mask_engine(cfg.mask_engine, getattr(cfg, "clip_text", "")) or "None"
    args.n_frames = _frames(args.clip)
    if args.n_frames < 600:
        print(f"[ab] WARNING: {args.n_frames} frames; acceptance claims need 600")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a.partition("|")[0] not in C.VALID_ENHANCERS:
            raise SystemExit(f"[ab] {a!r} is not an enhancer name roop.core matches")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True, check=False).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "roop"],
                                capture_output=True, text=True, check=False).stdout.strip())
    print(f"[ab] {os.path.basename(args.clip)} {args.n_frames} frames  swapper {swapper}  "
          f"mask {mask}  threads {args.threads}  HEAD {head} dirty={dirty}", flush=True)
    rows = []
    order = arms + arms[::-1]
    for i, arm in enumerate(order):
        row = run_arm(arm, i, args, swapper, mask)
        rows.append(row)
        print(f"\n[ab] {i + 1}/{len(order)} {arm:14s} {row['fps']:.2f} fps  swapped "
              f"{row['swapped']} ({row['swap_pct']}%) seen {row['seen']}", flush=True)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump({"head": head, "dirty": dirty, "rows": rows}, f, indent=1)
    print("\n[ab] summary (mean of the two counterbalanced arms):", flush=True)
    for arm in arms:
        f = [r["fps"] for r in rows if r["arm"] == arm]
        print(f"[ab]   {arm:14s} {sum(f) / len(f):6.2f} fps  arms {[round(x, 2) for x in f]}",
              flush=True)


if __name__ == "__main__":
    main()
