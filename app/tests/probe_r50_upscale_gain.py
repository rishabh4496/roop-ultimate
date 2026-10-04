"""Does `_rescue_upscaled` EVER gain a face on retinaface_r50?

    env/Scripts/python.exe tests/probe_r50_upscale_gain.py --date 2026-10-04

The question behind it: `_detect_faces` runs a 2x-upscale rescue when the first pass
finds nothing. retinaface_r50 resizes its input square to 640 either way, so a 2x
upscale should reach the network as a near-identical picture (and a face that was
under the size it can see stays under it), and the upscaled frame can also trip the
multi-scale pyramid's >=500 px close-up trigger. If the rescue never gains a face
there, it is two extra detector runs on every empty frame for nothing.

This measures it instead of arguing it, on the baseline clips' own frames:

  pass 1 (engine = retinaface_r50): scan each clip's baseline window, keep frames where
          the r50 FIRST pass (rescue=False) returns nothing, until N are collected
          (quota split across the clips, shortfall made up from the others), and on each
          call `_rescue_upscaled` directly, recording the faces it returns;
  pass 2 (engine = scrfd, one engine switch): count the faces scrfd finds on the same
          frames. A frame r50 misses but scrfd sees is where the rescue had a chance;
          a frame with no face at all cannot show a gain and would make "0 gained"
          meaningless on its own.

Measurement only: it never edits the code under test.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import fixtures                      # noqa: E402

# Baseline windows (docs/perf/baseline_2026-10-04.md).
CLIPS = [("d4", "double/d4.mp4", 0, 600),
         ("Love", "Love.mp4", 1100, 1700),
         ("d1", "double/d1.mp4", 0, 418),
         ("d6", "double/d6.mp4", 0, 488)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-10-04")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--engine", default="retinaface_r50")
    args = ap.parse_args()

    from settings import Settings
    import angle_bench as ab
    cfg = Settings(os.path.join(APP, "config.yaml"))
    g = ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), "None", "None", sync_config=True)
    g.g_desired_face_analysis = ["landmark_3d_68", "landmark_2d_106", "detection", "recognition"]
    g.lm68_lazy = False
    g.detector_engine = args.engine
    g.rescue_small_faces = True
    from roop import face_util as fu

    quota = max(1, args.frames // len(CLIPS))
    print("[probe] engine=%s det_size=%s thresh=%s quota/clip=%d"
          % (g.detector_engine, g.face_detector_size, g.face_detector_threshold, quota), flush=True)

    found = []          # dicts: clip, idx, gained, frame (kept for pass 2)
    scanned = {}
    per_clip_empty = {}
    t0 = time.time()
    # Two sweeps: first honour the per-clip quota, then top up from whatever is left.
    for sweep, cap_per_clip in (("quota", quota), ("top-up", 10 ** 9)):
        for name, rel, lo, hi in CLIPS:
            if len(found) >= args.frames:
                break
            cap = cv2.VideoCapture(fixtures.clip(rel, required=True))
            start = (scanned.get(name, lo) if sweep == "top-up" else lo)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            i = start
            while i < hi and len(found) < args.frames:
                ok, fr = cap.read()
                if not ok:
                    break
                i += 1
                scanned[name] = i
                if sweep == "quota" and per_clip_empty.get(name, 0) >= cap_per_clip:
                    break
                first = fu._detect_faces(fr, rescue=False)
                if first:
                    continue
                gained = fu._rescue_upscaled(fr) or []
                per_clip_empty[name] = per_clip_empty.get(name, 0) + 1
                found.append({"clip": name, "idx": i - 1, "gained": len(gained),
                              "frame": fr, "gained_boxes": [[float(v) for v in f.bbox] for f in gained]})
            cap.release()
    print("[probe] pass 1 done: %d empty frames in %.0fs; scanned %s"
          % (len(found), time.time() - t0, scanned), flush=True)

    # pass 2: scrfd on the same frames (one engine switch)
    g.detector_engine = "scrfd"
    for rec in found:
        rec["scrfd_faces"] = len(fu._detect_faces(rec["frame"], rescue=False) or [])
    g.detector_engine = args.engine

    for rec in found:
        rec.pop("frame")
    by_clip = {}
    for rec in found:
        d = by_clip.setdefault(rec["clip"], {"empty_frames": 0, "upscale_gained_frames": 0,
                                             "upscale_faces": 0, "scrfd_sees_face": 0,
                                             "scrfd_sees_face_and_upscale_gained": 0})
        d["empty_frames"] += 1
        d["upscale_gained_frames"] += int(rec["gained"] > 0)
        d["upscale_faces"] += rec["gained"]
        d["scrfd_sees_face"] += int(rec["scrfd_faces"] > 0)
        d["scrfd_sees_face_and_upscale_gained"] += int(rec["scrfd_faces"] > 0 and rec["gained"] > 0)
    total = {k: sum(v[k] for v in by_clip.values()) for k in next(iter(by_clip.values()))} if by_clip else {}
    out = {"engine": args.engine, "frames_requested": args.frames, "per_clip": by_clip, "total": total,
           "frames": found, "scanned_to": scanned,
           "decision": "SKIP _rescue_upscaled on retinaface_r50" if total and total["upscale_faces"] == 0
                       else "KEEP _rescue_upscaled (it gained a face)"}
    print(json.dumps({k: out[k] for k in ("engine", "per_clip", "total", "decision")}, indent=1))
    path = os.path.join(REPO, "docs", "perf", "r50_upscale_gain_%s.json" % args.date)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
