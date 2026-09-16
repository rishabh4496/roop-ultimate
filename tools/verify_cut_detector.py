"""Run the SHIPPED StreamingStabilizationHistory over a real clip and report
the cuts it now finds.

This exercises roop.one_euro directly -- the same object the render path
constructs -- rather than a copy of its maths, so the number printed here is
the number the pipeline will act on.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from roop.one_euro import StreamingStabilizationHistory  # noqa: E402


def open_capture(pattern):
    path = pattern
    if any(ch in pattern for ch in "*?[") or not os.path.exists(pattern):
        matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if matches:
            path = matches[0]
    shown = path
    if os.name == "nt":
        try:
            import ctypes

            buf = ctypes.create_unicode_buffer(1024)
            if ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 1024) and buf.value:
                path = buf.value
        except Exception:
            pass
    return cv2.VideoCapture(path), shown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    cap, shown = open_capture(args.video)
    if not cap.isOpened():
        print(f"cannot open {args.video}")
        return 2
    print(f"clip: {shown}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    hist = StreamingStabilizationHistory()
    print(f"detector: floor={hist.cut_floor} ratio={hist.cut_ratio} "
          f"window={hist.CUT_WINDOW} explicit_threshold={hist.cut_threshold}")

    cuts = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if hist.observe_frame(frame, idx):
            cuts.append(idx)
        idx += 1
        if args.limit and idx >= args.limit:
            break
        if idx % 2000 == 0:
            print(f"JCODE_PROGRESS {json.dumps({'current': idx, 'total': total, 'unit': 'frames', 'message': 'scanning'})}",
                  flush=True)
    cap.release()

    dur_min = idx / max(fps, 1e-6) / 60.0
    print(f"\nframes {idx}, cuts {len(cuts)} ({len(cuts) / max(dur_min, 1e-6):.1f}/min)")
    if cuts:
        shots = np.diff(np.concatenate(([0], cuts, [idx])))
        shots = shots[shots > 0]
        print(f"shot frames: min={shots.min()} median={int(np.median(shots))} "
              f"mean={shots.mean():.1f} max={shots.max()}")
        print(f"shots <= 10 frames: {(shots <= 10).sum()}   "
              f"<= 15 frames: {(shots <= 15).sum()}")
        print(f"first 20 cuts: {cuts[:20]}")
    assert hist.scene_cuts == len(cuts), "scene_cuts counter disagrees with returns"
    print("scene_cuts counter agrees with returned cuts")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"frames": idx, "fps": fps, "cuts": cuts}, fh)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
