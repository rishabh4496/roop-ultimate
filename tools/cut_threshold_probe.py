"""Find where the real hard cuts are in a clip, and what the shipped
threshold actually catches.

CONTEXT. StreamingStabilizationHistory (roop/one_euro.py) declares a hard cut
when the mean absolute difference of two consecutive 64px luma thumbnails is
>= 0.32. Both consumers of that signal treat a cut as a real discontinuity:
the swap phase resets the kps/enhancer/mask stabilisers, and the tracking
pre-pass collects garbage. On the clip under test the LARGEST difference any
pair of adjacent frames produces is 0.20, so the gate never once fires and
neither consumer ever sees a cut -- on footage that is visibly a montage.

This script reports the full distribution plus the cuts an adaptive,
scale-free rule would find, so the replacement threshold is chosen from the
data rather than guessed.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def signature(frame):
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    stride = max(1, int(max(arr.shape[:2]) / 64))
    sample = arr[::stride, ::stride, :3].astype(np.float32)
    return (0.114 * sample[:, :, 0] + 0.587 * sample[:, :, 1] +
            0.299 * sample[:, :, 2]) / 255.0


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
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    cap, shown = open_capture(args.video)
    if not cap.isOpened():
        print(f"cannot open {args.video}")
        return 2
    print(f"clip: {shown}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    diffs = []
    prev = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        sig = signature(frame)
        if sig is not None:
            if prev is not None and prev.shape == sig.shape:
                diffs.append(float(np.mean(np.abs(sig - prev))))
            prev = sig
        idx += 1
        if idx % 2000 == 0:
            print(f"JCODE_PROGRESS {json.dumps({'current': idx, 'total': total, 'unit': 'frames', 'message': 'scanning'})}",
                  flush=True)
    cap.release()

    d = np.asarray(diffs, dtype=np.float64)
    if d.size == 0:
        print("no frames")
        return 2

    print(f"\nframes {idx} @ {fps:.2f}fps, {d.size} adjacent pairs")
    print("\nframe-to-frame luma difference distribution")
    for q in (50, 75, 90, 95, 98, 99, 99.5, 99.9):
        print(f"  p{q:<5} {np.percentile(d, q):.4f}")
    print(f"  max    {d.max():.4f}")
    print(f"\nshipped fixed gate 0.32 fires on {(d >= 0.32).sum()} pair(s)  <-- the bug")

    # Adaptive rule: a cut is a difference that is both absolutely
    # significant and a large multiple of the local median. Scale-free, so it
    # behaves the same on a high-contrast action clip and a soft-graded one.
    win = 61
    half = win // 2
    padded = np.pad(d, (half, half), mode="edge")
    local = np.array([np.median(padded[i:i + win]) for i in range(d.size)])
    ratio = d / np.maximum(local, 1e-4)

    print("\nadaptive rule: diff >= floor AND diff >= k * local median (window 61)")
    for floor in (0.030, 0.045, 0.060):
        for k in (4.0, 6.0, 8.0):
            hits = int(((d >= floor) & (ratio >= k)).sum())
            per_min = hits / max(idx / max(fps, 1e-6) / 60.0, 1e-6)
            print(f"  floor={floor:.3f} k={k:.1f} -> {hits:5d} cuts "
                  f"({per_min:.1f}/min, mean shot {idx / max(hits + 1, 1) / max(fps, 1e-6):.2f}s)")

    floor, k = 0.045, 6.0
    cuts = np.flatnonzero((d >= floor) & (ratio >= k)) + 1
    print(f"\nchosen floor={floor} k={k}: {len(cuts)} cuts")
    if len(cuts):
        shots = np.diff(np.concatenate(([0], cuts, [idx])))
        shots = shots[shots > 0]
        print(f"  shot length frames: min={shots.min()} median={int(np.median(shots))} "
              f"mean={shots.mean():.1f} max={shots.max()}")
        print(f"  shots <= 10 frames (interp gap): {(shots <= 10).sum()}")
        print(f"  shots <= 15 frames (coast limit): {(shots <= 15).sum()}")
        print(f"  first 25 cut frames: {cuts[:25].tolist()}")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"frames": idx, "fps": fps,
                       "cuts": cuts.tolist(),
                       "diff_p99": float(np.percentile(d, 99)),
                       "diff_max": float(d.max())}, fh)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
