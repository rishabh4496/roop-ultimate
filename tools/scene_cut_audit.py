"""Measure scene cuts in the source clip and score the tracking pre-pass
guards against them.

WHY. roop's swap phase resets its stabilisers on a hard cut
(ProcessMgr.process_frame -> _stab_history.observe_frame), but the whole-clip
tracking pre-pass in procmgr_tracking._precompute_tracks only calls
gc.collect() on the same event. Nothing there stops a track, its
interpolated gap-fill, or its Kalman coast from crossing a shot boundary --
so on cut-heavy footage a face can be invented in a shot the person is not
in. This script quantifies that on the real clip instead of assuming it.

Outputs:
  - number of hard cuts and the shot-length distribution
  - how many cuts sit inside a gap short enough for interpolation (<= gap)
    or for coasting (<= coast), i.e. how many are actually bridgeable today
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
    """Same luma thumbnail StreamingStabilizationHistory._signature builds."""
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    stride = max(1, int(max(arr.shape[:2]) / 64))
    sample = arr[::stride, ::stride, :3].astype(np.float32)
    return (0.114 * sample[:, :, 0] + 0.587 * sample[:, :, 1] +
            0.299 * sample[:, :, 2]) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--threshold", type=float, default=0.32,
                    help="same default as StreamingStabilizationHistory")
    ap.add_argument("--gap", type=int, default=10, help="ROOP_TEMPORAL_GAP")
    ap.add_argument("--coast", type=int, default=15, help="ROOP_COAST_FRAMES")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    # The clip names carry emoji, and on Windows cv2.VideoCapture cannot open a
    # path it cannot encode in the ANSI code page. Resolve a glob here so the
    # shell never has to round-trip the name, and fall back to the short (8.3)
    # path for the handle itself.
    if any(ch in args.video for ch in "*?[") or not os.path.exists(args.video):
        matches = sorted(glob.glob(args.video), key=os.path.getmtime, reverse=True)
        if matches:
            args.video = matches[0]
    print(f"clip: {args.video}")

    path = args.video
    if os.name == "nt":
        try:
            import ctypes

            buf = ctypes.create_unicode_buffer(1024)
            if ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 1024):
                if buf.value:
                    path = buf.value
        except Exception:
            pass

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"cannot open {args.video}")
        return 2
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    cuts = []
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
                d = float(np.mean(np.abs(sig - prev)))
                diffs.append(d)
                if d >= args.threshold:
                    cuts.append(idx)
            prev = sig
        idx += 1
        if idx % 1000 == 0:
            print(f"JCODE_PROGRESS {json.dumps({'current': idx, 'total': total, 'unit': 'frames', 'message': 'scanning'})}",
                  flush=True)
    cap.release()

    shots = []
    boundaries = [0] + cuts + [idx]
    for a, b in zip(boundaries, boundaries[1:]):
        if b > a:
            shots.append(b - a)

    print(f"\nframes scanned : {idx} @ {fps:.3f} fps ({idx / max(fps, 1e-6):.1f}s)")
    print(f"hard cuts      : {len(cuts)} at threshold {args.threshold}")
    if shots:
        arr = np.asarray(shots)
        print(f"shots          : {len(shots)}  "
              f"min={arr.min()} p25={int(np.percentile(arr, 25))} "
              f"median={int(np.median(arr))} mean={arr.mean():.1f} max={arr.max()} frames")
        print(f"               : {(arr <= args.gap).sum()} shots are <= the "
              f"{args.gap}-frame interpolation gap limit")
        print(f"               : {(arr <= args.coast).sum()} shots are <= the "
              f"{args.coast}-frame coast limit")
    if diffs:
        d = np.asarray(diffs)
        print(f"frame-diff     : median={np.median(d):.4f} p99={np.percentile(d, 99):.4f} max={d.max():.4f}")

    # The number that matters: a cut is CROSSABLE by gap-fill when the tracker
    # can plausibly lose the face just before it and re-acquire just after,
    # which is exactly what a short shot on either side allows.
    crossable_interp = 0
    crossable_coast = 0
    for k, c in enumerate(cuts):
        before = c - (boundaries[k] if k < len(boundaries) else 0)
        after = (boundaries[k + 2] if k + 2 < len(boundaries) else idx) - c
        if min(before, after) <= args.gap:
            crossable_interp += 1
        if min(before, after) <= args.coast:
            crossable_coast += 1
    print(f"\ncuts with a shot shorter than the interpolation gap on one side : "
          f"{crossable_interp} / {len(cuts)}")
    print(f"cuts with a shot shorter than the coast limit on one side       : "
          f"{crossable_coast} / {len(cuts)}")
    print("\nEvery one of those is a boundary today's pre-pass may bridge: it "
          "detects the cut but only runs gc.collect().")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"frames": idx, "fps": fps, "cuts": cuts,
                       "shots": shots}, fh)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
