"""Calibrate Enhance_UltraMax._CONTENT_TOL against real footage.

The reuse policy is only sound while the crop the anchor was built from still
looks like the crop in front of it. That is an empirical property of the
footage, not something to pick by eye, so this replays real aligned face crops
through the policy at a range of tolerances and reports what each one costs and
what it lets through.

Two numbers matter and they pull against each other:
  cf_rate   -- share of faces that pay for real CodeFormer inference (speed)
  held_p99  -- 99th pct of the per-block luminance gap between the crop the
               output was actually built from and the crop it should have been
               built from (fidelity; this is the frozen-expression error)
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np

import angle_bench as ab
from roop.processors.Enhance_UltraMax import Enhance_UltraMax as UM

TOLS = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0, 1e9]
REFRESH = 4


def _centre(f):
    return (float(f.bbox[0] + f.bbox[2]) * 0.5, float(f.bbox[1] + f.bbox[3]) * 0.5)


def crops_for_clip(path, max_frames):
    """Aligned crops of ONE person, one per frame.

    Nearest-centroid to the previous pick, not largest-in-frame: on a two-person
    clip "largest" alternates between the two people as they move, and every
    alternation reads as a colossal content change that has nothing to do with
    the face moving. The real pipeline keys the cache per track, so the
    calibration has to follow one person too or it measures the wrong thing.
    """
    from roop.face_util import get_all_faces, align_crop
    cap = cv2.VideoCapture(path)
    out, prev = [], None
    while len(out) < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        faces = get_all_faces(frame)
        if not faces:
            out.append(None)
            continue
        if prev is None:
            f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        else:
            f = min(faces, key=lambda x: (_centre(x)[0] - prev[0]) ** 2
                                         + (_centre(x)[1] - prev[1]) ** 2)
        prev = _centre(f)
        crop, _ = align_crop(frame, f.kps, 512, mode='arcface')
        out.append(crop)
    cap.release()
    return out


def replay(sigs, tol, refresh=REFRESH):
    """Run the real policy. Returns (cf_calls, faces, held-error samples)."""
    anchor, age, calls, held = None, 0, 0, []
    for s in sigs:
        if s is None:
            anchor = None
            continue
        due = anchor is None or age >= refresh or float(np.abs(s - anchor).max()) > tol
        if due:
            anchor, age, calls = s, 0, calls + 1
            held.append(0.0)
        else:
            age += 1
            held.append(float(np.abs(s - anchor).max()))
    return calls, len([s for s in sigs if s is not None]), held


def main():
    ab.init_pipeline('tensorrt', 'realswap', 'UltraMax', 'mask_realityux', 0.0)

    clips = [
        (r"G:/pinokio/roop-keep/expression/Face Expression Videos, Download The BEST Free 4k Stock Video Footage & Face Expression HD Video Clips_3.mp4", 400),
        (r"G:/pinokio/roop-keep/duo/d2.mp4", 400),
    ]

    for path, n in clips:
        name = os.path.basename(path)[:44]
        print(f"\n=== {name} ({n} frames) ===")
        crops = crops_for_clip(path, n)
        sigs = [UM._content_sig(c) if c is not None else None for c in crops]
        got = len([s for s in sigs if s is not None])
        print(f"    faces found: {got}/{len(crops)}")
        if got < 20:
            print("    too few faces; skipping")
            continue

        # Frame-to-frame movement, for scale.
        consec = [float(np.abs(sigs[i] - sigs[i - 1]).max())
                  for i in range(1, len(sigs)) if sigs[i] is not None and sigs[i - 1] is not None]
        print(f"    consecutive-frame block delta: median {np.median(consec):.1f}  "
              f"p90 {np.percentile(consec, 90):.1f}  p99 {np.percentile(consec, 99):.1f}  "
              f"max {max(consec):.1f}")

        print(f"    {'tol':>7}{'cf_rate':>10}{'held_mean':>11}{'held_p99':>10}{'held_max':>10}")
        for tol in TOLS:
            calls, faces, held = replay(sigs, tol)
            h = np.asarray(held) if held else np.zeros(1)
            label = 'off' if tol > 1e8 else f"{tol:g}"
            print(f"    {label:>7}{100.0 * calls / faces:>9.1f}%{h.mean():>11.2f}"
                  f"{np.percentile(h, 99):>10.2f}{h.max():>10.2f}")


if __name__ == '__main__':
    main()
