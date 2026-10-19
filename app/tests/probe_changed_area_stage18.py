"""Stage 18: does the swapped region of the OUTPUT hold still when something crosses it?

For each render, per frame, inside the target's plate face hull (the hull is dilated
so a swap that spills past it still counts): the area where the output differs from
the plate (mean-channel |out - plate| > thr), as a fraction of the hull. The series is
what a viewer sees as the swap "jumping" between covering and not covering a region.

  jump       frames where that area moves by more than `--jump` of the hull between
             consecutive frames (a macro switch, not an edge wobble)
  jitter     mean and p95 of |d area| frame to frame
  partial    share of the changed area whose difference is only a PARTIAL blend
             (thr < diff < 3 x thr): the semi-transparent "ghost" look

Read arm against arm on the SAME clip, same flags, one thing changed.

Usage (cwd = app):
    env/Scripts/python.exe tests/probe_changed_area_stage18.py --plate <clip.mp4> \
        --det <plate_det.npz> --render <a.mp4> --render <b.mp4>
"""
import argparse
import os

import cv2
import numpy as np


def hull(shape, pts, grow):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(np.asarray(pts, np.float32)).astype(np.int32)), 255)
    if grow > 0:
        m = cv2.dilate(m, np.ones((2 * grow + 1,) * 2, np.uint8))
    return m > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plate', required=True)
    ap.add_argument('--det', required=True)
    ap.add_argument('--render', action='append', required=True)
    ap.add_argument('--thr', type=float, default=12.0)
    ap.add_argument('--jump', type=float, default=0.15)
    ap.add_argument('--grow', type=int, default=12)
    ap.add_argument('--lo', type=int, default=0)
    ap.add_argument('--hi', type=int, default=10 ** 9)
    args = ap.parse_args()
    det = list(np.load(args.det, allow_pickle=True)['det'])
    plate = cv2.VideoCapture(args.plate)
    caps = [cv2.VideoCapture(r) for r in args.render]
    series = [[] for _ in caps]
    partial = [[] for _ in caps]
    frames = 0
    k = 0
    while k < len(det):
        ok, pf = plate.read()
        outs = [c.read() for c in caps]
        if not ok or not all(o[0] for o in outs):
            break
        if args.lo <= k < args.hi and det[k]:
            f = sorted(det[k], key=lambda f: f[0][0] + f[0][2])[0]       # the left face is the target
            h = hull(pf.shape, f[1], args.grow)
            for i, (_, of) in enumerate(outs):
                d = np.abs(of.astype(np.int16) - pf.astype(np.int16)).mean(2)
                ch = (d > args.thr) & h
                series[i].append((k, ch.sum() / max(1, h.sum())))
                if ch.sum() > 50:
                    partial[i].append(((d > args.thr) & (d < 3 * args.thr) & h).sum() / ch.sum())
        k += 1
    for i, path in enumerate(args.render):
        s = series[i]
        ks = np.array([a for a, _ in s])
        v = np.array([b for _, b in s])
        dv = np.abs(np.diff(v))[np.diff(ks) == 1]
        print(f'{os.path.basename(os.path.dirname(os.path.dirname(path))):22s} '
              f'frames {len(v):3d}  changed-area mean {100*v.mean():5.1f}%  '
              f'|d area| mean {100*dv.mean():5.2f}% p95 {100*np.percentile(dv, 95):5.2f}%  '
              f'jumps(>{100*args.jump:.0f}%): {int((dv > args.jump).sum()):3d}   '
              f'partial-blend share {100*np.mean(partial[i]):4.1f}%')


if __name__ == '__main__':
    main()
