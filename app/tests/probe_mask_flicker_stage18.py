"""Stage 18: how much does the swapped face flicker when something crosses it?

Three symptoms, each asked of the OUTPUT against the PLATE (the untouched source
footage), inside the target's face hull:

  chatter   motion-compensated flip rate of the "changed" map along its own edge.
            C_t = mean-channel |out - plate| > thr. Frame t-1's C is warped to
            frame t with dense flow computed on the PLATE (so legitimate boundary
            motion is explained) and compared with C_t in a band around the edge.
            What is left is the mask boundary changing its mind, which is the
            "oscillates across the cutoff" the brief names.
  boil      excess temporal difference at that edge: |out_t - warp(out_{t-1})| minus
            the plate's own |plate_t - warp(plate_{t-1})| in the same band.
  flash     frame-to-frame jump of the mean Lab of the swapped interior (the changed
            region eroded away from its edge), high-passed against a 5-frame mean
            and with the plate's own jump over the same pixels subtracted.

Run on arms that differ in ONE thing (enhancer on/off, mask stabilizer on/off) and
read the difference, not the absolute number.

Usage (cwd = app):
    env/Scripts/python.exe tests/probe_mask_flicker_stage18.py --plate <clip.mp4> \
        --det <plate_det.npz> --render <armA.mp4> --render <armB.mp4> [--lo 10 --hi 80]
"""

import argparse
import os
import sys

import cv2
import numpy as np


def _hull(shape, pts):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(np.asarray(pts, np.float32)).astype(np.int32)), 255)
    return m > 0


def _flow(prev_gray, cur_gray):
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return dis.calc(prev_gray, cur_gray, None)


def _warp(img, flow):
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plate', required=True)
    ap.add_argument('--det', required=True, help='npz from probe_isolation_stage17 --cache')
    ap.add_argument('--render', action='append', required=True)
    ap.add_argument('--target-side', default='left')
    ap.add_argument('--thr', type=float, default=12.0)
    ap.add_argument('--band', type=int, default=5)
    ap.add_argument('--lo', type=int, default=0)
    ap.add_argument('--hi', type=int, default=10 ** 9)
    args = ap.parse_args()

    det = list(np.load(args.det, allow_pickle=True)['det'])
    plate = cv2.VideoCapture(args.plate)
    caps = [cv2.VideoCapture(r) for r in args.render]
    n = min(len(det), min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps),
            int(plate.get(cv2.CAP_PROP_FRAME_COUNT)))
    P, O = [], [[] for _ in caps]
    for _ in range(n):
        ok, f = plate.read()
        outs = [c.read() for c in caps]
        if not ok or not all(o[0] for o in outs):
            break
        P.append(f)
        for i, o in enumerate(outs):
            O[i].append(o[1])
    n = len(P)

    def target_hull(k):
        fs = det[k]
        if len(fs) == 0:
            return None
        fs = sorted(fs, key=lambda f: f[0][0] + f[0][2])
        f = fs[0] if args.target_side == 'left' else fs[-1]
        return _hull(P[k].shape, f[1])

    k5 = np.ones((2 * args.band + 1,) * 2, np.uint8)
    for ai, path in enumerate(args.render):
        chat, boil, flash_series, plate_series = [], [], [], []
        prev = None
        for k in range(max(1, args.lo), min(n, args.hi)):
            h0, h1 = target_hull(k - 1), target_hull(k)
            if h0 is None or h1 is None:
                prev = None
                continue
            hull = h0 & h1
            gp0, gp1 = cv2.cvtColor(P[k - 1], cv2.COLOR_BGR2GRAY), cv2.cvtColor(P[k], cv2.COLOR_BGR2GRAY)
            fl = _flow(gp1, gp0)                        # backward flow: samples frame k-1 at frame k's pixels
            C1 = (np.abs(O[ai][k].astype(np.int16) - P[k].astype(np.int16)).mean(2) > args.thr)
            C0 = (np.abs(O[ai][k - 1].astype(np.int16) - P[k - 1].astype(np.int16)).mean(2) > args.thr)
            C0w = _warp(C0.astype(np.float32), fl) > 0.5
            edge = ((cv2.dilate(C1.astype(np.uint8), k5) > 0) & ~(cv2.erode(C1.astype(np.uint8), k5) > 0)) & hull
            if edge.sum() >= 200:
                chat.append(float((C1 != C0w)[edge].mean()))
                ow = _warp(O[ai][k - 1].astype(np.float32), fl)
                pw = _warp(P[k - 1].astype(np.float32), fl)
                d_o = np.abs(O[ai][k].astype(np.float32) - ow).mean(2)[edge].mean()
                d_p = np.abs(P[k].astype(np.float32) - pw).mean(2)[edge].mean()
                boil.append(float(d_o - d_p))
            inner = cv2.erode(C1.astype(np.uint8), np.ones((4 * args.band + 1,) * 2, np.uint8)) > 0
            inner &= hull
            if inner.sum() >= 200:
                lab_o = cv2.cvtColor(O[ai][k], cv2.COLOR_BGR2LAB).astype(np.float32)[inner].mean(0)
                lab_p = cv2.cvtColor(P[k], cv2.COLOR_BGR2LAB).astype(np.float32)[inner].mean(0)
                flash_series.append((k, lab_o))
                plate_series.append((k, lab_p))
        # high-pass the swapped-interior colour against a 5-frame mean, minus the plate's own
        def highpass(series):
            v = np.array([s[1] for s in series])
            if len(v) < 6:
                return np.zeros(0)
            hp = np.array([np.abs(v[i] - np.mean(v[max(0, i - 2):i + 3], axis=0)) for i in range(len(v))])
            return hp
        ho, hp_ = highpass(flash_series), highpass(plate_series)
        ch = np.array(chat)
        print(f'\n=== {os.path.basename(path)}   frames {args.lo}..{min(n, args.hi)}')
        if len(ch):
            print(f'  chatter (MC edge flip rate)   mean {100*ch.mean():5.2f}%  p95 {100*np.percentile(ch, 95):5.2f}%  n={len(ch)}')
        if boil:
            b = np.array(boil)
            print(f'  boil (excess edge |dI|)       mean {b.mean():5.2f}  p95 {np.percentile(b, 95):5.2f}')
        if len(ho):
            ex = np.maximum(ho - hp_, 0)
            ks = [x[0] for x in flash_series]
            top = np.argsort(-ex[:, 0])[:3]
            print('  largest L flashes at frames ' + ', '.join(f'{ks[i]} (+{ex[i, 0]:.1f})' for i in top))
            print(f'  flash (Lab high-pass, excess) L mean {ex[:, 0].mean():4.2f} max {ex[:, 0].max():4.2f} | '
                  f'a {ex[:, 1].mean():4.2f} max {ex[:, 1].max():4.2f} | b {ex[:, 2].mean():4.2f} max {ex[:, 2].max():4.2f}'
                  f'   (swapped-interior frames: {len(ho)})')


if __name__ == '__main__':
    main()
