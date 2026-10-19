"""Stage 17: do swapped pixels land on the bystander, and is the target over-protected?

Two faces touch, ONE person is the swap target, the other is a bystander who
must keep their own face. The outcome is asked of the OUTPUT, per frame:

  bleed     fraction of the bystander's face hull (plate, 106-point) whose pixels
            changed between plate and output -- swapped skin on the wrong person
  coverage  fraction of the TARGET's face hull that changed -- a drop here with
            nothing else different is a hole cut into the foreground face
  seam      changed pixels within `--ring` px OUTSIDE the target hull and inside
            the bystander hull (the contested band specifically)

Both are binned by how much the two face boxes overlap (IoU), because the
isolation code only does anything when they do. "Changed" is mean-channel
|out - plate| > `--thr` (the documented render noise floor is 0.71/255 mean, 22
max, so 25 is clear of it).

Faces in the PLATE are found with the production detector and labelled by side
(`--target-side left|right`), which is only valid on a clip where the two people
do not swap sides; Weeds 960-1170 is such a stretch (the woman is the left face
throughout, verified on pixels).

Usage (cwd = app):
    env/Scripts/python.exe tests/probe_isolation_stage17.py --plate <clip> \
        --start 960 --render <output.mp4> --target-side left [--render <arm_b.mp4> ...]
"""

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _hull_mask(shape, pts):
    m = np.zeros(shape[:2], np.uint8)
    h = cv2.convexHull(np.round(np.asarray(pts, np.float32)).astype(np.int32))
    cv2.fillConvexPoly(m, h, 255)
    return m > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plate', required=True)
    ap.add_argument('--start', type=int, required=True, help='plate frame index of render frame 0')
    ap.add_argument('--render', action='append', required=True)
    ap.add_argument('--target-side', choices=('left', 'right'), default='left')
    ap.add_argument('--thr', type=float, default=25.0)
    ap.add_argument('--ring', type=int, default=12)
    ap.add_argument('--provider', default=None)
    ap.add_argument('--cache', default=None, help='npz to save/load plate detections')
    args = ap.parse_args()

    caps = [cv2.VideoCapture(r) for r in args.render]
    n = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps)

    plate = cv2.VideoCapture(args.plate)
    plate.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    det = None
    if args.cache and os.path.exists(args.cache):
        det = list(np.load(args.cache, allow_pickle=True)['det'])
    if det is None:
        import two_face_video as tfv
        import angle_bench as ab
        tfv._apply_startup_runtime_environment()
        import yaml
        cfg = yaml.safe_load(open(os.path.join(APP, 'config.yaml'))) or {}
        ab.init_pipeline(args.provider or cfg.get('provider') or 'cuda',
                         cfg.get('swap_model') or 'inswapper', 'None', 'None', sync_config=True)
        from roop.face_util import get_all_faces
        det = []
        for _ in range(n):
            ok, fr = plate.read()
            if not ok:
                break
            fs = get_all_faces(fr) or []
            det.append([(np.asarray(f.bbox, np.float32),
                         np.asarray(f.landmark_2d_106, np.float32)[:, :2])
                        for f in fs if getattr(f, 'landmark_2d_106', None) is not None])
        if args.cache:
            np.savez_compressed(args.cache, det=np.array(det, dtype=object))
        plate.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    bands = [(0.0, 0.02), (0.02, 0.10), (0.10, 0.30), (0.30, 1.01)]
    acc = [{b: [] for b in bands} for _ in args.render]
    for k in range(min(n, len(det))):
        ok, pf = plate.read()
        outs = []
        for c in caps:
            o, of = c.read()
            outs.append(of if o else None)
        if not ok or any(o is None for o in outs):
            break
        faces = det[k]
        if len(faces) != 2:
            continue
        faces = sorted(faces, key=lambda f: (f[0][0] + f[0][2]) * 0.5)
        t, b = (faces[0], faces[1]) if args.target_side == 'left' else (faces[1], faces[0])
        iou = _iou(t[0], b[0])
        th = _hull_mask(pf.shape, t[1])
        bh = _hull_mask(pf.shape, b[1])
        ring = cv2.dilate(th.astype(np.uint8), np.ones((2 * args.ring + 1,) * 2, np.uint8)) > 0
        contested = ring & ~th & bh
        pure = bh & ~th              # the bystander's face with the target's own hull removed
        for i, of in enumerate(outs):
            if of.shape != pf.shape:
                of = cv2.resize(of, (pf.shape[1], pf.shape[0]))
            ch = np.abs(of.astype(np.int16) - pf.astype(np.int16)).mean(axis=2) > args.thr
            row = (float((ch & bh).sum()) / max(1, bh.sum()),
                   float((ch & th).sum()) / max(1, th.sum()),
                   float((ch & contested).sum()) / max(1, contested.sum()) if contested.any() else float('nan'),
                   float((ch & pure).sum()) / max(1, pure.sum()) if pure.sum() > 50 else float('nan'))
            for band in bands:
                if band[0] <= iou < band[1]:
                    acc[i][band].append(row)
                    break

    for i, path in enumerate(args.render):
        print(f'\n=== {os.path.basename(path)}')
        print('  box IoU band |  n  | bleed on bystander (mean / p95 / worst) | target coverage (mean / min) | contested-band bleed | PURE bleed (bystander hull minus target hull)')
        for band in bands:
            rows = acc[i][band]
            if not rows:
                continue
            a = np.array(rows, np.float32)
            cb = a[:, 2]
            cb = cb[np.isfinite(cb)]
            print(f'  [{band[0]:.2f},{band[1]:.2f}) | {len(a):3d} | '
                  f'{100*a[:,0].mean():5.1f}% / {100*np.percentile(a[:,0],95):5.1f}% / {100*a[:,0].max():5.1f}%'
                  f'        | {100*a[:,1].mean():5.1f}% / {100*a[:,1].min():5.1f}%'
                  f'         | {(100*cb.mean() if len(cb) else float("nan")):5.1f}%'
                  f'         | {(100*np.nanmean(a[:,3]) if np.isfinite(a[:,3]).any() else float("nan")):5.1f}% mean / {(100*np.nanmax(a[:,3]) if np.isfinite(a[:,3]).any() else float("nan")):5.1f}% worst')


if __name__ == '__main__':
    main()
