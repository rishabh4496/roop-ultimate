"""Stage 16: does optical flow beat geometry-only filling of a detector dropout?

Ground truth is the detector itself on the UNOCCLUDED frames. A run of frames is
"dropped": the filler is not allowed to see their detections. It must place the
106 landmarks from the anchors on either side (and, for flow, the pixels).
Error is mean landmark distance over the inter-ocular distance (NME, %).

Methods, on identical trials:

  hermite   what `_build_temporal_faces` does today between two anchors
            (`ConfidenceAwareInterpolator.interpolate_face`)
  kalman    what `_coast_track_gaps` does today: forward and backward
            `FaceTracker` coast, the younger prediction wins
  flow      LK chained from the leading anchor through the dropped frames; a
            frame LK refuses falls back to KALMAN, so this column never sees the
            closing anchor and is comparable with `kalman`
  flow+br   flow, then the residual to the closing anchor spread linearly over
            the gap (flow supplies the shape of the path, the anchor pins its end);
            falls back to HERMITE, so it is comparable with `hermite`

Conditions: clean, and an occluder pasted over 60% / 85% of the face box on the
dropped frames only. The occluder is a STATIONARY patch (a pole, a wall edge)
copied from elsewhere in the frame, so while the head moves behind it LK sees
two motions -- the hard case. Flow reports how often LK refused a step.

Usage (cwd = app):
    env/Scripts/python.exe tests/probe_flow_stage16.py --video <clip> --frames 400
"""

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _lm(face):
    v = face.get('landmark_2d_106')
    return None if v is None else np.asarray(v, np.float32)[:, :2]


def _iod(face):
    k = np.asarray(face.kps, np.float32)
    return float(np.linalg.norm(k[0] - k[1]))


def nme(pred, truth, iod):
    return float(np.mean(np.linalg.norm(pred - truth, axis=1)) / max(iod, 1e-6) * 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--frames', type=int, default=400)
    ap.add_argument('--every', type=int, default=9, help='trial spacing in frames')
    ap.add_argument('--lengths', default='3,6,10')
    ap.add_argument('--provider', default=None)
    ap.add_argument('--max-side', type=int, default=960,
                    help='downscale frames so the longest side is this (detector sees the same)')
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(',')]

    import two_face_video as tfv          # noqa: E402
    import angle_bench as ab              # noqa: E402
    tfv._apply_startup_runtime_environment()
    import yaml
    with open(os.path.join(APP, 'config.yaml')) as fh:
        cfg = yaml.safe_load(fh) or {}
    provider = args.provider or (cfg.get('provider') or 'cuda')
    ab.init_pipeline(provider, cfg.get('swap_model') or 'inswapper', 'None', 'None',
                     sync_config=True)
    from roop.face_util import get_all_faces
    from roop.flow_coast import FlowCoaster, bridge_gap
    from roop.temporal_state_machine import ConfidenceAwareInterpolator as CAI
    from roop.tracker import FaceTracker

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    gray, det = [], []
    shape = None
    for _ in range(args.frames):
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        s = args.max_side / float(max(h, w))
        if s < 1.0:
            fr = cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        shape = fr.shape
        faces = get_all_faces(fr) or []
        good = [f for f in faces if _lm(f) is not None and f.kps is not None]
        det.append(good[0] if len(faces) == 1 and len(good) == 1 else None)
        gray.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    n = len(gray)
    print(f'[flow] {os.path.basename(args.video)}: {n} frames, '
          f'{sum(d is not None for d in det)} single-face detections, {shape}')

    rng = np.random.default_rng(0)
    results = {}          # (L, cond, method) -> list[nme]
    fallbacks = {}        # (L, cond) -> [n_trials, n_trials_with_fallback]

    def occlude(g, face, frac, side):
        """Paste a stationary patch over `frac` of the face box."""
        b = np.asarray(face.bbox, np.float32)
        x0, y0, x1, y1 = [int(v) for v in b]
        bw, bh = x1 - x0, y1 - y0
        k = int(round(bw * frac))
        if side == 0:
            ox0, ox1 = x0, x0 + k
        else:
            ox0, ox1 = x1 - k, x1
        oy0, oy1 = y0 - 5, y1 + 5
        H, W = g.shape
        ox0, ox1 = max(0, ox0), min(W, ox1)
        oy0, oy1 = max(0, oy0), min(H, oy1)
        out = g.copy()
        # source patch from the far side of the frame (not face-like, textured)
        sx = 5 if ox0 > W // 2 else W - 5 - (ox1 - ox0)
        sy = max(0, min(H - (oy1 - oy0), 5))
        patch = g[sy:sy + (oy1 - oy0), sx:sx + (ox1 - ox0)]
        if patch.shape == out[oy0:oy1, ox0:ox1].shape:
            out[oy0:oy1, ox0:ox1] = patch
        return out

    trials = 0
    for L in lengths:
        for k in range(8, n - L - 2, args.every):
            need = list(range(k - 7, k)) + list(range(k, k + L + 1)) + list(range(k + L + 1, min(n, k + L + 7)))
            if any(det[i] is None for i in need if 0 <= i < n):
                continue
            a, b = det[k - 1], det[k + L]
            gt = {g: _lm(det[g]) for g in range(k, k + L)}
            iod = {g: _iod(det[g]) for g in range(k, k + L)}
            emb = np.asarray(a.embedding, np.float32)
            for cond in ('clean', 'occ60', 'occ85'):
                frac = {'clean': 0.0, 'occ60': 0.6, 'occ85': 0.85}[cond]
                side = int(rng.integers(0, 2))
                frames = {g: (occlude(gray[g], det[g], frac, side) if frac else gray[g])
                          for g in range(k, k + L)}
                # -- hermite
                herm = {}
                for g in range(k, k + L):
                    w = (g - (k - 1)) / float(L + 1)
                    herm[g] = np.asarray(CAI.interpolate_face(a, b, w, emb)['landmark_2d_106'],
                                         np.float32)[:, :2]
                # -- kalman (fwd + bwd, younger wins)
                def run(order, hole):
                    tr = FaceTracker(max_age=30, max_coast=15, min_hits_to_coast=3)
                    found = {}
                    for step, i in enumerate(order):
                        real = None if i in hole else det[i]
                        tr.update([real] if real is not None else [], step)
                        if i in hole:
                            p = tr.coast(step, frame_shape=shape, occupied=[])
                            if p:
                                found[i] = p[0]
                    return found
                hole = set(range(k, k + L))
                fwd = run(list(range(k - 7, k + L + 1)), hole)
                bwd = run(list(range(k + L + 7 if k + L + 7 < n else n - 1, k - 2, -1)), hole)
                kal = {}
                for g in hole:
                    fa, fb = fwd.get(g), bwd.get(g)
                    if fa is None and fb is None:
                        kal[g] = herm[g]
                    else:
                        pick = fa if fb is None or (fa is not None and fa.get('_coast_age', 1)
                                                    <= fb.get('_coast_age', 1)) else fb
                        kal[g] = np.asarray(pick['landmark_2d_106'], np.float32)[:, :2]
                # -- flow
                c = FlowCoaster()
                flow, fell = {}, False
                if c.anchor(gray[k - 1], a.bbox, _lm(a), np.asarray(a.kps, np.float32)):
                    for g in range(k, k + L):
                        r = c.step(frames[g])
                        if r is None or not r.consistent:
                            fell = True
                            break
                        flow[g] = r.landmarks
                    end = None
                    if not fell:
                        r = c.step(gray[k + L])
                        end = r.landmarks if (r is not None and r.consistent) else None
                else:
                    fell = True
                flow_full = {g: flow.get(g, kal[g]) for g in range(k, k + L)}   # anchorless fallback
                bridged = {g: herm[g] for g in range(k, k + L)}
                if not fell and end is not None:
                    path = bridge_gap([flow[g] for g in range(k, k + L)], end,
                                      _lm(b), _iod(b))
                    if path is None:
                        fell = True                 # flow missed the anchor
                    else:
                        bridged = {g: path[g - k] for g in range(k, k + L)}
                tag = (L, cond)
                fb_ = fallbacks.setdefault(tag, [0, 0])
                fb_[0] += 1
                fb_[1] += int(fell or end is None)
                for name, src in (('hermite', herm), ('kalman', kal),
                                  ('flow', flow_full), ('flow+br', bridged)):
                    results.setdefault((L, cond, name), []).extend(
                        nme(src[g], gt[g], iod[g]) for g in range(k, k + L))
                trials += 1
    print(f'[flow] {trials} trials')
    print('\n   L  cond   |   hermite        kalman         flow           flow+bridge   | fallback')
    print('             |  mean  p95      mean  p95      mean  p95      mean  p95     |')
    for L in lengths:
        for cond in ('clean', 'occ60', 'occ85'):
            row = []
            for name in ('hermite', 'kalman', 'flow', 'flow+br'):
                v = results.get((L, cond, name))
                row.append('   n/a        ' if not v else
                           f'{np.mean(v):5.2f} {np.percentile(v, 95):5.2f}   ')
            fb = fallbacks.get((L, cond), [0, 0])
            pct = 100.0 * fb[1] / fb[0] if fb[0] else 0.0
            print(f'  {L:2d}  {cond:6s} | ' + '   '.join(row) + f'| {pct:4.0f}% of {fb[0]}')


if __name__ == '__main__':
    main()
