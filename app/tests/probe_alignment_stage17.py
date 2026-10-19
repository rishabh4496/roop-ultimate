"""Stage 17: how far does a partly covered face pull the swap crop, and does
visibility weighting put it back?

The ground truth is free: take a real frame, run the real detector on it (clean),
then paste an occluder over part of the face on the SAME frame and run the real
detector again. The clean detection's alignment matrix is the truth; the
occluded detection is what production would have been handed. Nothing is
simulated about the detector's response -- only the occluder is.

Occluders (a flat skin-coloured patch with grain and soft edges, the hard case:
it looks like a face part, unlike a background-coloured one):
    eye     one eye, radius 0.55 x inter-ocular
    mouth   both mouth corners
    half    the left half of the face box

Methods on identical trials (error = mean distance, in 256-px-crop pixels, between
where each method puts the 106 clean landmarks and where the clean alignment does):
    base    estimate_norm on the occluded detection (what production does)
    w5      weighted Umeyama, weights from P_occ sampled at the detected keypoints
    recon   w5 + occluded primary keypoints rebuilt from the previous clean frame
            through the visible dense landmarks (visibility_alignment)
    clamp   recon + KinematicClamp fed by the clean matrices of the two prior frames
    flow    Lucas-Kanade from the previous CLEAN frame over the visible pixels only
            (`roop.flow_coast`, occluder mask passed in) carries the previous
            keypoints onto this frame; the detector's occluded keypoints are not used
    hyb     detector keypoints where visible (w >= 0.5), flow-carried ones where not

P_occ here is the ORACLE (the pasted mask). It is an upper bound on what any
occlusion estimator can deliver; production would have to get it from XSeg/SAM 2.

Usage (cwd = app):
    env/Scripts/python.exe tests/probe_alignment_stage17.py --video <clip> --frames 300
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

CROP = 256


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def occluder(shape, face, kind, rng):
    """(mask float32 0..1, kind label coverage of the face hull)."""
    h, w = shape[:2]
    k = np.asarray(face.kps, np.float32)
    iod = float(np.linalg.norm(k[0] - k[1]))
    m = np.zeros((h, w), np.float32)
    if kind == 'eye':
        c = k[int(rng.integers(0, 2))]
        cv2.ellipse(m, (int(c[0]), int(c[1])), (int(0.55 * iod), int(0.5 * iod)), 0, 0, 360, 1.0, -1)
    elif kind == 'mouth':
        c = (k[3] + k[4]) * 0.5
        cv2.ellipse(m, (int(c[0]), int(c[1])), (int(0.75 * iod), int(0.45 * iod)), 0, 0, 360, 1.0, -1)
    else:
        b = np.asarray(face.bbox, np.float32)
        m[int(b[1]) - 5:int(b[3]) + 5, int(b[0]) - 5:int(max(b[0], (b[0] + b[2]) * 0.5))] = 1.0
    m = cv2.GaussianBlur(m, (0, 0), max(1.5, 0.04 * iod))
    return np.clip(m, 0, 1)


def paste(frame, face, mask, rng):
    b = np.asarray(face.bbox, np.int32)
    x0, y0, x1, y1 = max(0, b[0]), max(0, b[1]), min(frame.shape[1], b[2]), min(frame.shape[0], b[3])
    # skin colour sampled from the lower cheek band of the same face
    ch = frame[y0 + (y1 - y0) * 3 // 5:y1 - (y1 - y0) // 8, x0 + (x1 - x0) // 4:x1 - (x1 - x0) // 4]
    col = np.median(ch.reshape(-1, 3), axis=0) if ch.size else np.array([150, 160, 200])
    patch = np.empty_like(frame, dtype=np.float32)
    patch[:] = col
    patch += rng.normal(0, 6.0, frame.shape).astype(np.float32)
    patch = cv2.GaussianBlur(patch, (0, 0), 1.2)
    out = frame.astype(np.float32) * (1 - mask[..., None]) + patch * mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--every', type=int, default=3)
    ap.add_argument('--max-side', type=int, default=960)
    ap.add_argument('--provider', default=None)
    args = ap.parse_args()

    import two_face_video as tfv          # noqa: E402
    import angle_bench as ab              # noqa: E402
    tfv._apply_startup_runtime_environment()
    import yaml
    cfg = yaml.safe_load(open(os.path.join(APP, 'config.yaml'))) or {}
    ab.init_pipeline(args.provider or cfg.get('provider') or 'cuda',
                     cfg.get('swap_model') or 'inswapper', 'None', 'None', sync_config=True)
    from roop.face_util import get_all_faces, estimate_norm, swap_template_points
    from roop.visibility_alignment import (KinematicClamp, Observation, apply, decompose,
                                           reconstruct_keypoints, sample_probability,
                                           solve_alignment, visibility_weights)

    template = swap_template_points(CROP, 'arcface')
    from roop.flow_coast import FlowCoaster
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    frames, faces = [], []
    for _ in range(args.frames):
        ok, fr = cap.read()
        if not ok:
            break
        s = args.max_side / float(max(fr.shape[:2]))
        if s < 1.0:
            fr = cv2.resize(fr, (int(fr.shape[1] * s), int(fr.shape[0] * s)), interpolation=cv2.INTER_AREA)
        fs = get_all_faces(fr) or []
        good = [f for f in fs if getattr(f, 'landmark_2d_106', None) is not None]
        frames.append(fr)
        faces.append(good[0] if len(fs) == 1 and len(good) == 1 else None)
    cap.release()
    print(f'[align] {os.path.basename(args.video)}: {len(frames)} frames, '
          f'{sum(f is not None for f in faces)} clean single-face')

    rng = np.random.default_rng(3)
    methods = ('base', 'w5', 'recon', 'clamp', 'flow', 'hyb')
    res = {}      # (kind, method) -> list of errors
    scale_err, rot_err = {}, {}
    miss = {}
    flow_fail = {}
    sanity = []
    for t in range(2, len(frames), args.every):
        if any(faces[i] is None for i in (t - 2, t - 1, t)):
            continue
        truth = faces[t]
        M_ref = estimate_norm(np.asarray(truth.kps, np.float32), CROP, 'arcface')
        lm_true = np.asarray(truth.landmark_2d_106, np.float32)[:, :2]
        prior = Observation(np.asarray(faces[t - 1].kps, np.float32),
                            np.asarray(faces[t - 1].landmark_2d_106, np.float32)[:, :2])
        for kind in ('eye', 'mouth', 'half'):
            mask = occluder(frames[t].shape, truth, kind, rng)
            occ_frame = paste(frames[t], truth, mask, rng)
            fs = get_all_faces(occ_frame) or []
            best = max(fs, key=lambda f: _iou(f.bbox, truth.bbox), default=None)
            if best is None or _iou(best.bbox, truth.bbox) < 0.3 or getattr(best, 'landmark_2d_106', None) is None:
                miss[kind] = miss.get(kind, 0) + 1
                continue
            kps = np.asarray(best.kps, np.float32)
            lm = np.asarray(best.landmark_2d_106, np.float32)[:, :2]
            w_k = visibility_weights(sample_probability(mask, kps))
            w_l = visibility_weights(sample_probability(mask, lm))
            occ_frac = float((sample_probability(mask, lm) > 0.5).mean())

            ests = {}
            ests['base'] = estimate_norm(kps, CROP, 'arcface')
            sanity.append(float(np.abs(ests['base'] - solve_alignment(kps, np.ones(5), template)).max()))
            ests['w5'] = solve_alignment(kps, w_k, template)
            k2, w2, n = reconstruct_keypoints(kps, w_k, lm, w_l, prior)
            ests['recon'] = solve_alignment(k2, w2, template)
            clamp = KinematicClamp(crop=float(CROP))
            for j in (t - 2, t - 1):
                clamp.update(estimate_norm(np.asarray(faces[j].kps, np.float32), CROP, 'arcface'), 0.0)
            ests['clamp'] = clamp.update(ests['recon'], occ_frac)
            # flow from the previous clean frame, visible pixels only
            gp = cv2.cvtColor(frames[t - 1], cv2.COLOR_BGR2GRAY)
            go = cv2.cvtColor(occ_frame, cv2.COLOR_BGR2GRAY)
            fc = FlowCoaster()
            kps_flow = None
            if fc.anchor(gp, faces[t - 1].bbox, prior.landmarks, prior.kps):
                r = fc.step(go, occluder=(mask > 0.5).astype(np.uint8))
                if r is not None and r.consistent and r.kps is not None:
                    kps_flow = np.asarray(r.kps, np.float32)
            if kps_flow is not None:
                ests['flow'] = estimate_norm(kps_flow, CROP, 'arcface')
                hyb = np.where((w_k >= 0.5)[:, None], kps, kps_flow)
                ests['hyb'] = estimate_norm(hyb, CROP, 'arcface')
            else:
                flow_fail[kind] = flow_fail.get(kind, 0) + 1
            for name in methods:
                M = ests.get(name)
                if M is None:
                    continue
                e = float(np.linalg.norm(apply(M, lm_true) - apply(M_ref, lm_true), axis=1).mean())
                res.setdefault((kind, name), []).append(e)
                if name in ('base', 'recon'):
                    s_e, r_e, _ = decompose(M)
                    s_r, r_r, _ = decompose(M_ref)
                    scale_err.setdefault((kind, name), []).append(abs(s_e / s_r - 1.0) * 100)
                    rot_err.setdefault((kind, name), []).append(abs(r_e - r_r))

    print(f'[align] solve_alignment(w=1) vs estimate_norm: max |dM| = {max(sanity):.2e}')
    print('\n   occluder | n   | miss |   mean err (p95) px @256 crop   |  >3 px     | scale err % | rot err deg')
    print('            |     |      |   base         w5       recon      clamp |  base recon | base  recon | base  recon')
    for kind in ('eye', 'mouth', 'half'):
        n = len(res.get((kind, 'base'), []))
        if not n:
            print(f'   {kind:8s} | no trials (miss {miss.get(kind, 0)})')
            continue
        cells = []
        for name in methods:
            v = np.array(res.get((kind, name), [np.nan]))
            cells.append(f'{np.nanmean(v):5.2f}({np.nanpercentile(v, 95):5.2f})')
        print(f'   {kind:8s} | {n:3d} | {miss.get(kind, 0):4d} | ' + ' '.join(cells)
              + f'   (flow n={len(res.get((kind, "flow"), []))})')


if __name__ == '__main__':
    main()
