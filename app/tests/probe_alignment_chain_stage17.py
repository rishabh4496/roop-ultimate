"""Stage 17: alignment through a MULTI-FRAME, MOVING occluder, and without an oracle.

`probe_alignment_stage17.py` occludes one frame and gives flow the previous frame
clean plus the exact occluder mask. Production has neither luxury: an occluder is
present for several frames before it is a problem, and nothing hands the aligner
a mask. This closes both gaps.

For each trial a skin-coloured occluder slides in over K frames and ends covering
an eye / the mouth / half the face at frame t. Frame t-K is the last clean one
(the flow anchor); the detector runs on every occluded frame. Ground truth is the
clean detection at frame t. Arms, error in 256-px-crop pixels:

    base       estimate_norm on the detector's occluded keypoints (production)
    flow       chain LK from the clean anchor through the occluded frames, with the
               exact occluder mask excluded at each step (oracle)
    flow_nm    the same chain with NO mask: only LK's own rejection (forward-
               backward, patch error, competing motions) protects it
    flow_nm+d  flow_nm, but a chain LK refuses falls back to the detector's keypoints

Usage (cwd = app):
    env/Scripts/python.exe tests/probe_alignment_chain_stage17.py --video <clip> --chain 6
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
from probe_alignment_stage17 import _iou, occluder, paste, CROP   # noqa: E402


def shifted(mask, dx):
    m = np.zeros_like(mask)
    h, w = mask.shape
    if dx >= 0:
        m[:, dx:] = mask[:, :w - dx] if dx < w else 0
    else:
        m[:, :w + dx] = mask[:, -dx:]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--chain', type=int, default=6)
    ap.add_argument('--every', type=int, default=7)
    ap.add_argument('--max-side', type=int, default=960)
    ap.add_argument('--provider', default=None)
    args = ap.parse_args()
    K = args.chain

    import two_face_video as tfv
    import angle_bench as ab
    tfv._apply_startup_runtime_environment()
    import yaml
    cfg = yaml.safe_load(open(os.path.join(APP, 'config.yaml'))) or {}
    ab.init_pipeline(args.provider or cfg.get('provider') or 'cuda',
                     cfg.get('swap_model') or 'inswapper', 'None', 'None', sync_config=True)
    from roop.face_util import get_all_faces, estimate_norm
    from roop.flow_coast import FlowCoaster
    from roop.visibility_alignment import apply

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
    print(f'[chain] {os.path.basename(args.video)} K={K}: {len(frames)} frames, '
          f'{sum(f is not None for f in faces)} clean single-face')

    rng = np.random.default_rng(5)
    GATES = (0.05, 0.10, 0.20)
    TH = (0.5, 0.6, 0.7, 0.8)
    arms = ('base', 'flow', 'flow_nm', 'flow_nm+d') + tuple(f'gate{g:.2f}' for g in GATES) + tuple(f'surv{x:.1f}' for x in TH)
    res, miss, refused = {}, {}, {}
    discs = {}
    clean_disc, clean_refused = [], [0]
    clean_surv = []
    surv = {}
    TH = (0.5, 0.6, 0.7, 0.8)
    for t in range(K, len(frames), args.every):
        if any(faces[i] is None for i in range(t - K, t + 1)):
            continue
        truth = faces[t]
        M_ref = estimate_norm(np.asarray(truth.kps, np.float32), CROP, 'arcface')
        lm_true = np.asarray(truth.landmark_2d_106, np.float32)[:, :2]
        iod = float(np.linalg.norm(np.asarray(truth.kps)[0] - np.asarray(truth.kps)[1]))
        # CLEAN control: the same chain with nothing pasted. The gate must stay quiet
        # here, or it swaps a good detection for a (noisier) flow estimate.
        ga0 = cv2.cvtColor(frames[t - K], cv2.COLOR_BGR2GRAY)
        fc0 = FlowCoaster()
        a0 = faces[t - K]
        if fc0.anchor(ga0, a0.bbox, np.asarray(a0.landmark_2d_106, np.float32)[:, :2],
                      np.asarray(a0.kps, np.float32)):
            r0 = None
            for j in range(t - K + 1, t + 1):
                r0 = fc0.step(cv2.cvtColor(frames[j], cv2.COLOR_BGR2GRAY))
                if r0 is None or not r0.consistent:
                    r0 = None
                    break
            if r0 is not None and r0.kps is not None:
                clean_surv.append(r0.survival)
                clean_disc.append(float(np.mean(np.linalg.norm(
                    np.asarray(truth.kps, np.float32) - r0.kps, axis=1))) / max(iod, 1e-6))
            else:
                clean_refused[0] += 1
        for kind in ('eye', 'mouth', 'half'):
            final = occluder(frames[t].shape, truth, kind, rng)
            occ_frames, masks = {}, {}
            for j in range(t - K + 1, t + 1):
                if kind != 'half':
                    m = shifted(final, int(round((t - j) * 0.35 * iod)))
                else:
                    m = final * min(1.0, (j - (t - K)) / K)
                masks[j] = np.clip(m, 0, 1)
                occ_frames[j] = paste(frames[j], faces[j], masks[j], rng)
            fs = get_all_faces(occ_frames[t]) or []
            best = max(fs, key=lambda f: _iou(f.bbox, truth.bbox), default=None)
            det_ok = (best is not None and _iou(best.bbox, truth.bbox) >= 0.3
                      and getattr(best, 'kps', None) is not None)
            if not det_ok:
                miss[kind] = miss.get(kind, 0) + 1
            ests = {}
            if det_ok:
                ests['base'] = estimate_norm(np.asarray(best.kps, np.float32), CROP, 'arcface')
            a = faces[t - K]
            kps_by_arm = {}
            last_survival = 1.0
            ga = cv2.cvtColor(frames[t - K], cv2.COLOR_BGR2GRAY)
            for name, use_mask in (('flow', True), ('flow_nm', False)):
                fc = FlowCoaster()
                kps_f = None
                if fc.anchor(ga, a.bbox, np.asarray(a.landmark_2d_106, np.float32)[:, :2],
                             np.asarray(a.kps, np.float32)):
                    r = None
                    for j in range(t - K + 1, t + 1):
                        g = cv2.cvtColor(occ_frames[j], cv2.COLOR_BGR2GRAY)
                        occ = (masks[j] > 0.5).astype(np.uint8) if use_mask else None
                        r = fc.step(g, occluder=occ)
                        if r is None or not r.consistent:
                            r = None
                            break
                    if r is not None and r.kps is not None:
                        kps_f = np.asarray(r.kps, np.float32)
                        if not use_mask:
                            surv.setdefault(kind, []).append(r.survival)
                            last_survival = r.survival
                if kps_f is not None:
                    kps_by_arm[name] = kps_f
                    ests[name] = estimate_norm(kps_f, CROP, 'arcface')
                else:
                    refused[(kind, name)] = refused.get((kind, name), 0) + 1
            # Gate: no ground truth involved. Trust the detector unless the
            # flow-carried keypoints disagree with it by more than tau x IOD, in
            # which case the detector has most likely been dragged by the occluder.
            if det_ok and kps_by_arm.get('flow_nm') is not None:
                disc = float(np.mean(np.linalg.norm(
                    np.asarray(best.kps, np.float32) - kps_by_arm['flow_nm'], axis=1))) / max(iod, 1e-6)
                for g in GATES:
                    ests[f'gate{g:.2f}'] = ests['flow_nm'] if disc > g else ests['base']
                discs.setdefault(kind, []).append(disc)
            elif kps_by_arm.get('flow_nm') is not None:
                for g in GATES:
                    ests[f'gate{g:.2f}'] = ests['flow_nm']        # no detection at all: flow is all there is
            elif det_ok:
                for g in GATES:
                    ests[f'gate{g:.2f}'] = ests['base']
            for x in TH:
                if 'flow_nm' in ests and last_survival < x:
                    ests[f'surv{x:.1f}'] = ests['flow_nm']
                elif 'base' in ests:
                    ests[f'surv{x:.1f}'] = ests['base']
                elif 'flow_nm' in ests:
                    ests[f'surv{x:.1f}'] = ests['flow_nm']
            if 'flow_nm' in ests:
                ests['flow_nm+d'] = ests['flow_nm']
            elif 'base' in ests:
                ests['flow_nm+d'] = ests['base']
            for name in arms:
                if name in ests:
                    e = float(np.linalg.norm(apply(ests[name], lm_true) - apply(M_ref, lm_true), axis=1).mean())
                    res.setdefault((kind, name), []).append(e)

    if clean_disc:
        cd = np.array(clean_disc)
        print(f'[chain] CLEAN control, K={K}: detector-vs-flow keypoint disagreement / IOD  '
              f'median {np.median(cd):.3f}  p90 {np.percentile(cd, 90):.3f}  p99 {np.percentile(cd, 99):.3f}  '
              + '  '.join(f'>{g:.2f}: {100 * np.mean(cd > g):4.1f}%' for g in (0.05, 0.10, 0.20))
              + f'   (n={len(cd)}, chains refused {clean_refused[0]})')
    if clean_surv:
        cs = np.array(clean_surv)
        print(f'[chain] CLEAN survival K={K}: median {np.median(cs):.2f}  p10 {np.percentile(cs, 10):.2f}  '
              + '  '.join(f'<{x:.1f}: {100 * np.mean(cs < x):4.1f}%' for x in TH))
    for kind, v in surv.items():
        a = np.array(v)
        print(f'[chain] {kind:5s} survival: median {np.median(a):.2f}  p10 {np.percentile(a, 10):.2f}  p90 {np.percentile(a, 90):.2f}  '
              + '  '.join(f'<{x:.1f}: {100 * np.mean(a < x):4.1f}%' for x in TH))
    print(f'[chain] LK chains refused (no estimate): {dict(refused)}   detector misses: {miss}')
    print('\n   occluder |  mean err (p95) px @256 crop [n]')
    print('            |   base            flow_nm        surv<0.5       surv<0.6       surv<0.7       surv<0.8')
    for kind in ('eye', 'mouth', 'half'):
        cells = []
        for name in ('base', 'flow_nm', 'surv0.5', 'surv0.6', 'surv0.7', 'surv0.8'):
            v = res.get((kind, name))
            cells.append('   n/a        ' if not v else f'{np.mean(v):5.2f}({np.percentile(v, 95):5.2f})[{len(v)}]')
        print(f'   {kind:8s} | ' + '  '.join(cells))
    for kind, d in discs.items():
        a = np.array(d)
        print(f'   detector-vs-flow keypoint disagreement / IOD, {kind}: median {np.median(a):.3f}  p90 {np.percentile(a, 90):.3f}')


if __name__ == '__main__':
    main()
