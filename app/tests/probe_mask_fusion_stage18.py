"""Stage 18: offline A/B of mask temporal filters on CAPTURED production masks.

`capture_masks_stage18.py` records, per swapped frame and in order, the occluder
model's raw mask (256, what `_ms.apply` receives), the production final mask (after
the existing mask stabilizer) and the aligned crop. Masks sit in the aligned crop,
so frame-to-frame change is mask instability without flow compensation.

Arms (all on the SAME raw input X):
  raw        X as the stabilizer receives it
  prod       the production final mask (existing stabilizer)
  fuse       FlowHysteresisFuser(X)
  fuse+gf    fuse, then the guided matte (r=4, eps=1e-3) on the crop
  gf         guided matte alone on X

Metrics (128x128 grid):
  flip       % of all pixels that cross 0.5 between consecutive frames
  chatter    % of the ever-uncertain band that goes A,B,A within three frames
             (an oscillation, not a transition)
  ghost      % of pixels held OCCLUDED while X is clearly clear (< 0.15) in BOTH
             this and the previous frame: history that outlived the object
  lag        % of pixels held CLEAR while X is clearly occluded (> 0.85) in both:
             an occluder the filter has not caught up with
  edge       mean crop gradient magnitude on the mask's 0.5 contour (higher = the
             edge lies on a real image edge)
Ghost and lag are the price of smoothing; a filter that wins chatter by losing
these has not removed flicker, it has moved it.

Usage (cwd = app):
    env/Scripts/python.exe tests/probe_mask_fusion_stage18.py --pkl masks.pkl
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from roop.mask_fusion import FlowHysteresisFuser, guided_matte      # noqa: E402

G = 128


def small(a):
    return a if a.shape[0] == G else cv2.resize(a, (G, G), interpolation=cv2.INTER_AREA)


def metrics(name, arms, raw, crops):
    A = np.stack([small(a) for a in arms])
    R = np.stack([small(a) for a in raw])
    B = A > 0.5
    unc = ((A > 0.15) & (A < 0.85)).any(0)
    flip = (B[1:] != B[:-1]).mean() * 100
    aba = ((B[:-2] == B[2:]) & (B[:-2] != B[1:-1]))
    chat = aba[:, unc].mean() * 100 if unc.any() else float('nan')
    clear2 = (R[1:] < 0.15) & (R[:-1] < 0.15)
    occl2 = (R[1:] > 0.85) & (R[:-1] > 0.85)
    ghost = (B[1:] & clear2).mean() * 100
    lag = ((~B[1:]) & occl2).mean() * 100
    grads = []
    for a, c in zip(arms, crops):
        g = cv2.cvtColor(cv2.resize(c, (a.shape[1], a.shape[0])), cv2.COLOR_BGR2GRAY).astype(np.float32)
        gm = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        cont = np.abs(a - 0.5) < 0.05
        if cont.sum() > 20:
            grads.append(float(gm[cont].mean()))
    edge = float(np.mean(grads)) if grads else float('nan')
    print(f'  {name:26s} flip {flip:5.2f}%  chatter {chat:5.2f}%  ghost {ghost:5.2f}%  lag {lag:5.2f}%  edge-grad {edge:6.1f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--lo', type=int, default=0)
    ap.add_argument('--hi', type=int, default=10 ** 9)
    ap.add_argument('--sweep', action='store_true')
    args = ap.parse_args()
    items = pickle.load(open(args.pkl, 'rb'))
    oc = [i for i in items if i['raw'].shape == (256, 256) and i['final'] is not None][args.lo:args.hi]
    raw = [i['raw'] for i in oc]
    prod = [i['final'] for i in oc]
    crops = [i['crop'] for i in oc]
    print(f'{len(oc)} frames (target face, in order)')

    def run_fuser(**kw):
        f = FlowHysteresisFuser(**kw)
        return [f.update(r, c, key=0, frame_index=k) for k, (r, c) in enumerate(zip(raw, crops))], f

    # The existing stabilizer, rebuilt as production builds it (strength = config
    # stabilize_mask_strength 0.5, motion_beta 0), once plain and once with its own
    # opt-in flow warp (`mask_flow_warp`, off in config.yaml).
    from roop.one_euro import MaskStabilizer
    kps0 = np.array([[80, 90], [170, 90], [125, 130], [95, 180], [155, 180]], np.float32)

    def run_ms(flow):
        ms = MaskStabilizer(strength=0.5, motion_beta=0.0, flow_warp=flow)
        return [ms.apply(r, kps0, k, guide=c, track_id=0) for k, (r, c) in enumerate(zip(raw, crops))], ms

    ms_plain, _ = run_ms(False)
    ms_flow, msf = run_ms(True)
    print('existing MaskStabilizer reproduces the captured production mask: '
          f'max |diff| {max(float(np.abs(a - b).max()) for a, b in zip(ms_plain, prod)):.4f}; '
          f'flow-warp stats {msf.flow_stats}')
    fused, f0 = run_fuser()
    fused_gf = [guided_matte(m, c, 4, 1e-3) for m, c in zip(fused, crops)]
    gf = [guided_matte(m, c, 4, 1e-3) for m, c in zip(raw, crops)]
    print(f'fuser: applied {f0.applied} seeded {f0.seeded} skipped {f0.skipped}, mean fraction of pixels flow-reset {100*f0.reset_pixels/max(1,f0.applied):.1f}%')
    for name, arm in (('raw (stabilizer input)', raw), ('prod (existing stabilizer)', prod),
                      ('existing + mask_flow_warp', ms_flow),
                      ('fuse', fused), ('fuse + guided', fused_gf), ('guided alone', gf)):
        metrics(name, arm, raw, crops)
    if args.sweep:
        print('\nsweep (alpha, t_high/t_low, sigma):')
        for a in (0.5, 0.7, 0.85):
            for hl in ((0.60, 0.38), (0.55, 0.45), (0.70, 0.30)):
                for sg in (8.0, 14.0, 24.0):
                    out, _ = run_fuser(alpha=a, t_high=hl[0], t_low=hl[1], sigma=sg)
                    metrics(f'a={a} {hl[0]}/{hl[1]} s={sg:.0f}', out, raw, crops)


if __name__ == '__main__':
    main()
