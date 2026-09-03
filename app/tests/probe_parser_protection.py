"""Measure what the RealityUX accessory gate actually does to hair and glasses.

Written because `Mask_RealityUX.Run` gates BiSeNet's non-face subtraction on
`accessory_allowed = clip((xseg_mask - 0.05) / 0.20)`, and `xseg_mask` is the
"keep the original pixel" mask. The parser is therefore permitted to exclude a
pixel only in proportion to how much XSeg was ALREADY excluding it. The
disagreement case -- BiSeNet says hair, XSeg says swap -- is the case the gate
suppresses, and it is the only case that can protect a fringe XSeg missed.

This probe changes nothing. It reports, per protected class, the distribution
of `xseg_mask` underneath it and the share of those pixels that still land in
the swap region -- i.e. the rate at which the pipeline paints over hair and
glasses -- plus how much of that residue the gate is structurally unable to
rescue ("gate-blocked": xseg < 0.05, where permission is exactly zero).

    env/Scripts/python.exe tests/probe_parser_protection.py \
        --video G:/pinokio/roop-keep/single/s4.mp4 --frames 40
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
sys.path.insert(0, APP)
sys.path.insert(0, HERE)

# The classes this task is about, plus those already in _NONFACE_OPAQUE.
WATCH = {17: 'hair/bangs', 6: 'glasses', 18: 'hat', 7: 'ear(l)', 8: 'ear(r)'}


def new_acc():
    return {c: {'xseg': [], 'n': 0, 'painted_xseg': 0, 'painted_fused': 0,
                'blocked': 0, 'resid': []} for c in WATCH}


def table(a_map, title):
    print('\n' + title)
    hdr = (f'{"class":<12}{"pixels":>10}{"xseg p50":>10}{"painted XSeg":>14}'
           f'{"painted fused":>15}{"gate-blocked":>14}{"resid xseg p50":>16}')
    print(hdr)
    print('-' * len(hdr))
    for c, name in WATCH.items():
        a = a_map[c]
        if a['n'] == 0:
            continue
        v = np.concatenate(a['xseg'])
        px = a['painted_xseg'] / a['n'] * 100
        pf = a['painted_fused'] / a['n'] * 100
        bl = a['blocked'] / a['n'] * 100
        rv = (f"{np.percentile(np.concatenate(a['resid']), 50):.3f}"
              if a['resid'] else '-')
        print(f'{name:<12}{a["n"]:>10}{np.percentile(v, 50):>10.3f}'
              f'{px:>13.1f}%{pf:>14.1f}%{bl:>13.1f}%{rv:>16}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True, nargs='+')
    ap.add_argument('--frames', type=int, default=30)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--crop', type=int, default=512)
    args = ap.parse_args()

    from angle_bench import init_pipeline
    from settings import Settings
    cfg = Settings(os.path.join(APP, 'config.yaml'))
    init_pipeline(provider=cfg.provider, swap_model=cfg.swap_model,
                  enhancer=None, mask_engine=cfg.mask_engine,
                  sync_config=True)

    from roop.face_util import get_all_faces, align_crop
    from roop.processors.Mask_XSeg import Mask_XSeg
    from roop.processors.Mask_FaceParser import Mask_FaceParser
    from roop.processors.Mask_RealityUX import _NONFACE_OPAQUE
    import roop.globals as g

    prov = str(g.execution_providers).lower()
    opts = {'devicename': 'cuda' if ('cuda' in prov or 'tensorrt' in prov) else 'cpu'}
    xseg = Mask_XSeg()
    xseg.Initialize(opts)
    parser = Mask_FaceParser()
    parser.Initialize(opts)

    acc = new_acc()
    per_video = {}
    crops = 0

    for vp in args.video:
        cap = cv2.VideoCapture(vp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            print(f'could not open {vp}')
            cap.release()
            continue
        step = max(1, (total - args.start) // max(1, args.frames))
        vacc = new_acc()

        for i in range(args.frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.start + i * step)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            for face in (get_all_faces(frame) or [])[:2]:
                crop = align_crop(frame, face.kps, args.crop)
                if isinstance(crop, tuple):
                    crop = crop[0]
                if crop is None or crop.size == 0:
                    continue
                labels = parser.RunLabels(crop)
                xm = np.asarray(xseg.Run(crop, ''), dtype=np.float32)
                if xm.ndim == 3:
                    xm = xm[..., 0] if xm.shape[-1] == 1 else xm.mean(axis=-1)
                h, w = labels.shape[:2]
                if xm.shape[:2] != (h, w):
                    xm = cv2.resize(xm, (w, h), interpolation=cv2.INTER_LINEAR)

                # Reproduce Mask_RealityUX.Run exactly.
                is_acc = np.isin(labels, _NONFACE_OPAQUE).astype(np.float32)
                is_acc = cv2.GaussianBlur(is_acc, (0, 0), sigmaX=3)
                allowed = np.clip((xm - 0.05) / 0.20, 0.0, 1.0)
                fused = np.maximum(xm, np.clip(is_acc * allowed, 0.0, 1.0))

                for c in WATCH:
                    sel = labels == c
                    n = int(sel.sum())
                    if n == 0:
                        continue
                    still = sel & (fused < 0.5)
                    for a in (acc[c], vacc[c]):
                        a['n'] += n
                        a['xseg'].append(xm[sel])
                        a['painted_xseg'] += int((xm[sel] < 0.5).sum())
                        a['painted_fused'] += int(still.sum())
                        a['blocked'] += int((sel & (xm < 0.05)).sum())
                        if still.any():
                            a['resid'].append(xm[still])
                crops += 1
        cap.release()
        per_video[vp] = vacc

    for vp, vacc in per_video.items():
        table(vacc, os.path.basename(vp))
    print(f'\ncrops graded: {crops}   (crop {args.crop}px, mask space 512)')
    print('"painted" = the mask routes that pixel to the SWAPPED face.')
    table(acc, 'ALL CLIPS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
