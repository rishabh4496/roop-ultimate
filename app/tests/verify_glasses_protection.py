"""Before/after on real footage, through the SHIPPED engine.

`probe_parser_protection.py` answers "what does the accessory gate read". This
answers a different question -- "did the change actually reach the render" --
and it answers it by calling `Mask_RealityUX.Run` itself rather than by
reproducing the fusion inline, because a harness that reimplements the thing it
grades stops grading it the moment the two drift apart.

The two arms are the same engine with `glasses_frame_protect` toggled, so the
OFF arm is the pre-change behaviour by construction. That is checked, not
assumed: `--selfcheck` asserts the OFF arm is bit-identical to an independent
reproduction of the old fusion.

    env/Scripts/python.exe tests/verify_glasses_protection.py \
        --video G:/pinokio/roop-keep/single/s5.mp4 --frames 24 --dump out.png
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

GLASSES, HAIR = 6, 17


def old_fusion(xseg, labels):
    """Independent reproduction of the pre-change fusion, for the self-check."""
    from roop.processors.Mask_RealityUX import _NONFACE_OPAQUE
    acc = cv2.GaussianBlur(np.isin(labels, _NONFACE_OPAQUE).astype(np.float32),
                           (0, 0), sigmaX=3)
    allowed = np.clip((xseg - 0.05) / 0.20, 0.0, 1.0)
    return np.maximum(xseg, np.clip(acc * allowed, 0.0, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True, nargs='+')
    ap.add_argument('--frames', type=int, default=24)
    ap.add_argument('--crop', type=int, default=512)
    ap.add_argument('--dump', default=None)
    ap.add_argument('--selfcheck', action='store_true', default=True)
    args = ap.parse_args()

    from angle_bench import init_pipeline
    from settings import Settings
    cfg = Settings(os.path.join(APP, 'config.yaml'))
    init_pipeline(provider=cfg.provider, swap_model=cfg.swap_model,
                  enhancer=None, mask_engine=cfg.mask_engine, sync_config=True)

    from roop.face_util import get_all_faces, align_crop
    from roop.processors.Mask_RealityUX import Mask_RealityUX
    import roop.globals as g

    prov = str(g.execution_providers).lower()
    eng = Mask_RealityUX()
    eng.Initialize({'devicename': 'cuda' if ('cuda' in prov or 'tensorrt' in prov)
                    else 'cpu'})

    stats = {c: {'n': 0, 'off': 0, 'on': 0} for c in (GLASSES, HAIR)}
    selfcheck_max = 0.0
    shots = []
    crops = 0

    for vp in args.video:
        cap = cv2.VideoCapture(vp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            print(f'could not open {vp}')
            cap.release()
            continue
        step = max(1, total // max(1, args.frames))
        for i in range(args.frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            for face in (get_all_faces(frame) or [])[:1]:
                crop = align_crop(frame, face.kps, args.crop)
                if isinstance(crop, tuple):
                    crop = crop[0]
                if crop is None or crop.size == 0:
                    continue

                g.glasses_frame_protect = False
                off = np.asarray(eng.Run(crop, ''), dtype=np.float32).copy()
                g.glasses_frame_protect = True
                on = np.asarray(eng.Run(crop, ''), dtype=np.float32).copy()
                labels = eng._parser.RunLabels(crop)

                if args.selfcheck:
                    xm = cv2.resize(
                        np.asarray(eng._xseg.Run(crop, ''), np.float32).squeeze(),
                        off.shape[::-1], interpolation=cv2.INTER_LINEAR)
                    selfcheck_max = max(
                        selfcheck_max,
                        float(np.abs(off - old_fusion(xm, labels)).max()))

                for c in (GLASSES, HAIR):
                    sel = labels == c
                    n = int(sel.sum())
                    if not n:
                        continue
                    stats[c]['n'] += n
                    stats[c]['off'] += int((off[sel] < 0.5).sum())
                    stats[c]['on'] += int((on[sel] < 0.5).sum())

                if args.dump and (labels == GLASSES).sum() > 2000 and len(shots) < 3:
                    v = cv2.resize(crop, (256, 256))
                    def panel(m, tint):
                        o = v.copy()
                        o[cv2.resize(m, (256, 256)) > 0.5] = tint
                        return o
                    shots.append(np.hstack([
                        v, panel(off, (0, 0, 255)), panel(on, (0, 255, 0))]))
                crops += 1
        cap.release()

    print(f'\ncrops graded: {crops}')
    if args.selfcheck:
        verdict = 'OK' if selfcheck_max == 0.0 else 'DIVERGED'
        print(f'selfcheck: OFF arm vs independent old-fusion reproduction, '
              f'max |diff| = {selfcheck_max:.6f}  [{verdict}]')

    print(f'\n{"class":<10}{"pixels":>10}{"painted BEFORE":>16}'
          f'{"painted AFTER":>15}{"change":>10}')
    print('-' * 61)
    for c, name in ((GLASSES, 'glasses'), (HAIR, 'hair')):
        s = stats[c]
        if not s['n']:
            continue
        b = s['off'] / s['n'] * 100
        a = s['on'] / s['n'] * 100
        print(f'{name:<10}{s["n"]:>10}{b:>15.1f}%{a:>14.1f}%{a - b:>9.1f}%')

    if args.dump and shots:
        cv2.imwrite(args.dump, np.vstack(shots))
        print(f'\nwrote {args.dump}  '
              '(left: crop | middle RED: kept before | right GREEN: kept after)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
