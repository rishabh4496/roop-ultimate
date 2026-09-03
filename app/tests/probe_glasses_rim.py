"""Can the eyeglass FRAME be separated from the LENS by shape alone?

`Mask_RealityUX` deliberately excludes BiSeNet's glasses class (6) from the
protected set, and the reason recorded there is real: BiSeNet labels the whole
LENS AREA as glasses, including the eye visible through it. Protecting class 6
wholesale means the eye under the lens never gets swapped, so the original
person's eye looks out of the swapped face.

The frame is a THIN structure; the lens is a BROAD fill. A morphological
opening removes structures thinner than its kernel, so `glasses - open(glasses)`
should retain rim, bridge and temple arms while discarding lens interiors.

This probe measures whether that actually separates on real footage:
  rim share      -- how much of class 6 survives as rim
  eye in rim     -- BiSeNet eye/brow pixels caught inside the rim (must be ~0,
                    it is the exact harm the exclusion was written to avoid)
  eye in glasses -- the same for protecting class 6 wholesale (the baseline)

    env/Scripts/python.exe tests/probe_glasses_rim.py \
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

EYE_CLASSES = [2, 3, 4, 5]      # brows + eyes: what must stay swappable


def rim_of(glasses_u8, k):
    """Thin-structure residue: glasses minus its own morphological opening."""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    lens = cv2.morphologyEx(glasses_u8, cv2.MORPH_OPEN, ker)
    return cv2.subtract(glasses_u8, lens), lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True, nargs='+')
    ap.add_argument('--frames', type=int, default=24)
    ap.add_argument('--crop', type=int, default=512)
    ap.add_argument('--dump', default=None)
    args = ap.parse_args()

    from angle_bench import init_pipeline
    from settings import Settings
    cfg = Settings(os.path.join(APP, 'config.yaml'))
    init_pipeline(provider=cfg.provider, swap_model=cfg.swap_model,
                  enhancer=None, mask_engine=cfg.mask_engine, sync_config=True)

    from roop.face_util import get_all_faces, align_crop
    from roop.processors.Mask_FaceParser import Mask_FaceParser
    import roop.globals as g

    prov = str(g.execution_providers).lower()
    parser = Mask_FaceParser()
    parser.Initialize({'devicename': 'cuda' if ('cuda' in prov or 'tensorrt' in prov)
                       else 'cpu'})

    KS = [5, 9, 15, 21, 31]
    tot = {k: {'rim': 0, 'eye_in_rim': 0} for k in KS}
    n_glass = n_eye = 0
    shots = []

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
                labels = parser.RunLabels(crop)
                gl = (labels == 6).astype(np.uint8)
                if gl.sum() < 200:
                    continue
                eye = np.isin(labels, EYE_CLASSES)
                n_glass += int(gl.sum())
                n_eye += int((eye & (gl > 0)).sum())
                for k in KS:
                    rim, _lens = rim_of(gl, k)
                    tot[k]['rim'] += int(rim.sum())
                    tot[k]['eye_in_rim'] += int((eye & (rim > 0)).sum())
                if args.dump and len(shots) < 3:
                    rim, lens = rim_of(gl, 15)
                    vis = cv2.resize(crop, (256, 256))
                    ov = vis.copy()
                    r = cv2.resize(rim * 255, (256, 256))
                    l = cv2.resize(lens * 255, (256, 256))
                    ov[r > 127] = (0, 255, 0)       # rim  -> protect
                    ov[(l > 127) & (r <= 127)] = (0, 0, 255)  # lens -> swap through
                    shots.append(np.hstack([vis, ov]))
        cap.release()

    if n_glass == 0:
        print('no glasses pixels found')
        return 1

    print(f'\nglasses pixels: {n_glass}')
    print(f'eye/brow pixels inside class 6 wholesale: '
          f'{n_eye} ({n_eye / n_glass * 100:.2f}% of class 6)')
    print(f'\n{"open k":>8}{"rim share":>12}{"eye in rim":>13}{"eye/rim %":>12}')
    print('-' * 45)
    for k in KS:
        rim = tot[k]['rim']
        e = tot[k]['eye_in_rim']
        print(f'{k:>8}{rim / n_glass * 100:>11.1f}%{e:>13}'
              f'{(e / rim * 100 if rim else 0):>11.2f}%')

    if args.dump and shots:
        cv2.imwrite(args.dump, np.vstack(shots))
        print(f'\nwrote {args.dump}  (green = rim/protect, red = lens/swap through)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
