"""Do the eyes land where `swap_template_points` says, in the mask crop?

A glasses-frame protection has to keep the LENS-OVER-EYE swappable while
protecting rim/bridge/arms. BiSeNet cannot help locate the eye there: it labels
the eye behind a lens as glasses, so the eye classes are empty exactly where the
answer is needed (measured: 0 eye pixels inside class 6). The socket therefore
has to come from crop GEOMETRY.

`align_crop` fits the 5 keypoints to a fixed template, so the eyes land at a
known place. That is an assumption worth measuring rather than trusting -- the
`swap_template_points` docstring records a since-removed feature built on the
obvious-but-wrong `arcface_dst * size/112` guess.

Measures, over faces WITHOUT glasses (so the eye classes are real): the
centroid of each parsed eye against the template position, in 512-mask-space
pixels, plus the eye's own radius so a socket can be sized from data.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True, nargs='+')
    ap.add_argument('--frames', type=int, default=20)
    ap.add_argument('--crop', type=int, default=512)
    args = ap.parse_args()

    from angle_bench import init_pipeline
    from settings import Settings
    cfg = Settings(os.path.join(APP, 'config.yaml'))
    init_pipeline(provider=cfg.provider, swap_model=cfg.swap_model,
                  enhancer=None, mask_engine=cfg.mask_engine, sync_config=True)

    from roop.face_util import get_all_faces, align_crop, swap_template_points
    from roop.processors.Mask_FaceParser import Mask_FaceParser
    import roop.globals as g

    prov = str(g.execution_providers).lower()
    parser = Mask_FaceParser()
    parser.Initialize({'devicename': 'cuda' if ('cuda' in prov or 'tensorrt' in prov)
                       else 'cpu'})

    # Template eye points for the real crop size, expressed in 512 mask space.
    dst = swap_template_points(args.crop, 'arcface')
    s = 512.0 / float(args.crop)
    tmpl = dst[:2] * s                       # left eye, right eye
    print(f'template eyes @crop {args.crop} -> 512 space: '
          f'L={tmpl[0].round(1)} R={tmpl[1].round(1)}')

    dl, dr, rad = [], [], []
    for vp in args.video:
        cap = cv2.VideoCapture(vp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
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
                if (labels == 6).sum() > 200:
                    continue                  # glasses present: eyes unusable
                # BiSeNet l_eye(4) is the SUBJECT's left -> image RIGHT -> tmpl[1].
                for cls, store, t in ((4, dl, tmpl[1]), (5, dr, tmpl[0])):
                    ys, xs = np.nonzero(labels == cls)
                    if xs.size < 40:
                        continue
                    c = np.array([xs.mean(), ys.mean()])
                    store.append(np.linalg.norm(c - t))
                    rad.append(float(np.sqrt(((xs - c[0]) ** 2
                                              + (ys - c[1]) ** 2).mean())))
        cap.release()

    for name, d in (('left eye', dl), ('right eye', dr)):
        if not d:
            print(f'{name}: no samples')
            continue
        d = np.array(d)
        print(f'{name}: n={len(d)}  offset from template  '
              f'p50={np.percentile(d,50):.1f}px  p90={np.percentile(d,90):.1f}px  '
              f'max={d.max():.1f}px')
    if rad:
        r = np.array(rad)
        print(f'parsed eye rms radius: p50={np.percentile(r,50):.1f}px  '
              f'p90={np.percentile(r,90):.1f}px  max={r.max():.1f}px')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
