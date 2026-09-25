"""Restore Ultra: linear vs frequency-split recombination, graded on real swaps.

Every Restore Ultra call made while swapping real frames (roop.core.live_swap,
live config) is intercepted, and the exact pair it saw is kept: the
FFHQ-realigned SWAP crop it was handed, the raw RestoreFormer++ output and the
Ultra-finished output. The variants below are then built OFFLINE from those
same pairs by the production function (roop.enhance_blend), so every arm is
graded on identical inputs.

METRICS, on the 512 crop:

  id_src   cosine to the source faceset's embedding (the app's own
           recognizer). Higher = more of the person we are swapping IN.
  id_tgt   cosine to the ORIGINAL target crop. Higher = more of the person
           being replaced leaking back (the restorer's prior pulling toward
           the plate it never saw is unlikely; toward "a generic face" is the
           usual drift, which shows as id_src falling instead).
  skin_hf  band-pass (sigma 0.7 - 2.5) luma std on cheek/forehead skin,
           divided by the same on the ORIGINAL target crop at the same
           geometry: 1.0 = as textured as the real footage. The waxy look is
           this falling well below 1.
  tone     mean |Lab difference| of heavily blurred (sigma 12) output vs the
           swap input over the face: how far the restorer moved lighting and
           colour. The frequency arms are ~0 BY CONSTRUCTION on this one; it
           is here to size what the linear arm moves, not to rank.

    env/Scripts/python.exe tests/restore_ultra_bench.py
    env/Scripts/python.exe tests/restore_ultra_bench.py --clips single/s5.mp4 --out d/
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

import fixtures                       # noqa: E402
import angle_bench as ab              # noqa: E402
from two_face_video import load_library_faceset, map_mask_engine  # noqa: E402

DEFAULT_CLIPS = ['single/s%d.mp4' % i for i in (1, 2, 3, 4, 5, 6)] + \
                ['double/d1.mp4', 'double/d6.mp4']


def unit(v):
    v = np.asarray(v, np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def skin_mask(lm, shape):
    """Cheeks + forehead: the face hull minus grown eyes/brows/nose/mouth."""
    from roop.enhance_blend import INNER_GROUPS
    h, w = shape[:2]
    face = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(face, cv2.convexHull(
        np.round(lm[list(range(0, 33)) + list(range(43, 52))
                    + list(range(97, 106))]).astype(np.int32)), 255)
    feat = np.zeros((h, w), np.uint8)
    for idx in INNER_GROUPS:
        cv2.fillConvexPoly(feat, cv2.convexHull(
            np.round(lm[list(idx)]).astype(np.int32)), 255)
    k = max(3, int(0.06 * w) | 1)
    feat = cv2.dilate(feat, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    face = cv2.erode(face, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return (face > 0) & (feat == 0)


def band_std(img, m):
    y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.GaussianBlur(y, (0, 0), 0.7) - cv2.GaussianBlur(y, (0, 0), 2.5)
    return float(b[m].std()) if m.any() else float('nan')


def tone(img, ref, m):
    a = cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 12), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(cv2.GaussianBlur(ref, (0, 0), 12), cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.abs(a - b)[m].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=DEFAULT_CLIPS)
    ap.add_argument("--source", default="akansha")
    ap.add_argument("--start", type=int, default=60)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    swap_model, mask_engine = str(cfg.swap_model), str(cfg.mask_engine)
    g = ab.init_pipeline(str(cfg.provider), swap_model, "Restore Ultra",
                         mask_engine, sync_config=True)
    g.selected_enhancer = "Restore Ultra"
    options = ab.build_options(g, swap_model, map_mask_engine(mask_engine))
    src_fs = load_library_faceset(args.source)
    src_emb = unit(src_fs.faces[0].embedding)

    from roop.core import live_swap
    from roop.face_util import get_first_face, estimate_norm
    from roop.processors.Enhance_RestoreUltra import Enhance_RestoreUltra
    from roop.processors.Enhance_RestoreFormerPPlus import Enhance_RestoreFormerPPlus
    from roop.processors.enhance_common import enhance_restore_ultra
    from roop import enhance_blend as eb

    captured = []
    cur = {}

    def hooked(self, source_faceset, target_face, temp_frame):
        raw, sf = Enhance_RestoreFormerPPlus.Run(self, source_faceset,
                                                 target_face, temp_frame)
        ref = temp_frame
        if ref.shape[:2] != raw.shape[:2]:
            ref = cv2.resize(ref, (raw.shape[1], raw.shape[0]),
                             interpolation=cv2.INTER_CUBIC)
        fin = enhance_restore_ultra(raw, ref, target_face=target_face)
        plate = cur['frame']
        M = estimate_norm(np.asarray(target_face.kps, np.float32), 512,
                          mode='ffhq_512')
        orig = cv2.warpAffine(plate, M, (512, 512),
                              borderMode=cv2.BORDER_REPLICATE)
        captured.append(dict(swap=temp_frame.copy(), raw=raw.copy(),
                             ultra=fin.copy(), orig=orig, clip=cur['clip']))
        return fin, sf

    Enhance_RestoreUltra.Run = hooked

    for clip in args.clips:
        path = fixtures.clip(clip)
        if not os.path.isfile(path):
            print("[ru] skip %s: missing" % clip)
            continue
        cap = cv2.VideoCapture(path)
        for k in range(args.frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.start + k * args.stride)
            ok, fr = cap.read()
            if not ok:
                break
            cur.update(frame=fr, clip=clip)
            live_swap(fr.copy(), options, input_facesets=[src_fs])
        cap.release()
        print("[ru] %s: %d restorer calls so far" % (clip, len(captured)),
              flush=True)
    if not captured:
        raise SystemExit("Restore Ultra was never called -- nothing graded. "
                         "Is it the selected enhancer and did any face swap?")

    def variants(c):
        s512 = cv2.resize(c['swap'], (512, 512), interpolation=cv2.INTER_CUBIC) \
            if c['swap'].shape[:2] != (512, 512) else c['swap']
        u = c['ultra'] if c['ultra'].shape[:2] == (512, 512) else \
            cv2.resize(c['ultra'], (512, 512), interpolation=cv2.INTER_AREA)
        r = c['raw'] if c['raw'].shape[:2] == (512, 512) else \
            cv2.resize(c['raw'], (512, 512), interpolation=cv2.INTER_AREA)
        v = {
            'swap (no restore)': s512,
            'restoreformer raw': r,
            'restore ultra (shipped)': u,
            'linear 0.75': cv2.addWeighted(u, 0.75, s512, 0.25, 0),
        }
        for w in (0.5, 0.75, 1.0):
            v['freq w%.2f s3' % w] = eb.frequency_blend(s512, u, weight=w, sigma=3.0)
        for s in (2.0, 5.0):
            v['freq w0.75 s%g' % s] = eb.frequency_blend(s512, u, weight=0.75, sigma=s)
        return s512, v

    rows = {}
    n_ok = 0
    panel = None
    for c in captured:
        s512, v = variants(c)
        fo = get_first_face(c['orig'])
        fu = get_first_face(v['restore ultra (shipped)'])
        if fo is None or fu is None or getattr(fu, 'landmark_2d_106', None) is None:
            continue
        lm = np.asarray(fu.landmark_2d_106, np.float32)
        sm = skin_mask(lm, (512, 512))
        if sm.sum() < 500:
            continue
        fm = np.zeros((512, 512), np.uint8)
        cv2.fillConvexPoly(fm, cv2.convexHull(np.round(lm[:33]).astype(np.int32)), 1)
        fm = fm.astype(bool)
        tgt_emb = unit(fo.embedding)
        base_hf = band_std(c['orig'], sm)
        inner = eb.inner_feature_weight(lm, (512, 512))
        v['inner-only freq w0.75'] = eb.frequency_blend(
            s512, v['restore ultra (shipped)'], weight=0.75, sigma=3.0, region=inner)
        n_ok += 1
        for name, img in v.items():
            f = get_first_face(img)
            if f is None:
                continue
            e = unit(f.embedding)
            rows.setdefault(name, []).append((
                float(e @ src_emb), float(e @ tgt_emb),
                band_std(img, sm) / max(base_hf, 1e-6), tone(img, s512, fm)))
        if panel is None and args.out:
            keys = ['swap (no restore)', 'restore ultra (shipped)',
                    'freq w0.75 s3', 'inner-only freq w0.75']
            panel = np.hstack([c['orig']] + [v[k] for k in keys])

    print("\n[ru] %d restorer calls, %d graded (faces found in every crop)"
          % (len(captured), n_ok))
    print("%-26s %4s %8s %8s %8s %7s   %s" % ('arm', 'n', 'id_src', 'id_tgt',
                                          'skin_hf', 'tone',
                                          'id_src > shipped (paired)'))
    ship = np.array(rows.get('restore ultra (shipped)', []))
    for name, vals in rows.items():
        a = np.array(vals)
        paired = '-'
        if name != 'restore ultra (shipped)' and len(a) == len(ship):
            d = a[:, 0] - ship[:, 0]
            paired = '%d/%d  (mean %+.4f, sd %.4f)' % (int((d > 0).sum()),
                                                        len(d), d.mean(), d.std())
        print("%-26s %4d %8.4f %8.4f %8.3f %7.2f   %s"
              % (name, len(a), a[:, 0].mean(), a[:, 1].mean(),
                 np.nanmean(a[:, 2]), a[:, 3].mean(), paired))
    if args.out and panel is not None:
        os.makedirs(args.out, exist_ok=True)
        p = os.path.join(args.out, 'orig_swap_ultra_freq_inner.png')
        cv2.imwrite(p, panel)
        print("[ru] wrote %s" % p)


if __name__ == "__main__":
    main()
