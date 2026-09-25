"""Does the XSeg edge refinement / flow-warped mask stabilizer actually help?

Two features, one question each:

  mask_guided_filter  -- does XSeg's boundary land closer to the real face edge?
  mask_flow_warp      -- does the stabilized mask track the face better than
                         the in-place EMA it replaces, at no flicker cost?

THE REFERENCE, AND WHY IT IS NOT A SYNTHETIC OCCLUDER. The first version of
this harness composited a textured ellipse (occlusion_ground_truth.py's) and
graded against its exact mask. XSeg does not see such a patch as an occluder
at all -- it labels most of it "face" -- so the ellipse edge was not an edge
XSeg was even trying to draw, and every number graded against it measured
nothing. A metric built from image gradients is no better: a guided filter
moves the mask onto gradients by construction, so it would grade itself.

So the reference is an INDEPENDENT segmenter: BiSeNet's raw per-pixel class
map (Mask_FaceParser.RunLabels, 512px, no blur, no grow), face = skin + brows
+ eyes + nose + mouth, the parser's own default region. Pixels whose class is
ambiguous between the two models' definitions of "face" (glasses, ears, neck,
cloth, hat) are not graded. This is AGREEMENT with a second model, not truth;
it is strictly weaker than a hand-labelled matte and is reported as such.

Everything runs on the REAL XSeg model through the app's own init (TensorRT on
the live config), on real consecutive frames, on crops from the app's own
align_crop. The filters are the production functions, called exactly as
process_mask calls them (procmgr_masking.refine_mask_edges, then
one_euro.MaskStabilizer.apply with the crop as guide).

METRICS, in the 256px mask space, inside a 6px band around XSeg's raw 0.5
contour (where either feature can change anything):

  ref_mae   mean |mask - parser_restore|    lower = closer to the reference
  edge_err  mean |(mask>0.5) - (parser>0.5)| -- boundary PLACEMENT only
  flicker   mean |M_t - M_{t-1}| in the band. For the flow arm this is not a
            pure noise number: it is allowed to move where the content moved.
  flicker_mc  the same after warping M_{t-1} along an INDEPENDENT flow
            (Farneback at 256px, not the feature's DIS at 128): the change
            the content's motion does not explain, i.e. the jitter.

    env/Scripts/python.exe tests/xseg_refine_bench.py
    env/Scripts/python.exe tests/xseg_refine_bench.py --clips single/s5.mp4 --frames 200
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import fixtures                       # noqa: E402
import angle_bench as ab              # noqa: E402

FACE_CLASSES = (1, 2, 3, 4, 5, 10, 11, 12, 13)
AMBIGUOUS = (6, 7, 8, 9, 14, 15, 16, 18)
DEFAULT_CLIPS = ['single/s%d.mp4' % i for i in range(1, 8)] + \
                ['double/d%d.mp4' % i for i in (1, 4, 6)]


def read_frames(path, start, n):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def squeeze(m):
    m = np.asarray(m, np.float32)
    return m[..., 0] if m.ndim == 3 else m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=DEFAULT_CLIPS)
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--strength", type=float, default=None,
                    help="stabilize_mask_strength (default: config.yaml)")
    ap.add_argument("--gf", nargs="*", default=["4:1e-3"],
                    help="guided-filter radius:eps pairs to grade")
    ap.add_argument("--band", type=int, default=6)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), "None",
                     "DFL XSeg", sync_config=True)
    strength = (float(args.strength) if args.strength is not None
                else float(cfg.stabilize_mask_strength))

    from roop.face_util import get_all_faces, align_crop
    from roop.processors.Mask_XSeg import Mask_XSeg
    from roop.processors.Mask_FaceParser import Mask_FaceParser
    from roop.one_euro import MaskStabilizer
    from roop import procmgr_masking as pm

    xseg = Mask_XSeg()
    xseg.Initialize({"devicename": "cuda"})
    parser = Mask_FaceParser()
    parser.Initialize({"devicename": "cuda"})
    print("[xseg] providers: %s" % xseg.model_xseg.get_providers(), flush=True)

    gf_sets = []
    for s in args.gf:
        r, e = s.split(":")
        gf_sets.append((int(r), float(e)))
    arms = [('raw', None, False, False)]
    for r, e in gf_sets:
        arms.append(('gf r%d e%g' % (r, e), (r, e), False, False))
    arms += [('stab', None, True, False), ('stab+flow', None, True, True)]
    r0, e0 = gf_sets[0]
    arms += [('gf+stab', (r0, e0), True, False),
             ('gf+stab+flow', (r0, e0), True, True)]

    kband = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * args.band + 1, 2 * args.band + 1))
    per_clip = {}
    t_x, t_gf, t_st = [], [], {'plain': [], 'flow': []}
    lines = {}

    for clip in args.clips:
        path = fixtures.clip(clip)
        if not os.path.isfile(path):
            print("[xseg] skip %s: missing" % clip)
            continue
        frames = read_frames(path, args.start, args.frames)
        rec = []
        for fr in frames:
            faces = get_all_faces(fr) or []
            if not faces:
                rec.append(None)
                continue
            face = max(faces, key=lambda f: float(f.bbox[2] - f.bbox[0]))
            kps = np.asarray(face.kps, np.float32)
            crop, _ = align_crop(fr, kps, args.crop, mode="arcface")
            a = time.perf_counter()
            raw = xseg.Run(crop, "")
            t_x.append(time.perf_counter() - a)
            labels = parser.RunLabels(crop).astype(np.uint8)
            lab256 = cv2.resize(labels, (256, 256),
                                interpolation=cv2.INTER_NEAREST)
            face_ref = np.isin(labels, FACE_CLASSES).astype(np.float32)
            ref = 1.0 - cv2.resize(face_ref, (256, 256),
                                   interpolation=cv2.INTER_AREA)
            r2 = squeeze(raw)
            contour = cv2.morphologyEx((r2 > 0.5).astype(np.uint8),
                                       cv2.MORPH_GRADIENT, kband) > 0
            band = contour & ~np.isin(lab256, AMBIGUOUS)
            gray = cv2.cvtColor(cv2.resize(crop, (256, 256),
                                           interpolation=cv2.INTER_AREA),
                                cv2.COLOR_BGR2GRAY)
            rec.append(dict(kps=kps, crop=crop, raw=raw, ref=ref, band=band,
                            gray=gray))
        # Independent motion compensation for the flicker metric: Farneback
        # at 256px, NOT the DIS-at-128 the flow-warp feature uses, so the
        # flow arm is not graded by its own field. Backward flow: for each
        # current pixel, where it was in the previous crop.
        for t in range(1, len(rec)):
            if rec[t] is not None and rec[t - 1] is not None:
                fl_ = cv2.calcOpticalFlowFarneback(
                    rec[t]['gray'], rec[t - 1]['gray'], None,
                    0.5, 3, 15, 3, 5, 1.2, 0)
                gx, gy = np.meshgrid(np.arange(256, dtype=np.float32),
                                     np.arange(256, dtype=np.float32))
                rec[t]['map'] = (gx + fl_[..., 0], gy + fl_[..., 1])
        n_ok = sum(r is not None for r in rec)
        if n_ok < 10:
            print("[xseg] skip %s: faces on %d frames" % (clip, n_ok))
            continue

        res = {}
        for name, gfp, use_stab, use_flow in arms:
            st = (MaskStabilizer(strength=strength, motion_beta=0.0,
                                 flow_warp=use_flow) if use_stab else None)
            mae, edge, fl, flm, prev = [], [], [], [], None
            for t, r in enumerate(rec):
                if r is None:
                    prev = None
                    continue
                m = r['raw']
                if gfp is not None:
                    a = time.perf_counter()
                    m = pm.refine_mask_edges(m, r['crop'], radius=gfp[0],
                                             eps=gfp[1])
                    t_gf.append(time.perf_counter() - a)
                if st is not None:
                    a = time.perf_counter()
                    m = st.apply(m, r['kps'], t, guide=r['crop'])
                    t_st['flow' if use_flow else 'plain'].append(
                        time.perf_counter() - a)
                m = squeeze(m)
                b = r['band']
                if b.any():
                    mae.append(float(np.abs(m - r['ref'])[b].mean()))
                    edge.append(float(((m > 0.5) != (r['ref'] > 0.5))[b].mean()))
                    if prev is not None:
                        fl.append(float(np.abs(m - prev)[b].mean()))
                        if 'map' in r:
                            wp = cv2.remap(prev, r['map'][0], r['map'][1],
                                           cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_REPLICATE)
                            flm.append(float(np.abs(m - wp)[b].mean()))
                prev = m
            res[name] = (np.mean(mae), np.mean(edge), np.mean(fl), np.mean(flm))
            if st is not None and use_flow:
                lines[(clip, name)] = st.flow_summary_line()
        per_clip[clip] = (n_ok, res)
        print("[xseg] %-16s %3d faces  raw ref_mae %.4f  %s"
              % (clip, n_ok, res['raw'][0],
                 "  ".join("%s %+.4f" % (k, v[0] - res['raw'][0])
                           for k, v in res.items() if k != 'raw')),
              flush=True)

    if not per_clip:
        raise SystemExit("no clip graded")
    print("\n[xseg] %d clips x %d frames from %d, crop %d, band %dpx, "
          "stabilize_mask_strength %.2f"
          % (len(per_clip), args.frames, args.start, args.crop, args.band,
             strength))
    print("%-18s %9s %9s %9s %10s   %s"
          % ('arm', 'ref_mae', 'edge_err', 'flicker', 'flicker_mc',
             'clips better than raw: ref_mae / flicker_mc'))
    for name, *_ in arms:
        vals = np.array([per_clip[c][1][name] for c in per_clip])
        raw = np.array([per_clip[c][1]['raw'] for c in per_clip])
        better = int(np.sum(vals[:, 0] < raw[:, 0]))
        bmc = int(np.sum(vals[:, 3] < raw[:, 3]))
        print("%-18s %9.4f %9.4f %9.5f %10.5f   %s"
              % (name, vals[:, 0].mean(), vals[:, 1].mean(), vals[:, 2].mean(),
                 vals[:, 3].mean(),
                 '-' if name == 'raw' else '%d/%d  /  %d/%d'
                 % (better, len(per_clip), bmc, len(per_clip))))
    for (clip, name), line in lines.items():
        if line:
            print("  %-16s %-13s %s" % (clip, name, line))
    ms = lambda v: 1e3 * float(np.median(v)) if v else float('nan')
    print("[xseg] median ms/face, one thread: xseg %.2f | guided filter %.3f | "
          "stab %.3f | stab+flow %.3f"
          % (ms(t_x), ms(t_gf), ms(t_st['plain']), ms(t_st['flow'])))


if __name__ == "__main__":
    main()
