"""Swap the expression/ clips and measure whether the EXPRESSION survived.

These clips exist to exercise what identity metrics cannot see. A swap can score
perfectly on identity while flattening a blink, damping a smile, or holding a
mouth half-shut -- the face is the right person and the performance is gone.
Requirements 5, 6 and 17 all rest on that, and until now they rested on looking
at three still frames.

Measured per frame, plate against output, from the 106-point landmarks:

  EAR   eye aspect ratio -- lid separation over eye width. Drops toward 0 on a
        blink, so its TIME SERIES is the blink track.
  MAR   mouth aspect ratio -- lip separation over mouth width.

What matters is not the absolute value (a different face has differently shaped
eyes, and should) but whether the output MOVES WITH the plate. So the headline
is the correlation of the two series, plus how much of the plate's range the
output reproduces: a swap that damps a blink to half depth still correlates at
1.0 and is still wrong, which is why both are reported.

Usage:
    env/Scripts/python.exe tests/expression_bench.py --source harjot
    env/Scripts/python.exe tests/expression_bench.py --only e3 --tag mine
"""

import argparse
import glob
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

import angle_bench as ab                                          # noqa: E402
from angle_video import ensure_ffmpeg                             # noqa: E402
from two_face_video import (load_library_faceset, faceset_mean,   # noqa: E402
                            cos, auto_capture_targets)
import sample_bench as sb                                         # noqa: E402

EXPRESSION_DIR = r"G:/pinokio/roop-keep/expression"

# insightface 106-point indices.
L_EYE = [33, 35, 36, 37, 39, 42]
R_EYE = [87, 89, 90, 91, 93, 96]
MOUTH_V = (52, 61)          # upper lip centre, lower lip centre
MOUTH_H = (53, 59)          # left corner, right corner


def _ratio(pts, idx_v, idx_h):
    v = np.linalg.norm(pts[idx_v[0]] - pts[idx_v[1]])
    h = np.linalg.norm(pts[idx_h[0]] - pts[idx_h[1]])
    return float(v / h) if h > 1e-6 else float("nan")


def ear(pts, idx):
    """Lid separation over eye width, averaged over the two vertical pairs."""
    try:
        p = pts[idx]
    except Exception:
        return float("nan")
    w = np.linalg.norm(p[0] - p[3])
    if w <= 1e-6:
        return float("nan")
    return float((np.linalg.norm(p[1] - p[5]) +
                  np.linalg.norm(p[2] - p[4])) / (2.0 * w))


def measure(face):
    lm = getattr(face, "landmark_2d_106", None)
    if lm is None:
        return None
    p = np.asarray(lm, dtype=np.float64)
    if p.shape[0] < 106:
        return None
    e = float(np.nanmean([ear(p, L_EYE), ear(p, R_EYE)]))
    m = _ratio(p, MOUTH_V, MOUTH_H)
    return {"ear": e, "mar": m}


def pair_stats(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 8 or a.std() < 1e-9 or b.std() < 1e-9:
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    rng = (float((b.max() - b.min()) / (a.max() - a.min()))
           if a.max() > a.min() else float("nan"))
    return {"r": r, "range": rng, "n": int(len(a)),
            "plate_swing": float(a.max() - a.min())}


def fmt(d, k):
    return f"{d[k]:.3f}" if d else "  --  "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="harjot")
    ap.add_argument("--provider", default="tensorrt")
    ap.add_argument("--swap-model", default="realswap")
    ap.add_argument("--mask-engine", default=None, help="default: config.yaml")
    ap.add_argument("--enhancer", default="None")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--tag", default="expr")
    ap.add_argument("--only", default="", help="e1..e4, comma separated")
    args = ap.parse_args()

    ensure_ffmpeg()
    from settings import Settings
    cfg = Settings("config.yaml")
    me_name = args.mask_engine or cfg.mask_engine
    me = (sb.map_mask_engine(me_name)
          if me_name not in ("", "None", None) else "None")

    g = ab.init_pipeline(
        args.provider, args.swap_model, args.enhancer, me,
        float(getattr(cfg, "swap_model_mask_strength", 0.0) or 0.0))
    g.execution_threads = args.threads
    g.video_encoder = "libx264"
    g.video_quality = 12
    g.face_swap_mode = "selected"
    g.track_identities = True
    g.CFG.track_identities = True
    g.temporal_detection = True
    g.CFG.temporal_detection = True
    options = ab.build_options(g, args.swap_model, me)

    fs = load_library_faceset(args.source)
    src_mean = faceset_mean(fs)
    out_dir = os.path.join(APP, "output", f"expression_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)

    clips = sorted(glob.glob(os.path.join(EXPRESSION_DIR, "*.mp4")))
    want = [w.strip() for w in args.only.split(",") if w.strip()]
    print(f"[expr] {len(clips)} clip(s); model={args.swap_model} "
          f"mask={me_name} enhancer={args.enhancer} threads={args.threads} "
          f"swap_model_mask={g.swap_model_mask_strength} source={args.source}",
          flush=True)

    from roop.face_util import get_all_faces
    rows = []
    for i, clip in enumerate(clips, 1):
        name = f"e{i}"
        if want and name not in want:
            continue
        print(f"\n[expr] === {name} : {os.path.basename(clip)[:64]} ===",
              flush=True)
        targets, groups = auto_capture_targets(clip, expect=1,
                                               log_prefix="[expr]",
                                               strict=False)
        if targets is None:
            cap_idx, _, faces = sb.first_face_frame(clip)
            targets, groups = [sb.select_primary_face(faces)], [0]
            print(f"[expr] fell back to first-face capture, frame {cap_idx}",
                  flush=True)
        out, elapsed, _ = sb.run_swap(clip, [fs], targets, groups, options,
                                      out_dir)
        if not out:
            print(f"[expr] FAILED: no output for {name}", flush=True)
            continue
        final = os.path.join(out_dir, f"{name}__{args.source}.mp4")
        if os.path.abspath(out) != os.path.abspath(final):
            if os.path.exists(final):
                os.remove(final)
            os.rename(out, final)

        cp, co = cv2.VideoCapture(clip), cv2.VideoCapture(final)
        pe, po, pm, pmo, ids, det, tot = [], [], [], [], [], 0, 0
        while True:
            okp, fp = cp.read()
            oko, fo = co.read()
            if not (okp and oko):
                break
            tot += 1
            fpl = get_all_faces(fp) or []
            fol = get_all_faces(fo) or []
            if not fpl or not fol:
                continue
            a = max(fpl, key=lambda q: q.bbox[2] - q.bbox[0])
            b = max(fol, key=lambda q: q.bbox[2] - q.bbox[0])
            ma, mb = measure(a), measure(b)
            if ma and mb:
                det += 1
                pe.append(ma["ear"])
                po.append(mb["ear"])
                pm.append(ma["mar"])
                pmo.append(mb["mar"])
            if src_mean is not None and getattr(b, "embedding", None) is not None:
                ids.append(cos(b.embedding, src_mean))
        cp.release()
        co.release()

        es, ms = pair_stats(pe, po), pair_stats(pm, pmo)
        idm = float(np.mean(ids)) if ids else float("nan")
        rows.append((name, final, tot, det, idm, es, ms, elapsed))
        print(f"[expr] {name}: {elapsed:.1f}s -> {final}", flush=True)

    print("\n" + "=" * 96)
    print("EXPRESSION RESULTS   r = does the output move WITH the plate;  "
          "range = how much of the plate's swing it keeps")
    print("=" * 96)
    print(f"{'clip':<5}{'frames':>7}{'graded':>8}{'identity':>10}"
          f"{'EYE r':>8}{'EYE range':>11}{'MOUTH r':>9}{'MOUTH range':>13}")
    for name, final, tot, det, idm, es, ms, el in rows:
        print(f"{name:<5}{tot:>7}{det:>8}{idm:>10.3f}"
              f"{fmt(es, 'r'):>8}{fmt(es, 'range'):>11}"
              f"{fmt(ms, 'r'):>9}{fmt(ms, 'range'):>13}")
    print("\nOUTPUT FILES:")
    for name, final, *_ in rows:
        print(f"  {name}: {final}")


if __name__ == "__main__":
    main()
