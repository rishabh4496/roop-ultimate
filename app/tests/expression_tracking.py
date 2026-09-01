"""Do the target's eyes and mouth still drive the swapped face?

The campaign asks for two things the source must NOT control: eye state
(blinking, squinting, winking) and mouth/expression. Both are acceptance
criteria and neither had a run behind it on either GPU.

WHAT IS MEASURED. Per frame, on 68-point landmarks derived from the detector's
106, two normalised apertures:

    EAR    eye aspect ratio -- vertical lid separation over eye width
    MAR    mouth aspect ratio -- lip separation over mouth width

Both are scale-free, so a face that moves toward or away from camera does not
change them. Then, across the clip, the PLATE's series is correlated against
the SWAPPED output's.

    r near 1     the output follows the target's own eyes and mouth
    r near 0     the output ignores them -- the source is driving expression
    swapped range near 0    a frozen face: it never opens or closes at all

The range check matters as much as the correlation. A pipeline that emits a
constant half-open eye scores an undefined or meaningless correlation, and a
pipeline that damps a blink to a tenth of its depth can still correlate at 0.95
while looking obviously wrong. So range is reported beside r, as a fraction of
the target's own range.

WHY A CLIP WITH ACTUAL EXPRESSION. Correlating two flat series measures noise.
`survey_fixtures` was used to pick material where the face is large, single, and
doing something; the default here is one of those. A clip whose target EAR
barely moves is reported as INCONCLUSIVE rather than scored.

Frames are swapped in process through `live_swap`, one at a time. That means no
temporal engines and no tracker -- which is the STRICTER test for this
question: any expression tracking measured here comes from the per-frame path
rather than from smoothing across neighbours.

    env/Scripts/python.exe tests/expression_tracking.py
    env/Scripts/python.exe tests/expression_tracking.py --frames 90 --stride 2
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

# 68-point indices. Standard iBUG layout, which `angle_bench.lm68` maps to.
L_EYE = (36, 37, 38, 39, 40, 41)
R_EYE = (42, 43, 44, 45, 46, 47)
MOUTH_OUTER = (48, 51, 54, 57)          # left, top, right, bottom
MOUTH_INNER = (60, 62, 64, 66)


def _ear(p, idx):
    """Eye aspect ratio: mean lid separation over corner-to-corner width."""
    a, b, c, d, e, f = [np.asarray(p[i], dtype=np.float64) for i in idx]
    width = np.linalg.norm(a - d)
    if width < 1e-6:
        return None
    return float((np.linalg.norm(b - f) + np.linalg.norm(c - e)) / (2.0 * width))


def _mar(p):
    """Mouth aspect ratio from the inner lip contour."""
    l, t, r, b = [np.asarray(p[i], dtype=np.float64) for i in MOUTH_INNER]
    width = np.linalg.norm(l - r)
    if width < 1e-6:
        return None
    return float(np.linalg.norm(t - b) / width)


def measure(face):
    pts = ab.lm68(face)
    if pts is None:
        return None
    le, re, mar = _ear(pts, L_EYE), _ear(pts, R_EYE), _mar(pts)
    if None in (le, re, mar):
        return None
    return {"ear": (le + re) / 2.0, "mar": mar}


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=fixtures.clip(
        "3d model/Flexpressions.mp4"),
        help="needs a single, large face that actually changes expression")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--source", default="harjot")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--swap-model", default=None)
    ap.add_argument("--enhancer", default=None)
    ap.add_argument("--mask-engine", default=None)
    ap.add_argument("--min-r", type=float, default=0.60,
                    help="Pearson r between the target's series and the "
                         "output's, below which the output is not following "
                         "the target")
    ap.add_argument("--min-range-frac", type=float, default=0.50,
                    help="output range as a fraction of the target's. Catches "
                         "a damped blink, which can still correlate at 0.95")
    ap.add_argument("--min-target-ear-range", type=float, default=0.05,
                    help="if the TARGET's own eyes barely move, there is "
                         "nothing to track and the run is inconclusive")
    args = ap.parse_args()

    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    provider = args.provider or str(cfg.provider)
    swap_model = args.swap_model or str(cfg.swap_model)
    enhancer = args.enhancer or str(cfg.selected_enhancer)
    mask_engine = args.mask_engine or str(cfg.mask_engine)

    print("[expr] %s / %s / %s / %s | %s"
          % (swap_model, mask_engine, enhancer, provider,
             os.path.basename(args.video)), flush=True)

    g = ab.init_pipeline(provider, swap_model, enhancer, mask_engine,
                         sync_config=True)
    options = ab.build_options(g, swap_model, map_mask_engine(mask_engine))
    src_fs = load_library_faceset(args.source)
    from roop.core import live_swap

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("could not open %s" % args.video)

    t_ear, t_mar, s_ear, s_mar, misses = [], [], [], [], 0
    for i in range(args.frames):
        idx = args.start + i * args.stride
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, plate = cap.read()
        if not ok:
            break
        pf = ab.biggest_face(plate)
        if pf is None:
            misses += 1
            continue
        pm = measure(pf)
        if pm is None:
            misses += 1
            continue
        swapped = live_swap(plate.copy(), options, input_facesets=[src_fs])
        if swapped is None:
            misses += 1
            continue
        sf = ab.biggest_face(swapped)
        sm = measure(sf) if sf is not None else None
        if sm is None:
            # The face became undetectable, or lost its landmarks, AFTER the
            # swap. Counted, because that is a failure of its own and must not
            # quietly shrink the sample.
            misses += 1
            continue
        t_ear.append(pm["ear"]); t_mar.append(pm["mar"])
        s_ear.append(sm["ear"]); s_mar.append(sm["mar"])
    cap.release()

    n = len(t_ear)
    print("[expr] %d frames measured, %d unusable" % (n, misses))
    if n < 8:
        print("[expr] INCONCLUSIVE -- too few usable frames")
        return 1

    def rng(v):
        return float(np.percentile(v, 95) - np.percentile(v, 5))

    rows = []
    for label, t, s in (("eyes (EAR)", t_ear, s_ear),
                        ("mouth (MAR)", t_mar, s_mar)):
        r = _pearson(t, s)
        tr, sr = rng(t), rng(s)
        frac = (sr / tr) if tr > 1e-9 else None
        rows.append((label, r, tr, sr, frac))
        print("[expr] %-12s r=%6s   target range %.4f   output range %.4f   "
              "%s of target"
              % (label, ("%.3f" % r) if r is not None else "n/a", tr, sr,
                 ("%.0f%%" % (100 * frac)) if frac is not None else "n/a"))

    ear_row = rows[0]
    if ear_row[2] < args.min_target_ear_range:
        print("\n[expr] INCONCLUSIVE -- the TARGET's own eye aperture barely "
              "moves on this clip (range %.4f). Correlating two flat series "
              "measures noise; pick material with real blinks."
              % ear_row[2])
        return 1

    failures = []
    for label, r, tr, sr, frac in rows:
        if r is None:
            failures.append("%s: no correlation computable (a flat series)"
                            % label)
            continue
        if r < args.min_r:
            failures.append("%s: r=%.3f -- the output is not following the "
                            "target" % (label, r))
        elif frac is not None and frac < args.min_range_frac:
            failures.append("%s: r=%.3f but the output moves only %.0f%% as "
                            "far as the target -- damped, not tracked"
                            % (label, r, 100 * frac))

    if failures:
        print("\n[expr] %d FAILURE(S):" % len(failures))
        for f in failures:
            print("[expr]   %s" % f)
        return 1
    print("\n[expr] PASS -- eye and mouth aperture in the output follow the "
          "target's own, at full amplitude, on the per-frame path with no "
          "temporal smoothing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
