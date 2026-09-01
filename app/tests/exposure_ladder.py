"""Does the swapped face stay as dark as the scene it is in?

WHY SYNTHETIC. The campaign's lighting phase asks whether a swap in a dark
scene comes out realistically dark, and specifically whether the restorer
"hallucinates a bright daytime face". There is no real night footage on this
machine: a survey of all 32 clips found the darkest REAL material is a clip with
dark passages (5th-percentile luma 13.9) and nothing with a dark median. One
clip qualifies on median luma and it is a CGI head rotating, not a lit scene.

Rather than grade a phase on material that does not exist, this builds an
exposure ladder from a clip that IS well lit: the same frames, the same face,
the same pose, at 1.0 / 0.55 / 0.30 / 0.15 / 0.08 of their original exposure.
That is PAIRED, which real night footage can never be -- the only variable is
how much light there is.

WHAT IT MEASURES. Not the face's absolute brightness, which must fall as the
scene darkens. The invariant is the RATIO between the swapped face and the
scene around it. If exposure is respected, that ratio is roughly constant down
the ladder. If the pipeline lifts dark faces, the ratio climbs as the scene
darkens, and the climb IS the hallucination -- a face brighter than the room it
is in.

Each rung is also compared against the plate's own face at the same exposure,
which is the ground truth for "how bright should this face be".

WHAT IT IS NOT. Scaling exposure is not the same as shooting in the dark: there
is no sensor noise, no colour shift, and no lost shadow detail. So this isolates
the exposure question cleanly and does NOT establish behaviour on real night
footage. That remains open and needs material this machine does not have.

    env/Scripts/python.exe tests/exposure_ladder.py
    env/Scripts/python.exe tests/exposure_ladder.py --appearance off --frames 6
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

# Multiplicative, in linear-ish space via a gamma round trip, so a rung looks
# like less light rather than like a crushed curve.
RUNGS = (1.00, 0.55, 0.30, 0.15, 0.08)


def expose(frame, gain):
    if gain >= 0.999:
        return frame.copy()
    lin = np.power(frame.astype(np.float32) / 255.0, 2.2) * float(gain)
    return np.clip(np.power(lin, 1 / 2.2) * 255.0, 0, 255).astype(np.uint8)


def region_luma(img, box, shrink=0.75):
    """Mean luma inside a box, shrunk toward its centre.

    Shrunk because the detector box includes hair and background at the corners,
    and this is meant to be a measurement of the FACE.
    """
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hw, hh = (x1 - x0) * shrink / 2.0, (y1 - y0) * shrink / 2.0
    h, w = img.shape[:2]
    a = max(0, int(cy - hh)), max(0, int(cx - hw))
    b = min(h, int(cy + hh)), min(w, int(cx + hw))
    if b[0] <= a[0] or b[1] <= a[1]:
        return None
    crop = img[a[0]:b[0], a[1]:b[1]]
    return float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())


def scene_luma(img, box):
    """Mean luma of everything OUTSIDE the face box -- the room, in effect."""
    m = np.ones(img.shape[:2], dtype=bool)
    x0, y0, x1, y1 = [int(v) for v in box]
    m[max(0, y0):y1, max(0, x0):x1] = False
    if not m.any():
        return None
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[m].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=fixtures.clip("single/s5.mp4"))
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--source", default="harjot")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--swap-model", default=None)
    ap.add_argument("--enhancer", default=None)
    ap.add_argument("--mask-engine", default=None)
    ap.add_argument("--appearance", choices=("config", "on", "off"),
                    default="config",
                    help="target-conditioned appearance; 'config' uses "
                         "config.yaml, which on this machine has it ON")
    ap.add_argument("--max-ratio-climb", type=float, default=0.35,
                    help="how much the face/scene luma ratio may rise from the "
                         "brightest rung to the darkest before the face is "
                         "called brighter than its scene")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    provider = args.provider or str(cfg.provider)
    swap_model = args.swap_model or str(cfg.swap_model)
    enhancer = args.enhancer or str(cfg.selected_enhancer)
    mask_engine = args.mask_engine or str(cfg.mask_engine)

    engine_key = map_mask_engine(mask_engine)
    g = ab.init_pipeline(provider, swap_model, enhancer, mask_engine,
                         sync_config=True)
    if args.appearance != "config":
        g.target_conditioned_appearance = (args.appearance == "on")
    print("[expo] %s / %s / %s / %s | target_conditioned_appearance=%s"
          % (swap_model, mask_engine, enhancer, provider,
             getattr(g, "target_conditioned_appearance", False)), flush=True)

    options = ab.build_options(g, swap_model, engine_key)
    src_fs = load_library_faceset(args.source)
    from roop.core import live_swap

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("could not open %s" % args.video)
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    per_rung = {g_: [] for g_ in RUNGS}
    for i in range(args.frames):
        idx = args.start + i * args.stride
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        for gain in RUNGS:
            plate = expose(frame, gain)
            face = ab.biggest_face(plate)
            if face is None:
                # A rung dark enough to lose the face is recorded, not skipped:
                # "the detector stops finding faces below this exposure" is
                # itself a result for this phase.
                per_rung[gain].append(None)
                continue
            swapped = live_swap(plate.copy(), options, input_facesets=[src_fs])
            if swapped is None:
                per_rung[gain].append(None)
                continue
            box = [float(v) for v in face.bbox]
            f_sw = region_luma(swapped, box)
            f_pl = region_luma(plate, box)
            scene = scene_luma(plate, box)
            if None in (f_sw, f_pl, scene) or scene < 0.5:
                per_rung[gain].append(None)
                continue
            per_rung[gain].append({"face_sw": f_sw, "face_plate": f_pl,
                                   "scene": scene,
                                   "ratio": f_sw / scene,
                                   "plate_ratio": f_pl / scene})
            if args.out:
                cv2.imwrite(os.path.join(
                    args.out, "f%06d_g%03d.png" % (idx, int(gain * 100))),
                    swapped)
    cap.release()

    print("\n[expo] gain   scene luma   plate face   swapped face   "
          "face/scene   plate face/scene   detected")
    ratios = {}
    for gain in RUNGS:
        rows = [r for r in per_rung[gain] if r]
        found = len(rows)
        total = len(per_rung[gain])
        if not rows:
            print("[expo] %5.2f   %s" % (gain, "no face detected at this rung"))
            continue
        m = lambda k: float(np.mean([r[k] for r in rows]))   # noqa: E731
        ratios[gain] = m("ratio")
        print("[expo] %5.2f   %10.2f   %10.2f   %12.2f   %10.3f   %16.3f   %d/%d"
              % (gain, m("scene"), m("face_plate"), m("face_sw"),
                 m("ratio"), m("plate_ratio"), found, total))

    if len(ratios) < 2:
        print("\n[expo] INCONCLUSIVE -- fewer than two usable rungs")
        return 1
    bright = ratios[max(ratios)]
    dark = ratios[min(ratios)]
    climb = (dark - bright) / max(bright, 1e-6)
    print("\n[expo] face/scene luma ratio: %.3f at gain %.2f -> %.3f at gain "
          "%.2f  (%+.1f%%)"
          % (bright, max(ratios), dark, min(ratios), 100.0 * climb))
    if climb > args.max_ratio_climb:
        print("[expo] FAIL -- the swapped face grows brighter RELATIVE to its "
              "scene as the scene darkens. That is the hallucinated-daylight-"
              "face signature.")
        return 1
    print("[expo] PASS -- the face tracks the scene's exposure down the ladder. "
          "NOT a claim about real night footage: scaled exposure has no sensor "
          "noise, no colour shift and no lost shadow detail.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
