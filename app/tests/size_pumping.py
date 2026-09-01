"""Does the swapped face change SIZE from frame to frame while the jaw moves?

Requirement 16. "Pumping" is a TEMPORAL artifact -- the face breathing bigger
and smaller as the mouth opens and closes -- so a spread of stills cannot show
it, and the eye is poor at judging it from two frames. What matters is not the
absolute size (the head really does move) but the RATIO of the output's face to
the plate's on the SAME frame: a swap that preserves size holds that ratio flat,
and a swap that pumps makes it wobble.

So the plate's own size change over the run is the control. A ratio that is
steady while the face moves a lot means the swap is tracking size correctly.

Three measures, because they fail differently:
  bbox_w / bbox_h  what a viewer perceives, but detector boxes are noisy and
                   jump when the pose changes;
  eye_mouth        an INTERNAL distance between the 5 keypoints, immune to box
                   noise -- the one to trust when the two disagree.

Usage:
    env/Scripts/python.exe tests/size_pumping.py --arm d5_platecrop --clip d5.mp4 \
        --start 2830 --count 60
"""

import argparse
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

import fixtures
import angle_bench as ab  # noqa: E402

ARMS = os.path.join(APP, "output", "bench_two_face")
CLIPS = fixtures.clip_dir('double')


def out_video(tag):
    work = os.path.join(ARMS, tag, "work")
    v = [f for f in os.listdir(work) if f.endswith(".mp4") and "resume" not in f]
    v.sort(key=lambda f: os.path.getsize(os.path.join(work, f)))
    return os.path.join(work, v[-1])


def measures(face):
    b = np.asarray(face.bbox, dtype=np.float64)
    kps = np.asarray(face.kps, dtype=np.float64).reshape(5, 2)
    eye_mid = (kps[0] + kps[1]) / 2.0
    mouth_mid = (kps[3] + kps[4]) / 2.0
    return {"bbox_w": float(b[2] - b[0]),
            "bbox_h": float(b[3] - b[1]),
            "eye_mouth": float(np.linalg.norm(mouth_mid - eye_mid))}


def nearest(faces, cx, cy):
    best, bd = None, float("inf")
    for f in faces:
        b = f.bbox
        d = ((float(b[0]) + float(b[2])) / 2 - cx) ** 2 + \
            ((float(b[1]) + float(b[3])) / 2 - cy) ** 2
        if d < bd:
            best, bd = f, d
    return best, bd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--provider", default="tensorrt")
    args = ap.parse_args()

    ab.init_pipeline(args.provider, "inswapper", "None", "None")
    from roop.face_util import get_all_faces

    clip = args.clip if os.path.exists(args.clip) else os.path.join(CLIPS, args.clip)
    cp, co = cv2.VideoCapture(clip), cv2.VideoCapture(out_video(args.arm))
    cp.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    co.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    series = []
    for i in range(args.count):
        okp, plate = cp.read()
        oko, outp = co.read()
        if not (okp and oko):
            break
        pf = sorted(get_all_faces(plate) or [], key=lambda f: float(f.bbox[0]))
        of = get_all_faces(outp) or []
        if not pf or not of:
            continue
        for f in pf:
            b = f.bbox
            cx = (float(b[0]) + float(b[2])) / 2
            cy = (float(b[1]) + float(b[3])) / 2
            m, d = nearest(of, cx, cy)
            if m is None or d > ((float(b[2]) - float(b[0])) * 0.6) ** 2:
                continue
            mp, mo = measures(f), measures(m)
            series.append({"frame": args.start + i, "x": cx,
                           **{f"p_{k}": v for k, v in mp.items()},
                           **{f"o_{k}": v for k, v in mo.items()}})
    cp.release()
    co.release()

    if len(series) < 6:
        raise SystemExit(f"only {len(series)} comparable faces; pick another range")

    xs = np.array([r["x"] for r in series])
    groups = {"all": series}
    if xs.max() - xs.min() > 100:
        mid = (xs.max() + xs.min()) / 2
        groups = {"left": [r for r in series if r["x"] < mid],
                  "right": [r for r in series if r["x"] >= mid]}

    print(f"\nframes {args.start}..{args.start + args.count - 1}, arm {args.arm}\n")
    for name, rows in groups.items():
        if len(rows) < 6:
            continue
        print(f"  {name}: {len(rows)} faces")
        print(f"    {'measure':<12}{'plate swing':>13}{'ratio mean':>12}"
              f"{'ratio sd':>10}{'ratio swing':>13}")
        for k in ("bbox_w", "bbox_h", "eye_mouth"):
            pv = np.array([r[f"p_{k}"] for r in rows])
            ov = np.array([r[f"o_{k}"] for r in rows])
            ratio = ov / pv
            ps = (pv.max() - pv.min()) / pv.mean()
            print(f"    {k:<12}{100 * ps:>12.1f}%{ratio.mean():>12.4f}"
                  f"{100 * ratio.std(ddof=1) / ratio.mean():>9.2f}%"
                  f"{100 * (ratio.max() - ratio.min()) / ratio.mean():>12.1f}%")
        print()
    print("plate swing = how much the face genuinely changes size over the run.")
    print("ratio sd/swing = how much the SWAP changes it on top of that.")
    print("Pumping is a ratio that moves while the plate's own size is steady.")


if __name__ == "__main__":
    main()
