"""Side-by-side PLATE vs OUTPUT face crops, tiled into PNG sheets to LOOK at.

Requirements 3, 5, 10, 16 and 17 of the phase-4 checklist are appearance
questions -- is a hand painted over, is a mole kept, do the lips double when the
mouth opens -- and no metric in this project answers them. They need eyes on
pixels. Every existing visual tool here (verify_swap_visual, verify_swap_visual2)
measures identity numerically instead; this one produces images.

Face boxes come from an arm's rows.csv, so there is no re-detection and the
plate and the output are cropped on exactly the same rectangle -- any difference
in the pair is the swap, not a detector disagreement.

Usage:
    env/Scripts/python.exe tests/review_sheet.py --arm d5_platecrop --clip d5.mp4
    env/Scripts/python.exe tests/review_sheet.py --arm d5_platecrop --clip d5.mp4 \
        --frames 120,340,900 --tag mouth
"""

import argparse
import csv
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ARMS = os.path.join(APP, "output", "bench_two_face")
CLIPS = r"G:/pinokio/roop-keep/double"


def rows_for(tag):
    with open(os.path.join(ARMS, tag, "rows.csv"), newline="",
              encoding="utf-8") as f:
        return list(csv.DictReader(f))


def out_video(tag):
    work = os.path.join(ARMS, tag, "work")
    v = [f for f in os.listdir(work) if f.endswith(".mp4") and "resume" not in f]
    v.sort(key=lambda f: os.path.getsize(os.path.join(work, f)))
    return os.path.join(work, v[-1])


def crop(img, box, pad=0.35, size=320):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx, cy = x0 + w / 2.0, y0 + h / 2.0
    half = max(w, h) * (1 + pad) / 2.0
    a, b = int(cx - half), int(cy - half)
    c, d = int(cx + half), int(cy + half)
    H, W = img.shape[:2]
    pl, pt = max(0, -a), max(0, -b)
    pr, pb = max(0, c - W), max(0, d - H)
    a, b, c, d = max(0, a), max(0, b), min(W, c), min(H, d)
    if c - a < 8 or d - b < 8:
        return None
    sub = img[b:d, a:c]
    if pl or pt or pr or pb:
        sub = cv2.copyMakeBorder(sub, pt, pb, pl, pr, cv2.BORDER_CONSTANT,
                                 value=(20, 20, 20))
    return cv2.resize(sub, (size, size), interpolation=cv2.INTER_CUBIC)


def label(img, text, colour=(255, 255, 255)):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,
                cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frames", default="", help="comma-separated; default = an even spread")
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--per-sheet", type=int, default=4)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--tag", default="spread")
    ap.add_argument("--person", type=int, default=None)
    args = ap.parse_args()

    clip = args.clip if os.path.exists(args.clip) else os.path.join(CLIPS, args.clip)
    rows = rows_for(args.arm)
    by_frame = {}
    for r in rows:
        if args.person is not None and str(r.get("person")) != str(args.person):
            continue
        try:
            by_frame.setdefault(int(r["frame"]), []).append(
                (int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"]),
                 r.get("why", ""), r.get("who", "")))
        except (TypeError, ValueError):
            continue

    if args.frames.strip():
        want = [int(x) for x in args.frames.split(",") if x.strip()]
    else:
        keys = sorted(by_frame)
        step = max(1, len(keys) // args.count)
        want = keys[::step][:args.count]

    cap_p, cap_o = cv2.VideoCapture(clip), cv2.VideoCapture(out_video(args.arm))
    pairs = []
    for fr in want:
        cap_p.set(cv2.CAP_PROP_POS_FRAMES, fr)
        cap_o.set(cv2.CAP_PROP_POS_FRAMES, fr)
        okp, plate = cap_p.read()
        oko, outp = cap_o.read()
        if not (okp and oko):
            continue
        for (x0, y0, x1, y1, why, who) in by_frame.get(fr, []):
            cp = crop(plate, (x0, y0, x1, y1), size=args.size)
            co = crop(outp, (x0, y0, x1, y1), size=args.size)
            if cp is None or co is None:
                continue
            sw = "swapped" if "swapped (" in why else "NOT SWAPPED"
            row = np.hstack([label(cp.copy(), f"f{fr} PLATE p{who}"),
                             label(co.copy(), f"f{fr} OUTPUT [{sw}]",
                                   (120, 255, 120) if sw == "swapped"
                                   else (120, 120, 255))])
            pairs.append(row)
    cap_p.release()
    cap_o.release()

    if not pairs:
        raise SystemExit("no frames rendered")

    outdir = os.path.join(ARMS, args.arm, "review")
    os.makedirs(outdir, exist_ok=True)
    written = []
    for i in range(0, len(pairs), args.per_sheet):
        chunk = pairs[i:i + args.per_sheet]
        sheet = np.vstack(chunk)
        p = os.path.join(outdir, f"{args.tag}_{i // args.per_sheet:02d}.png")
        cv2.imwrite(p, sheet)
        written.append(p)
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
