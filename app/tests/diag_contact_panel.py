"""Look at the frames the contact tier decided, side by side with the plate.

The one check that is NOT circular. Whether a face in contact was swapped as
the right person cannot be settled by an embedding: the crop whose identity is
in question is the contaminated one, which is the reason the tier exists
(face_contact.py -- "on two faces in contact that measurement is exactly the
contaminated one this file now refuses to trust"). So the evidence has to be
the picture.

Builds one PNG per sampled frame: the original plate, the render with the tier
off, and the render with it on, cropped around the face and captioned. A wrong
identity, a face painted onto the junction between two heads, or a swap pasted
over the neighbour is visible at a glance and invisible to every number.

Usage:
    env/Scripts/python.exe tests/diag_contact_panel.py \
        --plate "<clip>.mp4" --off <off>.mp4 --on <on>.mp4 \
        --start 4600 --frames 4650,4700,4750 --out output/contact_panel
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


def frame_at(path, index):
    cap = cv2.VideoCapture(path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, img = cap.read()
        return img if ok else None
    finally:
        cap.release()


def label(img, text, height=34):
    bar = np.full((height, img.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, text, (10, height - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", required=True, help="the original clip")
    ap.add_argument("--off", required=True, help="render with the tier off")
    ap.add_argument("--on", required=True, help="render with the tier on")
    ap.add_argument("--start", type=int, default=0,
                    help="first plate frame the two renders correspond to")
    ap.add_argument("--frames", required=True,
                    help="comma-separated PLATE frame indices to sample")
    ap.add_argument("--crop", default="",
                    help="optional x,y,w,h to zoom into; default full frame")
    ap.add_argument("--width", type=int, default=560,
                    help="per-panel width in the montage")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    box = [int(v) for v in args.crop.split(",")] if args.crop.strip() else None

    for n in [int(x) for x in args.frames.split(",") if x.strip()]:
        panels = []
        for path, name, idx in ((args.plate, "original", n),
                                (args.off, "tier OFF (shipped)", n - args.start),
                                (args.on, "tier ON", n - args.start)):
            img = frame_at(path, idx)
            if img is None:
                print(f"[panel] frame {n}: {name} has no frame {idx}", flush=True)
                break
            if box:
                x, y, w, h = box
                img = img[max(0, y):y + h, max(0, x):x + w]
            scale = args.width / float(img.shape[1])
            img = cv2.resize(img, (args.width, max(1, int(img.shape[0] * scale))))
            panels.append(label(img, f"{name}  ·  frame {n}"))
        if len(panels) != 3:
            continue
        h = max(p.shape[0] for p in panels)
        panels = [np.vstack([p, np.full((h - p.shape[0], p.shape[1], 3), 24, np.uint8)])
                  if p.shape[0] < h else p for p in panels]
        out = os.path.join(args.out, f"frame_{n:06d}.png")
        cv2.imwrite(out, np.hstack(panels))
        print(f"[panel] {out}", flush=True)


if __name__ == "__main__":
    main()
