"""Dump specific timestamps from a plate + swapped video pair as PNGs for
visual inspection, side by side. Time-based (not frame-index) so it works
correctly across mismatched fps between plate and output.

Usage:
    env/Scripts/python.exe tests/extract_frames.py --plate P.mp4 --swapped S.mp4 \
        --times 0,30,90,150 --out G:/pinokio/cache/TEMP/.../scratchpad/frames
"""
import argparse
import os
import cv2


def grab(path, t_sec, out_path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t_sec * fps)))
    ok, fr = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(out_path, fr)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", required=True)
    ap.add_argument("--swapped", required=True)
    ap.add_argument("--times", required=True, help="comma-separated seconds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="frame")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for t in [float(x) for x in args.times.split(",")]:
        pp = os.path.join(args.out, f"{args.tag}_{t:.0f}s_plate.png")
        sp = os.path.join(args.out, f"{args.tag}_{t:.0f}s_swapped.png")
        ok1 = grab(args.plate, t, pp)
        ok2 = grab(args.swapped, t, sp)
        print(f"t={t:.0f}s plate={'OK' if ok1 else 'FAIL'} swapped={'OK' if ok2 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
