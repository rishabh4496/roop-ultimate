"""Bisect d11's confirmed t~24-26s swap-dropout: trim the SOURCE clip to N
frames and run it through the exact same capture+swap pipeline
`run_all_samples.py` uses for the real `baseline_double` output (not
`two_face_video.py`'s simplified capture, which does NOT reproduce this bug —
confirmed by a mismatched first attempt), to find the shortest length that
still reproduces the dropout.

Usage:
    env/Scripts/python.exe tests/diag_d11_bisect.py --end 2000
    env/Scripts/python.exe tests/diag_d11_bisect.py --end 4000
"""
import argparse
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import fixtures
import run_all_samples as ras  # noqa: E402
from angle_video import ensure_ffmpeg  # noqa: E402
from two_face_video import load_library_faceset  # noqa: E402

SRC = fixtures.clip('double/d11.mp4')
NAMES = ["harjot", "shambhavi"]


def trim(video, end, out_path):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = 0
    while i < end:
        ok, fr = cap.read()
        if not ok:
            break
        vw.write(fr)
        i += 1
    cap.release()
    vw.release()
    print(f"[bisect] trimmed {i} frames -> {out_path}", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--out", default=os.path.join(APP, "output", "d11_bisect"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    clip = trim(SRC, args.end, os.path.join(args.out, f"d11_trim{args.end}.mp4"))

    ensure_ffmpeg()
    g, options = ras._pipeline()
    facesets = [load_library_faceset(n) for n in NAMES]
    tag_dir = os.path.join(args.out, f"n{args.end}")
    os.makedirs(tag_dir, exist_ok=True)
    ras.run_one(clip, NAMES, facesets, options, tag_dir)


if __name__ == "__main__":
    main()
