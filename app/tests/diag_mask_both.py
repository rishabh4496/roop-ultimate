"""Like diag_mask.py but dumps the aligned crop + XSeg mask + overlay for
EVERY detected face in the frame (not just the widest), for investigating
mask boundary behaviour when two faces are close/touching (d7 "nose
disappears" complaint).

Usage:
    env/Scripts/python.exe tests/diag_mask_both.py --video PATH --time 37.4 --out DIR
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--time", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()

    import angle_bench as ab
    from settings import Settings
    from roop.face_util import get_all_faces, align_crop

    cfg = Settings("config.yaml")
    provider = args.provider or cfg.provider or "cuda"
    g = ab.init_pipeline(provider, cfg.swap_model, "None", None)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(args.time * fps)))
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("could not read frame")

    faces = get_all_faces(frame) or []
    if not faces:
        raise SystemExit("no face detected")
    faces = sorted(faces, key=lambda f: float(f.bbox[0]))
    print(f"[diag] {len(faces)} faces found", flush=True)

    from roop.processors.Mask_XSeg import Mask_XSeg
    xseg = Mask_XSeg()
    xseg.Initialize({"devicename": "cuda"})

    os.makedirs(args.out, exist_ok=True)
    for i, face in enumerate(faces):
        print(f"[diag] face {i} bbox={[round(float(v),1) for v in face.bbox]}", flush=True)
        aligned, M = align_crop(frame, face.kps, 512, mode="arcface")
        cv2.imwrite(os.path.join(args.out, f"aligned_crop_{i}.png"), aligned)

        mask = xseg.Run(aligned, "")
        mask_vis = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        mask_vis = cv2.resize(mask_vis, (512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(args.out, f"xseg_mask_{i}.png"), mask_vis)

        overlay = aligned.copy()
        contours, _ = cv2.findContours((mask_vis > 25).astype(np.uint8),
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(args.out, f"mask_overlay_{i}.png"), overlay)

    print(f"[diag] saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
