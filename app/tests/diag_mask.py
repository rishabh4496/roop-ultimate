"""Render the actual XSeg mask (and the aligned crop it runs on) for one exact
frame, to check whether the mask is cutting off the forehead at an extreme
backward-pitch pose -- the leading theory for s4's "small face" distortion
after two alignment-scale hypotheses were rejected.

Usage:
    env/Scripts/python.exe tests/diag_mask.py --video PATH --time 119.577
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
    face = max(faces, key=lambda f: (float(f.bbox[2]) - float(f.bbox[0])))
    print(f"[diag] bbox={[round(float(v),1) for v in face.bbox]}", flush=True)

    aligned, M = align_crop(frame, face.kps, 512, mode="arcface")
    os.makedirs(args.out, exist_ok=True)
    cv2.imwrite(os.path.join(args.out, "aligned_crop.png"), aligned)

    from roop.processors.Mask_XSeg import Mask_XSeg
    xseg = Mask_XSeg()
    xseg.Initialize({"devicename": "cuda"})
    mask = xseg.Run(aligned, "")
    mask_vis = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    mask_vis = cv2.resize(mask_vis, (512, 512), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(args.out, "xseg_mask.png"), mask_vis)

    # Overlay: mask boundary drawn on the aligned crop for direct comparison.
    overlay = aligned.copy()
    contours, _ = cv2.findContours((mask_vis > 25).astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(args.out, "mask_overlay.png"), overlay)
    print(f"[diag] saved aligned_crop.png, xseg_mask.png, mask_overlay.png to {args.out}", flush=True)


if __name__ == "__main__":
    main()
