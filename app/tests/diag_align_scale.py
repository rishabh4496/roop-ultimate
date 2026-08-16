"""Diagnostic: does the align_crop-fitted similarity-transform scale disagree
with the face bbox's own size, specifically at extreme pitch? If a normal
frontal frame's (fitted_scale / bbox_diag) ratio holds steady but an extreme
chin-up frame's ratio drops, that's the mechanism behind "face looks smaller
than it should" at extreme tilt -- the 5-point landmark spacing foreshortens
under pitch faster than the bbox (whole-face detection) does.

Usage:
    env/Scripts/python.exe tests/diag_align_scale.py --video PATH --times 0,19,69,98
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

import angle_bench as ab  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--times", required=True)
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()

    from settings import Settings
    cfg_probe = Settings("config.yaml")
    provider = args.provider or cfg_probe.provider or "cuda"
    ab.init_pipeline(provider, cfg_probe.swap_model, "None", None)

    from roop.face_util import get_all_faces, estimate_norm, solve_pose_5pt

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"\n{'t(s)':>6} {'bbox_w':>7} {'bbox_h':>7} {'bbox_diag':>9} "
          f"{'fit_scale':>9} {'scale*diag':>10} {'yaw':>7} {'pitch':>7}", flush=True)
    for t in [float(x) for x in args.times.split(",")]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, fr = cap.read()
        if not ok:
            print(f"{t:6.1f}  read failed")
            continue
        faces = get_all_faces(fr) or []
        if not faces:
            print(f"{t:6.1f}  no face")
            continue
        face = max(faces, key=lambda f: (float(f.bbox[2]) - float(f.bbox[0])))
        bw = float(face.bbox[2]) - float(face.bbox[0])
        bh = float(face.bbox[3]) - float(face.bbox[1])
        diag = float(np.hypot(bw, bh))
        M = estimate_norm(face.kps, 512, "arcface")
        fit_scale = float(np.hypot(M[0, 0], M[0, 1]))
        pose = solve_pose_5pt(face.kps)
        yaw, pitch = (float(pose[0]), float(pose[1])) if pose is not None else (float("nan"), float("nan"))
        print(f"{t:6.1f} {bw:7.1f} {bh:7.1f} {diag:9.1f} {fit_scale:9.4f} "
              f"{fit_scale*diag:10.2f} {yaw:7.1f} {pitch:7.1f}", flush=True)


if __name__ == "__main__":
    main()
