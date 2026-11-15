"""Scan a clip for the head-yaw / motion distribution the matte constants act on.

Run BEFORE tuning (AGENTS.md: measure the distribution the gate reads). Detects the
largest face every STRIDE frames with the app's own detector, stores kps + yaw proxy,
prints the yaw histogram and the 600-frame window richest in both frontal (<15 deg)
and profile (>35 deg) frames.

    env/Scripts/python.exe tests/tune_matte_scan.py --video D:/jennifer.mp4 --out <npz>
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--window", type=int, default=600)
    a = ap.parse_args()
    video = os.path.abspath(a.video)

    import sample_bench as sb                      # applies perf env first
    import angle_bench as ab
    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    ab.init_pipeline(cfg.provider or "cuda", cfg.swap_model, cfg.selected_enhancer or "None",
                     sb.map_mask_engine(cfg.mask_engine))
    from roop.face_util import get_all_faces
    from roop.unified_matte import yaw_from_kps

    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    kps = np.full((n, 5, 2), np.nan, np.float32)
    yaw = np.full(n, np.nan, np.float32)
    size = np.full(n, np.nan, np.float32)
    i = -1
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i % a.stride:
            continue
        faces = get_all_faces(fr) or []
        if not faces:
            continue
        f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        kps[i] = np.asarray(f.kps, np.float32)
        yaw[i] = yaw_from_kps(f.kps)
        size[i] = f.bbox[2] - f.bbox[0]
    cap.release()
    np.savez(a.out, kps=kps, yaw=yaw, size=size, stride=a.stride)

    ok = ~np.isnan(yaw)
    print(f"frames {n}, sampled with a face: {ok.sum()}")
    for lo, hi in ((0, 15), (15, 35), (35, 50), (50, 90)):
        m = ok & (np.abs(yaw) >= lo) & (np.abs(yaw) < hi)
        print(f"  |yaw| {lo:>2}-{hi:<2}: {m.sum():5d} samples ({100 * m.sum() / max(ok.sum(), 1):.1f}%)")
    best, best_s = 0, -1
    for s in range(0, max(n - a.window, 1), a.stride * 5):
        w = slice(s, s + a.window)
        fr_, pr_ = (ok[w] & (np.abs(yaw[w]) < 15)).sum(), (ok[w] & (np.abs(yaw[w]) > 35)).sum()
        score = min(fr_, pr_)
        if score > best_s:
            best, best_s = s, score
    w = slice(best, best + a.window)
    print(f"best {a.window}-frame window starts at {best}: "
          f"frontal {(ok[w] & (np.abs(yaw[w]) < 15)).sum()}, "
          f"profile {(ok[w] & (np.abs(yaw[w]) > 35)).sum()} samples")


if __name__ == "__main__":
    main()
