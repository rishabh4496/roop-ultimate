"""Dump the actual frame images capture_targets_best_frontal() would pick for
each person, for visual sanity-checking a reference capture (does the
off-axis-degrees NUMBER match what the pixels actually look like -- past
lessons in this project say don't trust the metric alone). Re-implements the
same scan as `two_face_video.capture_targets_best_frontal` but additionally
keeps the raw frame + bbox so a crop can be saved to disk.

Usage:
    env/Scripts/python.exe tests/diag_dump_capture.py --video PATH --out DIR
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()

    import angle_bench as ab
    from angle_video import ensure_ffmpeg
    from settings import Settings
    from two_face_video import capture_targets, separated_frame_with_fallback, _landmarks_plausible
    from roop.face_util import get_all_faces, solve_pose_5pt, offaxis_deg, align_crop
    from roop.utilities import compute_cosine_distance

    ensure_ffmpeg()
    cfg = Settings("config.yaml")
    provider = args.provider or cfg.provider or "cuda"
    ab.init_pipeline(provider, cfg.swap_model, "None", None)

    cap_idx, cap_frame = separated_frame_with_fallback(args.video)
    seed_targets, seed_groups = capture_targets(cap_frame)
    seed_embs = {g_: np.asarray(seed_targets[i].embedding, np.float64)
                 for i, g_ in enumerate(seed_groups)}
    print(f"[diag] seed frame {cap_idx}, groups={seed_groups}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    for i, grp in enumerate(seed_groups):
        crop, _ = align_crop(cap_frame, seed_targets[i].kps, 512, mode="arcface")
        cv2.imwrite(os.path.join(args.out, f"seed_person_{grp}.png"), crop)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    samples = int(min(2000, max(200, (total / fps) * 3)))
    stride = max(1, total // samples)

    best = {}  # group -> (offaxis_deg, fpos, face, frame)
    for fpos in range(0, total, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fpos)
        ok, fr = cap.read()
        if not ok:
            continue
        for f in (get_all_faces(fr) or []):
            e = getattr(f, 'embedding', None)
            if e is None:
                continue
            e = np.asarray(e, np.float64)
            g_best, d_best = None, 1e9
            for grp2, se in seed_embs.items():
                d = compute_cosine_distance(e, se)
                if d < d_best:
                    d_best, g_best = d, grp2
            if d_best > 0.85:
                continue
            if float(f.bbox[2]) - float(f.bbox[0]) < 60:
                continue
            if not _landmarks_plausible(f.bbox, f.kps, det_score=float(getattr(f, 'det_score', 1.0))):
                continue
            pose = solve_pose_5pt(f.kps)
            if pose is None:
                continue
            off = offaxis_deg(pose[0], pose[1])
            cur = best.get(g_best)
            if cur is None or off < cur[0]:
                best[g_best] = (off, fpos, f, fr.copy())
    cap.release()

    for grp, (off, fpos, f, fr) in sorted(best.items()):
        print(f"[diag] person {grp}: best frame {fpos} ({fpos/fps:.1f}s), off-axis {off:.1f} deg", flush=True)
        crop, _ = align_crop(fr, f.kps, 512, mode="arcface")
        cv2.imwrite(os.path.join(args.out, f"best_person_{grp}.png"), crop)

    print(f"[diag] saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
