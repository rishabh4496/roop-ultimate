"""Two-person target capture that picks each person's MOST FRONTAL frame
independently, rather than one shared frame (which may not have both people
looking reasonably frontal at the same time -- measured true on d1.mp4, a
"neck rotation" exercise video that alternates extreme up/down/profile poses
for both people through its entire runtime, with individual close-ups of
each person occurring at DIFFERENT times than the wide two-shot).

Seeds identity from the existing capture_targets() (one shared frame, left-
right), then scans the whole video the way auto_angles does, tracks the
best (lowest off-axis) frame found FOR EACH PERSON SEPARATELY by nearest
embedding to that seed, and replaces each person's single target angle with
their own best frame's capture.

Usage:
    env/Scripts/python.exe tests/diag_best_frontal_capture.py --video PATH \
        --swap-video PATH --sources harjot,shambhavi
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="video to scan for the best frame per person")
    ap.add_argument("--swap-video", default=None)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()
    swap_video = args.swap_video or args.video

    import angle_bench as ab
    from angle_video import ensure_ffmpeg
    from settings import Settings
    from two_face_video import load_library_faceset, separated_frame, capture_targets
    from roop.face_util import get_all_faces, solve_pose_5pt, offaxis_deg
    from roop.utilities import compute_cosine_distance
    import roop.globals as g
    import cv2

    ensure_ffmpeg()
    cfg = Settings("config.yaml")
    provider = args.provider or cfg.provider or "cuda"
    ab.init_pipeline(provider, cfg.swap_model, "None", None)

    cap_idx, cap_frame = separated_frame(swap_video)
    seed_targets, seed_groups = capture_targets(cap_frame)
    seed_embs = {g_: np.asarray(seed_targets[i].embedding, np.float64)
                 for i, g_ in enumerate(seed_groups)}
    print(f"[diag] seed identities from frame {cap_idx}: groups={seed_groups}", flush=True)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, total // args.samples)

    best = {}  # group -> (offaxis_deg, frame_idx, face)
    for fpos in range(0, total, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fpos)
        ok, fr = cap.read()
        if not ok:
            continue
        faces = get_all_faces(fr) or []
        for f in faces:
            e = getattr(f, 'embedding', None)
            if e is None:
                continue
            e = np.asarray(e, np.float64)
            g_best, d_best = None, 1e9
            for grp, se in seed_embs.items():
                d = compute_cosine_distance(e, se)
                if d < d_best:
                    d_best, g_best = d, grp
            if d_best > 0.85:
                continue  # not clearly either seeded person
            pose = solve_pose_5pt(f.kps)
            if pose is None:
                continue
            off = offaxis_deg(pose[0], pose[1])
            bw = float(f.bbox[2]) - float(f.bbox[0])
            if bw < 60:
                continue  # too small to be a reliable pose read
            cur = best.get(g_best)
            if cur is None or off < cur[0]:
                best[g_best] = (off, fpos, f, fr.copy())
    cap.release()

    for grp, (off, fpos, f, fr) in sorted(best.items()):
        print(f"[diag] person {grp}: best frame {fpos}, off-axis {off:.1f} deg", flush=True)

    new_targets, new_groups = [], []
    for grp in seed_groups if len(best) < len(set(seed_groups)) else sorted(best.keys()):
        if grp in best:
            new_targets.append(best[grp][2])
        else:
            idx = seed_groups.index(grp)
            new_targets.append(seed_targets[idx])
        new_groups.append(grp)

    g.TARGET_FACES = new_targets
    g.TARGET_FACE_GROUP = new_groups

    from sample_bench import run_swap, map_mask_engine
    g.video_encoder = g.CFG.output_video_codec
    g.video_quality = g.CFG.video_quality
    g.execution_threads = 8
    g.face_swap_mode = "selected"
    g.track_identities = bool(g.CFG.track_identities)
    g.CFG.track_identities = g.track_identities
    g.temporal_detection = bool(g.CFG.temporal_detection)
    g.CFG.temporal_detection = g.temporal_detection
    g.stabilize_face = bool(g.CFG.stabilize_face)
    options = ab.build_options(g, cfg.swap_model, map_mask_engine(cfg.mask_engine),
                               bool(g.CFG.use_source_bank))
    options.stabilize_face = g.stabilize_face

    names = [s.strip() for s in args.sources.split(",") if s.strip()]
    facesets = [load_library_faceset(n) for n in names]
    out_dir = os.path.join(APP, "output", "best_frontal_diag")
    out, elapsed, face_log = run_swap(swap_video, facesets, new_targets, new_groups, options, out_dir)
    print(f"[diag] swap with best-frontal targets -> {out} in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
