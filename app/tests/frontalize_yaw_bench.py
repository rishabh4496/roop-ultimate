"""Frontalize-by-yaw bench — is the existing affine frontalizer
(`use_frontalization`, roop/face_frontalize.py) better or worse than no
frontalization, per yaw band, on real footage?

The question this answers is whether extreme-yaw work needs a NEW unwarp. A
broader pose-dependent alignment was shipped and deleted on 2026-08-08 after
costing 0.07-0.11 identity at every yaw, so nothing new goes in until this
says the current one is the limit.

Frames come from real clips, binned by |yaw| from solve_pose_5pt (good to
~15-20 deg per person, see pose-solve-head-shape-limit: fine for BINNING, not
for gating). Each frame is graded with angle_bench.grade_frame, the same
drift / identity / ghost columns the angle sweeps use. Arms:

  off        use_frontalization False
  front_30   on, the shipped default threshold (30 deg)
  front_60   on, only past 60 deg -- the "extreme yaw only" gate

    app/env/Scripts/python.exe tests/frontalize_yaw_bench.py --faceset akansha
"""
import argparse
import csv
import glob
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
sys.path.insert(0, APP)
sys.path.insert(0, HERE)

import fixtures  # noqa: E402

ARMS = {"off": (False, 30.0), "front_30": (True, 30.0), "front_60": (True, 60.0)}
BANDS = [(0, 30), (30, 45), (45, 60), (60, 75), (75, 91)]


def band(yaw):
    a = abs(yaw)
    for lo, hi in BANDS:
        if lo <= a < hi:
            return f"{lo}-{hi if hi < 91 else 90}"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faceset", default="akansha")
    ap.add_argument("--clips", default=None,
                    help="glob; default angle.mp4 + angle_inverted.mp4 via fixtures")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--per-band", type=int, default=120,
                    help="cap frames per yaw band so frontal frames don't dominate")
    ap.add_argument("--arms", default="off,front_30,front_60,off")
    ap.add_argument("--out", default=os.path.join(APP, "output", "bench_frontalize_yaw"))
    args = ap.parse_args()

    from angle_bench import init_pipeline, build_options, load_faceset, biggest_face, grade_frame
    from two_face_video import map_mask_engine
    import yaml
    with open(os.path.join(APP, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = cfg.get("swap_model", "hyperswap")
    g = init_pipeline(cfg.get("provider", "cuda"), model, cfg.get("selected_enhancer", "None"),
                      cfg.get("mask_engine", "None"), sync_config=True)
    options = build_options(g, model, map_mask_engine(cfg.get("mask_engine", "None")) or "None")
    from roop.core import live_swap
    from roop.face_util import solve_pose_5pt

    src_fs = load_faceset(os.path.join(APP, "facesets", args.faceset + ".fsz"))
    src_embed = np.mean([f.normed_embedding for f in src_fs.faces], axis=0)

    # material: frames binned by yaw, capped per band
    material, counts = [], {}
    if args.clips:
        clips = sorted(glob.glob(args.clips))
    else:
            clips = [fixtures.clip(n, required=True)
                 for n in ("angle.mp4", "angle_inverted.mp4")]
    if not clips:
        raise SystemExit(f"no clips match {args.clips}")
    for clip in clips:
        cap = cv2.VideoCapture(clip)
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % args.stride == 0:
                face = biggest_face(fr)
                pose = solve_pose_5pt(face.kps) if face is not None else None
                if pose is not None:
                    b = band(pose[0])
                    if counts.get(b, 0) < args.per_band:
                        counts[b] = counts.get(b, 0) + 1
                        material.append((os.path.basename(clip), i, fr, face, float(pose[0]), b))
            i += 1
    print(f"[front_yaw] {len(material)} frames; per band {dict(sorted(counts.items()))}; "
          f"model={model}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    allrows = []
    last = time.time()
    for k, arm in enumerate(args.arms.split(",")):
        use, thr = ARMS[arm]
        options.use_frontalization = use
        options.frontalization_threshold = thr
        t0 = time.time()
        for n, (clip, i, fr, face, yaw, b) in enumerate(material):
            out = live_swap(fr.copy(), options, input_facesets=[src_fs])
            if out is None:
                out = fr
            row = grade_frame(fr, out, src_embed, face)
            row.update(arm=arm, rep=k, clip=clip, frame=i, yaw=round(yaw, 1), band=b,
                       swapped=int(float(np.abs(out.astype(np.int16) - fr.astype(np.int16)).mean()) > 0.5))
            allrows.append(row)
            if time.time() - last > 180:
                print(f"  [{arm}] {n + 1}/{len(material)} {(n + 1) / (time.time() - t0):.2f} fps", flush=True)
                last = time.time()
        print(f"[front_yaw] arm {arm} done in {time.time() - t0:.0f}s", flush=True)

    cols = ["arm", "rep", "clip", "frame", "yaw", "band", "swapped", "detected", "nose",
            "mouth", "eyes", "shift", "id_source", "id_plate", "ghost", "note"]
    with open(os.path.join(args.out, "rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(allrows)

    def mean(rows, key):
        v = [float(r[key]) for r in rows if r.get(key) not in ("", None)]
        return float(np.mean(v)) if v else float("nan")

    print(f"\n{'arm':<10}{'rep':>4}{'band':>7}{'n':>5}{'det':>5}{'swap':>6}"
          f"{'id_src':>8}{'nose':>8}{'mouth':>8}{'eyes':>8}{'ghost':>8}")
    for k, arm in enumerate(args.arms.split(",")):
        for lo, hi in BANDS:
            b = f"{lo}-{hi if hi < 91 else 90}"
            rows = [r for r in allrows if r["rep"] == k and r["band"] == b]
            if not rows:
                continue
            print(f"{arm:<10}{k:>4}{b:>7}{len(rows):>5}{sum(r['detected'] for r in rows):>5}"
                  f"{sum(r['swapped'] for r in rows):>6}{mean(rows, 'id_source'):>8.3f}"
                  f"{mean(rows, 'nose'):>8.3f}{mean(rows, 'mouth'):>8.3f}"
                  f"{mean(rows, 'eyes'):>8.3f}{mean(rows, 'ghost'):>8.3f}")


if __name__ == "__main__":
    main()
