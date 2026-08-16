"""Ground-truth check: did an output video actually get a face swapped in, or
does it just look like the plate while the pipeline's own SWAP AUDIT claims
success?

Session 3 of the roop-recode project: SWAP AUDIT reported 564/568 (99.3%) of
s1.mp4's frames as "swapped", but the user watched the output and reported "not
swapped at all". This script settles the disagreement by actually measuring
pixel change and post-swap identity in the face box, frame by frame, instead of
trusting either side's summary.

Usage:
    env/Scripts/python.exe tests/verify_swap_visual.py \
        --plate "G:/pinokio/roop-keep/single/s1.mp4" \
        --swapped "G:/pinokio/api/roop-ultimate/app/output/baseline_single/s1__harjot.mp4" \
        --sources harjot
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

import angle_bench as ab                                             # noqa: E402
from two_face_video import load_library_faceset, cos, faceset_mean  # noqa: E402
from sample_bench import first_face_frame, select_primary_face       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", required=True)
    ap.add_argument("--swapped", required=True)
    ap.add_argument("--sources", required=True, help="comma-separated faceset name(s)")
    ap.add_argument("--samples", type=int, default=25)
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()

    # Face detection/recognition models must be brought up the same way the
    # real pipeline does (prepare_environment etc.) before get_all_faces() or
    # faceset ingestion will find anything — this is a read-only verification
    # pass, so swap_model/mask_engine/enhancer don't matter, only the detector.
    from settings import Settings
    cfg_probe = Settings("config.yaml")
    provider = args.provider or cfg_probe.provider or "cuda"
    ab.init_pipeline(provider, cfg_probe.swap_model, "None", None)

    from roop.face_util import get_all_faces

    names = [s.strip() for s in args.sources.split(",") if s.strip()]
    facesets = [load_library_faceset(n) for n in names]
    means = [faceset_mean(fs) for fs in facesets]

    cap_idx, cap_frame, faces = first_face_frame(args.plate)
    target = select_primary_face(faces)
    own_mean = np.asarray(target.embedding, np.float64).ravel()
    print(f"[verify] target captured from plate frame {cap_idx}", flush=True)

    cap_p = cv2.VideoCapture(args.plate)
    cap_s = cv2.VideoCapture(args.swapped)
    fps_p = cap_p.get(cv2.CAP_PROP_FPS) or 30.0
    fps_s = cap_s.get(cv2.CAP_PROP_FPS) or 30.0
    n_p = int(cap_p.get(cv2.CAP_PROP_FRAME_COUNT))
    n_s = int(cap_s.get(cv2.CAP_PROP_FRAME_COUNT))
    dur_p = n_p / fps_p
    dur_s = n_s / fps_s
    dur = min(dur_p, dur_s)
    print(f"[verify] plate frames={n_p} @ {fps_p:.2f}fps ({dur_p:.1f}s), "
          f"swapped frames={n_s} @ {fps_s:.2f}fps ({dur_s:.1f}s) — aligning by TIME, "
          f"not frame index, since fps differs", flush=True)

    times = np.linspace(0, dur - (1.0 / min(fps_p, fps_s)), args.samples)
    idxs = [(int(round(t * fps_p)), int(round(t * fps_s))) for t in times]

    rows = []
    for idx_p, idx_s in idxs:
        idx = idx_p  # reported/plotted against the plate's own frame numbering
        cap_p.set(cv2.CAP_PROP_POS_FRAMES, idx_p)
        cap_s.set(cv2.CAP_PROP_POS_FRAMES, idx_s)
        ok_p, fp = cap_p.read()
        ok_s, fs = cap_s.read()
        if not ok_p or not ok_s:
            continue
        pfaces = get_all_faces(fp) or []
        if not pfaces:
            rows.append((idx, None, None, None, "no-face-in-plate"))
            continue
        pf = max(pfaces, key=lambda f: (float(f.bbox[2]) - float(f.bbox[0])))
        x0, y0, x1, y1 = [int(round(v)) for v in pf.bbox]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(fp.shape[1], x1), min(fp.shape[0], y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            rows.append((idx, None, None, None, "box-too-small"))
            continue
        touched = float(np.abs(fp[y0:y1, x0:x1].astype(np.float32)
                                - fs[y0:y1, x0:x1].astype(np.float32)).mean())
        sfaces = get_all_faces(fs) or []
        if not sfaces:
            rows.append((idx, touched, None, None, "no-face-in-swapped"))
            continue
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        best = min(sfaces, key=lambda f: ((float(f.bbox[0]) + float(f.bbox[2])) / 2.0 - cx) ** 2
                   + ((float(f.bbox[1]) + float(f.bbox[3])) / 2.0 - cy) ** 2)
        emb = np.asarray(best.embedding, np.float64).ravel()
        d_own = cos(emb, own_mean)
        d_src = [cos(emb, m) if m is not None else float("nan") for m in means]
        rows.append((idx, touched, d_own, d_src, None))

    print(f"\n{'frame':>6} {'touched':>8} {'d_own':>7} " +
          " ".join(f"d_{n}" for n in names) + "   note", flush=True)
    for idx, touched, d_own, d_src, note in rows:
        if touched is None or d_own is None:
            print(f"{idx:6d} {'--':>8} {'--':>7}   {note or ''}", flush=True)
            continue
        src_str = " ".join(f"{d:.3f}" for d in d_src) if d_src else "--"
        print(f"{idx:6d} {touched:8.2f} {d_own:7.3f} {src_str}   {note or ''}", flush=True)

    touched_vals = [r[1] for r in rows if r[1] is not None]
    d_own_vals = [r[2] for r in rows if r[2] is not None]
    if touched_vals:
        print(f"\n[verify] mean pixel change in face box: {np.mean(touched_vals):.2f} "
              f"(near 0 = box literally unchanged)", flush=True)
    if d_own_vals:
        closer_to_own = sum(1 for idx, touched, d_own, d_src, note in rows
                             if d_own is not None and d_src and d_own < min(d_src))
        print(f"[verify] frames where output face is closer to the ORIGINAL person "
              f"than to any source faceset: {closer_to_own} / {len(d_own_vals)}", flush=True)


if __name__ == "__main__":
    main()
