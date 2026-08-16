"""Ground-truth check for 2-person clips: does the SAME screen position keep
the SAME source identity throughout, or does it interchange?

Independent of the pipeline's own track/source bookkeeping -- detects faces
directly in the swapped output, matches each to whichever faceset it most
resembles, and reports a timeline of (left/right position -> identity) so any
interchange or unswapped stretch is directly visible, not inferred from logs.

Usage:
    env/Scripts/python.exe tests/verify_swap_visual2.py \
        --swapped output/baseline_double/d2__harjot-shambhavi.mp4 \
        --sources harjot,shambhavi --samples 60
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swapped", required=True)
    ap.add_argument("--sources", required=True, help="comma-separated, e.g. harjot,shambhavi")
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()

    from settings import Settings
    cfg_probe = Settings("config.yaml")
    provider = args.provider or cfg_probe.provider or "cuda"
    ab.init_pipeline(provider, cfg_probe.swap_model, "None", None)

    from roop.face_util import get_all_faces

    names = [s.strip() for s in args.sources.split(",") if s.strip()]
    facesets = [load_library_faceset(n) for n in names]
    means = [faceset_mean(fs) for fs in facesets]

    cap = cv2.VideoCapture(args.swapped)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    dur = n / fps

    idxs = sorted(set(int(round(t * fps)) for t in np.linspace(0, dur - 1.0 / fps, args.samples)))

    print(f"\n{'t(s)':>6} {'#faces':>7}  left-position                right-position", flush=True)
    header_names = "/".join(names)
    left_hist, right_hist = [], []
    n_no_face = 0
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok:
            continue
        faces = get_all_faces(fr) or []
        faces = sorted(faces, key=lambda f: float(f.bbox[0]))
        t = idx / fps
        if not faces:
            n_no_face += 1

        # A face that was never swapped (still the plate actor) is close to
        # NEITHER faceset -- forcing it to whichever mean is marginally
        # nearer manufactures a fake "flip" every time the unswapped face's
        # pose/lighting changes, which has nothing to do with identity.
        # UNSWAPPED_FLOOR is the same territory as the pipeline's own 0.6
        # same-person gate: above it, neither candidate is a real match.
        UNSWAPPED_FLOOR = 0.75

        def _describe(f):
            emb = getattr(f, 'embedding', None)
            if emb is None:
                return "no-embedding", None
            emb = np.asarray(emb, np.float64)
            ds = [cos(emb, m) if m is not None else float("nan") for m in means]
            best = int(np.argmin(ds))
            gap = sorted(ds)[1] - sorted(ds)[0] if len(ds) > 1 else 0.0
            if ds[best] > UNSWAPPED_FLOOR:
                return f"UNSWAPPED? d={ds[best]:.2f}", None
            ambiguous = " (AMBIGUOUS)" if gap < 0.08 else ""
            return f"{names[best]} d={ds[best]:.2f}{ambiguous}", names[best]

        left, left_id = _describe(faces[0]) if len(faces) >= 1 else ("MISSING", None)
        right, right_id = _describe(faces[1]) if len(faces) >= 2 else (("MISSING", None) if len(faces) >= 1 else ("", None))
        # "left"/"right" is whichever face is currently leftmost/rightmost --
        # when only ONE person is on screen they become "leftmost" by default
        # even if they are really the usual right-hand person, which reads as
        # a fake flip on both edges of the gap. Only trust the position label
        # on frames where both people are actually present.
        left_hist.append(left_id if len(faces) >= 2 else None)
        right_hist.append(right_id if len(faces) >= 2 else None)
        print(f"{t:6.1f} {len(faces):7d}  {left:<28} {right}", flush=True)

    def _flip_count(hist):
        hist = [h for h in hist if h is not None]
        return sum(1 for a, b in zip(hist, hist[1:]) if a != b)

    def _confident(hist):
        return [h for h in hist if h is not None]

    print(f"\n[verify2] left-position identity changes across CONFIDENTLY-matched "
          f"consecutive samples: {_flip_count(left_hist)} / "
          f"{max(0, len(_confident(left_hist))-1)} "
          f"({len(left_hist)-len(_confident(left_hist))} unswapped/no-embedding samples excluded)",
          flush=True)
    print(f"[verify2] right-position identity changes across CONFIDENTLY-matched "
          f"consecutive samples: {_flip_count(right_hist)} / "
          f"{max(0, len(_confident(right_hist))-1)} "
          f"({len(right_hist)-len(_confident(right_hist))} unswapped/no-embedding samples excluded)",
          flush=True)
    print(f"[verify2] samples with 0 faces detected: {n_no_face} / {len(idxs)}", flush=True)


if __name__ == "__main__":
    main()
