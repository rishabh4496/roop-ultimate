"""Did the swap land on the RIGHT face? Asked of the OUTPUT, per face.

A pixel diff between two renders cannot answer this. The stabilizers carry
state across frames, so an arm that swaps a face the other arm refused diverges
on later frames and on pixels near the face as well -- which made the change
look, on this clip, as though it had been painted onto the neighbour. (It had
not; the same comparison with every stabilizer off put it squarely on the
intended face.)

The question the diff was standing in for is simpler and can be asked directly:
after the render, does each detected face look like the SOURCE identity? That
comparison is not contaminated the way the target side is -- the source faceset
is a clean set of stills -- so it is evidence where a target-side embedding is
not.

For every sampled frame it prints, per detected face in the OUTPUT, the cosine
distance to the source mean, and flags the ones that changed identity between
two renders. A face that reads as the source in the ON arm and not in the OFF
arm was painted by whatever the ON arm turned on.

Usage:
    env/Scripts/python.exe tests/diag_contact_identity.py \
        --off <off>.mp4 --on <on>.mp4 --source "riya shukla" \
        --start 7880 --stride 10
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

import two_face_video as tfv          # noqa: E402
import angle_bench as ab              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True)
    ap.add_argument("--on", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--start", type=int, default=0, help="plate index of frame 0")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--project", default=None,
                    help="the saved project, for the captured TARGET angles. "
                         "With it, each painted face also reports what the "
                         "PLATE face at that box was -- which is the only way "
                         "to say whether the right person was painted.")
    ap.add_argument("--near", type=float, default=0.60,
                    help="distance under which a face reads as the source")
    args = ap.parse_args()

    tfv._apply_startup_runtime_environment()
    import yaml
    with open(os.path.join(APP, "config.yaml")) as fh:
        cfg = yaml.safe_load(fh) or {}
    ab.init_pipeline(cfg.get("provider") or "cuda",
                     cfg.get("swap_model") or "inswapper", "None", "None",
                     sync_config=True)
    from roop.face_util import get_all_faces
    from roop.utilities import compute_cosine_distance

    fs = tfv.load_library_faceset(args.source)
    src = [np.asarray(f.embedding, np.float32) for f in fs.faces
           if getattr(f, 'embedding', None) is not None]
    print(f"[id] source {args.source}: {len(src)} faces", flush=True)

    tgt = []
    if args.project:
        import json
        with open(args.project, encoding="utf-8") as fh:
            proj = json.load(fh)
        tgt = [np.asarray(t["data"]["embedding"], np.float32)
               for t in (proj.get("inputs", {}).get("target_faces") or [])]
        print(f"[id] captured target angles: {len(tgt)}", flush=True)

    a, b = cv2.VideoCapture(args.off), cv2.VideoCapture(args.on)
    idx = 0
    painted = 0          # faces that became the source in the ON arm
    both = 0             # already the source in both
    rows = 0
    while True:
        oka, fa = a.read()
        okb, fb = b.read()
        if not (oka and okb):
            break
        if idx % args.stride == 0:
            fo = get_all_faces(fa) or []
            fn = get_all_faces(fb) or []
            # pair by position: the same face in two renders of one frame
            for face_on in fn:
                cx = (face_on.bbox[0] + face_on.bbox[2]) * 0.5
                cy = (face_on.bbox[1] + face_on.bbox[3]) * 0.5
                best, bd = None, 1e9
                for face_off in fo:
                    ox = (face_off.bbox[0] + face_off.bbox[2]) * 0.5
                    oy = (face_off.bbox[1] + face_off.bbox[3]) * 0.5
                    d = (ox - cx) ** 2 + (oy - cy) ** 2
                    if d < bd:
                        best, bd = face_off, d
                if best is None:
                    continue
                d_on = min(compute_cosine_distance(e, face_on.embedding) for e in src)
                d_off = min(compute_cosine_distance(e, best.embedding) for e in src)
                rows += 1
                if d_on <= args.near and d_off > args.near:
                    painted += 1
                    who = ""
                    if tgt:
                        # The PLATE face at this box, against the captured
                        # target person. The selected person reads 0.2-0.6
                        # here; the other person's own tracks measured
                        # 0.81-0.97 on this clip.
                        dt = min(compute_cosine_distance(e, best.embedding)
                                 for e in tgt)
                        who = (f"  plate-vs-TARGET {dt:.2f} "
                               + ("(the selected person)" if dt <= 0.75
                                  else "<<< NOT the selected person"))
                    print(f"  frame {args.start + idx}  box "
                          f"{[int(v) for v in face_on.bbox]}  "
                          f"source-distance OFF {d_off:.2f} -> ON {d_on:.2f}"
                          f"{who}", flush=True)
                elif d_on <= args.near:
                    both += 1
        idx += 1
    a.release()
    b.release()
    print(f"[id] {rows} face comparisons over {idx} frames "
          f"(stride {args.stride})", flush=True)
    print(f"[id] already the source in both arms: {both}", flush=True)
    print(f"[id] BECAME the source in the ON arm: {painted}", flush=True)


if __name__ == "__main__":
    main()
