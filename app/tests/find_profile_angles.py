"""Find frames worth banking as extra PROFILE angles for a two-person clip.

Why this is not just "pick some frames where a face looks turned":

  * `two_face_video.capture_targets` banks extra angles by LEFT-TO-RIGHT
    POSITION in each extra frame. If the pair cross sides between the seed and
    the extra frame, the angle lands on the wrong person -- and one polluted
    angle makes a stranger's whole track measure ~0 to that person, because
    every swap-time match takes the MINIMUM over the bank. So each candidate is
    identity-checked against the seed here, not merely position-sorted.
  * A contaminated crop (the neighbour's face inside this face's recognition
    box) is exactly what must never enter the bank, so candidates are rejected
    on `crop_contamination` before anything else.

Prints frames ready to paste into `--capture-extra`.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np

import angle_bench as ab


def cos(a, b):
    if a is None or b is None:
        return float('nan')
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return 1.0 - float(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=r"G:/pinokio/roop-keep/duo/d2.mp4")
    ap.add_argument("--seed", type=int, default=675)
    ap.add_argument("--person", type=int, default=0,
                    help="which person (left=0) we want profile angles FOR")
    ap.add_argument("--want", type=int, default=6)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-contam", type=float, default=0.20)
    ap.add_argument("--max-id", type=float, default=0.60,
                    help="a candidate must be at least this close to the seed face "
                         "of its own person, or it is somebody else")
    args = ap.parse_args()

    ab.init_pipeline('tensorrt', 'realswap', 'None', 'None')
    from roop.face_util import get_all_faces
    from roop.face_contact import crop_contamination

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.seed)
    ok, seed_frame = cap.read()
    if not ok:
        raise SystemExit("could not read the seed frame")
    seed = sorted(get_all_faces(seed_frame) or [], key=lambda f: float(f.bbox[0]))
    if len(seed) != 2:
        raise SystemExit("seed frame does not have exactly 2 faces")
    seed_emb = [f.embedding for f in seed]
    print("[seed] frame %d, the two people are %.3f apart"
          % (args.seed, cos(seed_emb[0], seed_emb[1])), flush=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cands, idx = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % args.stride:
            idx += 1
            continue
        faces = sorted(get_all_faces(fr) or [], key=lambda f: float(f.bbox[0]))
        if len(faces) == 2:
            contam = crop_contamination(faces)
            if max(contam) <= args.max_contam:
                # Identity-ordered? left must be person 0, right person 1.
                d00 = cos(faces[0].embedding, seed_emb[0])
                d11 = cos(faces[1].embedding, seed_emb[1])
                d01 = cos(faces[0].embedding, seed_emb[1])
                d10 = cos(faces[1].embedding, seed_emb[0])
                ordered = (d00 < d01) and (d11 < d10)
                if ordered and d00 <= args.max_id and d11 <= args.max_id:
                    f = faces[args.person]
                    w = float(f.bbox[2]) - float(f.bbox[0])
                    h = float(f.bbox[3]) - float(f.bbox[1])
                    if h > 0 and w > 40:
                        cands.append({
                            'frame': idx, 'wh': w / h, 'w': w,
                            'id_self': (d00, d11)[args.person],
                            'contam': float(max(contam)),
                        })
        idx += 1
    cap.release()

    if not cands:
        raise SystemExit("no clean, identity-ordered two-face frames found")

    print("[scan] %d clean identity-ordered candidate frames" % len(cands))
    wh = np.array([c['wh'] for c in cands])
    print("[scan] person %d width/height over those: median %.3f  min %.3f"
          % (args.person, np.median(wh), wh.min()))

    # Most profile first, then spread them over the clip so the bank gains
    # variety rather than six views of one moment.
    cands.sort(key=lambda c: c['wh'])
    picked, seen = [], []
    for c in cands:
        if len(picked) >= args.want:
            break
        if all(abs(c['frame'] - s) > 60 for s in seen):
            picked.append(c)
            seen.append(c['frame'])
    picked.sort(key=lambda c: c['frame'])

    print("\n  frame    w/h   width  id-to-seed  contam")
    for c in picked:
        print("  %5d  %.3f  %5.0f      %.3f   %.3f"
              % (c['frame'], c['wh'], c['w'], c['id_self'], c['contam']))
    print("\n--capture-extra %s" % ",".join(str(c['frame']) for c in picked))


if __name__ == "__main__":
    main()
