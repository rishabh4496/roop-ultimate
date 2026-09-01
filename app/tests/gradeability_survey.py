"""How many frames of a clip can two_face_video actually GRADE for identity?

two_face_video blanks its `own`/`other` identity columns whenever a face's
recognition crop is shared with its neighbour (`contam >= GRADE_CONTAM_MAX`).
That is deliberate and correct -- the grader suffers the same contamination the
pipeline does, so reading identity there reports the two people as each other
whatever the swap did. But it has a consequence nobody had measured: on a
contact-heavy clip the identity columns can be blank on essentially EVERY frame,
and the arm then proves nothing about the swap model no matter how long it took
to render.

Measured on the existing d1 arms: `own` is populated on 1 row of 282
(d1_hyperswap_noenh) and 0 of 282 (d1_lidband_noenh). Rendering more d1 arms to
compare swap models cannot work, and two sessions have now queued exactly that.

This runs DETECTION ONLY over sampled plate frames -- no swap, no enhancer -- so
it costs a fraction of a render and answers the question before the render is
started. Plate contamination is the proxy for the output's: the grader measures
the swapped frame, but crop overlap is a geometric property of where the two
heads are, which the swap does not move.

Usage:
    env/Scripts/python.exe tests/gradeability_survey.py --every 10
    env/Scripts/python.exe tests/gradeability_survey.py --clips d2,d6 --every 5
"""

import argparse
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import fixtures
import angle_bench as ab  # noqa: E402

DOUBLE_DIR = fixtures.clip_dir('double')
GRADE_CONTAM_MAX = 0.35     # must track two_face_video.GRADE_CONTAM_MAX


def survey(path, every, limit):
    from roop.face_util import get_all_faces
    from roop.face_contact import crop_contamination

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sampled = gradeable = two_face = 0
    faces_seen = faces_clean = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or (limit and sampled >= limit):
            break
        if idx % every == 0:
            faces = sorted(get_all_faces(frame) or [],
                           key=lambda f: float(f.bbox[0]))
            contam = crop_contamination(faces) if faces else []
            sampled += 1
            if len(faces) >= 2:
                two_face += 1
                # The arm can grade this frame only if BOTH people's crops are
                # clean -- `own` and `other` are read from the same row, and the
                # summary keeps only rows where both are present.
                clean = [c for c in contam if c < GRADE_CONTAM_MAX]
                if len(clean) >= 2:
                    gradeable += 1
            faces_seen += len(faces)
            faces_clean += sum(1 for c in contam if c < GRADE_CONTAM_MAX)
        idx += 1
    cap.release()
    return {"total": total, "sampled": sampled, "two_face": two_face,
            "gradeable": gradeable, "faces": faces_seen, "clean": faces_clean}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="d1,d2,d3,d4,d5,d6")
    ap.add_argument("--every", type=int, default=10,
                    help="sample every Nth frame")
    ap.add_argument("--limit", type=int, default=400,
                    help="max sampled frames per clip (0 = whole clip)")
    ap.add_argument("--provider", default="tensorrt")
    args = ap.parse_args()

    # Detection only, but through the app's own init: without it TensorRT's DLLs
    # are off PATH and ORT falls back to CPU silently, which turns a 4 ms model
    # into a 210 ms one and makes this survey slower than the render it saves.
    ab.init_pipeline(args.provider, "inswapper", "None", "None")

    print(f"\n{'clip':>6}{'frames':>9}{'sampled':>9}{'2+ faces':>10}"
          f"{'GRADEABLE':>11}{'faces':>8}{'clean':>8}")
    for name in [c.strip() for c in args.clips.split(",") if c.strip()]:
        path = os.path.join(DOUBLE_DIR, f"{name}.mp4")
        if not os.path.exists(path):
            print(f"{name:>6}   missing ({path})")
            continue
        r = survey(path, args.every, args.limit)
        if r is None:
            print(f"{name:>6}   unreadable")
            continue
        pct = (100.0 * r["gradeable"] / r["sampled"]) if r["sampled"] else 0.0
        fpct = (100.0 * r["clean"] / r["faces"]) if r["faces"] else 0.0
        print(f"{name:>6}{r['total']:>9}{r['sampled']:>9}{r['two_face']:>10}"
              f"{r['gradeable']:>7} {pct:>4.0f}%{r['faces']:>8}"
              f"{r['clean']:>5} {fpct:>3.0f}%")
    print("\nGRADEABLE = sampled frames where BOTH faces have a clean enough "
          f"recognition crop (contam < {GRADE_CONTAM_MAX}) for two_face_video "
          "to fill in\nits identity columns. A clip near 0% cannot compare swap "
          "models however long it renders.")


if __name__ == "__main__":
    main()
