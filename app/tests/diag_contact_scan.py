"""Where in this clip do two faces actually SHARE a recognition crop?

"The swap flickers when another face is near it" is a claim about particular
stretches of footage, and picking windows by eye -- or from the spans in the
[Track] table -- does not find them: two people can be on screen together for
thousands of frames with their crops nowhere near each other. Co-occurrence is
not contact. (Measured the expensive way first: four windows chosen from
overlapping track spans contained ZERO contaminated detections.)

So this asks the question directly, with the detector alone and no swapping:
per sampled frame, run the same detection the pipeline runs and read
`face_contact.crop_contamination` off the result. Prints the stretches where it
is material, which is what a measurement window has to contain to be about this
defect at all.

Usage:
    env/Scripts/python.exe tests/diag_contact_scan.py \
        --video "<clip>.mp4" --stride 5 --floor 0.2
"""

import argparse
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import two_face_video as tfv          # noqa: E402  (perf env, as run.py does)
import angle_bench as ab              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0, help="0 = to the end")
    ap.add_argument("--stride", type=int, default=5,
                    help="a contact stretch lasts hundreds of frames, so a "
                         "stride this size cannot step over one")
    ap.add_argument("--floor", type=float, default=0.2,
                    help="crop fraction that counts as contact "
                         "(ROOP_CONTACT_TRACK's default)")
    ap.add_argument("--gap", type=int, default=60,
                    help="frames of quiet that end a stretch")
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()

    tfv._apply_startup_runtime_environment()
    import yaml
    with open(os.path.join(APP, "config.yaml")) as fh:
        cfg = yaml.safe_load(fh) or {}
    provider = args.provider or (cfg.get("provider") or "cuda")

    # Detector + recogniser only; the swapper/enhancer are never called.
    ab.init_pipeline(provider, cfg.get("swap_model") or "inswapper", "None",
                     "None", sync_config=True)
    from roop import face_contact
    from roop.face_util import get_all_faces

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    end = args.end if args.end > 0 else total
    print(f"[scan] {args.video}", flush=True)
    print(f"[scan] frames {args.start}..{end} of {total}, stride {args.stride}, "
          f"floor {args.floor}", flush=True)

    hits = []                      # (frame, worst contamination, n faces)
    idx = args.start
    t0 = time.time()
    scanned = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    while idx < end:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - args.start) % args.stride == 0:
            faces = get_all_faces(frame) or []
            scanned += 1
            if len(faces) > 1:
                worst = 0.0
                for f in faces:
                    try:
                        worst = max(worst, float(f.get('_emb_contam') or 0.0))
                    except (TypeError, ValueError, AttributeError):
                        pass
                if worst >= args.floor:
                    hits.append((idx, worst, len(faces)))
            if scanned % 200 == 0:
                rate = scanned / max(1e-6, time.time() - t0)
                print(f"[scan] {idx}/{end}  {rate:.1f} frames/s  "
                      f"{len(hits)} contact samples so far", flush=True)
        idx += 1
    cap.release()

    print(f"[scan] done: {scanned} frames sampled, {len(hits)} with a shared "
          f"crop >= {args.floor} ({100.0 * len(hits) / max(1, scanned):.1f}%)",
          flush=True)
    if not hits:
        print("[scan] no contact anywhere in this range -- a window from here "
              "cannot measure the contact tier.", flush=True)
        return

    runs = []
    for frame, worst, n in hits:
        if runs and frame - runs[-1][1] <= args.gap:
            runs[-1][1] = frame
            runs[-1][2] += 1
            runs[-1][3] = max(runs[-1][3], worst)
        else:
            runs.append([frame, frame, 1, worst])
    runs.sort(key=lambda r: -r[2])
    print("[scan] contact stretches, densest first "
          "(start, end, samples, worst fraction):", flush=True)
    for a, b, n, worst in runs[:12]:
        print(f"    {a:>7} .. {b:<7}  {n:>4} samples  worst {worst:.2f}",
              flush=True)
    print("[scan] windows=" + ",".join(
        f"{max(0, a - 50)}:{b + 50}" for a, b, n, w in runs[:4]), flush=True)


if __name__ == "__main__":
    main()
