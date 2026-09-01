"""Single-IMAGE face swap, end to end, graded on identity.

WHY THIS EXISTS. "Image face swap works" is an acceptance criterion of the
final validation campaign and neither GPU campaign had a run behind it. Every
end-to-end harness in this repository renders VIDEO -- `two_face_video`,
`baseline_controlled`, the phase benches, the enhancer sweep -- so the still
path (`roop.core.live_swap`, no tracker, no temporal engines, no stabilizer
geometry, no encoder) was covered only by unit tests.

WHAT IT ASSERTS, and why identity rather than pixels. Two renders of one
unchanged configuration differ on every frame on this pipeline (mean ~0.71/255,
see tests/measure_output_noise_floor.py), so "the output changed" is worth
nothing on its own. The acceptance is therefore directional and about identity:

  1. a face is detected in the plate at all -- otherwise the rest is vacuous;
  2. `live_swap` returns a frame and the FACE REGION actually moved;
  3. the swapped face's embedding is CLOSER to the source faceset's identity
     than the plate's own face was.

(3) is the one that cannot be faked by a filter, a colour transfer, or an
enhancer, and it is what distinguishes a real swap from a pipeline that ran
every stage and pasted the original back -- the failure this project has hit
repeatedly, most recently as a dedent that left frames unswapped while
reporting 100% and running 47% faster.

The stack comes from config.yaml, not from CLI defaults, for the same reason
every other harness here now does.

    env/Scripts/python.exe tests/image_swap_smoke.py
    env/Scripts/python.exe tests/image_swap_smoke.py --frames 5 --source gargee
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

import fixtures                       # noqa: E402
import angle_bench as ab              # noqa: E402
# The library ingest and the mask-engine name map both live in two_face_video.
# Reused rather than reimplemented: `angle_bench.load_faceset` runs
# `prepare_plate` first, which is right for the studio yaw strips it was written
# for and destructive for an ordinary faceset of loose photographs, and
# `build_options` wants the internal engine key (`mask_realityux`), not the UI
# label (`RealityUX`) -- passing the label raises KeyError inside
# ProcessMgr.initialize.
from two_face_video import load_library_faceset, map_mask_engine  # noqa: E402


def frame_at(video, index):
    cap = cv2.VideoCapture(video)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
    finally:
        cap.release()
    return frame if ok else None


def face_region_delta(plate, swapped, face):
    """Mean absolute difference inside the detected face box only.

    Whole-frame means are dominated by the untouched background, which makes a
    real swap and a no-op look similar.
    """
    x0, y0, x1, y1 = [int(v) for v in face.bbox]
    h, w = plate.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(cv2.absdiff(plate[y0:y1, x0:x1],
                             swapped[y0:y1, x0:x1]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=fixtures.clip("single/s1.mp4"),
                    help="frames are pulled from here; any clip with a face")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--stride", type=int, default=400)
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--source", default="harjot", help="faceset library name")
    ap.add_argument("--provider", default=None, help="defaults to config.yaml")
    ap.add_argument("--swap-model", default=None, help="defaults to config.yaml")
    ap.add_argument("--enhancer", default=None, help="defaults to config.yaml")
    ap.add_argument("--mask-engine", default=None, help="defaults to config.yaml")
    ap.add_argument("--min-region-delta", type=float, default=3.0,
                    help="mean abs diff inside the face box, /255. The "
                         "run-to-run noise floor is ~0.71 whole-frame, so this "
                         "is comfortably above it while staying far below a "
                         "real swap")
    ap.add_argument("--control", action="store_true",
                    help="run the identical grading with NO swap. Every "
                         "assertion must FAIL. Without this the harness could "
                         "be a rubber stamp and nobody would know -- the same "
                         "reason angle_bench carries a control arm")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    provider = args.provider or str(cfg.provider)
    swap_model = args.swap_model or str(cfg.swap_model)
    enhancer = args.enhancer or str(cfg.selected_enhancer)
    mask_engine = args.mask_engine or str(cfg.mask_engine)

    print("[image] %s / %s / %s / %s | source %s"
          % (swap_model, mask_engine, enhancer, provider, args.source),
          flush=True)

    engine_key = map_mask_engine(mask_engine)
    if mask_engine not in ("", "None", None) and engine_key is None:
        raise SystemExit("unknown mask engine %r -- two_face_video's map is the "
                         "authority on these names" % mask_engine)
    g = ab.init_pipeline(provider, swap_model, enhancer, mask_engine,
                         sync_config=True)
    options = ab.build_options(g, swap_model, engine_key)

    src_fs = load_library_faceset(args.source)
    if not src_fs.faces:
        raise SystemExit("faceset %s ingested zero faces" % args.source)
    src_embed = getattr(src_fs, "identity_embedding", None)
    if src_embed is None:
        src_embed = getattr(src_fs.faces[0], "embedding", None)
    print("[image] source faceset: %d faces" % len(src_fs.faces), flush=True)

    from roop.core import live_swap

    rows, failures = [], []
    for i in range(args.frames):
        idx = args.start + i * args.stride
        plate = frame_at(args.video, idx)
        if plate is None:
            failures.append("frame %d: could not be decoded" % idx)
            continue

        plate_face = ab.biggest_face(plate)
        if plate_face is None:
            # Not a swap failure; the fixture simply has no face here. Recorded
            # rather than silently skipped, because "no faces in this video" is
            # exactly what a broken detector also looks like.
            failures.append("frame %d: NO FACE DETECTED in the plate" % idx)
            continue

        swapped = (plate.copy() if args.control
                   else live_swap(plate.copy(), options, input_facesets=[src_fs]))
        if swapped is None:
            failures.append("frame %d: live_swap returned None" % idx)
            continue

        delta = face_region_delta(plate, swapped, plate_face)
        out_face = ab.biggest_face(swapped)
        before = ab.cosine(plate_face.embedding, src_embed)
        after = (ab.cosine(out_face.embedding, src_embed)
                 if out_face is not None else float("nan"))
        rows.append({"frame": idx, "delta": delta,
                     "before": before, "after": after})

        verdict = "ok"
        if delta < args.min_region_delta:
            verdict = ("FAIL: face region barely changed (%.2f/255) -- the "
                       "pipeline ran and pasted the original back" % delta)
        elif out_face is None:
            verdict = "FAIL: no face detectable in the swapped output"
        elif not (after > before):
            verdict = ("FAIL: identity did not move toward the source "
                       "(%.4f -> %.4f)" % (before, after))
        if verdict != "ok":
            failures.append("frame %d: %s" % (idx, verdict))
        print("[image] frame %-6d region delta %6.2f/255   identity to source "
              "%.4f -> %.4f   %s" % (idx, delta, before, after, verdict),
              flush=True)

        if args.out:
            os.makedirs(args.out, exist_ok=True)
            cv2.imwrite(os.path.join(args.out, "f%06d_plate.png" % idx), plate)
            cv2.imwrite(os.path.join(args.out, "f%06d_swap.png" % idx), swapped)

    if not rows:
        print("\n[image] FAILED -- no frame produced a gradable result")
        for f in failures:
            print("[image]   %s" % f)
        return 1

    print("\n[image] %d of %d frames graded; mean region delta %.2f/255, "
          "mean identity gain %+.4f"
          % (len(rows), args.frames,
             float(np.mean([r["delta"] for r in rows])),
             float(np.mean([r["after"] - r["before"] for r in rows]))))
    if args.control and failures:
        print("[image] control arm failed as required (%d problem(s)) -- the "
              "assertions discriminate" % len(failures))
        for f in failures:
            print("[image]   %s" % f)
        return 0
    if failures:
        print("[image] %d PROBLEM(S):" % len(failures))
        for f in failures:
            print("[image]   %s" % f)
        return 1
    print("[image] PASS -- the still path swaps and moves identity toward the "
          "source on every graded frame")
    if args.control:
        print("[image] BUT --control was set, so NOTHING WAS SWAPPED and these "
              "assertions are not discriminating. Fix the harness.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
