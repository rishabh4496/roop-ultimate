"""Render the angle sweep as VIDEO, swap it through the real pipeline, grade it.

The still bench (angle_bench.py) cannot exercise the orientation fix at all:
the roll ambiguity is resolved along a TRACK, and a still has no track. So this
builds a continuous roll sweep as an actual clip per yaw, runs it through
`batch_process_with_options` — the same path the app uses, tracking pre-pass and
all — and grades every output frame the way the still bench does, upright.

One clip per yaw rather than one long concatenated clip: a hard cut between two
different head angles is exactly the discontinuity the tracker is entitled to
stitch across, and starting each sweep at roll 0 (where the estimate is
unambiguous) gives the latch the same cold start it gets on real footage.

Usage:
    env/Scripts/python.exe tests/angle_video.py --tag after
    ROOP_ROLL_LATCH=0 env/Scripts/python.exe tests/angle_video.py --tag before
"""

import argparse
import csv
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import angle_bench as ab  # noqa: E402

YAWS = ab.YAW_LABEL

# Pinokio puts ffmpeg on the PATH of the shells it launches; a bench started
# from an ordinary terminal does not inherit that, and the encoder check then
# aborts every clip. Put the bundled binary in front rather than requiring the
# caller to remember, and leave any existing ffmpeg ahead of it untouched.
def ensure_ffmpeg():
    import shutil as _sh
    if _sh.which("ffmpeg"):
        return
    for cand in (r"G:/pinokio/bin/ffmpeg-env/Library/bin",):
        if os.path.exists(os.path.join(cand, "ffmpeg.exe")):
            os.environ["PATH"] = cand + os.pathsep + os.environ.get("PATH", "")
            return
    raise RuntimeError("ffmpeg not found; the video path cannot encode without it")


def render_clip(square, bg, path, roll_end=260.0, step=2.0, fps=30):
    """A continuous in-plane roll sweep, 0 -> roll_end, as a video file."""
    frames = int(round(roll_end / step)) + 1
    first = ab.roll_frame(square, bg, 0.0)
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"could not open {path} for writing")
    rolls = []
    for i in range(frames):
        roll = i * step
        vw.write(ab.roll_frame(square, bg, roll))
        rolls.append(roll)
    vw.release()
    return rolls


def read_frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        out.append(fr)
    cap.release()
    return out


def run_swap(clip_path, src_fs, options, out_dir):
    """Swap one clip through the real video path and return the output file."""
    import roop.globals as g
    from roop.core import batch_process_with_options
    from roop.ProcessEntry import ProcessEntry

    g.INPUT_FACESETS = [src_fs]
    g.TARGET_FACES = []
    g.output_path = out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Preserve the source cadence in the validation path.  A fixed 30 FPS
    # entry makes a 24/25 FPS clip appear shorter even when every frame was
    # processed, which turns a harness error into a false duration regression.
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    entry = ProcessEntry(clip_path, 0, 0, fps if fps and fps > 0 else 30.0)
    before = set(os.listdir(out_dir))
    batch_process_with_options([entry], options, None)

    # `entry.finalname` is not populated on the object the caller passes in, and
    # the output carries a run timestamp, so it is picked up from disk instead
    # of guessed. Newest new .mp4 that is not the input clip.
    if entry.finalname and os.path.exists(entry.finalname):
        return entry.finalname
    fresh = [f for f in os.listdir(out_dir)
             if f not in before and f.lower().endswith(".mp4")
             and os.path.join(out_dir, f) != clip_path]
    if not fresh:
        return None
    fresh.sort(key=lambda f: os.path.getmtime(os.path.join(out_dir, f)))
    return os.path.join(out_dir, fresh[-1])


def grade_clip(plate_frames, swap_frames, rolls, bg, src_embed, pair, yaw,
               ref_pts=None):
    """Per-frame drift and identity, graded upright.

    `ref_pts` is the un-rolled plate's landmarks at roll 0, so every row carries
    the same measurement floor the still bench does. Without it the floor column
    is empty, and the floor is the one column that catches a grader failing at
    an angle and being credited to the pipeline.
    """
    rows = []
    n = min(len(plate_frames), len(swap_frames))
    for i in range(n):
        roll = rolls[i]
        plate = ab.unroll(plate_frames[i], roll, bg)
        swapped = ab.unroll(swap_frames[i], roll, bg)
        pf = ab.biggest_face(plate)
        if pf is None:
            rows.append(dict(pair=pair, yaw=yaw, roll=roll, detected=0,
                             note="DETECT MISS on plate"))
            continue
        row = ab.grade_frame(plate, swapped, src_embed, pf, ref_pts)
        row.update(pair=pair, yaw=yaw, roll=roll)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--provider", default="cuda")
    ap.add_argument("--swap-model", default="inswapper")
    ap.add_argument("--enhancer", default="None")
    ap.add_argument("--mask-engine", default="None")
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--threads", type=int, default=20,
                    help="execution threads; the project benches at 20, which is "
                         "also what config.yaml runs")
    ap.add_argument("--swap-model-mask-strength", type=float, default=0.0,
                    help="the swap net's own face mask, 0-100. Production runs "
                         "25; the default is 0 because that is what every arm "
                         "saved before 2026-08-21 used, and changing it would "
                         "make this run incomparable to them.")
    ap.add_argument("--facesets", default="ashna,harjot")
    ap.add_argument("--yaws", default="",
                    help="comma-separated yaws to render, e.g. -90,90; default all five")
    # NOT under app/temp. roop.utilities.delete_temp_frames does an unguarded
    # shutil.rmtree of os.path.dirname(os.path.dirname(frame_path)) as part of
    # normal post-processing, so anything a bench leaves two levels inside the
    # app TEMP tree is deleted by the very run it is measuring. This bench lost
    # a complete set of results that way.
    ap.add_argument("--out", default=os.path.join(APP, "output", "bench_angle_video"))
    args = ap.parse_args()

    ensure_ffmpeg()
    a, b = args.facesets.split(",")
    # --mask-engine takes a UI display name ("RealityUX"); ProcessOptions keys
    # its processor dict by processorname ("mask_realityux"). Handing the display
    # name straight through builds a processor dict ProcessMgr cannot resolve —
    # the same defect `4310da4` fixed in frontal_roll_video.py, where it meant no
    # roll sweep in this project's history had ever run with a mask engine on.
    from sample_bench import map_mask_engine
    mask_engine = (map_mask_engine(args.mask_engine)
                   if args.mask_engine != "None" else "None")
    if mask_engine is None:
        raise SystemExit(f"unknown --mask-engine {args.mask_engine!r}; "
                         f"use a display name from sample_bench._MASK_ENGINE_MAP")
    g = ab.init_pipeline(args.provider, args.swap_model, args.enhancer,
                         mask_engine, args.swap_model_mask_strength)
    g.video_encoder = "libx264"
    g.video_quality = 12
    g.execution_threads = args.threads
    # The whole point is the tracking pre-pass, which is where the roll is
    # resolved. Without it every frame is detected independently and there is
    # no track to be continuous along.
    #
    # temporal_detection specifically: it is the ONLY mode that builds per-frame
    # Face objects up front (_build_temporal_faces), which is what the latch
    # stamps. The track_identities-only pre-pass stores source assignments, not
    # faces, so the latch has nothing to write on there.
    g.CFG.track_identities = True
    g.temporal_detection = True
    g.CFG.temporal_detection = True
    options = ab.build_options(g, args.swap_model, mask_engine, False)

    from roop import orientation
    outdir = os.path.join(args.out, args.tag)
    os.makedirs(outdir, exist_ok=True)
    work = os.path.join(outdir, "work")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    print(f"[angle_video] tag={args.tag} roll_latch={orientation.ENABLED} "
          f"model={args.swap_model} step={args.step}deg "
          f"mask_engine={args.mask_engine} enhancer={args.enhancer} "
          f"provider={args.provider} threads={args.threads} "
          f"swap_model_mask={args.swap_model_mask_strength}", flush=True)

    fs_dir = os.path.join(APP, "facesets")
    rows = []
    for src_name, tgt_name in ((a, b), (b, a)):
        src_fs = ab.load_faceset(os.path.join(fs_dir, src_name + ".fsz"))
        src_embed = getattr(src_fs.faces[0], "embedding", None)
        tgt = ab.plates(os.path.join(fs_dir, tgt_name + ".fsz"))
        pair = f"{src_name}->{tgt_name}"

        want = ({int(v) for v in args.yaws.split(",") if v.strip()}
                if args.yaws else None)
        for pi, plate in enumerate(tgt):
            yaw = YAWS.get(pi, pi)
            if want is not None and yaw not in want:
                continue
            square, bg = ab.prepare_plate(plate)
            if square is None:
                continue
            clip = os.path.join(work, f"{src_name}_{tgt_name}_yaw{yaw:+03d}.mp4")
            rolls = render_clip(square, bg, clip, step=args.step)
            final = run_swap(clip, src_fs, options, work)
            if not final or not os.path.exists(final):
                print(f"  {pair} yaw{yaw:+4d}: no output produced", flush=True)
                continue
            plate_frames = read_frames(clip)
            swap_frames = read_frames(final)
            ref_face = ab.biggest_face(ab.unroll(plate_frames[0], rolls[0], bg))
            ref_pts = ab.lm68(ref_face) if ref_face is not None else None
            got = grade_clip(plate_frames, swap_frames, rolls, bg, src_embed,
                             pair, yaw, ref_pts)
            rows += got
            ok = [r for r in got if r.get("detected")]
            ids = [float(r["id_source"]) for r in ok if r.get("id_source") != ""]
            print(f"  {pair} yaw{yaw:+4d}: {len(ok)}/{len(got)} graded, "
                  f"mean id {np.mean(ids):.3f}" if ids else
                  f"  {pair} yaw{yaw:+4d}: nothing graded", flush=True)

    cols = ["pair", "yaw", "roll", "detected", "shift", "jaw", "brows", "nose",
            "eyes", "mouth", "id_source", "id_plate", "ghost", "floor", "note"]
    with open(os.path.join(outdir, "rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    print("\n=== identity by yaw x roll band ===")
    print(f"{'pair':<16}{'yaw':>5}{'roll<90':>10}{'roll>=90':>10}{'all':>8}")
    for pair in sorted({r["pair"] for r in rows}):
        for yaw in sorted({r["yaw"] for r in rows if r["pair"] == pair}):
            sel = [r for r in rows if r["pair"] == pair and r["yaw"] == yaw
                   and r.get("detected") and r.get("id_source") != ""]
            if not sel:
                continue
            lo = [float(r["id_source"]) for r in sel if float(r["roll"]) < 90]
            hi = [float(r["id_source"]) for r in sel if float(r["roll"]) >= 90]
            al = [float(r["id_source"]) for r in sel]
            print(f"{pair:<16}{yaw:>5}{np.mean(lo) if lo else float('nan'):>10.3f}"
                  f"{np.mean(hi) if hi else float('nan'):>10.3f}{np.mean(al):>8.3f}")
    print(f"\nwrote {outdir}")


if __name__ == "__main__":
    main()
