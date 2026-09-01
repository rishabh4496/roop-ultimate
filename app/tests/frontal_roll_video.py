"""Roll a FRONTAL face through a full turn as video, swap it, grade every angle.

Why this exists alongside angle_video.py: that bench sweeps YAW (profile plates
from a studio faceset) and rolls them, and it grades in `all` swap mode. Two
reported failures live outside what it can see, and both are about a face that
is frontal and merely UPSIDE DOWN:

  1. an inverted face stops being recognised as the selected TARGET person, so
     it is never swapped at all — that is `selected` mode, which the studio
     bench does not exercise;
  2. on the frames that do swap, the pasted face comes out the other way up
     (eyes where the mouth should be).

So: one continuous 0..360 in-plane roll of a real frontal face out of the user's
own footage, run through `batch_process_with_options` exactly as the app does,
graded per frame.

THE PLATE CARRIES NO FABRICATED BACKGROUND. angle_bench works from letterboxed
studio strips and has to paint a synthetic backdrop before it can roll them;
here the material is a real 720x720 crop of the source frame, rolled with
BORDER_REFLECT_101. The head's corner radius is ~325px against an inscribed
radius of 360, so every pixel of the face is genuine footage at every angle and
only the four corner triangles are reflected.

GRADED UPRIGHT, ALWAYS. See angle_bench.unroll: the 106-point and recognition
models are trained on upright faces, and grading a rolled frame in place invents
a large roll-shaped failure on a frame where nothing happened. The `floor`
column is the plate's own round trip and is the tell that that has been avoided.

Usage:
    env/Scripts/python.exe tests/frontal_roll_video.py --tag before
    env/Scripts/python.exe tests/frontal_roll_video.py --tag after --source rhythm
"""

import argparse
import csv
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

import fixtures
import angle_bench as ab            # noqa: E402
from angle_video import ensure_ffmpeg, read_frames   # noqa: E402

SIDE = 720          # plate side; also the roll canvas, so nothing leaves frame


# ── material ─────────────────────────────────────────────────────────────────

def most_frontal_frame(path, stride=5):
    """Index of the frame whose largest face is nearest frontal.

    Scored on |yaw|+|pitch|+|roll| from solve_pose_5pt with a mild bonus for
    face area, so the sweep starts from a face the detector is unambiguously
    right about — the roll-0 reading is the ground truth every later angle is
    measured against, and a plate that starts ambiguous poisons the whole run.
    """
    from roop.face_util import solve_pose_5pt
    cap = cv2.VideoCapture(path)
    best, best_score, idx = None, None, -1
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        idx += 1
        if idx % stride:
            continue
        f = ab.biggest_face(fr)
        if f is None:
            continue
        pose = solve_pose_5pt(f.kps)
        if pose is None:
            continue
        yaw, pitch, roll = [float(v) for v in pose[:3]]
        area = float(f.bbox[2] - f.bbox[0]) * float(f.bbox[3] - f.bbox[1])
        score = abs(yaw) + abs(pitch) + abs(roll) - 4e-6 * area
        if best_score is None or score < best_score:
            best, best_score, best_i = fr.copy(), score, idx
    cap.release()
    if best is None:
        raise RuntimeError(f"no face with a pose solve anywhere in {path}")
    return best_i, best


def square_around_face(frame, side=SIDE):
    """A `side` square of real footage centred on the largest face."""
    f = ab.biggest_face(frame)
    if f is None:
        raise RuntimeError("no face to centre the plate on")
    cx = (float(f.bbox[0]) + float(f.bbox[2])) / 2.0
    cy = (float(f.bbox[1]) + float(f.bbox[3])) / 2.0
    h, w = frame.shape[:2]
    side = min(side, h, w)
    x0 = int(round(min(max(cx - side / 2.0, 0), w - side)))
    y0 = int(round(min(max(cy - side / 2.0, 0), h - side)))
    return frame[y0:y0 + side, x0:x0 + side].copy()


def roll_frame(square, deg):
    """In-plane roll about the square's centre, real pixels only."""
    d = square.shape[0]
    M = cv2.getRotationMatrix2D((d / 2.0, d / 2.0), float(deg), 1.0)
    return cv2.warpAffine(square, M, (d, d), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REFLECT_101)


def unroll(img, deg):
    return roll_frame(img, -deg)


def run_swap(clip_path, src_fs, options, out_dir):
    """Swap one clip through the real video path and return the output file.

    Deliberately NOT angle_video.run_swap: that one clears
    `roop.globals.TARGET_FACES`, which is correct for an `all`-mode sweep and
    fatal here — with no captured target the `selected` matcher has no person to
    assign, so every face is refused "over the identity threshold" and the run
    reports a 100% failure that is the harness's, at every angle including 0.
    """
    import roop.globals as g
    from roop.core import batch_process_with_options
    from roop.ProcessEntry import ProcessEntry

    g.INPUT_FACESETS = [src_fs]
    g.output_path = out_dir
    os.makedirs(out_dir, exist_ok=True)

    entry = ProcessEntry(clip_path, 0, 0, 30.0)
    before = set(os.listdir(out_dir))
    batch_process_with_options([entry], options, None)

    if entry.finalname and os.path.exists(entry.finalname):
        return entry.finalname
    fresh = [f for f in os.listdir(out_dir)
             if f not in before and f.lower().endswith(".mp4")
             and os.path.join(out_dir, f) != clip_path]
    if not fresh:
        return None
    fresh.sort(key=lambda f: os.path.getmtime(os.path.join(out_dir, f)))
    return os.path.join(out_dir, fresh[-1])


def render_clip(square, path, rolls, fps=30):
    d = square.shape[0]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (d, d))
    if not vw.isOpened():
        raise RuntimeError(f"could not open {path} for writing")
    for r in rolls:
        vw.write(roll_frame(square, r))
    vw.release()


# ── grading ──────────────────────────────────────────────────────────────────

def _grad(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float((a * b).sum() / d) if d > 1e-9 else float("nan")


def flip_gain(plate_up, swap_up, box):
    """How much BETTER the swapped face patch matches the plate when turned 180.

    The direct test for "eyes came out where the mouth should be", and it needs
    no landmarks on the composite — a swapped-upside-down face is exactly the
    case where re-detecting the result is least trustworthy. Positive means the
    pasted face is inverted relative to the head it sits on; a correct swap is
    comfortably negative, because a face correlates with itself far better the
    right way up than upside down.

    The box is symmetric about the face centre, so rotating the patch 180 keeps
    it registered with the plate's.
    """
    x0, y0, x1, y1 = box
    p = _grad(plate_up[y0:y1, x0:x1])
    s = _grad(swap_up[y0:y1, x0:x1])
    if p.size < 64 or s.shape != p.shape:
        return float("nan")
    return _ncc(cv2.rotate(s, cv2.ROTATE_180), p) - _ncc(s, p)


def touched(plate_up, swap_up, box):
    """Mean absolute difference inside the face box: 0 means never swapped."""
    x0, y0, x1, y1 = box
    a = plate_up[y0:y1, x0:x1].astype(np.float32)
    b = swap_up[y0:y1, x0:x1].astype(np.float32)
    return float(np.abs(a - b).mean())


def cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    n = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a.dot(b) / n) if n else float("nan")


# ── run ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--video", default=fixtures.clip('vidssave.mp4'))
    ap.add_argument("--source", default="rhythm", help="source faceset name")
    ap.add_argument("--frame", type=int, default=-1,
                    help="plate frame index; -1 picks the most frontal")
    ap.add_argument("--provider", default="cuda")
    ap.add_argument("--swap-model", default="inswapper")
    ap.add_argument("--enhancer", default="None")
    ap.add_argument("--mask-engine", default="None")
    ap.add_argument("--mode", default="selected", choices=["selected", "all"],
                    help="'selected' reproduces the target-face match; 'all' "
                         "bypasses it to isolate the orientation failure")
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--tracking", default="1",
                    help="0 disables the tracking pre-pass (roll latch is inert "
                         "without it, see roop.orientation)")
    ap.add_argument("--autorotate", default="1",
                    help="0 turns off the quarter-turn correction entirely")
    ap.add_argument("--refine", default="0",
                    help="1 enables the 106-point landmark refinement")
    # NOT under app/temp: utilities.delete_temp_frames rmtree's two levels up
    # from a frame path as normal post-processing and has eaten a bench's
    # results from there before.
    ap.add_argument("--out", default=os.path.join(APP, "output", "bench_frontal_roll"))
    args = ap.parse_args()

    ensure_ffmpeg()
    # The mask engine arrives as the display name the app and config.yaml use
    # ("RealityUX"); everything downstream wants the internal key
    # ("mask_realityux"). This bench passed the display name straight through,
    # so every engine except "None" died with a KeyError deep in build_options —
    # which is why a roll sweep had never been run with a mask engine on at all.
    from sample_bench import map_mask_engine
    mask_engine = map_mask_engine(args.mask_engine) if args.mask_engine != "None" else "None"
    if mask_engine is None:
        raise SystemExit(f"unknown --mask-engine {args.mask_engine!r}; "
                         f"use a display name from sample_bench._MASK_ENGINE_MAP")
    g = ab.init_pipeline(args.provider, args.swap_model, args.enhancer, mask_engine)
    g.video_encoder = "libx264"
    g.video_quality = 12
    g.execution_threads = 4
    g.face_swap_mode = args.mode
    g.autorotate_faces = args.autorotate != "0"
    g.refine_landmarks = args.refine != "0"
    track = args.tracking != "0"
    g.CFG.track_identities = track
    g.temporal_detection = track
    g.CFG.temporal_detection = track
    options = ab.build_options(g, args.swap_model, mask_engine, False)

    from roop import orientation
    outdir = os.path.join(args.out, args.tag)
    work = os.path.join(outdir, "work")
    os.makedirs(work, exist_ok=True)

    # ── plate ────────────────────────────────────────────────────────────────
    if args.frame >= 0:
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, base = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"frame {args.frame} unreadable")
        fidx = args.frame
    else:
        fidx, base = most_frontal_frame(args.video)
    square = square_around_face(base)

    ref = ab.biggest_face(square)
    if ref is None:
        raise RuntimeError("no face on the prepared plate")
    ref_emb = np.asarray(ref.embedding, np.float64).copy()
    # Face box for the patch metrics, centred on the face and square, so the
    # 180-degree rotation in flip_gain stays registered.
    cx = (float(ref.bbox[0]) + float(ref.bbox[2])) / 2.0
    cy = (float(ref.bbox[1]) + float(ref.bbox[3])) / 2.0
    half = int(0.62 * max(float(ref.bbox[2] - ref.bbox[0]),
                          float(ref.bbox[3] - ref.bbox[1])))
    d = square.shape[0]
    half = min(half, int(cx), int(cy), d - int(cx), d - int(cy))
    box = (int(cx) - half, int(cy) - half, int(cx) + half, int(cy) + half)

    print(f"[frontal_roll] tag={args.tag} frame={fidx} mode={args.mode} "
          f"tracking={track} autorotate={g.autorotate_faces} "
          f"refine={g.refine_landmarks} roll_latch={orientation.ENABLED} "
          f"model={args.swap_model} step={args.step}deg", flush=True)

    # ── the selected target face, captured at roll 0, as the UI would ────────
    import roop.globals as G
    if args.mode == "selected":
        G.TARGET_FACES = [ref]
        G.TARGET_FACE_GROUP = [0]
    else:
        G.TARGET_FACES = []
        G.TARGET_FACE_GROUP = []

    fs_dir = os.path.join(APP, "facesets")
    src_fs = ab.load_faceset(os.path.join(fs_dir, args.source + ".fsz"))
    src_embed = getattr(src_fs.faces[0], "embedding", None)

    rolls = [i * args.step for i in range(int(round(360.0 / args.step)))]
    clip = os.path.join(work, f"frontal_roll_{args.tag}.mp4")
    render_clip(square, clip, rolls)
    final = run_swap(clip, src_fs, options, work)
    if not final or not os.path.exists(final):
        raise RuntimeError("the pipeline produced no output clip")

    plate_frames = read_frames(clip)
    swap_frames = read_frames(final)
    n = min(len(plate_frames), len(swap_frames), len(rolls))
    print(f"  graded {n} frames  ({os.path.basename(final)})", flush=True)

    ref_pts = ab.lm68(ab.biggest_face(unroll(plate_frames[0], rolls[0])))

    rows = []
    for i in range(n):
        r = rolls[i]
        plate_up = unroll(plate_frames[i], r)
        swap_up = unroll(swap_frames[i], r)
        row = {"roll": r, "touched": round(touched(plate_up, swap_up, box), 3),
               "flip": "", "match_cos": "", "id_source": "", "detected": 0,
               "eyes": "", "mouth": "", "nose": "", "shift": "", "floor": "",
               "note": ""}
        fg = flip_gain(plate_up, swap_up, box)
        row["flip"] = round(fg, 4) if np.isfinite(fg) else ""

        # What `selected` mode actually sees: the ROLLED plate's embedding
        # against the captured target face. This is the number that decides
        # whether the face is swapped at all.
        rp = ab.biggest_face(plate_frames[i])
        row["match_cos"] = round(cos(getattr(rp, "embedding", []), ref_emb), 4) \
            if rp is not None else ""

        pf = ab.biggest_face(plate_up)
        if pf is None:
            row["note"] = "DETECT MISS on plate"
            rows.append(row)
            continue
        got = ab.grade_frame(plate_up, swap_up, src_embed, pf, ref_pts)
        for k in ("detected", "eyes", "mouth", "nose", "shift", "floor",
                  "id_source", "note"):
            row[k] = got.get(k, "")
        rows.append(row)

    cols = ["roll", "touched", "flip", "match_cos", "id_source", "detected",
            "shift", "nose", "eyes", "mouth", "floor", "note"]
    with open(os.path.join(outdir, "rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    report(rows)
    print(f"\nclip   {clip}\nswap   {final}\nrows   {os.path.join(outdir, 'rows.csv')}")


def report(rows, swap_floor=2.0):
    """Per-20-degree band: was it swapped, was it the right way up, is it him."""
    print(f"\n{'roll band':>12}{'swapped%':>10}{'inverted%':>11}{'match_cos':>11}"
          f"{'id_source':>11}{'mouth':>8}")
    for lo in range(0, 360, 20):
        sel = [r for r in rows if lo <= float(r["roll"]) < lo + 20]
        if not sel:
            continue
        sw = [r for r in sel if float(r["touched"] or 0) >= swap_floor]
        inv = [r for r in sw if r["flip"] != "" and float(r["flip"]) > 0]
        mc = [float(r["match_cos"]) for r in sel if r["match_cos"] != ""]
        ids = [float(r["id_source"]) for r in sw if r["id_source"] != ""]
        mo = [float(r["mouth"]) for r in sw if r["mouth"] != ""]
        print(f"{lo:>4}-{lo+20:<7}{100.0*len(sw)/len(sel):>10.0f}"
              f"{(100.0*len(inv)/len(sw) if sw else float('nan')):>11.0f}"
              f"{(np.mean(mc) if mc else float('nan')):>11.3f}"
              f"{(np.mean(ids) if ids else float('nan')):>11.3f}"
              f"{(np.mean(mo) if mo else float('nan')):>8.3f}")
    sw = [r for r in rows if float(r["touched"] or 0) >= swap_floor]
    print(f"\nswapped {len(sw)}/{len(rows)} frames; "
          f"inverted {sum(1 for r in sw if r['flip'] != '' and float(r['flip']) > 0)}")


if __name__ == "__main__":
    main()
