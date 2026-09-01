"""Render one unchanged configuration N times and report how much it moves.

WHY. Every pixel-level A/B this project runs -- "is this refactor equivalent",
"did this feature change anything", "is the parallel path seam-free" -- compares
two rendered outputs and reads a difference. None of them is interpretable
without knowing what TWO IDENTICAL RUNS produce, and that number had never been
measured here.

It is not zero. On the RTX 4070, `double/d4.mp4` frames 0..60, RealSwap /
RealityUX / GPEN 256 Pro / TensorRT / libx264, two renders of one unchanged
configuration differ on EVERY frame:

    mean |diff| 0.7466/255      max |diff| 28/255

and that floor is unmoved by the obvious suspects:

    threads 12 -> 1          mean 0.7469   (not worker scheduling, and not the
                                            RAM-derived stabilizer geometry)
    tensorrt -> cuda         mean 0.8921   (not a TensorRT tactic choice)
    PYTHONHASHSEED=0         mean 0.7804   (not set/dict iteration order)

Frame 0 already differs, at one worker, so it is not temporal state divergence
either; the detected bounding boxes agree exactly while the identity cosines in
`rows.csv` do not, so the divergence begins at or after the swap rather than in
detection. What remains is non-deterministic GPU reduction order, which neither
provider avoids.

CONSEQUENCE, and the reason this file is permanent: a pixel difference at or
below this floor is not evidence that a feature ran. That is not hypothetical --
`--identity-detail-strength 0.35` produced mean 0.766 against this 0.747 floor
while the `identity_detail` stage never appeared in the ROOP_PROFILE table at
all. Use `ROOP_PROFILE=1` and read the stage's call count to prove execution;
use this floor to decide whether a pixel delta means anything.

The floor is codec-inclusive on purpose: it is measured through the same lossy
encode that every A/B is measured through, so the two are comparable. It is not
a claim about the pipeline's internal divergence, which is smaller.

    env/Scripts/python.exe tests/measure_output_noise_floor.py --runs 3
    env/Scripts/python.exe tests/measure_output_noise_floor.py --runs 2 \
        --end 120 --threads 12
"""
import argparse
import glob
import itertools
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import fixtures   # noqa: E402


def _rendered_clip(out_dir):
    """The swapped output the harness wrote, not the trimmed plate beside it."""
    found = [p for p in glob.glob(os.path.join(out_dir, "**", "*.mp4"),
                                  recursive=True)
             if os.path.basename(p).startswith("clip_")]
    if not found:
        raise SystemExit("no rendered clip under %s -- the render failed; read "
                         "its log before reading any number here" % out_dir)
    return max(found, key=os.path.getmtime)


def compare(path_a, path_b):
    """Mean and max absolute per-pixel difference over the shorter clip."""
    ca, cb = cv2.VideoCapture(path_a), cv2.VideoCapture(path_b)
    try:
        total, peak, n, differing = 0.0, 0.0, 0, 0
        while True:
            ok_a, fa = ca.read()
            ok_b, fb = cb.read()
            if not (ok_a and ok_b):
                break
            d = cv2.absdiff(fa, fb)
            total += float(d.mean())
            peak = max(peak, float(d.max()))
            differing += 1 if d.max() > 0 else 0
            n += 1
    finally:
        ca.release()
        cb.release()
    if n == 0:
        raise SystemExit("no comparable frames; the two renders disagree on "
                         "length or failed to open")
    return {"frames": n, "differing": differing,
            "mean": total / n, "max": peak}


def render(tag, out_root, args):
    out = os.path.join(out_root, tag)
    cmd = [sys.executable, os.path.join(HERE, "two_face_video.py"),
           "--tag", tag, "--video", args.video, "--sources", args.sources,
           "--start", str(args.start), "--end", str(args.end),
           "--capture", str(args.capture), "--provider", args.provider,
           "--swap-model", args.swap_model, "--enhancer", args.enhancer,
           "--mask-engine", args.mask_engine, "--codec", args.codec,
           "--tracking", "1", "--threads", str(args.threads), "--out", out]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    log = os.path.join(out_root, tag + ".log")
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    if rc != 0:
        raise SystemExit("render %s returned %s; see %s" % (tag, rc, log))
    return _rendered_clip(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3,
                    help="renders of the SAME configuration (>=2)")
    ap.add_argument("--video", default=fixtures.clip("double/d4.mp4"))
    ap.add_argument("--sources", default="harjot,gargee")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=60)
    ap.add_argument("--capture", type=int, default=4930,
                    help="pinned; a wall-clock capture scan makes the fixture "
                         "itself a function of machine speed")
    ap.add_argument("--provider", default=None, help="defaults to config.yaml")
    ap.add_argument("--swap-model", default=None, help="defaults to config.yaml")
    ap.add_argument("--enhancer", default=None, help="defaults to config.yaml")
    ap.add_argument("--mask-engine", default=None, help="defaults to config.yaml")
    ap.add_argument("--codec", default=None, help="defaults to config.yaml")
    ap.add_argument("--threads", type=int, default=None,
                    help="defaults to config.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.runs < 2:
        raise SystemExit("--runs must be at least 2; a floor needs a pair")

    # Bench the stack the user actually runs, for the same reason the harness
    # itself now syncs config: a floor measured on a different pipeline is a
    # floor for a different pipeline.
    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    args.provider = args.provider or str(cfg.provider)
    args.swap_model = args.swap_model or str(cfg.swap_model)
    args.enhancer = args.enhancer or str(cfg.selected_enhancer)
    args.mask_engine = args.mask_engine or str(cfg.mask_engine)
    args.codec = args.codec or str(cfg.output_video_codec)
    args.threads = args.threads or int(cfg.max_threads)

    out_root = args.out or os.path.join(tempfile.gettempdir(), "roop_noise_floor")
    os.makedirs(out_root, exist_ok=True)
    print("[floor] %s frames %s..%s | %s / %s / %s / %s / %s / %s threads"
          % (args.runs, args.start, args.end, args.swap_model, args.mask_engine,
             args.enhancer, args.provider, args.codec, args.threads), flush=True)

    clips = []
    for i in range(args.runs):
        tag = "floor_%d" % i
        clips.append(render(tag, out_root, args))
        print("[floor] rendered %s" % tag, flush=True)

    results = []
    for i, j in itertools.combinations(range(len(clips)), 2):
        r = compare(clips[i], clips[j])
        results.append(r)
        print("[floor] run %d vs run %d: %d frames, %d differing, "
              "mean %.4f/255, max %.0f/255"
              % (i, j, r["frames"], r["differing"], r["mean"], r["max"]),
              flush=True)

    means = [r["mean"] for r in results]
    print("\n[floor] NOISE FLOOR over %d pair(s): mean %.4f/255 "
          "(worst pair %.4f), max %.0f/255"
          % (len(results), float(np.mean(means)), max(means),
             max(r["max"] for r in results)))
    print("[floor] A pixel delta at or below this is not evidence that a "
          "feature ran. Prove execution from ROOP_PROFILE call counts.")


if __name__ == "__main__":
    main()
