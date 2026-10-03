"""Regenerate app/tests/data/synthetic_face_jitter.npz and print the smoother comparison.

What it builds. One real face (a frame of ``single/s3.mp4``), rendered into two 24 fps
videos with independent sensor noise (sigma 2.5/255) and encoded with x264 CRF 20, so the
detector sees what it sees on a real file:

* ``static``  - the face does not move: every wobble in the detections is detector jitter.
* ``moving``  - a KNOWN path: 0.4 Hz sway (25 px) plus one 60 px head turn in 8 frames.

The app's own detector (``get_all_faces``) gives the 5-point keypoints; a single frame whose
detection jumped to another face (measured: 1 in 300, 158 px off) is replaced by
interpolation, as the production tracker would treat it as a track break.

The scores below (how much of the >2 Hz and >4 Hz keypoint jitter each smoother removes on
the static clip; lag and error against the exact truth on the moving clip) are ASSERTED by
app/tests/test_kalman_kps_stabilizer.py against the stored detections:

    smoother                              static >2 Hz   >4 Hz    sway lag   head-turn peak   rms
    AdaptiveLandmarkSmoother (shipped)       -67.4 %     -73.8 %    1.00 f      2.30 px       1.94 px
    One Euro 0.1 / 0.1 (non-default path)    -50.8 %     -58.0 %    0.50 f      1.65 px       1.13 px
    KalmanKpsStabilizer (option)             -77.0 %     -81.0 %    0.00 f      1.79 px       1.01 px
    raw detector                                                                1.41 px       0.90 px

The brief asked for >= 80% HF jitter reduction without lag. On a still head only the Kalman
option reaches it. On REAL conversational footage (heads move, mouths talk) neither it nor
any other filter reaches it, because the >4 Hz band there is partly real motion: Weeds and
Monica Bellucci clips, continuous single-face runs of 100-126 frames, >4 Hz band removed:
shipped 12.2% / 9.0%, Kalman 11.6% / 6.4%.

Run (app stopped; ~2 min)::

    app\\env\\Scripts\\python.exe tools/gen_jitter_fixture.py [--write]

``--write`` replaces the stored fixture; without it only the detector statistics are printed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app"
for p in (str(APP), str(APP / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

FPS = 24.0
N = 300
NOISE = 2.5


def media_dir() -> Path:
    env = os.environ.get("ROOP_KEEP_DIR")
    return Path(env) if env else REPO.parents[1] / "roop-keep"


def synth(work: Path, name, shift_fn, base, seed):
    import cv2
    import numpy as np

    from roop.ffmpeg_path import ffmpeg_binary
    rng = np.random.default_rng(seed)
    h, w = base.shape[:2]
    path = work / f"{name}.mp4"
    proc = subprocess.Popen([ffmpeg_binary(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20",
                             "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    shifts = []
    for t in range(N):
        dx, dy = shift_fn(t)
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(base, m, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        frame += rng.normal(0, NOISE, frame.shape).astype(np.float32)
        proc.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())
        shifts.append((dx, dy))
    proc.stdin.close()
    proc.wait()
    return path, np.array(shifts, np.float32)


def detect(path):
    import cv2
    import numpy as np

    from roop.face_util import get_all_faces
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = get_all_faces(frame)
        out.append(max(faces, key=lambda f: f.bbox[2] - f.bbox[0]).kps.astype(np.float64) if faces
                   else np.full((5, 2), np.nan))
    return np.stack(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write", action="store_true", help="replace the stored fixture")
    parser.add_argument("--clip", default=str(media_dir() / "single" / "s3.mp4"))
    parser.add_argument("--frame", type=int, default=20)
    args = parser.parse_args()

    import angle_bench as ab
    import cv2
    import numpy as np
    from settings import Settings

    cfg = Settings(str(APP / "config.yaml"))
    ab.init_pipeline(cfg.provider, cfg.swap_model, None, None, sync_config=True)

    cap = cv2.VideoCapture(args.clip)
    base = None
    for _ in range(args.frame + 1):
        ok, base = cap.read()
        if not ok:
            raise SystemExit(f"cannot read frame {args.frame} of {args.clip}")

    def burst(t):
        sway = 25 * np.sin(2 * np.pi * 0.4 * t / FPS)
        ramp = 60 * np.clip((t - 150) / 8.0, 0, 1) - 60 * np.clip((t - 200) / 8.0, 0, 1)
        return sway + ramp, 0.4 * sway

    work = Path(tempfile.mkdtemp(prefix="jitter_"))
    static_path, _ = synth(work, "static", lambda t: (0.0, 0.0), base, 7)
    moving_path, shift = synth(work, "moving", burst, base, 7)
    xs, xm = detect(static_path), detect(moving_path)
    base_kps = np.nanmedian(xs, axis=0)
    bad = ~(np.linalg.norm(xs - base_kps, axis=2).mean(1) < 20)       # a track break, not jitter
    for p in range(5):
        for c in range(2):
            v = xs[:, p, c]
            v[bad] = np.interp(np.where(bad)[0], np.where(~bad)[0], v[~bad])
    print(f"static: {int(bad.sum())} outlier frame(s) interpolated; detector sigma "
          f"{np.sqrt(np.mean(np.var(xs, axis=0))):.2f} px")

    fixture = APP / "tests" / "data" / "synthetic_face_jitter.npz"
    if args.write:
        fixture.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fixture, static_kps=xs.astype(np.float32), moving_kps=xm.astype(np.float32),
                            moving_shift=shift, base_kps=base_kps.astype(np.float32), fps=np.float32(FPS))
        print("wrote", fixture)
    print("Scores for these detections: pytest app/tests/test_kalman_kps_stabilizer.py -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
