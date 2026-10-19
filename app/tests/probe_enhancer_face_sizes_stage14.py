"""How many of the faces the restorer sees are small enough for a "small-face bypass" to matter? Measured on the roster clips.

    env/Scripts/python.exe tests/probe_enhancer_face_sizes_stage14.py [--frames 600] [--step 10]

Stage 14 proposed bypassing the 512 restorers (UltraMax / GPEN Realistic / RF++) when max(w, h) of the face in the source frame is
<= 128 px. The existing gate (`roop.enhance_gate`, `enhance_min_face_px`, default 0 = off) thresholds the SHORTER box side. This reports,
per clip, the share of detected faces under 80 / 100 / 128 px on BOTH definitions, via the app's own init and detector, so the gain
ceiling of turning the gate on (or defaulting it to 128) is a number rather than a guess. The saving is bounded by that share
of the enhancer stage (UltraMax: 99.75 ms/call, 11.0% of summed thread time in the 600-frame regression render, 2026-10-19).
Detection only; it renders nothing and says nothing about how a skipped face LOOKS (see the A/B in the CHANGELOG entry).
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

KEEP = os.environ.get("ROOP_MEDIA_DIR") or os.path.normpath(os.path.join(APP, "..", "..", "..", "roop-keep"))
CLIPS = ["double/d1.mp4", "double/d2.mp4", "double/d3.mp4", "double/d4.mp4", "double/d5.mp4", "double/d6.mp4",
         "single/s1.mp4", "single/s2.mp4", "single/s3.mp4", "single/s4.mp4", "single/s5.mp4", "single/s6.mp4", "single/s7.mp4",
         "Love.mp4", "Weeds.mp4", "Monica Bellucci .mp4", "b1.mp4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--step", type=int, default=10)
    args = ap.parse_args()
    import angle_bench as ab
    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), "None", "None", sync_config=True)
    from roop.face_util import get_all_faces
    print("clip | frame WxH | faces | short<80 <100 <128 | max<=80 <=100 <=128 | short side p10/p50/p90")
    allshort, allmax = [], []
    for rel in CLIPS:
        path = os.path.join(KEEP, rel)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(rel, "| cannot open")
            continue
        w, h = int(cap.get(3)), int(cap.get(4))
        short, mx = [], []
        for i in range(args.frames):
            if i % args.step:
                if not cap.grab():
                    break
                continue
            ok, fr = cap.read()
            if not ok:
                break
            for f in get_all_faces(fr) or []:
                x0, y0, x1, y1 = (float(v) for v in f.bbox[:4])
                short.append(min(x1 - x0, y1 - y0))
                mx.append(max(x1 - x0, y1 - y0))
        cap.release()
        if not short:
            print("%-22s | %dx%d | 0 faces" % (rel, w, h))
            continue
        s, m = np.array(short), np.array(mx)
        allshort += short
        allmax += mx
        pc = lambda a, t, op: 100.0 * np.mean(op(a, t))        # noqa: E731
        print("%-22s | %dx%d | %4d | %5.1f %5.1f %5.1f | %5.1f %5.1f %5.1f | %.0f/%.0f/%.0f" % (
            rel, w, h, len(s), pc(s, 80, np.less), pc(s, 100, np.less), pc(s, 128, np.less),
            pc(m, 80, np.less_equal), pc(m, 100, np.less_equal), pc(m, 128, np.less_equal),
            *np.percentile(s, [10, 50, 90])))
    s, m = np.array(allshort), np.array(allmax)
    print("ALL %d faces: short<128 %.1f%%  max<=128 %.1f%%  short<80 %.1f%%  median short side %.0f px" % (
        len(s), 100 * np.mean(s < 128), 100 * np.mean(m <= 128), 100 * np.mean(s < 80), np.median(s)))


if __name__ == "__main__":
    main()
