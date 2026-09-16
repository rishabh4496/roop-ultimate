"""Does the SHIPPED measure_expression actually track mouth opening?

Runs roop.temporal_expression.measure_expression -- the real function the
render path calls -- over real detected faces, and correlates its
`mouth_openness` output against a geometry-derived reference computed
independently from the mouth corners and lip centres.

A working measure correlates near +1. The indices this replaced scored
+0.000, which is why the mouth half of expression restore never fired.

Also reports the value's range, because the decision that consumes it
(`TemporalExpressionEngine.plan`) compares against absolute thresholds
(0.12, and a running closed-reference + 0.035): a measure in the wrong
numeric regime cannot cross them even if it does vary.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def open_clip(pattern):
    path = pattern
    if any(ch in pattern for ch in "*?[") or not os.path.exists(pattern):
        m = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if m:
            path = m[0]
    shown = path
    if os.name == "nt":
        try:
            import ctypes

            buf = ctypes.create_unicode_buffer(1024)
            if ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 1024) and buf.value:
                path = buf.value
        except Exception:
            pass
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {shown}")
    print(f"clip: {shown}")
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--faces", type=int, default=200)
    ap.add_argument("--stride", type=int, default=11)
    args = ap.parse_args()

    os.environ.setdefault("ROOP_EXECUTION_PROVIDER", "cpu")
    import roop.globals as G

    G.face_detector_threshold = 0.5
    G.face_detector_size = "512"
    G.default_det_size = True
    from roop.face_util import get_all_faces
    from roop.temporal_expression import (measure_expression, MOUTH_VERTICAL,
                                          MOUTH_HORIZONTAL, LEFT_EYE, RIGHT_EYE)

    print(f"MOUTH_VERTICAL={MOUTH_VERTICAL}  MOUTH_HORIZONTAL={MOUTH_HORIZONTAL}")

    cap = open_clip(args.video)
    measured, reference, eye_vals, norm = [], [], [], []
    idx = 0
    while len(measured) < args.faces:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx % args.stride == 0:
            try:
                faces = get_all_faces(frame) or []
            except Exception:
                faces = []
            for f in faces:
                lm = getattr(f, "landmark_2d_106", None)
                bbox = getattr(f, "bbox", None)
                if lm is None or bbox is None:
                    continue
                out = measure_expression(lm, kps=getattr(f, "kps", None), bbox=bbox)
                if not out or out.get("confidence", 0.0) <= 0.0:
                    continue
                lm = np.asarray(lm, np.float64)
                # Independent reference: lip-centre separation over corner
                # separation, both straight from the landmark geometry.
                ref = (np.linalg.norm(lm[67] - lm[53])
                       / max(1e-6, np.linalg.norm(lm[52] - lm[61])))
                measured.append(float(out["mouth_openness"]))
                reference.append(float(ref))
                eye_vals.append((float(out["left_eye_openness"]),
                                 float(out["right_eye_openness"])))
                x0, y0, x1, y1 = [float(v) for v in bbox]
                norm.append(np.stack([(lm[:, 0] - x0) / max(1.0, x1 - x0),
                                      (lm[:, 1] - y0) / max(1.0, y1 - y0)], 1))
        idx += 1
    cap.release()

    if len(measured) < 10:
        print("not enough faces")
        return 2
    m = np.asarray(measured)
    r = np.asarray(reference)
    print(f"\n{m.size} faces measured")
    print(f"mouth_openness  min={m.min():.3f} median={np.median(m):.3f} "
          f"max={m.max():.3f}")
    print(f"correlation with independent geometry : {np.corrcoef(m, r)[0,1]:+.3f}")

    # The decision thresholds in TemporalExpressionEngine.plan.
    base = float(np.percentile(m, 10))       # what the closed-reference tracks
    fires = int((m > max(0.12, base + 0.035)).sum())
    print(f"\nclosed reference (p10)  : {base:.3f}")
    print(f"frames the mouth event would fire on: {fires}/{m.size} "
          f"({100.0*fires/m.size:.1f}%)")
    if fires == 0:
        print("  -> the mouth restore can never trigger; the measure is in the "
              "wrong numeric regime for its thresholds")
    elif fires == m.size:
        print("  -> the mouth restore would fire on EVERY frame; the thresholds "
              "no longer discriminate")

    # Eye indices, checked the same way, since they sit beside the mouth ones.
    A = np.stack(norm)
    mean = A.mean(axis=0)
    print("\neye index positions (normalised to the face box):")
    for name, group in (("LEFT_EYE", LEFT_EYE), ("RIGHT_EYE", RIGHT_EYE)):
        pts = mean[list(group)]
        print(f"  {name:9s} x {pts[:,0].min():.3f}..{pts[:,0].max():.3f}  "
              f"y {pts[:,1].min():.3f}..{pts[:,1].max():.3f}")
    ev = np.asarray(eye_vals)
    print(f"  EAR left  min={ev[:,0].min():.3f} median={np.median(ev[:,0]):.3f} "
          f"max={ev[:,0].max():.3f}")
    print(f"  EAR right min={ev[:,1].min():.3f} median={np.median(ev[:,1]):.3f} "
          f"max={ev[:,1].max():.3f}")
    print("  (a plausible EAR sits around 0.2-0.45 open, near 0.1 closed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
