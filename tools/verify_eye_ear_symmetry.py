"""Confirm a symmetric, correct EAR ordering for both eyes.

The right eye's ordering was derived cleanly by tools/derive_eye_ear_order.py
(the contour is the contiguous run 87..96). The left eye's automatic search
was polluted by brow points sitting in the same height band, so this checks
the SYMMETRIC mapping instead: the 106 layout mirrors the two eye contours,
so the left-eye equivalent of a right-eye index is obtained by the same offset
within its own run.

Prints the resulting EAR range for both eyes. A correct EAR sits around
0.2-0.45 for an open eye and near 0.1 closed.
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


def ear(P, idx):
    q = P[list(idx)]
    w = max(1e-5, float(np.linalg.norm(q[0] - q[3])))
    return (float(np.linalg.norm(q[1] - q[5]))
            + float(np.linalg.norm(q[2] - q[4]))) / (2.0 * w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--faces", type=int, default=300)
    ap.add_argument("--stride", type=int, default=7)
    args = ap.parse_args()

    os.environ.setdefault("ROOP_EXECUTION_PROVIDER", "cpu")
    import roop.globals as G

    G.face_detector_threshold = 0.5
    G.face_detector_size = "512"
    G.default_det_size = True
    from roop.face_util import get_all_faces

    path = args.video
    if any(c in path for c in "*?[") or not os.path.exists(path):
        m = sorted(glob.glob(path), key=os.path.getmtime, reverse=True)
        if m:
            path = m[0]
    print(f"clip: {path}")
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
        print("cannot open")
        return 2
    N, i = [], 0
    while len(N) < args.faces:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        if i % args.stride == 0:
            try:
                faces = get_all_faces(fr) or []
            except Exception:
                faces = []
            for f in faces:
                lm = getattr(f, "landmark_2d_106", None)
                bb = getattr(f, "bbox", None)
                if lm is None or bb is None:
                    continue
                lm = np.asarray(lm, float)
                x0, y0, x1, y1 = [float(v) for v in bb]
                N.append(np.stack([(lm[:, 0] - x0) / max(1.0, x1 - x0),
                                   (lm[:, 1] - y0) / max(1.0, y1 - y0)], 1))
        i += 1
    cap.release()
    if len(N) < 20:
        print("not enough faces")
        return 2
    A = np.stack(N)
    mean = A.mean(0)
    print(f"{A.shape[0]} faces measured\n")

    # Derived for the right eye by tools/derive_eye_ear_order.py.
    right = (89, 95, 96, 93, 91, 87)
    # Same slots within the left contour. Left eye occupies 33..42, right
    # 87..96, both 10 points, so the offset within the run carries across.
    left = tuple(33 + (k - 87) for k in right)

    for name, order in (("LEFT ", left), ("RIGHT", right)):
        pts = mean[list(order)]
        v = np.array([ear(a, order) for a in A])
        print(f"{name} order {order}")
        print(f"   corners p0={order[0]} (x={pts[0,0]:.3f}) "
              f"p3={order[3]} (x={pts[3,0]:.3f})  width={abs(pts[3,0]-pts[0,0]):.3f}")
        print(f"   EAR min={v.min():.3f} median={np.median(v):.3f} max={v.max():.3f}")
        # Correlate with an independent openness reference from the same points.
        up = A[:, [order[1], order[2]]].mean(axis=1)
        lo = A[:, [order[4], order[5]]].mean(axis=1)
        w = np.linalg.norm(A[:, order[3]] - A[:, order[0]], axis=1)
        ref = np.linalg.norm(up - lo, axis=1) / np.maximum(w, 1e-6)
        if v.std() > 1e-9 and ref.std() > 1e-9:
            print(f"   correlation with lid separation: "
                  f"{np.corrcoef(v, ref)[0,1]:+.3f}")
        print()
    print("a correct EAR is ~0.2-0.45 open, ~0.1 closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
