"""Are the eye landmark index groups in temporal_expression correct, and in
the order the EAR formula requires?

WHY. `_ear(points, indices)` computes

    width = |p0 - p3|
    EAR   = (|p1 - p5| + |p2 - p4|) / (2 * width)

which is the standard eye-aspect-ratio, and it is only that if p0 and p3 are
the eye CORNERS (the widest-apart pair) and (p1,p5), (p2,p4) are the two
upper/lower lid pairs between them. A real EAR sits around 0.2-0.45 for an
open eye and near 0.1 when closed. The shipped groups produce 1.8-3.6, which
means p0/p3 are not the corners -- the ratio is upside down, so blink
detection reads a blink as an opening and vice versa.

This measures the real 2d106det output and reports, per group: each index's
position in the face box, which pair is actually the widest (the corners),
and the EAR the current ordering yields against the EAR a corrected ordering
yields.
"""
import argparse
import glob
import itertools
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


def ear(points, idx):
    p = points[list(idx)]
    width = max(1e-5, float(np.linalg.norm(p[0] - p[3])))
    return (float(np.linalg.norm(p[1] - p[5]))
            + float(np.linalg.norm(p[2] - p[4]))) / (2.0 * width)


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
    from roop.temporal_expression import LEFT_EYE, RIGHT_EYE

    cap = open_clip(args.video)
    norm = []
    idx = 0
    while len(norm) < args.faces:
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
                lm = np.asarray(lm, np.float64)
                x0, y0, x1, y1 = [float(v) for v in bbox]
                norm.append(np.stack([(lm[:, 0] - x0) / max(1.0, x1 - x0),
                                      (lm[:, 1] - y0) / max(1.0, y1 - y0)], 1))
        idx += 1
    cap.release()

    if len(norm) < 10:
        print("not enough faces")
        return 2
    A = np.stack(norm)
    mean = A.mean(axis=0)
    print(f"{A.shape[0]} faces measured\n")

    for name, group in (("LEFT_EYE", LEFT_EYE), ("RIGHT_EYE", RIGHT_EYE)):
        print(f"{name} = {group}")
        for slot, i in enumerate(group):
            print(f"  p{slot} = idx {i:3d}   x={mean[i,0]:.3f} y={mean[i,1]:.3f}")

        pts = mean[list(group)]
        # Which pair is genuinely the widest — those are the corners.
        best = max(itertools.combinations(range(6), 2),
                   key=lambda ab: np.linalg.norm(pts[ab[0]] - pts[ab[1]]))
        w_cur = float(np.linalg.norm(pts[0] - pts[3]))
        w_best = float(np.linalg.norm(pts[best[0]] - pts[best[1]]))
        print(f"  width used by the formula (p0-p3) : {w_cur:.4f}")
        print(f"  widest pair present  (p{best[0]}-p{best[1]})     : {w_best:.4f}"
              f"   -> indices {group[best[0]]},{group[best[1]]}")

        vals = np.array([ear(a, group) for a in A])
        print(f"  EAR as shipped : min={vals.min():.3f} "
              f"median={np.median(vals):.3f} max={vals.max():.3f}")

        # Reorder so p0/p3 are the true corners and the remaining four are
        # split into the two lid pairs by their position along the eye axis.
        corners = [group[best[0]], group[best[1]]]
        if mean[corners[0], 0] > mean[corners[1], 0]:
            corners.reverse()
        rest = [i for i in group if i not in corners]
        rest.sort(key=lambda i: mean[i, 0])
        upper = [i for i in rest if mean[i, 1] < mean[list(group), 1].mean()]
        lower = [i for i in rest if mean[i, 1] >= mean[list(group), 1].mean()]
        if len(upper) == 2 and len(lower) == 2:
            fixed = (corners[0], upper[0], upper[1],
                     corners[1], lower[1], lower[0])
            vals2 = np.array([ear(a, fixed) for a in A])
            print(f"  corrected order {fixed}")
            print(f"  EAR corrected  : min={vals2.min():.3f} "
                  f"median={np.median(vals2):.3f} max={vals2.max():.3f}")
        else:
            print(f"  could not split lids cleanly: upper={upper} lower={lower}")
        print()

    print("a real EAR is ~0.2-0.45 open and ~0.1 closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
