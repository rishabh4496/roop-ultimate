"""Derive a correct 6-point EAR ordering for the InsightFace 106 eye contours.

The 106-point set gives 8 points per eye, not the 6 the classic EAR formula
expects, and the two lids are not labelled. This finds, per eye:

  * the CORNERS, as the widest-separated pair;
  * the remaining points split into upper and lower lids by which side of the
    corner-to-corner line they fall on (a signed cross product, so it works on
    a tilted head, unlike comparing raw y);
  * a 6-point ordering (corner, up1, up2, corner, low2, low1) whose EAR is
    scored against a reference openness taken from the lid centroids.

Prints the candidate orderings with their EAR ranges so the constants can be
chosen on evidence.
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
    ap.add_argument("--faces", type=int, default=300)
    ap.add_argument("--stride", type=int, default=7)
    args = ap.parse_args()

    os.environ.setdefault("ROOP_EXECUTION_PROVIDER", "cpu")
    import roop.globals as G

    G.face_detector_threshold = 0.5
    G.face_detector_size = "512"
    G.default_det_size = True
    from roop.face_util import get_all_faces

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

    if len(norm) < 20:
        print("not enough faces")
        return 2
    A = np.stack(norm)
    mean = A.mean(axis=0)
    print(f"{A.shape[0]} faces measured\n")

    # The eye CONTOURS in the 106 layout are two contiguous runs of 8. A
    # position band alone also catches brow, temple and jaw points that happen
    # to sit at eye height, which is what made a first attempt report an
    # 0.43-wide "eye". So take the band, then keep only the longest contiguous
    # run of indices within it — a contour is consecutive by construction,
    # whereas the intruders are scattered.
    ys, xs = mean[:, 1], mean[:, 0]
    band = np.flatnonzero((ys > 0.33) & (ys < 0.48))

    def longest_run(indices):
        indices = sorted(int(i) for i in indices)
        if not indices:
            return []
        runs, run = [], [indices[0]]
        for a, b in zip(indices, indices[1:]):
            if b == a + 1:
                run.append(b)
            else:
                runs.append(run)
                run = [b]
        runs.append(run)
        return max(runs, key=len)

    left = longest_run(i for i in band if xs[i] < 0.45)
    right = longest_run(i for i in band if xs[i] > 0.55)
    print(f"left-eye band indices : {left}")
    print(f"right-eye band indices: {right}\n")

    for name, group in (("LEFT_EYE", left), ("RIGHT_EYE", right)):
        if len(group) < 6:
            print(f"{name}: only {len(group)} points, skipping")
            continue
        pts = mean[group]
        ci, cj = max(itertools.combinations(range(len(group)), 2),
                     key=lambda ab: np.linalg.norm(pts[ab[0]] - pts[ab[1]]))
        c0, c1 = group[ci], group[cj]
        if mean[c0, 0] > mean[c1, 0]:
            c0, c1 = c1, c0
        axis = mean[c1] - mean[c0]

        upper, lower = [], []
        for i in group:
            if i in (c0, c1):
                continue
            v = mean[i] - mean[c0]
            cross = axis[0] * v[1] - axis[1] * v[0]
            (lower if cross > 0 else upper).append(i)
        upper.sort(key=lambda i: mean[i, 0])
        lower.sort(key=lambda i: mean[i, 0])

        print(f"{name}")
        print(f"  corners : {c0} (x={mean[c0,0]:.3f}) .. {c1} (x={mean[c1,0]:.3f})"
              f"   width={np.linalg.norm(axis):.4f}")
        print(f"  upper lid: {upper}")
        print(f"  lower lid: {lower}")

        # Reference openness: perpendicular distance between lid centroids,
        # normalised by corner width. Independent of any EAR ordering.
        if not upper or not lower:
            print("  cannot form lid pairs\n")
            continue
        up_c = A[:, upper].mean(axis=1)
        lo_c = A[:, lower].mean(axis=1)
        width = np.linalg.norm(A[:, c1] - A[:, c0], axis=1)
        ref = np.linalg.norm(up_c - lo_c, axis=1) / np.maximum(width, 1e-6)
        print(f"  reference openness: min={ref.min():.3f} "
              f"median={np.median(ref):.3f} max={ref.max():.3f}")

        best = None
        for u in itertools.combinations(upper, 2):
            for lo in itertools.combinations(lower, 2):
                order = (c0, u[0], u[1], c1, lo[1], lo[0])
                vals = np.array([ear(a, order) for a in A])
                if vals.std() < 1e-9:
                    continue
                r = float(np.corrcoef(vals, ref)[0, 1])
                score = (r, -abs(float(np.median(vals)) - 0.30))
                if best is None or score > best[0]:
                    best = (score, order, vals, r)
        if best is not None:
            _, order, vals, r = best
            print(f"  best ordering {order}")
            print(f"    EAR min={vals.min():.3f} median={np.median(vals):.3f} "
                  f"max={vals.max():.3f}  r={r:+.3f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
