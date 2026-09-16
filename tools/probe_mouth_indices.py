"""Do the mouth landmark index pairs in temporal_expression actually measure
mouth opening and mouth width?

WHY. `temporal_expression` decides when to restore the target's own mouth by
watching a "mouth aspect ratio" built from two hard-coded index pairs:

    MOUTH_VERTICAL   = (52, 61)
    MOUTH_HORIZONTAL = (53, 59)

If those indices are not the points they are named for, the ratio does not
track mouth opening, the open/closed thresholds (0.48 / 0.70) never mean what
they say, and the expression restore fires on the wrong frames -- which is
what a "mouth opening is not good" report looks like from the inside.

This measures it rather than arguing from a table. It runs the real 2d106det
model over real faces, then:

  * locates the mouth corners and the lip centres by GEOMETRY (extreme x, and
    the vertical extremes near the mouth's horizontal centre);
  * scores every candidate pair by how strongly it CORRELATES with a
    geometry-derived openness signal across frames, which is the property the
    code actually needs;
  * prints what the shipped pairs measure, next to the best available.
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
        matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if matches:
            path = matches[0]
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
    ap.add_argument("--faces", type=int, default=150)
    ap.add_argument("--stride", type=int, default=13)
    args = ap.parse_args()

    os.environ.setdefault("ROOP_EXECUTION_PROVIDER", "cpu")
    import roop.globals as G

    G.face_detector_threshold = 0.5
    G.face_detector_size = "512"
    G.default_det_size = True
    from roop.face_util import get_all_faces

    cap = open_clip(args.video)
    samples = []
    idx = 0
    while len(samples) < args.faces:
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
                w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
                samples.append(np.stack([(lm[:, 0] - x0) / w,
                                         (lm[:, 1] - y0) / h], 1))
        idx += 1
    cap.release()

    if len(samples) < 10:
        print(f"only {len(samples)} faces; need more")
        return 2
    A = np.stack(samples)
    mean = A.mean(axis=0)
    print(f"{A.shape[0]} faces measured\n")

    # The mouth cluster, by position only.
    ys, xs = mean[:, 1], mean[:, 0]
    region = np.flatnonzero((ys > 0.66) & (ys < 0.95) & (xs > 0.30) & (xs < 0.72))

    left = int(region[np.argmin(mean[region, 0])])
    right = int(region[np.argmax(mean[region, 0])])
    cx = 0.5 * (mean[left, 0] + mean[right, 0])
    central = [int(i) for i in region if abs(mean[i, 0] - cx) < 0.045]
    top = min(central, key=lambda i: mean[i, 1])
    bottom = max(central, key=lambda i: mean[i, 1])

    print("measured by geometry:")
    print(f"  mouth corners      : {left} (x={mean[left,0]:.3f}) and "
          f"{right} (x={mean[right,0]:.3f})   width={mean[right,0]-mean[left,0]:.3f}")
    print(f"  central lip column : {sorted(central)}")
    print(f"  upper / lower      : {top} (y={mean[top,1]:.3f}) and "
          f"{bottom} (y={mean[bottom,1]:.3f})")

    # A reference openness signal per face: the vertical extent of the central
    # lip column, normalised by the corner-to-corner width. This is the thing
    # an aspect ratio is supposed to be.
    width = np.linalg.norm(A[:, right] - A[:, left], axis=1)
    ref = np.linalg.norm(A[:, top] - A[:, bottom], axis=1) / np.maximum(width, 1e-6)

    def score(pair):
        d = np.linalg.norm(A[:, pair[0]] - A[:, pair[1]], axis=1) / np.maximum(width, 1e-6)
        if d.std() < 1e-9 or ref.std() < 1e-9:
            return 0.0, d
        return float(np.corrcoef(d, ref)[0, 1]), d

    shipped_v = (52, 61)
    shipped_h = (53, 59)
    r_v, d_v = score(shipped_v)
    r_ref, _ = score((top, bottom))

    print("\nwhat the shipped constants actually are:")
    for name, pair in (("MOUTH_VERTICAL", shipped_v), ("MOUTH_HORIZONTAL", shipped_h)):
        a, b = pair
        print(f"  {name:17s} {pair}  "
              f"{a}(x={mean[a,0]:.3f},y={mean[a,1]:.3f})  "
              f"{b}(x={mean[b,0]:.3f},y={mean[b,1]:.3f})")

    sw = np.linalg.norm(A[:, shipped_h[0]] - A[:, shipped_h[1]], axis=1).mean()
    tw = width.mean()
    print(f"\n  MOUTH_HORIZONTAL spans {sw:.3f} of the box; the true mouth "
          f"width is {tw:.3f}  ({tw/max(sw,1e-6):.1f}x larger)")
    print(f"  MOUTH_VERTICAL correlation with real openness : {r_v:+.3f}")
    print(f"  best available pair ({top},{bottom})            : {r_ref:+.3f}")

    # Search every mouth-region pair for the best openness proxy, so the
    # replacement is chosen on evidence.
    best = []
    for i in region:
        for j in region:
            if j <= i:
                continue
            r, _ = score((int(i), int(j)))
            best.append((r, int(i), int(j)))
    best.sort(reverse=True)
    print("\ntop pairs by correlation with real mouth openness:")
    for r, i, j in best[:8]:
        print(f"  ({i:3d},{j:3d})  r={r:+.3f}   "
              f"dx={abs(mean[i,0]-mean[j,0]):.3f} dy={abs(mean[i,1]-mean[j,1]):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
