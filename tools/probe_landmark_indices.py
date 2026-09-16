"""Which indices of insightface's 106-point landmark set are the mouth?

WHY MEASURE INSTEAD OF READ. Two places in this codebase slice the mouth out
of `landmark_2d_106` by hard-coded index -- `create_mouth_mask` uses
`landmarks[52:71]` and `temporal_expression` reads MOUTH_VERTICAL = (52, 61).
The 106-point layout is a published but frequently mis-cited ordering, and a
wrong slice does not crash: it silently drives the mouth restore, the lip-sync
paste box and the expression event detector off a region that is only
partially the mouth. That is exactly the shape of a "mouth opening is not
good" report.

This runs the ACTUAL 2d106det model over real faces and reports, for each
index, where it sits in the normalised face box. The mouth indices are then
identified by geometry (lower third, central horizontally) rather than by
trusting any table, and the code's current slice is scored against them.
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


def frames_from(pattern, count, stride):
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
    out = []
    idx = 0
    while len(out) < count:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx % stride == 0:
            out.append(frame)
        idx += 1
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--faces", type=int, default=40)
    ap.add_argument("--stride", type=int, default=97)
    args = ap.parse_args()

    os.environ.setdefault("ROOP_EXECUTION_PROVIDER", "cpu")
    import roop.globals as G

    G.face_detector_threshold = 0.5
    G.face_detector_size = "512"
    G.default_det_size = True

    from roop.face_util import get_all_faces

    collected = []
    for frame in frames_from(args.video, args.faces * 3, args.stride):
        try:
            faces = get_all_faces(frame) or []
        except Exception as exc:
            print(f"detect failed: {exc}")
            continue
        for f in faces:
            lm = getattr(f, "landmark_2d_106", None)
            bbox = getattr(f, "bbox", None)
            if lm is None or bbox is None:
                continue
            lm = np.asarray(lm, np.float64)
            x0, y0, x1, y1 = [float(v) for v in bbox]
            w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
            collected.append(np.stack([(lm[:, 0] - x0) / w, (lm[:, 1] - y0) / h], 1))
        if len(collected) >= args.faces:
            break

    if not collected:
        print("no faces with 106 landmarks found")
        return 2

    A = np.stack(collected)          # (N, 106, 2) in normalised box coords
    mean = A.mean(axis=0)
    print(f"\n{A.shape[0]} faces measured\n")

    # The mouth is the cluster in the lower third of the face box, centred
    # horizontally. Identified purely by position so no index table is trusted.
    ys, xs = mean[:, 1], mean[:, 0]
    mouth = np.flatnonzero((ys > 0.60) & (ys < 0.95) & (xs > 0.22) & (xs < 0.78))

    # Vertical spread across faces separates the LIPS (which open and close)
    # from the static chin/jaw points that share the same region.
    spread = A[:, :, 1].std(axis=0)

    print("index | mean x | mean y | y-variability")
    for i in mouth:
        print(f"  {i:3d} | {mean[i,0]:6.3f} | {mean[i,1]:6.3f} | {spread[i]:.4f}")

    print(f"\ngeometric mouth-region indices: {sorted(mouth.tolist())}")
    contiguous = []
    run = [mouth[0]]
    for a, b in zip(mouth, mouth[1:]):
        if b == a + 1:
            run.append(b)
        else:
            contiguous.append(run)
            run = [b]
    contiguous.append(run)
    print("contiguous runs:")
    for r in contiguous:
        print(f"  {r[0]}..{r[-1]}  ({len(r)} points)")

    used = set(range(52, 71))
    found = set(mouth.tolist())
    print(f"\ncode currently uses landmarks[52:71] = {sorted(used)}")
    print(f"  of those, actually in the mouth region : {sorted(used & found)}")
    print(f"  of those, NOT in the mouth region      : {sorted(used - found)}")
    print(f"  mouth points the slice MISSES          : {sorted(found - used)}")

    if used - found:
        wrong = sorted(used - found)
        print("\nwhere the non-mouth points in the slice actually sit:")
        for i in wrong:
            print(f"  {i:3d} -> x={mean[i,0]:.3f} y={mean[i,1]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
