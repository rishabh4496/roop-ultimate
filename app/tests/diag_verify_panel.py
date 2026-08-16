"""Standing visual-verification format: original / swapped / amplified diff
heatmap, side by side, labeled with the frame's yaw/pitch/roll. Built per the
user's 2026-08-15 instruction after this session's two false "looks fixed"
calls (a swap silently discarded looks identical to the original at a glance;
only a diff/heatmap reliably tells them apart).

Usage:
    env/Scripts/python.exe tests/diag_verify_panel.py \
        --original PATH --swapped PATH --timestamps 3,7.5,12 \
        --out DIR [--crop X,Y,W,H] [--amplify 8]

Writes one PNG per timestamp: "{out}/panel_{t}.png", each a 3-up strip
(original | swapped | amplified |diff|) with a text header giving yaw/pitch/
roll (from the ORIGINAL frame's detected face, since that's the ground-truth
pose regardless of what the swap did to it) and the mean/max diff in the
compared region.
"""
import argparse
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


def _frame_at_index(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read {path} at frame {idx}")
    return frame


def _probe(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n


def _pose_label(frame):
    try:
        from roop.face_util import get_all_faces, solve_pose_5pt
        faces = get_all_faces(frame) or []
        if not faces:
            return "no face detected"
        f = faces[0]
        pose = solve_pose_5pt(getattr(f, 'kps', None))
        if pose is None:
            return "pose: n/a"
        yaw, pitch, roll = pose
        return f"yaw {yaw:+.1f}  pitch {pitch:+.1f}  roll {roll:+.1f}"
    except Exception as e:
        return f"pose: error ({e})"


def build_panel(original_path, swapped_path, t, out_dir, crop=None, amplify=8):
    # Neither wall-clock timestamp NOR raw frame index is safe to assume
    # blindly — the swap pipeline's output can differ from the source in
    # TWO independent ways, both seen on real sample clips:
    #   (a) same frame count, different container fps (d1.mp4: 25fps/832 in,
    #       30fps/832 out — every source frame processed exactly once, just
    #       re-labeled at a different fps, so INDEX correspondence holds and
    #       real-world duration does NOT: 33.28s in, 27.73s out).
    #   (b) different frame count, matching real-world duration (d8.mp4:
    #       60fps/2434 in, 30fps/1217 out — a genuine frame-rate DECIMATION,
    #       both represent the same ~40.6s, so DURATION correspondence holds
    #       and index does NOT).
    # A single proportional scale handles both without having to detect which
    # regime applies: resolve the source index from wall-clock time, then
    # scale it by the frame-COUNT ratio (not an fps ratio) to get the swap
    # index. When counts match this reduces to (a)'s index-for-index; when
    # counts differ proportionally to the fps change this reduces to (b)'s
    # duration-for-duration, i.e. effectively `t * swap_fps`.
    src_fps, src_n = _probe(original_path)
    _, swap_n = _probe(swapped_path)
    src_idx = round(t * src_fps)
    swap_idx = round(src_idx * swap_n / src_n) if src_n else src_idx
    ratio = swap_n / src_n if src_n else 1.0
    if not (0.98 <= ratio <= 1.02) and not (0.48 <= ratio <= 0.52):
        print(f"[panel] WARNING: frame-count ratio {ratio:.3f} (original={src_n}, "
              f"swapped={swap_n}) doesn't match a known pattern (1:1 or 2:1) — "
              f"proportional scaling may be wrong here; treat with suspicion.", flush=True)

    orig = _frame_at_index(original_path, src_idx)
    swap = _frame_at_index(swapped_path, swap_idx)
    if orig.shape != swap.shape:
        swap = cv2.resize(swap, (orig.shape[1], orig.shape[0]))

    pose = _pose_label(orig)

    if crop:
        x, y, w, h = crop
        oc = orig[y:y + h, x:x + w]
        sc = swap[y:y + h, x:x + w]
    else:
        oc, sc = orig, swap

    diff = np.abs(oc.astype(np.float32) - sc.astype(np.float32))
    mean_d = float(diff.mean())
    max_d = float(diff.max())
    heat = np.clip(diff.mean(axis=2) * amplify, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)

    h_target = 480
    def _resize(img):
        scale = h_target / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h_target))

    strip = np.hstack([_resize(oc), _resize(sc), _resize(heat_bgr)])
    header = np.zeros((60, strip.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, f"t={t}s  {pose}  |  diff mean={mean_d:.2f} max={max_d:.0f}  |  original / swapped / diff x{amplify}",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    out = np.vstack([header, strip])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"panel_{t}.png")
    cv2.imwrite(out_path, out)
    print(f"[panel] t={t}s {pose} diff_mean={mean_d:.2f} diff_max={max_d:.0f} -> {out_path}", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--swapped", required=True)
    ap.add_argument("--timestamps", required=True, help="comma-separated seconds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", default=None, help="x,y,w,h")
    ap.add_argument("--amplify", type=float, default=8)
    ap.add_argument("--no-pose", action="store_true",
                     help="skip model init/pose (fast, diff-only)")
    args = ap.parse_args()

    if not args.no_pose:
        import angle_bench as ab
        ab.init_pipeline("cuda", "hyperswap", "None", "None")
        import roop.globals as g
        g.g_desired_face_analysis = ["landmark_3d_68", "landmark_2d_106", "detection", "recognition"]

    crop = tuple(int(v) for v in args.crop.split(",")) if args.crop else None
    for t in [float(x) for x in args.timestamps.split(",")]:
        build_panel(args.original, args.swapped, t, args.out, crop=crop, amplify=args.amplify)


if __name__ == "__main__":
    main()
