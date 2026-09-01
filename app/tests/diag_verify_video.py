"""Standing visual-verification format: a single VIDEO with three synced
panels (original / newly-generated swap / amplified diff heatmap), each frame
headed with yaw/pitch/roll and diff mean/max. Supersedes both
`diag_verify_panel.py` (stills only, no temporal coverage) and a plain 2-panel
side-by-side video (no heatmap, no pose labels) per the user's 2026-08-15
instruction — see memory: visual-verification-standard.

Frame correspondence between original and swapped is derived from the
ORIGINAL's own fps/frame-count first (index correspondence when counts match,
proportional scaling when they don't — see `_swap_index`), never from each
video's own fps independently: source and swapped-output videos are frequently
re-encoded at a different fps even when every frame was processed exactly
once, which silently pulls mismatched frames if you convert timestamp->index
per-video (see the alignment-lesson note in the memory file above).

Usage:
    env/Scripts/python.exe tests/diag_verify_video.py \
        --original PATH --swapped PATH --out OUT.mp4 \
        --start-sec 20 --end-sec 30 [--amplify 8] [--stacked] [--pose-stride 1]
"""

import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _find_ffmpeg():
    import shutil as _sh
    exe = _sh.which("ffmpeg")
    if exe:
        return exe
    # Resolve under whichever PINOKIO_HOME is live rather than naming
    # the 4070 host's drive, which does not exist on the 3060.
    import fixtures as _fx
    for cand in (os.path.join(_fx.pinokio_home(), "bin",
                              "ffmpeg-env", "Library", "bin"),):
        p = os.path.join(cand, "ffmpeg.exe")
        if os.path.exists(p):
            return p
    raise RuntimeError("ffmpeg not found; set --ffmpeg explicitly")


def _probe(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n


def _swap_index(src_idx, src_n, swap_n):
    if src_n == swap_n or not src_n:
        return src_idx
    return int(round(src_idx * swap_n / src_n))


def _pose_label(frame, get_all_faces, solve_pose_5pt):
    try:
        faces = get_all_faces(frame) or []
        if not faces:
            return "no face", None
        f = faces[0]
        pose = solve_pose_5pt(getattr(f, 'kps', None))
        if pose is None:
            return "pose n/a", None
        yaw, pitch, roll = pose
        return f"yaw {yaw:+.1f} pitch {pitch:+.1f} roll {roll:+.1f}", pose
    except Exception as e:
        return f"pose err", None


def build(original, swapped, out, start_sec=0.0, end_sec=0.0, amplify=8.0,
          stacked=False, pose_stride=1, ffmpeg=None, no_pose=False):
    src_fps, src_n = _probe(original)
    _, swap_n = _probe(swapped)
    if src_n == 0:
        raise SystemExit(f"could not open {original}")
    if swap_n == 0:
        raise SystemExit(f"could not open {swapped}")
    ratio = swap_n / src_n if src_n else 1.0
    if not (0.98 <= ratio <= 1.02) and not (0.48 <= ratio <= 0.52):
        print(f"[verify-video] WARNING: frame-count ratio {ratio:.3f} "
              f"(original={src_n}, swapped={swap_n}) doesn't match a known "
              f"1:1 or 2:1 pattern — proportional scaling may be wrong here.",
              flush=True)

    get_all_faces = solve_pose_5pt = None
    if not no_pose:
        import angle_bench as ab
        ab.init_pipeline("cuda", "hyperswap", "None", "None")
        import roop.globals as g
        g.g_desired_face_analysis = ["landmark_3d_68", "landmark_2d_106",
                                      "detection", "recognition"]
        from roop.face_util import get_all_faces, solve_pose_5pt

    start_idx = max(0, min(int(round(start_sec * src_fps)), src_n - 1))
    end_idx = src_n if end_sec <= 0 else min(src_n, int(round(end_sec * src_fps)))

    ocap = cv2.VideoCapture(original)
    scap = cv2.VideoCapture(swapped)
    ocap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    ffmpeg = ffmpeg or _find_ffmpeg()
    proc = None
    written = 0
    last_pose_text = "pose n/a"

    for i in range(start_idx, end_idx):
        ok, oimg = ocap.read()
        if not ok:
            break
        s_idx = _swap_index(i, src_n, swap_n)
        scap.set(cv2.CAP_PROP_POS_FRAMES, s_idx)
        ok2, simg = scap.read()
        if not ok2:
            break
        if simg.shape != oimg.shape:
            simg = cv2.resize(simg, (oimg.shape[1], oimg.shape[0]))

        diff = np.abs(oimg.astype(np.float32) - simg.astype(np.float32))
        mean_d = float(diff.mean())
        max_d = float(diff.max())
        heat = np.clip(diff.mean(axis=2) * amplify, 0, 255).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)

        if not no_pose and (i - start_idx) % pose_stride == 0:
            last_pose_text, _ = _pose_label(oimg, get_all_faces, solve_pose_5pt)

        t = i / src_fps
        header_text = (f"t={t:.2f}s f{i}  {last_pose_text}  |  "
                        f"diff mean={mean_d:.2f} max={max_d:.0f}  |  "
                        f"original / new-output / diff x{amplify:g}")

        panels = [oimg, simg, heat_bgr]
        row = np.hstack(panels) if not stacked else None
        col = np.vstack(panels) if stacked else None
        body = col if stacked else row

        header = np.zeros((36, body.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, header_text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        frame_out = np.vstack([header, body])

        if proc is None:
            h, w = frame_out.shape[:2]
            # x264 requires even dimensions
            w -= w % 2
            h -= h % 2
            proc = subprocess.Popen([
                ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{w}x{h}", "-r", str(src_fps), "-i", "-",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", out,
            ], stdin=subprocess.PIPE)
            out_w, out_h = w, h

        if frame_out.shape[1] != out_w or frame_out.shape[0] != out_h:
            frame_out = cv2.resize(frame_out, (out_w, out_h))
        proc.stdin.write(frame_out.tobytes())
        written += 1

    if proc is not None:
        proc.stdin.close()
        proc.wait()
    print(f"[verify-video] wrote {written} frames [{start_idx}..{end_idx}) -> {out}",
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--swapped", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--end-sec", type=float, default=0.0, help="0 = to the end")
    ap.add_argument("--amplify", type=float, default=8.0)
    ap.add_argument("--stacked", action="store_true",
                     help="vertical stack instead of horizontal side-by-side")
    ap.add_argument("--pose-stride", type=int, default=1,
                     help="recompute pose every N frames (>1 speeds this up; "
                          "the label holds the last computed value between)")
    ap.add_argument("--no-pose", action="store_true",
                     help="skip model init/pose (fast, diff-only, no yaw label)")
    ap.add_argument("--ffmpeg", default=None)
    args = ap.parse_args()
    build(args.original, args.swapped, args.out, args.start_sec, args.end_sec,
          args.amplify, args.stacked, max(1, args.pose_stride), args.ffmpeg,
          args.no_pose)


if __name__ == "__main__":
    main()
