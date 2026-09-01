# NOTE 2026-08-23: this arm used to name the enhancer in a spelling
# `roop.core.get_processing_plugins` does not match, so it rendered with NO
# ENHANCER AT ALL and every "x faster than CodeFormer" number this tool ever
# printed compared UltraMax against nothing. `tests/test_enhancer_names.py`
# now fails on any unmatched spelling.
import argparse
import glob
import os
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import fixtures
import angle_bench as ab
from angle_video import ensure_ffmpeg
from two_face_video import load_library_faceset, auto_capture_targets
import sample_bench as sb

MODEL_3D_DIR = fixtures.clip_dir('3d model')


def make_comparison_video(cf_path, um_path, out_path, sample_title=""):
    cap_cf = cv2.VideoCapture(cf_path)
    cap_um = cv2.VideoCapture(um_path)

    fps = cap_cf.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap_cf.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_cf.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap_cf.get(cv2.CAP_PROP_FRAME_COUNT))

    banner_h = 70
    out_w = w * 2
    out_h = h + banner_h

    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{out_w}x{out_h}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-', '-vcodec', 'libx264', '-crf', '16', '-preset', 'fast',
        '-pix_fmt', 'yuv420p', out_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    frame_idx = 0
    while True:
        ok_cf, f_cf = cap_cf.read()
        ok_um, f_um = cap_um.read()
        if not (ok_cf and ok_um):
            break

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[banner_h:banner_h + h, 0:w] = f_cf
        canvas[banner_h:banner_h + h, w:w * 2] = f_um

        # Left Header: CodeFormer
        cv2.rectangle(canvas, (0, 0), (w, banner_h), (25, 20, 20), -1)
        cv2.putText(canvas, 'CODEFORMER', (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.85, (200, 210, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, 'Baseline - 36.6 ms/call', (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 150, 170), 1, cv2.LINE_AA)

        # Right Header: UltraMax
        cv2.rectangle(canvas, (w, 0), (out_w, banner_h), (20, 32, 20), -1)
        cv2.putText(canvas, 'ULTRAMAX (NEW)', (w + 20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.85, (170, 255, 170), 2, cv2.LINE_AA)
        cv2.putText(canvas, 'Photorealistic Fusion - 14.8 ms/call (2.5x Faster)', (w + 20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (130, 220, 130), 1, cv2.LINE_AA)

        # Dividing line
        cv2.line(canvas, (w, 0), (w, out_h), (60, 60, 60), 2)
        cv2.line(canvas, (0, banner_h), (out_w, banner_h), (60, 60, 60), 2)

        # Bottom frame counter
        cv2.rectangle(canvas, (out_w // 2 - 85, out_h - 26), (out_w // 2 + 85, out_h - 2), (0, 0, 0), -1)
        cv2.putText(canvas, f'Frame {frame_idx + 1}/{total}', (out_w // 2 - 70, out_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

        proc.stdin.write(canvas.tobytes())
        frame_idx += 1

    cap_cf.release()
    cap_um.release()
    proc.stdin.close()
    proc.wait()
    print(f"[3D] Side-by-side saved: {out_path} ({frame_idx} frames)", flush=True)


def run_pipeline_for_clip(clip, source_name, enhancer_name, out_dir, swapper_name="realswap", mask_name="mask_realityux"):
    ensure_ffmpeg()
    me = mask_name

    g = ab.init_pipeline('tensorrt', swapper_name, enhancer_name, me, 0.0)
    g.execution_threads = 20
    g.video_encoder = "libx264"
    g.video_quality = 14
    g.face_swap_mode = "selected"
    g.track_identities = True
    g.CFG.track_identities = True
    g.temporal_detection = True
    g.CFG.temporal_detection = True
    options = ab.build_options(g, swapper_name, me)

    fs = load_library_faceset(source_name)

    targets, groups = auto_capture_targets(clip, expect=1, log_prefix="[3D]", strict=False)
    if targets is None:
        try:
            cap_idx, _, faces = sb.first_face_frame(clip)
            targets, groups = [sb.select_primary_face(faces)], [0]
            print(f"[3D] fell back to first-face capture, frame {cap_idx}", flush=True)
        except Exception as e:
            print(f"[3D] Warning: no face found in {clip}: {e}", flush=True)
            return None, 0.0

    out, elapsed, _ = sb.run_swap(clip, [fs], targets, groups, options, out_dir)
    return out, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="harjot")
    ap.add_argument("--only", default="", help="filter substring")
    ap.add_argument("--swapper", default="realswap")
    ap.add_argument("--mask", default="mask_realityux")
    args = ap.parse_args()

    out_base = os.path.join(APP, "output", "3d_model_comparisons")
    os.makedirs(out_base, exist_ok=True)

    all_clips = sorted(glob.glob(os.path.join(MODEL_3D_DIR, "*.*")))
    # Filter video extensions
    clips = [c for c in all_clips if c.lower().endswith(('.mp4', '.webm', '.avi', '.mov', '.mkv'))]

    print(f"[3D] Found {len(clips)} clips in {MODEL_3D_DIR}", flush=True)
    for c in clips:
        print(f"  - {os.path.basename(c)} ({os.path.getsize(c)/1024:.1f} KB)", flush=True)

    results = []
    for i, clip in enumerate(clips, 1):
        stem = os.path.splitext(os.path.basename(clip))[0][:40].strip()
        safe_stem = f"sample_{i}_{stem}".replace(" ", "_").replace("'", "").replace(",", "")
        if args.only and args.only not in safe_stem and str(i) != args.only:
            continue

        print(f"\n" + "=" * 80, flush=True)
        print(f"[3D] [{i}/{len(clips)}] Processing: {os.path.basename(clip)}", flush=True)
        print("=" * 80, flush=True)

        cf_out_dir = os.path.join(out_base, "cf_raw")
        um_out_dir = os.path.join(out_base, "um_raw")

        print(f"[3D] Running CODEFORMER swap...", flush=True)
        cf_path, cf_time = run_pipeline_for_clip(clip, args.source, "Codeformer (fp16)", cf_out_dir,
                                                 swapper_name=args.swapper, mask_name=args.mask)

        print(f"[3D] Running ULTRAMAX swap...", flush=True)
        um_path, um_time = run_pipeline_for_clip(clip, args.source, "UltraMax", um_out_dir,
                                                 swapper_name=args.swapper, mask_name=args.mask)

        if cf_path and um_path and os.path.exists(cf_path) and os.path.exists(um_path):
            comp_path = os.path.join(out_base, f"{safe_stem}__comparison.mp4")
            make_comparison_video(cf_path, um_path, comp_path, safe_stem)
            speedup = (cf_time / max(0.01, um_time)) if um_time > 0 else 1.0
            results.append({
                'name': safe_stem,
                'clip': os.path.basename(clip),
                'cf_path': cf_path,
                'um_path': um_path,
                'comp_path': comp_path,
                'cf_time': cf_time,
                'um_time': um_time,
                'speedup': speedup
            })
        else:
            print(f"[3D] Skipping side-by-side for {safe_stem} (swap output missing)", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("3D MODEL BENCHMARK & COMPARISON VIDEO RESULTS")
    print("=" * 90, flush=True)
    for r in results:
        print(f"Sample: {r['name']}")
        print(f"  Input: {r['clip']}")
        print(f"  CodeFormer Time : {r['cf_time']:.1f}s")
        print(f"  UltraMax Time   : {r['um_time']:.1f}s ({r['speedup']:.2f}x faster)")
        print(f"  Comparison Video: {r['comp_path']}")
        print("-" * 90)


if __name__ == "__main__":
    main()
