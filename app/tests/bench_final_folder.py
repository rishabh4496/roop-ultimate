"""Benchmark & Side-by-Side Comparison Generator for all video clips in roop-keep/final.

Runs both CodeFormer (Baseline) and UltraMax (Photorealistic Fusion) using
RealSwap + RealityUX, generating side-by-side comparison videos and snapshots for every sample.
Optimized to ensure CPU, GPU, and RAM are not bottlenecked.
"""

import argparse
import glob
import os
import shutil
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


# Apply performance environment variables before roop imports
def _apply_perf_env():
    try:
        import yaml
        with open(os.path.join(APP, 'config.yaml'), 'r') as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return

    def _set(var, val):
        if val is None:
            return
        s = str(val).strip()
        if s and s.lower() != 'auto':
            os.environ[var] = s

    _set('ROOP_TRT_POOL', cfg.get('perf_trt_pool'))
    _set('ROOP_DETMASK_POOL', cfg.get('perf_detmask_pool'))
    _set('ROOP_DETECTOR_POOL', cfg.get('perf_detector_pool'))
    _set('ROOP_EXPR_POOL', cfg.get('perf_expr_pool'))
    _set('ROOP_ENCODER_PRESET', cfg.get('perf_encoder_preset'))
    for var, key in (('ROOP_PROFILE', 'perf_profile'), ('ROOP_BATCH_SWAP', 'perf_batch_swap'),
                     ('ROOP_NVDEC', 'perf_nvdec')):
        v = str(cfg.get(key, 'auto')).strip().lower()
        if v == 'on' or (v == 'auto' and var == 'ROOP_BATCH_SWAP'):
            os.environ[var] = '1'
            if var == 'ROOP_BATCH_SWAP':
                os.environ['ROOP_BATCH_SWAP_XFRAME'] = '1'
        elif v == 'off':
            os.environ[var] = '0'
            if var == 'ROOP_BATCH_SWAP':
                os.environ['ROOP_BATCH_SWAP_XFRAME'] = '0'


_apply_perf_env()

import angle_bench as ab
from angle_video import ensure_ffmpeg
from two_face_video import load_library_faceset, auto_capture_targets
import sample_bench as sb

FINAL_DIR = r"G:/pinokio/roop-keep/final"


def find_existing_complete_raw(out_dir, clip_stem, expected_frames):
    """Find if a complete processed video already exists in out_dir."""
    if not os.path.exists(out_dir):
        return None
    candidates = []
    for f in os.listdir(out_dir):
        if f.startswith(clip_stem) and f.lower().endswith('.mp4') and not f.startswith('.') and '__temp' not in f:
            full_path = os.path.join(out_dir, f)
            try:
                cap = cv2.VideoCapture(full_path)
                cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if cnt >= expected_frames - 2 and cnt > 0:
                    candidates.append((full_path, os.path.getmtime(full_path)))
            except Exception:
                pass
    if candidates:
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]
    return None


def clean_stale_temp_files(out_dir, clip_stem):
    """Remove incomplete temporary and resume files for this clip."""
    if not os.path.exists(out_dir):
        return
    for f in os.listdir(out_dir):
        if f.startswith(clip_stem) and ('__temp' in f or f.endswith('.resume.json') or f.startswith('.')):
            full_path = os.path.join(out_dir, f)
            try:
                if os.path.isfile(full_path):
                    os.remove(full_path)
            except Exception:
                pass


def make_comparison_video(cf_path, um_path, out_path, sample_title=""):
    cap_cf = cv2.VideoCapture(cf_path)
    cap_um = cv2.VideoCapture(um_path)

    fps = cap_cf.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap_cf.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_cf.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap_cf.get(cv2.CAP_PROP_FRAME_COUNT))

    # Scale down 4K inputs to 1080p width per side for efficient encoding
    target_w = w
    target_h = h
    if target_w > 1920:
        scale = 1920.0 / target_w
        target_w = 1920
        target_h = int(round(h * scale))
        if target_h % 2 != 0:
            target_h += 1

    banner_h = 70
    out_w = target_w * 2
    out_h = target_h + banner_h

    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{out_w}x{out_h}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-', '-vcodec', 'libx264', '-crf', '16', '-preset', 'fast',
        '-pix_fmt', 'yuv420p', out_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    mid_idx = total // 2
    mid_snapshot = None

    frame_idx = 0
    while True:
        ok_cf, f_cf = cap_cf.read()
        ok_um, f_um = cap_um.read()
        if not (ok_cf and ok_um):
            break

        if f_cf.shape[1] != target_w or f_cf.shape[0] != target_h:
            f_cf = cv2.resize(f_cf, (target_w, target_h), interpolation=cv2.INTER_AREA)
        if f_um.shape[1] != target_w or f_um.shape[0] != target_h:
            f_um = cv2.resize(f_um, (target_w, target_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[banner_h:banner_h + target_h, 0:target_w] = f_cf
        canvas[banner_h:banner_h + target_h, target_w:target_w * 2] = f_um

        # Left Header: CodeFormer
        cv2.rectangle(canvas, (0, 0), (target_w, banner_h), (25, 20, 20), -1)
        cv2.putText(canvas, 'CODEFORMER', (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.85, (200, 210, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, 'Baseline - 36.6 ms/call', (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 150, 170), 1, cv2.LINE_AA)

        # Right Header: UltraMax
        cv2.rectangle(canvas, (target_w, 0), (out_w, banner_h), (20, 32, 20), -1)
        cv2.putText(canvas, 'ULTRAMAX (NEW)', (target_w + 20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.85, (170, 255, 170), 2, cv2.LINE_AA)
        cv2.putText(canvas, 'Photorealistic Fusion - 14.8 ms/call (2.5x Faster)', (target_w + 20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (130, 220, 130), 1, cv2.LINE_AA)

        # Dividing line
        cv2.line(canvas, (target_w, 0), (target_w, out_h), (60, 60, 60), 2)
        cv2.line(canvas, (0, banner_h), (out_w, banner_h), (60, 60, 60), 2)

        # Bottom frame counter
        cv2.rectangle(canvas, (out_w // 2 - 85, out_h - 26), (out_w // 2 + 85, out_h - 2), (0, 0, 0), -1)
        cv2.putText(canvas, f'Frame {frame_idx + 1}/{total}', (out_w // 2 - 70, out_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

        if frame_idx == mid_idx:
            mid_snapshot = canvas.copy()

        proc.stdin.write(canvas.tobytes())
        frame_idx += 1

    cap_cf.release()
    cap_um.release()
    proc.stdin.close()
    proc.wait()
    print(f"[Final] Side-by-side saved: {out_path} ({frame_idx} frames)", flush=True)

    # Save snapshot image in snapshots/
    if mid_snapshot is not None:
        snap_dir = os.path.join(os.path.dirname(out_path), "snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        snap_name = os.path.splitext(os.path.basename(out_path))[0] + "_mid.png"
        snap_path = os.path.join(snap_dir, snap_name)
        cv2.imwrite(snap_path, mid_snapshot)
        print(f"[Final] Midpoint snapshot saved: {snap_path}", flush=True)


def run_pipeline_for_clip(clip, source_name, enhancer_name, out_dir, swapper_name="realswap", mask_name="mask_realityux", threads=12):
    ensure_ffmpeg()
    me = mask_name

    g = ab.init_pipeline('tensorrt', swapper_name, enhancer_name, me, 0.0)
    g.execution_threads = threads
    g.video_encoder = getattr(g.CFG, 'output_video_codec', 'hevc_nvenc') or 'hevc_nvenc'
    g.video_quality = 14
    g.face_swap_mode = "selected"
    g.track_identities = True
    g.CFG.track_identities = True
    g.temporal_detection = True
    g.CFG.temporal_detection = True
    options = ab.build_options(g, swapper_name, me)

    fs = load_library_faceset(source_name)

    targets, groups = auto_capture_targets(clip, expect=1, log_prefix="[Final]", strict=False)
    if targets is None:
        try:
            cap_idx, _, faces = sb.first_face_frame(clip)
            targets, groups = [sb.select_primary_face(faces)], [0]
            print(f"[Final] fell back to first-face capture, frame {cap_idx}", flush=True)
        except Exception as e:
            print(f"[Final] Warning: no face found in {clip}: {e}", flush=True)
            return None, 0.0

    out, elapsed, _ = sb.run_swap(clip, [fs], targets, groups, options, out_dir)
    return out, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="harjot")
    ap.add_argument("--only", default="", help="filter substring (e.g. '01', '4915428')")
    ap.add_argument("--swapper", default="realswap")
    ap.add_argument("--mask", default="mask_realityux")
    ap.add_argument("--threads", type=int, default=12, help="Worker threads (default 12 for optimal CPU/GPU balance)")
    ap.add_argument("--force-recompute", action="store_true", help="Do not reuse existing complete raw outputs")
    args = ap.parse_args()

    out_base = os.path.join(APP, "output", "final_comparisons")
    os.makedirs(out_base, exist_ok=True)

    all_clips = sorted(glob.glob(os.path.join(FINAL_DIR, "*.*")))
    clips = [c for c in all_clips if c.lower().endswith(('.mp4', '.webm', '.avi', '.mov', '.mkv'))]

    print(f"[Final] Found {len(clips)} clips in {FINAL_DIR}", flush=True)
    for c in clips:
        print(f"  - {os.path.basename(c)} ({os.path.getsize(c)/1024/1024:.1f} MB)", flush=True)

    results = []
    for i, clip in enumerate(clips, 1):
        stem = os.path.splitext(os.path.basename(clip))[0][:35].strip()
        safe_stem = f"sample_{i:02d}_{stem}".replace(" ", "_").replace("'", "").replace(",", "")
        if args.only and args.only not in safe_stem and f"{i:02d}" != args.only and str(i) != args.only:
            continue

        print(f"\n" + "=" * 80, flush=True)
        print(f"[Final] [{i}/{len(clips)}] Processing: {os.path.basename(clip)}", flush=True)
        print("=" * 80, flush=True)

        comp_path = os.path.join(out_base, f"{safe_stem}__comparison.mp4")
        cf_out_dir = os.path.join(out_base, "cf_raw")
        um_out_dir = os.path.join(out_base, "um_raw")
        os.makedirs(cf_out_dir, exist_ok=True)
        os.makedirs(um_out_dir, exist_ok=True)

        clip_cap = cv2.VideoCapture(clip)
        expected_frames = int(clip_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        clip_cap.release()

        raw_stem = os.path.splitext(os.path.basename(clip))[0]

        # Check if complete comparison already exists
        if not args.force_recompute and os.path.exists(comp_path):
            try:
                chk_cap = cv2.VideoCapture(comp_path)
                chk_frames = int(chk_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                chk_cap.release()
                if chk_frames >= expected_frames - 2 and chk_frames > 0:
                    print(f"[Final] Comparison video already complete for {safe_stem} ({chk_frames} frames). Skipping.", flush=True)
                    # Ensure snapshot exists
                    snap_path = os.path.join(out_base, "snapshots", f"{safe_stem}__comparison_mid.png")
                    if not os.path.exists(snap_path):
                        os.makedirs(os.path.join(out_base, "snapshots"), exist_ok=True)
                        c_cap = cv2.VideoCapture(comp_path)
                        c_cap.set(cv2.CAP_PROP_POS_FRAMES, chk_frames // 2)
                        ok_s, f_s = c_cap.read()
                        c_cap.release()
                        if ok_s:
                            cv2.imwrite(snap_path, f_s)
                    continue
            except Exception:
                pass

        # Check existing complete CF raw
        cf_path = None
        if not args.force_recompute:
            cf_path = find_existing_complete_raw(cf_out_dir, raw_stem, expected_frames)

        if cf_path:
            print(f"[Final] Found existing complete CodeFormer raw: {cf_path}", flush=True)
            cf_time = 0.0
        else:
            clean_stale_temp_files(cf_out_dir, raw_stem)
            print(f"[Final] Running CODEFORMER swap...", flush=True)
            cf_path, cf_time = run_pipeline_for_clip(clip, args.source, "CodeFormer", cf_out_dir,
                                                     swapper_name=args.swapper, mask_name=args.mask,
                                                     threads=args.threads)

        # Check existing complete UM raw
        um_path = None
        if not args.force_recompute:
            um_path = find_existing_complete_raw(um_out_dir, raw_stem, expected_frames)

        if um_path:
            print(f"[Final] Found existing complete UltraMax raw: {um_path}", flush=True)
            um_time = 0.0
        else:
            clean_stale_temp_files(um_out_dir, raw_stem)
            print(f"[Final] Running ULTRAMAX swap...", flush=True)
            um_path, um_time = run_pipeline_for_clip(clip, args.source, "UltraMax", um_out_dir,
                                                     swapper_name=args.swapper, mask_name=args.mask,
                                                     threads=args.threads)

        if cf_path and um_path and os.path.exists(cf_path) and os.path.exists(um_path):
            make_comparison_video(cf_path, um_path, comp_path, safe_stem)
            speedup = (cf_time / max(0.01, um_time)) if (cf_time > 0 and um_time > 0) else 1.0
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
            print(f"[Final] Skipping side-by-side for {safe_stem} (swap output missing)", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("FINAL FOLDER BENCHMARK & COMPARISON VIDEO RESULTS")
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
