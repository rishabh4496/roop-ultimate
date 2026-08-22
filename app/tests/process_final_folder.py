"""Pure Video Swapper for all clips in roop-keep/final.

Runs UltraMax + RealityUX + RealSwap ONLY (no CodeFormer baseline, no side-by-side).
Outputs direct full-quality swapped videos to app/output/final_swaps/.
"""

import argparse
import glob
import os
import shutil
import sys
import time

import cv2

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
OUT_DIR = os.path.join(APP, "output", "final_swaps")
RAW_UM_DIR = os.path.join(APP, "output", "final_comparisons", "um_raw")


def find_existing_complete(dir_path, clip_stem, expected_frames):
    if not os.path.exists(dir_path):
        return None
    for f in os.listdir(dir_path):
        if f.startswith(clip_stem) and f.lower().endswith('.mp4') and not f.startswith('.') and '__temp' not in f:
            full_path = os.path.join(dir_path, f)
            try:
                cap = cv2.VideoCapture(full_path)
                cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if cnt >= expected_frames - 2 and cnt > 0:
                    return full_path
            except Exception:
                pass
    return None


def run_ultramax_swap(clip, source_name="harjot", threads=12):
    ensure_ffmpeg()
    swapper_name = "realswap"
    enhancer_name = "UltraMax"
    mask_name = "mask_realityux"

    g = ab.init_pipeline('tensorrt', swapper_name, enhancer_name, mask_name, 0.0)
    g.execution_threads = threads
    g.video_encoder = getattr(g.CFG, 'output_video_codec', 'hevc_nvenc') or 'hevc_nvenc'
    g.video_quality = 14
    g.face_swap_mode = "selected"
    g.track_identities = True
    g.CFG.track_identities = True
    g.temporal_detection = True
    g.CFG.temporal_detection = True
    g.stabilize_face = True
    g.stabilize_mask = True
    g.upscale_after_swap = False
    g.CFG.upscale_after_swap = False
    options = ab.build_options(g, swapper_name, mask_name)
    options.stabilize_face = True
    options.stabilize_mask = True

    fs = load_library_faceset(source_name)

    targets, groups = auto_capture_targets(clip, expect=1, log_prefix="[Swap]", strict=False)
    if targets is None:
        try:
            cap_idx, _, faces = sb.first_face_frame(clip)
            targets, groups = [sb.select_primary_face(faces)], [0]
            print(f"[Swap] fell back to first-face capture, frame {cap_idx}", flush=True)
        except Exception as e:
            print(f"[Swap] Warning: no face found in {clip}: {e}", flush=True)
            return None, 0.0

    out, elapsed, _ = sb.run_swap(clip, [fs], targets, groups, options, OUT_DIR)
    return out, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="harjot")
    ap.add_argument("--only", default="")
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    all_clips = sorted(glob.glob(os.path.join(FINAL_DIR, "*.*")))
    clips = [c for c in all_clips if c.lower().endswith(('.mp4', '.webm', '.avi', '.mov', '.mkv'))]

    print(f"[Final Swaps] Found {len(clips)} clips in {FINAL_DIR}", flush=True)

    for i, clip in enumerate(clips, 1):
        stem = os.path.splitext(os.path.basename(clip))[0]
        safe_name = f"sample_{i:02d}_{stem}__ultramax.mp4"
        final_dest = os.path.join(OUT_DIR, safe_name)

        if args.only and args.only not in stem and f"{i:02d}" != args.only and str(i) != args.only:
            continue

        print(f"\n" + "=" * 80, flush=True)
        print(f"[Final Swaps] [{i}/{len(clips)}] Processing: {os.path.basename(clip)}", flush=True)
        print("=" * 80, flush=True)

        clip_cap = cv2.VideoCapture(clip)
        expected_frames = int(clip_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        clip_cap.release()

        # Check if already completed in OUT_DIR
        if os.path.exists(final_dest):
            chk_cap = cv2.VideoCapture(final_dest)
            chk_frames = int(chk_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            chk_cap.release()
            if chk_frames >= expected_frames - 2 and chk_frames > 0:
                print(f"[Final Swaps] Already complete in {final_dest} ({chk_frames} frames). Skipping.", flush=True)
                continue

        # Check if completed in um_raw
        existing_raw = find_existing_complete(RAW_UM_DIR, stem, expected_frames)
        if existing_raw:
            print(f"[Final Swaps] Copying existing verified UltraMax output from {existing_raw} -> {final_dest}", flush=True)
            shutil.copy2(existing_raw, final_dest)
            continue

        # Run fresh swap
        out, elapsed = run_ultramax_swap(clip, source_name=args.source, threads=args.threads)
        if out and os.path.exists(out):
            if os.path.abspath(out) != os.path.abspath(final_dest):
                shutil.move(out, final_dest)
            print(f"[Final Swaps] [DONE] Saved to: {final_dest} in {elapsed:.1f}s", flush=True)
        else:
            print(f"[Final Swaps] [ERROR] Failed to produce output for {stem}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("ALL REQUESTED FINAL SWAPS COMPLETE")
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
