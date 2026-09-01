"""Dedicated single clip tester for UltraMax + RealityUX + RealSwap.

Runs pure swapped output without side-by-side or AI upscaling.
"""

import argparse
import os
import shutil
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
def _apply_perf_env():
    try:
        import yaml
        with open(os.path.join(APP, 'config.yaml'), 'r') as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return

    def _set(var, val):
        if var in os.environ:
            return
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
        if var in os.environ:
            continue
        v = str(cfg.get(key, 'auto')).strip().lower()
        if v == 'on' or (v == 'auto' and var == 'ROOP_BATCH_SWAP'):
            os.environ[var] = '1'
            if var == 'ROOP_BATCH_SWAP' and 'ROOP_BATCH_SWAP_XFRAME' not in os.environ:
                os.environ['ROOP_BATCH_SWAP_XFRAME'] = '1'
        elif v == 'off':
            os.environ[var] = '0'
            if var == 'ROOP_BATCH_SWAP' and 'ROOP_BATCH_SWAP_XFRAME' not in os.environ:
                os.environ['ROOP_BATCH_SWAP_XFRAME'] = '0'


_apply_perf_env()

import angle_bench as ab
from angle_video import ensure_ffmpeg
from two_face_video import load_library_faceset, auto_capture_targets
import sample_bench as sb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=fixtures.clip('final/5155179-hd_1920_1080_30fps.mp4'))
    ap.add_argument("--source", default="harjot")
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    ensure_ffmpeg()
    swapper_name = "realswap"
    enhancer_name = "UltraMax"
    mask_name = "mask_realityux"

    out_dir = os.path.join(APP, "output")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print(f"[Single Test] Processing clip: {args.video}")
    print(f"[Single Test] Source: {args.source} | Swapper: {swapper_name} | Enhancer: {enhancer_name} | Mask: {mask_name}")
    print("=" * 80)

    g = ab.init_pipeline('tensorrt', swapper_name, enhancer_name, mask_name, 0.0)
    g.execution_threads = args.threads
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

    fs = load_library_faceset(args.source)

    targets, groups = auto_capture_targets(args.video, expect=1, log_prefix="[Single Test]", strict=False)
    if targets is None:
        cap_idx, _, faces = sb.first_face_frame(args.video)
        targets, groups = [sb.select_primary_face(faces)], [0]
        print(f"[Single Test] fell back to first-face capture, frame {cap_idx}", flush=True)

    out_file, elapsed, face_log = sb.run_swap(args.video, [fs], targets, groups, options, out_dir)

    stem = os.path.splitext(os.path.basename(args.video))[0]
    final_output = os.path.join(out_dir, f"{stem}__realswap_realityux_ultramax.mp4")

    if out_file and os.path.exists(out_file):
        if os.path.abspath(out_file) != os.path.abspath(final_output):
            if os.path.exists(final_output):
                os.remove(final_output)
            shutil.move(out_file, final_output)
        print(f"\n[Single Test] [SUCCESS] Completed in {elapsed:.1f}s")
        print(f"[Single Test] Final video saved: {final_output}")

        # Extract sample frames for inspection
        frames_dir = os.path.join(out_dir, f"{stem}__inspection_frames")
        os.makedirs(frames_dir, exist_ok=True)
        cap = cv2.VideoCapture(final_output)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        check_indices = [0, total_frames // 4, total_frames // 2, (total_frames * 3) // 4, total_frames - 1]
        for idx in check_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if ok:
                f_path = os.path.join(frames_dir, f"frame_{idx:04d}.png")
                cv2.imwrite(f_path, fr)
                print(f"[Single Test] Saved frame {idx} to: {f_path}")
        cap.release()
    else:
        print("[Single Test] [ERROR] No output produced!")


if __name__ == "__main__":
    main()
