"""Process and test all 4 duo video clips in G:/pinokio/roop-keep/duo/
using 2 source facesets (harjot and gargee) with RealSwap + RealityUX + UltraMax.
Extracts inspection preview frames and reports comprehensive processing data.
"""

import glob
import os
import sys
import time
import cv2
import torch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

os.environ['ROOP_TRT_POOL'] = '2'
os.environ['ROOP_DETMASK_POOL'] = '2'
os.environ['ROOP_DETECTOR_POOL'] = '2'
os.environ['ROOP_BATCH_SWAP'] = '1'
os.environ['ROOP_NVDEC'] = '1'
os.environ['ROOP_ENCODER_PRESET'] = 'p5'

import angle_bench as ab
from angle_video import ensure_ffmpeg
from two_face_video import load_library_faceset, auto_capture_targets
import sample_bench as sb
import roop.globals as g


def extract_inspection_frames(video_path, out_dir, prefix, num_samples=6):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = [int(i * (total - 1) / max(1, num_samples - 1)) for i in range(num_samples)]
    saved = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            fn = os.path.join(out_dir, f"{prefix}_frame_{idx:04d}.png")
            cv2.imwrite(fn, frame)
            saved.append(fn)
    cap.release()
    return saved


def main():
    ensure_ffmpeg()
    swapper_name = "realswap"
    enhancer_name = "UltraMax"
    mask_name = "mask_realityux"

    in_dir = r"G:/pinokio/roop-keep/duo"
    out_dir = os.path.join(APP, "output", "duo_swaps")
    inspect_dir = os.path.join(APP, "output", "duo_inspection_frames")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(inspect_dir, exist_ok=True)

    videos = sorted(glob.glob(os.path.join(in_dir, "*.mp4")))
    print(f"Loaded {len(videos)} duo videos from {in_dir}:")
    for i, v in enumerate(videos, 1):
        print(f"  {i}. {os.path.basename(v)}")

    # Initialize pipeline
    g = ab.init_pipeline('tensorrt', swapper_name, enhancer_name, mask_name, 0.0)
    g.execution_threads = 16
    g.video_encoder = 'hevc_nvenc'
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
    g.color_match_after_enhance = True
    g.detail_transfer_strength = 0.40

    options = ab.build_options(g, swapper_name, mask_name)
    options.stabilize_face = True
    options.stabilize_mask = True
    options.color_match_after_enhance = True
    options.detail_transfer_strength = 0.40

    # Ingest both source facesets
    print("\nLoading source facesets: 'harjot' and 'gargee'...")
    fs_harjot = load_library_faceset("harjot")
    fs_gargee = load_library_faceset("gargee")
    facesets = [fs_harjot, fs_gargee]
    print(f"  Person 0 Source -> 'harjot' ({len(fs_harjot.faces)} faces in faceset)")
    print(f"  Person 1 Source -> 'gargee' ({len(fs_gargee.faces)} faces in faceset)")

    records = []

    for i, vpath in enumerate(videos, 1):
        vname = os.path.basename(vpath)
        print("\n" + "=" * 80)
        print(f"[{i}/{len(videos)}] Processing Duo Video: {vname}")
        print("=" * 80)

        cap = cv2.VideoCapture(vpath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Capture both people present in the duo video
        print(f"Auto-capturing 2 target people in {vname}...")
        targets, groups = auto_capture_targets(vpath, expect=2, log_prefix="[DuoCapture]", strict=False)
        if targets is None or len(targets) < 2:
            print(f"Capture returned {len(targets) if targets else 0} faces; attempting multi-face scan fallback...")
            cap_idx, _, faces = sb.first_face_frame(vpath)
            if faces and len(faces) >= 2:
                targets = [faces[0], faces[1]]
                groups = [0, 1]
            elif faces:
                targets = [faces[0]]
                groups = [0]
            else:
                print(f"Skipping {vname}: no face could be captured.")
                continue

        print(f"Captured {len(targets)} target faces with group assignments {groups}.")

        t0 = time.perf_counter()
        out_file, elapsed, face_log = sb.run_swap(vpath, facesets, targets, groups, options, out_dir)
        total_time = time.perf_counter() - t0

        cap = cv2.VideoCapture(out_file)
        out_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        swap_fps = out_frames / max(0.001, elapsed)
        total_fps = out_frames / max(0.001, total_time)

        clean_name = "".join(c for c in vname[:15] if c.isalnum() or c in ('_', '-'))
        prefix = f"duo_clip{i}_{clean_name}"
        extracted = extract_inspection_frames(out_file, inspect_dir, prefix, num_samples=6)

        rec = {
            'index': i,
            'filename': vname,
            'resolution': f"{w}x{h}",
            'tested_frames': out_frames,
            'total_frames': total_frames,
            'swap_time_s': elapsed,
            'total_time_s': total_time,
            'swap_fps': swap_fps,
            'total_fps': total_fps,
            'out_file': out_file,
            'inspection_frames': extracted
        }
        records.append(rec)
        print(f"\nFINISHED DUO: {vname} ({out_frames} frames) in {total_time:.2f}s -> Swap FPS: {swap_fps:.1f}, Pipeline FPS: {total_fps:.1f}")

    print("\n" + "=" * 95)
    print("ALL 4 DUO CLIPS (2 FACESETS: HARJOT & GARGEE) TEST SUMMARY")
    print("=" * 95)
    print(f"{'#':<3}{'Resolution':<12}{'Frames':<10}{'Swap Time':<12}{'Total Time':<12}{'Swap FPS':<10}{'Total FPS':<10}{'Output File'}")
    print("-" * 95)
    for r in records:
        print(f"{r['index']:<3}{r['resolution']:<12}{r['tested_frames']:<10}{r['swap_time_s']:<12.2f}{r['total_time_s']:<12.2f}{r['swap_fps']:<10.1f}{r['total_fps']:<10.1f}{os.path.basename(r['out_file'])}")
    print("=" * 95)


if __name__ == "__main__":
    main()
