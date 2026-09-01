"""Benchmark matrix for RealSwap + RealityUX + UltraMax.

Evaluates pool sizing, thread scaling, temporal pre-pass stepping, and
enhancer cadence to find the maximum bottleneck-free throughput.
"""

import argparse
import os
import sys
import time
import psutil
import torch
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
if HERE not in sys.path:
    sys.path.insert(0, HERE)


import fixtures
def run_benchmark(video_path, source_name, threads, trt_pool, det_pool, temp_step, ultra_cadence):
    os.environ['ROOP_TRT_POOL'] = str(trt_pool)
    os.environ['ROOP_DETMASK_POOL'] = str(det_pool)
    os.environ['ROOP_DETECTOR_POOL'] = str(det_pool)
    os.environ['ROOP_TEMPORAL_STEP'] = str(temp_step)
    os.environ['ROOP_BATCH_SWAP'] = '1'
    os.environ['ROOP_BATCH_SWAP_XFRAME'] = '1'
    os.environ['ROOP_NVDEC'] = '1'
    os.environ['ROOP_ENCODER_PRESET'] = 'p5'

    # Clear cached pools
    import roop.session_pool as sp
    sp._pool_cache.clear()

    import angle_bench as ab
    from angle_video import ensure_ffmpeg
    from two_face_video import load_library_faceset, auto_capture_targets
    import sample_bench as sb
    import roop.globals as g

    ensure_ffmpeg()
    swapper_name = "realswap"
    enhancer_name = "UltraMax"
    mask_name = "mask_realityux"

    out_dir = os.path.join(APP, "output", "perf_bench")
    os.makedirs(out_dir, exist_ok=True)

    g = ab.init_pipeline('tensorrt', swapper_name, enhancer_name, mask_name, 0.0)
    g.execution_threads = threads
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

    options = ab.build_options(g, swapper_name, mask_name)
    options.stabilize_face = True
    options.stabilize_mask = True

    fs = load_library_faceset(source_name)
    targets, groups = auto_capture_targets(video_path, expect=1, log_prefix="[PerfBench]", strict=False)
    if targets is None:
        cap_idx, _, faces = sb.first_face_frame(video_path)
        targets, groups = [sb.select_primary_face(faces)], [0]

    # Run swap
    t0 = time.perf_counter()
    out_file, elapsed, face_log = sb.run_swap(video_path, [fs], targets, groups, options, out_dir)
    total_time = time.perf_counter() - t0

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    fps = total_frames / max(0.001, total_time)
    swap_fps = total_frames / max(0.001, elapsed)

    vram_used = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
    vram_res = torch.cuda.memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0

    print(f"\n[PerfBench Results] threads={threads}, trt_pool={trt_pool}, det_pool={det_pool}, step={temp_step}:")
    print(f"  Total Frames: {total_frames}")
    print(f"  Swap Stage Time: {elapsed:.2f}s ({swap_fps:.2f} fps)")
    print(f"  Total Run Time:  {total_time:.2f}s ({fps:.2f} fps)")
    print(f"  VRAM Allocated:  {vram_used:.2f} GB | Reserved: {vram_res:.2f} GB")

    return {
        'threads': threads,
        'trt_pool': trt_pool,
        'det_pool': det_pool,
        'temp_step': temp_step,
        'swap_fps': swap_fps,
        'total_fps': fps,
        'swap_time': elapsed,
        'total_time': total_time,
        'vram_gb': vram_res
    }


def main():
    video = fixtures.clip('final/5155179-hd_1920_1080_30fps.mp4')
    source = "harjot"

    configs = [
        {'threads': 12, 'trt_pool': 2, 'det_pool': 2, 'temp_step': 1, 'cadence': 4},
        {'threads': 16, 'trt_pool': 2, 'det_pool': 2, 'temp_step': 2, 'cadence': 4},
        {'threads': 16, 'trt_pool': 3, 'det_pool': 2, 'temp_step': 2, 'cadence': 4},
        {'threads': 16, 'trt_pool': 3, 'det_pool': 2, 'temp_step': 3, 'cadence': 4},
        {'threads': 20, 'trt_pool': 3, 'det_pool': 2, 'temp_step': 3, 'cadence': 4},
    ]

    results = []
    for cfg in configs:
        print("\n" + "=" * 70)
        print(f"Testing Config: threads={cfg['threads']}, trt_pool={cfg['trt_pool']}, step={cfg['temp_step']}")
        print("=" * 70)
        try:
            r = run_benchmark(video, source, cfg['threads'], cfg['trt_pool'], cfg['det_pool'], cfg['temp_step'], cfg['cadence'])
            results.append(r)
        except Exception as e:
            print(f"Config failed with error: {e}")

    print("\n" + "=" * 80)
    print(f"{'Threads':<8}{'TRT_Pool':<10}{'Det_Pool':<10}{'Step':<8}{'Swap FPS':<12}{'Total FPS':<12}{'VRAM (GB)':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['threads']:<8}{r['trt_pool']:<10}{r['det_pool']:<10}{r['temp_step']:<8}{r['swap_fps']:<12.2f}{r['total_fps']:<12.2f}{r['vram_gb']:<10.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
