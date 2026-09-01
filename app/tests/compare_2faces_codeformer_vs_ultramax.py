"""Comparative benchmark between CodeFormer and UltraMax on 8509564-uhd_3840_2160_25fps.mp4
swapping BOTH faces:
- Left person (Target 0) -> Harjot
- Right person (Target 1) -> Ashna
"""

# NOTE 2026-08-23: this arm used to name the enhancer in a spelling
# `roop.core.get_processing_plugins` does not match, so it rendered with NO
# ENHANCER AT ALL and every 'x faster than CodeFormer' number this tool ever
# printed compared UltraMax against nothing. `tests/test_enhancer_names.py`
# now fails on any unmatched spelling.

import os
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
os.environ['ROOP_TRT_POOL'] = '2'
os.environ['ROOP_DETMASK_POOL'] = '2'
os.environ['ROOP_DETECTOR_POOL'] = '2'
os.environ['ROOP_BATCH_SWAP'] = '1'
os.environ['ROOP_NVDEC'] = '1'
os.environ['ROOP_ENCODER_PRESET'] = 'p5'

import angle_bench as ab
from angle_video import ensure_ffmpeg
from two_face_video import load_library_faceset
import sample_bench as sb
import roop.face_util as fu


def create_side_by_side_video(video_cf_path, video_um_path, out_path, fps_cf, fps_um, total_fps_cf, total_fps_um):
    cap_cf = cv2.VideoCapture(video_cf_path)
    cap_um = cv2.VideoCapture(video_um_path)

    fps = cap_cf.get(cv2.CAP_PROP_FPS) or 25.0
    total_f = min(int(cap_cf.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap_um.get(cv2.CAP_PROP_FRAME_COUNT)))

    out_w, out_h = 1920, 1080
    half_w = out_w // 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    font = cv2.FONT_HERSHEY_DUPLEX

    for f_idx in range(total_f):
        ret_cf, frame_cf = cap_cf.read()
        ret_um, frame_um = cap_um.read()
        if not ret_cf or not ret_um:
            break

        half_cf = cv2.resize(frame_cf, (half_w, out_h), interpolation=cv2.INTER_AREA)
        half_um = cv2.resize(frame_um, (half_w, out_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:, :half_w] = half_cf
        canvas[:, half_w:] = half_um

        cv2.line(canvas, (half_w, 0), (half_w, out_h), (60, 60, 60), 3)

        # Top Header Banner
        cv2.rectangle(canvas, (0, 0), (out_w, 65), (15, 15, 15), -1)
        cv2.putText(canvas, "8509564 Inverted 4K Dual Face Swap Benchmark",
                    (out_w // 2 - 340, 30), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Frame: {f_idx + 1}/{total_f} | Left: Harjot | Right: Ashna | Swapper: RealSwap",
                    (out_w // 2 - 370, 55), font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        # Left HUD (CodeFormer)
        cv2.rectangle(canvas, (20, 80), (450, 195), (20, 20, 20), -1)
        cv2.rectangle(canvas, (20, 80), (450, 195), (0, 165, 255), 2)
        cv2.putText(canvas, "CodeFormer (FP16)", (35, 110), font, 0.75, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Swap Speed:     {fps_cf:.1f} FPS", (35, 140), font, 0.65, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Pipeline Speed: {total_fps_cf:.1f} FPS", (35, 168), font, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Status: Discrete VQGAN Codebook", (35, 188), font, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

        # Right HUD (UltraMax)
        cv2.rectangle(canvas, (half_w + 20, 80), (half_w + 450, 195), (20, 20, 20), -1)
        cv2.rectangle(canvas, (half_w + 20, 80), (half_w + 450, 195), (0, 255, 128), 2)
        cv2.putText(canvas, "UltraMax (Photoreal HD)", (half_w + 35, 110), font, 0.75, (0, 255, 128), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Swap Speed:     {fps_um:.1f} FPS", (half_w + 35, 140), font, 0.65, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Pipeline Speed: {total_fps_um:.1f} FPS", (half_w + 35, 168), font, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Status: Multi-Scale CIELAB HD + CLAHE", (half_w + 35, 188), font, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

        writer.write(canvas)

    cap_cf.release()
    cap_um.release()
    writer.release()
    print(f"[Side-by-Side] Saved comparison video to: {out_path}")


def main():
    ensure_ffmpeg()
    swapper_name = "realswap"
    mask_name = "mask_realityux"

    video_path = fixtures.clip('inverted/8509564-uhd_3840_2160_25fps.mp4')
    out_dir = os.path.join(APP, "output", "benchmark_8509564_dual")
    inspect_dir = os.path.join(out_dir, "inspection_frames")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(inspect_dir, exist_ok=True)

    print("=" * 80)
    print(f"[Benchmark Dual] Target Video: {video_path}")
    print(f"[Benchmark Dual] Sources: Left=Harjot, Right=Ashna")
    print("=" * 80)

    # 1. Initialize Pipeline & Global Configuration
    g = ab.init_pipeline('tensorrt', swapper_name, 'UltraMax', mask_name, 0.0)
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

    # 2. Ingest Source Facesets (Harjot for Left Person, Ashna for Right Person)
    fs_harjot = load_library_faceset("harjot")
    fs_ashna = load_library_faceset("ashna")
    sources = [fs_harjot, fs_ashna]

    # 3. Create representative 4K segment (50 frames)
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)

    num_eval_frames = 50
    eval_video_path = os.path.join(out_dir, "_eval_segment_4k.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(eval_video_path, fourcc, fps_in, (w, h))
    for _ in range(num_eval_frames):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    cap.release()
    writer.release()
    print(f"[Benchmark Dual] Prepared 4K segment ({num_eval_frames} frames, {w}x{h} @ {fps_in} FPS).")

    # 4. Capture both target faces from frame 20 (Left person = group 0, Right person = group 1)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 20)
    ret, cap_frame = cap.read()
    cap.release()
    faces_frame20 = sorted(fu.get_all_faces(cap_frame) or [], key=lambda f: float(f.bbox[0]))
    if len(faces_frame20) < 2:
        raise SystemExit(f"Expected 2 faces in frame 20, found {len(faces_frame20)}")
    
    # Left face -> Group 0 (Harjot), Right face -> Group 1 (Ashna)
    targets = [faces_frame20[0], faces_frame20[1]]
    groups = [0, 1]
    print(f"[Capture] Target 0 (Left Person, center_x={(targets[0].bbox[0]+targets[0].bbox[2])//2}) -> Harjot")
    print(f"[Capture] Target 1 (Right Person, center_x={(targets[1].bbox[0]+targets[1].bbox[2])//2}) -> Ashna")

    # ── ARM 1: CodeFormer (FP16) ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(">>> RUNNING ARM 1: CODEFORMER (FP16) - DUAL FACE SWAP")
    print("=" * 80)
    g1 = ab.init_pipeline('tensorrt', swapper_name, 'Codeformer (fp16)', mask_name, 0.0)
    g1.execution_threads = 16
    g1.video_encoder = 'hevc_nvenc'
    g1.video_quality = 14
    g1.face_swap_mode = "selected"
    g1.track_identities = True
    g1.CFG.track_identities = True
    g1.temporal_detection = True
    g1.CFG.temporal_detection = True
    g1.stabilize_face = True
    g1.stabilize_mask = True
    g1.upscale_after_swap = False
    g1.CFG.upscale_after_swap = False
    g1.color_match_after_enhance = True
    g1.detail_transfer_strength = 0.40

    opts_cf = ab.build_options(g1, swapper_name, mask_name)
    opts_cf.stabilize_face = True
    opts_cf.stabilize_mask = True
    opts_cf.color_match_after_enhance = True
    opts_cf.detail_transfer_strength = 0.40

    t0_cf = time.perf_counter()
    out_cf, swap_time_cf, _ = sb.run_swap(eval_video_path, sources, targets, groups, opts_cf, out_dir)
    total_time_cf = time.perf_counter() - t0_cf

    cap_res = cv2.VideoCapture(out_cf)
    frames_cf = int(cap_res.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_res.release()
    swap_fps_cf = frames_cf / max(0.001, swap_time_cf)
    total_fps_cf = frames_cf / max(0.001, total_time_cf)
    print(f"[ARM 1 CodeFormer Dual] {frames_cf} frames | Swap: {swap_time_cf:.2f}s ({swap_fps_cf:.2f} FPS) | Total: {total_time_cf:.2f}s ({total_fps_cf:.2f} FPS)")

    # ── ARM 2: UltraMax (Photoreal HD Fusion) ─────────────────────────────────
    print("\n" + "=" * 80)
    print(">>> RUNNING ARM 2: ULTRAMAX (PHOTOREAL HD) - DUAL FACE SWAP")
    print("=" * 80)
    g2 = ab.init_pipeline('tensorrt', swapper_name, 'UltraMax', mask_name, 0.0)
    g2.execution_threads = 16
    g2.video_encoder = 'hevc_nvenc'
    g2.video_quality = 14
    g2.face_swap_mode = "selected"
    g2.track_identities = True
    g2.CFG.track_identities = True
    g2.temporal_detection = True
    g2.CFG.temporal_detection = True
    g2.stabilize_face = True
    g2.stabilize_mask = True
    g2.upscale_after_swap = False
    g2.CFG.upscale_after_swap = False
    g2.color_match_after_enhance = True
    g2.detail_transfer_strength = 0.40

    opts_um = ab.build_options(g2, swapper_name, mask_name)
    opts_um.stabilize_face = True
    opts_um.stabilize_mask = True
    opts_um.color_match_after_enhance = True
    opts_um.detail_transfer_strength = 0.40

    t0_um = time.perf_counter()
    out_um, swap_time_um, _ = sb.run_swap(eval_video_path, sources, targets, groups, opts_um, out_dir)
    total_time_um = time.perf_counter() - t0_um

    cap_res = cv2.VideoCapture(out_um)
    frames_um = int(cap_res.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_res.release()
    swap_fps_um = frames_um / max(0.001, swap_time_um)
    total_fps_um = frames_um / max(0.001, total_time_um)
    print(f"[ARM 2 UltraMax Dual] {frames_um} frames | Swap: {swap_time_um:.2f}s ({swap_fps_um:.2f} FPS) | Total: {total_time_um:.2f}s ({total_fps_um:.2f} FPS)")

    # ── 5. Generate Side-by-Side Comparison Video ────────────────────────────
    sbs_path = os.path.join(out_dir, "compare_8509564_dual_codeformer_vs_ultramax.mp4")
    create_side_by_side_video(out_cf, out_um, sbs_path, swap_fps_cf, swap_fps_um, total_fps_cf, total_fps_um)

    # ── 6. Extract High-Res Inspection PNGs & Face Crops ─────────────────────
    cap_cf = cv2.VideoCapture(out_cf)
    cap_um = cv2.VideoCapture(out_um)
    for sample_idx in [0, 15, 30, 48]:
        cap_cf.set(cv2.CAP_PROP_POS_FRAMES, sample_idx)
        cap_um.set(cv2.CAP_PROP_POS_FRAMES, sample_idx)
        r1, f_cf = cap_cf.read()
        r2, f_um = cap_um.read()
        if r1 and r2:
            cv2.imwrite(os.path.join(inspect_dir, f"frame_{sample_idx:04d}_codeformer.png"), f_cf)
            cv2.imwrite(os.path.join(inspect_dir, f"frame_{sample_idx:04d}_ultramax.png"), f_um)
            
            # Extract Left and Right face crops
            ch, cw = f_cf.shape[:2]
            left_cf = f_cf[:, :cw//2]
            left_um = f_um[:, :cw//2]
            right_cf = f_cf[:, cw//2:]
            right_um = f_um[:, cw//2:]
            
            cv2.imwrite(os.path.join(inspect_dir, f"crop_left_harjot_frame_{sample_idx:04d}_cf.png"), left_cf)
            cv2.imwrite(os.path.join(inspect_dir, f"crop_left_harjot_frame_{sample_idx:04d}_um.png"), left_um)
            cv2.imwrite(os.path.join(inspect_dir, f"crop_right_ashna_frame_{sample_idx:04d}_cf.png"), right_cf)
            cv2.imwrite(os.path.join(inspect_dir, f"crop_right_ashna_frame_{sample_idx:04d}_um.png"), right_um)

    cap_cf.release()
    cap_um.release()

    print("\n" + "=" * 90)
    print("DUAL FACE SWAP BENCHMARK COMPLETE")
    print("=" * 90)
    print(f"Target Video:     8509564-uhd_3840_2160_25fps.mp4 (4K UHD Inverted)")
    print(f"Swapped Faces:    Left=Harjot, Right=Ashna (Both faces swapped)")
    print(f"CodeFormer (fp16): {swap_fps_cf:.2f} Swap FPS | {total_fps_cf:.2f} Pipeline FPS (Time: {swap_time_cf:.2f}s)")
    print(f"UltraMax (HD):    {swap_fps_um:.2f} Swap FPS | {total_fps_um:.2f} Pipeline FPS (Time: {swap_time_um:.2f}s)")
    print(f"Comparison Video: {sbs_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
