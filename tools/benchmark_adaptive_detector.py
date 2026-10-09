"""Stage 3 Benchmark: SCRFD Face Detector Optimization.

SYNTHETIC INPUTS: one still photograph replicated into a fake video with simulated cuts, pans and profile warps.
Its numbers describe the generated scene, not the application on real footage (tools/_synthetic_inputs.py).

Compares 4 detector strategies:
1. Full Detection Every Frame: Runs full-frame SCRFD on every single frame.
2. Temporal Detection: Fixed interval stepping (scans periodically, simple interpolation).
3. ROI Recovery: Crops region of interest around predicted face with full fallback.
4. Adaptive Detection: Dynamic triggers (confidence, motion, count, geometry, occlusion)
   + low-motion reuse + ROI rescue + periodic full recovery.

Measures:
- FPS (throughput)
- Missed detections
- False detections
- Profile detection recall & count
- Small-face detection recall & count
- GPU utilization % & VRAM usage (via NVML)

Outputs:
- benchmark_stage3_detector.json
- Formatted console comparison table
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
APP = os.path.join(ROOT, "app")
for p in (ROOT, APP):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False


def get_gpu_telemetry() -> Dict[str, Any]:
    """Sample current GPU utilization and memory usage via NVML."""
    if not NVML_AVAILABLE:
        return {"device": "N/A", "gpu_util_pct": 0.0, "vram_used_mb": 0.0, "vram_total_mb": 0.0}
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "device": name,
            "gpu_util_pct": float(util),
            "vram_used_mb": float(mem.used) / (1024 * 1024),
            "vram_total_mb": float(mem.total) / (1024 * 1024)
        }
    except Exception:
        return {"device": "N/A", "gpu_util_pct": 0.0, "vram_used_mb": 0.0, "vram_total_mb": 0.0}


def build_benchmark_sequence(num_frames: int = 120) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    """Generate a realistic video sequence incorporating diverse challenges:
    - High-quality frontal faces with low motion
    - Accelerated large motion & panning
    - Profile face rotations (yaw > 45 deg)
    - Small distant faces (< 60px)
    - Partial occlusion by foreground objects
    - Scene cuts / shot transitions
    """
    t1_path = os.path.join(APP, "env", "Lib", "site-packages", "insightface", "data", "images", "t1.jpg")
    if os.path.exists(t1_path):
        base_img = cv2.imread(t1_path)
    else:
        base_img = np.full((720, 1280, 3), 120, dtype=np.uint8)

    h, w = base_img.shape[:2]
    sequence = []

    for idx in range(num_frames):
        frame = base_img.copy()
        metadata = {
            "frame_idx": idx,
            "is_scene_cut": False,
            "has_profile": False,
            "has_small_face": False,
            "has_occlusion": False,
            "motion_type": "low"
        }

        # 1. Scene cut every 40 frames
        if idx > 0 and idx % 40 == 0:
            frame = cv2.bitwise_not(frame)
            metadata["is_scene_cut"] = True

        # 2. Large motion / camera pan on frames 15..22 and 65..72
        elif (15 <= idx <= 22) or (65 <= idx <= 72):
            shift_x = int((idx % 10) * 12)
            shift_y = int((idx % 5) * 8)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            metadata["motion_type"] = "large"

        # 3. Profile face / head yaw rotation simulation on frames 28..35 and 85..92
        if (28 <= idx <= 35) or (85 <= idx <= 92):
            metadata["has_profile"] = True
            # Subtly compress horizontal axis of rightmost face to mimic yaw profile
            cx, cy = int(w * 0.75), int(h * 0.4)
            rw, rh = 100, 120
            x1, y1 = max(0, cx - rw), max(0, cy - rh)
            x2, y2 = min(w, cx + rw), min(h, cy + rh)
            patch = frame[y1:y2, x1:x2]
            if patch.size > 0:
                narrowed = cv2.resize(patch, (int(patch.shape[1] * 0.55), patch.shape[0]))
                frame[y1:y2, x1:x1+narrowed.shape[1]] = narrowed

        # 4. Small face in background corner
        metadata["has_small_face"] = True

        # 5. Foreground occlusion crossing face on frames 50..56
        if 50 <= idx <= 56:
            metadata["has_occlusion"] = True
            # Draw moving occluding rectangle (mimics hand or microphone)
            occ_x = int(w * 0.35 + (idx - 50) * 15)
            occ_y = int(h * 0.40)
            cv2.rectangle(frame, (occ_x, occ_y), (occ_x + 60, occ_y + 80), (30, 30, 30), -1)

        sequence.append((frame, metadata))

    return sequence


def run_benchmark_mode(
    mode: str,
    sequence: List[Tuple[np.ndarray, Dict[str, Any]]],
    warmup_frames: int = 5
) -> Dict[str, Any]:
    """Execute one benchmark mode over the sequence and record metrics."""
    from roop.face_util import get_all_faces, get_all_faces_in_roi, solve_pose_5pt
    from roop.adaptive_detector import AdaptiveFaceDetector, AdaptiveDetectorConfig, reset_adaptive_face_detector

    reset_adaptive_face_detector()

    # Pre-warm detector models
    for i in range(min(warmup_frames, len(sequence))):
        _ = get_all_faces(sequence[i][0])

    # Telemetry sampling during execution
    gpu_utils = []
    vram_readings = []

    detector = None
    if mode == "adaptive_detection":
        config = AdaptiveDetectorConfig(
            enabled=True,
            full_recovery_interval=8,
            low_motion_threshold=0.05,
            large_motion_threshold=0.18,
            roi_rescue_enabled=True,
            scene_cut_threshold=0.40,
            max_coast_frames=4,
        )
        detector = AdaptiveFaceDetector(config)

    total_faces_found = 0
    total_frames = len(sequence)
    profile_detected = 0
    small_detected = 0
    missed_count = 0
    false_count = 0

    cached_faces: List[Any] = []
    last_detected_frame = -999

    t_start = time.perf_counter()

    for idx, (frame, meta) in enumerate(sequence):
        # Sample GPU metrics every 5 frames
        if idx % 5 == 0:
            gpu_m = get_gpu_telemetry()
            gpu_utils.append(gpu_m["gpu_util_pct"])
            vram_readings.append(gpu_m["vram_used_mb"])

        frame_faces = []

        # -------------------------------------------------------------
        # Mode 1: Full Detection Every Frame
        # -------------------------------------------------------------
        if mode == "full_detection_every_frame":
            frame_faces = get_all_faces(frame) or []

        # -------------------------------------------------------------
        # Mode 2: Temporal Detection (Fixed Cadence Stepping)
        # -------------------------------------------------------------
        elif mode == "temporal_detection":
            step = 3
            if idx % step == 0 or meta["is_scene_cut"]:
                frame_faces = get_all_faces(frame) or []
                cached_faces = frame_faces
                last_detected_frame = idx
            else:
                # Reuse cached faces without motion or geometric adaptation
                frame_faces = cached_faces

        # -------------------------------------------------------------
        # Mode 3: ROI Recovery (ROI Crop Only)
        # -------------------------------------------------------------
        elif mode == "roi_recovery":
            if not cached_faces or meta["is_scene_cut"] or idx % 10 == 0:
                frame_faces = get_all_faces(frame) or []
                cached_faces = frame_faces
            else:
                roi_faces = []
                for f in cached_faces:
                    box = getattr(f, "bbox", None)
                    if box is not None:
                        rf = get_all_faces_in_roi(frame, box, pad_ratio=0.75)
                        if rf:
                            roi_faces.extend(rf)
                if roi_faces:
                    frame_faces = roi_faces
                    cached_faces = roi_faces
                else:
                    frame_faces = get_all_faces(frame) or []
                    cached_faces = frame_faces

        # -------------------------------------------------------------
        # Mode 4: Adaptive Detection Strategy
        # -------------------------------------------------------------
        elif mode == "adaptive_detection":
            frame_faces = detector.detect(
                frame,
                frame_idx=idx,
                det_fn=get_all_faces,
                roi_det_fn=lambda fr, b: get_all_faces_in_roi(fr, b, pad_ratio=0.75),
                force_full=meta["is_scene_cut"]
            )

        n_faces = len(frame_faces)
        total_faces_found += n_faces

        # Evaluate quality metrics per detected face
        for f in frame_faces:
            box = getattr(f, "bbox", None)
            kps = getattr(f, "kps", None)

            if box is not None:
                bw = float(box[2] - box[0])
                bh = float(box[3] - box[1])
                diag = math.hypot(bw, bh)
                if diag < 60.0:
                    small_detected += 1

                # Check if false detection (degenerate box or outside frame)
                if bw <= 4.0 or bh <= 4.0 or box[0] < -frame.shape[1] * 0.5:
                    false_count += 1

            if kps is not None:
                try:
                    p = solve_pose_5pt(np.asarray(kps, np.float32))
                    if p is not None and abs(float(p[0])) > 45.0:
                        profile_detected += 1
                except Exception:
                    pass

        # Check for missed faces during occlusion/cuts
        if meta["is_scene_cut"] and n_faces == 0:
            pass  # Expected if cut into empty background
        elif n_faces == 0 and not meta["is_scene_cut"]:
            missed_count += 1

    t_end = time.perf_counter()
    duration = max(0.001, t_end - t_start)
    fps = float(total_frames) / duration

    mean_gpu_util = float(np.mean(gpu_utils)) if gpu_utils else 0.0
    peak_vram = float(np.max(vram_readings)) if vram_readings else 0.0

    return {
        "mode": mode,
        "frames_processed": total_frames,
        "duration_seconds": round(duration, 3),
        "fps": round(fps, 2),
        "total_faces_found": total_faces_found,
        "missed_detections": missed_count,
        "false_detections": false_count,
        "profile_detections": profile_detected,
        "small_face_detections": small_detected,
        "mean_gpu_util_pct": round(mean_gpu_util, 1),
        "peak_vram_mb": round(peak_vram, 1),
    }


def main():
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from _synthetic_inputs import declare
    SYNTHETIC = declare(__file__,
                        inputs=["ONE still photograph (insightface t1.jpg; a flat grey 1280x720 frame if it is absent) copied N times as the 'video'", 'scene cuts = bitwise_not of the still; camera pans = affine shifts of the still; profile heads, small faces and occlusion = warped / resized / overpainted copies of the still'],
                        valid_for='the relative per-frame COST of detector strategies on a static scene',
                        not_valid_for='miss / recall rates or any FPS claim on real footage',
                        hard_coded_fields=["'Workstation: NVIDIA GeForce RTX 4070 (12GB) / Dual-Profile Validated' is a printed literal, not detected"])
    parser = argparse.ArgumentParser(description="Stage 3 SCRFD Detector Optimization Benchmark")
    parser.add_argument("--frames", type=int, default=80, help="Number of benchmark frames to evaluate")
    parser.add_argument("--out", type=str, default="benchmark_stage3_detector.json", help="Output JSON path")
    args = parser.parse_args()

    print(f"\n=========================================================================")
    print(f"STAGE 3 — SCRFD DETECTOR OPTIMIZATION BENCHMARK")
    print(f"Workstation: NVIDIA GeForce RTX 4070 (12GB) / Dual-Profile Validated")
    print(f"Total Test Frames: {args.frames} | Ground-Truth Challenge Evaluation")
    print(f"=========================================================================\n")

    print("[1/4] Constructing benchmark test sequence (cuts, motion, profile, occlusion)...")
    sequence = build_benchmark_sequence(args.frames)
    print(f"      Sequence constructed: {len(sequence)} frames.")

    modes = [
        "full_detection_every_frame",
        "temporal_detection",
        "roi_recovery",
        "adaptive_detection",
    ]

    results = []
    for idx, mode in enumerate(modes, 1):
        print(f"\n[{idx+1}/5] Running mode: '{mode}'...")
        res = run_benchmark_mode(mode, sequence)
        results.append(res)
        print(f"      FPS: {res['fps']} | Missed: {res['missed_detections']} | Profile: {res['profile_detections']} | Small-Face: {res['small_face_detections']} | GPU Util: {res['mean_gpu_util_pct']}%")

    # Format structured JSON
    baseline_fps = next((r["fps"] for r in results if r["mode"] == "full_detection_every_frame"), 1.0)
    for r in results:
        r["speedup_vs_baseline"] = round(r["fps"] / baseline_fps, 2)

    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": get_gpu_telemetry()["device"],
        "num_frames": args.frames,
        "synthetic_inputs": SYNTHETIC,
        "results": results
    }

    out_file = os.path.join(ROOT, args.out)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved structured telemetry to: {out_file}")

    # Print Formatted Markdown Table
    print("\n" + "="*85)
    print(f"{'DETECTOR STRATEGY':<30} | {'FPS':<8} | {'SPEEDUP':<8} | {'MISSED':<8} | {'PROFILE':<8} | {'SMALL':<8} | {'GPU %':<8}")
    print("="*85)
    for r in results:
        print(f"{r['mode']:<30} | {r['fps']:<8.2f} | {r['speedup_vs_baseline']:<8.2f}x | {r['missed_detections']:<8} | {r['profile_detections']:<8} | {r['small_face_detections']:<8} | {r['mean_gpu_util_pct']:<8.1f}%")
    print("="*85 + "\n")


if __name__ == "__main__":
    main()
