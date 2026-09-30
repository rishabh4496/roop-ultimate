"""Stage 4 Benchmark: Face Alignment and Geometric Stability.

Measures alignment stability and geometric fidelity independently from face swap quality:
1. Baseline: Raw 5-point Umeyama (no temporal smoothing).
2. Fixed EMA Filter: Matrix-level EMA (alpha=0.85).
3. Stage 4 Pose-Aware Geometric Engine: Confidence weighting + 3-point profile anchor
   + Acceleration-Adaptive One-Euro Filter + Roll pre-normalization + Adaptive interpolation.

Measures across difficult cases (20–90° yaw, pitch, roll, small faces, border faces, occlusions, rapid motion):
- Landmark Jitter (px RMS)
- Translation Jitter (px RMS)
- Scale Jitter (RMS)
- Rotation Jitter (deg RMS)
- Dynamic Lag / Response Error on Rapid Motion (px)
- Extreme Profile Success Rate (%)
- Border / Edge Face Handling (%)
- Matrix Health & Non-Shear Compliance (%)
- Throughput (FPS)

Outputs:
- benchmark_stage4_alignment.json
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

from roop.geometric_alignment import (
    ARCFACE_DST_112,
    ConfidenceLandmarkSelector,
    GeometricAlignmentConfig,
    LandmarkGeometrySanity,
    OneEuroFilter1D,
    PoseAwareAlignmentSolver,
    StableInverseMapper,
    TemporalGeometryStabilizer,
)


def generate_geometric_test_sequence(num_frames: int = 120) -> List[Dict[str, Any]]:
    """Build a comprehensive sequence simulating real-world geometric challenges:
    - Frames 0..29: Stationary frontal face with detector subpixel jitter (1.2 px std dev)
    - Frames 30..49: Profile rotation from 0° up to 75° yaw
    - Frames 50..69: Pitch variations (-30° to +30°) and rapid roll tilt (up to 60°)
    - Frames 70..89: Rapid head motion / acceleration spike (sudden velocity of 25 px/frame)
    - Frames 90..104: Moving foreground occlusion crossing eye/mouth
    - Frames 105..119: Distant small face (< 50 px) and face positioned at image edge
    """
    frames_meta = []
    base_cx, base_cy = 320.0, 240.0
    frame_h, frame_w = 480, 640

    np.random.seed(1337)

    for i in range(num_frames):
        # 1. Base face parameters
        cx = base_cx
        cy = base_cy
        scale = 1.0
        yaw = 0.0
        pitch = 0.0
        roll = 0.0
        has_occlusion = False
        is_small = False
        is_border = False
        is_rapid = False

        # Phase 1: Stationary with subpixel jitter
        if 0 <= i < 30:
            pass  # Jitter added below

        # Phase 2: Extreme Profile Rotation (yaw 0 -> 75 deg)
        elif 30 <= i < 50:
            yaw = 15.0 + ((i - 30) / 19.0) * 60.0  # 15 deg -> 75 deg

        # Phase 3: Pitch and Roll Variations
        elif 50 <= i < 70:
            pitch = math.sin((i - 50) / 20.0 * math.pi) * 35.0
            roll = math.cos((i - 50) / 20.0 * math.pi) * 55.0

        # Phase 4: Rapid Motion Spike (head turn / camera jerk)
        elif 70 <= i < 90:
            is_rapid = True
            if 75 <= i <= 82:
                # Sudden high velocity translation and yaw snap
                cx += (i - 75) * 18.0
                cy += (i - 75) * 6.0
                yaw = 40.0

        # Phase 5: Foreground Occlusion
        elif 90 <= i < 105:
            has_occlusion = True

        # Phase 6: Small face & Border face
        else:
            is_small = True
            is_border = True
            scale = 0.35  # ~38 px face
            cx = 40.0     # Touching left border
            cy = 50.0

        # Generate ground-truth 5-point landmarks
        base_kps = ARCFACE_DST_112.copy()
        base_center = np.mean(base_kps, axis=0)
        shifted = (base_kps - base_center) * scale

        # Apply Roll
        theta = math.radians(roll)
        R_mat = np.array([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)]
        ], dtype=np.float32)
        rot_kps = (R_mat @ shifted.T).T

        # Apply Yaw foreshortening simulation
        if abs(yaw) > 10.0:
            is_right = yaw >= 0
            # Compress far side
            far_idx = 1 if is_right else 0
            far_m_idx = 4 if is_right else 3
            compress = max(0.2, 1.0 - (abs(yaw) / 90.0) * 0.8)
            rot_kps[far_idx, 0] = rot_kps[2, 0] + (rot_kps[far_idx, 0] - rot_kps[2, 0]) * compress
            rot_kps[far_m_idx, 0] = rot_kps[2, 0] + (rot_kps[far_m_idx, 0] - rot_kps[2, 0]) * compress

        gt_kps = (rot_kps + np.array([cx, cy])).astype(np.float32)

        # Inject realistic detector jitter (Gaussian noise std=1.2 px, except during small face: 0.5 px)
        noise_std = 0.5 if is_small else 1.2
        jitter_noise = np.random.normal(0, noise_std, gt_kps.shape).astype(np.float32)
        observed_kps = gt_kps + jitter_noise

        # Generate synthetic frame image
        frame = np.full((frame_h, frame_w, 3), 120, dtype=np.uint8)
        # Draw face ellipse
        cv2.ellipse(
            frame,
            (int(round(cx)), int(round(cy))),
            (int(round(40 * scale)), int(round(55 * scale))),
            roll, 0, 360, (200, 180, 170), -1
        )

        occ_mask = None
        if has_occlusion:
            occ_mask = np.zeros((frame_h, frame_w), dtype=np.float32)
            # Occlude right eye/mouth
            ox, oy = int(round(gt_kps[1, 0])), int(round(gt_kps[1, 1]))
            cv2.rectangle(frame, (ox - 15, oy - 15), (ox + 15, oy + 15), (20, 20, 20), -1)
            cv2.rectangle(occ_mask, (ox - 15, oy - 15), (ox + 15, oy + 15), 1.0, -1)

        frames_meta.append({
            "frame_idx": i,
            "frame": frame,
            "gt_kps": gt_kps,
            "observed_kps": observed_kps,
            "cx": cx, "cy": cy, "scale": scale,
            "yaw": yaw, "pitch": pitch, "roll": roll,
            "has_occlusion": has_occlusion,
            "occ_mask": occ_mask,
            "is_small": is_small,
            "is_border": is_border,
            "is_rapid": is_rapid
        })

    return frames_meta


def evaluate_alignment_strategy(
    strategy_name: str,
    sequence: List[Dict[str, Any]],
    crop_size: int = 128
) -> Dict[str, Any]:
    """Execute alignment strategy over sequence and measure jitter, lag, and matrix health."""
    solver = PoseAwareAlignmentSolver()
    stabilizer = TemporalGeometryStabilizer()
    stabilizer.reset_all()

    # Legacy EMA state
    legacy_prev_M = None
    legacy_alpha = 0.85

    landmark_errors = []
    translation_diffs = []
    scale_diffs = []
    rotation_diffs = []
    rapid_motion_lags = []
    matrix_health_valid = 0
    profile_success = 0
    profile_count = 0
    border_success = 0
    border_count = 0

    prev_cx = None
    prev_cy = None
    prev_scale = None
    prev_rot = None
    # Pre-warm solver
    _ = solver.align_face(sequence[0]["frame"], sequence[0]["observed_kps"], crop_size=crop_size)

    template_dst = solver.get_template(crop_size, "arcface")
    template_center_2d = np.mean(template_dst, axis=0)
    template_center_homo = np.array([template_center_2d[0], template_center_2d[1], 1.0], dtype=np.float32)

    t_start = time.perf_counter()

    for idx, item in enumerate(sequence):
        frame = item["frame"]
        obs_kps = item["observed_kps"].copy()
        gt_kps = item["gt_kps"]
        yaw = item["yaw"]
        pitch = item["pitch"]
        roll = item["roll"]
        occ_mask = item["occ_mask"]

        if abs(yaw) >= 45.0:
            profile_count += 1
        if item["is_border"]:
            border_count += 1

        M_forward = None

        # -------------------------------------------------------------
        # 1. Baseline: Raw 5-Point Umeyama (no temporal smoothing)
        # -------------------------------------------------------------
        if strategy_name == "baseline_raw_umeyama":
            dst = solver.get_template(crop_size, "arcface")
            M_forward = solver.solve_similarity(obs_kps, dst, weights=None)

        # -------------------------------------------------------------
        # 2. Fixed EMA Filter: Matrix-level EMA
        # -------------------------------------------------------------
        elif strategy_name == "fixed_matrix_ema":
            dst = solver.get_template(crop_size, "arcface")
            raw_M = solver.solve_similarity(obs_kps, dst, weights=None)
            if legacy_prev_M is None:
                M_forward = raw_M
            else:
                M_forward = (legacy_alpha * raw_M + (1.0 - legacy_alpha) * legacy_prev_M).astype(np.float32)
            legacy_prev_M = M_forward

        # -------------------------------------------------------------
        # 3. Stage 4: Pose-Aware + Adaptive One-Euro Stabilizer
        # -------------------------------------------------------------
        elif strategy_name == "stage4_pose_aware_stabilizer":
            # 1. Filter observed landmarks with acceleration-adaptive One-Euro filter
            filt_kps = stabilizer.filter_landmarks(obs_kps, track_id=0, frame_idx=idx)

            # 2. Align face with pose awareness (yaw/pitch/roll + confidence weighting)
            _, M_forward, _, meta = solver.align_face(
                frame, filt_kps, crop_size=crop_size,
                yaw_degrees=yaw, pitch_degrees=pitch, roll_degrees=roll,
                occlusion_mask=occ_mask
            )

        # Evaluate Matrix Health & Compliance
        is_valid, _ = LandmarkGeometrySanity.validate_affine_matrix(M_forward)
        if is_valid:
            matrix_health_valid += 1
            if abs(yaw) >= 45.0:
                profile_success += 1
            if item["is_border"]:
                border_success += 1

        # Extract translation, scale, rotation from M_forward
        linear = M_forward[:, :2]
        s_val = float(np.mean(np.linalg.norm(linear, axis=1)))
        rot_rad = math.atan2(linear[1, 0], linear[0, 0])
        rot_deg = math.degrees(rot_rad)

        inv_M = StableInverseMapper.invert_affine(M_forward)
        # Template center mapped to full frame
        mapped_cx, mapped_cy = (inv_M @ template_center_homo)[:2]

        # Landmark jitter error against GT
        if idx < 30:  # In stationary jitter phase
            trans_gt = (inv_M @ np.hstack([template_dst, np.ones((5, 1))]).T).T[:, :2]
            err = np.mean(np.linalg.norm(trans_gt - gt_kps, axis=1))
            landmark_errors.append(err)

            if prev_cx is not None:
                translation_diffs.append(math.hypot(mapped_cx - prev_cx, mapped_cy - prev_cy))
                scale_diffs.append(abs(s_val - prev_scale))
                rotation_diffs.append(abs(rot_deg - prev_rot))

        # Dynamic Lag on rapid motion spike
        if item["is_rapid"]:
            lag_dist = math.hypot(mapped_cx - item["cx"], mapped_cy - item["cy"])
            rapid_motion_lags.append(lag_dist)

        prev_cx = mapped_cx
        prev_cy = mapped_cy
        prev_scale = s_val
        prev_rot = rot_deg

    t_end = time.perf_counter()
    duration = max(0.001, t_end - t_start)
    fps = float(len(sequence)) / duration

    lm_jitter_rms = float(np.sqrt(np.mean(np.square(landmark_errors)))) if landmark_errors else 0.0
    trans_jitter_rms = float(np.sqrt(np.mean(np.square(translation_diffs)))) if translation_diffs else 0.0
    scale_jitter_rms = float(np.sqrt(np.mean(np.square(scale_diffs)))) if scale_diffs else 0.0
    rot_jitter_rms = float(np.sqrt(np.mean(np.square(rotation_diffs)))) if rotation_diffs else 0.0
    mean_lag = float(np.mean(rapid_motion_lags)) if rapid_motion_lags else 0.0

    profile_rate = (profile_success / max(1, profile_count)) * 100.0
    border_rate = (border_success / max(1, border_count)) * 100.0
    matrix_health_rate = (matrix_health_valid / len(sequence)) * 100.0

    return {
        "strategy": strategy_name,
        "fps": round(fps, 1),
        "landmark_jitter_rms_px": round(lm_jitter_rms, 3),
        "translation_jitter_rms_px": round(trans_jitter_rms, 3),
        "scale_jitter_rms": round(scale_jitter_rms, 5),
        "rotation_jitter_rms_deg": round(rot_jitter_rms, 3),
        "rapid_motion_lag_px": round(mean_lag, 2),
        "profile_success_pct": round(profile_rate, 1),
        "border_success_pct": round(border_rate, 1),
        "matrix_health_pct": round(matrix_health_rate, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 4 Geometric Alignment & Stability Benchmark")
    parser.add_argument("--frames", type=int, default=120, help="Number of benchmark sequence frames")
    args = parser.parse_args()

    print(f"=== STAGE 4 BENCHMARK: FACE ALIGNMENT & GEOMETRIC STABILITY ({args.frames} frames) ===")
    seq = generate_geometric_test_sequence(args.frames)

    strategies = [
        "baseline_raw_umeyama",
        "fixed_matrix_ema",
        "stage4_pose_aware_stabilizer"
    ]

    results = []
    for strat in strategies:
        print(f"Running strategy: '{strat}'...")
        res = evaluate_alignment_strategy(strat, seq)
        results.append(res)
        print(f"   Done -> FPS: {res['fps']} | Trans Jitter: {res['translation_jitter_rms_px']} px | "
              f"Lag: {res['rapid_motion_lag_px']} px | Profile: {res['profile_success_pct']}% | "
              f"Health: {res['matrix_health_pct']}%")

    out_file = os.path.join(ROOT, "benchmark_stage4_alignment.json")
    with open(out_file, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}, f, indent=2)
    print(f"\nSaved structured telemetry to: {out_file}\n")

    print("=" * 115)
    print(f"{'STRATEGY':<30} | {'FPS':<7} | {'LM JITTER':<10} | {'TRANS JIT':<10} | {'ROT JIT':<8} | {'LAG (PX)':<9} | {'PROFILE %':<9} | {'HEALTH %'}")
    print("=" * 115)
    for r in results:
        print(f"{r['strategy']:<30} | {r['fps']:<7} | {r['landmark_jitter_rms_px']:<10} | "
              f"{r['translation_jitter_rms_px']:<10} | {r['rotation_jitter_rms_deg']:<8} | "
              f"{r['rapid_motion_lag_px']:<9} | {r['profile_success_pct']:<9} | {r['matrix_health_pct']}")
    print("=" * 115)


if __name__ == "__main__":
    main()
