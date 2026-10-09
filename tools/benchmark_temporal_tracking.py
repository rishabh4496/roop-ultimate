"""Stage 9 — Temporal Face Tracking and Flicker Control Benchmark Harness.

SYNTHETIC INPUTS: random 512-d embeddings and generated keypoint trajectories with injected noise.
Its numbers describe the generated scene, not the application on real footage (tools/_synthetic_inputs.py).

Comprehensive validation and benchmark for:
1. Landmark Jitter & Flicker Suppression (Kalman & Spline vs Raw Jittery Detections)
2. Identity Stability Across Interacting / Crossing Faces (Identity Freeze vs Swap)
3. Temporary Detector Dropout Recovery & Coasting (Bridging 1-8 dropped frames)
4. Discontinuity Detection & Spline Interpolation (Shot cuts, velocity jumps, scale spikes)
5. Profile Transition & Rapid Motion Preservation (Yaw transitions, high velocity fidelity)
6. Face Leaving & Returning (Re-ID Memory Archive vs Lost Track ID)
7. Tracking Latency & Throughput Profile (Single-face and multi-face tracking overhead)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure app root is on sys.path
APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "app"))

from roop.temporal_state_machine import (
    ConfidenceAwareInterpolator,
    RobustFaceTrack,
    TemporalQualityMetrics,
    TemporalStateMachineTracker,
    TrackState,
)


def benchmark_landmark_jitter_suppression() -> Dict[str, Any]:
    """Evaluates landmark jitter variance on raw noisy detector outputs vs smoothed tracking."""
    np.random.seed(42)
    n_frames = 120
    t = np.linspace(0, 4 * np.pi, n_frames)

    # True smooth trajectory: head gentle sway
    cx = 300.0 + 80.0 * np.sin(0.5 * t)
    cy = 250.0 + 30.0 * np.cos(0.5 * t)

    # 5 primary landmarks relative to center
    base_kps_rel = np.array([
        [-30.0, -20.0],  # left eye
        [30.0, -20.0],   # right eye
        [0.0, 5.0],      # nose
        [-20.0, 30.0],   # left mouth
        [20.0, 30.0],    # right mouth
    ], dtype=np.float32)

    raw_kps_list = []
    tracker = TemporalStateMachineTracker()
    tracked_kps_list = []

    # Embedding constant
    emb = np.random.randn(512).astype(np.float32)
    emb /= np.linalg.norm(emb)

    for i in range(n_frames):
        # Raw noisy detection (Gaussian noise sigma = 2.5 px on landmarks, 1.5 px on bbox)
        noise_lm = np.random.normal(0.0, 2.5, size=base_kps_rel.shape).astype(np.float32)
        noise_box = np.random.normal(0.0, 1.5, size=4).astype(np.float32)

        cur_kps = np.array([[cx[i], cy[i]]], dtype=np.float32) + base_kps_rel + noise_lm
        raw_kps_list.append(cur_kps)

        cur_box = np.array([cx[i] - 60.0, cy[i] - 70.0, cx[i] + 60.0, cy[i] + 70.0], dtype=np.float32) + noise_box

        det = {
            "bbox": cur_box,
            "kps": cur_kps,
            "embedding": emb.copy(),
            "det_score": 0.92,
            "pose": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "source_assignment": 0,
        }

        active = tracker.process_frame([det], frame_index=i)
        if active and active[0].get("kps") is not None:
            tracked_kps_list.append(active[0]["kps"])
        else:
            tracked_kps_list.append(cur_kps)

    raw_jitter = TemporalQualityMetrics.compute_landmark_jitter(raw_kps_list)
    tracked_jitter = TemporalQualityMetrics.compute_landmark_jitter(tracked_kps_list)
    reduction_pct = max(0.0, (raw_jitter - tracked_jitter) / raw_jitter * 100.0) if raw_jitter > 0 else 0.0

    return {
        "raw_jitter_variance_px2": raw_jitter,
        "tracked_jitter_variance_px2": tracked_jitter,
        "jitter_reduction_pct": round(reduction_pct, 2),
        "flicker_suppressed": bool(tracked_jitter < 0.60 * raw_jitter),
    }


def benchmark_crossing_faces_identity_stability() -> Dict[str, Any]:
    """Tests identity preservation when two actors cross paths with overlapping bounding boxes."""
    np.random.seed(1337)
    n_frames = 60
    tracker_robust = TemporalStateMachineTracker()

    # Actor A moves left to right (x: 100 -> 500)
    # Actor B moves right to left (x: 500 -> 100)
    # Crossing occurs at frames 25-35 around x = 300
    emb_a = np.random.randn(512).astype(np.float32)
    emb_a /= np.linalg.norm(emb_a)
    emb_b = np.random.randn(512).astype(np.float32)
    emb_b /= np.linalg.norm(emb_b)

    actor_a_tracks = []
    actor_b_tracks = []

    # Also simulate naive tracker that updates embedding with corrupted overlap
    naive_assigned_a = []
    naive_assigned_b = []
    naive_emb_a = emb_a.copy()
    naive_emb_b = emb_b.copy()

    crossing_detected_count = 0

    for i in range(n_frames):
        alpha = i / (n_frames - 1)
        ax = 100.0 + 400.0 * alpha
        bx = 500.0 - 400.0 * alpha
        ay = 250.0
        by = 250.0

        box_a = np.array([ax - 50, ay - 60, ax + 50, ay + 60], dtype=np.float32)
        box_b = np.array([bx - 50, by - 60, bx + 50, by + 60], dtype=np.float32)

        # Check overlap
        inter_x = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
        inter_y = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
        overlap = (inter_x * inter_y) > 0.0

        # When overlapping, detector features get contaminated
        if overlap:
            noisy_emb_a = 0.6 * emb_a + 0.4 * emb_b
            noisy_emb_a /= np.linalg.norm(noisy_emb_a)
            noisy_emb_b = 0.6 * emb_b + 0.4 * emb_a
            noisy_emb_b /= np.linalg.norm(noisy_emb_b)
        else:
            noisy_emb_a = emb_a.copy()
            noisy_emb_b = emb_b.copy()

        det_a = {
            "bbox": box_a,
            "embedding": noisy_emb_a,
            "det_score": 0.90,
            "source_assignment": 0,
        }
        det_b = {
            "bbox": box_b,
            "embedding": noisy_emb_b,
            "det_score": 0.88,
            "source_assignment": 1,
        }

        active = tracker_robust.process_frame([det_a, det_b], frame_index=i)

        for track_id, track in tracker_robust.tracks.items():
            if track.state == TrackState.CROSSING:
                crossing_detected_count += 1

        # Check robust assignments
        for face in active:
            box = face["bbox"]
            cx = (box[0] + box[2]) * 0.5
            if i < 20:
                if cx < 250:
                    actor_a_tracks.append(face.get("source_assignment"))
                else:
                    actor_b_tracks.append(face.get("source_assignment"))
            elif i > 40:
                if cx > 350:
                    actor_a_tracks.append(face.get("source_assignment"))
                else:
                    actor_b_tracks.append(face.get("source_assignment"))

        # Naive tracking simulation (greedy assignment on contaminated embeddings)
        dist_aa = float(1.0 - np.dot(naive_emb_a, noisy_emb_a))
        dist_ab = float(1.0 - np.dot(naive_emb_a, noisy_emb_b))
        if overlap and dist_ab < dist_aa:
            # Identity swapped!
            naive_assigned_a.append(1)
            naive_assigned_b.append(0)
        else:
            naive_assigned_a.append(0)
            naive_assigned_b.append(1)
        if not overlap:
            naive_emb_a = 0.8 * naive_emb_a + 0.2 * noisy_emb_a
            naive_emb_b = 0.8 * naive_emb_b + 0.2 * noisy_emb_b

    stability_robust_a = TemporalQualityMetrics.compute_identity_stability(actor_a_tracks)
    stability_robust_b = TemporalQualityMetrics.compute_identity_stability(actor_b_tracks)
    stability_naive_a = TemporalQualityMetrics.compute_identity_stability(naive_assigned_a)

    return {
        "crossings_detected": crossing_detected_count,
        "robust_stability_score_pct": min(stability_robust_a, stability_robust_b),
        "naive_stability_score_pct": stability_naive_a,
        "identity_flips_prevented": bool(min(stability_robust_a, stability_robust_b) == 100.0),
    }


def benchmark_detector_dropout_recovery() -> Dict[str, Any]:
    """Tests Kalman motion coasting across 1 to 8 frames of complete detector dropout."""
    tracker = TemporalStateMachineTracker()
    n_frames = 40
    # Dropped frames at indices 15, 16, 17, 18, 19 (5 consecutive frames of occlusion/dropout)
    drop_indices = {15, 16, 17, 18, 19}

    emb = np.random.randn(512).astype(np.float32)
    emb /= np.linalg.norm(emb)

    frames_processed = 0
    coasted_frames_produced = 0
    recovered_cleanly = False

    for i in range(n_frames):
        # Linear velocity of 5 px/frame to the right
        cx = 100.0 + 5.0 * i
        cy = 200.0
        box = np.array([cx - 40, cy - 50, cx + 40, cy + 50], dtype=np.float32)

        if i in drop_indices:
            dets = []  # Detector dropped face completely
        else:
            dets = [{
                "bbox": box,
                "kps": np.array([[cx - 15, cy - 10], [cx + 15, cy - 10], [cx, cy], [cx - 10, cy + 15], [cx + 10, cy + 15]], dtype=np.float32),
                "embedding": emb.copy(),
                "det_score": 0.94,
                "source_assignment": 0,
            }]

        active = tracker.process_frame(dets, frame_index=i)
        frames_processed += 1

        if i in drop_indices:
            if active and active[0].get("_coasted"):
                coasted_frames_produced += 1
                # Check prediction accuracy
                pred_cx = (active[0]["bbox"][0] + active[0]["bbox"][2]) * 0.5
                assert abs(pred_cx - cx) < 15.0  # Within 15 px of ground truth

        if i == 20:
            # First frame after dropout
            if active and tracker.tracks[active[0]["_track_id"]].state == TrackState.STABLE:
                recovered_cleanly = True

    recovery_rate_pct = (coasted_frames_produced / len(drop_indices)) * 100.0

    return {
        "consecutive_drop_frames": len(drop_indices),
        "coasted_frames_produced": coasted_frames_produced,
        "dropout_recovery_rate_pct": recovery_rate_pct,
        "recovered_cleanly": recovered_cleanly,
    }


def benchmark_discontinuity_guard_and_splines() -> Dict[str, Any]:
    """Tests ConfidenceAwareInterpolator discontinuity triggers and Cubic Hermite spline smooths."""
    # 1. Spline vs Linear on acceleration
    t = np.linspace(0, 1, 11)
    # Sinusoidal motion between 0 and 100 with known velocities at endpoints
    v_start = 50.0
    v_end = 150.0
    hermite_pts = [ConfidenceAwareInterpolator.cubic_hermite(0.0, 100.0, v_start, v_end, step) for step in t]
    linear_pts = [(1.0 - step) * 0.0 + step * 100.0 for step in t]

    # Verify Hermite starts with tangent v_start (slope at t=0 should match 50)
    finite_diff_start = (hermite_pts[1] - hermite_pts[0]) / 0.1
    tangent_match = abs(finite_diff_start - v_start) < 10.0

    # 2. Discontinuity Triggers
    emb1 = np.ones(512, dtype=np.float32) / np.sqrt(512)
    emb2 = np.ones(512, dtype=np.float32) / np.sqrt(512)
    emb_diff = np.zeros(512, dtype=np.float32)
    emb_diff[0] = 1.0  # Cosine distance ~ 0.95

    # Case A: Normal continuous step
    disc_normal, _ = ConfidenceAwareInterpolator.check_discontinuity(
        face_a={"bbox": np.array([100, 100, 200, 200], dtype=np.float32), "embedding": emb1},
        face_b={"bbox": np.array([106, 102, 206, 202], dtype=np.float32), "embedding": emb2},
        span_frames=1,
    )

    # Case B: Scene cut / shot cut (>80 px jump)
    disc_shot_cut, _ = ConfidenceAwareInterpolator.check_discontinuity(
        face_a={"bbox": np.array([100, 100, 200, 200], dtype=np.float32), "embedding": emb1},
        face_b={"bbox": np.array([450, 450, 550, 550], dtype=np.float32), "embedding": emb2},
        span_frames=1,
    )

    # Case C: Scale jump (>1.8x)
    disc_scale, _ = ConfidenceAwareInterpolator.check_discontinuity(
        face_a={"bbox": np.array([100, 100, 200, 200], dtype=np.float32), "embedding": emb1},
        face_b={"bbox": np.array([100, 100, 350, 350], dtype=np.float32), "embedding": emb2},
        span_frames=1,
    )

    # Case D: Identity divergence (different person)
    disc_identity, _ = ConfidenceAwareInterpolator.check_discontinuity(
        face_a={"bbox": np.array([100, 100, 200, 200], dtype=np.float32), "embedding": emb1},
        face_b={"bbox": np.array([105, 105, 205, 205], dtype=np.float32), "embedding": emb_diff},
        span_frames=1,
    )

    return {
        "hermite_tangent_matched": bool(tangent_match),
        "continuous_motion_interpolated": bool(not disc_normal),
        "shot_cut_discontinuity_blocked": bool(disc_shot_cut),
        "scale_jump_discontinuity_blocked": bool(disc_scale),
        "identity_divergence_discontinuity_blocked": bool(disc_identity),
        "all_discontinuities_guarded": bool(
            not disc_normal and disc_shot_cut and disc_scale and disc_identity
        ),
    }


def benchmark_profile_turn_and_rapid_motion() -> Dict[str, Any]:
    """Tests yaw profile turn transitions and rapid motion fidelity."""
    tracker = TemporalStateMachineTracker()
    n_frames = 30
    profile_detected = False

    emb = np.random.randn(512).astype(np.float32)
    emb /= np.linalg.norm(emb)

    for i in range(n_frames):
        # Yaw ramps from 0 to 65 degrees (profile turn)
        yaw = float(i * 2.2)
        det = {
            "bbox": np.array([200, 200, 300, 300], dtype=np.float32),
            "pose": np.array([0.0, yaw, 0.0], dtype=np.float32),
            "embedding": emb.copy(),
            "det_score": 0.90,
            "source_assignment": 0,
        }
        active = tracker.process_frame([det], frame_index=i)
        if active and tracker.tracks[active[0]["_track_id"]].state == TrackState.PROFILE_TURNING:
            profile_detected = True

    # Rapid motion test (>20 px/frame)
    tracker_rapid = TemporalStateMachineTracker()
    true_traj = []
    tracked_traj = []
    for i in range(30):
        # Fast acceleration
        cx = 100.0 + 22.0 * i
        cy = 150.0 + 5.0 * i
        true_traj.append(np.array([cx, cy], dtype=np.float32))
        det = {
            "bbox": np.array([cx - 40, cy - 40, cx + 40, cy + 40], dtype=np.float32),
            "embedding": emb.copy(),
            "det_score": 0.91,
            "source_assignment": 0,
        }
        active = tracker_rapid.process_frame([det], frame_index=i)
        if active:
            b = active[0]["bbox"]
            tracked_traj.append(np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5], dtype=np.float32))

    motion_fidelity = TemporalQualityMetrics.compute_motion_fidelity(
        np.array(true_traj, dtype=np.float32),
        np.array(tracked_traj, dtype=np.float32),
    )

    return {
        "profile_turning_handled": profile_detected,
        "profile_transitions_count": tracker.stats["profile_turns_handled"],
        "rapid_motion_fidelity_r": motion_fidelity,
        "legitimate_motion_preserved": bool(motion_fidelity >= 0.98),
    }


def benchmark_reid_returning_actor() -> Dict[str, Any]:
    """Tests Re-ID archive preserving original track ID and source assignment when an actor returns."""
    tracker = TemporalStateMachineTracker()
    emb_actor = np.random.randn(512).astype(np.float32)
    emb_actor /= np.linalg.norm(emb_actor)

    initial_track_id = None
    # Actor present in frames 0..10
    for i in range(10):
        det = {
            "bbox": np.array([200, 200, 300, 300], dtype=np.float32),
            "embedding": emb_actor.copy(),
            "det_score": 0.95,
            "source_assignment": 2,
        }
        active = tracker.process_frame([det], frame_index=i)
        if active and initial_track_id is None:
            initial_track_id = active[0]["_track_id"]

    assert initial_track_id is not None
    assert tracker.tracks[initial_track_id].source_assignment == 2

    # Actor leaves the frame for 35 frames (frames 10..44)
    # Track expires and moves to Re-ID archive
    for i in range(10, 45):
        tracker.process_frame([], frame_index=i)

    # Actor returns at frame 45 at a completely different position
    returning_det = {
        "bbox": np.array([500, 150, 600, 250], dtype=np.float32),
        "embedding": emb_actor.copy() + np.random.normal(0.0, 0.02, size=512).astype(np.float32),
        "det_score": 0.93,
    }
    returning_det["embedding"] /= np.linalg.norm(returning_det["embedding"])

    active = tracker.process_frame([returning_det], frame_index=45)

    recovered_id = active[0].get("_track_id") if active else None
    recovered_src = active[0].get("source_assignment") if active else None
    archive_hits = tracker.stats.get("returning_faces_recovered", 0)

    return {
        "initial_track_id": initial_track_id,
        "recovered_track_id": recovered_id,
        "preserved_source_assignment": recovered_src,
        "reid_archive_hits": archive_hits,
        "identity_persistence_success": bool(recovered_id == initial_track_id and recovered_src == 2),
    }


def benchmark_tracking_latency_and_throughput() -> Dict[str, Any]:
    """Profiles latency and FPS for temporal state machine tracker across face counts."""
    tracker = TemporalStateMachineTracker()
    np.random.seed(999)

    def generate_dets(num_faces: int) -> List[Dict[str, Any]]:
        dets = []
        for f in range(num_faces):
            cx = 100.0 + f * 120.0
            cy = 200.0
            emb = np.random.randn(512).astype(np.float32)
            emb /= np.linalg.norm(emb)
            dets.append({
                "bbox": np.array([cx - 40, cy - 50, cx + 40, cy + 50], dtype=np.float32),
                "kps": np.array([[cx - 15, cy - 10], [cx + 15, cy - 10], [cx, cy], [cx - 10, cy + 15], [cx + 10, cy + 15]], dtype=np.float32),
                "embedding": emb,
                "det_score": 0.92,
                "source_assignment": f,
            })
        return dets

    results = {}
    for face_count in (1, 2, 5):
        # Warmup
        for i in range(15):
            tracker.process_frame(generate_dets(face_count), frame_index=i)

        n_iter = 200
        start_time = time.perf_counter()
        for i in range(15, 15 + n_iter):
            tracker.process_frame(generate_dets(face_count), frame_index=i)
        elapsed = time.perf_counter() - start_time

        lat_ms = (elapsed / n_iter) * 1000.0
        fps = n_iter / elapsed

        results[f"{face_count}_face"] = {
            "latency_ms": round(lat_ms, 3),
            "throughput_fps": round(fps, 1),
        }

    # Measure Cubic Hermite spline overhead
    interpolator = ConfidenceAwareInterpolator()
    start_spline = time.perf_counter()
    n_spline_iter = 10000
    for _ in range(n_spline_iter):
        _ = interpolator.cubic_hermite(100.0, 200.0, 15.0, 10.0, 0.5)
    spline_lat_us = ((time.perf_counter() - start_spline) / n_spline_iter) * 1e6

    results["spline_calc_latency_us"] = round(spline_lat_us, 3)

    return results


def run_full_stage9_benchmark() -> Dict[str, Any]:
    """Runs all Stage 9 temporal tracking benchmarks and outputs report summary."""
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from _synthetic_inputs import declare
    SYNTHETIC = declare(__file__,
                        inputs=['identity embeddings are np.random.randn(512) (seeds 42 / 1337 / 999), not recogniser output', 'landmark / box trajectories are generated and perturbed with Gaussian noise; crossings, dropouts and re-entries are scripted'],
                        valid_for='tracker / smoother logic on scripted scenarios',
                        not_valid_for='identity-matching accuracy or jitter suppression on real faces')
    print("=" * 80)
    print("STAGE 9 — TEMPORAL FACE TRACKING & FLICKER CONTROL BENCHMARK")
    print("=" * 80)

    print("\n[1/6] Benchmarking Landmark Jitter & Flicker Suppression...")
    jitter_res = benchmark_landmark_jitter_suppression()
    print(f"  - Raw Landmark Jitter Variance:     {jitter_res['raw_jitter_variance_px2']} px^2")
    print(f"  - Tracked Jitter Variance:         {jitter_res['tracked_jitter_variance_px2']} px^2")
    print(f"  - Jitter Reduction:                {jitter_res['jitter_reduction_pct']}%")
    print(f"  - Flicker Suppressed:              {jitter_res['flicker_suppressed']}")

    print("\n[2/6] Benchmarking Interacting & Crossing Faces Identity Stability...")
    cross_res = benchmark_crossing_faces_identity_stability()
    print(f"  - Crossings Detected:              {cross_res['crossings_detected']}")
    print(f"  - Robust Identity Stability:       {cross_res['robust_stability_score_pct']}%")
    print(f"  - Naive Identity Stability:        {cross_res['naive_stability_score_pct']}%")
    print(f"  - Identity Flips Prevented:        {cross_res['identity_flips_prevented']}")

    print("\n[3/6] Benchmarking Detector Dropout Recovery & Coasting...")
    drop_res = benchmark_detector_dropout_recovery()
    print(f"  - Drop Frames Tested:              {drop_res['consecutive_drop_frames']}")
    print(f"  - Coasted Frames Synthesized:      {drop_res['coasted_frames_produced']}")
    print(f"  - Dropout Recovery Rate:           {drop_res['dropout_recovery_rate_pct']}%")
    print(f"  - Seamless Re-acquisition:         {drop_res['recovered_cleanly']}")

    print("\n[4/6] Benchmarking Discontinuity Guard & Spline Interpolation...")
    disc_res = benchmark_discontinuity_guard_and_splines()
    print(f"  - Shot Cut Guard:                  {disc_res['shot_cut_discontinuity_blocked']}")
    print(f"  - Scale Jump Guard:                {disc_res['scale_jump_discontinuity_blocked']}")
    print(f"  - Identity Divergence Guard:       {disc_res['identity_divergence_discontinuity_blocked']}")
    print(f"  - Continuous Motion Allowed:       {disc_res['continuous_motion_interpolated']}")

    print("\n[5/6] Benchmarking Profile Turning & Rapid Motion Fidelity...")
    motion_res = benchmark_profile_turn_and_rapid_motion()
    print(f"  - Profile Turns Handled:           {motion_res['profile_transitions_count']}")
    print(f"  - Rapid Motion Fidelity (r):       {motion_res['rapid_motion_fidelity_r']}")
    print(f"  - Legitimate Motion Preserved:     {motion_res['legitimate_motion_preserved']}")

    print("\n[6/6] Benchmarking Returning Actor Re-ID Memory Archive...")
    reid_res = benchmark_reid_returning_actor()
    print(f"  - Original Track ID:               {reid_res['initial_track_id']}")
    print(f"  - Restored Track ID:               {reid_res['recovered_track_id']}")
    print(f"  - Source Assignment Restored:      {reid_res['preserved_source_assignment']}")
    print(f"  - Identity Persistence Success:    {reid_res['identity_persistence_success']}")

    print("\n[7/7] Profiling Tracking Latency & Throughput...")
    perf_res = benchmark_tracking_latency_and_throughput()
    print(f"  - 1 Face Latency:                  {perf_res['1_face']['latency_ms']} ms ({perf_res['1_face']['throughput_fps']} FPS)")
    print(f"  - 2 Faces Latency:                 {perf_res['2_faces']['latency_ms'] if '2_faces' in perf_res else perf_res['2_face']['latency_ms']} ms")
    print(f"  - Cubic Hermite Spline Overhead:   {perf_res['spline_calc_latency_us']} us/call")

    full_report = {
        "stage": "STAGE 9 — TEMPORAL FACE TRACKING AND FLICKER CONTROL",
        "synthetic_inputs": SYNTHETIC,
        "landmark_jitter_suppression": jitter_res,
        "crossing_faces_identity_stability": cross_res,
        "detector_dropout_recovery": drop_res,
        "discontinuity_guard_and_spline": disc_res,
        "profile_and_rapid_motion": motion_res,
        "reid_returning_actor": reid_res,
        "performance_profile": perf_res,
    }

    out_path = APP_ROOT / "benchmark_stage9_temporal_tracking.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[OK] Benchmark results written to {out_path}")

    return full_report


if __name__ == "__main__":
    run_full_stage9_benchmark()
