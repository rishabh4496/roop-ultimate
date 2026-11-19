"""Unit tests for app.tests.bench_metrics."""

import numpy as np
import pytest

from app.tests.bench_metrics import (
    BenchmarkMetricsTracker,
    FrameMetricRecord,
    YAW_BUCKETS,
    compute_landmark_jitter,
    compute_luminance_l1_delta,
    compute_mask_area,
    compute_mask_area_variance,
    compute_mask_relative_change,
    compute_seam_error,
    compute_ssim,
    compute_ssim_drop,
    detect_scene_cut,
    estimate_head_yaw,
    evaluate_benchmark_metrics,
    extract_face_roi,
    get_yaw_bucket,
)


def test_ssim_and_drop():
    img = np.full((128, 128, 3), 120, dtype=np.uint8)
    # Identical
    assert pytest.approx(compute_ssim(img, img), abs=1e-4) == 1.0
    assert pytest.approx(compute_ssim_drop(img, img), abs=1e-4) == 0.0

    # Slight modification
    img_mod = img.copy()
    img_mod[30:60, 30:60] = 160
    ssim_val = compute_ssim(img, img_mod)
    drop_val = compute_ssim_drop(img, img_mod)
    assert 0.0 < ssim_val < 1.0
    assert pytest.approx(drop_val, abs=1e-4) == 1.0 - ssim_val

    # Auto-resize mismatched shapes
    img_small = np.full((64, 64, 3), 120, dtype=np.uint8)
    assert pytest.approx(compute_ssim(img, img_small), abs=1e-4) == 1.0


def test_scene_cut_detection():
    # Consecutive smooth frames
    f1 = np.full((100, 100, 3), 100, dtype=np.uint8)
    f2 = np.full((100, 100, 3), 105, dtype=np.uint8)
    is_cut, diag = detect_scene_cut(f1, f2)
    assert not is_cut
    assert diag["ssim"] > 0.9

    # Abrupt scene cut: completely different scene
    f_black = np.zeros((100, 100, 3), dtype=np.uint8)
    f_white = np.full((100, 100, 3), 255, dtype=np.uint8)
    is_cut, diag = detect_scene_cut(f_black, f_white)
    assert is_cut
    assert diag["l1_diff"] > 100.0


def test_luminance_l1_delta():
    roi1 = np.full((64, 64, 3), 100, dtype=np.uint8)
    roi2 = np.full((64, 64, 3), 125, dtype=np.uint8)
    delta = compute_luminance_l1_delta(roi1, roi2)
    assert pytest.approx(delta, abs=1.0) == 25.0

    # Identical
    assert compute_luminance_l1_delta(roi1, roi1) == 0.0


def test_landmark_jitter():
    kps1 = np.array([[30.0, 40.0], [70.0, 40.0], [50.0, 60.0], [35.0, 80.0], [65.0, 80.0]])
    # Pure translation +10 x, +5 y
    kps_trans = kps1 + np.array([10.0, 5.0])
    jitter_comp = compute_landmark_jitter(kps1, kps_trans, compensate_translation=True)
    assert pytest.approx(jitter_comp, abs=1e-4) == 0.0

    # Actual jitter on single point
    kps_jitter = kps1.copy()
    kps_jitter[2] += np.array([3.0, 4.0])  # dist = 5.0 on point 2, 0 on others (mean 1.0 before translation comp)
    jitter = compute_landmark_jitter(kps1, kps_jitter, compensate_translation=True)
    assert jitter > 0.0


def test_mask_metrics():
    m1 = np.zeros((100, 100), dtype=np.uint8)
    m1[20:80, 20:80] = 255
    area1 = compute_mask_area(m1)
    assert area1 == 3600.0

    m2 = np.zeros((100, 100), dtype=np.uint8)
    m2[20:70, 20:80] = 255
    area2 = compute_mask_area(m2)
    assert area2 == 3000.0

    rel_change = compute_mask_relative_change(m1, m2)
    assert pytest.approx(rel_change, abs=1e-3) == (3600.0 - 3000.0) / 3600.0

    _, var = compute_mask_area_variance([m1, m2])
    assert var > 0.0


def test_yaw_and_bucketing():
    assert get_yaw_bucket(5.0) == "0-20"
    assert get_yaw_bucket(-15.0) == "0-20"
    assert get_yaw_bucket(25.0) == "20-40"
    assert get_yaw_bucket(-35.0) == "20-40"
    assert get_yaw_bucket(45.0) == "40-60"
    assert get_yaw_bucket(75.0) == "60+"

    # 5-point landmark yaw estimation
    kps_frontal = np.array([
        [38.0, 51.0],
        [73.0, 51.0],
        [55.5, 71.0],
        [41.0, 92.0],
        [70.0, 92.0]
    ])
    yaw = estimate_head_yaw(kps_frontal)
    assert abs(yaw) < 20.0
    assert get_yaw_bucket(yaw) == "0-20"


def test_seam_error():
    bg = np.full((120, 120, 3), 80, dtype=np.uint8)
    face = np.full((120, 120, 3), 200, dtype=np.uint8)

    # Empty mask
    empty_mask = np.zeros((120, 120), dtype=np.uint8)
    assert compute_seam_error(bg, empty_mask) == 0.0

    # Hard mask
    hard_mask = np.zeros((120, 120), dtype=np.uint8)
    hard_mask[30:90, 30:90] = 255
    comp_hard = bg.copy()
    comp_hard[hard_mask > 0] = face[hard_mask > 0]
    score_hard = compute_seam_error(comp_hard, hard_mask)

    # Smooth feathered mask
    import cv2
    smooth_mask = cv2.GaussianBlur(hard_mask.astype(np.float32), (15, 15), 5.0) / 255.0
    comp_smooth = (
        bg.astype(np.float32) * (1.0 - smooth_mask[..., None])
        + face.astype(np.float32) * smooth_mask[..., None]
    ).astype(np.uint8)
    score_smooth = compute_seam_error(comp_smooth, smooth_mask)

    assert score_hard > score_smooth > 0.0


def test_tracker_and_summary():
    tracker = BenchmarkMetricsTracker()

    # Add sequence of frames with known properties
    f_bg = np.full((100, 100, 3), 100, dtype=np.uint8)
    m = np.zeros((100, 100), dtype=np.uint8)
    m[20:80, 20:80] = 255
    kps = np.array([[38.0, 51.0], [73.0, 51.0], [55.5, 71.0], [41.0, 92.0], [70.0, 92.0]])

    # Frame 0
    tracker.add_frame(frame_index=0, image=f_bg, landmarks=kps, mask=m, is_detected=True, is_swapped=True, yaw=5.0)
    # Frame 1: slight delta
    f_bg1 = f_bg.copy()
    f_bg1[30:40, 30:40] = 110
    tracker.add_frame(frame_index=1, image=f_bg1, landmarks=kps, mask=m, is_detected=True, is_swapped=True, yaw=8.0)
    # Frame 2: turn to 30 deg
    tracker.add_frame(frame_index=2, image=f_bg, landmarks=kps, mask=m, is_detected=True, is_swapped=True, yaw=30.0)
    # Frame 3: profile 65 deg, not swapped
    tracker.add_frame(frame_index=3, image=f_bg, landmarks=kps, mask=m, is_detected=True, is_swapped=False, yaw=65.0)
    # Frame 4: scene cut
    f_white = np.full((100, 100, 3), 255, dtype=np.uint8)
    tracker.add_frame(frame_index=4, image=f_white, landmarks=kps, mask=m, is_detected=True, is_swapped=True, yaw=10.0)

    summary = tracker.compute_summary()

    assert "flicker" in summary
    assert "angle_buckets" in summary
    assert "overall_stability" in summary

    # Check flicker metrics
    flicker = summary["flicker"]
    assert flicker["valid_transitions"] > 0
    assert flicker["scene_cuts_excluded"] == 1
    assert 4 in flicker["scene_cut_indices"]

    # Check angle buckets
    buckets = summary["angle_buckets"]
    for b in YAW_BUCKETS:
        assert b in buckets
        assert "detection_recall" in buckets[b]
        assert "swap_coverage_rate" in buckets[b]
        assert "seam_error_score" in buckets[b]

    assert buckets["0-20"]["total_frames"] == 3  # frames 0, 1, 4
    assert buckets["20-40"]["total_frames"] == 1  # frame 2
    assert buckets["60+"]["total_frames"] == 1  # frame 3
    assert buckets["60+"]["swap_coverage_rate"] == 0.0  # frame 3 was not swapped


def test_evaluate_benchmark_metrics_convenience():
    data = [
        {"frame_index": 0, "is_detected": True, "is_swapped": True, "yaw": 12.0},
        {"frame_index": 1, "is_detected": True, "is_swapped": True, "yaw": 24.0},
    ]
    summary = evaluate_benchmark_metrics(data)
    assert summary["overall_stability"]["total_frames_analyzed"] == 2
    assert summary["angle_buckets"]["0-20"]["total_frames"] == 1
    assert summary["angle_buckets"]["20-40"]["total_frames"] == 1
