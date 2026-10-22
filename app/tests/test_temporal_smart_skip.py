"""Unit and integration tests for Motion-Compensated Temporal Frame Reuse (Smart-Skip)."""
import os
import numpy as np
import pytest

import roop.globals
from roop.temporal_smart_skip import (
    compute_inner_landmark_displacement,
    compute_roi_luma,
    compute_illumination_delta,
    is_smart_skip_enabled,
    mark_smart_skips_in_track,
    DenseFlowWarper,
    get_dense_flow_warper,
    set_track_patch,
    get_track_patch,
    clear_track_patches,
    MAX_CONSECUTIVE_SKIPS,
    DISPLACEMENT_THRESHOLD_PX,
    ILLUMINATION_DELTA_THRESHOLD,
)
from roop.trajectory_cache import FLAG_REUSE_FLOW_WARP, _flags_of, TrajectoryCache


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())


def test_inner_landmark_displacement():
    # 5 keypoints: 0: left eye, 1: right eye, 2: nose, 3: left mouth, 4: right mouth
    kps_base = np.array([
        [100.0, 100.0],
        [150.0, 100.0],
        [125.0, 130.0],
        [110.0, 160.0],
        [140.0, 160.0],
    ], dtype=np.float32)

    # Identical: disp == 0.0
    disp_zero = compute_inner_landmark_displacement(kps_base, kps_base)
    assert disp_zero == 0.0

    # Shift inner points by 0.1 px
    kps_shifted = kps_base.copy()
    kps_shifted[[0, 1, 3, 4], 0] += 0.1
    disp = compute_inner_landmark_displacement(kps_shifted, kps_base)
    assert pytest.approx(disp, abs=1e-4) == 0.1

    # Shift nose only (index 2) - should NOT affect inner landmark displacement!
    kps_nose = kps_base.copy()
    kps_nose[2] += [50.0, 50.0]
    disp_nose = compute_inner_landmark_displacement(kps_nose, kps_base)
    assert disp_nose == 0.0

    # Invalid cases
    assert compute_inner_landmark_displacement(None, kps_base) == float("inf")
    assert compute_inner_landmark_displacement(kps_base, None) == float("inf")
    assert compute_inner_landmark_displacement(np.array([[1.0, 2.0]]), kps_base) == float("inf")
    assert compute_inner_landmark_displacement(np.array([[np.nan, 0.0]] * 5), kps_base) == float("inf")


def test_roi_luma_and_delta():
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    luma = compute_roi_luma(frame, bbox=[10, 10, 50, 50])
    assert 0.45 < luma < 0.55

    frame_dark = np.full((100, 100, 3), 64, dtype=np.uint8)
    luma_dark = compute_roi_luma(frame_dark, bbox=[10, 10, 50, 50])
    delta = compute_illumination_delta(luma, luma_dark)
    assert delta > 0.2

    # Robustness on empty/None
    assert compute_roi_luma(None) == 0.5
    assert compute_roi_luma(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.5


def test_mark_smart_skips_in_track():
    kps = np.array([
        [100.0, 100.0],
        [150.0, 100.0],
        [125.0, 130.0],
        [110.0, 160.0],
        [140.0, 160.0],
    ], dtype=np.float32)

    # Sequence of 6 near-static frames
    merged = {
        i: {
            "kps": kps.copy() + np.random.uniform(-0.02, 0.02, kps.shape).astype(np.float32),
            "roi_luma": 0.5 + 0.005 * i,
        }
        for i in range(10, 16)
    }

    # Mark skips with max 2 consecutive skips
    count = mark_smart_skips_in_track(merged, max_consecutive=2, disp_threshold=0.15, illum_threshold=0.05)

    # Frame 10: First frame (anchor) -> Not skipped
    assert merged[10]["reuse_flow_warp"] is False
    # Frame 11: 1st skip
    assert merged[11]["reuse_flow_warp"] is True
    assert merged[11]["mode"] == "REUSE_FLOW_WARP"
    # Frame 12: 2nd skip
    assert merged[12]["reuse_flow_warp"] is True
    # Frame 13: Capped at 2 -> Not skipped!
    assert merged[13]["reuse_flow_warp"] is False
    # Frame 14: 1st skip again
    assert merged[14]["reuse_flow_warp"] is True
    # Frame 15: 2nd skip again
    assert merged[15]["reuse_flow_warp"] is True

    assert count == 4


def test_mark_smart_skips_scene_cuts_and_keyframes():
    kps = np.array([
        [100.0, 100.0],
        [150.0, 100.0],
        [125.0, 130.0],
        [110.0, 160.0],
        [140.0, 160.0],
    ], dtype=np.float32)

    merged = {
        i: {"kps": kps.copy(), "roi_luma": 0.5}
        for i in range(5)
    }

    # Cut on frame 1, keyframe on frame 3
    cuts = {1}
    keyframes = {3}
    count = mark_smart_skips_in_track(merged, cuts=cuts, keyframes=keyframes)

    # Frame 0: anchor
    assert merged[0]["reuse_flow_warp"] is False
    # Frame 1: Scene cut -> Protected, never skipped
    assert merged[1]["reuse_flow_warp"] is False
    # Frame 2: Can be skipped
    assert merged[2]["reuse_flow_warp"] is True
    # Frame 3: Keyframe -> Protected, never skipped
    assert merged[3]["reuse_flow_warp"] is False
    # Frame 4: Can be skipped
    assert merged[4]["reuse_flow_warp"] is True

    assert count == 2


def test_mark_smart_skips_large_displacement_rejected():
    kps_base = np.array([
        [100.0, 100.0],
        [150.0, 100.0],
        [125.0, 130.0],
        [110.0, 160.0],
        [140.0, 160.0],
    ], dtype=np.float32)

    # Large displacement: 0.5 px (> 0.15 px threshold)
    kps_moved = kps_base.copy()
    kps_moved[[0, 1, 3, 4]] += 0.5

    merged = {
        0: {"kps": kps_base, "roi_luma": 0.5},
        1: {"kps": kps_moved, "roi_luma": 0.5},
    }
    count = mark_smart_skips_in_track(merged, disp_threshold=0.15)
    assert count == 0
    assert merged[1]["reuse_flow_warp"] is False


def test_dense_flow_warper():
    warper = DenseFlowWarper(flow_tier="dis")
    img1 = np.zeros((64, 64, 3), dtype=np.uint8)
    img2 = np.zeros((64, 64, 3), dtype=np.uint8)
    img1[20:40, 20:40] = 255
    img2[22:42, 22:42] = 255

    flow = warper.compute_dense_flow(img2, img1)
    assert flow.shape == (64, 64, 2)
    assert np.isfinite(flow).all()

    warped = warper.warp(img1, flow)
    assert warped.shape == img1.shape
    assert warped.dtype == np.uint8


def test_track_patch_cache():
    clear_track_patches()
    assert get_track_patch(42) is None

    patch = np.full((128, 128, 3), 200, dtype=np.uint8)
    aligned = np.full((128, 128, 3), 150, dtype=np.uint8)

    set_track_patch(42, patch, aligned, frame_idx=10, has_enhanced=True)
    entry = get_track_patch(42)
    assert entry is not None
    assert entry["frame_idx"] == 10
    assert entry["has_enhanced"] is True
    assert np.array_equal(entry["patch"], patch)
    assert np.array_equal(entry["aligned_img"], aligned)

    clear_track_patches()
    assert get_track_patch(42) is None


def test_trajectory_cache_reuse_flag():
    face = {"reuse_flow_warp": True}
    flags = _flags_of(face)
    assert flags & FLAG_REUSE_FLOW_WARP == FLAG_REUSE_FLOW_WARP

    face_no_reuse = {"reuse_flow_warp": False}
    flags_no = _flags_of(face_no_reuse)
    assert flags_no & FLAG_REUSE_FLOW_WARP == 0


def test_smart_skip_enabled_toggle(monkeypatch):
    monkeypatch.delenv("ROOP_TEMPORAL_SMART_SKIP", raising=False)
    roop.globals.temporal_smart_skip = False
    assert is_smart_skip_enabled() is False

    roop.globals.temporal_smart_skip = True
    assert is_smart_skip_enabled() is True

    monkeypatch.setenv("ROOP_TEMPORAL_SMART_SKIP", "0")
    assert is_smart_skip_enabled() is False

    monkeypatch.setenv("ROOP_TEMPORAL_SMART_SKIP", "1")
    assert is_smart_skip_enabled() is True
