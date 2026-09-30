"""Unit and Integration Tests for Stage 4 — Face Alignment and Geometric Stability Engine."""

import math
import os
import sys
import numpy as np
import pytest

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
    get_geometric_stabilizer,
    get_pose_aware_solver,
    reset_geometric_stabilizer,
)


def _make_sample_kps(cx=256.0, cy=256.0, scale=1.0, roll_deg=0.0):
    """Generate synthetic 5-point facial landmarks around center."""
    base = ARCFACE_DST_112.copy()
    base_center = np.mean(base, axis=0)
    shifted = (base - base_center) * scale

    theta = math.radians(roll_deg)
    R = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ], dtype=np.float32)

    rot = (R @ shifted.T).T
    return (rot + np.array([cx, cy])).astype(np.float32)


# =====================================================================
# 1. Landmark Geometry Sanity Checks
# =====================================================================

def test_landmark_sanity_valid():
    kps = _make_sample_kps()
    is_valid, reason = LandmarkGeometrySanity.validate_landmarks(kps)
    assert is_valid is True
    assert reason == "valid"


def test_landmark_sanity_collapsed_eyes():
    kps = _make_sample_kps()
    # Collapse left and right eye to same point
    kps[1] = kps[0] + np.array([1.0, 0.0])
    is_valid, reason = LandmarkGeometrySanity.validate_landmarks(kps)
    assert is_valid is False
    assert "collapsed_inter_ocular" in reason


def test_landmark_sanity_non_finite():
    kps = _make_sample_kps()
    kps[2, 0] = np.nan
    is_valid, reason = LandmarkGeometrySanity.validate_landmarks(kps)
    assert is_valid is False
    assert reason == "non_finite_coordinates"


def test_matrix_sanity_valid_similarity():
    M = np.array([
        [1.2, -0.3, 10.0],
        [0.3,  1.2, 20.0]
    ], dtype=np.float32)
    is_valid, reason = LandmarkGeometrySanity.validate_affine_matrix(M)
    assert is_valid is True
    assert reason == "valid"


def test_matrix_sanity_sheared_fails():
    # Anisotropic non-uniform shear
    M = np.array([
        [2.5, 0.0, 10.0],
        [0.0, 0.5, 20.0]
    ], dtype=np.float32)
    is_valid, reason = LandmarkGeometrySanity.validate_affine_matrix(M)
    assert is_valid is False
    assert "anisotropic_shear" in reason


def test_matrix_sanity_inverted_det_fails():
    # Negative determinant (reflection/flip)
    M = np.array([
        [-1.0, 0.0, 0.0],
        [0.0,  1.0, 0.0]
    ], dtype=np.float32)
    is_valid, reason = LandmarkGeometrySanity.validate_affine_matrix(M)
    assert is_valid is False
    assert "negative_or_zero_determinant" in reason


# =====================================================================
# 2. Confidence-Aware Landmark Selection
# =====================================================================

def test_confidence_selector_frontal_symmetry():
    kps = _make_sample_kps()
    weights = ConfidenceLandmarkSelector.compute_weights(kps, yaw_degrees=0.0, pitch_degrees=0.0)
    assert weights.shape == (5,)
    assert np.isclose(np.sum(weights), 1.0)
    # Left and right eyes should have identical weights in frontal pose
    assert np.isclose(weights[0], weights[1], rtol=1e-3)
    assert np.isclose(weights[3], weights[4], rtol=1e-3)


def test_confidence_selector_asymmetric_yaw():
    kps = _make_sample_kps()
    # Turn right (+35 deg yaw): right eye and right mouth are far-side
    w_turn = ConfidenceLandmarkSelector.compute_weights(kps, yaw_degrees=35.0, pitch_degrees=0.0)
    assert w_turn[0] > w_turn[1]  # Near-side left eye weight > far-side right eye
    assert w_turn[3] > w_turn[4]  # Near-side left mouth > far-side right mouth
    assert w_turn[2] > w_turn[1]  # Nose weight boosted


def test_confidence_selector_occlusion_mask():
    kps = _make_sample_kps(cx=100, cy=100)
    mask = np.zeros((200, 200), dtype=np.float32)
    # Occlude right eye (kps[1])
    rx, ry = int(kps[1, 0]), int(kps[1, 1])
    mask[ry-5:ry+5, rx-5:rx+5] = 1.0

    w_occ = ConfidenceLandmarkSelector.compute_weights(kps, yaw_degrees=0.0, occlusion_mask=mask)
    assert w_occ[1] < w_occ[0] * 0.2  # Occluded eye severely attenuated


# =====================================================================
# 3. Temporal Smoothing without Oversmoothing (One-Euro Filter)
# =====================================================================

def test_one_euro_filter_jitter_reduction_at_low_velocity():
    filt = OneEuroFilter1D(min_cutoff=0.8, beta=0.05)
    np.random.seed(42)
    base_signal = 100.0
    noise = np.random.normal(0, 1.5, 60)  # 1.5 px jitter noise
    raw_samples = base_signal + noise

    filtered_samples = []
    for i in range(60):
        t = i / 30.0
        val = filt.filter(raw_samples[i], t)
        filtered_samples.append(val)

    # Variance of filtered samples should be at least 60% lower than raw noise
    raw_var = np.var(raw_samples[10:])
    filt_var = np.var(filtered_samples[10:])
    assert filt_var < raw_var * 0.40, f"Expected jitter variance reduction, got {filt_var} vs {raw_var}"


def test_one_euro_filter_preserves_rapid_motion_without_lag():
    """Verify that rapid motion spikes bypass smoothing (zero sluggish lag)."""
    filt = OneEuroFilter1D(min_cutoff=0.8, beta=0.05)
    # Constant baseline for 10 frames, then an abrupt rapid step of +80px at frame 11
    samples = [50.0] * 10 + [130.0] * 10
    filtered = []
    for i in range(20):
        t = i / 30.0
        filtered.append(filt.filter(samples[i], t, accel_kick_thresh=15.0))

    # At frame 11 (the instant of rapid movement), acceleration bypass snaps close to new target
    step_error = abs(filtered[11] - 130.0)
    assert step_error < 15.0, f"Filter lagged excessively on rapid movement, error={step_error}"


def test_temporal_geometry_stabilizer_multitrack():
    stabilizer = TemporalGeometryStabilizer()
    kps = _make_sample_kps()

    # Track 0 smooths across frames
    f0 = stabilizer.filter_landmarks(kps + np.random.normal(0, 0.5, kps.shape), track_id=0, frame_idx=0)
    f1 = stabilizer.filter_landmarks(kps + np.random.normal(0, 0.5, kps.shape), track_id=0, frame_idx=1)
    assert f0.shape == (5, 2)
    assert f1.shape == (5, 2)

    # Frame gap reset
    stabilizer.filter_landmarks(kps, track_id=0, frame_idx=20)
    # Track 1 independent
    f_t1 = stabilizer.filter_landmarks(kps, track_id=1, frame_idx=0)
    assert f_t1.shape == (5, 2)


# =====================================================================
# 4. Pose-Aware Alignment Solver
# =====================================================================

def test_pose_aware_solver_frontal():
    solver = PoseAwareAlignmentSolver()
    frame = np.full((400, 400, 3), 128, dtype=np.uint8)
    kps = _make_sample_kps(cx=200, cy=200, scale=1.0)

    crop, forward_M, inverse_M, meta = solver.align_face(
        frame, kps, crop_size=128, yaw_degrees=5.0, pitch_degrees=0.0, roll_degrees=0.0
    )
    assert crop.shape == (128, 128, 3)
    assert forward_M.shape == (2, 3)
    assert inverse_M.shape == (2, 3)
    assert meta["strategy"] == "frontal_5pt"


def test_pose_aware_solver_extreme_profile():
    solver = PoseAwareAlignmentSolver()
    frame = np.full((400, 400, 3), 128, dtype=np.uint8)
    kps = _make_sample_kps(cx=200, cy=200, scale=1.0)

    # 60 degrees yaw profile
    crop, forward_M, inverse_M, meta = solver.align_face(
        frame, kps, crop_size=128, yaw_degrees=60.0, pitch_degrees=0.0, roll_degrees=0.0
    )
    assert crop.shape == (128, 128, 3)
    assert meta["strategy"] == "profile_3pt"
    # Ensure forward matrix is non-sheared
    is_valid, _ = LandmarkGeometrySanity.validate_affine_matrix(forward_M)
    assert is_valid is True


def test_pose_aware_solver_canonical_roll_normalization():
    solver = PoseAwareAlignmentSolver()
    frame = np.full((500, 500, 3), 128, dtype=np.uint8)
    # Face tilted at 60 degrees roll
    kps = _make_sample_kps(cx=250, cy=250, scale=1.0, roll_deg=60.0)

    crop, forward_M, inverse_M, meta = solver.align_face(
        frame, kps, crop_size=128, yaw_degrees=0.0, pitch_degrees=0.0, roll_degrees=60.0
    )
    assert crop.shape == (128, 128, 3)
    assert meta["applied_roll_pre"] is True


def test_pose_aware_solver_small_face_upsampling():
    solver = PoseAwareAlignmentSolver()
    frame = np.full((400, 400, 3), 128, dtype=np.uint8)
    # Small face: scale 0.35 (diag < 60 px)
    kps = _make_sample_kps(cx=150, cy=150, scale=0.35)

    crop, forward_M, inverse_M, meta = solver.align_face(
        frame, kps, crop_size=128, yaw_degrees=0.0
    )
    assert crop.shape == (128, 128, 3)
    import cv2
    assert meta["interpolation"] == cv2.INTER_LANCZOS4


# =====================================================================
# 5. Stable Inverse Mapping
# =====================================================================

def test_stable_inverse_mapping_exactness():
    M = np.array([
        [0.85, -0.25, 45.0],
        [0.25,  0.85, 80.0]
    ], dtype=np.float32)

    inv_M = StableInverseMapper.invert_affine(M)
    # Compose forward and inverse: M_3 @ inv_M_3 should equal Identity 3x3
    M_3 = np.vstack([M, [0.0, 0.0, 1.0]])
    inv_M_3 = np.vstack([inv_M, [0.0, 0.0, 1.0]])
    prod = M_3 @ inv_M_3
    eye = np.eye(3, dtype=np.float32)
    assert np.allclose(prod, eye, atol=1e-5), f"Inverse matrix not identity: {prod}"


def test_stable_inverse_mapping_composition():
    A = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]], dtype=np.float32)
    B = np.array([[2.0, 0.0, 5.0],  [0.0, 2.0, 15.0]], dtype=np.float32)
    C = StableInverseMapper.compose_transforms(A, B)
    # (x * 2 + 5) + 10 = 2*x + 15
    assert np.isclose(C[0, 0], 2.0)
    assert np.isclose(C[0, 2], 15.0)
    assert np.isclose(C[1, 2], 35.0)


# =====================================================================
# 6. Global Accessor & Reset
# =====================================================================

def test_global_accessors():
    s1 = get_geometric_stabilizer()
    s2 = get_geometric_stabilizer()
    assert s1 is s2
    reset_geometric_stabilizer()
    p1 = get_pose_aware_solver()
    assert p1 is not None
