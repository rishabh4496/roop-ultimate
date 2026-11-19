# SPDX-License-Identifier: MIT
"""Unit tests for Component 2 (Persistent Identity Tracking with Anti-ID-Jump)
and Component 3 (Adaptive One-Euro Filter on Geometry).
"""

import math
import numpy as np
import pytest

from roop.tracker import FaceTracker, FaceTrack, _face_field, _set_face_field
from roop.multi_face_tracker import MultiFaceTracker
from roop.one_euro import AdaptiveGeometryStabilizer, OneEuroFilter
from roop.temporal_stabilizer import FaceTrackStabilizer
from roop.typing import Face


POINTS = np.array([
    [30.0, 28.0],
    [60.0, 28.0],
    [45.0, 46.0],
    [33.0, 67.0],
    [57.0, 67.0]
], dtype=np.float32)


def make_test_face(x=0.0, y=0.0, embedding=None, det_score=0.95, kps=None, bbox=None):
    if kps is None:
        p = POINTS.copy()
        p[:, 0] += x
        p[:, 1] += y
    else:
        p = np.asarray(kps, dtype=np.float32)

    if bbox is None:
        b = np.array([15.0 + x, 13.0 + y, 75.0 + x, 85.0 + y], dtype=np.float32)
    else:
        b = np.asarray(bbox, dtype=np.float32)

    if embedding is None:
        emb = np.array([1.0, 0.0] + [0.0] * 510, dtype=np.float32)
    else:
        emb = np.asarray(embedding, dtype=np.float32)
        if emb.size < 512:
            emb = np.pad(emb, (0, 512 - emb.size))
    norm = np.linalg.norm(emb)
    if norm > 1e-6:
        emb = emb / norm

    return Face(bbox=b, kps=p, embedding=emb, det_score=float(det_score))


def test_anti_id_jump_hysteresis_maintains_active_track():
    """Active track maintains ID even when appearance similarity drops (e.g. to 0.25)
    due to noisy detection or extreme profile, because hysteresis threshold is lower
    to maintain than to switch."""
    tracker = FaceTracker(max_cost=0.85)
    emb_a = np.array([1.0, 0.0] + [0.0] * 510, dtype=np.float32)
    emb_b = np.array([0.0, 1.0] + [0.0] * 510, dtype=np.float32)

    # Establish two active tracks across 3 frames
    for f_idx in range(3):
        faces = [
            make_test_face(x=30.0, embedding=emb_a),
            make_test_face(x=150.0, embedding=emb_b),
        ]
        tracked = tracker.update(faces, frame_index=f_idx)
        assert len(tracked) == 2

    # Check established tracks
    track_ids_frame2 = [f['_track_id'] for f in tracked]
    assert len(set(track_ids_frame2)) == 2
    tid_a, tid_b = track_ids_frame2[0], track_ids_frame2[1]

    # Frame 3: Face A has noisy embedding (sim ~0.26 against track A, but high spatial IoU)
    noisy_emb = 0.26 * emb_a + 0.96 * np.roll(emb_a, 5)
    noisy_emb = noisy_emb / np.linalg.norm(noisy_emb)

    faces_noisy = [
        make_test_face(x=32.0, embedding=noisy_emb),
        make_test_face(x=152.0, embedding=emb_b),
    ]
    tracked_noisy = tracker.update(faces_noisy, frame_index=3)
    assert len(tracked_noisy) == 2

    # Track IDs must remain stable without jumping or dropping
    assert tracked_noisy[0]['_track_id'] == tid_a
    assert tracked_noisy[1]['_track_id'] == tid_b


def test_anti_id_jump_hysteresis_rejects_id_switch_under_ambiguity():
    """Challenger face cannot steal an active track without clearing the higher
    SWITCH_SIM_THRESH and overcoming the HYSTERESIS_MARGIN."""
    tracker = MultiFaceTracker(max_cost=0.85)
    emb_alice = np.array([1.0, 0.0] + [0.0] * 510, dtype=np.float32)
    emb_bob = np.array([0.0, 1.0] + [0.0] * 510, dtype=np.float32)

    # Frames 0..3: Establish tracks Alice at x=80 and Bob at x=120
    for f in range(4):
        faces = [
            make_test_face(x=80.0, embedding=emb_alice),
            make_test_face(x=120.0, embedding=emb_bob),
        ]
        tracked = tracker.update(faces, frame_index=f)
        assert len(tracked) == 2

    tid_alice = tracked[0]['_track_id']
    tid_bob = tracked[1]['_track_id']
    assert tid_alice != tid_bob

    # Frame 4: Faces are close (e.g. crossing).
    # Introduce noisy frame where detection at x=82 has mild similarity to Bob (~0.35)
    # and lower similarity to Alice (~0.28).
    # Without hysteresis, Bob might steal detection 0.
    # With anti-ID-jump hysteresis, Alice has active continuity bonus and Bob cannot switch
    # because 0.35 is below SWITCH_SIM_THRESH (0.48).
    ambig_emb = 0.28 * emb_alice + 0.35 * emb_bob + 0.89 * np.roll(emb_alice, 10)
    ambig_emb = ambig_emb / np.linalg.norm(ambig_emb)

    faces_cross = [
        make_test_face(x=82.0, embedding=ambig_emb),
        make_test_face(x=118.0, embedding=emb_bob),
    ]
    tracked_cross = tracker.update(faces_cross, frame_index=4)
    assert len(tracked_cross) == 2
    assert tracked_cross[0]['_track_id'] == tid_alice
    assert tracked_cross[1]['_track_id'] == tid_bob


def test_miss_grace_buffer_propagates_geometry_and_decays_opacity():
    """Miss-grace buffer: if detector misses 1 to 3 frames, propagate smoothed
    Kalman prediction with decayed opacity, eliminating 1-frame unswapped flash."""
    tracker = FaceTracker(max_coast=3, min_hits_to_coast=2)
    emb = np.array([1.0, 0.0] + [0.0] * 510, dtype=np.float32)

    # Establish track over frames 0, 1, 2
    for f in range(3):
        faces = [make_test_face(x=50.0 + f * 2.0, embedding=emb)]
        tracked = tracker.update(faces, frame_index=f)
        assert len(tracked) == 1

    tid = tracked[0]['_track_id']

    # Frame 3: Detector misses frame completely (motion blur / occluder)
    tracked_empty = tracker.update([], frame_index=3)
    assert len(tracked_empty) == 0

    coasted_1 = tracker.coast(frame_index=3, frame_shape=(720, 1280, 3))
    assert len(coasted_1) == 1
    c1 = coasted_1[0]
    assert c1['_track_id'] == tid
    assert c1['_coasted'] is True
    assert c1['_coast_age'] == 1
    # 1st missed frame: opacity = 1.0 - 1 * 0.25 = 0.75
    assert 0.70 <= c1['opacity'] <= 0.80

    # Frame 4: Second missed frame
    tracker.update([], frame_index=4)
    coasted_2 = tracker.coast(frame_index=4, frame_shape=(720, 1280, 3))
    assert len(coasted_2) == 1
    c2 = coasted_2[0]
    assert c2['_coast_age'] == 2
    # 2nd missed frame: opacity = 1.0 - 2 * 0.25 = 0.50
    assert 0.45 <= c2['opacity'] <= 0.55

    # Frame 5: Third missed frame
    tracker.update([], frame_index=5)
    coasted_3 = tracker.coast(frame_index=5, frame_shape=(720, 1280, 3))
    assert len(coasted_3) == 1
    c3 = coasted_3[0]
    assert c3['_coast_age'] == 3
    # 3rd missed frame: opacity = 1.0 - 3 * 0.25 = 0.25
    assert 0.20 <= c3['opacity'] <= 0.30

    # Frame 6: Fourth missed frame exceeds miss grace (max_coast=3) -> must expire!
    tracker.update([], frame_index=6)
    coasted_4 = tracker.coast(frame_index=6, frame_shape=(720, 1280, 3))
    assert len(coasted_4) == 0


def test_adaptive_one_euro_landmarks_eliminates_subpixel_jitter():
    """Stationary faces lock rock-solid: One-Euro Filter on 5-point landmarks
    damps high-frequency sub-pixel detector jitter by more than 70%."""
    stabilizer = AdaptiveGeometryStabilizer(min_cutoff=0.05, beta=1.5, d_cutoff=2.0)
    rng = np.random.default_rng(42)

    raw_diffs = []
    smoothed_diffs = []
    prev_raw = None
    prev_smooth = None

    for i in range(40):
        # Sub-pixel noise ±0.4 px
        noise = rng.uniform(-0.4, 0.4, size=POINTS.shape).astype(np.float32)
        raw_kps = POINTS + noise
        out = stabilizer.apply(landmarks=raw_kps, bbox=[15, 13, 75, 85], t=i / 30.0, track_id=1)
        smooth_kps = out['landmarks']

        if prev_raw is not None:
            raw_diffs.append(np.abs(raw_kps - prev_raw).mean())
            smoothed_diffs.append(np.abs(smooth_kps - prev_smooth).mean())

        prev_raw = raw_kps
        prev_smooth = smooth_kps

    mean_raw_jitter = np.mean(raw_diffs)
    mean_smooth_jitter = np.mean(smoothed_diffs)
    # Jitter should be heavily suppressed
    assert mean_smooth_jitter < 0.08, f"Smooth jitter too high: {mean_smooth_jitter:.4f}"
    assert mean_smooth_jitter < mean_raw_jitter * 0.35, "Jitter suppression < 65%"


def test_adaptive_one_euro_landmarks_follows_fast_head_turn():
    """Fast head turns track with negligible lag: on a fast displacement step (18 px),
    the filter adapts and follows with error < 0.7 px."""
    stabilizer = AdaptiveGeometryStabilizer(min_cutoff=0.05, beta=2.5, d_cutoff=2.0)

    # 15 frames of resting face
    for i in range(15):
        stabilizer.apply(landmarks=POINTS, bbox=[15, 13, 75, 85], t=i / 30.0, track_id=1)

    # Sudden fast head turn step of 18 pixels
    step_kps = POINTS.copy()
    step_kps[:, 0] += 18.0
    step_bbox = np.array([15 + 18, 13, 75 + 18, 85], dtype=np.float32)

    # Apply step
    out_step = stabilizer.apply(landmarks=step_kps, bbox=step_bbox, t=15 / 30.0, track_id=1)
    smooth_step_kps = out_step['landmarks']
    smooth_step_bbox = out_step['bbox']

    kps_lag = np.max(np.abs(smooth_step_kps[:, 0] - step_kps[:, 0]))
    bbox_lag = np.max(np.abs(smooth_step_bbox - step_bbox))

    # Fast head turn lag must be negligible (< 0.7 px)
    assert kps_lag < 0.7, f"Keypoint lag too large: {kps_lag:.4f} px"
    assert bbox_lag < 0.7, f"Bounding box lag too large: {bbox_lag:.4f} px"


def test_adaptive_one_euro_affine_preserves_similarity_no_shear():
    """Affine warp parameters smoothed in similarity space [tx, ty, log(scale), angle]
    eliminate matrix jitter while preserving orthogonality (zero shear)."""
    stabilizer = AdaptiveGeometryStabilizer(min_cutoff=0.05, beta=1.5, d_cutoff=2.0)
    rng = np.random.default_rng(123)

    base_matrix = np.array([
        [1.2, -0.3, 50.0],
        [0.3,  1.2, 80.0]
    ], dtype=np.float32)

    for i in range(30):
        # Corrupt translation, scale, rotation with detector noise
        noise_tx = rng.uniform(-1.0, 1.0)
        noise_ty = rng.uniform(-1.0, 1.0)
        matrix = base_matrix.copy()
        matrix[0, 2] += noise_tx
        matrix[1, 2] += noise_ty

        out = stabilizer.apply(affine=matrix, t=i / 30.0, track_id=1)
        smooth_m = out['affine']

        assert smooth_m.shape == (2, 3)
        # Check similarity property: col0 and col1 must be orthogonal and equal length
        col0 = smooth_m[:, 0]
        col1 = smooth_m[:, 1]
        dot = float(np.dot(col0, col1))
        len0 = float(np.linalg.norm(col0))
        len1 = float(np.linalg.norm(col1))

        # Orthogonality: dot product near 0 (zero shear)
        assert abs(dot) < 1e-5, f"Shear detected in smoothed affine: dot={dot}"
        # Equal scaling: len0 == len1
        assert abs(len0 - len1) < 1e-5, f"Anisotropic scaling in affine: {len0} vs {len1}"


def test_adaptive_geometry_miss_grace_propagation():
    """Verify AdaptiveGeometryStabilizer propagates predicted geometry and decayed opacity
    during detector dropout, and stops after miss_grace frames."""
    stabilizer = AdaptiveGeometryStabilizer(miss_grace=3)
    kps = POINTS.copy()
    bbox = np.array([10.0, 10.0, 100.0, 100.0], dtype=np.float32)
    matrix = np.array([[1.0, 0.0, 20.0], [0.0, 1.0, 30.0]], dtype=np.float32)

    # Initial frame observed
    out0 = stabilizer.apply(landmarks=kps, bbox=bbox, affine=matrix, t=0.0, track_id=1, detected=True)
    assert out0['predicted'] is False
    assert out0['opacity'] == 1.0

    # Miss frame 1
    out1 = stabilizer.apply(t=1/30.0, track_id=1, detected=False)
    assert out1['predicted'] is True
    assert out1['landmarks'] is not None
    assert 0.70 <= out1['opacity'] <= 0.80

    # Miss frame 2
    out2 = stabilizer.apply(t=2/30.0, track_id=1, detected=False)
    assert out2['predicted'] is True
    assert 0.45 <= out2['opacity'] <= 0.55

    # Miss frame 3
    out3 = stabilizer.apply(t=3/30.0, track_id=1, detected=False)
    assert out3['predicted'] is True
    assert 0.20 <= out3['opacity'] <= 0.30

    # Miss frame 4 (exceeds miss_grace=3) -> must drop
    out4 = stabilizer.apply(t=4/30.0, track_id=1, detected=False)
    assert out4['predicted'] is False
    assert out4['landmarks'] is None
    assert out4['opacity'] == 0.0


def test_scene_cut_hard_reset_clears_all_buffers():
    """Scene cut: reset() flushes all temporal buffers, preventing ghosting across cuts."""
    stabilizer = AdaptiveGeometryStabilizer()
    stabilizer.apply(landmarks=POINTS, bbox=[10, 10, 80, 80], t=0.0, track_id=1)
    assert 1 in stabilizer.tracks

    # Cut triggers hard reset
    stabilizer.reset()
    assert len(stabilizer.tracks) == 0

    # Re-observed face starts fresh without memory from previous shot
    out = stabilizer.apply(landmarks=POINTS + 100.0, bbox=[110, 110, 180, 180], t=1.0, track_id=1)
    np.testing.assert_array_almost_equal(out['landmarks'], POINTS + 100.0, decimal=2)


def load_tests(loader, tests, pattern):
    """Expose this module's bare `test_*` functions to `unittest discover`."""
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:
        from unittest_shim import load_tests_for
    return load_tests_for(globals())

