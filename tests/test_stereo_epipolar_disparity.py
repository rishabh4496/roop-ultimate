"""Stage 21: Stereoscopic 3D (SBS/TB) Epipolar Disparity Lock & VR Viewport Tests.

Verifies:
1. Stereo format modes: MONO_2D, STEREO_SBS_HALF, STEREO_SBS_FULL, STEREO_TB, VR180_SBS.
2. Frame partitioning into Left and Right views and lossless recombination.
3. Epipolar Geometric Coupling:
   - Zero vertical epipolar drift (delta_y < 0.05 px) between coupled left and right face alignments.
   - Identical in-plane roll angles across the stereo pair.
   - Horizontal disparity smoothing via EMA filter (eliminates perceived depth pulsing).
4. Spatial & Identity pairing of multiple faces across L and R stereo views.
5. Latent noise seed, normalization statistics, and Stage 12 specular highlight mask synchronization.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from face_engine.core.config import EngineConfig, StereoMode
from face_engine.pipeline.stereo import (
    EpipolarCouplingResult,
    EpipolarStereoCoupler,
    StereoViewPartition,
    is_stereo_active,
    partition_stereo_frame,
    recombine_stereo_frame,
    synchronize_stereo_crops,
)
from app.roop.geometric_alignment import get_stereo_coupler


# Canonical 5-point face landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
CANONICAL_5PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


class TestStereoFormatModesAndPartitioning:
    """Tests stereo format modes enum, EngineConfig, and frame partitioning."""

    def test_stereo_modes_enum(self):
        modes = [
            StereoMode.MONO_2D,
            StereoMode.STEREO_SBS_HALF,
            StereoMode.STEREO_SBS_FULL,
            StereoMode.STEREO_TB,
            StereoMode.VR180_SBS,
        ]
        assert len(modes) == 5
        assert StereoMode("MONO_2D") == StereoMode.MONO_2D
        assert StereoMode("STEREO_SBS_HALF") == StereoMode.STEREO_SBS_HALF
        assert StereoMode("VR180_SBS") == StereoMode.VR180_SBS

    def test_engine_config_stereo_mode_default_and_override(self):
        cfg = EngineConfig()
        assert cfg.stereo_mode == StereoMode.MONO_2D

        cfg_sbs = EngineConfig(stereo_mode=StereoMode.STEREO_SBS_HALF)
        assert cfg_sbs.stereo_mode == StereoMode.STEREO_SBS_HALF

        cfg_vr = EngineConfig(stereo_mode=StereoMode.VR180_SBS)
        assert cfg_vr.stereo_mode == StereoMode.VR180_SBS

    def test_is_stereo_active(self):
        assert not is_stereo_active(StereoMode.MONO_2D)
        assert not is_stereo_active("MONO_2D")
        assert not is_stereo_active(None)

        assert is_stereo_active(StereoMode.STEREO_SBS_HALF)
        assert is_stereo_active(StereoMode.STEREO_SBS_FULL)
        assert is_stereo_active(StereoMode.STEREO_TB)
        assert is_stereo_active(StereoMode.VR180_SBS)

    def test_partition_and_recombine_sbs(self):
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Left half = red, right half = blue
        frame[:, : w // 2, 2] = 255
        frame[:, w // 2 :, 0] = 255

        partition = partition_stereo_frame(frame, StereoMode.STEREO_SBS_HALF)
        assert partition.left_frame.shape == (h, w // 2, 3)
        assert partition.right_frame.shape == (h, w // 2, 3)
        assert partition.left_roi == (0, 0, w // 2, h)
        assert partition.right_roi == (w // 2, 0, w // 2, h)

        # Check content separation
        assert np.all(partition.left_frame[:, :, 2] == 255)
        assert np.all(partition.right_frame[:, :, 0] == 255)

        # Recombine
        recombined = recombine_stereo_frame(partition.left_frame, partition.right_frame, StereoMode.STEREO_SBS_HALF)
        assert recombined.shape == frame.shape
        np.testing.assert_array_equal(recombined, frame)

    def test_partition_and_recombine_tb(self):
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Top half = green, bottom half = yellow
        frame[: h // 2, :, 1] = 255
        frame[h // 2 :, :, 0] = 255
        frame[h // 2 :, :, 1] = 255

        partition = partition_stereo_frame(frame, StereoMode.STEREO_TB)
        assert partition.left_frame.shape == (h // 2, w, 3)
        assert partition.right_frame.shape == (h // 2, w, 3)
        assert partition.left_roi == (0, 0, w, h // 2)
        assert partition.right_roi == (0, h // 2, w, h // 2)

        recombined = recombine_stereo_frame(partition.left_frame, partition.right_frame, StereoMode.STEREO_TB)
        assert recombined.shape == frame.shape
        np.testing.assert_array_equal(recombined, frame)

    def test_partition_mono_passthrough(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        partition = partition_stereo_frame(frame, StereoMode.MONO_2D)
        assert partition.right_frame is None
        np.testing.assert_array_equal(partition.left_frame, frame)

        recombined = recombine_stereo_frame(partition.left_frame, None, StereoMode.MONO_2D)
        np.testing.assert_array_equal(recombined, frame)


class TestEpipolarGeometricCoupling:
    """Verifies epipolar disparity lock and zero vertical drift."""

    def test_zero_vertical_epipolar_drift(self):
        """Verifies vertical epipolar drift delta_y < 0.05 px between coupled left and right face alignments."""
        coupler = EpipolarStereoCoupler(ema_alpha=0.70)

        # Simulate Left face at position (200, 300)
        left_kps = CANONICAL_5PTS.copy() + np.array([200.0, 300.0], dtype=np.float32)

        # Simulate Right face with 32.0 px horizontal disparity and significant vertical landmark jitter (e.g. 3.4 px)
        right_kps = CANONICAL_5PTS.copy() + np.array([200.0 - 32.0, 300.0 + 3.4], dtype=np.float32)
        # Add subpixel independent noise to individual landmarks
        rng = np.random.RandomState(42)
        left_kps += rng.normal(0, 0.8, size=left_kps.shape).astype(np.float32)
        right_kps += rng.normal(0, 0.8, size=right_kps.shape).astype(np.float32)

        # Before coupling: vertical drift is ~3-4 px
        raw_dy = np.max(np.abs(left_kps[:, 1] - right_kps[:, 1]))
        assert raw_dy > 1.5, f"Expected uncoupled vertical jitter, got {raw_dy:.3f}"

        # Execute Stage 4 / 9 Epipolar Coupling
        result: EpipolarCouplingResult = coupler.couple_alignment(left_kps, right_kps, track_id=1)

        # Verify strict epipolar lock: delta_y < 0.05 px across all landmarks
        vert_drift = result.vertical_drift_px
        assert vert_drift < 0.05, f"Epipolar vertical drift {vert_drift:.5f} exceeds strict 0.05 px tolerance!"

        # Pointwise verification
        for k in range(5):
            dy = abs(result.coupled_left_kps[k, 1] - result.coupled_right_kps[k, 1])
            assert dy < 0.05, f"Landmark {k} has delta_y {dy:.5f} >= 0.05 px"

    def test_roll_angle_synchronization(self):
        """Verifies in-plane roll angles are coupled to be identical across the stereo pair."""
        coupler = EpipolarStereoCoupler()

        # Left face rotated by +4.0 degrees
        angle_l = math.radians(4.0)
        c_l, s_l = math.cos(angle_l), math.sin(angle_l)
        rot_l = np.array([[c_l, -s_l], [s_l, c_l]], dtype=np.float32)
        left_kps = CANONICAL_5PTS @ rot_l.T + np.array([150.0, 200.0], dtype=np.float32)

        # Right face rotated by -2.5 degrees (6.5 deg binocular roll disparity)
        angle_r = math.radians(-2.5)
        c_r, s_r = math.cos(angle_r), math.sin(angle_r)
        rot_r = np.array([[c_r, -s_r], [s_r, c_r]], dtype=np.float32)
        right_kps = CANONICAL_5PTS @ rot_r.T + np.array([120.0, 200.0], dtype=np.float32)

        res = coupler.couple_alignment(left_kps, right_kps)

        roll_l_coupled = coupler.compute_roll_angle(res.coupled_left_kps)
        roll_r_coupled = coupler.compute_roll_angle(res.coupled_right_kps)

        assert abs(roll_l_coupled - roll_r_coupled) < 1e-4, "Coupled roll angles must be identical"
        # Coupled roll should be average of rotations added to base template roll
        base_roll = coupler.compute_roll_angle(CANONICAL_5PTS)
        expected_roll = base_roll + 0.5 * (angle_l + angle_r)
        assert abs(roll_l_coupled - expected_roll) < 1e-3
        assert res.vertical_drift_px < 0.05

    def test_horizontal_disparity_ema_filtering(self):
        """Verifies horizontal disparity EMA smoothing suppresses depth jitter across multiple frames."""
        coupler = EpipolarStereoCoupler(ema_alpha=0.60)
        track_id = 99

        true_disparity = 25.0
        rng = np.random.RandomState(1337)
        num_frames = 20

        raw_disparities = []
        smoothed_disparities = []
        max_drifts = []

        for f in range(num_frames):
            # Inject noisy horizontal disparity jitter (+- 3.0 px) and vertical jitter
            jitter_dx = rng.normal(0, 2.5)
            jitter_dy = rng.normal(0, 2.0)

            cur_raw_d = true_disparity + jitter_dx
            l_kps = CANONICAL_5PTS + np.array([250.0, 180.0], dtype=np.float32)
            r_kps = CANONICAL_5PTS + np.array([250.0 - cur_raw_d, 180.0 + jitter_dy], dtype=np.float32)

            res = coupler.couple_alignment(l_kps, r_kps, track_id=track_id)
            raw_disparities.append(res.raw_disparity_px)
            smoothed_disparities.append(res.smoothed_disparity_px)
            max_drifts.append(res.vertical_drift_px)

        # Zero vertical drift on all frames
        for f, d in enumerate(max_drifts):
            assert d < 0.05, f"Frame {f} vertical drift {d:.5f} >= 0.05 px"

        # Verify EMA reduced variance
        raw_std = float(np.std(raw_disparities))
        smooth_std = float(np.std(smoothed_disparities))
        assert smooth_std < raw_std, f"Smoothed disparity std {smooth_std:.3f} should be less than raw std {raw_std:.3f}"

    def test_pair_stereo_faces_matching(self):
        """Tests pairing of multiple faces across Left and Right stereo views."""
        coupler = EpipolarStereoCoupler()

        # Create two distinct people in Left view
        person_a_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        person_b_emb = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        class DummyFace:
            def __init__(self, name, x, y, emb):
                self.name = name
                self.kps = CANONICAL_5PTS + np.array([x, y], dtype=np.float32)
                self.embedding = emb

        left_faces = [
            DummyFace("PersonA_Left", 100.0, 200.0, person_a_emb),
            DummyFace("PersonB_Left", 400.0, 220.0, person_b_emb),
        ]

        # Right faces with matching embeddings and appropriate horizontal disparity
        right_faces = [
            DummyFace("PersonB_Right", 400.0 - 20.0, 222.0, person_b_emb),
            DummyFace("PersonA_Right", 100.0 - 22.0, 198.0, person_a_emb),
        ]

        pairs, un_l, un_r = coupler.pair_stereo_faces(left_faces, right_faces, frame_height=1080)
        assert len(pairs) == 2
        assert len(un_l) == 0
        assert len(un_r) == 0

        # Person A paired with Person A, Person B paired with Person B
        names = [(p[0].name, p[1].name) for p in pairs]
        assert ("PersonA_Left", "PersonA_Right") in names
        assert ("PersonB_Left", "PersonB_Right") in names


class TestStage6And12Synchronization:
    """Verifies synchronization of latent seeds, normalization statistics, and Stage 12 specular masks."""

    def test_synchronize_latent_noise_seeds(self):
        crop_l = np.ones((512, 512, 3), dtype=np.uint8) * 120
        crop_r = np.ones((512, 512, 3), dtype=np.uint8) * 125

        _, _, stats = synchronize_stereo_crops(crop_l, crop_r, seed=12345)
        assert stats["seed"] == 12345

    def test_synchronize_normalization_statistics(self):
        # Left eye slightly darker (mean 100), Right eye slightly brighter (mean 140)
        crop_l = np.full((128, 128, 3), 100, dtype=np.uint8)
        crop_r = np.full((128, 128, 3), 140, dtype=np.uint8)

        synced_l, synced_r, stats = synchronize_stereo_crops(crop_l, crop_r, sync_norm=True)
        mean_l = float(np.mean(synced_l))
        mean_r = float(np.mean(synced_r))

        # Normalized crops should converge toward joint mean (~120)
        assert abs(mean_l - mean_r) < 15.0  # discrepancy reduced from 40 down to < 15

    def test_synchronize_specular_highlight_masks(self):
        # Create corneal specular catchlight on Left eye
        spec_l = np.zeros((128, 128), dtype=np.float32)
        spec_l[50:56, 50:56] = 1.0  # left corneal reflection

        # Right eye has weaker catchlight (sensor angle difference)
        spec_r = np.zeros((128, 128), dtype=np.float32)
        spec_r[50:56, 50:56] = 0.3

        _, _, stats = synchronize_stereo_crops(
            np.zeros((128, 128, 3), dtype=np.uint8),
            np.zeros((128, 128, 3), dtype=np.uint8),
            specular_mask_L=spec_l,
            specular_mask_R=spec_r,
        )

        synced_mask = stats["specular_mask"]
        assert synced_mask is not None
        # Should be harmonized mean catchlight ~0.65
        assert np.isclose(synced_mask[52, 52], 0.65, atol=1e-3)


class TestGeometricStabilizerIntegration:
    """Verifies integration with Stage 4 get_stereo_coupler()."""

    def test_global_coupler_accessor(self):
        coupler1 = get_stereo_coupler()
        coupler2 = get_stereo_coupler()
        assert coupler1 is coupler2
        assert isinstance(coupler1, EpipolarStereoCoupler)
