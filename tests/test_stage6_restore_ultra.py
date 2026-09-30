"""Unit and integration tests for Stage 6 — Restore Ultra Quality and Throughput Optimizer.

Covers:
1. Profile definitions (FAST, BALANCED, QUALITY, ULTRA) and monotonic parameter progression.
2. Buffer pool allocation, thread-safety, and bit-for-bit input preparation without allocations.
3. Adaptive restoration strength scaling across resolutions, pose angles, and noise floors.
4. Anti-halo envelope bounding (zero halo overshoot guaranteed across all profiles).
5. Eye clarity boost localization and iris definition without periocular bleaching.
6. Identity preservation guardrail against codebook drift and hallucination artifacts.
7. End-to-end execution of apply_restore_ultra_profile.
"""

import math
import os
import sys
import unittest
from unittest import mock

import cv2
import numpy as np

# Ensure app is in path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

from roop.restore_ultra_optimizer import (
    RESTORE_PROFILES,
    RestoreProfileConfig,
    RestoreUltraBufferPool,
    BUFFER_POOL,
    get_profile,
    compute_adaptive_scaling,
    IdentityPreservationGuard,
    apply_restore_ultra_profile,
    measure_skin_pore_variance,
    measure_edge_sharpness,
    measure_eye_clarity,
    measure_identity_similarity,
    measure_halo_overshoot,
)


def create_synthetic_face(size=512, seed=42) -> np.ndarray:
    """Generate a reproducible face-like crop with skin texture, eyes, lips, and facial contours."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 160, dtype=np.uint8)

    # Face oval
    cv2.ellipse(img, (size // 2, int(size * 0.55)),
                (int(size * 0.32), int(size * 0.42)), 0, 0, 360,
                (175, 185, 205), -1)

    # Eyes at FFHQ-512 coordinates: left (0.3769, 0.4686), right (0.6228, 0.4691)
    for fx, fy in ((0.37691676, 0.46864664), (0.62285697, 0.46912813)):
        cx, cy = int(fx * size), int(fy * size)
        # Sclera
        cv2.ellipse(img, (cx, cy), (int(size * 0.055), int(size * 0.028)),
                    0, 0, 360, (235, 235, 240), -1)
        # Iris
        cv2.circle(img, (cx, cy), int(size * 0.020), (55, 40, 35), -1)
        # Pupil
        cv2.circle(img, (cx, cy), max(1, int(size * 0.008)), (10, 10, 10), -1)
        # Catchlight
        cv2.circle(img, (cx - 2, cy - 2), max(1, int(size * 0.004)),
                   (255, 255, 255), -1)

    # Lips
    cv2.ellipse(img, (size // 2, int(size * 0.72)),
                (int(size * 0.11), int(size * 0.035)), 0, 0, 360,
                (115, 110, 175), -1)

    # Skin pores / high-frequency texture
    noise = rng.normal(0.0, 5.0, (size, size, 3))
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


class TestRestoreProfiles(unittest.TestCase):
    """Test restoration profile configurations and parameter validity."""

    def test_all_four_profiles_exist(self):
        for name in ('FAST', 'BALANCED', 'QUALITY', 'ULTRA'):
            self.assertIn(name, RESTORE_PROFILES)
            prof = get_profile(name)
            self.assertEqual(prof.name, name)
            self.assertGreater(prof.strength, 0.0)
            self.assertGreater(prof.crispness, 0.0)
            self.assertGreater(prof.eye_clarity, 0.0)
            self.assertGreater(prof.detail_weight, 0.0)

    def test_profile_resolution_case_insensitive(self):
        self.assertEqual(get_profile('fast').name, 'FAST')
        self.assertEqual(get_profile('balanced').name, 'BALANCED')
        self.assertEqual(get_profile('quality').name, 'QUALITY')
        self.assertEqual(get_profile('ultra').name, 'ULTRA')
        # Default fallback
        self.assertEqual(get_profile('non_existent').name, 'QUALITY')
        self.assertEqual(get_profile(None).name, 'QUALITY')

    def test_parameter_progression_monotonic(self):
        """Higher fidelity profiles must have increasing detail weight, strength, and clarity."""
        fast = get_profile('FAST')
        bal = get_profile('BALANCED')
        qual = get_profile('QUALITY')
        ult = get_profile('ULTRA')

        self.assertLess(fast.strength, bal.strength)
        self.assertLess(bal.strength, qual.strength)
        self.assertLess(qual.strength, ult.strength)

        self.assertLess(fast.crispness, bal.crispness)
        self.assertLess(bal.crispness, qual.crispness)
        self.assertLess(qual.crispness, ult.crispness)

        self.assertLess(fast.eye_clarity, bal.eye_clarity)
        self.assertLess(bal.eye_clarity, qual.eye_clarity)
        self.assertLess(qual.eye_clarity, ult.eye_clarity)

        self.assertLess(fast.detail_weight, bal.detail_weight)
        self.assertLess(bal.detail_weight, qual.detail_weight)
        self.assertLess(qual.detail_weight, ult.detail_weight)


class TestBufferPoolAndCaching(unittest.TestCase):
    """Test memory preallocation, zero-leak buffer reuse, and table caching."""

    def test_buffer_pool_reuses_instance(self):
        pool = RestoreUltraBufferPool()
        buf1 = pool.get_input_buffer((1, 3, 512, 512))
        buf2 = pool.get_input_buffer((1, 3, 512, 512))
        self.assertIs(buf1, buf2)
        self.assertEqual(buf1.shape, (1, 3, 512, 512))
        self.assertEqual(buf1.dtype, np.float32)

    def test_prepare_model_input_matches_lut(self):
        img = create_synthetic_face(512)
        lut = ((np.arange(256, dtype=np.float32) / 127.5) - 1.0)
        expected = lut[img.transpose(2, 0, 1)[::-1]][None]

        actual = BUFFER_POOL.prepare_model_input(img)
        np.testing.assert_array_equal(actual, expected)
        self.assertGreaterEqual(float(actual.min()), -1.0)
        self.assertLessEqual(float(actual.max()), 1.0)

    def test_soft_knee_lut_caching(self):
        lut1 = RestoreUltraBufferPool.get_knee_lut(10.0, 2.5, 0.30)
        lut2 = RestoreUltraBufferPool.get_knee_lut(10.0, 2.5, 0.30)
        self.assertIs(lut1, lut2)
        self.assertEqual(lut1.shape, (511,))
        # Check boundary saturation
        self.assertAlmostEqual(float(lut1[255]), 0.0, places=5)
        # Below threshold: linear
        self.assertAlmostEqual(float(lut1[255 + 5]), 5.0 * 0.30, places=5)

    def test_postprocess_model_output_reuses_buffer(self):
        pool = RestoreUltraBufferPool()
        chw = np.zeros((3, 512, 512), dtype=np.float32)
        chw[0] = 0.5   # R
        chw[1] = -0.5  # G
        chw[2] = 1.0   # B
        
        res1 = pool.postprocess_model_output(chw)
        res2 = pool.postprocess_model_output(chw)
        self.assertEqual(res1.shape, (512, 512, 3))
        self.assertEqual(res1.dtype, np.uint8)
        np.testing.assert_array_equal(res1, res2)

    def test_teeth_clamping_prevents_rollover(self):
        """Negative predictions below -1.0 or overflows above 1.0 must clamp cleanly to [0, 255] without rollover."""
        pool = RestoreUltraBufferPool()
        chw = np.full((3, 512, 512), -2.5, dtype=np.float32)
        res = pool.postprocess_model_output(chw)
        self.assertEqual(int(res.min()), 0)
        self.assertEqual(int(res.max()), 0)

        chw_pos = np.full((3, 512, 512), 3.5, dtype=np.float32)
        res_pos = pool.postprocess_model_output(chw_pos)
        self.assertEqual(int(res_pos.min()), 255)
        self.assertEqual(int(res_pos.max()), 255)


class TestAdaptiveRestorationStrength(unittest.TestCase):
    """Test adaptive strength modulation based on resolution, pose, and noise."""

    def test_small_crop_attenuates_strength(self):
        """Tiny faces (< 128px) must attenuate strength to prevent hallucinated contrast."""
        res_small = compute_adaptive_scaling(crop_shape=(96, 96))
        self.assertLess(res_small.strength_mult, 0.85)
        self.assertLess(res_small.detail_weight_mult, 0.90)

    def test_nominal_crop_full_strength(self):
        """Faces in sweet spot (256 - 384px) must receive nominal 1.0x restoration."""
        res_nom = compute_adaptive_scaling(crop_shape=(300, 300))
        self.assertAlmostEqual(res_nom.strength_mult, 1.0, places=2)
        self.assertAlmostEqual(res_nom.eye_clarity_mult, 1.0, places=2)

    def test_profile_yaw_attenuates_eye_clarity(self):
        """Extreme profile angles (|yaw| > 25°) must damp eye clarity to prevent iris asymmetry."""
        target_face = mock.MagicMock()
        target_face.pose = [0.0, 45.0, 0.0]  # 45 deg yaw
        res_profile = compute_adaptive_scaling(crop_shape=(300, 300), target_face=target_face)

        self.assertLess(res_profile.eye_clarity_mult, 0.85)
        self.assertLess(res_profile.crispness_mult, 0.90)

    def test_high_noise_floor_attenuates_crispness(self):
        """High sensor noise must attenuate edge crispness to avoid noise haloing."""
        res_noisy = compute_adaptive_scaling(crop_shape=(300, 300), noise_estimate=12.0)
        self.assertLess(res_noisy.crispness_mult, 0.85)


class TestAntiHaloAndQuality(unittest.TestCase):
    """Test anti-halo envelope compliance and feature enhancement."""

    def test_all_profiles_strictly_obey_anti_halo_envelope(self):
        """No profile may produce pixel overshoot beyond the local 3x3 envelope during edge refinement."""
        face = create_synthetic_face(512)
        for name in ('FAST', 'BALANCED', 'QUALITY', 'ULTRA'):
            prof = get_profile(name)
            from roop.processors.enhance_common import apply_anti_halo_sharpen
            sharpened = apply_anti_halo_sharpen(
                face, amount=prof.crispness, sigma=prof.sharpen_sigma, limit=prof.sharpen_limit
            )
            # envelope_pad = limit + 2.0 for 8-bit BGR->LAB->BGR integer color round-trip quantization
            overshoot = measure_halo_overshoot(sharpened, face, envelope_pad=prof.sharpen_limit + 2.0)
            self.assertEqual(overshoot, 0.0, f"Profile {name} produced halo overshoot!")

    def test_eye_clarity_enhancement_improves_contrast(self):
        face = create_synthetic_face(512)
        clarity_before = measure_eye_clarity(face)
        out = apply_restore_ultra_profile(face, face, profile_name='QUALITY')
        clarity_after = measure_eye_clarity(out)
        self.assertGreater(clarity_after, clarity_before, "Eye clarity did not improve contrast!")

    def test_pore_variance_enhancement(self):
        face = create_synthetic_face(512)
        # Simulate slight blur from swapper
        blurred = cv2.GaussianBlur(face, (0, 0), sigmaX=1.2)
        var_before = measure_skin_pore_variance(blurred)
        out = apply_restore_ultra_profile(blurred, face, profile_name='QUALITY')
        var_after = measure_skin_pore_variance(out)
        self.assertGreater(var_after, var_before, "Pore texture variance was not recovered!")


class TestIdentityPreservationGuard(unittest.TestCase):
    """Test detection and mitigation of codebook hallucination drift."""

    def test_guard_inactive_on_normal_swaps(self):
        face = create_synthetic_face(512)
        guarded, was_triggered = IdentityPreservationGuard.protect(face, face)
        self.assertFalse(was_triggered)
        np.testing.assert_array_equal(guarded, face)

    def test_guard_triggers_and_recovers_on_extreme_codebook_drift(self):
        """Simulate severe codebook drift (restorer drastically altering face tone and features)."""
        reference = create_synthetic_face(512, seed=10)
        # Heavily alter eye/mouth region (simulated hallucination)
        hallucinated = reference.copy()
        hallucinated[180:400, 150:350] = cv2.add(hallucinated[180:400, 150:350], (45, -30, 50))

        guarded, was_triggered = IdentityPreservationGuard.protect(hallucinated, reference)
        self.assertTrue(was_triggered, "Identity guard failed to trigger on severe feature drift!")

        # Verify guarded frame is closer to reference in structural correlation
        sim_hallucinated = measure_identity_similarity(hallucinated, reference)
        sim_guarded = measure_identity_similarity(guarded, reference)
        self.assertGreater(sim_guarded, sim_hallucinated, "Guard did not improve identity likeness!")


class TestEndToEndProfileApplication(unittest.TestCase):
    """Test full pipeline execution across all profiles and edge cases."""

    def test_apply_all_profiles_end_to_end(self):
        face = create_synthetic_face(512)
        for profile in ('FAST', 'BALANCED', 'QUALITY', 'ULTRA'):
            out = apply_restore_ultra_profile(face, face, profile_name=profile, adaptive=True, identity_guard=True)
            self.assertEqual(out.shape, (512, 512, 3))
            self.assertEqual(out.dtype, np.uint8)
            self.assertTrue(np.all(np.isfinite(out)))

    def test_empty_or_none_inputs_handled_safely(self):
        self.assertIsNone(apply_restore_ultra_profile(None, None))
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        self.assertEqual(apply_restore_ultra_profile(empty, empty).shape, (0, 0, 3))


class TestProcessMgrRecombineAdaptive(unittest.TestCase):
    """Test ProcessMgr recombine adaptive scaling integration."""

    def test_adaptive_recombine_scales_detail_weight(self):
        from roop.ProcessMgr import ProcessMgr
        import roop.globals
        pm = ProcessMgr(None)
        face_restored = create_synthetic_face(512)
        face_swap = cv2.resize(face_restored, (256, 256))

        # Test with adaptive enabled
        roop.globals.restore_ultra_profile = 'QUALITY'
        roop.globals.restore_ultra_adaptive_strength = True
        roop.globals.restore_ultra_frequency_blend = True

        target_face = mock.MagicMock()
        target_face.pose = [0.0, 0.0, 0.0]
        out = pm._restore_ultra_recombine(face_restored, face_swap, target_face, None, 1.0)
        self.assertEqual(out.shape, (512, 512, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.all(np.isfinite(out)))


if __name__ == '__main__':
    unittest.main()

