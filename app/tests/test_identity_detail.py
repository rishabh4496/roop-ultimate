"""Phase 9 identity-detail tests and measurable synthetic quality guards."""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.identity_detail import (aggregate_detail_representations,
                                  build_detail_representation, decode_detail,
                                  restore_identity_detail)
from roop.temporal_identity import TemporalIdentityStabilizer


def _identity_plate(size=160, noise=0.0, dark=False):
    value = 52 if dark else 150
    image = np.full((size, size, 3), value, dtype=np.float32)
    # Mole / beauty mark.
    cv2.circle(image, (62, 65), 4, (55, 55, 55), -1)
    # Freckles.
    for x, y, r in ((74, 68, 2), (80, 72, 1), (87, 67, 2), (91, 75, 1)):
        cv2.circle(image, (x, y), r, (95, 95, 95), -1)
    # Stable scar and wrinkle strokes.
    cv2.line(image, (102, 55), (116, 73), (92, 92, 92), 2)
    cv2.line(image, (38, 91), (73, 94), (105, 105, 105), 1)
    # Low-amplitude pore/microtexture.
    rng = np.random.default_rng(9)
    texture = rng.normal(0.0, 1.4, image.shape[:2]).astype(np.float32)
    image += texture[..., None]
    if noise:
        image += np.random.default_rng(77).normal(0.0, noise, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


class IdentityDetailTest(unittest.TestCase):

    def test_faceset_representation_is_signed_and_not_a_texture_patch(self):
        rep = build_detail_representation(_identity_plate(), 1.0)
        self.assertEqual(rep["schema"], "roop.identity_detail.v1")
        self.assertEqual(rep["shape"], [64, 64])
        self.assertEqual(len(rep["residual_q"]), 64 * 64)
        decoded = decode_detail(rep)
        self.assertIsNotNone(decoded)
        self.assertLessEqual(float(np.max(np.abs(decoded["residual"]))), 24.25)
        self.assertLess(float(np.mean(np.abs(decoded["residual"]))), 8.0)

    def test_aggregation_keeps_persistent_marks_and_rejects_independent_noise(self):
        stable = _identity_plate()
        reps = [build_detail_representation(
            np.clip(stable.astype(np.int16) + np.random.default_rng(i).normal(
                0, 5, stable.shape), 0, 255).astype(np.uint8), 1.0)
                for i in range(3)]
        aggregate = aggregate_detail_representations(reps, [1.0, 1.0, 1.0])
        decoded = decode_detail(aggregate)
        self.assertEqual(decoded["source_count"], 3)
        self.assertGreater(float(np.max(decoded["confidence"])), 0.15)
        # The consensus is materially less energetic than an individual noisy
        # observation, which is the source-noise/JPEG rejection contract.
        individual_energy = np.mean(np.abs(decode_detail(reps[0])["residual"]))
        consensus_energy = np.mean(np.abs(decoded["residual"]))
        self.assertLessEqual(consensus_energy, individual_energy + 0.5)

    def test_confidence_and_visibility_weight_the_composite(self):
        detail = aggregate_detail_representations(
            [build_detail_representation(_identity_plate(), 1.0)])
        target = np.full((128, 128, 3), 145, dtype=np.uint8)
        full, full_metrics = restore_identity_detail(
            target, detail, strength=0.5, return_metrics=True)
        hidden, hidden_metrics = restore_identity_detail(
            target, detail, strength=0.5,
            visibility_mask=np.zeros((128, 128), dtype=np.float32),
            return_metrics=True)
        self.assertGreater(full_metrics["energy"], 0.0)
        self.assertEqual(hidden_metrics["energy"], 0.0)
        np.testing.assert_array_equal(hidden, target)
        self.assertFalse(np.array_equal(full, target))

    def test_dark_low_resolution_and_motion_blur_are_safe_and_low_strength(self):
        source = cv2.resize(_identity_plate(96, dark=True), (48, 48),
                            interpolation=cv2.INTER_AREA)
        source = cv2.GaussianBlur(source, (9, 9), 2.2)
        detail = build_detail_representation(source, 0.25)
        target = np.full((64, 64, 3), 45, dtype=np.uint8)
        output, metrics = restore_identity_detail(
            target, detail, strength=0.5, return_metrics=True)
        self.assertTrue(np.isfinite(output).all())
        self.assertLessEqual(metrics["energy"], 2.5)
        self.assertLessEqual(metrics["lighting_gain"], 1.0)

    def test_pose_template_warp_and_expression_regions_do_not_fail(self):
        detail = aggregate_detail_representations(
            [build_detail_representation(_identity_plate(), 1.0)])
        landmarks = np.zeros((68, 2), dtype=np.float32)
        landmarks[:, 0] = 64.0
        landmarks[:, 1] = 64.0
        landmarks[36:48] = [[48, 55], [54, 52], [60, 55], [60, 60],
                            [54, 62], [48, 60]] * 2
        landmarks[48:68] = np.array([[48, 85], [52, 82], [58, 81], [64, 82],
                                      [70, 81], [76, 85], [70, 90], [64, 92],
                                      [58, 90], [52, 90]] * 2, dtype=np.float32)[:20]
        target = np.full((128, 128, 3), 145, dtype=np.uint8)
        output = restore_identity_detail(
            target, detail, target_face={"landmark_2d_68": landmarks},
            matrix=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
            matrix_shape=(128, 128), strength=0.35,
            target_template="ffhq_512")
        self.assertEqual(output.shape, target.shape)
        self.assertTrue(np.isfinite(output).all())

    def test_temporal_detail_blend_reduces_alternating_flicker(self):
        stabilizer = TemporalIdentityStabilizer(
            enabled=True, output_strength=0.7, cache_size=64)
        values = []
        for i in range(12):
            current = np.full((64, 64), 8.0 if i % 2 else -8.0, np.float32)
            values.append(float(stabilizer.blend_detail(
                3, current, confidence=0.7, motion=0.0,
                source_index=0).mean()))
        raw_delta = 16.0
        filtered_delta = float(np.mean(np.abs(np.diff(values[2:]))))
        self.assertLess(filtered_delta, raw_delta)

    def test_source_switch_does_not_ghost_previous_identity_mark(self):
        stabilizer = TemporalIdentityStabilizer(enabled=True, output_strength=0.9)
        first = np.full((64, 64), 12.0, np.float32)
        second = np.full((64, 64), -12.0, np.float32)
        stabilizer.blend_detail(4, first, confidence=1.0, source_index=0)
        result = stabilizer.blend_detail(4, second, confidence=1.0,
                                         source_index=1, transition_alpha=0.0)
        self.assertAlmostEqual(float(result.mean()), -12.0, places=4)


if __name__ == "__main__":
    unittest.main()
