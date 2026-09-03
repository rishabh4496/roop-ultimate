"""Contracts for roop/temporal_smoother.py.

These assert PROPERTIES rather than thresholds wherever possible, per the
project's test convention: a number pinned to today's tuning fails on the next
legitimate retune and teaches nothing.

The properties that actually matter here, and would each have shipped a silent
no-op if untested:

  * the contiguity guard DECLINES rather than blending across a frame gap;
  * a per-block clone tallies into the SHARED counters, so `summary_line()` on
    the parent still reports the truth in the parallel path;
  * the distance ramp reaches exactly 0 and exactly 1 (an un-normalised
    sigmoid leaves a step at both ends, which is worse than the Gaussian it
    replaces);
  * the rim illumination match is EXACTLY zero where alpha saturates, so the
    identity region is bit-unchanged;
  * beta is continuous across its threshold.
"""

import unittest

import cv2
import numpy as np

from roop.temporal_smoother import (AdaptiveLandmarkSmoother,
                                    HighFrequencyFlowStabilizer,
                                    adaptive_margin_px,
                                    boundary_illumination_match,
                                    soft_distance_matte)


def _kps(dx=0.0, dy=0.0):
    base = np.array([[30., 40.], [70., 40.], [50., 60.], [35., 80.], [65., 80.]],
                    dtype=np.float32)
    return base + np.array([dx, dy], dtype=np.float32)


class AdaptiveBeta(unittest.TestCase):
    def test_endpoints_reproduce_the_specified_constants(self):
        s = AdaptiveLandmarkSmoother()
        self.assertAlmostEqual(s.beta_for(0.0), 0.35, places=6)
        self.assertAlmostEqual(s.beta_for(10.0), 0.90, places=6)

    def test_beta_is_continuous_across_the_threshold(self):
        """A step at the threshold is itself a visible artefact: the filter's
        lag changes discontinuously on the crossing frame, and a face hovering
        at the threshold alternates regimes frame to frame."""
        s = AdaptiveLandmarkSmoother()
        thr = s.velocity_threshold
        eps = 1e-4
        self.assertLess(abs(s.beta_for(thr + eps) - s.beta_for(thr - eps)), 1e-2)

    def test_beta_is_monotonic_in_velocity(self):
        s = AdaptiveLandmarkSmoother()
        vs = np.linspace(0.0, 0.06, 40)
        betas = [s.beta_for(float(v)) for v in vs]
        self.assertTrue(all(b <= a + 1e-9 for a, b in zip(betas[1:], betas[:-1])))

    def test_velocity_is_normalised_by_face_scale(self):
        """The same MOTION at two face sizes must pick the same beta, or the
        filter behaves differently on a wide shot than on a close-up."""
        s = AdaptiveLandmarkSmoother()
        small, big = _kps(), _kps() * 4.0
        # Move each by 3% of its own extent.
        shift_s = s._extent(small) * 0.03
        shift_b = s._extent(big) * 0.03
        s.smooth(small, None, track_id='a', frame_index=0)
        _, _, beta_small = s.smooth(small + shift_s, None, track_id='a', frame_index=1)
        s.smooth(big, None, track_id='b', frame_index=0)
        _, _, beta_big = s.smooth(big + shift_b, None, track_id='b', frame_index=1)
        self.assertAlmostEqual(beta_small, beta_big, places=5)


class ContiguityGuard(unittest.TestCase):
    def test_a_frame_gap_declines_instead_of_blending(self):
        """Round-robin dispatch means no worker sees adjacent frames. A filter
        that blended anyway would mix a face from N frames away."""
        s = AdaptiveLandmarkSmoother()
        s.smooth(_kps(), None, track_id=1, frame_index=0)
        moved = _kps(dx=25.0)
        out, _, beta = s.smooth(moved, None, track_id=1, frame_index=5)
        np.testing.assert_allclose(out, moved)          # untouched
        self.assertEqual(beta, 1.0)
        self.assertEqual(s.stats()['skipped_noncontiguous'], 1)
        self.assertEqual(s.stats()['applied'], 0)

    def test_contiguous_frames_do_blend(self):
        s = AdaptiveLandmarkSmoother()
        s.smooth(_kps(), None, track_id=1, frame_index=0)
        moved = _kps(dx=1.0)
        out, _, _ = s.smooth(moved, None, track_id=1, frame_index=1)
        self.assertTrue(np.all(out[:, 0] < moved[:, 0]))   # pulled back toward prev
        self.assertEqual(s.stats()['applied'], 1)

    def test_no_track_id_declines_and_is_counted(self):
        s = AdaptiveLandmarkSmoother()
        out, _, _ = s.smooth(_kps(), None, track_id=None, frame_index=3)
        np.testing.assert_allclose(out, _kps())
        self.assertEqual(s.stats()['skipped_no_key'], 1)

    def test_separate_tracks_do_not_share_state(self):
        s = AdaptiveLandmarkSmoother()
        s.smooth(_kps(), None, track_id=1, frame_index=0)
        far = _kps(dx=400.0)
        out, _, _ = s.smooth(far, None, track_id=2, frame_index=1)
        np.testing.assert_allclose(out, far)   # seeded, not blended with track 1


class DenseLandmarksShareBeta(unittest.TestCase):
    def test_dense_uses_the_same_beta_as_the_five_points(self):
        """The 5 points drive the alignment, the dense set drives the paste
        hull. Different rates open a sliver of plate at the jaw during motion."""
        s = AdaptiveLandmarkSmoother()
        dense0 = np.zeros((106, 2), dtype=np.float32)
        s.smooth(_kps(), dense0, track_id=1, frame_index=0)
        dense1 = np.full((106, 2), 10.0, dtype=np.float32)
        _, out_dense, beta = s.smooth(_kps(dx=1.0), dense1,
                                      track_id=1, frame_index=1)
        expected = beta * dense1 + (1.0 - beta) * dense0
        np.testing.assert_allclose(out_dense, expected, rtol=1e-5)

    def test_a_dense_shape_change_does_not_crash_or_blend(self):
        s = AdaptiveLandmarkSmoother()
        s.smooth(_kps(), np.zeros((106, 2), np.float32), track_id=1, frame_index=0)
        d68 = np.full((68, 2), 5.0, dtype=np.float32)
        _, out, _ = s.smooth(_kps(dx=1.0), d68, track_id=1, frame_index=1)
        np.testing.assert_allclose(out, d68)     # passed through, not mixed


class BlockCloneReportsToTheParent(unittest.TestCase):
    """A clone that tallied into itself would be discarded with the block,
    leaving the parent's summary at zero for a filter that ran on every face --
    identical, from the log, to the filter never having run."""

    def test_landmark_clone_tallies_into_shared_counters(self):
        parent = AdaptiveLandmarkSmoother()
        clone = parent.clone_for_block()
        clone.smooth(_kps(), None, track_id=1, frame_index=0)
        clone.smooth(_kps(dx=1.0), None, track_id=1, frame_index=1)
        self.assertEqual(parent.stats()['applied'], 1)
        self.assertIsNotNone(parent.summary_line())

    def test_landmark_clone_has_independent_state(self):
        parent = AdaptiveLandmarkSmoother()
        parent.smooth(_kps(), None, track_id=1, frame_index=0)
        clone = parent.clone_for_block()
        far = _kps(dx=100.0)
        out, _, _ = clone.smooth(far, None, track_id=1, frame_index=1)
        np.testing.assert_allclose(out, far)     # clone seeded; parent's state unseen

    def test_hf_clone_tallies_into_shared_counters_and_rebuilds_tls(self):
        parent = HighFrequencyFlowStabilizer()
        clone = parent.clone_for_block()
        self.assertIsNot(clone._tls, parent._tls)   # cv2 DIS is not thread-safe
        crop = (np.random.RandomState(0).rand(64, 64, 3) * 255).astype(np.uint8)
        clone.stabilize(crop, track_id=1, frame_index=0)
        clone.stabilize(crop, track_id=1, frame_index=1)
        self.assertEqual(parent.stats()['applied'], 1)


class WarmupIsDerived(unittest.TestCase):
    def test_landmark_warmup_matches_the_slow_beta(self):
        from roop.one_euro import ema_warmup_frames
        s = AdaptiveLandmarkSmoother()
        self.assertEqual(s.warmup_frames(), ema_warmup_frames(s.beta_slow))

    def test_hf_warmup_matches_the_carried_fraction(self):
        from roop.one_euro import ema_warmup_frames
        h = HighFrequencyFlowStabilizer()
        self.assertEqual(h.warmup_frames(), ema_warmup_frames(1.0 - h.weight))

    def test_a_stronger_filter_needs_a_longer_warmup(self):
        self.assertGreater(AdaptiveLandmarkSmoother(beta_slow=0.05).warmup_frames(),
                           AdaptiveLandmarkSmoother(beta_slow=0.60).warmup_frames())


class SoftDistanceMatte(unittest.TestCase):
    def _disc(self, r=80, n=256):
        m = np.zeros((n, n), np.uint8)
        cv2.circle(m, (n // 2, n // 2), r, 255, -1)
        return m

    def test_endpoints_are_exact(self):
        """An un-normalised logistic lands at ~0.018 / ~0.982 on [0,1], leaving
        a step of 2% of alpha at BOTH ends of the ramp -- two new hard edges
        either side of the soft one."""
        a = soft_distance_matte(self._disc())
        self.assertEqual(float(a.max()), 1.0)
        self.assertEqual(float(a.min()), 0.0)

    def test_alpha_is_a_function_of_distance_alone(self):
        """The defining property, and the whole reason for the distance
        transform: two pixels at the same distance from the boundary get the
        same alpha no matter WHERE on the rim they sit. Asserted exactly rather
        than over a tolerance band -- a band's own width dominates the spread
        and would have been measuring the test, not the ramp."""
        m = self._disc(r=80)
        a = soft_distance_matte(m, margin_px=12.0)
        dist = cv2.distanceTransform((m > 127).astype(np.uint8), cv2.DIST_L2, 5)
        # Group by exact distance value; every group must be single-valued.
        rim = (dist > 0) & (dist < 12.0)
        keys = np.round(dist[rim], 6)
        vals = a[rim]
        order = np.argsort(keys, kind='stable')
        keys, vals = keys[order], vals[order]
        edges = np.flatnonzero(np.diff(keys)) + 1
        for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(keys)]):
            if hi - lo > 1:
                self.assertLess(float(np.ptp(vals[lo:hi])), 1e-6)

    def test_a_gaussian_ramp_is_not_a_function_of_distance(self):
        """The contrast that justifies the mode existing. A Gaussian of a
        binary region narrows at convex geometry, so pixels equidistant from
        the boundary land at DIFFERENT alphas -- one `face_mask_blend` setting,
        two visible seam widths in the same frame."""
        m = np.zeros((256, 256), np.uint8)
        cv2.circle(m, (90, 128), 60, 255, -1)          # blunt, low curvature
        cv2.circle(m, (190, 128), 9, 255, -1)          # sharp, high curvature
        blurred = cv2.GaussianBlur(m, (25, 25), 0).astype(np.float32) / 255.0
        dist = cv2.distanceTransform((m > 127).astype(np.uint8), cv2.DIST_L2, 5)
        band = np.abs(dist - 3.0) < 0.2
        self.assertTrue(band.sum() > 20)
        gaussian_spread = float(np.ptp(blurred[band]))
        distance_spread = float(np.ptp(soft_distance_matte(m, margin_px=12.0)[band]))
        self.assertGreater(gaussian_spread, 0.1)       # curvature-dependent
        self.assertLess(distance_spread, gaussian_spread / 4.0)

    def test_alpha_is_saturated_well_inside_and_zero_outside(self):
        a = soft_distance_matte(self._disc(r=80), margin_px=10.0)
        self.assertEqual(float(a[128, 128]), 1.0)     # centre
        self.assertEqual(float(a[0, 0]), 0.0)         # corner, outside the disc

    def test_a_wider_margin_produces_a_wider_ramp(self):
        narrow = soft_distance_matte(self._disc(), margin_px=4.0)
        wide = soft_distance_matte(self._disc(), margin_px=16.0)
        self.assertGreater(int(((wide > 0.02) & (wide < 0.98)).sum()),
                           int(((narrow > 0.02) & (narrow < 0.98)).sum()))

    def test_empty_matte_returns_empty_not_an_exception(self):
        a = soft_distance_matte(np.zeros((32, 32), np.uint8))
        self.assertEqual(float(a.max()), 0.0)

    def test_margin_tracks_face_scale_between_the_stated_bounds(self):
        self.assertAlmostEqual(adaptive_margin_px(1.0), 8.0, places=6)
        self.assertAlmostEqual(adaptive_margin_px(4096.0), 18.0, places=6)
        self.assertLess(adaptive_margin_px(200.0), adaptive_margin_px(400.0))


class BoundaryIllumination(unittest.TestCase):
    def _pair(self, n=96):
        paste = np.full((n, n, 3), 100, np.uint8)
        target = np.full((n, n, 3), 160, np.uint8)
        alpha = np.zeros((n, n), np.float32)
        cv2.circle(alpha, (n // 2, n // 2), n // 3, 1.0, -1)
        alpha = cv2.GaussianBlur(alpha, (0, 0), 5.0)
        return paste, target, alpha

    def test_identity_region_is_bit_unchanged(self):
        """Weighted by alpha*(1-alpha): exactly zero where alpha saturates, so
        the face interior cannot be graded toward the plate."""
        paste, target, alpha = self._pair()
        out = boundary_illumination_match(paste, target, alpha, strength=1.0)
        interior = alpha > 0.995
        self.assertTrue(interior.any())
        np.testing.assert_array_equal(out[interior], paste[interior])

    def test_zero_strength_is_a_bit_identical_no_op(self):
        paste, target, alpha = self._pair()
        out = boundary_illumination_match(paste, target, alpha, strength=0.0)
        np.testing.assert_array_equal(out, paste)

    def test_the_rim_moves_toward_the_plate(self):
        paste, target, alpha = self._pair()
        out = boundary_illumination_match(paste, target, alpha, strength=1.0)
        rim = (alpha > 0.35) & (alpha < 0.65)
        self.assertTrue(rim.any())
        self.assertGreater(float(out[rim].mean()), float(paste[rim].mean()))

    def test_mismatched_shapes_return_the_input_rather_than_raising(self):
        paste, target, alpha = self._pair()
        out = boundary_illumination_match(paste, target[:32, :32], alpha,
                                          strength=1.0)
        np.testing.assert_array_equal(out, paste)


class HighFrequencyCarry(unittest.TestCase):
    def _crop(self, seed=0, n=96):
        return (np.random.RandomState(seed).rand(n, n, 3) * 255).astype(np.uint8)

    def test_zero_weight_is_a_bit_identical_no_op(self):
        h = HighFrequencyFlowStabilizer(weight=0.0)
        crop = self._crop()
        h.stabilize(crop, track_id=1, frame_index=0)
        np.testing.assert_array_equal(
            h.stabilize(crop, track_id=1, frame_index=1), crop)

    def test_a_frame_gap_declines(self):
        h = HighFrequencyFlowStabilizer()
        crop = self._crop()
        h.stabilize(crop, track_id=1, frame_index=0)
        out = h.stabilize(crop, track_id=1, frame_index=7)
        np.testing.assert_array_equal(out, crop)
        self.assertEqual(h.stats()['skipped_noncontiguous'], 1)

    def test_repeated_noise_is_damped_toward_its_own_history(self):
        """Two independent noise fields over the same underlying image is the
        restorer's failure mode. The carry must reduce the frame-to-frame
        difference; passing through would leave it unchanged."""
        rs = np.random.RandomState(3)
        base = cv2.GaussianBlur(
            (rs.rand(96, 96, 3) * 255).astype(np.uint8), (0, 0), 6.0)

        def noisy(seed):
            n = np.random.RandomState(seed).randn(96, 96, 3) * 12.0
            return np.clip(base.astype(np.float32) + n, 0, 255).astype(np.uint8)

        a, b = noisy(1), noisy(2)
        raw = float(np.mean(np.abs(b.astype(np.float32) - a.astype(np.float32))))
        h = HighFrequencyFlowStabilizer(weight=0.5)
        out_a = h.stabilize(a, track_id=1, frame_index=0)
        out_b = h.stabilize(b, track_id=1, frame_index=1)
        damped = float(np.mean(np.abs(out_b.astype(np.float32)
                                      - out_a.astype(np.float32))))
        self.assertLess(damped, raw)
        self.assertEqual(h.stats()['applied'], 1)

    def test_a_shape_change_declines_rather_than_raising(self):
        h = HighFrequencyFlowStabilizer()
        h.stabilize(self._crop(n=96), track_id=1, frame_index=0)
        small = self._crop(n=64)
        np.testing.assert_array_equal(
            h.stabilize(small, track_id=1, frame_index=1), small)

    def test_an_unrelated_frame_trips_the_residual_guard(self):
        """When flow cannot explain the change the carry must be dropped, not
        smeared: a stale texture over new content is worse than no filter."""
        h = HighFrequencyFlowStabilizer()
        h.stabilize(np.zeros((96, 96, 3), np.uint8), track_id=1, frame_index=0)
        bright = np.full((96, 96, 3), 255, np.uint8)
        out = h.stabilize(bright, track_id=1, frame_index=1)
        np.testing.assert_array_equal(out, bright)
        self.assertEqual(h.stats()['reset_residual'], 1)

    def test_disabled_is_a_pass_through_that_records_nothing(self):
        h = HighFrequencyFlowStabilizer(enabled=False)
        crop = self._crop()
        np.testing.assert_array_equal(
            h.stabilize(crop, track_id=1, frame_index=0), crop)
        self.assertIsNone(h.summary_line())


class SummaryAlwaysDistinguishesRanFromDeclined(unittest.TestCase):
    """The swap audit counts INTENT, so it reads the same whether these filters
    ran on every face or on none. These lines are the only discriminator."""

    def test_all_declined_reports_zero_percent_applied(self):
        s = AdaptiveLandmarkSmoother()
        for i in (0, 4, 8, 12):
            s.smooth(_kps(), None, track_id=1, frame_index=i)
        line = s.summary_line()
        self.assertIn('applied 0/', line)
        self.assertIn('non-contiguous', line)

    def test_all_applied_reports_a_high_percentage(self):
        s = AdaptiveLandmarkSmoother()
        for i in range(12):
            s.smooth(_kps(dx=float(i)), None, track_id=1, frame_index=i)
        self.assertIn('applied 11/12', s.summary_line())


if __name__ == '__main__':
    unittest.main()
