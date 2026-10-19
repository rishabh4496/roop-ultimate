"""Stage 17: occlusion-visibility-weighted alignment (pure numpy).

Every correction test has its control: the same inputs through the UNWEIGHTED
solve must be visibly wrong, or the test proves nothing about the fix.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.visibility_alignment import (KinematicClamp, Observation, apply,
                                       compose_similarity, consensus_transform,
                                       decompose, reconstruct_keypoints,
                                       sample_probability, solve_alignment,
                                       visibility_weights, weighted_similarity)

TEMPLATE = np.array([[38.3, 51.7], [73.5, 51.5], [56.0, 71.7],
                     [41.5, 92.4], [70.7, 92.2]], np.float32)


def _face(cx=300.0, cy=240.0, s=1.0):
    """Five keypoints and a 106-point layout around them, frame pixels."""
    k = (TEMPLATE - TEMPLATE.mean(0)) * 3.0 * s + [cx, cy]
    rng = np.random.default_rng(7)
    lm = np.concatenate([k, (rng.uniform(-1, 1, (101, 2)) * [70, 90]) * s + [cx, cy]])
    return k.astype(np.float32), lm.astype(np.float32)


def _move(pts, M):
    return apply(M, pts)


class WeightFormulaTest(unittest.TestCase):

    def test_formula_and_clamp(self):
        w = visibility_weights(np.array([0.0, 0.2, 0.5, 0.64, 0.9, 1.0]))
        np.testing.assert_allclose(w, [1.0, 0.7, 0.25, 0.05 + 0.0, 0.05, 0.05], atol=1e-6)

    def test_never_zero(self):
        self.assertGreaterEqual(float(visibility_weights(np.array([1.0]))[0]), 0.05)


class SampleTest(unittest.TestCase):

    def test_bilinear_and_off_map(self):
        p = np.zeros((10, 10), np.float32)
        p[4:6, 4:6] = 1.0
        v = sample_probability(p, np.array([[4.5, 4.5], [0.0, 0.0], [-3.0, 5.0], [50.0, 5.0]]))
        self.assertAlmostEqual(float(v[0]), 1.0, places=5)
        self.assertEqual(float(v[1]), 0.0)
        self.assertEqual(float(v[2]), 0.0)        # off the map reads 0, not the edge value
        self.assertEqual(float(v[3]), 0.0)


class WeightedSimilarityTest(unittest.TestCase):

    def test_exact_recovery(self):
        k, _ = _face()
        M = compose_similarity(0.31, 13.0, np.array([5.0, -9.0]))
        got = weighted_similarity(k, _move(k, M))
        np.testing.assert_allclose(got, M, atol=1e-3)

    def test_equal_weights_equal_unweighted(self):
        k, _ = _face()
        d = _move(k, compose_similarity(0.3, 5.0, np.array([1.0, 2.0]))) + np.random.default_rng(1).normal(0, .5, (5, 2))
        np.testing.assert_allclose(weighted_similarity(k, d, np.ones(5)),
                                   weighted_similarity(k, d), atol=1e-5)

    def test_a_wrong_point_wrecks_the_unweighted_fit_not_the_weighted_one(self):
        k, _ = _face()
        M = compose_similarity(0.3, 0.0, np.array([0.0, 0.0]))
        d = _move(k, M)
        bad = d.copy()
        bad[0] += [25.0, 18.0]                      # an eye keypoint dragged onto an occluder
        w = np.array([0.05, 1, 1, 1, 1], np.float32)
        err_unweighted = np.abs(apply(weighted_similarity(k, bad), k) - d).max()
        err_weighted = np.abs(apply(weighted_similarity(k, bad, w), k) - d).max()
        self.assertGreater(err_unweighted, 3.0, 'control: one bad point must hurt')
        self.assertLess(err_weighted, 0.3 * err_unweighted)

    def test_degenerate_inputs(self):
        self.assertIsNone(weighted_similarity(np.zeros((1, 2)), np.zeros((1, 2))))
        self.assertIsNone(weighted_similarity(np.ones((5, 2)), np.ones((5, 2))))
        k, _ = _face()
        self.assertIsNone(weighted_similarity(k, k, np.zeros(5)))


class ConsensusTest(unittest.TestCase):

    def test_points_on_the_occluder_are_voted_out(self):
        _, lm = _face()
        M = compose_similarity(1.02, 3.0, np.array([6.0, -4.0]))
        cur = _move(lm, M).copy()
        cur[:30] += [40.0, 25.0]                    # 30 of 106 landmarks stuck to a moving occluder
        T = consensus_transform(lm, cur, np.ones(len(lm)))
        self.assertLess(np.abs(apply(T, lm[30:]) - _move(lm, M)[30:]).max(), 0.5)
        plain = weighted_similarity(lm, cur)
        self.assertGreater(np.abs(apply(plain, lm[30:]) - _move(lm, M)[30:]).max(), 3.0,
                           'control: without the IRLS the outliers drag the fit')


class ReconstructionTest(unittest.TestCase):

    def setUp(self):
        self.k0, self.lm0 = _face()
        self.M = compose_similarity(1.03, 4.0, np.array([7.0, -5.0]))
        self.k_true = _move(self.k0, self.M)
        self.lm_true = _move(self.lm0, self.M)

    def _occluded_eye(self):
        k = self.k_true.copy()
        k[1] += [-22.0, 20.0]                       # right eye keypoint on the occluder
        w = np.ones(5, np.float32)
        w[1] = 0.05
        lm = self.lm_true.copy()
        wl = np.ones(len(lm), np.float32)
        near = np.linalg.norm(self.lm_true - self.k_true[1], axis=1) < 25.0
        lm[near] += [-22.0, 20.0]
        wl[near] = 0.05
        wl[1] = 0.05
        return k, w, lm, wl

    def test_rebuilds_the_occluded_keypoint_from_the_prior(self):
        k, w, lm, wl = self._occluded_eye()
        got, w2, n = reconstruct_keypoints(k, w, lm, wl, Observation(self.k0, self.lm0))
        self.assertEqual(n, 1)
        self.assertLess(float(np.linalg.norm(got[1] - self.k_true[1])), 1.0)
        self.assertAlmostEqual(float(w2[1]), 0.6, places=5)
        np.testing.assert_allclose(got[[0, 2, 3, 4]], k[[0, 2, 3, 4]])   # visible ones untouched

    def test_the_repaired_alignment_beats_the_raw_one(self):
        k, w, lm, wl = self._occluded_eye()
        truth = solve_alignment(self.k_true, np.ones(5), TEMPLATE)
        raw = solve_alignment(k, np.ones(5), TEMPLATE)
        fixed_k, fixed_w, _ = reconstruct_keypoints(k, w, lm, wl, Observation(self.k0, self.lm0))
        fixed = solve_alignment(fixed_k, fixed_w, TEMPLATE)
        probe = self.lm_true
        e_raw = np.linalg.norm(apply(raw, probe) - apply(truth, probe), axis=1).mean()
        e_fix = np.linalg.norm(apply(fixed, probe) - apply(truth, probe), axis=1).mean()
        self.assertGreater(e_raw, 2.0, 'control: the occluded keypoint must hurt the raw solve')
        self.assertLess(e_fix, 0.25 * e_raw)

    def test_nothing_occluded_changes_nothing(self):
        got, w2, n = reconstruct_keypoints(self.k_true, np.ones(5), self.lm_true,
                                           np.ones(len(self.lm_true)), Observation(self.k0, self.lm0))
        self.assertEqual(n, 0)
        np.testing.assert_array_equal(got, self.k_true)

    def test_no_prior_means_no_rebuild(self):
        k, w, lm, wl = self._occluded_eye()
        got, _, n = reconstruct_keypoints(k, w, lm, wl, None)
        self.assertEqual(n, 0)
        np.testing.assert_array_equal(got, k)

    def test_a_mismatched_dense_layout_is_refused(self):
        k, w, lm, wl = self._occluded_eye()
        got, _, n = reconstruct_keypoints(k, w, lm[:68], wl[:68], Observation(self.k0, self.lm0))
        self.assertEqual(n, 0)

    def test_too_little_visible_face_is_refused(self):
        k, w, lm, wl = self._occluded_eye()
        wl[:] = 0.05
        wl[:5] = 1.0                                # only 5 visible dense points
        _, _, n = reconstruct_keypoints(k, w, lm, wl, Observation(self.k0, self.lm0))
        self.assertEqual(n, 0)


class KinematicClampTest(unittest.TestCase):

    def _m(self, s=1.0, deg=0.0, tx=0.0, ty=0.0):
        return compose_similarity(s, deg, np.array([tx, ty]))

    def test_unoccluded_frames_pass_through_exactly(self):
        c = KinematicClamp()
        c.update(self._m(), 0.0)
        M = self._m(1.05, 6.0, 9.0, 4.0)
        np.testing.assert_allclose(c.update(M, 0.1), M, atol=1e-5)

    def test_a_jump_while_occluded_is_bounded(self):
        c = KinematicClamp(max_log_scale=0.02, max_deg=1.5, max_trans_frac=0.01, crop=256.0)
        c.update(self._m(), 0.0)
        c.update(self._m(), 0.0)
        out = c.update(self._m(1.20, 12.0, 40.0, -30.0), 0.7)       # detector went wrong
        s, deg, t = decompose(out)
        self.assertLessEqual(abs(np.log(s)), 0.02 + 1e-6)
        self.assertLessEqual(abs(deg), 1.5 + 1e-6)
        self.assertLessEqual(float(np.linalg.norm(t)), 2.56 + 1e-5)

    def test_legitimate_slow_motion_is_not_clamped(self):
        c = KinematicClamp()
        for k in range(10):
            M = self._m(1.0 + 0.002 * k, 0.2 * k, 0.5 * k, 0.0)
            out = c.update(M, 0.7)
            np.testing.assert_allclose(out, M, atol=0.05)

    def test_reset_forgets_the_state(self):
        c = KinematicClamp()
        c.update(self._m(), 0.0)
        c.reset()
        M = self._m(1.3, 20.0, 50.0, 50.0)
        np.testing.assert_allclose(c.update(M, 0.9), M, atol=1e-5)   # first frame after reset passes


if __name__ == '__main__':
    unittest.main()
