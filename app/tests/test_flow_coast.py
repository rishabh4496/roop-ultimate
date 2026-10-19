"""Stage 16: Lucas-Kanade landmark propagation through a detector dropout.

Synthetic, deterministic images: a blurred-noise "head" texture that can be
moved by a known similarity transform, with an occluder pasted over part of it.
Ground truth is the transform itself, so the assertions are in pixels.

The negative tests carry the weight: a propagator that returns a confident
wrong answer paints a swapped face where no face is. Each must come back
`consistent=False` (the caller then falls back to geometry-only filling).
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.flow_coast import FlowCoaster, bridge_gap, face_from_flow   # noqa: E402

H, W = 480, 640
CENTRE = np.array([320.0, 240.0], np.float32)


def _texture(seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (H, W), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 2.2)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _landmarks(n=106, rx=70.0, ry=90.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # a ring plus an interior spread, like a real 106-point layout
    ring = np.stack([CENTRE[0] + rx * np.cos(t), CENTRE[1] + ry * np.sin(t)], 1)
    rng = np.random.default_rng(3)
    inner = CENTRE + rng.uniform(-0.6, 0.6, (n // 3, 2)) * [rx, ry]
    return np.concatenate([ring[:n - len(inner)], inner]).astype(np.float32)


def _similarity(dx, dy, deg=0.0, scale=1.0):
    a = np.deg2rad(deg)
    c, s = scale * np.cos(a), scale * np.sin(a)
    # rotate/scale about the face centre, then translate
    m = np.array([[c, -s, 0.0], [s, c, 0.0]], np.float32)
    m[:, 2] = CENTRE + [dx, dy] - m[:, :2] @ CENTRE
    return m


def _warp(img, m):
    return cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def _apply(m, pts):
    return (pts @ m[:, :2].T + m[:, 2]).astype(np.float32)


def _bbox(pts):
    return np.array([pts[:, 0].min(), pts[:, 1].min(),
                     pts[:, 0].max(), pts[:, 1].max()], np.float32)


def _coaster(img, lm):
    c = FlowCoaster()
    assert c.anchor(img, _bbox(lm), lm, lm[:5]), 'anchor must succeed on a textured face'
    return c


class FlowPropagationTest(unittest.TestCase):

    def setUp(self):
        self.img = _texture()
        self.lm = _landmarks()

    def test_translation_is_recovered_to_subpixel(self):
        c = _coaster(self.img, self.lm)
        m = _similarity(4.0, -3.0)
        r = c.step(_warp(self.img, m))
        self.assertTrue(r.consistent, r.reason)
        err = np.linalg.norm(r.landmarks - _apply(m, self.lm), axis=1)
        self.assertLess(float(err.max()), 0.8)

    def test_rotation_and_scale_are_recovered(self):
        c = _coaster(self.img, self.lm)
        m = _similarity(2.0, 1.0, deg=4.0, scale=1.03)
        r = c.step(_warp(self.img, m))
        self.assertTrue(r.consistent, r.reason)
        err = np.linalg.norm(r.landmarks - _apply(m, self.lm), axis=1)
        self.assertLess(float(err.max()), 1.2)

    def test_steps_chain_over_a_ten_frame_dropout(self):
        c = _coaster(self.img, self.lm)
        total = np.eye(2, 3, dtype=np.float32)
        last = None
        for k in range(1, 11):
            m = _similarity(2.0 * k, -1.0 * k, deg=0.5 * k)
            last = c.step(_warp(self.img, m))
            self.assertTrue(last.consistent, f'step {k}: {last.reason}')
            total = m
        err = np.linalg.norm(last.landmarks - _apply(total, self.lm), axis=1)
        self.assertLess(float(err.mean()), 2.0)

    def test_hidden_landmarks_follow_the_visible_majority(self):
        """Occluder over ~half the face: the covered landmarks still move with the head."""
        c = _coaster(self.img, self.lm)
        m = _similarity(5.0, 2.0)
        moved = _warp(self.img, m)
        occ = np.zeros((H, W), np.uint8)
        occ[:, :] = 0
        occ[150:400, 240:330] = 255                  # covers the left-centre of the head
        junk = _texture(seed=99)
        moved = np.where(occ > 0, junk, moved).astype(np.uint8)
        r = c.step(moved, occluder=occ)
        self.assertTrue(r.consistent, r.reason)
        truth = _apply(m, self.lm)
        hidden = (self.lm[:, 0] > 240) & (self.lm[:, 0] < 330) & \
                 (self.lm[:, 1] > 150) & (self.lm[:, 1] < 400)
        self.assertGreater(int(hidden.sum()), 5)
        err = np.linalg.norm(r.landmarks - truth, axis=1)
        self.assertLess(float(err[hidden].max()), 1.5,
                        'landmarks under the occluder must follow the head, not the occluder')

    def test_occluder_without_a_mask_is_still_survived_by_the_fit(self):
        """No occluder mask supplied: RANSAC alone must reject the junk points."""
        c = _coaster(self.img, self.lm)
        m = _similarity(4.0, 1.0)
        moved = _warp(self.img, m)
        moved[200:330, 250:330] = _texture(seed=7)[200:330, 250:330]
        r = c.step(moved)
        self.assertTrue(r.consistent, r.reason)
        err = np.linalg.norm(r.landmarks - _apply(m, self.lm), axis=1)
        self.assertLess(float(np.median(err)), 1.0)


class HighResolutionTest(unittest.TestCase):
    """A 4K face is ~4x the pixels across of the one LK is tuned for. The chain
    must be resolution-normalised, with error reported in FULL-resolution px."""

    K = 4

    def setUp(self):
        k = self.K
        self.img = cv2.resize(_texture(), None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        self.lm = _landmarks() * k
        self.centre = CENTRE * k

    def _m(self, dx, dy, deg=0.0, scale=1.0):
        a = np.deg2rad(deg)
        c, s = scale * np.cos(a), scale * np.sin(a)
        m = np.array([[c, -s, 0.0], [s, c, 0.0]], np.float32)
        m[:, 2] = self.centre + [dx, dy] - m[:, :2] @ self.centre
        return m

    def _w(self, m):
        return cv2.warpAffine(self.img, m, (self.img.shape[1], self.img.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def test_a_large_face_is_tracked_at_flow_scale(self):
        c = FlowCoaster()
        self.assertTrue(c.anchor(self.img, _bbox(self.lm), self.lm, self.lm[:5]))
        self.assertLess(c._scale, 0.5, 'a ~700px face must be tracked downscaled')
        m = self._m(24.0, -16.0)
        r = c.step(self._w(m))
        self.assertTrue(r.consistent, r.reason)
        err = np.linalg.norm(r.landmarks - _apply(m, self.lm), axis=1)
        self.assertLess(float(err.max()), 0.8 * self.K / 1.0 * 1.5)

    def test_a_ten_frame_dropout_at_high_resolution(self):
        c = FlowCoaster()
        self.assertTrue(c.anchor(self.img, _bbox(self.lm), self.lm, self.lm[:5]))
        last = total = None
        for k in range(1, 11):
            total = self._m(8.0 * k, -4.0 * k, deg=0.5 * k)
            last = c.step(self._w(total))
            self.assertTrue(last.consistent, f'step {k}: {last.reason}')
        err = np.linalg.norm(last.landmarks - _apply(total, self.lm), axis=1)
        self.assertLess(float(err.mean()), 2.0 * self.K)

    def test_small_faces_are_not_downscaled(self):
        c = FlowCoaster()
        lm = _landmarks()
        c.anchor(_texture(), _bbox(lm), lm, lm[:5])
        self.assertEqual(c._scale, 1.0)

    def test_jump_beyond_range_is_still_refused_at_high_resolution(self):
        c = FlowCoaster()
        c.anchor(self.img, _bbox(self.lm), self.lm, self.lm[:5])
        self.assertFalse(c.step(self._w(self._m(560.0, 0.0))).consistent)


class FlowRefusalTest(unittest.TestCase):
    """Wrong answers must come back inconsistent, never confident."""

    def setUp(self):
        self.img = _texture()
        self.lm = _landmarks()

    def test_a_featureless_face_is_not_believed(self):
        """No texture means no motion evidence (aperture problem). Zero flow
        from a blank patch must not read as 'the head did not move'."""
        flat = np.full((H, W), 128, np.uint8)
        c = FlowCoaster()
        anchored = c.anchor(flat, _bbox(self.lm), self.lm, self.lm[:5])
        if anchored:
            r = c.step(flat)
            self.assertTrue(r is None or not r.consistent)

    def test_a_jump_beyond_lk_range_is_not_believed(self):
        c = _coaster(self.img, self.lm)
        r = c.step(_warp(self.img, _similarity(140.0, 0.0)))
        self.assertFalse(r.consistent)

    def test_two_equal_competing_motions_are_not_believed(self):
        """Half the cloud moving one way and half the other is not one head,
        and the evidence cannot say which half is. Split at the cloud's own
        median so the two groups really are equal."""
        c = _coaster(self.img, self.lm)
        cut = float(np.median(c._pts[:, 0]))
        a = _warp(self.img, _similarity(6.0, 0.0))
        b = _warp(self.img, _similarity(-6.0, 0.0))
        split = np.where(np.arange(W)[None, :] < cut, a, b).astype(np.uint8)
        r = c.step(split)
        self.assertFalse(r.consistent, r.reason)

    def test_a_dominant_motion_with_a_minority_outlier_group_is_believed(self):
        """~25% of the cloud under something that moves differently: the head is
        still the clear majority, so the fit stands."""
        c = _coaster(self.img, self.lm)
        cut = float(np.percentile(c._pts[:, 0], 25))
        head = _warp(self.img, _similarity(5.0, 0.0))
        other = _warp(self.img, _similarity(-7.0, 2.0))
        mixed = np.where(np.arange(W)[None, :] < cut, other, head).astype(np.uint8)
        r = c.step(mixed)
        self.assertTrue(r.consistent, r.reason)
        self.assertAlmostEqual(float(r.transform[0, 2]), 5.0, delta=0.5)

    def test_an_implausible_scale_step_is_rejected(self):
        c = _coaster(self.img, self.lm)
        r = c.step(_warp(self.img, _similarity(0.0, 0.0, scale=1.6)))
        self.assertFalse(r.consistent)

    def test_a_failed_step_ends_the_chain(self):
        c = _coaster(self.img, self.lm)
        self.assertFalse(c.step(_warp(self.img, _similarity(140.0, 0.0))).consistent)
        self.assertFalse(c.anchored)
        self.assertIsNone(c.step(_warp(self.img, _similarity(1.0, 0.0))))

    def test_step_before_anchor_returns_none(self):
        self.assertIsNone(FlowCoaster().step(self.img))

    def test_frame_size_change_is_refused(self):
        c = _coaster(self.img, self.lm)
        self.assertIsNone(c.step(self.img[:240, :320]))


class FaceFromFlowTest(unittest.TestCase):

    def test_stamps_match_the_existing_coast_contract(self):
        img, lm = _texture(), _landmarks()
        c = _coaster(img, lm)
        r = c.step(_warp(img, _similarity(2.0, 1.0)))
        template = {'bbox': _bbox(lm), 'kps': lm[:5].copy(),
                    'landmark_2d_106': lm.copy(),
                    'embedding': np.ones(4, np.float32), 'det_score': np.float32(0.9)}
        face = face_from_flow(template, r, coast_age=3)
        self.assertTrue(face['_coasted'])
        self.assertTrue(face['_interpolated'], 'the swap audit counts gap-filled faces by this flag')
        self.assertTrue(face['_flow_coasted'])
        self.assertEqual(face['_coast_age'], 3)
        self.assertEqual(face['occlusion_state'], 'coasted')
        np.testing.assert_allclose(face['embedding'], template['embedding'])
        self.assertFalse(np.allclose(face['bbox'], template['bbox']))

    def test_inconsistent_result_yields_no_face(self):
        img, lm = _texture(), _landmarks()
        c = _coaster(img, lm)
        r = c.step(_warp(img, _similarity(140.0, 0.0)))
        self.assertIsNone(face_from_flow({'bbox': _bbox(lm)}, r))


class BridgeGapTest(unittest.TestCase):
    """The closing detection is the one piece of evidence flow cannot fake."""

    def setUp(self):
        self.lm = _landmarks()

    def test_residual_is_spread_linearly_and_the_path_ends_on_the_anchor(self):
        # flow drifted by +2 px in x by the closing frame
        path = [self.lm + [1.0 * (j + 1), 0.0] for j in range(3)]     # true motion
        end = self.lm + [4.0, 0.0]
        closing = self.lm + [4.0 + 2.0, 0.0]                          # anchor says +6
        out = bridge_gap(path, end, closing, iod=40.0)
        self.assertIsNotNone(out)
        # frame j gets (j+1)/(L+1) of the 2 px shortfall
        for j, p in enumerate(out):
            expect = self.lm + [1.0 * (j + 1) + 2.0 * (j + 1) / 4.0, 0.0]
            np.testing.assert_allclose(p, expect, atol=1e-4)

    def test_a_flow_path_that_missed_the_anchor_is_refused(self):
        path = [self.lm + [1.0, 0.0]]
        end = self.lm + [1.0, 0.0]
        closing = self.lm + [30.0, 0.0]                  # 30 px off on a 40 px IOD
        self.assertIsNone(bridge_gap(path, end, closing, iod=40.0))

    def test_tolerance_boundary(self):
        path = [self.lm.copy()]
        end = self.lm.copy()
        self.assertIsNotNone(bridge_gap(path, end, self.lm + [11.0, 0.0], iod=40.0))  # 0.275 IOD
        self.assertIsNone(bridge_gap(path, end, self.lm + [13.0, 0.0], iod=40.0))     # 0.325 IOD

    def test_no_flow_end_means_no_bridge(self):
        self.assertIsNone(bridge_gap([self.lm], None, self.lm, iod=40.0))
        self.assertIsNone(bridge_gap([], self.lm, self.lm, iod=40.0))


if __name__ == '__main__':
    unittest.main()


class ChainPolicyTest(unittest.TestCase):
    """The decision a persistent chain makes on a DETECTED face (Stage 17)."""

    def setUp(self):
        from roop.flow_coast import (chain_policy, SUBSTITUTE, REFRESH, CONTINUE, REANCHOR)
        self.p, self.S, self.R, self.C, self.A = chain_policy, SUBSTITUTE, REFRESH, CONTINUE, REANCHOR

    def test_an_occluded_face_takes_the_chains_geometry(self):
        self.assertEqual(self.p(True, 0.55, 3, 0), self.S)

    def test_a_clean_young_chain_just_continues(self):
        self.assertEqual(self.p(True, 0.95, 2, 0), self.C)

    def test_a_clean_old_chain_is_refreshed_on_the_detection(self):
        self.assertEqual(self.p(True, 0.95, 4, 0), self.R)

    def test_a_chain_in_the_grey_zone_is_neither_trusted_nor_refreshed(self):
        # survival 0.8: not occluded enough to override the detector, not clean
        # enough to anchor on it
        self.assertEqual(self.p(True, 0.80, 5, 0), self.C)

    def test_a_failed_step_starts_over_from_the_detection(self):
        self.assertEqual(self.p(False, 0.2, 3, 0), self.A)

    def test_the_detector_is_trusted_again_after_the_substitution_budget(self):
        self.assertEqual(self.p(True, 0.5, 5, 8), self.C)
        self.assertEqual(self.p(True, 0.5, 8, 8), self.A)

    def test_nothing_runs_forever(self):
        self.assertEqual(self.p(True, 0.8, 8, 0), self.A)

    def test_thresholds_are_the_measured_ones(self):
        from roop import flow_coast as fc
        self.assertEqual((fc.OCCLUDED_BELOW, fc.REFRESH_ABOVE), (0.70, 0.90))


class SurvivalTest(unittest.TestCase):
    """`survival` is the mask-free occlusion estimate the policy leans on."""

    def test_clean_motion_keeps_survival_high(self):
        img, lm = _texture(), _landmarks()
        c = _coaster(img, lm)
        r = None
        for k in range(1, 5):
            r = c.step(_warp(img, _similarity(1.5 * k, -1.0 * k)))
            self.assertTrue(r.consistent, r.reason)
        self.assertGreater(r.survival, 0.85)

    def test_a_covering_occluder_lowers_survival(self):
        img, lm = _texture(), _landmarks()
        c = _coaster(img, lm)
        moved = _warp(img, _similarity(2.0, 1.0))
        moved[150:400, 240:330] = _texture(seed=42)[150:400, 240:330]
        r = c.step(moved)
        self.assertTrue(r.consistent, r.reason)
        self.assertLess(r.survival, 0.75)

    def test_a_failed_step_reports_unknown_not_occluded(self):
        img, lm = _texture(), _landmarks()
        c = _coaster(img, lm)
        r = c.step(_warp(img, _similarity(140.0, 0.0)))
        self.assertFalse(r.consistent)
        self.assertEqual(r.survival, 1.0)
