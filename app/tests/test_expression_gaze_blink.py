"""Expression Transfer Strength / Eye-Gaze Follow Ratio / Blink Sync.

The split must be free for every config written before it: with no gaze ratio
the keypoint maths has to be BIT-identical to the old single-strength path, not
merely close, because the regression benchmark compares renders byte for byte.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.processors.Expression_LivePortrait import (  # noqa: E402
    EYE_INDICES, LIP_INDICES, Expression_LivePortrait, blend_expression,
    blink_retarget_state, driving_keypoints, eye_close_ratio, eye_retarget_input,
    keypoint_weights)
from settings import expression_stage_active, legacy_gaze_follow  # noqa: E402

N = 21


def _legacy_driving(x_s, scale, exp_s, exp_d, strength, region):
    """The pre-split driving_keypoints, verbatim."""
    exp_mix = blend_expression(exp_s, exp_d, strength, region)
    delta = (exp_mix - np.asarray(exp_s, np.float32)).reshape(x_s.shape)
    return x_s + delta * np.asarray(scale, np.float32).reshape(-1, 1, 1)


class TestLegacyBitIdentity(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.x_s = rng.normal(size=(1, N, 3)).astype(np.float32)
        self.exp_s = rng.normal(scale=0.05, size=(1, N, 3)).astype(np.float32)
        self.exp_d = rng.normal(scale=0.05, size=(1, N, 3)).astype(np.float32)
        self.scale = np.array([[1.7]], np.float32)

    def test_gaze_none_is_the_old_path(self):
        for region in ("all", "lips", "eyes"):
            for s in (0.0, 0.4, 1.0, 1.2):
                old = _legacy_driving(self.x_s, self.scale, self.exp_s, self.exp_d, s, region)
                new = driving_keypoints(self.x_s, self.scale, self.exp_s, self.exp_d, s, region)
                self.assertTrue(np.array_equal(old, new), (region, s))

    def test_weights_reproduce_blend_expression_exactly(self):
        # The explicit-gaze path with gaze == the legacy value must also match
        # bit for bit: a saved config reloads with gaze derived from strength.
        for region in ("all", "lips", "eyes"):
            for s in (0.4, 1.0, 1.2):
                g = legacy_gaze_follow(s, region)
                old = _legacy_driving(self.x_s, self.scale, self.exp_s, self.exp_d, s, region)
                new = driving_keypoints(self.x_s, self.scale, self.exp_s, self.exp_d, s, region, g)
                self.assertTrue(np.array_equal(old, new), (region, s, g))


class TestSplit(unittest.TestCase):
    def test_weights(self):
        w = keypoint_weights(N, 0.7, 0.2, "all")
        for i in range(N):
            self.assertAlmostEqual(float(w[i]), 0.2 if i in EYE_INDICES else 0.7, places=6)

    def test_gaze_alone_moves_only_eye_keypoints(self):
        rng = np.random.default_rng(1)
        x_s = rng.normal(size=(1, N, 3)).astype(np.float32)
        e_s = rng.normal(size=(1, N, 3)).astype(np.float32)
        e_d = rng.normal(size=(1, N, 3)).astype(np.float32)
        x_d = driving_keypoints(x_s, np.ones((1, 1), np.float32), e_s, e_d, 0.0, "all", 1.0)
        moved = {i for i in range(N) if not np.array_equal(x_d[0, i], x_s[0, i])}
        self.assertEqual(moved, set(EYE_INDICES))

    def test_expression_alone_leaves_the_eyes(self):
        rng = np.random.default_rng(2)
        x_s = rng.normal(size=(1, N, 3)).astype(np.float32)
        e_s = rng.normal(size=(1, N, 3)).astype(np.float32)
        e_d = rng.normal(size=(1, N, 3)).astype(np.float32)
        x_d = driving_keypoints(x_s, np.ones((1, 1), np.float32), e_s, e_d, 1.0, "all", 0.0)
        for i in EYE_INDICES:
            self.assertTrue(np.array_equal(x_d[0, i], x_s[0, i]))
        self.assertFalse(np.array_equal(x_d[0, LIP_INDICES[0]], x_s[0, LIP_INDICES[0]]))

    def test_lips_region_with_gaze(self):
        w = keypoint_weights(N, 1.0, 0.5, "lips")
        for i in range(N):
            want = 0.5 if i in EYE_INDICES else (1.0 if i in LIP_INDICES else 0.0)
            self.assertEqual(float(w[i]), want)


class TestEyeRatio(unittest.TestCase):
    def _pts(self, gap):
        p = np.zeros((203, 2), np.float32)
        # left eye: corners 0 and 12, lids 6 and 18; right: 24/36, 30/42
        p[0], p[12] = (0, 0), (10, 0)
        p[6], p[18] = (5, -gap / 2), (5, gap / 2)
        p[24], p[36] = (20, 0), (30, 0)
        p[30], p[42] = (25, -gap), (25, gap)
        return p

    def test_ratio_is_gap_over_width(self):
        r = eye_close_ratio(self._pts(4.0))
        self.assertAlmostEqual(float(r[0]), 0.4, places=5)
        self.assertAlmostEqual(float(r[1]), 0.8, places=5)

    def test_closed_eye_is_zero(self):
        self.assertLess(float(eye_close_ratio(self._pts(0.0))[0]), 1e-6)

    def test_retarget_input_layout(self):
        x = np.arange(63, dtype=np.float32).reshape(1, 21, 3)
        v = eye_retarget_input(x, np.array([0.3, 0.35]), np.array([0.05, 0.4]))
        self.assertEqual(v.shape, (1, 66))
        self.assertTrue(np.array_equal(v[0, :63], np.arange(63, dtype=np.float32)))
        # the TARGET contributes its first eye only, as in LivePortrait
        np.testing.assert_allclose(v[0, 63:], [0.3, 0.35, 0.05], rtol=1e-6)


class TestBlinkDoesNotDoubleCount(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.x_s = rng.normal(size=(1, N, 3)).astype(np.float32)
        self.x_d = self.x_s + 0.1
        self.c_s = np.array([0.45, 0.47], np.float32)
        self.c_t = np.array([0.05, 0.06], np.float32)

    def test_weight_zero_is_the_reference_call(self):
        kp, c = blink_retarget_state(self.x_s, self.x_d, self.c_s, self.c_t, 0.0)
        self.assertIs(kp, self.x_s)
        self.assertTrue(np.array_equal(c, self.c_s))

    def test_full_weight_sees_lids_at_target(self):
        kp, c = blink_retarget_state(self.x_s, self.x_d, self.c_s, self.c_t, 1.0)
        self.assertIs(kp, self.x_d)
        np.testing.assert_allclose(c, self.c_t, rtol=1e-6)

    def test_half_weight_interpolates_and_clamps(self):
        _, c = blink_retarget_state(self.x_s, self.x_d, self.c_s, self.c_t, 0.5)
        np.testing.assert_allclose(c, (self.c_s + self.c_t) / 2, rtol=1e-6)
        _, c2 = blink_retarget_state(self.x_s, self.x_d, self.c_s, self.c_t, 1.8)
        np.testing.assert_allclose(c2, self.c_t, rtol=1e-6)


class TestGates(unittest.TestCase):
    def test_is_active(self):
        self.assertFalse(Expression_LivePortrait.is_active(0.0, None, False))
        self.assertFalse(Expression_LivePortrait.is_active(0.0, 0.0, False))
        self.assertTrue(Expression_LivePortrait.is_active(0.0, 0.3, False))
        self.assertTrue(Expression_LivePortrait.is_active(0.0, None, True))
        self.assertTrue(Expression_LivePortrait.is_active(0.5, None, False))

    def test_stage_active_agrees(self):
        get = lambda d: (lambda k, default: d.get(k, default))  # noqa: E731
        self.assertFalse(expression_stage_active(get({})))
        self.assertFalse(expression_stage_active(get({'expression_restore_strength': 0})))
        self.assertTrue(expression_stage_active(get({'expression_gaze_follow': 0.5})))
        self.assertTrue(expression_stage_active(get({'expression_blink_sync': True})))
        self.assertTrue(expression_stage_active(get({'expression_restore_strength': 0.8})))

    def test_legacy_gaze(self):
        self.assertEqual(legacy_gaze_follow(0.8, 'all'), 0.8)
        self.assertEqual(legacy_gaze_follow(0.8, 'eyes'), 0.8)
        self.assertEqual(legacy_gaze_follow(0.8, 'lips'), 0.0)
        self.assertEqual(legacy_gaze_follow(None, 'all'), 0.0)

    def test_settings_derives_gaze_for_an_old_config(self):
        import tempfile
        from settings import Settings
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'config.yaml')
            with open(p, 'w', encoding='utf-8') as f:
                f.write("expression_restore_strength: 1.2\nexpression_restore_region: all\n")
            s = Settings(p)
            self.assertEqual(float(s.expression_gaze_follow), 1.2)
            self.assertFalse(s.expression_blink_sync)


class TestProcessMgrGate(unittest.TestCase):
    """The render-path gate must run the stage for gaze-only and blink-only."""

    def test_call_site_reads_the_new_keys(self):
        src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'roop', 'ProcessMgr.py'), encoding='utf-8').read()
        i = src.index("_ex = float(getattr(roop.globals, 'expression_restore_strength'")
        block = src[i:i + 2500]
        self.assertIn("expression_gaze_follow", block)
        self.assertIn("expression_blink_sync", block)
        self.assertIn("gaze=_gz, blink=_bk", block)
        self.assertIn("restorer.prepare(_crop, aligned_img, _bk)", block)


if __name__ == '__main__':
    unittest.main()
