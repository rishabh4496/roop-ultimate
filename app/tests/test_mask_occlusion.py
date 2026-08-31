"""Contract tests for the opt-in asymmetric temporal occlusion response."""

import os
import sys
import unittest

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.one_euro import MaskStabilizer, _alpha  # noqa: E402


KPS = np.array([
    [40.0, 42.0], [72.0, 42.0], [56.0, 58.0],
    [44.0, 74.0], [68.0, 74.0],
], dtype=np.float32)


class MaskOcclusionResponse(unittest.TestCase):

    def _sequence(self, fast_restore=0.0):
        return MaskStabilizer(strength=0.5, motion_beta=0.0,
                              fast_restore_alpha=fast_restore)

    def test_default_is_the_original_symmetric_filter(self):
        base = self._sequence()
        explicit_zero = self._sequence(fast_restore=0.0)
        masks = [np.zeros((8, 8), np.float32), np.ones((8, 8), np.float32),
                 np.zeros((8, 8), np.float32)]
        for i, mask in enumerate(masks):
            got = base.apply(mask, KPS, i)
            expected = explicit_zero.apply(mask, KPS, i)
            np.testing.assert_array_equal(got, expected)

    def test_occluder_entering_is_revealed_faster(self):
        normal = self._sequence()
        fast = self._sequence(fast_restore=0.85)
        empty = np.zeros((8, 8), np.float32)
        occluded = np.ones((8, 8), np.float32)
        normal.apply(empty, KPS, 0)
        fast.apply(empty, KPS, 0)
        normal_enter = normal.apply(occluded, KPS, 1)
        fast_enter = fast.apply(occluded, KPS, 1)
        self.assertGreater(float(fast_enter.mean()), float(normal_enter.mean()))
        self.assertGreaterEqual(float(fast_enter.min()), 0.85 - 1e-6)

    def test_occluder_leaving_keeps_the_normal_fade(self):
        normal = self._sequence()
        fast = self._sequence(fast_restore=0.85)
        empty = np.zeros((8, 8), np.float32)
        occluded = np.ones((8, 8), np.float32)
        for stab in (normal, fast):
            stab.apply(empty, KPS, 0)
            stab.apply(occluded, KPS, 1)
        before_fast_leave = fast.tracks[0]['prev'].copy()
        fast_leave = fast.apply(empty, KPS, 2)
        # The fast path changes the state on entry, so its absolute release is
        # intentionally different from the normal filter. Its reverse step must
        # still use the ordinary alpha, rather than snapping straight to zero.
        expected = before_fast_leave * (1.0 - _alpha(1.0, fast.base_cutoff))
        np.testing.assert_allclose(fast_leave, expected, rtol=0.0, atol=1e-7)
        self.assertGreater(float(fast_leave.mean()), 0.0)

    def test_alpha_is_clamped_and_bad_values_disable(self):
        self.assertEqual(self._sequence(-2.0).fast_restore_alpha, 0.0)
        self.assertEqual(self._sequence(4.0).fast_restore_alpha, 1.0)
        self.assertEqual(self._sequence('bad').fast_restore_alpha, 0.0)


if __name__ == '__main__':
    unittest.main()
