"""Every restorer must refuse to hand back a non-finite frame.

The shared ending — clip to [-1, 1], rescale, cast to uint8 — silently turns a
NaN into a BLACK PIXEL, because `np.clip` propagates NaN rather than removing
it (inf clips fine; nan does not) and `uint8(nan)` is 0. A saturated graph
therefore produces a completely black face with no exception, no warning, and
a perfectly normal-looking (512, 512, 3) uint8 array on the way out:

    np.clip(nan, -1, 1)                       -> nan
    np.full(..., nan) -> post-process -> uint8 -> every value 0

This is not hypothetical in this repo. GPEN's 1024/2048 weights overflow in
FP16 under TensorRT and painted exactly that, which is why GPEN alone carried
a guard; the frame upscaler hit the same thing (ESRGAN x4 goes black under TRT
FP16). CodeFormer, GFPGAN, RestoreFormer++ and DMDNet ran the same three lines
unguarded — and CodeFormer has since gained a half-precision tier.

A black face reads as "the app is broken", not "this model overflowed", so it
costs a long time to trace. The guard is one `np.isfinite` over a 512² crop.
"""

import os
import re
import sys
import unittest

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

from roop.processors.enhance_common import is_usable, sized     # noqa: E402

PROC = os.path.join(APP, 'roop', 'processors')
ENHANCERS = ('Enhance_CodeFormer', 'Enhance_GFPGAN', 'Enhance_GPEN',
             'Enhance_RestoreFormerPPlus', 'Enhance_DMDNet')


def _src(name):
    with open(os.path.join(PROC, f'{name}.py'), encoding='utf-8') as fh:
        return fh.read()


class TheGuardItself(unittest.TestCase):
    def test_nan_and_inf_are_both_rejected(self):
        ok = np.random.default_rng(0).uniform(-1, 1, (3, 8, 8)).astype(np.float32)
        self.assertTrue(is_usable(ok))
        for bad in (np.nan, np.inf, -np.inf):
            x = ok.copy()
            x[0, 0, 0] = bad
            self.assertFalse(is_usable(x), f'{bad} slipped through')

    def test_the_failure_mode_is_what_the_guard_claims(self):
        """Pin the reason the guard exists, so nobody 'simplifies' it away by
        reasoning that np.clip already handles it."""
        self.assertTrue(np.isnan(np.clip(np.float32(np.nan), -1, 1)),
                        'np.clip is expected to PROPAGATE nan — if that ever '
                        'changes, this guard could be revisited')
        self.assertEqual(int(np.array([np.nan], np.float32).astype(np.uint8)[0]), 0,
                         'uint8(nan) is expected to be 0, i.e. black')


class TheScaleContract(unittest.TestCase):
    def test_a_model_smaller_than_the_crop_is_resized_not_zeroed(self):
        out, sf = sized(np.zeros((256, 256, 3), np.uint8), 512)
        self.assertEqual(out.shape[:2], (512, 512))
        self.assertEqual(sf, 1, 'int(256/512) is 0, which collapses the paste matrix')

    def test_larger_models_report_an_integer_factor(self):
        for width, crop, want in ((512, 512, 1), (1024, 512, 2), (2048, 512, 4),
                                  (512, 128, 4), (512, 256, 2)):
            _out, sf = sized(np.zeros((width, width, 3), np.uint8), crop)
            self.assertEqual(sf, want, f'{width} from a {crop} crop')

    def test_the_factor_is_never_zero(self):
        for width in (64, 128, 256, 511):
            _out, sf = sized(np.zeros((width, width, 3), np.uint8), 512)
            self.assertGreaterEqual(sf, 1)


class EveryEnhancerUsesIt(unittest.TestCase):
    def test_all_of_them_guard_their_output(self):
        for name in ENHANCERS:
            self.assertIn(
                'is_usable(', _src(name),
                f'{name} does not check its output for non-finite values, so a '
                'FP16 overflow or a torn session paints a silent black face')

    def test_none_of_them_still_compute_the_factor_by_hand(self):
        """One implementation, so the two traps are documented in one place."""
        for name in ENHANCERS:
            self.assertEqual(
                re.findall(r'scale_factor\s*=\s*(?:max\()?int\(', _src(name)), [],
                f'{name} computes scale_factor itself; use enhance_common.sized')


if __name__ == '__main__':
    unittest.main()
