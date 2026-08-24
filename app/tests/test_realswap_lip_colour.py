"""RealSwap's lip-colour transfer: properties that must hold without a GPU.

The rendered proof lives in `tests/verify_realswap_lip_colour.py`, which runs
both nets on real frames and grades against hififace. These are the invariants
that can be asserted from arithmetic alone, and every one of them is here
because the first build of `_lip_colour` broke it.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                            # noqa: E402
import cv2                                                    # noqa: E402

from roop.processors.FaceSwapInsightFace import (              # noqa: E402
    FaceSwapInsightFace as CLS)

SIZE = 256
TMPL = 'arcface'


def crop(fill=0.0, seed=0):
    """A [3,H,W] float32 RGB crop in [-1,1], the swap models' own layout."""
    rng = np.random.default_rng(seed)
    return (np.full((3, SIZE, SIZE), fill, np.float32)
            + rng.normal(0, 0.05, (3, SIZE, SIZE)).astype(np.float32))


def paint_lips(img, rgb):
    """Set the lip ellipse to a colour, so a transfer has something to move."""
    lip, _, _ = CLS._lip_masks(SIZE, TMPL)
    out = img.copy()
    for c in range(3):
        out[c] = out[c] * (1 - lip) + rgb[c] * lip
    return out


class Masks(unittest.TestCase):

    def test_the_lip_mask_sits_on_the_mouth(self):
        """Anchored to the TEMPLATE's mouth corners, not to a per-face fit."""
        from roop.face_util import swap_template_points
        lip, ring, _ = CLS._lip_masks(SIZE, TMPL)
        pts = np.asarray(swap_template_points(SIZE, TMPL), np.float32)
        cx, cy = (pts[3] + pts[4]) / 2.0
        ys, xs = np.nonzero(lip > 0.5 * lip.max())
        self.assertAlmostEqual(float(xs.mean()), float(cx), delta=2.0)
        self.assertAlmostEqual(float(ys.mean()), float(cy), delta=2.0)

    def test_the_ring_excludes_the_lips(self):
        """The reference is SKIN. A ring that overlapped the lips would measure
        the lip colour against itself and the offset would collapse."""
        lip, ring, _ = CLS._lip_masks(SIZE, TMPL)
        self.assertEqual(float((ring * (lip > 0.5)).sum()), 0.0)
        self.assertGreater(float(ring.sum()), float(lip.sum()))

    def test_the_masks_are_cached_per_size_and_template(self):
        a = CLS._lip_masks(SIZE, TMPL)
        b = CLS._lip_masks(SIZE, TMPL)
        self.assertIs(a[0], b[0])
        self.assertIsNot(CLS._lip_masks(128, TMPL)[0], a[0])

    def test_the_feather_is_a_sigma_not_a_kernel(self):
        """A fixed kernel is sigma ~1.7 px at a 58 px mouth -- fine enough that
        the mask's own edge reads as added structure, which is exactly what the
        sweep in the constants measured at 1.33/255. Sizing it as a fraction of
        the mouth keeps the edge soft at every crop size."""
        src = open(os.path.join(APP, 'roop', 'processors',
                                'FaceSwapInsightFace.py'), encoding='utf-8').read()
        self.assertIn('sigmaX=max(0.8, cls._LIP_FEATHER * mw)', src)
        for size in (128, 256, 512):
            lip, _, _ = CLS._lip_masks(size, TMPL)
            edge = np.abs(cv2.Laplacian(lip, cv2.CV_32F, ksize=3)).max()
            self.assertLess(float(edge), 0.35,
                            f"lip mask edge too abrupt at {size}")


class Transfer(unittest.TestCase):

    def test_it_changes_nothing_outside_the_lip_mask(self):
        """EXACTLY nothing -- this is the guarantee that makes 'only the lips'
        a fact rather than an intention."""
        base, sec = crop(0.0, 1), paint_lips(crop(0.0, 2), (0.6, -0.2, -0.2))
        out = CLS._lip_colour(base, sec, SIZE, TMPL)
        lip, _, _ = CLS._lip_masks(SIZE, TMPL)
        self.assertEqual(float(np.abs(out - base)[:, lip <= 0.0].max()), 0.0)

    def test_it_moves_the_lips_toward_the_secondary(self):
        base = paint_lips(crop(0.0, 1), (0.10, 0.0, 0.0))    # pale lips
        sec = paint_lips(crop(0.0, 1), (0.55, -0.1, -0.1))   # rich lips
        out = CLS._lip_colour(base, sec, SIZE, TMPL)
        lip, ring, _ = CLS._lip_masks(SIZE, TMPL)

        def offset(x, ch):
            return float((x[ch] * lip).sum() / lip.sum()
                         - (x[ch] * ring).sum() / ring.sum())

        for ch in range(3):
            before = abs(offset(base, ch) - offset(sec, ch))
            after = abs(offset(out, ch) - offset(sec, ch))
            self.assertLess(after, before * 0.2,
                            f"channel {ch} barely moved")

    def test_a_global_cast_between_the_nets_is_NOT_transferred(self):
        """THE BUG THE MEASUREMENT CAUGHT FIRST. The two nets differ by a
        whole-face colour shift (1.92 on the skin, 1.82 on the lips), and a
        transfer that compared their lips directly would paint that face-wide
        tint onto the mouth. Measuring lips-minus-skin in each net separately
        makes it cancel: a secondary that is the base plus a constant must
        produce no change at all."""
        base = paint_lips(crop(0.0, 3), (0.3, 0.0, 0.0))
        sec = base + np.float32(0.25)          # pure global cast
        out = CLS._lip_colour(base, sec, SIZE, TMPL)
        self.assertLess(float(np.abs(out - base).max()), 1e-5)

    def test_it_carries_no_structure_from_the_secondary(self):
        """The transferred quantity is three numbers, so a secondary with wildly
        different lip STRUCTURE and the same lip COLOUR must give the same
        answer as a flat one."""
        base = paint_lips(crop(0.0, 4), (0.2, 0.0, 0.0))
        flat = paint_lips(crop(0.0, 5), (0.6, -0.1, -0.1))
        lip, _, _ = CLS._lip_masks(SIZE, TMPL)
        rng = np.random.default_rng(7)
        # Same lip MEAN, violently different content. The pattern is added
        # THROUGH the mask, so what has to vanish from the lip-weighted mean is
        # sum(pattern * lip * lip) -- weighting by `lip` instead leaves a
        # residual and the test fails on its own arithmetic rather than on the
        # code's.
        pattern = rng.normal(0, 0.3, (3, SIZE, SIZE)).astype(np.float32)
        lip2 = lip * lip
        pattern -= ((pattern * lip2).reshape(3, -1).sum(1)[:, None, None]
                    / lip2.sum())
        textured = flat + pattern * lip
        a = CLS._lip_colour(base, flat, SIZE, TMPL)
        b = CLS._lip_colour(base, textured, SIZE, TMPL)
        self.assertLess(float(np.abs(a - b).max()), 1e-4)

    def test_strength_zero_is_a_bit_identical_no_op(self):
        base, sec = crop(0.0, 8), paint_lips(crop(0.0, 9), (0.6, 0, 0))
        out = CLS._lip_colour(base, sec, SIZE, TMPL, strength=0.0)
        self.assertIs(out, base)

    def test_strength_scales_the_offset_linearly(self):
        base = paint_lips(crop(0.0, 10), (0.1, 0.0, 0.0))
        sec = paint_lips(crop(0.0, 10), (0.5, 0.0, 0.0))
        full = CLS._lip_colour(base, sec, SIZE, TMPL, strength=1.0) - base
        half = CLS._lip_colour(base, sec, SIZE, TMPL, strength=0.5) - base
        self.assertLess(float(np.abs(full * 0.5 - half).max()), 1e-5)

    def test_the_delivered_offset_is_the_full_delta(self):
        """The mask is feathered, so without the normalisation the lip MEAN
        lands short of the target. Measured on real frames: 0.115 residual
        un-normalised against 0.063 normalised."""
        base = paint_lips(crop(0.0, 11), (0.10, 0.0, 0.0))
        sec = paint_lips(crop(0.0, 11), (0.50, 0.0, 0.0))
        out = CLS._lip_colour(base, sec, SIZE, TMPL)
        lip, ring, _ = CLS._lip_masks(SIZE, TMPL)

        def offset(x, ch):
            return float((x[ch] * lip).sum() / lip.sum()
                         - (x[ch] * ring).sum() / ring.sum())

        self.assertAlmostEqual(offset(out, 0), offset(sec, 0), delta=0.01)

    def test_non_finite_input_returns_the_base_unchanged(self):
        base = paint_lips(crop(0.0, 12), (0.2, 0, 0))
        sec = paint_lips(crop(0.0, 13), (0.6, 0, 0))
        sec[0, 0, 0] = np.nan
        out = CLS._lip_colour(base, sec, SIZE, TMPL)
        self.assertIs(out, base)


class Wiring(unittest.TestCase):

    def test_mix_outputs_applies_it(self):
        """A two-path feature that silently does not run looks exactly like one
        that ran and did nothing -- this processor has been bitten by that twice
        (the pose router, and the batched paths dropping the composite)."""
        src = open(os.path.join(APP, 'roop', 'processors',
                                'FaceSwapInsightFace.py'), encoding='utf-8').read()
        body = src[src.index('def _mix_outputs('):src.index('def mix_summary(')]
        self.assertIn('self._lip_colour(', body)

    def test_mix_summary_reports_it(self):
        src = open(os.path.join(APP, 'roop', 'processors',
                                'FaceSwapInsightFace.py'), encoding='utf-8').read()
        body = src[src.index('def mix_summary('):src.index('def _run_secondary(')]
        self.assertIn('_LIP_COLOUR', body)

    def test_it_defaults_ON(self):
        """Ship the fix, not the flag: a feature defaulting off is off for
        everyone."""
        self.assertGreater(CLS._LIP_COLOUR, 0.0)

    def test_the_env_override_exists_for_remeasuring(self):
        src = open(os.path.join(APP, 'roop', 'processors',
                                'FaceSwapInsightFace.py'), encoding='utf-8').read()
        self.assertIn('ROOP_REALSWAP_LIP_COLOUR', src)


class TheNegativeResultsAreRecorded(unittest.TestCase):
    """Three things were built here and measured away. Each cost a render to
    disprove, and each looks obviously correct on paper, so the reasons live in
    the source at the site rather than only in a session log."""

    def setUp(self):
        self.doc = CLS._lip_colour.__doc__ or ''

    def test_the_luminance_projection_is_recorded_as_rejected(self):
        self.assertIn('PROJECTING THE LUMINANCE OUT', self.doc)
        self.assertIn('0.910', self.doc)      # the gap it left

    def test_the_redness_gate_is_recorded_as_rejected(self):
        self.assertIn('REDNESS GATE', self.doc)

    def test_the_normalisation_reason_is_recorded(self):
        self.assertIn('0.115', self.doc)      # un-normalised residual


if __name__ == '__main__':
    unittest.main(verbosity=2)
