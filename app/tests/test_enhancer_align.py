"""Enhancer alignment, and the paste-scale guard next to it.

The restorers here learned their priors from FFHQ-aligned 512 faces, but the
crop they are handed comes from the SWAPPER's template, because the enhancer
reuses the swap crop. Measured against ffhq_512 at 512 px:

    arcface (inswapper family)  scale 0.856   mean landmark error 22.1 px
    arcface_112_v1 (ghost…)     scale 0.828                       16.3 px
    arcface_112_v2 (blendswap)  scale 0.744                       30.7 px
    mtcnn_512 (hififace)        scale 0.875                       13.2 px
    ffhq_512 (uniface…)         scale 1.000                        0.0 px

ProcessMgr can re-warp into the enhancer's own space and back. Two properties
carry that, and neither is visible in a diff:

  * THE ROUND TRIP IS EXACT. Two affines that are not inverses still produce a
    plausible-looking face, just drifting a few pixels per frame — which reads
    as the swap being unstable, not as a matrix being wrong.
  * THE SCALED INVERSE. The enhancer usually returns a LARGER buffer than it
    was given, and an affine expressed for a k-times bigger canvas is the same
    rotation and scale with a k-times bigger TRANSLATION. Scaling the whole
    matrix instead is the obvious mistake and it lands the face off-centre.

Plus the guard that lives beside them: paste_upscale multiplies the paste
matrix by scale_factor, so a 0 collapses it and blanks the face. It cannot
reach 0 today (biggest swapper output and biggest pixel-boost are both 512),
which is exactly why it needs a test — nothing else would notice the day a
1024 model lands.
"""

import os
import re
import sys
import unittest

import cv2
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

import roop.globals as g                                        # noqa: E402
from roop.ProcessMgr import _compose_affine, _invert_affine     # noqa: E402
from roop.face_util import estimate_norm                        # noqa: E402

PROC = os.path.join(APP, 'roop', 'processors')
KPS = np.array([[95., 105.], [160., 103.], [128., 140.], [100., 180.], [155., 178.]],
               dtype=np.float32)
PTS = np.array([[[40., 40.], [128., 128.], [210., 200.], [5., 250.]]], dtype=np.float32)


def realign(size=256, swap='arcface', enh='ffhq_512'):
    return _compose_affine(estimate_norm(KPS, size, mode=enh),
                           _invert_affine(estimate_norm(KPS, size, mode=swap)))


class TheRoundTripIsExact(unittest.TestCase):
    def test_forward_then_inverse_is_identity(self):
        for swap in ('arcface', 'arcface_112_v1', 'mtcnn_512', 'ffhq_512'):
            A = realign(swap=swap)
            back = cv2.transform(cv2.transform(PTS, A), cv2.invertAffineTransform(A))
            self.assertLess(np.abs(back - PTS).max(), 1e-3,
                            f'{swap}: the re-align round trip does not return to '
                            'where it started')

    def test_the_inverse_scales_by_translation_only(self):
        """What ProcessMgr does when the enhancer returns a bigger buffer."""
        A = realign()
        fwd = cv2.transform(PTS, A)
        for k in (1.0, 2.0, 4.0):
            B = cv2.invertAffineTransform(A)
            B[:, 2] *= k
            got = cv2.transform(fwd * k, B)
            self.assertLess(np.abs(got - PTS * k).max(), 1e-3,
                            f'k={k}: scaling the whole matrix instead of only its '
                            'translation puts the face off-centre')

    def test_compose_applies_the_right_one_first(self):
        """_compose_affine(A, B) must mean "B, then A" — the other order also
        produces a crop, just of the wrong part of the face."""
        A = np.array([[2., 0., 0.], [0., 2., 0.]])          # scale x2
        B = np.array([[1., 0., 10.], [0., 1., 0.]])         # shift x+10
        p = np.array([[[1., 1.]]], dtype=np.float32)
        got = cv2.transform(p, _compose_affine(A, B).astype(np.float32))
        self.assertAlmostEqual(float(got[0, 0, 0]), 22.0, places=4,
                               msg='expected shift-then-scale = (1+10)*2')


class MatchingTemplatesAreANoOp(unittest.TestCase):
    def test_the_same_template_gives_the_identity(self):
        """uniface/blendswap already crop to ffhq_512, so re-aligning them must
        change nothing — and ProcessMgr skips the warp entirely for them."""
        A = realign(swap='ffhq_512', enh='ffhq_512')
        self.assertLess(np.abs(cv2.transform(PTS, A) - PTS).max(), 1e-3)

    def test_a_mismatched_template_actually_moves_the_crop(self):
        """The other half: if this were also identity the feature would be a
        no-op dressed as a fix."""
        moved = np.abs(cv2.transform(PTS, realign(swap='arcface')) - PTS).max()
        self.assertGreater(moved, 5.0,
                           'arcface and ffhq_512 should differ by many pixels; '
                           'got {moved:.2f}')


class TheBorderComesFromThePlate(unittest.TestCase):
    """FFHQ framing is WIDER than every swap template, so warping the swap crop
    into it leaves a ring the crop does not contain. Filling that with
    replicated edge pixels would hand the restorer a correctly-framed face
    inside a thick smear — the same out-of-distribution input the re-align
    exists to remove, just a different flavour of it."""

    def _ring(self, swap):
        A = realign(size=512, swap=swap)
        cov = cv2.warpAffine(np.full((512, 512), 255, np.uint8), A, (512, 512),
                             flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return float((cov == 0).mean())

    def test_the_ring_is_far_too_large_to_replicate(self):
        for swap, floor in (('arcface', 0.20), ('arcface_112_v1', 0.20),
                            ('arcface_112_v2', 0.30), ('mtcnn_512', 0.15)):
            got = self._ring(swap)
            self.assertGreater(
                got, floor,
                f'{swap}: {got:.1%} of the enhancer input is outside the swap '
                'crop — this is why it is composited against the frame rather '
                'than edge-replicated')

    def test_a_matching_template_leaves_no_ring(self):
        self.assertLess(self._ring('ffhq_512'), 0.001)

    def test_processmgr_composites_against_the_frame(self):
        """The regression would be a one-line 'simplification' back to a plain
        warp with BORDER_REPLICATE, which still produces a plausible crop."""
        src = open(os.path.join(APP, 'roop', 'ProcessMgr.py'), encoding='utf-8').read()
        block = src.split('_M_e = estimate_norm', 1)[1][:2000]
        # `plate` is the untouched original frame — the same buffer as `frame`
        # for a single face, and deliberately NOT the running composite when
        # several faces overlap (the ring would otherwise pick up a neighbour's
        # already-swapped skin). Either name satisfies the property this guards.
        self.assertRegex(block, r'warpAffine\((frame|plate), _M_e',
                         'the enhancer input must take its outer region from the '
                         'original frame, which has real hair/neck/background there')
        self.assertIn('INTER_NEAREST', block,
                      'the coverage mask must be nearest-neighbour, or its own '
                      'interpolation softens the footprint it is describing')

    def test_the_swap_crop_survives_the_round_trip(self):
        """The reverse warp needs the whole swap crop to have landed inside the
        enhancer's box; anything that fell off the edge is gone for good."""
        for swap in ('arcface', 'arcface_112_v1', 'arcface_112_v2', 'mtcnn_512'):
            A = realign(size=512, swap=swap)
            corners = np.array([[[0, 0], [512, 0], [512, 512], [0, 512]]], np.float32)
            mapped = cv2.transform(corners, A)[0]
            self.assertTrue(
                (mapped >= -0.5).all() and (mapped <= 512.5).all(),
                f'{swap}: part of the swap crop maps outside the enhancer box, '
                f'so the trip back cannot restore it — corners {mapped.tolist()}')


class Declarations(unittest.TestCase):
    FFHQ = ('Enhance_CodeFormer', 'Enhance_GFPGAN',
            'Enhance_RestoreFormerPPlus', 'Enhance_GPEN')

    def test_every_ffhq_restorer_declares_its_template(self):
        for name in self.FFHQ:
            src = open(os.path.join(PROC, f'{name}.py'), encoding='utf-8').read()
            self.assertRegex(
                src, r"model_template\s*=\s*'ffhq_512'",
                f'{name} must declare the alignment it was trained on, or '
                'ProcessMgr silently leaves its crop in swap-template space')

    def test_the_paste_scale_can_never_be_zero(self):
        """paste_upscale does `M * scale_factor`; a 0 blanks the face."""
        for name in ('Enhance_CodeFormer', 'Enhance_GFPGAN',
                     'Enhance_RestoreFormerPPlus', 'Enhance_DMDNet',
                     'Enhance_KEEP'):
            src = open(os.path.join(PROC, f'{name}.py'), encoding='utf-8').read()
            bare = re.findall(r'scale_factor\s*=\s*int\(', src)
            self.assertEqual(
                bare, [],
                f'{name} computes scale_factor with a bare int(); use '
                'max(1, int(...)) so a model smaller than the crop cannot '
                'collapse the paste matrix')

    def test_realignment_is_off_by_default(self):
        """It changes every render, so a fresh install must not turn it on."""
        self.assertIs(getattr(g, 'enhancer_align'), False)
        self.assertIs(getattr(g, 'color_match_after_enhance'), False)


if __name__ == '__main__':
    unittest.main()
