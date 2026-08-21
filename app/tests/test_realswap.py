"""RealSwap — two swap nets behind one model entry, composited by region.

hyperswap is the base for the whole face; hififace contributes only the eyelid
and eyelash band. That split is by FEATURE, from the user's own brief: hyperswap
is the better model for faithfulness to the faceset and for the nose, eye
interior, mouth, chin and cheeks, hififace for eyelids, eyelashes and
expression.

It replaced a pose ROUTER that sent whole faces to hififace past ~80 deg of yaw.
That was measured on real contact footage and cost 0.26 of faceset identity
(0.168 vs hyperswap's 0.424) for no gain in sharpness. Do not revive it.

What has to stay true is mostly about the seams between two models that were
never meant to share a pipeline:

  * the two nets do NOT share an alignment (arcface vs mtcnn_512), so the
    second's crop is derived from the first's — and the round trip must not
    resample the detail out of the face (it did: bilinear kept 35.8%);
  * the composite band must contain the eyes and NOTHING else — the base model
    owns the nose, mouth and chin outright — and its edge must be feathered, on
    skin, so no feature can double (`angle-handling-three-layers`);
  * the secondary must not be able to recurse into another secondary.

These tests pin the MECHANISM, not the constants — the band's geometry is a
tuning surface and is expected to move.
"""
import os
import re
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from roop.processors.FaceSwapInsightFace import (              # noqa: E402
    SWAP_MODELS, FaceSwapInsightFace,
)

_API = os.path.join(os.path.dirname(__file__), '..', 'api.py')

# A frontal and a profile 5-point set. Real keypoint layouts: eyes, nose, mouth
# corners. The profile one has the nose pushed toward one eye, which is what a
# turned head does to this projection.
FRONTAL = np.array([[38., 52.], [74., 52.], [56., 72.], [41., 92.], [71., 92.]],
                   dtype=np.float32)
PROFILE = np.array([[38., 52.], [74., 52.], [69., 74.], [48., 92.], [72., 91.]],
                   dtype=np.float32)


class _Face(dict):
    """Stand-in for an insightface Face: attribute kps, dict-like .get."""

    def __init__(self, kps=None, track=None):
        super().__init__()
        self.kps = FRONTAL if kps is None else kps
        if track is not None:
            self['_track_id'] = track


def _proc():
    """A processor with the composite state initialised but no model loaded."""
    p = FaceSwapInsightFace()
    p.secondary = object()          # only its presence matters to the composite
    p.secondary_template = 'mtcnn_512'
    p.model_template = 'arcface'
    return p


class TestSpec(unittest.TestCase):

    def test_secondary_names_a_real_model(self):
        sec = SWAP_MODELS['realswap'].get('secondary')
        self.assertIn(sec, SWAP_MODELS,
                      'realswap.secondary must name an entry in this same table')

    def test_secondary_cannot_recurse(self):
        sec = SWAP_MODELS['realswap']['secondary']
        self.assertIsNone(SWAP_MODELS[sec].get('secondary'),
                          'the nested load would recurse without bound')

    def test_published_contract_is_the_primary_s(self):
        # ProcessMgr aligns, normalizes and pastes from the PUBLISHED contract,
        # so it has to describe the crop space the processor actually returns —
        # the primary's. A secondary with a different output size or range would
        # be re-warped into it, never published in place of it.
        spec = SWAP_MODELS['realswap']
        self.assertEqual(spec['template'], 'arcface')
        self.assertEqual(spec['output_size'], 256)

    def test_no_verify_tol_until_one_is_measured(self):
        # A mixed output's clean-swap band has not been measured. Inheriting
        # hififace's tighter tolerance because half the faces come from it would
        # be a fitted-looking number with nothing behind it.
        self.assertIsNone(SWAP_MODELS['realswap'].get('verify_tol'))

    def test_registered_in_the_api_model_list(self):
        # A swap model absent from this list cannot be selected, and the model
        # table alone gives no hint that it is unreachable.
        src = open(_API, encoding='utf-8').read()
        block = re.search(r'"swap_models":\s*\[(.*?)\]', src, re.S)
        self.assertIsNotNone(block, 'swap_models list not found in api.py')
        self.assertIn('"realswap"', block.group(1))


class TestCropTransform(unittest.TestCase):

    def test_round_trips_to_identity(self):
        M = FaceSwapInsightFace._crop_to_crop(FRONTAL, 256, 'arcface', 'mtcnn_512')
        import cv2
        back = cv2.invertAffineTransform(M)
        composed = (np.vstack([M, [0, 0, 1]]) @ np.vstack([back, [0, 0, 1]]))
        np.testing.assert_allclose(composed, np.eye(3), atol=1e-4)

    def test_is_stable_across_faces(self):
        # Pins the measurement that corrected this function's own docstring. The
        # transform was assumed to be strongly face-dependent (a profile fits a
        # frontal 5-point template far worse than a frontal head does); measured,
        # the residual enters only at second order and the matrix moves by ~1e-05
        # on a 256px crop. That is what keeps the composite stable: the crop the
        # secondary net receives does not drift as the head turns, so the eyelid
        # band it contributes stays registered with the base underneath it.
        a = FaceSwapInsightFace._crop_to_crop(FRONTAL, 256, 'arcface', 'mtcnn_512')
        b = FaceSwapInsightFace._crop_to_crop(PROFILE, 256, 'arcface', 'mtcnn_512')
        self.assertLess(np.abs(a - b).max(), 1e-3,
                        'if this ever becomes strongly face-dependent, the '
                        'derived crop is drifting with pose and the routed '
                        'swap is being fed a moving target')

    def test_identity_between_a_template_and_itself(self):
        M = FaceSwapInsightFace._crop_to_crop(PROFILE, 256, 'arcface', 'arcface')
        np.testing.assert_allclose(M, np.float32([[1, 0, 0], [0, 1, 0]]), atol=1e-4)

    def test_round_trip_keeps_most_of_the_detail(self):
        # The secondary's output is resampled twice, into its template space and
        # back. With INTER_LINEAR that pair of warps kept only 35.8% of the
        # detail of a real profile crop and shipped as a visible blur; LANCZOS4
        # keeps 81.6%. Pinned on noise, which is the hardest case for a
        # resampler, so the bar is deliberately below the measured figure.
        import cv2
        rng = np.random.default_rng(0)
        img = rng.random((3, 256, 256)).astype(np.float32)
        M = FaceSwapInsightFace._crop_to_crop(FRONTAL, 256, 'arcface', 'mtcnn_512')
        there = FaceSwapInsightFace._warp_chw(img, M, 256)
        back = FaceSwapInsightFace._warp_chw(there, cv2.invertAffineTransform(M), 256)
        # Compare variance in the interior, away from the replicated border.
        c = (slice(None), slice(32, 224), slice(32, 224))
        self.assertGreater(back[c].var() / img[c].var(), 0.30,
                           'the template round trip is destroying detail — check '
                           'the interpolation flag before blaming the swap net')

    def test_warp_preserves_layout(self):
        chw = np.random.rand(3, 64, 64).astype(np.float32)
        M = np.float32([[1, 0, 0], [0, 1, 0]])
        out = FaceSwapInsightFace._warp_chw(chw, M, 64)
        self.assertEqual(out.shape, (3, 64, 64))
        np.testing.assert_allclose(out, chw, atol=1e-5)


class TestRenormalize(unittest.TestCase):

    class _M:
        def __init__(self, mean, std):
            self.model_mean = mean
            self.model_standard_deviation = std

    def test_noop_for_the_shipped_pairing(self):
        p = _proc()
        same = self._M([0.5] * 3, [0.5] * 3)
        blob = np.random.rand(1, 3, 8, 8).astype(np.float32)
        out = p._renormalize(blob, same, same)
        self.assertIs(out, blob, 'an equal pairing should not pay for a copy')

    def test_converts_between_different_ranges(self):
        p = _proc()
        src = self._M([0.5] * 3, [0.5] * 3)        # [-1,1]
        dst = self._M([0.0] * 3, [1.0] * 3)        # [0,1]
        blob = np.zeros((1, 3, 4, 4), np.float32)  # mid grey in [-1,1]
        np.testing.assert_allclose(p._renormalize(blob, src, dst), 0.5, atol=1e-6)


class TestEyeBandComposite(unittest.TestCase):
    """hyperswap is the base; hififace contributes only the eyelid band.

    The user's brief, by feature rather than by pose: hyperswap is better at
    faithfulness to the faceset and at the nose, eye interior, mouth, chin and
    cheeks; hififace is better at eyelids, eyelashes and expression. So the
    composite must leave the great majority of the face untouched.
    """

    def test_band_is_the_lids_and_NOT_the_eye_itself(self):
        # The band is an annulus. The lid margins and lashes are the secondary's;
        # the eye aperture inside them stays the base model's, because ArcFace
        # draws more identity per pixel from the periocular interior than from
        # anywhere else — a disc over the whole eye took 16.6% of the crop area
        # and 34% of the available identity gap.
        from roop.face_util import swap_template_points
        m = FaceSwapInsightFace._eye_region_mask(256, 'arcface')
        pts = np.asarray(swap_template_points(256, 'arcface'), dtype=np.float32)
        sep = float(np.linalg.norm(pts[1] - pts[0]))
        for eye in (pts[0], pts[1]):
            x, y = int(eye[0]), int(eye[1])
            self.assertLess(m[y, x], 0.05,
                            'the eye APERTURE must stay with the base model')
            self.assertGreater(m[y - int(0.13 * sep), x], 0.5,
                               'the upper lid must come from the secondary')
            self.assertGreater(m[y + int(0.11 * sep), x], 0.5,
                               'the lower lid must come from the secondary')
        for name, q in (('nose', pts[2]), ('mouth-left', pts[3]),
                        ('mouth-right', pts[4])):
            self.assertLess(m[int(q[1]), int(q[0])], 0.02,
                            f'{name} belongs wholly to the base model')

    def test_band_is_a_minority_of_the_crop(self):
        # The brief is 80-85% base model. The mask is feathered, so its MEAN is
        # the honest measure of how much of the face is not purely the base.
        cov = float(FaceSwapInsightFace._eye_region_mask(256, 'arcface').mean())
        self.assertTrue(0.02 < cov < 0.25,
                        f'eye band covers {cov:.1%} of the crop; the base model '
                        f'must remain the great majority of the face')

    def test_edges_are_feathered_not_hard(self):
        # A hard edge between two swapped faces is a visible seam. The band has
        # to arrive and leave gradually, over skin.
        m = FaceSwapInsightFace._eye_region_mask(256, 'arcface')
        mid = m[(m > 0.02) & (m < 0.98)]
        self.assertGreater(mid.size, 400, 'the band has no transition zone')

    def test_mask_is_cached_per_size(self):
        a = FaceSwapInsightFace._eye_region_mask(256, 'arcface')
        b = FaceSwapInsightFace._eye_region_mask(256, 'arcface')
        self.assertIs(a, b, 'the crop IS the template, so this is face-'
                            'independent and must be built once')

    def test_mix_takes_base_outside_and_secondary_inside(self):
        p = _proc()
        p.model_template = 'arcface'
        base = np.zeros((3, 256, 256), np.float32)
        other = np.ones((3, 256, 256), np.float32)
        out = p._mix_outputs(base, other, 256)
        from roop.face_util import swap_template_points
        pts = np.asarray(swap_template_points(256, 'arcface'), dtype=np.float32)
        sep = float(np.linalg.norm(pts[1] - pts[0]))
        lid = (int(pts[0][1] - 0.13 * sep), int(pts[0][0]))
        self.assertGreater(out[0, lid[0], lid[1]], 0.5, 'lid comes from the secondary')
        self.assertLess(out[0, int(pts[0][1]), int(pts[0][0])], 0.05,
                        'the eye aperture stays with the base')
        self.assertLess(out[0, int(pts[2][1]), int(pts[2][0])], 0.02)
        self.assertLess(out[0, 8, 8], 0.02, 'crop corners are pure base')

    def test_counts_composited_faces(self):
        p = _proc()
        p.loaded_model_key, p.secondary = 'realswap', _proc()
        p.secondary.loaded_model_key = 'hififace'
        z = np.zeros((3, 256, 256), np.float32)
        p._mix_outputs(z, z, 256)
        p._mix_outputs(z, z, 256)
        self.assertEqual((p._mixed_faces, p._seen_faces), (2, 2))
        self.assertIn('2 of 2', p.mix_summary())

    def test_single_net_models_never_composite(self):
        p = FaceSwapInsightFace()
        self.assertIsNone(p.secondary)
        self.assertIsNone(p.mix_summary())


if __name__ == '__main__':
    unittest.main()
