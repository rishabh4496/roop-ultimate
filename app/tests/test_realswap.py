"""RealSwap — two swap nets behind one model entry, routed per face.

RealSwap loads hyperswap and hififace together and sends each face to ONE of
them. What has to stay true is mostly about the seams between two models that
were never meant to share a pipeline:

  * the two nets do NOT share an alignment (arcface vs mtcnn_512), so the second
    one's crop has to be derived from the first's — and derived PER FACE, not
    from a fixed template-to-template matrix, because estimate_norm is a
    least-squares fit whose residual depends on the face;
  * the routing decision has to be latched along the track. A bare threshold
    makes a head hovering near it alternate between two models with different
    identity strength frame to frame, which is the flicker requirement 11 is
    about, and the cross-fade that would otherwise smooth it is the uniform
    blend that doubles every feature (`angle-handling-three-layers`);
  * the latch must not survive into the next video, whose track ids restart;
  * the secondary must not be able to recurse into another secondary.

The numbers behind the threshold are in FaceSwapInsightFace._SECONDARY_ENTER_YAW
and the yaw-sweep table in RECODE_STATUS.md. These tests pin the MECHANISM, not
the constants — the constants are a measurement and are expected to move.
"""
import os
import re
import sys
import unittest
from unittest import mock

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
    """A processor with the routing state initialised but no model loaded."""
    p = FaceSwapInsightFace()
    p.secondary = object()          # only its presence matters to the routing
    p.secondary_template = 'mtcnn_512'
    p.model_template = 'arcface'
    return p


def _yaw(deg):
    """Patch the pose solver to report |yaw| = deg."""
    return mock.patch('roop.face_util.solve_pose_5pt',
                      return_value=(float(deg), 0.0, 0.0))


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
        # on a 256px crop. That is what makes the profile route safe: the crop
        # the secondary net receives does not drift as the head turns, which is
        # exactly where RealSwap sends faces.
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
        # A routed face is resampled twice, into the secondary's template space
        # and back. With INTER_LINEAR that pair of warps kept only 35.8% of the
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


class TestRouting(unittest.TestCase):

    def test_enters_only_past_the_entry_angle(self):
        p = _proc()
        with _yaw(p._SECONDARY_ENTER_YAW - 1):
            self.assertFalse(p._route_to_secondary(_Face(track=1)))
        with _yaw(p._SECONDARY_ENTER_YAW + 1):
            self.assertTrue(p._route_to_secondary(_Face(track=2)))

    def test_latch_holds_through_the_hysteresis_band(self):
        p = _proc()
        f = _Face(track=7)
        with _yaw(p._SECONDARY_ENTER_YAW + 5):
            self.assertTrue(p._route_to_secondary(f))
        # Between EXIT and ENTER: a bare threshold would drop back here, which
        # is the frame-to-frame alternation the latch exists to prevent.
        mid = (p._SECONDARY_ENTER_YAW + p._SECONDARY_EXIT_YAW) / 2.0
        with _yaw(mid):
            self.assertTrue(p._route_to_secondary(f))
        with _yaw(p._SECONDARY_EXIT_YAW - 1):
            self.assertFalse(p._route_to_secondary(f))

    def test_latch_is_per_track(self):
        p = _proc()
        with _yaw(p._SECONDARY_ENTER_YAW + 5):
            p._route_to_secondary(_Face(track='a'))
        mid = (p._SECONDARY_ENTER_YAW + p._SECONDARY_EXIT_YAW) / 2.0
        with _yaw(mid):
            self.assertTrue(p._route_to_secondary(_Face(track='a')))
            self.assertFalse(p._route_to_secondary(_Face(track='b')),
                             'one track latching must not route another')

    def test_untracked_faces_use_the_bare_threshold(self):
        p = _proc()
        mid = (p._SECONDARY_ENTER_YAW + p._SECONDARY_EXIT_YAW) / 2.0
        with _yaw(mid):
            self.assertFalse(p._route_to_secondary(_Face()))

    def test_no_pose_never_routes(self):
        p = _proc()
        with mock.patch('roop.face_util.solve_pose_5pt', return_value=None):
            self.assertFalse(p._route_to_secondary(_Face(track=1)))

    def test_single_net_models_never_route(self):
        p = FaceSwapInsightFace()          # secondary stays None
        self.assertIsNone(p.secondary)
        self.assertIsNone(p.route_summary())

    def test_counts_how_many_faces_routed(self):
        # Without this a clean result on real footage is ambiguous: "the routing
        # is safe" and "the routing never fired" read the same from the audit.
        p = _proc()
        p.loaded_model_key, p.secondary = 'realswap', _proc()
        p.secondary.loaded_model_key = 'hififace'
        with _yaw(p._SECONDARY_ENTER_YAW + 5):
            p._route_to_secondary(_Face(track=1))
        with _yaw(0.0):
            p._route_to_secondary(_Face(track=2))
            p._route_to_secondary(_Face(track=3))
        self.assertEqual((p._routed_faces, p._seen_faces), (1, 3))
        self.assertIn('1 of 3', p.route_summary())


if __name__ == '__main__':
    unittest.main()
