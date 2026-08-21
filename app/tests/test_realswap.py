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
        # Normalised by its own peak: this test is about WHERE the band is, and
        # the peak is a separately-measured tuning value (see TestBandOpacity).
        # Reading raw values here silently pinned "the band is fully opaque".
        m = FaceSwapInsightFace._eye_region_mask(256, 'arcface')
        m = m / float(m.max())
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

    def test_band_is_the_15_percent_the_user_asked_for(self):
        # The user's split: hififace 15%, hyperswap 85%. Measured against a FACE
        # OVAL, not the crop -- the crop is only 36.1% face, the rest hair and
        # background, so quoting "% of the crop" understates the split by ~2.8x.
        import cv2
        from roop.face_util import swap_template_points
        size = 256
        m = FaceSwapInsightFace._eye_region_mask(size, 'arcface')
        pts = np.asarray(swap_template_points(size, 'arcface'), dtype=np.float32)
        sep = float(np.linalg.norm(pts[1] - pts[0]))
        eye_mid, mouth_mid = (pts[0] + pts[1]) / 2.0, (pts[3] + pts[4]) / 2.0
        cx = (eye_mid[0] + mouth_mid[0]) / 2.0
        cy = (eye_mid[1] + mouth_mid[1]) / 2.0 + 0.10 * sep
        oval = np.zeros((size, size), np.uint8)
        cv2.ellipse(oval, (int(cx), int(cy)),
                    (int(1.05 * sep), int(1.45 * sep)), 0, 0, 360, 1, -1)
        share = float((m * oval).sum()) / float(oval.sum())
        self.assertTrue(0.12 < share < 0.18,
                        f'hififace covers {share:.1%} of the face; the brief is '
                        f'15% (hyperswap 85%)')

    def test_band_does_not_cross_the_nose_bridge(self):
        # The eye centres sit at +-0.5 sep, so a ring wider than ~0.45 runs the
        # two ellipses into each other over the bridge -- which is the base
        # model's outright, along with the nose below it.
        self.assertLess(FaceSwapInsightFace._OUTER_X, 0.45)

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
        # base=0 and other=1, so `out` IS the mask; normalise for the same
        # reason as above -- the question is which model owns which pixel.
        out = out / float(out.max())
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


class _FakeSecondary:
    """Enough of a processor for `_run_secondary`: it swaps, and it stashes a
    mask of its own that the caller is expected to drain."""

    loaded_model_key = 'hififace'
    model_mean = [0.5, 0.5, 0.5]
    model_standard_deviation = [0.5, 0.5, 0.5]
    model_denormalize = True

    def __init__(self):
        self.masks = None
        self.runs = 0

    def Run(self, source_face, target_face, blob):
        self.runs += 1
        self.masks = [np.full((256, 256), 0.25, np.float32)]   # "my" face mask
        return np.zeros((3, 256, 256), np.float32)

    def take_masks(self):
        m, self.masks = self.masks, None
        return m


class TestPublishedMaskIsThePrimarySown(unittest.TestCase):
    """The mask ProcessMgr pastes through must describe the face it is pasting.

    Under the composite that face is the PRIMARY's everywhere but the ~6% eye
    band, so the primary's mask is the one to publish. `_run_secondary` used to
    overwrite it with the secondary's — correct under the pose ROUTER it was
    written for, where the whole face was the secondary's, and left behind when
    the router became a region composite.
    """

    def _primed(self):
        p = _proc()
        p.secondary = _FakeSecondary()
        p.model_mean = [0.5, 0.5, 0.5]
        p.model_standard_deviation = [0.5, 0.5, 0.5]
        p.model_denormalize = True
        primary_mask = np.full((256, 256), 0.75, np.float32)
        p._mask_tls.masks = [primary_mask]                     # as Run stashes it
        return p, primary_mask

    def test_primary_mask_survives_the_secondary(self):
        p, primary_mask = self._primed()
        out = p._run_secondary(_Face(), _Face(), np.zeros((1, 3, 256, 256), np.float32))
        self.assertIsNotNone(out, 'the secondary should have produced a swap')
        published = p.take_masks()
        self.assertIsNotNone(published, 'the primary mask was dropped')
        self.assertTrue(np.allclose(published[0], 0.75),
                        "the secondary's mask was published in the primary's place")

    def test_secondary_mask_is_drained_not_left_for_the_next_face(self):
        # Left stashed on the sub-processor it would be served to a later face
        # on the same thread.
        p, _ = self._primed()
        p._run_secondary(_Face(), _Face(), np.zeros((1, 3, 256, 256), np.float32))
        self.assertIsNone(p.secondary.masks, "the secondary's mask was not drained")

    def test_a_maskless_secondary_does_not_null_the_primary_s(self):
        p, _ = self._primed()

        def _no_mask(source_face, target_face, blob):
            p.secondary.masks = None
            return np.zeros((3, 256, 256), np.float32)

        p.secondary.Run = _no_mask
        p._run_secondary(_Face(), _Face(), np.zeros((1, 3, 256, 256), np.float32))
        published = p.take_masks()
        self.assertIsNotNone(published,
                             'a secondary with no mask nulled the primary\'s too')
        self.assertTrue(np.allclose(published[0], 0.75))


class TestBatchedPathsStillComposite(unittest.TestCase):
    """A batched path that runs only the primary would BE a different swap model.

    Note this pins a property, not a fix: `Load` already sets
    `_batch_unsupported = True` whenever a secondary is configured, so the
    composite has always taken the sequential fallback. These tests exist so
    that if that flag is ever cleared — it means "this model cannot batch",
    which is a different proposition and could legitimately change — the eye
    band does not start disappearing silently. `mix_summary` could not report
    it either: it counts only faces that reached `_mix_outputs`, so it would
    print nothing rather than "0 of N".
    """

    def _spy(self):
        p = _proc()
        p.secondary = _FakeSecondary()
        calls = []
        p._sequential_fallback = lambda reqs: calls.append(reqs) or [
            r[2][0] for r in reqs]
        return p, calls

    def test_runbatch_defers_to_the_compositing_path(self):
        p, calls = self._spy()
        crops = [np.zeros((1, 3, 256, 256), np.float32) for _ in range(3)]
        p.RunBatch(_Face(), _Face(), crops)
        self.assertEqual(len(calls), 1, 'RunBatch ran the primary alone')
        self.assertEqual(len(calls[0]), 3)

    def test_runbatchmulti_defers_to_the_compositing_path(self):
        p, calls = self._spy()
        reqs = [(_Face(), _Face(), np.zeros((1, 3, 256, 256), np.float32))
                for _ in range(2)]
        p.RunBatchMulti(reqs)
        self.assertEqual(len(calls), 1, 'RunBatchMulti ran the primary alone')

    def test_single_net_models_keep_their_batching(self):
        # The opt-out is the composite's, not every model's: batching is a real
        # throughput win and nothing about one net needs it disabled.
        p = FaceSwapInsightFace()
        self.assertIsNone(p.secondary)
        called = []
        p._sequential_fallback = lambda reqs: called.append(reqs) or []
        p._batch_unsupported = False
        p._compute_source_input = lambda src: None      # stops before inference
        p.RunBatch(_Face(), _Face(), [np.zeros((1, 3, 256, 256), np.float32)])
        self.assertEqual(called, [], 'a single-net model lost its batching')


class TestBandOpacity(unittest.TestCase):
    """The band's opacity is the lever that reaches the identity cost.

    Measured on the yaw sweep at production settings: identity cost is strongly
    CONVEX in opacity -- at alpha 0.25 identity equals hyperswap's outright
    (0.7898 vs 0.7897) while eyelid drift still improves, and at 0.50 it costs
    15% of the full band's identity hit for 49% of its gain. Eye gain is roughly
    linear; ghost cost is concave (47% of the sharpness loss has arrived by
    0.25), which is why ghost is a BLENDING problem rather than a quantity one.
    """

    def setUp(self):
        self._alpha = FaceSwapInsightFace._EYE_ALPHA
        FaceSwapInsightFace._EYE_MASK_CACHE.clear()

    def tearDown(self):
        FaceSwapInsightFace._EYE_ALPHA = self._alpha
        FaceSwapInsightFace._EYE_MASK_CACHE.clear()

    def test_default_is_FULL_for_the_lash_band(self):
        # Opacity 0.5 belonged to the wide LID RING, where halving it bought
        # back most of the identity that band cost. The lash band is a different
        # instrument: it costs little because it is tiny and avoids periocular
        # skin, and the requirement is that the lashes ARE hififace's -- which a
        # 50% blend does not satisfy. So full opacity over a much smaller area.
        self.assertEqual(FaceSwapInsightFace._EYE_ALPHA, 1.0)

    def test_alpha_scales_the_peak(self):
        FaceSwapInsightFace._EYE_ALPHA = 0.5
        m = FaceSwapInsightFace._eye_region_mask(256, 'arcface')
        self.assertAlmostEqual(float(m.max()), 0.5, places=3)

    def test_alpha_does_not_change_the_band_SHAPE(self):
        # Opacity and area are different levers, and the whole point of this one
        # is that it is not a smaller band: shrinking the area was measured and
        # recovered only 29% of the identity its area predicted. So the support
        # must be identical and only the amplitude may move.
        FaceSwapInsightFace._EYE_ALPHA = 1.0
        full = FaceSwapInsightFace._eye_region_mask(256, 'arcface').copy()
        FaceSwapInsightFace._EYE_MASK_CACHE.clear()
        FaceSwapInsightFace._EYE_ALPHA = 0.25
        part = FaceSwapInsightFace._eye_region_mask(256, 'arcface')
        self.assertEqual(float((full > 1e-6).sum()), float((part > 1e-6).sum()),
                         'opacity changed the band footprint; that is the AREA '
                         'lever, which is already measured and rejected')
        import numpy as _np
        self.assertTrue(_np.allclose(part, full * 0.25, atol=1e-6),
                        'the edge profile must be unchanged, only the peak')

    def test_two_opacities_cannot_share_a_cache_entry(self):
        FaceSwapInsightFace._EYE_ALPHA = 1.0
        a = FaceSwapInsightFace._eye_region_mask(256, 'arcface')
        FaceSwapInsightFace._EYE_ALPHA = 0.25
        b = FaceSwapInsightFace._eye_region_mask(256, 'arcface')
        self.assertNotAlmostEqual(float(a.max()), float(b.max()), places=3,
                                  msg='the cache served one opacity for another')

    def test_zero_alpha_is_the_base_model_alone(self):
        FaceSwapInsightFace._EYE_ALPHA = 0.0
        p = _proc()
        base = np.zeros((3, 256, 256), np.float32)
        other = np.ones((3, 256, 256), np.float32)
        out = p._mix_outputs(base, other, 256)
        self.assertLess(float(out.max()), 1e-6,
                        'alpha 0 must reduce exactly to the base model')



class TestFirstGenerationCrop(unittest.TestCase):
    """The secondary net is cropped from the PLATE, not from the primary's crop.

    Its pixels used to survive three resamples -- plate -> arcface crop, that
    crop -> mtcnn_512, and back -- so the second net never saw plate detail at
    all. Cropping from the plate removes the one upstream of the net.

    The registration must not move a pixel while doing it. It provably does not:
    the old transform is `inv(_crop_to_crop) = M_a . inv(M_b)` and the new one is
    composed as exactly that, so this is a refactor of WHERE the pixels are
    sampled from, not of where they land.
    """

    SIZE = 256

    def _norms(self):
        from roop.face_util import estimate_norm
        kps = FRONTAL * (self.SIZE / 128.0)
        return kps, (estimate_norm(kps, self.SIZE, 'arcface'),
                     estimate_norm(kps, self.SIZE, 'mtcnn_512'))

    def test_plate_path_registers_identically_to_the_derived_path(self):
        # THE test. If these ever diverge, the eye band is landing somewhere
        # other than where the base model's eyes are, and every measurement of
        # the band is measuring a misregistration instead.
        import cv2
        kps, (M_a, M_b) = self._norms()
        derived = cv2.invertAffineTransform(
            FaceSwapInsightFace._crop_to_crop(kps, self.SIZE, 'arcface',
                                              'mtcnn_512'))
        plate = FaceSwapInsightFace._compose_affine(
            M_a, cv2.invertAffineTransform(M_b))
        np.testing.assert_allclose(plate, derived, atol=1e-4)

    def test_compose_affine_is_inner_then_outer(self):
        outer = np.float32([[2, 0, 5], [0, 2, -3]])
        inner = np.float32([[0, -1, 4], [1, 0, 7]])
        c = FaceSwapInsightFace._compose_affine(outer, inner)
        pt = np.float32([3, 11])
        step = inner[:, :2] @ pt + inner[:, 2]
        both = outer[:, :2] @ step + outer[:, 2]
        np.testing.assert_allclose(c[:, :2] @ pt + c[:, 2], both, atol=1e-4)

    def test_prepare_blob_matches_the_pipeline_s_own(self):
        # A differently-scaled input is invisible -- the net still returns a
        # face, just a worse one -- so this is pinned against the real thing.
        from roop.procmgr_tiling import PixelBoostMixin as _T
        rng = np.random.default_rng(3)
        crop = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)

        class _M:
            model_mean = [0.5, 0.5, 0.5]
            model_standard_deviation = [0.5, 0.5, 0.5]
        mine = FaceSwapInsightFace._prepare_blob(crop, _M)
        theirs = _T.prepare_crop_frame(None, crop.astype(np.float64), _M)
        np.testing.assert_allclose(mine, theirs, atol=1e-5)

    def test_context_is_withheld_when_the_crop_is_not_a_plain_align(self):
        # ProcessMgr's call, not the processor's: pixel boost and frontalization
        # both make the primary crop something other than align_crop(plate).
        p = _proc()
        p.set_plate_context(np.zeros((8, 8, 3), np.uint8), np.eye(2, 3), False)
        self.assertIsNone(p._plate_tls.ctx)
        p.set_plate_context(np.zeros((8, 8, 3), np.uint8), np.eye(2, 3), True)
        self.assertIsNotNone(p._plate_tls.ctx)

    def test_context_is_cleared_so_it_cannot_serve_the_next_face(self):
        p = _proc()
        p.set_plate_context(np.zeros((8, 8, 3), np.uint8), np.eye(2, 3), True)
        p.clear_plate_context()
        self.assertIsNone(p._plate_tls.ctx)

    def test_run_secondary_uses_the_plate_when_it_has_one(self):
        from roop.face_util import estimate_norm
        rng = np.random.default_rng(5)
        plate = rng.integers(0, 256, (400, 400, 3), dtype=np.uint8)
        kps = FRONTAL * 2.0 + 60.0
        p = _proc()
        p.secondary = _FakeSecondary()
        p.model_mean = [0.5, 0.5, 0.5]
        p.model_standard_deviation = [0.5, 0.5, 0.5]
        p.model_denormalize = True
        M_a = estimate_norm(kps, 256, 'arcface')
        p.set_plate_context(plate, M_a, True)
        out = p._run_secondary(_Face(kps=kps), _Face(kps=kps),
                               np.zeros((1, 3, 256, 256), np.float32))
        self.assertIsNotNone(out)
        self.assertEqual(tuple(out.shape), (3, 256, 256))
        self.assertEqual(p.secondary.runs, 1)

    def test_without_a_plate_it_still_works(self):
        # The fallback is the shipped path and has to keep working: pixel boost
        # and frontalization both land here.
        p = _proc()
        p.secondary = _FakeSecondary()
        p.model_mean = [0.5, 0.5, 0.5]
        p.model_standard_deviation = [0.5, 0.5, 0.5]
        p.model_denormalize = True
        p.clear_plate_context()
        out = p._run_secondary(_Face(), _Face(),
                               np.zeros((1, 3, 256, 256), np.float32))
        self.assertIsNotNone(out)
        self.assertEqual(tuple(out.shape), (3, 256, 256))



class TestCropSourceIsReported(unittest.TestCase):
    """A silent fallback to the derived crop looks exactly like the fix working.

    This processor has been bitten twice by exactly that: the pose router sent
    69% of a clip's faces to the wrong net while the audit read 100%/100%, and
    the batched paths dropped the composite entirely while mix_summary printed
    nothing rather than "0 of N".
    """

    def _p(self):
        p = _proc()
        p.loaded_model_key = 'realswap'
        p.secondary = _FakeSecondary()
        p.model_mean = [0.5, 0.5, 0.5]
        p.model_standard_deviation = [0.5, 0.5, 0.5]
        p.model_denormalize = True
        return p

    def test_counts_the_plate_path(self):
        from roop.face_util import estimate_norm
        rng = np.random.default_rng(11)
        plate = rng.integers(0, 256, (400, 400, 3), dtype=np.uint8)
        kps = FRONTAL * 2.0 + 60.0
        p = self._p()
        p.set_plate_context(plate, estimate_norm(kps, 256, 'arcface'), True)
        p._run_secondary(_Face(kps=kps), _Face(kps=kps),
                         np.zeros((1, 3, 256, 256), np.float32))
        self.assertEqual((p._plate_crops, p._derived_crops), (1, 0))

    def test_counts_the_fallback_separately(self):
        p = self._p()
        p.clear_plate_context()
        p._run_secondary(_Face(), _Face(), np.zeros((1, 3, 256, 256), np.float32))
        self.assertEqual((p._plate_crops, p._derived_crops), (0, 1))

    def test_summary_names_the_split_and_the_opacity(self):
        p = self._p()
        p.clear_plate_context()
        p._run_secondary(_Face(), _Face(), np.zeros((1, 3, 256, 256), np.float32))
        z = np.zeros((3, 256, 256), np.float32)
        p._mix_outputs(z, z, 256)
        line = p.mix_summary()
        self.assertIn('0% cropped from the plate', line)
        self.assertIn('fell back to the derived crop', line)
        self.assertIn('opacity 1', line)



class TestLashBandTargetsOnlyTheLashLine(unittest.TestCase):
    """hififace gets the lashes; hyperswap keeps the identity.

    The user's requirement, given directly: RealSwap's identity must match
    hyperswap's and only the EYELASHES come from hififace. So the band has to
    sit ON the lid margin, where lashes grow, and stay off three things that
    would cost identity -- the eye aperture, the lid/socket above it, and the
    brow. ArcFace reads identity most densely from exactly that periocular
    skin, which is why the wide lid ring it replaced cost 0.03 of identity.
    """

    def setUp(self):
        FaceSwapInsightFace._EYE_MASK_CACHE.clear()

    def test_lash_line_is_the_secondary_s_and_the_rest_is_not(self):
        from roop.face_util import swap_template_points
        m = FaceSwapInsightFace._eye_region_mask(256, 'arcface')
        peak = float(m.max())
        pts = np.asarray(swap_template_points(256, 'arcface'), dtype=np.float32)
        sep = float(np.linalg.norm(pts[1] - pts[0]))
        for eye in (pts[0], pts[1]):
            x, y = int(eye[0]), int(eye[1])
            self.assertLess(m[y, x], 0.05 * peak,
                            'the eye APERTURE must stay with the base model')
            self.assertGreater(m[y - int(0.10 * sep), x], 0.5 * peak,
                               'the UPPER LASH LINE must come from the secondary')
            self.assertLess(m[y - int(0.42 * sep), x], 0.10 * peak,
                            'the BROW belongs to the base model; the ring '
                            'reach it')
        for name, q in (('nose', pts[2]), ('mouth-left', pts[3])):
            self.assertLess(m[int(q[1]), int(q[0])], 0.02 * peak,
                            f'{name} belongs wholly to the base model')

    def test_lashes_are_FULLY_the_secondary_s_not_a_blend(self):
        # "Eyelashes must match hififace" is not satisfied by a 50% mix of two
        # models' lashes -- that is the doubling the whole design forbids.
        m = FaceSwapInsightFace._eye_region_mask(256, 'arcface')
        self.assertGreater(float(m.max()), 0.99)
        self.assertGreater(float((m > 0.95).sum()), 200,
                           'no pixel region is fully the secondary; the lash '
                           'line is a blend, not the secondary alone')


if __name__ == '__main__':
    unittest.main()
