"""DeepFaceLab merger post-ops — that each one is inert until asked for.

The whole premise of this family is "costs nothing when off". That is only true
if every op returns its input UNTOUCHED at its neutral value, and if the chain
short-circuits before doing any work when they all are. Those are the tests
that matter here; the rest assert the property each op is supposed to have
rather than a specific pixel value, so a better implementation of the same idea
does not fail them.
"""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals                                    # noqa: E402
from roop.procmgr_merger import MergerMixin            # noqa: E402
from roop.procmgr_masking import MaskingMixin          # noqa: E402

MERGER_KEYS = ("merger_hist_match", "merger_sharpen", "merger_motion_blur",
               "merger_grain_match", "merger_degrade", "merger_clarity")


class _Merger(MergerMixin):
    """MergerMixin is a mixin; it needs no state, so a bare subclass is enough."""


def _crop(seed=0, size=64):
    """A crop with real structure — flat noise would hide a blur regression."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, : size // 2] = 90
    img[:, size // 2:] = 170
    img[size // 3: 2 * size // 3, :] = 40
    return np.clip(img.astype(np.int16) + rng.integers(-8, 9, img.shape), 0, 255).astype(np.uint8)


class NeutralIsAFreeNoOp(unittest.TestCase):
    """The cost story: at defaults nothing runs and nothing changes."""

    def setUp(self):
        self._saved = {k: getattr(roop.globals, k) for k in MERGER_KEYS}
        for k in MERGER_KEYS:
            setattr(roop.globals, k, 0.0)

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(roop.globals, k, v)

    def test_chain_returns_the_very_same_object(self):
        """Not merely equal — identical. An `is` check is what proves no copy,
        no convert and no allocation happened on the default path."""
        m, face, ref = _Merger(), _crop(1), _crop(2)
        self.assertIs(m.apply_merger_post(face, ref), face)

    def test_each_op_is_bit_identical_at_neutral(self):
        m, face, ref = _Merger(), _crop(1), _crop(2)
        for out in (m.apply_hist_match(face, ref, 0.0),
                    m.apply_sharpen(face, 0.0),
                    m.apply_motion_blur(face, ref, 0.0),
                    m.apply_grain_match(face, ref, 0.0),
                    m.apply_degrade(face, 0.0),
                    m.apply_clarity(face, 0.0)):
            np.testing.assert_array_equal(out, face)

    def test_one_enabled_knob_wakes_the_chain(self):
        """Guard the guard: if the short-circuit were too eager, every knob
        would be dead and the tests above would still pass."""
        m, face, ref = _Merger(), _crop(1), _crop(2)
        for key in MERGER_KEYS:
            setattr(roop.globals, key, 0.9)
            try:
                self.assertFalse(np.array_equal(m.apply_merger_post(face, ref), face),
                                 f"{key} = 0.9 changed nothing — the knob is not wired")
            finally:
                setattr(roop.globals, key, 0.0)


class OpsDoWhatTheyClaim(unittest.TestCase):
    def test_hist_match_moves_toward_the_reference(self):
        m = _Merger()
        face = np.full((32, 32, 3), 60, dtype=np.uint8)
        ref = np.full((32, 32, 3), 200, dtype=np.uint8)
        self.assertGreater(float(m.apply_hist_match(face, ref, 1.0).mean()),
                           float(face.mean()) + 50)

    def test_hist_match_strength_interpolates(self):
        m = _Merger()
        face = np.full((32, 32, 3), 60, dtype=np.uint8)
        ref = np.full((32, 32, 3), 200, dtype=np.uint8)
        half = float(m.apply_hist_match(face, ref, 0.5).mean())
        self.assertTrue(60 < half < 200, f"half strength landed at {half}")

    def test_sharpen_is_signed(self):
        """Positive raises local contrast, negative lowers it. Measured as the
        variance of the Laplacian, the standard sharpness proxy."""
        import cv2
        m, face = _Merger(), _crop(3)

        def sharpness(img):
            return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                       cv2.CV_32F).var())

        base = sharpness(face)
        self.assertGreater(sharpness(m.apply_sharpen(face, 0.8)), base)
        self.assertLess(sharpness(m.apply_sharpen(face, -0.8)), base)

    def test_degrade_softens_and_keeps_the_shape(self):
        import cv2
        m, face = _Merger(), _crop(4)
        out = m.apply_degrade(face, 1.0)
        self.assertEqual(out.shape, face.shape)
        lap = lambda i: float(cv2.Laplacian(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY),  # noqa: E731
                                            cv2.CV_32F).var())
        self.assertLess(lap(out), lap(face))

    def test_grain_scales_with_the_plates_own_noise(self):
        """The level is MEASURED, not dialled in: a clean plate must get less
        grain than a noisy one at the same strength."""
        m = _Merger()
        face = np.full((64, 64, 3), 128, dtype=np.uint8)
        clean = np.full((64, 64, 3), 128, dtype=np.uint8)
        rng = np.random.default_rng(7)
        noisy = np.clip(128 + rng.normal(0, 18, (64, 64, 3)), 0, 255).astype(np.uint8)

        def added(ref):
            out = m.apply_grain_match(face, ref, 1.0).astype(np.float32)
            return float(np.std(out - face.astype(np.float32)))

        self.assertLess(added(clean), added(noisy))

    def test_grain_is_temporally_independent(self):
        """Real sensor noise does not track the face. Identical grain frame to
        frame would read as a static texture stuck to the skin."""
        m = _Merger()
        face = np.full((64, 64, 3), 128, dtype=np.uint8)
        rng = np.random.default_rng(8)
        ref = np.clip(128 + rng.normal(0, 15, (64, 64, 3)), 0, 255).astype(np.uint8)
        a = m.apply_grain_match(face, ref, 1.0)
        b = m.apply_grain_match(face, ref, 1.0)
        self.assertFalse(np.array_equal(a, b))

    def test_motion_blur_follows_the_plates_axis(self):
        """A plate smeared horizontally must blur the face horizontally. The
        direction is measured, so this is the property that would break if the
        structure-tensor angle were off by 90 degrees."""
        import cv2
        m = _Merger()
        rng = np.random.default_rng(9)
        base = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        # Smear the reference along x only.
        kernel = np.zeros((15, 15), np.float32)
        kernel[7, :] = 1.0 / 15
        ref = cv2.filter2D(base, -1, kernel)

        out = m.apply_motion_blur(base, ref, 1.0).astype(np.float32)
        gray = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Horizontal blur destroys horizontal gradients and spares vertical ones.
        gx = float(np.mean(np.abs(np.diff(gray, axis=1))))
        gy = float(np.mean(np.abs(np.diff(gray, axis=0))))
        self.assertLess(gx, gy, "blur did not run along the reference's smear axis")

    def test_every_op_preserves_shape_and_dtype(self):
        m, face, ref = _Merger(), _crop(5), _crop(6)
        for out in (m.apply_hist_match(face, ref, 1.0),
                    m.apply_sharpen(face, 1.0),
                    m.apply_motion_blur(face, ref, 1.0),
                    m.apply_grain_match(face, ref, 1.0),
                    m.apply_degrade(face, 1.0)):
            self.assertEqual(out.shape, face.shape)
            self.assertEqual(out.dtype, np.uint8)


class EveryEntryPointPopulatesTheMergerGlobals(unittest.TestCase):
    """The merger ops read roop.globals, so anything that starts a render has to
    populate them or the whole stage is silently off.

    This is not hypothetical. `api.py` populated them and `tests/angle_bench.py`
    did not, and roop.globals' own defaults are 0.0 -- so every arm ever
    rendered through the bench harness ran with hist/sharpen/grain/degrade OFF
    while production ran 0.4 / 0.35 / 0.45 / 0. It was invisible until a feature
    was moved INTO that stage and measured as doing nothing at all. Source-level
    so the suite stays fast; importing either module drags in the model stack.
    """

    def _src(self, rel):
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(os.path.dirname(here), rel), encoding='utf-8') as f:
            return f.read()

    def test_both_entry_points_name_every_merger_key(self):
        for rel in ('api.py', 'tests/angle_bench.py'):
            src = self._src(rel)
            for key in MERGER_KEYS:
                self.assertIn(key, src,
                              f"{rel} never sets {key}, so that op is dead in "
                              f"anything it launches")

class ClarityIsTheMovedUltraMaxFilter(unittest.TestCase):
    """`merger_clarity` at 1.0 must reproduce what UltraMax used to apply.

    The filter was lifted out of Enhance_UltraMax so every enhancer can have it
    (measured: it was the ENTIRE difference between UltraMax and the CodeFormer
    it wraps). A move is only safe if it is a move, so the old implementation is
    inlined here as the reference and compared pixel for pixel.
    """

    def _ultramax_original(self, face_img):
        """Enhance_UltraMax._harmonize_face as it stood at 3c530f9."""
        lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_chan, A_chan, B_chan = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        blur_L_fine = cv2.GaussianBlur(L_chan, (0, 0), sigmaX=1.0)
        clarity_fine_clamped = np.clip(L_chan - blur_L_fine, -18.0, 18.0)
        lum_midtone = np.clip(np.sin(np.pi * np.clip(L_chan / 255.0, 0.0, 1.0)), 0.0, 1.0)
        lab[:, :, 0] = np.clip(
            L_chan + 0.32 * clarity_fine_clamped * (0.65 + 0.35 * lum_midtone), 0.0, 255.0)
        a_mean, b_mean = float(A_chan.mean()), float(B_chan.mean())
        lab[:, :, 1] = a_mean + np.tanh((A_chan - a_mean) / 16.0) * 14.5
        lab[:, :, 2] = b_mean + np.tanh((B_chan - b_mean) / 18.0) * 16.0
        return cv2.cvtColor(np.clip(lab, 0.0, 255.0).astype(np.uint8), cv2.COLOR_LAB2BGR)

    def test_strength_one_matches_the_original_implementation(self):
        m = _Merger()
        for seed in (1, 2, 3):
            face = _crop(seed, size=128)
            np.testing.assert_array_equal(
                m.apply_clarity(face, 1.0), self._ultramax_original(face),
                "merger_clarity 1.0 no longer reproduces the filter it replaced")

    def test_it_actually_sharpens_luminance(self):
        """Guard against the op silently degrading to a no-op at 1.0."""
        m, face = _Merger(), _crop(4, size=128)
        before = cv2.Laplacian(cv2.cvtColor(face, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32),
                               cv2.CV_32F).var()
        after = cv2.Laplacian(cv2.cvtColor(m.apply_clarity(face, 1.0), cv2.COLOR_BGR2LAB)[:, :, 0]
                              .astype(np.float32), cv2.CV_32F).var()
        self.assertGreater(after, before * 1.05)

    def test_it_bounds_a_neon_chrominance_excursion(self):
        """The half of the filter that is not sharpening: pull an over-saturated
        crop back toward the skin gamut rather than leaving it neon."""
        m = _Merger()
        neon = np.zeros((64, 64, 3), np.uint8)
        neon[:, :, 1], neon[:, :, 2] = 120, 255      # hot orange/red
        neon[::2, ::2, 2] = 180                      # some spread to compress
        out = m.apply_clarity(neon, 1.0)
        lab_in = cv2.cvtColor(neon, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_out = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
        self.assertLessEqual(lab_out[:, :, 2].std(), lab_in[:, :, 2].std() + 1e-6)
        self.assertTrue(np.isfinite(lab_out).all())

    def test_partial_strength_lands_between(self):
        """0.5 must be a real interpolation, not a second full application."""
        m, face = _Merger(), _crop(5, size=128)
        half = m.apply_clarity(face, 0.5).astype(np.float32)
        full = m.apply_clarity(face, 1.0).astype(np.float32)
        base = face.astype(np.float32)
        self.assertLess(np.abs(half - base).mean(), np.abs(full - base).mean())


class OutputFaceScale(unittest.TestCase):
    """The paste-matrix half, which lives on MaskingMixin."""

    def setUp(self):
        self._saved = roop.globals.output_face_scale

    def tearDown(self):
        roop.globals.output_face_scale = self._saved

    @staticmethod
    def _identity_IM():
        return np.array([[1.0, 0.0, 100.0],
                         [0.0, 1.0, 50.0]], dtype=np.float64)

    def test_neutral_returns_the_same_objects(self):
        roop.globals.output_face_scale = 0.0
        IM, lm = self._identity_IM(), np.zeros((5, 2), np.float32)
        out_IM, out_lm = MaskingMixin._scale_paste(IM, (64, 64), lm)
        self.assertIs(out_IM, IM)
        self.assertIs(out_lm, lm)

    def test_the_crop_centre_is_the_fixed_point(self):
        """Scaling about the centre means the face grows in place rather than
        drifting across the frame — the whole point of the transform."""
        roop.globals.output_face_scale = 0.2
        IM = self._identity_IM()
        centre = np.array([32.0, 32.0, 1.0])
        before = IM @ centre
        out_IM, _ = MaskingMixin._scale_paste(IM, (64, 64), None)
        np.testing.assert_allclose(out_IM @ centre, before, atol=1e-6)

    def test_positive_grows_and_negative_shrinks(self):
        IM = self._identity_IM()
        corner = np.array([0.0, 0.0, 1.0])
        centre = np.array([32.0, 32.0, 1.0])
        base = np.linalg.norm((IM @ corner) - (IM @ centre))
        for scale, expect_bigger in ((0.2, True), (-0.2, False)):
            roop.globals.output_face_scale = scale
            out, _ = MaskingMixin._scale_paste(IM, (64, 64), None)
            got = np.linalg.norm((out @ corner) - (out @ centre))
            self.assertEqual(got > base, expect_bigger, f"scale {scale} went the wrong way")

    def test_landmarks_scale_with_the_face(self):
        """The landmark hull clips the mask in FRAME space. If it did not scale
        too, growing the face would just crop it back to the old outline."""
        roop.globals.output_face_scale = 0.2
        IM = self._identity_IM()
        lm = np.array([[100.0, 50.0], [120.0, 90.0], [140.0, 50.0]], dtype=np.float32)
        _, out_lm = MaskingMixin._scale_paste(IM, (64, 64), lm)
        f_c = IM @ np.array([32.0, 32.0, 1.0])
        for before, after in zip(lm, out_lm):
            np.testing.assert_allclose(after - f_c, (before - f_c) * 1.2, atol=1e-4)

    def test_the_matrix_stays_invertible(self):
        """paste_upscale hands IM straight to warpAffine; a singular matrix
        there is a crash, not a bad-looking frame."""
        for scale in (-0.2, -0.05, 0.05, 0.2):
            roop.globals.output_face_scale = scale
            out, _ = MaskingMixin._scale_paste(self._identity_IM(), (64, 64), None)
            self.assertTrue(np.all(np.isfinite(out)))
            self.assertGreater(abs(np.linalg.det(out[:, :2])), 1e-6)


class GpenSizeContract(unittest.TestCase):
    """The scale_factor contract, which a sub-crop-size model nearly broke.

    paste_upscale multiplies the paste matrix by scale_factor, so a model
    SMALLER than the crop giving int(256/512) = 0 would collapse that matrix
    and blank the face. Tested on the helper, so it needs no weights.

    The helper was GPEN's own `_sized` and is now enhance_common.sized: every
    restorer needed the same contract, and three of them had grown a private
    `max(1, int(...))` instead. Kept pointing here rather than deleted, because
    the 256-tier registration below is GPEN-specific.
    """

    @staticmethod
    def _helper():
        from roop.processors.enhance_common import sized
        return sized

    def test_the_256_tier_is_registered(self):
        from roop.processors.Enhance_GPEN import GPEN_MODELS
        self.assertIn(256, GPEN_MODELS)
        self.assertTrue(GPEN_MODELS[256]["url"].startswith("https://huggingface.co/"))

    def test_a_smaller_model_never_reports_zero(self):
        out, factor = self._helper()(np.zeros((256, 256, 3), np.uint8), 512)
        self.assertEqual(factor, 1)
        self.assertEqual(out.shape[:2], (512, 512),
                         "a sub-crop-size result must come back at crop size")

    def test_equal_and_larger_models_are_untouched(self):
        helper = self._helper()
        for model_px, crop_px, expect in ((512, 512, 1), (1024, 512, 2), (2048, 512, 4)):
            src = np.zeros((model_px, model_px, 3), np.uint8)
            out, factor = helper(src, crop_px)
            self.assertEqual(factor, expect)
            self.assertIs(out, src, "the upscaling path must not copy or resize")


if __name__ == "__main__":
    unittest.main()
