"""GPEN Ultimate and Restore Ultra: the two ported "ultimate quality" profiles.

WHAT THESE TWO PROFILES ACTUALLY ARE, because the name suggests more than the
change is. Neither adds a network. `Enhance_GPENUltimate` opens the same
`GPEN-BFR-512.onnx` as `Enhance_GPEN` and `Enhance_RestoreUltra` opens the same
`restoreformer_plus_plus.onnx` as `Enhance_RestoreFormerPPlus`, through the same
pooled io-binding session path, with the same provider policy and the same
non-finite guard. Two things differ, and this module pins both:

1. `force_align = True` -- a per-processor OVERRIDE of the global
   `enhancer_align` opt-in, because the finishing stages are keyed to the FFHQ
   template's own eye coordinates and are only anatomically correct on a crop
   genuinely in that space.

2. A CPU finish (`enhance_gpen_ultimate` / `enhance_restore_ultra`): bilateral
   detail injection, eye clarity, anti-halo sharpen.

The non-obvious risks these tests exist for are all in the SHARING, not in the
arithmetic: the base classes keep their session and their mutex in CLASS
attributes, so a subclass that does not re-declare them silently runs on its
parent's session, and the finish is applied in the subclass's `Run` so selecting
the plain arm must stay bit-identical to what it was before these existed.
"""

import os
import sys
import threading
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.processors.enhance_common import (  # noqa: E402
    _eye_region,
    _inject_bilateral_detail,
    _knee_lut,
    apply_anti_halo_sharpen,
    enhance_eyes_clarity,
    enhance_gpen_ultimate,
    enhance_restore_ultra,
    inject_reference_detail,
)

# The two eye keypoints of the ffhq_512 warp template, as fractions of the crop.
FFHQ_EYES = ((0.37691676, 0.46864664), (0.62285697, 0.46912813))


def face_like_crop(size=512, seed=7):
    """A deterministic crop with eyes, a mouth and skin-scale micro-texture.

    Not a real face, but it carries the two features the finish is keyed to:
    local step edges at the template's eye positions, and broadband texture for
    the bilateral/knee stage to act on.
    """
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 150, np.uint8)
    cv2.ellipse(img, (size // 2, int(size * 0.55)),
                (int(size * 0.30), int(size * 0.40)), 0, 0, 360,
                (172, 180, 196), -1)
    for fx, fy in FFHQ_EYES:
        cx, cy = int(fx * size), int(fy * size)
        cv2.ellipse(img, (cx, cy), (int(size * 0.055), int(size * 0.030)),
                    0, 0, 360, (240, 240, 240), -1)
        cv2.circle(img, (cx, cy), int(size * 0.020), (60, 45, 40), -1)
        cv2.circle(img, (cx - 2, cy - 2), max(1, int(size * 0.005)),
                   (255, 255, 255), -1)
    cv2.ellipse(img, (size // 2, int(size * 0.72)),
                (int(size * 0.10), int(size * 0.035)), 0, 0, 360,
                (110, 105, 160), -1)
    noise = rng.normal(0.0, 6.0, (size, size, 3))
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def local_envelope(img, pad=0.0):
    """The 3x3 min/max envelope every stage here is supposed to stay inside."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    lo = cv2.erode(img, k).astype(np.float32) - pad
    hi = cv2.dilate(img, k).astype(np.float32) + pad
    return lo, hi


# ── the source-level contract ────────────────────────────────────────────────
class TestProfileDeclarations(unittest.TestCase):
    """The two class-level facts the pipeline reads off these processors."""

    def test_both_profiles_force_alignment_and_declare_ffhq(self):
        """`force_align` is what makes the eye stage anatomically valid.

        Without it the profile would only realign when the user happened to
        have the global `enhancer_align` switch on, and the eye ellipses --
        placed from the TEMPLATE, not from the face -- would land wherever the
        swapper's template put the eyes instead.
        """
        from roop.processors.Enhance_GPENUltimate import Enhance_GPENUltimate
        from roop.processors.Enhance_RestoreUltra import Enhance_RestoreUltra
        for cls in (Enhance_GPENUltimate, Enhance_RestoreUltra):
            self.assertIs(cls.force_align, True, cls.__name__)
            self.assertEqual(cls.model_template, 'ffhq_512', cls.__name__)
            self.assertEqual(cls.type, 'enhance', cls.__name__)
            # Inherited from the bases: every session call goes through
            # `exclusive()`, so the enhance-stage lock can skip them.
            self.assertIs(getattr(cls, 'self_excluding', False), True,
                          cls.__name__)

    def test_processmgr_registers_both_keys(self):
        from roop.ProcessMgr import ProcessMgr
        self.assertEqual(ProcessMgr.plugins['gpen_ultimate'],
                         'Enhance_GPENUltimate')
        self.assertEqual(ProcessMgr.plugins['restore_ultra'],
                         'Enhance_RestoreUltra')

    def test_core_selector_maps_both_labels(self):
        """The UI label must reach the processor key, or the dropdown entry is
        a silent no-op that renders with no enhancer at all."""
        import re
        core = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'roop', 'core.py')
        with open(core, encoding='utf-8') as fh:
            src = fh.read()
        for label, key in (('GPEN Ultimate', 'gpen_ultimate'),
                           ('Restore Ultra', 'restore_ultra')):
            self.assertTrue(
                re.search(rf"selected_enhancer == '{label}'", src),
                f'core.py has no branch for {label!r}')
            self.assertIn(f'"{key}"', src)

    def test_each_subclass_owns_its_session_lock_and_session_slot(self):
        """THE SHARING BUG THIS PREVENTS.

        `_session_lock` and the session handle live in CLASS attributes on both
        bases. A subclass that does not re-declare them inherits the PARENT's
        objects, so:

          - `Enhance_RestoreUltra` would find the base's
            `model_restoreformerpplus` already set by a previous
            `Restoreformer++` selection and skip building its own session,
            then run the plain arm's session while reporting as Restore Ultra;
          - and the two would serialise against one another's mutex for no
            reason, since they hold separate contexts.
        """
        from roop.processors.Enhance_GPEN import Enhance_GPEN
        from roop.processors.Enhance_GPENUltimate import Enhance_GPENUltimate
        from roop.processors.Enhance_RestoreFormerPPlus import (
            Enhance_RestoreFormerPPlus)
        from roop.processors.Enhance_RestoreUltra import Enhance_RestoreUltra

        self.assertIsNot(Enhance_GPENUltimate._session_lock,
                         Enhance_GPEN._session_lock)
        self.assertIsNot(Enhance_RestoreUltra._session_lock,
                         Enhance_RestoreFormerPPlus._session_lock)
        for cls in (Enhance_GPENUltimate, Enhance_RestoreUltra):
            self.assertIsInstance(cls._session_lock, type(threading.Lock()))
        # Restore Ultra must not inherit the base's session slot.
        self.assertIn('model_restoreformerpplus',
                      Enhance_RestoreUltra.__dict__)
        self.assertIsNone(Enhance_RestoreUltra.model_restoreformerpplus)

    def test_the_finish_is_applied_by_the_subclass_not_the_base(self):
        """Selecting plain GPEN / Restoreformer++ must be unchanged.

        If the finish were called from the shared base it would apply to every
        GPEN tier and to plain Restoreformer++ as a side effect of these
        profiles existing.
        """
        root = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'roop', 'processors')
        for base in ('Enhance_GPEN.py', 'Enhance_RestoreFormerPPlus.py'):
            with open(os.path.join(root, base), encoding='utf-8') as fh:
                src = fh.read()
            self.assertNotIn('enhance_gpen_ultimate', src, base)
            self.assertNotIn('enhance_restore_ultra', src, base)


# ── the soft-knee detail table ───────────────────────────────────────────────
class TestKneeTable(unittest.TestCase):
    """The 511-entry LUT must BE the curve it replaces, not approximate it."""

    def test_table_is_elementwise_identical_to_the_where_form(self):
        threshold, softness, strength = 12.0, 3.0, 0.36
        d = np.arange(-255, 256, dtype=np.float32)
        expected = np.where(
            np.abs(d) <= threshold,
            d,
            np.sign(d) * (threshold + softness
                          * np.tanh((np.abs(d) - threshold) / softness))
        ) * np.float32(strength)
        got = _knee_lut(threshold, softness, strength)
        # Bit-for-bit: the table is built with the identical expression.
        np.testing.assert_array_equal(got, expected)

    def test_the_two_profiles_get_separate_tables(self):
        """Keyed by its own parameters, so GPEN's (12/3) and Restore's (10/2.5)
        curves cannot overwrite each other in the cache."""
        a = _knee_lut(12.0, 3.0, 0.36)
        b = _knee_lut(10.0, 2.5, 0.30)
        self.assertFalse(np.array_equal(a, b))
        self.assertIs(a, _knee_lut(12.0, 3.0, 0.36))  # cached, not rebuilt

    def test_below_the_knee_the_curve_is_the_identity_times_strength(self):
        lut = _knee_lut(12.0, 3.0, 1.0)
        for delta in range(-12, 13):
            self.assertAlmostEqual(float(lut[delta + 255]), float(delta),
                                   places=5)

    def test_above_the_knee_it_saturates_rather_than_growing(self):
        lut = _knee_lut(12.0, 3.0, 1.0)
        # A 200-level residual must not be transferred as 200 levels.
        self.assertLess(float(lut[200 + 255]), 16.0)
        self.assertGreater(float(lut[200 + 255]), 12.0)

    def test_detail_injection_stays_in_range_and_is_uint8(self):
        ref = face_like_crop()
        enhanced = cv2.GaussianBlur(ref, (0, 0), sigmaX=1.6)
        out = _inject_bilateral_detail(enhanced, ref, 22.0, 12.0, 3.0, 0.36)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, ref.shape)


# ── the anti-halo contract ───────────────────────────────────────────────────
class TestAntiHaloBounding(unittest.TestCase):
    """No stage may push a pixel outside its own neighbours' range.

    That property is the whole reason these profiles can sharpen aggressively
    without ringing: a halo IS an overshoot beyond the local envelope, so
    bounding to the envelope makes one unrepresentable rather than merely small.
    """

    def test_sharpen_never_leaves_the_local_envelope_on_a_step_edge(self):
        # A hard black/white step is the worst case: classic unsharp masking
        # produces its largest overshoot exactly here.
        img = np.zeros((64, 64, 3), np.uint8)
        img[:, 32:] = 255
        limit = 2.5
        out = apply_anti_halo_sharpen(img, amount=1.5, sigma=1.0, limit=limit)

        lab_in = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        lo, hi = local_envelope(lab_in, pad=limit)
        lab_out = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        # +-1 for the BGR->LAB->BGR round trip's own quantization.
        self.assertTrue((lab_out >= lo - 1.5).all())
        self.assertTrue((lab_out <= hi + 1.5).all())

    def test_zero_amount_is_an_exact_no_op(self):
        img = face_like_crop(128)
        np.testing.assert_array_equal(
            apply_anti_halo_sharpen(img, amount=0.0), img)

    def test_sharpen_actually_raises_edge_contrast(self):
        """A bound that also removed the effect would pass the test above."""
        img = face_like_crop(256)
        out = apply_anti_halo_sharpen(img, amount=0.8, sigma=1.0)
        def edge_energy(x):
            g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
            return float(np.abs(cv2.Laplacian(g, cv2.CV_32F)).mean())
        self.assertGreater(edge_energy(out), edge_energy(img) * 1.01)


class TestEyeClarity(unittest.TestCase):
    """The eye stage must be LOCAL, TONE-NEUTRAL and halo-free."""

    def test_it_only_touches_a_box_around_the_template_eyes(self):
        img = face_like_crop()
        out = enhance_eyes_clarity(img, template='ffhq_512', strength=0.52)
        changed = np.any(out != img, axis=2)
        self.assertTrue(changed.any(), 'the eye stage did nothing at all')

        region = _eye_region(img.shape[0], img.shape[1], 'ffhq_512', 0.52)
        self.assertIsNotNone(region)
        _weight, (x0, y0, x1, y1), _sigma = region
        outside = changed.copy()
        outside[y0:y1, x0:x1] = False
        self.assertFalse(outside.any(),
                         'the eye stage modified pixels outside its own box')

        # The box must actually be around the eyes, not a band across the face.
        h, w = img.shape[:2]
        for fx, fy in FFHQ_EYES:
            self.assertTrue(x0 <= fx * w <= x1)
            self.assertTrue(y0 <= fy * h <= y1)

    def test_the_box_is_eye_sized_not_a_band_across_the_whole_face(self):
        """Radii come from the template's own interocular distance.

        A fixed fraction of the crop made the pair meet over the nose bridge
        and reach the temples -- brows, sockets and upper cheeks included --
        which is what turns a "local" sharpen into a visible periocular band.
        """
        region = _eye_region(512, 512, 'ffhq_512', 0.5)
        _weight, (x0, y0, x1, y1), _sigma = region
        self.assertLess(y1 - y0, 512 * 0.22)
        self.assertLess(x1 - x0, 512 * 0.50)

    def test_it_does_not_shift_the_tone_of_the_region(self):
        """THE REGRESSION THIS PINS.

        A CLAHE-based version of this stage moved L over the eye ellipses by
        +28.5 / +23.4 levels on average -- a low-frequency brightening painted
        into an oval and feathered at its rim, i.e. exactly the halo the stage
        claims to avoid. What remains is a zero-mean high-pass, so the MEAN
        level of the region must be essentially unchanged.
        """
        img = face_like_crop()
        out = enhance_eyes_clarity(img, template='ffhq_512', strength=0.52)
        _w, (x0, y0, x1, y1), _s = _eye_region(512, 512, 'ffhq_512', 0.52)
        lin = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2LAB)[:, :, 0]
        lout = cv2.cvtColor(out[y0:y1, x0:x1], cv2.COLOR_BGR2LAB)[:, :, 0]
        drift = abs(float(lout.mean()) - float(lin.mean()))
        self.assertLess(drift, 1.0, f'eye region tone drifted {drift:.2f} levels')

    def test_it_stays_inside_the_original_luma_envelope(self):
        img = face_like_crop()
        out = enhance_eyes_clarity(img, template='ffhq_512', strength=1.0)
        _w, (x0, y0, x1, y1), _s = _eye_region(512, 512, 'ffhq_512', 1.0)
        lin = cv2.cvtColor(img[y0:y1, x0:x1],
                           cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        lout = cv2.cvtColor(out[y0:y1, x0:x1],
                            cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        lo, hi = local_envelope(lin)
        self.assertTrue((lout >= lo - 2.0).all())
        self.assertTrue((lout <= hi + 2.0).all())

    def test_zero_strength_is_an_exact_no_op(self):
        img = face_like_crop(128)
        np.testing.assert_array_equal(
            enhance_eyes_clarity(img, strength=0.0), img)

    def test_a_crop_too_small_for_a_pair_of_eyes_is_returned_untouched(self):
        tiny = np.full((6, 6, 3), 128, np.uint8)
        np.testing.assert_array_equal(
            enhance_eyes_clarity(tiny, strength=0.5), tiny)

    def test_the_mask_is_cached_per_geometry_and_strength(self):
        a = _eye_region(512, 512, 'ffhq_512', 0.52)
        self.assertIs(a, _eye_region(512, 512, 'ffhq_512', 0.52))
        self.assertIsNot(a, _eye_region(512, 512, 'ffhq_512', 0.40))


# ── the two composed finishes ────────────────────────────────────────────────
class TestComposedFinishes(unittest.TestCase):

    def _restored(self):
        """Stand in for a restorer output: slightly soft, same geometry."""
        ref = face_like_crop()
        return cv2.GaussianBlur(ref, (0, 0), sigmaX=1.4), ref

    def test_both_finishes_return_a_valid_same_shape_uint8_frame(self):
        enhanced, ref = self._restored()
        for fn in (enhance_gpen_ultimate, enhance_restore_ultra):
            out = fn(enhanced, ref)
            self.assertEqual(out.dtype, np.uint8, fn.__name__)
            self.assertEqual(out.shape, enhanced.shape, fn.__name__)
            self.assertTrue(np.isfinite(out.astype(np.float32)).all(),
                            fn.__name__)

    def test_both_finishes_add_detail_rather_than_leaving_the_input_alone(self):
        enhanced, ref = self._restored()
        def edge_energy(x):
            g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
            return float(np.abs(cv2.Laplacian(g, cv2.CV_32F)).mean())
        base = edge_energy(enhanced)
        for fn in (enhance_gpen_ultimate, enhance_restore_ultra):
            out = fn(enhanced, ref)
            self.assertFalse(np.array_equal(out, enhanced), fn.__name__)
            self.assertGreater(edge_energy(out), base, fn.__name__)

    def test_neither_finish_shifts_overall_exposure(self):
        """A finish is detail, not grading: the global mean must hold."""
        enhanced, ref = self._restored()
        for fn in (enhance_gpen_ultimate, enhance_restore_ultra):
            out = fn(enhanced, ref)
            drift = abs(float(out.mean()) - float(enhanced.mean()))
            self.assertLess(drift, 2.0, f'{fn.__name__} drifted {drift:.2f}')

    def test_restore_ultra_is_the_gentler_of_the_two(self):
        """Its constants are deliberately lower (18/10/0.8 against 22/12/1.0)
        because RestoreFormer++ already returns more micro-contrast."""
        enhanced, ref = self._restored()
        d_gpen = float(np.abs(enhance_gpen_ultimate(enhanced, ref).astype(np.float32)
                              - enhanced.astype(np.float32)).mean())
        d_rest = float(np.abs(enhance_restore_ultra(enhanced, ref).astype(np.float32)
                              - enhanced.astype(np.float32)).mean())
        self.assertLess(d_rest, d_gpen)

    def test_a_mismatched_reference_is_resized_not_rejected(self):
        """The 1024/2048 GPEN tiers return a buffer larger than the crop."""
        enhanced, ref = self._restored()
        small = cv2.resize(ref, (256, 256), interpolation=cv2.INTER_AREA)
        for fn in (enhance_gpen_ultimate, enhance_restore_ultra):
            out = fn(enhanced, small)
            self.assertEqual(out.shape, enhanced.shape, fn.__name__)

    def test_a_missing_reference_degrades_to_self_reference(self):
        enhanced, _ref = self._restored()
        for fn in (enhance_gpen_ultimate, enhance_restore_ultra):
            out = fn(enhanced, None)
            self.assertEqual(out.shape, enhanced.shape, fn.__name__)
            self.assertEqual(out.dtype, np.uint8, fn.__name__)

    def test_none_input_is_passed_through(self):
        for fn in (enhance_gpen_ultimate, enhance_restore_ultra):
            self.assertIsNone(fn(None, None))

    def test_a_flat_frame_does_not_become_noise(self):
        """Nothing to sharpen must mean nothing sharpened -- no amplification
        of the quantization floor into visible grain."""
        flat = np.full((256, 256, 3), 128, np.uint8)
        for fn in (enhance_gpen_ultimate, enhance_restore_ultra):
            out = fn(flat, flat)
            self.assertLessEqual(int(np.abs(out.astype(np.int16)
                                            - 128).max()), 2, fn.__name__)


class TestInjectReferenceDetail(unittest.TestCase):
    """The shared helper the Ultra profile's lineage is built on."""

    def test_neutral_parameters_are_a_strict_no_op(self):
        enhanced = face_like_crop(128)
        ref = face_like_crop(128, seed=9)
        out = inject_reference_detail(enhanced, ref, strength=0.0, crispness=0.0)
        np.testing.assert_array_equal(out, enhanced)

    def test_non_numeric_strengths_degrade_to_a_no_op(self):
        enhanced = face_like_crop(64)
        np.testing.assert_array_equal(
            inject_reference_detail(enhanced, enhanced, strength='x',
                                    crispness=None),
            enhanced)

    def test_it_transfers_reference_detail_into_a_blurred_frame(self):
        ref = face_like_crop(256)
        enhanced = cv2.GaussianBlur(ref, (0, 0), sigmaX=2.0)
        out = inject_reference_detail(enhanced, ref, strength=0.6,
                                      crispness=0.4)
        def edge_energy(x):
            g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
            return float(np.abs(cv2.Laplacian(g, cv2.CV_32F)).mean())
        self.assertGreater(edge_energy(out), edge_energy(enhanced))
        self.assertEqual(out.dtype, np.uint8)

    def test_a_single_channel_input_is_rejected_rather_than_crashing(self):
        gray = np.full((32, 32), 128, np.uint8)
        np.testing.assert_array_equal(
            inject_reference_detail(gray, gray, strength=0.5), gray)


def load_tests(loader, tests, pattern):
    """Expose bare `test_*` functions to `unittest discover`; see
    tests/unittest_shim.py. pytest never calls load_tests."""
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:  # discovery started from inside tests/
        from unittest_shim import load_tests_for
    return load_tests_for(globals())


if __name__ == '__main__':
    unittest.main(verbosity=2)
