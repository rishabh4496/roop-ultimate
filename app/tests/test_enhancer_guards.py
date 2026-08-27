"""The three lean-host-path enhancers: guards that cannot fire, and grain that
cannot flicker.

GPEN 256 Pro, GPEN Realistic and UltraMax share a post-processing lineage — a
LUT gather in, one `cv2.convertScaleAbs` out. That refactor moved the uint8 cast
EARLIER than in the other five restorers, and the non-finite guard went with it,
landing on the wrong side: `np.isfinite` is always True on an integer dtype, so
`is_usable(restored)` could never fire in any of the three. The real check is the
`np.isfinite(sum)` on the float before the cast, which is why nothing ever broke
— the second net was simply fictional. Fixed 2026-08-24; these tests stop it
coming back, in both directions.
"""

import os
import re
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.processors.enhance_common import is_usable, looks_collapsed   # noqa: E402

PROC = os.path.join(APP, 'roop', 'processors')
LEAN = ('Enhance_GPEN256Pro.py', 'Enhance_GPENRealistic.py', 'Enhance_UltraMax.py')
# These cast at the very END, so their is_usable sits on the raw float output
# and is correct. Named so the guard below cannot over-reach onto them.
RAW_FLOAT = ('Enhance_CodeFormer.py', 'Enhance_DMDNet.py', 'Enhance_GFPGAN.py',
             'Enhance_GPEN.py', 'Enhance_RestoreFormerPPlus.py')


def src(name):
    with open(os.path.join(PROC, name), encoding='utf-8') as fh:
        return fh.read()


def code_of(name):
    """Source with comment-only lines dropped, so a rule about CODE is not
    satisfied or broken by prose describing it."""
    return "\n".join(l for l in src(name).split("\n") if not l.strip().startswith('#'))


class TheDeadGuard(unittest.TestCase):

    def test_isfinite_is_blind_to_an_integer_dtype(self):
        """The premise. If this ever stops being true the rest is unnecessary."""
        import cv2
        nan = np.full((4, 4, 3), np.nan, dtype=np.float32)
        u8 = cv2.convertScaleAbs(nan, alpha=127.5, beta=127.5)
        self.assertEqual(u8.dtype, np.uint8)
        self.assertTrue((u8 == 0).all(), "NaN -> uint8 should be a black face")
        self.assertTrue(is_usable(u8), "is_usable cannot detect this — that IS the trap")
        self.assertFalse(is_usable(nan), "and it must still work on the float")

    def test_the_lean_enhancers_do_not_call_is_usable(self):
        """They cast before they check, so the call could only be decoration."""
        for name in LEAN:
            self.assertNotIn('is_usable(', code_of(name),
                             f"{name} calls is_usable after its uint8 cast, "
                             f"where np.isfinite is always True")

    def test_the_lean_enhancers_keep_the_float_guard(self):
        """Removing is_usable must not have removed the check that WORKS."""
        for name in LEAN:
            self.assertIn('np.isfinite(hwc.sum())', code_of(name),
                          f"{name} lost its pre-cast non-finite guard")

    def test_the_lean_enhancers_check_for_collapse(self):
        """Finite but flat is the failure that survives the cast — GFPGAN's FP16
        mode. Two of these are half-precision graphs."""
        for name in LEAN:
            self.assertIn('looks_collapsed(', code_of(name), name)

    def test_the_other_restorers_are_left_alone(self):
        """They check the RAW float straight out of ORT/torch, which is correct.
        This test exists so the rule above is never 'fixed' onto them."""
        for name in RAW_FLOAT:
            self.assertIn('is_usable(', code_of(name),
                          f"{name} checks the float output; do not remove it")

    def test_is_usable_documents_which_side_it_belongs_on(self):
        doc = is_usable.__doc__ or ''
        self.assertIn('BEFORE THE uint8 CAST', doc)


class CollapseDetection(unittest.TestCase):

    def test_a_flat_face_is_collapsed(self):
        rng = np.random.default_rng(0)
        source = rng.integers(0, 255, (64, 64, 3)).astype(np.uint8)
        self.assertTrue(looks_collapsed(np.full((64, 64, 3), 128, np.uint8), source))

    def test_a_real_restoration_is_not(self):
        rng = np.random.default_rng(0)
        source = rng.integers(0, 255, (64, 64, 3)).astype(np.uint8)
        self.assertFalse(looks_collapsed(source, source))

    def test_a_low_contrast_SOURCE_never_trips_it(self):
        """It must not fire on footage that is legitimately flat, or it would
        disable the enhancer on a dark or hazy shot."""
        flat = np.full((64, 64, 3), 100, np.uint8)
        self.assertFalse(looks_collapsed(flat, flat))


class UltraMaxEyeProtection(unittest.TestCase):
    """UltraMax must not redraw a second periocular structure over RealSwap."""

    def test_protects_swapped_eye_band_but_keeps_other_restoration(self):
        from roop.processors.Enhance_UltraMax import Enhance_UltraMax

        source = np.full((512, 512, 3), 32, dtype=np.uint8)
        restored = np.full((512, 512, 3), 224, dtype=np.uint8)
        out = Enhance_UltraMax._protect_swapped_eyes(restored, source)

        self.assertEqual(out.shape, restored.shape)
        self.assertTrue(np.isfinite(out).all())
        # The registered eye band is taken from the swapped input.
        # The eye aperture is intentionally a controlled source/UltraMax mix,
        # not a full paste; the source contribution must still be observable.
        self.assertLess(int(out[240, 193, 0]), 180)
        # A distant cheek pixel remains the enhancer's restored output.
        self.assertGreater(int(out[350, 350, 0]), 180)


class GrainMustNotFlicker(unittest.TestCase):
    """GPEN 256 Pro's synthetic micro-grain is deterministic ON PURPOSE.

    It is injected in aligned-crop space, so the field tracks the face. Drawing
    fresh noise per call would re-randomise it every frame on an otherwise stable
    face — flicker, which is the artefact this pipeline spends the most effort
    removing. The defect was never the determinism; it was re-seeding and
    re-drawing a 512x512 gaussian per face to produce a bit-identical array.
    """

    def setUp(self):
        from roop.processors.Enhance_GPEN256Pro import Enhance_GPEN256Pro
        self.P = Enhance_GPEN256Pro

    def test_the_field_is_identical_across_calls(self):
        self.assertTrue(np.array_equal(self.P._grain(256), self.P._grain(256)),
                        "a per-call redraw would flicker at frame rate")

    def test_it_is_cached_not_regenerated(self):
        self.assertIs(self.P._grain(256), self.P._grain(256))

    def test_it_is_read_only(self):
        """It is shared between every worker thread, so nothing may write to it."""
        self.assertFalse(self.P._grain(256).flags.writeable)

    def test_each_size_gets_its_own(self):
        a, b = self.P._grain(256), self.P._grain(512)
        self.assertEqual(a.shape, (256, 256, 1))
        self.assertEqual(b.shape, (512, 512, 1))

    def test_no_per_call_rng_is_left_in_the_filter(self):
        body = code_of('Enhance_GPEN256Pro.py')
        body = body.split('_enhance_textures_and_sharpness', 1)[1]
        self.assertNotIn('default_rng', body,
                         "the grain must come from the cache, not a fresh draw")


class SilentFailuresMustSpeak(unittest.TestCase):
    """A look filter must never take a render down — and must never fail mutely.

    GPEN 256 Pro's texture step is also what upsamples 256 -> 512, so an
    exception there returns a 256 image and `sized()` reports scale 1 instead of
    2: HALF the resolution reaches the frame and the processor silently becomes
    plain GPEN-256, the one outcome it exists to avoid.
    """

    def test_gpen256pro_warns_on_both_fallbacks(self):
        body = code_of('Enhance_GPEN256Pro.py')
        self.assertIn('_warned_texture', body)
        self.assertIn('_warned_colour', body)
        self.assertEqual(body.count('except Exception:'), 0,
                         "a bare silent except is what hid the halving")

    def test_gpen_realistic_warns_on_its_colour_fallback(self):
        self.assertIn('_warned_colour', code_of('Enhance_GPENRealistic.py'))

    def test_the_warnings_are_one_shot(self):
        """Per face would be thousands of lines and would break the progress bar."""
        for name, flags in (('Enhance_GPEN256Pro.py', ('_warned_texture', '_warned_colour')),
                            ('Enhance_GPENRealistic.py', ('_warned_colour',)),
                            ('Enhance_UltraMax.py', ('_warned_texture',))):
            body = code_of(name)
            for flag in flags:
                self.assertTrue(re.search(rf'{flag}\s*=\s*True', body),
                                f"{name}: {flag} is never latched")

    def test_failure_paths_return_the_unenhanced_input(self):
        """Not a cubic upsample of it — the same frame, at its own scale."""
        for name in LEAN:
            body = code_of(name)
            self.assertNotIn('return sized(src,', body, name)
            self.assertNotIn('return sized(src512,', body, name)


if __name__ == '__main__':
    unittest.main()
