"""GPEN Realistic: GPEN-512's luminance, the swapper's chrominance.

Two findings these tests encode, the second correcting the first build.

1. GPEN's problem is COLOUR, not detail. It pushes the face pink and paints
   magenta on the eyelids — chroma drift ~2.9 against the crop it was handed,
   where the input is 0. Keeping GPEN's LUMINANCE and taking chrominance from
   the swapper's crop removes it with detail completely unchanged.

2. But the size that matters is 512, because of the PASTE. `realswap` emits a
   256 crop, so a 256 restorer returns 256 and pastes at scale 1, while a 512
   restorer returns 512 and pastes at scale 2. Detail reaching the frame:

       swap input 2.67 | GPEN-256 2.82 | CodeFormer-512 4.11 | GPEN-512 5.14

   GPEN-256 is barely above the UNENHANCED input. The first version of this
   processor used 256 plus the colour fix and was reported as indistinguishable
   from plain GPEN-256 — correctly, and that is what
   `test_a_256_crop_comes_back_at_512_with_scale_2` now pins.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import roop.globals  # noqa: F401
import roop.processors.Enhance_GPENRealistic as GR

CLS = GR.Enhance_GPENRealistic


def _chroma_drift(a, b):
    la = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32)
    lb = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.abs(la[:, :, 1:] - lb[:, :, 1:]).mean())


def _detail(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float((g - cv2.GaussianBlur(g, (0, 0), 1.1)).std())


def _face_like(seed=0, size=256, tint=(0, 0, 0), detail=9.0):
    """A textured patch standing in for a face crop, optionally colour-cast."""
    rng = np.random.default_rng(seed)
    base = np.clip(rng.normal(140, detail, (size, size)), 0, 255).astype(np.uint8)
    img = np.repeat(base[:, :, None], 3, axis=2).astype(np.int16)
    for c in range(3):
        img[:, :, c] += tint[c]
    return np.clip(img, 0, 255).astype(np.uint8)


class TestColourComesFromTheSource(unittest.TestCase):
    def test_a_colour_cast_in_the_restored_image_is_removed(self):
        """The whole point: GPEN's pink must not reach the output."""
        source = _face_like(0)
        restored = _face_like(0, tint=(-18, -6, 30))      # pink-ish cast
        self.assertGreater(_chroma_drift(restored, source), 2.0,
                           "the fixture must actually be cast, or this proves nothing")
        out = CLS._keep_source_colour(restored, source)
        self.assertLess(_chroma_drift(out, source), 0.6,
                        "output colour must track the SOURCE, not the restorer")

    def test_luminance_detail_is_preserved(self):
        """Detail lives in luminance, so taking the source's colour must not
        cost any of it — that is why this is not a plain blend."""
        source = _face_like(0, detail=5.0)
        restored = _face_like(1, tint=(-18, -6, 30), detail=12.0)
        out = CLS._keep_source_colour(restored, source)
        self.assertGreater(_detail(out), _detail(source) * 1.3,
                           "the restorer's extra detail must survive")
        self.assertAlmostEqual(_detail(out), _detail(restored),
                               delta=_detail(restored) * 0.15)

    def test_a_plain_blend_would_lose_detail_this_does_not(self):
        """Contrast with the obvious alternative: averaging the two images fixes
        the colour too, but halves the detail. This is why the fix is a
        luminance-only edit."""
        source = _face_like(0, detail=5.0)
        restored = _face_like(1, tint=(-18, -6, 30), detail=12.0)
        blend = cv2.addWeighted(source, 0.5, restored, 0.5, 0.0)
        out = CLS._keep_source_colour(restored, source)
        self.assertGreater(_detail(out), _detail(blend))

    def test_chroma_1_returns_the_restorer_untouched(self):
        source = _face_like(0)
        restored = _face_like(1, tint=(-18, -6, 30))
        out = CLS._keep_source_colour(restored, source, chroma=1.0)
        self.assertLess(float(np.abs(out.astype(np.int16)
                                     - restored.astype(np.int16)).mean()), 1.0)

    def test_shape_dtype_and_finiteness(self):
        source, restored = _face_like(0), _face_like(1)
        out = CLS._keep_source_colour(restored, source)
        self.assertEqual(out.shape, source.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.isfinite(out).all())

    def test_identical_inputs_are_a_near_no_op(self):
        img = _face_like(3)
        out = CLS._keep_source_colour(img, img)
        self.assertLessEqual(int(np.abs(out.astype(np.int16)
                                        - img.astype(np.int16)).max()), 1)

    def test_the_exact_lab_form_agrees_with_the_cheap_one(self):
        """`_LAB_EXACT` is the more precise variant kept for re-measurement; it
        must reach the same place, or the cheap default is not a shortcut but a
        different operation."""
        source = _face_like(0)
        restored = _face_like(1, tint=(-18, -6, 30))
        cheap = CLS._keep_source_colour(restored, source)
        old = CLS._LAB_EXACT
        try:
            CLS._LAB_EXACT = True
            exact = CLS._keep_source_colour(restored, source)
        finally:
            CLS._LAB_EXACT = old
        self.assertLess(_chroma_drift(exact, source), 0.6)
        self.assertLess(float(np.abs(cheap.astype(np.int16)
                                     - exact.astype(np.int16)).mean()), 6.0)


class TestRunPath(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get('ROOP_GPENR_CHROMA')
        os.environ.pop('ROOP_GPENR_CHROMA', None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop('ROOP_GPENR_CHROMA', None)
        else:
            os.environ['ROOP_GPENR_CHROMA'] = self._saved

    def _make(self, out_chw=None, size=None):
        p = CLS()
        p.plugin_options = {'devicename': 'cuda'}
        p.devicename = 'cuda'
        p.size = size or CLS._SIZE
        p._lut = (np.arange(256, dtype=np.float32) / 127.5) - 1.0
        if out_chw is None:
            out_chw = np.zeros((3, p.size, p.size), np.float32)
        iob = MagicMock()
        iob.bound = {}
        iob.bind_cpu_input = MagicMock(side_effect=lambda n, v: iob.bound.__setitem__(n, v))
        iob.copy_outputs_to_cpu = MagicMock(return_value=[out_chw[None]])
        sess = MagicMock()
        sess.run_with_iobinding = MagicMock()
        p.session, p.io_binding = sess, iob
        p.in_name, p.out_name = 'input', 'output'
        return p

    def test_run_returns_a_valid_frame(self):
        p = self._make()
        out, scale = p.Run(None, None, _face_like(0, size=p.size))
        self.assertEqual(out.shape, (p.size, p.size, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(scale, 1)

    def test_input_is_normalised_rgb_chw(self):
        p = self._make()
        frame = np.zeros((p.size, p.size, 3), np.uint8)
        frame[:, :, 2] = 255                     # pure red, BGR in
        p.Run(None, None, frame)
        x = p.io_binding.bound['input']
        self.assertEqual(x.shape, (1, 3, p.size, p.size))
        self.assertEqual(x.dtype, np.float32)
        self.assertAlmostEqual(float(x[0, 0].mean()), 1.0, places=3)   # R
        self.assertAlmostEqual(float(x[0, 1].mean()), -1.0, places=3)  # G
        self.assertAlmostEqual(float(x[0, 2].mean()), -1.0, places=3)  # B

    def test_a_256_crop_comes_back_at_512_with_scale_2(self):
        """THE WHOLE REASON THIS RUNS AT 512. realswap emits a 256 crop; a 512
        restorer returns 512 and `sized` reports scale 2, so paste_upscale
        composites at twice the resolution. A 256 restorer returns 256 at scale
        1 and that extra detail never reaches the frame — which is why the first
        build of this processor was indistinguishable from plain GPEN-256."""
        p = self._make()
        out, scale = p.Run(None, None, _face_like(0, size=256))
        self.assertEqual(out.shape[:2], (512, 512))
        self.assertEqual(scale, 2)

    def test_the_256_tier_still_works_and_pastes_at_scale_1(self):
        p = self._make(size=256)
        out, scale = p.Run(None, None, _face_like(0, size=256))
        self.assertEqual(out.shape[:2], (256, 256))
        self.assertEqual(scale, 1)

    def test_non_finite_output_falls_back_to_the_unenhanced_crop(self):
        p0 = CLS()
        bad = np.zeros((3, p0._SIZE, p0._SIZE), np.float32)
        bad[0, 5, 5] = np.nan
        p = self._make(bad)
        frame = _face_like(0, size=p.size)
        out, scale = p.Run(None, None, frame)
        self.assertTrue(np.array_equal(out, frame))
        self.assertEqual(scale, 1)

    def test_handles_missing_or_empty_frames(self):
        p = self._make()
        out, _ = p.Run(None, None, None)
        self.assertIsNone(out)
        empty = np.zeros((0, 0, 3), np.uint8)
        out, _ = p.Run(None, None, empty)
        self.assertEqual(out.size, 0)

    def test_cost_summary_counts_faces(self):
        p = self._make()
        self.assertIsNone(p.cost_summary())
        p.Run(None, None, _face_like(0, size=p.size))
        self.assertIn('1 faces', p.cost_summary())


class TestWiring(unittest.TestCase):
    """An enhancer that is not registered everywhere renders as NOTHING, silently
    — see tests/test_enhancer_names.py for the four harnesses that shipped that
    bug. These are the registration points for this one."""

    NAME = 'GPEN Realistic'
    KEY = 'gpen_realistic'

    def _src(self, *parts):
        with open(os.path.join(APP, *parts), encoding='utf-8') as f:
            return f.read()

    def test_core_maps_the_display_name_to_the_processor(self):
        src = self._src('roop', 'core.py')
        self.assertIn(f"selected_enhancer == '{self.NAME}'", src)
        self.assertIn(f'{{"{self.KEY}": {{}}}}', src)

    def test_processmgr_maps_the_key_to_the_class(self):
        self.assertIn(f"'{self.KEY}'", self._src('roop', 'ProcessMgr.py'))
        self.assertIn('Enhance_GPENRealistic', self._src('roop', 'ProcessMgr.py'))

    def test_the_ui_offers_it(self):
        """The React dropdown renders meta.enhancers from api.py."""
        self.assertIn(f'"{self.NAME}"', self._src('api.py'))

    def test_it_declares_a_face_template(self):
        self.assertEqual(CLS.model_template, 'ffhq_512')
        self.assertEqual(CLS.type, 'enhance')
        self.assertEqual(CLS.processorname, self.KEY)


if __name__ == '__main__':
    unittest.main()
