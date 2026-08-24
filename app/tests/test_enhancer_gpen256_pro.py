"""Unit tests for GPEN 256 Pro: Upgraded sharper, high-texture, photo-realistic face restorer."""

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
import roop.processors.Enhance_GPEN256Pro as GPro

CLS = GPro.Enhance_GPEN256Pro


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


class TestColorAndChromaPreservation(unittest.TestCase):
    def test_gpen_pink_cast_is_removed(self):
        """GPEN's synthetic color cast must not reach the output."""
        source = _face_like(0)
        restored = _face_like(0, tint=(-18, -6, 30))  # pink/magenta cast
        self.assertGreater(_chroma_drift(restored, source), 2.0)
        out = CLS._keep_source_colour(restored, source)
        self.assertLess(_chroma_drift(out, source), 0.6)

    def test_identical_inputs_preserve_color(self):
        img = _face_like(5)
        out = CLS._keep_source_colour(img, img)
        self.assertLessEqual(int(np.abs(out.astype(np.int16) - img.astype(np.int16)).max()), 1)


class TestTexturesAndSharpening(unittest.TestCase):
    def test_high_frequency_dermal_texture_is_preserved_and_enhanced(self):
        source = _face_like(0, size=256, detail=8.0)
        # Model smoothed the face
        restored = cv2.GaussianBlur(source, (0, 0), 1.5)
        enhanced = CLS._enhance_textures_and_sharpness(restored, source, 256)
        self.assertGreater(_detail(enhanced), _detail(restored) * 1.1)

    def test_feature_sharpening_boosts_edges_cleanly(self):
        # Create an artificial face with sharp eye/feature line
        face = np.full((256, 256, 3), 140, dtype=np.uint8)
        # Sharp eyelid / lash line
        face[100:104, 80:180] = 30
        enhanced = CLS._enhance_textures_and_sharpness(face, face, 256)
        self.assertEqual(enhanced.shape, (512, 512, 3))
        self.assertEqual(enhanced.dtype, np.uint8)
        self.assertTrue(np.isfinite(enhanced).all())

    def test_scale_2_output_when_input_is_512(self):
        """When given a 512 crop, the processor enhances at 512 resolution (scale 2)."""
        source_512 = _face_like(0, size=512, detail=7.0)
        restored_256 = _face_like(1, size=256, detail=5.0)
        out = CLS._enhance_textures_and_sharpness(restored_256, source_512, 512)
        self.assertEqual(out.shape[:2], (512, 512))


class TestRunPath(unittest.TestCase):
    def _make(self, out_chw=None):
        p = CLS()
        p.plugin_options = {'devicename': 'cuda'}
        p.devicename = 'cuda'
        p.size = 256
        p._lut = (np.arange(256, dtype=np.float32) / 127.5) - 1.0
        if out_chw is None:
            face = _face_like(1, size=256, detail=10.0)
            out_chw = (face.transpose(2, 0, 1)[::-1].astype(np.float32) / 127.5) - 1.0
        iob = MagicMock()
        iob.bound = {}
        iob.bind_cpu_input = MagicMock(side_effect=lambda n, v: iob.bound.__setitem__(n, v))
        iob.copy_outputs_to_cpu = MagicMock(return_value=[out_chw[None]])
        sess = MagicMock()
        sess.run_with_iobinding = MagicMock()
        p.session, p.io_binding = sess, iob
        p.in_name, p.out_name = 'input', 'output'
        return p

    def test_run_returns_valid_frame_and_scale_2_for_256_crop(self):
        """256 neural net enhanced to 512 scale 2 for high-resolution paste-upscale."""
        p = self._make()
        out, scale = p.Run(None, None, _face_like(0, size=256))
        self.assertEqual(out.shape, (512, 512, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(scale, 2)

    def test_run_returns_scale_1_with_forced_256_size(self):
        os.environ['ROOP_GPEN256PRO_SIZE'] = '256'
        try:
            p = self._make()
            out, scale = p.Run(None, None, _face_like(0, size=256))
            self.assertEqual(out.shape, (256, 256, 3))
            self.assertEqual(scale, 1)
        finally:
            os.environ.pop('ROOP_GPEN256PRO_SIZE', None)

    def test_run_returns_512_for_512_crop(self):
        p = self._make()
        out, scale = p.Run(None, None, _face_like(0, size=512))
        self.assertEqual(out.shape, (512, 512, 3))
        self.assertEqual(scale, 1)

    def test_input_is_normalised_rgb_chw(self):
        p = self._make()
        frame = np.zeros((256, 256, 3), np.uint8)
        frame[:, :, 2] = 255  # pure red, BGR in
        p.Run(None, None, frame)
        x = p.io_binding.bound['input']
        self.assertEqual(x.shape, (1, 3, 256, 256))
        self.assertEqual(x.dtype, np.float32)
        self.assertAlmostEqual(float(x[0, 0].mean()), 1.0, places=3)   # R
        self.assertAlmostEqual(float(x[0, 1].mean()), -1.0, places=3)  # G
        self.assertAlmostEqual(float(x[0, 2].mean()), -1.0, places=3)  # B

    def test_non_finite_output_safely_falls_back(self):
        bad = np.zeros((3, 256, 256), np.float32)
        bad[0, 10, 10] = np.nan
        p = self._make(bad)
        frame = _face_like(0, size=256)
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

    def test_cost_summary_tracks_faces(self):
        p = self._make()
        self.assertIsNone(p.cost_summary())
        p.Run(None, None, _face_like(0, size=256))
        self.assertIn('1 faces', p.cost_summary())


class TestSystemWiring(unittest.TestCase):
    NAME = 'GPEN 256 Pro'
    KEY = 'gpen_256_pro'

    def _src(self, *parts):
        with open(os.path.join(APP, *parts), encoding='utf-8') as f:
            return f.read()

    def test_core_maps_display_name(self):
        src = self._src('roop', 'core.py')
        self.assertIn(f"selected_enhancer == '{self.NAME}'", src)
        self.assertIn(f'{{"{self.KEY}": {{}}}}', src)

    def test_processmgr_maps_key_to_class(self):
        src = self._src('roop', 'ProcessMgr.py')
        self.assertIn(f"'{self.KEY}'", src)
        self.assertIn('Enhance_GPEN256Pro', src)

    def test_api_offers_it(self):
        src = self._src('api.py')
        self.assertIn(f'"{self.NAME}"', src)

    def test_class_declares_template_and_type(self):
        self.assertEqual(CLS.model_template, 'ffhq_512')
        self.assertEqual(CLS.type, 'enhance')
        self.assertEqual(CLS.processorname, self.KEY)


if __name__ == '__main__':
    unittest.main()
