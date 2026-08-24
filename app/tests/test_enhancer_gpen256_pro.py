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


def _reference_filter(cls, restored, source, input_size):
    """The texture/sharpen filter as originally spelled, kept as the oracle.

    The shipping version computes the same thing in fewer passes over the
    512x512x3 buffers (it was 32 ms per face against the network's 4.3 ms --
    the reason the card idled through this enhancer). Rewrites of that kind are
    exactly where a look silently changes, so the old arithmetic stays here and
    the new output is held against it.
    """
    target_size = 512 if input_size <= 256 else input_size
    T = (target_size, target_size)
    rest_scaled = (cv2.resize(restored, T, interpolation=cv2.INTER_LANCZOS4)
                   if restored.shape[:2] != T else restored)
    src_scaled = (cv2.resize(source, T, interpolation=cv2.INTER_CUBIC)
                  if source.shape[:2] != T else source)
    rest_f = rest_scaled.astype(np.float32)
    src_f = src_scaled.astype(np.float32)
    rest_gray = cv2.cvtColor(rest_scaled, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(rest_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(rest_gray, cv2.CV_32F, 0, 1, ksize=3)
    skin_gate = (1.0 / (1.0 + (np.hypot(gx, gy) / 14.0) ** 2))[:, :, np.newaxis]
    feature_gate = 1.0 - skin_gate
    exposure_gate = cls._EXPOSURE_LUT[
        np.clip(rest_gray, 0, 255).astype(np.uint8)][:, :, np.newaxis]
    sigma_texture = max(1.0, target_size / 256.0)
    hf_texture = src_f - cv2.GaussianBlur(src_f, (0, 0), sigma_texture)
    core = np.exp(-((hf_texture / 16.0) ** 2))
    hf_std = float(np.std(hf_texture))
    injected = 0.85 * hf_texture * core * skin_gate * exposure_gate
    if hf_std < 3.5:
        injected = injected + (cls._grain(target_size) * skin_gate * exposure_gate
                               * (1.0 - min(1.0, hf_std / 3.5)))
    sigma_sharp = 0.8 * (target_size / 256.0)
    hf_restored = rest_f - cv2.GaussianBlur(rest_f, (0, 0), sigma_sharp)
    sharpened = hf_restored * (0.42 * feature_gate + 0.12 * skin_gate)
    return np.clip(rest_f + injected + sharpened, 0.0, 255.0).astype(np.uint8)


class TestFilterMatchesTheReference(unittest.TestCase):
    """The rewrite is a SPEED change, not a look change.

    The only permitted difference is the final cast: the reference truncates
    (np.clip + astype) and the shipping version rounds (cv2.add with
    dtype=CV_8U), so a pixel may differ by 1/255 and never more.
    """

    def _both(self, restored, source, input_size=256):
        cls = CLS
        cls._warned_texture = False
        got = cls._enhance_textures_and_sharpness(restored, source, input_size)
        self.assertFalse(cls._warned_texture,
                         "the filter raised and fell back to 256 -- see the "
                         "except in _enhance_textures_and_sharpness")
        return _reference_filter(cls, restored, source, input_size), got

    def _assert_matches(self, restored, source, input_size=256):
        want, got = self._both(restored, source, input_size)
        self.assertEqual(want.shape, got.shape)
        d = np.abs(want.astype(np.int16) - got.astype(np.int16))
        self.assertEqual(int((d > 1).sum()), 0,
                         f"max deviation {int(d.max())}/255 -- the rewrite may "
                         f"only differ by the final rounding")

    def test_matches_on_a_normally_textured_crop(self):
        self._assert_matches(_face_like(1, detail=9.0), _face_like(2, detail=9.0))

    def test_matches_on_a_512_crop(self):
        self._assert_matches(_face_like(3, size=512), _face_like(4, size=512), 512)

    def test_matches_on_the_blurry_grain_branch(self):
        """A blurred source takes the `hf_std < 3.5` path, which adds a
        SINGLE-CHANNEL grain field to a three-channel buffer by broadcast.

        This branch is why the test exists. It fires only on degraded input, so
        a rewrite can break it and every ordinary clip still renders -- the
        filter's own except would swallow the error and quietly hand back a 256
        image, i.e. plain GPEN-256 quality, which is the exact outcome the
        class exists to avoid.
        """
        source = cv2.GaussianBlur(_face_like(5, detail=9.0), (0, 0), 6.0)
        cls = CLS
        rest_f = cv2.resize(_face_like(6), (512, 512),
                            interpolation=cv2.INTER_CUBIC).astype(np.float32)
        src_f = cv2.resize(source, (512, 512),
                           interpolation=cv2.INTER_CUBIC).astype(np.float32)
        hf = src_f - cv2.GaussianBlur(src_f, (0, 0), 2.0)
        self.assertLess(float(np.std(hf)), 3.5,
                        "this fixture no longer reaches the grain branch")
        self._assert_matches(_face_like(6), source)

    def test_grain_stays_monochrome(self):
        """Broadcast, not per-channel: coloured grain would be visible noise."""
        g = CLS._grain(512)
        self.assertEqual(g.shape, (512, 512, 1))


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
