"""Unit tests for the UltraMax enhancer.

UltraMax runs codeformer.fp16.onnx directly (same weights as
`Codeformer (fp16)`) on a leaner host path, then restores dermal texture that
the restorer flattened. The tests that matter here are about the texture
restore, because that is the part with a look to get wrong: it must leave
everything the codebook drew as STRUCTURE alone, which is what stops it
printing the second eyelid crease the old unsharp-based filter produced.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import roop.globals
import roop.processors.Enhance_UltraMax as UM


def _fake_session(out_chw):
    """A stand-in for the ONNX session: records the bound inputs, replays
    `out_chw` as the model output."""
    iob = MagicMock()
    iob.bound = {}
    iob.bind_cpu_input = MagicMock(side_effect=lambda n, v: iob.bound.__setitem__(n, v))
    iob.bind_output = MagicMock()
    iob.copy_outputs_to_cpu = MagicMock(return_value=[out_chw[None]])
    sess = MagicMock()
    sess.io_binding = MagicMock(return_value=iob)
    sess.run_with_iobinding = MagicMock()
    return sess, iob


class TestUltraMaxTextureRestore(unittest.TestCase):
    """`_restore_texture` is a pure function — no session, no GPU."""

    def _noisy(self, base=128, amp=9, size=512, seed=0):
        rng = np.random.default_rng(seed)
        n = rng.normal(0.0, amp, (size, size)).astype(np.float32)
        img = np.clip(base + n, 0, 255).astype(np.uint8)
        return np.repeat(img[:, :, None], 3, axis=2)

    def test_shape_dtype_and_finiteness_preserved(self):
        src = self._noisy()
        restored = np.full((512, 512, 3), 128, np.uint8)
        out = UM.Enhance_UltraMax._restore_texture(restored, src, 0.55)
        self.assertEqual(out.shape, (512, 512, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.isfinite(out).all())

    def test_zero_gain_is_a_bit_identical_no_op(self):
        src = self._noisy()
        restored = np.full((512, 512, 3), 128, np.uint8)
        out = UM.Enhance_UltraMax._restore_texture(restored, src, 0.0)
        self.assertTrue(np.array_equal(out, restored))

    def test_flat_skin_gets_the_source_texture_back(self):
        """A flat restored face + a textured source = texture on the output."""
        src = self._noisy(amp=9)
        restored = np.full((512, 512, 3), 128, np.uint8)
        out = UM.Enhance_UltraMax._restore_texture(restored, src, 0.55)
        # The restored input has zero variance; the output must not.
        self.assertGreater(float(out.std()), 1.0)

    def test_structure_in_the_restored_face_is_protected(self):
        """Where the codebook drew an edge, nothing is injected.

        This is the property that separates this from the unsharp filter it
        replaced: the eye/lash/lip detail CodeFormer produced is the sharpest
        source in the pipeline, and every filter applied over it made it worse.
        """
        src = self._noisy(amp=9)
        restored = np.full((512, 512, 3), 128, np.uint8)
        # A hard edge, of the kind a lid crease or lip margin is.
        restored[:, 256:] = 200

        out = UM.Enhance_UltraMax._restore_texture(restored, src, 0.55)
        delta = out.astype(np.int16) - restored.astype(np.int16)

        # A band straddling the edge must be essentially untouched, while flat
        # skin well away from it must have received texture.
        on_edge = np.abs(delta[:, 248:264]).mean()
        off_edge = np.abs(delta[:, 40:180]).mean()
        self.assertLess(on_edge, off_edge * 0.5)

    def test_a_textureless_source_adds_nothing(self):
        """Self-limiting: no real texture in means nothing invented out."""
        src = np.full((512, 512, 3), 128, np.uint8)
        restored = np.full((512, 512, 3), 140, np.uint8)
        out = UM.Enhance_UltraMax._restore_texture(restored, src, 0.55)
        self.assertLessEqual(int(np.abs(out.astype(np.int16)
                                        - restored.astype(np.int16)).max()), 1)

    def test_does_not_sharpen_the_restored_face(self):
        """The old filter was an unsharp mask; this one must not be.

        With no texture in the source, a structured restored face must come back
        with its edge contrast unchanged — a sharpener would raise it.
        """
        src = np.full((512, 512, 3), 128, np.uint8)
        restored = np.full((512, 512, 3), 120, np.uint8)
        restored[:, 256:] = 190
        out = UM.Enhance_UltraMax._restore_texture(restored, src, 0.55)
        import cv2
        lap = lambda im: float(cv2.Laplacian(
            cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())
        self.assertAlmostEqual(lap(out), lap(restored), delta=lap(restored) * 0.02)


class TestUltraMaxRun(unittest.TestCase):
    def setUp(self):
        self._saved_pool_env = os.environ.get('ROOP_TRT_POOL')
        self._saved_gain = os.environ.get('ROOP_ULTRAMAX_TEXTURE')
        roop.globals.execution_threads = 16
        from roop import session_pool
        session_pool._pool_cache.clear()

    def tearDown(self):
        for var, saved in (('ROOP_TRT_POOL', self._saved_pool_env),
                           ('ROOP_ULTRAMAX_TEXTURE', self._saved_gain)):
            if saved is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = saved
        from roop import session_pool
        session_pool._pool_cache.clear()

    def _pools(self, n):
        os.environ['ROOP_TRT_POOL'] = str(n)
        from roop import session_pool
        session_pool._pool_cache.clear()

    def _make(self, out_chw=None):
        """An initialised processor with the ONNX session replaced, so the test
        exercises the host pre/post path without loading 180 MB of weights."""
        p = UM.Enhance_UltraMax()
        p.plugin_options = {'devicename': 'cuda'}
        p.devicename = 'cuda'
        p.in_dtype = np.float16
        p._lut = ((np.arange(256, dtype=np.float32) / 127.5) - 1.0).astype(np.float16)
        if out_chw is None:
            out_chw = np.zeros((3, 512, 512), np.float32)
        sess, iob = p._fake = _fake_session(out_chw)
        p.session, p.io_binding = sess, iob
        p.model_inputs = [MagicMock(name='i0'), MagicMock(name='i1')]
        p.model_inputs[0].name, p.model_inputs[1].name = 'x', 'w'
        p.model_outputs = [MagicMock()]
        return p

    def test_run_outputs_a_valid_frame_at_scale_one(self):
        p = self._make()
        frame = np.full((512, 512, 3), 128, np.uint8)
        out, scale = p.Run(None, None, frame)
        self.assertEqual(out.shape, (512, 512, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(scale, 1)

    def test_input_is_normalised_rgb_chw_in_the_model_dtype(self):
        """The LUT gather has to produce exactly what the graph expects."""
        p = self._make()
        frame = np.zeros((512, 512, 3), np.uint8)
        frame[:, :, 2] = 255          # pure red, BGR in
        p.Run(None, None, frame)
        x = p.io_binding.bound['x']
        self.assertEqual(x.shape, (1, 3, 512, 512))
        self.assertEqual(x.dtype, np.float16)
        # RGB order out: R saturated to +1, G and B at -1.
        self.assertAlmostEqual(float(x[0, 0].mean()), 1.0, places=2)
        self.assertAlmostEqual(float(x[0, 1].mean()), -1.0, places=2)
        self.assertAlmostEqual(float(x[0, 2].mean()), -1.0, places=2)

    def test_output_is_bgr_and_rescaled(self):
        """CHW RGB in [-1,1] must come back HWC BGR in [0,255]."""
        out_chw = np.stack([np.full((512, 512), 1.0, np.float32),    # R
                            np.full((512, 512), -1.0, np.float32),   # G
                            np.full((512, 512), -1.0, np.float32)])  # B
        p = self._make(out_chw)
        os.environ['ROOP_ULTRAMAX_TEXTURE'] = '0'
        out, _ = p.Run(None, None, np.zeros((512, 512, 3), np.uint8))
        self.assertLess(int(out[:, :, 0].mean()), 4)     # B low
        self.assertLess(int(out[:, :, 1].mean()), 4)     # G low
        self.assertGreater(int(out[:, :, 2].mean()), 250)  # R high

    def test_non_finite_output_falls_back_to_the_unenhanced_crop(self):
        """A half-precision graph is one overflow away from a black FACE:
        np.clip does not remove NaN and uint8(NaN) is 0."""
        out_chw = np.zeros((3, 512, 512), np.float32)
        out_chw[0, 5, 5] = np.nan
        p = self._make(out_chw)
        frame = np.full((512, 512, 3), 77, np.uint8)
        out, scale = p.Run(None, None, frame)
        self.assertTrue(np.array_equal(out, frame))
        self.assertEqual(scale, 1)

    def test_fidelity_is_bound_from_globals_every_call(self):
        p = self._make()
        old = getattr(roop.globals, 'codeformer_fidelity', 0.5)
        try:
            roop.globals.codeformer_fidelity = 0.83
            p.Run(None, None, np.zeros((512, 512, 3), np.uint8))
            self.assertAlmostEqual(float(p.io_binding.bound['w'][0]), 0.83)
        finally:
            roop.globals.codeformer_fidelity = old

    def test_smaller_crop_is_upsampled_and_reports_its_scale(self):
        p = self._make()
        out, scale = p.Run(None, None, np.full((256, 256, 3), 128, np.uint8))
        self.assertEqual(out.shape[:2], (512, 512))
        self.assertEqual(scale, 2)

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
        p.Run(None, None, np.zeros((512, 512, 3), np.uint8))
        line = p.cost_summary()
        self.assertIn("1 faces", line)
        self.assertIn("codeformer.fp16", line)

    def test_texture_gain_env_disables_the_restore(self):
        os.environ['ROOP_ULTRAMAX_TEXTURE'] = '0'
        p = self._make()
        p.Run(None, None, np.full((512, 512, 3), 128, np.uint8))
        self.assertEqual(p._textured, 0)
        self.assertEqual(p._faces, 1)


if __name__ == '__main__':
    unittest.main()
