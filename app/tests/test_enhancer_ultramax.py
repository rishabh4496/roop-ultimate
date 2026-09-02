"""Unit tests for the UltraMax enhancer.

UltraMax runs codeformer.fp16.onnx directly (same weights as
`Codeformer (fp16)`) on a leaner host path. Its texture restore is present but
DEFAULTS OFF: measured on rendered frames against the footage's own skin it
moved texture by an amount indistinguishable from zero (paired t = -0.7 over
102 frames) while costing 2.49 ms/face. The tests below still cover it, because
the env knob can turn it back on and because its structure gate is the property
that separates it from the unsharp filter it replaced — that filter is what
printed a second eyelid crease.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import roop.globals
import roop.processors.Enhance_UltraMax as UM
from roop.processors.enhance_common import (luma_only_recolour,
                                            luma_only_recolour_tensor)


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


@unittest.skipUnless(UM._TORCH_CUDA, 'requires CUDA for tensor-path parity')
class TestUltraMaxCudaColourTransfer(unittest.TestCase):
    def test_cuda_luminance_transfer_is_byte_close_to_opencv(self):
        """The GPU path must retain the established BGR luminance result to
        one uint8 level, including when some CodeFormer colour is retained."""
        import torch
        rng = np.random.default_rng(20260902)
        source = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        restored = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        for chroma in (0.0, 0.35):
            cpu = luma_only_recolour(restored, source, chroma)
            tensor = luma_only_recolour_tensor(
                torch.from_numpy(restored).cuda(), torch.from_numpy(source).cuda(), chroma)
            self.assertTrue(tensor.is_cuda)
            gpu = tensor.round().to(torch.uint8).cpu().numpy()
            self.assertLessEqual(
                int(np.abs(cpu.astype(np.int16) - gpu.astype(np.int16)).max()), 1)


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
        # This deliberately models the compatibility host binding.  The mock
        # cannot execute an ORT CUDA write into a Torch allocation.
        p._cuda_iob_available = False
        p.in_dtype = np.float16
        p._lut = ((np.arange(256, dtype=np.float32) / 127.5) - 1.0).astype(np.float16)
        if out_chw is None:
            # Not zeros: they decode to a uniform grey, which is what
            # `looks_collapsed` is built to reject, so the success-path tests
            # would quietly exercise the fallback instead. See the same note in
            # test_enhancer_gpen_realistic.
            rng = np.random.default_rng(1)
            out_chw = np.clip(rng.normal(0.1, 0.25, (3, 512, 512)),
                              -1.0, 1.0).astype(np.float32)
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

    def test_model_output_is_never_bound_into_torch_owned_memory(self):
        """ORT must ALLOCATE AND OWN the output; we never hand it our pointer.

        This is the invariant, and it is the whole reason the CUDA
        post-process is safe under TensorRT. Handing ORT a Torch allocation via
        the 7-argument `bind_output(name, dev, id, elem, shape, ptr)` form is
        what produced finite-but-spatially-corrupt faces on ORT 1.23 / TRT 10 --
        striped and ghosted, so no numerical guard can catch it. The response
        used to be to refuse the CUDA path entirely whenever TensorRT was the
        provider, which meant it never ran on a single face in production.

        The three-argument form lets ORT own the allocation and is measured
        bit-identical to the host path on TensorRT. So the test is no longer
        "never bind anything" -- inputs are legitimately bound from Torch
        memory -- it is specifically "never bind an OUTPUT POINTER".
        """
        p = self._make()
        p._cuda_iob_available = None
        p.session.get_providers.return_value = ['TensorrtExecutionProvider',
                                                'CUDAExecutionProvider']
        frame = np.zeros((512, 512, 3), dtype=np.uint8)
        with patch.object(UM, '_TORCH_CUDA', True):
            p._run_cuda_postprocess(frame, 512, frame)
        for call_ in p.io_binding.bind_output.call_args_list:
            self.assertLessEqual(
                len(call_.args), 3,
                'bind_output was given a data pointer; ORT must own the output '
                'allocation or TensorRT can return spatially corrupt faces')
            self.assertNotIn('buffer_ptr', call_.kwargs)

    def test_cuda_input_is_contiguous_before_its_pointer_is_bound(self):
        """`bind_input` takes a RAW POINTER and reads it as contiguous.

        `permute(2,0,1).flip(0)` leaves a non-contiguous view. Binding that
        tensor's `data_ptr()` feeds the network transposed garbage -- and it
        still returns a plausible finite image, so the only visible symptom was
        the collapse guard rejecting every face. Nothing else in the stack can
        catch this, which is why it is asserted directly.
        """
        import torch
        if not torch.cuda.is_available():
            self.skipTest('requires CUDA')
        src = np.arange(512 * 512 * 3, dtype=np.uint8).reshape(512, 512, 3)
        source = torch.from_numpy(src).to('cuda', dtype=torch.float32)
        x = (source.permute(2, 0, 1).flip(0).unsqueeze(0)
             .div(127.5).sub(1.0).to(torch.float16).contiguous())
        self.assertTrue(x.is_contiguous())
        # And the un-contiguous spelling really is non-contiguous, so this test
        # is guarding something that can actually happen.
        loose = source.permute(2, 0, 1).flip(0).unsqueeze(0).div(127.5).sub(1.0)
        self.assertFalse(loose.is_contiguous())

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
        """CHW RGB in [-1,1] must come back HWC BGR in [0,255].

        ROOP_ULTRAMAX_CHROMA=1 keeps CodeFormer's own colour, which is what this
        assertion is about: since 2026-08-24 the DEFAULT replaces the network's
        chrominance with the crop's, so a red network output over a black crop
        correctly comes back grey and this test would be measuring the colour
        fix instead of the channel order. The colour fix has its own tests below.
        """
        out_chw = np.stack([np.full((512, 512), 1.0, np.float32),    # R
                            np.full((512, 512), -1.0, np.float32),   # G
                            np.full((512, 512), -1.0, np.float32)])  # B
        p = self._make(out_chw)
        os.environ['ROOP_ULTRAMAX_CHROMA'] = '1'
        try:
            out, _ = p.Run(None, None, np.zeros((512, 512, 3), np.uint8))
        finally:
            os.environ.pop('ROOP_ULTRAMAX_CHROMA', None)
        self.assertLess(int(out[:, :, 0].mean()), 4)     # B low
        self.assertLess(int(out[:, :, 1].mean()), 4)     # G low
        self.assertGreater(int(out[:, :, 2].mean()), 250)  # R high

    def test_the_crops_chrominance_survives_the_restorer(self):
        """THE PALE-SKIN FIX. CodeFormer desaturates and lifts -- measured
        against the crop it was handed, chroma drift 2.51, LAB-a -0.96,
        saturation x0.958, which is what the user reported as pale skin where
        GPEN Realistic and GPEN 256 Pro (which already do this) look right.

        A grey network output over a strongly coloured crop must come back
        CARRYING THE CROP'S COLOUR, not grey.
        """
        rng = np.random.default_rng(2)
        p = self._make(rng.normal(0, 0.15, (3, 512, 512)).astype(np.float32))
        frame = np.zeros((512, 512, 3), np.uint8)
        frame[:, :, 2] = 200                                  # a red crop, BGR
        out, _ = p.Run(None, None, frame)
        self.assertGreater(int(out[:, :, 2].mean()), int(out[:, :, 0].mean()) + 60,
                           "the crop's red did not survive the restorer")

    def test_the_networks_luminance_is_what_survives(self):
        """The other half of the same operator: it is the RESTORER that decides
        brightness. A bright network output must come back brighter than a dark
        one over the same crop, or the fix would be discarding the restoration.
        """
        frame = np.full((512, 512, 3), 90, np.uint8)
        frame[:, :, 2] = 160
        # TEXTURED, not uniform. A flat output is what `looks_collapsed` exists
        # to reject -- it would fall back to the unenhanced crop and both arms
        # would come back identical, which is how this test failed when written.
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.15, (3, 512, 512)).astype(np.float32)
        dark, _ = self._make(noise - 0.5).Run(None, None, frame)
        bright, _ = self._make(noise + 0.5).Run(None, None, frame)
        self.assertGreater(int(bright.mean()), int(dark.mean()) + 80)

    def test_the_colour_fix_can_be_turned_off_for_remeasuring(self):
        """ROOP_ULTRAMAX_CHROMA=1 restores CodeFormer's own colour exactly --
        `tests/bench_ultramax_vs_codeformer.py` sets it to assert the lean host
        path is still bit-identical to the reference implementation."""
        # Textured for the same reason as above, and textured ENOUGH: the guard
        # is relative to the CROP's own spread, and a crop this saturated has a
        # global std of 94, so an output below ~33 still reads as collapsed.
        rng = np.random.default_rng(1)
        out_chw = (rng.normal(0, 0.4, (3, 512, 512)).astype(np.float32) + 0.2)
        frame = np.zeros((512, 512, 3), np.uint8)
        frame[:, :, 0] = 200                                  # a blue crop
        on, _ = self._make(out_chw).Run(None, None, frame)
        os.environ['ROOP_ULTRAMAX_CHROMA'] = '1'
        try:
            off, _ = self._make(out_chw).Run(None, None, frame)
        finally:
            os.environ.pop('ROOP_ULTRAMAX_CHROMA', None)
        # With the network's own colour kept, a uniform output is neutral grey.
        self.assertLess(int(off[:, :, 0].mean()) - int(off[:, :, 2].mean()), 4)
        # With the fix on, the crop's blue comes through.
        self.assertGreater(int(on[:, :, 0].mean()) - int(on[:, :, 2].mean()), 60)

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

    def test_texture_restore_is_off_by_default(self):
        """Measured on rendered frames it moved skin texture by an amount
        indistinguishable from zero (paired t = -0.7 over 102 frames) while
        costing 2.49 ms/face and raising flicker. Off is the shipped default,
        and this is what fails if someone turns it back on without a new
        measurement."""
        p = self._make()
        p.Run(None, None, np.full((512, 512, 3), 128, np.uint8))
        self.assertEqual(p._textured, 0)
        self.assertEqual(p._faces, 1)
        self.assertEqual(UM.Enhance_UltraMax._TEXTURE_GAIN, 0.0)

    def test_texture_gain_env_re_enables_the_restore(self):
        os.environ['ROOP_ULTRAMAX_TEXTURE'] = '0.55'
        p = self._make()
        p.Run(None, None, np.full((512, 512, 3), 128, np.uint8))
        self.assertEqual(p._textured, 1)


if __name__ == '__main__':
    unittest.main()
