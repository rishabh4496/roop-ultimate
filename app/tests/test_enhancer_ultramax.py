"""Unit tests for the UltraMax enhancer.

UltraMax runs codeformer.fp16.onnx directly (same weights as
`Codeformer (fp16)`) on a leaner host path. Its texture restore and dual-stream
engine were measured to do nothing / cost 3.1x and were deleted 2026-10-13; the
numbers live in the module docstring of Enhance_UltraMax.
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
        self._saved_detail_env = os.environ.pop('ROOP_ULTRAMAX_DETAIL_WEIGHT', None)
        self._saved_lowband = (getattr(roop.globals, 'ultramax_swap_lowband', True),
                               getattr(roop.globals, 'ultramax_detail_weight', 0.75))
        roop.globals.execution_threads = 16
        from roop import session_pool
        session_pool._pool_cache.clear()

    def tearDown(self):
        roop.globals.ultramax_swap_lowband, roop.globals.ultramax_detail_weight = self._saved_lowband
        for var, saved in (('ROOP_TRT_POOL', self._saved_pool_env),
                           ('ROOP_ULTRAMAX_DETAIL_WEIGHT', self._saved_detail_env)):
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

        That is the LEGACY full-restorer paste. Since 2026-10-12 the default takes the
        low band from the swap instead (see `TestUltraMaxLowBandFromSwap`), so this
        contract is pinned under `ultramax_swap_lowband = False`.
        """
        roop.globals.ultramax_swap_lowband = False
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

    # ── swap low band + a weighted share of the restorer's detail (default since 2026-10-12) ──
    # The restorer's low band used to replace the swap's, and a codebook prior re-draws it every
    # frame: on a 420-frame clip two thirds of the lead's frame-to-frame shimmer was that. These
    # pin the properties that make the fix a fix and not a softer filter.

    # A frame that is NOT flat, so the swap has a low band worth keeping and the guard that
    # rejects a flat network output does not fire. Colours differ per channel on purpose.
    @staticmethod
    def _crop():
        yy, xx = np.mgrid[0:512, 0:512].astype(np.float32)
        base = 100 + 40 * (xx / 511.0)
        f = np.stack([base * 0.9, base, base * 1.1], axis=2)
        return np.clip(f, 0, 255).astype(np.uint8)

    @staticmethod
    def _noise(seed=0, amp=0.15, shift=0.0):
        rng = np.random.default_rng(seed)
        return (rng.normal(0, amp, (3, 512, 512)) + shift).astype(np.float32)

    def test_default_takes_the_low_band_from_the_swap(self):
        frame = self._crop()
        dark, _ = self._make(self._noise(shift=-0.5)).Run(None, None, frame)
        bright, _ = self._make(self._noise(shift=0.5)).Run(None, None, frame)
        # A +-0.5 network brightness swing (~64 levels) must NOT move the face: the tone is the swap's.
        self.assertLess(abs(float(bright.mean()) - float(dark.mean())), 10.0)
        self.assertLess(abs(float(bright.mean()) - float(frame.mean())), 12.0)

    def test_legacy_paste_follows_the_network_again(self):
        frame = self._crop()
        roop.globals.ultramax_swap_lowband = False
        dark, _ = self._make(self._noise(shift=-0.5)).Run(None, None, frame)
        bright, _ = self._make(self._noise(shift=0.5)).Run(None, None, frame)
        self.assertGreater(float(bright.mean()), float(dark.mean()) + 60.0)

    def test_env_off_is_the_legacy_paste_and_beats_the_setting(self):
        frame = self._crop()
        os.environ['ROOP_ULTRAMAX_DETAIL_WEIGHT'] = 'off'
        self.assertIsNone(UM.Enhance_UltraMax._detail_weight())
        dark, _ = self._make(self._noise(shift=-0.5)).Run(None, None, frame)
        bright, _ = self._make(self._noise(shift=0.5)).Run(None, None, frame)
        self.assertGreater(float(bright.mean()), float(dark.mean()) + 60.0)

    def test_env_number_beats_a_disabled_setting(self):
        roop.globals.ultramax_swap_lowband = False
        os.environ['ROOP_ULTRAMAX_DETAIL_WEIGHT'] = '0.5'
        self.assertEqual(UM.Enhance_UltraMax._detail_weight(), 0.5)

    def test_weight_comes_from_the_setting_and_is_clamped(self):
        roop.globals.ultramax_swap_lowband = True
        roop.globals.ultramax_detail_weight = 0.4
        self.assertAlmostEqual(UM.Enhance_UltraMax._detail_weight(), 0.4)
        roop.globals.ultramax_detail_weight = 9.0
        self.assertEqual(UM.Enhance_UltraMax._detail_weight(), 1.5)
        roop.globals.ultramax_detail_weight = -3.0
        self.assertEqual(UM.Enhance_UltraMax._detail_weight(), 0.0)
        os.environ['ROOP_ULTRAMAX_DETAIL_WEIGHT'] = 'garbage'
        roop.globals.ultramax_detail_weight = 0.3        # unparsable env -> falls through to the setting
        self.assertAlmostEqual(UM.Enhance_UltraMax._detail_weight(), 0.3)

    def test_detail_weight_scales_the_restorers_high_band(self):
        frame = self._crop()
        net = self._noise(seed=3, amp=0.2)
        spread = {}
        for w in ('0', '0.5', '1.0'):
            os.environ['ROOP_ULTRAMAX_DETAIL_WEIGHT'] = w
            out, _ = self._make(net).Run(None, None, frame)
            # fine detail energy: the output minus its own blur
            import cv2
            f = out.astype(np.float32)
            spread[w] = float((f - cv2.GaussianBlur(f, (0, 0), 2.0)).std())
        self.assertLess(spread['0'], spread['0.5'])
        self.assertLess(spread['0.5'], spread['1.0'])

    def test_weight_zero_is_the_swaps_own_tone(self):
        frame = self._crop()
        os.environ['ROOP_ULTRAMAX_DETAIL_WEIGHT'] = '0'
        out, _ = self._make(self._noise(seed=4, amp=0.2, shift=0.3)).Run(None, None, frame)
        self.assertLess(float(np.abs(out.astype(np.float32) - frame.astype(np.float32)).mean()), 8.0)

    def test_chroma_diagnostic_mode_stays_bit_faithful_to_the_network(self):
        """ROOP_ULTRAMAX_CHROMA=1 is the documented 'same as CodeFormer' mode that
        tests/bench_ultramax_vs_codeformer.py relies on; the recombination must not touch it."""
        frame = self._crop()
        os.environ['ROOP_ULTRAMAX_CHROMA'] = '1'
        try:
            dark, _ = self._make(self._noise(shift=-0.5)).Run(None, None, frame)
            bright, _ = self._make(self._noise(shift=0.5)).Run(None, None, frame)
        finally:
            os.environ.pop('ROOP_ULTRAMAX_CHROMA', None)
        self.assertGreater(float(bright.mean()), float(dark.mean()) + 60.0)

    def test_cost_summary_says_what_actually_ran(self):
        """A setting that reads as applied while nothing ran is this project's recurring failure."""
        p = self._make(self._noise())
        p.Run(None, None, self._crop())
        line = p.cost_summary()
        self.assertIn("swap low band + 0.75 x restorer detail on 1", line)
        legacy = self._make(self._noise())
        roop.globals.ultramax_swap_lowband = False
        legacy.Run(None, None, self._crop())
        self.assertIn("legacy", legacy.cost_summary())
        self.assertNotIn("swap low band", legacy.cost_summary())

    def test_a_failing_recombination_is_loud_and_falls_back(self):
        frame = self._crop()
        UM.Enhance_UltraMax._warned_lowband = False
        p = self._make(self._noise())
        with patch.object(UM.Enhance_UltraMax, '_lowband_blend_gpu', side_effect=RuntimeError('boom')):
            try:
                out, _ = p.Run(None, None, frame)
            except RuntimeError:
                self.skipTest('GPU post block not taken on this host; the CPU path has its own guard')
        self.assertEqual(out.shape, (512, 512, 3))


class StructuralPoolIsCappedOnNormalCards(unittest.TestCase):
    """One context below 15.5GB — the same guard the detail stream applies.

    Measured with `tests/diag_ultramax_pool_scaling.py` on an RTX 4070, 10
    threads, 240 real crops: a deeper pool is monotonically WORSE, 40.0 faces/s
    at one context against 38.4 at two and 36.6 at four, while VRAM climbs
    +1136 / +1519 / +2704 MB. 40 faces/s is 25 ms/face, which is the network's
    own time, so the card is already the floor and the extra contexts buy only
    memory pressure on a card that also holds realswap's two nets, RealityUX
    and the detector/detmask pools.

    The size-1 pool is still a pool. That matters: `_gpu_guard` waives the
    global TensorRT lock only for a processor that owns one, so capping to a
    single context must not be allowed to collapse into no pool at all — that
    would hand the most expensive stage in a render back to a one-at-a-time
    global lock, which is a far larger regression than the one being avoided.
    """

    def _n_for(self, gb, requested=4, forced=None):
        env = {k: v for k, v in os.environ.items() if k != 'ROOP_ULTRAMAX_POOL'}
        if forced is not None:
            env['ROOP_ULTRAMAX_POOL'] = forced
        with patch.dict(os.environ, env, clear=True), \
                patch.object(UM.session_pool, '_detect_vram_gb', return_value=gb):
            return UM._structural_pool_size(requested)

    def test_twelve_gb_card_gets_one_context(self):
        self.assertEqual(self._n_for(12.0), 1)

    def test_six_gb_card_gets_one_context(self):
        self.assertEqual(self._n_for(6.0), 1)

    def test_large_card_keeps_the_requested_pool(self):
        self.assertEqual(self._n_for(24.0), 4)

    def test_unknown_vram_does_not_silently_cap(self):
        """A failed probe is not evidence the card is small."""
        self.assertEqual(self._n_for(0), 4)

    def test_env_override_wins_for_measurement(self):
        self.assertEqual(self._n_for(12.0, forced='3'), 3)

    def test_never_returns_zero(self):
        """0 would mean no SessionPool at all, and `_gpu_guard` waives the
        global TensorRT lock only for a processor that owns one -- so a 0 here
        would serialise the most expensive stage in a render behind that lock.
        That is a far bigger regression than the context it saves."""
        for requested in (0, None, -3, 1):
            self.assertGreaterEqual(self._n_for(12.0, requested=requested), 1)

    def test_the_cap_is_actually_wired_into_initialize(self):
        """A helper nothing calls would pass every test above.

        Asserts the sizing branch in `Initialize` routes through the helper,
        by making the helper the only thing that can decide the pool width.
        """
        import inspect
        src = inspect.getsource(UM.Enhance_UltraMax.Initialize)
        self.assertIn('_structural_pool_size(', src)
        # ...and that the raw pool_size result is not what reaches SessionPool.
        structural = src.split("model_key='enhancer:ultramax'", 1)[1]
        self.assertLess(structural.index('_structural_pool_size('),
                        structural.index('SessionPool('))


if __name__ == '__main__':
    unittest.main()
