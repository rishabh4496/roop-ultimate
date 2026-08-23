"""FP16 precision failures that the NaN guard cannot see.

`is_usable` rejects non-finite output, which catches the LOUD failure: GPEN at
1024/2048 overflows under TensorRT FP16, `np.clip` does not strip NaN, and
uint8(NaN) is 0, so the face comes out solid black.

GFPGAN v1.4 fails a different way, and it shipped undetected because of it. Its
FP16 engine does not overflow — it COLLAPSES. Measured 2026-08-24, RTX 4070,
same input and the same pre/post on both sides:

    TRT fp16   raw range [-0.47, -0.14]   pixel std 16.0   detail 0.08
    TRT fp32   raw range [-1.00,  1.00]   pixel std 65.2   detail 4.35
    CUDA       raw range [-1.00,  1.00]   pixel std 65.2   detail 4.35

fp32 matches the CUDA reference to 0.03/255 and fp16 differs from it by 59/255.
Every fp16 value is finite, so nothing fired, and the enhancer returned a
uniform grey face that still looked like a valid image. It was also FAST that
way (23 ms against the fixed 41.7 ms), which is how it came to be documented in
the UI as the cheapest restorer.

Two defences, both tested here: force FP32 for the models that need it, and a
guard that notices a finite-but-degenerate result.
"""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.processors.enhance_common import (fp32_trt_providers, is_usable,
                                            looks_collapsed)

TRT = ('TensorrtExecutionProvider', {
    'device_id': 0,
    'trt_fp16_enable': True,
    'trt_engine_cache_path': 'models/trt_cache/mixed',
})


class TestForcedFp32Providers(unittest.TestCase):
    def test_it_disables_fp16_and_moves_the_engine_cache(self):
        out = fp32_trt_providers([TRT, 'CUDAExecutionProvider'], 'gfpgan')
        name, opts = out[0]
        self.assertEqual(name, 'TensorrtExecutionProvider')
        self.assertFalse(opts['trt_fp16_enable'])
        self.assertTrue(opts['trt_engine_cache_path'].endswith('_gfpgan_fp32'),
                        "an FP32 engine must not share a cache with the FP16 "
                        "engines built for detection and the other stages")

    def test_each_model_gets_its_own_cache(self):
        a = fp32_trt_providers([TRT], 'gfpgan')[0][1]['trt_engine_cache_path']
        b = fp32_trt_providers([TRT], 'gpen')[0][1]['trt_engine_cache_path']
        self.assertNotEqual(a, b)

    def test_non_tensorrt_providers_pass_through_untouched(self):
        src = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.assertEqual(fp32_trt_providers(src, 'gfpgan'), src)

    def test_the_original_provider_list_is_not_mutated(self):
        original = [TRT, 'CUDAExecutionProvider']
        fp32_trt_providers(original, 'gfpgan')
        self.assertTrue(original[0][1]['trt_fp16_enable'],
                        "callers share this list; mutating it would silently "
                        "force every other stage to FP32 too")

    def test_the_env_override_opts_back_into_fp16(self):
        key = 'ROOP_GFPGAN_FP16'
        saved = os.environ.get(key)
        try:
            os.environ[key] = '1'
            out = fp32_trt_providers([TRT], 'gfpgan')
            self.assertTrue(out[0][1]['trt_fp16_enable'])
        finally:
            if saved is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = saved


class TestCollapseGuard(unittest.TestCase):
    def _crop(self, std=40.0, seed=0, size=256):
        rng = np.random.default_rng(seed)
        a = np.clip(rng.normal(120, std, (size, size, 3)), 0, 255)
        return a.astype(np.uint8)

    def test_it_catches_the_real_failure(self):
        """The measured case: input with normal spread, output flat."""
        source = self._crop(std=40)
        collapsed = np.clip(np.random.default_rng(1).normal(84, 16.0 / 3,
                                                           source.shape), 0, 255
                            ).astype(np.uint8)
        self.assertTrue(is_usable(collapsed.astype(np.float32)),
                        "it is finite — which is exactly why is_usable misses it")
        self.assertTrue(looks_collapsed(collapsed, source))

    def test_it_does_not_fire_on_a_real_restoration(self):
        source = self._crop(std=40, seed=0)
        restored = self._crop(std=46, seed=2)          # a bit more contrast
        self.assertFalse(looks_collapsed(restored, source))

    def test_it_does_not_fire_on_a_legitimately_soft_face(self):
        """A low-contrast but honest restoration must pass. The guard is
        deliberately conservative: a false positive silently disables the
        enhancer, which is worse than the bug it protects against."""
        source = self._crop(std=40, seed=0)
        soft = self._crop(std=22, seed=3)              # 55% of the input's spread
        self.assertFalse(looks_collapsed(soft, source))

    def test_it_stays_quiet_when_the_input_itself_is_flat(self):
        """A uniform input (a blown-out or masked crop) legitimately produces a
        uniform output; there is nothing to conclude from that."""
        flat = np.full((256, 256, 3), 130, np.uint8)
        self.assertFalse(looks_collapsed(flat, flat))

    def test_it_never_raises(self):
        for bad in (None, np.zeros((0, 0, 3), np.uint8), 'not an image'):
            with self.subTest(value=type(bad).__name__):
                self.assertIsInstance(looks_collapsed(bad, bad), bool)


class TestGfpganIsWiredToTheFix(unittest.TestCase):
    def test_it_forces_fp32_and_checks_for_collapse(self):
        with open(os.path.join(APP, 'roop', 'processors', 'Enhance_GFPGAN.py'),
                  encoding='utf-8') as f:
            src = f.read()
        self.assertIn('fp32_trt_providers', src,
                      "GFPGAN must not build an FP16 TensorRT engine")
        self.assertIn('looks_collapsed', src)
        self.assertNotIn('providers=roop.globals.execution_providers', src,
                         "the raw provider list is the FP16 path this fixes")


if __name__ == '__main__':
    unittest.main()
