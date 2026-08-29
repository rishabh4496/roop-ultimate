"""Unit tests for the model-specific precision gate.

These tests deliberately use provider descriptions and tiny temporary model
files; they do not claim to be GPU benchmarks.  The hardware gates are tested
by the real benchmark harness and the two-device validation records.
"""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from roop.precision_policy import (  # noqa: E402
    POLICIES,
    canonical_model_key,
    decision_cache_key,
    matrix,
    providers_for,
    resolve,
)


TRT = [('TensorrtExecutionProvider', {
    'trt_fp16_enable': True,
    'trt_engine_cache_path': 'test-trt-cache',
})]


class TestPrecisionPolicy(unittest.TestCase):
    @staticmethod
    def _hardware(**overrides):
        values = {
            'cuda_available': True,
            'tensorrt_available': True,
            'fp16_supported': True,
            'bf16_supported': False,
            'int8_supported': True,
            'fp8_supported': True,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_matrix_has_all_precision_and_fallback_columns(self):
        required = {
            'model', 'backend', 'fp32', 'fp16', 'mixed', 'bf16', 'int8',
            'fp8', 'cuda_fallback', 'cpu_fallback', 'recommended',
            'trt_supported',
        }
        self.assertGreaterEqual(len(matrix()), 20)
        for row in matrix():
            self.assertTrue(required.issubset(row))
            self.assertIn(row['bf16'], ('unsupported', 'not-validated', 'safe', 'unsafe', 'candidate'))
            self.assertIn(row['int8'], ('unsupported', 'not-validated', 'safe', 'unsafe', 'candidate'))
            self.assertIn(row['fp8'], ('unsupported', 'not-validated', 'safe', 'unsafe', 'candidate'))

    def test_aliases_cover_requested_families(self):
        cases = {
            'GPEN 256': 'gpen_256',
            'GPEN 256 Pro': 'gpen_256_pro',
            'GPEN Realistic': 'gpen_realistic',
            'GPEN 1024': 'gpen_1024',
            'GPEN 2048': 'gpen_2048',
            'CodeFormer': 'codeformer',
            'UltraMax': 'codeformer_fp16',
            'GFPGAN': 'gfpgan',
            'RestoreFormer++': 'restoreformer_pp',
            'real_esrgan_x4': 'frame_upscaler',
            'liveportrait:warping': 'liveportrait',
            'face_detection:yoloface': 'face_detection',
            'recognition:adaface': 'recognition',
            'face_swap:inswapper': 'face_swap',
            'masking:xseg': 'masking',
            'masking_no_trt:mobilesam': 'masking_no_trt',
        }
        for alias, expected in cases.items():
            self.assertEqual(canonical_model_key(alias), expected, alias)

    def test_known_fp32_safeguards_are_not_bypassed(self):
        for model in ('gpen_1024', 'gpen_2048', 'gfpgan'):
            decision = resolve(model, 'mixed', TRT)
            self.assertEqual(decision.effective, 'fp32', model)
            self.assertTrue(decision.fallback, model)
            out, _ = providers_for(model, TRT, requested='mixed')
            self.assertFalse(out[0][1]['trt_fp16_enable'], model)
            self.assertNotEqual(out[0][1]['trt_engine_cache_path'], 'test-trt-cache')

    def test_safe_fp16_graph_and_mixed_candidate_are_preserved(self):
        decision = resolve('codeformer_fp16', 'fp16', TRT)
        self.assertEqual(decision.effective, 'fp16')
        out, _ = providers_for('codeformer_fp16', TRT, requested='fp16')
        self.assertTrue(out[0][1]['trt_fp16_enable'])

        decision = resolve('face_swap:inswapper', 'mixed', TRT)
        self.assertEqual(decision.effective, 'mixed')

    def test_bf16_candidate_is_explicit_and_isolated(self):
        out, decision = providers_for('liveportrait:stitching', TRT, requested='bf16')
        self.assertEqual(decision.effective, 'bf16')
        self.assertTrue(out[0][1]['trt_bf16_enable'])
        self.assertFalse(out[0][1]['trt_fp16_enable'])
        self.assertTrue(out[0][1]['trt_engine_cache_path'].endswith('_liveportrait_bf16'))

    def test_hardware_gate_rejects_exposed_but_unvalidated_modes(self):
        unsupported = self._hardware()
        decision = resolve('liveportrait:stitching', 'bf16', TRT,
                           hardware=unsupported)
        self.assertEqual(decision.effective, 'fp32')
        self.assertTrue(decision.fallback)

        # INT8/FP8 exposure is not enough: this project has no calibrated,
        # quality-validated provider implementation for either mode.
        for precision in ('int8', 'fp8'):
            decision = resolve('liveportrait:stitching', precision, TRT,
                               hardware=self._hardware(bf16_supported=True))
            self.assertEqual(decision.effective, 'fp32', precision)

    def test_hardware_gate_allows_only_the_real_bf16_provider_path(self):
        decision = resolve('liveportrait:stitching', 'bf16', TRT,
                           hardware=self._hardware(bf16_supported=True))
        self.assertEqual(decision.effective, 'bf16')
        out, _ = providers_for('liveportrait:stitching', TRT,
                                requested='bf16', hardware=self._hardware(
                                    bf16_supported=True))
        self.assertTrue(out[0][1]['trt_bf16_enable'])

    def test_frame_and_sam_paths_remove_trt(self):
        for model in ('frame_upscaler:real_esrgan_x4', 'rife', 'masking_no_trt:mobilesam'):
            out, decision = providers_for(model, TRT, requested='mixed')
            self.assertFalse(any('tensorrt' in str(p).lower() for p in out))
            self.assertEqual(decision.policy.trt_supported, 'no')
            self.assertEqual(decision.backend, 'cpu')

    def test_provider_input_is_not_mutated(self):
        original = [('TensorrtExecutionProvider', {
            'trt_fp16_enable': True,
            'trt_engine_cache_path': 'original',
        })]
        providers_for('gfpgan', original, requested='mixed')
        self.assertTrue(original[0][1]['trt_fp16_enable'])
        self.assertEqual(original[0][1]['trt_engine_cache_path'], 'original')

    def test_cache_identity_is_model_and_hardware_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model.onnx')
            with open(path, 'wb') as handle:
                handle.write(b'first model')
            with mock.patch('roop.precision_policy.cache_namespace', return_value={'gpu': 'RTX 3060', 'sm': '8.6'}):
                laptop = decision_cache_key('gpen_256', path, 'mixed', 'mixed', 0)
            with open(path, 'wb') as handle:
                handle.write(b'changed model')
            with mock.patch('roop.precision_policy.cache_namespace', return_value={'gpu': 'RTX 4070', 'sm': '8.9'}):
                desktop = decision_cache_key('gpen_256', path, 'mixed', 'mixed', 0)
            self.assertNotEqual(laptop, desktop)


if __name__ == '__main__':
    unittest.main()
