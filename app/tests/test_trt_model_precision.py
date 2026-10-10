"""Per-model TensorRT precision (roop/precision_policy.PRECISION_OVERRIDES / ROOP_TRT_MODEL_PRECISION).

`trt_precision` is one global setting. The measured table (docs/perf/trt_precision_fidelity.md) moves xseg, 2d106det and
1k3d68 to FP32 and leaves w600k_r50 mixed. What must hold:

  * the shipped table is exactly that, so a regression in the table is a red test, not a silent change in a render;
  * an entry changes ONLY that model's precision and its engine cache directory (the FP32 engine can never be loaded for the
    mixed request or the reverse), and a model with no entry is returned exactly as the global setting gives it;
  * an explicit `requested=` is authoritative (the harnesses name a precision and must get it);
  * the env var wins over the table, `global` drops an entry, a typo is loud and inert, swappers are refused;
  * the bundle seam (`bundle_member_providers`, used for buffalo_l's files) applies the precision step and nothing else.

No GPU: provider chains are plain tuples and the GPU-probing helpers are stubbed.
"""
import copy
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from roop import precision_policy as pp                         # noqa: E402


def _trt(cache, fp16=True):
    return [('TensorrtExecutionProvider', {
        'device_id': 0, 'trt_fp16_enable': fp16, 'trt_build_heuristics_enable': True,
        'trt_builder_optimization_level': 3, 'trt_engine_cache_enable': True,
        'trt_engine_cache_path': cache, 'trt_timing_cache_path': cache}),
        'CUDAExecutionProvider', 'CPUExecutionProvider']


def _opts(providers):
    return next(p[1] for p in providers if isinstance(p, tuple) and 'tensorrt' in p[0].lower())


class Base(unittest.TestCase):
    def setUp(self):
        pp._precision_announced.clear()
        del pp.precision_log[:]
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = os.path.join(self._tmp.name, 'mixed_cache')
        os.makedirs(self.cache)
        self._patches = [
            mock.patch.object(pp, 'cache_namespace', lambda precision, device_id=0: 'ns_' + precision),
            mock.patch.object(pp, 'write_decision_cache', lambda decision, directory=None: ''),
            mock.patch.object(pp, '_finalize', lambda key, path, providers, device_id=0: list(providers)),
            mock.patch.object(pp, '_global_precision', lambda: self.global_precision),
            mock.patch.dict(os.environ, {}, clear=False),
        ]
        self.global_precision = 'mixed'
        for p in self._patches:
            p.start()
        os.environ.pop(pp.PRECISION_OVERRIDE_ENV, None)

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self._tmp.cleanup()

    def chain(self, model, key, **kw):
        with redirect_stdout(StringIO()):
            return pp.providers_for(key, _trt(self.cache), model, **kw)


class ShippedTable(Base):
    def test_table_is_the_measured_one(self):
        self.assertEqual(pp.PRECISION_OVERRIDES, {'xseg': 'fp32', '2d106det': 'fp32', '1k3d68': 'fp32'})

    def test_w600k_r50_keeps_the_global_mixed(self):
        chain, decision = self.chain('models/buffalo_l/w600k_r50.onnx', 'recognition:buffalo_l')
        self.assertTrue(_opts(chain)['trt_fp16_enable'])
        self.assertEqual(decision.effective, 'mixed')

    def test_models_not_in_the_table_are_unchanged(self):
        original = _trt(self.cache)
        before = copy.deepcopy(original)
        with redirect_stdout(StringIO()):
            chain, decision = pp.providers_for('face_detection:r50', original, 'models/retinaface_r50.onnx')
        self.assertEqual(chain, before)
        self.assertEqual(decision.effective, 'mixed')


class OverrideApplies(Base):
    def test_xseg_is_fp32_with_its_own_cache(self):
        chain, decision = self.chain('models/xseg.onnx', 'masking:xseg')
        self.assertEqual(decision.requested, 'fp32')
        self.assertFalse(_opts(chain)['trt_fp16_enable'])
        self.assertEqual(_opts(chain)['trt_engine_cache_path'], self.cache + '_masking_fp32')
        self.assertEqual(_opts(chain)['trt_timing_cache_path'], self.cache + '_masking_fp32')

    def test_the_callers_chain_is_not_mutated(self):
        original = _trt(self.cache)
        before = copy.deepcopy(original)
        with redirect_stdout(StringIO()):
            pp.providers_for('masking:xseg', original, 'models/xseg.onnx')
        self.assertEqual(original, before)

    def test_it_is_announced_once(self):
        out = StringIO()
        with redirect_stdout(out):
            pp.providers_for('masking:xseg', _trt(self.cache), 'models/xseg.onnx')
            pp.providers_for('masking:xseg', _trt(self.cache), 'models/xseg.onnx')
        self.assertEqual(out.getvalue().count('precision for xseg: fp32 (per-model; global mixed)'), 1)

    def test_explicit_requested_is_authoritative(self):
        chain, decision = self.chain('models/xseg.onnx', 'masking:xseg', requested='mixed')
        self.assertTrue(_opts(chain)['trt_fp16_enable'])
        self.assertEqual(decision.requested, 'mixed')

    def test_non_trt_chain_is_untouched(self):
        with redirect_stdout(StringIO()):
            chain, decision = pp.providers_for('masking:xseg', ['CUDAExecutionProvider', 'CPUExecutionProvider'],
                                               'models/xseg.onnx')
        self.assertEqual(chain, ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.assertFalse(decision.trt_enabled)

    def test_the_decision_key_differs_between_the_two_precisions(self):
        _c, fp32 = self.chain('models/xseg.onnx', 'masking:xseg')
        _c, mixed = self.chain('models/xseg.onnx', 'masking:xseg', requested='mixed')
        self.assertNotEqual(fp32.cache_key, mixed.cache_key)


class EnvOverride(Base):
    def test_env_moves_a_model_back_to_mixed(self):
        os.environ[pp.PRECISION_OVERRIDE_ENV] = 'xseg:mixed'
        chain, _d = self.chain('models/xseg.onnx', 'masking:xseg')
        self.assertTrue(_opts(chain)['trt_fp16_enable'])
        self.assertEqual(_opts(chain)['trt_engine_cache_path'], self.cache)

    def test_global_drops_the_table_entry(self):
        os.environ[pp.PRECISION_OVERRIDE_ENV] = 'xseg:global'
        self.assertIsNone(pp.precision_override_for('masking:xseg', 'models/xseg.onnx'))
        self.global_precision = 'fp16'
        chain, decision = self.chain('models/xseg.onnx', 'masking:xseg')
        self.assertEqual(decision.requested, 'fp16')

    def test_env_adds_a_model(self):
        os.environ[pp.PRECISION_OVERRIDE_ENV] = 'w600k_r50:fp32'
        chain, _d = self.chain('models/buffalo_l/w600k_r50.onnx', 'recognition:buffalo_l')
        self.assertFalse(_opts(chain)['trt_fp16_enable'])

    def test_a_typo_is_loud_and_inert(self):
        os.environ[pp.PRECISION_OVERRIDE_ENV] = 'xseg:fp64'
        out = StringIO()
        with redirect_stdout(out):
            self.assertIsNone(pp.precision_override_for('masking:xseg', 'models/xseg.onnx'))
        self.assertIn('IGNORED', out.getvalue())
        self.assertTrue(any('malformed' in str(e.get('refused')) for e in pp.precision_log))

    def test_parse(self):
        self.assertEqual(pp.parse_precision_overrides('XSeg:FP32 ; a:global'), {'xseg': 'fp32', 'a': 'global'})
        for bad in ('xseg', 'xseg:', ':fp32', 'xseg:fp64'):
            with self.assertRaises(ValueError):
                pp.parse_precision_overrides(bad)

    def test_swappers_are_refused(self):
        os.environ[pp.PRECISION_OVERRIDE_ENV] = 'hyperswap_1a:fp32'
        out = StringIO()
        with redirect_stdout(out):
            self.assertIsNone(pp.precision_override_for('swap:hyperswap', 'models/hyperswap_1a.onnx'))
        self.assertIn('refused', out.getvalue())


class BundleSeam(Base):
    def member(self, model, **kw):
        with redirect_stdout(StringIO()):
            return pp.bundle_member_providers('recognition:buffalo_l', _trt(self.cache), model, **kw)

    def test_1k3d68_is_forced_fp32_into_the_recognition_fp32_cache(self):
        chain = self.member('models/buffalo_l/1k3d68.onnx')
        self.assertFalse(_opts(chain)['trt_fp16_enable'])
        self.assertEqual(_opts(chain)['trt_engine_cache_path'], self.cache + '_recognition_fp32')

    def test_w600k_r50_gets_no_override(self):
        self.assertIsNone(self.member('models/buffalo_l/w600k_r50.onnx'))

    def test_only_the_precision_step_runs(self):
        with mock.patch.object(pp, '_finalize', side_effect=AssertionError('shape profile / build override must not run')):
            self.member('models/buffalo_l/2d106det.onnx')

    def test_a_chain_without_tensorrt_is_left_alone(self):
        with redirect_stdout(StringIO()):
            self.assertIsNone(pp.bundle_member_providers('recognition:buffalo_l', ['CUDAExecutionProvider'],
                                                         'models/buffalo_l/1k3d68.onnx'))

    def test_loosening_under_a_global_fp32_is_refused_not_guessed(self):
        self.global_precision = 'fp32'
        os.environ[pp.PRECISION_OVERRIDE_ENV] = '1k3d68:mixed'
        out = StringIO()
        with redirect_stdout(out):
            got = pp.bundle_member_providers('recognition:buffalo_l', _trt(self.cache, fp16=False),
                                             'models/buffalo_l/1k3d68.onnx')
        self.assertIsNone(got)
        self.assertIn('ignored inside a bundle', out.getvalue())

    def test_an_entry_equal_to_the_global_setting_is_a_no_op(self):
        self.global_precision = 'fp32'
        self.assertIsNone(self.member('models/buffalo_l/1k3d68.onnx'))


class DigestMemo(unittest.TestCase):
    def test_a_model_file_is_read_once_per_change(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'm.onnx')
            open(path, 'wb').write(b'abc')
            first = pp._model_digest(path)
            with mock.patch('builtins.open', side_effect=AssertionError('re-read')):
                self.assertEqual(pp._model_digest(path), first)
            open(path, 'wb').write(b'abcd')
            os.utime(path, ns=(1, 2 * 10 ** 18))
            self.assertNotEqual(pp._model_digest(path), first)


if __name__ == '__main__':
    unittest.main()
