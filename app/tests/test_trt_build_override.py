"""Per-model TensorRT build overrides (roop/precision_policy.apply_build_override).

core.py sets one heuristics flag and one builder level for EVERY engine.  The override layer lets one model differ, and the
two things that must hold are: (1) with no override the providers and the engine-cache namespace are untouched, so an
install that never opts in keeps every engine it has built; (2) every distinct effective configuration - heuristics,
level, build tag - gets its OWN engine and timing cache, so a build made one way can never be loaded for another.
Swappers are refused.  No GPU: provider chains are plain tuples.
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


def _trt(cache, heur=True, level=3):
    return [('TensorrtExecutionProvider', {
        'device_id': 0, 'trt_fp16_enable': True, 'trt_build_heuristics_enable': heur,
        'trt_builder_optimization_level': level, 'trt_engine_cache_enable': True,
        'trt_engine_cache_path': cache, 'trt_timing_cache_path': cache}),
        'CUDAExecutionProvider', 'CPUExecutionProvider']


def _opts(providers):
    return next(p[1] for p in providers if isinstance(p, tuple) and 'tensorrt' in p[0].lower())


class Base(unittest.TestCase):
    def setUp(self):
        pp._override_announced.clear()
        del pp.override_log[:]
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = os.path.join(self._tmp.name, 'mixed_cache')
        os.makedirs(self.cache)
        self._saved = dict(pp.BUILD_OVERRIDES)
        pp.BUILD_OVERRIDES.clear()
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop(pp.BUILD_OVERRIDE_ENV, None)

    def tearDown(self):
        self._env.stop()
        pp.BUILD_OVERRIDES.clear()
        pp.BUILD_OVERRIDES.update(self._saved)
        self._tmp.cleanup()

    def apply(self, spec, model='models/retinaface_r50.onnx', key='face_detection:r50', providers=None):
        os.environ[pp.BUILD_OVERRIDE_ENV] = spec
        return pp.apply_build_override(providers if providers is not None else _trt(self.cache), key, model)


class NoOverrideMeansNoChange(Base):
    def test_providers_and_namespace_are_untouched(self):
        original = _trt(self.cache)
        before = copy.deepcopy(original)
        out = pp.apply_build_override(original, 'face_detection:r50', 'models/retinaface_r50.onnx')
        self.assertEqual(out, before)
        self.assertEqual(os.listdir(self.tmp_root()), ['mixed_cache'])          # no cache directory was created

    def tmp_root(self):
        return self._tmp.name

    def test_an_entry_equal_to_the_global_options_is_also_a_no_op(self):
        out = self.apply('retinaface_r50:h=1,l=3')
        self.assertEqual(_opts(out)['trt_engine_cache_path'], self.cache)
        self.assertEqual(pp.override_log, [])

    def test_other_models_are_not_touched(self):
        out = self.apply('retinaface_r50:h=0,l=5', model='models/xseg.onnx', key='masking:xseg')
        self.assertEqual(_opts(out)['trt_engine_cache_path'], self.cache)


class OverrideApplies(Base):
    def test_options_and_both_cache_paths_change(self):
        original = _trt(self.cache)
        before = copy.deepcopy(original)
        out = self.apply('retinaface_r50:h=0,l=5', providers=original)
        o = _opts(out)
        self.assertFalse(o['trt_build_heuristics_enable'])
        self.assertEqual(o['trt_builder_optimization_level'], 5)
        self.assertTrue(o['trt_engine_cache_path'].startswith(self.cache + '_ovh0l5_'))
        self.assertEqual(o['trt_timing_cache_path'], o['trt_engine_cache_path'])    # tactics must not leak across configs
        self.assertTrue(os.path.isdir(o['trt_engine_cache_path']))
        self.assertEqual(original, before, 'the caller\'s providers were mutated')
        self.assertEqual(out[1:], ['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def test_every_distinct_config_gets_its_own_cache(self):
        paths = set()
        for h in (0, 1):
            for l in (3, 5):
                for tag in ('r1', 'r2'):
                    out = self.apply('retinaface_r50:h=%d,l=%d,tag=%s' % (h, l, tag))
                    paths.add(_opts(out)['trt_engine_cache_path'])
        self.assertEqual(len(paths), 8, 'two builds of one config, or two configs, would share an engine cache')

    def test_a_tag_alone_forces_a_fresh_cache_even_when_options_match(self):
        out = self.apply('retinaface_r50:h=1,l=3,tag=r1')
        self.assertNotEqual(_opts(out)['trt_engine_cache_path'], self.cache)

    def test_the_hash_covers_the_tag(self):
        a = _opts(self.apply('retinaface_r50:h=0,l=5,tag=a'))['trt_engine_cache_path']
        b = _opts(self.apply('retinaface_r50:h=0,l=5,tag=b'))['trt_engine_cache_path']
        self.assertEqual(a.split('_ovh0l5_')[1].split('_')[0], 'a')
        self.assertNotEqual(a.rsplit('_', 1)[1], b.rsplit('_', 1)[1])

    def test_table_entry_applies_and_env_wins_over_it(self):
        pp.BUILD_OVERRIDES['xseg'] = {'heuristics': False, 'level': 3}
        out = pp.apply_build_override(_trt(self.cache), 'masking:xseg', 'models/xseg.onnx')
        self.assertFalse(_opts(out)['trt_build_heuristics_enable'])
        os.environ[pp.BUILD_OVERRIDE_ENV] = 'xseg:h=1,l=5'
        out = pp.apply_build_override(_trt(self.cache), 'masking:xseg', 'models/xseg.onnx')
        self.assertTrue(_opts(out)['trt_build_heuristics_enable'])
        self.assertEqual(_opts(out)['trt_builder_optimization_level'], 5)

    def test_non_tensorrt_chains_are_left_alone(self):
        chain = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.assertEqual(self.apply('retinaface_r50:h=0,l=5', providers=chain), chain)

    def test_an_applied_override_announces_itself(self):
        buf = StringIO()
        with redirect_stdout(buf):
            self.apply('retinaface_r50:h=0,l=5')
        self.assertIn('[TRT] build override for retinaface_r50: heuristics=False level=5', buf.getvalue())
        self.assertEqual(pp.override_log[0]['stem'], 'retinaface_r50')

    def test_an_unwritable_cache_skips_the_override_instead_of_failing(self):
        blocked = os.path.join(self.cache, 'file')
        open(blocked, 'w').close()
        out = self.apply('retinaface_r50:h=0,l=5', providers=_trt(os.path.join(blocked, 'sub')))
        self.assertTrue(_opts(out)['trt_build_heuristics_enable'])


class SwappersAreProtected(Base):
    def test_a_swapper_is_refused_even_when_named_explicitly(self):
        buf = StringIO()
        with redirect_stdout(buf):
            out = self.apply('hyperswap_1a_256:h=0,l=5', model='models/hyperswap_1a_256.onnx',
                             key='face_swap:realswap')
        self.assertTrue(_opts(out)['trt_build_heuristics_enable'])
        self.assertEqual(_opts(out)['trt_engine_cache_path'], self.cache)
        self.assertIn('swappers keep the global build options', buf.getvalue())
        self.assertEqual(pp.override_log[-1]['refused'], 'swapper')

    def test_the_table_cannot_override_a_swapper_either(self):
        pp.BUILD_OVERRIDES['hififace_unofficial_256'] = {'heuristics': False}
        out = pp.apply_build_override(_trt(self.cache), 'face_swap:hififace', 'models/hififace_unofficial_256.onnx')
        self.assertTrue(_opts(out)['trt_build_heuristics_enable'])


class MalformedSpecsAreLoud(Base):
    def test_a_bad_env_value_is_reported_and_ignored(self):
        for bad in ('retinaface_r50:h=2', 'retinaface_r50:l=9', 'retinaface_r50:x=1', 'retinaface_r50', ':h=1',
                    'retinaface_r50:tag=a b', 'retinaface_r50:h'):
            pp._override_announced.clear()
            buf = StringIO()
            with redirect_stdout(buf):
                out = self.apply(bad)
            self.assertEqual(_opts(out)['trt_engine_cache_path'], self.cache, bad)
            self.assertIn('IGNORED', buf.getvalue(), bad)

    def test_parse_roundtrip(self):
        self.assertEqual(
            pp.parse_build_overrides('retinaface_r50:h=0,l=5,tag=r1;xseg:h=1'),
            {'retinaface_r50': {'heuristics': False, 'level': 5, 'tag': 'r1'}, 'xseg': {'heuristics': True}})
        self.assertEqual(pp.parse_build_overrides(''), {})


class ThroughFinalize(Base):
    def test_finalize_applies_the_override_after_the_shape_profile(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'models', 'retinaface_r50.onnx')
        if not os.path.exists(path):
            self.skipTest('retinaface_r50.onnx not installed')
        os.environ[pp.BUILD_OVERRIDE_ENV] = 'retinaface_r50:h=0,l=5'
        out = pp._finalize('face_detection:r50', path, _trt(self.cache))
        cache = _opts(out)['trt_engine_cache_path']
        self.assertIn('_sp', cache)                         # the profile namespace is still there ...
        self.assertTrue(cache.rsplit('_ovh0l5_', 1)[0].endswith(cache.split('_ovh0l5_')[0][-10:]))
        self.assertIn('_ovh0l5_', cache)                    # ... and the override namespace comes last
        self.assertLess(cache.index('_sp'), cache.index('_ovh0l5_'))

    def test_finalize_without_an_override_is_what_it_was(self):
        out = pp._finalize('masking:xseg', 'models/xseg.onnx', _trt(self.cache))
        self.assertEqual(_opts(out)['trt_engine_cache_path'], self.cache)


if __name__ == '__main__':
    unittest.main()
