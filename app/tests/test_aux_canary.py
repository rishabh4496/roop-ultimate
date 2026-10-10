"""Startup canary for the small TensorRT models (roop/aux_canary.py), cached by engine hash.

What makes the guard trustworthy, and what these pin:

  * it FAILS a wrong picture (an engine answering for another face, a collapsed or non-finite output) and PASSES an engine that
    matches, including real FP16 noise;
  * a verdict is cached under the CONTENT hash of the engine, so the second start does not build a reference session, and any
    change to the engine, the model, a build option or the cache directory is a miss - nothing is trusted across a rebuild;
  * it never turns "could not check" into a failure or into a cached verdict, and never runs where nothing can drift;
  * on a failed verdict the model is rebuilt on TensorRT FP32 and re-checked, then on CUDA/CPU.

No GPU: sessions are fakes with the ORT surface the canary touches.
"""
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from roop import aux_canary as ac                                  # noqa: E402

GRAPH = 'main_graph'


class _Meta:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class _ModelMeta:
    graph_name = GRAPH


SHAPES = {'xseg': ('xseg_input:0', ['unk__1', 256, 256, 3]), 'w600k_r50': ('input.1', ['None', 3, 112, 112]),
          '2d106det': ('data', ['None', 3, 192, 192]), '1k3d68': ('data', ['None', 3, 192, 192])}


def _fn(stem):
    """A deterministic stand-in for the model: a smooth function of the input with the real output shape."""
    def run(x):
        v = np.asarray(x, np.float32)
        if stem == 'xseg':
            return 1.0 / (1.0 + np.exp(-(v.mean(axis=3, keepdims=True) - 0.5) * 6.0))
        if stem == 'w600k_r50':
            return np.tile(v.reshape(1, -1)[:, :512] + 0.01 * v.mean(), (1, 1))
        n = 212 if stem == '2d106det' else 3309
        base = np.linspace(-0.8, 0.8, n, dtype=np.float32)[None]
        return base + 0.05 * float(v.mean()) / (255.0 if v.max() > 2 else 1.0)
    return run


class FakeSession:
    def __init__(self, stem, fn=None, cache_dir=None, trt=True, fp16=True):
        self.stem = stem
        self._fn = fn or _fn(stem)
        self._cache = cache_dir
        self._trt = trt
        self._fp16 = fp16
        self.calls = 0

    def get_inputs(self):
        return [_Meta(*SHAPES[self.stem])]

    def get_modelmeta(self):
        return _ModelMeta()

    def get_providers(self):
        return ['TensorrtExecutionProvider', 'CPUExecutionProvider'] if self._trt else ['CUDAExecutionProvider', 'CPUExecutionProvider']

    def get_provider_options(self):
        return {'TensorrtExecutionProvider': {'trt_engine_cache_path': self._cache, 'trt_fp16_enable': '1' if self._fp16 else '0'}}

    def run(self, _names, feed):
        self.calls += 1
        return [self._fn(next(iter(feed.values())))]


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = os.path.join(self._tmp.name, 'trt_cache')
        os.makedirs(self.cache)
        self.model = os.path.join(self._tmp.name, 'm.onnx')
        open(self.model, 'wb').write(b'onnx-bytes')
        self.engine = os.path.join(self.cache, 'TensorrtExecutionProvider_TRTKernel_graph_%s_123_0_0_fp16_sm89.engine' % GRAPH)
        open(self.engine, 'wb').write(b'engine-one')
        self._patches = [
            mock.patch.object(ac, '_store_path', lambda: os.path.join(self._tmp.name, 'aux_canary.json')),
            mock.patch.dict(os.environ, {}, clear=False),
        ]
        for p in self._patches:
            p.start()
        os.environ.pop(ac.CANARY_ENV, None)
        ac._MEMO.clear()

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        ac._MEMO.clear()
        self._tmp.cleanup()

    def session(self, stem='xseg', **kw):
        return FakeSession(stem, cache_dir=self.cache, **kw)

    def check(self, stem, session, ref=None, **kw):
        made = []

        def factory():
            made.append(1)
            return ref if ref is not None else FakeSession(stem, cache_dir=self.cache)
        self.refs_built = made
        with redirect_stdout(StringIO()):
            return ac.check_session(stem, session, self.model, reference_factory=factory, **kw)


class Metrics(unittest.TestCase):
    def test_identical_outputs_pass_every_model(self):
        for stem in ac.SPECS:
            out = _fn(stem)(np.random.RandomState(0).rand(*([1, 256, 256, 3] if stem == 'xseg' else [1, 3, 112, 112])))
            self.assertTrue(ac.judge(stem, ac.compare(stem, out, out)), stem)

    def test_non_finite_candidate_fails_on_every_metric(self):
        for stem in ac.SPECS:
            out = _fn(stem)(np.zeros((1, 256, 256, 3) if stem == 'xseg' else (1, 3, 112, 112)))
            bad = out.copy()
            bad.flat[0] = np.nan
            self.assertFalse(ac.judge(stem, ac.compare(stem, bad, out)), stem)

    def test_shape_mismatch_fails(self):
        a = np.zeros((1, 212), np.float32)
        self.assertFalse(ac.judge('2d106det', ac.compare('2d106det', a, np.zeros((1, 213), np.float32))))

    def test_embedding_of_another_face_fails(self):
        rng = np.random.RandomState(1)
        a, b = rng.randn(1, 512), rng.randn(1, 512)
        self.assertFalse(ac.judge('w600k_r50', ac.compare('w600k_r50', a, b)))
        self.assertTrue(ac.judge('w600k_r50', ac.compare('w600k_r50', a, a * 1.0001 + 1e-4)))

    def test_landmarks_are_decoded_like_Landmark_get(self):
        # a uniform shift of +0.01 in normalised units is 0.96 crop px on a 192 crop, for both models
        for stem, n, dim in (('2d106det', 212, 2), ('1k3d68', 3309, 3)):
            ref = np.zeros((1, n), np.float32)
            got = ref + 0.01
            m = ac.compare(stem, got, ref, 192)
            self.assertAlmostEqual(m['mean_px'], 0.01 * 96.0 * np.sqrt(2), places=4)
            self.assertTrue(ac.judge(stem, m))

    def test_a_large_landmark_error_fails(self):
        ref = np.zeros((1, 212), np.float32)
        self.assertFalse(ac.judge('2d106det', ac.compare('2d106det', ref + 0.2, ref)))

    def test_mask_collapse_fails(self):
        ref = np.full((1, 256, 256, 1), 0.9, np.float32)
        self.assertFalse(ac.judge('xseg', ac.compare('xseg', np.zeros_like(ref), ref)))

    def test_a_half_precision_sized_error_passes(self):
        ref = np.random.RandomState(2).rand(1, 256, 256, 1).astype(np.float32)
        self.assertTrue(ac.judge('xseg', ac.compare('xseg', ref + 2e-3, ref)))


class Feeds(unittest.TestCase):
    def test_each_kind_is_scaled_like_its_models_preprocessing(self):
        xs = ac.build_feeds('xseg', FakeSession('xseg'))
        self.assertEqual(xs[0]['xseg_input:0'].shape, (1, 256, 256, 3))
        self.assertTrue(0.0 <= xs[0]['xseg_input:0'].min() and xs[0]['xseg_input:0'].max() <= 1.0)
        em = ac.build_feeds('w600k_r50', FakeSession('w600k_r50'))
        self.assertEqual(em[0]['input.1'].shape, (1, 3, 112, 112))
        self.assertTrue(-1.0 <= em[0]['input.1'].min() and em[0]['input.1'].max() <= 1.0)
        lm = ac.build_feeds('1k3d68', FakeSession('1k3d68'))
        self.assertEqual(lm[0]['data'].shape, (1, 3, 192, 192))
        self.assertGreater(lm[0]['data'].max(), 100.0)                  # 0..255, mean 0 / std 1
        for feeds in (xs, em, lm):
            self.assertEqual(len(feeds), 2)
            self.assertEqual(feeds[0][next(iter(feeds[0]))].dtype, np.float32)

    def test_deterministic(self):
        a = ac.build_feeds('xseg', FakeSession('xseg'))
        b = ac.build_feeds('xseg', FakeSession('xseg'))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x['xseg_input:0'], y['xseg_input:0'])

    def test_unknown_model_is_not_guessed(self):
        self.assertIsNone(ac.build_feeds('genderage', FakeSession('xseg')))


class EngineHash(Base):
    def test_matches_by_graph_name_and_hashes_content(self):
        other = os.path.join(self.cache, 'TensorrtExecutionProvider_TRTKernel_graph_other_9_0_0_fp16_sm89.engine')
        open(other, 'wb').write(b'not-ours')
        h = ac.engine_hashes(self.cache, GRAPH)
        self.assertEqual(len(h), 1)
        open(self.engine, 'wb').write(b'engine-two')
        ac._MEMO.clear()
        self.assertNotEqual(ac.engine_hashes(self.cache, GRAPH), h)

    def test_same_size_and_content_change_is_still_seen(self):
        h = ac.engine_hashes(self.cache, GRAPH)
        st = os.stat(self.engine)
        open(self.engine, 'wb').write(b'engine-ONE')                    # same length
        os.utime(self.engine, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))   # a rewrite moves mtime by >> the 100 ns FS tick
        self.assertNotEqual(ac.engine_hashes(self.cache, GRAPH), h)

    def test_no_cache_dir_means_no_identity(self):
        self.assertEqual(ac.engine_hashes(None, GRAPH), [])
        self.assertEqual(ac.engine_hashes(os.path.join(self.cache, 'missing'), GRAPH), [])

    def test_a_file_is_hashed_once(self):
        store = ac._load_store()
        ac.engine_hashes(self.cache, GRAPH, store)
        with mock.patch('builtins.open', side_effect=AssertionError('re-read')):
            ac.engine_hashes(self.cache, GRAPH, store)


class Verdicts(Base):
    def test_matching_engine_passes(self):
        res = self.check('xseg', self.session('xseg'))
        self.assertTrue(res.passed)

    def test_every_model_passes_against_itself(self):
        for stem in ac.SPECS:
            res = self.check(stem, self.session(stem))
            self.assertTrue(res.passed, (stem, res.reason))

    def test_wrong_picture_fails(self):
        bad = FakeSession('xseg', fn=lambda x: 1.0 - _fn('xseg')(x), cache_dir=self.cache)
        res = self.check('xseg', bad)
        self.assertTrue(res.failed)

    def test_non_finite_output_fails(self):
        bad = FakeSession('w600k_r50', fn=lambda x: np.full((1, 512), np.nan, np.float32), cache_dir=self.cache)
        self.assertTrue(self.check('w600k_r50', bad).failed)

    def test_landmarks_from_another_face_fail(self):
        shifted = FakeSession('1k3d68', fn=lambda x: _fn('1k3d68')(x) + 0.3, cache_dir=self.cache)
        self.assertTrue(self.check('1k3d68', shifted).failed)

    def test_second_start_uses_the_cached_verdict_without_a_reference(self):
        first = self.check('xseg', self.session('xseg'))
        self.assertEqual(len(self.refs_built), 1)
        ac._MEMO.clear()                                                # a new process: only the file survives
        second = self.check('xseg', self.session('xseg'))
        self.assertEqual(self.refs_built, [])
        self.assertTrue(second.passed)
        self.assertIn('cached by engine hash', second.reason)
        self.assertTrue(os.path.isfile(os.path.join(self._tmp.name, 'aux_canary.json')))

    def test_a_rebuilt_engine_is_a_miss(self):
        self.check('xseg', self.session('xseg'))
        open(self.engine, 'wb').write(b'engine-rebuilt-differently')
        ac._MEMO.clear()
        self.check('xseg', self.session('xseg'))
        self.assertEqual(len(self.refs_built), 1)

    def test_a_changed_model_option_or_cache_dir_is_a_miss(self):
        self.check('xseg', self.session('xseg', fp16=True))
        self.check('xseg', self.session('xseg', fp16=False))
        self.assertEqual(len(self.refs_built), 1)                       # fp16 -> fp32 is a different engine identity
        other = os.path.join(self._tmp.name, 'trt_cache_fp32')
        os.makedirs(other)
        open(os.path.join(other, os.path.basename(self.engine)), 'wb').write(b'engine-one')
        self.check('xseg', FakeSession('xseg', cache_dir=other))
        self.assertEqual(len(self.refs_built), 1)

    def test_a_changed_model_is_a_miss(self):
        self.check('xseg', self.session('xseg'))
        open(self.model, 'wb').write(b'a different onnx')
        from roop import precision_policy
        precision_policy._DIGESTS.clear()
        self.check('xseg', self.session('xseg'))
        self.assertEqual(len(self.refs_built), 1)

    def test_a_failed_verdict_is_cached_too(self):
        bad = FakeSession('xseg', fn=lambda x: 1.0 - _fn('xseg')(x), cache_dir=self.cache)
        self.assertTrue(self.check('xseg', bad).failed)
        ac._MEMO.clear()
        again = self.check('xseg', bad)
        self.assertTrue(again.failed)
        self.assertEqual(self.refs_built, [])

    def test_the_two_models_sharing_a_graph_name_have_their_own_verdicts(self):
        self.check('2d106det', self.session('2d106det'))
        self.check('1k3d68', self.session('1k3d68'))
        self.assertEqual(len(self.refs_built), 1)                       # second call: its own key -> built its own reference
        stems = {v['stem'] for v in ac._load_store()['verdicts'].values()}
        self.assertEqual(stems, {'2d106det', '1k3d68'})


class Skips(Base):
    def test_not_on_tensorrt_is_skipped(self):
        res = self.check('xseg', self.session('xseg', trt=False))
        self.assertIsNone(res.passed)
        self.assertEqual(self.refs_built, [])

    def test_disabled_by_env(self):
        os.environ[ac.CANARY_ENV] = '0'
        res = self.check('xseg', self.session('xseg'))
        self.assertIsNone(res.passed)
        self.assertEqual(self.refs_built, [])

    def test_unknown_model(self):
        self.assertIsNone(self.check('genderage', self.session('xseg')).passed)

    def test_no_reference_is_a_skip_and_not_cached(self):
        with redirect_stdout(StringIO()):
            res = ac.check_session('xseg', self.session('xseg'), self.model, reference_factory=lambda: None)
        self.assertIsNone(res.passed)
        self.assertEqual(ac._load_store()['verdicts'], {})

    def test_a_reference_that_raises_is_a_skip_and_not_cached(self):
        def boom():
            raise RuntimeError('cuda init failed')
        with redirect_stdout(StringIO()):
            res = ac.check_session('xseg', self.session('xseg'), self.model, reference_factory=boom)
        self.assertIsNone(res.passed)
        self.assertEqual(ac._load_store()['verdicts'], {})

    def test_a_session_that_raises_mid_check_is_a_skip(self):
        class Boom(FakeSession):
            def run(self, *_a, **_k):
                raise RuntimeError('execution failed')
        res = self.check('xseg', Boom('xseg', cache_dir=self.cache))
        self.assertIsNone(res.passed)

    def test_no_engine_cache_still_checks_but_never_caches(self):
        res = self.check('xseg', FakeSession('xseg', cache_dir=None))
        self.assertTrue(res.passed)
        self.assertEqual(ac._load_store()['verdicts'], {})

    def test_an_unwritable_store_costs_a_recheck_not_a_failure(self):
        with mock.patch.object(ac, '_store_path', lambda: os.path.join(self.model, 'nope', 'aux.json')):
            res = self.check('xseg', self.session('xseg'))
        self.assertTrue(res.passed)


class Guard(Base):
    def run_guard(self, session, build, providers=None):
        providers = providers or [('TensorrtExecutionProvider', {'trt_fp16_enable': True, 'trt_engine_cache_path': self.cache}),
                                  'CUDAExecutionProvider', 'CPUExecutionProvider']
        with redirect_stdout(StringIO()) as out:
            got = ac.guard('xseg', session, self.model, providers, build)
        self.out = out.getvalue()
        return got

    def test_good_engine_is_left_alone(self):
        s = self.session('xseg')
        built = []
        with mock.patch.object(ac, '_reference_session', lambda f: FakeSession('xseg', cache_dir=self.cache)):
            got, providers, res = self.run_guard(s, lambda chain: built.append(chain))
        self.assertIs(got, s)
        self.assertEqual(built, [])
        self.assertTrue(res.passed)
        self.assertIn('[AuxCanary] xseg: OK', self.out)

    def test_corrupt_fp16_engine_is_rebuilt_on_trt_fp32(self):
        bad = FakeSession('xseg', fn=lambda x: 1.0 - _fn('xseg')(x), cache_dir=self.cache)
        good = FakeSession('xseg', cache_dir=os.path.join(self._tmp.name, 'fp32'), fp16=False)
        os.makedirs(good._cache)
        open(os.path.join(good._cache, os.path.basename(self.engine)), 'wb').write(b'fp32-engine')
        chains = []

        def build(chain):
            chains.append(chain)
            return good
        with mock.patch.object(ac, '_reference_session', lambda f: FakeSession('xseg', cache_dir=self.cache)):
            got, providers, res = self.run_guard(bad, build)
        self.assertIs(got, good)
        self.assertEqual(len(chains), 1)
        opts = next(p[1] for p in chains[0] if isinstance(p, tuple) and 'tensorrt' in p[0].lower())
        self.assertFalse(opts['trt_fp16_enable'])
        self.assertIn('TensorRT FP32 engine instead', self.out)

    def test_fp32_engine_failing_too_falls_back_to_cuda(self):
        bad = FakeSession('xseg', fn=lambda x: 1.0 - _fn('xseg')(x), cache_dir=self.cache)
        also_bad = FakeSession('xseg', fn=lambda x: 0.0 * x[..., :1], cache_dir=os.path.join(self._tmp.name, 'fp32b'), fp16=False)
        os.makedirs(also_bad._cache)
        open(os.path.join(also_bad._cache, os.path.basename(self.engine)), 'wb').write(b'fp32-engine-b')
        cuda = FakeSession('xseg', trt=False)
        seen = []

        def build(chain):
            seen.append(chain)
            return also_bad if len(seen) == 1 else cuda
        with mock.patch.object(ac, '_reference_session', lambda f: FakeSession('xseg', cache_dir=self.cache)):
            got, providers, res = self.run_guard(bad, build)
        self.assertIs(got, cuda)
        self.assertEqual(len(seen), 2)
        self.assertFalse(any('tensorrt' in (p[0] if isinstance(p, tuple) else p).lower() for p in seen[1]))
        self.assertIn('falling back to CUDAExecutionProvider', self.out)

    def test_an_fp32_engine_that_fails_goes_straight_to_cuda(self):
        bad = FakeSession('xseg', fn=lambda x: 1.0 - _fn('xseg')(x), cache_dir=self.cache, fp16=False)
        providers = [('TensorrtExecutionProvider', {'trt_fp16_enable': False, 'trt_engine_cache_path': self.cache}),
                     'CUDAExecutionProvider', 'CPUExecutionProvider']
        seen = []
        cuda = FakeSession('xseg', trt=False)

        def build(chain):
            seen.append(chain)
            return cuda
        with mock.patch.object(ac, '_reference_session', lambda f: FakeSession('xseg', cache_dir=self.cache)):
            got, used, res = self.run_guard(bad, build, providers)
        self.assertIs(got, cuda)
        self.assertEqual(len(seen), 1)

    def test_a_skipped_check_never_rebuilds(self):
        s = self.session('xseg')
        built = []
        with mock.patch.object(ac, '_reference_session', lambda f: None):
            got, providers, res = self.run_guard(s, lambda chain: built.append(chain))
        self.assertIs(got, s)
        self.assertEqual(built, [])
        self.assertIn('skipped', self.out)


if __name__ == '__main__':
    unittest.main()
