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
    def __init__(self, name, shape, type_='tensor(float)'):
        self.name, self.shape, self.type = name, shape, type_


class _ModelMeta:
    graph_name = GRAPH


SHAPES = {'xseg': ('xseg_input:0', ['unk__1', 256, 256, 3]), 'w600k_r50': ('input.1', ['None', 3, 112, 112]),
          '2d106det': ('data', ['None', 3, 192, 192]), '1k3d68': ('data', ['None', 3, 192, 192]),
          'restoreformer_plus_plus': ('input', [1, 3, 512, 512]), 'gpen-bfr-512': ('input', [1, 3, 512, 512]),
          'gpen_bfr_256': ('input', [1, 3, 256, 256]), 'codeformer.fp16': ('x', ['batch', 3, 512, 512])}
RESTORERS = ('restoreformer_plus_plus', 'gpen-bfr-512', 'gpen_bfr_256', 'codeformer.fp16')


def _fn(stem):
    """A deterministic stand-in for the model: a smooth function of the input with the real output shape."""
    def run(x):
        v = np.asarray(x, np.float32)
        if stem in RESTORERS:       # a smooth, structure-preserving "restorer": mild contrast + a blur-free tone curve
            return np.tanh(1.2 * v) / np.tanh(1.2)
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
        if self.stem == 'codeformer.fp16':      # the real graph: fp16 pixels `x` and a float64 fidelity scalar `w`
            return [_Meta(*SHAPES[self.stem], 'tensor(float16)'), _Meta('w', [], 'tensor(double)')]
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
        with open(self.model, 'wb') as f:
            f.write(b'onnx-bytes')
        self.engine = os.path.join(self.cache, 'TensorrtExecutionProvider_TRTKernel_graph_%s_123_0_0_fp16_sm89.engine' % GRAPH)
        with open(self.engine, 'wb') as f:
            f.write(b'engine-one')
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

    def _forget_verdicts(self):
        ac._MEMO.clear()
        store = ac._store_path()
        if os.path.exists(store):
            os.remove(store)

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
            out = _fn(stem)(np.random.RandomState(0).rand(*([1, 256, 256, 3] if stem == 'xseg' else [1, 3, 112, 112])) * 2 - 1)
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
            ref = np.linspace(-0.8, 0.8, n, dtype=np.float32)[None]     # a real output is never the zero vector (cosine is undefined there)
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


class RestorerCanary(Base):
    """Kind "image": the swap canary's SSIM floor, for the face restorers."""

    def test_feeds_are_in_the_restorers_range(self):
        feeds = ac.build_feeds('restoreformer_plus_plus', FakeSession('restoreformer_plus_plus'))
        x = feeds[0]['input']
        self.assertEqual(x.shape, (1, 3, 512, 512))
        self.assertLess(x.min(), -0.3)                      # really spans [-1, 1], not [0, 1]
        self.assertLessEqual(x.max(), 1.0)
        self.assertGreaterEqual(x.min(), -1.0)

    def test_matching_restorer_passes_with_half_precision_noise(self):
        noisy = FakeSession('restoreformer_plus_plus', fn=lambda x: _fn('restoreformer_plus_plus')(x) + 4e-4,
                            cache_dir=self.cache)
        self.assertTrue(self.check('restoreformer_plus_plus', noisy).passed)

    def test_a_grey_or_smeared_output_fails(self):
        grey = FakeSession('restoreformer_plus_plus', fn=lambda x: np.zeros_like(x), cache_dir=self.cache)
        self.assertTrue(self.check('restoreformer_plus_plus', grey).failed)
        import cv2
        smear = FakeSession('restoreformer_plus_plus', cache_dir=self.cache,
                            fn=lambda x: cv2.GaussianBlur(_fn('restoreformer_plus_plus')(x)[0].transpose(1, 2, 0), (0, 0), 6)
                            .transpose(2, 0, 1)[None].astype(np.float32))
        self.assertTrue(self.check('restoreformer_plus_plus', smear).failed)

    def test_non_finite_restorer_output_fails(self):
        bad = FakeSession('restoreformer_plus_plus', fn=lambda x: np.full_like(x, np.nan), cache_dir=self.cache)
        self.assertTrue(self.check('restoreformer_plus_plus', bad).failed)

    def test_rf_pp_uses_the_cpu_reference(self):
        asked = []
        with mock.patch.object(ac, '_reference_session', lambda f, cpu=False, extended=False: asked.append(cpu) or FakeSession('restoreformer_plus_plus', cache_dir=self.cache)):
            with redirect_stdout(StringIO()):
                ac.check_session('restoreformer_plus_plus', FakeSession('restoreformer_plus_plus', cache_dir=self.cache), self.model)
        self.assertEqual(asked, [True])

class CodeFormerCanary(Base):
    """UltraMax / the CodeFormer fp16 tier: a two-input graph (fp16 pixels + a float64 `w`), reference on the CPU at EXTENDED."""
    STEM = 'codeformer.fp16'

    def test_feeds_match_what_run_binds(self):
        feeds = ac.build_feeds(self.STEM, FakeSession(self.STEM))
        self.assertEqual(len(feeds), 2)
        for feed in feeds:
            self.assertEqual(set(feed), {'x', 'w'})
            self.assertEqual(feed['x'].shape, (1, 3, 512, 512))
            self.assertEqual(feed['x'].dtype, np.float16)
            self.assertLess(float(feed['x'].min()), -0.3)           # spans [-1, 1], not [0, 1]
            self.assertEqual(feed['w'].dtype, np.float64)
            self.assertEqual(feed['w'].shape, (1,))
            self.assertEqual(float(feed['w'][0]), 0.5)

    def test_a_session_with_the_wrong_number_of_inputs_is_not_fed(self):
        self.assertIsNone(ac.build_feeds(self.STEM, FakeSession('restoreformer_plus_plus')))
        self.assertIsNone(ac.build_feeds('restoreformer_plus_plus', FakeSession(self.STEM)))

    def test_single_input_models_keep_float32_feeds(self):
        for stem in ('xseg', 'w600k_r50', '2d106det', '1k3d68', 'restoreformer_plus_plus'):
            for feed in ac.build_feeds(stem, FakeSession(stem)):
                self.assertEqual(len(feed), 1, stem)
                self.assertEqual(next(iter(feed.values())).dtype, np.float32, stem)

    def test_matching_engine_passes_with_half_precision_noise(self):
        noisy = FakeSession(self.STEM, fn=lambda x: _fn(self.STEM)(x) + 4e-4, cache_dir=self.cache)
        self.assertTrue(self.check(self.STEM, noisy).passed)

    def test_grey_smeared_and_non_finite_outputs_fail(self):
        import cv2
        grey = FakeSession(self.STEM, fn=lambda x: np.zeros_like(x), cache_dir=self.cache)
        self.assertTrue(self.check(self.STEM, grey).failed)
        smear = FakeSession(self.STEM, cache_dir=self.cache,
                            fn=lambda x: cv2.GaussianBlur(_fn(self.STEM)(x)[0].transpose(1, 2, 0).astype(np.float32), (0, 0), 6)
                            .transpose(2, 0, 1)[None])
        self.assertTrue(self.check(self.STEM, smear).failed)
        for bad_value in (np.nan, np.inf):
            bad = FakeSession(self.STEM, fn=lambda x, v=bad_value: np.full(x.shape, v, np.float16), cache_dir=self.cache)
            self.assertTrue(self.check(self.STEM, bad).failed, bad_value)

    def test_reference_is_the_cpu_session_at_extended(self):
        asked = []
        with mock.patch.object(ac, '_reference_session',
                               lambda f, cpu=False, extended=False: asked.append((cpu, extended)) or FakeSession(self.STEM, cache_dir=self.cache)):
            with redirect_stdout(StringIO()):
                ac.check_session(self.STEM, FakeSession(self.STEM, cache_dir=self.cache), self.model)
        self.assertEqual(asked, [(True, True)])

    def test_identity_carries_the_new_fields_only_where_they_apply(self):
        store = ac._load_store()
        _key, cf = ac.verdict_key(self.STEM, FakeSession(self.STEM, cache_dir=self.cache), self.model, store)
        self.assertEqual(cf['aux'], [[0.5], True])
        _key, rf = ac.verdict_key('xseg', FakeSession('xseg', cache_dir=self.cache), self.model, store)
        self.assertNotIn('aux', rf)         # an existing model's cached verdicts must not be invalidated by this change

    def test_a_failed_verdict_rebuilds_on_trt_fp32_then_falls_back(self):
        bad = FakeSession(self.STEM, fn=lambda x: np.zeros_like(x), cache_dir=self.cache)
        chain = [('TensorrtExecutionProvider', {'trt_fp16_enable': True, 'trt_engine_cache_path': self.cache}), 'CUDAExecutionProvider']
        built = []

        def build(c):
            built.append(c)
            return FakeSession(self.STEM, cache_dir=self.cache, fp16=False) if len(built) == 1 else FakeSession(self.STEM, trt=False)
        with mock.patch.object(ac, '_make_reference', lambda s, f: FakeSession(self.STEM, cache_dir=self.cache)):
            with redirect_stdout(StringIO()):
                session, used, verdict = ac.guard(self.STEM, bad, self.model, chain, build)
        self.assertTrue(session.get_providers()[0] == 'TensorrtExecutionProvider')   # the TensorRT FP32 rebuild passed
        self.assertIs(built[0][0][1]['trt_fp16_enable'], False)

    def test_gpen_canary_is_a_collapse_detector_not_a_drift_detector(self):
        """GPEN's shipped TensorRT mixed engines are cosine 0.992-0.998 / SSIM ~0.85-0.89 from FP32 by design (the 4070's tuned look).
        An SSIM floor, or the brief's cosine >= 0.995, would 'fail' healthy engines and rebuild them FP32; the floor is a cosine
        collapse floor only (docs/perf/stage2_canary_cosine_2026-10-18.md)."""
        for stem in ('gpen-bfr-512', 'gpen_bfr_256'):
            floors = dict(ac.SPECS[stem].floors)
            self.assertEqual(set(floors), {'cosine'}, 'an SSIM floor would reject the healthy FP16 look')
            self.assertLess(floors['cosine'], 0.992, 'above the worst healthy engine measured: it would fail good engines')
            self.assertGreater(floors['cosine'], 0.92, 'below the 24 px shifted-output control: it would pass a wrong picture')
            ref = _fn(stem)

            def healthy(x, ref=ref):    # a structured FP16-sized perturbation: well under SSIM 0.9, cosine ~0.995
                v = ref(x)
                return v + 0.07 * np.sin(np.arange(v.size, dtype=np.float32).reshape(v.shape) * 0.9)
            probe = np.random.RandomState(3).rand(1, 3, 64, 64).astype(np.float32) * 2 - 1
            cos = ac.compare(stem, healthy(probe), ref(probe))['cosine']
            self.assertTrue(0.97 < cos < 0.999, cos)
            self._forget_verdicts()
            self.assertTrue(self.check(stem, self.session(stem, fn=healthy)).passed)
            for label, fn in (('grey', lambda x: np.zeros_like(ref(x))), ('negated', lambda x: -ref(x)),
                              ('nan', lambda x: np.full_like(ref(x), np.nan))):
                with self.subTest(stem=stem, wrong=label):
                    self._forget_verdicts()     # the fake engine file is the same bytes: a real wrong engine has other bytes
                    self.assertFalse(self.check(stem, self.session(stem, fn=fn)).passed)

    def test_cosine_floors_match_the_stage_2_policy(self):
        """xseg / landmarks >= 0.998, RF++ / UltraMax >= 0.995 (the brief); GPEN's is the measured 0.97."""
        want = {'xseg': 0.998, '2d106det': 0.998, '1k3d68': 0.998, 'restoreformer_plus_plus': 0.995,
                'codeformer.fp16': 0.995, 'gpen-bfr-512': 0.97, 'gpen_bfr_256': 0.97}
        for stem, floor in want.items():
            self.assertEqual(dict(ac.SPECS[stem].floors).get('cosine'), floor, stem)

    def test_every_gpen_realistic_model_file_has_a_canary_spec(self):
        """The canary is selected by file stem; a size added to GPEN Realistic without a spec would run unguarded."""
        import re
        src = open(os.path.join(os.path.dirname(__file__), '..', 'roop', 'processors', 'Enhance_GPENRealistic.py'),
                   encoding='utf-8').read()
        files = re.findall(r"\(\s*'([\w\-]+\.onnx)'", src)
        self.assertTrue(files, 'could not read _SIZES from Enhance_GPENRealistic.py')
        for f in files:
            self.assertIn(ac.stem_of(f), ac.SPECS, f)
        self.assertIn('aux_canary.guard(', src)

    def test_the_rfpp_native_engine_is_canary_checked_and_falls_back(self):
        src = open(os.path.join(os.path.dirname(__file__), '..', 'roop', 'processors', 'Enhance_RestoreFormerPPlus.py'),
                   encoding='utf-8').read()
        start = src.index('The native engine used to skip the canary entirely')
        native = src[start:src.index('if not native_engine:', start)]
        self.assertIn("check_session('restoreformer_plus_plus'", native)
        self.assertIn("_record_downgrade('restoreformer_plus_plus', 'tensorrt-native', 'tensorrt-ort'", native)
        self.assertIn('_build = _build_ort', native)

    def test_restorer_canary_uses_structured_inputs_not_the_flat_defaults(self):
        cases = ac.SPECS['restoreformer_plus_plus'].cases
        self.assertTrue(cases and all(k.startswith('skin+n') for k, _ in cases))
        a = ac.build_feeds('restoreformer_plus_plus', FakeSession('restoreformer_plus_plus'))
        self.assertEqual(len(a), 2)
        self.assertFalse(np.array_equal(a[0]['input'], a[1]['input']))
        # the noise really is in the image: its local variance is well above the plain skin image's
        plain = ac._synth('skin', 1, 64, 64)
        noisy = ac._synth('skin+n0.08', 1, 64, 64)
        self.assertGreater(float(np.abs(np.diff(noisy, axis=2)).mean()), 3 * float(np.abs(np.diff(plain, axis=2)).mean()))

    def test_the_cases_are_part_of_the_verdict_key(self):
        sess = FakeSession('restoreformer_plus_plus', cache_dir=self.cache)
        store = ac._load_store()
        k1 = ac.verdict_key('restoreformer_plus_plus', sess, self.model, store)[0]
        old = ac.SPECS['restoreformer_plus_plus']
        try:
            ac.SPECS['restoreformer_plus_plus'] = ac.Spec(old.kind, old.floors, in_lo=old.in_lo, in_hi=old.in_hi, ref_cpu=True,
                                                          cases=(('skin+n0.08', 5),))
            k2 = ac.verdict_key('restoreformer_plus_plus', sess, self.model, store)[0]
        finally:
            ac.SPECS['restoreformer_plus_plus'] = old
        self.assertNotEqual(k1, k2)

    def test_only_the_first_output_is_fetched(self):
        seen = []

        class Wide(FakeSession):
            def get_outputs(self):
                return [_Meta('2701', [1, 3, 512, 512]), _Meta('input.1', [1, 64, 512, 512])]

            def run(self, names, feed):
                seen.append(names)
                return super().run(names, feed)
        self.check('restoreformer_plus_plus', Wide('restoreformer_plus_plus', cache_dir=self.cache),
                   ref=Wide('restoreformer_plus_plus', cache_dir=self.cache))
        self.assertTrue(seen and all(n == ['2701'] for n in seen))

    def test_a_failing_restorer_is_rebuilt_on_fp32(self):
        bad = FakeSession('restoreformer_plus_plus', fn=lambda x: np.zeros_like(x), cache_dir=self.cache)
        good = FakeSession('restoreformer_plus_plus', cache_dir=os.path.join(self._tmp.name, 'fp32'), fp16=False)
        os.makedirs(good._cache)
        open(os.path.join(good._cache, os.path.basename(self.engine)), 'wb').write(b'fp32')
        chains = []
        providers = [('TensorrtExecutionProvider', {'trt_fp16_enable': True, 'trt_engine_cache_path': self.cache}), 'CPUExecutionProvider']
        with mock.patch.object(ac, '_reference_session', lambda f, cpu=False: FakeSession('restoreformer_plus_plus', cache_dir=self.cache)):
            with redirect_stdout(StringIO()):
                got, used, res = ac.guard('restoreformer_plus_plus', bad, self.model, providers, lambda c: chains.append(c) or good)
        self.assertIs(got, good)
        self.assertFalse(next(p[1] for p in chains[0] if isinstance(p, tuple))['trt_fp16_enable'])


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


class FakeNative:
    """The surface roop.trt_native_runner.NativeSession gives the canary: a file path, no ORT options, ORT-shaped run()."""

    def __init__(self, path, fn):
        self.engine_path, self._fn = path, fn

    def get_inputs(self):
        return [_Meta('input', [1, 3, 512, 512])]

    def get_providers(self):
        return ['TensorrtNativeEngine']

    def get_provider_options(self):
        return {}

    def run(self, _names, feed):
        return [self._fn(next(iter(feed.values())))]


class NativeEngineCanary(Base):
    """The opt-in native RF++ engine used to be the one restorer path with no canary at all."""
    STEM = 'restoreformer_plus_plus'

    def setUp(self):
        super().setUp()
        self.native = os.path.join(self._tmp.name, 'rfpp.engine')
        open(self.native, 'wb').write(b'native-engine-one')

    def run_check(self, fn):
        return self.check(self.STEM, FakeNative(self.native, fn), ref=FakeSession(self.STEM, cache_dir=self.cache))

    def test_a_good_native_engine_passes_and_is_cached_by_file_content(self):
        self.assertTrue(self.run_check(_fn(self.STEM)).passed)
        self.assertEqual(len(ac._load_store()['verdicts']), 1)
        again = self.run_check(_fn(self.STEM))
        self.assertTrue(again.passed)
        self.assertEqual(self.refs_built, [], 'the second start must not build a reference session')

    def test_a_wrong_native_engine_fails(self):
        for label, fn in (('grey', lambda x: np.zeros_like(x)), ('negated', lambda x: -_fn(self.STEM)(x))):
            with self.subTest(wrong=label):
                self.assertFalse(self.run_check(fn).passed)

    def test_rebuilding_the_native_engine_file_is_a_cache_miss(self):
        self.assertTrue(self.run_check(_fn(self.STEM)).passed)
        open(self.native, 'wb').write(b'native-engine-two-rebuilt')
        self.run_check(_fn(self.STEM))
        self.assertEqual(len(ac._load_store()['verdicts']), 2)

    def test_native_session_exposes_what_the_canary_needs(self):
        import inspect
        from roop.trt_native_runner import NativeSession
        for attr in ('run', 'get_provider_options', 'get_providers', 'get_inputs'):
            self.assertTrue(callable(getattr(NativeSession, attr, None)), attr)
        self.assertIn('engine_path', inspect.getsource(NativeSession.__init__))


class VerdictIdentity(Base):
    def test_the_tensorrt_version_and_gpu_architecture_are_part_of_the_identity(self):
        s = self.session('xseg')
        store = ac._load_store()
        with mock.patch.object(ac, 'runtime_stamp', lambda: {'trt': '10.9.0.34', 'sm': 'sm89'}):
            key_a, ident = ac.verdict_key('xseg', s, self.model, store)
        with mock.patch.object(ac, 'runtime_stamp', lambda: {'trt': '10.9.0.34', 'sm': 'sm86'}):
            key_b, _ = ac.verdict_key('xseg', s, self.model, store)
        with mock.patch.object(ac, 'runtime_stamp', lambda: {'trt': '10.12.0.1', 'sm': 'sm89'}):
            key_c, _ = ac.verdict_key('xseg', s, self.model, store)
        self.assertEqual(ident['runtime'], {'trt': '10.9.0.34', 'sm': 'sm89'})
        self.assertEqual(len({key_a, key_b, key_c}), 3, 'a different GPU or TensorRT build must never inherit a verdict')

    def test_the_identity_names_the_model_precision_and_profile_shapes(self):
        _key, ident = ac.verdict_key('xseg', self.session('xseg'), self.model, ac._load_store())
        self.assertIn('model', ident)                                   # ONNX digest
        for option in ('trt_fp16_enable', 'trt_profile_min_shapes', 'trt_profile_opt_shapes', 'trt_profile_max_shapes'):
            self.assertIn(option, ident['options'])
        self.assertEqual(set(ident['runtime']), {'trt', 'sm'})

    def test_the_runtime_stamp_never_raises(self):
        ac._MEMO.pop('runtime_stamp', None)
        stamp = ac.runtime_stamp()
        self.assertEqual(set(stamp), {'trt', 'sm'})


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
        with mock.patch.object(ac, '_reference_session', lambda f, cpu=False: FakeSession('xseg', cache_dir=self.cache)):
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
        with mock.patch.object(ac, '_reference_session', lambda f, cpu=False: FakeSession('xseg', cache_dir=self.cache)):
            got, providers, res = self.run_guard(bad, build)
        self.assertIs(got, good)
        self.assertEqual(len(chains), 1)
        opts = next(p[1] for p in chains[0] if isinstance(p, tuple) and 'tensorrt' in p[0].lower())
        self.assertFalse(opts['trt_fp16_enable'])
        self.assertIn('TensorRT FP32 engine instead', self.out)
        self.assertEqual(self._canary_records()[-1]['active'], 'tensorrt-fp32')

    def _canary_records(self):
        from roop import backend_manager
        return [r for r in backend_manager.provider_degradations()
                if r['stage'] == 'canary' and r['model'] == 'xseg']

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
        with mock.patch.object(ac, '_reference_session', lambda f, cpu=False: FakeSession('xseg', cache_dir=self.cache)):
            got, providers, res = self.run_guard(bad, build)
        self.assertIs(got, cuda)
        self.assertEqual(len(seen), 2)
        self.assertFalse(any('tensorrt' in (p[0] if isinstance(p, tuple) else p).lower() for p in seen[1]))
        self.assertIn('falling back to CUDAExecutionProvider', self.out)
        self.assertEqual(self._canary_records()[-1]['active'], 'CUDAExecutionProvider')
        self.assertIn('diverges', self._canary_records()[-1]['reason'])

    def test_an_fp32_engine_that_fails_goes_straight_to_cuda(self):
        bad = FakeSession('xseg', fn=lambda x: 1.0 - _fn('xseg')(x), cache_dir=self.cache, fp16=False)
        providers = [('TensorrtExecutionProvider', {'trt_fp16_enable': False, 'trt_engine_cache_path': self.cache}),
                     'CUDAExecutionProvider', 'CPUExecutionProvider']
        seen = []
        cuda = FakeSession('xseg', trt=False)

        def build(chain):
            seen.append(chain)
            return cuda
        with mock.patch.object(ac, '_reference_session', lambda f, cpu=False: FakeSession('xseg', cache_dir=self.cache)):
            got, used, res = self.run_guard(bad, build, providers)
        self.assertIs(got, cuda)
        self.assertEqual(len(seen), 1)

    def test_a_skipped_check_never_rebuilds(self):
        s = self.session('xseg')
        built = []
        with mock.patch.object(ac, '_reference_session', lambda f, cpu=False: None):
            got, providers, res = self.run_guard(s, lambda chain: built.append(chain))
        self.assertIs(got, s)
        self.assertEqual(built, [])
        self.assertIn('skipped', self.out)


if __name__ == '__main__':
    unittest.main()
