"""Guards on the hardware benchmark's decision rules and on its wiring.

The benchmark this replaced measured a batched `torch.matmul` in ONE thread,
called the batch dimension "threads", reported the result as "FPS" (18,698 of
them on a card whose real swap throughput is single digits) and fed the winner
straight into `Settings.resolve_threads`. Nothing about that was detectable from
its output: the numbers looked like numbers and the UI showed a green tick.

So what is pinned here is not the measurements — those are per machine — but the
things whose breakage would again be invisible:

  * the knee rule, which is what turns a curve into a setting;
  * best-of-N, which is what makes a 4% margin mean anything under 30% noise;
  * the feed builder, which is what makes a per-call time a real per-call time;
  * the catalogue's tile arithmetic, which is the difference between "9 ms" and
    "38 ms per face" on the same model;
  * the start gate, whose obvious wrong version silently measures zero;
  * and the seam to `resolve_threads`, which is the whole point of running it.
"""

import os
import sys
import threading
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import numpy as np                                          # noqa: E402

import roop.globals as g                                    # noqa: E402
from roop import bench                                      # noqa: E402


class _FakeInput:
    def __init__(self, name, shape, typ='tensor(float)'):
        self.name = name
        self.shape = shape
        self.type = typ


class _FakeSession:
    """Only what `make_feeds` reads. A real session needs a GPU; the contract
    being tested is 'what does it build from a signature', which does not."""

    def __init__(self, inputs):
        self._inputs = inputs

    def get_inputs(self):
        return self._inputs


class KneePicksTheSmallestThatIsAsGood(unittest.TestCase):
    """The knee is the recommendation. Argmax is not."""

    def test_plateau_takes_the_cheap_end(self):
        # 4 is 0.5% better than 2 and costs twice the VRAM: not worth it.
        self.assertEqual(bench._knee([100, 198, 199], [1, 2, 4], 0.04), 2)

    def test_a_real_gain_is_taken(self):
        self.assertEqual(bench._knee([100, 150, 300], [1, 2, 4], 0.04), 4)

    def test_a_regression_at_the_top_is_not_chosen(self):
        # More instances measuring SLOWER is the oversubscription case, and the
        # answer there is the level before it, never the widest tried.
        self.assertEqual(bench._knee([100, 200, 120], [1, 2, 4], 0.04), 2)

    def test_single_level_is_that_level(self):
        self.assertEqual(bench._knee([42], [1], 0.04), 1)

    def test_no_data_does_not_raise(self):
        self.assertEqual(bench._knee([], [], 0.04), 1)


class BestOfTakesTheFastestRun(unittest.TestCase):
    """Noise here is one-sided — interference can only ever slow a run down."""

    def test_returns_the_maximum(self):
        vals = iter([120.0, 300.0, 90.0])
        self.assertEqual(bench.best_of(3, lambda: next(vals)), 300.0)

    def test_unwraps_the_throughput_tuple(self):
        # `throughput` returns (calls_s, total); best_of must compare the rate,
        # not the tuple, or it silently compares call COUNTS between runs of
        # different length.
        vals = iter([(10.0, 5), (20.0, 3)])
        self.assertEqual(bench.best_of(2, lambda: next(vals)), 20.0)

    def test_reps_below_one_still_runs_once(self):
        calls = []
        bench.best_of(0, lambda: calls.append(1) or 1.0)
        self.assertEqual(len(calls), 1)


class FeedsComeFromTheSessionSignature(unittest.TestCase):

    def test_dynamic_batch_axis_takes_the_batch(self):
        s = _FakeSession([_FakeInput('x', ['N', 3, 112, 112])])
        feeds = bench.make_feeds(s, batch=4)
        self.assertEqual(feeds['x'].shape, (4, 3, 112, 112))

    def test_dynamic_spatial_axes_take_the_hint(self):
        # retinaface_r50 declares ('b', 3, 'h', 'w'). Without the hint it would
        # be timed at 1x1 and report a detector 400x faster than it is.
        s = _FakeSession([_FakeInput('input', ['b', 3, 'h', 'w'])])
        feeds = bench.make_feeds(s, shape_hint=(640, 640))
        self.assertEqual(feeds['input'].shape, (1, 3, 640, 640))

    def test_static_shapes_are_left_alone(self):
        s = _FakeSession([_FakeInput('target', [1, 3, 256, 256]),
                          _FakeInput('source', [1, 512])])
        feeds = bench.make_feeds(s, shape_hint=(640, 640))
        self.assertEqual(feeds['target'].shape, (1, 3, 256, 256))
        self.assertEqual(feeds['source'].shape, (1, 512))

    def test_dtype_comes_from_the_model(self):
        s = _FakeSession([_FakeInput('i', [1, 4], 'tensor(int64)'),
                          _FakeInput('h', [1, 4], 'tensor(float16)')])
        feeds = bench.make_feeds(s)
        self.assertEqual(feeds['i'].dtype, np.int64)
        self.assertEqual(feeds['h'].dtype, np.float16)

    def test_scalar_input_is_a_scalar(self):
        # CodeFormer's fidelity weight is declared with an empty shape.
        s = _FakeSession([_FakeInput('w', [])])
        self.assertEqual(bench.make_feeds(s)['w'].shape, ())

    def test_only_the_declared_inputs_are_fed(self):
        # GPEN's export declares every weight as a graph input; onnxruntime
        # resolves those against initialisers and does not list them. Feeding
        # anything not in get_inputs() would upload hundreds of MB per call and
        # time the PCIe bus instead of the network.
        s = _FakeSession([_FakeInput('input', [1, 3, 512, 512])])
        self.assertEqual(list(bench.make_feeds(s)), ['input'])


class StartGateStartsTheClockAfterWarmup(unittest.TestCase):
    """The obvious version of this races, and the race reports zero work.

    Main-thread-waits-on-barrier-then-sets-the-deadline releases the workers at
    the same instant, so they read a deadline of 0, decide time is up, and count
    nothing. The deadline therefore has to be set by the barrier's own action.
    """

    def test_deadline_is_live_the_moment_a_worker_is_released(self):
        gate = bench._StartGate(2, seconds=5.0)
        seen = []

        def worker():
            time.sleep(0.05)            # stand in for warm-up
            gate.ready()
            seen.append(gate.expired())

        ts = [threading.Thread(target=worker) for _ in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=10)
        self.assertEqual(seen, [False, False])

    def test_elapsed_excludes_the_warmup(self):
        gate = bench._StartGate(1, seconds=1.0)
        time.sleep(0.2)
        gate.ready()
        self.assertLess(gate.elapsed(), 0.15)

    def test_a_missing_party_does_not_hang_forever(self):
        gate = bench._StartGate(2, seconds=1.0)
        gate._barrier.abort()
        self.assertFalse(gate.ready())


class CatalogueFollowsTheSettings(unittest.TestCase):
    """The benchmark has to measure the models the user renders with."""

    def setUp(self):
        from settings import Settings
        self._saved_cfg = g.CFG
        self._saved_prov = g.execution_providers
        g.CFG = Settings(os.path.join(HERE, 'no-such-config.yaml'))
        g.execution_providers = ['CPUExecutionProvider']

    def tearDown(self):
        g.CFG = self._saved_cfg
        g.execution_providers = self._saved_prov

    def _stage(self, key):
        stages, _ = bench.build_catalogue(1.0)
        return next((s for s in stages if s.key == key), None)

    def test_pixel_boost_tiles_are_counted_per_face(self):
        # A 128px model at the 512px setting runs SIXTEEN times per face. Timing
        # one call and calling it the swap cost understates it by that factor —
        # the trap `tools/bench_stages.py` documents in its own header.
        g.CFG.swap_model = 'inswapper'          # 128 output
        g.CFG.subsample_upscale = '512px'
        self.assertEqual(self._stage('swap').calls_per_frame, 16.0)

        g.CFG.subsample_upscale = '256px'
        self.assertEqual(self._stage('swap').calls_per_frame, 4.0)

        g.CFG.swap_model = 'hyperswap'          # 256 output, 1 tile at 256
        self.assertEqual(self._stage('swap').calls_per_frame, 1.0)

    def test_per_face_stages_scale_with_faces_per_frame(self):
        stages, _ = bench.build_catalogue(2.5)
        by_key = {s.key: s for s in stages}
        self.assertEqual(by_key['recognize'].calls_per_frame, 2.5)
        # The detector runs once per FRAME however many faces are in it.
        self.assertEqual(by_key['detect'].calls_per_frame, 1.0)

    def test_workload_modes_compose_the_way_the_api_resolves_them(self):
        # api.py picks 'enhanced' when an enhancer is set and 'heavy' when a mask
        # engine or expression restore is on. The composite has to agree, or the
        # thread count recommended for a mode is for a different pipeline.
        g.CFG.selected_enhancer = 'Restoreformer++'
        g.CFG.mask_engine = 'DFL XSeg'
        stages, _ = bench.build_catalogue(1.0)
        modes = {s.key: s.in_modes for s in stages}
        self.assertNotIn('standard', modes['enhance'])
        self.assertIn('enhanced', modes['enhance'])
        self.assertEqual(modes.get('mask_xseg'), ('heavy',))
        self.assertIn('standard', modes['swap'])

    def test_an_unpooled_enhancer_is_reported_as_unpooled(self):
        # GPEN and GFPGAN have no SessionPool, so under TensorRT they take the
        # global GPU lock and serialise every other stage behind them. That is
        # the single most useful thing the report can say about a configuration,
        # and it is derived here.
        g.CFG.selected_enhancer = 'GPEN'
        self.assertFalse(self._stage('enhance').pooled)
        g.CFG.selected_enhancer = 'Restoreformer++'
        # The catalogue follows the runtime pool capability. Make that
        # capability explicit here so this wiring test is independent of the
        # machine running the test suite (the physical 6 GB laptop disables
        # pools by design).
        with mock.patch('roop.session_pool.pooling_enabled', return_value=True):
            self.assertTrue(self._stage('enhance').pooled)

    def test_a_selected_model_that_is_not_on_disk_warns_instead_of_failing(self):
        g.CFG.selected_enhancer = 'DMDNet'      # a .pth, not single-file ONNX
        stages, warnings = bench.build_catalogue(1.0)
        self.assertIsNone(next((s for s in stages if s.key == 'enhance'), None))
        self.assertTrue(any('DMDNet' in w for w in warnings))

    def test_codeformer_fp16_catalogue_uses_the_fp16_graph(self):
        path, note = bench._enhancer_model('Codeformer (fp16)')
        self.assertTrue(path.endswith(os.path.join('CodeFormer',
                                                   'codeformer.fp16.onnx')))
        self.assertIn('distinct FP16', note)

    def test_stress_profile_is_a_multi_face_sustained_test(self):
        profile = bench.PROFILES['stress']
        self.assertGreaterEqual(profile['default_faces'], 2.0)
        self.assertGreaterEqual(profile['min_faces'], 2.0)
        self.assertGreaterEqual(profile['measure_sec'], 10.0)
        self.assertGreaterEqual(profile['warm_sec'], 1.0)
        self.assertGreaterEqual(profile['composite_reps'], 3)

    def test_composite_metrics_keep_repeated_run_distribution(self):
        import inspect
        src = inspect.getsource(bench.measure_composite)
        self.assertIn("'runs_fps'", src)
        self.assertIn("'spread_pct'", src)


class StandaloneRunReadsTheRealConfig(unittest.TestCase):
    """The CLI has to measure the same configuration the UI would.

    The catalogue is built off `g.CFG` and `g.execution_providers`, and under the
    UI both are populated before the panel can start anything — `core.run` assigns
    CFG, `ui.main` resolves the provider. Run as `python -m roop.bench` nothing
    does, and the failure is quiet: CFG falls back to a defaults Settings and the
    module-level provider list is taken for a decision. The observed CLI report
    read "providers: CUDAExecutionProvider, CPUExecutionProvider (config: /)" and
    benchmarked inswapper @128 + SCRFD against a config.yaml that selects
    tensorrt, hyperswap and RestoreFormer++ — right models, wrong ones.
    """

    def setUp(self):
        self._saved = (g.CFG, g.execution_providers, g.execution_threads)

    def tearDown(self):
        g.CFG, g.execution_providers, g.execution_threads = self._saved

    def test_an_unconfigured_process_is_detected_despite_a_populated_provider_list(self):
        # THE bug in the first version of this guard. `execution_providers` is
        # initialised in roop/globals.py to a two-provider list, so `if not
        # g.execution_providers` is False on a process that has resolved
        # nothing — the check has to hang off CFG, which really is None.
        self.assertTrue(g.execution_providers,
                        'globals no longer pre-populates execution_providers; '
                        'this guard can be simplified')
        g.CFG = None
        self.assertTrue(bench.bootstrap_globals())

    def test_bootstrap_reads_the_apps_config_file(self):
        g.CFG = None
        bench.bootstrap_globals()
        from settings import Settings
        on_disk = Settings(os.path.join(APP, 'config.yaml'))
        for field in ('swap_model', 'selected_enhancer', 'detector_engine',
                      'mask_engine', 'provider', 'subsample_upscale'):
            self.assertEqual(getattr(g.CFG, field), getattr(on_disk, field),
                             f'{field} does not match config.yaml')

    def test_config_is_found_from_any_working_directory(self):
        # Resolved against the module, not the shell's cwd. A relative miss does
        # not raise — it silently yields a defaults Settings, which is the exact
        # failure this class exists for.
        cwd = os.getcwd()
        try:
            os.chdir(os.path.dirname(APP))
            g.CFG = None
            bench.bootstrap_globals()
            from settings import Settings
            self.assertEqual(g.CFG.swap_model,
                             Settings(os.path.join(APP, 'config.yaml')).swap_model)
        finally:
            os.chdir(cwd)

    def test_the_configured_provider_reaches_the_provider_list(self):
        g.CFG = None
        g.execution_providers = ['CPUExecutionProvider']
        bench.bootstrap_globals()
        names = [p if isinstance(p, str) else p[0] for p in g.execution_providers]
        if g.CFG.provider == 'tensorrt':
            try:
                import tensorrt                          # noqa: F401
            except ImportError:
                self.assertNotIn('TensorrtExecutionProvider', names)
            else:
                import torch
                small_gpu = (torch.cuda.is_available() and
                             torch.cuda.get_device_properties(0).total_memory /
                             (1024 ** 3) < 7.0)
                if small_gpu:
                    self.assertEqual(names[0], 'CUDAExecutionProvider')
                else:
                    self.assertEqual(names[0], 'TensorrtExecutionProvider')
                    # CUDA behind it, as ui/main.py does, so a node TensorRT
                    # cannot build lands on CUDA instead of falling through to
                    # CPU.
                    self.assertIn('CUDAExecutionProvider', names)
        else:
            self.assertNotIn('TensorrtExecutionProvider', names)

    def test_a_booted_app_is_left_alone(self):
        # api.py calls run_benchmark inside the live process. Overwriting the
        # provider list there would benchmark one configuration and render with
        # another.
        from settings import Settings
        sentinel = Settings(os.path.join(HERE, 'no-such-config.yaml'))
        g.CFG = sentinel
        g.execution_providers = ['CPUExecutionProvider']
        self.assertFalse(bench.bootstrap_globals())
        self.assertIs(g.CFG, sentinel)
        self.assertEqual(g.execution_providers, ['CPUExecutionProvider'])

    def test_main_bootstraps_before_it_measures(self):
        # The order is the whole point: build_catalogue reads the globals, so a
        # bootstrap after run_benchmark would be a no-op that still passes a
        # "does it get called" test.
        import ast
        with open(os.path.join(APP, 'roop', 'bench.py'), encoding='utf-8') as fh:
            tree = ast.parse(fh.read())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == 'main')
        called = [i for i, node in enumerate(ast.walk(fn))
                  if isinstance(node, ast.Call)
                  and getattr(node.func, 'id', '') == 'bootstrap_globals']
        self.assertTrue(called, 'main() never calls bootstrap_globals')
        src = ast.dump(fn)
        self.assertLess(src.index('bootstrap_globals'), src.index('run_benchmark'),
                        'bootstrap_globals must run before run_benchmark')


class TheResultReachesResolveThreads(unittest.TestCase):
    """The seam that makes the benchmark matter at all.

    `Settings.resolve_threads` reads `benchmark_results['best_threads'][mode]`
    and hands the answer to `roop.globals.execution_threads` for every run. A
    report that renames that key measures a machine and changes nothing.
    """

    def test_resolve_threads_reads_the_reports_own_key(self):
        from settings import Settings
        cfg = Settings(os.path.join(HERE, 'no-such-config.yaml'))
        cfg.auto_thread_selection = True
        cfg.benchmark_results = {'best_threads': {'standard': 7, 'enhanced': 5,
                                                  'heavy': 3}}
        self.assertEqual(cfg.resolve_threads('standard'), 7)
        self.assertEqual(cfg.resolve_threads('enhanced'), 5)
        self.assertEqual(cfg.resolve_threads('heavy'), 3)

    def test_manual_mode_still_wins(self):
        from settings import Settings
        cfg = Settings(os.path.join(HERE, 'no-such-config.yaml'))
        cfg.auto_thread_selection = False
        cfg.max_threads = 6
        cfg.benchmark_results = {'best_threads': {'standard': 16}}
        self.assertEqual(cfg.resolve_threads('standard'), 6)

    def test_auto_threads_ignore_benchmark_for_different_enhancer(self):
        from settings import Settings
        cfg = Settings(os.path.join(HERE, 'no-such-config.yaml'))
        cfg.auto_thread_selection = True
        cfg.selected_enhancer = 'UltraMax'
        cfg.benchmark_results = {
            'best_threads': {'heavy': 6},
            'settings_measured': {
                'provider': cfg.provider,
                'swap_model': cfg.swap_model,
                'enhancer': 'GPEN 256 Pro',
                'mask_engine': cfg.mask_engine,
                'detector_engine': cfg.detector_engine,
                'subsample_upscale': cfg.subsample_upscale,
            },
        }
        self.assertNotEqual(cfg.resolve_threads('heavy'), 6)

    def test_recommend_emits_that_key(self):
        device = {'cpu_logical': 16, 'cpu_physical': 8}
        curves = {'standard': {'1': 5.0, '2': 9.0, '4': 9.1},
                  'enhanced': {'1': 3.0, '2': 5.5},
                  'heavy': {'1': 2.0, '2': 3.9, '4': 6.0}}
        rec = bench.recommend([], device, {'trt_pool': 2}, curves, [], {}, {})
        self.assertEqual(rec['threads']['standard'], 2)   # 4 adds 1%: not worth it
        self.assertEqual(rec['threads']['enhanced'], 2)
        self.assertEqual(rec['threads']['heavy'], 4)

    def test_threads_never_exceed_the_logical_cores(self):
        device = {'cpu_logical': 4, 'cpu_physical': 2}
        curves = {'standard': {'1': 1.0, '16': 99.0}}
        rec = bench.recommend([], device, {}, curves, [], {}, {})
        self.assertEqual(rec['threads']['standard'], 4)


class PoolRecommendationsRespectExplicitOverrides(unittest.TestCase):
    def test_small_card_benchmark_does_not_persist_wide_pools(self):
        from settings import Settings
        saved = g.CFG
        try:
            g.CFG = Settings(os.path.join(HERE, 'no-such-config.yaml'))
            g.CFG.save = lambda: None
            g.CFG.perf_trt_pool = 'auto'
            g.CFG.perf_detmask_pool = 'auto'
            g.CFG.perf_detector_pool = 'auto'
            g.CFG.perf_expr_pool = 'auto'
            fake_props = type('Props', (), {'total_memory': 6 * 1024 ** 3})()
            with mock.patch('torch.cuda.is_available', return_value=True), \
                    mock.patch('torch.cuda.get_device_properties',
                               return_value=fake_props):
                out = bench.apply_recommendation({'recommend': {
                    'pools': {'trt_pool': 3, 'detmask_pool': 3,
                              'detector_pool': 2, 'expr_pool': 1}}})
            self.assertEqual(g.CFG.perf_trt_pool, 'auto')
            self.assertEqual(g.CFG.perf_detmask_pool, 'auto')
            self.assertEqual(g.CFG.perf_detector_pool, 'auto')
            self.assertEqual(g.CFG.perf_expr_pool, 'auto')
            self.assertNotIn('safety_resets', out['pending_restart'])
        finally:
            g.CFG = saved

    def test_numeric_pool_settings_are_not_replaced(self):
        from settings import Settings
        saved = g.CFG
        try:
            g.CFG = Settings(os.path.join(HERE, 'no-such-config.yaml'))
            g.CFG.save = lambda: None
            g.CFG.perf_trt_pool = '2'
            g.CFG.perf_detmask_pool = '2'
            g.CFG.perf_detector_pool = '2'
            g.CFG.perf_expr_pool = '2'
            fake_props = type('Props', (), {'total_memory': 12 * 1024 ** 3})()
            with mock.patch('torch.cuda.is_available', return_value=True), \
                    mock.patch('torch.cuda.get_device_properties',
                               return_value=fake_props):
                out = bench.apply_recommendation({'recommend': {
                    'pools': {'trt_pool': 3, 'detmask_pool': 4,
                              'detector_pool': 6, 'expr_pool': 3}}})
            self.assertEqual(g.CFG.perf_trt_pool, '2')
            self.assertEqual(g.CFG.perf_detmask_pool, '2')
            self.assertEqual(g.CFG.perf_detector_pool, '2')
            self.assertEqual(g.CFG.perf_expr_pool, '2')
            self.assertEqual(set(out['pending_restart']['explicit_overrides']), {
                'perf_trt_pool', 'perf_detmask_pool', 'perf_detector_pool',
                'perf_expr_pool'})
        finally:
            g.CFG = saved


class TheEncoderIsChosenToKeepUpNotToWin(unittest.TestCase):
    """Encoding streams alongside the swap, so "fastest" is the wrong question.

    `ProcessMgr.write_frames_thread` feeds a live FFMPEG_VideoWriter while the
    pipeline runs. An encoder already ahead of the pipeline therefore buys no
    wall clock at all, and the fastest row measured is the one that costs the
    most per frame stored — on the reference machine hevc_nvenc p5 ran 170 f/s
    against libx265 medium's 42.5 for a file 15x larger.
    """

    ROWS = [
        {'name': 'libx265 medium', 'fps': 42.5, 'mb': 0.10},
        {'name': 'libx265 faster', 'fps': 94.8, 'mb': 0.40},
        {'name': 'libx264 medium', 'fps': 91.9, 'mb': 0.20},
        {'name': 'libx264 faster', 'fps': 150.4, 'mb': 0.60},
        {'name': 'h264_nvenc p5', 'fps': 121.5, 'mb': 2.90},
        {'name': 'hevc_nvenc p5', 'fps': 170.0, 'mb': 1.50},
    ]
    THREADS = {'standard': 8, 'enhanced': 12, 'heavy': 4}
    CURVES = {'standard': {'8': 28.4}, 'enhanced': {'12': 13.4}, 'heavy': {'4': 9.9}}

    def _rec(self, rows=None, current='libx265'):
        return bench._recommend_encoder(rows or self.ROWS, self.THREADS,
                                        self.CURVES, current)

    def test_the_fastest_encoder_is_not_the_recommendation(self):
        out = self._rec()
        self.assertEqual(out['encoder_fastest'], 'hevc_nvenc p5')
        self.assertNotEqual(out['encoder'], 'hevc_nvenc p5')

    def test_it_moves_the_preset_and_keeps_the_codec(self):
        # libx265 medium (42.5) does not clear 2x28.4; libx265 faster does. The
        # preset is the quality-neutral lever — same CRF, same perceptual
        # quality, slightly larger file — so it is the one that should move.
        self.assertEqual(self._rec()['encoder'], 'libx265 faster')

    def test_size_never_decides_ACROSS_codecs(self):
        """The bug this rule was rewritten to avoid.

        Ranking every fast-enough row by megabytes picks libx264 medium (0.20 MB)
        over libx265 faster (0.40 MB) and silently changes the codec. CRF 18 does
        not mean the same quality to x264 and x265, so that comparison is between
        two different qualities and cannot justify anything.
        """
        self.assertEqual(self._rec(current='libx265')['encoder'].split()[0], 'libx265')
        self.assertEqual(self._rec(current='libx264')['encoder'].split()[0], 'libx264')

    def test_a_codec_that_cannot_keep_up_does_move(self):
        # 4K, or a CPU busy enough that no software preset clears the bar. NVENC
        # is right there for a second reason: it runs on the dedicated encode
        # engine, off the cores the workers are competing for.
        slow = [dict(r, fps=r['fps'] / 6.0) for r in self.ROWS]
        out = self._rec(slow)
        self.assertEqual(out['encoder'], 'hevc_nvenc p5')
        self.assertIn('clears', out['encoder_reason'])

    def test_an_encoder_already_fast_enough_is_left_completely_alone(self):
        self.assertEqual(self._rec(current='libx264')['encoder'], 'libx264 medium')
        self.assertEqual(self._rec(current='hevc_nvenc')['encoder'], 'hevc_nvenc p5')

    def test_the_reason_does_not_compare_the_winner_to_itself(self):
        # Already on the fastest encoder, the sentence read "hevc_nvenc p5 is
        # faster but the render cannot use it" — about hevc_nvenc p5.
        out = self._rec(current='hevc_nvenc')
        self.assertEqual(out['encoder'], out['encoder_fastest'])
        self.assertNotIn('is faster but', out['encoder_reason'])

    def test_no_thread_curve_falls_back_to_the_fastest(self):
        out = bench._recommend_encoder(self.ROWS, {}, {}, 'libx265')
        self.assertEqual(out['encoder'], 'hevc_nvenc p5')

    def test_a_missing_size_cannot_win_on_absent_data(self):
        rows = [{'name': 'libx265 medium', 'fps': 99.0},          # no 'mb'
                {'name': 'libx265 faster', 'fps': 99.0, 'mb': 0.40}]
        self.assertEqual(self._rec(rows)['encoder'], 'libx265 faster')

    def test_recommend_actually_routes_through_the_rule(self):
        """The wiring, not the rule.

        Every assertion above calls `_recommend_encoder` directly, so restoring
        the old one-liner in `recommend()` — sort by fps, take the first — left
        all of them green while the report went back to recommending the fastest
        encoder. The seam has to be tested from the caller's side.
        """
        from settings import Settings
        saved = g.CFG
        try:
            g.CFG = Settings(os.path.join(HERE, 'no-such-config.yaml'))
            g.CFG.output_video_codec = 'libx265'
            rec = bench.recommend([], {'cpu_logical': 32}, {}, self.CURVES, [],
                                  {'encode': self.ROWS}, {})
        finally:
            g.CFG = saved
        self.assertEqual(rec['encoder'], 'libx265 faster')
        self.assertEqual(rec['encoder_fastest'], 'hevc_nvenc p5')
        self.assertIn('encoder_reason', rec)


class TheEncoderRecommendationIsApplied(unittest.TestCase):
    """It was measured, printed, and then acted on by nobody.

    The codec and the preset have different lifetimes and the report has to say
    so: `_run_swap` re-reads output_video_codec from config on every run, while
    perf_encoder_preset leaves as ROOP_ENCODER_PRESET, which run.py exports once
    at startup.
    """

    def setUp(self):
        from settings import Settings
        self._saved = g.CFG
        g.CFG = Settings(os.path.join(HERE, 'no-such-config.yaml'))
        g.CFG.save = lambda: None               # never touch a real config.yaml

    def tearDown(self):
        g.CFG = self._saved

    def _apply(self, encoder):
        return bench.apply_recommendation({'recommend': {'encoder': encoder}})

    def test_a_preset_change_is_reported_as_needing_a_restart(self):
        g.CFG.output_video_codec = 'libx265'
        g.CFG.perf_encoder_preset = 'auto'
        out = self._apply('libx265 faster')
        self.assertEqual(out['pending_restart'].get('perf_encoder_preset'), 'faster')
        self.assertEqual(g.CFG.perf_encoder_preset, 'faster')
        self.assertNotIn('output_video_codec', out['applied_now'])

    def test_a_codec_change_is_reported_as_live(self):
        g.CFG.output_video_codec = 'libx265'
        out = self._apply('hevc_nvenc p5')
        self.assertEqual(out['applied_now'].get('output_video_codec'), 'hevc_nvenc')
        self.assertEqual(g.CFG.output_video_codec, 'hevc_nvenc')
        self.assertNotIn('output_video_codec', out['pending_restart'])

    def test_an_nvenc_preset_is_not_written_into_the_x264_style_knob(self):
        # ffmpeg_writer consults ROOP_ENCODER_PRESET for libx264/libx265 only and
        # validates against x264 preset names; 'p5' would fail that check and
        # silently become 'faster'. NVENC's preset is ROOP_NVENC_PRESET.
        g.CFG.output_video_codec = 'libx265'
        g.CFG.perf_encoder_preset = 'auto'
        self._apply('hevc_nvenc p5')
        self.assertEqual(g.CFG.perf_encoder_preset, 'auto')

    def test_a_codec_outside_the_apps_own_list_is_refused(self):
        g.CFG.output_video_codec = 'libx265'
        self._apply('av1_qsv fast')
        self.assertEqual(g.CFG.output_video_codec, 'libx265')

    def test_the_codec_list_matches_the_one_the_ui_offers(self):
        # bench keeps its own copy so the CLI does not have to import api.py and
        # boot FastAPI. Copies drift; this is what stops it.
        import re
        with open(os.path.join(APP, 'api.py'), encoding='utf-8') as fh:
            m = re.search(r'^_VIDEO_CODECS\s*=\s*\[(.*?)\]', fh.read(), re.M | re.S)
        self.assertIsNotNone(m, '_VIDEO_CODECS not found in api.py')
        self.assertEqual(sorted(re.findall(r'"([^"]+)"', m.group(1))),
                         sorted(bench.APP_CODECS))


class PerInstanceVramIsMarginal(unittest.TestCase):
    """The first session also pays for the CUDA context, the engine
    deserialisation and the arena; charging that to instance two overstates a
    pool by hundreds of MB and stops the sweep before it has found the knee."""

    def test_cost_is_the_slope_not_the_intercept(self):
        scaling = [{'n': 1, 'free_vram_gb': 10.0}, {'n': 4, 'free_vram_gb': 8.5}]
        self.assertAlmostEqual(bench._per_instance_mb(scaling),
                               (1.5 * 1024) / 3, places=3)

    def test_one_level_reports_nothing_rather_than_guessing(self):
        self.assertEqual(bench._per_instance_mb([{'n': 1, 'free_vram_gb': 9.0}]), 0.0)

    def test_free_vram_going_up_does_not_report_a_negative_cost(self):
        scaling = [{'n': 1, 'free_vram_gb': 8.0}, {'n': 2, 'free_vram_gb': 9.0}]
        self.assertEqual(bench._per_instance_mb(scaling), 0.0)


class PhasesRunWithoutAGpu(unittest.TestCase):
    """Walk the control flow of the phases that need a session, with fakes.

    These exist because of a live bug the other checks could not see: an edit to
    `measure_provider_ab` dropped the two lines that build `feeds` and `outs`,
    and every stage then failed at run time with `NameError`. It was invisible to
    `test_no_undefined_names` by construction — that check treats a name bound
    ANYWHERE in a module as defined everywhere in it, and both names are bound in
    `measure_stage` and `measure_composite`. It was invisible to the compiler for
    the same reason, and to the unit tests above because they test pure
    functions. The only thing that catches it is executing the function, and the
    only thing stopping that was needing a GPU — so the GPU is faked.
    """

    class _Sess:
        def __init__(self, provider='CUDAExecutionProvider'):
            self._p = provider

        def get_providers(self):
            return [self._p, 'CPUExecutionProvider']

        def get_inputs(self):
            return [_FakeInput('x', [1, 3, 8, 8])]

        def get_outputs(self):
            return [type('O', (), {'name': 'y'})()]

        def run(self, _names, _feeds):
            return [np.zeros((1, 3, 8, 8), dtype=np.float32)]

    def setUp(self):
        self.cfg = {'measure_sec': 0.02, 'warm_sec': 0.0, 'reps': 1}
        self.logs = []
        self.report = lambda **kw: self.logs.append(kw.get('log') or kw.get('status'))
        self._build = bench._build_session
        self._tp = bench.throughput
        bench.throughput = lambda *a, **k: (100.0, 10)

    def tearDown(self):
        bench._build_session = self._build
        bench.throughput = self._tp

    def _stage(self, key='swap'):
        st = bench.Stage(key, 'Fake stage', 'fake.onnx',
                         [('TensorrtExecutionProvider', {}), 'CUDAExecutionProvider'],
                         'trt_pool', 1.0, in_modes=('standard',))
        st.ms_call = 5.0
        return st

    def test_provider_ab_produces_a_row(self):
        bench._build_session = lambda st, providers=None: self._Sess()
        rows = bench.measure_provider_ab([self._stage()], self.cfg, self.report,
                                         lambda: False)
        self.assertEqual(len(rows), 1)
        self.assertNotIn('error', rows[0], f'phase raised internally: {rows[0]}')
        self.assertEqual(rows[0]['cuda_ms'], 10.0)          # 100 calls/s
        self.assertFalse(rows[0]['cuda_fell_back'])

    def test_provider_ab_flags_a_silent_cpu_fallback(self):
        # onnxruntime does not raise when the CUDA EP refuses a node; it drops
        # the EP and rebuilds on CPU. The row must say so, or a CPU time is read
        # as "TensorRT is 18x faster than CUDA".
        bench._build_session = lambda st, providers=None: self._Sess('CPUExecutionProvider')
        rows = bench.measure_provider_ab([self._stage()], self.cfg, self.report,
                                         lambda: False)
        self.assertTrue(rows[0]['cuda_fell_back'])
        self.assertEqual(rows[0]['cuda_provider'], 'CPU')

    def test_the_fallback_is_detected_when_it_happens_ON_THE_FIRST_RUN(self):
        """The real shape of it, and what the first version of this guard missed.

        onnxruntime reports CUDA at construction and only drops the EP when a
        kernel actually fails to initialise — which is during the first
        inference. Measured on this machine with one RestoreFormer++ session:

            before run: ['CUDAExecutionProvider', 'CPUExecutionProvider']
            first run:  1948 ms          <- the EP is dropped in here
            after run:  ['CPUExecutionProvider']

        A guard placed before the measurement reads the intention and passes, so
        a 1005 ms CPU time went into the report as "CUDA", 17.83x slower than
        TensorRT, with `stages_cuda_refused: 0`.
        """
        class _FlipsOnRun(PhasesRunWithoutAGpu._Sess):
            def __init__(self):
                super().__init__('CUDAExecutionProvider')

            def run(self, names, feeds):
                self._p = 'CPUExecutionProvider'     # what ORT does, when it does it
                return super().run(names, feeds)

        sess = _FlipsOnRun()
        bench._build_session = lambda st, providers=None: sess
        # A throughput stub that actually RUNS, so the flip can happen; the
        # module-level stub in setUp never touches the session.
        bench.throughput = lambda sessions, feeds, outs, *a, **k: (
            sessions[0].run(outs, feeds[0]) and None) or (100.0, 10)

        rows = bench.measure_provider_ab([self._stage()], self.cfg, self.report,
                                         lambda: False)
        self.assertTrue(rows[0]['cuda_fell_back'],
                        'the provider was read before the run — a CPU time is '
                        'being reported as a CUDA time')
        self.assertEqual(rows[0]['cuda_provider'], 'CPU')

        # And the verdict must drop the row, not average a CPU time into it —
        # while still SAYING that it dropped one. With this the only row, the
        # verdict has nothing left to compare, and reporting nothing at all
        # would read as "the provider was never measured".
        rec = bench.recommend([], {'cpu_logical': 8}, {}, {}, rows, {}, {})
        self.assertEqual(rec['provider']['stages_cuda_refused'], 1)
        self.assertEqual(rec['provider']['stages_compared'], 0)
        self.assertIsNone(rec['provider']['cuda_ms_frame'])
        self.assertEqual(rec['provider']['recommend'], 'tensorrt')

    def test_the_provider_row_is_printed_with_the_provider_that_ran(self):
        # `_print_report` hardcoded the word CUDA, so the flagged row looked
        # exactly like a genuine 17x TensorRT win in the terminal report.
        import ast
        with open(os.path.join(APP, 'roop', 'bench.py'), encoding='utf-8') as fh:
            tree = ast.parse(fh.read())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == '_print_report')
        src = ast.dump(fn)
        self.assertIn('cuda_fell_back', src,
                      '_print_report never surfaces the fallback flag')
        self.assertIn('cuda_provider', src,
                      '_print_report labels every row CUDA regardless of what ran')

    def test_a_row_that_cannot_run_on_cuda_at_all_still_prints(self):
        # cuda_ms is None there, and a `:7.2f` on None raises out of the report
        # after the whole measured run has completed.
        res = {'summary': 's', 'stages': [], 'thread_curve': {},
               'provider_ab': [{'stage': 'x', 'trt_ms': 5.0, 'cuda_ms': None,
                                'trt_speedup': None, 'calls_per_frame': 1.0,
                                'error': 'RuntimeError: nope'}],
               'recommend': {}}
        bench._print_report(res)            # must not raise

    def test_a_fallen_back_row_is_left_out_of_the_provider_verdict(self):
        rows = [{'stage': 'a', 'trt_ms': 5.0, 'cuda_ms': 1000.0, 'calls_per_frame': 1.0,
                 'cuda_fell_back': True},
                {'stage': 'b', 'trt_ms': 5.0, 'cuda_ms': 6.0, 'calls_per_frame': 1.0,
                 'cuda_fell_back': False}]
        rec = bench.recommend([], {'cpu_logical': 8}, {}, {}, rows, {}, {})
        self.assertEqual(rec['provider']['cuda_ms_frame'], 6.0)
        self.assertEqual(rec['provider']['stages_cuda_refused'], 1)

    def test_batch_swap_phase_runs(self):
        bench._build_session = lambda st, providers=None: self._Sess()
        out = bench.measure_batch_swap([self._stage('swap')], self.cfg,
                                       self.report, lambda: False)
        self.assertNotIn('error', out, f'phase raised internally: {out}')
        self.assertIn(out['recommend'], ('on', 'off'))


class TheSyntheticWorkloadIsGone(unittest.TestCase):
    """The specific thing that must never come back.

    A benchmark that runs its own invented tensor workload cannot be wrong in a
    way anyone notices — it produces plausible numbers on any hardware, for a
    pipeline that does not exist. Both files are checked because the measuring
    moved out of api.py and the shape of the mistake would move with it.
    """

    def _src(self, rel):
        """The module's CODE — comments and docstrings stripped.

        Both of these files explain the old benchmark in prose, quoting the very
        call it used. A plain grep is answered by that explanation and fails on a
        file that is correct, which is the trap `test_detect_cost.py` documents:
        the assertion has to be made against what runs, not what is written.
        """
        import ast
        with open(os.path.join(APP, rel), encoding='utf-8') as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            body = getattr(node, 'body', None)
            if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                  ast.AsyncFunctionDef))
                    and body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
        return ast.unparse(tree)

    def test_the_check_can_actually_fail(self):
        # A stripper that returned '' would pass every assertion below forever.
        import ast
        src = self._src(os.path.join('roop', 'bench.py'))
        self.assertIn('def run_benchmark', src)
        self.assertIn('InferenceSession', src)
        ast.parse(src)

    def test_api_no_longer_benchmarks_a_matmul(self):
        src = self._src('api.py')
        for banned in ('torch.matmul', 'torch.randn'):
            self.assertNotIn(banned, src,
                             f'{banned} is back in api.py — the benchmark is '
                             'measuring a synthetic workload again')

    def test_bench_measures_onnx_sessions_not_torch(self):
        src = self._src(os.path.join('roop', 'bench.py'))
        self.assertNotIn('torch.matmul', src)
        self.assertNotIn('torch.randn', src)
        # torch is still allowed, but only for what it is the right tool for.
        self.assertIn('mem_get_info', src)

    def test_vram_is_read_whole_device(self):
        # `max_memory_allocated` sees torch's allocator only. Every byte that
        # matters here is allocated by onnxruntime and TensorRT, so the old
        # guard read ~0.06 GB while TensorRT held gigabytes and called an
        # oversubscribed configuration safe.
        src = self._src(os.path.join('roop', 'bench.py'))
        self.assertNotIn('max_memory_allocated', src)


class DetectorSizeFollowsConfigTests(unittest.TestCase):
    """The bench must size the detector the way the app does, not by default.

    `roop/globals.py` sets `face_detector_size = '640'` as a MODULE DEFAULT, and
    `_detector_model` reads `getattr(globals, ...) or getattr(CFG, ...)`. A
    truthy '640' short-circuits that `or`, so the CFG arm was dead code and a
    machine configured for 512 was benchmarked at 640 -- under a report headed
    "from the current settings". This project measured 640 -> 512 as 1.30x at
    the detect stage, so the detector rows described a configuration production
    does not run.
    """

    def test_configured_size_wins_over_the_module_default(self):
        import roop.globals
        from roop.bench import _detector_model
        prev = roop.globals.face_detector_size
        try:
            roop.globals.face_detector_size = '512'
            self.assertEqual(_detector_model('retinaface_r50')[2], 512)
            roop.globals.face_detector_size = '640'
            self.assertEqual(_detector_model('retinaface_r50')[2], 640)
        finally:
            roop.globals.face_detector_size = prev

    def test_module_default_is_truthy_so_the_or_fallback_cannot_be_relied_on(self):
        """Pins the reason the sync is required rather than the `or` fallback."""
        import importlib
        mod = importlib.import_module('roop.globals')
        default = getattr(mod, 'face_detector_size', None)
        self.assertTrue(
            default,
            "globals.face_detector_size is falsy; if this ever becomes None the "
            "`or CFG` fallback in _detector_model works and the explicit sync in "
            "roop/bench.py could be simplified -- but until then it is required")


if __name__ == '__main__':
    unittest.main()
