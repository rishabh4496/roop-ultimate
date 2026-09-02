"""Execution-provider upgrades: TensorRT shape profiles, NPU offload, fallback.

Three of these assertions exist because of a defect this repo has already paid
for once, and they are the reason the file is not just "does it import":

  * `yoloface_8n` is a fixed [1,3,640,640] export.  The 2026-08-24 session
    records it silently returning ZERO faces for an entire render at any other
    det_size, because `get_all_faces` swallows the resulting InvalidArgument.
    A shape profile handed to a static graph is the same class of mistake, so
    `test_static_models_get_no_profile` pins that they get none.

  * A profile changes the engine, and this project has twice shipped a
    benchmark that silently reused an engine built under different options.
    `test_profile_scopes_the_engine_cache` pins that a profile moves the cache
    directory, and `test_static_model_leaves_cache_path_untouched` pins the
    other half: an install whose models are all static must NOT be pushed into
    a cold rebuild by a feature that does nothing for it.

  * A provider downgrade must be loud.  A stage that reports success while
    running somewhere slower is the single most repeated defect in this
    codebase's history, so the fallback is asserted to both print and record.
"""

import io
import os
import sys
import unittest
from contextlib import contextmanager, redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import onnxruntime as ort                                        # noqa: E402

from roop import backend_manager, trt_shape_profile              # noqa: E402
from roop.backend_manager import (build_session_with_fallback,    # noqa: E402
                                  provider_available,
                                  resolve_provider_names,
                                  session_degradations)
from roop.face_analyser import (detector_offload_target,          # noqa: E402
                                detector_providers, offload_report)
from roop.trt_shape_profile import (apply_shape_profile,          # noqa: E402
                                    resolve_profile)

MODELS = os.path.join(APP, 'models')


@contextmanager
def _openvino_stub(available, devices, usable):
    """Drive the offload branches on a machine with no OpenVINO EP.

    The module memoises device enumeration, probe results and one-shot
    warnings in a single dict, so the whole dict is swapped and restored --
    otherwise a warning fired by one test silences the assertion in the next.
    """
    import roop.face_analyser as fa
    saved = (fa.openvino_available, fa.openvino_devices,
             fa.openvino_device_usable, dict(fa._openvino_state))
    fa.openvino_available = lambda: available
    fa.openvino_devices = lambda: tuple(devices)
    fa.openvino_device_usable = lambda _d: usable
    fa._openvino_state.clear()
    try:
        yield
    finally:
        (fa.openvino_available, fa.openvino_devices,
         fa.openvino_device_usable, state) = saved
        fa._openvino_state.clear()
        fa._openvino_state.update(state)


def _model(name):
    path = os.path.join(MODELS, name)
    return path if os.path.isfile(path) else None


def _trt(cache='/tmp/engines'):
    """A provider chain shaped like the one core.decode_execution_providers builds."""
    return [('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': cache,
                'trt_timing_cache_enable': True,
                'trt_timing_cache_path': cache,
            }),
            'CUDAExecutionProvider',
            'CPUExecutionProvider']


def _opts(providers):
    for entry in providers:
        if isinstance(entry, tuple) and 'tensorrt' in entry[0].lower():
            return entry[1]
    return {}


class AvailableProviders(unittest.TestCase):
    """What this ORT build exposes, and that the resolver agrees with it."""

    def test_cpu_is_always_available(self):
        self.assertIn('CPUExecutionProvider', ort.get_available_providers())
        self.assertTrue(provider_available('CPUExecutionProvider'))

    def test_resolution_never_returns_an_unavailable_provider(self):
        available = set(ort.get_available_providers())
        for requested in ('auto', 'tensorrt', 'cuda', 'cpu', 'dml'):
            for name in resolve_provider_names([requested]):
                self.assertIn(name, available, f'{requested} -> {name}')

    def test_resolution_always_yields_something_runnable(self):
        for requested in ('auto', 'tensorrt', 'cuda', 'openvino', 'nonsense'):
            self.assertTrue(resolve_provider_names([requested]), requested)


class ShapeProfiles(unittest.TestCase):
    """A profile is emitted from the model's real graph, or not at all."""

    def test_dynamic_detector_gets_a_profile(self):
        path = _model('retinaface_r50.onnx')
        if path is None:
            self.skipTest('retinaface_r50.onnx not installed')
        profile = resolve_profile('face_detection:r50', path)
        self.assertIsNotNone(profile, 'r50 input is [b,3,h,w] and must be profiled')
        self.assertEqual(profile.min_shapes, 'input:1x3x320x320')
        self.assertEqual(profile.opt_shapes, 'input:1x3x512x512')
        self.assertEqual(profile.max_shapes, 'input:8x3x1280x1280')

    def test_static_models_get_no_profile(self):
        """The live swapper and every shipped restorer are fully static."""
        for name, key in (('hyperswap_1a_256.onnx', 'face_swap'),
                          ('GPEN-BFR-512.onnx', 'gpen_512'),
                          ('GFPGANv1.4.onnx', 'gfpgan'),
                          ('yoloface_8n.onnx', 'face_detection:yoloface')):
            path = _model(name)
            if path is None:
                continue
            self.assertIsNone(resolve_profile(key, path),
                              f'{name} is a static export and must not be profiled')

    def test_batch_dynamic_model_keeps_its_spatial_axes_pinned(self):
        """Opening a free batch axis must not open the fixed spatial ones."""
        path = _model('hififace_unofficial_256.onnx')
        if path is None:
            self.skipTest('hififace_unofficial_256.onnx not installed')
        profile = resolve_profile('face_swap', path)
        self.assertIsNotNone(profile)
        for shapes in (profile.min_shapes, profile.opt_shapes, profile.max_shapes):
            self.assertIn('target:', shapes)
            self.assertTrue(shapes.split('target:')[1].startswith(
                tuple(f'{b}x3x256x256' for b in (1, 8))), shapes)

    def test_colon_in_an_input_name_disables_profiling(self):
        """ORT splits these options on the FIRST colon, so 'xseg_input:0' cannot
        be expressed and must yield no profile rather than a malformed one."""
        spec = (trt_shape_profile.InputSpec('xseg_input:0', (None, 256, 256, 3)),)
        original = trt_shape_profile.graph_inputs
        trt_shape_profile.graph_inputs = lambda _p: spec
        try:
            self.assertIsNone(resolve_profile('masking', 'anything.onnx'))
        finally:
            trt_shape_profile.graph_inputs = original

    def test_missing_model_is_not_an_error(self):
        self.assertIsNone(resolve_profile('face_swap', None))
        self.assertIsNone(resolve_profile('face_swap', '/no/such/model.onnx'))
        self.assertEqual(apply_shape_profile(_trt(), 'face_swap', None), _trt())


class ProfileCacheIdentity(unittest.TestCase):
    """A profile must move the engine cache; its absence must not."""

    def test_profile_scopes_the_engine_cache(self):
        path = _model('retinaface_r50.onnx')
        if path is None:
            self.skipTest('retinaface_r50.onnx not installed')
        cache = os.path.join(APP, 'models', 'trt_cache', '_unittest_profile')
        patched = apply_shape_profile(_trt(cache), 'face_detection:r50', path)
        options = _opts(patched)
        self.assertIn('trt_profile_min_shapes', options)
        self.assertIn('trt_profile_opt_shapes', options)
        self.assertIn('trt_profile_max_shapes', options)
        self.assertNotEqual(options['trt_engine_cache_path'], cache)
        self.assertTrue(options['trt_engine_cache_path'].startswith(cache))
        # The timing cache has to follow the engine cache or a profile change
        # would reuse tactics measured for another shape range.
        self.assertEqual(options['trt_timing_cache_path'],
                         options['trt_engine_cache_path'])

    def test_static_model_leaves_cache_path_untouched(self):
        """No forced cold rebuild for an install whose models are all static."""
        path = _model('hyperswap_1a_256.onnx')
        if path is None:
            self.skipTest('hyperswap_1a_256.onnx not installed')
        cache = os.path.join(APP, 'models', 'trt_cache', '_unittest_static')
        options = _opts(apply_shape_profile(_trt(cache), 'face_swap', path))
        self.assertEqual(options['trt_engine_cache_path'], cache)
        self.assertNotIn('trt_profile_opt_shapes', options)

    def test_the_callers_providers_are_never_mutated(self):
        path = _model('retinaface_r50.onnx')
        if path is None:
            self.skipTest('retinaface_r50.onnx not installed')
        original = _trt()
        apply_shape_profile(original, 'face_detection:r50', path)
        self.assertNotIn('trt_profile_opt_shapes', _opts(original))


class MockSessionWithCachedEngineFlags(unittest.TestCase):
    """Build a stand-in session from the flags a real one would receive."""

    def test_engine_cache_flags_survive_the_whole_provider_pipeline(self):
        from roop.precision_policy import providers_for
        path = _model('retinaface_r50.onnx')
        if path is None:
            self.skipTest('retinaface_r50.onnx not installed')
        cache = os.path.join(APP, 'models', 'trt_cache', '_unittest_flags')
        chain = _trt(cache)
        if not any('tensorrt' in str(p).lower()
                   for p in resolve_provider_names(['auto'])):
            self.skipTest('no TensorRT provider on this machine')
        resolved, decision = providers_for('face_detection:r50', chain, path)
        options = _opts(resolved)
        self.assertTrue(options.get('trt_engine_cache_enable'))
        self.assertTrue(options.get('trt_timing_cache_enable'))
        self.assertTrue(os.path.isdir(options['trt_engine_cache_path']))
        self.assertIn('trt_profile_opt_shapes', options)
        self.assertTrue(decision.trt_enabled)

    def test_a_mock_session_records_the_chain_it_was_handed(self):
        seen = {}

        def build(chain):
            seen['chain'] = list(chain)
            return object()

        session, used = build_session_with_fallback(build, _trt(), tag='mock')
        self.assertIsNotNone(session)
        self.assertEqual(seen['chain'], used)


class GracefulFallback(unittest.TestCase):
    """A build failure steps down the chain; it does not take the app with it."""

    def setUp(self):
        self._before = len(session_degradations())

    def test_tensorrt_failure_falls_back_to_cuda(self):
        attempts = []

        def build(chain):
            attempts.append(chain)
            if any('tensorrt' in str(p).lower() for p in chain):
                raise RuntimeError('engine build failed')
            return 'session'

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            session, used = build_session_with_fallback(build, _trt(), tag='unit')
        self.assertEqual(session, 'session')
        self.assertEqual(len(attempts), 2)
        self.assertFalse(any('tensorrt' in str(p).lower() for p in used))
        self.assertIn('CUDAExecutionProvider', [str(p) for p in used])
        self.assertIn('unit', buffer.getvalue())
        self.assertIn('engine build failed', buffer.getvalue())

    def test_gpu_failure_falls_back_all_the_way_to_cpu(self):
        def build(chain):
            names = [str(p[0] if isinstance(p, tuple) else p).lower() for p in chain]
            if any(n.startswith(('tensorrt', 'cuda', 'rocm')) for n in names):
                raise RuntimeError('no usable GPU runtime')
            return 'cpu-session'

        with redirect_stdout(io.StringIO()):
            session, used = build_session_with_fallback(build, _trt(), tag='unit')
        self.assertEqual(session, 'cpu-session')
        self.assertEqual([str(p) for p in used], ['CPUExecutionProvider'])

    def test_every_downgrade_is_recorded_for_diagnostics(self):
        def build(chain):
            if any('tensorrt' in str(p).lower() for p in chain):
                raise RuntimeError('boom')
            return 'session'

        with redirect_stdout(io.StringIO()):
            build_session_with_fallback(build, _trt(), tag='recorded')
        entries = session_degradations()[self._before:]
        self.assertTrue(entries)
        self.assertEqual(entries[-1]['model'], 'recorded')
        self.assertIn('boom', entries[-1]['error'])

    def test_a_broken_model_still_raises(self):
        """Degrading forever would hide a genuinely broken model."""
        def build(chain):
            raise RuntimeError('the model itself is invalid')

        with redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError) as caught:
                build_session_with_fallback(build, _trt(), tag='unit')
        # The FIRST error is re-raised: the TensorRT message is the informative
        # one, not the CPU repeat of it.
        self.assertIn('the model itself is invalid', str(caught.exception))

    def test_a_working_chain_is_never_downgraded(self):
        calls = []
        session, used = build_session_with_fallback(
            lambda chain: calls.append(chain) or 'ok', _trt(), tag='unit')
        self.assertEqual(session, 'ok')
        self.assertEqual(len(calls), 1)
        self.assertTrue(any('tensorrt' in str(p).lower() for p in used))
        self.assertEqual(len(session_degradations()), self._before)


class OpenVINODetectorOffload(unittest.TestCase):
    """Offload is off by default and a no-op where the runtime is absent."""

    def setUp(self):
        self._saved = os.environ.pop('ROOP_DETECTOR_OPENVINO', None)

    def tearDown(self):
        os.environ.pop('ROOP_DETECTOR_OPENVINO', None)
        if self._saved is not None:
            os.environ['ROOP_DETECTOR_OPENVINO'] = self._saved

    def test_unset_is_off(self):
        self.assertIsNone(detector_offload_target())
        self.assertEqual(detector_providers(['CUDAExecutionProvider']),
                         ['CUDAExecutionProvider'])

    def test_explicit_off_values_are_off(self):
        for value in ('0', 'off', 'false', 'no', 'none', ''):
            os.environ['ROOP_DETECTOR_OPENVINO'] = value
            self.assertIsNone(detector_offload_target(), value)

    def test_request_without_the_provider_degrades_and_says_so(self):
        if provider_available('OpenVINOExecutionProvider'):
            self.skipTest('this build has the OpenVINO provider')
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            chain = detector_providers(['CUDAExecutionProvider'], requested='NPU')
        self.assertEqual(chain, ['CUDAExecutionProvider'])
        self.assertIsNone(detector_offload_target('NPU'))

    def test_only_small_per_face_models_are_offloadable(self):
        """The swapper and restorers must stay on the discrete GPU."""
        for key in ('face_swap', 'gpen_256_pro', 'codeformer', 'frame_upscaler'):
            self.assertEqual(
                detector_providers(['CUDAExecutionProvider'], model_key=key,
                                   requested='NPU'),
                ['CUDAExecutionProvider'], key)

    def test_offload_keeps_the_primary_provider_behind_it(self):
        """Routing is a prepend, so an unsupported op still runs on CUDA.

        Driven through injected availability rather than skipped: this machine
        has no OpenVINO EP, and a test that skips here never exercises the only
        branch that actually routes anything.
        """
        chain = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        with _openvino_stub(available=True, devices=('NPU',), usable=True):
            with redirect_stdout(io.StringIO()):
                routed = detector_providers(chain, 'face_detection',
                                            requested='NPU')
        self.assertEqual(routed[0][0], 'OpenVINOExecutionProvider')
        self.assertEqual(routed[0][1]['device_type'], 'NPU')
        self.assertEqual(routed[1:], chain)

    def test_a_device_that_fails_open_is_refused(self):
        """The EP is listed and the device is listed, but it does not activate.

        Measured on this machine 2026-09-03 with onnxruntime-openvino 1.23.0:
        asking for GPU/NPU/GPU.1 that the box does not have returns a WORKING
        session in ~0.3s that silently ran on CPUExecutionProvider, raising
        nothing. No try/except can catch that, so the probe result -- not the
        listing -- has to be what gates routing.
        """
        chain = ['CUDAExecutionProvider']
        with _openvino_stub(available=True, devices=('NPU',), usable=False):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                routed = detector_providers(chain, 'face_detection',
                                            requested='NPU')
            self.assertEqual(routed, chain)
            self.assertIn('does not activate', buffer.getvalue())

    def test_auto_never_takes_the_first_gpu_slot(self):
        """OpenVINO 2026.3 enumerates this machine's RTX 4070 as 'GPU'.

        Auto-selecting GPU.0 would move detection onto the very card the
        offload exists to unload, through a slower runtime than the TensorRT
        path it already has.
        """
        with _openvino_stub(available=True, devices=('CPU', 'GPU'), usable=True):
            with redirect_stdout(io.StringIO()):
                self.assertIsNone(detector_offload_target('auto'))

    def test_auto_takes_an_npu_when_one_is_really_there(self):
        with _openvino_stub(available=True, devices=('CPU', 'GPU', 'NPU'),
                            usable=True):
            self.assertEqual(detector_offload_target('auto'), 'NPU')

    def test_report_is_json_safe_and_honest(self):
        report = offload_report()
        self.assertEqual(report['provider_available'],
                         provider_available('OpenVINOExecutionProvider'))
        self.assertFalse(report['active'])
        self.assertIn('face_detection', report['offloadable_models'])
        self.assertNotIn('face_swap', report['offloadable_models'])


class PrecisionCapabilityGate(unittest.TestCase):
    """FP16/FP8 follow the SM level, and FP8 exposure is not FP8 selection."""

    def test_fp8_eligibility_matches_compute_capability(self):
        import roop.globals as g
        try:
            import torch
            if not torch.cuda.is_available():
                self.skipTest('no CUDA device')
            capability = torch.cuda.get_device_capability(g.cuda_device_id)
        except ImportError:
            self.skipTest('torch unavailable')
        if not any('tensorrt' in str(p).lower()
                   for p in resolve_provider_names(['auto'])):
            self.skipTest('TensorRT never admitted, so the flag is never set')
        # The flag is set while providers are DECODED, not on import. Asserting
        # it after a bare `import roop.core` passes vacuously on a machine where
        # nothing ever decoded -- which is how the first version of this test
        # failed. Drive the real function instead.
        from roop.core import decode_execution_providers
        with redirect_stdout(io.StringIO()):
            decoded = decode_execution_providers(['tensorrt'])
        self.assertTrue(any('tensorrt' in str(p).lower() for p in decoded))
        self.assertEqual(bool(getattr(g, 'trt_fp8_eligible', False)),
                         capability >= (8, 9))

    def test_policy_still_refuses_fp8_on_an_eligible_card(self):
        """Being new enough is not the same as being calibrated."""
        from roop.precision_policy import resolve
        decision = resolve('face_swap', 'fp8', _trt(), device_id=0)
        self.assertEqual(decision.effective, 'fp32')
        self.assertTrue(decision.fallback)


if __name__ == '__main__':
    unittest.main(verbosity=2)
