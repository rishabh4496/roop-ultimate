"""Opt-in TensorRT bound/graph sessions and the build-heuristics switch: both must be INERT unless asked.

The A/B levers added 2026-10-06/09 are env flags (ROOP_TRT_BOUND, ROOP_TRT_BUILD_HEURISTICS). The property
worth pinning is the one this repo keeps losing: "defaults unchanged". Unset must give the shipped provider
options and the shipped engine-cache namespace byte for byte, and the pure helpers must never mutate the
provider list the app built.
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from roop.trt_bound_session import bound_mode, with_stream_and_graph      # noqa: E402


class _Env:
    def __init__(self, **kv):
        self.kv, self.old = kv, {}

    def __enter__(self):
        for k, v in self.kv.items():
            self.old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def __exit__(self, *_):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class BoundModeTests(unittest.TestCase):
    def test_unset_is_none(self):
        with _Env(ROOP_TRT_BOUND=None):
            self.assertIsNone(bound_mode('xseg'))

    def test_parses_a_list_and_is_per_model(self):
        with _Env(ROOP_TRT_BOUND='xseg=graph, rfpp = bound'):
            self.assertEqual(bound_mode('xseg'), 'graph')
            self.assertEqual(bound_mode('rfpp'), 'bound')
            self.assertIsNone(bound_mode('det'))

    def test_garbage_and_unknown_modes_are_off(self):
        for raw in ('', 'xseg', 'xseg=', 'xseg=fast', 'graph', '=graph', 'xseg:graph'):
            with _Env(ROOP_TRT_BOUND=raw):
                self.assertIsNone(bound_mode('xseg'), raw)


class ProviderChainTests(unittest.TestCase):
    TRT = {'trt_fp16_enable': True, 'trt_engine_cache_enable': True}

    def chain(self):
        return [('TensorrtExecutionProvider', dict(self.TRT)),
                ('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']

    def test_input_is_never_mutated(self):
        src = self.chain()
        with_stream_and_graph(src, 1234, True)
        self.assertEqual(src, self.chain())

    def test_tensorrt_gets_stream_and_a_real_bool_graph_flag(self):
        out = with_stream_and_graph(self.chain(), 1234, True)
        opts = out[0][1]
        self.assertEqual(opts['user_compute_stream'], '1234')
        # ORT 1.23 rejects '1'/'0' for TRT bool options and silently drops TRT AND CUDA to CPU
        self.assertIs(opts['trt_cuda_graph_enable'], True)
        self.assertNotIn('has_user_compute_stream', opts)         # a CUDA-EP key; TRT rejects it
        self.assertIs(with_stream_and_graph(self.chain(), 1, False)[0][1]['trt_cuda_graph_enable'], False)

    def test_production_tensorrt_options_survive(self):
        opts = with_stream_and_graph(self.chain(), 7, False)[0][1]
        for k, v in self.TRT.items():
            self.assertEqual(opts[k], v)

    def test_cuda_ep_gets_the_stream_with_its_own_key(self):
        cuda = with_stream_and_graph(self.chain(), 55, False)[1][1]
        self.assertEqual(cuda['has_user_compute_stream'], '1')
        self.assertEqual(cuda['user_compute_stream'], '55')
        self.assertEqual(cuda['device_id'], 0)

    def test_cpu_entry_is_left_alone(self):
        self.assertEqual(with_stream_and_graph(self.chain(), 7, True)[2], 'CPUExecutionProvider')


class BuildHeuristicsFlagTests(unittest.TestCase):
    """Drive the real decode: unset must equal the shipped options AND namespace."""

    def decode(self, value):
        import roop.globals as g
        try:
            import torch
            if not torch.cuda.is_available():
                self.skipTest('no CUDA device')
        except ImportError:
            self.skipTest('torch unavailable')
        from roop.backend_manager import resolve_provider_names
        if not any('tensorrt' in str(p).lower() for p in resolve_provider_names(['auto'])):
            self.skipTest('TensorRT never admitted')
        from roop.core import decode_execution_providers
        if getattr(g, 'CFG', None) is not None and getattr(g.CFG, 'trt_precision', 'mixed') != 'mixed':
            self.skipTest('config is not trt_precision: mixed')
        with _Env(ROOP_TRT_BUILD_HEURISTICS=value), redirect_stdout(io.StringIO()):
            decoded = decode_execution_providers(['tensorrt'])
        trt = next(p for p in decoded if 'tensorrt' in str(p[0] if isinstance(p, tuple) else p).lower())
        return trt[1]

    def test_unset_is_the_shipped_heuristics_on_mixed_namespace(self):
        o = self.decode(None)
        self.assertTrue(o['trt_build_heuristics_enable'])
        self.assertIn('_lnfp32_seq_heur_', o['trt_engine_cache_path'])

    def test_off_gets_its_own_namespace_and_never_the_shipped_one(self):
        shipped = self.decode(None)['trt_engine_cache_path']
        off = self.decode('0')
        self.assertFalse(off['trt_build_heuristics_enable'])
        self.assertNotEqual(off['trt_engine_cache_path'], shipped)
        self.assertNotIn('_heur_', off['trt_engine_cache_path'])

    def test_explicit_on_equals_the_shipped_namespace(self):
        self.assertEqual(self.decode('1')['trt_engine_cache_path'], self.decode(None)['trt_engine_cache_path'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
