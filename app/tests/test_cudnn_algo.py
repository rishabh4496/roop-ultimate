"""Guards for the per-model cuDNN conv-algorithm policy.

The policy's failure modes are all SILENT, which is what these cover:

* a suspect key that no processor actually passes never probes, so the model
  keeps the broken HEURISTIC and nothing says so;
* an error matcher that misses the real signature makes the probe record
  nothing (this happened once already -- ORT's automatic CPU-EP fallback
  replaced the cuDNN text with a graph-fusion error, so the probe silently
  gave up);
* an over-broad matcher would demote healthy models to the 2-3x slower
  DEFAULT path;
* a non-suspect model must never pay a probe or get an override.
"""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import cudnn_algo


class SuspectKeysAreReachable(unittest.TestCase):
    """Every suspect key must be a key some processor really passes."""

    def test_every_suspect_key_appears_at_a_providers_for_call_site(self):
        proc_dir = os.path.join(APP, 'roop', 'processors')
        blob = []
        for name in os.listdir(proc_dir):
            if name.endswith('.py'):
                with open(os.path.join(proc_dir, name), 'r', encoding='utf-8',
                          errors='ignore') as fh:
                    blob.append(fh.read())
        source = '\n'.join(blob)
        quoted = set(re.findall(r"['\"]([A-Za-z0-9_:+.]+)['\"]", source))
        missing = sorted(k for k in cudnn_algo.SUSPECT_MODEL_KEYS
                         if k not in quoted)
        self.assertEqual(missing, [],
                         'suspect model keys that no processor passes, so they '
                         'would never be probed: %s' % missing)


class ErrorMatcher(unittest.TestCase):

    def test_matches_the_real_frontend_failures(self):
        # Both halves of the real signature, verbatim from the RTX 3060 logs.
        planning = ("Non-zero status code returned while running Conv node. "
                    "Name:'/blocks.3/conv/Conv' Status Message: Failed to "
                    "initialize CUDNN Frontend ... CUDNN_FE failure 8: "
                    "HEURISTIC_QUERY_FAILED ; GPU=0")
        execution = ("CUDNN_FE failure 7: GRAPH_EXECUTION_FAILED ; GPU=0 ; "
                     "file=conv.cc ; line=485")
        self.assertTrue(cudnn_algo.is_cudnn_frontend_error(planning))
        self.assertTrue(cudnn_algo.is_cudnn_frontend_error(execution))

    def test_does_not_match_unrelated_failures(self):
        # An out-of-memory or a graph-fusion error must NOT demote a model to
        # the slower algo path.
        for other in ("CUDA out of memory",
                      "Attempting to get index by a name which does not exist:"
                      "InsertedPrecisionFreeCast_/ft_layers.0/norm1/Constant",
                      "TypeError: 'NoneType' object is not subscriptable"):
            self.assertFalse(cudnn_algo.is_cudnn_frontend_error(other), other)


class ApplyAlgo(unittest.TestCase):

    def test_rewrites_only_the_cuda_entry(self):
        provs = [('CUDAExecutionProvider', {'device_id': 0,
                                            'cudnn_conv_algo_search': 'HEURISTIC'}),
                 'CPUExecutionProvider']
        out = cudnn_algo.apply_algo(provs, 'DEFAULT')
        self.assertEqual(out[0][0], 'CUDAExecutionProvider')
        self.assertEqual(out[0][1]['cudnn_conv_algo_search'], 'DEFAULT')
        self.assertEqual(out[0][1]['device_id'], 0, 'other options must survive')
        self.assertEqual(out[1], 'CPUExecutionProvider')

    def test_does_not_mutate_the_caller_list(self):
        opts = {'device_id': 0, 'cudnn_conv_algo_search': 'HEURISTIC'}
        provs = [('CUDAExecutionProvider', opts)]
        cudnn_algo.apply_algo(provs, 'DEFAULT')
        self.assertEqual(opts['cudnn_conv_algo_search'], 'HEURISTIC')

    def test_none_is_a_no_op(self):
        provs = [('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'HEURISTIC'}),
                 'CPUExecutionProvider']
        self.assertEqual(cudnn_algo.apply_algo(provs, None), provs)

    def test_leaves_a_tensorrt_only_chain_alone(self):
        provs = ['TensorrtExecutionProvider', 'CPUExecutionProvider']
        self.assertEqual(cudnn_algo.apply_algo(provs, 'DEFAULT'), provs)


class ProbeScope(unittest.TestCase):

    def test_non_suspect_model_is_never_probed(self):
        # No CUDA session may be built for a model outside the suspect set --
        # the probe must return before touching onnxruntime at all.
        provs = [('CUDAExecutionProvider', {}), 'CPUExecutionProvider']
        self.assertIsNone(
            cudnn_algo.probe('gpen_256_pro', __file__, provs))

    def test_cpu_only_chain_is_never_probed(self):
        self.assertIsNone(
            cudnn_algo.probe('codeformer_fp16', __file__,
                             ['CPUExecutionProvider']))

    def test_missing_model_file_is_not_probed(self):
        # A cached verdict short-circuits ahead of the file check, and this
        # machine may legitimately have one for a real suspect key, so the
        # cache lookup is neutralised to test the path that is actually
        # under test rather than whatever this host happens to have learned.
        provs = [('CUDAExecutionProvider', {}), 'CPUExecutionProvider']
        original = cudnn_algo.known_algo
        cudnn_algo.known_algo = lambda *a, **k: None
        try:
            self.assertIsNone(
                cudnn_algo.probe('codeformer_fp16',
                                 os.path.join(APP, 'no_such_model.onnx'), provs))
        finally:
            cudnn_algo.known_algo = original

    def test_a_cached_verdict_short_circuits_the_probe(self):
        # The fast path: once learned, no session is ever built again. The
        # bogus model path proves no probe ran.
        provs = [('CUDAExecutionProvider', {}), 'CPUExecutionProvider']
        key = 'codeformer_fp16'
        original = cudnn_algo.known_algo
        cudnn_algo.known_algo = lambda k, *a, **kw: 'DEFAULT' if k == key else None
        try:
            self.assertEqual(
                cudnn_algo.probe(key, os.path.join(APP, 'no_such.onnx'), provs),
                'DEFAULT')
        finally:
            cudnn_algo.known_algo = original

    def test_has_cuda_ep_accepts_both_shapes(self):
        self.assertTrue(cudnn_algo.has_cuda_ep(['CUDAExecutionProvider']))
        self.assertTrue(cudnn_algo.has_cuda_ep([('CUDAExecutionProvider', {})]))
        self.assertFalse(cudnn_algo.has_cuda_ep(['CPUExecutionProvider']))


class CacheRoundTrip(unittest.TestCase):

    def test_recorded_verdict_is_read_back(self):
        key = '_unit_test_model_key'
        try:
            cudnn_algo.record(key, 'DEFAULT')
            self.assertEqual(cudnn_algo.known_algo(key), 'DEFAULT')
        finally:
            cudnn_algo._MEMO.pop(key, None)

    def test_unknown_model_has_no_verdict(self):
        self.assertIsNone(cudnn_algo.known_algo('_never_recorded_key_'))


if __name__ == '__main__':
    unittest.main()
