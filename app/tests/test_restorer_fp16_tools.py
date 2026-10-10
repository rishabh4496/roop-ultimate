"""tools/restorer_fp16_ranking.py and app/roop/trt_native_runner.py - the parts that need no GPU.

The ranking tool simulates FP16 on a CPU graph by routing a module's weights, inputs and outputs through Cast(fp16) ->
Cast(fp32). The failure to guard is the quiet one: an injection that does nothing (ORT folds the casts away, or the
selection matches no node) would rank every module as harmless. So these tests run a tiny real graph and pin that the
injection (a) changes the result, by an amount on the order of fp16 rounding, (b) saturates past 65504 like a real FP16
tensor, (c) leaves the source model alone, and (d) with an empty selection is a no-op.
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'tools'))

import onnx                                                     # noqa: E402
import onnxruntime as ort                                       # noqa: E402
from onnx import TensorProto, helper, numpy_helper              # noqa: E402

import restorer_fp16_ranking as rk                              # noqa: E402


def tiny_model(scale=1.0):
    """input -> /enc/conv/Conv -> /enc/act/Sigmoid -> /dec/head/Mul(x, 3) -> output: a stand-in for an encoder / decoder pair."""
    rng = np.random.RandomState(0)
    w = (rng.randn(4, 3, 3, 3) * scale).astype(np.float32)
    c = np.full((1, 4, 1, 1), 3.0, np.float32)
    nodes = [
        helper.make_node('Conv', ['input', 'w'], ['conv_out'], name='/enc/conv/Conv', pads=[1, 1, 1, 1]),
        helper.make_node('Sigmoid', ['conv_out'], ['sig_out'], name='/enc/act/Sigmoid'),
        helper.make_node('Mul', ['sig_out', 'c'], ['output'], name='/dec/head/Mul'),
    ]
    graph = helper.make_graph(
        nodes, 'tiny', [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 16, 16])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4, 16, 16])],
        initializer=[numpy_helper.from_array(w, 'w'), numpy_helper.from_array(c, 'c')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def dtypes_of(model):
    inferred = onnx.shape_inference.infer_shapes(model)
    d = {v.name: v.type.tensor_type.elem_type
         for v in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output)}
    d.update({i.name: i.data_type for i in model.graph.initializer})
    return d


def run(model, x):
    s = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    return s.run(None, {'input': x})[0]


class ModuleOf(unittest.TestCase):
    def test_parent_path(self):
        self.assertEqual(rk.module_of('/encoder/block.0/norm1/InstanceNormalization'), '/encoder/block.0/norm1')
        self.assertEqual(rk.module_of('/encoder/conv_in/Conv'), '/encoder/conv_in')
        self.assertEqual(rk.module_of('/quantize/Add_1'), '/quantize')

    def test_a_flat_name_is_its_own_module(self):
        self.assertEqual(rk.module_of('Conv_12'), 'Conv_12')


class Injection(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.dtypes = dtypes_of(self.model)
        self.x = np.random.RandomState(1).uniform(-1, 1, (1, 3, 16, 16)).astype(np.float32)
        self.ref = run(self.model, self.x)

    def indices(self, prefix):
        return {i for i, n in enumerate(self.model.graph.node) if n.name.startswith(prefix)}

    def test_empty_selection_is_exactly_the_original(self):
        np.testing.assert_array_equal(run(rk.inject(self.model, self.dtypes, set()), self.x), self.ref)

    def test_the_source_model_is_not_modified(self):
        before = self.model.SerializeToString()
        rk.inject(self.model, self.dtypes, self.indices('/enc'))
        self.assertEqual(self.model.SerializeToString(), before)

    def test_a_selected_module_changes_the_result_by_about_fp16_rounding(self):
        got = run(rk.inject(self.model, self.dtypes, self.indices('/enc')), self.x)
        err = float(np.abs(got - self.ref).max())
        self.assertGreater(err, 0.0, 'the injection did nothing: the Cast pair was folded away or nothing was selected')
        self.assertLess(err, 5e-3)          # fp16 has ~1e-3 relative precision; the output is a scale-3 sigmoid

    def test_only_the_head_selected_gives_a_rounded_copy_of_the_exact_output(self):
        got = run(rk.inject(self.model, self.dtypes, self.indices('/dec')), self.x)
        want = self.ref.astype(np.float16).astype(np.float32)
        self.assertLess(float(np.abs(got - want).max()), 2e-3)

    def test_values_past_the_fp16_range_saturate_like_a_real_fp16_tensor(self):
        # A Conv-only graph whose outputs reach ~1e5 (a sigmoid after it would hide the overflow: sigmoid(inf) is exactly 1).
        w = (np.random.RandomState(0).randn(4, 3, 3, 3) * 12000.0).astype(np.float32)
        graph = helper.make_graph(
            [helper.make_node('Conv', ['input', 'w'], ['output'], name='/enc/conv/Conv', pads=[1, 1, 1, 1])], 'big',
            [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 16, 16])],
            [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4, 16, 16])],
            initializer=[numpy_helper.from_array(w, 'w')])
        big = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
        big.ir_version = 8
        x = np.random.RandomState(2).uniform(0.5, 1.0, (1, 3, 16, 16)).astype(np.float32)
        ref = run(big, x)
        self.assertTrue(np.isfinite(ref).all())
        self.assertGreater(float(np.abs(ref).max()), 65504.0, 'the fixture must overflow fp16')
        sim = run(rk.inject(big, dtypes_of(big), {0}), x)
        self.assertTrue(np.isinf(sim).any(), 'a >65504 activation went through the fp16 round trip unchanged: no saturation simulated')

    def test_non_float_tensors_are_never_cast(self):
        d = dict(self.dtypes)
        d['conv_out'] = TensorProto.INT64   # pretend it is an index tensor: it must be skipped
        for n in rk.inject(self.model, d, {0}).graph.node:
            if n.op_type == 'Cast':
                self.assertNotIn('conv_out', list(n.input))


class Metrics(unittest.TestCase):
    def test_identical_outputs_score_perfectly(self):
        y = np.random.RandomState(0).uniform(-1, 1, (3, 3, 64, 64)).astype(np.float32)
        m = rk.metrics(list(y), list(y))
        self.assertAlmostEqual(m['ssim_mean'], 1.0, places=6)
        self.assertEqual(m['mae_levels'], 0.0)

    def test_a_perturbation_lowers_ssim_and_is_measured_in_8bit_levels(self):
        y = np.random.RandomState(0).uniform(-1, 1, (2, 3, 64, 64)).astype(np.float32)
        m = rk.metrics(list(y + 0.02), list(y))
        self.assertAlmostEqual(m['mae_levels'], 0.02 * 127.5, places=3)
        self.assertLess(m['ssim_mean'], 1.0)


class NativeSessionShim(unittest.TestCase):
    """The ORT-shaped wrapper the restorer processors use for a native engine (no GPU: the engine is a fake)."""

    def test_shim_speaks_the_slice_of_ort_the_processor_uses(self):
        from roop import trt_native_runner as tnr

        class FakeEngine:
            in_shape, out_shape = (1, 3, 8, 8), (1, 3, 8, 8)

            def __init__(self, *a, **k):
                pass

            def run(self, arr):
                return arr * 2.0
        orig = tnr.NativeEngine
        tnr.NativeEngine = FakeEngine
        try:
            s = tnr.NativeSession('x.engine')
            self.assertEqual(s.get_providers(), ['TensorrtNativeEngine'])
            self.assertEqual(s.get_inputs()[0].name, 'input')
            self.assertEqual(s.get_inputs()[0].shape, [1, 3, 8, 8])
            iob = s.io_binding()
            iob.bind_output(s.get_outputs()[0].name, 'cuda')
            x = np.random.RandomState(0).rand(1, 3, 8, 8).astype(np.float32)
            iob.bind_cpu_input('input', x)
            s.run_with_iobinding(iob)
            np.testing.assert_allclose(iob.copy_outputs_to_cpu()[0], x * 2.0)
        finally:
            tnr.NativeEngine = orig


if __name__ == '__main__':
    unittest.main()
