"""GPEN Realistic and RestoreFormer++ on the zero-copy ring: the real Run(), both paths, same bytes.

The processors are built around a tiny synthetic ONNX graph on the CUDA execution provider (no
180 MB weights, no TensorRT build), so what is under test is the WIRING: the GPU normalise into a
ring slot, ORT reading/writing the slot by pointer, the on-GPU finite / collapse / recolour finish,
the single uint8 download, and every fallback to the host path.

Why byte equality and not "close": the finishing steps replace cv2 calls (`convertScaleAbs`,
`cvtColor`, `meanStdDev`), and an off-by-one level there shifts every face by a level without any
numeric guard noticing. The ported ops are pinned separately in test_enhance_common_gpu.py.
"""
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import torch                                                       # noqa: E402

import roop.globals                                                # noqa: E402
import roop.processors.Enhance_GPENRealistic as GR                 # noqa: E402
import roop.processors.Enhance_RestoreFormerPPlus as RF            # noqa: E402
from roop.optimized_processor import get_zero_copy_pool, release_zero_copy_pools   # noqa: E402

CUDA = torch.cuda.is_available()
S = 512


def _model(kind):
    """input [1,3,S,S] -> output [1,3,S,S]; 'face' keeps structure, 'nan' poisons, 'flat' collapses."""
    from onnx import TensorProto, helper
    inits = [helper.make_tensor('a', TensorProto.FLOAT, [], [0.9]),
             helper.make_tensor('b', TensorProto.FLOAT, [], [0.07]),
             helper.make_tensor('z', TensorProto.FLOAT, [], [0.0])]
    nodes = {
        'face': [helper.make_node('Mul', ['input', 'a'], ['m']), helper.make_node('Add', ['m', 'b'], ['output'])],
        'nan': [helper.make_node('Sqrt', ['input'], ['output'])],                 # NaN for x < 0
        'flat': [helper.make_node('Mul', ['input', 'z'], ['output'])],
    }[kind]
    graph = helper.make_graph(
        nodes, kind,
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, S, S])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, S, S])],
        initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8
    return model.SerializeToString()


def _session(kind='face'):
    import onnxruntime as ort
    sess = ort.InferenceSession(_model(kind), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    if sess.get_providers()[0] != 'CUDAExecutionProvider':
        raise unittest.SkipTest('CUDA execution provider not active')
    return sess


def _crops():
    rng = np.random.RandomState(7)
    base = rng.randint(0, 256, (S, S, 3)).astype(np.uint8)
    smooth = np.clip(np.cumsum(np.cumsum(rng.randn(S, S, 3), 0), 1) * 0.3 + 128, 0, 255).astype(np.uint8)
    small = np.ascontiguousarray(base[:256, :256])            # 256 in -> resized to 512, scale 2... as the host does
    return [base, smooth, small]


def _gpen(kind='face'):
    p = GR.Enhance_GPENRealistic()
    p.devicename = 'cuda'
    p.size = S
    p.session = _session(kind)
    p.io_binding = p.session.io_binding()
    p.in_name, p.out_name = 'input', 'output'
    p._lut = ((np.arange(256, dtype=np.float32) / 127.5) - 1.0)
    return p


def _rf(kind='face'):
    p = RF.Enhance_RestoreFormerPPlus()
    p.devicename = 'cuda'
    p.model_restoreformerpplus = _session(kind)
    p.io_binding = p.model_restoreformerpplus.io_binding()
    p.model_inputs = p.model_restoreformerpplus.get_inputs()
    p.pool = None
    RF.Enhance_RestoreFormerPPlus._zc_faces = 0
    return p


class _Env:
    """Set an env var for a block."""
    def __init__(self, **kv):
        self.kv, self.saved = kv, {}

    def __enter__(self):
        for k, v in self.kv.items():
            self.saved[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def __exit__(self, *a):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@unittest.skipUnless(CUDA, 'CUDA required')
class TestGpenRealisticZeroCopy(unittest.TestCase):
    def tearDown(self):
        release_zero_copy_pools()

    def _run(self, p, crop, zero_copy):
        with _Env(ROOP_ZEROCOPY_ENHANCE='1' if zero_copy else '0'):
            return p.Run(None, None, crop)

    def test_the_zero_copy_path_returns_the_host_paths_bytes(self):
        p = _gpen()
        for crop in _crops():
            host, hs = self._run(p, crop, False)
            zc, zs = self._run(p, crop, True)
            self.assertEqual(hs, zs)
            self.assertTrue(np.array_equal(host, zc), 'zero-copy changed pixels')
        self.assertEqual(p._zc_faces, 3, 'the zero-copy path did not actually run')

    def test_off_means_the_host_path_and_no_ring_is_built(self):
        p = _gpen()
        self._run(p, _crops()[0], False)
        self.assertEqual(p._zc_faces, 0)
        self.assertEqual(get_zero_copy_pool(0).stats()['rings'], {})

    def test_the_host_path_still_works_after_a_zero_copy_call_on_the_same_binding(self):
        """slot.run clears the binding; a host call right after must bind its own output."""
        p = _gpen()
        crop = _crops()[0]
        ref, _ = self._run(p, crop, False)
        for _ in range(3):
            self.assertTrue(np.array_equal(self._run(p, crop, True)[0], ref))
            self.assertTrue(np.array_equal(self._run(p, crop, False)[0], ref))
        self.assertEqual(p._zc_faces, 3)

    def test_chroma_blend_keeps_the_host_path(self):
        p = _gpen()
        with _Env(ROOP_GPENR_CHROMA='0.3'):
            self._run(p, _crops()[0], True)
        self.assertEqual(p._zc_faces, 0)

    def test_non_finite_output_falls_back_to_the_unenhanced_crop_on_both_paths(self):
        p = _gpen('nan')
        crop = _crops()[0]
        host, _ = self._run(p, crop, False)
        zc, _ = self._run(p, crop, True)
        self.assertTrue(np.array_equal(host, crop))
        self.assertTrue(np.array_equal(zc, crop))
        self.assertEqual(p._zc_faces, 0, 'a refused face must not count as enhanced')

    def test_a_collapsed_output_falls_back_on_both_paths(self):
        p = _gpen('flat')
        crop = _crops()[0]
        self.assertTrue(np.array_equal(self._run(p, crop, False)[0], crop))
        self.assertTrue(np.array_equal(self._run(p, crop, True)[0], crop))
        self.assertEqual(p._zc_faces, 0)

    def test_an_error_inside_the_ring_path_disables_it_and_the_host_path_answers(self):
        p = _gpen()
        crop = _crops()[0]
        ref, _ = self._run(p, crop, False)
        with patch('roop.optimized_processor.ZeroCopySlot.run', side_effect=RuntimeError('boom')):
            out, _ = self._run(p, crop, True)
        self.assertTrue(np.array_equal(out, ref))
        self.assertFalse(p._zc_ok)
        out2, _ = self._run(p, crop, True)           # stays on the host path afterwards
        self.assertTrue(np.array_equal(out2, ref))
        self.assertEqual(p._zc_faces, 0)

    def test_a_dynamic_or_wrong_sized_graph_is_not_eligible(self):
        p = _gpen()
        p.size = 256                                  # the graph says 512
        self.assertFalse(p._zero_copy_ready())

    def test_concurrent_workers_get_their_own_pixels(self):
        import threading
        p = _gpen()
        crops = _crops()
        refs = [self._run(p, c, False)[0] for c in crops]
        errors = []

        def work(k):
            try:
                with _Env(ROOP_ZEROCOPY_ENHANCE='1'):
                    for i in range(15):
                        j = (k + i) % len(crops)
                        if not np.array_equal(p.Run(None, None, crops[j])[0], refs[j]):
                            errors.append((k, i))
            except Exception as error:
                errors.append(repr(error))

        ts = [threading.Thread(target=work, args=(k,)) for k in range(8)]
        for t in ts: t.start()
        for t in ts: t.join(120)
        self.assertFalse(any(t.is_alive() for t in ts))
        self.assertEqual(errors, [])


@unittest.skipUnless(CUDA, 'CUDA required')
class TestRestoreFormerZeroCopy(unittest.TestCase):
    def tearDown(self):
        release_zero_copy_pools()

    def _run(self, p, crop, zero_copy):
        with _Env(ROOP_ZEROCOPY_ENHANCE='1' if zero_copy else '0'):
            return p.Run(None, None, crop)

    def test_the_zero_copy_path_returns_the_host_paths_bytes(self):
        p = _rf()
        for crop in _crops():
            host, hs = self._run(p, crop, False)
            zc, zs = self._run(p, crop, True)
            self.assertEqual(hs, zs)
            self.assertTrue(np.array_equal(host, zc), 'zero-copy changed pixels')
        self.assertEqual(RF.Enhance_RestoreFormerPPlus._zc_faces, 3)

    def test_host_and_zero_copy_interleave_on_one_binding(self):
        p = _rf()
        crop = _crops()[1]
        ref, _ = self._run(p, crop, False)
        for _ in range(3):
            self.assertTrue(np.array_equal(self._run(p, crop, True)[0], ref))
            self.assertTrue(np.array_equal(self._run(p, crop, False)[0], ref))

    def test_non_finite_and_collapsed_outputs_fall_back(self):
        for kind in ('nan', 'flat'):
            p = _rf(kind)
            crop = _crops()[0]
            self.assertTrue(np.array_equal(self._run(p, crop, False)[0], crop), kind)
            self.assertTrue(np.array_equal(self._run(p, crop, True)[0], crop), kind)
            self.assertEqual(RF.Enhance_RestoreFormerPPlus._zc_faces, 0, kind)

    def test_the_native_tensorrt_engine_keeps_the_host_path(self):
        p = _rf()
        p.model_restoreformerpplus = object()          # not an onnxruntime.InferenceSession
        self.assertFalse(p._zero_copy_ready())


if __name__ == '__main__':
    unittest.main()
