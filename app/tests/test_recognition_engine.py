"""Recognition engine: fallback chain, post-warm-up verification, preprocessing.

The fallback tests drive a FAKE onnxruntime session, because the behaviour under test is
exactly the one a real run cannot be made to produce on demand: a provider that is
accepted at construction and dropped during the first inference.
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
    import onnxruntime as ort
    from roop import recognition_engine as eng
    from roop.recognition_registry import get_model_spec
    _IMPORT_ERROR = None
except ImportError as exc:                      # light profile: no cv2 / onnxruntime
    _IMPORT_ERROR = exc


def setUpModule():
    if _IMPORT_ERROR is not None:
        raise unittest.SkipTest(f"recognition engine needs cv2/onnxruntime: {_IMPORT_ERROR}")


class _Meta:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class _FakeSession:
    """Stands in for InferenceSession. `behaviour[label]` decides what each tier does."""

    built = []
    plan = {}
    out_dim = 512

    def __init__(self, path, sess_options=None, providers=None):
        _FakeSession.built.append(providers)
        lead = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
        step = _FakeSession.plan.get(lead, "ok")
        if step == "build_raises":
            raise RuntimeError("EP could not be created\nsecond line")
        self._providers = list(p[0] if isinstance(p, tuple) else p for p in providers)
        if step == "dropped_at_construction":
            self._providers = ["CPUExecutionProvider"]
        self._drop_on_run = step == "dropped_on_run"
        self._run_raises = step == "run_raises"
        self.options = sess_options

    def get_inputs(self):
        return [_Meta("input.1", [None, 3, 112, 112])]

    def get_outputs(self):
        return [_Meta("out", [1, self.out_dim])]

    def get_providers(self):
        return list(self._providers)

    def run(self, names, feed):
        if self._run_raises:
            raise RuntimeError("engine build failed")
        if self._drop_on_run:
            self._providers = ["CPUExecutionProvider"]
        x = next(iter(feed.values()))
        assert x.shape == (1, 3, 112, 112) and x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]
        return [np.arange(1, self.out_dim + 1, dtype=np.float32).reshape(1, self.out_dim)]


_ALL = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider",
        "CPUExecutionProvider"]
_ADA = {"name": "Fake 4070", "capability": (8, 9), "total_bytes": 12 << 30,
        "free_bytes": 10 << 30, "arch": "ada"}


class _EngineCase(unittest.TestCase):
    def build(self, plan, *, model="default", device="auto", available=_ALL, gpu=_ADA, **kw):
        _FakeSession.plan, _FakeSession.built = plan, []
        _FakeSession.out_dim = get_model_spec(model).output_dim
        patches = [
            mock.patch.object(eng, "resolve_model_path", return_value=os.path.abspath("fake.onnx")),
            mock.patch.object(eng.ort, "InferenceSession", _FakeSession),
            mock.patch.object(eng.ort, "get_available_providers", return_value=list(available)),
            mock.patch.object(eng, "_gpu_info", return_value=gpu),
            mock.patch("roop.gpu_preflight.register_gpu_runtime_dirs", return_value=[]),
            mock.patch("roop.trt_shape_profile.graph_inputs", return_value=()),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        with mock.patch.object(eng.RecognitionInferenceEngine, "_trt_options",
                               return_value={"device_id": 0, "trt_fp16_enable": True}):
            return eng.RecognitionInferenceEngine(model, ".", device, **kw)


class TestFallbackChain(_EngineCase):
    def test_healthy_tensorrt_is_kept(self):
        e = self.build({})
        self.assertEqual(e.active_providers[0], "TensorrtExecutionProvider")
        self.assertFalse(e.degraded)
        self.assertEqual(len(_FakeSession.built), 1)

    def test_provider_dropped_during_first_inference_is_caught(self):
        """The case a try/except cannot see: constructs fine, loses the EP on run #1."""
        e = self.build({"TensorrtExecutionProvider": "dropped_on_run"})
        self.assertEqual(e.active_providers[0], "CUDAExecutionProvider")
        self.assertTrue(any("TensorRT requested but active providers" in m for m in e.fallback_log))
        self.assertFalse(e.degraded)

    def test_provider_dropped_at_construction_is_caught(self):
        e = self.build({"TensorrtExecutionProvider": "dropped_at_construction"})
        self.assertEqual(e.active_providers[0], "CUDAExecutionProvider")

    def test_build_and_run_exceptions_fall_through(self):
        e = self.build({"TensorrtExecutionProvider": "build_raises", "CUDAExecutionProvider": "run_raises"})
        self.assertEqual(e.active_providers, ["DmlExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(len(e.fallback_log), 2)

    def test_everything_failing_lands_on_cpu_and_says_so(self):
        e = self.build({"TensorrtExecutionProvider": "dropped_on_run",
                        "CUDAExecutionProvider": "dropped_on_run",
                        "DmlExecutionProvider": "dropped_on_run"}, device="cuda")
        self.assertEqual(e.active_providers, ["CPUExecutionProvider"])
        self.assertTrue(e.degraded)
        self.assertTrue(e.describe()["degraded"])

    def test_strict_raises_instead_of_degrading(self):
        with self.assertRaisesRegex(RuntimeError, "CPU-only"):
            self.build({"CUDAExecutionProvider": "dropped_on_run"}, device="cuda",
                       available=["CUDAExecutionProvider", "CPUExecutionProvider"], strict=True)

    def test_cpu_request_is_not_degraded(self):
        e = self.build({}, device="cpu")
        self.assertEqual(e.active_providers, ["CPUExecutionProvider"])
        self.assertFalse(e.degraded)

    def test_auto_on_a_machine_without_gpu_is_not_degraded(self):
        e = self.build({}, device="auto", available=["CPUExecutionProvider"], gpu=None)
        self.assertEqual(e.active_providers, ["CPUExecutionProvider"])
        self.assertFalse(e.degraded)

    def test_chain_only_goes_down(self):
        """A 'cuda' request never builds TensorRT; 'directml' never builds CUDA."""
        self.build({}, device="cuda")
        leads = [t[0][0] if isinstance(t[0], tuple) else t[0] for t in _FakeSession.built]
        self.assertNotIn("TensorrtExecutionProvider", leads)
        self.build({}, device="directml")
        self.assertEqual(_FakeSession.built[0][0][0], "DmlExecutionProvider")

    def test_tensorrt_not_offered_below_7gib(self):
        small = dict(_ADA, name="Fake 3060", capability=(8, 6), arch="ampere", total_bytes=6 << 30)
        e = self.build({}, device="tensorrt", gpu=small)
        self.assertEqual(e.active_providers[0], "CUDAExecutionProvider")
        self.assertTrue(any("7 GiB" in m for m in e.fallback_log))

    def test_auto_uses_tensorrt_only_on_ada(self):
        ampere = dict(_ADA, capability=(8, 6), arch="ampere")
        self.assertEqual(self.build({}, device="auto", gpu=ampere).active_providers[0], "CUDAExecutionProvider")

    def test_unknown_device_and_algo_are_rejected(self):
        with self.assertRaises(ValueError):
            self.build({}, device="tpu")
        with self.assertRaises(ValueError):
            self.build({}, cudnn_algo="FASTEST")


class TestThreadSafety(_EngineCase):
    def _hammer(self, e):
        import threading
        inside, worst = [0], [0]
        guard = threading.Lock()
        real = e.session.run

        def counting_run(*a, **k):
            with guard:
                inside[0] += 1
                worst[0] = max(worst[0], inside[0])
            try:
                import time
                time.sleep(0.002)
                return real(*a, **k)
            finally:
                with guard:
                    inside[0] -= 1
        e.session.run = counting_run
        crop = np.zeros((112, 112, 3), np.uint8)
        threads = [threading.Thread(target=lambda: [e.compute_embedding(crop) for _ in range(5)])
                   for _ in range(6)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        return worst[0]

    def test_tensorrt_session_runs_one_inference_at_a_time(self):
        e = self.build({})                                   # TensorRT-led
        self.assertEqual(self._hammer(e), 1)

    def test_cuda_session_is_not_serialised(self):
        e = self.build({}, device="cuda")
        self.assertIsNone(e._run_lock)
        self.assertGreater(self._hammer(e), 1)


class TestHardwareProfiles(_EngineCase):
    def cuda_options(self, gpu, **kw):
        e = self.build({}, device="cuda", gpu=gpu, **kw)
        return e._cuda_options()

    def test_ada_profile(self):
        o = self.cuda_options(_ADA)
        self.assertEqual((o["cudnn_conv_algo_search"], o["arena_extend_strategy"]),
                         ("EXHAUSTIVE", "kNextPowerOfTwo"))
        self.assertTrue(o["do_copy_in_default_stream"])
        self.assertEqual(o["gpu_mem_limit"], 4 << 30)                # 40% of 10 GiB, capped at 4 GiB

    def test_ampere_profile(self):
        o = self.cuda_options(dict(_ADA, capability=(8, 6), arch="ampere", total_bytes=6 << 30))
        self.assertEqual((o["cudnn_conv_algo_search"], o["arena_extend_strategy"]),
                         ("HEURISTIC", "kSameAsRequested"))
        self.assertNotIn("gpu_mem_limit", o)

    def test_algo_override_wins(self):
        self.assertEqual(self.cuda_options(_ADA, cudnn_algo="heuristic")["cudnn_conv_algo_search"], "HEURISTIC")

    def test_arena_cap_has_a_floor(self):
        o = self.cuda_options(dict(_ADA, free_bytes=1 << 30))
        self.assertEqual(o["gpu_mem_limit"], 1 << 30)

    def test_cpu_threads_are_session_options(self):
        e = self.build({}, device="cpu")
        self.assertEqual(e.session.options.inter_op_num_threads, 1)
        self.assertEqual(e.session.options.intra_op_num_threads, eng._physical_cores())

    def test_directml_disables_mem_pattern(self):
        e = self.build({}, device="directml")
        self.assertFalse(e.session.options.enable_mem_pattern)


class TestPreprocess(_EngineCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)

    def crop(self, size=112):
        return self.rng.randint(0, 256, (size, size, 3), dtype=np.uint8)

    def test_matches_opencv_blob_reference(self):
        """RGB specs == blobFromImage(swapRB=True); BGR specs == swapRB=False."""
        for model, swap in (("default", True), ("adaface", False)):
            e = self.build({}, model=model, device="cpu")
            spec = get_model_spec(model)
            img = self.crop()
            ref = cv2.dnn.blobFromImage(img, 1.0 / spec.std[0], (112, 112), spec.mean, swapRB=swap)
            got = e.preprocess(img)
            self.assertEqual((got.shape, got.dtype), ((1, 3, 112, 112), np.float32))
            self.assertTrue(got.flags["C_CONTIGUOUS"])
            np.testing.assert_allclose(got, ref, atol=1e-5, err_msg=model)

    def test_sface_is_raw_bgr(self):
        e = self.build({}, model="facerecognizersf", device="cpu")
        img = self.crop()
        np.testing.assert_array_equal(e.preprocess(img)[0], img.astype(np.float32).transpose(2, 0, 1))

    def test_wrong_size_is_resized(self):
        e = self.build({}, device="cpu")
        self.assertEqual(e.preprocess(self.crop(160)).shape, (1, 3, 112, 112))
        self.assertEqual(e.preprocess(self.crop(64)).shape, (1, 3, 112, 112))

    def test_bad_input_is_rejected(self):
        e = self.build({}, device="cpu")
        for bad in (None, np.zeros((112, 112), np.uint8), np.zeros((112, 112, 4), np.uint8)):
            with self.assertRaises(ValueError):
                e.preprocess(bad)

    def test_non_contiguous_view_is_handled(self):
        e = self.build({}, device="cpu")
        view = self.crop(224)[::2, ::2]
        self.assertFalse(view.flags["C_CONTIGUOUS"])
        self.assertTrue(e.preprocess(view).flags["C_CONTIGUOUS"])


class TestEmbedding(_EngineCase):
    def test_unit_length_and_quality(self):
        crop = np.zeros((112, 112, 3), np.uint8)
        e = self.build({}, device="cpu")
        v, q = e.compute_embedding(crop)
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)
        self.assertEqual((v.shape, v.dtype, q), ((512,), np.float32, 1.0))

    def test_quality_is_the_raw_norm_only_when_the_spec_says_so(self):
        e = self.build({}, device="cpu")
        raw_norm = float(np.linalg.norm(np.arange(1, 513, dtype=np.float32)))
        e.spec = type(e.spec)(**{**e.spec.__dict__, "extract_quality_score": True})
        self.assertAlmostEqual(e.compute_embedding(np.zeros((112, 112, 3), np.uint8))[1], raw_norm, places=2)

    def test_degenerate_output_is_flagged_not_hidden(self):
        e = self.build({}, device="cpu")
        for bad in (np.zeros((1, 512), np.float32), np.full((1, 512), np.nan, np.float32)):
            e.session.run = lambda *a, _b=bad, **k: [_b]
            v, q = e.compute_embedding(np.zeros((112, 112, 3), np.uint8))
            self.assertEqual((float(np.abs(v).sum()), q), (0.0, 0.0))
        self.assertEqual(e.invalid_outputs, 2)

    def test_output_width_must_match_the_registry(self):
        with mock.patch.object(_FakeSession, "run", lambda self, n, f: [np.ones((1, 128), np.float32)]):
            with self.assertRaisesRegex(ValueError, "output_dim"):
                self.build({}, device="cpu")


class TestRealModel(unittest.TestCase):
    """The shipped w600k_r50 on the CPU provider: the one check with no fakes."""

    MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

    def test_cpu_engine_on_the_real_model(self):
        if not os.path.isfile(os.path.join(self.MODELS, "buffalo_l", "w600k_r50.onnx")):
            self.skipTest("w600k_r50.onnx not installed")
        e = eng.RecognitionInferenceEngine("default", self.MODELS, "cpu")
        self.assertEqual(e.active_providers, ["CPUExecutionProvider"])
        rng = np.random.RandomState(1)
        a = rng.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        va, _ = e.compute_embedding(a)
        vb, _ = e.compute_embedding(a)
        np.testing.assert_array_equal(va, vb)
        self.assertAlmostEqual(float(np.linalg.norm(va)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
