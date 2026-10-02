"""Startup canary for TensorRT swapper engines (roop/swap_canary.py).

The failure it exists for: an inswapper TensorRT FP16/"mixed" engine that builds,
warms up, reports the TensorRT provider and runs at full speed, yet emits the
wrong picture for every face (measured: SSIM 0.75 / identity 0.762 -> 0.087 when
one build option changed).  Nothing in the build path can see that, so these tests
pin the three things that make the guard trustworthy:

  * it FAILS a corrupt engine (non-finite output, or output far from the reference),
  * it PASSES an engine that matches, including the real engine's small FP16 noise,
  * it never turns "could not check" into a failure, and never runs where nothing
    can drift (FP32 engine, CUDA, CPU, no CUDA reference).

No GPU is needed: sessions are fakes with the ORT surface the canary touches.
"""
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from roop import swap_canary                                       # noqa: E402
from roop.processors import FaceSwapInsightFace as fsi             # noqa: E402


class _Meta:
    def __init__(self, name, shape, type_="tensor(float)"):
        self.name, self.shape, self.type = name, shape, type_


class FakeSession:
    """The surface the canary uses: get_inputs() and run()."""

    def __init__(self, fn, inputs=None):
        self._fn = fn
        self._inputs = inputs or [_Meta("target", [1, 3, 128, 128]), _Meta("source", [1, 512])]
        self.calls = 0

    def get_inputs(self):
        return self._inputs

    def run(self, _outputs, feed):
        self.calls += 1
        return [self._fn(feed)]

    def get_providers(self):
        return ["TensorrtExecutionProvider", "CPUExecutionProvider"]


def _reference_fn(feed):
    """A deterministic stand-in 'swapper': a smooth function of both inputs."""
    img = feed["target"].astype(np.float32)
    lat = feed["source"].astype(np.float32)
    return np.clip(0.8 * img + 0.2 * float(np.tanh(lat.sum())), 0.0, 1.0)


_EMAP = np.random.RandomState(7).randn(512, 512).astype(np.float32)


class TestSsim(unittest.TestCase):
    def test_agrees_with_skimage(self):
        try:
            from skimage.metrics import structural_similarity
        except Exception:
            self.skipTest("scikit-image not installed")
        rng = np.random.RandomState(1)
        a = rng.rand(96, 96, 3) * 255
        for noise in (0.0, 4.0, 40.0):
            b = np.clip(a + rng.randn(*a.shape) * noise, 0, 255)
            want = structural_similarity(a, b, channel_axis=2, data_range=255)
            got = swap_canary.ssim(a, b, 255.0)
            self.assertAlmostEqual(got, want, delta=2e-3, msg=f"noise={noise}")

    def test_identical_is_one_and_different_is_low(self):
        rng = np.random.RandomState(2)
        a = rng.rand(64, 64, 3) * 255
        self.assertAlmostEqual(swap_canary.ssim(a, a, 255.0), 1.0, places=6)
        self.assertLess(swap_canary.ssim(a, rng.rand(64, 64, 3) * 255, 255.0), 0.2)


class TestProviderPredicates(unittest.TestCase):
    def test_trt_fp16_active(self):
        trt = lambda v: ("TensorrtExecutionProvider", {"trt_fp16_enable": v})
        self.assertTrue(swap_canary.trt_fp16_active([trt(True), "CUDAExecutionProvider"]))
        self.assertTrue(swap_canary.trt_fp16_active([trt("True")]))
        self.assertFalse(swap_canary.trt_fp16_active([trt(False)]))
        self.assertFalse(swap_canary.trt_fp16_active([trt("False")]))
        self.assertFalse(swap_canary.trt_fp16_active(["CUDAExecutionProvider", "CPUExecutionProvider"]))
        self.assertFalse(swap_canary.trt_fp16_active(None))
        self.assertFalse(swap_canary.trt_fp16_active([("TensorrtExecutionProvider", {})]))

    def test_non_trt_providers_keeps_order_and_options(self):
        cuda = ("CUDAExecutionProvider", {"device_id": 0})
        got = swap_canary.non_trt_providers([("TensorrtExecutionProvider", {}), cuda, "CPUExecutionProvider"])
        self.assertEqual(got, [cuda, "CPUExecutionProvider"])

    def test_env_switch(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(swap_canary.CANARY_ENV, None)
            self.assertTrue(swap_canary.enabled())
            os.environ[swap_canary.CANARY_ENV] = "0"
            self.assertFalse(swap_canary.enabled())
            os.environ[swap_canary.CANARY_ENV] = "1"
            self.assertTrue(swap_canary.enabled())


class TestBuildFeeds(unittest.TestCase):
    def test_inswapper_shaped_session(self):
        feeds = swap_canary.build_feeds(FakeSession(_reference_fn), emap=_EMAP)
        self.assertEqual(len(feeds), 2)
        for feed in feeds:
            self.assertEqual(feed["target"].shape, (1, 3, 128, 128))
            self.assertEqual(feed["source"].shape, (1, 512))
            self.assertEqual(feed["target"].dtype, np.float32)
            self.assertAlmostEqual(float(np.linalg.norm(feed["source"])), 1.0, places=4)
            self.assertGreaterEqual(float(feed["target"].min()), 0.0)
            self.assertLessEqual(float(feed["target"].max()), 1.0)

    def test_deterministic_and_two_distinct_cases(self):
        a = swap_canary.build_feeds(FakeSession(_reference_fn), emap=_EMAP)
        b = swap_canary.build_feeds(FakeSession(_reference_fn), emap=_EMAP)
        np.testing.assert_array_equal(a[0]["target"], b[0]["target"])
        np.testing.assert_array_equal(a[1]["source"], b[1]["source"])
        self.assertFalse(np.array_equal(a[0]["target"], a[1]["target"]))

    def test_half_precision_inputs_follow_the_model_dtype(self):
        s = FakeSession(_reference_fn, [_Meta("target", [1, 3, 256, 256], "tensor(float16)"),
                                        _Meta("source", [1, 512], "tensor(float16)")])
        feed = swap_canary.build_feeds(s)[0]
        self.assertEqual(feed["target"].dtype, np.float16)
        self.assertEqual(feed["source"].dtype, np.float16)
        self.assertEqual(feed["target"].shape, (1, 3, 256, 256))

    def test_unrecognised_inputs_are_not_guessed(self):
        self.assertIsNone(swap_canary.build_feeds(FakeSession(
            _reference_fn, [_Meta("x", [1, 7, 9, 9, 3])])))
        self.assertIsNone(swap_canary.build_feeds(FakeSession(
            _reference_fn, [_Meta("a", [1, 3, 64, 64]), _Meta("b", [1, 3, 64, 64])])))


class TestCheckEngine(unittest.TestCase):
    def _check(self, engine_fn, reference_factory=None):
        engine = FakeSession(engine_fn)
        factory = reference_factory or (lambda: FakeSession(_reference_fn))
        return swap_canary.check_engine(engine, factory, "swapper:test", emap=_EMAP), engine

    def test_matching_engine_passes(self):
        res, engine = self._check(_reference_fn)
        self.assertTrue(res.passed, res.reason)
        self.assertGreater(res.min_ssim, 0.999)
        self.assertEqual(engine.calls, 2)               # two canary cases

    def test_small_half_precision_noise_still_passes(self):
        """The real mixed engine scores 0.996-0.9975 (not 1.0); that must pass."""
        def noisy(feed):
            out = _reference_fn(feed)
            rng = np.random.RandomState(3)
            return np.clip(out + rng.randn(*out.shape).astype(np.float32) * 0.004, 0, 1)
        res, _ = self._check(noisy)
        self.assertTrue(res.passed, f"{res.reason} {res.per_case}")

    def test_non_finite_output_fails(self):
        res, _ = self._check(lambda f: np.full((1, 3, 128, 128), np.nan, np.float32))
        self.assertTrue(res.failed)
        self.assertEqual(res.min_ssim, -1.0)

    def test_wrong_picture_fails(self):
        """The measured corrupt engine: finite, plausible, and the wrong image."""
        def wrong(feed):
            rng = np.random.RandomState(5)
            return rng.rand(1, 3, 128, 128).astype(np.float32)
        res, _ = self._check(wrong)
        self.assertTrue(res.failed)
        self.assertLess(res.min_ssim, swap_canary.SSIM_FLOOR)

    def test_one_bad_case_is_enough(self):
        """Two input statistics so a lucky pattern cannot hide an overflow."""
        state = {"n": 0}

        def second_case_broken(feed):
            state["n"] += 1
            if state["n"] == 2:
                return np.full((1, 3, 128, 128), np.inf, np.float32)
            return _reference_fn(feed)
        res, _ = self._check(second_case_broken)
        self.assertTrue(res.failed)

    def test_reference_that_cannot_be_built_is_a_skip_not_a_failure(self):
        def boom():
            raise RuntimeError("no CUDA reference available")
        res, _ = self._check(_reference_fn, boom)
        self.assertIsNone(res.passed)
        self.assertFalse(res.failed)
        self.assertFalse(res.ran)

    def test_unknown_inputs_skip(self):
        engine = FakeSession(_reference_fn, [_Meta("x", [1, 5])])
        res = swap_canary.check_engine(engine, lambda: FakeSession(_reference_fn), "t")
        self.assertIsNone(res.passed)


class TestForceFp32Providers(unittest.TestCase):
    def test_switches_only_trt_and_isolates_the_cache(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cuda = ("CUDAExecutionProvider", {"device_id": 0})
            src = [("TensorrtExecutionProvider", {"trt_fp16_enable": True,
                                                 "trt_engine_cache_path": os.path.join(d, "mixed_x"),
                                                 "trt_layer_norm_fp32_fallback": True}),
                   cuda, "CPUExecutionProvider"]
            out = fsi._force_fp32_providers(src)
            self.assertFalse(out[0][1]["trt_fp16_enable"])
            self.assertTrue(out[0][1]["trt_engine_cache_path"].endswith("_swap_fp32"))
            self.assertTrue(out[0][1]["trt_layer_norm_fp32_fallback"])
            self.assertEqual(out[1:], [cuda, "CPUExecutionProvider"])
            self.assertTrue(src[0][1]["trt_fp16_enable"], "input list must not be mutated")
            again = fsi._force_fp32_providers(out)         # idempotent: no double suffix
            self.assertEqual(again[0][1]["trt_engine_cache_path"],
                             out[0][1]["trt_engine_cache_path"])


class TestCanaryGate(unittest.TestCase):
    """FaceSwapInsightFace._canary_gate: pass / fail->FP32 / fail->CUDA fallbacks."""

    MIXED = [("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
             ("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]

    def _swapper(self, providers):
        s = fsi.FaceSwapInsightFace.__new__(fsi.FaceSwapInsightFace)
        s._swap_providers = list(providers)
        s.embedding_mode = "normed_emap"
        s.emap = _EMAP
        s._model_arg = "inswapper_128.onnx"
        s._trt_disabled = False
        s.pool = None
        s._io_bindings = {}
        s.loaded_model_key = "inswapper"
        s.model_swap_insightface = FakeSession(_reference_fn)
        return s

    def _run(self, swapper, engine_fn, fp32_builder=None):
        """Drive the gate with an `onnxruntime.InferenceSession` that hands out
        the corrupt engine's reference (CUDA) and, on request, an FP32 session."""
        built = []

        def factory(model, options=None, providers=None, **_):
            built.append(list(providers))
            tr = any("tensorrt" in str(p[0] if isinstance(p, tuple) else p).lower() for p in providers)
            if tr and fp32_builder is not None:
                return fp32_builder(providers)
            return FakeSession(_reference_fn)

        swapper.model_swap_insightface = FakeSession(engine_fn)
        with mock.patch.object(fsi.onnxruntime, "InferenceSession", side_effect=factory), \
             mock.patch.object(fsi, "get_onnx_session_options", return_value=None), \
             mock.patch("roop.predictor.verify_and_warmup", return_value=[]):
            swapper._canary_gate("inswapper", {"output_size": 128})
        return built

    def test_good_engine_is_left_alone(self):
        s = self._swapper(self.MIXED)
        built = self._run(s, _reference_fn)
        self.assertTrue(s.swap_canary.passed)
        self.assertIs(s._swap_providers[0][1]["trt_fp16_enable"], True)
        self.assertEqual(len(built), 1, "only the CUDA reference may be built")
        self.assertNotIn("tensorrt", str(built[0]).lower())

    def test_corrupt_engine_is_rebuilt_on_trt_fp32(self):
        s = self._swapper(self.MIXED)
        bad = lambda f: np.random.RandomState(9).rand(1, 3, 128, 128).astype(np.float32)
        built = self._run(s, bad, fp32_builder=lambda p: FakeSession(_reference_fn))
        self.assertTrue(s.swap_canary.passed)
        self.assertIs(s._swap_providers[0][1]["trt_fp16_enable"], False)
        self.assertFalse(s._trt_disabled, "FP32 TensorRT is still TensorRT")
        self.assertTrue(any("tensorrt" in str(p).lower() for b in built for p in b))

    def test_fp32_rebuild_failure_falls_back_to_cuda(self):
        s = self._swapper(self.MIXED)
        bad = lambda f: np.full((1, 3, 128, 128), np.nan, np.float32)

        def broken(_providers):
            raise RuntimeError("engine build failed")
        self._run(s, bad, fp32_builder=broken)
        self.assertTrue(s._trt_disabled)

    def test_skipped_when_nothing_can_drift(self):
        for providers in ([("TensorrtExecutionProvider", {"trt_fp16_enable": False}), "CUDAExecutionProvider"],
                          ["CUDAExecutionProvider", "CPUExecutionProvider"],
                          [("TensorrtExecutionProvider", {"trt_fp16_enable": True}), "CPUExecutionProvider"]):
            s = self._swapper(providers)
            built = self._run(s, lambda f: np.zeros((1, 3, 128, 128), np.float32))
            self.assertEqual(built, [], f"no session may be built for {providers}")
            self.assertIsNone(s.swap_canary)

    def test_disabled_by_env(self):
        s = self._swapper(self.MIXED)
        with mock.patch.dict(os.environ, {swap_canary.CANARY_ENV: "0"}):
            built = self._run(s, lambda f: np.zeros((1, 3, 128, 128), np.float32))
        self.assertEqual(built, [])


if __name__ == "__main__":
    unittest.main()
