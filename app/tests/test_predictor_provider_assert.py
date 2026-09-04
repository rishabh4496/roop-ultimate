"""The provider assertion must fire on a silent EP drop and nowhere else.

The defect it exists for is invisible by construction: onnxruntime returns a
working session when an execution provider fails to register, so there is no
exception to catch and no wrong output to notice -- only a session running
somewhere slower than asked.  These tests drive the check over stub sessions
that report a chosen provider list, which is the only observable ORT gives.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from roop import predictor  # noqa: E402


class StubMeta:
    def __init__(self, name, shape, type_="tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = type_


class StubSession:
    """Minimal InferenceSession surface: providers, inputs, run."""

    def __init__(self, providers, inputs=None, fail=None):
        self._providers = list(providers)
        self._inputs = inputs or []
        self._fail = fail
        self.ran = []

    def get_providers(self):
        return list(self._providers)

    def get_inputs(self):
        return list(self._inputs)

    def run(self, outputs, feed):
        if self._fail:
            raise self._fail
        self.ran.append({k: v.shape for k, v in feed.items()})
        return [np.zeros((1, 3, 8, 8), np.float32)]


TRT = "TensorrtExecutionProvider"
CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"


class ProviderAssertionTest(unittest.TestCase):

    def setUp(self):
        predictor.reset()

    def test_tensorrt_requested_and_active_passes(self):
        session = StubSession([TRT, CUDA, CPU])
        active = predictor.assert_session_providers(session, [TRT, CUDA, CPU], "m")
        self.assertEqual(active[0], TRT)

    def test_tensorrt_requested_but_dropped_raises(self):
        # The whole defect: the session works, it just is not on TensorRT.
        session = StubSession([CUDA, CPU])
        with self.assertRaises(predictor.ProviderAssertionError) as ctx:
            predictor.assert_session_providers(session, [TRT, CUDA, CPU], "swapper",
                                               strict=True)
        message = str(ctx.exception)
        self.assertIn("swapper", message)
        self.assertIn("TensorRT was REQUESTED", message)
        # The point of the message is to be actionable, so it must name what
        # actually ran and report the library search.
        self.assertIn(CUDA, message)
        self.assertIn("onnxruntime", message)

    def test_tensorrt_not_requested_never_raises(self):
        """The sub-7GB tier and the FP32-forced models must not trip this.

        `resolve_provider_names` strips TensorRT on a small card and
        `precision_policy` routes several models to CUDA on purpose; in both
        cases TensorRT is absent from the REQUESTED list, which is the input to
        this check.
        """
        session = StubSession([CPU])
        self.assertEqual(
            predictor.assert_session_providers(session, [CUDA, CPU], "m"), [CPU])
        self.assertEqual(
            predictor.assert_session_providers(session, [CPU], "m"), [CPU])

    def test_tuple_form_providers_are_recognised(self):
        """Provider options arrive as ("Name", {...}); a bare `in` test misses them.

        Every TensorRT session in this project is built with the tuple form,
        because that is how the engine cache path and precision flags are set --
        so a check that only understood plain strings would never fire at all.
        """
        requested = [(TRT, {"trt_fp16_enable": True}), (CUDA, {}), CPU]
        self.assertTrue(predictor.wants_tensorrt(requested))
        with self.assertRaises(predictor.ProviderAssertionError):
            predictor.assert_session_providers(StubSession([CUDA, CPU]), requested,
                                               "m", strict=True)

    def test_non_strict_warns_and_records_a_degradation(self):
        from roop import backend_manager
        before = len(backend_manager.session_degradations())
        active = predictor.assert_session_providers(
            StubSession([CUDA, CPU]), [TRT, CPU], "m", strict=False)
        self.assertEqual(active, [CUDA, CPU])
        self.assertGreater(len(backend_manager.session_degradations()), before)

    def test_strict_flag_reads_the_environment(self):
        with mock.patch.dict(os.environ, {"ROOP_STRICT_PROVIDER": "0"}):
            self.assertFalse(predictor.strict_enabled())
        with mock.patch.dict(os.environ, {"ROOP_STRICT_PROVIDER": "1"}):
            self.assertTrue(predictor.strict_enabled())
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ROOP_STRICT_PROVIDER", None)
            self.assertTrue(predictor.strict_enabled())  # strict by default

    def test_a_session_that_is_not_a_session_is_not_fatal(self):
        class Opaque:
            pass
        self.assertEqual(
            predictor.assert_session_providers(Opaque(), [TRT], "m"), [])


class WarmupTest(unittest.TestCase):

    def setUp(self):
        predictor.reset()

    def test_warmup_feeds_the_declared_static_shape(self):
        session = StubSession([CUDA], inputs=[StubMeta("in", [1, 3, 256, 256])])
        self.assertTrue(predictor.warmup_session(session, "gpen"))
        self.assertEqual(session.ran, [{"in": (1, 3, 256, 256)}])

    def test_symbolic_axes_resolve_to_the_pipeline_shapes(self):
        # A dynamic export declares its spatial axes as strings; guessing them
        # wrong builds a TensorRT engine for a shape the pipeline never feeds.
        session = StubSession([CUDA],
                              inputs=[StubMeta("img", ["N", 3, "H", "W"]),
                                      StubMeta("emb", ["N", 512])])
        self.assertTrue(predictor.warmup_session(session, "swap",
                                                 default_hw=(112, 256)))
        self.assertEqual(session.ran[0]["img"], (1, 3, 256, 256))
        self.assertEqual(session.ran[0]["emb"], (1, 512))

    def test_warmup_runs_once_per_tag(self):
        session = StubSession([CUDA], inputs=[StubMeta("in", [1, 3, 112, 112])])
        predictor.warmup_session(session, "once")
        predictor.warmup_session(session, "once")
        self.assertEqual(len(session.ran), 1)

    def test_a_failing_warmup_reports_and_does_not_raise(self):
        """The dummy pass surfaces a first-inference failure; it must not BE one.

        Its job is to move a TensorRT engine build off frame 0, so a model that
        cannot warm up should say so and leave the run's own error handling to
        the render.
        """
        session = StubSession([CUDA], inputs=[StubMeta("in", [1, 3, 112, 112])],
                              fail=RuntimeError("engine build failed"))
        self.assertFalse(predictor.warmup_session(session, "broken"))

    def test_dtype_follows_the_input_declaration(self):
        session = StubSession([CUDA],
                              inputs=[StubMeta("i", [1, 3, 8, 8], "tensor(float16)")])
        predictor.warmup_session(session, "half")
        self.assertTrue(predictor.warmup_session(session, "half"))

    def test_verify_and_warmup_asserts_before_it_warms(self):
        """Warming a session that silently landed on CPU proves the wrong thing."""
        session = StubSession([CPU], inputs=[StubMeta("in", [1, 3, 112, 112])])
        with self.assertRaises(predictor.ProviderAssertionError):
            predictor.verify_and_warmup(session, [TRT, CPU], "m")
        self.assertEqual(session.ran, [])

    def test_warmup_can_be_disabled(self):
        with mock.patch.dict(os.environ, {"ROOP_WARMUP": "0"}):
            self.assertFalse(predictor.warmup_enabled())
            session = StubSession([CUDA], inputs=[StubMeta("in", [1, 3, 8, 8])])
            predictor.verify_and_warmup(session, [CUDA], "m")
            self.assertEqual(session.ran, [])


class EnvironmentReportTest(unittest.TestCase):

    def test_report_is_json_safe_and_names_what_it_looked_for(self):
        report = predictor.environment_report()
        for key in ("onnxruntime", "available_providers", "found", "missing",
                    "searched_roots", "CUDA_PATH"):
            self.assertIn(key, report)
        text = predictor.format_environment(report)
        self.assertIn("onnxruntime", text)
        # Either list must be rendered; a report that silently showed neither
        # would make the raised error useless.
        self.assertTrue("MISSING libraries" in text or "resolved libraries" in text)

    def test_ort_package_directory_is_searched(self):
        """ORT's own EP libraries are NOT on PATH on Windows.

        They are found through the DLL directories ORT adds at import time, so a
        probe that only walked PATH would report the TensorRT EP missing on a
        perfectly working install and turn this diagnostic into a liar.
        """
        try:
            import onnxruntime
        except Exception:
            self.skipTest("onnxruntime unavailable")
        roots = [r.lower() for r in predictor._search_roots()]
        ort_dir = os.path.dirname(os.path.abspath(onnxruntime.__file__)).lower()
        self.assertIn(os.path.normpath(ort_dir), roots)


if __name__ == "__main__":
    unittest.main()
