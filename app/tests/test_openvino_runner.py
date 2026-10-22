"""roop/openvino_runner.py: device policy, ORT-surface parity on the REAL SCRFD / ArcFace / XSeg graphs,
concurrent-caller isolation, error recovery and the INT8 fail-closed rule.

The parity tests need the ``openvino`` package, which the app venv does not ship (the offload is opt-in and
neither validated machine has an Intel accelerator). They run on the OpenVINO ``CPU`` device - a correctness
target, not a speed one - and SKIP, loudly, where the package is absent:

    <scratch venv with openvino + nncf + onnxruntime>\\Scripts\\python -m pytest app/tests/test_openvino_runner.py

The policy tests (no package needed) always run.
"""
from __future__ import annotations

import json
import os
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from roop import openvino_runner as ovr

_MODELS = Path(__file__).resolve().parents[2] / ".cache" / "models"
_HAVE_OV = ovr.available()
try:
    import onnxruntime as _ort
except Exception:                                      # noqa: BLE001
    _ort = None


def _ref_session(name: str):
    return _ort.InferenceSession(str(_MODELS / f"{name}.onnx"), providers=["CPUExecutionProvider"])


needs_ov = unittest.skipUnless(_HAVE_OV and _ort is not None,
                               "openvino / onnxruntime not importable in this interpreter - run in the OpenVINO scratch venv")


def needs_model(name: str):
    return unittest.skipUnless((_MODELS / f"{name}.onnx").is_file(), f"{name}.onnx not in {_MODELS}")


class DevicePolicy(unittest.TestCase):
    def setUp(self):
        ovr._reset_for_tests()
        self.addCleanup(ovr._reset_for_tests)

    def test_unset_and_off_values_are_off(self):
        for v in (None, "", "0", "off", "False", "none"):
            with mock.patch.dict(os.environ, {ovr.ENV_DEVICE: v or ""}):
                self.assertIsNone(ovr.resolve_device(), v)

    def test_missing_package_is_an_honest_noop(self):
        with mock.patch.object(ovr, "available", return_value=False):
            self.assertIsNone(ovr.resolve_device("auto"))
            self.assertIsNone(ovr.maybe_session("xseg", "x.onnx", requested="auto"))

    def test_auto_never_takes_the_discrete_nvidia_gpu(self):
        names = {"CPU": "Intel i9", "GPU": "NVIDIA GeForce RTX 4070 (dGPU)"}
        with mock.patch.object(ovr, "available", return_value=True), \
                mock.patch.object(ovr, "devices", return_value=("CPU", "GPU")), \
                mock.patch.object(ovr, "device_name", side_effect=lambda d: names.get(d, "")):
            self.assertIsNone(ovr.resolve_device("auto"))
            # an explicit request is the user's call and is honoured
            self.assertEqual(ovr.resolve_device("GPU"), "GPU")

    def test_auto_prefers_npu_then_intel_gpu(self):
        names = {"GPU.0": "Intel(R) Arc(TM) Graphics (iGPU)", "NPU": "Intel(R) AI Boost"}
        with mock.patch.object(ovr, "available", return_value=True), \
                mock.patch.object(ovr, "devices", return_value=("CPU", "GPU.0", "NPU")), \
                mock.patch.object(ovr, "device_name", side_effect=lambda d: names.get(d, "")):
            self.assertEqual(ovr.resolve_device("auto"), "NPU")
        with mock.patch.object(ovr, "available", return_value=True), \
                mock.patch.object(ovr, "devices", return_value=("CPU", "GPU.0")), \
                mock.patch.object(ovr, "device_name", side_effect=lambda d: names.get(d, "")):
            self.assertEqual(ovr.resolve_device("auto"), "GPU.0")

    def test_unlisted_device_is_refused(self):
        with mock.patch.object(ovr, "available", return_value=True), \
                mock.patch.object(ovr, "devices", return_value=("CPU",)):
            self.assertIsNone(ovr.resolve_device("NPU"))

    def test_only_the_small_graphs_are_offloadable(self):
        with mock.patch.object(ovr, "resolve_device", return_value="NPU"):
            for key in ("inswapper", "hyperswap", "gpen", "restoreformer"):
                self.assertIsNone(ovr.maybe_session(key, "x.onnx"), key)

    def test_a_build_failure_returns_none_instead_of_raising(self):
        with mock.patch.object(ovr, "resolve_device", return_value="CPU"), \
                mock.patch.object(ovr, "OpenVinoAsyncSession", side_effect=RuntimeError("boom")):
            self.assertIsNone(ovr.maybe_session("xseg", "x.onnx"))


@needs_ov
class RealModelParity(unittest.TestCase):
    """Same inputs through ORT-CPU and through OpenVinoAsyncSession on the OpenVINO CPU device."""

    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(7)

    def _pair(self, name):
        ref = _ref_session(name)
        ov = ovr.OpenVinoAsyncSession(str(_MODELS / f"{name}.onnx"), "CPU", precision="fp32", model_key=name)
        return ref, ov

    @needs_model("xseg")
    def test_xseg_matches_ort_and_exposes_the_ort_surface(self):
        ref, ov = self._pair("xseg")
        self.assertEqual([i.name for i in ov.get_inputs()], [i.name for i in ref.get_inputs()])
        self.assertEqual([o.name for o in ov.get_outputs()], [o.name for o in ref.get_outputs()])
        self.assertEqual(ov.get_providers(), ["OpenVINOExecutionProvider"])
        x = self.rng.random((1, 256, 256, 3), dtype=np.float32)
        feed = {ref.get_inputs()[0].name: x}
        a, b = ref.run(None, feed)[0], ov.run(None, feed)[0]
        self.assertEqual(a.shape, b.shape)
        self.assertLess(float(np.abs(a - b).max()), 2e-3)

    @needs_model("arcface_w600k_r50")
    def test_arcface_matches_ort_across_batch_sizes(self):
        ref, ov = self._pair("arcface_w600k_r50")
        name = ref.get_inputs()[0].name
        for n in (1, 3):
            x = self.rng.standard_normal((n, 3, 112, 112)).astype(np.float32)
            a, b = ref.run(None, {name: x})[0], ov.run(None, {name: x})[0]
            self.assertEqual(a.shape, b.shape)
            cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
            self.assertGreater(float(cos.min()), 0.9999)
        self.assertEqual(len(ov._compiled), 2, "one compiled model per distinct batch shape")

    @needs_model("scrfd_10g_bnkps")
    def test_scrfd_matches_ort_on_two_resolutions_by_output_name(self):
        ref, ov = self._pair("scrfd_10g_bnkps")
        name = ref.get_inputs()[0].name
        outs = [o.name for o in ref.get_outputs()]
        for hw in (640, 480):
            x = self.rng.standard_normal((1, 3, hw, hw)).astype(np.float32)
            a, b = ref.run(outs, {name: x}), ov.run(outs, {name: x})
            self.assertEqual(len(a), len(b))
            for ra, rb in zip(a, b):
                self.assertEqual(ra.shape, rb.shape)
                self.assertLess(float(np.abs(ra - rb).max()), 5e-3)
        # selecting a subset, in a caller-chosen order, must follow the names, not the graph order
        sub = [outs[3], outs[0]]
        b_sub = ov.run(sub, {name: x})
        np.testing.assert_allclose(b_sub[0], b[3], atol=1e-6)
        np.testing.assert_allclose(b_sub[1], b[0], atol=1e-6)


@needs_ov
@needs_model("xseg")
class ConcurrencyAndErrors(unittest.TestCase):
    def setUp(self):
        self.ov = ovr.OpenVinoAsyncSession(str(_MODELS / "xseg.onnx"), "CPU", precision="fp32", model_key="xseg")
        self.ref = _ref_session("xseg")
        self.name = self.ref.get_inputs()[0].name

    def test_concurrent_callers_each_get_their_own_answer(self):
        """16 threads x 6 distinct inputs. A completion routed to the wrong waiter fails an assert here."""
        rng = np.random.default_rng(3)
        inputs = [rng.random((1, 256, 256, 3), dtype=np.float32) for _ in range(96)]
        expected = [self.ref.run(None, {self.name: x})[0] for x in inputs]
        got, errors = [None] * len(inputs), []

        def work(lo):
            try:
                for i in range(lo, lo + 6):
                    got[i] = self.ov.run(None, {self.name: inputs[i]})[0]
            except Exception as exc:                   # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(k * 6,)) for k in range(16)]
        [t.start() for t in threads]
        [t.join(120) for t in threads]
        self.assertEqual(errors, [])
        for i, (e, g) in enumerate(zip(expected, got)):
            self.assertIsNotNone(g, f"call {i} never returned")
            self.assertLess(float(np.abs(e - g).max()), 2e-3, f"call {i} got another caller's result")
        info = self.ov.info()
        self.assertEqual(info["calls"], 96)
        self.assertGreater(info["peak_in_flight"], 1, "requests never overlapped: the queue was not exercised")
        self.assertGreater(info["queue_requests"][0], 1)

    def test_run_many_returns_results_in_submission_order(self):
        rng = np.random.default_rng(5)
        inputs = [rng.random((1, 256, 256, 3), dtype=np.float32) for _ in range(20)]
        expected = [self.ref.run(None, {self.name: x})[0] for x in inputs]
        got = self.ov.run_many(None, [{self.name: x} for x in inputs])
        for e, g in zip(expected, got):
            self.assertLess(float(np.abs(e - g[0]).max()), 2e-3)

    def test_a_bad_input_raises_and_the_session_keeps_working(self):
        with self.assertRaises(Exception):
            self.ov.run(None, {self.name: np.zeros((1, 100, 100, 3), np.float32)})
        with self.assertRaises(KeyError):
            self.ov.run(None, {"nope": np.zeros((1, 256, 256, 3), np.float32)})
        x = np.random.default_rng(1).random((1, 256, 256, 3), dtype=np.float32)
        self.assertLess(float(np.abs(self.ref.run(None, {self.name: x})[0] - self.ov.run(None, {self.name: x})[0]).max()), 2e-3)

    def test_a_completion_that_never_fires_times_out_instead_of_hanging(self):
        with mock.patch.object(ovr, "_WAIT_S", 0.05):
            job = ovr._Job()
            with self.assertRaises(TimeoutError):
                self.ov._collect(job, None)


@needs_ov
@needs_model("arcface_w600k_r50")
class Int8FailsClosed(unittest.TestCase):
    """The mechanism, not the accuracy: an INT8 IR is used iff its sidecar recorded a PASSING gate."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict(os.environ, {ovr.ENV_CACHE: self.tmp.name})
        env.start()
        self.addCleanup(env.stop)
        self.model = str(_MODELS / "arcface_w600k_r50.onnx")
        rng = np.random.default_rng(11)
        self.samples = [{"input": rng.standard_normal((1, 3, 112, 112)).astype(np.float32)} for _ in range(16)]

    def _quantize(self, gate):
        with mock.patch.dict(ovr.GATES, {"arcface": gate}):
            return ovr.quantize_int8(self.model, "arcface", self.samples)

    def test_failed_gate_writes_a_negative_sidecar_and_is_not_usable(self):
        side = self._quantize(lambda r, g: ("forced", 0.0, False))
        self.assertFalse(side["passed"])
        self.assertIsNone(ovr.usable_int8(self.model, "arcface"))
        with mock.patch.object(ovr, "resolve_device", return_value="CPU"):
            sess = ovr.maybe_session("arcface", self.model, precision="int8")
        self.assertIsNotNone(sess)
        self.assertEqual(sess.precision, "fp16", "int8 must fall back to fp16 when the gate did not pass")

    def test_passed_gate_is_usable_and_the_int8_graph_really_runs(self):
        side = self._quantize(lambda r, g: ("forced", 1.0, True))
        self.assertTrue(side["passed"])
        xml = ovr.usable_int8(self.model, "arcface")
        self.assertTrue(xml and os.path.isfile(xml))
        with mock.patch.object(ovr, "resolve_device", return_value="CPU"):
            sess = ovr.maybe_session("arcface", self.model, precision="int8")
        self.assertEqual(sess.precision, "int8")
        out = sess.run(None, self.samples[0])[0]
        self.assertEqual(out.shape, (1, 512))
        # it must be a quantized graph, not the FP32 one under a new name
        ops = {op.get_type_name() for op in ovr._core().read_model(xml).get_ops()}
        self.assertIn("FakeQuantize", ops)

    def test_a_sidecar_for_a_different_model_file_is_not_trusted(self):
        self._quantize(lambda r, g: ("forced", 1.0, True))
        _xml, sidecar = ovr.int8_paths(self.model, "arcface")
        data = json.load(open(sidecar, encoding="utf-8"))
        data["model_sha"] = "0" * 64
        json.dump(data, open(sidecar, "w", encoding="utf-8"))
        self.assertIsNone(ovr.usable_int8(self.model, "arcface"))

    def test_too_few_samples_and_unknown_model_are_refused(self):
        with self.assertRaises(ValueError):
            ovr.quantize_int8(self.model, "arcface", self.samples[:3])
        with self.assertRaises(ValueError):
            ovr.quantize_int8(self.model, "inswapper", self.samples)

    def test_recorder_keeps_only_real_inputs_it_is_offered(self):
        rec = ovr.CalibrationRecorder("arcface", self.tmp.name, every=2, limit=3)
        for s in self.samples:
            rec.offer(s)
        self.assertEqual(rec.kept, 3)
        path = rec.save()
        back = ovr.load_calibration(path)
        self.assertEqual(len(back), 3)
        np.testing.assert_array_equal(back[0]["input"], self.samples[1]["input"])


if __name__ == "__main__":
    unittest.main()
