"""`bench.measure_provider_matrix`: the per-stage CPU / primary / OpenVINO table and the four ARMS built from it.

Real onnxruntime on a tiny generated ONNX model drives the CPU arm and `make_feeds`/`_time_calls`/`throughput`
for real. The OpenVINO arm is exercised through a stand-in session with the same surface (the real runner has its
own parity tests in test_openvino_runner.py, which need the `openvino` package this venv deliberately lacks).

What must hold: an arm that cannot run carries a REASON and no number (never an estimate), an error in one stage's
arm is recorded rather than aborting the matrix, and the serial per-frame figure is the sum of ms x calls/frame.
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

try:
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper
except ImportError:                                           # light profile
    onnx = None

from roop import bench, openvino_runner  # noqa: E402

CFG = {"measure_sec": 0.05, "warm_sec": 0.01}


def _tiny_model(path, channels=3):
    w = helper.make_tensor("w", TensorProto.FLOAT, [4, channels, 3, 3],
                           np.random.default_rng(0).standard_normal((4, channels, 3, 3)).astype(np.float32).ravel())
    node = helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])
    graph = helper.make_graph(
        [node], "tiny", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, channels, 32, 32])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 32, 32])], initializer=[w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)


class _FakeOv:
    """The ORT-shaped surface the matrix needs from `OpenVinoAsyncSession`, backed by a CPU ORT session."""

    def __init__(self, path, device, precision="auto", model_key=None, **_):
        self._s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def get_inputs(self):
        return self._s.get_inputs()

    def get_outputs(self):
        return self._s.get_outputs()

    def run(self, outs, feeds):
        return self._s.run(outs, feeds)

    def info(self):
        return {"queue_requests": [2]}


@unittest.skipIf(onnx is None, "needs onnx + onnxruntime (full profile)")
class ProviderMatrix(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "tiny.onnx")
        _tiny_model(self.path)

    def stage(self, calls=2.0, ms=1.5, error=''):
        s = bench.Stage("tiny", "Tiny conv", self.path, ["CPUExecutionProvider"], None, calls, shape_hint=(32, 32))
        s.ms_call, s.provider_used, s.error = ms, "CPU", error
        return s

    def matrix(self, stages, ov_device=None, family=None):
        patches = [mock.patch.object(openvino_runner, "resolve_device", return_value=ov_device),
                   mock.patch.object(openvino_runner, "family_for_file", return_value=family)]
        if ov_device:
            patches += [mock.patch.object(openvino_runner, "OpenVinoAsyncSession", _FakeOv),
                        mock.patch.object(openvino_runner, "device_name", return_value="Fake NPU")]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return bench.measure_provider_matrix(stages, CFG, 1.0, lambda **kw: None, lambda: False)

    def arm(self, m, key):
        return next(a for a in m["arms"] if a["key"] == key)

    def test_the_cpu_arm_is_measured_for_real(self):
        m = self.matrix([self.stage(calls=2.0)])
        row = m["stages"][0]
        self.assertGreater(row["cpu_ms"], 0)
        cpu = self.arm(m, "cpu")
        self.assertTrue(cpu["available"])
        self.assertAlmostEqual(cpu["serial_ms_frame"], round(row["cpu_ms"] * 2.0, 2), places=1)
        self.assertAlmostEqual(cpu["serial_fps"], 1000.0 / cpu["serial_ms_frame"], delta=0.1)

    def test_primary_arm_is_the_summed_per_call_cost(self):
        m = self.matrix([self.stage(calls=2.0, ms=1.5), self.stage(calls=1.0, ms=4.0)])
        self.assertEqual(self.arm(m, "primary")["serial_ms_frame"], 7.0)

    def test_an_arm_that_cannot_run_has_a_reason_and_no_number(self):
        m = self.matrix([self.stage()])
        aux = self.arm(m, "trt_openvino_aux")
        self.assertFalse(aux["available"])
        self.assertIsNone(aux["serial_ms_frame"])
        self.assertIsNone(aux["serial_fps"])
        self.assertIn("Intel", aux["reason"])
        gpu = self.arm(m, "trt_cvcuda_nvofa")
        self.assertFalse(gpu["available"])
        self.assertIsNone(gpu["serial_ms_frame"])
        self.assertTrue(gpu["reason"].startswith("cannot run here"))
        self.assertIn("CV-CUDA", gpu["reason"])

    def test_the_openvino_arm_runs_only_for_an_offloadable_family_on_a_found_device(self):
        none = self.matrix([self.stage()], ov_device="NPU", family=None)
        self.assertIsNone(none["stages"][0]["openvino_ms"])
        self.assertFalse(self.arm(none, "trt_openvino_aux")["available"])
        self.assertIn("no offloadable stage", self.arm(none, "trt_openvino_aux")["reason"])

        hit = self.matrix([self.stage(calls=2.0)], ov_device="NPU", family="scrfd")
        row = hit["stages"][0]
        self.assertGreater(row["openvino_ms"], 0)
        self.assertGreater(row["openvino_calls_per_s"] or 0, 0, row)
        self.assertNotIn("openvino_error", row)
        self.assertIn("NPU", row["openvino_device"])
        hybrid = self.arm(hit, "trt_openvino_aux")
        self.assertTrue(hybrid["available"])
        self.assertAlmostEqual(hybrid["serial_ms_frame"], round(row["openvino_ms"] * 2.0, 2), places=1)

    def test_a_cpu_failure_is_recorded_not_raised(self):
        broken = self.stage()
        broken.path = os.path.join(self.tmp.name, "missing.onnx")
        m = self.matrix([broken, self.stage()])
        self.assertEqual(len(m["stages"]), 2)
        self.assertIn("cpu_error", m["stages"][0])
        self.assertIsNone(m["stages"][0]["cpu_ms"])
        # one stage with no CPU number makes the whole CPU arm unavailable: a partial sum is not a frame cost
        self.assertFalse(self.arm(m, "cpu")["available"])
        self.assertIn("cannot run on the CPU provider", self.arm(m, "cpu")["reason"])
        self.assertIn("Tiny conv", self.arm(m, "cpu")["reason"])
        self.assertTrue(self.arm(m, "primary")["available"])

    def test_a_stage_that_failed_to_measure_is_listed_and_no_arm_pretends_to_be_a_frame(self):
        m = self.matrix([self.stage(error="boom"), self.stage(ms=0.0), self.stage()])
        self.assertEqual(len(m["stages"]), 1)
        self.assertEqual([u["reason"] for u in m["unmeasured"]], ["boom", "no timing was produced"])
        # a partial sum would understate the frame cost: every serial arm is unavailable and says which stage
        for key in ("cpu", "primary"):
            arm = self.arm(m, key)
            self.assertFalse(arm["available"], key)
            self.assertIsNone(arm["serial_ms_frame"], key)
            self.assertIn("not measured", arm["reason"])
            self.assertIn("Tiny conv", arm["reason"])
            self.assertIn("boom", arm["reason"])

    def test_nothing_unmeasured_means_an_empty_list(self):
        self.assertEqual(self.matrix([self.stage()])["unmeasured"], [])

    def test_cancellation_stops_the_matrix(self):
        with mock.patch.object(openvino_runner, "resolve_device", return_value=None):
            with self.assertRaises(bench.Cancelled):
                bench.measure_provider_matrix([self.stage()], CFG, 1.0, lambda **kw: None, lambda: True)

    def test_warp_blend_and_flow_report_unavailable_backends_with_reasons(self):
        m = self.matrix([self.stage()])
        by_key = {b["key"]: b for b in m["warp_blend"]["backends"] + m["optical_flow"]["backends"]}
        self.assertTrue(by_key["cv2_cpu"]["available"])
        self.assertGreater(by_key["cv2_cpu"]["ms"], 0)
        for key in ("cvcuda", "nvof"):
            self.assertFalse(by_key[key]["available"], key)
            self.assertIsNone(by_key[key]["ms"], key)
            self.assertTrue(by_key[key]["reason"], key)
        self.assertTrue(by_key["cv"]["available"])

    def test_the_result_is_json_safe(self):
        import json
        json.dumps(self.matrix([self.stage()]))


if __name__ == "__main__":
    unittest.main()
