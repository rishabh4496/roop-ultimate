"""A hardware path that gives way to a slower one must say so where the report and the HUD read it.

The failure class (AGENTS.md, "Something reports success while not running"): a broad ``except`` around PyNvVideoCodec /
Triton / cuda-python / NVOF caught a RUNTIME bug, continued on the CPU, and the label that said "NVDEC-DLPack" stayed.
Each test below drives one of those sites with a fault and reads the consumer - the telemetry snapshot (per-job report,
HARDWARE section), ``flat()`` (the 4 Hz frame behind the HUD badge) or ``roop.degrade.report()`` - never the wrapper.

None of it needs a GPU; the fault is injected where the real one would surface.
"""
from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import degrade                                     # noqa: E402
from roop import video_io_hw as hw                           # noqa: E402

BANNER = "[WARNING: HARDWARE FALLBACK TRIGGERED]"


def _captured(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn()
    return buf.getvalue(), result


class TelemetryFallbacks(unittest.TestCase):
    def setUp(self):
        hw.telemetry.reset()
        degrade.reset()

    def test_a_runtime_fallback_prints_the_banner_once_relabels_the_stage_and_rides_the_frame(self):
        t = hw.telemetry
        t.register_decoder("stage1-keyframes", hw.BACKEND_NVDEC_DLPACK)
        out, _ = _captured(lambda: [t.hardware_fallback("decode", "stage1-keyframes", hw.BACKEND_NVDEC_DLPACK,
                                                        hw.BACKEND_CPU, "ValueError: stride") for _ in range(3)])
        self.assertEqual(out.count(BANNER), 1, "the same reason is announced once, not per frame")
        self.assertIn("stage1-keyframes: NVDEC-DLPack -> CPU-Fallback (ValueError: stride)", out)
        snap = t.snapshot()
        self.assertEqual(snap["decoders"]["stage1-keyframes"], hw.BACKEND_CPU,
                         "the reader registered itself as NVDEC when it opened; the label must follow the work")
        self.assertEqual(snap["decode_backend"], hw.BACKEND_CPU)
        self.assertEqual(snap["fallback_count"], 1)
        self.assertEqual(len(snap["fallbacks"]), 1)
        flat = t.flat()
        self.assertEqual(flat["io_fallbacks"], 1)
        self.assertIn("ValueError: stride", flat["io_fallback"])

    def test_an_encoder_fallback_relabels_the_encoder_not_the_decoder(self):
        t = hw.telemetry
        t.register_decoder("decode", hw.BACKEND_NVDEC_DLPACK)
        t.register_encoder("encode", hw.BACKEND_NVENC_DIRECT)
        _captured(lambda: t.hardware_fallback("encode", "encode", hw.BACKEND_NVENC_DIRECT, hw.BACKEND_CPU, "x"))
        snap = t.snapshot()
        self.assertEqual(snap["encoders"]["encode"], hw.BACKEND_CPU)
        self.assertEqual(snap["decoders"]["decode"], hw.BACKEND_NVDEC_DLPACK)

    def test_an_optional_qsv_skip_is_recorded_but_never_raises_the_badge(self):
        out, _ = _captured(lambda: hw.telemetry.note_fallback("decode: QSV-oneVPL skipped (no adapter)", warn=False))
        self.assertNotIn(BANNER, out)
        self.assertEqual(hw.telemetry.flat()["io_fallbacks"], 0)
        self.assertEqual(hw.snapshot()["fallbacks"], ["decode: QSV-oneVPL skipped (no adapter)"],
                         "still in the report for anyone who wants it")

    def test_the_open_ladders_announce_a_rung_that_cannot_start(self):
        with mock.patch.object(hw, "_load_nvc", side_effect=hw.HwUnavailable("no nvc (test)")), \
                mock.patch.dict(os.environ, {"ROOP_QSV": "0"}):
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "x.mp4")
                out, w = _captured(lambda: hw.open_writer(path, 321, 181, 25.0))
                with contextlib.suppress(Exception):
                    w.close()
        self.assertIn(BANNER, out)
        self.assertEqual(hw.telemetry.flat()["io_fallbacks"], 1)

    def test_idle_frame_fields_are_constant(self):
        a, b = hw.telemetry.flat(), hw.telemetry.flat()
        self.assertEqual(a, b)
        self.assertEqual((a["io_fallbacks"], a["io_fallback"]), (0, ""))


class SparseScanRuntimeFailure(unittest.TestCase):
    """roop/sparse_range_scan.py: a hardware scan that dies on a real bug drops to cv2 - and says so."""

    def setUp(self):
        hw.telemetry.reset()
        degrade.reset()
        fd, self.path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

    def tearDown(self):
        with contextlib.suppress(OSError):
            os.remove(self.path)

    def _scan(self, exc):
        from roop import sparse_range_scan as srs
        hw.telemetry.register_decoder("stage1-keyframes", hw.BACKEND_NVDEC_DLPACK)   # what the reader did at open
        with mock.patch.object(srs, "_scan_active_ranges_hw", side_effect=exc):
            return _captured(lambda: srs.scan_active_ranges(self.path, frame_count=300))

    def test_a_runtime_bug_in_the_hardware_scan_relabels_and_reports(self):
        out, result = self._scan(ValueError("strides (3, 640) not contiguous"))
        self.assertIn(BANNER, out)
        snap = hw.snapshot()
        self.assertEqual(snap["decoders"]["stage1-keyframes"], hw.BACKEND_CPU)
        self.assertEqual(snap["fallback_count"], 1)
        self.assertIn("strides (3, 640) not contiguous", snap["fallbacks"][0])
        self.assertIn("sparse_range_scan:hw_scan", [e["site"] for e in degrade.report()])
        self.assertEqual(result[2].get("error"), "cannot_open_video", "the cv2 rung ran (and this file is not a video)")

    def test_an_unavailable_package_is_reported_the_same_way(self):
        out, _ = self._scan(hw.HwUnavailable("PyNvVideoCodec/torch unavailable: ImportError: DLL load failed"))
        self.assertIn(BANNER, out)
        self.assertEqual(hw.snapshot()["decoders"]["stage1-keyframes"], hw.BACKEND_CPU)

    def test_strict_mode_turns_the_swallow_into_a_failure(self):
        with mock.patch.dict(os.environ, {"ROOP_STRICT_FALLBACK": "1"}):
            with self.assertRaises(ValueError):
                self._scan(ValueError("boom"))


class DegradeFellBack(unittest.TestCase):
    def setUp(self):
        degrade.reset()

    def test_fell_back_counts_prints_once_and_honours_strict(self):
        out, _ = _captured(lambda: [degrade.fell_back("t:site", "no runtime", "host path") for _ in range(3)])
        self.assertEqual(out.count("[Fallback] t:site"), 1)
        (entry,) = degrade.report()
        self.assertEqual((entry["site"], entry["count"]), ("t:site", 3))
        self.assertIn("no runtime", entry["first_error"])
        with self.assertRaises(RuntimeError):
            degrade.fell_back("t:strict", "x", strict=True)


class ImportReasonsSurvive(unittest.TestCase):
    def setUp(self):
        degrade.reset()

    def test_triton_import_failure_reason_is_kept_and_a_cuda_tensor_downgrade_is_recorded(self):
        fp = __import__("core.fused_postprocess", fromlist=["x"])
        cuda_tensor = types.SimpleNamespace(is_cuda=True)
        with mock.patch.object(fp, "HAVE_TRITON", False), \
                mock.patch.object(fp, "TRITON_IMPORT_ERROR", "ImportError: libtriton.dll: invalid ELF header"):
            out, backend = _captured(lambda: fp.select_backend(cuda_tensor, "auto"))
            self.assertEqual(backend, "torch")
            self.assertIn("invalid ELF header", out)
            self.assertIn("core/fused_postprocess.py:triton_unavailable", [e["site"] for e in degrade.report()])
            with self.assertRaisesRegex(RuntimeError, "invalid ELF header"):
                fp.select_backend(cuda_tensor, "triton")
            with self.assertRaisesRegex(RuntimeError, "invalid ELF header"):
                fp.fused_postprocess(None, None, None, None)

    def test_a_cpu_tensor_taking_the_reference_is_a_choice_not_a_fallback(self):
        fp = __import__("core.fused_postprocess", fromlist=["x"])
        with mock.patch.object(fp, "HAVE_TRITON", False):
            _captured(lambda: fp.select_backend(types.SimpleNamespace(is_cuda=False), "auto"))
        self.assertEqual(degrade.report(), [])

    def test_cuda_python_import_failure_reason_is_in_available(self):
        l2 = __import__("core.l2_persistence", fromlist=["x"])
        with mock.patch.object(l2, "HAVE_CUDA_PYTHON", False), \
                mock.patch.object(l2, "CUDA_PYTHON_IMPORT_ERROR", "ImportError: cudart64_12.dll not found"):
            ok, why = l2.available()
        self.assertFalse(ok)
        self.assertIn("cudart64_12.dll not found", why)


class CudartNoneIsNotSilent(unittest.TestCase):
    def setUp(self):
        degrade.reset()

    def test_a_missing_runtime_dll_is_recorded_when_the_cuda_copy_is_declined(self):
        ec = __import__("roop.processors.enhance_common", fromlist=["x"])
        ort_value = types.SimpleNamespace(device_name=lambda: "cuda")
        with mock.patch.object(ec, "_cudart", return_value=None):
            out, result = _captured(lambda: ec.ort_cuda_output_to_torch(ort_value, None))
        self.assertIsNone(result)
        self.assertIn("roop/processors/enhance_common.py:cudart", [e["site"] for e in degrade.report()])

    def test_a_non_cuda_value_stays_quiet(self):
        ec = __import__("roop.processors.enhance_common", fromlist=["x"])
        out, result = _captured(lambda: ec.ort_cuda_output_to_torch(types.SimpleNamespace(device_name=lambda: "cpu"), None))
        self.assertIsNone(result)
        self.assertEqual(degrade.report(), [])


class NvofCreateFailureIsRecorded(unittest.TestCase):
    def setUp(self):
        degrade.reset()

    def test_smart_skip_records_an_nvof_engine_that_cannot_be_created(self):
        import cv2
        from roop import temporal_smart_skip as tss
        warper = tss.DenseFlowWarper()
        bad = types.SimpleNamespace(create=mock.Mock(side_effect=cv2.error("NVOF: no OFA engine on this device")))
        with mock.patch.object(tss.DenseFlowWarper, "_cuda_available", staticmethod(lambda: True)), \
                mock.patch.object(tss.DenseFlowWarper, "_nvof_class", staticmethod(lambda: bad)):
            out, engine = _captured(lambda: warper._nvof_engine((90, 160)))
        self.assertIsNone(engine)
        self.assertIn("roop/temporal_smart_skip.py:nvof_create", [e["site"] for e in degrade.report()])
        self.assertIn("160x90", out)

    def test_the_swapper_records_it_too(self):
        import cv2
        try:
            from roop.processors.frame import face_swapper as fs
        except Exception as exc:                                    # pragma: no cover - heavy import, light profile
            self.skipTest(f"face_swapper does not import here: {type(exc).__name__}: {exc}")
        smoother = fs.TemporalMaskSmoother()
        bad = types.SimpleNamespace(create=mock.Mock(side_effect=cv2.error("NVOF: no OFA engine on this device")))
        with mock.patch.object(fs.TemporalMaskSmoother, "_cuda_available", staticmethod(lambda: True)), \
                mock.patch.object(fs.TemporalMaskSmoother, "_nvof_class", staticmethod(lambda: bad)):
            out, engine = _captured(lambda: smoother._nvof_engine((90, 160)))
        self.assertIsNone(engine)
        self.assertIn("face_swapper.py:nvof_create", [e["site"] for e in degrade.report()])


class Fp8CalibrationIsNotASilentPass(unittest.TestCase):
    """core/trt_precision_surgeon.py: the activation probes used to be wrapped in a bare ``except: pass``."""

    @staticmethod
    def _model():
        import numpy as np
        import onnx
        from onnx import TensorProto, helper
        rng = np.random.default_rng(0)

        def conv(name, x, y, cin, cout):
            w = helper.make_tensor(f"{name}_w", TensorProto.FLOAT, [cout, cin, 3, 3],
                                   rng.standard_normal(cout * cin * 9).astype("float32").tolist())
            return w, helper.make_node("Conv", [x, f"{name}_w"], [y], name=name, pads=[1, 1, 1, 1])

        w1, c1 = conv("c1", "input", "t1", 3, 4)
        w2, c2 = conv("c2", "t1", "t2", 4, 4)
        w3, c3 = conv("c3", "t2", "output", 4, 4)
        graph = helper.make_graph([c1, c2, c3], "g",
                                  [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])],
                                  [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 8, 8])],
                                  initializer=[w1, w2, w3])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        return model, np

    def _run(self, feeds):
        try:
            import onnx_graphsurgeon                                 # noqa: F401
            import onnxruntime                                       # noqa: F401
            from core import trt_precision_surgeon as tps
        except Exception as exc:
            self.skipTest(f"onnx stack not importable here: {type(exc).__name__}: {exc}")
        model, _ = self._model()
        quant = tps.ModelOptQuantizer.__new__(tps.ModelOptQuantizer)       # skip the hardware probe
        quant.device_id, quant.hardware, quant.last_calibration = 0, {}, {}
        with self.assertLogs(tps.logger, level="WARNING") as logs:
            logging.getLogger(tps.logger.name).warning("sentinel")           # assertLogs needs at least one record
            quant._quantize_native_fp8(model, "test", feeds, None)
        return quant.last_calibration, "\n".join(logs.output)

    def test_a_feed_that_cannot_run_is_counted_and_warned_about(self):
        _, np = self._model()
        last, log = self._run([{"input": np.zeros((1, 3, 5, 5), np.float32)}])      # the graph is declared 8x8
        self.assertEqual(last["runs"], 0)
        self.assertGreater(last["failed"], 0)
        self.assertIn("NO feed succeeded", log)

    def test_a_good_feed_calibrates_without_a_warning(self):
        _, np = self._model()
        last, log = self._run([{"input": np.ones((1, 3, 8, 8), np.float32)}])
        self.assertEqual((last["runs"], last["failed"], last["tensors"]), (1, 0, 1),
                         "the interior conv's input is an intermediate tensor: ORT only returns it when promoted")
        self.assertNotIn("calibration feeds failed", log)


if __name__ == "__main__":
    unittest.main()
