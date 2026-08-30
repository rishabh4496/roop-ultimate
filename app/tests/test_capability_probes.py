"""Guards against capability probes that can never succeed.

Two probes in this project asked for attributes that do not exist:
``torch._C._cuda_getDriverVersion`` and ``builder.platform_has_fast_fp8``.
Both were wrapped in ``getattr(..., default)``, so each returned a plausible
negative ("driver unknown", "FP8 unsupported") on every machine instead of
failing.  A negative capability that cannot be distinguished from a broken
probe is the failure these tests exist to prevent, so they assert the probed
attribute EXISTS rather than asserting a particular capability value.

The driver half was found independently on both validation targets; the
engine-cache side of it is owned by ``test_backend_manager`` (the driver
stays in the cache key).  What is asserted here is that the probes
themselves can still answer.
"""
import os
import sys
import unittest
import unittest.mock

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import backend_manager


def _torch_cuda():
    try:
        import torch
        return torch if torch.cuda.is_available() else None
    except Exception:
        return None


def _trt():
    try:
        import tensorrt
        return tensorrt
    except Exception:
        return None


class TestDriverProbe(unittest.TestCase):
    @unittest.skipIf(_torch_cuda() is None, "no CUDA device")
    def test_driver_version_is_actually_resolved(self):
        """nvidia-smi is the only source that reports the display driver."""
        driver = backend_manager._driver_from_smi(0)
        self.assertTrue(driver, "display driver did not resolve on a CUDA host")
        self.assertNotEqual(driver.lower(), "unknown")
        self.assertRegex(driver, r"^\d+\.\d+")

    def test_torch_driver_probe_is_still_absent(self):
        """Pin the reason the old fallback was dead, so a revival is caught."""
        torch = _torch_cuda()
        if torch is None:
            self.skipTest("no CUDA device")
        self.assertIsNone(
            getattr(getattr(torch, "_C", None), "_cuda_getDriverVersion", None),
            "torch now exposes a driver probe; display_driver_version may use "
            "it, but only after checking it returns the DISPLAY driver and not "
            "the CUDA driver API version")


class TestPrecisionProbe(unittest.TestCase):
    @unittest.skipIf(_trt() is None, "TensorRT not installed")
    def test_probed_builder_attributes_exist(self):
        trt = _trt()
        builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
        for name in ("platform_has_fast_fp16", "platform_has_fast_int8"):
            self.assertTrue(hasattr(builder, name),
                            f"the FP16/INT8 probe reads a missing '{name}'")

    @unittest.skipIf(_trt() is None, "TensorRT not installed")
    def test_fp8_is_probed_through_an_answerable_question(self):
        trt = _trt()
        builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
        self.assertFalse(
            hasattr(builder, "platform_has_fast_fp8"),
            "TensorRT now has platform_has_fast_fp8; prefer it over the "
            "BuilderFlag/DataType + SM gate")
        self.assertTrue(hasattr(trt.BuilderFlag, "FP8"))
        self.assertTrue(hasattr(trt.DataType, "FP8"))

    @unittest.skipIf(_trt() is None or _torch_cuda() is None,
                     "needs TensorRT and a CUDA device")
    def test_fp8_exposure_follows_compute_capability(self):
        from roop.runtime_optimizer import HardwareProfiler
        torch = _torch_cuda()
        compute = torch.cuda.get_device_capability(0)
        _fp16, _bf16, _int8, fp8, _cores = HardwareProfiler._precision_capabilities(
            torch, 0, compute, True)
        self.assertEqual(fp8, compute >= (8, 9))


if __name__ == "__main__":
    unittest.main()
