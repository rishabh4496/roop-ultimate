"""Exercise startup code paths when TensorRT is not installed or optional.

Validates that missing TensorRT packages do not emit [Fallback] or
ModuleNotFoundError diagnostic messages that Pinokio's shell runner or
launcher script could misinterpret as fatal startup failures.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest
from unittest.mock import patch

from roop import degrade


# Needs the ML stack at runtime (onnxruntime/cv2/torch behaviour, not just imports);
# skipped in the light profile / CI, run on the GPU machines. See conftest.py.
import pytest
pytestmark = pytest.mark.gpu


class TestStartupOptionalTensorRT(unittest.TestCase):
    def setUp(self):
        degrade.reset()
        self.addCleanup(degrade.reset)

    def test_enable_tensorrt_runtime_silent_when_not_installed(self):
        """When tensorrt is not installed, _enable_tensorrt_runtime must not emit fallback errors."""
        from settings import _enable_tensorrt_runtime
        orig_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "tensorrt":
                raise ModuleNotFoundError("No module named 'tensorrt'")
            return orig_import(name, *args, **kwargs)

        buf = io.StringIO()
        with patch("builtins.__import__", side_effect=fake_import):
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                _enable_tensorrt_runtime()

        output = buf.getvalue()
        self.assertNotIn("ModuleNotFoundError", output)
        self.assertNotIn("Fallback", output)
        self.assertNotIn("Error:", output)
        self.assertEqual(degrade.report(), [])

    def test_detect_hardware_silent_when_tensorrt_not_installed(self):
        """When tensorrt is not installed, detect_hardware must succeed without emitting ModuleNotFoundError."""
        import settings
        settings._HARDWARE_CACHE = None
        orig_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "tensorrt":
                raise ModuleNotFoundError("No module named 'tensorrt'")
            return orig_import(name, *args, **kwargs)

        buf = io.StringIO()
        with patch("builtins.__import__", side_effect=fake_import):
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                hw = settings.detect_hardware()

        output = buf.getvalue()
        self.assertNotIn("ModuleNotFoundError", output)
        self.assertNotIn("[Fallback] settings.py:26", output)
        self.assertNotIn("[Fallback] settings.py:171", output)
        self.assertEqual(hw.get("tensorrt", ""), "")
        # Clean up cache
        settings._HARDWARE_CACHE = None

    def test_backend_manager_cache_namespace_silent_when_tensorrt_not_installed(self):
        """cache_namespace should resolve gracefully without logging a fallback error when tensorrt is absent."""
        from roop.backend_manager import cache_namespace
        orig_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "tensorrt":
                raise ModuleNotFoundError("No module named 'tensorrt'")
            return orig_import(name, *args, **kwargs)

        buf = io.StringIO()
        with patch("builtins.__import__", side_effect=fake_import):
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                ns = cache_namespace(precision="fp16", device_id=0)

        output = buf.getvalue()
        self.assertNotIn("ModuleNotFoundError", output)
        self.assertNotIn("[Fallback] roop/backend_manager.py:275", output)
        self.assertIn("trtunknown", ns)

    def test_runtime_optimizer_profile_silent_when_tensorrt_not_installed(self):
        """HardwareProfiler.profile must not emit ModuleNotFoundError when tensorrt is absent."""
        from roop.runtime_optimizer import HardwareProfiler
        orig_import = __import__

        def fake_import(name, *args, **kwargs):
            if name in ("tensorrt", "_trt"):
                raise ModuleNotFoundError("No module named 'tensorrt'")
            return orig_import(name, *args, **kwargs)

        buf = io.StringIO()
        with patch("builtins.__import__", side_effect=fake_import):
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                prof = HardwareProfiler().profile()

        output = buf.getvalue()
        self.assertNotIn("ModuleNotFoundError", output)
        self.assertNotIn("[Fallback] roop/runtime_optimizer.py:902", output)
        self.assertEqual(prof.tensorrt_version, "")


if __name__ == "__main__":
    unittest.main()
