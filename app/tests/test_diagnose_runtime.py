#!/usr/bin/env python3
"""Tests for standalone diagnostic mode (--diagnose-runtime)."""
from __future__ import annotations

import io
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

from roop.runtime_diagnostics import (
    collect_runtime_diagnostics,
    format_diagnostics_report,
    run_diagnose_runtime,
)


# Needs the ML stack at runtime (onnxruntime/cv2/torch behaviour, not just imports);
# skipped in the light profile / CI, run on the GPU machines. See conftest.py.
import pytest
pytestmark = pytest.mark.gpu


class TestDiagnoseRuntime(unittest.TestCase):
    def test_report_contains_all_required_section_headers(self):
        """The output report must contain all 7 required section headings."""
        data, code = collect_runtime_diagnostics()
        report = format_diagnostics_report(data)

        expected_sections = [
            "=== Python ===",
            "=== ONNX Runtime ===",
            "=== CUDA ===",
            "=== TensorRT ===",
            "=== Provider Decision ===",
            "=== Config ===",
            "=== Result ===",
        ]
        for section in expected_sections:
            self.assertIn(section, report, f"Missing required section heading: {section}")

    def test_report_contains_all_required_fields(self):
        """The output report must contain all required field labels."""
        data, code = collect_runtime_diagnostics()
        report = format_diagnostics_report(data)

        required_fields = [
            "python version:",
            "python executable:",
            "module path:",
            "version:",
            "get_available_providers API:",
            "available providers:",
            "PyTorch version:",
            "CUDA runtime version:",
            "GPU name:",
            "compute capability:",
            "driver:",
            "TensorRT Python package:",
            "TensorRT version:",
            "TensorRT DLL directories:",
            "TensorRT provider library:",
            "minimal TensorRT ORT session result:",
            "requested:",
            "admitted:",
            "active:",
            "fallback reason:",
            "provider:",
            "trt_precision:",
            "trt_builder_optimization_level:",
        ]
        for field in required_fields:
            self.assertIn(field, report, f"Missing required field: {field}")

    def test_healthy_tensorrt_machine_yields_pass_and_exit_0(self):
        """On a healthy environment, the result is PASS with exit code 0."""
        data, code = collect_runtime_diagnostics()
        self.assertIn(code, (0, 1))  # 0 on RTX 4070, 1 on sub-7GB
        if code == 0:
            self.assertEqual(data["result"]["status"], "PASS")

    def test_degraded_environment_yields_degraded_and_exit_1(self):
        """When TensorRT is unavailable and falls back, result is DEGRADED with exit code 1."""
        mock_preflight = {
            "available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": False,
            "tensorrt_session_usable": False,
            "active_provider": "CUDAExecutionProvider",
            "failure_stage": "tensorrt_initialization_failure",
            "failure_reason": "TensorRT failed to initialize and session dropped to CUDA",
        }
        with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight):
            data, code = collect_runtime_diagnostics()
            self.assertEqual(code, 1)
            self.assertEqual(data["result"]["status"], "DEGRADED")
            report = format_diagnostics_report(data)
            self.assertIn("DEGRADED", report)

    def test_fatal_environment_yields_fail_and_exit_2(self):
        """When core dependency is broken (e.g. NumPy 2.x), result is FAIL with exit code 2."""
        fake_np = MagicMock()
        fake_np.__version__ = "2.2.0"
        with patch.dict(sys.modules, {"numpy": fake_np}):
            data, code = collect_runtime_diagnostics()
            self.assertEqual(code, 2)
            self.assertEqual(data["result"]["status"], "FAIL")
            report = format_diagnostics_report(data)
            self.assertIn("FAIL", report)


if __name__ == "__main__":
    unittest.main()
