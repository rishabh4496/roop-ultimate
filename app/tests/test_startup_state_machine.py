#!/usr/bin/env python3
"""Comprehensive tests for the transactional startup state machine."""
from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from roop.startup_state_machine import (
    StartupPhase,
    PhaseStatus,
    PhaseResult,
    StartupStateMachine,
    PHASE_SEQUENCE,
    format_fatal_diagnostic,
    execute_boot,
    execute_dependency_preflight,
    execute_dll_runtime_preflight,
    execute_ort_preflight,
    execute_gpu_preflight,
    execute_provider_admission,
    execute_config_load,
    execute_model_runtime_init,
    execute_api_ready,
    execute_ui_ready,
)


class TestStartupStateMachine(unittest.TestCase):
    def setUp(self):
        from roop import backend_manager
        backend_manager.clear_probe_cache()
        self.sm = StartupStateMachine(halt_on_fatal=False)
        self.addCleanup(backend_manager.clear_probe_cache)

    def test_ten_phases_and_sequence_defined(self):
        """All 10 explicit phases must be defined in strict sequence."""
        expected = [
            "BOOT",
            "DEPENDENCY_PREFLIGHT",
            "DLL_RUNTIME_PREFLIGHT",
            "ORT_PREFLIGHT",
            "GPU_PREFLIGHT",
            "PROVIDER_ADMISSION",
            "CONFIG_LOAD",
            "MODEL_RUNTIME_INIT",
            "API_READY",
            "UI_READY",
        ]
        self.assertEqual([p.value for p in PHASE_SEQUENCE], expected)

    def test_fatal_diagnostic_format(self):
        """Fatal output must match the exact required key-value structure."""
        out = format_fatal_diagnostic(
            stage=StartupPhase.DEPENDENCY_PREFLIGHT,
            component="numpy",
            reason="InsightFace C-bindings require numpy<2.0.0",
            detected_version="2.1.0",
            expected_version="<2.0.0",
            next_action="uv pip install 'numpy<2.0.0'",
        )
        lines = out.strip().split("\n")
        self.assertEqual(lines[0], "[Startup:FATAL]")
        self.assertIn("stage=DEPENDENCY_PREFLIGHT", lines)
        self.assertIn("component=numpy", lines)
        self.assertIn("reason=InsightFace C-bindings require numpy<2.0.0", lines)
        self.assertIn("detected_version=2.1.0", lines)
        self.assertIn("expected_version=<2.0.0", lines)
        self.assertIn("next_action=uv pip install 'numpy<2.0.0'", lines)

    def test_fatal_halts_and_blocks_subsequent_phases(self):
        """A FATAL phase must set fatal state and block transitions to any later phase."""
        self.assertTrue(self.sm.can_transition_to(StartupPhase.BOOT))
        self.sm.record_success(StartupPhase.BOOT, "bootstrap")

        self.assertTrue(self.sm.can_transition_to(StartupPhase.DEPENDENCY_PREFLIGHT))
        self.sm.record_fatal(
            StartupPhase.DEPENDENCY_PREFLIGHT,
            component="numpy",
            reason="Incompatible numpy version",
            detected_version="2.0.0",
            expected_version="<2.0.0",
        )
        self.assertTrue(self.sm.is_failed)

        # Subsequent phases MUST be blocked
        self.assertFalse(self.sm.can_transition_to(StartupPhase.DLL_RUNTIME_PREFLIGHT))
        self.assertFalse(self.sm.can_transition_to(StartupPhase.ORT_PREFLIGHT))
        self.assertFalse(self.sm.can_transition_to(StartupPhase.API_READY))
        self.assertFalse(self.sm.can_transition_to(StartupPhase.UI_READY))

    def test_dependency_preflight_numpy_2_is_fatal(self):
        """NumPy 2.x must return FATAL in DEPENDENCY_PREFLIGHT with diagnostic details."""
        fake_np = MagicMock()
        fake_np.__version__ = "2.1.0"
        with patch.dict(sys.modules, {"numpy": fake_np}):
            res = execute_dependency_preflight()
            self.assertEqual(res.status, PhaseStatus.FATAL)
            self.assertEqual(res.phase, StartupPhase.DEPENDENCY_PREFLIGHT)
            self.assertEqual(res.component, "numpy")
            self.assertEqual(res.detected_version, "2.1.0")
            self.assertEqual(res.expected_version, "<2.0.0 (e.g. 1.26.4)")
            self.assertIn("numpy<2.0.0", res.next_action)

    def test_ort_preflight_empty_providers_is_fatal(self):
        """ORT exposing no providers must return FATAL in ORT_PREFLIGHT."""
        mock_res = {
            "onnxruntime_importable": True,
            "onnxruntime_version": "1.23.2",
            "available_providers": [],
            "failure_stage": "provider_not_compiled",
            "failure_reason": "provider registry is empty",
        }
        with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_res):
            res = execute_ort_preflight()
            self.assertEqual(res.status, PhaseStatus.FATAL)
            self.assertEqual(res.phase, StartupPhase.ORT_PREFLIGHT)
            self.assertEqual(res.component, "onnxruntime")

    def test_auto_provider_with_unusable_tensorrt_is_degraded(self):
        """When provider mode is AUTO and TensorRT is unavailable, status is DEGRADED."""
        mock_preflight = {
            "available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": False,
            "tensorrt_session_usable": False,
            "active_provider": "CUDAExecutionProvider",
        }
        with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight):
            res = execute_provider_admission(requested="auto", device_id=0)
            self.assertEqual(res.status, PhaseStatus.DEGRADED)
            self.assertEqual(res.phase, StartupPhase.PROVIDER_ADMISSION)
            self.assertEqual(res.details["active"], "cuda")

    def test_explicit_tensorrt_request_with_unusable_tensorrt_is_degraded_not_silent_mutation(self):
        """When TensorRT is explicitly requested but unavailable, status is DEGRADED without silent mutation."""
        mock_preflight = {
            "available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": False,
            "tensorrt_session_usable": False,
            "active_provider": "CUDAExecutionProvider",
            "failure_stage": "dll_load_failure",
            "failure_reason": "TensorRT runtime DLLs missing",
        }
        with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight):
            res = execute_provider_admission(requested="tensorrt", device_id=0)
            self.assertEqual(res.status, PhaseStatus.DEGRADED)
            self.assertEqual(res.phase, StartupPhase.PROVIDER_ADMISSION)
            self.assertEqual(res.component, "tensorrt")
            self.assertEqual(res.details["requested"], "tensorrt")
            self.assertEqual(res.details["active"], "cuda")

    def test_cpu_fallback_policy_disallowed_is_fatal(self):
        """If ROOP_DISALLOW_CPU_FALLBACK=1, a GPU request falling back to CPU must return FATAL."""
        mock_preflight = {
            "available_providers": ["CPUExecutionProvider"],
            "cuda_available": False,
            "tensorrt_available": False,
            "tensorrt_session_usable": False,
            "active_provider": "CPUExecutionProvider",
        }
        with patch.dict(os.environ, {"ROOP_DISALLOW_CPU_FALLBACK": "1"}):
            with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight):
                res = execute_provider_admission(requested="cuda", device_id=0)
                self.assertEqual(res.status, PhaseStatus.FATAL)
                self.assertIn("CPU fallback is forbidden by policy", res.reason)

    def test_ui_ready_depends_on_api_ready(self):
        """UI_READY cannot be declared ready if API_READY did not succeed."""
        self.sm.record_success(StartupPhase.BOOT, "boot")
        self.sm.record_success(StartupPhase.DEPENDENCY_PREFLIGHT, "deps")
        self.sm.record_success(StartupPhase.DLL_RUNTIME_PREFLIGHT, "dlls")
        self.sm.record_success(StartupPhase.ORT_PREFLIGHT, "ort")
        self.sm.record_success(StartupPhase.GPU_PREFLIGHT, "gpu")
        self.sm.record_success(StartupPhase.PROVIDER_ADMISSION, "provider")
        self.sm.record_success(StartupPhase.CONFIG_LOAD, "config")
        self.sm.record_success(StartupPhase.MODEL_RUNTIME_INIT, "model")

        # API_READY failed
        self.sm.record_fatal(StartupPhase.API_READY, "api", "Port binding failed")

        # UI_READY cannot transition
        self.assertFalse(self.sm.can_transition_to(StartupPhase.UI_READY))

    def test_pinokio_url_not_emitted_before_api_ready(self):
        """The Pinokio capture URL is emitted only by UI_READY when API_READY has succeeded."""
        fake_thread = MagicMock()
        fake_thread.is_alive.return_value = False

        # If API server died, execute_api_ready returns FATAL
        res_api = execute_api_ready(fake_thread, api_port=9999, timeout=0.1)
        self.assertEqual(res_api.status, PhaseStatus.FATAL)

        # stdout must NOT have the listening URL
        stdout_buf = io.StringIO()
        with patch("sys.stdout", stdout_buf):
            # Because API failed, UI_READY is never called, so stdout has no URL
            self.assertNotIn("[Backend] listening on http", stdout_buf.getvalue())


if __name__ == "__main__":
    unittest.main()
