#!/usr/bin/env python3
"""Tests for authoritative canonical provider selection pipeline."""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from roop import backend_manager
import settings
from settings import Settings


# Needs the ML stack at runtime (onnxruntime/cv2/torch behaviour, not just imports);
# skipped in the light profile / CI, run on the GPU machines. See conftest.py.
import pytest
pytestmark = pytest.mark.gpu


class TestCanonicalProviderDecision(unittest.TestCase):
    def setUp(self):
        backend_manager.clear_probe_cache()
        settings._DEFAULT_PROVIDER_CACHE = None
        self.addCleanup(backend_manager.clear_probe_cache)

    def test_canonical_result_distinguishes_four_states(self):
        """Preflight must distinguish requested, admitted, available, and active."""
        decision = backend_manager.canonical_provider_decision("tensorrt")
        d = decision.as_dict()
        self.assertIn("requested", d)
        self.assertIn("admitted", d)
        self.assertIn("available", d)
        self.assertIn("active", d)
        self.assertIn("active_chain", d)
        self.assertIn("degraded", d)
        self.assertIn("degradation_reason", d)
        self.assertIn("degradation_stage", d)
        self.assertEqual(decision.requested, "tensorrt")

    def test_downgrade_records_why_and_stage(self):
        """When TensorRT cannot bind, decision must record why and emit degradation."""
        backend_manager.clear_probe_cache()
        mock_preflight = {
            "onnxruntime_importable": True,
            "onnxruntime_version": "1.23.2",
            "onnxruntime_path": "/path/ort",
            "available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": False,
            "tensorrt_session_usable": False,
            "active_provider": "CUDAExecutionProvider",
            "failure_stage": "provider_not_compiled",
            "failure_reason": "TensorrtExecutionProvider not compiled into ORT",
        }

        with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight):
            decision = backend_manager.canonical_provider_decision("tensorrt")
            self.assertEqual(decision.requested, "tensorrt")
            self.assertEqual(decision.active, "CUDAExecutionProvider")
            self.assertTrue(decision.degraded)
            self.assertIsNotNone(decision.degradation_reason)
            self.assertIsNotNone(decision.degradation_stage)

            degs = backend_manager.provider_degradations()
            self.assertTrue(any(e["requested"] == "tensorrt" and e["active"] == "cuda" for e in degs))

    def test_settings_preserves_requested_without_rewriting_config(self):
        """Settings must expose provider_requested and provider_active without silently rewriting saved config."""
        backend_manager.clear_probe_cache()
        settings._DEFAULT_PROVIDER_CACHE = None
        mock_preflight = {
            "onnxruntime_importable": True,
            "onnxruntime_version": "1.23.2",
            "onnxruntime_path": "/path/ort",
            "available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": False,
            "tensorrt_session_usable": False,
            "active_provider": "CUDAExecutionProvider",
            "failure_stage": "dll_load_failure",
            "failure_reason": "nvinfer_10.dll missing",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.yaml"
            cfg_path.write_text("provider: tensorrt\n", encoding="utf-8")

            with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight):
                cfg = Settings(str(cfg_path))
                # provider and provider_requested must stay tensorrt
                self.assertEqual(cfg.provider, "tensorrt")
                self.assertEqual(cfg.provider_requested, "tensorrt")
                # provider_active must report cuda
                self.assertEqual(cfg.provider_active, "cuda")
                self.assertIsNotNone(cfg.degradation_reason)

                # Saving must preserve tensorrt in config file
                cfg.save()
                saved_text = cfg_path.read_text(encoding="utf-8")
                self.assertIn("provider: tensorrt", saved_text)

    def test_cpu_fallback_preserved(self):
        """CPU provider requested must resolve to CPUExecutionProvider with zero degradation."""
        decision = backend_manager.canonical_provider_decision("cpu")
        self.assertEqual(decision.requested, "cpu")
        self.assertEqual(decision.admitted, "cpu")
        self.assertEqual(decision.active, "CPUExecutionProvider")
        self.assertFalse(decision.degraded)


if __name__ == "__main__":
    unittest.main()
