#!/usr/bin/env python3
"""Tests for TensorRT settings visibility, provider distinction, and admission state."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import settings
from settings import Settings
import roop.globals as roop_globals
from roop import backend_manager


class TestTrtSettingsVisibility(unittest.TestCase):
    def setUp(self):
        backend_manager.clear_probe_cache()
        settings._DEFAULT_PROVIDER_CACHE = None
        self._saved_cfg = roop_globals.CFG
        self._saved_provs = roop_globals.execution_providers
        self.addCleanup(self._restore)

    def _restore(self):
        backend_manager.clear_probe_cache()
        settings._DEFAULT_PROVIDER_CACHE = None
        roop_globals.CFG = self._saved_cfg
        roop_globals.execution_providers = self._saved_provs

    def test_suggest_execution_providers_preserves_tensorrt_when_not_selected(self):
        """When CUDA is active or selected, TensorRT must remain selectable in available providers."""
        mock_preflight = {
            "available_providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": True,
            "tensorrt_session_usable": True,
            "active_provider": "CUDAExecutionProvider",
        }
        with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight):
            from roop.core import suggest_execution_providers
            providers = suggest_execution_providers()
            self.assertIn("tensorrt", providers)
            self.assertIn("cuda", providers)
            self.assertIn("cpu", providers)

    def test_provider_distinction_requested_active_admitted(self):
        """Settings and get_meta must cleanly distinguish requested, admitted, and active providers."""
        mock_preflight = {
            "available_providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": False,
            "tensorrt_session_usable": False,
            "active_provider": "CUDAExecutionProvider",
            "failure_stage": "tensorrt_initialization_failure",
            "failure_reason": "TensorRT failed to initialize and session dropped to CUDA",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.yaml"
            cfg_path.write_text("provider: tensorrt\ntrt_precision: mixed\n", encoding="utf-8")

            with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight):
                cfg = Settings(str(cfg_path))
                roop_globals.CFG = cfg
                roop_globals.execution_providers = [
                    ("CUDAExecutionProvider", {"device_id": 0}),
                    "CPUExecutionProvider",
                ]

                from api import get_meta, _public_settings
                meta = get_meta()
                pub = _public_settings(cfg)

                # 1. UI payload must distinguish requested, active, available
                self.assertEqual(meta["requested_provider"], "tensorrt")
                self.assertEqual(meta["active_provider"], "cuda")
                self.assertIn("tensorrt", meta["providers"])
                self.assertFalse(meta["tensorrt_active"])
                self.assertIsNotNone(meta["degradation_reason"])

                # 2. Public settings payload must preserve tensorrt as requested setting
                self.assertEqual(pub["provider"], "tensorrt")
                self.assertEqual(pub["provider_requested"], "tensorrt")
                self.assertEqual(pub["provider_active"], "cuda")
                self.assertEqual(pub["trt_precision"], "mixed")

                # 3. Saving settings must never silently rewrite provider to cuda
                cfg.save()
                saved_text = cfg_path.read_text(encoding="utf-8")
                self.assertIn("provider: tensorrt", saved_text)

    def test_sub_7gb_safety_admission_state_reflected(self):
        """When sub-7GB GPU safety policy rejects TensorRT, admission state must be 'cuda' while requested is 'tensorrt'."""
        mock_preflight = {
            "available_providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": True,
            "tensorrt_session_usable": True,
            "active_provider": "TensorrtExecutionProvider",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.yaml"
            cfg_path.write_text("provider: tensorrt\n", encoding="utf-8")

            with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight), \
                 patch("roop.backend_manager._small_gpu", return_value=True), \
                 patch.dict(os.environ, {"ROOP_ALLOW_TRT_SMALL_GPU": "0"}):

                cfg = Settings(str(cfg_path))
                self.assertEqual(cfg.provider, "tensorrt")
                self.assertEqual(cfg.provider_requested, "tensorrt")
                self.assertEqual(cfg.provider_admitted, "cuda")
                self.assertEqual(cfg.provider_active, "cuda")
                self.assertEqual(cfg.degradation_stage, "admission_rejected")
                self.assertIn("sub-7GB safety policy", cfg.degradation_reason)

    def test_tensorrt_active_state_exposes_trt_active_flag(self):
        """When TensorRT is genuinely active, meta and CFG reflect tensorrt active."""
        mock_preflight = {
            "available_providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": True,
            "tensorrt_session_usable": True,
            "active_provider": "TensorrtExecutionProvider",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.yaml"
            cfg_path.write_text("provider: tensorrt\ntrt_precision: fp16\n", encoding="utf-8")

            with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight), \
                 patch("roop.backend_manager._small_gpu", return_value=False):

                cfg = Settings(str(cfg_path))
                roop_globals.CFG = cfg
                roop_globals.execution_providers = [
                    ("TensorrtExecutionProvider", {}),
                    ("CUDAExecutionProvider", {}),
                    "CPUExecutionProvider",
                ]

                from api import get_meta
                meta = get_meta()
                self.assertEqual(meta["requested_provider"], "tensorrt")
                self.assertEqual(meta["active_provider"], "tensorrt")
                self.assertTrue(meta["tensorrt_active"])
                self.assertFalse(meta["provider_status"]["degraded"])


if __name__ == "__main__":
    unittest.main()
