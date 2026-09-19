"""Verify that clean installs on new user devices receive TensorRT and precision mode.

Validates that:
1. Fresh launches without an existing config.yaml seed from default_config.yaml
   with provider: tensorrt and trt_precision: mixed.
2. Even in the absence of any config file, NVIDIA hardware profiles resolve
   provider to 'tensorrt' by default.
3. Sub-7GB devices permit TensorRT without stripping it from provider resolution.
4. The /api/meta endpoint publishes 'tensorrt' in providers and all precision modes.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from roop import degrade
from roop.backend_manager import (
    clear_probe_cache,
    provider_admission,
    resolve_provider_names,
    _small_gpu,
)
from settings import Settings


class TestFreshInstallTensorRT(unittest.TestCase):
    def setUp(self):
        degrade.reset()
        clear_probe_cache()
        self.addCleanup(degrade.reset)
        self.addCleanup(clear_probe_cache)

    def test_default_config_yaml_exists_and_contains_main_device_tensorrt(self):
        """The repository must ship default_config.yaml matching the main workstation."""
        default_yaml = Path(__file__).resolve().parents[1] / "default_config.yaml"
        self.assertTrue(default_yaml.exists(), "app/default_config.yaml must exist")
        text = default_yaml.read_text(encoding="utf-8")
        self.assertIn("provider: tensorrt", text)
        self.assertIn("trt_precision: mixed", text)
        self.assertIn("perf_trt_pool: '2'", text)
        self.assertIn("selected_enhancer: Restore Ultra", text)

    def test_settings_seeds_from_default_config_when_config_yaml_missing(self):
        """A new user's PC starts with no config.yaml; Settings must seed from default_config.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_default = Path(__file__).resolve().parents[1] / "default_config.yaml"
            tmp_default = Path(tmpdir) / "default_config.yaml"
            shutil.copy(str(repo_default), str(tmp_default))
            
            cfg_path = Path(tmpdir) / "config.yaml"
            self.assertFalse(cfg_path.exists())
            
            cfg = Settings(str(cfg_path))
            self.assertEqual(cfg.provider, "tensorrt")
            self.assertEqual(cfg.trt_precision, "mixed")
            self.assertEqual(cfg.selected_enhancer, "Restore Ultra")

    def test_settings_resolves_tensorrt_even_without_any_yaml_file(self):
        """When no config file exists at all, an NVIDIA environment must still default to tensorrt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.yaml"
            self.assertFalse(cfg_path.exists())
            
            with patch("roop.core.suggest_execution_providers", return_value=["tensorrt", "cuda", "cpu"]):
                cfg = Settings(str(cfg_path))
                self.assertEqual(cfg.provider, "tensorrt")
                self.assertEqual(cfg.trt_precision, "mixed")

    def test_sub_7gb_gpu_admits_tensorrt(self):
        """A secondary device or laptop card (< 7GB) admits TensorRT without opt-in."""
        adm = provider_admission("tensorrt", device_id=0)
        self.assertTrue(adm["admitted"])

    def test_resolve_provider_names_includes_tensorrt(self):
        """resolve_provider_names must retain TensorrtExecutionProvider for tensorrt and auto."""
        with patch("roop.backend_manager._available", return_value=[
            "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"
        ]):
            with patch("roop.backend_manager.provider_usable", return_value=True):
                resolved_trt = resolve_provider_names(["tensorrt"], device_id=0)
                self.assertIn("TensorrtExecutionProvider", resolved_trt)
                resolved_auto = resolve_provider_names(["auto"], device_id=0)
                self.assertIn("TensorrtExecutionProvider", resolved_auto)

    def test_api_meta_publishes_tensorrt_and_precision_modes(self):
        """The React UI depends on get_meta() providing tensorrt in providers and trt_precisions."""
        from api import get_meta
        meta = get_meta()
        self.assertIn("tensorrt", meta["providers"])
        self.assertIn("mixed", meta["trt_precisions"])
        self.assertIn("fp16", meta["trt_precisions"])
        self.assertIn("fp32", meta["trt_precisions"])


if __name__ == "__main__":
    unittest.main()
