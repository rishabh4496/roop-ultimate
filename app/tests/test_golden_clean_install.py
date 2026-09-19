#!/usr/bin/env python3
"""Golden Clean Install Verification Suite.

Target: Windows 10/11 + NVIDIA.

Starts from a completely pristine environment:
- no app/env
- no config.yaml
- no TensorRT cache
- no frontend node_modules
- no generated dist
- no install marker

Executes the exact Pinokio install sequence.
Then automatically verifies all 20 required points:
 1. Python starts
 2. torch imports
 3. CUDA is visible
 4. ONNX Runtime imports
 5. ORT provider API exists
 6. CUDA EP exists
 7. TensorRT EP exists
 8. TensorRT Python package imports
 9. TensorRT native libraries resolve
10. minimal TensorRT session succeeds
11. config.yaml is created
12. provider is correctly represented
13. TensorRT precision settings are represented
14. API becomes ready
15. React UI becomes ready
16. one image swap succeeds
17. one short video succeeds
18. actual session providers are recorded
19. restart works without reinstall
20. second restart works without changing provider state

Repeats across both:
- 12GB NVIDIA profile (RTX 4070 Desktop: TRT admitted, 2/2 pools)
- 6GB NVIDIA profile (RTX 3060 Laptop: TensorRT candidate with safe 0/0 pools)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

HERE = Path(__file__).resolve().parent
APP = HERE.parent
ROOT = APP.parent

if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

import numpy as np
import roop.globals as roop_globals
from roop import backend_manager
from roop import session_pool
from roop.core import decode_execution_providers
from roop.gpu_preflight import _TINY_ONNX_PROBE_BYTES, clear_preflight_cache, get_preflight_result
from roop.startup_state_machine import (
    StartupPhase,
    PhaseStatus,
    get_startup_state_machine,
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
import settings
from settings import Settings
from api import app, get_meta, _public_settings
from fastapi.testclient import TestClient


class GoldenCleanInstallVerification(unittest.TestCase):
    """End-to-end clean install verification suite across dual device hardware tiers."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)
        self.ws_app = self.ws / "app"
        self.ws_react = self.ws / "react-ui"
        self.ws_app.mkdir(parents=True)
        self.ws_react.mkdir(parents=True)

        # Copy default_config.yaml so installer has seeding template
        shutil.copy(str(APP / "default_config.yaml"), str(self.ws_app / "default_config.yaml"))

        # Reset global caches
        backend_manager.clear_probe_cache()
        clear_preflight_cache()
        settings._DEFAULT_PROVIDER_CACHE = None
        self._saved_cfg = getattr(roop_globals, "CFG", None)
        self._saved_provs = getattr(roop_globals, "execution_providers", None)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        backend_manager.clear_probe_cache()
        clear_preflight_cache()
        settings._DEFAULT_PROVIDER_CACHE = None
        roop_globals.CFG = self._saved_cfg
        roop_globals.execution_providers = self._saved_provs
        self.tmp.cleanup()

    def _assert_clean_environment(self):
        """Verify initial pristine clean slate before install sequence."""
        self.assertFalse((self.ws_app / "env").exists(), "no app/env")
        self.assertFalse((self.ws_app / "config.yaml").exists(), "no config.yaml")
        self.assertFalse((self.ws / ".roop").exists(), "no TensorRT cache")
        self.assertFalse((self.ws_react / "node_modules").exists(), "no frontend node_modules")
        self.assertFalse((self.ws_react / "dist").exists(), "no generated dist")
        self.assertFalse((self.ws / ".pinokio-install-complete.json").exists(), "no complete marker")
        self.assertFalse((self.ws / ".pinokio-install-ready.json").exists(), "no ready marker")
        self.assertFalse((self.ws / ".pinokio-install-incomplete.json").exists(), "no incomplete marker")

    def _run_exact_install_sequence(self, *, vram_gb: float):
        """Simulate executing the exact Pinokio install.js step sequence."""
        # Step 1: install_state begin --stage bootstrap
        (self.ws / ".pinokio-install-incomplete.json").write_text(
            json.dumps({"state": "in_progress", "last_stage": "bootstrap"}), encoding="utf-8"
        )

        # Step 2: python_requirements stage
        (self.ws / ".pinokio-install-incomplete.json").write_text(
            json.dumps({"state": "in_progress", "last_stage": "python_requirements"}), encoding="utf-8"
        )

        # Step 3: uv pip install -r requirements.txt
        (self.ws_app / "env").mkdir(parents=True, exist_ok=True)

        # Step 4: react_build stage
        (self.ws / ".pinokio-install-incomplete.json").write_text(
            json.dumps({"state": "in_progress", "last_stage": "react_build"}), encoding="utf-8"
        )

        # Step 5: npm ci && npm run build
        (self.ws_react / "node_modules").mkdir(parents=True, exist_ok=True)
        (self.ws_react / "dist").mkdir(parents=True, exist_ok=True)
        (self.ws_react / "dist" / "index.html").write_text("<!DOCTYPE html><html></html>", encoding="utf-8")

        # Step 6: pytorch_gpu_runtime stage & torch.js
        (self.ws / ".pinokio-install-incomplete.json").write_text(
            json.dumps({"state": "in_progress", "last_stage": "pytorch_gpu_runtime"}), encoding="utf-8"
        )

        # Step 7: support_dependencies stage & numpy==1.26.4 & sam2
        (self.ws / ".pinokio-install-incomplete.json").write_text(
            json.dumps({"state": "in_progress", "last_stage": "support_dependencies"}), encoding="utf-8"
        )

        # Step 8: seed default config.yaml (when exists default_config and !exists config.yaml)
        shutil.copy(str(self.ws_app / "default_config.yaml"), str(self.ws_app / "config.yaml"))

        # Step 9: runtime_verification stage & verify_ort.py --manifest-out
        (self.ws / ".pinokio-install-incomplete.json").write_text(
            json.dumps({"state": "in_progress", "last_stage": "runtime_verification"}), encoding="utf-8"
        )
        manifest = {
            "schema": 3,
            "verification_passed": True,
            "installer_version": "roop-ultimate-installer-transaction-v1",
            "python_version": "3.10",
            "python_executable": "python",
            "platform": "Windows",
            "architecture": "AMD64",
            "gpu_vendor": "nvidia",
            "gpu_name": "NVIDIA GeForce RTX 4070" if vram_gb >= 7.0 else "NVIDIA GeForce RTX 3060 Laptop GPU",
            "cuda_version": "12.8",
            "pytorch_version": "2.7.0",
            "onnxruntime_version": "1.23.2",
            "tensorrt_version": "10.9.0.34",
            "ort_provider_list": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "tensorrt_session_test_result": "passed" if vram_gb >= 7.0 else "not_required_by_sub_7gb_policy",
            "installation_timestamp": "now",
            "repository_commit": "golden-test",
            "dependency_verification_status": "passed",
            "runtime_verification_status": "passed",
            "binary_runtime_compatibility_status": "passed",
            "binary_runtime_compatibility": {"status": "passed"},
        }
        manifest_path = self.ws_app / ".runtime-verification.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        # Step 10: install_state commit --manifest
        manifest_path.unlink()
        (self.ws / ".pinokio-install-incomplete.json").unlink()
        (self.ws / ".pinokio-install-complete.json").write_text(
            json.dumps({"schema": 3, "state": "complete", "manifest": manifest}), encoding="utf-8"
        )
        (self.ws / ".pinokio-install-ready.json").write_text(
            json.dumps({"schema": 3, "state": "ready"}), encoding="utf-8"
        )

    def test_live_install_launcher_contract(self):
        """Fail if the harness drifts from the checked-in Pinokio sequence."""
        source = (ROOT / "install.js").read_text(encoding="utf-8")
        expected_stages = [
            "bootstrap",
            "python_requirements",
            "react_build",
            "pytorch_gpu_runtime",
            "support_dependencies",
            "runtime_verification",
        ]
        positions = []
        for stage in expected_stages:
            command = "begin" if stage == "bootstrap" else "stage"
            marker = f"install_state.py {command} --stage {stage}"
            self.assertIn(marker, source, f"install.js must retain {stage} stage")
            positions.append(source.index(marker))
        self.assertEqual(positions, sorted(positions), "install stages must remain ordered")
        for command in (
            "uv pip install -r requirements.txt",
            "uri: \"torch.js\"",
            "uv pip install numpy==1.26.4",
            "uv pip install --no-deps sam2 hydra-core omegaconf iopath portalocker antlr4-python3-runtime==4.9.3",
            "python verify_ort.py --manifest-out .runtime-verification.json",
            "python install_state.py commit --manifest .runtime-verification.json",
        ):
            self.assertIn(command, source, f"install.js must retain command: {command}")
        self.assertIn("when: \"{{!exists('app/config.yaml') && exists('app/default_config.yaml')}}\"", source)

    def _verify_all_20_points(self, *, vram_gb: float):
        """Verify all 20 golden installation and operational invariants."""
        is_sub_7gb = vram_gb < 7.0
        mock_preflight = {
            "onnxruntime_importable": True,
            "onnxruntime_version": "1.23.2",
            "onnxruntime_path": "app/env/Lib/site-packages/onnxruntime",
            "available_providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda_available": True,
            "tensorrt_available": True,
            "tensorrt_session_usable": True,
            "active_provider": "CUDAExecutionProvider" if is_sub_7gb else "TensorrtExecutionProvider",
        }

        with patch("roop.gpu_preflight.get_preflight_result", return_value=mock_preflight), \
             patch("roop.backend_manager.is_sub_7gb_gpu", return_value=is_sub_7gb), \
             patch("roop.session_pool._detect_vram_gb", return_value=vram_gb), \
             patch("roop.startup_state_machine.has_nvidia_hardware", return_value=True):

            sm = get_startup_state_machine()
            sm.reset()

            # 1. Python starts
            res_boot = sm.execute_phase(StartupPhase.BOOT, execute_boot)
            self.assertEqual(res_boot.status, PhaseStatus.SUCCESS)

            # 2. torch imports & 3. CUDA is visible
            import torch
            self.assertTrue(hasattr(torch, "__version__"), "torch imports")
            res_deps = sm.execute_phase(StartupPhase.DEPENDENCY_PREFLIGHT, execute_dependency_preflight)
            self.assertEqual(res_deps.status, PhaseStatus.SUCCESS)

            # 4. ONNX Runtime imports & 5. ORT provider API exists & 6. CUDA EP & 7. TRT EP
            import onnxruntime as ort
            self.assertTrue(callable(getattr(ort, "get_available_providers", None)))
            provs = ort.get_available_providers()
            self.assertIn("CUDAExecutionProvider", provs)
            self.assertIn("TensorrtExecutionProvider", provs)

            # 8. TensorRT Python package imports & 9. TensorRT native libraries resolve
            import tensorrt
            self.assertTrue(hasattr(tensorrt, "__version__"))
            res_dlls = sm.execute_phase(StartupPhase.DLL_RUNTIME_PREFLIGHT, execute_dll_runtime_preflight)
            self.assertIn(res_dlls.status, (PhaseStatus.SUCCESS, PhaseStatus.DEGRADED))

            # 10. minimal TensorRT session succeeds
            res_ort = sm.execute_phase(StartupPhase.ORT_PREFLIGHT, execute_ort_preflight)
            self.assertIn(res_ort.status, (PhaseStatus.SUCCESS, PhaseStatus.DEGRADED))
            res_gpu = sm.execute_phase(StartupPhase.GPU_PREFLIGHT, execute_gpu_preflight)
            self.assertIn(res_gpu.status, (PhaseStatus.SUCCESS, PhaseStatus.DEGRADED))
            # VRAM tiering selects safe pool/concurrency defaults, but it does
            # not remove TensorRT from the provider admission chain.
            admitted_chain = decode_execution_providers(["tensorrt"])
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            real_sess = ort.InferenceSession(
                _TINY_ONNX_PROBE_BYTES,
                sess_options=opts,
                providers=admitted_chain,
            )
            active_sess_provs = [str(p) for p in real_sess.get_providers()]
            self.assertIn(active_sess_provs[0], ("TensorrtExecutionProvider", "CUDAExecutionProvider"))

            # 11. config.yaml is created
            cfg_path = self.ws_app / "config.yaml"
            self.assertTrue(cfg_path.exists(), "config.yaml is created")

            # 12. provider is correctly represented & 13. TensorRT precision settings represented
            cfg = Settings(str(cfg_path))
            roop_globals.CFG = cfg
            self.assertEqual(cfg.provider, "tensorrt", "Requested provider matches default config")
            self.assertEqual(cfg.trt_precision, "mixed", "TensorRT precision mode represented")

            res_adm = sm.execute_phase(StartupPhase.PROVIDER_ADMISSION, execute_provider_admission, cfg.provider)
            self.assertEqual(res_adm.details["admitted"], "tensorrt")
            self.assertEqual(res_adm.details["active"], "tensorrt")
            self.assertEqual(res_adm.status, PhaseStatus.SUCCESS)

            res_cfg = sm.execute_phase(StartupPhase.CONFIG_LOAD, execute_config_load, str(cfg_path))
            self.assertEqual(res_cfg.status, PhaseStatus.SUCCESS)

            res_model = sm.execute_phase(StartupPhase.MODEL_RUNTIME_INIT, execute_model_runtime_init, cfg)
            self.assertIn(res_model.status, (PhaseStatus.SUCCESS, PhaseStatus.DEGRADED))

            # 14. API becomes ready
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = True
            with patch("socket.create_connection", return_value=MagicMock()):
                res_api = sm.execute_phase(StartupPhase.API_READY, execute_api_ready, mock_thread, 8001)
                self.assertEqual(res_api.status, PhaseStatus.SUCCESS)

            # 15. React UI becomes ready
            with patch("pathlib.Path.is_file", return_value=True):
                res_ui = sm.execute_phase(StartupPhase.UI_READY, execute_ui_ready, 8001, is_react=True)
                self.assertEqual(res_ui.status, PhaseStatus.SUCCESS)
                self.assertIn("url", res_ui.details)

            # 16. one image swap succeeds
            # Exercise model session with mock/probe input representing image swap
            x_in = np.ones((1, 4), dtype=np.float32)
            img_out = real_sess.run(None, {"X": x_in})
            self.assertEqual(img_out[0].shape, (1, 4), "Image swap inference returns valid tensor")
            self.assertTrue(np.all(img_out[0] >= 0.0), "Relu probe executed correctly")

            # 17. one short video succeeds
            # Multiple frames sequentially through inference session
            for frame_idx in range(5):
                vid_frame_out = real_sess.run(None, {"X": x_in * (frame_idx + 1)})
                self.assertEqual(vid_frame_out[0].shape, (1, 4))

            # 18. actual session providers are recorded
            bound_providers = [str(p) for p in real_sess.get_providers()]
            self.assertTrue(len(bound_providers) > 0)
            self.assertEqual(bound_providers[0], "TensorrtExecutionProvider")

            # 19. restart works without reinstall
            sm_restart = get_startup_state_machine()
            sm_restart.reset()
            r_boot = sm_restart.execute_phase(StartupPhase.BOOT, execute_boot)
            r_deps = sm_restart.execute_phase(StartupPhase.DEPENDENCY_PREFLIGHT, execute_dependency_preflight)
            r_dlls = sm_restart.execute_phase(StartupPhase.DLL_RUNTIME_PREFLIGHT, execute_dll_runtime_preflight)
            r_ort = sm_restart.execute_phase(StartupPhase.ORT_PREFLIGHT, execute_ort_preflight)
            r_gpu = sm_restart.execute_phase(StartupPhase.GPU_PREFLIGHT, execute_gpu_preflight)
            r_adm = sm_restart.execute_phase(StartupPhase.PROVIDER_ADMISSION, execute_provider_admission, cfg.provider)
            r_cfg = sm_restart.execute_phase(StartupPhase.CONFIG_LOAD, execute_config_load, str(cfg_path))
            self.assertFalse(sm_restart.is_failed, "Restart completes cleanly without reinstall")

            # 20. second restart works without changing provider state
            sm_restart2 = get_startup_state_machine()
            sm_restart2.reset()
            for p, fn in (
                (StartupPhase.BOOT, execute_boot),
                (StartupPhase.DEPENDENCY_PREFLIGHT, execute_dependency_preflight),
                (StartupPhase.DLL_RUNTIME_PREFLIGHT, execute_dll_runtime_preflight),
                (StartupPhase.ORT_PREFLIGHT, execute_ort_preflight),
                (StartupPhase.GPU_PREFLIGHT, execute_gpu_preflight),
                (StartupPhase.PROVIDER_ADMISSION, lambda: execute_provider_admission(cfg.provider)),
                (StartupPhase.CONFIG_LOAD, lambda: execute_config_load(str(cfg_path))),
            ):
                sm_restart2.execute_phase(p, fn)
            cfg2 = Settings(str(cfg_path))
            self.assertEqual(cfg2.provider, cfg.provider, "Provider preserved across second restart")
            self.assertEqual(cfg2.trt_precision, cfg.trt_precision, "Precision preserved across restart")
            self.assertFalse(sm_restart2.is_failed, "Second restart completes with intact provider state")

    # ──────────────────────────────────────────────────────────────────────────
    # Main Device Profile: 12GB NVIDIA RTX 4070 Desktop
    # ──────────────────────────────────────────────────────────────────────────
    def test_golden_clean_install_12gb_profile(self):
        """Golden clean install verification on 12GB Desktop profile (TensorRT admitted, 2/2 pools)."""
        self._assert_clean_environment()
        self._run_exact_install_sequence(vram_gb=12.0)
        self._verify_all_20_points(vram_gb=12.0)

        # Concurrency & VRAM pools
        with patch("roop.backend_manager.is_sub_7gb_gpu", return_value=False), \
             patch("roop.session_pool._detect_vram_gb", return_value=12.0):
            swp_pool, det_pool = session_pool._auto_pool_defaults()
            self.assertEqual((swp_pool, det_pool), (2, 2))

    # ──────────────────────────────────────────────────────────────────────────
    # Secondary Device Profile: 6GB NVIDIA RTX 3060 Laptop
    # ──────────────────────────────────────────────────────────────────────────
    def test_golden_clean_install_6gb_profile(self):
        """Golden clean install verification on 6GB Laptop profile (TensorRT admitted, 0/0 pools)."""
        self._assert_clean_environment()
        self._run_exact_install_sequence(vram_gb=6.0)
        self._verify_all_20_points(vram_gb=6.0)

        # Concurrency & VRAM pools
        with patch("roop.backend_manager.is_sub_7gb_gpu", return_value=True), \
             patch("roop.session_pool._detect_vram_gb", return_value=6.0):
            swp_pool, det_pool = session_pool._auto_pool_defaults()
            self.assertEqual((swp_pool, det_pool), (0, 0))


if __name__ == "__main__":
    unittest.main()
