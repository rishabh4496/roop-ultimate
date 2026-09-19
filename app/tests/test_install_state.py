"""Transactional installation-state tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import install_state


class TestInstallState(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.paths = {
            "ROOT": root,
            "COMPLETE": root / ".pinokio-install-complete.json",
            "READY": root / ".pinokio-install-ready.json",
            "INCOMPLETE": root / ".pinokio-install-incomplete.json",
        }
        self.patches = [patch.object(install_state, name, value) for name, value in self.paths.items()]
        for item in self.patches:
            item.start()
        self.commit_patch = patch.object(install_state, "repository_commit", return_value="commit-test")
        self.commit_patch.start()

    def tearDown(self):
        self.commit_patch.stop()
        for item in reversed(self.patches):
            item.stop()
        self.tempdir.cleanup()

    def manifest(self):
        return {
            "schema": 3,
            "verification_passed": True,
            "installer_version": install_state.INSTALLER_VERSION,
            "python_version": "3.10",
            "python_executable": "python",
            "platform": "Windows",
            "architecture": "AMD64",
            "gpu_vendor": "nvidia",
            "gpu_name": "RTX test",
            "cuda_version": "12.8",
            "pytorch_version": "2.7.0",
            "onnxruntime_version": "1.23.2",
            "tensorrt_version": "10.9.0.34",
            "ort_provider_list": ["TensorrtExecutionProvider"],
            "tensorrt_session_test_result": "passed",
            "installation_timestamp": "now",
            "repository_commit": "commit-test",
            "dependency_verification_status": "passed",
            "runtime_verification_status": "passed",
            "binary_runtime_compatibility_status": "not_applicable",
            "binary_runtime_compatibility": {"status": "not_applicable"},
        }

    def test_begin_replaces_ready_state_and_records_stage(self):
        self.paths["COMPLETE"].write_text("old", encoding="utf-8")
        self.paths["READY"].write_text("old", encoding="utf-8")
        install_state.begin("python_requirements")
        state = json.loads(self.paths["INCOMPLETE"].read_text(encoding="utf-8"))
        self.assertEqual(state["state"], "in_progress")
        self.assertEqual(state["last_stage"], "python_requirements")
        self.assertFalse(self.paths["COMPLETE"].exists())
        self.assertFalse(self.paths["READY"].exists())

    def test_recover_records_interrupted_stage(self):
        install_state.begin("pytorch_gpu_runtime")
        install_state.recover()
        state = json.loads(self.paths["INCOMPLETE"].read_text(encoding="utf-8"))
        self.assertEqual(state["state"], "failed")
        self.assertEqual(state["failed_stage"], "pytorch_gpu_runtime")

    def test_commit_publishes_manifest_and_ready_state_last(self):
        install_state.begin("runtime_verification")
        manifest_path = Path(self.tempdir.name) / ".runtime-verification.json"
        manifest_path.write_text(json.dumps(self.manifest()), encoding="utf-8")
        install_state.commit(str(manifest_path))
        complete = json.loads(self.paths["COMPLETE"].read_text(encoding="utf-8"))
        ready = json.loads(self.paths["READY"].read_text(encoding="utf-8"))
        self.assertEqual(complete["state"], "complete")
        self.assertEqual(complete["schema"], 3)
        self.assertEqual(ready["state"], "ready")
        self.assertFalse(self.paths["INCOMPLETE"].exists())
        self.assertFalse(manifest_path.exists())

    def test_invalid_commit_keeps_installation_incomplete(self):
        install_state.begin("runtime_verification")
        manifest_path = Path(self.tempdir.name) / ".runtime-verification.json"
        manifest_path.write_text(json.dumps({"schema": 3}), encoding="utf-8")
        with self.assertRaises(install_state.InstallationStateError):
            install_state.commit(str(manifest_path))
        state = json.loads(self.paths["INCOMPLETE"].read_text(encoding="utf-8"))
        self.assertEqual(state["state"], "failed")
        self.assertEqual(state["failed_stage"], "manifest_commit")
        self.assertFalse(self.paths["COMPLETE"].exists())


if __name__ == "__main__":
    unittest.main()
