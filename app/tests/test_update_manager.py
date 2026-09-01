import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import update_manager


def _state():
    return {
        "python": "3.10.20",
        "platform": "win32",
        "provider": "cuda",
        "available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "hardware": {"profile": "rtx4070_12gb", "compute_capability": "8.9"},
        "runtime": {
            "torch": "2.7.0+cu128",
            "onnxruntime": "1.23.2",
            "tensorrt": "10.9.0.34",
            "cuda": "12.8",
            "ffmpeg": "ffmpeg version 8.1.2",
        },
        "tracked_file_hashes": {
            path: f"hash-{index}" for index, path in enumerate(update_manager.SENSITIVE_FILES)
        },
        "dirty": False,
    }


def _manifest(state, sha="a" * 40):
    return {
        "schema_version": 1,
        "source_commit": sha,
        "activation": "fast_forward_only",
        "compatibility": {
            "platforms": ["win32"],
            "python": {"min": "3.10", "max": "3.13"},
            "providers": ["cuda", "tensorrt", "cpu"],
            "hardware_profiles": ["rtx4070_12gb", "rtx3060_laptop_6gb"],
            "gpu_architectures": ["8.9", "8.6"],
            "application_contract": {
                "project_schema": 1,
                "processing_contract": "segmented-video-v1",
            },
            "application_requirements": {"policy": "unchanged"},
            "models": {"policy": "unchanged"},
            "runtime": {
                "torch": "==2.7.0+cu128",
                "onnxruntime": "==1.23.2",
                "tensorrt": "==10.9.0.34",
                "cuda": "==12.8",
            },
        },
        "critical_runtime_changes": [],
        "dependency_changes": [],
        "model_changes": [],
        "tracked_file_hashes": copy.deepcopy(state["tracked_file_hashes"]),
    }


class UpdateManagerTests(unittest.TestCase):
    def test_explicit_compatible_manifest_is_safe(self):
        state = _state()
        result = update_manager.evaluate_manifest(_manifest(state), "a" * 40,
                                                   state, state["tracked_file_hashes"])
        self.assertEqual(result["classification"], "SAFE")

    def test_unknown_provider_is_not_safe(self):
        state = _state()
        state["provider"] = None
        result = update_manager.evaluate_manifest(_manifest(state), "a" * 40,
                                                   state, state["tracked_file_hashes"])
        self.assertEqual(result["classification"], "UNVERIFIED")
        self.assertTrue(any("provider" in item for item in result["reasons"]))

    def test_critical_runtime_change_requires_review(self):
        state = _state()
        manifest = _manifest(state)
        manifest["critical_runtime_changes"] = ["onnxruntime 1.24"]
        result = update_manager.evaluate_manifest(manifest, "a" * 40,
                                                   state, state["tracked_file_hashes"])
        self.assertEqual(result["classification"], "REQUIRES REVIEW")

    def test_provider_mismatch_is_incompatible(self):
        state = _state()
        manifest = _manifest(state)
        manifest["compatibility"]["providers"] = ["cpu"]
        result = update_manager.evaluate_manifest(manifest, "a" * 40,
                                                   state, state["tracked_file_hashes"])
        self.assertEqual(result["classification"], "INCOMPATIBLE")

    def test_provider_unavailable_in_current_onnx_runtime_is_incompatible(self):
        state = _state()
        state["available_providers"] = ["CPUExecutionProvider"]
        result = update_manager.evaluate_manifest(_manifest(state), "a" * 40,
                                                   state, state["tracked_file_hashes"])
        self.assertEqual(result["classification"], "INCOMPATIBLE")

    def test_cuda_mismatch_is_incompatible(self):
        state = _state()
        state["runtime"]["cuda"] = "12.4"
        result = update_manager.evaluate_manifest(_manifest(state), "a" * 40,
                                                   state, state["tracked_file_hashes"])
        self.assertEqual(result["classification"], "INCOMPATIBLE")

    def test_missing_mandatory_hardware_profile_is_incompatible(self):
        state = _state()
        manifest = _manifest(state)
        manifest["compatibility"]["hardware_profiles"] = ["rtx4070_12gb"]
        result = update_manager.evaluate_manifest(manifest, "a" * 40,
                                                   state, state["tracked_file_hashes"])
        self.assertEqual(result["classification"], "INCOMPATIBLE")

    def test_missing_mandatory_gpu_architecture_is_incompatible(self):
        state = _state()
        manifest = _manifest(state)
        manifest["compatibility"]["gpu_architectures"] = ["8.9"]
        result = update_manager.evaluate_manifest(manifest, "a" * 40,
                                                   state, state["tracked_file_hashes"])
        self.assertEqual(result["classification"], "INCOMPATIBLE")

    def test_missing_model_policy_is_unverified(self):
        state = _state()
        manifest = _manifest(state)
        del manifest["compatibility"]["models"]
        result = update_manager.evaluate_manifest(manifest, "a" * 40,
                                                   state, state["tracked_file_hashes"])
        self.assertEqual(result["classification"], "UNVERIFIED")

    def test_sensitive_dependency_change_requires_review(self):
        state = _state()
        manifest = _manifest(state)
        candidate = copy.deepcopy(state["tracked_file_hashes"])
        candidate["app/requirements.txt"] = "new-hash"
        manifest["tracked_file_hashes"] = candidate
        result = update_manager.evaluate_manifest(manifest, "a" * 40,
                                                   state, candidate)
        self.assertEqual(result["classification"], "REQUIRES REVIEW")
        self.assertTrue(any("requirements.txt" in item for item in result["reasons"]))

    def test_persisted_work_requires_review(self):
        state = _state()
        state["active_work"] = ["project demo is PAUSED"]
        result = update_manager.evaluate_manifest(_manifest(state), "a" * 40,
                                                   state, state["tracked_file_hashes"])
        self.assertEqual(result["classification"], "REQUIRES REVIEW")
        self.assertTrue(any("PAUSED" in item for item in result["reasons"]))

    def test_manifest_commit_mismatch_is_unverified(self):
        state = _state()
        result = update_manager.evaluate_manifest(_manifest(state, "b" * 40), "a" * 40,
                                                   state, state["tracked_file_hashes"])
        self.assertEqual(result["classification"], "UNVERIFIED")

    def test_invalid_manifest_is_unverified(self):
        result = update_manager.evaluate_manifest(None, "a" * 40, _state())
        self.assertEqual(result["classification"], "UNVERIFIED")

    def test_snapshot_contains_git_identity_and_config_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "app" / "config.yaml"
            config.parent.mkdir()
            config.write_text("provider: cuda\n", encoding="utf-8")
            current = _state()
            current["sha"] = "a" * 40
            with mock.patch.object(update_manager, "ROOT", root), \
                    mock.patch.object(update_manager, "SNAPSHOT_ROOT", root / "snapshots"), \
                    mock.patch.object(update_manager, "_run", return_value=None):
                snapshot = update_manager._create_snapshot(current)
            metadata = __import__("json").loads(
                (snapshot / "snapshot.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["prior_commit"], "a" * 40)
            self.assertTrue((snapshot / "config.yaml").is_file())
            self.assertEqual(metadata["files"]["app/config.yaml"]["sha256"],
                             update_manager._sha256_file(config))

    def test_apply_does_not_report_success_before_post_health(self):
        candidate = "b" * 40
        report = {"available": True, "classification": "SAFE",
                  "candidate_ref": "origin/main", "candidate_sha": candidate,
                  "current": {"sha": "a" * 40}}
        fake_snapshot = Path("snapshot-under-test")
        health_failure = {"healthy": False, "checks": [{"name": "inference", "ok": False}]}
        with mock.patch.object(update_manager, "check", return_value=report), \
                mock.patch.object(update_manager, "_transaction"), \
                mock.patch.object(update_manager, "_run_health",
                                  side_effect=[{"healthy": True}, health_failure]), \
                mock.patch.object(update_manager, "_create_snapshot", return_value=fake_snapshot), \
                mock.patch.object(update_manager, "_stage_candidate", return_value={"health": {"healthy": True}}), \
                mock.patch.object(update_manager, "_record_snapshot"), \
                mock.patch.object(update_manager, "_rollback", return_value={"ok": True, "detail": "restored"}) as rollback_mock, \
                mock.patch.object(update_manager, "_run", return_value=mock.Mock(returncode=0)), \
                mock.patch.object(update_manager, "_git", side_effect=["a" * 40, "", "b" * 40]):
            result = update_manager.apply()
        self.assertEqual(result, 3)
        rollback_mock.assert_called_once_with(fake_snapshot, "a" * 40)


if __name__ == "__main__":
    unittest.main()
