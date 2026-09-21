import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys
from pathlib import Path

# `app.update_manager` is only importable when the REPOSITORY ROOT is on sys.path.
# Both documented suite commands must collect this module:
#   from app/ : python -m unittest discover -s tests -t . -p "test_*.py"
#   from root : python -m unittest discover -s app/tests -p "test_*.py"
# Only the second puts the repository root on sys.path, so under the
# app-relative command this module raised ImportError and unittest reported an
# ERROR instead of running its tests -- a whole module silently uncollected on
# one of the two commands the project documents.  Bootstrapping here makes the
# module self-sufficient under either.
# app.update_manager itself imports roop.* from app/, so app/ must be on the
# path too; otherwise this module only collects when an earlier test module
# happened to add it (it was order-dependent under pytest).
_ROOT = Path(__file__).resolve().parents[2]
for _path in (_ROOT, _ROOT / "app"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

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


def _manifest(state):
    return {
        "schema_version": update_manager.MANIFEST_SCHEMA_VERSION,
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

    def test_schema_1_manifest_is_unverified(self):
        # Schema 1 bound identity to a source_commit no committed file can
        # satisfy; the checker does not accept it.
        state = _state()
        manifest = _manifest(state)
        manifest["schema_version"] = 1
        manifest["source_commit"] = "a" * 40
        result = update_manager.evaluate_manifest(manifest, "a" * 40,
                                                   state, state["tracked_file_hashes"])
        self.assertEqual(result["classification"], "UNVERIFIED")

    def test_manifest_hash_not_in_candidate_tree_is_unverified(self):
        # The identity binding: every declared hash must be the fetched tree's.
        state = _state()
        manifest = _manifest(state)
        manifest["tracked_file_hashes"]["app/requirements.txt"] = "not-the-tree"
        result = update_manager.evaluate_manifest(manifest, "a" * 40,
                                                   state, state["tracked_file_hashes"])
        self.assertEqual(result["classification"], "UNVERIFIED")
        self.assertTrue(any("requirements.txt" in item for item in result["reasons"]))

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

    def test_manifest_integrity_is_the_gated_question(self):
        # Valid = evaluable by any machine: schema, activation and every hash
        # equal to the fetched tree. Compatibility with THIS machine is not
        # part of it, so a foreign provider still reads as gated.
        state = _state()
        manifest = _manifest(state)
        self.assertTrue(update_manager.manifest_integrity(manifest, state["tracked_file_hashes"])["valid"])
        stale = copy.deepcopy(state["tracked_file_hashes"])
        stale["torch.js"] = "tree-moved-on"
        verdict = update_manager.manifest_integrity(manifest, stale)
        self.assertFalse(verdict["valid"])
        self.assertTrue(verdict["present"])
        self.assertTrue(any("torch.js" in item for item in verdict["problems"]))
        missing = update_manager.manifest_integrity(None, stale)
        self.assertEqual((missing["present"], missing["valid"]), (False, False))

    def test_report_names_an_ungated_newer_commit(self):
        """Remote ahead, no manifest at the candidate: the report must say a
        newer commit EXISTS and that it is not gated -- with its date -- rather
        than collapsing to an anonymous UNVERIFIED."""
        state = _state()
        state.update({"branch": "main", "sha": "a" * 40, "remote": "https://example/repo",
                      "date": "2026-09-01T00:00:00+00:00"})
        candidate = "b" * 40

        def fake_run(command, cwd=None, check=True, timeout=None):
            if command[:2] == ["git", "ls-remote"]:
                return mock.Mock(returncode=0, stdout=f"{candidate}\trefs/heads/main\n")
            if command[:2] == ["git", "fetch"]:
                return mock.Mock(returncode=0, stdout="")
            if command[:2] == ["git", "merge-base"]:
                return mock.Mock(returncode=0, stdout="")
            raise AssertionError(f"unexpected command {command}")

        with mock.patch.object(update_manager, "_run", side_effect=fake_run), \
                mock.patch.object(update_manager, "_load_candidate_manifest", return_value=None), \
                mock.patch.object(update_manager, "_candidate_file_hashes", return_value={}), \
                mock.patch.object(update_manager, "_commit_date", return_value="2026-09-21T12:00:00+00:00"):
            report = update_manager._candidate_report(state)
        self.assertTrue(report["available"])
        self.assertEqual(report["classification"], "UNVERIFIED")
        self.assertEqual(report["candidate_sha"], candidate)
        self.assertEqual(report["candidate_date"], "2026-09-21T12:00:00+00:00")
        self.assertFalse(report["candidate_manifest"]["present"])
        self.assertFalse(report["candidate_manifest"]["valid"])

    def test_apply_channel_gated_reads_update_js(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "update.js").write_text('message: ["git pull"]', encoding="utf-8")
            with mock.patch.object(update_manager, "ROOT", root):
                self.assertFalse(update_manager.apply_channel_gated())
            (root / "update.js").write_text('message: ["python update_manager.py apply"]', encoding="utf-8")
            with mock.patch.object(update_manager, "ROOT", root):
                self.assertTrue(update_manager.apply_channel_gated())
            (root / "update.js").unlink()
            with mock.patch.object(update_manager, "ROOT", root):
                self.assertIsNone(update_manager.apply_channel_gated())


if __name__ == "__main__":
    unittest.main()
