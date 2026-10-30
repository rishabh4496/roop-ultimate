"""Real filesystem tests. No inference or protected-write check is mocked."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

AUDIT = Path(__file__).absolute().parents[1] / "audit"
sys.path.insert(0, str(AUDIT))
from isolation import AuditError, inventory, safe_path, verify, write_json


class IsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path(__file__).absolute().parents[1] / ".runtime" / "tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temporary.name)
        (self.root / "app").mkdir()
        (self.root / "app" / "legacy.py").write_bytes(b"original")
        (self.root / ".gitignore").write_bytes(b"models/\n")
        self.baseline = inventory(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_unchanged_tree_passes(self) -> None:
        self.assertTrue(verify(self.root, self.baseline)["ok"])

    def test_equal_size_mutation_fails_by_hash(self) -> None:
        (self.root / "app" / "legacy.py").write_bytes(b"modified")
        result = verify(self.root, self.baseline)
        self.assertFalse(result["ok"])
        self.assertEqual(result["violations"][0]["status"], "HASH_MISMATCH")

    def test_deleted_file_fails(self) -> None:
        (self.root / "app" / "legacy.py").unlink()
        self.assertEqual(verify(self.root, self.baseline)["violations"][0]["status"], "MISSING")

    def test_new_protected_file_fails(self) -> None:
        (self.root / "app" / "new.py").write_bytes(b"unauthorized")
        self.assertEqual(verify(self.root, self.baseline)["violations"][0]["status"], "ADDED")

    def test_new_empty_protected_directory_fails(self) -> None:
        (self.root / "app" / "new_empty_directory").mkdir()
        self.assertEqual(verify(self.root, self.baseline)["violations"][0]["status"], "DIRECTORY_ADDED")

    def test_removed_empty_protected_directory_fails(self) -> None:
        empty = self.root / "app" / "empty"
        empty.mkdir()
        baseline = inventory(self.root)
        empty.rmdir()
        self.assertEqual(verify(self.root, baseline)["violations"][0]["status"], "DIRECTORY_MISSING")

    def test_all_authorized_roots_are_excluded(self) -> None:
        for relative in ("roop_3d/audit/test.json", "roop_3d_ui/src/test.json", ".git/info/exclude"):
            write_json(self.root, relative, {"ok": True})
        (self.root / "start_roop3d.js").write_bytes(b"// allowed")
        self.assertTrue(verify(self.root, self.baseline)["ok"])

    def test_prefix_siblings_and_legacy_writes_rejected(self) -> None:
        for value in ("app/x.py", "react-ui/x", "roop_3d_evil/x", "start_roop3d.js/child", ".git/config"):
            with self.subTest(path=value), self.assertRaises(AuditError):
                safe_path(self.root, value, write=True)

    def test_traversal_absolute_drives_and_streams_rejected(self) -> None:
        for value in ("../x", "roop_3d/../app/x", "/absolute", "C:/absolute", "C:relative",
                      "roop_3d//x", "./roop_3d/x", "roop_3d/x:stream", "\\\\host\\share", "",
                      "roop_3d/.. /app/x", "roop_3d/dir./x", "roop_3d/dir /x"):
            with self.subTest(path=value), self.assertRaises(AuditError):
                safe_path(self.root, value, write=True)

    def test_empty_manifest_fails_closed(self) -> None:
        with self.assertRaises(AuditError):
            verify(self.root, {"files": []})

    def test_duplicate_case_insensitive_paths_rejected(self) -> None:
        manifest = copy.deepcopy(self.baseline)
        duplicate = dict(manifest["files"][0])
        duplicate["path"] = duplicate["path"].upper()
        manifest["files"].append(duplicate)
        with self.assertRaises(AuditError):
            verify(self.root, manifest)

    def test_manifest_traversal_rejected(self) -> None:
        manifest = copy.deepcopy(self.baseline)
        manifest["files"][0]["path"] = "../outside"
        with self.assertRaises(AuditError):
            verify(self.root, manifest)

    def test_json_writer_is_consumed(self) -> None:
        write_json(self.root, "roop_3d/audit/result.json", self.baseline)
        data = json.loads((self.root / "roop_3d/audit/result.json").read_text("utf-8"))
        self.assertTrue(verify(self.root, data)["ok"])

    def test_atomic_write_does_not_mutate_a_legacy_hardlink_target(self) -> None:
        (self.root / "roop_3d").mkdir()
        os.link(self.root / "app/legacy.py", self.root / "roop_3d/linked.json")
        write_json(self.root, "roop_3d/linked.json", {"replacement": True})
        self.assertEqual((self.root / "app/legacy.py").read_bytes(), b"original")
        self.assertTrue(verify(self.root, self.baseline)["ok"])

    def prepare_cli(self) -> Path:
        audit = self.root / "roop_3d/audit"
        audit.mkdir(parents=True)
        for name in ("isolation.py", "runtime_policy.py", "stage00.py", "verify_isolation.py"):
            (audit / name).write_bytes((AUDIT / name).read_bytes())
        write_json(self.root, "roop_3d/audit/protected_paths.json", self.baseline)
        return audit / "verify_isolation.py"

    def test_cli_requires_comprehensive_baseline(self) -> None:
        script = self.prepare_cli()
        result = subprocess.run([sys.executable, "-B", str(script)], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("BASELINE_MISSING", result.stdout)

    def test_cli_reads_and_writes_real_gate_report(self) -> None:
        script = self.prepare_cli()
        write_json(self.root, "roop_3d/audit/protected_all_paths.json", self.baseline)
        result = subprocess.run([sys.executable, "-B", str(script), "--report"], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((self.root / "roop_3d/audit/stage00_verification.json").read_text("utf-8"))
        self.assertTrue(report["ok"])
        self.assertTrue(report["comprehensive"]["detect_additions"])

    def test_cli_fails_if_new_protected_file_appears(self) -> None:
        script = self.prepare_cli()
        write_json(self.root, "roop_3d/audit/protected_all_paths.json", self.baseline)
        (self.root / "app/injected.py").write_bytes(b"unauthorized")
        result = subprocess.run([sys.executable, "-B", str(script)], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('"ADDED"', result.stdout)

    def test_cli_cannot_overwrite_existing_snapshot(self) -> None:
        script = self.prepare_cli()
        write_json(self.root, "roop_3d/audit/protected_all_paths.json", self.baseline)
        path = self.root / "roop_3d/audit/protected_all_paths.json"
        before = path.read_bytes()
        result = subprocess.run([sys.executable, "-B", str(script), "--snapshot"], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("BASELINE_EXISTS", result.stdout)
        self.assertEqual(path.read_bytes(), before)

    def test_real_junction_or_symlink_cannot_write_into_legacy(self) -> None:
        (self.root / "roop_3d").mkdir()
        junction = self.root / "roop_3d" / "escape"
        if os.name == "nt":
            import _winapi
            _winapi.CreateJunction(str(self.root / "app"), str(junction))
        else:
            junction.symlink_to(self.root / "app", target_is_directory=True)
        try:
            with self.assertRaises(AuditError):
                write_json(self.root, "roop_3d/escape/injected.json", {})
            self.assertFalse((self.root / "app/injected.json").exists())
        finally:
            if os.name == "nt":
                os.rmdir(junction)
            else:
                junction.unlink()

    def test_protected_junction_replacement_cannot_hide_legacy(self) -> None:
        legacy = self.root / "app" / "legacy.py"
        legacy.unlink()
        (self.root / "app").rmdir()
        target = self.root / "roop_3d" / "target"
        target.mkdir(parents=True)
        (target / "legacy.py").write_bytes(b"original")
        junction = self.root / "app"
        if os.name == "nt":
            import _winapi
            _winapi.CreateJunction(str(target), str(junction))
        else:
            junction.symlink_to(target, target_is_directory=True)
        try:
            self.assertFalse(verify(self.root, self.baseline)["ok"])
        finally:
            if os.name == "nt":
                os.rmdir(junction)
            else:
                junction.unlink()


if __name__ == "__main__":
    unittest.main()
