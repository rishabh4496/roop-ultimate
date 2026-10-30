from pathlib import Path
import sys
import unittest
import tempfile

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "audit"))
from isolation import inventory
from runtime_policy import verify_policy


class RuntimePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path(__file__).absolute().parents[1] / ".runtime/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temporary.name)
        (self.root / "app").mkdir()
        (self.root / "app/legacy.py").write_bytes(b"original")
        self.baseline = inventory(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_runtime_activity_is_reported_without_rewriting_source_baseline(self) -> None:
        original = (self.root / "app/legacy.py").read_bytes()
        output = self.root / "app/output"
        output.mkdir()
        (output / "render.mp4").write_bytes(b"runtime data")
        result = verify_policy(self.root, self.baseline)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["runtime_activity"]["observed_files"], 1)
        self.assertEqual((self.root / "app/legacy.py").read_bytes(), original)

    def test_runtime_policy_cannot_hide_source_changes(self) -> None:
        (self.root / "app/legacy.py").write_bytes(b"modified")
        self.assertFalse(verify_policy(self.root, self.baseline)["ok"])

    def test_runtime_prefix_sibling_is_still_protected(self) -> None:
        (self.root / "app/output_evil").mkdir()
        (self.root / "app/output_evil/code.py").write_bytes(b"forbidden")
        self.assertFalse(verify_policy(self.root, self.baseline)["ok"])

    def test_only_explicit_history_files_are_runtime_not_config(self) -> None:
        (self.root / "app/run_history.json").write_bytes(b"runtime")
        assert verify_policy(self.root, self.baseline)["ok"]
        (self.root / "app/config.yaml").write_bytes(b"protected")
        assert not verify_policy(self.root, self.baseline)["ok"]

    def test_profile_json_is_runtime_but_model_weights_remain_protected(self) -> None:
        profiles = self.root / "app/models/runtime_profiles"
        profiles.mkdir(parents=True)
        # The models directory itself is part of the protected directory baseline.
        baseline = inventory(self.root)
        (profiles / "runtime.json").write_bytes(b"{}")
        assert verify_policy(self.root, baseline)["ok"]
        (profiles / "weights.onnx").write_bytes(b"model")
        assert not verify_policy(self.root, baseline)["ok"]
