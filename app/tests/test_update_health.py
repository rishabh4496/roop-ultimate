import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import update_health


class _Input:
    def __init__(self, name, shape, input_type):
        self.name = name
        self.shape = shape
        self.type = input_type


class _Session:
    def __init__(self):
        self.inputs = [
            _Input("image", [1, 3, 256, 256], "tensor(float)"),
            _Input("embedding", [1, 512], "tensor(float)"),
        ]

    def get_inputs(self):
        return self.inputs


class UpdateHealthTests(unittest.TestCase):
    def test_smoke_feeds_follow_actual_session_shapes(self):
        feeds = update_health._smoke_feeds(_Session(), 256)
        self.assertEqual(feeds["image"].shape, (1, 3, 256, 256))
        self.assertEqual(feeds["embedding"].shape, (1, 512))
        self.assertEqual(str(feeds["image"].dtype), "float32")

    def test_dynamic_dimensions_are_bounded_to_a_single_smoke_batch(self):
        session = _Session()
        session.inputs[0] = _Input("image", ["batch", 3, "height", "width"], "tensor(float16)")
        feeds = update_health._smoke_feeds(session, 128)
        self.assertEqual(feeds["image"].shape, (1, 3, 128, 128))
        self.assertEqual(str(feeds["image"].dtype), "float16")

    def test_dependency_check_fails_closed_for_missing_distribution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app").mkdir()
            (root / "app" / "requirements.txt").write_text(
                "definitely-not-installed-roop-test-package==1.0\n", encoding="utf-8")
            result = update_health._requirements_check(root)
        self.assertFalse(result["ok"])
        self.assertIn("definitely-not-installed-roop-test-package", result["missing"])

    def test_node_dependency_check_fails_closed_without_npm(self):
        with mock.patch.object(update_health.shutil, "which", return_value=None):
            result = update_health._node_dependencies_check(Path("source"), Path("data"))
        self.assertFalse(result["ok"])
        self.assertEqual(result["classification"], "UNVERIFIED")


if __name__ == "__main__":
    unittest.main()
