import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys
from pathlib import Path

# `app.update_health` is only importable when the REPOSITORY ROOT is on sys.path.
# Both documented suite commands must collect this module:
#   from app/ : python -m unittest discover -s tests -t . -p "test_*.py"
#   from root : python -m unittest discover -s app/tests -p "test_*.py"
# Only the second puts the repository root on sys.path, so under the
# app-relative command this module raised ImportError and unittest reported an
# ERROR instead of running its tests -- a whole module silently uncollected on
# one of the two commands the project documents.  Bootstrapping here makes the
# module self-sufficient under either.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
        """Genuinely unresolvable npm must still fail closed.

        Patching `shutil.which` alone is no longer sufficient: npm is now also
        looked for inside the Pinokio toolchain, because a bare PATH lookup
        MISSED an installed, working npm and reported the whole health check
        unhealthy on a healthy machine. The resolver is patched instead, which
        is what "npm cannot be found" now means.
        """
        with mock.patch.object(update_health, "_resolve_npm", return_value=None):
            result = update_health._node_dependencies_check(Path("source"), Path("data"))
        self.assertFalse(result["ok"])
        self.assertEqual(result["classification"], "UNVERIFIED")

    def test_npm_is_resolved_beyond_the_callers_path(self):
        """The regression: PATH misses it, the Pinokio toolchain has it."""
        import os
        home = os.path.join(os.sep, "fake-pinokio")
        expected = os.path.join(home, "bin", "miniforge", "npm.cmd")
        with mock.patch.object(update_health.shutil, "which", return_value=None),                 mock.patch.dict(os.environ, {"PINOKIO_HOME": home}, clear=False),                 mock.patch.object(update_health.os.path, "isdir",
                                  lambda p: p == home),                 mock.patch.object(update_health.os.path, "isfile",
                                  lambda p: p == expected):
            self.assertEqual(update_health._resolve_npm(), expected)

    def test_a_path_npm_still_wins(self):
        with mock.patch.object(update_health.shutil, "which",
                               return_value="/usr/bin/npm"):
            self.assertEqual(update_health._resolve_npm(), "/usr/bin/npm")


if __name__ == "__main__":
    unittest.main()
