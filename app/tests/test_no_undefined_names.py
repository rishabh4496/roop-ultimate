"""Every module must define, import or receive every name it uses.

This exists because the decomposition broke three call sites and none of the
other checks noticed. Splitting a module verifies the moved code has what it
needs — but the code LEFT BEHIND can just as easily lose a helper that went with
the move, and a surface diff will not see it: the name is simply gone from a
function body that nobody imports.

All three were live: `_run_swap` (the swap run path) called `_gpu_name`,
`get_file` called `_faceset_library_dir`, and `source_refresh_thumbs` called
`_frontal_crop_from_images`. Each would have raised NameError on first use.

The check is a light static approximation of pyflakes, scoped to this project so
it needs no new dependency. It resolves module-level and local bindings and
ignores attribute access, which is enough to catch a name that no longer exists
anywhere in a module.
"""

import ast
import builtins
import os
import sys
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

# Module dunders are injected by the import machinery, not bound in the source.
MODULE_DUNDERS = {"__file__", "__name__", "__doc__", "__package__", "__spec__",
                  "__loader__", "__builtins__", "__path__", "__debug__"}
BUILTINS = set(dir(builtins)) | MODULE_DUNDERS

# Third-party/large vendored trees are not ours to police.
SKIP_DIRS = {"env", "tests", "clip", "installer", "models", "output", "temp",
             "__pycache__", "facesets", "docs", "sidecar_keep", "ui"}


def _module_files():
    for path in sorted(APP.glob("*.py")):
        yield path
    for path in sorted((APP / "roop").glob("*.py")):
        yield path
    for path in sorted((APP / "roop" / "processors").glob("*.py")):
        yield path


def undefined_names(path):
    """Names loaded in *path* that it never binds. Approximate but conservative:
    every binding form is treated as defining the name for the whole module, so
    this under-reports rather than crying wolf."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    defined, used = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.alias):
            defined.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
        elif isinstance(node, ast.Global):
            defined.update(node.names)
        elif isinstance(node, ast.Name):
            (defined if isinstance(node.ctx, ast.Store) else used).add(node.id)
    return sorted(used - defined - BUILTINS)


class TestNoUndefinedNames(unittest.TestCase):
    def test_every_module_resolves_its_names(self):
        broken = {}
        for path in _module_files():
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            missing = undefined_names(path)
            if missing:
                broken[str(path.relative_to(APP))] = missing
        self.assertEqual(
            broken, {},
            "these modules reference names they never define or import — a "
            "NameError waiting for the first call:\n"
            + "\n".join(f"  {f}: {names}" for f, names in sorted(broken.items())))

    def test_the_check_actually_detects_a_missing_name(self):
        """Guard against the guard silently passing everything."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "broken.py"
            p.write_text("def f():\n    return some_helper_that_moved_away()\n", encoding="utf-8")
            self.assertIn("some_helper_that_moved_away", undefined_names(p))


if __name__ == "__main__":
    unittest.main()
