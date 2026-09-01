"""Benchmark harnesses must resolve fixtures, not name one machine's drive.

TWO defects are pinned here, and the second was introduced while fixing the
first.

1. **Hardcoded `G:/pinokio/roop-keep/...`.** That is the RTX 4070
   workstation's layout. On the RTX 3060 laptop `PINOKIO_HOME` is `C:\\pinokio`
   and there is no `G:` drive at all, so every documented dual-GPU command died
   on a missing file -- or worse, swept an empty directory and reported a clean
   result. `tests/fixtures.py` exists precisely to resolve this at runtime; it
   was written and then wired into only two of ~35 harnesses.

2. **`import fixtures` inside `if HERE not in sys.path:`.** Python puts a
   script's own directory on `sys.path` before running it, so that branch is
   False in the normal case and the import silently never ran. Every migrated
   harness then raised `NameError` at its first `fixtures.` reference -- and a
   syntax check passes happily, because the code is perfectly valid. The guard
   therefore asserts the import is reachable at MODULE level, not merely
   present.
"""

import ast
import glob
import io
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fixtures

# A drive-lettered pinokio path written as a STRING LITERAL in code. Prose in a
# docstring or comment may still cite the other machine's layout as history.
_LITERAL = re.compile(r"""r?(['"])[A-Za-z]:[/\\]{1,2}pinokio[^'"]*\1""")


def _harnesses():
    return sorted(glob.glob(os.path.join(HERE, "*.py")))


def _code_literals(path):
    """Drive-lettered pinokio literals that are real code, not docstrings."""
    src = io.open(path, encoding="utf-8").read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ["<unparseable>"]

    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.add(doc)

    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in docstrings:
                continue
            if re.search(r"[A-Za-z]:[/\\]{1,2}pinokio", node.value):
                # A multi-line usage example embedded in a non-docstring string
                # is prose too; only single-path values are the defect.
                if "\n" not in node.value:
                    hits.append(node.value)
    return hits


class NoHardcodedFixtureRootsTest(unittest.TestCase):

    def test_no_harness_hardcodes_a_drive_lettered_pinokio_path(self):
        offenders = {}
        for path in _harnesses():
            hits = _code_literals(path)
            if hits:
                offenders[os.path.basename(path)] = hits
        self.assertEqual(
            offenders, {},
            "these harnesses name one machine's drive in code; use "
            "fixtures.clip()/clip_dir() so both GPUs can run them:\n%s"
            % offenders)


class FixturesImportIsReachableTest(unittest.TestCase):

    def test_every_user_imports_fixtures_at_module_level(self):
        """Present is not the same as reachable.

        `import fixtures` nested in `if HERE not in sys.path:` never executes
        when the file is run as a script, which is how every one of these is
        run.
        """
        bad = []
        for path in _harnesses():
            src = io.open(path, encoding="utf-8").read()
            if "fixtures." not in src:
                continue
            if "import fixtures as _fx" in src:
                continue          # deliberate local import inside a helper
            try:
                tree = ast.parse(src)
            except SyntaxError as exc:
                bad.append((os.path.basename(path), "syntax: %s" % exc))
                continue
            top_level = any(
                isinstance(n, ast.Import)
                and any(a.name == "fixtures" for a in n.names)
                for n in tree.body)
            if not top_level:
                bad.append((os.path.basename(path),
                            "uses fixtures. but has no module-level import"))
        self.assertEqual(bad, [], "unreachable fixtures import:\n%s" % bad)


class ResolverContractTest(unittest.TestCase):

    def test_missing_fixture_is_returned_unchanged_not_substituted(self):
        """A fixture swap silently invalidates a cross-target comparison."""
        missing = "definitely/not/here_%s.mp4" % os.getpid()
        self.assertEqual(fixtures.clip(missing), missing)
        self.assertFalse(fixtures.available(missing))

    def test_missing_directory_is_returned_unchanged(self):
        missing = "no_such_category_%s" % os.getpid()
        self.assertEqual(fixtures.clip_dir(missing), missing)
        self.assertFalse(fixtures.dir_available(missing))

    def test_required_raises_with_the_search_list(self):
        with self.assertRaises(SystemExit) as ctx:
            fixtures.clip("nope_%s.mp4" % os.getpid(), required=True)
        self.assertIn("Searched", str(ctx.exception))

    def test_pinokio_home_is_not_assumed_to_be_a_drive(self):
        home = fixtures.pinokio_home()
        self.assertTrue(os.path.isabs(home))


if __name__ == "__main__":
    unittest.main()
