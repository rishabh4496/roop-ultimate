"""`benchmark_pipeline.py` must find its clip, not name one machine's drive.

It defaulted to ``r"D:\\k1.mp4"`` in three places (docstring, constructor, ``--video``). The existing guard
(`test_fixture_paths.py`) missed it twice: it scans only `app/tests/*.py`, and only for ``X:\\pinokio`` literals, while this was a
root-level script with a bare ``D:\\k1.mp4``. On the 3060 laptop there is no such drive, so the documented command died on a missing
file; on the 4070 it silently depended on one machine's layout.

Pinned here:
  * `fixtures.resolve_target` finds a NAMED clip in the fixture roots (flat or nested), and uses an operator-typed absolute path
    exactly as given -- if that path is missing it FAILS rather than falling back to a same-named clip under the root, which would
    run a different workload under the operator's chosen name;
  * a clip that cannot be found raises FileNotFoundError naming every root searched (the run is PENDING, never substituted);
  * `BenchmarkSuite` resolves the clip BEFORE it builds the model registry, and `main()` turns the failure into exit code 2;
  * no root-level benchmark harness writes a drive-lettered path into code.
"""
import ast
import glob
import io
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
BENCH_DIR = os.path.join(ROOT, "scripts", "benchmarks")      # benchmark_pipeline.py / run_stage0_benchmark.py live here
for _p in (APP, HERE, ROOT, BENCH_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fixtures  # noqa: E402

# The fixture roots always include <PINOKIO_HOME>/roop-keep, which on a developer machine holds real clips: every name used
# here is made up so nothing real can satisfy it by accident.
NAME = "zz_stage_clip_resolution_probe.mp4"


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"\x00")
    return path


class ResolveTargetTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.other = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.addCleanup(self.other.cleanup)
        patcher = mock.patch.dict(os.environ, {"ROOP_CLIP_ROOT": self.root.name})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_default_name_is_found_flat_in_the_clip_root(self):
        want = _touch(os.path.join(self.root.name, NAME))
        self.assertEqual(os.path.normcase(fixtures.resolve_target(None, NAME)), os.path.normcase(want))

    def test_a_name_is_found_in_a_category_folder(self):
        want = _touch(os.path.join(self.root.name, "double", NAME))
        self.assertEqual(os.path.normcase(fixtures.resolve_target(NAME, "unused.mp4")), os.path.normcase(want))

    def test_an_explicit_existing_path_is_used_as_given(self):
        _touch(os.path.join(self.root.name, NAME))                      # a same-named clip in the root must not win
        mine = _touch(os.path.join(self.other.name, NAME))
        self.assertEqual(fixtures.resolve_target(mine, "unused.mp4"), mine)

    def test_a_missing_explicit_absolute_path_does_not_fall_back_to_a_same_named_clip(self):
        _touch(os.path.join(self.root.name, NAME))
        missing = os.path.join(self.other.name, "gone", NAME)           # absolute, does not exist
        with self.assertRaises(FileNotFoundError) as cm:
            fixtures.resolve_target(missing, "unused.mp4")
        self.assertIn("not searched", str(cm.exception))

    def test_an_unfindable_clip_raises_and_names_every_root_searched(self):
        with self.assertRaises(FileNotFoundError) as cm:
            fixtures.resolve_target(None, NAME)
        msg = str(cm.exception)
        self.assertIn("PENDING", msg)
        self.assertIn("ROOP_CLIP_ROOT", msg)
        self.assertIn(self.root.name, msg)                              # the override root is in the searched list

    def test_the_failure_never_returns_some_other_clip(self):
        _touch(os.path.join(self.root.name, "a_different_clip.mp4"))
        with self.assertRaises(FileNotFoundError):
            fixtures.resolve_target(None, NAME)


class BenchmarkPipelineWiringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import benchmark_pipeline as bp
        except Exception as exc:                                       # torch / face_engine unavailable (light CI profile)
            raise unittest.SkipTest("benchmark_pipeline needs the GPU stack to import: %s" % exc)
        cls.bp = bp

    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        for patcher in (mock.patch.dict(os.environ, {"ROOP_CLIP_ROOT": self.root.name}),
                        mock.patch.object(self.bp, "DEFAULT_CLIP", NAME)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_the_default_is_a_name_not_a_path(self):
        import benchmark_pipeline as real
        self.assertFalse(os.path.isabs(real.DEFAULT_CLIP))
        self.assertFalse(re.match(r"^[A-Za-z]:", real.DEFAULT_CLIP))

    def test_the_clip_is_resolved_before_any_model_registry_is_built(self):
        boom = mock.Mock(side_effect=AssertionError("the model registry was built before the clip was found"))
        with mock.patch.object(self.bp, "build_default_registry", boom):
            with self.assertRaises(FileNotFoundError):
                self.bp.BenchmarkSuite()
        self.assertEqual(boom.call_count, 0)

    def test_main_reports_pending_with_exit_code_2(self):
        with mock.patch.object(sys, "argv", ["benchmark_pipeline.py"]), \
                mock.patch.object(self.bp, "build_default_registry", mock.Mock(side_effect=AssertionError("registry"))):
            with self.assertRaises(SystemExit) as cm:
                self.bp.main()
        self.assertEqual(cm.exception.code, 2)


class NoDriveLettersInBenchmarkHarnessesTest(unittest.TestCase):
    """`test_fixture_paths.py` covers app/tests; these are the harnesses that live elsewhere and slipped past it."""

    SCRIPTS = (os.path.join("scripts", "benchmarks", "benchmark_pipeline.py"), os.path.join("tests", "benchmark_pipeline.py"),
               os.path.join("scripts", "benchmarks", "run_stage0_benchmark.py"), "pinokio_batch_runner.py",
               os.path.join("scripts", "verification", "verify_pipeline.py"))
    DRIVE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")

    def _code_literals(self, path):
        tree = ast.parse(io.open(path, encoding="utf-8").read())
        docs = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    docs.add(doc)
        return [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value not in docs
                and "\n" not in n.value and self.DRIVE.search(n.value)]

    def test_no_benchmark_harness_writes_a_drive_letter_into_code(self):
        offenders = {}
        for rel in self.SCRIPTS:
            path = os.path.join(ROOT, rel)
            self.assertTrue(os.path.isfile(path), "%s moved: update this guard" % rel)
            hits = self._code_literals(path)
            if hits:
                offenders[rel] = hits
        self.assertEqual(offenders, {}, "resolve the path (fixtures.resolve_target / PINOKIO_HOME), do not name a drive")

    def test_the_guard_would_catch_the_original_defect(self):
        """Mutation check: the exact lines that shipped must be flagged by this scan."""
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "bad.py")
            with open(bad, "w", encoding="utf-8") as fh:
                fh.write('def f(video_path: str = r"D:\\k1.mp4"):\n    pass\nparser.add_argument("--video", default=r"D:\\k1.mp4")\n')
            self.assertEqual(len(self._code_literals(bad)), 2)


if __name__ == "__main__":
    unittest.main()
