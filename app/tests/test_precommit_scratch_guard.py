"""Repo hygiene: scratch output at the repo root cannot be committed again.

86ace34 ("commit remaining working-tree files") committed scratch_capture.log,
scratch_eval.log, scratch_probe.log and scratch_tf32.log at the repo root next to a
real harness and a test edit. Nothing stopped it: .gitignore had no rule for them and
the pre-commit hook only regenerated the update manifest.

What these pin:
  * the hook refuses staged scratch output at the root (scratch_*, *.log) with exit 1;
  * it does NOT refuse docs/perf/**/*.log -- deliberate evidence logs the changelog cites;
  * deleting one from the index (git rm --cached) is always allowed;
  * .gitignore covers new root scratch output and leaves docs/perf logs trackable;
  * nothing scratch-named is tracked any more, and the trt_matrix harness is kept.
The hook runs for real, in a throwaway repository, through its ROOP_PRECOMMIT_GUARD_ONLY seam
(otherwise the allowed cases would fail later, on the manifest generator, for the wrong reason).
"""
import os
import shutil
import subprocess
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOOK = os.path.join(REPO, ".githooks", "pre-commit")
GIT = shutil.which("git")
SH = shutil.which("sh")


def _git(cwd, *args, check=True):
    return subprocess.run([GIT, *args], cwd=cwd, capture_output=True, text=True, check=check)


@unittest.skipUnless(GIT and SH and os.path.isfile(HOOK), "needs git and sh")
class HookRefusesRootScratch(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="precommit_guard_")
        _git(self.tmp, "init", "-q")
        _git(self.tmp, "config", "user.email", "t@example.invalid")
        _git(self.tmp, "config", "user.name", "t")
        shutil.copy(HOOK, os.path.join(self.tmp, "pre-commit"))
        # an initial commit, so `--diff-filter=D` cases have something to delete
        self._write("keep.txt", "x")
        _git(self.tmp, "add", "keep.txt")
        _git(self.tmp, "commit", "-q", "-m", "init", "--no-verify")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, rel, text="log\n"):
        path = os.path.join(self.tmp, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)

    def _hook(self):
        env = dict(os.environ, ROOP_PRECOMMIT_GUARD_ONLY="1")
        return subprocess.run([SH, os.path.join(self.tmp, "pre-commit")], cwd=self.tmp, env=env,
                              capture_output=True, text=True)

    def _stage(self, *paths):
        for p in paths:
            self._write(p)
        _git(self.tmp, "add", "-f", *paths)

    def test_scratch_prefixed_file_at_the_root_is_refused(self):
        self._stage("scratch_probe.log")
        r = self._hook()
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("scratch_probe.log", r.stderr)

    def test_any_root_level_log_is_refused(self):
        self._stage("run.log")
        self.assertEqual(self._hook().returncode, 1)

    def test_scratch_without_an_extension_is_refused_too(self):
        self._stage("scratch_notes")
        self.assertEqual(self._hook().returncode, 1)

    def test_every_offender_is_named(self):
        self._stage("scratch_a.log", "scratch_b.log")
        r = self._hook()
        self.assertIn("scratch_a.log", r.stderr)
        self.assertIn("scratch_b.log", r.stderr)

    def test_evidence_logs_under_docs_perf_are_allowed(self):
        self._stage("docs/perf/sustained_throughput_2026-10-11/final_arm1.log")
        self.assertEqual(self._hook().returncode, 0, self._hook().stderr)

    def test_a_nested_scratch_prefixed_log_is_not_a_root_file(self):
        self._stage("docs/perf/x/scratch_run.log")
        self.assertEqual(self._hook().returncode, 0)

    def test_removing_a_tracked_scratch_log_from_the_index_is_allowed(self):
        self._stage("scratch_old.log")
        _git(self.tmp, "-c", "core.hooksPath=/dev/null", "commit", "-q", "-m", "legacy", "--no-verify")
        _git(self.tmp, "rm", "--cached", "-q", "scratch_old.log")
        self.assertEqual(self._hook().returncode, 0)

    def test_a_clean_commit_passes(self):
        self._stage("app/notes.md")
        self.assertEqual(self._hook().returncode, 0)


@unittest.skipUnless(GIT, "needs git")
class RepositoryIsClean(unittest.TestCase):

    def test_new_root_scratch_output_is_ignored(self):
        for name in ("scratch_anything.log", "scratch_tf32.log", "capture.log", "scratch_dump"):
            r = _git(REPO, "check-ignore", "-q", name, check=False)
            self.assertEqual(r.returncode, 0, "%s should be gitignored" % name)

    def test_docs_perf_logs_stay_trackable(self):
        r = _git(REPO, "check-ignore", "-q", "docs/perf/x/run.log", check=False)
        self.assertEqual(r.returncode, 1, "docs/perf evidence logs must not be ignored")

    def test_no_scratch_output_is_tracked_at_the_root(self):
        tracked = _git(REPO, "ls-files").stdout.split("\n")
        bad = [p for p in tracked if p and "/" not in p and (p.startswith("scratch_") or p.endswith(".log"))]
        self.assertEqual(bad, [], "scratch output is tracked again: %s" % bad)

    def test_the_trt_matrix_harness_that_shared_that_commit_is_kept(self):
        self.assertTrue(os.path.isfile(os.path.join(REPO, "app", "tests", "trt_matrix.py")))
        tracked = _git(REPO, "ls-files", "app/tests/trt_matrix.py").stdout.strip()
        self.assertEqual(tracked, "app/tests/trt_matrix.py")


if __name__ == "__main__":
    unittest.main()
