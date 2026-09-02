"""A checkpoint write must survive a transient rename, and never wedge a render.

WHY THIS EXISTS.  On the physical RTX 3060 host a project's PROCESSING state
update raised `PermissionError: [WinError 5]` from `os.replace`.  Windows raises
that whenever any other process holds a handle to either file for a moment --
Defender scanning the temporary that was just fsynced, the Search indexer, a
backup agent.  Nothing in this application was holding them.

The consequences were out of all proportion to the cause.  The call sits at the
top of `api._run_swap`, BEFORE that function's own `try`, so the exception
escaped the worker thread: the render never started, the `finally` that clears
`_progress["processing"]` never ran, and the API reported `processing: true,
progress: 0.0, desc: 'Starting...', error: ''` indefinitely.  A user sees a job
generating forever that is not running, produces nothing, and reports no error.

Three layers are asserted here: the rename retries, a checkpoint failure that
does get through is reported rather than raised, and -- because the first two
are per-fault while the wedge is structural -- `_run_swap`'s own `try` now
covers its whole preamble, so ANY statement failing there still clears the
processing flag.
"""

import os
import sys
import unittest
from unittest import mock

_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP not in sys.path:
    sys.path.insert(0, _APP)

import project_checkpoint  # noqa: E402


class AtomicWriteRetryTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.directory = tempfile.mkdtemp(prefix="ckpt-test-")
        self.path = os.path.join(self.directory, "record.json")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_transient_permission_error_is_retried_and_succeeds(self):
        real = os.replace
        calls = {"n": 0}

        def flaky(src, dst):
            calls["n"] += 1
            if calls["n"] < 3:
                raise PermissionError(5, "Access is denied")
            return real(src, dst)

        with mock.patch.object(project_checkpoint.os, "replace", flaky), \
                mock.patch.object(project_checkpoint.time, "sleep", lambda _s: None):
            project_checkpoint._atomic_write(self.path, {"id": "abc", "state": "PROCESSING"})

        self.assertEqual(calls["n"], 3)
        self.assertTrue(os.path.isfile(self.path))
        import json
        with open(self.path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["state"], "PROCESSING")

    def test_a_persistent_permission_error_still_raises(self):
        """The retry must not mask a genuinely unwritable projects directory."""
        def always(_src, _dst):
            raise PermissionError(5, "Access is denied")

        with mock.patch.object(project_checkpoint.os, "replace", always), \
                mock.patch.object(project_checkpoint.time, "sleep", lambda _s: None):
            with self.assertRaises(PermissionError):
                project_checkpoint._atomic_write(self.path, {"id": "abc"})

    def test_the_temporary_file_is_not_left_behind_on_failure(self):
        def always(_src, _dst):
            raise PermissionError(5, "Access is denied")

        with mock.patch.object(project_checkpoint.os, "replace", always), \
                mock.patch.object(project_checkpoint.time, "sleep", lambda _s: None):
            with self.assertRaises(PermissionError):
                project_checkpoint._atomic_write(self.path, {"id": "abc"})
        leftovers = [n for n in os.listdir(self.directory) if n.startswith(".checkpoint-")]
        self.assertEqual(leftovers, [])

    def test_retry_budget_is_bounded(self):
        self.assertGreaterEqual(project_checkpoint._REPLACE_ATTEMPTS, 2)
        self.assertLessEqual(project_checkpoint._REPLACE_ATTEMPTS, 32)


class ProjectStateIsNotFatalTests(unittest.TestCase):
    """The second layer: bookkeeping must not abort the run it describes."""

    def test_a_failing_checkpoint_update_does_not_propagate(self):
        import api

        def boom(*_a, **_k):
            raise PermissionError(5, "Access is denied")

        logged = []
        with mock.patch.object(api._project_checkpoint, "update_state", boom), \
                mock.patch.object(api, "_push_log",
                                  lambda msg, **_k: logged.append(msg)):
            # Must not raise -- this call sits before _run_swap's own try.
            api._set_processing_project_state("some-project", "PROCESSING")

        self.assertTrue(logged, "a persistence failure must be reported, not swallowed")
        self.assertIn("project state could not be saved", logged[0])

    def test_no_project_id_is_still_a_no_op(self):
        import api
        with mock.patch.object(api._project_checkpoint, "update_state") as update:
            api._set_processing_project_state("", "PROCESSING")
        update.assert_not_called()


class ThePreambleCannotWedgeTheApp(unittest.TestCase):
    """The third layer, and the only structural one.

    Guarding the checkpoint call fixes the fault that was observed.  It does
    not fix the SHAPE of the failure: `_start_existing_project` claims
    `_progress["processing"]` synchronously, and only `_run_swap`'s `finally`
    releases it, so every statement above that `try` is another way to leave
    the app generating forever with no error and no output.  The preamble is
    now inside the try, which is asserted here by failing on its FIRST
    statement -- one the checkpoint guard cannot reach.
    """

    def test_a_preamble_failure_still_clears_the_processing_flag(self):
        import api

        controller = api._procmgr_runtime.pause_controller
        api._progress.update({"processing": True, "error": ""})

        def boom():
            raise RuntimeError("preamble failed")

        with mock.patch.object(controller, "start", boom):
            api._run_swap({})

        self.assertFalse(api._progress["processing"],
                         "a failed preamble left the app wedged at 409")
        self.assertEqual(api._progress["error"], "preamble failed")

    def test_the_preamble_is_inside_the_try_at_source_level(self):
        """Read the structure, not just one behaviour.

        The behavioural test above can only fail on the one statement it
        patches; this fails if the `try` ever drifts back down past any of
        them.
        """
        import inspect
        import api

        body = inspect.getsource(api._run_swap).split("\n")
        opens_try = next(i for i, line in enumerate(body) if line.strip() == "try:")
        for name in ("pause_controller.start()", "_log_lines.clear()",
                     "segment_writer.reset_parts()", "live_preview.reset()",
                     "_procmgr_runtime.reset_eta()", '_progress.update({"processing": True'):
            at = next(i for i, line in enumerate(body) if name in line)
            self.assertGreater(at, opens_try,
                               f"{name} sits above _run_swap's try and can wedge the app")


if __name__ == "__main__":
    unittest.main()
