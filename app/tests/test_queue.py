"""The batch queue must not lose work.

That is the whole reason it moved out of the browser, so these test the ways it
could still lose a job rather than the happy path:

  * an app restart mid-job (the job has to come back runnable, not stuck);
  * a restart while the batch was running (it must NOT resume unattended);
  * a stale client sending a reorder that does not mention every job;
  * a target removed after the job was queued (resolve by name, never by a
    stored index — an index silently swaps the wrong file);
  * a job that raises (the rest of the batch has to keep going);
  * a non-ASCII console on Windows (a single '->' arrow raised out of print,
    out of the runner loop, and stopped the batch with no visible error).

The runner is driven with a fake _run_swap so a test is milliseconds, not a
render.
"""

import os
import sys
import tempfile
import time
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

import routes_queue as q  # noqa: E402


class _Entry:
    """Stand-in for a roop ProcessEntry — only the fields the runner touches."""
    def __init__(self, filename, total=1000):
        self.filename = filename
        self.startframe = 0
        self.endframe = total
        self.total_frames = total
        self.fps = 30


class QueueTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="queue_test_")
        self._old_file = q.QUEUE_FILE
        q.QUEUE_FILE = os.path.join(self._tmp, "queue.json")
        q._queue["jobs"] = []
        q._queue.update({"running": False, "paused": False, "current": None})

        self.ran = []                       # payloads the fake runner received
        self.progress = {"processing": False, "progress": 0.0, "error": "", "paused": False}
        self.entries = []
        q._progress = self.progress
        q.list_files_process = self.entries
        q._stop_current = lambda: None
        q._run_swap = self._fake_run
        # Reset the output-recording hooks too: a test that installs them must
        # not leak into the next one, which would then assert against a stale
        # fake rather than the module's real default.
        q._snapshot_outputs = None
        q._outputs_since = None
        # Same reasoning: a test that pretends the benchmark is running must not
        # leave every later test dispatching into a fake benchmark.
        q._benchmark_running = lambda: False

    def tearDown(self):
        q._queue["running"] = False
        q.QUEUE_FILE = self._old_file

    def _fake_run(self, payload):
        self.ran.append(dict(payload))
        self.progress["progress"] = 1.0
        self.progress["processing"] = False

    def _add(self, name, **kw):
        body = {"target_name": name, "source_index": 0, "payload": {"swap_model": "inswapper"}}
        body.update(kw)
        return q.queue_add(body)

    def _drain(self, timeout=5.0):
        """Run the queue to completion on this thread's clock."""
        import roop.globals as g
        g.processing = True
        q.queue_start()
        deadline = time.time() + timeout
        while q._queue["running"] and time.time() < deadline:
            time.sleep(0.01)
        self.assertFalse(q._queue["running"], "runner did not finish")


class Persistence(QueueTestBase):
    def test_jobs_survive_a_restart(self):
        self._add("a.mp4")
        self._add("b.mp4")
        q._queue["jobs"] = []               # simulate a fresh process
        q.load()
        self.assertEqual([j["target_name"] for j in q._snapshot()["jobs"]],
                         ["a.mp4", "b.mp4"])

    def test_a_job_interrupted_by_a_restart_becomes_runnable_again(self):
        self._add("a.mp4")
        q._queue["jobs"][0]["status"] = "running"
        q._save()
        q._queue["jobs"] = []
        q.load()
        job = q._snapshot()["jobs"][0]
        self.assertEqual(job["status"], "pending",
                         "a job left 'running' by a crash could never run again")
        self.assertIn("restart", job["error"])

    def test_a_restart_does_not_resume_the_batch_by_itself(self):
        """Runner flags are not persisted on purpose: coming back up swapping,
        with nobody watching, is worse than coming back up idle."""
        self._add("a.mp4")
        q._queue["running"] = True
        q._save()
        q._queue.update({"running": False, "jobs": []})
        q.load()
        self.assertFalse(q._snapshot()["running"])

    def test_a_failed_write_does_not_raise_into_a_handler(self):
        q.QUEUE_FILE = os.path.join(self._tmp, "no-such-dir", "queue.json")
        self._add("a.mp4")              # must not raise
        self.assertEqual(len(q._snapshot()["jobs"]), 1)


class Editing(QueueTestBase):
    def test_reorder_from_a_stale_client_cannot_drop_a_job(self):
        self._add("a.mp4"); self._add("b.mp4")
        ids = [j["id"] for j in q._snapshot()["jobs"]]
        self._add("c.mp4")              # the client does not know about this one
        q.queue_reorder({"ids": [ids[1], ids[0]]})
        names = [j["target_name"] for j in q._snapshot()["jobs"]]
        self.assertEqual(names, ["b.mp4", "a.mp4", "c.mp4"])

    def test_duplicate_lands_next_to_its_original_and_is_pending(self):
        self._add("a.mp4"); self._add("b.mp4")
        first = q._snapshot()["jobs"][0]
        first["status"] = "finished"
        out = q.queue_duplicate({"id": first["id"]})["jobs"]
        self.assertEqual([j["target_name"] for j in out], ["a.mp4", "a.mp4", "b.mp4"])
        self.assertEqual(out[1]["status"], "pending")
        self.assertNotEqual(out[0]["id"], out[1]["id"])

    def test_editing_a_finished_job_re_queues_it(self):
        self._add("a.mp4")
        job = q._snapshot()["jobs"][0]
        q._find(job["id"])["status"] = "finished"
        out = q.queue_update({"id": job["id"], "payload": {"swap_model": "ghost_1"}})
        self.assertEqual(out["jobs"][0]["status"], "pending")
        self.assertEqual(out["jobs"][0]["payload"]["swap_model"], "ghost_1")

    def test_a_running_job_cannot_be_edited_or_removed(self):
        self._add("a.mp4")
        job = q._snapshot()["jobs"][0]
        q._find(job["id"])["status"] = "running"
        q._queue["current"] = job["id"]
        self.assertEqual(q.queue_update({"id": job["id"], "label": "x"}).status_code, 409)
        self.assertEqual(q.queue_remove({"id": job["id"]}).status_code, 409)

    def test_clear_keeps_the_job_that_is_mid_render(self):
        self._add("a.mp4"); self._add("b.mp4")
        cur = q._snapshot()["jobs"][0]["id"]
        q._queue["current"] = cur
        left = q.queue_clear({})["jobs"]
        self.assertEqual([j["id"] for j in left], [cur])


class Runner(QueueTestBase):
    def test_the_target_is_resolved_by_name_not_by_a_stored_index(self):
        """The queued index goes stale the moment a target is removed, and the
        job then renders a different file than the one it names."""
        self.entries.extend([_Entry("/media/a.mp4"), _Entry("/media/b.mp4")])
        self._add("b.mp4", source_index=0)
        self.entries.pop(0)             # 'b' is now index 0, not 1
        self._drain()
        self.assertEqual(len(self.ran), 1)
        self.assertEqual(self.ran[0]["target_index"], 0)

    def test_a_target_that_is_gone_fails_that_job_only(self):
        self.entries.append(_Entry("/media/b.mp4"))
        self._add("gone.mp4")
        self._add("b.mp4")
        self._drain()
        jobs = q._snapshot()["jobs"]
        self.assertEqual(jobs[0]["status"], "failed")
        self.assertIn("no longer loaded", jobs[0]["error"])
        self.assertEqual(jobs[1]["status"], "finished")
        self.assertEqual(len(self.ran), 1, "the second job still had to run")

    def test_a_job_that_raises_does_not_kill_the_batch(self):
        self.entries.append(_Entry("/media/a.mp4"))
        self._add("a.mp4"); self._add("a.mp4")
        calls = {"n": 0}

        def boom(payload):
            calls["n"] += 1
            self.progress["processing"] = False
            if calls["n"] == 1:
                raise RuntimeError("kaboom")
            self.progress["progress"] = 1.0
        q._run_swap = boom

        self._drain()
        jobs = q._snapshot()["jobs"]
        self.assertEqual(jobs[0]["status"], "failed")
        self.assertEqual(jobs[0]["error"], "kaboom")
        self.assertEqual(jobs[1]["status"], "finished")

    def test_a_segment_sets_the_targets_trim_range_before_the_run(self):
        """The range lives on the ProcessEntry, not in the swap payload, so each
        segment job has to set it immediately before dispatching."""
        entry = _Entry("/media/a.mp4", total=500)
        self.entries.append(entry)
        seen = []
        q._run_swap = lambda p: (seen.append((entry.startframe, entry.endframe)),
                                 self.progress.update({"processing": False, "progress": 1.0}))
        self._add("a.mp4", frame_start=100, frame_end=200)
        self._add("a.mp4", frame_start=300, frame_end=9999)   # clamped to total
        self._drain()
        self.assertEqual(seen, [(100, 200), (300, 500)])

    def test_finished_jobs_are_skipped_so_a_restart_resumes(self):
        self.entries.append(_Entry("/media/a.mp4"))
        self._add("a.mp4"); self._add("a.mp4")
        q._queue["jobs"][0]["status"] = "finished"
        self._drain()
        self.assertEqual(len(self.ran), 1)

    def test_start_refuses_when_a_single_run_is_already_processing(self):
        self.entries.append(_Entry("/media/a.mp4"))
        self._add("a.mp4")
        self.progress["processing"] = True
        self.assertEqual(q.queue_start().status_code, 409)
        self.assertFalse(q._queue["running"])

    def test_start_refuses_an_exhausted_queue(self):
        self._add("a.mp4")
        q._queue["jobs"][0]["status"] = "finished"
        self.assertEqual(q.queue_start().status_code, 400)


class Segments(QueueTestBase):
    def test_each_job_records_the_files_it_produced(self):
        """The join needs to know which outputs belong to which segment; guessing
        from filenames would break the moment the output template changes."""
        self.entries.append(_Entry("/media/a.mp4"))
        made = ["/out/a_0001.mp4"]
        q._snapshot_outputs = lambda: {}
        q._outputs_since = lambda _before: made
        self._add("a.mp4", frame_start=1, frame_end=50)
        self._drain()
        self.assertEqual(q._snapshot()["jobs"][0]["outputs"], made)

    def test_join_refuses_fewer_than_two_segments(self):
        self._add("a.mp4")
        job = q._snapshot()["jobs"][0]
        q._find(job["id"])["outputs"] = ["/out/only.mp4"]
        self.assertEqual(q.queue_join({"ids": [job["id"]]}).status_code, 400)

    def test_join_refuses_mixed_formats(self):
        """A concat stream-copy across formats produces a file that plays wrong
        or not at all — better to say so than to hand back a broken render."""
        import tempfile
        d = tempfile.mkdtemp(prefix="join_")
        paths = [os.path.join(d, "a.mp4"), os.path.join(d, "b.mkv")]
        for p in paths:
            open(p, "wb").close()
        self._add("a.mp4"); self._add("a.mp4")
        jobs = q._snapshot()["jobs"]
        for job, p in zip(jobs, paths):
            q._find(job["id"])["outputs"] = [p]
        res = q.queue_join({"ids": [j["id"] for j in jobs]})
        self.assertEqual(res.status_code, 400)
        self.assertIn("different formats", res.body.decode())

    def test_join_ignores_outputs_that_no_longer_exist(self):
        self._add("a.mp4"); self._add("a.mp4")
        jobs = q._snapshot()["jobs"]
        q._find(jobs[0]["id"])["outputs"] = ["/gone/one.mp4"]
        q._find(jobs[1]["id"])["outputs"] = ["/gone/two.mp4"]
        self.assertEqual(q.queue_join({"ids": [j["id"] for j in jobs]}).status_code, 400)


class RunnerLifetime(QueueTestBase):
    def test_a_superseded_runner_retires_instead_of_dispatching(self):
        """Stop → Start must not leave two runners walking one queue.

        The window is real rather than theoretical. queue_stop only sets flags
        and returns, and _run_swap clears _progress['processing'] in its own
        finally — so while the previous runner is still alive finishing its
        bookkeeping (an _outputs_since listdir and a queue.json write), BOTH of
        queue_start's guards already pass: nothing is 'running' and nothing is
        'processing'. A new runner starts legitimately. The old one then comes
        round its loop, sees running=True again, and takes the NEXT pending job
        — two renders into one output directory through one set of ProcessMgr
        globals. The generation token is what retires it.
        """
        self._add("a.mp4")
        self._add("b.mp4")
        q._queue["running"] = True          # a new batch is live…
        q._generation += 1                  # …started by somebody else
        q._loop(q._generation - 1)          # the old thread comes round its loop
        self.assertEqual(self.ran, [],
                         "a superseded runner dispatched into the new batch")
        self.assertEqual([j["status"] for j in q._snapshot()["jobs"]],
                         ["pending", "pending"])

    def test_the_current_runner_is_not_retired_by_its_own_token(self):
        """Guard the guard: if the token comparison were inverted or the
        generation bumped anywhere else, every batch would exit immediately and
        the queue would look like it silently refused to run."""
        self.entries.append(_Entry("a.mp4"))
        self._add("a.mp4")
        self._drain()
        self.assertEqual(len(self.ran), 1, "the live runner dispatched nothing")
        self.assertEqual(q._snapshot()["jobs"][0]["status"], "finished")


class ConsoleOutput(QueueTestBase):
    def test_a_non_ascii_message_cannot_raise_out_of_the_runner(self):
        """This is not hypothetical: a '->' arrow in the completion line raised
        UnicodeEncodeError on a cp1252 console, out of the loop, and stopped the
        batch after the finished job had already been saved — so the only
        symptom was a queue that quietly stopped."""
        q._say("[Queue] → ✓ café \U0001F600")   # must not raise

    def test_the_runner_reports_through_the_safe_helper_only(self):
        with open(os.path.join(APP, "routes_queue.py"), encoding="utf-8") as fh:
            src = fh.read()
        body = src.split("def _say(", 1)[1]
        after = body.split("\n\n\n", 1)[1] if "\n\n\n" in body else body
        self.assertNotIn("\n    print(", after,
                         "everything after _say() must report through it, or a "
                         "non-ASCII path/filename can kill the runner again")


class BenchmarkAndTheQueueDoNotShareTheGpu(QueueTestBase):
    """The queue is the third door into `_run_swap`, and it was unguarded.

    /api/swap refuses to start while the benchmark runs, and the benchmark
    refuses to start while `_progress['processing']` is set. Both guards live in
    api.py, and the queue calls `_run_swap` directly — so a batch could dispatch
    straight into a benchmark that is holding several pools of TensorRT contexts.
    On a 12GB card that is an OOM, not merely two wrong measurements.
    """

    def test_start_is_refused_while_the_benchmark_holds_the_gpu(self):
        self.entries.append(_Entry("a.mp4"))
        self._add("a.mp4")
        q._benchmark_running = lambda: True
        resp = q.queue_start()
        self.assertEqual(getattr(resp, "status_code", 200), 409)
        self.assertFalse(q._queue["running"])
        self.assertEqual(self.ran, [])

    def test_the_runner_holds_between_jobs_instead_of_dispatching(self):
        """The window the start guard cannot cover.

        Between two jobs nothing is processing, so the benchmark may legitimately
        start there. The runner has to notice on its next pass — and HOLD, not
        fail the batch: the jobs are still valid, the GPU is merely busy.
        """
        self.entries.extend([_Entry("a.mp4"), _Entry("b.mp4")])
        self._add("a.mp4")
        self._add("b.mp4")

        busy = {"on": False}
        q._benchmark_running = lambda: busy["on"]
        real_run = self._fake_run

        def run_then_start_a_benchmark(payload):
            real_run(payload)
            busy["on"] = True               # starts in the gap after job 1
        q._run_swap = run_then_start_a_benchmark

        import roop.globals as g
        g.processing = True
        q.queue_start()
        deadline = time.time() + 3.0
        while len(self.ran) < 1 and time.time() < deadline:
            time.sleep(0.01)
        time.sleep(0.4)                     # long enough for several loop passes
        self.assertEqual(len(self.ran), 1,
                         "the second job was dispatched into a running benchmark")
        self.assertTrue(q._queue["running"], "the batch was abandoned, not held")
        self.assertEqual(q._snapshot()["jobs"][1]["status"], "pending")

        busy["on"] = False                  # benchmark finishes
        deadline = time.time() + 3.0
        while q._queue["running"] and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(self.ran), 2, "the batch did not resume by itself")

    def test_api_injects_the_predicate(self):
        # The module default is `lambda: False`, so a missing injection disables
        # both guards above while every test in this file still passes.
        with open(os.path.join(APP, "api.py"), encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("_routes_queue._benchmark_running", src,
                      "api.py never injects the benchmark predicate — the queue "
                      "guards are wired to a constant False")


class CanonicalLifecycle(QueueTestBase):
    def test_new_jobs_expose_required_state_schema_and_order(self):
        self._add("a.mp4")
        self._add("b.mp4")
        snapshot = q._snapshot()
        self.assertEqual(snapshot["schema_version"], 2)
        self.assertEqual(snapshot["job_states"], list(q.JOB_STATES))
        self.assertEqual([job["state"] for job in snapshot["jobs"]], ["QUEUED", "QUEUED"])
        self.assertEqual([job["position"] for job in snapshot["jobs"]], [1, 2])
        self.assertIn("fraction", snapshot["jobs"][0]["progress"])

    def test_legacy_running_record_migrates_to_recoverable(self):
        self._add("a.mp4")
        job = q._find(q._snapshot()["jobs"][0]["id"])
        job.pop("state", None)
        job["status"] = "running"
        q._save()
        q._queue["jobs"] = []
        q.load()
        restored = q._snapshot()["jobs"][0]
        self.assertEqual(restored["state"], "RECOVERABLE")
        self.assertEqual(restored["status"], "pending")
        self.assertIn("restart", restored["error"])

    def test_queued_cancel_isolated_from_other_jobs(self):
        self.entries.append(_Entry("/media/a.mp4"))
        self._add("a.mp4")
        self._add("missing.mp4")
        jobs = q._snapshot()["jobs"]
        q.queue_cancel({"id": jobs[1]["id"]})
        self._drain()
        result = q._snapshot()["jobs"]
        self.assertEqual(result[0]["state"], "COMPLETED")
        self.assertEqual(result[1]["state"], "CANCELLED")
        self.assertEqual(len(self.ran), 1)

    def test_current_cancel_becomes_cancelled_after_cooperative_stop(self):
        self.entries.append(_Entry("/media/a.mp4"))
        self._add("a.mp4")
        job = q._snapshot()["jobs"][0]
        q._queue["current"] = job["id"]
        q._find(job["id"])["state"] = "PROCESSING"
        q._find(job["id"])["status"] = "running"
        called = []
        q._stop_current = lambda: called.append(True)
        result = q.queue_cancel({"id": job["id"]})
        self.assertEqual(result["jobs"][0]["state"], "PROCESSING")
        self.assertTrue(q._find(job["id"])["cancel_requested"])
        self.assertEqual(called, [True])


if __name__ == "__main__":
    unittest.main()
