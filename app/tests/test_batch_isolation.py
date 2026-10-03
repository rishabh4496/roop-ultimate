"""Isolated video batch: one spawn child per video, and a queue that survives a bad video.

Found 2026-10-03 by running `pinokio_batch_runner.py` for real over a 50-clip queue: it died
on the first frame of the first video with ``TypeError: ... got multiple values for keyword
argument 'total'`` -- nothing had ever run it, because nothing tested it. And the manager
stopped the WHOLE queue at the first failed video, so a 50-job run died at job N. These tests
use real `spawn` children with tiny stub workers (no GPU), so what is asserted is the process
and queue behaviour that the contract ("reset to zero leaked bytes between files") rests on.
"""
import os
import sys

import pytest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(APP)
for _p in (APP, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from roop.process_manager import IsolatedVideoBatch, VideoJob, WorkerFailed  # noqa: E402
from tests import batch_stub_workers as stubs  # noqa: E402


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())


def _jobs(*names):
    return [VideoJob("C:/in/" + n, "C:/out/" + n, "single", ("f",)) for n in names]


def test_every_video_runs_in_its_own_fresh_process():
    results = IsolatedVideoBatch(stubs.pid_worker, join_seconds=20).run(_jobs("a.mp4", "b.mp4", "c.mp4"))
    pids = [r["pid"] for r in results]
    assert [r["status"] for r in results] == ["completed"] * 3
    assert len(set(pids)) == 3                       # three processes, none reused
    assert os.getpid() not in pids                   # and none of them is the parent


def test_progress_events_reach_the_parent():
    events = []
    IsolatedVideoBatch(stubs.pid_worker, join_seconds=20).run(_jobs("a.mp4"), on_progress=events.append)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "started" and kinds[-1] == "completed"
    assert any(e["type"] == "progress" and e.get("frame_index") == 1 for e in events)


def test_by_default_the_first_failure_stops_the_queue():
    with pytest.raises(WorkerFailed, match="synthetic failure for bad.mp4"):
        IsolatedVideoBatch(stubs.fail_on_bad_worker, join_seconds=20).run(
            _jobs("a.mp4", "bad.mp4", "c.mp4"))


def test_keep_going_records_the_failure_and_finishes_the_queue():
    events = []
    results = IsolatedVideoBatch(stubs.fail_on_bad_worker, join_seconds=20).run(
        _jobs("a.mp4", "bad.mp4", "c.mp4"), on_progress=events.append, keep_going=True)
    assert [r["status"] for r in results] == ["completed", "failed", "completed"]
    assert "synthetic failure" in results[1]["error"]
    assert results[0]["pid"] != results[2]["pid"]
    assert any(e["type"] == "job_failed" and e["input_path"].endswith("bad.mp4") for e in events)


def test_a_child_that_dies_without_a_word_is_a_failure_not_a_hang():
    """A CUDA fault kills the process; there is no terminal event, only an exit code."""
    with pytest.raises(WorkerFailed, match="exit code 7"):
        IsolatedVideoBatch(stubs.crash_worker, join_seconds=20).run(_jobs("a.mp4"))
    results = IsolatedVideoBatch(stubs.crash_worker, join_seconds=20).run(_jobs("a.mp4"), keep_going=True)
    assert results[0]["status"] == "failed"


def test_stop_requested_terminates_the_running_child():
    results = IsolatedVideoBatch(stubs.slow_worker, poll_seconds=0.05, join_seconds=5).run(
        _jobs("a.mp4", "b.mp4"), stop_requested=lambda: True)
    assert results == []                              # stopped before anything started


def test_stop_requested_mid_job_kills_the_child():
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 3                          # let the child start, then stop
    results = IsolatedVideoBatch(stubs.slow_worker, poll_seconds=0.05, join_seconds=5).run(
        _jobs("a.mp4"), stop_requested=stop)
    assert results[0]["status"] == "stopped" and results[0]["terminated"]


# --- pinokio_batch_runner: the progress class ProcessMgr instantiates -------------------
def test_worker_progress_accepts_the_keywords_processmgr_passes():
    """The regression: ProcessMgr calls ChunkedProgress(total=N, desc=..., unit=...)."""
    import pinokio_batch_runner as runner
    events = []
    cls = runner.make_worker_progress(24, lambda **e: events.append(e))
    with cls(total=24, desc="Processing", unit="frames", dynamic_ncols=True, bar_format="x") as bar:
        bar.update(1)
        bar.update(1)
    assert bar.total == 24 and bar.n == 2
    assert events and events[0]["frame_total"] == 24


def test_worker_progress_uses_the_declared_total_when_the_frame_count_is_unknown():
    import pinokio_batch_runner as runner
    cls = runner.make_worker_progress(0, lambda **e: None)       # cv2 could not count
    assert cls(total=7, desc="Processing").total == 7
    assert cls(desc="Processing").total == 0


def test_main_exit_code_reflects_failures_and_keep_going(tmp_path, monkeypatch):
    import pinokio_batch_runner as runner
    (tmp_path / "single").mkdir()
    for n in ("a.mp4", "bad.mp4"):
        (tmp_path / "single" / n).write_bytes(b"x")
    seen = {}

    class FakeBatch:
        def __init__(self, worker):
            pass

        def run(self, jobs, on_progress=None, keep_going=False):
            seen["keep_going"] = keep_going
            return [{"job": j.as_dict(), "status": "failed" if "bad" in j.input_path else "completed",
                     "error": "boom"} for j in jobs]
    monkeypatch.setattr(runner, "IsolatedVideoBatch", FakeBatch)
    argv = ["--root", str(tmp_path), "--single-faceset", "f", "--double-facesets", "f,g", "--keep-going"]
    assert runner.main(argv) == 1 and seen["keep_going"] is True
    assert runner.main(argv[:-1]) == 1 and seen["keep_going"] is False
