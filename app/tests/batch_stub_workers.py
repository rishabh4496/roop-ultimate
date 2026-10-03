"""Module-level workers for tests/test_batch_isolation.py.

A `multiprocessing.spawn` child unpickles its target by module name, so these cannot live
inside a test function or a test module that pytest imports under a mangled name.
"""
import os
import time


def pid_worker(job, report):
    report(message="working", frame_index=1, frame_total=2)
    return {"pid": os.getpid(), "name": os.path.basename(job.input_path)}


def fail_on_bad_worker(job, report):
    if "bad" in os.path.basename(job.input_path):
        raise RuntimeError("synthetic failure for " + os.path.basename(job.input_path))
    return {"pid": os.getpid(), "name": os.path.basename(job.input_path)}


def crash_worker(job, report):
    """Dies without a terminal event, like a CUDA fault taking the process down."""
    os._exit(7)


def slow_worker(job, report):
    time.sleep(30)
    return {}
