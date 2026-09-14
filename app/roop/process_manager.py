"""Crash-contained batch video supervision.

This module deliberately does not import torch, onnxruntime, or any of the
renderer modules in the parent process.  A Windows ``spawn`` child is created
for exactly one video and is allowed to exit before the next one starts.  CUDA
allocations, TensorRT/ONNX sessions, DirectShow handles, and ffmpeg child
processes are consequently owned by one OS process and cannot accumulate over
a folder run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import time
import traceback
from typing import Any, Callable, Iterable, Mapping, Optional


ProgressCallback = Callable[[Mapping[str, Any]], None]
Worker = Callable[["VideoJob", Callable[..., None]], Optional[Mapping[str, Any]]]


@dataclass(frozen=True)
class VideoJob:
    """A pickle-safe description of one isolated video render."""

    input_path: str
    output_path: str
    group: str
    facesets: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class WorkerFailed(RuntimeError):
    """The child exited with an error after reporting its traceback."""


def _release_gpu_resources() -> None:
    """Best-effort cleanup; process exit remains the reclamation guarantee."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def _worker_entry(job: VideoJob, worker: Worker, events: Any) -> None:
    """Spawn target.  Keep this module-level for Windows pickling."""
    started = time.monotonic()

    def report(**event: Any) -> None:
        event.setdefault("type", "progress")
        event.setdefault("input_path", job.input_path)
        event.setdefault("group", job.group)
        event.setdefault("elapsed_seconds", time.monotonic() - started)
        try:
            events.put(event, timeout=1.0)
        except Exception:
            # UI reporting must never stall a CUDA worker.
            pass

    try:
        report(type="started")
        result = worker(job, report) or {}
        report(type="completed", result=dict(result))
    except BaseException as exc:
        report(type="failed", error=f"{type(exc).__name__}: {exc}",
               traceback=traceback.format_exc())
        raise
    finally:
        _release_gpu_resources()


class IsolatedVideoBatch:
    """Run videos one at a time in fresh ``multiprocessing.spawn`` workers.

    ``worker`` must be a module-level function (or another pickle-safe
    callable).  The enforced sequential policy is intentional: two renderers
    compete for the same GPU and defeat both the VRAM reclamation and the
    laptop's single-context safety policy.
    """

    def __init__(self, worker: Worker, *, poll_seconds: float = 0.25,
                 join_seconds: float = 15.0) -> None:
        self._worker = worker
        self._poll_seconds = max(0.05, float(poll_seconds))
        self._join_seconds = max(1.0, float(join_seconds))
        self._context = mp.get_context("spawn")

    def run(self, jobs: Iterable[VideoJob], *, on_progress: Optional[ProgressCallback] = None,
            stop_requested: Optional[Callable[[], bool]] = None) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for job in jobs:
            if stop_requested and stop_requested():
                break
            results.append(self._run_one(job, on_progress, stop_requested))
        return results

    def _run_one(self, job: VideoJob, on_progress: Optional[ProgressCallback],
                 stop_requested: Optional[Callable[[], bool]]) -> dict[str, Any]:
        events = self._context.Queue()
        process = self._context.Process(
            target=_worker_entry, args=(job, self._worker, events),
            name=f"roop-video-{Path(job.input_path).stem}", daemon=False)
        terminal: Optional[dict[str, Any]] = None
        terminated = False
        process.start()
        try:
            while process.is_alive():
                if stop_requested and stop_requested():
                    process.terminate()
                    terminated = True
                    terminal = {"type": "stopped", "input_path": job.input_path}
                    break
                try:
                    event = events.get(timeout=self._poll_seconds)
                except Empty:
                    continue
                if on_progress:
                    on_progress(event)
                if event.get("type") in {"completed", "failed"}:
                    terminal = event
            # The final event can be queued just before the child exits.
            while True:
                try:
                    event = events.get_nowait()
                except Empty:
                    break
                if on_progress:
                    on_progress(event)
                if event.get("type") in {"completed", "failed"}:
                    terminal = event
        finally:
            process.join(self._join_seconds)
            if process.is_alive():
                process.terminate()
                terminated = True
                process.join(self._join_seconds)
            exitcode = process.exitcode
            events.close()
            events.join_thread()

        if terminal is None:
            raise WorkerFailed(
                f"Worker exited without a terminal event for {job.input_path!r} "
                f"(exit code {exitcode}).")
        if terminal.get("type") == "failed":
            raise WorkerFailed(
                f"Worker failed for {job.input_path!r}: {terminal.get('error')}\n"
                f"{terminal.get('traceback', '')}")
        if terminal.get("type") == "stopped":
            return {"job": job.as_dict(), "status": "stopped", "exitcode": exitcode,
                    "terminated": terminated}
        if exitcode not in (0, None):
            raise WorkerFailed(
                f"Worker reported completion but exited {exitcode} for {job.input_path!r}.")
        return {"job": job.as_dict(), "status": "completed", "exitcode": exitcode,
                "terminated": terminated, **dict(terminal.get("result") or {})}
