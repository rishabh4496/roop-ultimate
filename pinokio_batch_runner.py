"""Run the roop-keep video folders with one fresh GPU process per video.

Examples:
    python pinokio_batch_runner.py --dry-run
    python pinokio_batch_runner.py

The parent process is intentionally lightweight.  It never imports the Roop
application or CUDA; all GPU state is created and destroyed in a spawn child
for each input video.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable

from roop.process_manager import IsolatedVideoBatch, VideoJob, WorkerFailed


PROJECT_ROOT = Path(__file__).resolve().parent
APP_ROOT = PROJECT_ROOT / "app"
APP_TESTS = APP_ROOT / "tests"
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def _reexec_in_app_environment() -> None:
    """Use Pinokio's app venv when this wrapper was launched by system Python."""
    app_python = APP_ROOT / "env" / "Scripts" / "python.exe"
    if not app_python.exists() or os.environ.get("ROOP_BATCH_APP_PYTHON") == "1":
        return
    try:
        if Path(sys.executable).resolve() == app_python.resolve():
            return
    except OSError:
        return
    environment = os.environ.copy()
    environment["ROOP_BATCH_APP_PYTHON"] = "1"
    os.execve(str(app_python), [str(app_python), str(Path(__file__).resolve()), *sys.argv[1:]],
               environment)


class QueueProgress:
    """Small tqdm-compatible meter forwarding progress over multiprocessing."""

    def __init__(self, total: int | None = None, **_: Any) -> None:
        self.total = int(total or 0)
        self.n = 0
        self._started = time.monotonic()
        self._last_report = 0.0
        self.rolling_rate = 0.0
        self.format_dict: dict[str, float] = {"rate": 0.0}
        self._report = None

    def bind(self, report: Any) -> "QueueProgress":
        self._report = report
        return self

    def set_postfix(self, *_: Any, **__: Any) -> None:
        return None

    def update(self, count: int = 1) -> None:
        self.n += int(count)
        elapsed = max(time.monotonic() - self._started, 1e-6)
        self.rolling_rate = self.n / elapsed
        self.format_dict["rate"] = self.rolling_rate
        now = time.monotonic()
        if self._report and (now - self._last_report >= 0.5 or
                             (self.total and self.n >= self.total)):
            self._last_report = now
            remaining = max(self.total - self.n, 0)
            self._report(frame_index=self.n, frame_total=self.total,
                         fps=self.rolling_rate,
                         eta_seconds=(remaining / self.rolling_rate
                                      if self.rolling_rate else None))

    def close(self) -> None:
        return None

    def __enter__(self) -> "QueueProgress":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _frame_count(video_path: str) -> int:
    import cv2

    capture = cv2.VideoCapture(video_path)
    try:
        return max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()


def render_video_worker(job: VideoJob, report: Any) -> dict[str, Any]:
    """Render a job inside the spawn child; imports GPU code only here."""
    # sample_bench is the repository's existing, direct batch-pipeline entry.
    # Place app first so imports resolve the real application package instead of
    # the small root-level launcher helper package.
    for path in (str(APP_TESTS), str(APP_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import os
    os.chdir(APP_ROOT)

    import angle_bench as ab
    import sample_bench as sb
    from angle_video import ensure_ffmpeg
    from two_face_video import auto_capture_targets, load_library_faceset
    from settings import Settings
    import roop.ProcessMgr as process_mgr_module

    input_path = str(Path(job.input_path).resolve())
    output_path = Path(job.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = _frame_count(input_path)
    report(message=f"initialising renderer ({total or 'unknown'} frames)", frame_total=total)

    class WorkerProgress(QueueProgress):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(total=total or kwargs.pop("total", None), **kwargs)
            self.bind(report)

    # ProcessMgr resolves ChunkedProgress from its module global.  Replacing it
    # here keeps the standard renderer unchanged while giving the parent real
    # frame/FPS/ETA events.
    process_mgr_module.ChunkedProgress = WorkerProgress
    ensure_ffmpeg()

    cfg = Settings("config.yaml")
    swap_model = cfg.swap_model
    mask_engine_display = cfg.mask_engine
    mask_engine = sb.map_mask_engine(mask_engine_display)
    enhancer = cfg.selected_enhancer or "None"
    provider = cfg.provider or "cuda"
    threads = cfg.max_threads
    globals_module = ab.init_pipeline(provider, swap_model, enhancer, mask_engine)
    globals_module.video_encoder = globals_module.CFG.output_video_codec
    globals_module.video_quality = globals_module.CFG.video_quality
    globals_module.execution_threads = threads
    globals_module.face_swap_mode = "selected"
    globals_module.track_identities = bool(globals_module.CFG.track_identities)
    globals_module.temporal_detection = bool(globals_module.CFG.temporal_detection)
    globals_module.CFG.track_identities = globals_module.track_identities
    globals_module.CFG.temporal_detection = globals_module.temporal_detection
    globals_module.stabilize_face = bool(globals_module.CFG.stabilize_face)
    options = ab.build_options(globals_module, swap_model, mask_engine,
                               bool(globals_module.CFG.use_source_bank))
    options.swap_mode = "selected"
    options.stabilize_face = globals_module.stabilize_face

    facesets = [load_library_faceset(name) for name in job.facesets]
    targets, groups = auto_capture_targets(input_path, expect=len(facesets),
                                           log_prefix="[pinokio-batch]", strict=False)
    if targets is None:
        frame_index, _, faces = sb.first_face_frame(input_path)
        targets, groups = [sb.select_primary_face(faces)], [0]
        report(message=f"target auto-capture fell back to frame {frame_index}")

    rendered, elapsed, _face_log = sb.run_swap(
        input_path, facesets, targets, groups, options, str(output_path.parent))
    if not rendered:
        raise RuntimeError("renderer completed without producing an output video")
    rendered_path = Path(rendered).resolve()
    if rendered_path != output_path:
        if output_path.exists():
            output_path.unlink()
        shutil.move(str(rendered_path), str(output_path))
    return {"output_path": str(output_path), "elapsed_seconds": elapsed,
            "frames": total}


def discover_jobs(root: Path, single_faceset: str, double_facesets: tuple[str, ...],
                  overwrite: bool) -> tuple[list[VideoJob], list[Path]]:
    jobs: list[VideoJob] = []
    skipped: list[Path] = []
    for group, facesets in (("single", (single_faceset,)), ("double", double_facesets)):
        source = root / group
        destination = root / f"{group}_results"
        destination.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            continue
        for video in sorted(path for path in source.rglob("*")
                            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS):
            # Preserve nested source folders so equal filenames never collide.
            relative = video.relative_to(source).with_suffix(".mp4")
            output = destination / relative
            if output.exists() and not overwrite:
                skipped.append(video)
                continue
            jobs.append(VideoJob(str(video.resolve()), str(output.resolve()), group, facesets))
    return jobs, skipped


def _configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pinokio_batch_runner")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(r"G:\pinokio\roop-keep"),
                        help="folder containing single/ and double/")
    parser.add_argument("--single-faceset", default="rhythm")
    parser.add_argument("--double-facesets", default="ashna,rhythm",
                        help="two comma-separated faceset names, left to right")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace already completed outputs")
    parser.add_argument("--dry-run", action="store_true",
                        help="create result folders and list jobs without rendering")
    parser.add_argument("--log", type=Path, default=None,
                        help="defaults to <root>/pinokio_batch_runner.log")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    double_facesets = tuple(item.strip() for item in args.double_facesets.split(",") if item.strip())
    if len(double_facesets) != 2:
        parser.error("--double-facesets must contain exactly two names")
    logger = _configure_logging(args.log.resolve() if args.log else root / "pinokio_batch_runner.log")
    jobs, skipped = discover_jobs(root, args.single_faceset, double_facesets, args.overwrite)
    logger.info("root=%s jobs=%d skipped=%d", root, len(jobs), len(skipped))
    for job in jobs:
        logger.info("%s: %s -> %s", job.group, job.input_path, job.output_path)
    if args.dry_run:
        return 0

    # Keep --dry-run usable with a plain system Python; actual rendering must
    # use the dependency-complete Pinokio environment before it spawns workers.
    if argv is None:
        _reexec_in_app_environment()

    def on_progress(event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "progress":
            logger.info("%s frame=%s/%s fps=%s eta=%s", Path(event["input_path"]).name,
                        event.get("frame_index"), event.get("frame_total"),
                        f"{event.get('fps', 0):.2f}", event.get("eta_seconds"))
        elif event_type in {"started", "completed", "failed"}:
            logger.info("%s %s", event_type.upper(), event)

    try:
        results = IsolatedVideoBatch(render_video_worker).run(jobs, on_progress=on_progress)
    except WorkerFailed as exc:
        logger.exception("batch stopped: %s", exc)
        return 1
    logger.info("batch finished: %d video(s) completed", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
