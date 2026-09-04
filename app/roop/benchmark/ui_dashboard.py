"""Presentation and control layer for the benchmark.

Everything a user-facing surface needs -- the React panel, the Gradio tab, the
CLI, and the HTTP API -- is built here ONCE and rendered four ways.  The rule
is that no surface computes a number: a score, a badge or a comparison row that
appeared in the React panel and not in ``--benchmark`` output would be a second
implementation with its own bugs.

    PreBenchmarkPrompt  -> the pre-run modal (face complexity, mode, the models
                           that are locked for the run)
    BenchmarkSession    -> starts the run, owns its thread, and publishes a
                           ProgressSnapshot for the live gauges
    DashboardReport     -> the results screen: score, avg/1% low, bottleneck
                           badge, the current-vs-recommended table, three presets
    apply_/decline_recommended_settings -> the accept/decline workflow

WHAT THIS LAYER REFUSES TO DO
-----------------------------
A results screen is a persuasion surface, which makes it the easiest place in
the project to launder an unmeasured number into something that looks like a
fact.  Three guards, each for a mistake this repo has already made:

* **A projection is labelled as one.**  "Expected FPS 17.8 (+59%)" is only
  printed when a run actually rendered that configuration.  Otherwise the cell
  reads "not measured" and the delta is omitted.  Stage share is never turned
  into a projected speedup -- ``detect = 42.4%`` predicted ~10% off the wall
  clock and measured +1%.

* **"Current" is read from the live configuration**, never from module
  defaults.  Three sources can express a setting here (``config.yaml``,
  ``settings.py``, the React ``defaults.js``) and they have silently drifted by
  34 keys before, which made a whole session's work inert.

* **The score is an index, not a measurement**, and it says what it is relative
  to.  It is comparable only between runs of the SAME workload mode on the same
  machine; a solo score and a group score are different tests.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from roop.benchmark.asset_manager import (
    STANDARDIZED_WINDOW_FRAMES,
    WorkloadMode,
    WorkloadSelector,
)
from roop.benchmark.storage import (
    get_latest_optimal_settings,
    load_benchmark_history,
    save_benchmark_result,
    update_setting_status,
)

SCHEMA_VERSION = 1

# Frame windows for the two offered durations.
#
# These are PRODUCT choices -- what a user is willing to wait for -- not
# acceptance windows. This pipeline runs at roughly 3-6 fps on the validation
# hardware, so 90 frames is about the promised 30 seconds and 270 about 90.
#
# Read the caveat in the mode detail text and mean it: this project's own rule
# puts the floor for an ACCEPTANCE claim at 600 frames, because a short window
# is dominated by warm-up and has reversed the sign of a result on both
# validation GPUs. These windows rank configurations; they do not settle them.
QUICK_WINDOW_FRAMES = 90
FULL_WINDOW_FRAMES = 270


# --------------------------------------------------------------------------
# choices presented by the pre-benchmark modal
# --------------------------------------------------------------------------

# "Target Face Complexity". The labels are the user's language; the values are
# the WorkloadMode the engine already defines, so this list cannot drift from
# what the runner can actually execute.
FACE_COMPLEXITY_CHOICES: Tuple[Dict[str, Any], ...] = (
    {"value": "1", "mode": WorkloadMode.SOLO.value, "label": "1 Face (Default)",
     "detail": "One target face per frame -- the common case."},
    {"value": "2", "mode": WorkloadMode.DUO.value, "label": "2 Faces",
     "detail": "Two targets, which is where source binding and per-face cost "
               "start to matter."},
    {"value": "all", "mode": WorkloadMode.GROUP.value, "label": "3+ Faces / Crowd",
     "detail": "Crowd frames. The heaviest case: per-face work multiplies "
               "while the frame budget does not."},
)

# "Benchmark Mode". The seconds are what the user is promised, so the frame
# windows are derived from them and from the standardized clip's own rate --
# not typed in twice.
BENCHMARK_MODE_CHOICES: Tuple[Dict[str, Any], ...] = (
    {"value": "quick", "label": "Quick Profile (30s / 90 frames)", "seconds": 30,
     "frames": QUICK_WINDOW_FRAMES,
     "detail": "One standardized window. Enough to rank settings; a short "
               "window measures warm-up, so it is not an acceptance claim."},
    {"value": "full", "label": "Full Stress & Thermal Test (90s / 270 frames)",
     "seconds": 90, "frames": FULL_WINDOW_FRAMES,
     "detail": "Three windows back to back, so sustained clocks and thermal "
               "retention become visible rather than the first-window peak."},
)

_MODE_BY_VALUE = {choice["value"]: choice for choice in BENCHMARK_MODE_CHOICES}
_FACES_BY_VALUE = {choice["value"]: choice for choice in FACE_COMPLEXITY_CHOICES}

# The notice the Decline path must show, verbatim, on every surface.
DECLINE_NOTICE = ("Settings saved to your profiles. You can review or apply "
                  "them anytime from Settings > Optimization Profiles.")

# The score's reference point.
#
# A "performance index" with no stated denominator is a number that cannot be
# checked. This one is explicit: 1000 points is REFERENCE_FPS on the
# standardized solo workload. It is a fixed constant so two runs on the same
# machine are comparable over time; it is NOT a claim that this rate is good,
# normal, or achievable on any particular GPU.
REFERENCE_FPS = 12.0
# Solo is the unit workload. Heavier modes do strictly more work per frame, so
# their raw fps is divided by the frame's face count before indexing --
# otherwise every machine would score worst on the test that stresses it most,
# which measures the test rather than the machine. These are the presets' own
# target face counts.
_WORKLOAD_WEIGHT = {WorkloadMode.SOLO.value: 1.0,
                    WorkloadMode.DUO.value: 2.0,
                    WorkloadMode.GROUP.value: 3.0}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if result != result or result in (float("inf"), float("-inf")) else result


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _setting(settings: Any, name: str, default: Any = None) -> Any:
    if settings is None:
        return default
    if isinstance(settings, Mapping):
        return settings.get(name, default)
    return getattr(settings, name, default)


def _live_config() -> Any:
    """The configuration the application is ACTUALLY running.

    Not ``settings.py``'s defaults and not the React snapshot: those three have
    drifted before and the results screen is the one place that difference
    would be invisible.
    """
    try:
        import roop.globals
        return roop.globals.CFG
    except Exception:
        return None


# --------------------------------------------------------------------------
# 1. the pre-benchmark modal
# --------------------------------------------------------------------------

@dataclass
class PreBenchmarkPrompt:
    """Everything the pre-run dialog renders, on any surface."""

    face_choices: Tuple[Dict[str, Any], ...] = FACE_COMPLEXITY_CHOICES
    mode_choices: Tuple[Dict[str, Any], ...] = BENCHMARK_MODE_CHOICES
    default_faces: str = "1"
    default_mode: str = "quick"
    active_models: Dict[str, str] = field(default_factory=dict)
    model_summary: str = ""
    warnings: List[str] = field(default_factory=list)
    can_run: bool = True

    @classmethod
    def build(cls, runner: Any = None) -> "PreBenchmarkPrompt":
        """Read the locked models from the live pipeline, never from a list.

        The models are held CONSTANT for the run -- the benchmark measures the
        user's configuration, not a stand-in for it. Four harnesses here once
        ran a "CodeFormer" arm with no enhancer at all because the name was
        matched by string and a miss silently added nothing, so the summary
        below is read back from the pipeline rather than echoed from the UI.
        """
        models: Dict[str, str] = {}
        warnings: List[str] = []
        try:
            if runner is None:
                from roop.benchmark.runner import BenchmarkRunner
                runner = BenchmarkRunner()
            models = dict(runner.inspect_active_models() or {})
        except Exception as exc:
            warnings.append("Could not read the active models (%s: %s). The "
                            "benchmark would not be measuring a known "
                            "configuration." % (type(exc).__name__, exc))
        return cls(active_models=models,
                   model_summary=cls.summarize(models),
                   warnings=warnings,
                   can_run=bool(models))

    @staticmethod
    def summarize(models: Mapping[str, str]) -> str:
        if not models:
            return "Testing with active models: unavailable"
        return ("Testing with active models: Swapper: %s, Enhancer: %s, Mask: %s"
                % (models.get("swapper") or "unknown",
                   models.get("enhancer") or "None",
                   models.get("mask_engine") or "None"))

    def as_dict(self) -> dict:
        result = asdict(self)
        result["face_choices"] = [dict(choice) for choice in self.face_choices]
        result["mode_choices"] = [dict(choice) for choice in self.mode_choices]
        result["decline_notice"] = DECLINE_NOTICE
        return result


def resolve_selection(faces: Any = "1", mode: Any = "quick") -> Dict[str, Any]:
    """Normalize the modal's two answers into engine arguments.

    Accepts what each surface naturally sends -- the CLI's ``1``/``2``/``all``,
    the React value, or a WorkloadMode -- and resolves through the engine's own
    ``WorkloadSelector`` so an unknown value degrades to solo rather than
    raising in a background thread where nobody sees it.
    """
    face_key = str(faces).strip().lower()
    face_choice = _FACES_BY_VALUE.get(face_key)
    if face_choice is None:
        # "3", "crowd", "group", a WorkloadMode, or junk: let the engine decide.
        workload = WorkloadSelector.get_workload(faces)
        face_choice = next((c for c in FACE_COMPLEXITY_CHOICES
                            if c["mode"] == workload.mode.value),
                           FACE_COMPLEXITY_CHOICES[0])
    else:
        workload = WorkloadSelector.get_workload(face_choice["mode"])

    mode_key = str(mode).strip().lower()
    mode_choice = _MODE_BY_VALUE.get(mode_key, _MODE_BY_VALUE["quick"])
    return {
        "faces": face_choice["value"],
        "face_label": face_choice["label"],
        "workload_mode": workload.mode.value,
        "workload": workload,
        "mode": mode_choice["value"],
        "mode_label": mode_choice["label"],
        "frame_window": int(mode_choice["frames"]),
        "estimated_seconds": int(mode_choice["seconds"]),
    }


# --------------------------------------------------------------------------
# 2. the live progress screen
# --------------------------------------------------------------------------

@dataclass
class ProgressSnapshot:
    """One poll of the running benchmark, shaped for gauges and graphs."""

    running: bool = False
    phase: str = "idle"
    status: str = "Idle"
    frame: int = 0
    total_frames: int = 0
    frames_remaining: int = 0
    progress_pct: float = 0.0
    current_fps: float = 0.0
    average_fps: float = 0.0
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    vram_pct: float = 0.0
    cpu_pct: float = 0.0
    gpu_pct: Optional[float] = None
    elapsed_sec: float = 0.0
    eta_sec: Optional[float] = None
    # Bounded series for the sparklines. Kept small on purpose: this is polled
    # about once a second and the whole snapshot is serialised each time.
    fps_series: List[float] = field(default_factory=list)
    vram_series: List[float] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    error: str = ""
    done: bool = False
    # Terminal without a result. Without this a cancelled run leaves
    # ``running`` False, ``done`` False and ``error`` empty, which is
    # indistinguishable from "not started yet" -- so a poller waiting for
    # any of the three polls forever.
    cancelled: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


_SERIES_LIMIT = 180        # ~3 minutes of once-a-second samples
_LOG_LIMIT = 200


class BenchmarkSession:
    """Owns one benchmark run and publishes its progress.

    Deliberately a single active run per process. Two concurrent benchmarks
    would each measure a GPU the other is using, and both answers would be
    wrong in a way nothing downstream could detect.
    """

    def __init__(self, runner_factory: Optional[Callable[[], Any]] = None,
                 clock: Callable[[], float] = time.time):
        self._runner_factory = runner_factory
        self._clock = clock
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self._reset()

    def _reset(self) -> None:
        self._snapshot = ProgressSnapshot()
        self._result: Optional[Any] = None
        self._report: Optional["DashboardReport"] = None
        self._selection: Dict[str, Any] = {}
        self._started_at = 0.0
        self._fps_samples: List[float] = []

    # -- lifecycle -------------------------------------------------------
    @property
    def running(self) -> bool:
        with self._lock:
            return self._snapshot.running

    def start(self, faces: Any = "1", mode: Any = "quick",
              persist: bool = True) -> Dict[str, Any]:
        """Begin a run. Returns immediately; poll ``snapshot()``."""
        with self._lock:
            if self._snapshot.running:
                return {"status": "running",
                        "message": "A benchmark is already in progress."}
            selection = resolve_selection(faces, mode)
            self._reset()
            self._cancel.clear()
            self._selection = selection
            self._started_at = self._clock()
            self._snapshot = ProgressSnapshot(
                running=True, phase="prepare", status="Loading models…",
                total_frames=selection["frame_window"],
                frames_remaining=selection["frame_window"])
            thread = threading.Thread(
                target=self._worker, args=(selection, bool(persist)),
                name="roop-benchmark-session", daemon=True)
            self._thread = thread
        thread.start()
        return {"status": "started",
                "workload": selection["workload_mode"],
                "mode": selection["mode"],
                "frames": selection["frame_window"],
                "estimated_seconds": selection["estimated_seconds"]}

    def cancel(self) -> Dict[str, str]:
        self._cancel.set()
        with self._lock:
            if self._snapshot.running:
                self._snapshot.status = "Cancelling…"
        return {"status": "cancelling"}

    def snapshot(self) -> ProgressSnapshot:
        with self._lock:
            # Copy the mutable series: the worker appends to these while
            # FastAPI is serialising the return value, and handing the
            # serialiser a list under mutation fails mid-response as a silently
            # broken poll.
            snap = ProgressSnapshot(**asdict(self._snapshot))
            snap.fps_series = list(self._snapshot.fps_series)
            snap.vram_series = list(self._snapshot.vram_series)
            snap.logs = list(self._snapshot.logs[-40:])
            return snap

    def report(self) -> Optional["DashboardReport"]:
        with self._lock:
            return self._report

    # -- the run ---------------------------------------------------------
    def _log(self, message: str) -> None:
        with self._lock:
            self._snapshot.logs.append(message)
            if len(self._snapshot.logs) > _LOG_LIMIT:
                del self._snapshot.logs[:-_LOG_LIMIT]

    def _on_progress(self, current: int = 0, total: int = 0,
                     fps: float = 0.0, **extra: Any) -> None:
        """Progress callback handed to the runner.

        Signature is permissive on purpose: the engine calls it positionally as
        ``(current, total, fps)`` and may grow keyword telemetry later. A
        callback that raises inside the runner's frame loop would abort a
        benchmark for a cosmetic reason.
        """
        if self._cancel.is_set():
            raise KeyboardInterrupt("benchmark cancelled by user")
        now = self._clock()
        fps = max(0.0, _number(fps))
        with self._lock:
            snap = self._snapshot
            snap.phase = "measuring"
            snap.status = "Measuring"
            snap.frame = _integer(current, snap.frame)
            snap.total_frames = _integer(total, snap.total_frames) or snap.total_frames
            snap.frames_remaining = max(0, snap.total_frames - snap.frame)
            snap.current_fps = fps
            if fps > 0:
                self._fps_samples.append(fps)
                snap.average_fps = sum(self._fps_samples) / len(self._fps_samples)
            snap.elapsed_sec = max(0.0, now - self._started_at)
            snap.progress_pct = (100.0 * snap.frame / snap.total_frames
                                 if snap.total_frames else 0.0)
            # ETA from the run's own average rather than the instantaneous
            # rate: the per-frame number swings far more than the machine does.
            if snap.average_fps > 0 and snap.frames_remaining:
                snap.eta_sec = snap.frames_remaining / snap.average_fps
            for key, target in (("vram_used_mb", "vram_used_mb"),
                                ("vram_total_mb", "vram_total_mb"),
                                ("cpu_pct", "cpu_pct"), ("gpu_pct", "gpu_pct")):
                if extra.get(key) is not None:
                    setattr(snap, target, _number(extra[key]))
            if snap.vram_total_mb:
                snap.vram_pct = 100.0 * snap.vram_used_mb / snap.vram_total_mb
            snap.fps_series.append(round(fps, 3))
            snap.vram_series.append(round(snap.vram_used_mb, 1))
            if len(snap.fps_series) > _SERIES_LIMIT:
                del snap.fps_series[:-_SERIES_LIMIT]
                del snap.vram_series[:-_SERIES_LIMIT]

    def _worker(self, selection: Mapping[str, Any], persist: bool) -> None:
        try:
            if self._runner_factory is not None:
                runner = self._runner_factory()
            else:
                from roop.benchmark.runner import BenchmarkRunner
                runner = BenchmarkRunner()
            self._log("Workload: %s, %s" % (selection["face_label"],
                                            selection["mode_label"]))
            result = runner.run(
                workload=selection["workload_mode"],
                frame_window=selection["frame_window"],
                persist=persist,
                progress_cb=self._on_progress)
            report = DashboardReport.from_result(result, selection)
            with self._lock:
                self._result = result
                self._report = report
                self._snapshot.running = False
                self._snapshot.done = True
                self._snapshot.phase = "complete"
                self._snapshot.status = "Benchmark complete"
                self._snapshot.progress_pct = 100.0
                self._snapshot.frames_remaining = 0
                if report.average_fps:
                    self._snapshot.average_fps = report.average_fps
            self._log("Complete: %.2f FPS average, score %d"
                      % (report.average_fps, report.score))
        except KeyboardInterrupt:
            with self._lock:
                self._snapshot.running = False
                self._snapshot.cancelled = True
                self._snapshot.phase = "cancelled"
                self._snapshot.status = "Cancelled"
            self._log("Cancelled by user.")
        except Exception as exc:
            message = "%s: %s" % (type(exc).__name__, exc)
            with self._lock:
                self._snapshot.running = False
                self._snapshot.phase = "failed"
                self._snapshot.status = "Failed"
                self._snapshot.error = message
            self._log("ERROR " + message)


_SESSION_LOCK = threading.Lock()
_SESSION: Optional[BenchmarkSession] = None


def get_session() -> BenchmarkSession:
    """The process-wide session the API and CLI both drive."""
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            _SESSION = BenchmarkSession()
        return _SESSION


# --------------------------------------------------------------------------
# 3. the results dashboard
# --------------------------------------------------------------------------

# Verdict -> the badge the results screen shows. The badge says what is
# limiting AND whether that is a healthy state, because "GPU Bound" alone reads
# as a fault when it is usually the goal.
_BADGES = {
    "GPU compute bound": ("GPU Bound — Optimum VRAM Utilization", "good",
                          "The GPU is the limit and memory is comfortable. "
                          "This is the healthy state for this pipeline."),
    "GPU VRAM bound": ("VRAM Limited — Reduce Memory Pressure", "critical",
                       "Peak VRAM is in the band where the driver starts "
                       "paging over PCIe. It fails as a slowdown, not an "
                       "error."),
    "CPU bound": ("CPU Bound — GPU Underfed", "warn",
                  "Host-side per-face work is starving the GPU."),
    "Disk I/O bound": ("Storage Bound — Slow Temp Volume", "warn",
                       "Frame reads and writes are the limit."),
    "RAM-bound": ("System RAM Limited", "warn",
                  "Host memory pressure is limiting the run."),
    "synchronization bound": ("Sync Bound — Concurrency Limited", "warn",
                              "Nothing is saturated; workers are waiting on "
                              "each other."),
    "unknown": ("Not Determined", "neutral",
                "The signals needed for a verdict were not reported."),
}


@dataclass
class ComparisonRow:
    """One row of the current-vs-recommended table."""

    setting: str
    key: str = ""
    current: str = ""
    recommended: str = ""
    delta: str = ""
    note: str = ""
    changed: bool = False
    # "measured" when a run rendered this configuration; "not measured" when
    # the value is a recommendation nothing has executed. The results screen
    # renders the second differently -- see the module docstring.
    evidence: str = "measured"
    requires_restart: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DashboardReport:
    """The full 3DMark-style results payload, surface-independent."""

    run_id: str = ""
    timestamp: str = ""
    score: int = 0
    score_basis: str = ""
    average_fps: float = 0.0
    p1_low_fps: float = 0.0
    peak_vram_mb: float = 0.0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    frames_processed: int = 0
    bottleneck: str = ""
    badge: str = ""
    badge_tone: str = "neutral"
    badge_detail: str = ""
    bottleneck_evidence: List[str] = field(default_factory=list)
    thermal: Dict[str, Any] = field(default_factory=dict)
    throttling_detected: bool = False
    active_models: Dict[str, str] = field(default_factory=dict)
    workload: Dict[str, Any] = field(default_factory=dict)
    device: Dict[str, Any] = field(default_factory=dict)
    comparison: List[ComparisonRow] = field(default_factory=list)
    presets: Dict[str, Any] = field(default_factory=dict)
    recommended_settings: Dict[str, Any] = field(default_factory=dict)
    applied: bool = False
    warnings: List[str] = field(default_factory=list)
    decline_notice: str = DECLINE_NOTICE

    @classmethod
    def from_result(cls, result: Any, selection: Optional[Mapping[str, Any]] = None,
                    config: Any = None) -> "DashboardReport":
        data = result.to_dict() if hasattr(result, "to_dict") else dict(result or {})
        metrics = dict(data.get("metrics") or {})
        thermal = dict(data.get("thermal_stability") or {})
        workload = dict(data.get("workload") or {})
        recommended = dict(data.get("recommended_settings") or {})
        config = config if config is not None else _live_config()

        average_fps = _number(metrics.get("avg_fps"))
        p1_low = _number(metrics.get("p1_low_fps"))
        score, basis = compute_score(metrics, thermal, workload)
        badge, tone, detail, verdict, evidence = classify_badge(
            metrics, thermal, data.get("device_specs") or {})

        report = cls(
            run_id=str(data.get("run_id", "")),
            timestamp=str(data.get("timestamp", "")),
            score=score, score_basis=basis,
            average_fps=average_fps, p1_low_fps=p1_low,
            peak_vram_mb=_number(metrics.get("peak_vram_mb")),
            avg_latency_ms=_number(metrics.get("avg_latency_ms")),
            p99_latency_ms=_number(metrics.get("p99_latency_ms")),
            frames_processed=_integer(metrics.get("frames_processed")),
            bottleneck=verdict, badge=badge, badge_tone=tone,
            badge_detail=detail, bottleneck_evidence=list(evidence),
            thermal=thermal,
            throttling_detected=bool(thermal.get("throttling_detected")),
            active_models=dict(data.get("active_models") or {}),
            workload=workload, device=dict(data.get("device_specs") or {}),
            recommended_settings=recommended,
            applied=bool(data.get("applied")),
        )
        report.comparison = build_comparison(recommended, config, metrics)
        report.presets = build_presets(recommended, metrics)
        if report.throttling_detected:
            report.warnings.append(
                "Throughput fell over the run (%.1f%% retention): this machine "
                "is thermally limited, so a short benchmark overstates what a "
                "long render will do."
                % _number(thermal.get("retention_pct"), 100.0))
        if not any(row.evidence == "measured" and row.changed
                   for row in report.comparison):
            report.warnings.append(
                "The recommended configuration was not itself rendered, so its "
                "throughput is an estimate rather than a measurement.")
        if selection:
            report.workload.setdefault("mode_label", selection.get("mode_label", ""))
            report.workload.setdefault("face_label", selection.get("face_label", ""))
        return report

    def as_dict(self) -> dict:
        result = asdict(self)
        result["comparison"] = [row.as_dict() for row in self.comparison]
        result["schema_version"] = SCHEMA_VERSION
        return result

    def summary_text(self) -> str:
        """The CLI rendering of the same dashboard."""
        lines = [
            "=" * 68,
            "  roop-ultimate Benchmark Results",
            "=" * 68,
            "  Score           : %d  (%s)" % (self.score, self.score_basis),
            "  Average FPS     : %.2f" % self.average_fps,
            "  1%% Low FPS      : %.2f" % self.p1_low_fps,
            "  Peak VRAM       : %.0f MiB" % self.peak_vram_mb,
            "  Bottleneck      : %s" % self.badge,
            "                    %s" % self.badge_detail,
            "  Models          : %s" % PreBenchmarkPrompt.summarize(self.active_models),
            "-" * 68,
            "  %-24s %-18s %s" % ("Setting", "Current", "Recommended"),
            "-" * 68,
        ]
        for row in self.comparison:
            recommended = row.recommended
            if row.delta:
                recommended = "%s (%s)" % (recommended, row.delta)
            if row.evidence != "measured":
                recommended = "%s [%s]" % (recommended, row.evidence)
            lines.append("  %-24s %-18s %s" % (row.setting, row.current, recommended))
        lines.append("-" * 68)
        for warning in self.warnings:
            lines.append("  ! " + warning)
        lines.append("=" * 68)
        return "\n".join(lines)


def compute_score(metrics: Mapping[str, Any], thermal: Mapping[str, Any],
                  workload: Mapping[str, Any]) -> Tuple[int, str]:
    """The performance index, and a plain-English statement of what it means.

    Three deliberate choices:

    * **The 1% low is part of the score, not a footnote.**  A run that averages
      well and stutters is worse to use than its average says, so the index is
      75% average and 25% the 1% low floor.
    * **Thermal retention scales it.**  A machine that starts fast and fades has
      not earned the peak it showed in the first window.
    * **Heavier workloads are weighted by their face count**, so a group run is
      not automatically a lower score than a solo run on the same machine --
      that would measure the test, not the hardware.
    """
    average = _number(metrics.get("avg_fps"))
    if average <= 0:
        return 0, "no measured throughput"
    p1_low = _number(metrics.get("p1_low_fps"), average)
    mode = str(workload.get("mode", WorkloadMode.SOLO.value))
    weight = _WORKLOAD_WEIGHT.get(mode, _number(workload.get("target_faces"), 1.0) or 1.0)

    effective = 0.75 * average + 0.25 * min(p1_low, average)
    retention = _number(thermal.get("retention_pct"), 100.0) / 100.0
    # Retention above 1.0 means the run got FASTER (a cold first window). That
    # is not a bonus -- it is warm-up -- so it is clamped rather than rewarded.
    retention = max(0.5, min(1.0, retention))
    index = 1000.0 * (effective * max(1.0, weight)) / REFERENCE_FPS * retention
    basis = ("1000 = %.0f FPS on the 1-face workload; 75%% average + 25%% 1%% "
             "low, scaled by %.0f%% thermal retention, weighted x%.0f for the "
             "%s workload. Comparable only to other runs of the same workload "
             "on this machine." % (REFERENCE_FPS, retention * 100.0, weight, mode))
    return int(round(index)), basis


def classify_badge(metrics: Mapping[str, Any], thermal: Mapping[str, Any],
                   device: Mapping[str, Any]) -> Tuple[str, str, str, str, List[str]]:
    """Run the optimizer's bottleneck engine over this run's telemetry.

    The classification is NOT re-implemented here. ``BottleneckAnalyzer`` owns
    the thresholds and the rule that absent telemetry yields ``unknown``
    instead of a confident guess; this only turns its verdict into a badge.
    """
    from roop.benchmark.optimizer import BottleneckAnalyzer, Measurement
    from roop.runtime_optimizer import HardwareProfile

    total_vram_gb = _number(device.get("vram_total_mb")) / 1024.0
    if not total_vram_gb:
        total_vram_gb = _number(device.get("vram_total_gb"))
    hardware = HardwareProfile(
        gpu_name=str(device.get("gpu_name", "")),
        vram_total_gb=total_vram_gb,
        cpu_logical_cores=_integer(device.get("cpu_logical_cores"), 1) or 1,
        cpu_physical_cores=_integer(device.get("cpu_physical_cores"), 1) or 1,
        ram_total_gb=_number(device.get("ram_total_gb")))

    measurement = Measurement.from_mapping({
        "fps": _number(metrics.get("avg_fps")),
        "frames": _integer(metrics.get("frames_processed")),
        "faces_seen": _integer(metrics.get("faces_detected")),
        "faces_swapped": _integer(metrics.get("faces_swapped")),
        "peak_vram_gb": _number(metrics.get("peak_vram_mb")) / 1024.0,
        "cpu_utilization_pct": _number(metrics.get("avg_cpu_pct"),
                                       _number(metrics.get("peak_cpu_pct"))),
        "per_core_peak_pct": metrics.get("peak_cpu_pct"),
        "gpu_utilization_pct": metrics.get("avg_gpu_pct",
                                           metrics.get("peak_gpu_pct")),
        "frame_time_median_ms": _number(metrics.get("avg_latency_ms")),
        "frame_time_p99_ms": _number(metrics.get("p99_latency_ms")),
        "disk_wait_pct": metrics.get("disk_wait_pct"),
    })
    verdict = BottleneckAnalyzer().classify(measurement, hardware)
    badge, tone, detail = _BADGES.get(
        verdict.kind, (verdict.kind.title(), "neutral", verdict.recommendation))
    return badge, tone, detail, verdict.kind, list(verdict.evidence)


# The rows of the comparison table: the label the user reads, the settings key
# it maps to, the key the engine's recommendation uses, and whether changing it
# needs a restart. Pool/provider values are env-backed and read once at process
# start by run.py, so promising a live effect for them would be a lie.
_COMPARISON_FIELDS: Tuple[Tuple[str, str, str, bool], ...] = (
    ("Execution Threads", "max_threads", "execution_threads", False),
    ("Temp Image Format", "output_image_format", "temp_frame_format", False),
    ("Provider Memory Limit", "perf_gpu_mem_limit", "gpu_memory_limit_mb", True),
    ("ONNX Memory Arena", "perf_ort_arena_strategy", "arena_extend_strategy", True),
    ("cudnn_conv_search", "perf_cudnn_conv_algo", "cudnn_conv_algo_search", True),
    ("Execution Provider", "provider", "execution_provider", True),
    ("Video Encoder", "output_video_codec", "output_video_encoder", False),
)


# The engine and the settings layer were written in different sessions and do
# not agree on every spelling. Translating here -- once, at the boundary --
# beats teaching either side about the other, and beats the alternative that
# was actually happening: an unrecognised key is simply absent from the table
# and never applied, which is indistinguishable from a successful run that
# changed nothing.
_REC_KEY_ALIASES = {
    "temp_format": "temp_frame_format",
    "temp_image_format": "temp_frame_format",
    "threads": "execution_threads",
    "provider": "execution_provider",
    "gpu_mem_limit_mb": "gpu_memory_limit_mb",
    "video_encoder": "output_video_encoder",
}

# ``roop.globals.CFG.provider`` stores the SHORT name and is tested by
# membership (`if self.provider in ['cuda', 'tensorrt']`), while the runner
# recommends the onnxruntime class name. Writing the latter into the config
# does not raise -- it just quietly fails that test and turns the accelerated
# paths off.
_PROVIDER_SHORT_NAMES = {
    "cudaexecutionprovider": "cuda",
    "cpuexecutionprovider": "cpu",
    "tensorrtexecutionprovider": "tensorrt",
    "rocmexecutionprovider": "rocm",
    "dmlexecutionprovider": "dml",
}


def normalize_recommendation(recommended: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Canonicalize an engine recommendation for the settings layer.

    Returns a new mapping; the input is never mutated. An unknown provider is
    passed through untouched rather than guessed at -- a value the application
    rejects loudly is better than one silently rewritten into something else.
    """
    result: Dict[str, Any] = {}
    for key, value in dict(recommended or {}).items():
        result[_REC_KEY_ALIASES.get(str(key), str(key))] = value
    provider = result.get("execution_provider")
    if isinstance(provider, str):
        result["execution_provider"] = _PROVIDER_SHORT_NAMES.get(
            provider.strip().lower(), provider)
    return result


def _format_value(value: Any) -> str:
    if value is None or value == "":
        return "Unset"
    if isinstance(value, bool):
        return "On" if value else "Off"
    if isinstance(value, float):
        return ("%.2f" % value).rstrip("0").rstrip(".")
    return str(value)


def build_comparison(recommended: Mapping[str, Any], config: Any,
                     metrics: Mapping[str, Any]) -> List[ComparisonRow]:
    """Build the current-vs-recommended table from the LIVE configuration."""
    recommended = normalize_recommendation(recommended)
    rows: List[ComparisonRow] = []
    for label, config_key, rec_key, restart in _COMPARISON_FIELDS:
        if rec_key not in recommended:
            continue
        current_value = _setting(config, config_key)
        recommended_value = recommended.get(rec_key)
        current_text = _format_value(current_value)
        recommended_text = _format_value(recommended_value)
        changed = current_text != recommended_text
        row = ComparisonRow(
            setting=label, key=config_key, current=current_text,
            recommended=recommended_text, changed=changed,
            requires_restart=restart and changed,
            evidence="measured" if not changed else "projected")
        if restart and changed:
            row.note = "Takes effect on the next application start."
        if config_key == "output_image_format" and changed:
            # The one recommendation that is a QUALITY decision wearing a speed
            # number: these frames are the encoder's input, so a lossy format
            # puts a lossy generation inside the delivered video.
            if str(recommended_value).lower() in ("jpg", "jpeg", "webp"):
                row.note = ("Lossy. Temp frames are the encoder's input, so "
                            "this re-compresses every swapped face before the "
                            "video is written.")
                row.evidence = "projected — quality trade-off"
        rows.append(row)

    measured_fps = _number(metrics.get("avg_fps"))
    projected_fps = _number(recommended.get("expected_fps"))
    fps_row = ComparisonRow(
        setting="Expected FPS", key="expected_fps",
        current="%.1f FPS" % measured_fps if measured_fps else "not measured",
        recommended="not measured", evidence="not measured")
    if projected_fps > 0 and measured_fps > 0:
        fps_row.recommended = "%.1f FPS" % projected_fps
        fps_row.delta = "%+.0f%%" % ((projected_fps - measured_fps) / measured_fps * 100.0)
        fps_row.changed = True
        # The engine supplies this as an estimate unless a run rendered the
        # recommended configuration; nothing here upgrades it to a measurement.
        fps_row.evidence = str(recommended.get("expected_fps_evidence", "estimated"))
        fps_row.note = ("Estimated from the recommended settings; it has not "
                        "been rendered." if fps_row.evidence != "measured" else "")
    rows.append(fps_row)
    return rows


def build_presets(recommended: Mapping[str, Any],
                  metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """The three selectable preset buttons.

    Built from the engine's recommendation when it supplies presets, and
    otherwise derived here so the buttons always exist -- a results screen with
    a dead button is worse than one with a conservative answer.
    """
    recommended = normalize_recommendation(recommended)
    supplied = recommended.get("presets")
    if isinstance(supplied, Mapping) and supplied:
        return {str(key): dict(value) if isinstance(value, Mapping) else value
                for key, value in supplied.items()}

    threads = _integer(recommended.get("execution_threads"), 0)
    if threads <= 0:
        config = _live_config()
        threads = _integer(_setting(config, "max_threads", 4), 4)
    return {
        "max_throughput": {
            "name": "Max Throughput", "tuning": {"max_threads": threads},
            "rationale": "The measured best, within safe VRAM headroom.",
        },
        "balanced": {
            "name": "Balanced",
            "tuning": {"max_threads": max(1, min(threads, max(1, threads)))},
            "rationale": "The knee: the smallest footprint that keeps "
                         "essentially all of the throughput.",
            "recommended": True,
        },
        "stable_low_power": {
            "name": "Stable / Quiet",
            "tuning": {"max_threads": max(1, threads // 2)},
            "rationale": "Half the workers for background rendering and "
                         "quieter fans.",
        },
    }


# --------------------------------------------------------------------------
# 4. accept / decline
# --------------------------------------------------------------------------

# Recommendation key -> the live settings attribute it writes.
_APPLY_MAP: Dict[str, str] = {
    "execution_threads": "max_threads",
    "temp_frame_format": "output_image_format",
    "gpu_memory_limit_mb": "perf_gpu_mem_limit",
    "execution_provider": "provider",
    "output_video_encoder": "output_video_codec",
    "arena_extend_strategy": "perf_ort_arena_strategy",
    "cudnn_conv_algo_search": "perf_cudnn_conv_algo",
}

# Applied settings that ALSO have a live runtime mirror in roop.globals.
#
# CFG is the durable truth and the next render reads it (core.py sets
# execution_threads from CFG.max_threads at render start), but a user who
# accepts a recommendation expects it to be live now, not after a restart --
# and a `--benchmark-apply` CLI session has no later render to pick it up. So
# the applicable subset is mirrored immediately.
#
# `provider` needs translating on the way out: CFG stores the short name and
# roop.globals.execution_providers holds onnxruntime's class names.
_PROVIDER_ORT_NAMES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "dml": "DmlExecutionProvider",
}


def _mirror_to_globals(config_key: str, value: Any) -> Optional[Tuple[str, Any]]:
    """Return the ``(globals attribute, value)`` this setting drives, if any."""
    try:
        import roop.globals as runtime
    except Exception:
        return None
    if config_key == "max_threads":
        threads = _integer(value, 0)
        if threads > 0:
            runtime.execution_threads = threads
            return "execution_threads", threads
    elif config_key == "output_video_codec" and value:
        runtime.video_encoder = str(value)
        return "video_encoder", str(value)
    # `provider` is deliberately absent. ORT sessions already built keep the
    # provider they were built with, so mirroring it would leave this process
    # running two providers at once -- worse than a clean "takes effect on the
    # next launch", which is what _RESTART_REQUIRED reports for it.
    return None
# These are read once at process start (run.py exports them as ROOP_* env), so
# writing them changes the NEXT launch, not this one. Saying "applied" for them
# would have the user measure a change that has not happened.
_RESTART_REQUIRED = {"perf_gpu_mem_limit", "provider", "perf_ort_arena_strategy",
                     "perf_cudnn_conv_algo"}


def apply_recommended_settings(recommended: Optional[Mapping[str, Any]] = None,
                               run_id: str = "", config: Any = None,
                               storage_path: Any = None,
                               allow_lossy_temp_frames: bool = False) -> Dict[str, Any]:
    """Write the recommendation into the live configuration.

    Returns ``applied`` (in effect now), ``pending`` (written, needs a
    restart), and ``skipped`` with a reason for each -- the same three-way
    split ``roop.bench.apply_recommendation`` uses, because "applied" has to
    mean the render will actually behave differently.
    """
    if recommended is None:
        recommended = get_latest_optimal_settings(storage_path) or {}
    recommended = normalize_recommendation(recommended)
    config = config if config is not None else _live_config()
    applied: Dict[str, Any] = {}
    pending: Dict[str, Any] = {}
    skipped: Dict[str, str] = {}
    live: Dict[str, Any] = {}          # what changed in roop.globals right now

    if config is None:
        return {"status": "error", "applied": {}, "pending": {}, "skipped": {},
                "message": "No live configuration is loaded; nothing was changed."}

    for rec_key, config_key in _APPLY_MAP.items():
        if rec_key not in recommended:
            continue
        value = recommended[rec_key]
        if not hasattr(config, config_key):
            skipped[config_key] = "not a setting on this build"
            continue
        if (config_key == "output_image_format"
                and str(value).lower() in ("jpg", "jpeg", "webp")
                and not allow_lossy_temp_frames):
            # Refused by default, and the reason is the delivered video rather
            # than the scratch directory. An explicit opt-in is required.
            skipped[config_key] = (
                "lossy temp frames re-compress every swapped face before the "
                "video is encoded; re-apply with allow_lossy_temp_frames=true "
                "to accept that trade")
            continue
        current = getattr(config, config_key)
        if _format_value(current) == _format_value(value):
            skipped[config_key] = "already set to the recommended value"
            continue
        setattr(config, config_key, value)
        if config_key in _RESTART_REQUIRED:
            pending[config_key] = value
        else:
            applied[config_key] = value
            mirrored = _mirror_to_globals(config_key, value)
            if mirrored:
                live[mirrored[0]] = mirrored[1]

    try:
        config.save()
    except Exception as exc:
        return {"status": "error", "applied": applied, "pending": pending,
                "skipped": skipped, "live_globals": live,
                "message": "Settings could not be saved: %s: %s"
                           % (type(exc).__name__, exc)}

    marked = False
    if run_id:
        try:
            marked = bool(update_setting_status(run_id, True, storage_path))
        except Exception:
            marked = False

    message = "Applied %d setting(s)." % len(applied)
    if pending:
        message += (" %d more were saved and take effect on the next "
                    "application start." % len(pending))
    return {"status": "applied", "applied": applied, "pending": pending,
            "skipped": skipped, "live_globals": live,
            "restart_required": bool(pending),
            "history_updated": marked, "message": message}


def decline_recommended_settings(result: Any = None, run_id: str = "",
                                 storage_path: Any = None) -> Dict[str, Any]:
    """Keep the current settings, but never lose the run.

    The benchmark is the expensive part; the decision is cheap and reversible.
    A declined run is persisted in full so it can be applied later from
    Settings > Optimization Profiles.
    """
    saved_id = run_id
    if result is not None:
        if hasattr(result, "to_storage_record"):
            payload = result.to_storage_record()
        elif isinstance(result, dict) and "best_metrics" in result and "baseline_metrics" in result:
            payload = dict(result)
        elif isinstance(result, dict):
            from roop.benchmark.runner import BenchmarkRunResult
            payload = BenchmarkRunResult.from_dict(result).to_storage_record()
        else:
            payload = dict(result)
        payload["status"] = "declined"
        try:
            saved_id = save_benchmark_result(payload, storage_path)
        except Exception as exc:
            return {"status": "error", "applied": False,
                    "message": "The run could not be saved: %s: %s"
                               % (type(exc).__name__, exc)}
    elif run_id:
        try:
            update_setting_status(run_id, False, storage_path)
        except Exception:
            pass
    return {"status": "declined", "applied": False, "run_id": saved_id,
            "message": DECLINE_NOTICE, "notice": DECLINE_NOTICE}


# Every setting the benchmark is allowed to write. Revert restores exactly
# these and nothing else: a user clearing benchmark overrides has not asked to
# lose their theme, their faceset, or any of the other ~200 settings.
BENCHMARK_OWNED_SETTINGS: Tuple[str, ...] = tuple(sorted(set(_APPLY_MAP.values())))


def stock_defaults(keys: Sequence[str] = ()) -> Dict[str, Any]:
    """The shipped default for each benchmark-owned setting.

    Read from a throwaway ``Settings`` built on a path that does not exist, so
    the values are the ones a fresh install would have. That instance is never
    saved -- bringing the file into being would create a second config beside
    the user's.
    """
    keys = tuple(keys or BENCHMARK_OWNED_SETTINGS)
    try:
        import os
        import tempfile
        import settings as settings_module
        with tempfile.TemporaryDirectory(prefix="roop_defaults_") as tmp:
            probe = settings_module.Settings(
                os.path.join(tmp, "does-not-exist.yaml"))
            return {name: getattr(probe, name) for name in keys
                    if hasattr(probe, name)}
    except Exception:
        return {}


def revert_to_default_settings(config: Any = None,
                               keys: Sequence[str] = ()) -> Dict[str, Any]:
    """Clear benchmark overrides, restoring the shipped defaults.

    Scoped deliberately to ``BENCHMARK_OWNED_SETTINGS``. "Revert to default"
    on a results screen means "undo what this benchmark changed", not "reset
    the application", and the second reading would be a destructive surprise.
    """
    config = config if config is not None else _live_config()
    if config is None:
        return {"status": "error", "reverted": {}, "unchanged": {},
                "message": "No live configuration is loaded; nothing changed."}

    defaults = stock_defaults(keys)
    if not defaults:
        return {"status": "error", "reverted": {}, "unchanged": {},
                "message": "The shipped defaults could not be read; nothing "
                           "was changed rather than guessing at them."}

    reverted: Dict[str, Any] = {}
    unchanged: Dict[str, Any] = {}
    live: Dict[str, Any] = {}
    for name, default in defaults.items():
        if not hasattr(config, name):
            continue
        current = getattr(config, name)
        if _format_value(current) == _format_value(default):
            unchanged[name] = current
            continue
        setattr(config, name, default)
        reverted[name] = default
        mirrored = _mirror_to_globals(name, default)
        if mirrored:
            live[mirrored[0]] = mirrored[1]
    try:
        config.save()
    except Exception as exc:
        return {"status": "error", "reverted": reverted, "unchanged": unchanged,
                "live_globals": live,
                "message": "Defaults could not be saved: %s: %s"
                           % (type(exc).__name__, exc)}
    return {
        "status": "reverted", "reverted": reverted, "unchanged": unchanged,
        "live_globals": live,
        "message": ("Restored %d benchmark setting(s) to their shipped "
                    "defaults." % len(reverted) if reverted else
                    "Already at the shipped defaults; nothing changed."),
    }


def _bootstrap_config(log: Callable[[str], None] = print) -> Any:
    """Ensure roop.globals.CFG exists before anything reads the configuration.

    Idempotent: an entry point that already built CFG (core.run does) keeps
    its instance untouched. Only a path that has not -- run.py's benchmark
    branch -- gets one created here, pointed at the same config.yaml the
    application uses.
    """
    try:
        import roop.globals as runtime
    except Exception:
        return None
    if getattr(runtime, "CFG", None) is not None:
        return runtime.CFG
    try:
        from settings import Settings
        runtime.CFG = Settings("config.yaml")
        # Mirror the same runtime globals core.run() derives from CFG, so the
        # benchmark measures the configured thread count rather than None.
        if getattr(runtime, "execution_threads", None) in (None, 0):
            runtime.execution_threads = runtime.CFG.max_threads
        if getattr(runtime, "video_encoder", None) in (None, ""):
            runtime.video_encoder = runtime.CFG.output_video_codec
        if getattr(runtime, "video_quality", None) is None:
            runtime.video_quality = runtime.CFG.video_quality
        return runtime.CFG
    except Exception as exc:
        log("  ! could not load config.yaml (%s: %s); the benchmark would be "
            "measuring module defaults rather than your configuration"
            % (type(exc).__name__, exc))
        return None


def run_cli_benchmark(faces: str = "1", mode: str = "quick",
                      apply_result: bool = False,
                      log: Callable[[str], None] = print) -> int:
    """Headless benchmark: the same engine and dashboard, rendered as text.

    Shared by both entry points (`run.py` for the React launcher, `core.py`
    for the classic CLI) so the terminal cannot disagree with the panel about
    a score, a badge or a recommendation.

    Deliberately does NOT start the API or the Gradio UI: a benchmark shares
    the GPU with whatever else the process is doing, and a run that also
    served a render would be measuring a busy card.
    """
    from roop.benchmark.runner import BenchmarkRunner

    # Load the configuration if this entry point has not already.
    #
    # MEASURED, 2026-09-05: `run.py --benchmark` reported
    # "Swapper: inswapper, Enhancer: None, Mask: None" against a config.yaml
    # holding realswap / UltraMax / RealityUX. Only `core.run()` creates
    # roop.globals.CFG, and run.py's benchmark branch fires before it, so the
    # model readback and every "Current Value" cell fell through to module
    # defaults -- the same silent wrong-source failure the readback was fixed
    # for once already, arriving by a different route.
    _bootstrap_config(log)

    selection = resolve_selection(faces, mode)
    try:
        runner = BenchmarkRunner()
    except Exception as exc:
        log("Benchmark could not start: %s: %s" % (type(exc).__name__, exc))
        return 1

    prompt = PreBenchmarkPrompt.build(runner)
    log("")
    log(prompt.model_summary)
    for warning in prompt.warnings:
        log("  ! " + warning)
    log("Workload: %s | %s | %d frames"
        % (selection["face_label"], selection["mode_label"],
           selection["frame_window"]))
    log("")

    state = {"last": 0.0}

    def progress(current=0, total=0, fps=0.0, **_extra):
        # One line every ~2 seconds. A per-frame print would cost more than
        # the thing it is reporting on.
        now = time.time()
        if now - state["last"] < 2.0 and current < total:
            return
        state["last"] = now
        pct = (100.0 * current / total) if total else 0.0
        log("  [%5.1f%%] frame %d/%d  %.2f FPS" % (pct, current, total, fps))

    try:
        result = runner.run(workload=selection["workload_mode"],
                            frame_window=selection["frame_window"],
                            persist=True, progress_cb=progress)
    except KeyboardInterrupt:
        log("\nBenchmark cancelled.")
        return 130
    except Exception as exc:
        log("\nBenchmark failed: %s: %s" % (type(exc).__name__, exc))
        return 1

    report = DashboardReport.from_result(result, selection)
    log(report.summary_text())

    if apply_result:
        outcome = apply_recommended_settings(
            recommended=report.recommended_settings, run_id=report.run_id)
        log("  " + str(outcome.get("message", "")))
        for key, value in (outcome.get("live_globals") or {}).items():
            log("    live now: roop.globals.%s = %s" % (key, value))
        for key, value in (outcome.get("pending") or {}).items():
            log("    pending (needs a restart): %s = %s" % (key, value))
        for key, reason in (outcome.get("skipped") or {}).items():
            log("    skipped: %s -- %s" % (key, reason))
    else:
        outcome = decline_recommended_settings(run_id=report.run_id)
        log("  " + str(outcome.get("message", "")))
    log("")
    return 0


def list_saved_profiles(storage_path: Any = None,
                        limit: int = 20) -> List[Dict[str, Any]]:
    """History for the Settings > Optimization Profiles list."""
    try:
        history = load_benchmark_history(storage_path)
    except Exception:
        return []
    rows: List[Dict[str, Any]] = []
    for record in reversed(history[-max(1, int(limit)):]):
        metrics = record.get("best_metrics") or {}
        workload = record.get("workload") or {}
        score, _ = compute_score(metrics, record.get("thermal_stability") or {},
                                 workload)
        rows.append({
            "run_id": record.get("run_id", ""),
            "timestamp": record.get("timestamp", ""),
            "score": score,
            "avg_fps": round(_number(metrics.get("avg_fps")), 2),
            "p1_low_fps": round(_number(metrics.get("p1_low_fps")), 2),
            "workload": workload.get("name") or workload.get("mode", ""),
            "applied": record.get("status") == "accepted",
            "active_models": record.get("active_models") or {},
            "recommended_settings": {
                "execution_threads": (record.get("presets") or {}).get("balanced", {}).get("threads", 0),
                "temp_format": (record.get("presets") or {}).get("balanced", {}).get("temp_format", ""),
                "provider_options": (record.get("presets") or {}).get("balanced", {}).get("provider_options", {}),
            },
        })
    return rows


__all__ = [
    "BENCHMARK_MODE_CHOICES",
    "DECLINE_NOTICE",
    "FACE_COMPLEXITY_CHOICES",
    "REFERENCE_FPS",
    "BenchmarkSession",
    "ComparisonRow",
    "DashboardReport",
    "PreBenchmarkPrompt",
    "ProgressSnapshot",
    "apply_recommended_settings",
    "build_comparison",
    "build_presets",
    "classify_badge",
    "compute_score",
    "decline_recommended_settings",
    "get_session",
    "revert_to_default_settings",
    "run_cli_benchmark",
    "stock_defaults",
    "BENCHMARK_OWNED_SETTINGS",
    "normalize_recommendation",
    "list_saved_profiles",
    "resolve_selection",
]
