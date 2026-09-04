"""Guided parameter search, bottleneck analysis, and preset generation.

WHAT THIS ADDS, AND WHAT IT DELIBERATELY DOES NOT
-------------------------------------------------
The measurement layer already exists and is not duplicated here:

* ``roop.runtime_optimizer`` -- ``HardwareProfiler`` (GPU/CPU/RAM facts),
  ``WorkloadProfile``, ``RuntimeTuning`` + ``ResourceManager`` (the safe
  bounds), ``RuntimeMonitor`` (rolling per-stage telemetry), and
  ``RuntimeAutotuner`` (a bounded staged search over backend/precision/pools).
* ``roop.bench`` -- isolated per-stage costs, pool knees, encoder/decoder I/O.

This module is the *search and decision* layer on top of them:

1. an explicit, inspectable parameter search space (threads, ORT execution
   provider threading + arena allocator, temp frame format, memory limits);
2. a guided multi-pass search -- a coarse thread sweep that locates the
   contention inflection, then a VRAM headroom pass, then a disk I/O pass --
   instead of an exhaustive grid;
3. a bottleneck classifier that reports the *evidence* for its verdict and
   says ``unknown`` when the signals it needs were never measured;
4. three presets: Max Throughput, Balanced, Stable / Low-Power.

THE FIVE RULES THIS FILE IS BUILT AROUND
----------------------------------------
Every one was learned by shipping the opposite, and every one is enforced in
code below rather than left to the caller's discipline.

R1. **The acceptance threshold is measured on the live machine, never fixed.**
    ``RuntimeAutotuner.MIN_IMPROVEMENT`` was a constant 1%.  Two runs of the
    same deterministic search on the same RTX 4070 four days apart returned
    "promote nothing" and "+3.59%, promote trt_context_count=1" -- and that
    second promotion would have *halved* a TensorRT pool measured as worth
    +46% at 2.  The run-to-run spread is ~1.6% at 60 frames, ~8% at 600 on the
    4070, and ~3.3% at 600 on the RTX 3060.  A constant cannot be right on two
    machines.  See ``NoiseFloor``.

R2. **Counterbalance every A/B (A/B then B/A).**  The first arm of a process
    pays the TensorRT engine build and reads several fps slow.  Two changes
    measured as NEUTRAL read +21.8% and +9.8% when a forward-only order was
    trusted.  See ``counterbalanced_pairs`` and ``order_corrected``.

R3. **A configuration that goes faster by doing less work has not got faster.**
    A dedent in ``_build_temporal_faces`` disabled the swap on the shipped
    default path; throughput rose 12.9 -> 19.0 fps (+47%), the return code was
    0, the swap audit read 100%, and 1575 tests passed.  Every candidate here
    carries ``faces_seen``/``faces_swapped`` and is rejected as NOT COMPARABLE
    when its work differs from the baseline's.  See ``Measurement.comparable_to``.

R4. **Stage share is not a speedup budget.**  ``ROOP_PROFILE`` reports thread
    time summed across workers.  ``detect = 42.4%`` predicted ~10% off the wall
    clock; the measured end-to-end effect was +1%.  The classifier below names
    a dominant stage but never converts its share into a projected speedup.

R5. **Prove the path executed before believing "no effect".**  The adaptive
    controller was wired to the sequential writer while production renders
    through the parallel one, so it declined to act on neither GPU -- it was
    never reached.  Phase C therefore *probes reachability* of the temp-frame
    path before sweeping its format, rather than sweeping a path a real render
    never takes.

MEASUREMENT IS INJECTED
-----------------------
``GuidedOptimizer`` requires a ``measure`` callback exactly as
``RuntimeAutotuner`` does.  There is no synthetic fallback: a caller must not
be able to mistake an isolated GPU probe for an end-to-end pipeline result.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from roop.runtime_optimizer import (
    HardwareProfile,
    HardwareProfiler,
    ResourceManager,
    RuntimeTuning,
    WorkloadProfile,
)

SCHEMA_VERSION = 1

# The minimum window an acceptance claim may be made over.
#
# MEASURED, on both validation GPUs, in OPPOSITE DIRECTIONS -- which is what
# makes it a rule rather than a preference:
#
#     3060, p_plus_e vs auto:   +19.6% at 120 frames,  -0.5% at 600
#     4070, p_only  vs auto:    -10.7% at 120 frames,  +1.4% at 600
#
# Absolute throughput is also ~2-2.5x higher at 600 frames than at 120 on the
# same clip: a short window is dominated by warm-up, not by steady state.
ACCEPTANCE_FRAMES = 600

# A shorter window is allowed for *ranking* passes that only need an ordering
# and will be confirmed at full length.  Anything measured there is marked
# ``provisional`` and can never be promoted on its own.
RANKING_FRAMES = 120

# Fraction of the best value a smaller/cheaper setting may give up and still be
# preferred.  These reproduce ``roop.bench.THREAD_GAIN`` / ``POOL_GAIN``, which
# were fitted to real device curves: a measured heavy thread curve ran 9.09 f/s
# at 12 threads and 9.58 at 32, so a 2% bar bought a near-tripling of the
# worker count for 5%.
THREAD_GAIN = 0.05
POOL_GAIN = 0.04

# VRAM policy, from the requirement and from measurement.
#
# 0.85 is the hard admission ceiling used by Phase B: a level whose peak would
# exceed it is not offered at all.  On an RTX 3060 6GB the "keep the enhancer"
# arms peak at 91-94% of 6144 MB, which is the band where this pipeline stops
# OOMing and starts *thrashing* -- the driver pages contexts over PCIe, the
# card sits near 100% "utilisation" at a third of its power limit, and
# throughput collapses from 45.3 fps to 2-2.5 without ever raising an error.
# A hang, not an OOM, is what this ceiling exists to prevent.
VRAM_ADMISSION_CEILING = 0.82
# Balanced additionally refuses anything above this, per the preset spec.
VRAM_BALANCED_CEILING = 0.80
# Balanced targets this share of the best admissible arm: enough of the peak to
# be worth having, cheap enough to be quiet and OOM-free on a long render.
BALANCED_FPS_BAND = (0.90, 0.95)

# TensorRT allocates context memory on the FIRST INFERENCE, not at session
# build time, so a build-time free-VRAM check sees nothing.  Leave this much
# unallocated so a level measured with 200 MB to spare is never promoted.
VRAM_RESERVE_GB = 1.25


# --------------------------------------------------------------------------
# coercion helpers
#
# Kept local so this module stays stdlib-only importable, matching
# runtime_optimizer.  roop.bench pulls onnxruntime at import time and cannot be
# a module-scope dependency of a decision layer that tests must import cheaply.
# --------------------------------------------------------------------------

def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(result) or math.isinf(result) else result


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _mean(values: Sequence[Optional[float]]) -> float:
    usable = [v for v in values if v is not None]
    return sum(usable) / len(usable) if usable else 0.0


def knee(values: Sequence[float], labels: Sequence[Any], gain: float) -> Any:
    """Smallest label whose value is within ``gain`` of the best.

    Reproduces ``roop.bench._knee``.  It is re-stated rather than imported
    because ``roop.bench`` imports onnxruntime at module scope; the semantics
    are asserted equal against the original in the test suite.

    Picking the argmax alone buys a fraction of a percent for a doubling of
    threads or VRAM.
    """
    if not values:
        return labels[0] if labels else 1
    best = max(values)
    for value, label in zip(values, labels):
        if value >= best * (1.0 - gain):
            return label
    return labels[max(range(len(values)), key=values.__getitem__)]


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

@dataclass
class Measurement:
    """One arm of the search, normalized.

    ``faces_seen``/``faces_swapped`` are not diagnostics -- they are the
    comparability guard (R3).  A run that swapped fewer faces did less work,
    and its throughput is not on the same axis as the baseline's.
    """

    config: Dict[str, Any] = field(default_factory=dict)
    label: str = ""
    fps: float = 0.0
    frames: int = 0
    faces_seen: int = 0
    faces_swapped: int = 0
    peak_vram_gb: float = 0.0
    peak_ram_gb: float = 0.0
    cpu_utilization_pct: float = 0.0
    gpu_utilization_pct: Optional[float] = None
    per_core_peak_pct: Optional[float] = None
    disk_read_mb_s: Optional[float] = None
    disk_write_mb_s: Optional[float] = None
    disk_wait_pct: Optional[float] = None
    frame_time_p99_ms: Optional[float] = None
    frame_time_median_ms: Optional[float] = None
    stage_seconds: Dict[str, float] = field(default_factory=dict)
    queue_depths: Dict[str, float] = field(default_factory=dict)
    startup_seconds: float = 0.0
    stable: bool = True
    error: str = ""
    # R2: which slot of the counterbalanced sequence produced this arm.  Slot 0
    # paid the cold TensorRT engine build; it is reported, never trusted alone.
    position: int = 0
    # Window discipline: an arm shorter than ACCEPTANCE_FRAMES may rank but may
    # never promote.
    provisional: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def swap_rate(self) -> float:
        return (self.faces_swapped / self.faces_seen) if self.faces_seen else 0.0

    @property
    def work_verified(self) -> bool:
        """Did this arm report how much work it actually did?

        Without face counts the R3 guard has nothing to compare and passes
        everything -- which is absence of evidence read as a pass, the same
        failure this file rejects in the bottleneck classifier. Callers get a
        warning rather than a silent green.  ``swap_log_counts`` derives these
        from ``ProcessMgr._SWAP_LOG`` for the real harness.
        """
        return bool(self.faces_seen or self.faces_swapped)

    @classmethod
    def from_mapping(cls, value: Any, config: Optional[Mapping[str, Any]] = None,
                     label: str = "", position: int = 0) -> "Measurement":
        if isinstance(value, cls):
            return replace(value, position=position,
                           label=value.label or label)
        data = dict(value or {})

        vram = data.get("peak_vram_gb")
        if vram is None:
            vram = _number(data.get("peak_vram_mb")) / 1024.0
        ram = data.get("peak_ram_gb", data.get("peak_rss_gb"))
        stable = data.get("stable", True)
        if isinstance(stable, str):
            stable = stable.strip().lower() in ("1", "true", "yes", "on", "pass")
        frames = _integer(data.get("frames", 0))

        def optional(*names: str) -> Optional[float]:
            """None and 0.0 mean different things: never sampled vs measured idle."""
            for name in names:
                if data.get(name) is not None:
                    return _number(data[name])
            return None

        return cls(
            config=dict(config or data.get("config") or {}),
            label=label or str(data.get("label", "")),
            fps=max(0.0, _number(data.get("end_to_end_fps", data.get("fps")))),
            frames=frames,
            faces_seen=_integer(data.get("faces_seen", data.get("faces", 0))),
            faces_swapped=_integer(data.get("faces_swapped", data.get("swapped", 0))),
            peak_vram_gb=max(0.0, _number(vram)),
            peak_ram_gb=max(0.0, _number(ram)),
            cpu_utilization_pct=max(0.0, _number(
                data.get("cpu_utilization_pct", data.get("mean_cpu_pct")))),
            gpu_utilization_pct=optional("gpu_utilization_pct", "mean_gpu_util_pct"),
            per_core_peak_pct=optional("per_core_peak_pct", "max_core_pct"),
            disk_read_mb_s=optional("disk_read_mb_s"),
            disk_write_mb_s=optional("disk_write_mb_s"),
            disk_wait_pct=optional("disk_wait_pct"),
            frame_time_p99_ms=optional("frame_time_p99_ms"),
            frame_time_median_ms=optional("frame_time_median_ms"),
            stage_seconds={str(k): _number(v) for k, v in
                           (data.get("stage_seconds") or {}).items()},
            queue_depths={str(k): _number(v) for k, v in
                          (data.get("queue_depths") or {}).items()},
            startup_seconds=max(0.0, _number(data.get("startup_seconds"))),
            stable=bool(stable) and not data.get("error"),
            error=str(data.get("error", "") or ""),
            position=position,
            provisional=bool(frames and frames < ACCEPTANCE_FRAMES),
            raw=data,
        )

    def comparable_to(self, baseline: "Measurement",
                      tolerance: float = 0.02) -> Tuple[bool, str]:
        """Did this arm do the SAME WORK as the baseline? (R3)

        Returns ``(comparable, reason)``.  A faster arm that saw or swapped
        materially fewer faces is not an optimization; it is a different code
        path, and on this project that has twice looked exactly like a large
        speedup.  ``faces_seen`` is also a free path discriminator: on the
        locked 600-frame fixture 679 means one sequential pass while >750 means
        the parallel path re-processed its warm-up frames.
        """
        if not self.stable:
            return False, self.error or "arm did not complete"
        if baseline.faces_seen and self.faces_seen:
            drift = abs(self.faces_seen - baseline.faces_seen) / baseline.faces_seen
            if drift > tolerance:
                return False, ("faces_seen %d vs baseline %d (%.1f%%): different "
                               "code path or detector outcome, not a speedup"
                               % (self.faces_seen, baseline.faces_seen, drift * 100.0))
        if baseline.faces_swapped and self.faces_swapped:
            drift = (abs(self.faces_swapped - baseline.faces_swapped)
                     / baseline.faces_swapped)
            if drift > tolerance:
                return False, ("faces_swapped %d vs baseline %d (%.1f%%): less "
                               "work done, throughput is not comparable"
                               % (self.faces_swapped, baseline.faces_swapped,
                                  drift * 100.0))
        if baseline.frames and self.frames and self.frames != baseline.frames:
            return False, ("frames %d vs baseline %d: different window"
                           % (self.frames, baseline.frames))
        if not (self.work_verified and baseline.work_verified):
            # Passing, but say so. A guard that cannot see the work it guards
            # must not report the same clean answer as one that checked.
            return True, ("work counts not reported by the measure callback; "
                          "comparability NOT verified")
        return True, ""

    def as_dict(self) -> dict:
        result = asdict(self)
        result.pop("raw", None)
        result["swap_rate"] = round(self.swap_rate, 4)
        return result


@dataclass(frozen=True)
class NoiseFloor:
    """This machine's run-to-run spread, measured (R1).

    The acceptance threshold is derived from repeated runs of the UNCHANGED
    baseline, so the search adapts to the rig it is on.  A candidate must beat
    the baseline's BEST replicate by more than the observed spread: the
    baseline is what already ships, so doubt resolves in its favour.
    """

    # Floor only, for the degenerate case of a single usable replicate.  Never
    # the operating value when two or more replicates exist -- a constant here
    # is exactly what promoted noise on the 4070.
    MIN_THRESHOLD_PCT = 1.0

    replicates: Tuple[float, ...] = ()
    spread_pct: float = 0.0
    threshold_pct: float = MIN_THRESHOLD_PCT
    best: float = 0.0
    median: float = 0.0
    frames: int = 0
    measured: bool = False
    source: str = "not measured"

    @classmethod
    def from_replicates(cls, values: Sequence[float], frames: int = 0) -> "NoiseFloor":
        usable = [v for v in values if v and v > 0.0]
        if not usable:
            return cls(frames=frames,
                       source="no usable replicate; floor applied, NOT measured")
        best = max(usable)
        median = statistics.median(usable)
        if len(usable) < 2:
            return cls(replicates=tuple(usable), best=best, median=median,
                       frames=frames,
                       source="single replicate; floor applied, NOT measured")
        spread = (max(usable) - min(usable)) / _mean(usable) * 100.0
        return cls(replicates=tuple(usable), spread_pct=spread,
                   threshold_pct=max(cls.MIN_THRESHOLD_PCT, spread), best=best,
                   median=median, frames=frames, measured=True,
                   source="measured over %d replicates of the unchanged baseline"
                          % len(usable))

    def resolves(self, delta_pct: float) -> bool:
        """Is an effect of this size distinguishable from noise on this rig?

        Used to refuse a claim, not only to refuse a promotion.  A 1% effect on
        a rig with an 8% spread would need ~25 arms per side to establish; the
        honest report is that it is not measurable here.
        """
        return abs(delta_pct) > self.threshold_pct

    def accepts(self, candidate_fps: float) -> bool:
        """Does a candidate beat the baseline's best showing by more than noise?"""
        if self.best <= 0.0 or candidate_fps <= 0.0:
            return False
        return (candidate_fps - self.best) / self.best * 100.0 > self.threshold_pct

    def improvement_pct(self, candidate_fps: float) -> float:
        """Improvement against the baseline's MEDIAN.

        Deliberately not the first replicate: dividing by whichever replicate
        happened to run first made a search that promoted nothing still report
        "+3.59%" whenever that run was the slow one.
        """
        if self.median <= 0.0:
            return 0.0
        return (candidate_fps - self.median) / self.median * 100.0

    def beats(self, candidate_fps: float, reference_fps: float) -> bool:
        """Does a candidate clear the noise floor against an ARBITRARY reference?

        A later phase's reference is not the phase-0 baseline: by then the
        worker count has already moved, and comparing a memory axis against the
        original baseline silently credits that axis with the thread gain.  The
        spread is a property of the machine and carries over; the reference it
        is applied to does not.
        """
        if reference_fps <= 0.0 or candidate_fps <= 0.0:
            return False
        return ((candidate_fps - reference_fps) / reference_fps * 100.0
                > self.threshold_pct)

    def as_dict(self) -> dict:
        result = asdict(self)
        result["replicates"] = [round(v, 4) for v in self.replicates]
        result["spread_pct"] = round(self.spread_pct, 3)
        result["threshold_pct"] = round(self.threshold_pct, 3)
        return result


def swap_log_counts(swap_log: Any) -> Dict[str, int]:
    """Derive ``faces_seen``/``faces_swapped`` from ``ProcessMgr._SWAP_LOG``.

    This is the adapter a real ``measure`` callback needs: the pipeline already
    records, per frame, every face it considered and what it decided, and that
    audit is the only place the R3 comparability guard can get its numbers.

    Counting is deliberately from the pipeline's OWN decision at the composite
    rather than by re-detecting the output.  Re-detection hits the same shared
    recognition-crop problem the pipeline does, so on exactly the contact frames
    where a two-faceset bug lives it reports each person as the other regardless
    of what the swap actually did.

    Note the standing caveat: this counts INTENT over the faces the pipeline was
    handed.  It cannot see a face the detector never found, and it cannot see an
    enhancer that failed on every frame while the swap succeeded -- both have
    happened here and both read as 100%.  It is the comparability guard, not a
    correctness oracle.
    """
    seen = swapped = 0
    if not swap_log:
        return {"faces_seen": 0, "faces_swapped": 0}
    frames = swap_log.values() if isinstance(swap_log, Mapping) else swap_log
    for entries in frames:
        for entry in (entries or ()):
            seen += 1
            if isinstance(entry, Mapping):
                decision = entry.get("swapped", entry.get("result"))
            else:
                decision = getattr(entry, "swapped", None)
            if _bool(decision, False) or decision == "swapped":
                swapped += 1
    return {"faces_seen": seen, "faces_swapped": swapped}


def counterbalanced_pairs(a: Any, b: Any, rounds: int = 1) -> List[Any]:
    """Return an ABBA(ABBA...) ordering for a two-arm comparison (R2).

    Forward-only ordering is not a shortcut, it is a wrong answer: two changes
    that measure NEUTRAL when counterbalanced read +21.8% and +9.8% forward-
    only, because the first arm of a process pays the TensorRT engine build.
    On one rig the SECOND arm of a pair was faster in 5 of 6 pairs regardless
    of which treatment it carried.
    """
    order: List[Any] = []
    for _ in range(max(1, int(rounds))):
        order.extend((a, b, b, a))
    return order


def counterbalanced_sweep(levels: Sequence[Any]) -> List[Any]:
    """Forward then reversed, so every level is measured in both halves.

    For a sweep of more than two levels the ABBA form generalises to a
    palindrome: each level appears once early and once late, so position and
    treatment are no longer confounded.
    """
    levels = list(levels)
    return levels + list(reversed(levels))


def order_corrected(results: Sequence[Measurement], key: Callable[[Measurement], Any],
                    a_key: Any, b_key: Any) -> Dict[str, Any]:
    """Mean of each arm across a counterbalanced sequence, plus a position control.

    ``position_bias_pairs`` counts how often the second arm of a pair won
    regardless of treatment.  When that is most pairs, the sequence measured
    order, not the change, and the delta must not be reported as an effect.
    """
    a_values = [m.fps for m in results if key(m) == a_key and m.fps > 0]
    b_values = [m.fps for m in results if key(m) == b_key and m.fps > 0]
    a_mean, b_mean = _mean(a_values), _mean(b_values)
    second_wins = pairs = 0
    for index in range(0, len(results) - 1, 2):
        first, second = results[index], results[index + 1]
        if first.fps > 0 and second.fps > 0:
            pairs += 1
            second_wins += int(second.fps > first.fps)
    return {
        "a_mean_fps": round(a_mean, 4),
        "b_mean_fps": round(b_mean, 4),
        "delta_pct": round((b_mean - a_mean) / a_mean * 100.0, 3) if a_mean else 0.0,
        "position_bias_pairs": "%d of %d" % (second_wins, pairs),
        "position_bias_suspected": bool(pairs) and second_wins >= max(1, pairs - 1),
        "arms": len(results),
    }


# --------------------------------------------------------------------------
# the search space
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Axis:
    """One searchable dimension: its values, how it is applied, and its cost."""

    name: str
    values: Tuple[Any, ...]
    applies_to: str            # the RuntimeTuning field or setting key it drives
    env_var: str = ""          # the ROOP_* variable a child process consumes
    phase: str = "A"
    reachable: bool = True
    unreachable_reason: str = ""
    note: str = ""
    # True when this axis can change OUTPUT rather than only speed, so
    # throughput alone must not promote it. The faces_seen/faces_swapped guard
    # does NOT cover this case: when cuDNN planning fails, the SWAP still
    # succeeds and only the enhancer silently falls back to the original
    # frame, so every count and every integrity check reads clean while the
    # picture is wrong -- and the broken arm is the FASTEST, because not
    # enhancing is cheap.
    requires_output_check: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


class SearchSpace:
    """The parameter space, with models held CONSTANT.

    The models are not searched.  Swapper, enhancer, mask engine and detector
    are read from the user's live configuration and passed through unchanged:
    a benchmark that quietly substitutes a cheaper model is measuring a
    different application.  (This has invalidated whole sessions here twice --
    four harnesses once ran their "CodeFormer" arm with no enhancer at all,
    because ``get_processing_plugins`` matches enhancer names by exact string
    and a miss silently adds nothing.)
    """

    # Coarse, geometric, and bounded.  Phase A locates the inflection between
    # these; it does not test every integer, because the thread curve's tail is
    # nearly flat and each extra arm costs a full render.
    COARSE_THREADS = (2, 4, 8, 12, 16)

    # ORT threading.  intra = threads inside one op, inter = ops in parallel.
    # ResourceManager bounds them at (1,4) and (1,2) respectively: this
    # pipeline runs many concurrent sessions, so a large per-session pool
    # multiplies against the worker count instead of adding to it.
    ORT_INTRA = (1, 2, 4)
    ORT_INTER = (1, 2)

    # Arena allocator.  kSameAsRequested stops the arena growing geometrically,
    # which is what turns a 12 GB card into a paging one when several pooled
    # contexts each hold their own arena.
    ARENA_STRATEGIES = ("kNextPowerOfTwo", "kSameAsRequested")

    # Ordered cheapest-to-verify first. DEFAULT is the safe fallback the
    # per-model probe uses, so it leads: if the sweep never gets further, the
    # device is left on the mode that is known to work everywhere.
    CUDNN_ALGOS = ("DEFAULT", "HEURISTIC", "EXHAUSTIVE")

    # The three formats the application actually offers (ui/tabs/settings_tab.py
    # ``image_formats``).  Anything else would not be selectable by a user.
    FRAME_FORMATS = ("png", "jpg", "webp")

    def __init__(self, hardware: HardwareProfile,
                 workload: Optional[WorkloadProfile] = None,
                 settings: Any = None):
        self.hardware = hardware
        self.workload = workload or WorkloadProfile()
        self.settings = settings

    # -- threads ---------------------------------------------------------
    def thread_levels(self) -> Tuple[int, ...]:
        """Coarse thread sweep, capped by logical cores and the safe bounds."""
        logical = max(1, _integer(self.hardware.cpu_logical_cores, 1))
        levels = sorted({
            ResourceManager.clamp("worker_count", value, self.hardware)
            for value in self.COARSE_THREADS if value <= logical
        })
        if not levels:
            levels = [ResourceManager.clamp("worker_count", 1, self.hardware)]
        # Always include the full logical width as the top of the sweep, so the
        # curve's flat tail is observed rather than assumed.
        top = ResourceManager.clamp("worker_count", logical, self.hardware)
        if top not in levels:
            levels.append(top)
        return tuple(sorted(levels))

    # -- ONNX Runtime execution provider ---------------------------------
    def ort_axes(self) -> List[Axis]:
        return [
            Axis("ort_intra_threads",
                 tuple(v for v in self.ORT_INTRA
                       if v <= max(1, self.hardware.cpu_physical_cores)),
                 applies_to="ort_intra_threads",
                 env_var="ROOP_ORT_INTRA_THREADS", phase="B",
                 note="threads inside one op; multiplies against worker_count"),
            Axis("ort_inter_threads", self.ORT_INTER,
                 applies_to="ort_inter_threads",
                 env_var="ROOP_ORT_INTER_THREADS", phase="B",
                 note="ops in parallel within one session"),
            Axis("opencv_threads",
                 tuple(v for v in (1, 2, 4)
                       if v <= max(1, self.hardware.cpu_physical_cores)),
                 applies_to="opencv_threads",
                 env_var="ROOP_CV_THREADS", phase="B",
                 note="host pre/post-processing; competes with the workers"),
            Axis("arena_extend_strategy", self.ARENA_STRATEGIES,
                 applies_to="arena_extend_strategy",
                 env_var="ROOP_CUDA_ARENA_STRATEGY", phase="B",
                 note="kSameAsRequested caps arena growth under pooled contexts"),
            self.cudnn_axis(),
        ]

    # The most dangerous axis in this package, and the reason it is not a
    # plain throughput sweep.
    #
    # `cudnn_conv_algo_search` selects how ORT's CUDA EP plans convolutions.
    # HEURISTIC (the shipped default) and EXHAUSTIVE both go through cuDNN's
    # frontend graph API; DEFAULT uses the legacy path. Measured on an RTX 3060
    # Laptop, HEURISTIC is 55-241% FASTER than DEFAULT across this app's
    # models -- and on that same device it makes the CodeFormer family
    # (Codeformer, Codeformer fp16, UltraMax, Restoreformer++) fail every
    # convolution with CUDNN_FE HEURISTIC_QUERY_FAILED.
    #
    # The failure does not raise anywhere a benchmark would see it: ProcessMgr
    # catches the per-frame GPU error and writes the ORIGINAL frame, and the
    # swap audit still reports "swapped 100.0%". Four enhancers produced 60/60
    # unswapped frames while every throughput and integrity check passed.
    #
    # So a sweep scored on FPS would pick HEURISTIC on exactly the machine
    # where it silently disables the enhancer -- and it would be the FASTEST
    # arm, because not enhancing is cheap. This axis is therefore:
    #
    #   * gated on `roop.cudnn_algo`'s per-device probe rather than measured
    #     here. That probe already runs on the live device, tests only the
    #     suspect models, and lowers just those -- a strictly better answer
    #     than any single global value this search could promote;
    #   * offered for measurement ONLY when the probe reports the device
    #     healthy, so the sweep can never re-enable a mode the probe rejected.
    def cudnn_axis(self) -> Axis:
        healthy, reason = self.cudnn_probe_state()
        return Axis(
            "cudnn_conv_algo_search",
            self.CUDNN_ALGOS if healthy else (),
            applies_to="cudnn_conv_algo_search",
            env_var="ROOP_CUDNN_CONV_ALGO", phase="B",
            reachable=bool(healthy and self.hardware.cuda_available),
            unreachable_reason=reason if not healthy else (
                "" if self.hardware.cuda_available else
                "no CUDA device: cuDNN convolution planning does not apply"),
            requires_output_check=True,
            note=("correctness-gated: scored on output equivalence first, "
                  "throughput second"))

    def cudnn_probe_state(self) -> Tuple[bool, str]:
        """Ask the live-device probe whether the frontend path is usable.

        Returns ``(healthy, reason)``.  A device where the probe has lowered
        any model is NOT offered this axis: promoting a global value there
        would either re-break the lowered model or force every other model
        onto the slow path.
        """
        try:
            from roop import cudnn_algo
        except Exception as exc:
            return False, ("cuDNN policy module unavailable (%s); the axis is "
                           "not swept blind" % type(exc).__name__)
        lowered = ()
        for name in ("lowered_models", "downgraded_models", "unsafe_models"):
            getter = getattr(cudnn_algo, name, None)
            if callable(getter):
                try:
                    lowered = tuple(getter() or ())
                    break
                except Exception:
                    continue
        if lowered:
            return False, (
                "this device's cuDNN probe already lowered %s to DEFAULT; a "
                "global value here would either re-break it or slow every "
                "other model down" % ", ".join(sorted(map(str, lowered))))
        return True, ""

    # -- memory ----------------------------------------------------------
    def memory_axes(self) -> List[Axis]:
        """Pool/batch/limit axes, admitted only where the VRAM tier allows.

        A sub-7GB card is given single-context axes only.  That is not caution
        for its own sake: pool 8 on a 12GB card was measured at 2-2.5 fps
        against pool 2's 45.3 on the same clip, and it presents as a hang.
        """
        small = 0 < self.hardware.vram_total_gb < 7.0
        axes = [
            Axis("trt_context_count",
                 (1,) if small else (1, 2, 3),
                 applies_to="trt_context_count",
                 env_var="ROOP_TRT_POOL", phase="B",
                 reachable=bool(self.hardware.tensorrt_available),
                 unreachable_reason=("" if self.hardware.tensorrt_available
                                     else "TensorRT is not admitted on this device"),
                 note="each context allocates on FIRST INFERENCE, not at build"),
            Axis("detmask_pool_size",
                 (0,) if small else (0, 2),
                 applies_to="detmask_pool_size",
                 env_var="ROOP_DETMASK_POOL", phase="B",
                 reachable=bool(self.hardware.cuda_available),
                 unreachable_reason=("" if self.hardware.cuda_available else
                                     "no CUDA device: the detect/mask stages "
                                     "have no GPU pool to size")),
            Axis("gpu_mem_limit_gb",
                 self._gpu_mem_limits(),
                 applies_to="gpu_mem_limit_gb",
                 env_var="ROOP_CUDA_MEM_LIMIT", phase="B",
                 reachable=bool(self.hardware.cuda_available),
                 unreachable_reason=("" if self.hardware.cuda_available else
                                     "no CUDA device: there is no GPU "
                                     "allocator to cap"),
                 note="ORT CUDA/TRT allocator ceiling; caps arena growth"),
        ]
        faces = _number(self.workload.faces_per_frame, 1.0)
        axes.append(Axis(
            "batch_size", (1, 2) if faces >= 2.0 else (1,),
            applies_to="batch_size", env_var="ROOP_RUNTIME_BATCH_SIZE", phase="B",
            reachable=faces >= 2.0,
            unreachable_reason=("" if faces >= 2.0 else
                                "workload averages %.2f faces/frame; a batch "
                                "dimension above 1 has nothing to fill it" % faces),
            note="cross-frame batching only pays with multiple faces per frame"))
        return axes

    def _gpu_mem_limits(self) -> Tuple[Any, ...]:
        total = _number(self.hardware.vram_total_gb)
        if total <= 0:
            return (None,)
        ceiling = max(1.0, total * VRAM_ADMISSION_CEILING)
        balanced = max(1.0, total * VRAM_BALANCED_CEILING)
        return (None, round(balanced, 2), round(ceiling, 2))

    # -- disk ------------------------------------------------------------
    def frame_format_axis(self, reachable: bool, reason: str = "") -> Axis:
        """Temp frame format -- gated on whether the disk path can even run (R5).

        ``output_image_format`` is the temp frame format: it is what
        ``util_ffmpeg.extract_frames`` writes and what
        ``utilities.get_temp_frame_paths`` globs back.  But the shipped video
        path is a zero-disk rawvideo pipe at BOTH ends, and the frame-extraction
        path only runs when one of three triggers fires (``keep_frames``, the
        legacy non-``use_new_method`` route, or per-frame masks).  Sweeping this
        on a machine where none of them fires would be measuring a code path
        that a real render never takes.
        """
        return Axis("output_image_format", self.FRAME_FORMATS,
                    applies_to="output_image_format", phase="C",
                    reachable=reachable, unreachable_reason=reason,
                    note="temp frame format; only reached by the disk fallback path")

    def as_dict(self) -> dict:
        return {
            "models_held_constant": True,
            "thread_levels": list(self.thread_levels()),
            "ort_axes": [axis.as_dict() for axis in self.ort_axes()],
            "memory_axes": [axis.as_dict() for axis in self.memory_axes()],
            "frame_formats": list(self.FRAME_FORMATS),
        }


# --------------------------------------------------------------------------
# Phase C support: is the disk path reachable, and how fast is the temp volume
# --------------------------------------------------------------------------

def temp_path_triggers(settings: Any = None) -> Dict[str, bool]:
    """The three conditions under which a video render writes temp frames.

    Read from ``roop.core``'s own branch: the disk route is taken when
    ``keep_frames`` is set, when the legacy (non ``use_new_method``) route is
    selected, or when per-frame masks force a second pass over saved frames.
    """
    def get(name: str, default: Any = False) -> Any:
        if settings is None:
            return default
        if isinstance(settings, Mapping):
            return settings.get(name, default)
        return getattr(settings, name, default)

    return {
        "keep_frames": _bool(get("keep_frames", False)),
        "legacy_frame_route": not _bool(get("use_new_method", True), True),
        "per_frame_masks": _bool(get("has_per_frame_masks", False)),
    }


def probe_temp_volume(path: Optional[str] = None, payload_mb: int = 64,
                      probe=None) -> dict:
    """Measure write/read throughput of the volume temp frames land on.

    THE MEASUREMENT IS NOT DONE HERE.  ``hardware_probe.probe_disk_io`` already
    performs it -- same sequential write, read-back and guaranteed cleanup --
    and this is the one place in the package that needs its answer reshaped for
    the search.  Re-implementing it was a duplicate that only existed because
    this module was written against a different package layout.

    What this adds is the two things the search needs and the probe does not
    provide: a ``class`` for the storage tier, and key names the rest of this
    module already speaks.

    The temp directory is NOT the system temp directory:
    ``get_temp_directory_path`` places it beside the *target file*, so this is a
    property of whichever drive the user's footage sits on.
    """
    if probe is None:
        from roop.benchmark.hardware_probe import probe_disk_io as probe
    try:
        raw = probe(path, size_mb=max(1, int(payload_mb)))
    except Exception as exc:                      # a probe must never abort a run
        return {"path": path or "", "write_mb_s": None, "read_mb_s": None,
                "payload_mb": payload_mb, "class": "unknown",
                "note": "probe failed: %s: %s" % (type(exc).__name__, exc)}
    raw = dict(raw or {})
    ok = bool(raw.get("success"))
    write = _number(raw.get("write_mb_per_sec")) if ok else None
    read = _number(raw.get("read_mb_per_sec")) if ok else None
    return {
        "path": raw.get("temp_directory", "") or (path or ""),
        "write_mb_s": round(write, 1) if write else None,
        "read_mb_s": round(read, 1) if read else None,
        "payload_mb": round(_number(raw.get("bytes_tested")) / (1024.0 ** 2), 1)
                      or payload_mb,
        "class": classify_volume(write),
        # Read-back is warm in the page cache by construction, so it is an
        # upper bound on the medium rather than its cold-read speed.
        "read_is_cached": True,
        "note": "" if ok else str(raw.get("error") or "probe reported failure"),
    }


def classify_volume(write_mb_s: Optional[float]) -> str:
    """Name the storage tier from measured sequential write throughput."""
    if not write_mb_s:
        return "unknown"
    if write_mb_s >= 1200.0:
        return "nvme"
    if write_mb_s >= 300.0:
        return "sata-ssd"
    if write_mb_s >= 80.0:
        return "fast-hdd-or-network"
    return "slow"


def frame_format_cost(width: int, height: int, formats: Sequence[str] = (),
                      samples: int = 8,
                      clock: Callable[[], float] = time.perf_counter) -> dict:
    """Encode/decode cost and file size per temp frame format, at the real resolution.

    Resolution is not a detail here: JPEG's advantage over PNG grows with pixel
    count, so measuring at a fixed 1080p and applying the answer to a 4K render
    ranks the formats on a workload the user does not have.

    The frame content is a moving gradient with a flat block, never noise:
    noise is incompressible and makes every codec look equally slow, which is
    the one comparison this must not get wrong.
    """
    formats = tuple(formats or SearchSpace.FRAME_FORMATS)
    out: Dict[str, Any] = {"width": width, "height": height, "rows": [],
                           "notes": []}
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        out["notes"].append("cv2/numpy unavailable: %s" % exc)
        return out

    width = max(16, int(width))
    height = max(16, int(height))
    xs = np.linspace(0, 255, width, dtype=np.float32)
    base = np.repeat(xs[None, :], height, axis=0)
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = base
    frame[:, :, 1] = np.roll(base, width // 7, axis=1)
    frame[:, :, 2] = 255 - base
    frame[height // 4:height // 2, width // 8:width // 3] = 240

    params = {
        # Quality is pinned per format so the comparison is cost-at-equal-intent.
        # PNG level 1 is what a frame-dump path wants: level 9 triples the write
        # cost for a file that is deleted minutes later.
        "png": [cv2.IMWRITE_PNG_COMPRESSION, 1],
        "jpg": [cv2.IMWRITE_JPEG_QUALITY, 95],
        "webp": [cv2.IMWRITE_WEBP_QUALITY, 95],
    }

    for name in formats:
        extension = "." + name
        encode_params = params.get(name, [])
        try:
            ok, buffer = cv2.imencode(extension, frame, encode_params)
        except cv2.error as exc:
            out["notes"].append("%s: not supported by this OpenCV build (%s)"
                                % (name, exc))
            continue
        if not ok:
            out["notes"].append("%s: encoder refused the frame" % name)
            continue

        start = clock()
        for _ in range(samples):
            cv2.imencode(extension, frame, encode_params)
        encode_ms = (clock() - start) * 1000.0 / samples

        start = clock()
        for _ in range(samples):
            cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        decode_ms = (clock() - start) * 1000.0 / samples

        size_mb = len(buffer) / (1024.0 * 1024.0)
        # Lossy formats are recorded as such. A temp frame is re-read and
        # re-encoded into the final video, so a lossy round trip is a QUALITY
        # decision the throughput number must not silently make.
        out["rows"].append({
            "format": name,
            "encode_ms": round(encode_ms, 3),
            "decode_ms": round(decode_ms, 3),
            "size_mb": round(size_mb, 4),
            "lossless": name == "png",
        })
    return out


def frame_format_recommendation(cost: Mapping[str, Any],
                                write_mb_s: Optional[float],
                                allow_lossy: bool = False) -> dict:
    """Weigh disk write latency against compression overhead.

    Total per-frame cost is ``encode_ms + size_mb / write_mb_s``: a format that
    compresses harder pays CPU to save disk time, and which side wins is a
    property of the volume, not of the format.  On an NVMe the disk term nearly
    vanishes and the cheapest encoder wins; on a slow volume the smallest file
    does.

    ``allow_lossy`` DEFAULTS TO FALSE, and that is the load-bearing part.  It is
    tempting to reason that a temp frame is scratch and may as well be a cheap
    JPEG -- but these frames are not scratch.  ``util_ffmpeg.create_video``
    builds the DELIVERED video straight out of the temp directory
    (``-i <temp>/%06d.<output_image_format>``), so ``jpg`` inserts a lossy
    generation *inside the output chain* and every swapped face is re-quantised
    before the encoder ever sees it.  A throughput search must not make that
    trade silently: the caller has to ask for it.

    ``webp`` at quality 95 is lossy too (OpenCV only writes lossless WebP above
    100), so PNG is the sole lossless option among the three the UI offers.
    """
    rows = list(cost.get("rows") or [])
    if not rows:
        return {"choice": None, "reason": "no format could be measured"}
    considered = [row for row in rows if allow_lossy or row.get("lossless")]
    if not considered:
        return {"choice": "png",
                "reason": ("the temp frames are the encoder's input, so the "
                           "format must stay lossless; only PNG qualifies")}

    scored = []
    for row in considered:
        disk_ms = ((row["size_mb"] / write_mb_s) * 1000.0) if write_mb_s else 0.0
        scored.append({**row, "disk_ms": round(disk_ms, 3),
                       "total_ms": round(row["encode_ms"] + disk_ms, 3)})
    scored.sort(key=lambda row: row["total_ms"])
    best = scored[0]
    if len(scored) == 1:
        # Do not dress a forced choice as a won comparison.
        reason = ("%s is the only admissible format (lossy formats are excluded "
                  "because the temp frames are the encoder's input)"
                  % best["format"])
    else:
        runner_up = scored[1]
        margin = ((runner_up["total_ms"] - best["total_ms"])
                  / max(1e-9, runner_up["total_ms"]) * 100.0)
        reason = ("lowest encode+write cost per frame at %dx%d on a %s volume; "
                  "%.0f%% cheaper than %s"
                  % (cost.get("width", 0), cost.get("height", 0),
                     classify_volume(write_mb_s), margin, runner_up["format"]))
    return {
        "choice": best["format"],
        "total_ms": best["total_ms"],
        "reason": reason,
        "disk_term_included": bool(write_mb_s),
        "allow_lossy": allow_lossy,
        "candidates_considered": len(scored),
        "ranked": scored,
    }


# --------------------------------------------------------------------------
# bottleneck analysis
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BottleneckVerdict:
    """A classification, the evidence behind it, and what it licenses."""

    kind: str
    confidence: str = "low"
    evidence: Tuple[str, ...] = ()
    missing_signals: Tuple[str, ...] = ()
    dominant_stage: str = ""
    recommendation: str = ""

    def badge_headline(self) -> str:
        """A one-line label for the results screen.

        States the limiter AND whether that is a healthy place to be, because
        "GPU Bound" on its own reads as a fault when it is usually the goal.
        """
        return {
            "GPU compute bound": "GPU Bound - Optimum VRAM Utilization",
            "GPU VRAM bound": "VRAM Limited - Reduce Memory Pressure",
            "CPU bound": "CPU Bound - GPU Underfed",
            "Disk I/O bound": "Storage Bound - Slow Temp Volume",
            "synchronization bound": "Sync Bound - Concurrency Limited",
            "unknown": "Not Determined",
        }.get(self.kind, self.kind)

    def as_dict(self) -> dict:
        result = asdict(self)
        result["headline"] = self.badge_headline()
        return result


class BottleneckAnalyzer:
    """Classify the primary limiter from measured telemetry.

    Thresholds are the specified ones.  Two design points matter more than the
    numbers:

    * **Absence of evidence is never rendered as a diagnosis.**  The previous
      classifier fell through to ``"I/O-bound"`` whenever queue depths were
      0.0 -- which is also exactly what an UNINSTRUMENTED run looks like.  Both
      validation GPUs reported "I/O-bound" on runs whose decode cost 3.3 ms of
      a 244.8 ms frame.  When the signals are missing this returns ``unknown``
      and names what is missing.

    * **Utilization is time-coverage, not work.**  A 6-context arm reached
      94.5% GPU utilization at 1.51 fps while paging.  High GPU utilization is
      therefore corroborating evidence for GPU-bound only when throughput is
      not simultaneously collapsing, which is why VRAM pressure is tested first.
    """

    GPU_COMPUTE_PCT = 90.0
    GPU_COMPUTE_CPU_CEILING = 65.0
    VRAM_PRESSURE_PCT = 85.0
    CPU_CORE_SATURATED_PCT = 99.0
    CPU_BOUND_GPU_CEILING = 70.0
    DISK_WAIT_PCT = 20.0

    # The persisted vocabulary. `storage._BOTTLE_NECKS` accepts exactly these
    # four, so a record carrying anything else is refused outright.
    #
    # The internal verdicts are deliberately RICHER than the four -- this
    # analyzer also returns "unknown" and "synchronization bound", and those
    # exist for a reason: the previous classifier fell through to a confident
    # "I/O-bound" whenever telemetry was missing, and both validation GPUs
    # reported it on runs whose decode cost 3.3 ms of a 244.8 ms frame.
    # ``storage_name`` therefore refuses to invent one of the four for an
    # undetermined verdict rather than rounding "I do not know" up to a
    # diagnosis that will be shown to a user as fact.
    STORAGE_NAMES = {
        "GPU compute bound": "GPU Compute Bound",
        "GPU VRAM bound": "GPU VRAM Bound",
        "CPU bound": "CPU Bound",
        "Disk I/O bound": "Disk I/O Bound",
    }
    # A frame-time p99 this many times the median is hitching, not jitter.
    HITCH_RATIO = 3.0

    def classify(self, measurement: Measurement,
                 hardware: Optional[HardwareProfile] = None,
                 disk: Optional[Mapping[str, Any]] = None) -> BottleneckVerdict:
        evidence: List[str] = []
        missing: List[str] = []

        gpu = measurement.gpu_utilization_pct
        cpu = measurement.cpu_utilization_pct
        core_peak = measurement.per_core_peak_pct
        if gpu is None:
            missing.append("gpu_utilization_pct (nvml/nvidia-smi not sampled)")
        if core_peak is None:
            missing.append("per_core_peak_pct (per-core sampling not reported)")
        if measurement.disk_wait_pct is None and not disk:
            missing.append("disk_wait_pct (no I/O telemetry)")

        vram_pct = 0.0
        if hardware and hardware.vram_total_gb and measurement.peak_vram_gb:
            vram_pct = measurement.peak_vram_gb / hardware.vram_total_gb * 100.0
        hitching = self._hitching(measurement)

        # 1. VRAM first. It is tested ahead of GPU utilization because a paging
        #    card reports HIGH utilization while throughput collapses, so the
        #    utilization test alone would call a thrashing run "GPU compute
        #    bound" and recommend exactly the wrong action.
        if vram_pct >= self.VRAM_PRESSURE_PCT:
            evidence.append("peak VRAM %.1f%% of %.1f GB (>= %.0f%%)"
                            % (vram_pct, hardware.vram_total_gb if hardware else 0.0,
                               self.VRAM_PRESSURE_PCT))
            if hitching:
                evidence.append(hitching)
            return BottleneckVerdict(
                "GPU VRAM bound",
                confidence="high" if hitching else "medium",
                evidence=tuple(evidence), missing_signals=tuple(missing),
                dominant_stage=self.dominant_stage(measurement),
                recommendation=(
                    "Reduce context/pool sizes or the enhancer tier before "
                    "anything else. This fails as a THRASH, not an OOM: the "
                    "card sits near 100% utilisation at a fraction of its "
                    "power limit while the driver pages over PCIe."))

        # 2. Disk.
        disk_verdict = self._disk_bound(measurement, disk, evidence)
        if disk_verdict is not None:
            return replace(disk_verdict, missing_signals=tuple(missing),
                           dominant_stage=self.dominant_stage(measurement))

        # 3. GPU compute.
        if gpu is not None and gpu >= self.GPU_COMPUTE_PCT and cpu < self.GPU_COMPUTE_CPU_CEILING:
            evidence.append("GPU %.1f%% (>= %.0f%%), process CPU %.1f%% (< %.0f%%)"
                            % (gpu, self.GPU_COMPUTE_PCT, cpu,
                               self.GPU_COMPUTE_CPU_CEILING))
            evidence.append("VRAM %.1f%%, so utilization is work and not paging"
                            % vram_pct)
            return BottleneckVerdict(
                "GPU compute bound", confidence="high",
                evidence=tuple(evidence), missing_signals=tuple(missing),
                dominant_stage=self.dominant_stage(measurement),
                recommendation=(
                    "Only REMOVING GPU work moves this: a cheaper enhancer "
                    "tier, a smaller detector canvas, or fewer per-face "
                    "passes. More threads, deeper queues and larger pools "
                    "redistribute thread time and have measured neutral."))

        # 4. CPU.
        core_saturated = core_peak is not None and core_peak >= self.CPU_CORE_SATURATED_PCT
        if (core_saturated or cpu >= self.CPU_CORE_SATURATED_PCT) and (
                gpu is None or gpu < self.CPU_BOUND_GPU_CEILING):
            if core_saturated:
                evidence.append("a core is pinned at %.1f%%" % core_peak)
            else:
                evidence.append("process CPU %.1f%%" % cpu)
            evidence.append("GPU %s (< %.0f%%)"
                            % ("not sampled" if gpu is None else "%.1f%%" % gpu,
                               self.CPU_BOUND_GPU_CEILING))
            return BottleneckVerdict(
                "CPU bound",
                confidence="medium" if gpu is None else "high",
                evidence=tuple(evidence), missing_signals=tuple(missing),
                dominant_stage=self.dominant_stage(measurement),
                recommendation=(
                    "Host-side per-face work is starving the GPU. Raise worker "
                    "threads toward the Phase A knee, and reduce host "
                    "pre/post-processing on the dominant stage."))

        # 5. Nothing is saturated. Say so, and say what would decide it.
        stage = self.dominant_stage(measurement)
        if not self._has_telemetry(measurement):
            return BottleneckVerdict(
                "unknown", confidence="none",
                evidence=("no utilization, queue or per-core signal was "
                          "reported for this run",),
                missing_signals=tuple(missing), dominant_stage=stage,
                recommendation=(
                    "Enable ROOP_RUNTIME_MONITOR and re-measure. Do not infer "
                    "a limiter from stage share alone: it is thread time "
                    "summed across workers, and a 42.4%% detect share "
                    "predicted 10%% that measured as 1%%."))

        evidence.append("GPU %s, process CPU %.1f%%, VRAM %.1f%% -- nothing saturated"
                        % ("not sampled" if gpu is None else "%.1f%%" % gpu,
                           cpu, vram_pct))
        if stage:
            evidence.append("costliest stage: %s" % stage)
        return BottleneckVerdict(
            "synchronization bound", confidence="medium",
            evidence=tuple(evidence), missing_signals=tuple(missing),
            dominant_stage=stage,
            recommendation=(
                "Concurrency, not stage cost, is the limiter: workers are "
                "waiting on each other or on a serializing lock. Threads, "
                "contexts, queues and affinity have each been measured as dead "
                "ends here, so the productive direction is reducing per-face "
                "work on the %s stage." % (stage or "dominant")))

    def _disk_bound(self, measurement: Measurement,
                    disk: Optional[Mapping[str, Any]],
                    evidence: List[str]) -> Optional[BottleneckVerdict]:
        wait = measurement.disk_wait_pct
        if wait is not None and wait >= self.DISK_WAIT_PCT:
            evidence.append("frame read/write wait %.1f%% of frame time (>= %.0f%%)"
                            % (wait, self.DISK_WAIT_PCT))
            saturation = self._disk_saturation(measurement, disk)
            if saturation:
                evidence.append(saturation)
            return BottleneckVerdict(
                "Disk I/O bound", confidence="high", evidence=tuple(evidence),
                recommendation=(
                    "Move the target file to a faster volume, or keep the "
                    "render on the zero-disk rawvideo path (do not enable "
                    "keep_frames). If frames must be written, prefer the "
                    "format Phase C selected for this volume."))
        return None

    def _disk_saturation(self, measurement: Measurement,
                         disk: Optional[Mapping[str, Any]]) -> str:
        if not disk:
            return ""
        capacity = _number(disk.get("write_mb_s"))
        observed = _number(measurement.disk_write_mb_s)
        if capacity and observed:
            return ("write %.0f MB/s against a probed %.0f MB/s ceiling (%.0f%%)"
                    % (observed, capacity, observed / capacity * 100.0))
        return "volume probed at %s MB/s (%s)" % (disk.get("write_mb_s"),
                                                  disk.get("class", "unknown"))

    def _hitching(self, measurement: Measurement) -> str:
        p99 = measurement.frame_time_p99_ms
        median = measurement.frame_time_median_ms
        if p99 and median and p99 >= median * self.HITCH_RATIO:
            return ("frame time p99 %.1f ms against median %.1f ms (%.1fx): "
                    "hitching consistent with paging" % (p99, median, p99 / median))
        return ""

    @staticmethod
    def _has_telemetry(measurement: Measurement) -> bool:
        return (measurement.gpu_utilization_pct is not None
                or measurement.per_core_peak_pct is not None
                or bool(measurement.queue_depths)
                or measurement.cpu_utilization_pct > 0.0)

    @classmethod
    def storage_name(cls, verdict: "BottleneckVerdict") -> Optional[str]:
        """The persisted name, or None when the verdict is undetermined.

        Returning None is the point: a run whose limiter could not be
        established must not be filed under one of the four as though it had
        been. The caller decides what to do about an unstorable verdict; it
        does not get silently rounded to the nearest label.
        """
        return cls.STORAGE_NAMES.get(verdict.kind)

    def advise(self, verdict: "BottleneckVerdict", measurement: Measurement,
               hardware: Optional[HardwareProfile] = None,
               disk: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Turn a verdict into advice, with the size of the win BOUNDED.

        A results screen that says "+35% FPS" is making a promise, and this
        project's rule is that an unmeasured number is labelled as one. So the
        estimate here is not a guess: it is an UPPER BOUND derived from the
        share of frame time the bottleneck actually consumed. Removing a
        bottleneck cannot return more than the time it was taking, and it
        usually returns less because something else becomes the limit -- three
        stage-level wins here measured well in isolation and NEUTRAL end to
        end, so the ceiling is reported as a ceiling.
        """
        advice: Dict[str, Any] = {
            "headline": verdict.badge_headline(),
            "action": verdict.recommendation,
            "estimated_gain_pct": None,
            "estimate_basis": "not estimated",
            "confidence": verdict.confidence,
        }
        if verdict.kind == "Disk I/O bound":
            wait = _number(measurement.disk_wait_pct)
            current = classify_volume((disk or {}).get("write_mb_s"))
            # The ceiling: if frames stopped waiting on disk entirely, the run
            # gets back exactly the share of frame time it spent waiting.
            ceiling = wait / max(1e-9, 100.0 - wait) * 100.0 if wait < 100 else None
            advice["estimated_gain_pct"] = round(ceiling, 1) if ceiling else None
            advice["estimate_basis"] = (
                "upper bound: frame time spent waiting on disk was %.1f%%, so "
                "eliminating that wait returns at most %.1f%%. A faster volume "
                "reduces the wait, it does not remove it."
                % (wait, ceiling or 0.0))
            advice["action"] = (
                "Move the target file and its temp directory off this %s volume "
                "onto an NVMe drive, or keep the render on the zero-disk path "
                "(leave keep_frames off)." % current)
        elif verdict.kind == "GPU VRAM bound":
            advice["action"] = (
                "Reduce memory pressure before anything else: a smaller "
                "context/pool size, or a lighter enhancer tier. Above this "
                "line the driver pages over PCIe, which presents as a "
                "slowdown rather than an error.")
            advice["estimate_basis"] = (
                "not estimated: a paging run's throughput is not a stable "
                "baseline to project from")
        elif verdict.kind == "CPU bound":
            has_gpu = bool(hardware and (hardware.cuda_available
                                         or hardware.vram_total_gb))
            if not has_gpu:
                # "The GPU is being starved" is nonsense on a machine that has
                # none, and advice a user can see must not describe hardware
                # they do not own.
                advice["action"] = (
                    "Every model is running on the CPU, so the CPU is the "
                    "pipeline. ONNX intra/inter-op threading is the lever "
                    "Stage B tunes here; beyond that, the only large win is a "
                    "CUDA-capable GPU.")
                advice["estimate_basis"] = (
                    "not estimated: there is no idle accelerator to reclaim "
                    "time from")
                return advice
            advice["action"] = (
                "The GPU is being starved by host-side per-face work. Raise "
                "worker threads toward the Stage A knee; if it is already "
                "there, the remaining lever is reducing per-face host work on "
                "the %s stage." % (verdict.dominant_stage or "dominant"))
            gpu = measurement.gpu_utilization_pct
            if gpu is not None and gpu > 0:
                # If the GPU could be fed continuously it would do at most
                # 100/gpu times the work it is doing now.
                advice["estimated_gain_pct"] = round(
                    (100.0 / max(1.0, gpu) - 1.0) * 100.0, 1)
                advice["estimate_basis"] = (
                    "upper bound: the GPU idled at %.0f%% utilization, so "
                    "feeding it perfectly could not exceed %.0fx the current "
                    "rate" % (gpu, 100.0 / max(1.0, gpu)))
        elif verdict.kind == "GPU compute bound":
            advice["action"] = (
                "This is the healthy state. Only REMOVING GPU work moves it "
                "further: a cheaper enhancer tier, a smaller detector canvas, "
                "or fewer per-face passes. More threads, deeper queues and "
                "larger pools redistribute thread time and have measured "
                "neutral here.")
            advice["estimate_basis"] = (
                "not estimated: no idle resource to reclaim")
        return advice

    @staticmethod
    def dominant_stage(measurement: Measurement) -> str:
        """The costliest non-aggregate stage.

        Named for diagnosis only.  ``frame_total`` is the sum of the others and
        is excluded, or it would always win; and the share is NOT converted
        into a projected speedup (R4).
        """
        stages = {name: value for name, value in measurement.stage_seconds.items()
                  if name != "frame_total"}
        if not stages:
            return ""
        return max(stages.items(), key=lambda item: item[1])[0]


# --------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------

@dataclass
class Preset:
    """One recommended configuration, with the constraint that shaped it."""

    name: str
    tuning: Dict[str, Any] = field(default_factory=dict)
    measured_fps: Optional[float] = None
    projected_vram_pct: Optional[float] = None
    constraints: Tuple[str, ...] = ()
    rationale: str = ""
    provisional: bool = False
    # True when NO measured arm could satisfy this preset's own constraint, so
    # the tuning below is the least-bad option rather than a compliant one. A
    # preset that states "peak VRAM <= 80%" while sitting at 92% is lying; on a
    # 4GB card that is the honest answer and it has to be said out loud.
    constraint_violated: bool = False
    violation: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class PresetBuilder:
    """Turn the search results into Max Throughput / Balanced / Stable presets.

    Balanced is the recommended default and is the only one permitted to differ
    from the shipped configuration on evidence weaker than the measured noise
    floor -- which it never does, because a candidate that does not clear the
    floor is not promoted at all.
    """

    # Low-power caps the worker pool at half the knee. Threads were measured
    # flat across a 5x range on a 12GB desktop (12.41 / 12.35 / 12.32 fps at
    # 4 / 10 / 20 workers, trending slightly DOWN), so giving them up costs
    # very little and buys a quiet machine.
    LOW_POWER_THREAD_FRACTION = 0.5

    # Balanced accepts the smallest footprint still inside this share of the
    # best admissible arm. The band's floor (0.90) is what decides it: a
    # tighter bar just reproduces Max Throughput, and the whole point of the
    # preset is to trade a few percent for headroom and quiet.
    BALANCED_GAIN = 1.0 - BALANCED_FPS_BAND[0]

    def build(self, baseline: RuntimeTuning, hardware: HardwareProfile,
              thread_curve: Sequence[Tuple[int, float]],
              vram_by_threads: Mapping[int, float],
              noise: NoiseFloor,
              frame_format: Optional[str] = None,
              ort: Optional[Mapping[str, Any]] = None) -> Dict[str, Preset]:
        levels = [level for level, _ in thread_curve]
        values = [fps for _, fps in thread_curve]
        total_vram = _number(hardware.vram_total_gb)

        def vram_pct(level: int) -> Optional[float]:
            peak = _number(vram_by_threads.get(level))
            if not peak or not total_vram:
                return None
            return peak / total_vram * 100.0

        def apply(level: int, **extra: Any) -> Dict[str, Any]:
            tuning = dict(baseline.as_dict())
            tuning["worker_count"] = ResourceManager.clamp(
                "worker_count", level, hardware)
            if ort:
                for name in ("ort_intra_threads", "ort_inter_threads",
                             "opencv_threads"):
                    if ort.get(name) is not None:
                        tuning[name] = ResourceManager.clamp(name, ort[name],
                                                             hardware)
            if frame_format:
                tuning["output_image_format"] = frame_format
            tuning.update(extra)
            return tuning

        presets: Dict[str, Preset] = {}

        # -- Max Throughput: the argmax, subject only to the hard admission
        #    ceiling. It is allowed to sit above the Balanced VRAM bar; it is
        #    not allowed into the paging band.
        admissible = [(level, fps) for level, fps in thread_curve
                      if (vram_pct(level) or 0.0) <= VRAM_ADMISSION_CEILING * 100.0]
        max_violation = ""
        if admissible:
            best_level, best_fps = max(admissible, key=lambda item: item[1])
        elif thread_curve:
            # Nothing on this device fits under the ceiling. Take the smallest
            # footprint rather than the fastest -- every arm is already in the
            # paging band, so throughput here is not a meaningful ranking.
            best_level, best_fps = min(thread_curve, key=lambda item: item[0])
            max_violation = (
                "no measured arm fits under %.0f%% of this device's %.1f GB: "
                "the smallest footprint is offered instead, and this device is "
                "under-provisioned for the configured models"
                % (VRAM_ADMISSION_CEILING * 100, total_vram))
        else:
            best_level, best_fps = baseline.worker_count, 0.0
        presets["max_throughput"] = Preset(
            "Max Throughput", apply(best_level), best_fps, vram_pct(best_level),
            constraints=("peak VRAM <= %.0f%% of total" % (VRAM_ADMISSION_CEILING * 100),),
            rationale=(max_violation or
                       ("highest measured sustained throughput among admissible "
                        "arms (%d workers)" % best_level)),
            provisional=not noise.measured,
            constraint_violated=bool(max_violation), violation=max_violation)

        # -- Balanced: the KNEE, not the argmax, and additionally capped at the
        #    80% VRAM bar. The knee is what the shipped pool tiers were chosen
        #    on: past it a bigger number buys a fraction of a percent for a
        #    doubling of resources, and the next run's noise could reverse it.
        balanced_curve = [(level, fps) for level, fps in thread_curve
                          if (vram_pct(level) or 0.0) <= VRAM_BALANCED_CEILING * 100.0]
        balanced_violation = ""
        if balanced_curve:
            knee_level = knee([fps for _, fps in balanced_curve],
                              [level for level, _ in balanced_curve],
                              self.BALANCED_GAIN)
            knee_fps = dict(balanced_curve).get(knee_level)
        else:
            knee_level, knee_fps = best_level, best_fps
            balanced_violation = (
                "no measured arm fits under %.0f%% of this device's %.1f GB, so "
                "the Balanced VRAM contract cannot be met here; reduce the "
                "enhancer tier or the mask engine before trusting this preset"
                % (VRAM_BALANCED_CEILING * 100, total_vram))
        presets["balanced"] = Preset(
            "Balanced", apply(knee_level), knee_fps, vram_pct(knee_level),
            constraints=("peak VRAM <= %.0f%% of total" % (VRAM_BALANCED_CEILING * 100),
                         "within %.0f-%.0f%% of the best admissible arm"
                         % (BALANCED_FPS_BAND[0] * 100, BALANCED_FPS_BAND[1] * 100)),
            rationale=(balanced_violation or
                       ("smallest worker count still delivering %.0f%% of the "
                        "best admissible arm, so the thermal and VRAM cost is "
                        "not paid for a few percent of throughput"
                        % (BALANCED_FPS_BAND[0] * 100))),
            provisional=not noise.measured,
            constraint_violated=bool(balanced_violation),
            violation=balanced_violation)

        # -- Stable / Low-Power.
        low_level = max(1, int(round(knee_level * self.LOW_POWER_THREAD_FRACTION)))
        if levels:
            low_level = min(levels, key=lambda level: abs(level - low_level))
        low_fps = dict(thread_curve).get(low_level)
        low_tuning = apply(low_level,
                           queue_depth=max(1, min(2, baseline.queue_depth)),
                           in_flight_frames=max(1, min(2, baseline.in_flight_frames)))
        cost_pct = None
        if low_fps and knee_fps:
            cost_pct = (knee_fps - low_fps) / knee_fps * 100.0
        presets["stable_low_power"] = Preset(
            "Quiet / Low Power", low_tuning, low_fps, vram_pct(low_level),
            constraints=("worker count halved from the knee",
                         "queue depth and in-flight frames capped at 2"),
            rationale=("reduced thread and buffer footprint for background "
                       "rendering%s"
                       % ("" if cost_pct is None
                          else "; costs %.1f%% throughput against Balanced"
                               % cost_pct)),
            provisional=not noise.measured)

        return presets


# Internal preset key -> the key `storage._PRESET_KEYS` requires.
#
# The internal name is kept because the React panel, the dashboard and the
# existing tests all address it; renaming it there to satisfy a storage schema
# would be the schema dictating the UI's vocabulary. The translation lives
# here, at the one boundary that needs it.
_STORAGE_PRESET_KEYS = {
    "max_throughput": "max_throughput",
    "balanced": "balanced",
    "stable_low_power": "quiet",
}


def to_storage_presets(presets: Mapping[str, Any],
                       temp_format: str = "png") -> Dict[str, Dict[str, Any]]:
    """Reshape presets into the exact record `storage` accepts.

    Storage requires all three keys, each holding exactly ``threads``,
    ``provider_options`` and ``temp_format``. Anything else -- including the
    measured FPS and the rationale this package cares about -- is refused, so
    the richer Preset stays in memory and only these three fields are filed.
    """
    result: Dict[str, Dict[str, Any]] = {}
    for internal, stored in _STORAGE_PRESET_KEYS.items():
        preset = presets.get(internal)
        tuning = {}
        if isinstance(preset, Mapping):
            tuning = preset.get("tuning") or {}
        elif preset is not None:
            tuning = getattr(preset, "tuning", {}) or {}
        provider_options = {
            name: tuning[name] for name in
            ("ort_intra_threads", "ort_inter_threads", "opencv_threads",
             "arena_extend_strategy", "cudnn_conv_algo_search",
             "gpu_mem_limit_gb", "trt_context_count")
            if name in tuning and tuning[name] is not None}
        result[stored] = {
            "threads": max(1, _integer(tuning.get("worker_count"), 1)),
            "provider_options": provider_options,
            "temp_format": str(tuning.get("output_image_format")
                               or temp_format or "png"),
        }
    return result


# --------------------------------------------------------------------------
# report + the guided search itself
# --------------------------------------------------------------------------

@dataclass
class OptimizerReport:
    schema_version: int = SCHEMA_VERSION
    created_at: float = field(default_factory=time.time)
    hardware: Dict[str, Any] = field(default_factory=dict)
    workload: Dict[str, Any] = field(default_factory=dict)
    search_space: Dict[str, Any] = field(default_factory=dict)
    noise_floor: Dict[str, Any] = field(default_factory=dict)
    phase_a: Dict[str, Any] = field(default_factory=dict)
    phase_b: Dict[str, Any] = field(default_factory=dict)
    phase_c: Dict[str, Any] = field(default_factory=dict)
    bottleneck: Dict[str, Any] = field(default_factory=dict)
    presets: Dict[str, Any] = field(default_factory=dict)
    measurements: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


class GuidedOptimizer:
    """Progressive multi-pass search: threads, then VRAM headroom, then disk.

    Not a grid.  Each phase changes ONE concern from the current best and stops
    as soon as the curve flattens, because every arm is a real render: an
    exhaustive grid over the axes in ``SearchSpace`` would be several hundred
    renders and the machine's noise floor would swamp most of the cells anyway.

    The ``measure`` callback must run the real pipeline and return a mapping
    ``Measurement.from_mapping`` understands.  It is required for the same
    reason ``RuntimeAutotuner`` requires one: an isolated GPU probe must not be
    mistakeable for an end-to-end result.
    """

    BASELINE_REPLICATES = 3

    def __init__(self, measure: Callable[[Mapping[str, Any], int], Any],
                 hardware: Optional[HardwareProfile] = None,
                 workload: Optional[WorkloadProfile] = None,
                 settings: Any = None,
                 baseline: Optional[RuntimeTuning] = None,
                 log: Optional[Callable[[str], None]] = None):
        if measure is None:
            raise ValueError(
                "GuidedOptimizer requires an end-to-end measure callback; "
                "there is deliberately no synthetic fallback")
        self.measure = measure
        self.hardware = hardware or HardwareProfiler().profile()
        self.workload = workload or WorkloadProfile()
        self.settings = settings
        self.baseline = baseline or RuntimeTuning()
        self.space = SearchSpace(self.hardware, self.workload, settings)
        self.analyzer = BottleneckAnalyzer()
        self.presets = PresetBuilder()
        self.log = log or (lambda message: None)
        self._measurements: List[Measurement] = []
        self._position = 0

    # -- one arm ---------------------------------------------------------
    def _run(self, config: Mapping[str, Any], frames: int,
             label: str = "") -> Measurement:
        started = time.perf_counter()
        try:
            raw = self.measure(dict(config), frames)
        except Exception as exc:                      # a failed arm is data
            raw = {"error": "%s: %s" % (type(exc).__name__, exc), "stable": False}
        result = Measurement.from_mapping(raw, config=config, label=label,
                                          position=self._position)
        if not result.frames:
            result.frames = frames
            result.provisional = frames < ACCEPTANCE_FRAMES
        # The arm's own wall clock, independent of whatever the callback chose
        # to report. It is what makes a cold first arm visible: slot 0 pays the
        # TensorRT engine build, and that shows up here as seconds the fps
        # number alone does not explain.
        result.raw["arm_wall_seconds"] = round(time.perf_counter() - started, 3)
        self._position += 1
        self._measurements.append(result)
        self.log("  [%02d] %-28s %6.2f fps  faces %d/%d  vram %.2f GB%s"
                 % (result.position, label or "arm", result.fps,
                    result.faces_swapped, result.faces_seen, result.peak_vram_gb,
                    "" if result.stable else "  FAILED: " + result.error))
        return result

    # -- noise floor (R1) ------------------------------------------------
    def measure_noise_floor(self, frames: int = ACCEPTANCE_FRAMES) -> NoiseFloor:
        """Re-measure the UNCHANGED baseline and derive the acceptance threshold.

        This runs before any candidate.  Skipping it is how a search comes to
        promote whichever arm the noise favoured.
        """
        self.log("Phase 0: baseline replicates (acceptance threshold is measured, "
                 "not assumed)")
        config = dict(self.baseline.as_dict())
        replicates = [self._run(config, frames, "baseline r%d" % (index + 1))
                      for index in range(max(2, self.BASELINE_REPLICATES))]
        floor = NoiseFloor.from_replicates([m.fps for m in replicates if m.stable],
                                           frames=frames)
        self.log("  spread %.2f%% -> acceptance threshold %.2f%% (%s)"
                 % (floor.spread_pct, floor.threshold_pct, floor.source))
        return floor

    # -- Phase A ---------------------------------------------------------
    def phase_a_threads(self, noise: NoiseFloor,
                        frames: int = RANKING_FRAMES,
                        confirm_frames: int = ACCEPTANCE_FRAMES) -> Dict[str, Any]:
        """Coarse thread sweep, counterbalanced, then confirm the winner at length.

        The sweep locates the inflection where added workers stop feeding the
        GPU and start contending for it.  The ranking pass may use a short
        window -- an ordering survives warm-up better than an absolute value --
        but the winner is re-measured at ``confirm_frames`` before it is
        allowed to become a recommendation.
        """
        levels = self.space.thread_levels()
        self.log("Phase A: thread sweep %s (counterbalanced)" % (list(levels),))
        order = counterbalanced_sweep(levels)
        results: Dict[int, List[Measurement]] = {}
        for level in order:
            config = dict(self.baseline.as_dict())
            config["worker_count"] = level
            measurement = self._run(config, frames, "threads=%d" % level)
            results.setdefault(level, []).append(measurement)

        curve: List[Tuple[int, float]] = []
        vram: Dict[int, float] = {}
        for level in levels:
            arms = [m for m in results.get(level, []) if m.stable and m.fps > 0]
            if not arms:
                continue
            curve.append((level, _mean([m.fps for m in arms])))
            vram[level] = max(m.peak_vram_gb for m in arms)

        inflection = self._inflection(curve, noise)
        knee_level = knee([fps for _, fps in curve], [level for level, _ in curve],
                          THREAD_GAIN) if curve else self.baseline.worker_count

        confirmation: Dict[str, Any] = {}
        if confirm_frames > frames and curve:
            best_level = max(curve, key=lambda item: item[1])[0]
            config = dict(self.baseline.as_dict())
            config["worker_count"] = best_level
            confirmed = self._run(config, confirm_frames,
                                  "confirm threads=%d" % best_level)
            confirmation = {
                "level": best_level, "fps": confirmed.fps,
                "frames": confirm_frames,
                "provisional": confirmed.provisional,
                "note": ("a short ranking window is trusted for ORDER only; "
                         "this arm is the one an acceptance claim may cite"),
            }

        flat = (not noise.resolves(self._curve_span_pct(curve))) if curve else True
        return {
            "levels": list(levels),
            "curve": [{"threads": level, "fps": round(fps, 4),
                       "peak_vram_gb": round(vram.get(level, 0.0), 3)}
                      for level, fps in curve],
            "knee": knee_level,
            "inflection": inflection,
            "confirmation": confirmation,
            "curve_span_pct": round(self._curve_span_pct(curve), 3),
            "flat_within_noise": flat,
            "verdict": ("thread count does not resolve above this machine's "
                        "%.1f%% noise floor across a %dx range; the knee is "
                        "chosen for the smallest footprint, not for speed"
                        % (noise.threshold_pct,
                           (max(levels) // max(1, min(levels))) if levels else 1)
                        if flat else
                        "throughput responds to worker count; knee at %d"
                        % knee_level),
        }

    def _curve_span_pct(self, curve: Sequence[Tuple[int, float]]) -> float:
        values = [fps for _, fps in curve if fps > 0]
        if len(values) < 2:
            return 0.0
        return (max(values) - min(values)) / _mean(values) * 100.0

    def _inflection(self, curve: Sequence[Tuple[int, float]],
                    noise: NoiseFloor) -> Dict[str, Any]:
        """The first level where adding workers stops paying, or None.

        "Stops paying" is defined against the MEASURED noise floor, not a fixed
        percentage: on a rig with an 8% spread a 3% step down is not a real
        inflection and must not be reported as one.
        """
        if len(curve) < 2:
            return {"level": None, "reason": "not enough levels measured"}
        for index in range(1, len(curve)):
            previous_level, previous_fps = curve[index - 1]
            level, fps = curve[index]
            if previous_fps <= 0:
                continue
            delta = (fps - previous_fps) / previous_fps * 100.0
            if delta < 0 and noise.resolves(delta):
                return {
                    "level": previous_level,
                    "reason": ("%d -> %d workers costs %.1f%%, which clears the "
                               "%.1f%% noise floor: contention, not scaling"
                               % (previous_level, level, delta,
                                  noise.threshold_pct)),
                }
        return {"level": None,
                "reason": ("no resolvable decrease across %d..%d workers; the "
                           "curve is flat within noise"
                           % (curve[0][0], curve[-1][0]))}

    # -- Phase B ---------------------------------------------------------
    def phase_b_vram(self, threads: int, noise: NoiseFloor,
                     frames: int = ACCEPTANCE_FRAMES) -> Dict[str, Any]:
        """Admit only memory configurations that stay under the headroom ceiling.

        Two things make this a measurement rather than arithmetic:

        * TensorRT allocates a context's memory on its FIRST INFERENCE, so the
          footprint of a pool level cannot be predicted before frames flow;
        * the failure above the ceiling is a THRASH, not an OOM -- nothing
          raises, so only throughput and peak VRAM together reveal it.
        """
        total = _number(self.hardware.vram_total_gb)
        ceiling_gb = max(0.0, total * VRAM_ADMISSION_CEILING - 0.0)
        reserve_gb = max(0.0, total - VRAM_RESERVE_GB)
        budget_gb = min(ceiling_gb, reserve_gb) if total else 0.0
        self.log("Phase B: VRAM headroom -- budget %.2f GB of %.2f GB total"
                 % (budget_gb, total))

        # Stage B is the deep provider pass: ONNX threading and the arena
        # allocator are tuned HERE, at the thread tier Stage A settled on,
        # because their effect is conditional on the worker count they compete
        # with. Memory axes ride along in the same staged loop.
        candidate_axes = list(self.space.ort_axes()) + list(self.space.memory_axes())
        axes = [axis for axis in candidate_axes if axis.reachable and axis.values]
        skipped = [{"axis": axis.name,
                    "reason": axis.unreachable_reason or "no values to try"}
                   for axis in candidate_axes
                   if not (axis.reachable and axis.values)]

        rows: List[Dict[str, Any]] = []
        accepted = dict(self.baseline.as_dict())
        accepted["worker_count"] = threads

        if not axes:
            # Nothing to search: a CPU-only or non-TensorRT device reaches none
            # of these axes. Return BEFORE the reference arm -- measuring a
            # reference for a comparison that will never happen is a whole
            # 600-frame render spent on nothing.
            self.log("    no reachable memory axis on this device; phase skipped")
            return {
                "total_vram_gb": round(total, 2),
                "budget_gb": round(budget_gb, 2),
                "admission_ceiling_pct": VRAM_ADMISSION_CEILING * 100.0,
                "reference_fps": None,
                "rows": [], "skipped_axes": skipped, "accepted": {},
                "note": ("no memory axis is reachable on this device, so no "
                         "arm was rendered for this phase"),
            }

        # Phase A has already moved the worker count, so the phase-0 baseline is
        # no longer the right reference: comparing a memory axis against it
        # would credit that axis with the thread gain and promote it for an
        # improvement it did not produce. Re-measure the reference HERE, at the
        # thread count this phase actually runs.
        reference = self._run(accepted, frames,
                              "phase B reference (threads=%d)" % threads)
        reference_fps = reference.fps
        self.log("    reference at the Phase A thread count: %.2f fps" % reference_fps)

        for axis in axes:
            for value in axis.values:
                if value == accepted.get(axis.applies_to):
                    continue
                config = dict(accepted)
                config[axis.applies_to] = value
                measurement = self._run(config, frames,
                                        "%s=%s" % (axis.name, value))
                comparable, reason = measurement.comparable_to(reference)
                over_budget = bool(budget_gb) and measurement.peak_vram_gb > budget_gb
                row = {
                    "axis": axis.name, "value": value,
                    "fps": round(measurement.fps, 4),
                    "peak_vram_gb": round(measurement.peak_vram_gb, 3),
                    "vram_pct": (round(measurement.peak_vram_gb / total * 100.0, 1)
                                 if total else None),
                    "admitted": bool(comparable and not over_budget
                                     and measurement.stable),
                }
                if over_budget:
                    row["rejected"] = ("peak %.2f GB exceeds the %.2f GB headroom "
                                       "budget" % (measurement.peak_vram_gb, budget_gb))
                elif not comparable:
                    row["rejected"] = reason        # R3
                row["vs_reference_pct"] = (
                    round((measurement.fps - reference_fps) / reference_fps * 100.0, 2)
                    if reference_fps else None)
                rows.append(row)
                # Staged, not competitive: each axis is judged against the
                # configuration currently in hand, and a promotion moves the
                # reference so the next axis is judged against the new best.
                if axis.requires_output_check and not _output_verified(measurement):
                    row["admitted"] = False
                    row["rejected"] = (
                        "this axis can change the OUTPUT, and the measure "
                        "callback reported no output-equivalence signal. It is "
                        "not promoted on throughput alone: the failure mode is "
                        "a silently unenhanced frame on the FASTEST arm.")
                    self.log("    NOT promoted %s=%s -- unverified output"
                             % (axis.name, value))
                if row["admitted"] and noise.beats(measurement.fps, reference_fps):
                    # CONFIRM BEFORE PROMOTING. A single arm clearing the bar is
                    # not enough: the spread estimated from a handful of
                    # replicates is itself noisy (measured 1.64% on one run and
                    # 3.92% on the next of the same search), and the low
                    # estimate is exactly when a noise winner gets through. The
                    # candidate has to do it twice.
                    confirm = self._run(config, frames,
                                        "confirm %s=%s" % (axis.name, value))
                    confirm_ok, confirm_reason = confirm.comparable_to(reference)
                    row["confirmation_fps"] = round(confirm.fps, 4)
                    if confirm_ok and noise.beats(confirm.fps, reference_fps):
                        self.log("    promoted %s=%s (+%.1f%% then +%.1f%% on "
                                 "confirmation, clears the %.2f%% floor twice)"
                                 % (axis.name, value, row["vs_reference_pct"],
                                    (confirm.fps - reference_fps) / reference_fps * 100.0,
                                    noise.threshold_pct))
                        accepted[axis.applies_to] = value
                        # Promote on the CONFIRMATION, not on the arm that won
                        # the first draw: taking the luckier of two numbers as
                        # the new reference biases every later axis against
                        # itself.
                        reference_fps = confirm.fps
                        reference = confirm
                        row["confirmed"] = True
                    else:
                        row["confirmed"] = False
                        row["rejected"] = (
                            confirm_reason or
                            "cleared the floor once (%+.1f%%) but not on "
                            "re-measurement (%+.1f%%): not separable from noise"
                            % (row["vs_reference_pct"],
                               (confirm.fps - reference_fps) / reference_fps * 100.0))
                        self.log("    NOT promoted %s=%s -- %s"
                                 % (axis.name, value, row["rejected"]))
        return {
            "total_vram_gb": round(total, 2),
            "budget_gb": round(budget_gb, 2),
            "admission_ceiling_pct": VRAM_ADMISSION_CEILING * 100.0,
            "reference_fps": round(reference_fps, 4),
            "rows": rows,
            "skipped_axes": skipped,
            "accepted": {axis.applies_to: accepted.get(axis.applies_to)
                         for axis in axes},
            "note": ("each axis is compared against a reference re-measured at "
                     "the Phase A thread count, and must clear this machine's "
                     "measured %.2f%% noise floor" % noise.threshold_pct),
        }

    # -- Phase C ---------------------------------------------------------
    def phase_c_disk(self, temp_path: Optional[str] = None) -> Dict[str, Any]:
        """Temp volume speed and frame format -- gated on reachability (R5).

        The shipped video path is a zero-disk rawvideo pipe at both ends. The
        frame format only reaches a render when one of three triggers fires, so
        this reports the measurement AND whether it currently applies, instead
        of quietly recommending a setting nothing reads.
        """
        triggers = temp_path_triggers(self.settings)
        reachable = any(triggers.values())
        reason = ("" if reachable else
                  "the render path is a zero-disk rawvideo pipe; none of "
                  "keep_frames / legacy route / per-frame masks is active, so "
                  "no temp frame is written")
        self.log("Phase C: disk I/O -- temp frame path %s"
                 % ("REACHABLE" if reachable else "not reached by this configuration"))

        volume = probe_temp_volume(temp_path)
        width = self.workload.input_width or 1920
        height = self.workload.input_height or 1080
        cost = frame_format_cost(width, height)
        # Opt-in only. The temp frames ARE the encoder's input, so a lossy
        # format here is a lossy generation in the delivered video -- not a
        # scratch-space economy. See frame_format_recommendation.
        allow_lossy = _bool(_setting(self.settings,
                                     "benchmark_allow_lossy_temp_frames", False))
        recommendation = frame_format_recommendation(
            cost, volume.get("write_mb_s"), allow_lossy=allow_lossy)

        axis = self.space.frame_format_axis(reachable, reason)
        return {
            "triggers": triggers,
            "reachable": reachable,
            "unreachable_reason": reason,
            "volume": volume,
            "format_cost": cost,
            "recommendation": recommendation,
            "axis": axis.as_dict(),
            "applies_now": reachable,
            "note": ("measured regardless of reachability so the answer is "
                     "ready if a trigger is enabled; it is NOT applied to a "
                     "configuration that never writes a frame"),
        }

    # -- orchestration ---------------------------------------------------
    def run(self, ranking_frames: int = RANKING_FRAMES,
            acceptance_frames: int = ACCEPTANCE_FRAMES,
            temp_path: Optional[str] = None) -> OptimizerReport:
        report = OptimizerReport(
            hardware=self.hardware.as_dict(),
            workload=self.workload.as_dict(),
            search_space=self.space.as_dict(),
        )

        noise = self.measure_noise_floor(acceptance_frames)
        report.noise_floor = noise.as_dict()
        if not noise.measured:
            report.warnings.append(
                "the acceptance threshold was NOT measured on this machine "
                "(%s); every promotion below rests on a %.1f%% floor that was "
                "chosen, not observed" % (noise.source, noise.threshold_pct))

        report.phase_a = self.phase_a_threads(noise, ranking_frames,
                                              acceptance_frames)
        threads = _integer(report.phase_a.get("knee"), self.baseline.worker_count)
        report.phase_b = self.phase_b_vram(threads, noise, acceptance_frames)
        report.phase_c = self.phase_c_disk(temp_path)

        best = self._best_measurement()
        disk = report.phase_c.get("volume")
        verdict = self.analyzer.classify(best, self.hardware, disk)
        report.bottleneck = verdict.as_dict()

        curve = [(row["threads"], row["fps"]) for row in report.phase_a["curve"]]
        vram_by_threads = {row["threads"]: row["peak_vram_gb"]
                           for row in report.phase_a["curve"]}
        frame_format = None
        if report.phase_c.get("applies_now"):
            frame_format = (report.phase_c.get("recommendation") or {}).get("choice")
        tuned = replace(self.baseline, **{
            name: value for name, value in
            (report.phase_b.get("accepted") or {}).items()
            if value is not None and hasattr(self.baseline, name)
        })
        presets = self.presets.build(tuned, self.hardware, curve, vram_by_threads,
                                     noise, frame_format=frame_format)
        report.presets = {key: preset.as_dict() for key, preset in presets.items()}
        report.measurements = [m.as_dict() for m in self._measurements]

        if report.phase_a.get("flat_within_noise"):
            report.warnings.append(
                "the thread curve is flat within this machine's noise floor: "
                "the presets differ in footprint, not in measurable speed")
        for preset in presets.values():
            if preset.constraint_violated:
                report.warnings.append("%s: %s" % (preset.name, preset.violation))
        if not any(m.work_verified for m in self._measurements):
            report.warnings.append(
                "no arm reported faces_seen/faces_swapped, so the comparability "
                "guard could not run: a candidate that went faster by doing "
                "less work would have been accepted. Wire swap_log_counts() "
                "into the measure callback before trusting any promotion here")
        if verdict.kind == "unknown":
            report.warnings.append(
                "the bottleneck could not be classified: " +
                ", ".join(verdict.missing_signals))
        return report

    def _best_measurement(self) -> Measurement:
        arms = [m for m in self._measurements if m.stable and m.fps > 0]
        if not arms:
            return self._measurements[-1] if self._measurements else Measurement()
        return max(arms, key=lambda m: m.fps)


def _output_verified(measurement: Measurement) -> bool:
    """Did the measure callback prove this arm produced the RIGHT pixels?

    Accepts any of the signals the harnesses in this repo already produce. An
    absent signal is not a pass -- that is the whole lesson of the cuDNN
    finding, where every count and integrity check read clean while four
    enhancers wrote unenhanced frames.
    """
    raw = measurement.raw or {}
    for key in ("output_equivalent", "output_verified", "identity_ok"):
        if key in raw:
            return _bool(raw[key], False)
    for key in ("identity_cosine", "identity"):
        if key in raw:
            # A good swap sits near 0.41-0.45 here; an unswapped/unenhanced
            # frame re-detects against its own original at ~0.96.
            return _number(raw[key], 1.0) < 0.7
    return False


def _setting(settings: Any, name: str, default: Any = None) -> Any:
    if settings is None:
        return default
    if isinstance(settings, Mapping):
        return settings.get(name, default)
    return getattr(settings, name, default)
