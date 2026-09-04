"""End-to-end simulation of the optimizer across four hardware profiles.

``test_benchmark_optimizer.py`` unit-tests the pieces.  This drives the WHOLE
search -- noise floor, Phase A/B/C, bottleneck classification, preset
generation -- against a simulated pipeline whose physics differ per machine,
and asserts the optimizer reaches the right conclusion on each.

THE SIMULATOR IS THE INTERESTING PART
-------------------------------------
A mock that returns a constant proves nothing: every phase would "pass" while
measuring an inert callback.  ``SimulatedMachine`` therefore models the four
mechanisms this pipeline actually exhibits, each taken from a measurement in
the project record:

1. **Throughput saturates, then contends.**  Workers help until the GPU is
   saturated; past that they cost.  (Measured: 12.41 / 12.35 / 12.32 fps at
   4 / 10 / 20 workers on a 12GB desktop -- flat, trending down.)
2. **VRAM pressure is a cliff, not a slope.**  Above the paging threshold
   throughput COLLAPSES while utilization stays high and nothing raises.
   (Measured: pool 8 on a 12GB card ran 2-2.5 fps against pool 2's 45.3, and
   presented as a hang.)
3. **A host-limited machine cannot use a bigger GPU.**  On the 24GB profile
   the per-face host work caps throughput with the GPU half idle -- the
   pipeline's real steady state.
4. **Run-to-run noise is a property of the machine**, so each profile carries
   its own spread and the optimizer has to discover it.

Each profile declares the verdict it should produce, and the tests assert the
optimizer arrives there without being told.
"""
import os
import random
import sys
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.benchmark.optimizer import (
    VRAM_ADMISSION_CEILING,
    VRAM_BALANCED_CEILING,
    BottleneckAnalyzer,
    GuidedOptimizer,
    Measurement,
    NoiseFloor,
)
from roop.runtime_optimizer import HardwareProfile, RuntimeTuning, WorkloadProfile


# --------------------------------------------------------------------------
# the simulated pipeline
# --------------------------------------------------------------------------

@dataclass
class SimulatedMachine:
    """A deterministic stand-in for one machine's end-to-end pipeline.

    ``measure(config, frames)`` returns exactly the mapping a real
    ``measure`` callback returns, so the optimizer cannot tell it apart from
    a render.  Determinism comes from a seeded RNG consumed in call order,
    which is fixed because the search is single-threaded.
    """

    name: str
    hardware: HardwareProfile
    # Throughput model.
    per_worker_fps: float = 1.6        # host-side scaling before saturation
    gpu_ceiling_fps: float = 12.0      # what the GPU can sustain, whatever the host does
    host_ceiling_fps: float = 1e9      # what the HOST can sustain (profile 3's limiter)
    contention_fps_per_worker: float = 0.0   # cost of each worker past saturation
    saturation_workers: int = 8
    # Memory model, in GB.
    vram_base: float = 4.0
    vram_per_worker: float = 0.02
    vram_per_context: float = 1.8
    # Above this fraction of total VRAM the driver pages and throughput dies.
    paging_fraction: float = 0.90
    paging_collapse: float = 0.06      # residual throughput while thrashing
    # Utilization model.
    cpu_base_pct: float = 18.0
    cpu_per_worker_pct: float = 1.2
    core_peak_pct: Optional[float] = 45.0
    gpu_reports_utilization: bool = True
    # Disk model.
    disk_wait_pct: Optional[float] = None
    disk_write_mb_s: Optional[float] = None
    # This machine's run-to-run spread, as a percentage.
    noise_pct: float = 1.0
    seed: int = 11
    # Work done. Constant across arms, so the R3 comparability guard passes and
    # any test that wants to break it must do so deliberately.
    faces_seen: int = 900
    faces_swapped: int = 880

    calls: list = field(default_factory=list)

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    # -- the model -------------------------------------------------------
    def _contexts(self, config: Mapping[str, Any]) -> int:
        return max(1, int(config.get("trt_context_count", 1) or 1)) + \
            max(0, int(config.get("detmask_pool_size", 0) or 0)) // 2

    def peak_vram(self, config: Mapping[str, Any]) -> float:
        if not self.hardware.vram_total_gb:
            return 0.0
        workers = max(1, int(config.get("worker_count", 1)))
        return (self.vram_base
                + self.vram_per_worker * workers
                + self.vram_per_context * (self._contexts(config) - 1))

    def true_fps(self, config: Mapping[str, Any]) -> Tuple[float, bool]:
        """Noise-free throughput, and whether the card is paging."""
        workers = max(1, int(config.get("worker_count", 1)))
        contexts = self._contexts(config)

        # Host-side scaling up to saturation, then contention.
        fps = self.per_worker_fps * min(workers, self.saturation_workers)
        if workers > self.saturation_workers:
            fps -= self.contention_fps_per_worker * (workers - self.saturation_workers)

        # More GPU contexts raise the GPU ceiling, with diminishing returns --
        # the measured shape (+46% at 2 contexts, little beyond 3).
        ceiling = self.gpu_ceiling_fps * (1.0 + 0.46 * min(contexts - 1, 1)
                                          + 0.04 * max(0, contexts - 2))
        fps = min(fps, ceiling, self.host_ceiling_fps)

        paging = False
        total = self.hardware.vram_total_gb
        if total and self.peak_vram(config) > total * self.paging_fraction:
            # The cliff. Utilization stays high; throughput does not.
            paging = True
            fps *= self.paging_collapse
        return max(0.01, fps), paging

    def measure(self, config: Mapping[str, Any], frames: int) -> Dict[str, Any]:
        self.calls.append((dict(config), frames))
        fps, paging = self.true_fps(config)
        jitter = 1.0 + self._rng.uniform(-self.noise_pct, self.noise_pct) / 100.0
        fps *= jitter

        workers = max(1, int(config.get("worker_count", 1)))
        vram = self.peak_vram(config)
        cpu = min(99.0, self.cpu_base_pct + self.cpu_per_worker_pct * workers)

        gpu = None
        if self.gpu_reports_utilization:
            if paging:
                # THE TRAP: a thrashing card reports near-full utilization. A
                # classifier that reads utilization before VRAM calls this
                # "GPU compute bound" and recommends exactly the wrong action.
                gpu = 99.0
            else:
                headroom = fps / max(1e-9, self.gpu_ceiling_fps)
                gpu = min(99.0, 100.0 * min(1.0, headroom))

        median_ms = 1000.0 / max(1e-6, fps)
        payload = {
            "fps": fps,
            "frames": frames,
            "faces_seen": self.faces_seen,
            "faces_swapped": self.faces_swapped,
            "peak_vram_gb": vram,
            "peak_ram_gb": min(self.hardware.ram_total_gb * 0.5, 3.5 + 0.1 * workers),
            "cpu_utilization_pct": cpu,
            "gpu_utilization_pct": gpu,
            "per_core_peak_pct": self.core_peak_pct,
            "frame_time_median_ms": median_ms,
            # Paging shows up as hitching, which is the corroborating signal
            # the VRAM verdict raises its confidence on.
            "frame_time_p99_ms": median_ms * (8.0 if paging else 1.3),
            "stage_seconds": {"frame_total": 100.0, "detect": 28.0, "swap": 34.0,
                              "mask": 22.0, "enhance": 14.0, "decode": 1.4,
                              "encode": 0.6},
            "stable": True,
        }
        if self.disk_wait_pct is not None:
            payload["disk_wait_pct"] = self.disk_wait_pct
        if self.disk_write_mb_s is not None:
            payload["disk_write_mb_s"] = self.disk_write_mb_s
        return payload


# --------------------------------------------------------------------------
# the four profiles
# --------------------------------------------------------------------------

def _hardware(**kwargs) -> HardwareProfile:
    base = dict(gpu_vendor="nvidia", ram_available_gb=8.0, fp16_supported=True,
                nvenc_available=True, nvdec_available=True)
    base.update(kwargs)
    return HardwareProfile(**base)


LOW_END = HardwareProfile(
    gpu_name="GTX 1650 4GB", gpu_vendor="nvidia", vram_total_gb=4.0,
    vram_available_gb=3.4, cuda_available=True, tensorrt_available=True,
    fp16_supported=True, nvenc_available=True,
    cpu_physical_cores=4, cpu_logical_cores=8,
    ram_total_gb=8.0, ram_available_gb=3.0)

MID_RANGE = HardwareProfile(
    gpu_name="RTX 3070 8GB", gpu_vendor="nvidia", vram_total_gb=8.0,
    vram_available_gb=6.8, cuda_available=True, tensorrt_available=True,
    fp16_supported=True, nvenc_available=True,
    cpu_physical_cores=8, cpu_logical_cores=16,
    ram_total_gb=16.0, ram_available_gb=8.0)

HIGH_END = HardwareProfile(
    gpu_name="RTX 4090 24GB", gpu_vendor="nvidia", vram_total_gb=24.0,
    vram_available_gb=22.0, cuda_available=True, tensorrt_available=True,
    fp16_supported=True, bf16_supported=True, nvenc_available=True,
    cpu_physical_cores=16, cpu_logical_cores=32,
    ram_total_gb=64.0, ram_available_gb=40.0)

CPU_ONLY = HardwareProfile(
    gpu_name="", gpu_vendor="unknown", vram_total_gb=0.0, vram_available_gb=0.0,
    cuda_available=False, tensorrt_available=False, nvenc_available=False,
    cpu_physical_cores=8, cpu_logical_cores=16,
    ram_total_gb=32.0, ram_available_gb=20.0)


def low_end_machine(**overrides) -> SimulatedMachine:
    """4GB: the model working set alone sits in the paging band.

    3.7 GB of 4.0 is 92.5%. This is the card where a second context is not a
    tuning choice -- it is the difference between rendering and thrashing.
    """
    params = dict(
        name="low-end 4GB", hardware=LOW_END, per_worker_fps=0.9,
        gpu_ceiling_fps=4.5, saturation_workers=4,
        contention_fps_per_worker=0.25, vram_base=3.66, vram_per_worker=0.01,
        vram_per_context=1.9, paging_fraction=0.98, cpu_base_pct=22.0,
        core_peak_pct=55.0, noise_pct=1.0, seed=101)
    params.update(overrides)
    return SimulatedMachine(**params)


def mid_range_machine(**overrides) -> SimulatedMachine:
    """8GB: the GPU is the limit and there is headroom to feed it harder."""
    params = dict(
        name="mid-range 8GB", hardware=MID_RANGE, per_worker_fps=1.7,
        gpu_ceiling_fps=11.0, saturation_workers=8,
        contention_fps_per_worker=0.10, vram_base=4.6, vram_per_worker=0.02,
        vram_per_context=1.5, cpu_base_pct=20.0, cpu_per_worker_pct=1.4,
        core_peak_pct=52.0, noise_pct=1.2, seed=202)
    params.update(overrides)
    return SimulatedMachine(**params)


def high_end_machine(**overrides) -> SimulatedMachine:
    """24GB: a GPU too big for its host.

    ``host_ceiling_fps`` caps throughput well under the GPU ceiling, so the
    card idles and a core saturates -- the project's own measured steady state,
    where threads/contexts/queues were each a dead end and only removing
    per-face host work moved the clock.
    """
    params = dict(
        name="high-end 24GB", hardware=HIGH_END, per_worker_fps=2.4,
        gpu_ceiling_fps=60.0, host_ceiling_fps=22.0, saturation_workers=10,
        contention_fps_per_worker=0.08, vram_base=6.0, vram_per_worker=0.05,
        vram_per_context=2.2, cpu_base_pct=30.0, cpu_per_worker_pct=2.2,
        core_peak_pct=100.0, noise_pct=1.0, seed=303)
    params.update(overrides)
    return SimulatedMachine(**params)


def cpu_only_machine(**overrides) -> SimulatedMachine:
    """No GPU at all: no utilization signal, and no memory axis to search."""
    params = dict(
        name="CPU-only", hardware=CPU_ONLY, per_worker_fps=0.22,
        gpu_ceiling_fps=1e9, host_ceiling_fps=2.0, saturation_workers=8,
        contention_fps_per_worker=0.05, vram_base=0.0, vram_per_worker=0.0,
        vram_per_context=0.0, cpu_base_pct=70.0, cpu_per_worker_pct=2.0,
        core_peak_pct=100.0, gpu_reports_utilization=False,
        noise_pct=1.5, seed=404)
    params.update(overrides)
    return SimulatedMachine(**params)


# factory -> (expected bottleneck, does the thread curve show contention?)
#
# Not every machine HAS an inflection, and pretending otherwise would be a
# fixture that lies. On the two GPU-ceiling-bound profiles the curve goes
# flat before contention can show, so the honest expectation there is
# "no resolvable decrease" -- which is also what the real 12GB desktop
# measured across a 5x worker range.
ALL_PROFILES = {
    "low_end_4gb": (low_end_machine, "GPU VRAM bound", True),
    "mid_range_8gb": (mid_range_machine, "GPU compute bound", False),
    "high_end_24gb": (high_end_machine, "CPU bound", False),
    "cpu_only": (cpu_only_machine, "CPU bound", True),
}


WORKLOAD = WorkloadProfile(input_width=1280, input_height=720,
                           faces_per_frame=1.0, enhancement_enabled=True,
                           video_length_frames=13305)
SETTINGS = {"keep_frames": False, "use_new_method": True}


def optimize(machine: SimulatedMachine, baseline: Optional[RuntimeTuning] = None,
             settings: Optional[dict] = None):
    optimizer = GuidedOptimizer(
        machine.measure, hardware=machine.hardware, workload=WORKLOAD,
        settings=dict(settings or SETTINGS),
        baseline=baseline or RuntimeTuning(worker_count=4, trt_context_count=1,
                                           queue_depth=2, in_flight_frames=2))
    return optimizer.run(ranking_frames=120, acceptance_frames=600)


class ProfileFixtureTests(unittest.TestCase):
    """The simulator must actually exhibit the physics it claims to."""

    def test_the_four_profiles_are_genuinely_different_machines(self):
        seen = set()
        for factory, _, _contention in ALL_PROFILES.values():
            hardware = factory().hardware
            seen.add((hardware.vram_total_gb, hardware.cpu_logical_cores,
                      hardware.cuda_available))
        self.assertEqual(len(seen), 4)

    def test_the_simulator_is_deterministic(self):
        """A flaky fixture would make every assertion below meaningless."""
        first = optimize(mid_range_machine())
        second = optimize(mid_range_machine())
        self.assertEqual([row["fps"] for row in first.phase_a["curve"]],
                         [row["fps"] for row in second.phase_a["curve"]])

    def test_the_low_end_card_runs_pressured_and_pages_on_a_second_context(self):
        """Two different bands, and the distinction is the whole profile.

        At one context the card sits ABOVE the classifier's 90% pressure line
        but still renders -- that is a 4GB card running this workload at all.
        A second context crosses into actual paging, where throughput dies.
        """
        machine = low_end_machine()
        one = {"worker_count": 4, "trt_context_count": 1}
        two = {"worker_count": 4, "trt_context_count": 2}
        pressure = machine.peak_vram(one) / LOW_END.vram_total_gb * 100.0
        self.assertGreaterEqual(pressure, 90.0, "the profile must be VRAM-pressured")
        self.assertFalse(machine.true_fps(one)[1], "one context must still render")
        self.assertTrue(machine.true_fps(two)[1], "a second context must page")
        self.assertLess(machine.true_fps(two)[0], machine.true_fps(one)[0] * 0.2)

    def test_a_thrashing_arm_still_reports_high_gpu_utilization(self):
        """The trap the classifier has to survive: 99% util at collapsed fps."""
        machine = low_end_machine()
        payload = machine.measure({"worker_count": 4, "trt_context_count": 2}, 600)
        self.assertGreaterEqual(payload["gpu_utilization_pct"], 95.0)
        self.assertLess(payload["fps"], 1.0)

    def test_the_high_end_gpu_is_deliberately_starved_by_its_host(self):
        machine = high_end_machine()
        payload = machine.measure({"worker_count": 16, "trt_context_count": 2}, 600)
        self.assertLess(payload["gpu_utilization_pct"], 70.0)
        self.assertEqual(payload["per_core_peak_pct"], 100.0)

    def test_the_cpu_only_profile_reports_no_gpu_signal(self):
        payload = cpu_only_machine().measure({"worker_count": 8}, 600)
        self.assertIsNone(payload["gpu_utilization_pct"])


class BottleneckIdentificationTests(unittest.TestCase):
    """Requirement 3: the right limiter, on each machine, from telemetry alone."""

    def test_each_profile_is_classified_correctly(self):
        for name, (factory, expected, _contention) in ALL_PROFILES.items():
            with self.subTest(profile=name):
                report = optimize(factory())
                self.assertEqual(report.bottleneck["kind"], expected)

    def test_the_low_end_verdict_is_vram_not_gpu_compute(self):
        """Utilization must not outrank VRAM: the paging arm reports 99%."""
        report = optimize(low_end_machine())
        self.assertEqual(report.bottleneck["kind"], "GPU VRAM bound")
        self.assertIn("THRASH", report.bottleneck["recommendation"])
        self.assertTrue(any("peak VRAM" in item
                            for item in report.bottleneck["evidence"]))

    def test_hitching_raises_the_vram_verdict_from_medium_to_high(self):
        """Pressure alone is a medium verdict; pressure PLUS hitching is high.

        The best arm on the low-end profile is pressured but still rendering,
        so there is no hitching to cite and the verdict must not overclaim. An
        arm that is actually paging supplies the second signal.
        """
        pressured = optimize(low_end_machine())
        self.assertEqual(pressured.bottleneck["confidence"], "medium")
        self.assertFalse(any("hitching" in item
                             for item in pressured.bottleneck["evidence"]))

        thrashing = BottleneckAnalyzer().classify(
            Measurement.from_mapping(low_end_machine().measure(
                {"worker_count": 4, "trt_context_count": 2}, 600)), LOW_END)
        self.assertEqual(thrashing.kind, "GPU VRAM bound")
        self.assertEqual(thrashing.confidence, "high")
        self.assertTrue(any("hitching" in item for item in thrashing.evidence))

    def test_the_mid_range_verdict_names_removing_gpu_work(self):
        report = optimize(mid_range_machine())
        self.assertEqual(report.bottleneck["kind"], "GPU compute bound")
        self.assertEqual(report.bottleneck["confidence"], "high")
        # R4: the advice must be to remove GPU work, not to add threads.
        self.assertIn("REMOVING GPU work", report.bottleneck["recommendation"])

    def test_the_high_end_verdict_is_the_host_not_the_card(self):
        """A 24GB card at 6GB used and half idle is not short of anything."""
        report = optimize(high_end_machine())
        self.assertEqual(report.bottleneck["kind"], "CPU bound")
        self.assertEqual(report.bottleneck["confidence"], "high")

    def test_the_cpu_only_verdict_is_lower_confidence_and_says_why(self):
        """Same verdict as the 24GB box, reached with strictly less evidence."""
        report = optimize(cpu_only_machine())
        self.assertEqual(report.bottleneck["kind"], "CPU bound")
        self.assertEqual(report.bottleneck["confidence"], "medium")
        self.assertTrue(any("gpu_utilization_pct" in signal
                            for signal in report.bottleneck["missing_signals"]))

    def test_a_slow_volume_with_frames_on_disk_is_identified_as_disk_bound(self):
        """The fourth class, on the profile most likely to hit it."""
        machine = low_end_machine(disk_wait_pct=38.0, disk_write_mb_s=95.0,
                                  vram_base=2.0, vram_per_context=0.2)
        report = optimize(machine,
                          settings={"keep_frames": True, "use_new_method": True})
        self.assertEqual(report.bottleneck["kind"], "Disk I/O bound")
        self.assertIn("volume", report.bottleneck["recommendation"])

    def test_every_verdict_carries_its_evidence(self):
        for name, (factory, _, _contention) in ALL_PROFILES.items():
            with self.subTest(profile=name):
                report = optimize(factory())
                self.assertTrue(report.bottleneck["evidence"],
                                "a verdict with no evidence is an assertion")


class ThreadSelectionTests(unittest.TestCase):
    """Requirement 2 Phase A: the inflection, and the knee, per machine."""

    def test_the_sweep_never_offers_more_workers_than_the_machine_has(self):
        for name, (factory, _, _contention) in ALL_PROFILES.items():
            with self.subTest(profile=name):
                machine = factory()
                report = optimize(machine)
                self.assertLessEqual(max(report.phase_a["levels"]),
                                     machine.hardware.cpu_logical_cores)

    def test_each_profile_finds_its_own_contention_point(self):
        """The inflection must track the machine, not a constant.

        Two of the four profiles are GPU-ceiling-bound, so their curves go flat
        before contention can show. "No resolvable decrease" is the correct
        answer there and must not be manufactured into an inflection.
        """
        for name, (factory, _, has_contention) in ALL_PROFILES.items():
            with self.subTest(profile=name):
                machine = factory()
                report = optimize(machine)
                level = report.phase_a["inflection"]["level"]
                if has_contention:
                    self.assertIsNotNone(
                        level, "contention was modelled but not found")
                    # The inflection is the last level that still paid, so it
                    # sits at or above the saturation point.
                    self.assertGreaterEqual(level, machine.saturation_workers)
                else:
                    self.assertIsNone(
                        level, "a flat curve must not be reported as contention")
                    self.assertIn("flat within noise",
                                  report.phase_a["inflection"]["reason"])

    def test_the_selected_thread_count_is_at_or_below_the_inflection(self):
        """Selecting past the inflection would buy contention with watts."""
        for name, (factory, _, _contention) in ALL_PROFILES.items():
            with self.subTest(profile=name):
                report = optimize(factory())
                inflection = report.phase_a["inflection"]["level"]
                if inflection is None:
                    self.skipTest("no contention on this profile")
                for preset in report.presets.values():
                    self.assertLessEqual(preset["tuning"]["worker_count"],
                                         inflection)

    def test_a_weak_host_is_given_fewer_workers_than_a_strong_one(self):
        low = optimize(low_end_machine()).presets["balanced"]["tuning"]
        high = optimize(high_end_machine()).presets["balanced"]["tuning"]
        self.assertLess(low["worker_count"], high["worker_count"])

    def test_the_confirmation_arm_runs_at_the_acceptance_window(self):
        """A short ranking window may order arms; it may not promote one."""
        report = optimize(mid_range_machine())
        confirmation = report.phase_a["confirmation"]
        self.assertEqual(confirmation["frames"], 600)
        self.assertFalse(confirmation["provisional"])


class MemorySelectionTests(unittest.TestCase):
    """Requirement 2 Phase B: memory settings, per VRAM tier."""

    def test_the_low_end_card_is_never_given_a_second_context(self):
        """The whole point of the tier: 2 contexts on 4GB is a thrash."""
        report = optimize(low_end_machine())
        accepted = report.phase_b["accepted"]
        self.assertEqual(accepted.get("trt_context_count"), 1)
        self.assertEqual(report.presets["balanced"]["tuning"]["trt_context_count"], 1)

    def test_no_admitted_arm_exceeds_the_headroom_budget(self):
        for name, (factory, _, _contention) in ALL_PROFILES.items():
            with self.subTest(profile=name):
                report = optimize(factory())
                budget = report.phase_b["budget_gb"]
                if not budget:
                    continue
                for row in report.phase_b["rows"]:
                    if row["admitted"]:
                        self.assertLessEqual(row["peak_vram_gb"], budget)

    def test_an_over_budget_arm_is_rejected_with_the_reason_recorded(self):
        report = optimize(low_end_machine())
        rejected = [row for row in report.phase_b["rows"] if not row["admitted"]]
        self.assertTrue(rejected)
        self.assertTrue(any("headroom budget" in row.get("rejected", "")
                            for row in rejected))

    def test_a_host_bound_card_is_not_given_vram_it_cannot_use(self):
        """24GB of headroom is not a reason to spend it.

        On this profile the host caps throughput below the GPU ceiling, so a
        second context measures as ZERO gain for 2.2 GB. Admitting it because
        the VRAM was available would be spending memory for nothing -- the
        ceiling is an upper bound, never a target.
        """
        machine = high_end_machine()
        self.assertEqual(
            machine.true_fps({"worker_count": 16, "trt_context_count": 1})[0],
            machine.true_fps({"worker_count": 16, "trt_context_count": 2})[0],
            "fixture: this profile must gain nothing from a second context")
        report = optimize(machine)
        self.assertEqual(report.phase_b["accepted"].get("trt_context_count"), 1)
        admitted = [row for row in report.phase_b["rows"] if row["admitted"]]
        self.assertTrue(admitted, "the arm was measured and fit the budget")
        self.assertTrue(all(row.get("peak_vram_gb", 0) < 24.0 for row in admitted))

    def test_the_mid_range_card_accepts_a_context_its_vram_can_hold(self):
        report = optimize(mid_range_machine())
        accepted = report.phase_b["accepted"].get("trt_context_count")
        self.assertGreaterEqual(accepted, 2)
        peak = max(row["peak_vram_gb"] for row in report.phase_b["rows"]
                   if row["admitted"])
        self.assertLessEqual(peak, MID_RANGE.vram_total_gb * VRAM_ADMISSION_CEILING)

    def test_the_cpu_only_profile_searches_no_memory_axis_and_renders_nothing(self):
        """A phase with nothing to compare must not spend an arm on a reference."""
        machine = cpu_only_machine()
        report = optimize(machine)
        self.assertEqual(report.phase_b["rows"], [])
        self.assertEqual(report.phase_b["accepted"], {})
        self.assertIsNone(report.phase_b["reference_fps"])
        self.assertTrue(report.phase_b["skipped_axes"])
        for call, _ in machine.calls:
            self.assertNotIn("phase B", str(call))

    def test_unreachable_axes_are_reported_with_a_reason(self):
        report = optimize(cpu_only_machine())
        reasons = {row["axis"]: row["reason"]
                   for row in report.phase_b["skipped_axes"]}
        self.assertIn("trt_context_count", reasons)
        self.assertTrue(all(reasons.values()), "a skip with no reason is a mystery")


class PresetTests(unittest.TestCase):
    """Requirement 4: three presets, on every machine, inside their contracts."""

    def test_every_profile_produces_all_three_presets(self):
        for name, (factory, _, _contention) in ALL_PROFILES.items():
            with self.subTest(profile=name):
                report = optimize(factory())
                self.assertEqual(set(report.presets),
                                 {"max_throughput", "balanced", "stable_low_power"})

    def test_balanced_either_meets_its_vram_contract_or_says_it_cannot(self):
        """A preset stating "<= 80%" while sitting at 92% is lying.

        On a 4GB card no arm fits, and the honest answer is to flag the
        violation rather than present a compliant-looking number.
        """
        for name, (factory, _, _contention) in ALL_PROFILES.items():
            with self.subTest(profile=name):
                preset = optimize(factory()).presets["balanced"]
                pct = preset["projected_vram_pct"]
                if pct is None or pct <= VRAM_BALANCED_CEILING * 100.0:
                    self.assertFalse(preset["constraint_violated"])
                else:
                    self.assertTrue(preset["constraint_violated"])
                    self.assertTrue(preset["violation"])

    def test_the_under_provisioned_card_is_the_one_that_flags_a_violation(self):
        self.assertTrue(
            optimize(low_end_machine()).presets["balanced"]["constraint_violated"])
        self.assertFalse(
            optimize(mid_range_machine()).presets["balanced"]["constraint_violated"])

    def test_a_violated_constraint_is_raised_to_the_run_warnings(self):
        report = optimize(low_end_machine())
        self.assertTrue(any("Balanced" in warning for warning in report.warnings))

    def test_max_throughput_stays_out_of_the_paging_band_or_flags_it(self):
        for name, (factory, _, _contention) in ALL_PROFILES.items():
            with self.subTest(profile=name):
                preset = optimize(factory()).presets["max_throughput"]
                pct = preset["projected_vram_pct"]
                if pct is None or pct <= VRAM_ADMISSION_CEILING * 100.0:
                    self.assertFalse(preset["constraint_violated"])
                else:
                    self.assertTrue(preset["constraint_violated"])

    def test_an_under_provisioned_card_is_offered_the_smallest_footprint(self):
        """When every arm pages, throughput is not a meaningful ranking."""
        report = optimize(low_end_machine())
        preset = report.presets["max_throughput"]
        self.assertTrue(preset["constraint_violated"])
        self.assertEqual(preset["tuning"]["worker_count"],
                         min(report.phase_a["levels"]))
        self.assertIn("under-provisioned", preset["violation"])

    def test_low_power_is_always_the_smallest_footprint(self):
        for name, (factory, _, _contention) in ALL_PROFILES.items():
            with self.subTest(profile=name):
                presets = optimize(factory()).presets
                low = presets["stable_low_power"]["tuning"]
                balanced = presets["balanced"]["tuning"]
                self.assertLessEqual(low["worker_count"], balanced["worker_count"])
                self.assertLessEqual(low["queue_depth"], 2)
                self.assertLessEqual(low["in_flight_frames"], 2)

    def test_max_throughput_is_never_slower_than_balanced(self):
        for name, (factory, _, _contention) in ALL_PROFILES.items():
            with self.subTest(profile=name):
                presets = optimize(factory()).presets
                best = presets["max_throughput"]["measured_fps"]
                balanced = presets["balanced"]["measured_fps"]
                if best is not None and balanced is not None:
                    self.assertGreaterEqual(best, balanced)

    def test_balanced_gives_up_little_for_its_smaller_footprint(self):
        """The knee's contract: within THREAD_GAIN of the best admissible arm."""
        for name, (factory, _, _contention) in ALL_PROFILES.items():
            with self.subTest(profile=name):
                presets = optimize(factory()).presets
                best = presets["max_throughput"]["measured_fps"]
                balanced = presets["balanced"]["measured_fps"]
                if best and balanced:
                    self.assertGreaterEqual(balanced, best * 0.95)

    def test_every_preset_stays_inside_the_safe_bounds(self):
        for name, (factory, _, _contention) in ALL_PROFILES.items():
            machine = factory()
            for key, preset in optimize(machine).presets.items():
                with self.subTest(profile=name, preset=key):
                    workers = preset["tuning"]["worker_count"]
                    self.assertGreaterEqual(workers, 1)
                    self.assertLessEqual(workers,
                                         machine.hardware.cpu_logical_cores)


class NoiseDisciplineTests(unittest.TestCase):
    """Requirement 1/R1: the acceptance threshold is this machine's, measured."""

    def test_each_profile_measures_its_own_floor(self):
        floors = {}
        for name, (factory, _, _contention) in ALL_PROFILES.items():
            report = optimize(factory())
            with self.subTest(profile=name):
                self.assertTrue(report.noise_floor["measured"])
                self.assertEqual(len(report.noise_floor["replicates"]), 3)
            floors[name] = report.noise_floor["threshold_pct"]
        self.assertGreater(len(set(floors.values())), 1,
                           "a floor identical on four machines was not measured")

    def test_a_noisy_machine_refuses_a_gain_a_quiet_one_accepts(self):
        """The same simulated pipeline, differing only in run-to-run spread."""
        quiet = optimize(mid_range_machine(noise_pct=0.5))
        noisy = optimize(mid_range_machine(noise_pct=25.0))
        self.assertLess(quiet.noise_floor["threshold_pct"],
                        noisy.noise_floor["threshold_pct"])
        quiet_accepted = quiet.phase_b["accepted"].get("trt_context_count")
        noisy_accepted = noisy.phase_b["accepted"].get("trt_context_count")
        self.assertGreaterEqual(quiet_accepted, 2)
        self.assertLessEqual(noisy_accepted, quiet_accepted)

    def test_a_machine_too_noisy_to_measure_promotes_nothing(self):
        """A 60% spread cannot resolve a 46% effect; the answer is 'no'."""
        report = optimize(mid_range_machine(noise_pct=60.0))
        self.assertEqual(report.phase_b["accepted"].get("trt_context_count"), 1)

    def test_a_flat_curve_is_reported_as_flat_rather_than_ranked(self):
        machine = mid_range_machine(per_worker_fps=50.0, saturation_workers=1,
                                    contention_fps_per_worker=0.0, noise_pct=0.5)
        report = optimize(machine)
        self.assertTrue(report.phase_a["flat_within_noise"])
        self.assertTrue(any("footprint" in warning for warning in report.warnings))


class WorkVerificationTests(unittest.TestCase):
    """R3, through the whole search rather than at the unit."""

    def test_an_arm_that_stopped_swapping_is_never_promoted(self):
        """The +47% regression, injected into the memory phase of a real run."""
        class Sabotaged(SimulatedMachine):
            def measure(self, config, frames):
                payload = super().measure(config, frames)
                if int(config.get("trt_context_count", 1) or 1) > 1:
                    # Twice as fast, because it stopped doing the work.
                    payload.update(fps=payload["fps"] * 2.0 + 20.0,
                                   faces_seen=6, faces_swapped=2)
                return payload

        machine = Sabotaged(name="sabotaged", hardware=MID_RANGE,
                            per_worker_fps=1.7, gpu_ceiling_fps=11.0,
                            saturation_workers=8, vram_base=4.6,
                            noise_pct=1.0, seed=505)
        report = optimize(machine)
        self.assertEqual(report.phase_b["accepted"].get("trt_context_count"), 1)
        self.assertTrue(any("faces_seen" in row.get("rejected", "")
                            for row in report.phase_b["rows"]))

    def test_a_run_reporting_real_counts_raises_no_verification_warning(self):
        for name, (factory, _, _contention) in ALL_PROFILES.items():
            with self.subTest(profile=name):
                report = optimize(factory())
                self.assertFalse(
                    any("comparability guard could not run" in warning
                        for warning in report.warnings))


class DiskPhaseTests(unittest.TestCase):
    """Requirement 2 Phase C, and R5."""

    def test_the_default_zero_disk_path_is_measured_but_not_applied(self):
        for name, (factory, _, _contention) in ALL_PROFILES.items():
            with self.subTest(profile=name):
                report = optimize(factory())
                self.assertFalse(report.phase_c["applies_now"])
                self.assertIn("volume", report.phase_c)
                self.assertNotIn("output_image_format",
                                 report.presets["balanced"]["tuning"])

    def test_enabling_a_trigger_makes_the_format_apply(self):
        report = optimize(mid_range_machine(),
                          settings={"keep_frames": True, "use_new_method": True})
        self.assertTrue(report.phase_c["applies_now"])
        self.assertEqual(report.presets["balanced"]["tuning"]["output_image_format"],
                         "png")

    def test_the_chosen_format_is_lossless_by_default(self):
        """Temp frames are the encoder's input, so this is an output decision."""
        report = optimize(mid_range_machine(),
                          settings={"keep_frames": True, "use_new_method": True})
        self.assertEqual(report.phase_c["recommendation"]["choice"], "png")
        self.assertFalse(report.phase_c["recommendation"]["allow_lossy"])


class CostTests(unittest.TestCase):
    """The search must stay bounded: every arm is a real render."""

    def test_no_profile_needs_an_unreasonable_number_of_renders(self):
        for name, (factory, _, _contention) in ALL_PROFILES.items():
            with self.subTest(profile=name):
                machine = factory()
                optimize(machine)
                self.assertLessEqual(len(machine.calls), 40,
                                     "the search stopped being progressive")

    def test_the_cpu_only_search_is_the_cheapest(self):
        gpu_machine, cpu_machine = mid_range_machine(), cpu_only_machine()
        optimize(gpu_machine)
        optimize(cpu_machine)
        self.assertLess(len(cpu_machine.calls), len(gpu_machine.calls))


if __name__ == "__main__":
    unittest.main()
