"""Tests for roop.benchmark.optimizer.

These assert PROPERTIES of the search, not thresholds: a test that pins a
constant only records what the code currently says.  The properties chosen are
the ones whose violation has actually shipped here before -- a fixed acceptance
threshold promoting noise, a forward-only A/B, a faster arm that was doing less
work, a confident bottleneck verdict built from absent telemetry, and a sweep
over a code path that never runs.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.benchmark.optimizer import (
    ACCEPTANCE_FRAMES,
    VRAM_ADMISSION_CEILING,
    VRAM_BALANCED_CEILING,
    BottleneckAnalyzer,
    GuidedOptimizer,
    Measurement,
    NoiseFloor,
    PresetBuilder,
    SearchSpace,
    counterbalanced_pairs,
    counterbalanced_sweep,
    frame_format_recommendation,
    knee,
    order_corrected,
    swap_log_counts,
    temp_path_triggers,
)
from roop.runtime_optimizer import HardwareProfile, RuntimeTuning, WorkloadProfile


def hardware(vram=12.0, logical=32, physical=24, tensorrt=True, cuda=True):
    return HardwareProfile(
        gpu_name="test", gpu_vendor="nvidia", vram_total_gb=vram,
        vram_available_gb=max(1.0, vram - 2.0), cuda_available=cuda,
        tensorrt_available=tensorrt, fp16_supported=True, nvenc_available=True,
        cpu_physical_cores=physical, cpu_logical_cores=logical,
        ram_total_gb=32.0, ram_available_gb=16.0)


class KneeTests(unittest.TestCase):
    def test_matches_the_bench_implementation_it_restates(self):
        """The local copy must not drift from roop.bench._knee."""
        try:
            from roop.bench import _knee
        except Exception as exc:                       # onnxruntime unavailable
            self.skipTest("roop.bench not importable here: %s" % exc)
        cases = [([1, 2, 3], [1, 2, 3]), ([9.09, 9.5, 9.58], [12, 20, 32]),
                 ([5, 4, 3], [1, 2, 4]), ([1, 9, 2], [1, 2, 3]), ([], [1])]
        for values, labels in cases:
            self.assertEqual(knee(values, labels, 0.05),
                             _knee(values, labels, 0.05), (values, labels))

    def test_prefers_the_smallest_label_within_the_gain(self):
        # 12 threads at 9.25 is 3.4% below 32 threads at 9.58 -- inside the 5%
        # bar -- so the knee is 12: a near-tripling of the worker count must
        # earn more than that.
        self.assertEqual(knee([9.25, 9.5, 9.58], [12, 20, 32], 0.05), 12)

    def test_a_label_just_outside_the_gain_does_not_win(self):
        # The real measured curve: 9.09 at 12 threads against 9.58 at 32 is
        # 5.11% down, just OUTSIDE a 5% bar, so 12 does not qualify. The bar is
        # a real boundary and the nearest qualifying label wins instead.
        self.assertEqual(knee([9.09, 9.5, 9.58], [12, 20, 32], 0.05), 20)


class NoiseFloorTests(unittest.TestCase):
    def test_threshold_is_measured_from_the_replicates(self):
        floor = NoiseFloor.from_replicates([10.0, 10.5, 10.2])
        self.assertTrue(floor.measured)
        self.assertGreater(floor.threshold_pct, NoiseFloor.MIN_THRESHOLD_PCT)
        self.assertAlmostEqual(floor.threshold_pct, floor.spread_pct)

    def test_a_single_replicate_is_reported_as_NOT_measured(self):
        """A floor that was assumed must never be presentable as observed."""
        floor = NoiseFloor.from_replicates([12.0])
        self.assertFalse(floor.measured)
        self.assertIn("NOT measured", floor.source)
        self.assertEqual(floor.threshold_pct, NoiseFloor.MIN_THRESHOLD_PCT)

    def test_no_replicate_at_all_is_also_not_measured(self):
        floor = NoiseFloor.from_replicates([])
        self.assertFalse(floor.measured)
        self.assertIn("NOT measured", floor.source)

    def test_a_noisy_rig_refuses_what_a_quiet_rig_accepts(self):
        """The same candidate must be judged differently on different machines.

        This is the whole point of R1: a fixed 1% promoted a noise winner on a
        rig whose spread was 8%.
        """
        quiet = NoiseFloor.from_replicates([10.0, 10.05, 10.02])
        noisy = NoiseFloor.from_replicates([9.0, 10.0, 11.0])
        candidate = 10.4                       # ~+3.5% over the quiet best
        self.assertTrue(quiet.accepts(candidate))
        self.assertFalse(noisy.accepts(candidate))

    def test_improvement_is_reported_against_the_median_not_the_first_run(self):
        """Dividing by whichever replicate ran first reported +3.59% for nothing."""
        floor = NoiseFloor.from_replicates([8.0, 10.0, 10.1])
        self.assertAlmostEqual(floor.improvement_pct(10.0), 0.0, places=6)
        self.assertLess(floor.improvement_pct(10.0), 25.0)

    def test_beats_uses_the_reference_it_is_given(self):
        floor = NoiseFloor.from_replicates([10.0, 10.2, 10.1])
        self.assertFalse(floor.beats(12.0, 12.0))
        self.assertTrue(floor.beats(14.0, 12.0))


class ComparabilityTests(unittest.TestCase):
    """R3 -- a faster arm that did less work is not an optimization."""

    def test_the_swap_disabled_speedup_is_rejected(self):
        baseline = Measurement.from_mapping(
            {"fps": 12.9, "frames": 600, "faces_seen": 894, "faces_swapped": 819})
        # The real regression: throughput rose 47% because the swap stopped
        # running. Return code 0, audit 100%, 1575 tests green.
        broken = Measurement.from_mapping(
            {"fps": 19.0, "frames": 600, "faces_seen": 5, "faces_swapped": 3})
        comparable, reason = broken.comparable_to(baseline)
        self.assertFalse(comparable)
        self.assertIn("faces_seen", reason)

    def test_equal_work_is_comparable(self):
        baseline = Measurement.from_mapping(
            {"fps": 12.0, "frames": 600, "faces_seen": 853, "faces_swapped": 847})
        candidate = Measurement.from_mapping(
            {"fps": 13.5, "frames": 600, "faces_seen": 853, "faces_swapped": 847})
        comparable, reason = candidate.comparable_to(baseline)
        self.assertTrue(comparable, reason)

    def test_a_different_window_is_not_comparable(self):
        baseline = Measurement.from_mapping({"fps": 12.0, "frames": 600})
        candidate = Measurement.from_mapping({"fps": 20.0, "frames": 120})
        comparable, reason = candidate.comparable_to(baseline)
        self.assertFalse(comparable)
        self.assertIn("window", reason)

    def test_a_failed_arm_is_never_comparable(self):
        baseline = Measurement.from_mapping({"fps": 12.0, "frames": 600})
        failed = Measurement.from_mapping({"error": "boom", "stable": False})
        self.assertFalse(failed.comparable_to(baseline)[0])

    def test_short_arms_are_marked_provisional(self):
        self.assertTrue(Measurement.from_mapping({"frames": 120}).provisional)
        self.assertFalse(
            Measurement.from_mapping({"frames": ACCEPTANCE_FRAMES}).provisional)

    def test_an_unverifiable_comparison_says_so_instead_of_passing_silently(self):
        """A guard that cannot see the work it guards must not read as clean."""
        baseline = Measurement.from_mapping({"fps": 12.0, "frames": 600})
        candidate = Measurement.from_mapping({"fps": 19.0, "frames": 600})
        comparable, reason = candidate.comparable_to(baseline)
        self.assertTrue(comparable)
        self.assertIn("NOT verified", reason)
        self.assertFalse(candidate.work_verified)

    def test_a_verified_comparison_carries_no_caveat(self):
        baseline = Measurement.from_mapping(
            {"fps": 12.0, "frames": 600, "faces_seen": 100, "faces_swapped": 99})
        candidate = Measurement.from_mapping(
            {"fps": 13.0, "frames": 600, "faces_seen": 100, "faces_swapped": 99})
        comparable, reason = candidate.comparable_to(baseline)
        self.assertTrue(comparable)
        self.assertEqual(reason, "")
        self.assertTrue(candidate.work_verified)

    def test_unsampled_utilization_stays_None_and_is_not_zero(self):
        """None (never sampled) and 0.0 (measured idle) must not collapse."""
        self.assertIsNone(Measurement.from_mapping({}).gpu_utilization_pct)
        self.assertEqual(
            Measurement.from_mapping({"gpu_utilization_pct": 0.0}).gpu_utilization_pct,
            0.0)


class SwapLogAdapterTests(unittest.TestCase):
    """The R3 guard is only live if the callback feeds it real counts."""

    def test_counts_intent_from_the_pipeline_audit(self):
        log = {
            0: [{"swapped": True}, {"swapped": False}],
            1: [{"swapped": True}, {"swapped": True}],
        }
        self.assertEqual(swap_log_counts(log),
                         {"faces_seen": 4, "faces_swapped": 3})

    def test_an_empty_or_absent_log_reports_zero_rather_than_guessing(self):
        for value in (None, {}, []):
            self.assertEqual(swap_log_counts(value),
                             {"faces_seen": 0, "faces_swapped": 0})

    def test_a_sequence_of_frames_works_as_well_as_a_mapping(self):
        self.assertEqual(swap_log_counts([[{"swapped": True}], [{"swapped": False}]]),
                         {"faces_seen": 2, "faces_swapped": 1})


class CounterbalanceTests(unittest.TestCase):
    """R2 -- forward-only ordering is a wrong answer, not a shortcut."""

    def test_pairs_are_ABBA(self):
        self.assertEqual(counterbalanced_pairs("a", "b"), ["a", "b", "b", "a"])

    def test_every_level_appears_in_both_halves_of_a_sweep(self):
        order = counterbalanced_sweep([2, 4, 8])
        self.assertEqual(order, [2, 4, 8, 8, 4, 2])
        first, second = order[:3], order[3:]
        for level in (2, 4, 8):
            self.assertIn(level, first)
            self.assertIn(level, second)

    def test_position_bias_is_reported_when_the_second_arm_always_wins(self):
        # Every pair's second arm is faster regardless of treatment: the
        # sequence measured order, not the change.
        results = [Measurement(fps=fps, label=label) for fps, label in
                   [(10.0, "a"), (11.0, "b"), (10.0, "b"), (11.0, "a")]]
        summary = order_corrected(results, lambda m: m.label, "a", "b")
        self.assertTrue(summary["position_bias_suspected"])
        self.assertEqual(summary["position_bias_pairs"], "2 of 2")
        # And with the order corrected the two arms are equal.
        self.assertAlmostEqual(summary["delta_pct"], 0.0)


class BottleneckTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = BottleneckAnalyzer()
        self.hardware = hardware()

    def test_gpu_compute_bound(self):
        verdict = self.analyzer.classify(Measurement.from_mapping(
            {"gpu_utilization_pct": 95.0, "cpu_utilization_pct": 40.0,
             "peak_vram_gb": 6.0}), self.hardware)
        self.assertEqual(verdict.kind, "GPU compute bound")
        self.assertEqual(verdict.confidence, "high")

    def test_vram_bound_wins_over_gpu_utilization(self):
        """A thrashing card reports HIGH utilization; utilization must not win.

        Measured: a 6-context arm reached 94.5% GPU utilization at 1.51 fps
        while paging. Classifying that as GPU-compute-bound would recommend
        exactly the wrong action.
        """
        verdict = self.analyzer.classify(Measurement.from_mapping(
            {"gpu_utilization_pct": 99.0, "cpu_utilization_pct": 20.0,
             "peak_vram_gb": 11.4, "frame_time_p99_ms": 900.0,
             "frame_time_median_ms": 80.0}), self.hardware)
        self.assertEqual(verdict.kind, "GPU VRAM bound")
        self.assertTrue(any("hitching" in item for item in verdict.evidence))
        self.assertIn("THRASH", verdict.recommendation)

    def test_cpu_bound_on_a_single_saturated_core(self):
        verdict = self.analyzer.classify(Measurement.from_mapping(
            {"gpu_utilization_pct": 45.0, "cpu_utilization_pct": 30.0,
             "per_core_peak_pct": 100.0, "peak_vram_gb": 5.0}), self.hardware)
        self.assertEqual(verdict.kind, "CPU bound")

    def test_cpu_bound_requires_the_gpu_to_be_idle(self):
        verdict = self.analyzer.classify(Measurement.from_mapping(
            {"gpu_utilization_pct": 95.0, "cpu_utilization_pct": 30.0,
             "per_core_peak_pct": 100.0, "peak_vram_gb": 5.0}), self.hardware)
        self.assertNotEqual(verdict.kind, "CPU bound")

    def test_disk_io_bound(self):
        verdict = self.analyzer.classify(
            Measurement.from_mapping(
                {"gpu_utilization_pct": 30.0, "cpu_utilization_pct": 25.0,
                 "peak_vram_gb": 5.0, "disk_wait_pct": 35.0,
                 "disk_write_mb_s": 180.0}),
            self.hardware, {"write_mb_s": 200.0, "class": "sata-ssd"})
        self.assertEqual(verdict.kind, "Disk I/O bound")
        self.assertTrue(any("ceiling" in item for item in verdict.evidence))

    def test_absent_telemetry_returns_unknown_not_a_confident_verdict(self):
        """The defect this replaces: both GPUs reported 'I/O-bound' on runs
        whose decode cost 3.3 ms of a 244.8 ms frame, because absent queue
        depths look exactly like idle ones."""
        verdict = self.analyzer.classify(Measurement.from_mapping(
            {"fps": 5.0, "stage_seconds": {"frame_total": 100.0, "detect": 42.4}}),
            self.hardware)
        self.assertEqual(verdict.kind, "unknown")
        self.assertEqual(verdict.confidence, "none")
        self.assertTrue(verdict.missing_signals)

    def test_an_empty_stage_table_does_not_produce_a_verdict(self):
        verdict = self.analyzer.classify(Measurement(), self.hardware)
        self.assertEqual(verdict.kind, "unknown")

    def test_stage_share_is_never_turned_into_a_projected_speedup(self):
        """R4: detect at 42.4% predicted ~10%; it measured +1%."""
        verdict = self.analyzer.classify(Measurement.from_mapping(
            {"gpu_utilization_pct": 28.0, "cpu_utilization_pct": 17.0,
             "per_core_peak_pct": 40.0, "peak_vram_gb": 6.0,
             "stage_seconds": {"detect": 42.4, "swap": 20.0,
                               "frame_total": 100.0}}), self.hardware)
        self.assertEqual(verdict.dominant_stage, "detect")
        blob = " ".join(verdict.evidence) + verdict.recommendation
        for forbidden in ("42.4% faster", "speedup of", "would save 42"):
            self.assertNotIn(forbidden, blob)

    def test_frame_total_never_wins_the_dominant_stage(self):
        measurement = Measurement.from_mapping(
            {"stage_seconds": {"frame_total": 999.0, "mask": 50.0}})
        self.assertEqual(BottleneckAnalyzer.dominant_stage(measurement), "mask")


class SearchSpaceTests(unittest.TestCase):
    def test_models_are_not_in_the_search_space(self):
        """Swapping the model changes the application, not its configuration."""
        space = SearchSpace(hardware()).as_dict()
        self.assertTrue(space["models_held_constant"])
        names = {axis["name"] for axis in space["ort_axes"] + space["memory_axes"]}
        for model_axis in ("swap_model", "selected_enhancer", "mask_engine",
                           "face_detector"):
            self.assertNotIn(model_axis, names)

    def test_thread_levels_never_exceed_the_logical_core_count(self):
        for logical in (4, 8, 20, 32):
            levels = SearchSpace(hardware(logical=logical)).thread_levels()
            self.assertTrue(levels)
            self.assertLessEqual(max(levels), logical)
            self.assertEqual(list(levels), sorted(set(levels)))

    def test_the_full_logical_width_is_always_measured(self):
        """The flat tail must be observed, not assumed."""
        self.assertIn(20, SearchSpace(hardware(logical=20)).thread_levels())

    def test_batch_axis_is_unreachable_below_two_faces_per_frame(self):
        space = SearchSpace(hardware(),
                            WorkloadProfile(faces_per_frame=1.0))
        axis = [a for a in space.memory_axes() if a.name == "batch_size"][0]
        self.assertFalse(axis.reachable)
        self.assertIn("nothing to fill it", axis.unreachable_reason)

    def test_trt_axis_is_unreachable_without_tensorrt(self):
        space = SearchSpace(hardware(tensorrt=False))
        axis = [a for a in space.memory_axes()
                if a.name == "trt_context_count"][0]
        self.assertFalse(axis.reachable)

    def test_a_small_card_is_offered_single_context_only(self):
        """pool 8 on a 12GB card measured 2-2.5 fps against pool 2's 45.3."""
        space = SearchSpace(hardware(vram=6.0))
        axis = [a for a in space.memory_axes()
                if a.name == "trt_context_count"][0]
        self.assertEqual(axis.values, (1,))


class FrameFormatTests(unittest.TestCase):
    COST = {
        "width": 1920, "height": 1080,
        "rows": [
            {"format": "png", "encode_ms": 20.0, "decode_ms": 5.0,
             "size_mb": 3.0, "lossless": True},
            {"format": "jpg", "encode_ms": 4.0, "decode_ms": 3.0,
             "size_mb": 0.5, "lossless": False},
        ],
    }

    def test_lossy_is_excluded_by_default(self):
        """The temp frames ARE the encoder's input: create_video reads them."""
        result = frame_format_recommendation(self.COST, 1500.0)
        self.assertEqual(result["choice"], "png")
        self.assertFalse(result["allow_lossy"])

    def test_lossy_requires_an_explicit_opt_in(self):
        result = frame_format_recommendation(self.COST, 1500.0, allow_lossy=True)
        self.assertEqual(result["choice"], "jpg")

    def test_a_slow_volume_shifts_the_answer_toward_the_smaller_file(self):
        """Which side wins is a property of the volume, not of the format."""
        cost = {"width": 1920, "height": 1080, "rows": [
            {"format": "a", "encode_ms": 1.0, "size_mb": 3.0, "lossless": True},
            {"format": "b", "encode_ms": 8.0, "size_mb": 0.3, "lossless": True}]}
        self.assertEqual(
            frame_format_recommendation(cost, 5000.0)["choice"], "a")
        self.assertEqual(
            frame_format_recommendation(cost, 50.0)["choice"], "b")

    def test_a_forced_choice_is_not_described_as_a_won_comparison(self):
        result = frame_format_recommendation(self.COST, 1500.0)
        self.assertEqual(result["candidates_considered"], 1)
        self.assertIn("only admissible", result["reason"])

    def test_no_measurable_format_returns_no_choice(self):
        result = frame_format_recommendation({"rows": []}, 1500.0)
        self.assertIsNone(result["choice"])


class TempPathReachabilityTests(unittest.TestCase):
    """R5 -- do not sweep a code path a real render never takes."""

    def test_the_default_zero_disk_path_writes_no_temp_frame(self):
        triggers = temp_path_triggers({"keep_frames": False,
                                       "use_new_method": True})
        self.assertFalse(any(triggers.values()))

    def test_each_trigger_makes_the_path_reachable(self):
        for setting in ({"keep_frames": True},
                        {"use_new_method": False},
                        {"has_per_frame_masks": True}):
            self.assertTrue(any(temp_path_triggers(setting).values()), setting)


class PresetTests(unittest.TestCase):
    # 16 threads is the argmax by 0.4% over 8, and sits at exactly the 80% bar
    # so BOTH ceilings admit it: the knee-vs-argmax property is then the only
    # thing separating Balanced from Max Throughput.
    CURVE = [(2, 4.0), (4, 8.0), (8, 11.8), (12, 12.0), (16, 12.05)]
    VRAM = {2: 5.0, 4: 5.2, 8: 5.6, 12: 9.0, 16: 9.6}

    def build(self):
        floor = NoiseFloor.from_replicates([12.0, 12.1, 11.9])
        return PresetBuilder().build(
            RuntimeTuning(worker_count=8, queue_depth=3, in_flight_frames=4),
            hardware(), self.CURVE, self.VRAM, floor)

    def test_balanced_takes_the_knee_not_the_argmax(self):
        """16 threads is 0.4% faster than 8; that must not cost 8 more workers."""
        presets = self.build()
        self.assertEqual(presets["balanced"].tuning["worker_count"], 8)
        self.assertEqual(presets["max_throughput"].tuning["worker_count"], 16)

    def test_balanced_respects_the_80_percent_vram_bar(self):
        presets = self.build()
        self.assertLessEqual(presets["balanced"].projected_vram_pct,
                             VRAM_BALANCED_CEILING * 100.0)

    def test_max_throughput_stays_out_of_the_paging_band(self):
        presets = self.build()
        self.assertLessEqual(presets["max_throughput"].projected_vram_pct,
                             VRAM_ADMISSION_CEILING * 100.0)

    def test_an_arm_above_the_admission_ceiling_is_never_max_throughput(self):
        floor = NoiseFloor.from_replicates([12.0, 12.1, 11.9])
        presets = PresetBuilder().build(
            RuntimeTuning(worker_count=8), hardware(),
            [(8, 11.0), (16, 30.0)], {8: 5.0, 16: 11.8}, floor)
        self.assertEqual(presets["max_throughput"].tuning["worker_count"], 8)

    def test_low_power_uses_fewer_workers_and_smaller_buffers(self):
        presets = self.build()
        low = presets["stable_low_power"]
        self.assertLess(low.tuning["worker_count"],
                        presets["balanced"].tuning["worker_count"])
        self.assertLessEqual(low.tuning["queue_depth"], 2)
        self.assertLessEqual(low.tuning["in_flight_frames"], 2)

    def test_low_power_states_what_the_quiet_costs(self):
        self.assertIn("costs", self.build()["stable_low_power"].rationale)

    def test_presets_are_provisional_when_the_floor_was_not_measured(self):
        presets = PresetBuilder().build(
            RuntimeTuning(worker_count=8), hardware(), self.CURVE, self.VRAM,
            NoiseFloor.from_replicates([12.0]))
        self.assertTrue(all(preset.provisional for preset in presets.values()))

    def test_every_preset_stays_inside_the_safe_bounds(self):
        for preset in self.build().values():
            self.assertGreaterEqual(preset.tuning["worker_count"], 1)
            self.assertLessEqual(preset.tuning["worker_count"], 32)


class GuidedOptimizerTests(unittest.TestCase):
    HARDWARE = hardware(logical=16)
    WORKLOAD = WorkloadProfile(input_width=1280, input_height=720,
                               faces_per_frame=1.0)

    def optimizer(self, measure, **kwargs):
        return GuidedOptimizer(
            measure, hardware=self.HARDWARE, workload=self.WORKLOAD,
            settings={"keep_frames": False, "use_new_method": True},
            baseline=RuntimeTuning(worker_count=4, trt_context_count=1),
            **kwargs)

    def test_a_measure_callback_is_required(self):
        """No synthetic fallback: an isolated probe must not pass for a render."""
        with self.assertRaises(ValueError):
            GuidedOptimizer(None, hardware=self.HARDWARE)

    def test_a_raising_callback_becomes_a_failed_arm_not_a_crash(self):
        def measure(config, frames):
            raise RuntimeError("engine build failed")
        result = self.optimizer(measure)._run({}, 600, "boom")
        self.assertFalse(result.stable)
        self.assertIn("engine build failed", result.error)

    def test_the_noise_floor_is_measured_before_any_candidate(self):
        seen = []

        def measure(config, frames):
            seen.append(dict(config))
            return {"fps": 10.0, "frames": frames, "faces_seen": 100,
                    "faces_swapped": 99}
        optimizer = self.optimizer(measure)
        optimizer.measure_noise_floor(600)
        self.assertGreaterEqual(len(seen), 2)
        # Every replicate is the UNCHANGED baseline.
        self.assertEqual({config["worker_count"] for config in seen}, {4})

    def test_phase_b_re_measures_its_reference_at_the_phase_a_thread_count(self):
        """Regression: comparing a memory axis against the phase-0 baseline
        credited that axis with Phase A's thread gain."""
        labels = []

        def measure(config, frames):
            labels.append(config.get("worker_count"))
            return {"fps": 10.0 + config.get("worker_count", 0) * 0.1,
                    "frames": frames, "faces_seen": 100, "faces_swapped": 99,
                    "peak_vram_gb": 5.0}
        optimizer = self.optimizer(measure)
        floor = NoiseFloor.from_replicates([10.4, 10.4, 10.4])
        report = optimizer.phase_b_vram(12, floor, frames=600)
        self.assertIn("reference_fps", report)
        # The reference ran at 12 workers, not at the baseline's 4.
        self.assertAlmostEqual(report["reference_fps"], 11.2, places=3)
        self.assertIn("Phase A thread count", report["note"])

    def test_phase_b_rejects_an_arm_over_the_vram_budget(self):
        def measure(config, frames):
            pool = config.get("trt_context_count", 1)
            return {"fps": 100.0 * pool, "frames": frames, "faces_seen": 100,
                    "faces_swapped": 99, "peak_vram_gb": 2.0 + 5.0 * pool}
        optimizer = self.optimizer(measure)
        floor = NoiseFloor.from_replicates([10.0, 10.1, 10.05])
        report = optimizer.phase_b_vram(4, floor, frames=600)
        over = [row for row in report["rows"] if not row["admitted"]]
        self.assertTrue(over)
        self.assertTrue(any("headroom budget" in row.get("rejected", "")
                            for row in over))
        # A hugely faster but over-budget arm is still refused.
        self.assertNotEqual(report["accepted"].get("trt_context_count"), 3)

    def test_phase_b_rejects_an_arm_that_did_less_work(self):
        def measure(config, frames):
            if config.get("trt_context_count", 1) > 1:
                return {"fps": 99.0, "frames": frames, "faces_seen": 5,
                        "faces_swapped": 3, "peak_vram_gb": 5.0}
            return {"fps": 10.0, "frames": frames, "faces_seen": 100,
                    "faces_swapped": 99, "peak_vram_gb": 5.0}
        optimizer = self.optimizer(measure)
        floor = NoiseFloor.from_replicates([10.0, 10.1, 10.05])
        report = optimizer.phase_b_vram(4, floor, frames=600)
        self.assertEqual(report["accepted"].get("trt_context_count"), 1)
        self.assertTrue(any("faces_seen" in row.get("rejected", "")
                            for row in report["rows"]))

    def test_phase_a_is_counterbalanced(self):
        order = []

        def measure(config, frames):
            order.append(config["worker_count"])
            return {"fps": 10.0, "frames": frames, "faces_seen": 100,
                    "faces_swapped": 99}
        optimizer = self.optimizer(measure)
        optimizer.phase_a_threads(NoiseFloor.from_replicates([10.0, 10.1]),
                                  frames=120, confirm_frames=0)
        self.assertEqual(order, list(reversed(order)))

    def test_a_flat_curve_is_reported_as_flat_rather_than_ranked(self):
        """0.7% across a 5x worker range must not be sold as an optimization."""
        def measure(config, frames):
            return {"fps": 12.4, "frames": frames, "faces_seen": 100,
                    "faces_swapped": 99}
        optimizer = self.optimizer(measure)
        floor = NoiseFloor.from_replicates([12.4, 12.35, 12.32])
        report = optimizer.phase_a_threads(floor, frames=120, confirm_frames=0)
        self.assertTrue(report["flat_within_noise"])
        self.assertIsNone(report["inflection"]["level"])
        self.assertIn("noise floor", report["verdict"])

    def test_the_inflection_is_found_where_contention_starts(self):
        def measure(config, frames):
            workers = config["worker_count"]
            fps = 2.0 * workers if workers <= 8 else 16.0 - (workers - 8) * 1.5
            return {"fps": fps, "frames": frames, "faces_seen": 100,
                    "faces_swapped": 99}
        optimizer = self.optimizer(measure)
        floor = NoiseFloor.from_replicates([16.0, 16.05, 15.95])
        report = optimizer.phase_a_threads(floor, frames=120, confirm_frames=0)
        self.assertEqual(report["inflection"]["level"], 8)
        self.assertFalse(report["flat_within_noise"])

    def test_phase_c_does_not_apply_a_format_the_render_never_writes(self):
        def measure(config, frames):
            return {"fps": 10.0, "frames": frames, "faces_seen": 100,
                    "faces_swapped": 99}
        report = self.optimizer(measure).phase_c_disk()
        self.assertFalse(report["reachable"])
        self.assertFalse(report["applies_now"])
        self.assertIn("zero-disk", report["unreachable_reason"])
        # It is still measured, so the answer is ready if a trigger is enabled.
        self.assertIn("volume", report)

    def test_a_full_run_produces_all_three_presets_and_a_verdict(self):
        def measure(config, frames):
            workers = config.get("worker_count", 1)
            return {"fps": min(12.0, 1.5 * workers), "frames": frames,
                    "faces_seen": 100, "faces_swapped": 99,
                    "peak_vram_gb": 5.0 + 0.05 * workers,
                    "gpu_utilization_pct": 95.0, "cpu_utilization_pct": 30.0}
        report = self.optimizer(measure).run(ranking_frames=120,
                                             acceptance_frames=600)
        self.assertEqual(set(report.presets),
                         {"max_throughput", "balanced", "stable_low_power"})
        self.assertEqual(report.bottleneck["kind"], "GPU compute bound")
        self.assertTrue(report.measurements)
        self.assertTrue(report.noise_floor["measured"])

    def test_a_run_without_face_counts_warns_that_the_guard_could_not_run(self):
        """Absence of the guard's input must not read as the guard passing."""
        def measure(config, frames):
            return {"fps": 10.0, "frames": frames, "gpu_utilization_pct": 95.0,
                    "cpu_utilization_pct": 30.0, "peak_vram_gb": 5.0}
        report = self.optimizer(measure).run(ranking_frames=120,
                                             acceptance_frames=600)
        self.assertTrue(any("comparability guard could not run" in warning
                            for warning in report.warnings), report.warnings)

    def test_a_run_with_face_counts_does_not_raise_that_warning(self):
        def measure(config, frames):
            return {"fps": 10.0, "frames": frames, "faces_seen": 100,
                    "faces_swapped": 99, "gpu_utilization_pct": 95.0,
                    "cpu_utilization_pct": 30.0, "peak_vram_gb": 5.0}
        report = self.optimizer(measure).run(ranking_frames=120,
                                             acceptance_frames=600)
        self.assertFalse(any("comparability guard could not run" in warning
                             for warning in report.warnings))

    def test_a_run_on_an_unmeasurable_rig_warns_instead_of_recommending(self):
        def measure(config, frames):
            return {"error": "no", "stable": False}
        report = self.optimizer(measure).run(ranking_frames=120,
                                             acceptance_frames=600)
        self.assertTrue(report.warnings)
        self.assertTrue(any("NOT measured" in warning
                            for warning in report.warnings))


if __name__ == "__main__":
    unittest.main()
