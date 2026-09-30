"""Tests for Stage 15 — Unified CPU/RAM/GPU Runtime Scheduler.

Covers:
1. All 8 monitored pipeline metrics:
   - GPU utilization (%)
   - GPU memory (VRAM allocated, free, total)
   - CPU utilization (%)
   - RAM (used, available, total)
   - Queue depth (current, max)
   - Stage latencies (8 pipeline stages)
   - Decoder state (OK, STARVING, BLOCKED, IDLE, BACKPRESSURE)
   - Encoder state (OK, STARVING, BLOCKED, IDLE, BACKPRESSURE)
2. All 8 dynamically regulated knobs:
   - Worker count
   - Queue depth
   - Inference batch size
   - Detector frequency
   - Restoration concurrency
   - Preprocessing workers
   - Decode workers
   - Encode workers
3. Hardware capacity tiers & safety limits:
   - RTX 4070 Desktop (12GB VRAM, 9.5GB safe cap, 2.5GB headroom, 16 batch, 20 workers)
   - RTX 3060 Laptop (6GB VRAM, 4.9GB safe cap, 1.2GB headroom, RSS < 2.5GB, 4 batch, 6 workers)
   - CPU fallback
4. Core scheduling rules & guards:
   - VRAM safety limit enforcement (urgent downward throttle, no delay)
   - System RAM cap / RSS cap guard
   - Queue backpressure guard (throttling decode workers)
   - Decoder starvation rescue (boosting decode workers and queue depth)
   - Encoder starvation rescue (boosting preprocessing and worker pool)
   - CPU starvation prevention (throttling worker pool when CPU > 92%)
5. Asymmetric hysteresis & anti-oscillation:
   - Immediate downscale on danger
   - Cooldown period and sustained multi-cycle headroom required for upward scaling
   - Cap enforcement (batch and worker ceiling)
6. Diagnostics ledger and human-readable decision explanations.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Ensure app path is available
ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app"
if str(APP_PATH) not in sys.path:
    sys.path.insert(0, str(APP_PATH))

from roop.unified_runtime_scheduler import (
    ComponentState,
    HardwareCapacity,
    PipelineStage,
    RuntimeTelemetry,
    SchedulerDecision,
    SchedulerKnobs,
    UnifiedRuntimeScheduler,
    detect_system_capacity,
)


class TestPipelineStagesAndTelemetry(unittest.TestCase):
    """Test stage definitions, component states, telemetry, and knobs dataclasses."""

    def test_eight_pipeline_stages_exist(self):
        expected_stages = {
            "DECODE",
            "PREPROCESS",
            "DETECT_TRACK",
            "SWAP",
            "RESTORE",
            "MASK",
            "COMPOSITE",
            "ENCODE",
        }
        actual_stages = {s.value for s in PipelineStage}
        self.assertEqual(actual_stages, expected_stages)

    def test_component_states_exist(self):
        expected_states = {"OK", "STARVING", "BLOCKED", "IDLE", "BACKPRESSURE"}
        actual_states = {s.value for s in ComponentState}
        self.assertEqual(actual_states, expected_states)

    def test_telemetry_captures_all_eight_metric_streams(self):
        latencies = {s: 4.5 for s in PipelineStage}
        telemetry = RuntimeTelemetry(
            timestamp=1000.0,
            frame_index=42,
            gpu_utilization_pct=88.5,
            gpu_vram_allocated_mb=6500.0,
            gpu_vram_free_mb=5500.0,
            gpu_vram_total_mb=12288.0,
            cpu_utilization_pct=45.2,
            ram_used_mb=5120.0,
            ram_available_mb=27000.0,
            ram_total_mb=32768.0,
            queue_depth_current=3,
            queue_depth_max=6,
            stage_latencies_ms=latencies,
            decoder_state=ComponentState.OK,
            encoder_state=ComponentState.OK,
        )
        d = telemetry.to_dict()
        self.assertEqual(d["frame_index"], 42)
        self.assertAlmostEqual(d["gpu_utilization_pct"], 88.5)
        self.assertAlmostEqual(d["gpu_vram_allocated_mb"], 6500.0)
        self.assertAlmostEqual(d["cpu_utilization_pct"], 45.2)
        self.assertAlmostEqual(d["ram_used_mb"], 5120.0)
        self.assertEqual(d["queue_depth_current"], 3)
        self.assertEqual(len(d["stage_latencies_ms"]), 8)
        self.assertEqual(d["decoder_state"], "OK")
        self.assertEqual(d["encoder_state"], "OK")

    def test_scheduler_knobs_regulates_all_eight_parameters(self):
        knobs = SchedulerKnobs(
            worker_count=8,
            queue_depth=5,
            inference_batch_size=4,
            detector_frequency=1,
            restoration_concurrency=2,
            preprocessing_workers=3,
            decode_workers=2,
            encode_workers=2,
        )
        d = knobs.to_dict()
        self.assertEqual(len(d), 8)
        self.assertEqual(d["worker_count"], 8)
        self.assertEqual(d["queue_depth"], 5)
        self.assertEqual(d["inference_batch_size"], 4)
        self.assertEqual(d["detector_frequency"], 1)
        self.assertEqual(d["restoration_concurrency"], 2)
        self.assertEqual(d["preprocessing_workers"], 3)
        self.assertEqual(d["decode_workers"], 2)
        self.assertEqual(d["encode_workers"], 2)


class TestHardwareCapacityProfiles(unittest.TestCase):
    """Test hardware capacity definitions and detection."""

    def test_rtx_4070_desktop_capacity(self):
        cap = HardwareCapacity.rtx_4070_desktop()
        self.assertEqual(cap.tier_name, "RTX_4070_DESKTOP")
        self.assertAlmostEqual(cap.total_vram_mb, 12288.0)
        self.assertAlmostEqual(cap.safe_vram_limit_mb, 9728.0)
        self.assertAlmostEqual(cap.vram_headroom_target_mb, 2560.0)
        self.assertEqual(cap.max_batch_cap, 16)
        self.assertEqual(cap.max_worker_cap, 20)

    def test_rtx_3060_laptop_capacity(self):
        cap = HardwareCapacity.rtx_3060_laptop()
        self.assertEqual(cap.tier_name, "RTX_3060_LAPTOP")
        self.assertAlmostEqual(cap.total_vram_mb, 6144.0)
        self.assertAlmostEqual(cap.safe_vram_limit_mb, 4915.0)
        self.assertAlmostEqual(cap.vram_headroom_target_mb, 1228.0)
        self.assertAlmostEqual(cap.safe_ram_limit_mb, 2560.0)  # RSS < 2.5 GB cap
        self.assertEqual(cap.max_batch_cap, 4)
        self.assertEqual(cap.max_worker_cap, 6)

    def test_cpu_fallback_capacity(self):
        cap = HardwareCapacity.cpu_fallback()
        self.assertEqual(cap.tier_name, "CPU_FALLBACK")
        self.assertEqual(cap.max_batch_cap, 1)
        self.assertEqual(cap.max_worker_cap, 4)

    def test_detect_system_capacity_runs_safely(self):
        cap = detect_system_capacity()
        self.assertIsInstance(cap, HardwareCapacity)
        self.assertTrue(cap.max_batch_cap >= 1)
        self.assertTrue(cap.max_worker_cap >= 1)


class TestSchedulerRuleEngine(unittest.TestCase):
    """Test closed-loop scheduling decisions under simulated workloads."""

    def setUp(self):
        self.cap_4070 = HardwareCapacity.rtx_4070_desktop()
        self.cap_3060 = HardwareCapacity.rtx_3060_laptop()

    def _make_telemetry(
        self,
        frame: int = 1,
        vram_alloc: float = 4000.0,
        ram_used: float = 2000.0,
        queue_depth: int = 2,
        queue_max: int = 4,
        cpu_util: float = 30.0,
        decoder_state: ComponentState = ComponentState.OK,
        encoder_state: ComponentState = ComponentState.OK,
    ) -> RuntimeTelemetry:
        latencies = {s: 5.0 for s in PipelineStage}
        return RuntimeTelemetry(
            timestamp=1000.0 + frame * 0.033,
            frame_index=frame,
            gpu_utilization_pct=60.0,
            gpu_vram_allocated_mb=vram_alloc,
            gpu_vram_free_mb=max(0.0, 12288.0 - vram_alloc),
            gpu_vram_total_mb=12288.0,
            cpu_utilization_pct=cpu_util,
            ram_used_mb=ram_used,
            ram_available_mb=10000.0,
            ram_total_mb=32768.0,
            queue_depth_current=queue_depth,
            queue_depth_max=queue_max,
            stage_latencies_ms=latencies,
            decoder_state=decoder_state,
            encoder_state=encoder_state,
        )

    def test_rule1_vram_pressure_throttles_immediately(self):
        scheduler = UnifiedRuntimeScheduler(
            capacity=self.cap_4070,
            initial_knobs=SchedulerKnobs(
                inference_batch_size=8, restoration_concurrency=2, queue_depth=6
            ),
        )
        # Exceed safe VRAM limit (9728 MB)
        telemetry = self._make_telemetry(frame=10, vram_alloc=10200.0)
        decision = scheduler.observe_and_schedule(telemetry)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.trigger_reason, "VRAM_PRESSURE_HIGH")
        # Halved batch size from 8 to 4
        self.assertEqual(decision.updated_knobs["inference_batch_size"], 4)
        self.assertEqual(decision.updated_knobs["restoration_concurrency"], 1)
        self.assertEqual(decision.updated_knobs["queue_depth"], 5)
        self.assertIn("VRAM exceeded safe threshold", decision.explanation)

    def test_rule2_ram_exhaustion_guard_on_3060_laptop(self):
        scheduler = UnifiedRuntimeScheduler(
            capacity=self.cap_3060,
            initial_knobs=SchedulerKnobs(worker_count=4, queue_depth=3),
        )
        # 3060 safe RAM limit is 2560 MB (strict RSS < 2.5 GB)
        telemetry = self._make_telemetry(frame=15, ram_used=2900.0)
        decision = scheduler.observe_and_schedule(telemetry)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.trigger_reason, "RAM_EXHAUSTION_GUARD")
        self.assertEqual(decision.updated_knobs["worker_count"], 3)
        self.assertEqual(decision.updated_knobs["queue_depth"], 2)
        self.assertIn("Host RAM", decision.explanation)

    def test_rule3_queue_backpressure_throttles_decode_workers(self):
        scheduler = UnifiedRuntimeScheduler(
            capacity=self.cap_4070,
            initial_knobs=SchedulerKnobs(decode_workers=2, queue_depth=4),
        )
        # Queue saturated: current >= max
        telemetry = self._make_telemetry(frame=20, queue_depth=4, queue_max=4)
        decision = scheduler.observe_and_schedule(telemetry)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.trigger_reason, "QUEUE_BACKPRESSURE_ACTIVE")
        self.assertEqual(decision.updated_knobs["decode_workers"], 1)
        self.assertIn("Queue depth saturated", decision.explanation)

    def test_rule4_decoder_starvation_rescue(self):
        scheduler = UnifiedRuntimeScheduler(
            capacity=self.cap_4070,
            initial_knobs=SchedulerKnobs(decode_workers=1, queue_depth=3),
        )
        # Queue empty, decoder starving
        telemetry = self._make_telemetry(
            frame=25,
            queue_depth=0,
            decoder_state=ComponentState.STARVING,
            encoder_state=ComponentState.OK,
        )
        decision = scheduler.observe_and_schedule(telemetry)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.trigger_reason, "DECODER_STARVATION_RESCUE")
        self.assertEqual(decision.updated_knobs["decode_workers"], 2)
        self.assertEqual(decision.updated_knobs["queue_depth"], 4)
        self.assertIn("Decoder starvation observed", decision.explanation)

    def test_rule5_encoder_starvation_rescue(self):
        scheduler = UnifiedRuntimeScheduler(
            capacity=self.cap_4070,
            initial_knobs=SchedulerKnobs(worker_count=8, preprocessing_workers=2),
        )
        # Encoder starving for frames
        telemetry = self._make_telemetry(
            frame=30,
            queue_depth=2,
            encoder_state=ComponentState.STARVING,
        )
        decision = scheduler.observe_and_schedule(telemetry)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.trigger_reason, "ENCODER_STARVATION_RESCUE")
        self.assertEqual(decision.updated_knobs["preprocessing_workers"], 3)
        self.assertEqual(decision.updated_knobs["worker_count"], 9)
        self.assertIn("Encoder starving", decision.explanation)

    def test_rule6_cpu_starvation_prevention(self):
        scheduler = UnifiedRuntimeScheduler(
            capacity=self.cap_4070,
            initial_knobs=SchedulerKnobs(worker_count=12, preprocessing_workers=4),
            ewma_alpha=1.0,  # Fast EWMA for testing
        )
        # CPU pegged at 96%
        telemetry = self._make_telemetry(frame=35, cpu_util=96.0)
        decision = scheduler.observe_and_schedule(telemetry)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.trigger_reason, "CPU_STARVATION_PREVENTION")
        self.assertEqual(decision.updated_knobs["worker_count"], 10)
        self.assertEqual(decision.updated_knobs["preprocessing_workers"], 3)
        self.assertIn("High CPU utilization", decision.explanation)


class TestHysteresisAndAntiOscillation(unittest.TestCase):
    """Test hysteresis logic, cooldown suppression, and multi-cycle sustained headroom."""

    def setUp(self):
        self.cap_4070 = HardwareCapacity.rtx_4070_desktop()

    def _make_safe_telemetry(self, frame: int) -> RuntimeTelemetry:
        latencies = {s: 3.0 for s in PipelineStage}
        return RuntimeTelemetry(
            timestamp=1000.0 + frame * 0.033,
            frame_index=frame,
            gpu_utilization_pct=40.0,
            gpu_vram_allocated_mb=3000.0,  # Well below 75% of 9728 MB safe cap
            gpu_vram_free_mb=9288.0,
            gpu_vram_total_mb=12288.0,
            cpu_utilization_pct=25.0,
            # 1800 MB is below both the 3060 RSS cap (2560 MB) and the 4070
            # safe RAM limit (24576 MB) so this helper is safe for all tier tests.
            ram_used_mb=1800.0,
            ram_available_mb=20000.0,
            ram_total_mb=32768.0,
            queue_depth_current=2,
            queue_depth_max=6,
            stage_latencies_ms=latencies,
            decoder_state=ComponentState.OK,
            encoder_state=ComponentState.OK,
        )

    def test_hysteresis_prevents_immediate_expansion_without_sustained_cycles(self):
        scheduler = UnifiedRuntimeScheduler(
            capacity=self.cap_4070,
            initial_knobs=SchedulerKnobs(inference_batch_size=4, worker_count=8),
            hysteresis_cooldown_frames=30,
        )
        # Cycle 1: Frame 50 (cooldown passed from initial -30) -> first headroom observation
        t1 = self._make_safe_telemetry(frame=50)
        d1 = scheduler.observe_and_schedule(t1)
        # Should NOT expand yet on cycle 1 (requires at least 2 consecutive cycles)
        self.assertIsNone(d1)
        self.assertEqual(scheduler.get_current_knobs().inference_batch_size, 4)

        # Cycle 2: Frame 51 -> second consecutive headroom observation
        t2 = self._make_safe_telemetry(frame=51)
        d2 = scheduler.observe_and_schedule(t2)
        # Now sustained headroom is proven!
        self.assertIsNotNone(d2)
        self.assertEqual(d2.trigger_reason, "STABLE_HEADROOM_EXPANSION")
        self.assertEqual(d2.updated_knobs["inference_batch_size"], 8)
        self.assertEqual(d2.updated_knobs["worker_count"], 10)

    def test_cooldown_suppresses_rapid_re_expansion(self):
        scheduler = UnifiedRuntimeScheduler(
            capacity=self.cap_4070,
            initial_knobs=SchedulerKnobs(inference_batch_size=4, worker_count=8),
            hysteresis_cooldown_frames=30,
        )
        # Expand at frame 10 (using force_evaluation to simulate an expansion)
        t = self._make_safe_telemetry(frame=10)
        d = scheduler.observe_and_schedule(t, force_evaluation=True)
        self.assertIsNotNone(d)
        self.assertEqual(d.frame_index, 10)

        # Frame 15: Only 5 frames since last decision (cooldown is 30)
        # Even with safe headroom, should NOT expand
        for f in range(11, 20):
            t_soon = self._make_safe_telemetry(frame=f)
            d_soon = scheduler.observe_and_schedule(t_soon)
            self.assertIsNone(d_soon)

    def test_expansion_respects_hardware_caps(self):
        # Cap batch at 4 and workers at 6 for 3060
        cap_3060 = HardwareCapacity.rtx_3060_laptop()
        scheduler = UnifiedRuntimeScheduler(
            capacity=cap_3060,
            initial_knobs=SchedulerKnobs(inference_batch_size=4, worker_count=6),
            hysteresis_cooldown_frames=10,
        )
        # Even under ideal conditions, knobs should not exceed hardware caps
        t1 = self._make_safe_telemetry(frame=20)
        d = scheduler.observe_and_schedule(t1, force_evaluation=True)
        # Already at max caps (batch 4, workers 6) -> no change
        self.assertIsNone(d)
        knobs = scheduler.get_current_knobs()
        self.assertEqual(knobs.inference_batch_size, 4)
        self.assertEqual(knobs.worker_count, 6)


class TestDiagnosticsAndAuditLedger(unittest.TestCase):
    """Test diagnostics summary, audit ledger history, and live probe integration."""

    def test_decision_history_and_diagnostics_summary(self):
        scheduler = UnifiedRuntimeScheduler(
            capacity=HardwareCapacity.rtx_4070_desktop(),
            initial_knobs=SchedulerKnobs(worker_count=8, queue_depth=4),
        )
        # Trigger an urgent backpressure decision
        latencies = {s: 2.0 for s in PipelineStage}
        t = RuntimeTelemetry(
            timestamp=1000.0,
            frame_index=12,
            gpu_utilization_pct=50.0,
            gpu_vram_allocated_mb=3000.0,
            gpu_vram_free_mb=9288.0,
            gpu_vram_total_mb=12288.0,
            cpu_utilization_pct=20.0,
            ram_used_mb=2000.0,
            ram_available_mb=20000.0,
            ram_total_mb=32768.0,
            queue_depth_current=4,
            queue_depth_max=4,
            stage_latencies_ms=latencies,
            decoder_state=ComponentState.OK,
            encoder_state=ComponentState.OK,
        )
        decision = scheduler.observe_and_schedule(t)
        self.assertIsNotNone(decision)

        history = scheduler.get_decision_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].trigger_reason, "QUEUE_BACKPRESSURE_ACTIVE")

        diag = scheduler.get_diagnostics_summary()
        self.assertEqual(diag["total_decisions_made"], 1)
        self.assertEqual(diag["last_decision_frame"], 12)
        self.assertEqual(diag["active_knobs"]["decode_workers"], 0 if decision.updated_knobs["decode_workers"] == 0 else 1)
        self.assertIn("smoothed_stage_latencies_ms", diag)

    def test_probe_live_telemetry_constructs_valid_telemetry(self):
        scheduler = UnifiedRuntimeScheduler()
        telemetry = scheduler.probe_live_telemetry(
            frame_index=1,
            queue_depth=2,
            stage_latencies={PipelineStage.SWAP: 12.5},
            decoder_state=ComponentState.OK,
            encoder_state=ComponentState.OK,
        )
        self.assertIsInstance(telemetry, RuntimeTelemetry)
        self.assertEqual(telemetry.frame_index, 1)
        self.assertEqual(telemetry.queue_depth_current, 2)
        self.assertEqual(telemetry.stage_latencies_ms[PipelineStage.SWAP], 12.5)
        self.assertTrue(telemetry.ram_total_mb > 0)


if __name__ == "__main__":
    unittest.main()
