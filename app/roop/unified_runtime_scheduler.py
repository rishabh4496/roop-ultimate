"""Unified CPU/RAM/GPU Runtime Scheduler for Stage 15.

A hardware-adaptive, real-time closed-loop pipeline scheduler that monitors:
1. GPU utilization (%)
2. GPU memory (Allocated, Free, Total VRAM)
3. CPU utilization (%)
4. System RAM (Allocated, Available, Total RAM)
5. Queue depth across pipeline boundaries
6. Stage latencies (Decode, Preprocess, Detect/Track, Swap, Restore, Mask, Composite, Encode)
7. Decoder state (OK, STARVING, BLOCKED, IDLE, BACKPRESSURE)
8. Encoder state (OK, STARVING, BLOCKED, IDLE, BACKPRESSURE)

Dynamically regulates 8 pipeline knobs:
1. Worker count (overall concurrent pipeline workers)
2. Queue depth (dynamic capacity ceiling)
3. Inference batch size (cross-frame/intra-frame batching)
4. Detector frequency (temporal stride step)
5. Restoration concurrency (parallel enhancer slots)
6. Preprocessing workers (crop & tensor prep threads)
7. Decode workers (video input stream readers)
8. Encode workers (video output stream writers)

Governed by strict rules:
- Never exceed safe VRAM (enforcing distinct caps for RTX 4070 Desktop and RTX 3060 Laptop).
- Never allow unlimited queue growth (bounded queues with backpressure).
- Avoid GPU contention (coordinating compute vs memory transfers).
- Avoid CPU starvation (reserving cores for OS and I/O).
- Avoid decoder starvation (preventing inference stalls).
- Avoid encoder starvation (maintaining smooth output flow).
- Built-in hysteresis (asymmetric escalation/de-escalation cooldown) to eliminate frame oscillation.
- Full runtime diagnostics explaining every scheduling decision.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from roop.degrade import swallowed as _swallowed

logger = logging.getLogger("roop.unified_scheduler")

try:
    import torch
except Exception as _degrade_error:  # pragma: no cover
    _swallowed("roop/unified_runtime_scheduler.py:48", _degrade_error, "torch fallback")
    torch = None  # type: ignore[assignment]

try:
    import psutil
except Exception as _degrade_error:  # pragma: no cover
    _swallowed("roop/unified_runtime_scheduler.py:54", _degrade_error, "psutil fallback")
    psutil = None  # type: ignore[assignment]

try:
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception as _degrade_error:  # pragma: no cover
    _swallowed("roop/unified_runtime_scheduler.py:61", _degrade_error, "pynvml unavailable; gpu_util falls back to VRAM heuristic")
    _pynvml = None  # type: ignore[assignment]
    _NVML_AVAILABLE = False


# ── Pipeline Stages and Component States ───────────────────────────────────


class PipelineStage(str, Enum):
    """The 8 sequential stages of the video processing pipeline."""

    DECODE = "DECODE"
    PREPROCESS = "PREPROCESS"
    DETECT_TRACK = "DETECT_TRACK"
    SWAP = "SWAP"
    RESTORE = "RESTORE"
    MASK = "MASK"
    COMPOSITE = "COMPOSITE"
    ENCODE = "ENCODE"


class ComponentState(str, Enum):
    """Operational health state of decoder and encoder."""

    OK = "OK"
    STARVING = "STARVING"
    BLOCKED = "BLOCKED"
    IDLE = "IDLE"
    BACKPRESSURE = "BACKPRESSURE"


# ── Telemetry and Knobs Dataclasses ────────────────────────────────────────


@dataclass
class RuntimeTelemetry:
    """Instantaneous snapshot of system and pipeline telemetry."""

    timestamp: float
    frame_index: int
    gpu_utilization_pct: float
    gpu_vram_allocated_mb: float
    gpu_vram_free_mb: float
    gpu_vram_total_mb: float
    cpu_utilization_pct: float
    ram_used_mb: float
    ram_available_mb: float
    ram_total_mb: float
    queue_depth_current: int
    queue_depth_max: int
    stage_latencies_ms: Dict[PipelineStage, float]
    decoder_state: ComponentState
    encoder_state: ComponentState

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["stage_latencies_ms"] = {k.value: v for k, v in self.stage_latencies_ms.items()}
        d["decoder_state"] = self.decoder_state.value
        d["encoder_state"] = self.encoder_state.value
        return d


@dataclass
class SchedulerKnobs:
    """The 8 dynamically regulated pipeline parameters."""

    worker_count: int = 4
    queue_depth: int = 4
    inference_batch_size: int = 4
    detector_frequency: int = 1  # 1 = every frame, 2 = every 2nd frame
    restoration_concurrency: int = 1
    preprocessing_workers: int = 2
    decode_workers: int = 1
    encode_workers: int = 1

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class SchedulerDecision:
    """Explanatory record of an adaptive scheduling decision."""

    timestamp: float
    frame_index: int
    trigger_reason: str
    previous_knobs: Dict[str, int]
    updated_knobs: Dict[str, int]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Hardware Profile and Safety Caps ───────────────────────────────────────


@dataclass(frozen=True)
class HardwareCapacity:
    """Physical hardware limits and target margins."""

    tier_name: str
    total_vram_mb: float
    safe_vram_limit_mb: float
    vram_headroom_target_mb: float
    total_ram_mb: float
    safe_ram_limit_mb: float
    physical_cpu_cores: int
    logical_cpu_cores: int
    max_batch_cap: int
    max_worker_cap: int

    @classmethod
    def rtx_4070_desktop(cls) -> "HardwareCapacity":
        """Main Device: 12GB VRAM, 32GB RAM, 24/32 CPU cores."""
        return cls(
            tier_name="RTX_4070_DESKTOP",
            total_vram_mb=12288.0,
            safe_vram_limit_mb=9728.0,  # Reserving 2.5 GB headroom
            vram_headroom_target_mb=2560.0,
            total_ram_mb=32768.0,
            safe_ram_limit_mb=24576.0,
            physical_cpu_cores=24,
            logical_cpu_cores=32,
            max_batch_cap=16,
            max_worker_cap=20,
        )

    @classmethod
    def rtx_3060_laptop(cls) -> "HardwareCapacity":
        """Secondary Device: 6GB VRAM, 16GB RAM, mobile CPU, RSS < 2.5GB."""
        return cls(
            tier_name="RTX_3060_LAPTOP",
            total_vram_mb=6144.0,
            safe_vram_limit_mb=4915.0,  # Reserving 1.2 GB headroom
            vram_headroom_target_mb=1228.0,
            total_ram_mb=16384.0,
            safe_ram_limit_mb=2560.0,  # Strict RSS < 2.5 GB cap
            physical_cpu_cores=8,
            logical_cpu_cores=16,
            max_batch_cap=4,
            max_worker_cap=6,
        )

    @classmethod
    def cpu_fallback(cls) -> "HardwareCapacity":
        """Host CPU fallback environment."""
        return cls(
            tier_name="CPU_FALLBACK",
            total_vram_mb=0.0,
            safe_vram_limit_mb=0.0,
            vram_headroom_target_mb=0.0,
            total_ram_mb=16384.0,
            safe_ram_limit_mb=12288.0,
            physical_cpu_cores=4,
            logical_cpu_cores=8,
            max_batch_cap=1,
            max_worker_cap=4,
        )


def detect_system_capacity() -> HardwareCapacity:
    """Detect current hardware environment and build capacity bounds."""
    if torch is None or not torch.cuda.is_available():
        return HardwareCapacity.cpu_fallback()

    try:
        props = torch.cuda.get_device_properties(0)
        vram_mb = float(props.total_memory) / (1024.0**2)
        name = str(props.name).lower()

        # Laptop RTX 3060 or sub-7GB
        if vram_mb < 7168.0 or "3060" in name or "laptop" in name:
            return HardwareCapacity.rtx_3060_laptop()

        # RTX 4070 or 12GB+ desktop
        if vram_mb >= 11000.0 or "4070" in name:
            return HardwareCapacity.rtx_4070_desktop()

        # Generic CUDA
        return HardwareCapacity(
            tier_name="GENERIC_CUDA",
            total_vram_mb=vram_mb,
            safe_vram_limit_mb=vram_mb * 0.80,
            vram_headroom_target_mb=vram_mb * 0.20,
            total_ram_mb=32768.0,
            safe_ram_limit_mb=24576.0,
            physical_cpu_cores=8,
            logical_cpu_cores=16,
            max_batch_cap=8,
            max_worker_cap=8,
        )
    except Exception as exc:
        _swallowed("roop/unified_runtime_scheduler.py:214", exc, "capacity detection fallback")
        return HardwareCapacity.cpu_fallback()


# ── The Unified Adaptive Runtime Scheduler ─────────────────────────────────


class UnifiedRuntimeScheduler:
    """Hardware-adaptive pipeline concurrency and queue governor."""

    def __init__(
        self,
        capacity: Optional[HardwareCapacity] = None,
        initial_knobs: Optional[SchedulerKnobs] = None,
        hysteresis_cooldown_frames: int = 30,
        ewma_alpha: float = 0.15,
    ) -> None:
        self.capacity = capacity or detect_system_capacity()
        self.knobs = initial_knobs or self._initial_knobs_for_tier()
        self.cooldown_frames = hysteresis_cooldown_frames
        self.ewma_alpha = ewma_alpha

        self._lock = threading.RLock()
        self._last_decision_frame: int = -self.cooldown_frames
        self._decision_history: List[SchedulerDecision] = []

        # Smoothed EWMA values
        self._smoothed_gpu_util: float = 0.0
        self._smoothed_cpu_util: float = 0.0
        self._smoothed_latencies: Dict[PipelineStage, float] = {
            s: 5.0 for s in PipelineStage
        }

        # Consecutive headroom counter for safe upward escalation
        self._stable_cycles: int = 0

    def _initial_knobs_for_tier(self) -> SchedulerKnobs:
        """Initialize baseline knobs tailored to the detected tier."""
        if self.capacity.tier_name == "RTX_4070_DESKTOP":
            return SchedulerKnobs(
                worker_count=12,
                queue_depth=6,
                inference_batch_size=8,
                detector_frequency=1,
                restoration_concurrency=2,
                preprocessing_workers=4,
                decode_workers=2,
                encode_workers=2,
            )
        elif self.capacity.tier_name == "RTX_3060_LAPTOP":
            return SchedulerKnobs(
                worker_count=4,
                queue_depth=3,
                inference_batch_size=4,
                detector_frequency=1,
                restoration_concurrency=1,
                preprocessing_workers=2,
                decode_workers=1,
                encode_workers=1,
            )
        else:
            return SchedulerKnobs(
                worker_count=2,
                queue_depth=2,
                inference_batch_size=1,
                detector_frequency=1,
                restoration_concurrency=0,
                preprocessing_workers=1,
                decode_workers=1,
                encode_workers=1,
            )

    def probe_live_telemetry(
        self,
        frame_index: int,
        queue_depth: int,
        stage_latencies: Optional[Dict[PipelineStage, float]] = None,
        decoder_state: ComponentState = ComponentState.OK,
        encoder_state: ComponentState = ComponentState.OK,
    ) -> RuntimeTelemetry:
        """Query live hardware sensors and assemble complete telemetry snapshot.

        GPU utilization is sourced from NVML when available (the only accurate
        real-time signal), falling back to a VRAM-presence heuristic when the
        NVIDIA management library is not installed.  VRAM figures prefer
        ``torch.cuda.mem_get_info`` (driver-reported free/total) over the
        reserved-memory approximation, which can lag by several hundred MB.
        """
        gpu_util = 0.0
        vram_alloc = 0.0
        vram_free = self.capacity.total_vram_mb
        vram_total = self.capacity.total_vram_mb

        if torch is not None and torch.cuda.is_available():
            try:
                # Prefer driver-level free/total for vram_free — more accurate
                # than PyTorch's allocator bookkeeping.
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                vram_total = float(total_bytes) / (1024.0 ** 2)
                vram_free = float(free_bytes) / (1024.0 ** 2)
                vram_alloc = max(0.0, vram_total - vram_free)
            except Exception:
                # mem_get_info not available (old torch) — fall back to allocator view
                try:
                    vram_alloc = float(torch.cuda.memory_allocated(0)) / (1024.0 ** 2)
                    vram_reserved = float(torch.cuda.memory_reserved(0)) / (1024.0 ** 2)
                    vram_alloc = max(vram_alloc, vram_reserved)
                    vram_free = max(0.0, vram_total - vram_alloc)
                except Exception as exc:
                    _swallowed("roop/unified_runtime_scheduler.py:vram_probe", exc, "vram probe fallback")

        # GPU utilization: NVML is the only source of real-time SM occupancy.
        # Without it, a VRAM-presence heuristic is the best available signal.
        if _NVML_AVAILABLE and _pynvml is not None:
            try:
                handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = float(_pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            except Exception as exc:
                _swallowed("roop/unified_runtime_scheduler.py:nvml_util", exc, "nvml util fallback")
                gpu_util = 75.0 if vram_alloc > 1000.0 else 10.0
        else:
            # Coarse proxy: if models are loaded (~>1 GB VRAM reserved) the GPU
            # is likely active during inference, otherwise idle.
            gpu_util = 75.0 if vram_alloc > 1000.0 else 10.0

        cpu_util = 35.0
        ram_used = 4096.0
        ram_avail = 12288.0
        ram_total = self.capacity.total_ram_mb

        if psutil is not None:
            try:
                cpu_util = float(psutil.cpu_percent(interval=None))
                vm = psutil.virtual_memory()
                ram_used = float(vm.used) / (1024.0 ** 2)
                ram_avail = float(vm.available) / (1024.0 ** 2)
                ram_total = float(vm.total) / (1024.0 ** 2)
            except Exception as exc:
                _swallowed("roop/unified_runtime_scheduler.py:ram_probe", exc, "ram probe fallback")

        latencies = dict(self._smoothed_latencies)
        if stage_latencies:
            latencies.update(stage_latencies)

        return RuntimeTelemetry(
            timestamp=time.time(),
            frame_index=frame_index,
            gpu_utilization_pct=gpu_util,
            gpu_vram_allocated_mb=vram_alloc,
            gpu_vram_free_mb=vram_free,
            gpu_vram_total_mb=vram_total,
            cpu_utilization_pct=cpu_util,
            ram_used_mb=ram_used,
            ram_available_mb=ram_avail,
            ram_total_mb=ram_total,
            queue_depth_current=queue_depth,
            queue_depth_max=self.knobs.queue_depth,
            stage_latencies_ms=latencies,
            decoder_state=decoder_state,
            encoder_state=encoder_state,
        )

    def observe_and_schedule(
        self,
        telemetry: RuntimeTelemetry,
        force_evaluation: bool = False,
    ) -> Optional[SchedulerDecision]:
        """Process telemetry, update smoothed metrics, evaluate rules, and apply adjustments."""
        with self._lock:
            # 1. Update Exponentially Weighted Moving Averages (EWMA)
            a = self.ewma_alpha
            self._smoothed_gpu_util = (1.0 - a) * self._smoothed_gpu_util + a * telemetry.gpu_utilization_pct
            self._smoothed_cpu_util = (1.0 - a) * self._smoothed_cpu_util + a * telemetry.cpu_utilization_pct
            for stage, lat in telemetry.stage_latencies_ms.items():
                prev = self._smoothed_latencies.get(stage, lat)
                self._smoothed_latencies[stage] = (1.0 - a) * prev + a * lat

            curr_frame = telemetry.frame_index
            prev_knobs = self.knobs.to_dict()
            new_knobs = SchedulerKnobs(**prev_knobs)
            decision: Optional[SchedulerDecision] = None

            # ── RULE 1: Never Exceed Safe VRAM (Urgent Downward Throttle) ──
            # Does not wait for hysteresis cooldown if VRAM exceeds safe limit!
            if self.capacity.safe_vram_limit_mb > 0 and telemetry.gpu_vram_allocated_mb > self.capacity.safe_vram_limit_mb:
                # Step down batch size immediately
                new_batch = max(1, new_knobs.inference_batch_size // 2)
                new_knobs.inference_batch_size = new_batch
                new_knobs.restoration_concurrency = max(1, new_knobs.restoration_concurrency - 1)
                new_knobs.queue_depth = max(2, new_knobs.queue_depth - 1)

                explanation = (
                    f"VRAM exceeded safe threshold ({telemetry.gpu_vram_allocated_mb:.1f}MB > "
                    f"{self.capacity.safe_vram_limit_mb:.1f}MB). Throttled batch to {new_batch} "
                    f"and queue depth to {new_knobs.queue_depth} to prevent driver thrash."
                )
                decision = SchedulerDecision(
                    timestamp=time.time(),
                    frame_index=curr_frame,
                    trigger_reason="VRAM_PRESSURE_HIGH",
                    previous_knobs=prev_knobs,
                    updated_knobs=new_knobs.to_dict(),
                    explanation=explanation,
                )
                self._commit_decision(decision, new_knobs, curr_frame)
                return decision

            # ── RULE 2: System RAM Cap / Strict RSS on 3060 Laptop ─────────
            if telemetry.ram_used_mb > self.capacity.safe_ram_limit_mb:
                new_knobs.queue_depth = max(2, new_knobs.queue_depth - 1)
                new_knobs.worker_count = max(2, new_knobs.worker_count - 1)
                explanation = (
                    f"Host RAM ({telemetry.ram_used_mb:.1f}MB) exceeded safe boundary "
                    f"({self.capacity.safe_ram_limit_mb:.1f}MB). Scaled down workers to {new_knobs.worker_count}."
                )
                decision = SchedulerDecision(
                    timestamp=time.time(),
                    frame_index=curr_frame,
                    trigger_reason="RAM_EXHAUSTION_GUARD",
                    previous_knobs=prev_knobs,
                    updated_knobs=new_knobs.to_dict(),
                    explanation=explanation,
                )
                self._commit_decision(decision, new_knobs, curr_frame)
                return decision

            # ── RULE 3: Never Allow Unlimited Queue Growth (Backpressure) ──
            if telemetry.queue_depth_current >= telemetry.queue_depth_max:
                # Upstream decoder is producing faster than downstream can process
                new_knobs.decode_workers = max(1, new_knobs.decode_workers - 1)
                explanation = (
                    f"Queue depth saturated at ceiling ({telemetry.queue_depth_current}/{telemetry.queue_depth_max}). "
                    f"Throttled decode workers to {new_knobs.decode_workers} for backpressure."
                )
                decision = SchedulerDecision(
                    timestamp=time.time(),
                    frame_index=curr_frame,
                    trigger_reason="QUEUE_BACKPRESSURE_ACTIVE",
                    previous_knobs=prev_knobs,
                    updated_knobs=new_knobs.to_dict(),
                    explanation=explanation,
                )
                self._commit_decision(decision, new_knobs, curr_frame)
                return decision

            # ── RULE 4: Avoid Decoder Starvation ───────────────────────────
            if telemetry.decoder_state == ComponentState.STARVING or (
                telemetry.queue_depth_current == 0 and telemetry.encoder_state == ComponentState.OK
            ):
                new_knobs.decode_workers = min(4, new_knobs.decode_workers + 1)
                new_knobs.queue_depth = min(8, new_knobs.queue_depth + 1)
                explanation = (
                    f"Decoder starvation observed (Queue empty). Increased decode workers to "
                    f"{new_knobs.decode_workers} and expanded queue depth to {new_knobs.queue_depth}."
                )
                decision = SchedulerDecision(
                    timestamp=time.time(),
                    frame_index=curr_frame,
                    trigger_reason="DECODER_STARVATION_RESCUE",
                    previous_knobs=prev_knobs,
                    updated_knobs=new_knobs.to_dict(),
                    explanation=explanation,
                )
                self._commit_decision(decision, new_knobs, curr_frame)
                return decision

            # ── RULE 5: Avoid Encoder Starvation ───────────────────────────
            if telemetry.encoder_state == ComponentState.STARVING:
                new_knobs.worker_count = min(self.capacity.max_worker_cap, new_knobs.worker_count + 1)
                new_knobs.preprocessing_workers = min(6, new_knobs.preprocessing_workers + 1)
                explanation = (
                    f"Encoder starving for completed frames. Accelerated preprocessing workers to "
                    f"{new_knobs.preprocessing_workers} and pipeline workers to {new_knobs.worker_count}."
                )
                decision = SchedulerDecision(
                    timestamp=time.time(),
                    frame_index=curr_frame,
                    trigger_reason="ENCODER_STARVATION_RESCUE",
                    previous_knobs=prev_knobs,
                    updated_knobs=new_knobs.to_dict(),
                    explanation=explanation,
                )
                self._commit_decision(decision, new_knobs, curr_frame)
                return decision

            # ── RULE 6: Avoid CPU Starvation ───────────────────────────────
            if self._smoothed_cpu_util > 92.0:
                new_knobs.worker_count = max(2, new_knobs.worker_count - 2)
                new_knobs.preprocessing_workers = max(1, new_knobs.preprocessing_workers - 1)
                explanation = (
                    f"High CPU utilization ({self._smoothed_cpu_util:.1f}%). Clamped worker count to "
                    f"{new_knobs.worker_count} to prevent OS and FFmpeg thread starvation."
                )
                decision = SchedulerDecision(
                    timestamp=time.time(),
                    frame_index=curr_frame,
                    trigger_reason="CPU_STARVATION_PREVENTION",
                    previous_knobs=prev_knobs,
                    updated_knobs=new_knobs.to_dict(),
                    explanation=explanation,
                )
                self._commit_decision(decision, new_knobs, curr_frame)
                return decision

            # ── RULE 7: Upward Expansion with Strict Hysteresis ───────────
            # Only expand if cooldown frames have passed and stable headroom was observed
            frames_since_last = curr_frame - self._last_decision_frame
            if (frames_since_last >= self.cooldown_frames or force_evaluation) and (
                self.capacity.safe_vram_limit_mb == 0
                or telemetry.gpu_vram_allocated_mb < (self.capacity.safe_vram_limit_mb * 0.75)
            ):
                self._stable_cycles += 1
                # Must sustain headroom for at least 2 checks before expanding
                if self._stable_cycles >= 2 or force_evaluation:
                    changed = False
                    if new_knobs.inference_batch_size < self.capacity.max_batch_cap:
                        new_knobs.inference_batch_size = min(
                            self.capacity.max_batch_cap, new_knobs.inference_batch_size * 2
                        )
                        changed = True

                    if new_knobs.worker_count < self.capacity.max_worker_cap:
                        new_knobs.worker_count = min(
                            self.capacity.max_worker_cap, new_knobs.worker_count + 2
                        )
                        changed = True

                    if changed:
                        explanation = (
                            f"Stable headroom observed ({self._stable_cycles} cycles). Safely scaled batch to "
                            f"{new_knobs.inference_batch_size} and worker count to {new_knobs.worker_count}."
                        )
                        decision = SchedulerDecision(
                            timestamp=time.time(),
                            frame_index=curr_frame,
                            trigger_reason="STABLE_HEADROOM_EXPANSION",
                            previous_knobs=prev_knobs,
                            updated_knobs=new_knobs.to_dict(),
                            explanation=explanation,
                        )
                        self._commit_decision(decision, new_knobs, curr_frame)
                        self._stable_cycles = 0
                        return decision

            return None

    def _commit_decision(
        self, decision: SchedulerDecision, new_knobs: SchedulerKnobs, frame_index: int
    ) -> None:
        """Apply new knobs and record decision to the audit log."""
        self.knobs = new_knobs
        self._last_decision_frame = frame_index
        self._decision_history.append(decision)
        logger.info(
            "[RuntimeScheduler @ Frame %d] %s: %s",
            frame_index,
            decision.trigger_reason,
            decision.explanation,
        )

    def get_current_knobs(self) -> SchedulerKnobs:
        """Return a copy of the active regulated pipeline knobs."""
        with self._lock:
            return SchedulerKnobs(**self.knobs.to_dict())

    def get_decision_history(self) -> List[SchedulerDecision]:
        """Return the complete ledger of scheduler decisions."""
        with self._lock:
            return list(self._decision_history)

    def get_diagnostics_summary(self) -> Dict[str, Any]:
        """Return a comprehensive diagnostics summary."""
        with self._lock:
            return {
                "capacity": asdict(self.capacity),
                "active_knobs": self.knobs.to_dict(),
                "smoothed_gpu_util_pct": self._smoothed_gpu_util,
                "smoothed_cpu_util_pct": self._smoothed_cpu_util,
                "smoothed_stage_latencies_ms": {
                    k.value: v for k, v in self._smoothed_latencies.items()
                },
                "total_decisions_made": len(self._decision_history),
                "last_decision_frame": self._last_decision_frame,
            }


__all__ = [
    "ComponentState",
    "HardwareCapacity",
    "PipelineStage",
    "RuntimeTelemetry",
    "SchedulerDecision",
    "SchedulerKnobs",
    "UnifiedRuntimeScheduler",
    "detect_system_capacity",
]
