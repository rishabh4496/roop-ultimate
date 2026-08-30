"""Unified resource scheduler for the video pipeline.

The application has several independently safe optimizers (CPU policy,
TensorRT pools, CUDA streams, NVDEC/NVENC selection, and stabilization
chunking).  This module is the small coordination layer that gives those
decisions one owner without taking ownership of model/session lifetimes.

Two execution modes are intentional:

* ``run`` is a frame pipeline for workloads whose processing callback is safe
  to run concurrently.  Decode, processing, and encode use bounded queues and
  can occupy different frames at the same time.
* ``admit``/``observe`` is the control plane used by the existing chunked
  stabilizer.  Temporal stabilizers remain chunk-owned because changing their
  execution order would change output pixels; the coordinator still owns the
  resource budget and safe-boundary limits.

All limits are derived from the detected hardware profile and workload.  No
GPU model, VRAM capacity, or architecture is selected by name here.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import Any, Callable, Mapping, Optional

from roop.runtime_optimizer import (
    HardwareProfile,
    RuntimeTuning,
    WorkloadProfile,
)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _enabled(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class SchedulerBudget:
    """Admission budget for decoded frames and in-flight processing."""

    frame_bytes: int
    ram_budget_bytes: int
    queue_capacity: int
    in_flight_limit: int
    worker_limit: int
    worker_buffer_frames: int
    stabilization_chunk_frames: int
    estimated_host_bytes: int
    vram_margin_ratio: float
    ram_margin_ratio: float
    pinned_host_memory: bool = False
    pinned_memory_reason: str = "not used for mutable OpenCV-owned frames"

    @classmethod
    def from_profile(cls, hardware: HardwareProfile,
                     workload: WorkloadProfile,
                     tuning: RuntimeTuning) -> "SchedulerBudget":
        width = max(1, int(workload.input_width or workload.output_width or 1))
        height = max(1, int(workload.input_height or workload.output_height or 1))
        # BGR frames are three one-byte channels.  Processing may create one
        # destination copy, so the budget counts two frame-sized host buffers
        # per admitted frame rather than pretending one array is sufficient.
        frame_bytes = width * height * 3
        configured_mb = max(1, int(tuning.ram_buffer_mb or 1))
        available_ram = _number(hardware.ram_available_gb)
        if available_ram > 0:
            # Keep the scheduler's buffer budget below a fraction of currently
            # available RAM.  The fraction is a safety margin, not a device
            # capacity assumption; the user can lower the configured budget.
            ram_budget = min(configured_mb * 2**20,
                             int(available_ram * 2**30 * 0.25))
        else:
            ram_budget = configured_mb * 2**20
        ram_budget = max(frame_bytes * 2, ram_budget)

        requested_queue = max(1, int(tuning.queue_depth or 1))
        requested_inflight = max(1, int(tuning.in_flight_frames or 1))
        workers = max(1, int(tuning.worker_count or 1))
        worker_buffer_frames = workers
        stabilization_chunk_frames = (max(0, int(tuning.stabilization_chunk_size or 0))
                                       if workload.stabilization_enabled else 0)
        # Reserve host memory for worker destinations and any stateful
        # stabilizer chunk before admitting queue slots. This is an estimate,
        # not an allocation: the existing stabilizer owns its chunk buffers.
        per_frame_bytes = frame_bytes * 2
        reserved_bytes = frame_bytes * (worker_buffer_frames +
                                        stabilization_chunk_frames)
        queue_budget = max(per_frame_bytes,
                           ram_budget - min(reserved_bytes,
                                            max(0, ram_budget - per_frame_bytes)))
        memory_slots = max(2, queue_budget // max(1, per_frame_bytes))
        queue_capacity = max(1, min(requested_queue, memory_slots // 2 or 1))
        in_flight = max(1, min(requested_inflight,
                               max(1, memory_slots - queue_capacity * 2)))
        in_flight = min(in_flight, workers + queue_capacity)
        buffered_frames = queue_capacity * 2 + in_flight
        estimated_host_bytes = frame_bytes * (
            buffered_frames * 2 + worker_buffer_frames +
            stabilization_chunk_frames)

        # Margins are environment-tunable fractions of detected capacity.  A
        # small-card policy therefore gets the same rule as a desktop card,
        # with different measured free/total memory and tuning inputs.
        vram_margin = _number(os.environ.get(
            "ROOP_SCHEDULER_VRAM_MARGIN", "0.12"), 0.12)
        ram_margin = _number(os.environ.get(
            "ROOP_SCHEDULER_RAM_MARGIN", "0.20"), 0.20)
        return cls(
            frame_bytes=frame_bytes,
            ram_budget_bytes=ram_budget,
            queue_capacity=queue_capacity,
            in_flight_limit=in_flight,
            worker_limit=workers,
            worker_buffer_frames=worker_buffer_frames,
            stabilization_chunk_frames=stabilization_chunk_frames,
            estimated_host_bytes=estimated_host_bytes,
            vram_margin_ratio=max(0.01, min(0.50, vram_margin)),
            ram_margin_ratio=max(0.01, min(0.50, ram_margin)),
        )


@dataclass
class SchedulerMetrics:
    decoded: int = 0
    processed: int = 0
    encoded: int = 0
    dropped: int = 0
    queue_wait_seconds: dict = field(default_factory=dict)
    stage_seconds: dict = field(default_factory=dict)
    stage_calls: dict = field(default_factory=dict)
    max_queue_depths: dict = field(default_factory=dict)
    resource_samples: deque = field(default_factory=lambda: deque(maxlen=64))
    actions: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def add_stage(self, name: str, elapsed: float) -> None:
        key = str(name)
        self.stage_seconds[key] = self.stage_seconds.get(key, 0.0) + max(0.0, elapsed)
        self.stage_calls[key] = self.stage_calls.get(key, 0) + 1

    def observe_queue(self, name: str, depth: int) -> None:
        value = max(0, int(depth))
        self.max_queue_depths[name] = max(value, self.max_queue_depths.get(name, 0))


@dataclass
class _FramePacket:
    index: int
    frame: Any
    submitted_at: float = field(default_factory=time.perf_counter)


class UnifiedRuntimeScheduler:
    """Coordinate bounded pipeline work and safe adaptive control.

    The scheduler never closes a model session, destroys a CUDA context, or
    mutates an active queue's capacity.  Pressure can reduce admission for
    future work; queue geometry and encoder changes are deferred to the next
    run by the caller.
    """

    STAGE_NAMES = ("decode", "preprocess", "detect", "track", "swap",
                   "enhance", "stabilize", "postprocess", "encode")

    def __init__(self, hardware: HardwareProfile, workload: WorkloadProfile,
                 tuning: RuntimeTuning, settings: Any = None,
                 monitor: Any = None, adaptive: Any = None):
        self.hardware = hardware
        self.workload = workload
        self.tuning = tuning
        self.settings = settings
        self.monitor = monitor
        self.adaptive = adaptive
        self.budget = SchedulerBudget.from_profile(hardware, workload, tuning)
        self.metrics = SchedulerMetrics()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._effective_inflight = self.budget.in_flight_limit
        self._healthy_samples = 0
        self._pressure_samples = 0
        self._last_resource_sample = 0.0
        self._last_bottleneck = "unknown"
        self._active = 0
        self._last_snapshot = None

    @property
    def worker_count(self) -> int:
        return max(1, min(self.budget.worker_limit, self.tuning.worker_count))

    @property
    def queue_capacity(self) -> int:
        return self.budget.queue_capacity

    @property
    def effective_inflight(self) -> int:
        with self._lock:
            return max(1, int(self._effective_inflight))

    def frame_pipeline_allowed(self, stateful_stabilization: bool = False) -> bool:
        """Whether frame-level processing is safe for this workload."""
        if stateful_stabilization and self.worker_count > 1:
            return False
        return _enabled(os.environ.get("ROOP_SCHEDULER_FRAME_PIPELINE", "1"), True)

    def stage(self, name: str):
        """Return a context manager recording a scheduler-owned stage."""
        scheduler = self
        key = str(name)

        class _Stage:
            def __enter__(self):
                self.started = time.perf_counter()
                scheduler.admit(key)
                return self

            def __exit__(self, exc_type, exc, tb):
                scheduler.release(key)
                scheduler.record_stage(key, time.perf_counter() - self.started)
                return False

        return _Stage()

    def admit(self, stage: str = "process") -> None:
        """Account for work without blocking active GPU resources unsafely."""
        if self._stop.is_set():
            raise RuntimeError("runtime scheduler is stopping")
        with self._lock:
            self._active += 1

    def release(self, stage: str = "process") -> None:
        with self._lock:
            self._active = max(0, self._active - 1)

    def record_stage(self, name: str, elapsed: float, calls: int = 1) -> None:
        if calls <= 0:
            return
        with self._lock:
            for _ in range(int(calls)):
                self.metrics.add_stage(name, elapsed / max(1, int(calls)))
        if self.monitor is not None:
            try:
                self.monitor.record_stage(name, elapsed, calls=calls)
            except Exception:
                pass

    def observe_queue(self, name: str, depth: int) -> None:
        with self._lock:
            self.metrics.observe_queue(name, depth)

    @staticmethod
    def _process_rss_gb() -> Optional[float]:
        try:
            import psutil
            return float(psutil.Process(os.getpid()).memory_info().rss) / 2**30
        except Exception:
            return None

    def _resource_snapshot(self) -> dict:
        result = {"time": time.time(), "active": self._active,
                  "effective_inflight": self.effective_inflight}
        rss = self._process_rss_gb()
        if rss is not None:
            result["ram_used_gb"] = rss
        try:
            import psutil
            memory = psutil.virtual_memory()
            result["ram_available_gb"] = float(memory.available) / 2**30
            result["ram_total_gb"] = float(memory.total) / 2**30
            result["cpu_utilization_pct"] = float(psutil.cpu_percent(interval=None))
        except Exception:
            pass
        try:
            import torch
            device_id = int(getattr(self.hardware, "device_id", 0) or 0)
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(device_id)
                result["vram_free_gb"] = int(free) / 2**30
                result["vram_total_gb"] = int(total) / 2**30
        except Exception:
            pass
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(
                int(getattr(self.hardware, "device_id", 0) or 0))
            result["gpu_utilization_pct"] = float(
                pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        except Exception:
            # NVML is optional. The scheduler still has queue, stage, CPU, and
            # VRAM signals when a vendor telemetry binding is unavailable.
            pass
        if result.get("vram_total_gb"):
            result["vram_pressure_pct"] = max(
                0.0, min(100.0, 100.0 * (1.0 - result["vram_free_gb"] /
                                          result["vram_total_gb"])))
        return result

    def observe(self, queue_depths: Optional[Mapping[str, int]] = None,
                force: bool = False) -> dict:
        """Collect a coarse sample and adjust only future admission."""
        now = time.perf_counter()
        interval = _number(os.environ.get("ROOP_SCHEDULER_SAMPLE_INTERVAL", "1.0"), 1.0)
        if not force and now - self._last_resource_sample < max(0.25, interval):
            return self._last_snapshot or self.snapshot()
        self._last_resource_sample = now
        resource = self._resource_snapshot()
        resource["queue_depths"] = dict(queue_depths or {})
        for name, depth in resource["queue_depths"].items():
            self.observe_queue(name, depth)
        with self._lock:
            self.metrics.resource_samples.append(resource)
        self._update_admission(resource)
        self._last_snapshot = self.snapshot()
        return self._last_snapshot

    def _update_admission(self, resource: Mapping[str, Any]) -> None:
        vram_total = _number(resource.get("vram_total_gb"))
        vram_free = _number(resource.get("vram_free_gb"))
        ram_total = _number(resource.get("ram_total_gb"))
        ram_available = _number(resource.get("ram_available_gb"))
        vram_pressure = bool(vram_total and
                             vram_free <= vram_total * self.budget.vram_margin_ratio)
        ram_pressure = bool(ram_total and
                            ram_available <= ram_total * self.budget.ram_margin_ratio)
        queue_depths = resource.get("queue_depths") or {}
        queue_pressure = any(
            _number(depth) >= self.queue_capacity for depth in queue_depths.values())
        pressured = vram_pressure or ram_pressure or queue_pressure
        with self._lock:
            if pressured:
                self._pressure_samples += 1
                self._healthy_samples = 0
            else:
                self._healthy_samples += 1
                self._pressure_samples = 0
            old = self._effective_inflight
            if self._pressure_samples >= 2:
                self._effective_inflight = max(1, old - 1)
                self._pressure_samples = 0
            elif self._healthy_samples >= 4:
                self._effective_inflight = min(self.budget.in_flight_limit, old + 1)
                self._healthy_samples = 0
            new = self._effective_inflight
        if new != old:
            action = {
                "reason": ("resource_pressure" if new < old else "resources_healthy"),
                "changes": {"in_flight_frames": new},
                "scope": "future_work",
                "safe_boundary": True,
            }
            with self._lock:
                self.metrics.actions.append(action)
            if self.adaptive is not None:
                try:
                    self.adaptive.tuning = self.adaptive.tuning  # keep ownership explicit
                except Exception:
                    pass

    def safe_boundary(self, queue_depths: Optional[Mapping[str, int]] = None) -> dict:
        """Observe after a completed encode, the only safe reconfiguration point."""
        # Resource probes are deliberately coarse.  The completed encode is a
        # safe control point, but probing CUDA/psutil for every frame would add
        # a new synchronization/I/O bottleneck to the hot path.
        result = self.observe(queue_depths=queue_depths, force=False)
        if self.adaptive is not None:
            try:
                summary = self.monitor.summary() if self.monitor is not None else result
                action = self.adaptive.update(summary, safe_boundary=True)
                if action:
                    with self._lock:
                        self.metrics.actions.append(dict(action))
                    changes = action.get("changes", {})
                    if "in_flight_frames" in changes:
                        with self._lock:
                            self._effective_inflight = max(
                                1, min(self.budget.in_flight_limit,
                                       _integer(changes["in_flight_frames"],
                                                self._effective_inflight)))
            except Exception as exc:
                with self._lock:
                    self.metrics.errors.append("adaptive: %s" % exc)
        self._last_bottleneck = self.classify_bottleneck(result)
        return result

    def classify_bottleneck(self, snapshot: Optional[Mapping[str, Any]] = None) -> str:
        """Use rolling queue/resource signals, prioritizing safety pressure."""
        data = snapshot or self.snapshot()
        samples = data.get("resource_samples") or []
        latest = samples[-1] if samples else {}
        if _number(latest.get("vram_pressure_pct")) >= 100.0 * (1.0 - self.budget.vram_margin_ratio):
            return "VRAM-bound"
        if (latest.get("ram_total_gb") and
                _number(latest.get("ram_available_gb")) <=
                _number(latest.get("ram_total_gb")) * self.budget.ram_margin_ratio):
            return "RAM-bound"
        queues = latest.get("queue_depths") or {}
        if _number(queues.get("encode", 0)) >= self.queue_capacity:
            return "encode-bound"
        if _number(queues.get("decode", 0)) <= 0 and self.metrics.decoded:
            return "decode-bound"
        gpu = latest.get("gpu_utilization_pct")
        cpu = latest.get("cpu_utilization_pct")
        if gpu is not None and _number(gpu) >= 80.0:
            return "GPU-bound"
        if cpu is not None and _number(cpu) >= 80.0:
            return "CPU-bound"
        if self.metrics.max_queue_depths:
            return "synchronization-bound"
        return "unknown"

    def snapshot(self) -> dict:
        with self._lock:
            metrics = self.metrics
            return {
                "hardware_profile_key": getattr(self.hardware, "as_dict", lambda: {})().get(
                    "hardware_profile_key"),
                "workload_profile": getattr(self.workload, "as_dict", lambda: {})(),
                "runtime_tuning": getattr(self.tuning, "as_dict", lambda: {})(),
                "queue_capacity": self.queue_capacity,
                "in_flight_limit": self.budget.in_flight_limit,
                "effective_inflight": self._effective_inflight,
                "worker_limit": self.worker_count,
                "frame_bytes": self.budget.frame_bytes,
                "ram_budget_bytes": self.budget.ram_budget_bytes,
                "worker_buffer_frames": self.budget.worker_buffer_frames,
                "stabilization_chunk_frames": self.budget.stabilization_chunk_frames,
                "estimated_host_bytes": self.budget.estimated_host_bytes,
                "pinned_host_memory": self.budget.pinned_host_memory,
                "pinned_memory_reason": self.budget.pinned_memory_reason,
                "decoded": metrics.decoded,
                "processed": metrics.processed,
                "encoded": metrics.encoded,
                "dropped": metrics.dropped,
                "stage_seconds": dict(metrics.stage_seconds),
                "stage_calls": dict(metrics.stage_calls),
                "max_queue_depths": dict(metrics.max_queue_depths),
                "resource_samples": list(metrics.resource_samples),
                "actions": list(metrics.actions),
                "errors": list(metrics.errors),
                "bottleneck": self._last_bottleneck,
            }

    def run(self, decode: Callable[[], Any], process: Callable[[Any, int], Any],
            encode: Callable[[Any, int], None],
            should_continue: Optional[Callable[[], bool]] = None,
            on_frame: Optional[Callable[[int], None]] = None,
            workers: Optional[int] = None) -> dict:
        """Run a bounded decode/process/encode pipeline in frame order."""
        should_continue = should_continue or (lambda: not self._stop.is_set())
        worker_count = max(1, min(self.worker_count,
                                  _integer(workers, self.worker_count)
                                  if workers is not None else self.worker_count))
        decode_q: Queue = Queue(maxsize=self.queue_capacity)
        encode_q: Queue = Queue(maxsize=self.queue_capacity)
        errors = []
        decoder_done = object()
        processor_done = object()

        def put_until(queue: Queue, item: Any) -> bool:
            while not self._stop.is_set():
                try:
                    queue.put(item, timeout=0.25)
                    return True
                except Full:
                    self.observe({"decode": decode_q.qsize(),
                                  "encode": encode_q.qsize()})
                    if not should_continue():
                        self._stop.set()
                        return False
            return False

        def decoder() -> None:
            index = 0
            try:
                while should_continue() and not self._stop.is_set():
                    started = time.perf_counter()
                    frame = decode()
                    self.record_stage("decode", time.perf_counter() - started)
                    if frame is None:
                        break
                    packet = _FramePacket(index=index, frame=frame)
                    if not put_until(decode_q, packet):
                        break
                    with self._lock:
                        self.metrics.decoded += 1
                    index += 1
            except Exception as exc:
                errors.append(exc)
                self._stop.set()
            finally:
                put_until(decode_q, decoder_done)

        def processor() -> None:
            pending = {}
            try:
                with ThreadPoolExecutor(max_workers=worker_count,
                                        thread_name_prefix="roop-scheduler") as pool:
                    input_done = False
                    while not self._stop.is_set() and (not input_done or pending):
                        while not input_done and len(pending) < self.effective_inflight:
                            try:
                                item = decode_q.get(timeout=0.25)
                            except Empty:
                                if not should_continue():
                                    self._stop.set()
                                    break
                                continue
                            self.observe({"decode": decode_q.qsize(),
                                          "encode": encode_q.qsize()})
                            if item is decoder_done:
                                input_done = True
                                break
                            future = pool.submit(self._process_one, process, item)
                            pending[future] = item
                        if not pending:
                            continue
                        done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                        for future in done:
                            item = pending.pop(future)
                            result = future.result()
                            if result is None:
                                with self._lock:
                                    self.metrics.dropped += 1
                                continue
                            item.frame = result
                            if not put_until(encode_q, item):
                                return
                            with self._lock:
                                self.metrics.processed += 1
            except Exception as exc:
                errors.append(exc)
                self._stop.set()
            finally:
                put_until(encode_q, processor_done)

        def encoder() -> None:
            expected = 0
            pending = {}
            try:
                while not self._stop.is_set():
                    try:
                        item = encode_q.get(timeout=0.25)
                    except Empty:
                        if not should_continue() and not pending:
                            break
                        continue
                    if item is processor_done:
                        for index in sorted(pending):
                            packet = pending[index]
                            self._encode_one(encode, packet, on_frame)
                        pending.clear()
                        break
                    pending[item.index] = item
                    while expected in pending:
                        packet = pending.pop(expected)
                        self._encode_one(encode, packet, on_frame)
                        expected += 1
            except Exception as exc:
                errors.append(exc)
                self._stop.set()

        threads = [
            threading.Thread(target=decoder, name="roop-scheduler-decode", daemon=True),
            threading.Thread(target=processor, name="roop-scheduler-process", daemon=True),
            threading.Thread(target=encoder, name="roop-scheduler-encode", daemon=True),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self._stop.set()
        if errors:
            with self._lock:
                self.metrics.errors.extend("%s: %s" % (type(e).__name__, e)
                                           for e in errors)
            raise errors[0]
        self.safe_boundary({"decode": decode_q.qsize(), "encode": encode_q.qsize()})
        return self.snapshot()

    def _process_one(self, process: Callable[[Any, int], Any], packet: _FramePacket):
        started = time.perf_counter()
        try:
            return process(packet.frame, packet.index)
        finally:
            self.record_stage("process", time.perf_counter() - started)

    def _encode_one(self, encode: Callable[[Any, int], None], packet: _FramePacket,
                    on_frame: Optional[Callable[[int], None]]) -> None:
        started = time.perf_counter()
        encode(packet.frame, packet.index)
        self.record_stage("encode", time.perf_counter() - started)
        with self._lock:
            self.metrics.encoded += 1
        if on_frame is not None:
            on_frame(packet.index)
        self.safe_boundary({"decode": 0, "encode": 0})


__all__ = ["SchedulerBudget", "SchedulerMetrics", "UnifiedRuntimeScheduler"]
