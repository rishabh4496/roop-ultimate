"""Stage timing instrumentation and GPU/VRAM telemetry for roop-ultimate benchmarks.

Provides clean hooks into ``app/roop/procmgr_runtime.py`` via ``set_stage_sink``
to capture per-frame and per-stage timings, p50/p95 latency percentiles, effective
FPS throughput, and GPU/VRAM telemetry via pynvml (with graceful torch/zero fallbacks).

Usage Example:
    from app.tests.bench_telemetry import StageTelemetry

    telemetry = StageTelemetry()
    with telemetry:
        # Run benchmark or roop rendering pipeline
        ...
    summary = telemetry.get_summary()
    telemetry.print_summary()
"""
from __future__ import annotations

from collections import defaultdict
import contextlib
import json
import math
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

# Canonical 8 stage groups defined by benchmark requirements
STAGE_GROUPS: Dict[str, Dict[str, Any]] = {
    "video_decode": {
        "name": "Video Decode",
        "stages": ["decode", "track_decode"],
    },
    "detection_landmarks": {
        "name": "Face Detection & Landmark Extraction",
        "stages": ["detect", "detection", "track_detect", "alignment"],
    },
    "pose_source_selection": {
        "name": "Pose & Source Embedding Selection",
        "stages": ["faceset_lookup"],
    },
    "swapper_inference": {
        "name": "Swapper Inference",
        "stages": ["swap"],
    },
    "mask_generation": {
        "name": "Occluder & Face Mask Generation",
        "stages": [
            "mask",
            "mask_refine",
            "mask_stabilize",
            "occlusion_analysis",
            "interaction_occlusion",
        ],
    },
    "color_transfer_blending": {
        "name": "Color Transfer & Blending",
        "stages": [
            "lighting",
            "blend",
            "realism",
            "identity_detail",
            "detail_transfer",
        ],
    },
    "face_restoration": {
        "name": "Face Restoration",
        "stages": ["enhance", "enhance_recombine", "stabilize"],
    },
    "video_encode_mux": {
        "name": "Video Encode + Audio Mux",
        "stages": ["encode", "encode_finalize"],
    },
}

# Reverse index: stage_name -> group_key
STAGE_TO_GROUP: Dict[str, str] = {}
for _grp_key, _grp_info in STAGE_GROUPS.items():
    for _stg in _grp_info["stages"]:
        STAGE_TO_GROUP[_stg] = _grp_key


def compute_percentiles(values: Sequence[float]) -> Dict[str, float]:
    """Compute distribution statistics (p50, p95, p99, mean, min, max, std, count).

    Returns 0.0 for all statistics if values is empty.
    """
    if not values:
        return {
            "count": 0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
        }

    clean_vals = [float(v) for v in values]
    n = len(clean_vals)
    val_mean = sum(clean_vals) / n
    variance = sum((x - val_mean) ** 2 for x in clean_vals) / n
    val_std = math.sqrt(variance)

    try:
        import numpy as np

        p50 = float(np.percentile(clean_vals, 50))
        p95 = float(np.percentile(clean_vals, 95))
        p99 = float(np.percentile(clean_vals, 99))
    except Exception:
        sorted_vals = sorted(clean_vals)

        def _calc_pct(pct: float) -> float:
            k = (len(sorted_vals) - 1) * (pct / 100.0)
            f = int(k)
            c = min(f + 1, len(sorted_vals) - 1)
            d = k - f
            return float(sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f]))

        p50 = _calc_pct(50.0)
        p95 = _calc_pct(95.0)
        p99 = _calc_pct(99.0)

    return {
        "count": n,
        "p50": round(p50, 4),
        "p95": round(p95, 4),
        "p99": round(p99, 4),
        "mean": round(val_mean, 4),
        "min": round(min(clean_vals), 4),
        "max": round(max(clean_vals), 4),
        "std": round(val_std, 4),
    }


class GpuTelemetrySampler:
    """Background sampling thread for device VRAM (MB) and GPU utilization (%).

    Uses pynvml when available, with automatic fallback to torch.cuda or 0.0
    if running in headless, CPU-only, or unsupported environments.
    """

    def __init__(self, device_id: int = 0, interval_s: float = 0.05):
        self.device_id = int(device_id)
        self.interval_s = float(max(0.01, interval_s))
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.device_name: str = "Unknown"
        self.total_vram_mb: float = 0.0
        self.source: str = "none"

        self.vram_samples_mb: List[float] = []
        self.gpu_util_samples: List[float] = []
        self.initial_vram_mb: Optional[float] = None
        self.final_vram_mb: Optional[float] = None
        self.peak_vram_mb: float = 0.0
        self.peak_gpu_util: float = 0.0

        self._nvml_initialized = False
        self._nvml_handle = None
        self._init_backend()

    def _init_backend(self) -> None:
        """Initialize the GPU query backend (pynvml -> torch -> fallback)."""
        # 1. Attempt pynvml
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            idx = min(max(0, self.device_id), device_count - 1)
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            raw_name = pynvml.nvmlDeviceGetName(handle)
            name_str = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            self._nvml_handle = handle
            self._nvml_initialized = True
            self.device_name = name_str
            self.total_vram_mb = float(mem_info.total) / (1024.0 * 1024.0)
            self.source = "pynvml"
            return
        except Exception:
            self._nvml_initialized = False
            self._nvml_handle = None

        # 2. Attempt PyTorch CUDA
        try:
            import torch

            if torch.cuda.is_available():
                idx = min(max(0, self.device_id), torch.cuda.device_count() - 1)
                self.device_name = str(torch.cuda.get_device_name(idx))
                try:
                    free_b, total_b = torch.cuda.mem_get_info(idx)
                    self.total_vram_mb = float(total_b) / (1024.0 * 1024.0)
                except Exception:
                    self.total_vram_mb = 0.0
                self.source = "torch"
                return
        except Exception:
            pass

        # 3. Fallback
        self.source = "fallback"
        self.device_name = "CPU / Fallback"
        self.total_vram_mb = 0.0

    def sample_once(self) -> Tuple[float, float]:
        """Query instantaneous VRAM used (MB) and GPU utilization (%)."""
        if self._nvml_initialized and self._nvml_handle is not None:
            try:
                import pynvml

                mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                vram_mb = float(mem.used) / (1024.0 * 1024.0)
                gpu_util = float(util.gpu)
                return vram_mb, gpu_util
            except Exception:
                pass

        if self.source == "torch":
            try:
                import torch

                if torch.cuda.is_available():
                    try:
                        free_b, total_b = torch.cuda.mem_get_info(self.device_id)
                        vram_mb = float(total_b - free_b) / (1024.0 * 1024.0)
                    except Exception:
                        vram_mb = float(torch.cuda.max_memory_allocated(self.device_id)) / (1024.0 * 1024.0)
                    return vram_mb, 0.0
            except Exception:
                pass

        return 0.0, 0.0

    def _record_sample(self, vram_mb: float, gpu_util: float) -> None:
        with self._lock:
            self.vram_samples_mb.append(vram_mb)
            self.gpu_util_samples.append(gpu_util)
            if self.initial_vram_mb is None:
                self.initial_vram_mb = vram_mb
            self.final_vram_mb = vram_mb
            if vram_mb > self.peak_vram_mb:
                self.peak_vram_mb = vram_mb
            if gpu_util > self.peak_gpu_util:
                self.peak_gpu_util = gpu_util

    def _run(self) -> None:
        while not self._stop_event.is_set():
            vram, util = self.sample_once()
            self._record_sample(vram, util)
            self._stop_event.wait(self.interval_s)

    def start(self) -> None:
        """Start the background sampling thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self.reset()
            # Take immediate baseline sample
            vram, util = self.sample_once()
            self._record_sample(vram, util)
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run, name="bench_gpu_sampler", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop sampling and record final snapshot."""
        # Final sample at stage completion
        vram, util = self.sample_once()
        self._record_sample(vram, util)

        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.5)
        self._thread = None

    def reset(self) -> None:
        """Reset sample buffers."""
        with self._lock:
            self.vram_samples_mb.clear()
            self.gpu_util_samples.clear()
            self.initial_vram_mb = None
            self.final_vram_mb = None
            self.peak_vram_mb = 0.0
            self.peak_gpu_util = 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregated GPU and VRAM telemetry statistics."""
        with self._lock:
            v_stats = compute_percentiles(self.vram_samples_mb)
            u_stats = compute_percentiles(self.gpu_util_samples)

            peak_vram = self.peak_vram_mb if self.peak_vram_mb > 0.0 else v_stats["max"]
            peak_util = self.peak_gpu_util if self.peak_gpu_util > 0.0 else u_stats["max"]

            return {
                "source": self.source,
                "device_name": self.device_name,
                "total_vram_mb": round(self.total_vram_mb, 2),
                "peak_vram_mb": round(peak_vram, 2),
                "mean_vram_mb": round(v_stats["mean"], 2),
                "initial_vram_mb": round(self.initial_vram_mb or 0.0, 2),
                "final_vram_mb": round(self.final_vram_mb or 0.0, 2),
                "peak_gpu_util_pct": round(peak_util, 2),
                "mean_gpu_util_pct": round(u_stats["mean"], 2),
                "p50_gpu_util_pct": round(u_stats["p50"], 2),
                "p95_gpu_util_pct": round(u_stats["p95"], 2),
                "samples_count": len(self.vram_samples_mb),
            }


class StageTelemetry:
    """Telemetry collector and timing observer for roop-ultimate pipeline stages.

    Hooks cleanly into ``app/roop/procmgr_runtime.set_stage_sink`` without modifying
    production runtime behavior. Records per-frame and per-stage timings, p50/p95
    latency percentiles, total throughput (FPS), and GPU telemetry.
    """

    def __init__(self, device_id: int = 0, sample_interval_s: float = 0.05):
        self.device_id = int(device_id)
        self.sample_interval_s = float(sample_interval_s)

        self._lock = threading.RLock()
        self._tls = threading.local()

        # Raw stage durations in milliseconds: stage -> list of durations [ms, ms, ...]
        self._raw_stage_samples_ms: Dict[str, List[float]] = defaultdict(list)

        # Thread-safe frame records: list of {"frame_idx": int, "total_ms": float, "stages": {stage: ms}}
        self._frame_records: List[Dict[str, Any]] = []

        # Per-frame decode and encode times in ms (1 sample per frame)
        self._decode_samples_ms: List[float] = []
        self._encode_samples_ms: List[float] = []
        self._frame_total_samples_ms: List[float] = []

        self._started_at: Optional[float] = None
        self._stopped_at: Optional[float] = None
        self._explicit_frame_counter: int = 0

        self.gpu_sampler = GpuTelemetrySampler(
            device_id=self.device_id, interval_s=self.sample_interval_s
        )
        self._prev_sink: Optional[Callable[[str, float], None]] = None
        self._is_hooked = False

    def reset(self) -> None:
        """Reset all recorded metrics and timings."""
        with self._lock:
            self._raw_stage_samples_ms.clear()
            self._frame_records.clear()
            self._decode_samples_ms.clear()
            self._encode_samples_ms.clear()
            self._frame_total_samples_ms.clear()
            self._started_at = None
            self._stopped_at = None
            self._explicit_frame_counter = 0
            self.gpu_sampler.reset()

    def record_stage(self, stage: str, dt_s: float) -> None:
        """Sink callback for stage telemetry: ``sink(stage, dt_s)``.

        Called by ``procmgr_runtime._prof`` upon exiting a stage context.
        Thread-safe and low overhead.
        """
        dt_ms = float(dt_s) * 1000.0
        stage_name = str(stage)

        with self._lock:
            self._raw_stage_samples_ms[stage_name].append(dt_ms)

            # Record dedicated I/O stage samples
            if stage_name in ("decode", "track_decode"):
                self._decode_samples_ms.append(dt_ms)
            elif stage_name in ("encode", "encode_finalize"):
                self._encode_samples_ms.append(dt_ms)
            elif stage_name == "frame_total":
                self._frame_total_samples_ms.append(dt_ms)

        # Thread-local frame stage accumulation
        pending = getattr(self._tls, "pending_stages", None)
        if pending is None:
            self._tls.pending_stages = defaultdict(float)
            pending = self._tls.pending_stages

        if stage_name == "frame_total":
            # A full frame has completed on this thread
            frame_breakdown = dict(pending)
            pending.clear()
            with self._lock:
                idx = len(self._frame_records)
                self._frame_records.append({
                    "frame_idx": idx,
                    "total_ms": dt_ms,
                    "stages": frame_breakdown,
                })
        else:
            pending[stage_name] += dt_ms

        # Forward to any previously active sink
        prev = self._prev_sink
        if prev is not None and callable(prev):
            try:
                prev(stage, dt_s)
            except Exception:
                pass

    def start(self) -> "StageTelemetry":
        """Hook into procmgr_runtime and start GPU sampling."""
        with self._lock:
            self.reset()
            self._started_at = time.perf_counter()

            # Hook procmgr_runtime
            try:
                from roop import procmgr_runtime

                self._prev_sink = getattr(procmgr_runtime, "_STAGE_SINK", None)
                procmgr_runtime.set_stage_sink(self.record_stage)
                self._is_hooked = True
            except Exception:
                self._is_hooked = False

            # Start GPU sampler
            self.gpu_sampler.start()
        return self

    def stop(self) -> "StageTelemetry":
        """Unhook from procmgr_runtime, stop GPU sampler, and mark elapsed time."""
        with self._lock:
            self._stopped_at = time.perf_counter()

            # Flush any pending thread-local frame stages on current thread
            pending = getattr(self._tls, "pending_stages", None)
            if pending:
                frame_breakdown = dict(pending)
                pending.clear()
                total_ms = sum(frame_breakdown.values())
                idx = len(self._frame_records)
                self._frame_records.append({
                    "frame_idx": idx,
                    "total_ms": total_ms,
                    "stages": frame_breakdown,
                })

            # Unhook procmgr_runtime
            if self._is_hooked:
                try:
                    from roop import procmgr_runtime

                    procmgr_runtime.set_stage_sink(self._prev_sink)
                except Exception:
                    pass
                self._is_hooked = False

            # Stop GPU sampler
            self.gpu_sampler.stop()
        return self

    def __enter__(self) -> "StageTelemetry":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def start_frame(self, frame_idx: Optional[int] = None) -> None:
        """Explicitly demarcate the start of a frame on the calling thread."""
        if frame_idx is None:
            with self._lock:
                frame_idx = self._explicit_frame_counter
                self._explicit_frame_counter += 1

        self._tls.current_frame_idx = int(frame_idx)
        self._tls.frame_start_time = time.perf_counter()
        self._tls.pending_stages = defaultdict(float)

    def end_frame(self, frame_idx: Optional[int] = None) -> Dict[str, Any]:
        """Explicitly demarcate the end of a frame on the calling thread."""
        t_end = time.perf_counter()
        t_start = getattr(self._tls, "frame_start_time", t_end)
        frame_total_ms = max(0.0, (t_end - t_start) * 1000.0)

        f_idx = getattr(self._tls, "current_frame_idx", frame_idx or 0)
        pending = getattr(self._tls, "pending_stages", None) or {}
        frame_breakdown = dict(pending)

        record = {
            "frame_idx": int(f_idx),
            "total_ms": frame_total_ms,
            "stages": frame_breakdown,
        }

        with self._lock:
            self._frame_records.append(record)
            self._frame_total_samples_ms.append(frame_total_ms)

        self._tls.pending_stages = defaultdict(float)
        return record

    @contextlib.contextmanager
    def frame_scope(self, frame_idx: Optional[int] = None):
        """Context manager to scope all stage calls within one frame."""
        self.start_frame(frame_idx)
        try:
            yield
        finally:
            self.end_frame(frame_idx)

    def get_summary(self, total_frames: Optional[int] = None) -> Dict[str, Any]:
        """Aggregate per-frame and per-stage timings, p50/p95 latencies, FPS, and GPU telemetry.

        Parameters:
            total_frames: Optional override for frame count. If not supplied,
                          infers from frame records, frame_total samples, decode calls,
                          or stage calls.
        """
        with self._lock:
            t0 = self._started_at or time.perf_counter()
            t1 = self._stopped_at or time.perf_counter()
            wall_time_s = max(0.0001, t1 - t0)

            # Determine frame count
            inferred_frames = max(
                len(self._frame_records),
                len(self._frame_total_samples_ms),
                len(self._decode_samples_ms),
            )
            if inferred_frames == 0 and self._raw_stage_samples_ms:
                inferred_frames = max(len(samples) for samples in self._raw_stage_samples_ms.values())

            num_frames = int(total_frames if total_frames is not None else max(1, inferred_frames))

            # 1. Compute Individual Stage Metrics
            stages_summary: Dict[str, Dict[str, Any]] = {}
            total_stage_time_ms = 0.0

            for stage_name, samples_ms in self._raw_stage_samples_ms.items():
                stats = compute_percentiles(samples_ms)
                tot_ms = sum(samples_ms)
                total_stage_time_ms += tot_ms
                stage_time_s = tot_ms / 1000.0
                effective_fps = round(stats["count"] / stage_time_s, 2) if stage_time_s > 0 else 0.0

                stages_summary[stage_name] = {
                    "calls": stats["count"],
                    "total_ms": round(tot_ms, 2),
                    "mean_ms": stats["mean"],
                    "p50_ms": stats["p50"],
                    "p95_ms": stats["p95"],
                    "p99_ms": stats["p99"],
                    "min_ms": stats["min"],
                    "max_ms": stats["max"],
                    "effective_fps": effective_fps,
                }

            # 2. Compute Canonical Stage Groups Metrics
            groups_summary: Dict[str, Dict[str, Any]] = {}

            for grp_key, grp_def in STAGE_GROUPS.items():
                grp_label = grp_def["name"]
                member_stages = grp_def["stages"]

                # Sum durations per frame across member stages
                per_frame_group_ms: List[float] = []

                if self._frame_records:
                    for rec in self._frame_records:
                        stages_dict = rec.get("stages", {})
                        f_ms = sum(stages_dict.get(stg, 0.0) for stg in member_stages)
                        per_frame_group_ms.append(f_ms)
                elif grp_key == "video_decode" and self._decode_samples_ms:
                    per_frame_group_ms = list(self._decode_samples_ms)
                elif grp_key == "video_encode_mux" and self._encode_samples_ms:
                    per_frame_group_ms = list(self._encode_samples_ms)
                else:
                    # Fallback when no frame records exist: aggregate all calls across member stages
                    agg_samples: List[float] = []
                    for stg in member_stages:
                        agg_samples.extend(self._raw_stage_samples_ms.get(stg, []))
                    per_frame_group_ms = agg_samples

                # Statistics across frames
                grp_stats = compute_percentiles(per_frame_group_ms)

                # Total calls and total ms across member stages
                grp_calls = sum(len(self._raw_stage_samples_ms.get(stg, [])) for stg in member_stages)
                grp_total_ms = sum(sum(self._raw_stage_samples_ms.get(stg, [])) for stg in member_stages)
                grp_time_s = grp_total_ms / 1000.0

                # Stage group throughput
                grp_fps = round(num_frames / grp_time_s, 2) if grp_time_s > 0 else 0.0

                groups_summary[grp_key] = {
                    "name": grp_label,
                    "stages": member_stages,
                    "calls": grp_calls,
                    "total_ms": round(grp_total_ms, 2),
                    "p50_ms": grp_stats["p50"],
                    "p95_ms": grp_stats["p95"],
                    "p99_ms": grp_stats["p99"],
                    "mean_ms": grp_stats["mean"],
                    "min_ms": grp_stats["min"],
                    "max_ms": grp_stats["max"],
                    "effective_fps": grp_fps,
                    "pct_share": round((100.0 * grp_total_ms / total_stage_time_ms), 2)
                    if total_stage_time_ms > 0
                    else 0.0,
                }

            # 3. Overall Pipeline Totals
            total_fps = round(num_frames / wall_time_s, 2)
            frame_tot_stats = compute_percentiles(self._frame_total_samples_ms)

            # 4. GPU Telemetry
            gpu_stats = self.gpu_sampler.get_stats()

            summary = {
                "total_frames": num_frames,
                "wall_time_s": round(wall_time_s, 4),
                "total_fps": total_fps,
                "frame_latency_ms": {
                    "p50": frame_tot_stats["p50"],
                    "p95": frame_tot_stats["p95"],
                    "p99": frame_tot_stats["p99"],
                    "mean": frame_tot_stats["mean"],
                    "min": frame_tot_stats["min"],
                    "max": frame_tot_stats["max"],
                },
                # Top-level direct mirrors for rapid CLI integration
                "peak_vram_mb": gpu_stats["peak_vram_mb"],
                "gpu_util_pct": gpu_stats["mean_gpu_util_pct"],
                "frame_p50_ms": frame_tot_stats["p50"],
                "frame_p95_ms": frame_tot_stats["p95"],
                "stage_groups": groups_summary,
                "stages": stages_summary,
                "gpu": gpu_stats,
            }
            return summary

    def to_dict(self, total_frames: Optional[int] = None) -> Dict[str, Any]:
        """Alias for get_summary()."""
        return self.get_summary(total_frames=total_frames)

    def to_json(self, filepath: Optional[str] = None, indent: int = 2) -> str:
        """Serialize benchmark summary to JSON string, optionally writing to file."""
        data = self.get_summary()
        json_str = json.dumps(data, indent=indent, sort_keys=True)
        if filepath:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    def print_summary(self, total_frames: Optional[int] = None) -> None:
        """Print clean ASCII / terminal summary tables to stdout."""
        summary = self.get_summary(total_frames=total_frames)
        table = format_timing_table(summary)
        print(table, flush=True)


class BenchmarkTelemetryContext:
    """Convenience context manager wrapper around StageTelemetry."""

    def __init__(self, device_id: int = 0, sample_interval_s: float = 0.05):
        self.telemetry = StageTelemetry(device_id=device_id, sample_interval_s=sample_interval_s)

    def __enter__(self) -> StageTelemetry:
        return self.telemetry.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.telemetry.stop()


def hook_telemetry(telemetry: Optional[StageTelemetry] = None) -> StageTelemetry:
    """Attach a StageTelemetry instance to procmgr_runtime's stage sink."""
    t = telemetry or StageTelemetry()
    return t.start()


def unhook_telemetry() -> None:
    """Detach any active stage sink from procmgr_runtime."""
    try:
        from roop import procmgr_runtime

        procmgr_runtime.set_stage_sink(None)
    except Exception:
        pass


def format_timing_table(summary: Dict[str, Any]) -> str:
    """Format structured summary dict into a readable CLI table."""
    lines: List[str] = []
    lines.append("=" * 90)
    lines.append(f"{'STAGE-BY-STAGE TIMING BREAKDOWN':^90}")
    lines.append("=" * 90)
    lines.append(
        f"{'Stage Group':<40} {'Calls':>7} {'Total (s)':>10} {'Share':>7} {'p50 (ms)':>9} {'p95 (ms)':>9} {'FPS':>8}"
    )
    lines.append("-" * 90)

    stage_groups = summary.get("stage_groups", {})
    for grp_key, grp in stage_groups.items():
        name = grp.get("name", grp_key)
        calls = grp.get("calls", 0)
        tot_s = grp.get("total_ms", 0.0) / 1000.0
        share = grp.get("pct_share", 0.0)
        p50 = grp.get("p50_ms", 0.0)
        p95 = grp.get("p95_ms", 0.0)
        fps = grp.get("effective_fps", 0.0)
        lines.append(
            f"{name:<40} {calls:>7d} {tot_s:>10.2f} {share:>6.1f}% {p50:>9.2f} {p95:>9.2f} {fps:>8.1f}"
        )

    lines.append("-" * 90)
    frames = summary.get("total_frames", 0)
    wall_s = summary.get("wall_time_s", 0.0)
    tot_fps = summary.get("total_fps", 0.0)
    lat = summary.get("frame_latency_ms", {})
    p50_frame = lat.get("p50", 0.0)
    p95_frame = lat.get("p95", 0.0)

    lines.append(
        f"Pipeline Total: {frames} frames in {wall_s:.2f}s ({tot_fps:.2f} FPS) | Frame Latency: p50={p50_frame:.1f}ms, p95={p95_frame:.1f}ms"
    )

    gpu = summary.get("gpu", {})
    dev_name = gpu.get("device_name", "N/A")
    peak_vram = gpu.get("peak_vram_mb", 0.0)
    tot_vram = gpu.get("total_vram_mb", 0.0)
    peak_util = gpu.get("peak_gpu_util_pct", 0.0)
    mean_util = gpu.get("mean_gpu_util_pct", 0.0)
    source = gpu.get("source", "none")

    vram_str = f"{peak_vram:.1f} MB" + (f" / {tot_vram:.1f} MB" if tot_vram > 0 else "")
    lines.append(
        f"GPU [{dev_name}] ({source}): Peak VRAM: {vram_str} | Util: mean={mean_util:.1f}%, peak={peak_util:.1f}%"
    )
    lines.append("=" * 90)
    return "\n".join(lines)


def compare_telemetry(
    current: Dict[str, Any], baseline: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate absolute and percentage differences between current run and a baseline."""
    comparison: Dict[str, Any] = {
        "fps": {
            "baseline": baseline.get("total_fps", 0.0),
            "current": current.get("total_fps", 0.0),
            "delta": round(current.get("total_fps", 0.0) - baseline.get("total_fps", 0.0), 2),
            "pct_change": _calc_pct_change(
                baseline.get("total_fps", 0.0), current.get("total_fps", 0.0)
            ),
        },
        "peak_vram_mb": {
            "baseline": baseline.get("peak_vram_mb", 0.0),
            "current": current.get("peak_vram_mb", 0.0),
            "delta": round(current.get("peak_vram_mb", 0.0) - baseline.get("peak_vram_mb", 0.0), 2),
            "pct_change": _calc_pct_change(
                baseline.get("peak_vram_mb", 0.0), current.get("peak_vram_mb", 0.0)
            ),
        },
        "frame_latency_p50_ms": {
            "baseline": baseline.get("frame_p50_ms", 0.0),
            "current": current.get("frame_p50_ms", 0.0),
            "delta": round(current.get("frame_p50_ms", 0.0) - baseline.get("frame_p50_ms", 0.0), 2),
            "pct_change": _calc_pct_change(
                baseline.get("frame_p50_ms", 0.0), current.get("frame_p50_ms", 0.0)
            ),
        },
        "stage_groups": {},
    }

    curr_groups = current.get("stage_groups", {})
    base_groups = baseline.get("stage_groups", {})

    all_keys = sorted(set(list(curr_groups.keys()) + list(base_groups.keys())))
    for key in all_keys:
        cg = curr_groups.get(key, {})
        bg = base_groups.get(key, {})
        name = cg.get("name") or bg.get("name") or key

        cg_p50 = cg.get("p50_ms", 0.0)
        bg_p50 = bg.get("p50_ms", 0.0)

        comparison["stage_groups"][key] = {
            "name": name,
            "baseline_p50_ms": bg_p50,
            "current_p50_ms": cg_p50,
            "delta_p50_ms": round(cg_p50 - bg_p50, 2),
            "pct_change_p50": _calc_pct_change(bg_p50, cg_p50),
        }

    return comparison


def format_comparison_table(current: Dict[str, Any], baseline: Dict[str, Any]) -> str:
    """Format baseline vs current comparison into an ASCII CLI table."""
    comp = compare_telemetry(current, baseline)
    lines: List[str] = []
    lines.append("=" * 86)
    lines.append(f"{'BENCHMARK REGRESSION / SPEEDUP COMPARISON':^86}")
    lines.append("=" * 86)
    lines.append(f"{'Metric':<35} {'Baseline':>15} {'Current':>15} {'Delta':>15}")
    lines.append("-" * 86)

    # Top-level metrics
    fps = comp["fps"]
    fps_delta_str = f"{fps['delta']:+.2f} ({fps['pct_change']:+.1f}%)"
    lines.append(f"{'Throughput (FPS)':<35} {fps['baseline']:>15.2f} {fps['current']:>15.2f} {fps_delta_str:>15}")

    vram = comp["peak_vram_mb"]
    vram_delta_str = f"{vram['delta']:+.1f} MB ({vram['pct_change']:+.1f}%)"
    lines.append(f"{'Peak VRAM (MB)':<35} {vram['baseline']:>15.1f} {vram['current']:>15.1f} {vram_delta_str:>15}")

    lat = comp["frame_latency_p50_ms"]
    lat_delta_str = f"{lat['delta']:+.2f} ms ({lat['pct_change']:+.1f}%)"
    lines.append(f"{'Frame Latency p50 (ms)':<35} {lat['baseline']:>15.2f} {lat['current']:>15.2f} {lat_delta_str:>15}")

    lines.append("-" * 86)
    lines.append(f"{'Stage Group (p50 ms)':<35} {'Baseline':>15} {'Current':>15} {'Delta':>15}")
    lines.append("-" * 86)

    for grp_key, grp_data in comp.get("stage_groups", {}).items():
        name = grp_data["name"]
        b_val = grp_data["baseline_p50_ms"]
        c_val = grp_data["current_p50_ms"]
        d_val = grp_data["delta_p50_ms"]
        pct_val = grp_data["pct_change_p50"]
        d_str = f"{d_val:+.2f} ({pct_val:+.1f}%)"
        lines.append(f"{name:<35} {b_val:>15.2f} {c_val:>15.2f} {d_str:>15}")

    lines.append("=" * 86)
    return "\n".join(lines)


def _calc_pct_change(base_val: float, curr_val: float) -> float:
    if abs(base_val) < 1e-9:
        return 0.0
    return round(100.0 * (curr_val - base_val) / base_val, 2)


__all__ = [
    "STAGE_GROUPS",
    "STAGE_TO_GROUP",
    "compute_percentiles",
    "GpuTelemetrySampler",
    "StageTelemetry",
    "BenchmarkTelemetryContext",
    "hook_telemetry",
    "unhook_telemetry",
    "format_timing_table",
    "compare_telemetry",
    "format_comparison_table",
]
