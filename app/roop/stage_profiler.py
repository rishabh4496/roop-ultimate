"""Opt-in end-to-end stage profiler.

This module is observational only. It is enabled with ``ROOP_PROFILE_DETAIL=1``
and is attached to the existing ``procmgr_runtime._prof`` context, so it cannot
change scheduling, session ownership, or frame ordering. GPU synchronization is
deliberately optional because explicit fences perturb throughput; a report records
whether the GPU fields are event-based, synchronized, or unavailable.
"""

from collections import defaultdict
import json
import threading
import time


REQUIRED_STAGES = (
    "detection", "tracking", "alignment", "faceset_lookup", "swap",
    "expression_analysis", "occlusion_analysis", "detail_restoration",
    "enhancement", "lighting", "mask", "blending", "encoding",
)

_CANONICAL = {
    "detect": "detection",
    "track_detect": "tracking", "track_decode": "tracking",
    "track_wait": "tracking", "track_consume": "tracking",
    "expression": "expression_analysis",
    "identity_detail": "detail_restoration", "detail_transfer": "detail_restoration",
    "enhance": "enhancement",
    "blend": "blending",
    "encode": "encoding", "encode_finalize": "encoding",
}


def _cuda_module():
    try:
        import torch
        if torch.cuda.is_available():
            return torch
    except Exception:
        pass
    return None


def _memory_snapshot(torch):
    try:
        allocated = float(torch.cuda.memory_allocated()) / (1024 ** 2)
        reserved = float(torch.cuda.memory_reserved()) / (1024 ** 2)
        stats = torch.cuda.memory_stats()
        allocated_total = float(stats.get("allocated_bytes.all.allocated", 0)) / (1024 ** 2)
        freed_total = float(stats.get("allocated_bytes.all.freed", 0)) / (1024 ** 2)
        return {"allocated_mb": allocated, "reserved_mb": reserved,
                "allocated_total_mb": allocated_total,
                "freed_total_mb": freed_total}
    except Exception:
        return None


class StageProfiler:
    """Collect per-stage CPU/GPU/allocator evidence without owning execution."""

    def __init__(self, gpu_sync=False, device_id=0):
        self.gpu_sync = bool(gpu_sync)
        self.device_id = int(device_id or 0)
        self.torch = _cuda_module()
        self._stages = defaultdict(lambda: {
            "calls": 0, "cpu_seconds": 0.0, "gpu_event_seconds": 0.0,
            "gpu_event_samples": 0, "gpu_sync_window_seconds": 0.0,
            "sync_seconds": 0.0, "sync_samples": 0,
            "alloc_delta_mb": 0.0, "reserved_delta_mb": 0.0,
            "alloc_peak_mb": None, "reserved_peak_mb": None,
            "steady_state_allocated_mb": [], "steady_state_reserved_mb": [],
            "transfer_h2d_bytes": 0, "transfer_d2h_bytes": 0,
            "transfer_attribution": "unavailable: ORT provider transfers are opaque",
        })
        self._lock = threading.RLock()
        self._calls = 0
        self._started = time.perf_counter()
        self._peak_allocated_mb = None
        self._peak_reserved_mb = None

    def begin(self, stage):
        token = {"stage": str(stage), "cpu_start": time.perf_counter(),
                 "mem_start": _memory_snapshot(self.torch) if self.torch else None,
                 "event_start": None, "sync_start": None}
        if self.torch:
            try:
                if self.gpu_sync:
                    sync_start = time.perf_counter()
                    self.torch.cuda.synchronize(self.device_id)
                    token["sync_start"] = sync_start
                token["event_start"] = self.torch.cuda.Event(enable_timing=True)
                token["event_start"].record()
            except Exception:
                token["event_start"] = None
        return token

    def end(self, token):
        if token is None:
            return
        stage = token["stage"]
        cpu_seconds = max(0.0, time.perf_counter() - token["cpu_start"])
        gpu_event_seconds = None
        sync_seconds = 0.0
        sync_window = None
        if self.torch:
            try:
                event_end = self.torch.cuda.Event(enable_timing=True)
                event_end.record()
                wait_start = time.perf_counter()
                if self.gpu_sync:
                    self.torch.cuda.synchronize(self.device_id)
                wait_end = time.perf_counter()
                sync_seconds = max(0.0, wait_end - wait_start)
                if token.get("event_start") is not None:
                    gpu_event_seconds = max(0.0, token["event_start"].elapsed_time(event_end) / 1000.0)
                if token.get("sync_start") is not None:
                    sync_window = max(0.0, wait_end - token["sync_start"])
            except Exception:
                gpu_event_seconds = None
        mem_end = _memory_snapshot(self.torch) if self.torch else None
        with self._lock:
            data = self._stages[stage]
            data["calls"] += 1
            data["cpu_seconds"] += cpu_seconds
            self._calls += 1
            if gpu_event_seconds is not None:
                data["gpu_event_seconds"] += gpu_event_seconds
                data["gpu_event_samples"] += 1
            if sync_window is not None:
                data["gpu_sync_window_seconds"] += sync_window
            if self.torch and self.gpu_sync:
                data["sync_seconds"] += sync_seconds
                data["sync_samples"] += 1
            if token.get("mem_start") and mem_end:
                before = token["mem_start"]
                data["alloc_delta_mb"] += float(mem_end["allocated_mb"] - before["allocated_mb"])
                data["reserved_delta_mb"] += float(mem_end["reserved_mb"] - before["reserved_mb"])
                for key, peak_key, steady_key in (
                        ("allocated_mb", "alloc_peak_mb", "steady_state_allocated_mb"),
                        ("reserved_mb", "reserved_peak_mb", "steady_state_reserved_mb")):
                    value = float(mem_end[key])
                    data[peak_key] = value if data[peak_key] is None else max(data[peak_key], value)
                    if data["calls"] > 3:
                        data[steady_key].append(value)
                    if key == "allocated_mb":
                        self._peak_allocated_mb = value if self._peak_allocated_mb is None else max(self._peak_allocated_mb, value)
                    else:
                        self._peak_reserved_mb = value if self._peak_reserved_mb is None else max(self._peak_reserved_mb, value)

    def record_transfer(self, stage, direction, byte_count):
        """Allow an explicit tensor boundary to contribute transfer bytes."""
        direction = "h2d" if str(direction).lower() in ("h2d", "host_to_device") else "d2h"
        try:
            byte_count = max(0, int(byte_count))
        except (TypeError, ValueError):
            return
        with self._lock:
            data = self._stages[str(stage)]
            data["transfer_" + direction + "_bytes"] += byte_count
            data["transfer_attribution"] = "explicit boundary counters"

    @staticmethod
    def _mean(values):
        return (sum(values) / len(values)) if values else None

    def report(self):
        with self._lock:
            stages = {}
            for name, source in self._stages.items():
                item = dict(source)
                item["cpu_ms_total"] = item.pop("cpu_seconds") * 1000.0
                item["gpu_event_ms_total"] = item.pop("gpu_event_seconds") * 1000.0
                item["gpu_sync_window_ms_total"] = item.pop("gpu_sync_window_seconds") * 1000.0
                item["sync_ms_total"] = item.pop("sync_seconds") * 1000.0
                item["cpu_ms_per_call"] = item["cpu_ms_total"] / max(1, item["calls"])
                item["gpu_event_ms_per_call"] = item["gpu_event_ms_total"] / max(1, item["gpu_event_samples"])
                item["sync_ms_per_call"] = item["sync_ms_total"] / max(1, item["sync_samples"])
                item["steady_state_allocated_mb"] = self._mean(item["steady_state_allocated_mb"])
                item["steady_state_reserved_mb"] = self._mean(item["steady_state_reserved_mb"])
                stages[name] = item
            canonical = {}
            for name in REQUIRED_STAGES:
                members = [key for key in stages
                           if _CANONICAL.get(key, key) == name]
                if not members:
                    canonical[name] = self._empty_report_stage()
                    continue
                merged = self._empty_report_stage()
                merged["status"] = "measured"
                for key in members:
                    source = stages[key]
                    for field in ("calls", "gpu_event_samples", "sync_samples",
                                  "cpu_ms_total", "gpu_event_ms_total",
                                  "gpu_sync_window_ms_total", "sync_ms_total",
                                  "alloc_delta_mb", "reserved_delta_mb",
                                  "transfer_h2d_bytes", "transfer_d2h_bytes"):
                        merged[field] += source.get(field, 0) or 0
                    for field in ("alloc_peak_mb", "reserved_peak_mb"):
                        value = source.get(field)
                        if value is not None:
                            merged[field] = value if merged[field] is None else max(merged[field], value)
                    for field in ("steady_state_allocated_mb", "steady_state_reserved_mb"):
                        value = source.get(field)
                        if value is not None:
                            merged[field] = (merged[field] or []) + [value]
                    if source.get("transfer_attribution", "").startswith("explicit"):
                        merged["transfer_attribution"] = "explicit boundary counters"
                merged["cpu_ms_per_call"] = merged["cpu_ms_total"] / max(1, merged["calls"])
                merged["gpu_event_ms_per_call"] = merged["gpu_event_ms_total"] / max(1, merged["gpu_event_samples"])
                merged["sync_ms_per_call"] = merged["sync_ms_total"] / max(1, merged["sync_samples"])
                merged["steady_state_allocated_mb"] = self._mean(merged["steady_state_allocated_mb"])
                merged["steady_state_reserved_mb"] = self._mean(merged["steady_state_reserved_mb"])
                canonical[name] = merged
            return {
                "schema": "roop-phase14-stage-profile-v1",
                "mode": "synchronized" if self.gpu_sync else "event-only",
                "invasive": bool(self.gpu_sync),
                "torch_cuda_available": self.torch is not None,
                "device_id": self.device_id,
                "total_calls": self._calls,
                "elapsed_seconds": max(0.0, time.perf_counter() - self._started),
                "peak_allocated_mb": self._peak_allocated_mb,
                "peak_reserved_mb": self._peak_reserved_mb,
                "host_device_transfer_note": (
                    "Explicit boundary bytes only; ONNX Runtime provider transfers "
                    "remain opaque unless a boundary calls record_transfer()."),
                "gpu_time_note": (
                    "CUDA event time covers the current PyTorch stream. In synchronized "
                    "mode the sync window also fences other device work, but includes "
                    "concurrent CPU/device activity."),
                "full_card_memory_note": (
                    "Stage allocator values are PyTorch allocator values; full-card "
                    "peak/steady VRAM comes from the external telemetry harness."),
                "stages": stages,
                "required_stages": list(REQUIRED_STAGES),
                "canonical_stages": canonical,
            }

    @staticmethod
    def _empty_report_stage():
        return {
            "status": "not_observed", "calls": 0, "gpu_event_samples": 0,
            "sync_samples": 0, "cpu_ms_total": 0.0,
            "gpu_event_ms_total": 0.0, "gpu_sync_window_ms_total": 0.0,
            "sync_ms_total": 0.0, "cpu_ms_per_call": 0.0,
            "gpu_event_ms_per_call": 0.0, "sync_ms_per_call": 0.0,
            "alloc_delta_mb": 0.0, "reserved_delta_mb": 0.0,
            "alloc_peak_mb": None, "reserved_peak_mb": None,
            "steady_state_allocated_mb": None, "steady_state_reserved_mb": None,
            "transfer_h2d_bytes": 0, "transfer_d2h_bytes": 0,
            "transfer_attribution": "unavailable: ORT provider transfers are opaque",
        }

    def print_report(self):
        print("\n==== DETAILED STAGE PROFILE (ROOP_PROFILE_DETAIL) ====", flush=True)
        print(json.dumps(self.report(), sort_keys=True), flush=True)
        print("=======================================================\n", flush=True)


__all__ = ["StageProfiler"]
