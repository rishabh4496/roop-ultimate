"""Hardware-isolated validation records for the two required GPU targets.

This module only assembles reports.  It never invents measurements and never
selects runtime settings from a GPU model name.  A benchmark record is
accepted only when its stable hardware/workload key matches the target record;
otherwise the target remains pending.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


VALIDATION_TARGETS = ("RTX 3060", "RTX 4070")
REQUIRED_METRICS = (
    "baseline_fps", "final_fps", "improvement_pct", "peak_vram_mb",
    "average_vram_mb", "cpu_utilization_pct", "gpu_utilization_pct",
    "decode_throughput_fps", "inference_throughput_fps",
    "enhancement_throughput_fps", "encode_throughput_fps", "latency_ms",
    "stability", "output_quality",
)

_HARDWARE_FIELDS = (
    "device_id", "gpu_name", "gpu_vendor", "architecture",
    "compute_capability", "vram_total_gb", "vram_tier", "driver_version",
    "cuda_version", "tensorrt_version", "onnxruntime_version",
    "tensor_core_capabilities", "fp16_supported", "bf16_supported",
    "int8_supported", "fp8_supported", "nvdec_available", "nvdec_codecs",
    "nvenc_available", "nvenc_codecs", "ram_total_gb", "platform",
)


def hardware_profile_key(hardware: Mapping[str, Any], workload: Mapping[str, Any] | None = None) -> str:
    """Return a stable key from detected identity and workload facts.

    ``vram_available_gb``/``free_vram_gb`` are intentionally transient and are
    not included.  Total VRAM is read from the runtime and remains part of the
    identity; no target capacity is assumed.
    """
    identity = {key: hardware.get(key) for key in _HARDWARE_FIELDS}
    if workload:
        identity["workload"] = dict(workload)
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _pending(target: str, reason: str = "no physical measurement recorded") -> dict:
    return {
        "target": target,
        "status": "pending",
        "hardware_profile_key": None,
        "metrics": {name: None for name in REQUIRED_METRICS},
        "notes": reason,
    }


def _measurement(target: str, record: Mapping[str, Any]) -> dict | None:
    hardware = record.get("hardware")
    supplied_key = record.get("hardware_profile_key")
    if isinstance(hardware, Mapping):
        expected_key = hardware_profile_key(hardware, record.get("workload"))
        if not supplied_key or supplied_key != expected_key:
            return None
    metrics = {name: record.get(name) for name in REQUIRED_METRICS}
    return {
        "target": target,
        "status": record.get("status", "measured"),
        "hardware_profile_key": record.get("hardware_profile_key"),
        "metrics": metrics,
        "notes": record.get("notes", ""),
    }


def build_dual_target_report(records: Mapping[str, Mapping[str, Any]] | None = None) -> dict:
    """Build separate RTX 3060/RTX 4070 tables without cross-filling rows.

    ``records`` is keyed by the explicit validation target label.  Callers
    should pass a record only after running on that physical target.  Missing
    or malformed records are represented as ``pending``.
    """
    records = records or {}
    targets = {}
    for target in VALIDATION_TARGETS:
        record = records.get(target)
        if not isinstance(record, Mapping):
            targets[target] = _pending(target)
            continue
        measurement = _measurement(target, record)
        targets[target] = measurement or _pending(
            target, "hardware profile key does not match the supplied runtime identity")
    return {
        "schema_version": 1,
        "targets": targets,
        "required_metrics": list(REQUIRED_METRICS),
        "separate_tables": True,
        "notes": (
            "Results are hardware-specific. A target without a physical run "
            "remains pending; no result is copied from the other target."
        ),
    }


def classify_optimization(
    rtx3060: Mapping[str, Any] | None,
    rtx4070: Mapping[str, Any] | None,
    *,
    improvement_tolerance_pct: float = 0.0,
) -> str:
    """Classify an optimization using measured target outcomes only.

    Returns the acceptance labels required by the project.  Missing target
    results are never treated as success on that target.
    """
    a = rtx3060 or {}
    b = rtx4070 or {}
    if not a or not b or a.get("status") == "pending" or b.get("status") == "pending":
        return "PENDING"
    a_change = a.get("improvement_pct")
    b_change = b.get("improvement_pct")
    if not isinstance(a_change, (int, float)) or not isinstance(b_change, (int, float)):
        return "F. UNSAFE / REJECTED"
    tolerance = abs(float(improvement_tolerance_pct))
    a_good = a_change > tolerance
    b_good = b_change > tolerance
    if a_good and b_good:
        return "A. BENEFICIAL ON BOTH"
    if a_good and not b_good:
        return "B. RTX 3060-SPECIFIC"
    if b_good and not a_good:
        return "C. RTX 4070-SPECIFIC"
    if abs(a_change) <= tolerance and abs(b_change) <= tolerance:
        return "D. NEUTRAL"
    return "E. REGRESSION ON ONE GPU"


__all__ = [
    "REQUIRED_METRICS", "VALIDATION_TARGETS", "build_dual_target_report",
    "classify_optimization", "hardware_profile_key",
]
