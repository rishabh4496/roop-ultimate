"""Hardware-isolated validation records for the two required GPU targets.

This module only assembles reports.  It never invents measurements and never
selects runtime settings from a GPU model name.  A benchmark record is
accepted only when its stable hardware/workload key matches the target record;
otherwise the target remains pending.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
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
    "cpu_name", "cpu_physical_cores", "cpu_logical_cores",
    "cpu_max_frequency_mhz", "cpu_simd_capabilities",
    "cpu_performance_cores", "cpu_efficiency_cores", "cpu_topology_source",
    "os_affinity_supported",
)

_TARGET_GPU_PATTERNS = {
    # Validation labels are deliberately stricter than the runtime policy.  A
    # GPU name is used only to verify that a submitted result belongs to the
    # named target; architecture and capabilities still come from runtime
    # probes in HardwareProfiler.
    "RTX 3060": re.compile(
        r"\brtx\s+3060\b(?!\s*(?:ti|super)\b)", re.IGNORECASE),
    "RTX 4070": re.compile(
        r"\brtx\s+4070\b(?!\s*(?:ti|super)\b)", re.IGNORECASE),
}

def target_matches_hardware(target: str, hardware: Mapping[str, Any]) -> bool:
    """Return whether detected NVIDIA identity matches a validation target.

    This is an audit guard, not a capability detector.  It prevents a result
    recorded under one target label from being accepted for another target,
    while all runtime choices continue to use the probed architecture,
    compute capability, memory, and exposed software capabilities.
    """
    pattern = _TARGET_GPU_PATTERNS.get(str(target))
    if pattern is None or not isinstance(hardware, Mapping):
        return False
    vendor = str(hardware.get("gpu_vendor", "")).strip().lower()
    name = str(hardware.get("gpu_name", "")).strip()
    return bool(
        name and
        (vendor in ("", "unknown", "nvidia") or "nvidia" in name.lower()) and
        pattern.search(name)
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
        "missing_metrics": list(REQUIRED_METRICS),
        "notes": reason,
    }


def _measurement(target: str, record: Mapping[str, Any]) -> dict | None:
    hardware = record.get("hardware")
    supplied_key = record.get("hardware_profile_key")
    if not isinstance(hardware, Mapping):
        return None
    if not target_matches_hardware(target, hardware):
        return None
    expected_key = hardware_profile_key(hardware, record.get("workload"))
    if not supplied_key or supplied_key != expected_key:
        return None
    metrics = {name: record.get(name) for name in REQUIRED_METRICS}
    missing = [name for name, value in metrics.items() if value is None]
    status = record.get("status", "measured")
    # A partial stage/calibration record remains useful evidence, but it must
    # never look like a complete final validation row.  Complete acceptance
    # requires every metric in REQUIRED_METRICS to be present.
    if status == "measured" and missing:
        status = "measured_partial"
    return {
        "target": target,
        "status": status,
        "hardware_profile_key": record.get("hardware_profile_key"),
        "metrics": metrics,
        "missing_metrics": missing,
        "hardware": dict(hardware),
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
    complete_statuses = {"measured", "validated", "complete"}
    if (not a or not b or a.get("status") not in complete_statuses or
            b.get("status") not in complete_statuses):
        return "PENDING"
    a_change = a.get("improvement_pct")
    b_change = b.get("improvement_pct")
    if (not isinstance(a_change, (int, float)) or
            not isinstance(b_change, (int, float)) or
            not math.isfinite(float(a_change)) or
            not math.isfinite(float(b_change))):
        return "F. UNSAFE / REJECTED"
    tolerance = abs(float(improvement_tolerance_pct))
    a_good = a_change > tolerance
    b_good = b_change > tolerance
    a_regression = a_change < -tolerance
    b_regression = b_change < -tolerance
    if a_regression or b_regression:
        return "E. REGRESSION ON ONE GPU"
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
    "classify_optimization", "hardware_profile_key", "target_matches_hardware",
]
