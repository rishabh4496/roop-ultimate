"""Hardware-isolated Phase 11 matrix schema and report assembly."""

import hashlib
import json

from roop.enhancer_inventory import entries


MATRIX_FIELDS = (
    "Enhancer", "Backend", "Precision", "Input", "Output", "Batch",
    "Contexts", "Streams", "FPS", "Latency", "VRAM", "CPU", "Notes",
)


def hardware_key(hardware: dict, workload=None) -> str:
    """Create a stable hardware/workload key for a benchmark profile."""
    identity = {
        key: hardware.get(key) for key in (
            "device_id", "gpu_name", "architecture", "compute_capability",
            "vram_total_gb", "vram_available_gb", "driver_version", "cuda_version",
            "tensorrt_version", "onnxruntime_version", "tensor_core_capabilities",
            "fp16_supported", "bf16_supported", "int8_supported", "fp8_supported",
            "nvdec_available", "nvdec_codecs", "nvenc_available", "nvenc_codecs",
        )
    }
    if workload:
        identity["workload"] = {
            key: workload.get(key) for key in (
                "enhancer", "model", "model_hash", "precision", "input",
                "output", "batch", "workload_characteristics",
            ) if key in workload
        }
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _row(entry, profile_key):
    return {
        "Enhancer": entry["label"],
        "Backend": entry["backend"],
        "Precision": entry["precision"],
        "Input": entry["input"],
        "Output": entry["output"],
        "Batch": entry["batch"],
        "Contexts": entry["contexts"],
        "Streams": entry["streams"],
        "FPS": None,
        "Latency": None,
        "VRAM": None,
        "CPU": None,
        "Notes": (
            f"pending runtime measurement; source={entry['source']} "
            f"class={entry['class']} Run={entry['run']} "
            f"Initialize={entry['initialize']} Release={entry['release']} "
            f"profile={profile_key}"
        ),
        "id": entry["id"],
        "model": entry["model"],
        "quality_guards": entry.get("quality_guards", ""),
        "status": "pending",
        "hardware_profile_key": profile_key,
    }


def create_matrix(hardware: dict, measurements=None, include_adjacent=True):
    """Return one independent row per discovered enhancement path.

    ``measurements`` is keyed by registry ``id`` and may contain only values
    actually observed on the supplied hardware.  Missing values stay pending;
    this is intentional for unavailable RTX 3060/RTX 4070 validation targets.
    """
    hardware_profile_key = hardware_key(hardware)
    measured = measurements or {}
    result = []
    for entry in entries(include_adjacent=include_adjacent):
        workload = {
            "enhancer": entry["id"],
            "model": entry["model"],
            "precision": entry["precision"],
            "input": entry["input"],
            "output": entry["output"],
            "batch": entry["batch"],
        }
        profile_key = hardware_key(hardware, workload)
        row = _row(entry, profile_key)
        row["hardware_key"] = hardware_profile_key
        values = measured.get(entry["id"])
        if values:
            for key in ("FPS", "Latency", "VRAM", "CPU", "Notes"):
                if key in values:
                    row[key] = values[key]
            row["status"] = values.get("status", "measured")
        result.append(row)
    return result


__all__ = ["MATRIX_FIELDS", "create_matrix", "hardware_key"]
