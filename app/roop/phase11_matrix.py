"""Hardware-isolated Phase 11 matrix schema and report assembly."""

from roop.enhancer_inventory import entries
from roop.hardware_validation import hardware_profile_key


MATRIX_FIELDS = (
    "Enhancer", "Backend", "Precision", "Input", "Output", "Batch",
    "Contexts", "Streams", "FPS", "Latency", "VRAM", "CPU", "Notes",
)


def hardware_key(hardware: dict, workload=None) -> str:
    """Create a stable hardware/workload key for a benchmark profile.

    Available VRAM is deliberately excluded.  It is a live measurement that
    changes as models, pools, and other applications allocate memory; using it
    as an identity field made the same GPU produce a different profile key
    during one benchmark.  Total VRAM and all runtime capability fields remain
    part of the key, so profiles cannot cross GPU/runtime boundaries.
    """
    return hardware_profile_key(hardware, workload)


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
    base_hardware_key = hardware_key(hardware)
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
        row["hardware_key"] = base_hardware_key
        values = measured.get(entry["id"])
        if values:
            for key in ("FPS", "Latency", "VRAM", "CPU", "Notes"):
                if key in values:
                    row[key] = values[key]
            row["status"] = values.get("status", "measured")
        result.append(row)
    return result


__all__ = ["MATRIX_FIELDS", "create_matrix", "hardware_key"]
