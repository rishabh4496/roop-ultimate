"""Target-bound NVIDIA discovery shared by the performance harnesses."""
import csv
import io
import subprocess

from roop.hardware_validation import target_matches_hardware


def query_gpus():
    """Return (rows, raw) for every adapter, including its stable index."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader,nounits"], text=True, timeout=10)
    except Exception as exc:
        return [], "unavailable: %s" % exc
    rows = []
    for values in csv.reader(io.StringIO(out)):
        if len(values) < 5:
            continue
        try:
            index = int(values[0].strip())
        except ValueError:
            continue
        rows.append({
            "index": index,
            "name": values[1].strip(),
            "memory_total": values[2].strip(),
            "driver": values[3].strip(),
            "compute_capability": values[4].strip(),
        })
    return rows, out.strip() or "unknown"


def selected_gpu(rows, device_id):
    try:
        wanted = int(device_id)
    except (TypeError, ValueError):
        wanted = 0
    return next((row for row in rows if row["index"] == wanted), None)


def target_on_device(target, rows, device_id):
    row = selected_gpu(rows, device_id)
    hardware = ({"gpu_name": row["name"], "gpu_vendor": "nvidia"}
                if row else None)
    return bool(row and target_matches_hardware(target, hardware)), row


def format_selected(rows, raw, device_id):
    row = selected_gpu(rows, device_id)
    if row is None:
        return "selected GPU %s unavailable; detected %s" % (device_id, raw)
    return "%s, %s MiB, driver %s, compute %s (device %s)" % (
        row["name"], row["memory_total"], row["driver"],
        row["compute_capability"], row["index"])
