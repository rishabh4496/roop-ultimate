"""Phase 11 inventory and hardware-isolated matrix contracts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.enhancer_inventory import entries  # noqa: E402
from roop.phase11_matrix import create_matrix, hardware_key  # noqa: E402


def test_inventory_contains_all_source_discovered_paths():
    labels = {row["label"] for row in entries()}
    assert {
        "GPEN 256", "GPEN 512", "GPEN 1024", "GPEN 2048",
        "GPEN 256 Pro", "GPEN Realistic 256", "GPEN Realistic 512",
        "UltraMax", "CodeFormer", "CodeFormer FP16", "GFPGAN",
        "RestoreFormer++", "DMDNet", "KEEP (sidecar)",
        "Real-ESRGAN x2", "Real-ESRGAN x4", "Real-ESRGAN Anime x4",
        "LSiDIR x4", "UltraSharp x4", "Clear Reality x4", "SPAN x4",
        "Compact ESRGAN x4", "NOMOS 8K x4",
    } <= labels
    assert len(entries()) == 29


def test_matrix_keeps_missing_measurements_pending_per_hardware():
    a = {"gpu_name": "NVIDIA GeForce RTX 3060", "architecture": "Ampere",
         "compute_capability": "8.6", "vram_total_gb": 6.0,
         "driver_version": "a", "cuda_version": "12", "tensorrt_version": "10",
         "onnxruntime_version": "1", "fp16_supported": True}
    b = dict(a, gpu_name="NVIDIA GeForce RTX 4070", architecture="Ada Lovelace",
             compute_capability="8.9", vram_total_gb=12.0)
    ma = create_matrix(a, {"gpen_256": {"FPS": 10.0, "status": "measured"}},
                       include_adjacent=False)
    mb = create_matrix(b, include_adjacent=False)
    ra = next(row for row in ma if row["id"] == "gpen_256")
    rb = next(row for row in mb if row["id"] == "gpen_256")
    assert ra["status"] == "measured"
    assert ra["hardware_key"] == hardware_key(a)
    assert ra["hardware_profile_key"] != hardware_key(a)
    assert rb["status"] == "pending"
    assert ra["hardware_profile_key"] != rb["hardware_profile_key"]
