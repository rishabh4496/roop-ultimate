import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.hardware_validation import (  # noqa: E402
    REQUIRED_METRICS,
    build_dual_target_report,
    classify_optimization,
    hardware_profile_key,
)


def _hardware(free=4.0):
    return {
        "device_id": 0,
        "gpu_name": "NVIDIA GeForce RTX 3060",
        "gpu_vendor": "nvidia",
        "architecture": "Ampere",
        "compute_capability": "8.6",
        "vram_total_gb": 6.0,
        "vram_tier": "small",
        "vram_available_gb": free,
        "driver_version": "test",
        "cuda_version": "12.8",
        "tensorrt_version": "10.9",
        "onnxruntime_version": "1.23",
        "tensor_core_capabilities": ["fp16"],
        "fp16_supported": True,
        "bf16_supported": False,
        "int8_supported": True,
        "fp8_supported": False,
        "nvdec_available": True,
        "nvdec_codecs": ["h264_cuvid"],
        "nvenc_available": True,
        "nvenc_codecs": ["h264_nvenc"],
        "ram_total_gb": 16.0,
        "platform": "test",
    }


def test_profile_key_ignores_transient_available_vram():
    a = hardware_profile_key(_hardware(4.0))
    b = hardware_profile_key(_hardware(1.0))
    assert a == b


def test_dual_target_report_keeps_missing_target_pending():
    report = build_dual_target_report({
        "RTX 3060": {
            "status": "measured",
            "hardware": _hardware(),
            "hardware_profile_key": hardware_profile_key(_hardware()),
            "final_fps": 5.0,
        }
    })
    assert report["separate_tables"] is True
    assert report["targets"]["RTX 3060"]["status"] == "measured_partial"
    assert "baseline_fps" in report["targets"]["RTX 3060"]["missing_metrics"]
    assert report["targets"]["RTX 4070"]["status"] == "pending"
    assert set(report["targets"]["RTX 4070"]["metrics"]) == set(REQUIRED_METRICS)


def test_mismatched_runtime_identity_is_not_accepted():
    record = {
        "status": "measured",
        "hardware": _hardware(),
        "hardware_profile_key": hardware_profile_key(_hardware(), {"model": "a"}),
    }
    report = build_dual_target_report({"RTX 3060": record})
    assert report["targets"]["RTX 3060"]["status"] == "pending"


def test_wrong_physical_target_label_is_not_accepted():
    hardware = _hardware()
    record = {
        "status": "measured",
        "hardware": hardware,
        "hardware_profile_key": hardware_profile_key(hardware),
        **{name: 1 for name in REQUIRED_METRICS},
    }
    report = build_dual_target_report({"RTX 4070": record})
    assert report["targets"]["RTX 4070"]["status"] == "pending"
    assert "does not match" in report["targets"]["RTX 4070"]["notes"]


def test_classification_requires_both_physical_results():
    assert classify_optimization(
        {"status": "measured", "improvement_pct": 10}, None
    ) == "PENDING"
    assert classify_optimization(
        {"status": "measured", "improvement_pct": 10},
        {"status": "measured", "improvement_pct": 5},
    ) == "A. BENEFICIAL ON BOTH"
    assert classify_optimization(
        {"status": "measured", "improvement_pct": 10},
        {"status": "measured", "improvement_pct": -2},
    ) == "E. REGRESSION ON ONE GPU"
    assert classify_optimization(
        {"status": "measured", "improvement_pct": 10},
        {"status": "measured", "improvement_pct": 0},
    ) == "B. RTX 3060-SPECIFIC"
