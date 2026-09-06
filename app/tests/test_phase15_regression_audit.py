"""Unit contracts for the Phase 15 cross-hardware audit."""

from pathlib import Path
from types import SimpleNamespace

from roop.regression_audit import (
    BACKENDS,
    ENHANCERS,
    LIFECYCLE_CHECKS,
    QUALITY_MODES,
    WORKFLOWS,
    build_report,
    core_enhancer_names,
    coverage_rows,
    inspect_cache_roots,
    runtime_capabilities,
    summarize_coverage,
)


class FakeOrt:
    __version__ = "test-ort"

    @staticmethod
    def get_available_providers():
        return ["CPUExecutionProvider", "CUDAExecutionProvider"]


class FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 1

    @staticmethod
    def get_device_name(_index):
        return "test NVIDIA"


class FakeTorch:
    __version__ = "test-torch"
    version = SimpleNamespace(cuda="12.0", hip=None)
    cuda = FakeCuda()


def test_matrix_contains_all_requested_backend_families_and_precisions():
    ids = {row["id"] for row in BACKENDS}
    assert {"cuda_fp32", "cuda_fp16", "cuda_mixed", "tensorrt_fp32", "tensorrt_fp16", "tensorrt_mixed", "rocm_fp32", "directml_fp32", "coreml_fp32", "cpu_fp32"} <= ids
    assert {row["family"] for row in BACKENDS} == {"nvidia", "amd", "directml", "apple", "cpu"}


def test_unavailable_provider_is_not_reported_as_pass():
    report = runtime_capabilities(ort_module=FakeOrt(), torch_module=FakeTorch(), provider_usable=lambda *_args: True)
    rows = {row["id"]: row for row in report["backends"]}
    assert rows["cuda_fp32"]["status"] == "available_not_validated"
    assert rows["tensorrt_fp32"]["status"] == "unavailable"
    assert rows["rocm_fp32"]["status"] == "unavailable"
    assert rows["coreml_fp32"]["status"] == "unavailable"
    assert rows["cpu_fp32"]["status"] == "available_not_validated"


def test_coverage_has_every_workflow_on_every_backend_and_lifecycle_check():
    rows = coverage_rows()
    assert len(rows) == len(BACKENDS) * len(WORKFLOWS) + len(LIFECYCLE_CHECKS)
    assert all(row["status"] == "not_run" for row in rows)
    assert summarize_coverage(rows)["complete"] is False


def test_required_enhancers_and_quality_modes_are_explicit():
    assert {"Adaptive", "GPEN 256 Pro", "GPEN Realistic", "GPEN 256 Ultra", "UltraMax", "KEEP (sidecar)"} <= set(ENHANCERS)
    assert set(core_enhancer_names()) <= set(ENHANCERS)
    assert QUALITY_MODES == ("FAST", "BALANCED", "REALISTIC", "MAX QUALITY")


def test_cache_audit_flags_driverless_and_legacy_namespaces_without_deleting(tmp_path: Path):
    root = tmp_path / "trt_cache"
    stale = root / "fp16_NVIDIA_sm89_drvunknown"
    stale.mkdir(parents=True)
    (stale / "engine.engine").write_bytes(b"test")
    legacy = root / "fp16"
    legacy.mkdir()
    (legacy / "engine.engine").write_bytes(b"test")
    active = root / "fp16_NVIDIA_sm89_drv616.56_trt10_ort1"
    active.mkdir()
    (active / "engine.engine").write_bytes(b"test")

    result = inspect_cache_roots([root], ["fp16_NVIDIA_sm89_drv616.56_trt10_ort1"])
    assert result["stale_candidates"] == 2
    assert any("driver identity" in reason for row in result["entries"] for reason in row["reasons"])
    assert stale.exists()
    assert legacy.exists()


def test_report_keeps_cache_and_execution_gaps_explicit(tmp_path: Path):
    report = build_report(cache_roots=[tmp_path], active_namespaces=["fp32_active"], ort_module=FakeOrt(), torch_module=FakeTorch(), provider_usable=lambda *_args: True)
    assert report["rules"]["availability_is_not_validation"] is True
    assert report["coverage_summary"]["complete"] is False
    assert report["enhancers"]["status"] == "not_run"
    assert report["enhancers"]["missing_from_audit"] == []
    assert report["quality_modes"]["status"] == "not_run"


def load_tests(loader, tests, pattern):
    """Expose this module's bare `test_*` functions to `unittest discover`.

    Without this, `unittest` collects nothing here and reports OK; see
    tests/unittest_shim.py. pytest never calls load_tests, so it is unaffected.
    """
    try:
        from tests.unittest_shim import load_tests_for
    except ImportError:  # discovery started from inside tests/
        from unittest_shim import load_tests_for
    return load_tests_for(globals())
