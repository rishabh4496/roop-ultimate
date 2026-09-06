"""Regression tests for the opt-in Phase 14 stage profiler."""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.stage_profiler import REQUIRED_STAGES, StageProfiler


def test_cpu_profile_has_required_stage_contract():
    profiler = StageProfiler(gpu_sync=False)
    token = profiler.begin("detect")
    time.sleep(0.001)
    profiler.end(token)
    profiler.record_transfer("swap", "h2d", 128)
    report = profiler.report()

    assert report["schema"] == "roop-phase14-stage-profile-v1"
    assert report["mode"] == "event-only"
    assert report["required_stages"] == list(REQUIRED_STAGES)
    assert report["canonical_stages"]["detection"]["status"] == "measured"
    assert report["canonical_stages"]["detection"]["calls"] == 1
    assert report["canonical_stages"]["swap"]["transfer_h2d_bytes"] == 128
    assert report["canonical_stages"]["swap"]["transfer_attribution"].startswith("explicit")


def test_empty_profile_is_explicit_about_missing_measurements():
    report = StageProfiler().report()
    for name in REQUIRED_STAGES:
        stage = report["canonical_stages"][name]
        assert stage["status"] == "not_observed"
        assert stage["calls"] == 0
        assert stage["cpu_ms_total"] == 0.0
        assert stage["transfer_h2d_bytes"] == 0
        assert stage["transfer_d2h_bytes"] == 0


def test_end_none_is_safe_for_optional_profiler_hook():
    StageProfiler().end(None)


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
