"""Regression tests for Phase 14 profile parsing and bottleneck reporting.

WHY THIS IS unittest AND NOT pytest. This module used to `import pytest` at the
top. `pytest` is not installed in `app/env`, and the project's suite is run with
`python -m unittest discover`, so the import failed at collection and **every
test in this file silently never ran** -- including
`test_blend_roi_warp_matches_legacy_full_frame`, which is the only check that
`ROOP_BLEND_ROI_WARP` produces output identical to the full-frame path it
replaces. A discovery error looks like an environment complaint, not like an
untested optimization, so it survived.

`pytest.importorskip` and the `monkeypatch` fixture are replaced with the
stdlib equivalents; the assertions themselves are unchanged.
"""

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for path in (APP, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from baseline_controlled import parse_detailed_stage_profile
from phase14_bottleneck_report import compare


def _profile():
    return {
        "schema": "roop-phase14-stage-profile-v1",
        "mode": "event-only",
        "canonical_stages": {
            "detection": {
                "status": "measured", "calls": 2, "cpu_ms_total": 20.0,
                "gpu_event_ms_total": 4.0, "sync_ms_total": 0.0,
                "gpu_sync_window_ms_total": 0.0, "alloc_peak_mb": 10.0,
                "steady_state_allocated_mb": 8.0,
                "steady_state_reserved_mb": 12.0,
                "transfer_h2d_bytes": 0, "transfer_d2h_bytes": 0,
                "transfer_attribution": "opaque",
            }
        },
    }


class Phase14ProfileTest(unittest.TestCase):

    def test_detailed_profile_parser_reads_marked_json(self):
        text = ("==== DETAILED STAGE PROFILE (ROOP_PROFILE_DETAIL) ====\n"
                '{"schema": "roop-phase14-stage-profile-v1"}\n'
                "=======================================================\n")
        self.assertEqual(parse_detailed_stage_profile(text)["schema"],
                         "roop-phase14-stage-profile-v1")

    def test_bottleneck_report_keeps_unobserved_stage_gaps_explicit(self):
        record = {"workload": {"clip_id": "fixture"},
                  "detailed_stage_profile": _profile()}
        report = compare(record)
        self.assertEqual(report["bottlenecks_by_cpu_time"][0]["stage"],
                         "detection")
        self.assertIn("swap", report["measurement"]["required_stage_gaps"])
        self.assertTrue(report["optimization_decision"].startswith("INCOMPLETE"))


class BlendRoiWarpEquivalenceTest(unittest.TestCase):
    """`ROOP_BLEND_ROI_WARP=1` must be pixel-identical to the legacy path.

    This is an optimization that narrows the warp to a region of interest. If it
    is not bit-identical it is a silent quality change on every composited face,
    which is why the assertion is exact equality rather than a tolerance.
    """

    def setUp(self):
        try:
            from roop.procmgr_masking import MaskingMixin
        except Exception as exc:                      # pragma: no cover
            raise unittest.SkipTest("roop.procmgr_masking unavailable: %s" % exc)
        self.MaskingMixin = MaskingMixin
        self._saved = os.environ.get("ROOP_BLEND_ROI_WARP")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("ROOP_BLEND_ROI_WARP", None)
        else:
            os.environ["ROOP_BLEND_ROI_WARP"] = self._saved

    def test_blend_roi_warp_matches_legacy_full_frame(self):
        class Bench(self.MaskingMixin):
            def __init__(self):
                self.options = SimpleNamespace(
                    show_face_area_overlay=False, blend_ratio=1.0)

        rng = np.random.default_rng(14)
        fake = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        source = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        matrix = np.array([[1.0, 0.0, 23.0], [0.0, 1.0, 19.0]], dtype=np.float32)
        bench = Bench()

        os.environ["ROOP_BLEND_ROI_WARP"] = "0"
        legacy = bench.paste_upscale(
            fake, fake, matrix, source.copy(), 1, [0, 0, 0, 0, 0, 0],
            inplace=True)
        os.environ["ROOP_BLEND_ROI_WARP"] = "1"
        optimized = bench.paste_upscale(
            fake, fake, matrix, source.copy(), 1, [0, 0, 0, 0, 0, 0],
            inplace=True)
        self.assertTrue(np.array_equal(optimized, legacy),
                        "ROI warp diverged from the full-frame path")


if __name__ == "__main__":
    unittest.main()
