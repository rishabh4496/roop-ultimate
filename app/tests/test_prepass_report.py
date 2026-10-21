"""What the pre-pass says about itself in the per-job report (Stage 2 of `roop/stage_telemetry.py`).

A render that ran with `prepass_detect_interval` > 1 found fewer faces and swapped fewer frames than the same render
at 1 (measured: docs/perf/phase14/README.md), so the report must be able to say which it was. And a run whose Stage 2
was restored from a checkpoint did not run the pre-pass at all: it must report NOTHING for these fields, never the
previous job's numbers.
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import stage_telemetry as st                                    # noqa: E402

S2 = st.STAGE_2_ACTIVE_PREPASS
FIELDS = ("detect_interval", "detector_calls", "scan_frames", "coast_frames", "flow_backend", "landmark_filter",
          "trajectory_faces")


def _src(rel):
    return open(os.path.join(APP, rel), encoding="utf-8").read()


class Stage2Report(unittest.TestCase):
    def setUp(self):
        st.begin_run(900)
        self.addCleanup(st.end_run)

    def test_the_schedule_the_prepass_ran_reaches_the_report(self):
        st.enter(S2)
        st.put(S2, detect_interval=4, detector_calls=233, scan_frames=900, coast_frames=667, flow_backend="cv",
               landmark_filter="one_euro", trajectory_faces=1177)
        st.leave(S2)
        stage2 = st.end_run(ok=True)["stages"][S2]
        self.assertEqual({k: stage2[k] for k in FIELDS},
                         dict(detect_interval=4, detector_calls=233, scan_frames=900, coast_frames=667,
                              flow_backend="cv", landmark_filter="one_euro", trajectory_faces=1177))
        json.dumps(stage2)                                           # JSON-safe: it goes to /ws/telemetry and the history

    def test_nothing_measured_is_none_never_a_fabricated_zero(self):
        st.enter(S2)
        st.leave(S2)
        stage2 = st.end_run(ok=True)["stages"][S2]
        self.assertEqual({k: stage2[k] for k in FIELDS}, {k: None for k in FIELDS})

    def test_a_restored_stage2_reports_none_for_what_it_did_not_run(self):
        """The batch runner hands the fields over only when Stage 2 actually ran in THIS job."""
        text = _src("roop/procmgr_batch.py")
        self.assertIn("_sched = {} if _stage2_restored else dict(getattr(self, '_prepass_schedule', None) or {})", text)
        self.assertIn("flow_backend=None if _stage2_restored else", text)
        self.assertIn("landmark_filter=None if _stage2_restored else", text)

    def test_each_job_starts_with_no_schedule_of_its_own(self):
        text = _src("roop/procmgr_batch.py")
        for reset in ("self._prepass_schedule = {}", "self._flow_backend_name = None", "self._prepass_filter = None"):
            self.assertIn(reset, text)

    def test_the_prepass_writes_what_the_batch_runner_reads(self):
        tracking = _src("roop/procmgr_tracking.py")
        for written in ("self._prepass_schedule = dict(", "self._prepass_filter = _kind", "self._flow_backend_name = "):
            self.assertIn(written, tracking)

    def test_the_ui_row_reads_the_same_field_names(self):
        js = _src(os.path.join("..", "react-ui", "src", "components", "faceswap", "reportFormat.js"))
        for name in ("detector_calls", "scan_frames", "detect_interval"):
            self.assertIn("s2." + name, js)


if __name__ == "__main__":
    unittest.main()
