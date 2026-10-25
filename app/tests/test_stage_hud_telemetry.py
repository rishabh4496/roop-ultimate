"""Per-stage HUD telemetry: Stage 1 keyframes/s + scene cuts, Stage 2 flow-tracked vs neural-detected frames,
Stage 3 ms per operation with the backend each ran on, and the OpenVINO `aux_offload` block.

Each field is driven through the writer production uses (`procmgr_runtime._prof` for the Stage 3 ops) and read
back through the two consumers (`build_summary` for the per-job report, `snapshot` for the live frame). A counter
nothing increments, or a label nothing reads, looks wired (AGENTS.md: "Something reports success while not
running").
"""
import os
import sys
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import procmgr_runtime, stage_telemetry as st  # noqa: E402

S1, S2, S3 = st.STAGE_1_SPARSE_PRESCAN, st.STAGE_2_ACTIVE_PREPASS, st.STAGE_3_MAIN_SWAP


class _Base(unittest.TestCase):
    def setUp(self):
        st.reset()
        self.t = [1000.0]
        patcher = mock.patch.object(st.time, "time", lambda: self.t[0])
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(st.reset)
        st._vram.stop()
        st._vram._probe = lambda: None
        vram_start = mock.patch.object(st._vram, "start", lambda: st._vram.reset())
        vram_start.start()
        self.addCleanup(vram_start.stop)

    def advance(self, seconds):
        self.t[0] += seconds


class Stage3Ops(_Base):
    def test_ops_are_recorded_through_prof_only_while_stage3_runs(self):
        st.begin_run(10)
        with procmgr_runtime._prof("swap"):           # Stage 3 not entered yet: must not be counted
            pass
        st.enter(S3, 10)
        for _ in range(3):
            with procmgr_runtime._prof("swap"):
                time.sleep(0.002)
        with procmgr_runtime._prof("blend"):
            time.sleep(0.002)
        st.leave(S3)
        with procmgr_runtime._prof("swap"):           # left again: must not be counted
            pass
        ops = st.build_summary()["stages"][S3]["ops"]
        self.assertEqual(ops["model"]["calls"], 3)
        self.assertGreaterEqual(ops["model"]["ms"], 1.5)
        self.assertEqual(ops["blend"]["calls"], 1)

    def test_a_column_that_never_ran_is_none_not_zero(self):
        st.begin_run(10)
        st.enter(S3, 10)
        with procmgr_runtime._prof("decode"):
            pass
        ops = st._ops_report()
        self.assertIsNotNone(ops["decode"])
        for col in ("warp", "model", "restore", "blend", "encode"):
            self.assertIsNone(ops[col], col)

    def test_each_column_maps_to_a_stage_name_production_actually_uses(self):
        """OPS_COLUMNS names `_prof` stages; a name nothing emits would be a permanently empty HUD column."""
        src = ""
        for rel in ("roop/ProcessMgr.py", "roop/procmgr_batch.py", "roop/procmgr_runtime.py"):
            with open(os.path.join(APP, rel), encoding="utf-8", errors="replace") as fh:
                src += fh.read()
        for column, name in st.OPS_COLUMNS:
            self.assertTrue(f"_prof('{name}')" in src or f'_prof("{name}")' in src,
                            f"column {column}: no _prof('{name}') site in the pipeline")

    def test_a_new_stage3_starts_from_zero(self):
        st.begin_run(10)
        st.enter(S3, 10)
        with procmgr_runtime._prof("swap"):
            pass
        st.leave(S3)
        st.enter(S3, 10)
        self.assertIsNone(st._ops_report()["model"])

    def test_reset_and_end_run_switch_collection_off(self):
        st.begin_run(10)
        st.enter(S3, 10)
        self.assertTrue(st._ops_on)
        st.end_run(ok=True)
        self.assertFalse(st._ops_on)
        st.enter(S3, 10)
        st.reset()
        self.assertFalse(st._ops_on)
        self.assertIsNone(st._ops_report()["model"])

    def _labels(self, providers, encoder="h264_nvenc"):
        import roop.globals as g
        with mock.patch.object(g, "execution_providers", providers, create=True),                 mock.patch.object(g, "video_encoder", encoder, create=True),                 mock.patch.object(st, "_video_io_snapshot", return_value=None):
            return st._backends()

    def test_backend_labels_are_honest(self):
        labels = self._labels(["TensorrtExecutionProvider", "CPUExecutionProvider"])
        # The GPU warp/paste has no production caller, so warp / blend never claim a GPU backend.
        self.assertEqual(labels["warp"], "cv2 (CPU)")
        self.assertEqual(labels["blend"], "cv2 (CPU)")
        self.assertEqual(labels["model"], "TensorRT")
        self.assertEqual(labels["encode"], "h264_nvenc")

    def test_a_provider_given_as_a_name_options_pair_is_still_named(self):
        """Production passes (name, options) pairs for TensorRT; str() of one never read as TensorRT."""
        labels = self._labels([("TensorrtExecutionProvider", {"trt_fp16_enable": True}), "CPUExecutionProvider"])
        self.assertEqual(labels["model"], "TensorRT")
        self.assertEqual(self._labels([("CUDAExecutionProvider", {})])["model"], "CUDA")
        self.assertEqual(self._labels(["CPUExecutionProvider"])["model"], "CPU")

    def test_snapshot_has_ops_only_once_stage3_started(self):
        st.begin_run(10)
        self.assertNotIn("ops", st.snapshot()["stages"][S3])
        st.enter(S3, 10)
        with procmgr_runtime._prof("alignment"):
            pass
        live = st.snapshot()["stages"][S3]["ops"]
        self.assertEqual(live["warp"]["calls"], 1)
        self.assertEqual(live["warp"]["backend"], "cv2 (CPU)")


class Stage1And2(_Base):
    def _summary_stage(self, name):
        return st.build_summary()["stages"][name]

    def test_stage1_keyframes_per_second_and_cuts(self):
        st.begin_run(600)
        st.enter(S1)
        st.put(S1, sampled_frames=40, samples_total=40, scene_cuts=3, scan_mode="hardware-keyframe")
        self.advance(4.0)
        st.leave(S1)
        s1 = self._summary_stage(S1)
        self.assertEqual(s1["keyframes_per_sec"], 10.0)
        self.assertEqual(s1["scene_cuts"], 3)
        self.assertEqual(s1["scan_mode"], "hardware-keyframe")

    def test_stage1_rate_is_none_when_nothing_was_sampled(self):
        st.begin_run(600)
        st.enter(S1)
        self.advance(1.0)
        st.leave(S1)
        self.assertIsNone(self._summary_stage(S1)["keyframes_per_sec"])

    def test_stage2_flow_vs_neural_split(self):
        st.begin_run(600)
        st.enter(S2)
        st.put(S2, detector_calls=25, coast_frames=75, scan_frames=100)
        self.advance(5.0)
        st.leave(S2)
        s2 = self._summary_stage(S2)
        self.assertEqual(s2["neural_detected_frames"], 25)
        self.assertEqual(s2["flow_tracked_frames"], 75)
        self.assertEqual(s2["flow_ratio_pct"], 75.0)
        self.assertEqual(s2["fps"], 20.0)

    def test_stage2_ratio_is_none_without_either_count(self):
        self.assertIsNone(st._ratio_pct(None, None))
        self.assertIsNone(st._ratio_pct(0, 0))
        self.assertEqual(st._ratio_pct(0, 10), 0.0)

    def test_live_snapshot_carries_stage_fps_and_stage2_split(self):
        st.begin_run(600)
        st.enter(S2)
        st.put(S2, detector_calls=10, coast_frames=30)
        st.progress(S2, 80, 200)
        self.advance(4.0)
        live = st.snapshot()["stages"][S2]
        self.assertEqual(live["fps"], 20.0)
        self.assertEqual(live["flow_ratio_pct"], 75.0)
        self.assertIsNone(st.snapshot()["stages"][S3]["fps"])      # a pending stage has no rate

    def test_live_snapshot_names_the_flow_engine_only_when_one_was_reported(self):
        # The HUD labels Stage 2's tracked share "NVOFA tracked" from this field; without it, plain "flow".
        st.begin_run(600)
        st.enter(S2)
        st.put(S2, detector_calls=10, coast_frames=30)
        self.assertIsNone(st.snapshot()["stages"][S2]["flow_backend"])
        st.put(S2, flow_backend="nvof")
        self.assertEqual(st.snapshot()["stages"][S2]["flow_backend"], "nvof")
        self.assertEqual(st.build_summary()["stages"][S2]["flow_backend"], "nvof")   # the finished report agrees


class AuxOffload(_Base):
    def test_absent_when_the_runner_was_never_loaded(self):
        with mock.patch.dict(sys.modules):
            sys.modules.pop("roop.openvino_runner", None)
            self.assertIsNone(st._aux_offload())

    def test_reported_only_when_a_session_exists(self):
        fake = mock.Mock()
        fake.report.return_value = {"sessions": []}
        with mock.patch.dict(sys.modules, {"roop.openvino_runner": fake}):
            self.assertIsNone(st._aux_offload())
            fake.report.return_value = {"sessions": [{"family": "arcface", "device": "NPU"}]}
            self.assertEqual(st._aux_offload()["sessions"][0]["family"], "arcface")

    def test_a_raising_report_cannot_take_down_the_summary(self):
        fake = mock.Mock()
        fake.report.side_effect = RuntimeError("boom")
        with mock.patch.dict(sys.modules, {"roop.openvino_runner": fake}):
            self.assertIsNone(st._aux_offload())
            self.assertIn("aux_offload", st.build_summary())


if __name__ == "__main__":
    unittest.main()
