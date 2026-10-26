"""Every fps in the report divides frames the OUTPUT WRITER accepted, never planned or merely-processed frames.

Before: `processing_fps` and `effective_fps` divided the trim's planned `total_frames` by wall time (a stage that died at
frame 40 of 600 reported 600 / wall), and the live counter ticked when a stabilization worker put a frame in its result
dict, up to a chunk before the ordered writer wrote it. These tests drive the registry's real writer (`retire`), the real
`_run_stab_parallel` writer thread, and scan every `write_frame` site (AGENTS.md: a counter nothing increments looks
completely wired).
"""
import ast
import os
import sys
import threading
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import stage_telemetry as st  # noqa: E402

S1, S2, S3 = st.STAGE_1_SPARSE_PRESCAN, st.STAGE_2_ACTIVE_PREPASS, st.STAGE_3_MAIN_SWAP


class _Clocks:
    """Settable wall clock (`time.time`: stage walls) and monotonic clock (`retire` window)."""
    def __init__(self):
        self.wall = 1000.0
        self.mono = 50.0


class _Base(unittest.TestCase):
    def setUp(self):
        st.reset()
        self.c = _Clocks()
        for name, attr in (("time", "wall"), ("monotonic", "mono")):
            p = mock.patch.object(st.time, name, lambda a=attr: getattr(self.c, a))
            p.start()
            self.addCleanup(p.stop)
        st._vram.stop()
        st._vram._probe = lambda: None
        mock.patch.object(st._vram, "start", lambda: st._vram.reset()).start()
        self.addCleanup(mock.patch.stopall)
        self.addCleanup(st.reset)
        mock.patch.object(st, "_smart_skip_enabled", lambda: True).start()

    def stage3(self, total=600, wall=30.0):
        st.begin_run(total)
        st.skip(S1)
        st.skip(S2)
        st.enter(S3, total)
        self.c.wall += wall


class RetireIsGatedAndCounted(_Base):
    def test_ignored_before_stage3_and_after_it(self):
        st.begin_run(10)
        st.retire(5)                                   # Stage 3 not started: a preview swap
        st.enter(S3, 10)
        st.retire(3)
        st.leave(S3)
        st.retire(4)                                   # a late write after the stage ended
        self.assertEqual(st.stage3_rates()["frames_retired"], 3)

    def test_fps_is_retired_over_wall_not_planned_over_wall(self):
        self.stage3(total=600, wall=30.0)
        st.retire(40)                                  # the stage died at frame 40 of 600
        st.leave(S3, ok=False)
        rep = st.end_run(ok=False)
        s3 = rep["stages"][S3]
        self.assertEqual(s3["frames_retired"], 40)
        self.assertEqual(s3["effective_output_fps"], round(40 / 30.0, 2))
        self.assertEqual(s3["processing_fps"], round(40 / 30.0, 2))
        self.assertNotEqual(s3["processing_fps"], round(600 / 30.0, 2))      # the old formula
        self.assertEqual(rep["effective_fps"], round(40 / 30.0, 2))

    def test_a_stage_nothing_wrote_to_is_none_not_a_made_up_rate(self):
        self.stage3(total=600, wall=30.0)
        st.leave(S3)
        s3 = st.end_run(ok=True)["stages"][S3]
        self.assertEqual(s3["frames_retired"], 0)       # entered, wrote nothing: an honest zero
        self.assertIsNone(s3["effective_output_fps"])
        self.assertIsNone(s3["processing_fps"])
        self.assertIsNone(st.build_summary()["effective_fps"])

    def test_a_stage_that_never_ran_has_no_retired_count(self):
        st.begin_run(100)
        self.assertIsNone(st.stage3_rates())
        s3 = st.build_summary()["stages"][S3]
        self.assertIsNone(s3["frames_retired"])
        self.assertIsNone(s3["effective_output_fps"])

    def test_the_basis_is_named_in_the_report(self):
        self.stage3()
        st.retire(10)
        st.leave(S3)
        rep = st.end_run(ok=True)
        self.assertIn("retired", rep["effective_fps_basis"])
        self.assertIn("retired", rep["stages"][S3]["fps_basis"])


class LiveRate(_Base):
    def test_window_rate_follows_writes_and_decays_when_they_stop(self):
        st.begin_run(1000)
        st.enter(S3, 1000)
        self.assertIsNone(st.retired_view())            # nothing written yet: caller keeps the labelled bar rate
        for _ in range(3):
            self.c.mono += 1.0
            st.retire(20)                               # 20 frames written per second
        self.assertEqual(st.retired_view()["now_fps"], 20.0)
        self.assertEqual(st.retired_view()["frames"], 60)
        self.c.mono += 10.0                             # the writer stalls for a full window
        self.assertEqual(st.retired_view()["now_fps"], 0.0)

    def test_live_stage3_fps_in_the_snapshot_is_the_retired_rate(self):
        self.stage3(total=1000, wall=0.0)
        for _ in range(3):
            self.c.mono += 1.0
            st.retire(20)
        st.set_main_progress(900, 1000)                 # workers finished 900 frames; the writer accepted 60
        s3 = st.snapshot()["stages"][S3]
        self.assertEqual(s3["frames_retired"], 60)
        self.assertEqual(s3["fps"], 20.0)
        self.assertEqual(s3["effective_output_fps"], 20.0)


class SmartSkip(_Base):
    def test_ratio_and_neural_rate_are_counted_from_executed_faces(self):
        self.stage3(wall=30.0)
        st.add(S3, "neural_inferences", 75)
        st.add(S3, "smart_skip_faces", 25)
        st.retire(600)
        st.leave(S3)
        s3 = st.end_run(ok=True)["stages"][S3]
        self.assertEqual(s3["neural_inferences"], 75)
        self.assertEqual(s3["smart_skip_faces"], 25)
        self.assertEqual(s3["smart_skip_ratio_pct"], 25.0)
        self.assertEqual(s3["neural_inference_fps"], 2.5)
        self.assertEqual(s3["effective_output_fps"], 20.0)
        self.assertGreater(s3["effective_output_fps"], s3["neural_inference_fps"])

    def test_no_faces_at_all_is_none(self):
        self.stage3()
        st.retire(600)
        st.leave(S3)
        s3 = st.end_run(ok=True)["stages"][S3]
        self.assertIsNone(s3["smart_skip_ratio_pct"])
        self.assertIsNone(s3["neural_inferences"])
        self.assertIsNone(s3["neural_inference_fps"])

    def test_every_face_skipped_reads_100_percent_not_none(self):
        self.stage3()
        st.add(S3, "smart_skip_faces", 8)
        st.retire(8)
        st.leave(S3)
        self.assertEqual(st.end_run(ok=True)["stages"][S3]["smart_skip_ratio_pct"], 100.0)

    def test_counters_ignore_warm_up_frames_via_the_processmgr_guard(self):
        from roop.ProcessMgr import ProcessMgr
        mgr = ProcessMgr.__new__(ProcessMgr)
        mgr._tls = threading.local()
        self.stage3()
        mgr._count_stage3("neural_inferences")
        mgr._tls.stage_count_off = True                  # a stabilization warm-up / OOM rerun
        mgr._count_stage3("neural_inferences")
        mgr._count_stage3("smart_skip_faces")
        st.leave(S3)
        s3 = st.end_run(ok=True)["stages"][S3]
        self.assertEqual(s3["neural_inferences"], 1)
        self.assertIsNone(s3["smart_skip_faces"])

    def test_patch_is_reusable_only_for_a_real_warpable_patch(self):
        import numpy as np
        from roop.temporal_smart_skip import patch_is_reusable
        self.assertTrue(patch_is_reusable({"patch": np.zeros((4, 4, 3), np.uint8)}))
        self.assertFalse(patch_is_reusable({"patch": np.zeros((0, 4, 3), np.uint8)}))
        self.assertFalse(patch_is_reusable({"patch": [1, 2, 3]}))
        self.assertFalse(patch_is_reusable({"patch": None}))
        self.assertFalse(patch_is_reusable(None))


class ClosingLine(_Base):
    def test_line_carries_the_three_numbers_and_a_short_count(self):
        self.stage3(wall=30.0)
        st.add(S3, "neural_inferences", 75)
        st.add(S3, "smart_skip_faces", 25)
        st.retire(590)
        st.leave(S3)
        line = st.stage3_line(600)
        self.assertIn("retired 590/600 frames", line)       # a short render is printed as short
        self.assertIn("19.67 Effective Output FPS", line)
        self.assertIn("75 neural inferences = 2.5/s", line)
        self.assertIn("Smart-Skip 25.0% of 100 faces", line)

    def test_smart_skip_off_beats_a_computed_zero_percent(self):
        self.stage3(wall=30.0)
        st.add(S3, "neural_inferences", 50)
        st.retire(600)
        st.leave(S3)
        with mock.patch.object(st, "_smart_skip_enabled", lambda: False):
            self.assertIn("Smart-Skip off", st.stage3_line(600))
        self.assertIn("Smart-Skip 0.0% of 50 faces", st.stage3_line(600))     # enabled, nothing reused

    def test_none_before_stage3(self):
        st.begin_run(10)
        self.assertIsNone(st.stage3_line(10))


class WriteSitesRetire(_Base):
    """The claim "frames are counted when written" needs the real writer, not a stand-in for the counter."""

    def _manager(self, writer):
        from roop.ProcessMgr import ProcessMgr
        m = ProcessMgr.__new__(ProcessMgr)
        m._runtime_scheduler = None
        m._stab_chunk_queue_capacity = 1
        m._stab_warmup = 0
        m._runtime_stab_small = False
        m._stab_frame_bytes = 3
        m._kps_stab_factory = m._enh_stab_factory = m._mask_stab_factory = None
        m._tls = threading.local()
        m.output_to_file = writer is not None
        m.output_to_cam = False
        m.videowriter = writer
        m.streamwriter = None
        m._temporal_faces = m._track_assignments = m._precomputed_kps = None
        m._runtime_monitor = None
        m._runtime_adaptive = None
        m.process_frame = lambda frame, frame_idx=None, output_pending=False: frame
        return m

    def _run(self, manager, frames):
        import roop.globals as g
        previous = g.processing
        g.processing = True
        ticks = []
        try:
            with mock.patch.object(type(manager), "_stab_parallel_geometry", return_value=(0, 1, 1, 1)):
                manager._run_stab_parallel("unused", frames, 0, len(frames), len(frames), 1,
                                           lambda: ticks.append(1))
        finally:
            g.processing = previous
        return ticks

    def test_stabilized_writer_retires_each_frame_after_its_write_returns(self):
        seen = []

        class Writer:
            def write_frame(self, frame):
                seen.append(("before", st.stage3_rates()["frames_retired"], len(seen)))

        self.stage3(total=5, wall=1.0)
        manager = self._manager(Writer())
        self._run(manager, [f"f{i}" for i in range(5)])
        # While write N is in progress the counter has only seen the N frames whose writes already returned.
        self.assertEqual([(r, w) for _, r, w in seen], [(i, i) for i in range(5)])
        self.assertEqual(st.stage3_rates()["frames_retired"], 5)

    def test_no_output_means_nothing_is_retired(self):
        self.stage3(total=3, wall=1.0)
        self._run(self._manager(None), ["a", "b", "c"])
        self.assertEqual(st.stage3_rates()["frames_retired"], 0)

    def test_every_write_frame_site_in_processmgr_retires(self):
        """Ratchet: a new `.write_frame(` that does not retire would silently under-report fps."""
        src_path = os.path.join(APP, "roop", "ProcessMgr.py")
        with open(src_path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        def enclosing_function(node):
            while node in parents:
                node = parents[node]
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return node
            return None

        def retires(func):
            return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "retire"
                       and isinstance(n.func.value, ast.Name) and n.func.value.id == "_stage_telemetry"
                       for n in ast.walk(func))

        sites = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "write_frame"]
        self.assertGreaterEqual(len(sites), 3)           # sequential writer thread, unified scheduler, stabilized
        for call in sites:
            func = enclosing_function(call)
            self.assertTrue(func is not None and retires(func),
                            f"ProcessMgr.py:{call.lineno} writes a frame without _stage_telemetry.retire()")


if __name__ == "__main__":
    unittest.main()
