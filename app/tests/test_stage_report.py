"""Phase 9: the per-job report (stage wall clocks, seek counts, passthrough / GPU frames, peak VRAM, Stage 4,
speedup estimate) and the three places it must reach: `/ws/telemetry`, the history entry, `/api/runtime/state`.

Every number is checked at the site that produces it and again at the consumer that reads it. A metric
nothing reads, or a counter nothing increments, looks completely wired (AGENTS.md: "Something reports success
while not running"), so each counter has a test that drives the real writer.
"""
import json
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import range_seek, stage_telemetry as st  # noqa: E402
from roop.vram_peak import GIB, PeakSampler  # noqa: E402

S1, S2, S3 = st.STAGE_1_SPARSE_PRESCAN, st.STAGE_2_ACTIVE_PREPASS, st.STAGE_3_MAIN_SWAP


class Clock:
    """A settable time.time for the registry (wall clocks must not depend on the machine's speed)."""
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class _Base(unittest.TestCase):
    def setUp(self):
        st.reset()
        self.clock = Clock()
        patcher = mock.patch.object(st.time, "time", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(st.reset)
        # No real sampler thread, no torch: the registry's sampler gets a scripted probe.
        self.probe_values = []
        st._vram.stop()
        st._vram._probe = lambda: self.probe_values.pop(0) if self.probe_values else None
        self.addCleanup(lambda: setattr(st._vram, "_probe", __import__("roop.vram_peak", fromlist=["x"]).default_probe))
        mock.patch.object(st._vram, "start", lambda: st._vram.reset()).start()
        self.addCleanup(mock.patch.stopall)

    def run_pipeline(self, total=600, active=200, s1=2.0, s2=10.0, s3=30.0, retired=None):
        st.begin_run(total)
        st.enter(S1)
        self.clock.t += s1
        st.set_ranges(2, active, total)
        st.put(S1, sampled_frames=21, samples_total=21, stride=30, workers=4, worker_utilization_pct=63.5)
        st.leave(S1)
        st.enter(S2)
        self.clock.t += s2
        st.set_verified(180)
        st.put(S2, grab_calls=410, set_calls=2, seek_walks=1)
        st.leave(S2)
        st.enter(S3, total)
        for _ in range(total - active):
            st.add(S3, "passthrough_frames")
        for _ in range(active):
            st.add(S3, "gpu_frames")
        for _ in range(active - 10):
            st.add(S3, "swapped_frames")
        # The output writer accepts every frame of the trim (ProcessMgr's three write sites call `retire`).
        st.retire(total if retired is None else retired)
        self.clock.t += s3
        st.put(S3, faces_swapped=190)
        st.leave(S3)


class Wall(_Base):
    def test_stage_wall_clocks_are_measured_not_guessed(self):
        self.run_pipeline(s1=2.0, s2=10.0, s3=30.0)
        rep = st.end_run(ok=True)
        stages = rep["stages"]
        self.assertEqual(stages[S1]["wall_s"], 2.0)
        self.assertEqual(stages[S2]["wall_s"], 10.0)
        self.assertEqual(stages[S3]["wall_s"], 30.0)
        self.assertEqual(rep["total_frames"], 600)
        self.assertEqual(rep["effective_fps"], round(600 / 42.0, 2))

    def test_a_running_stage_reports_its_elapsed_time_live(self):
        st.begin_run(100)
        st.enter(S2)
        self.clock.t += 4.5
        self.assertEqual(st.snapshot()["stages"][S2]["wall_s"], 4.5)

    def test_an_unmeasured_stage_is_none_never_zero(self):
        st.begin_run(100)
        st.skip(S1)
        st.skip(S2)
        st.enter(S3, 100)
        self.clock.t += 3
        st.leave(S3)
        rep = st.end_run(ok=True)
        self.assertIsNone(rep["stages"][S1]["wall_s"])
        self.assertIsNone(rep["stages"][S1]["sampled_frames"])
        self.assertIsNone(rep["stages"][S2]["grab_calls"])
        self.assertIsNone(rep["stages"][S2]["set_calls"])
        self.assertIsNone(rep["stages"][S4_KEY])      # no finalize recorded


S4_KEY = st.STAGE_4_FINALIZE


class Counters(_Base):
    def test_counts_land_only_while_their_stage_is_running(self):
        st.begin_run(10)
        st.add(S3, "passthrough_frames")                 # stage 3 PENDING: ignored (a preview swap)
        st.enter(S3, 10)
        st.add(S3, "passthrough_frames", 3)
        st.leave(S3)
        st.add(S3, "passthrough_frames")                 # a late worker after the stage: ignored
        self.assertEqual(st.build_summary()["stages"][S3]["passthrough_frames"], 3)

    def test_passthrough_plus_gpu_frames_account_for_every_frame(self):
        self.run_pipeline(total=600, active=200)
        s3 = st.end_run(ok=True)["stages"][S3]
        self.assertEqual(s3["passthrough_frames"], 400)
        self.assertEqual(s3["gpu_frames"], 200)
        self.assertEqual(s3["passthrough_frames"] + s3["gpu_frames"], 600)
        self.assertEqual(s3["swapped_frames"], 190)
        self.assertEqual(s3["faces_swapped"], 190)

    def test_stage1_and_stage2_facts_come_through(self):
        self.run_pipeline()
        stages = st.end_run(ok=True)["stages"]
        self.assertEqual((stages[S1]["sampled_frames"], stages[S1]["workers"],
                          stages[S1]["worker_utilization_pct"]), (21, 4, 63.5))
        self.assertEqual(stages[S1]["active_ranges_count"], 2)
        self.assertEqual(stages[S1]["skipped_frames_pct"], round(100.0 * 400 / 600, 1))
        self.assertEqual((stages[S2]["grab_calls"], stages[S2]["set_calls"]), (410, 2))
        self.assertEqual(stages[S2]["verified_target_frames"], 180)


class Seeks(unittest.TestCase):
    """cap.set vs cap.grab, counted at the one class every Stage 2 reader goes through."""

    class FakeCap:
        def __init__(self):
            self.sets, self.grabs = 0, 0

        def set(self, *_):
            self.sets += 1

        def grab(self):
            self.grabs += 1
            return True

        def read(self):
            return True, object()

    def test_a_long_gap_is_one_set_and_a_short_gap_is_grabs(self):
        range_seek.reset_counts()
        cap = self.FakeCap()
        seeker = range_seek.RangeSeeker(cap, sequential=False, seq_gap_max=10)
        seeker.goto(5)                                   # gap 5 <= 10: walked with grab()
        seeker.goto(500)                                 # gap 495 > 10: one cap.set
        seeker.goto(502)                                 # gap 2: walked
        totals = range_seek.counts()
        self.assertEqual(totals, {"set_calls": 1, "grab_calls": 7, "seek_walks": 2})
        self.assertEqual((cap.sets, cap.grabs), (1, 7), "the counters must equal what the capture really saw")

    def test_a_backward_move_re_opens_at_zero_with_one_set(self):
        range_seek.reset_counts()
        cap = self.FakeCap()
        seeker = range_seek.RangeSeeker(cap, sequential=True, seq_gap_max=10)
        seeker.goto(20)
        seeker.goto(8)                                   # backward: set(0) then walk 8
        self.assertEqual(range_seek.counts()["set_calls"], cap.sets)
        self.assertEqual(range_seek.counts()["grab_calls"], cap.grabs)


class Vram(unittest.TestCase):
    def test_peak_is_the_largest_used_value_per_stage_and_overall(self):
        feed = iter([(8 * GIB, 12 * GIB), (5 * GIB, 12 * GIB), (2 * GIB, 12 * GIB), (6 * GIB, 12 * GIB)])
        stage = {"v": S2}
        sampler = PeakSampler(probe=lambda: next(feed), stage_of=lambda: stage["v"])
        sampler.sample()                                 # used 4 (base), stage 2
        sampler.sample()                                 # used 7, stage 2
        stage["v"] = S3
        sampler.sample()                                 # used 10, stage 3
        sampler.sample()                                 # used 6, stage 3
        rep = sampler.report()
        self.assertEqual((rep["base_gb"], rep["peak_gb"], rep["total_gb"]), (4.0, 10.0, 12.0))
        self.assertEqual(rep["by_stage_gb"], {S2: 7.0, S3: 10.0})
        self.assertEqual(rep["samples"], 4)

    def test_no_cuda_is_none_not_zero(self):
        sampler = PeakSampler(probe=lambda: None)
        sampler.sample()
        rep = sampler.report()
        self.assertIsNone(rep["peak_gb"])
        self.assertEqual(rep["samples"], 0)

    def test_a_raising_probe_cannot_take_down_the_run(self):
        from roop import vram_peak
        with mock.patch.dict(sys.modules, {"torch": mock.Mock(cuda=mock.Mock(
                is_available=mock.Mock(side_effect=RuntimeError("driver")), mem_get_info=mock.Mock()))}):
            self.assertIsNone(vram_peak.default_probe())

    def test_the_registry_report_carries_peak_vram_for_stage3(self):
        st.reset()
        probe = iter([(10 * GIB, 12 * GIB), (3 * GIB, 12 * GIB)])
        with mock.patch.object(st._vram, "_probe", lambda: next(probe)), \
                mock.patch.object(st._vram, "start", lambda: st._vram.reset()):
            st.begin_run(10)
            st.enter(S3, 10)
            st._vram.sample()
            st._vram.sample()
            st.leave(S3)
            rep = st.build_summary()
        st.reset()
        self.assertEqual(rep["stages"][S3]["peak_vram_gb"], 9.0)
        self.assertEqual((rep["vram"]["base_gb"], rep["vram"]["peak_gb"]), (2.0, 9.0))


class Speedup(unittest.TestCase):
    def test_estimate_arithmetic(self):
        # Stage 2 + 3 = 40 s for 200 active frames -> a dense render of 600 frames ~ 120 s; ran in 42 s.
        out = st.estimate_speedup(2.0, 10.0, 30.0, 600, 200)
        self.assertEqual((out["baseline_s"], out["actual_s"], out["multiplier"]), (120.0, 42.0, 2.86))
        self.assertTrue(out["estimated"])
        self.assertIn("not run", out["method"])

    def test_no_sparse_gain_is_below_one_because_stage1_is_overhead(self):
        out = st.estimate_speedup(3.0, 10.0, 30.0, 600, 600)
        self.assertEqual(out["multiplier"], round(40.0 / 43.0, 2))
        self.assertLess(out["multiplier"], 1.0)

    def test_dense_run_with_the_scan_skipped_is_exactly_one(self):
        self.assertEqual(st.estimate_speedup(None, None, 30.0, 600, None)["multiplier"], 1.0)

    def test_a_restored_stage_has_no_estimate(self):
        out = st.estimate_speedup(None, None, 30.0, 600, 200, restored=True)
        self.assertIsNone(out["multiplier"])
        self.assertIn("checkpoint", out["reason"])

    def test_an_unfinished_stage3_has_no_estimate(self):
        self.assertIsNone(st.estimate_speedup(2.0, 10.0, None, 600, 200)["multiplier"])


class Report(_Base):
    def finalize_result(self, ok=True):
        return {"stage": st.STAGE_4_FINALIZE, "ok": ok, "output": "o.mp4", "wall_s": 0.8, "mux_s": 0.6,
                "expected_frames": 600, "problems": [] if ok else ["the audio mux failed"], "orphans": [],
                "temp_deleted": ok,
                "verify": {"frames": 600, "audio_present": True, "audio_start_offset_ms": 1.2,
                           "color": {"color_space": "bt709"}, "color_preserved": True, "bytes": 123}}

    def test_stage4_is_in_the_report_and_a_failed_one_fails_the_job(self):
        self.run_pipeline()
        st.set_finalize(self.finalize_result(ok=True))
        rep = st.end_run(ok=True)
        self.assertTrue(rep["ok"])
        s4 = rep["stages"][S4_KEY]
        self.assertEqual((s4["state"], s4["frames"], s4["audio_start_offset_ms"], s4["color_preserved"]),
                         ("DONE", 600, 1.2, True))
        # effective fps includes the finalize time
        self.assertEqual(rep["effective_fps"], round(600 / (42.0 + 0.8), 2))

    def test_failed_finalize_makes_the_report_not_ok_and_keeps_the_reason(self):
        self.run_pipeline()
        st.set_finalize(self.finalize_result(ok=False))
        rep = st.end_run(ok=True)
        self.assertFalse(rep["ok"])
        self.assertEqual(rep["stages"][S4_KEY]["state"], "FAILED")
        self.assertEqual(rep["stages"][S4_KEY]["problems"], ["the audio mux failed"])

    def test_the_report_is_json_safe_and_end_run_is_idempotent(self):
        self.run_pipeline()
        a = st.end_run(ok=True)
        json.dumps(a)
        b = st.end_run(ok=True)
        self.assertEqual(a["stages"][S3], b["stages"][S3])

    def test_a_new_run_forgets_the_last_report(self):
        self.run_pipeline()
        st.end_run(ok=True)
        self.assertTrue(st.summary())
        st.begin_run(50)
        self.assertEqual(st.summary(), {})

    def test_snapshot_carries_the_summary_only_after_the_run_ended(self):
        self.run_pipeline()
        self.assertEqual(st.snapshot()["summary"], {})
        st.end_run(ok=True)
        self.assertEqual(st.snapshot()["summary"]["total_frames"], 600)

    def test_an_aborted_run_still_produces_a_report(self):
        st.begin_run(100)
        st.enter(S3, 100)
        self.clock.t += 2
        rep = st.end_run(ok=False)                       # a stage still RUNNING is cut short
        self.assertFalse(rep["ok"])
        self.assertEqual(rep["stages"][S3]["state"], st.FAILED)
        self.assertEqual(rep["stages"][S3]["wall_s"], 2.0)


class Consumers(unittest.TestCase):
    """The three readers: /ws/telemetry, the history entry, /api/runtime/state."""

    @classmethod
    def setUpClass(cls):
        import api
        from fastapi.testclient import TestClient
        cls.api = api
        cls.client = TestClient(api.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def setUp(self):
        st.reset()
        self.addCleanup(st.reset)
        mock.patch.object(st._vram, "start", lambda: st._vram.reset()).start()
        self.addCleanup(mock.patch.stopall)

    def _finished_run(self):
        st.begin_run(300)
        st.enter(S3, 300)
        st.add(S3, "gpu_frames", 300)
        st.leave(S3)
        st.set_finalize({"stage": st.STAGE_4_FINALIZE, "ok": True, "wall_s": 0.5, "problems": [], "orphans": [],
                         "verify": {"frames": 300}})
        return st.end_run(ok=True)

    def test_the_report_reaches_the_websocket(self):
        self._finished_run()
        with self.client.websocket_connect("/ws/telemetry") as ws:
            hello = ws.receive_json()
        self.assertEqual(hello["summary"]["total_frames"], 300)
        self.assertEqual(hello["summary"]["stages"][S3]["gpu_frames"], 300)
        self.assertEqual(hello["summary"]["stages"][S4_KEY]["frames"], 300)

    def test_the_report_is_absent_from_an_idle_frame_and_present_after(self):
        idle = self.api._telemetry_snapshot()
        self.assertNotIn("summary", idle)      # idle snapshot stays the constant {"stage": None, "stages": {}}
        self._finished_run()
        self.assertIn("summary", self.api._telemetry_snapshot())

    def test_the_history_entry_carries_the_report(self):
        report = self._finished_run()
        tmp = tempfile.mkdtemp()
        hist = os.path.join(tmp, "run_history.json")
        with mock.patch.object(self.api, "HISTORY_FILE", hist):
            self.api._record_run_history({"enhancer": "none"}, [os.path.join(tmp, "out.mp4")])
            entries = self.api._load_history()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["report"]["stages"][S3]["gpu_frames"], 300)
        self.assertEqual(entries[0]["report"]["total_frames"], report["total_frames"])
        json.dumps(entries[0])

    def test_a_run_with_no_report_records_none_rather_than_an_empty_one(self):
        tmp = tempfile.mkdtemp()
        hist = os.path.join(tmp, "run_history.json")
        with mock.patch.object(self.api, "HISTORY_FILE", hist):
            self.api._record_run_history({"enhancer": "none"}, [os.path.join(tmp, "out.mp4")])
            entries = self.api._load_history()
        self.assertNotIn("report", entries[0])

    def test_runtime_state_keeps_14_sections_and_carries_the_report(self):
        from roop import runtime_state
        before = runtime_state.snapshot()
        self.assertEqual(len(before["sections"]), 14)
        self.assertEqual(before["sections"]["PERFORMANCE"]["values"]["pipeline_report"], runtime_state.UNKNOWN)
        self._finished_run()
        after = runtime_state.snapshot()
        self.assertEqual(len(after["sections"]), 14)
        self.assertEqual(after["sections"]["PERFORMANCE"]["values"]["pipeline_report"]["total_frames"], 300)


if __name__ == "__main__":
    unittest.main()


class CountingSuppression(unittest.TestCase):
    """`ProcessMgr._count_stage3` counts frames WRITTEN: warm-up priming and OOM reruns must not count.

    The first version counted inside `process_frame` unconditionally and read 624 frames for a 600-frame trim
    (tests/verify_stage4_report.py R1), because a parallel stabilization block re-runs its warm-up frames
    through `process_frame` and discards them."""

    def setUp(self):
        st.reset()
        self.addCleanup(st.reset)
        mock.patch.object(st._vram, "start", lambda: st._vram.reset()).start()
        self.addCleanup(mock.patch.stopall)
        import threading
        from roop.ProcessMgr import ProcessMgr
        self.PM = ProcessMgr
        self.fake = mock.Mock()
        self.fake._tls = threading.local()
        self.fake._count_stage3 = lambda key: ProcessMgr._count_stage3(self.fake, key)
        st.begin_run(10)
        st.enter(S3, 10)

    def passthrough(self):
        return (st.build_summary()["stages"][S3].get("passthrough_frames") or 0)

    def test_a_counted_frame_counts(self):
        self.PM._count_stage3(self.fake, "passthrough_frames")
        self.assertEqual(self.passthrough(), 1)

    def test_the_flag_suppresses_counting_and_is_per_thread(self):
        import threading
        self.fake._tls.stage_count_off = True
        self.PM._count_stage3(self.fake, "passthrough_frames")
        self.assertEqual(self.passthrough(), 0)
        seen = []
        t = threading.Thread(target=lambda: (self.PM._count_stage3(self.fake, "passthrough_frames"), seen.append(1)))
        t.start()
        t.join()
        self.assertEqual(self.passthrough(), 1, "another thread's frame must still count")

    def test_an_uncounted_rerun_restores_the_flag_even_when_the_frame_raises(self):
        pm = self.fake
        pm._tls.stage_count_off = False

        def boom(frame, **kw):
            self.assertTrue(pm._tls.stage_count_off, "the rerun must run with counting off")
            raise RuntimeError("oom again")

        pm.process_frame = boom
        with self.assertRaises(RuntimeError):
            self.PM._process_frame_uncounted(pm, object(), frame_idx=3)
        self.assertFalse(pm._tls.stage_count_off, "a raised rerun left counting switched off for the next frame")
