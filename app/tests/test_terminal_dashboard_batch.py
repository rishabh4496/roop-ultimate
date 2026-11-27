"""The terminal dashboard's batch panel, system line and full-width layout (`roop/terminal_dashboard.py`).

Driven through the producers production uses: the queue's own reader (`routes_queue._dashboard_batch_source`), the
manager's `begin_job` / `_tick` / `end_job`, and the registry the render feeds (`roop.stage_telemetry`). The live block
is checked on the same small VT emulator as `test_terminal_dashboard.py`, so "fits the terminal" and "is not
duplicated in the scrollback" are observed on a screen.
"""
import io
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
from roop import terminal_dashboard as td  # noqa: E402
from tests.test_terminal_dashboard import Clock, VtScreen, os_size  # noqa: E402

S1, S2, S3 = st.STAGE_1_SPARSE_PRESCAN, st.STAGE_2_ACTIVE_PREPASS, st.STAGE_3_MAIN_SWAP
T0 = 1_000_000.0

SYS = {"gpu_name": "NVIDIA GeForce RTX 4070", "gpu_util_pct": 91, "gpu_temp_c": 68, "gpu_power_w": 171.0,
       "gpu_power_limit_w": 200.0, "gpu_mem_used_gb": 9.12, "gpu_mem_total_gb": 11.99, "nvenc_pct": 22, "nvdec_pct": 9,
       "cpu_pct": 37.0, "ram_used_gb": 18.4, "ram_total_gb": 31.7, "proc_rss_gb": 6.21, "proc_threads": 74}


def queue_src(current="j3", run_started=T0 - 600, extra=()):
    jobs = [
        {"id": "j1", "name": "a.mp4", "state": "COMPLETED", "started": T0 - 600, "finished": T0 - 400, "output": "a_out.mp4"},
        {"id": "j2", "name": "b.mp4", "state": "COMPLETED", "started": T0 - 400, "finished": T0 - 100},
        {"id": "j3", "name": "c.mp4", "state": "PROCESSING", "started": T0 - 90},
        {"id": "j4", "name": "d.mp4", "state": "QUEUED"},
        {"id": "j5", "name": "e.mp4", "state": "RECOVERABLE"},
    ] + list(extra)
    return {"source": "queue", "jobs": jobs, "current": current, "run_started": run_started, "paused": False}


class BatchView(unittest.TestCase):
    def test_counts_and_the_eta_is_the_mean_of_this_runs_finished_jobs(self):
        live = {"stage": "S3", "eta_s": 30.0, "frames_done": 100, "frames_total": 400, "fps": 10.0, "job_frac": 0.25}
        v = td.build_batch_view(queue_src(), {}, T0, live)
        self.assertEqual(v["counts"], {"done": 2, "failed": 0, "stopped": 0, "running": 1, "queued": 2})
        self.assertEqual(v["current_pos"], 3)
        self.assertEqual(v["per_job_s"], 250.0)                    # (200 + 300) / 2
        self.assertEqual(v["eta_s"], 30 + 2 * 250)                 # what is left of j3 + two queued jobs
        self.assertIn("2 finished", v["eta_basis"])
        self.assertEqual(v["pct"], round(100 * 2.25 / 5, 1))
        self.assertEqual(v["elapsed_s"], 600.0)
        cur = v["rows"][2]
        self.assertTrue(cur["current"])
        self.assertEqual((cur["frames_done"], cur["frames_total"], cur["fps"]), (100, 400, 10.0))
        self.assertEqual(cur["elapsed_s"], 90.0)

    def test_a_job_finished_before_this_run_is_not_in_the_estimate(self):
        old = {"id": "j0", "name": "old.mp4", "state": "COMPLETED", "started": T0 - 99_000, "finished": T0 - 90_000}
        v = td.build_batch_view(queue_src(extra=[old]), {}, T0, {"stage": "S3", "eta_s": 30.0})
        self.assertEqual(v["per_job_s"], 250.0)
        self.assertEqual(v["counts"]["done"], 3)

    def test_with_nothing_finished_the_current_job_projects_itself(self):
        src = queue_src()
        for j in src["jobs"][:2]:
            j.update(state="QUEUED", started=None, finished=None)
        v = td.build_batch_view(src, {}, T0, {"stage": "S3", "eta_s": 30.0})
        self.assertEqual(v["per_job_s"], 120.0)                    # 90 s elapsed + 30 s left
        self.assertEqual(v["eta_s"], 30 + 4 * 120)
        self.assertEqual(v["eta_basis"], "current job's projection")

    def test_no_estimate_is_made_up_before_stage_3_has_a_rate(self):
        src = queue_src()
        for j in src["jobs"][:2]:
            j.update(state="QUEUED", started=None, finished=None)
        v = td.build_batch_view(src, {}, T0, {"stage": "S2", "eta_s": 5.0})
        self.assertIsNone(v["eta_s"])

    def test_a_finished_queue_has_no_eta(self):
        src = queue_src(current=None)
        for j in src["jobs"]:
            j.update(state="COMPLETED", started=T0 - 50, finished=T0 - 10)
        v = td.build_batch_view(src, {}, T0, None)
        self.assertIsNone(v["eta_s"])
        self.assertIsNone(v["eta_basis"])
        self.assertEqual(v["pct"], 100.0)

    def test_a_single_job_is_not_a_batch(self):
        self.assertIsNone(td.build_batch_view({"jobs": [{"id": "x", "state": "PROCESSING"}]}, {}, T0))
        self.assertIsNone(td.build_batch_view(None, {}, T0))

    def test_finished_rows_take_frames_and_fps_from_the_history(self):
        hist = {"j1": {"ok": True, "elapsed_s": 200.0, "frames": 1800, "frames_total": 1800, "fps_avg": 9.0}}
        v = td.build_batch_view(queue_src(), hist, T0, {"stage": "S3", "eta_s": 1.0})
        r = v["rows"][0]
        self.assertEqual((r["frames_done"], r["frames_total"], r["fps"]), (1800, 1800, 9.0))
        self.assertEqual(r["note"], "a_out.mp4")


class BatchRendering(unittest.TestCase):
    def view(self, n_queued=0):
        extra = [{"id": f"q{i}", "name": f"queued_{i}.mp4", "state": "QUEUED"} for i in range(n_queued)]
        return td.build_batch_view(queue_src(extra=extra), {}, T0,
                                   {"stage": "S3", "eta_s": 30.0, "frames_done": 100, "frames_total": 400, "pct": 25.0})

    def test_every_line_fits_the_width_in_both_renderings(self):
        v = self.view(10)
        for width in (90, 120, 160, 220):
            for color in (True, False):
                for ln in td.render_batch_lines(v, color=color, width=width, max_rows=6, now=T0):
                    self.assertLessEqual(td.visible_len(ln), width, (width, color, ln))
                    if not color:
                        self.assertNotIn("\x1b", ln)

    def test_the_rule_and_the_bar_span_the_full_width(self):
        lines = td.render_batch_lines(self.view(), color=True, width=150, now=T0)
        self.assertEqual(td.visible_len(lines[0]), 150)
        self.assertGreaterEqual(td.visible_len(lines[1]), 140)

    def test_a_long_queue_keeps_the_running_job_and_says_what_is_hidden(self):
        v = self.view(20)
        lines = [td.strip_ansi(x) for x in td.render_batch_lines(v, color=True, width=160, max_rows=5, now=T0)]
        body = "\n".join(lines)
        self.assertIn("c.mp4", body)                                 # the running job
        self.assertIn("running", body)
        self.assertIn("more below", lines[-1])
        self.assertIn("queued)", lines[-1])
        self.assertEqual(len(lines), 3 + 5 + 1)                      # rule, bar, header, 5 rows, "more"

    def test_each_state_has_its_own_colour(self):
        v = td.build_batch_view(queue_src(extra=[{"id": "f", "name": "bad.mp4", "state": "FAILED", "error": "boom"}]),
                                {}, T0, {"stage": "S3", "eta_s": 1.0})
        lines = td.render_batch_lines(v, color=True, width=160, now=T0)
        rows = {td.strip_ansi(ln).split()[3]: ln for ln in lines[3:]}
        self.assertIn(td.GREEN, rows["a.mp4"])
        self.assertIn(td.GOLD, rows["c.mp4"])
        self.assertIn(td.SKY, rows["d.mp4"])
        self.assertIn(td.RED, rows["bad.mp4"])
        self.assertIn("boom", td.strip_ansi(rows["bad.mp4"]))

    def test_log_records_are_ascii_key_value(self):
        lines = td.render_batch_log_lines(self.view(), "12:00:00", rows=True)
        self.assertTrue(lines[0].startswith("[Batch] t=12:00:00 job=3/5 done=2 failed=0 stopped=0 running=1 queued=2"))
        self.assertEqual(len(lines), 1 + 5)
        for ln in lines:
            self.assertNotIn("\x1b", ln)

    def test_the_summary_lists_every_job_and_totals(self):
        src = queue_src(current=None, extra=[{"id": f"q{i}", "name": f"q{i}.mp4", "state": "COMPLETED",
                                              "started": T0 - 10, "finished": T0} for i in range(15)])
        for j in src["jobs"]:
            if j["state"] != "COMPLETED":
                j.update(state="COMPLETED", started=T0 - 10, finished=T0)
        hist = {"j1": {"frames": 600, "fps_avg": 3.0}, "j2": {"frames": 900, "fps_avg": 3.0}}
        lines = [td.strip_ansi(x) for x in td.render_batch_summary(td.build_batch_view(src, hist, T0), color=True, width=140)]
        self.assertIn("BATCH FINISHED", lines[0])
        self.assertEqual(sum(1 for ln in lines if ".mp4" in ln), 20)  # no row cap in the summary
        self.assertIn("1,500 frames written", "\n".join(lines))


class Layout(unittest.TestCase):
    def snap(self):
        mgr = td.TerminalProgressManager(clock=Clock(T0), wall=Clock(T0))
        mgr.TICK_S = 3600
        mgr._sys.sample = lambda now: SYS
        with mock.patch.object(td, "_batch_source", lambda: queue_src()), mock.patch.dict(os.environ, {"ROOP_DASHBOARD": "log"}), \
                mock.patch("builtins.print"):
            st.reset()
            mgr.begin_job({"job_id": "j3", "batch_index": 3, "batch_count": 5, "label": "c.mp4", "mode": "video"},
                          td.collect_swap_profile(None, {"job_id": "j3"}, None, 30.0, 400))
            st.begin_run(400)
            st.enter(S3, 400)
            for _ in range(40):
                st.retire()
            mgr._tick()
            snap = mgr.snapshot()
            mgr.end_job(ok=False, _quiet=True)
        st.reset()
        return snap

    def test_the_snapshot_carries_the_batch_and_the_system_load(self):
        snap = self.snap()
        self.assertEqual(snap["batch"]["current_pos"], 3)
        self.assertEqual(snap["batch"]["rows"][2]["frames_done"], 40)
        self.assertEqual(snap["system"]["gpu_util_pct"], 91)

    def test_the_block_never_exceeds_the_terminal_height(self):
        snap = self.snap()
        for height in range(10, 40):
            lines = td.compose_live_lines(snap, color=True, width=140, height=height, footer="footer", batch_rows=50)
            self.assertLessEqual(len(lines), max(height, 12), height)
        tall = td.compose_live_lines(snap, color=True, width=140, height=60, footer="footer", batch_rows=50)
        self.assertTrue(any("e.mp4" in td.strip_ansi(x) for x in tall))

    def test_header_and_rules_span_the_full_width(self):
        snap = self.snap()
        for width in (100, 140, 200):
            lines = td.compose_live_lines(snap, color=True, width=width, height=60)
            self.assertIn(td.visible_len(lines[0]), (width, width - 1), td.strip_ansi(lines[0]))
            rules = [ln for ln in lines if td.strip_ansi(ln).startswith("──")]
            self.assertTrue(rules)
            for ln in rules:
                self.assertEqual(td.visible_len(ln), width)

    def test_stage_bars_grow_with_the_terminal(self):
        snap = self.snap()
        narrow = td.strip_ansi(td.render_stage_lines(snap, color=True, width=110)[2]).count("░")
        wide = td.strip_ansi(td.render_stage_lines(snap, color=True, width=200)[2]).count("░")
        self.assertGreater(wide, narrow)
        for ln in td.render_stage_lines(snap, color=True, width=200):
            self.assertLessEqual(td.visible_len(ln), 200)

    def test_the_system_line_fits_and_drops_detail_when_narrow(self):
        wide = td.render_system_line(SYS, color=True, width=200)
        self.assertIn("RTX 4070", wide)
        self.assertIn("NVDEC", wide)
        narrow = td.render_system_line(SYS, color=True, width=100)
        self.assertLessEqual(td.visible_len(narrow), 100)
        self.assertIn("VRAM 9.12/11.99 GB", td.strip_ansi(narrow))
        self.assertEqual(td.render_system_line({}, color=True), "")
        self.assertTrue(td.render_system_log_line(SYS, "t").startswith("[System] t=t gpu_util_pct=91"))

    def test_the_footer_shortens_instead_of_running_off_the_edge(self):
        self.assertLessEqual(td.visible_len(td.render_hotkey_footer(width=80, batch=True)), 80)
        self.assertIn("[B]", td.render_hotkey_footer(width=200, batch=True))
        self.assertNotIn("[B]", td.render_hotkey_footer(width=200, batch=False))


class ManagerBatch(unittest.TestCase):
    def setUp(self):
        st.reset()
        td.reset()
        self.clock = Clock(T0)
        for p in (mock.patch.object(st.time, "time", self.clock), mock.patch.object(st._vram, "start", lambda: st._vram.reset()),
                  mock.patch.dict(os.environ, {"ROOP_DASHBOARD": "log"})):
            p.start()
            self.addCleanup(p.stop)
        st._vram.stop()
        st._vram._probe = lambda: None
        self.addCleanup(st.reset)
        self.mgr = td.TerminalProgressManager(clock=self.clock, wall=self.clock)
        self.mgr.TICK_S = 3600
        self.mgr._sys.sample = lambda now: SYS
        self.addCleanup(lambda: self.mgr.end_job(ok=False, _quiet=True))

    def run_job(self, job, frames, out=None):
        with mock.patch("builtins.print", side_effect=(lambda *a, **k: out.append(a[0])) if out is not None else None):
            self.mgr.begin_job(job, None)
            st.begin_run(frames)
            st.enter(S3, frames)
            for i in range(frames):
                st.retire()
                if i % 10 == 9:
                    self.clock.advance(1.0)
                    self.mgr._tick()
            st.leave(S3)
            self.clock.advance(1.0)
            self.mgr.end_job(True)
        st.reset()

    def test_files_batch_lists_every_target_and_remembers_finished_ones(self):
        files = ["one.mp4", "two.mp4", "three.mp4"]
        self.run_job({"job_id": None, "label": "one.mp4", "batch_index": 1, "batch_count": 3, "batch_files": files}, 30)
        with mock.patch("builtins.print"):
            self.mgr.begin_job({"label": "two.mp4", "batch_index": 2, "batch_count": 3, "batch_files": files}, None)
            self.clock.advance(5.0)
            self.mgr._tick()
        v = self.mgr.snapshot()["batch"]
        self.assertEqual([r["category"] for r in v["rows"]], ["done", "running", "queued"])
        self.assertEqual(v["rows"][0]["frames_done"], 30)
        self.assertIsNotNone(v["rows"][0]["fps"])
        self.assertEqual(v["rows"][1]["elapsed_s"], 5.0)
        self.assertEqual(v["elapsed_s"], round(self.clock() - T0, 1))  # since job 1 began

    def test_a_busy_queue_keeps_the_last_list_and_never_waits(self):
        answers = [queue_src(), {"busy": True}]
        with mock.patch.object(td, "_batch_source", lambda: answers.pop(0) if answers else {"busy": True}), \
                mock.patch("builtins.print"):
            self.mgr.begin_job({"job_id": "j3", "batch_index": 3, "batch_count": 5}, None)
            for _ in range(3):
                self.clock.advance(1.5)
                self.mgr._tick()
            self.assertEqual(self.mgr.snapshot()["batch"]["total"], 5)

    def test_a_queue_that_stopped_running_drops_the_panel(self):
        answers = [queue_src(), None]
        with mock.patch.object(td, "_batch_source", lambda: answers.pop(0) if answers else None), mock.patch("builtins.print"):
            self.mgr.begin_job({"job_id": "j3"}, None)
            self.assertIsNotNone(self.mgr.snapshot()["batch"])
            self.clock.advance(1.5)
            self.mgr._tick()
            self.assertIsNone(self.mgr.snapshot()["batch"])

    def test_a_source_that_raises_does_not_reach_the_render(self):
        def boom():
            raise RuntimeError("queue exploded")
        with mock.patch.object(td, "_batch_source", boom), mock.patch("builtins.print"):
            self.mgr.begin_job({"job_id": "j3"}, None)
            self.assertTrue(self.mgr.active)
            self.assertIsNone(self.mgr.snapshot()["batch"])

    def test_log_mode_prints_the_job_table_once_then_only_the_summary(self):
        out = []
        with mock.patch.object(td, "_batch_source", lambda: queue_src()):
            self.run_job({"job_id": "j3", "batch_index": 3, "batch_count": 5, "label": "c.mp4"}, 50, out)
        rows = [ln for ln in out if ln.startswith("[Batch] #")]
        heads = [ln for ln in out if ln.startswith("[Batch] t=")]
        self.assertEqual(len(rows), 5)
        self.assertGreaterEqual(len(heads), 2)
        self.assertTrue(any(ln.startswith("[System]") for ln in out))
        self.assertFalse(any("\x1b" in ln for ln in out))
        self.assertEqual(self.mgr._history["j3"]["frames"], 50)

    def test_b_toggles_the_full_list(self):
        self.assertEqual(self.mgr._batch_row_cap(), 8)
        self.assertIn("ALL JOBS", self.mgr.handle_key("b"))
        self.assertEqual(self.mgr._batch_row_cap(), 200)
        self.mgr.handle_key("B")
        with mock.patch.dict(os.environ, {"ROOP_DASHBOARD_BATCH_ROWS": "3"}):
            self.assertEqual(self.mgr._batch_row_cap(), 3)


class LiveHeight(unittest.TestCase):
    def test_a_block_taller_than_the_screen_is_cut_not_smeared_into_the_scrollback(self):
        scr = VtScreen()
        saved = (sys.stdout, sys.stderr)
        sys.stdout, sys.stderr = scr, VtScreen()
        try:
            term = td._LiveTerminal(scr)
            term.install()
            with mock.patch.object(td.shutil, "get_terminal_size", return_value=os_size(80, 10)):
                for n in range(6):
                    term.set_lines([f"L{i}-{n}" for i in range(30)])
            term.uninstall([])
        finally:
            sys.stdout, sys.stderr = saved
        self.assertEqual([r for r in scr.screen() if r], [])         # erased cleanly: nothing left behind
        self.assertEqual(term.drawn, 0)


class QueueSource(unittest.TestCase):
    """routes_queue's reader: the producer the batch panel reads in production."""

    @classmethod
    def setUpClass(cls):
        import routes_queue
        cls.q = routes_queue

    def setUp(self):
        q = self.q
        self.saved = (list(q._queue["jobs"]), dict(q._queue))
        self.addCleanup(self.restore)
        q._queue["jobs"] = [
            {"id": "a", "target_name": r"C:\clips\a.mp4", "label": "", "state": "COMPLETED", "status": "finished",
             "started": 10.0, "finished": 20.0, "outputs": [r"C:\out\a_swapped.mp4"], "error": ""},
            {"id": "b", "target_name": "b.mp4", "label": "Interview", "state": "PROCESSING", "status": "running",
             "started": 21.0, "finished": 0.0, "frame_start": 0, "frame_end": 300},
            {"id": "c", "target_name": "c.mp4", "state": "QUEUED", "status": "pending"},
        ]
        q._queue.update(running=True, current="b", run_started=5.0, paused=False)

    def restore(self):
        jobs, rest = self.saved
        self.q._queue.clear()
        self.q._queue.update(rest)
        self.q._queue["jobs"] = jobs

    def test_it_is_registered_with_the_dashboard(self):
        self.assertIs(td._batch_source, self.q._dashboard_batch_source)

    def test_it_reports_the_jobs_the_queue_holds(self):
        src = self.q._dashboard_batch_source()
        self.assertEqual([j["name"] for j in src["jobs"]], ["a.mp4", "Interview", "c.mp4"])
        self.assertEqual([j["state"] for j in src["jobs"]], ["COMPLETED", "PROCESSING", "QUEUED"])
        self.assertEqual(src["jobs"][0]["output"], "a_swapped.mp4")
        self.assertEqual(src["jobs"][1]["frames_total"], 300)
        self.assertEqual((src["current"], src["run_started"]), ("b", 5.0))
        v = td.build_batch_view(src, {}, 100.0, {"stage": "S3", "eta_s": 10.0})
        self.assertEqual(v["current_pos"], 2)

    def test_a_queue_that_is_not_running_is_not_a_batch(self):
        self.q._queue["running"] = False
        self.assertIsNone(self.q._dashboard_batch_source())
        self.assertEqual(len(self.q._dashboard_batch_source(include_idle=True)["jobs"]), 3)

    def test_it_never_waits_on_a_held_lock(self):
        held, release = threading.Event(), threading.Event()

        def holder():
            with self.q._lock:
                held.set()
                release.wait(5)
        t = threading.Thread(target=holder)
        t.start()
        try:
            held.wait(5)
            self.assertEqual(self.q._dashboard_batch_source(), {"busy": True})
        finally:
            release.set()
            t.join(5)

    def test_the_end_of_queue_summary_prints_ascii_records_into_a_pipe(self):
        for j in self.q._queue["jobs"]:
            j.update(state="COMPLETED", status="finished", started=1.0, finished=2.0)
        self.q._queue.update(running=False, current=None)
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"ROOP_DASHBOARD": "log"}), mock.patch.object(sys, "stdout", buf):
            self.q._print_batch_summary()
        out = buf.getvalue()
        self.assertIn("BATCH FINISHED", out)
        self.assertIn("Interview", out)
        self.assertNotIn("\x1b", out)
        self.assertTrue(all(ln.startswith("[Batch]") for ln in out.splitlines() if ln))


class Wiring(unittest.TestCase):
    def src(self, rel):
        with open(os.path.join(APP, rel), encoding="utf-8") as fh:
            return fh.read()

    def test_core_hands_the_target_list_to_the_dashboard(self):
        core = self.src(os.path.join("roop", "core.py"))
        self.assertIn("batch_files=[x.filename for x in videofiles]", core)
        self.assertIn("job['batch_files']", core)

    def test_the_queue_prints_the_summary_when_it_runs_dry(self):
        rq = self.src("routes_queue.py")
        self.assertIn("finished_all = True", rq)
        self.assertIn("_print_batch_summary()", rq)
        self.assertIn('_queue["run_started"] = time.time()', rq)


if __name__ == "__main__":
    unittest.main()
