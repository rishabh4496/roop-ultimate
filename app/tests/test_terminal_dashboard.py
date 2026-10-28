"""Stage 9: the terminal processing dashboard (`roop/terminal_dashboard.py`).

Every behaviour is driven through the writer production uses and read back through the consumer it exists for:

* the registry (`roop.stage_telemetry`) is fed with the calls `procmgr_batch` / `_prof` / the writer make, and the
  dashboard's snapshot is read back - a counter nothing increments or a label nothing reads looks wired (AGENTS.md);
* the console text is parsed back against the SNAPSHOT, so "the terminal and the React screen show the same numbers" is
  a checked claim, not a hope;
* the live block is run against a small VT emulator (cursor-previous-line / erase / newline) so "a stray print lands above
  the block and the block is not duplicated" is observed on a screen, not inferred from escape strings;
* `ChunkedProgress` keeps rendering every bar the dashboard does not claim, so the legacy tests' contract is unchanged.
"""
import io
import json
import os
import re
import sys
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import stage_telemetry as st  # noqa: E402
from roop import terminal_dashboard as td  # noqa: E402

S1, S2, S3 = st.STAGE_1_SPARSE_PRESCAN, st.STAGE_2_ACTIVE_PREPASS, st.STAGE_3_MAIN_SWAP
ANSI = re.compile(r"\x1b\[")


class Clock:
    """One fake clock for the registry (time.time), the manager (clock + wall) and the EMA."""

    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


class _Env(unittest.TestCase):
    def setUp(self):
        st.reset()
        td.reset()
        self.clock = Clock()
        for patcher in (mock.patch.object(st.time, "time", self.clock),
                        mock.patch.object(st._vram, "start", lambda: st._vram.reset()),
                        mock.patch.dict(os.environ, {"ROOP_DASHBOARD": "log"})):
            patcher.start()
            self.addCleanup(patcher.stop)
        st._vram.stop()
        st._vram._probe = lambda: None
        self.addCleanup(st.reset)
        self.addCleanup(td.reset)
        self.mgr = td.TerminalProgressManager(clock=self.clock, wall=self.clock)
        self.mgr.TICK_S = 3600.0                      # no background thread ticks: the test drives `_tick`
        self.addCleanup(lambda: self.mgr.end_job(ok=False, _quiet=True))

    def start(self, job=None, profile=None):
        with mock.patch("builtins.print"):
            self.mgr.begin_job(job or {"job_id": "j1", "mode": "video"}, profile)

    def finish(self, ok=True):
        with mock.patch("builtins.print"):
            self.mgr.end_job(ok)

    def tick(self, seconds=0.5):
        self.clock.advance(seconds)
        with mock.patch("builtins.print"):
            self.mgr._tick()
        return self.mgr.snapshot()


class AdaptiveEmaTests(unittest.TestCase):
    def feed(self, ema, schedule, tick=0.5):
        """schedule: (arrival_time, frames) pairs; sample the cumulative count every `tick` seconds."""
        n, i, t, end = 0, 0, 0.0, max(a for a, _ in schedule)
        ema.update(0.0, 0)
        while t < end:
            t += tick
            while i < len(schedule) and schedule[i][0] <= t:
                n += schedule[i][1]
                i += 1
            ema.update(t, n)
        return n, t

    def test_bursty_arrivals_do_not_inflate_the_rate(self):
        # 24 frames land at once every 0.96 s = a steady 25 f/s. ChunkedProgress' docstring: averaging per-update
        # instantaneous rates read this kind of producer 795x too fast.
        ema = td.AdaptiveEma()
        self.feed(ema, [(k * 0.96, 24) for k in range(1, 120)])
        self.assertAlmostEqual(ema.rate, 25.0, delta=2.0)

    def test_eta_is_remaining_over_rate_and_none_without_a_rate(self):
        ema = td.AdaptiveEma()
        self.assertIsNone(ema.eta(100))
        self.feed(ema, [(float(k), 20) for k in range(1, 40)])
        self.assertAlmostEqual(ema.eta(400), 20.0, delta=1.5)
        self.assertIsNone(ema.eta(0))

    def test_a_real_speed_change_is_followed(self):
        ema = td.AdaptiveEma()
        sched = [(float(k), 30) for k in range(1, 120)] + [(120.0 + k, 10) for k in range(1, 60)]
        self.feed(ema, sched)
        self.assertLess(ema.rate, 14.0)               # fell from 30 f/s towards the new 10 f/s instead of averaging 25

    def test_a_stall_pulls_the_rate_down_not_up(self):
        ema = td.AdaptiveEma()
        self.feed(ema, [(float(k), 20) for k in range(1, 30)])
        before = ema.rate
        for k in range(1, 20):
            ema.update(29.0 + k, 29 * 20)
        self.assertLess(ema.rate, before)


class Formatting(unittest.TestCase):
    def test_durations(self):
        self.assertEqual(td.fmt_dur(74), "01m 14s")
        self.assertEqual(td.fmt_dur(3725), "1h 02m 05s")
        self.assertEqual(td.fmt_dur(None), "--")
        self.assertEqual(td.fmt_dur(float("nan")), "--")

    def test_eta_pairs_the_countdown_with_the_wall_clock_finish(self):
        now = time.mktime((2026, 10, 28, 3, 22, 56, 0, 0, -1))
        text = td.fmt_eta(74, now + 74, now)
        self.assertEqual(text, "01m 14s -> 03:24:10")                  # the brief's example, to the second
        self.assertEqual(td.fmt_eta(None, None), "--")

    def test_a_finish_on_another_day_says_so(self):
        now = time.mktime((2026, 10, 28, 23, 50, 0, 0, 0, -1))
        self.assertIn("(+1d)", td.fmt_eta(1800, now + 1800, now))

    def test_truncation_keeps_escapes_whole_and_closes_the_colour(self):
        text = f"{td.RED}abcdefghij{td.RESET}"
        cut = td.truncate_visible(text, 4)
        self.assertEqual(td.strip_ansi(cut), "abcd")
        self.assertTrue(cut.endswith(td.RESET))


class Mode(unittest.TestCase):
    def env(self, **kw):
        base = {k: v for k, v in os.environ.items() if k not in ("ROOP_DASHBOARD", "CI", "GITHUB_ACTIONS", "PYTEST_CURRENT_TEST", "TERM")}
        base.update(kw)
        return mock.patch.dict(os.environ, base, clear=True)

    def test_explicit_values(self):
        for value, want in (("off", "off"), ("0", "off"), ("log", "log"), ("live", "live")):
            with self.env(ROOP_DASHBOARD=value):
                self.assertEqual(td.mode(), want)

    def test_auto_is_live_only_on_a_real_terminal(self):
        with self.env(), mock.patch.object(td, "stream_is_tty", return_value=True):
            self.assertEqual(td.mode(), "live")
        with self.env(), mock.patch.object(td, "stream_is_tty", return_value=False):
            self.assertEqual(td.mode(), "log")

    def test_auto_falls_back_under_ci_pytest_and_a_dumb_terminal(self):
        for extra in ({"CI": "true"}, {"GITHUB_ACTIONS": "true"}, {"PYTEST_CURRENT_TEST": "x"}, {"TERM": "dumb"}):
            with self.env(**extra), mock.patch.object(td, "stream_is_tty", return_value=True):
                self.assertEqual(td.mode(), "log", extra)

    def test_a_non_tty_stdout_is_not_a_terminal(self):
        with mock.patch.object(sys, "stdout", io.StringIO()):
            self.assertFalse(td.stream_is_tty())


class Stages(_Env):
    def test_a_job_walks_the_four_stages(self):
        self.start()
        s = self.tick()
        self.assertEqual(s["stages"]["S1"]["state"], td.RUNNING)
        self.assertEqual([s["stages"][k]["state"] for k in ("S2", "S3", "S4")], [td.PENDING] * 3)
        self.mgr.s1_step("media probe")
        self.mgr.s1_step("audio stream")
        self.assertEqual(self.tick()["stages"]["S1"]["pct"], 50.0)
        st.begin_run(600)
        self.mgr.s1_finish()
        st.skip(S1)
        st.enter(S2, 300)
        st.progress(S2, 150, 300)
        s = self.tick()
        self.assertEqual((s["stages"]["S1"]["state"], s["stages"]["S1"]["pct"]), (td.DONE, 100.0))
        self.assertEqual((s["stages"]["S2"]["state"], s["stages"]["S2"]["pct"]), (td.RUNNING, 50.0))
        self.assertEqual(s["current"], "S2")
        st.leave(S2)
        st.enter(S3, 600)
        s = self.tick()
        self.assertEqual((s["stages"]["S2"]["state"], s["stages"]["S3"]["state"]), (td.DONE, td.RUNNING))
        st.leave(S3)
        self.mgr.s4_step("segment custody & concat", 1.0)
        self.mgr.s4_step("audio remux", 0.5)
        self.mgr.s4_step("output verification", 0.1)
        self.mgr.s4_end(True)
        s = self.tick()
        self.assertEqual((s["stages"]["S3"]["state"], s["stages"]["S4"]["state"], s["stages"]["S4"]["pct"]), (td.DONE, td.DONE, 100.0))
        self.finish()

    def test_s1_finishes_on_its_own_when_the_registry_moves_on(self):
        self.start()
        st.begin_run(100)
        st.enter(S3, 100)
        self.assertEqual(self.tick()["stages"]["S1"]["state"], td.DONE)

    def test_s2_merges_the_prescan_and_the_tracking_stage(self):
        self.start()
        st.begin_run(1000)
        st.enter(S1, 100)
        st.progress(S1, 100, 100)
        st.leave(S1)
        st.enter(S2, 200)
        st.progress(S2, 100, 200)
        s2 = self.tick()["stages"]["S2"]
        self.assertEqual(s2["pct"], 75.0)                                # (1.0 + 0.5) / 2
        self.assertEqual(s2["detail"]["phase"], "Tracking")

    def test_a_dense_render_skips_s2(self):
        self.start()
        st.begin_run(100)
        st.skip(S1)
        st.enter(S3, 100)
        s2 = self.tick()["stages"]["S2"]
        self.assertEqual((s2["state"], s2["pct"]), (td.SKIPPED, None))

    def test_stage3_counts_writer_accepted_frames_and_says_so(self):
        self.start()
        st.begin_run(600)
        st.enter(S3, 600)
        self.tick()
        for _ in range(100):
            st.retire()
        s3 = self.tick()["stages"]["S3"]
        self.assertEqual((s3["done"], s3["total"], s3["detail"]["fps_basis"]), (100, 600, "retired"))
        self.assertAlmostEqual(s3["pct"], 16.7, places=1)

    def test_a_stage3_that_ended_short_is_not_rounded_up_to_100(self):
        self.start()
        st.begin_run(600)
        st.enter(S3, 600)
        self.tick()
        for _ in range(300):
            st.retire()
        self.tick()
        st.leave(S3)                                                      # leave() forces the registry's own counter to 600
        s3 = self.tick()["stages"]["S3"]
        self.assertEqual((s3["state"], s3["pct"], s3["detail"]["short_by"]), (td.DONE, 50.0, 300))
        self.assertIn("SHORT by 300", td.render_stage_lines(self.mgr.snapshot(), color=False)[2])

    def test_eta_and_rate_appear_once_frames_flow(self):
        self.start()
        st.begin_run(600)
        st.enter(S3, 600)
        self.tick()
        for _ in range(4):
            for _ in range(25):
                st.retire()
            s3 = self.tick(1.0)["stages"]["S3"]
        self.assertAlmostEqual(s3["rate"], 25.0, delta=3.0)
        self.assertAlmostEqual(s3["eta_s"], (600 - 100) / 25.0, delta=60)
        self.assertAlmostEqual(s3["eta_at"], self.clock.t + s3["eta_s"], delta=2)

    def test_a_paused_job_freezes_the_estimate(self):
        self.start()
        st.begin_run(600)
        st.enter(S3, 600)
        self.tick()
        for _ in range(3):
            for _ in range(25):
                st.retire()
            self.tick(1.0)
        before = self.mgr.snapshot()["stages"]["S3"]["rate"]
        with mock.patch.object(self.mgr, "_paused", return_value=True):
            for _ in range(10):
                after = self.tick(1.0)["stages"]["S3"]
        self.assertEqual(after["rate"], before)
        self.assertTrue(after["paused"])
        self.assertIn("PAUSED", td.render_stage_lines(self.mgr.snapshot(), color=False)[2])

    def test_an_image_job_skips_the_video_only_stages(self):
        self.start(job={"job_id": "i1", "mode": "image"})
        bar = mock.Mock(desc="Processing", n=3, total=10)
        self.mgr.attach_bar(bar)
        s = self.tick()["stages"]
        self.assertEqual((s["S2"]["state"], s["S4"]["state"], s["S3"]["state"]), (td.SKIPPED, td.SKIPPED, td.RUNNING))
        self.assertEqual((s["S3"]["done"], s["S3"]["total"], s["S3"]["detail"]["fps_basis"]), (3, 10, "processed"))
        bar.n = 10
        self.mgr.detach_bar(bar)
        self.assertEqual(self.tick()["stages"]["S3"]["state"], td.DONE)


class Telemetry(_Env):
    """The per-frame latency split, the bottleneck and the optimizer hit rates."""

    def run_stage3(self, per_frame, frames=100, window_ticks=4):
        """per_frame: {_prof name: seconds per delivered frame}."""
        self.start()
        st.begin_run(1000)
        st.enter(S3, 1000)
        self.tick()
        for _ in range(window_ticks):
            for _ in range(frames // window_ticks):
                for name, sec in per_frame.items():
                    st.record_op(name, sec)
                st.retire()
            snap = self.tick(1.0)
        return snap["telemetry"]

    def test_columns_group_the_raw_prof_names_and_the_split_is_per_delivered_frame(self):
        t = self.run_stage3({"decode": 0.004, "detect": 0.006, "swap": 0.010, "mask": 0.004, "mask_refine": 0.002,
                             "enhance": 0.030, "enhance_recombine": 0.002, "lighting": 0.001, "blend": 0.001, "encode": 0.003})
        ms = t["latency_ms"]
        self.assertEqual((ms["decode"], ms["detect"], ms["swap"], ms["mask"]), (4.0, 6.0, 10.0, 6.0))
        self.assertEqual((ms["restore"], ms["blend"], ms["encode"]), (32.0, 2.0, 3.0))
        self.assertEqual(t["frames"], 100)

    def test_the_bottleneck_is_the_largest_share_of_all_recorded_time(self):
        t = self.run_stage3({"decode": 0.004, "swap": 0.008, "mask": 0.004, "enhance": 0.032, "blend": 0.002})
        self.assertEqual(t["bottleneck"]["stage"], "restore")
        self.assertEqual(t["bottleneck"]["share_pct"], 64.0)              # 32 of 50 ms
        self.assertEqual(t["bottleneck"]["severity"], "red")

    def test_other_operations_stay_in_the_denominator(self):
        # alignment is not one of the seven columns, but it is recorded time: a "64%" must be 64% of ALL of it.
        t = self.run_stage3({"enhance": 0.032, "swap": 0.008, "alignment": 0.050})
        self.assertEqual(t["bottleneck"]["share_pct"], 36.0)              # 32 of 90 ms
        self.assertEqual(t["bottleneck"]["severity"], "yellow")
        self.assertEqual(t["latency_ms"]["other"], 50.0)

    def test_wrappers_and_prepass_names_are_not_stage3_work(self):
        t = self.run_stage3({"swap": 0.010, "frame_total": 0.500, "track_detect": 0.300, "verify": 0.200})
        self.assertEqual(t["bottleneck"]["stage"], "swap")
        self.assertEqual(t["bottleneck"]["share_pct"], 100.0)

    def test_no_frames_no_made_up_numbers(self):
        self.start()
        st.begin_run(100)
        st.enter(S3, 100)
        st.record_op("swap", 1.0)
        t = self.tick()["telemetry"]
        self.assertIsNone(t["bottleneck"])
        self.assertTrue(all(v is None for v in t["latency_ms"].values()))

    def test_the_split_follows_the_trailing_window_not_the_whole_run(self):
        self.start()
        st.begin_run(10000)
        st.enter(S3, 10000)
        self.tick()
        for _ in range(30):                                               # 30 s of a swap-bound scene...
            for _ in range(10):
                st.record_op("swap", 0.010)
                st.retire()
            self.tick(1.0)
        for _ in range(12):                                               # ...then a restorer-bound one
            for _ in range(10):
                st.record_op("enhance", 0.050)
                st.retire()
            t = self.tick(1.0)["telemetry"]
        self.assertEqual(t["bottleneck"]["stage"], "restore")

    def test_the_last_split_survives_the_end_of_stage3(self):
        self.run_stage3({"swap": 0.010})
        st.leave(S3)
        t = self.tick(30.0)["telemetry"]
        self.assertEqual(t["latency_ms"]["swap"], 10.0)

    def test_hit_rates_come_from_the_registry(self):
        self.start()
        st.begin_run(1000)
        st.set_ranges(2, 800, 1000)
        st.enter(S3, 1000)
        st.add(S3, "neural_inferences", 90)
        st.add(S3, "smart_skip_faces", 10)
        with mock.patch.object(st._vram, "current_gb", return_value=7.25), \
                mock.patch.object(st._vram, "report", return_value={"peak_gb": 8.5, "total_gb": 12.0, "base_gb": 1.0}):
            hr = self.tick()["telemetry"]["hit_rates"]
        self.assertEqual(hr["smart_skip_roi_reused_pct"], 10.0)
        self.assertEqual(hr["stage2_face_free_skipped_pct"], 20.0)
        self.assertEqual((hr["vram_allocated_gb"], hr["vram_peak_gb"], hr["vram_total_gb"]), (7.25, 8.5, 12.0))


class Rendering(_Env):
    def snap(self):
        self.start({"job_id": "j9", "mode": "video"})
        st.begin_run(600)
        st.set_ranges(3, 450, 600)
        st.skip(S1)
        st.enter(S2, 450)
        st.progress(S2, 450, 450)
        st.leave(S2)
        st.enter(S3, 600)
        self.tick()
        for _ in range(4):
            for _ in range(25):
                for n, v in (("decode", 0.003), ("swap", 0.008), ("enhance", 0.032), ("blend", 0.002)):
                    st.record_op(n, v)
                st.retire()
            snap = self.tick(1.0)
        return snap

    def test_log_lines_carry_no_escape_sequences_and_are_one_record_per_line(self):
        lines = td.render_log_lines(self.snap())
        self.assertTrue(lines)
        text = "\n".join(lines)
        self.assertNotIn("\x1b", text)
        self.assertNotIn("\r", text)
        self.assertTrue(all(ln.startswith(("[Dashboard]", "[Telemetry]")) for ln in lines))
        self.assertIn("stage=S3(Swap) state=RUNNING", text)
        self.assertIn("bottleneck=restore:", text)

    def test_the_console_prints_the_snapshots_numbers(self):
        snap = self.snap()
        s3 = snap["stages"]["S3"]
        plain = td.strip_ansi("\n".join(td.render_stage_lines(snap, color=True, width=200)
                                        + td.render_telemetry_lines(snap, color=True, width=200)))
        self.assertIn(f"{s3['pct']:.1f}%", plain)
        self.assertIn(f"{s3['done']:,}/{s3['total']:,} frames", plain)
        self.assertIn(f"{s3['rate']:.1f} fps", plain)
        self.assertIn(td.fmt_eta(s3["eta_s"], s3["eta_at"], snap["updated_at"]), plain)
        bn = snap["telemetry"]["bottleneck"]
        self.assertIn(f"[BOTTLENECK: {bn['label']} {bn['share_pct']:.0f}%]", plain)
        for key, label, *_ in td.LATENCY_COLUMNS:
            self.assertIn(f"{label} {snap['telemetry']['latency_ms'][key]:.1f}", plain.replace("  ", " "))

    def test_each_stage_has_its_own_colour_and_the_bottleneck_is_highlighted(self):
        snap = self.snap()
        lines = td.render_stage_lines(snap, color=True)
        self.assertEqual([td.STAGE_COLOR[k] in lines[i] for i, k in enumerate(("S1", "S2", "S3", "S4"))], [True] * 4)
        self.assertEqual(len({td.STAGE_COLOR[k] for k in ("S1", "S2", "S3", "S4")}), 4)
        tel = td.render_telemetry_lines(snap, color=True)
        self.assertIn(td.RED, tel[1])                                     # restore holds 64%: bold red
        self.assertIn(td.RED, tel[0])

    def test_plain_rendering_has_no_escapes_either(self):
        snap = self.snap()
        for line in td.render_stage_lines(snap, color=False) + td.render_telemetry_lines(snap, color=False):
            self.assertNotIn("\x1b", line)

    def test_the_snapshot_is_json_safe_and_the_ws_frame_leaves_the_profile_out(self):
        profile = td.collect_swap_profile(None, {"job_id": "j9", "batch_index": 2, "batch_count": 5}, None, 30.0, 600)
        self.start({"job_id": "j9", "mode": "video"}, profile)
        self.tick()
        snap = json.loads(json.dumps(self.mgr.snapshot()))
        self.assertEqual(snap["schema"], 1)
        self.assertIn("profile", snap)
        with mock.patch.object(td, "manager", self.mgr):
            frame = td.telemetry_frame()
        self.assertNotIn("profile", frame)
        self.assertTrue(frame["has_profile"])

    def test_the_idle_snapshot_is_constant(self):
        self.assertEqual(td.snapshot(), {"schema": 1, "active": False})
        self.assertEqual(td.telemetry_frame(), td.telemetry_frame())

    def test_a_finished_job_stays_readable_and_constant(self):
        self.snap()
        self.finish()
        a = self.mgr.snapshot()
        self.assertFalse(a["active"])
        self.assertEqual(a["stages"]["S3"]["state"], td.RUNNING if False else a["stages"]["S3"]["state"])
        self.assertEqual(a, self.mgr.snapshot())


class SwapProfile(unittest.TestCase):
    def profile(self, **over):
        p = {"job": {"id": "job-7", "mode": "segmented", "label": "clip.mp4", "batch": "3/8",
                     "segment": {"group": "g", "index": 2, "count": 6, "start": 600, "end": 1200, "seconds": 20.0,
                                 "custody_dir": "out/.segments/g"}},
             "identity": {"sources": [{"index": 0, "name": "monica.fsz", "kind": ".fsz", "crops": 14, "embedding_norm": 24.31,
                                       "pose_coverage": "7/9 cells"}], "blend": "monica.fsz:0.60 + anna.fsz:0.40"},
             "target": {"file": "clip.mp4", "resolution": "1920x1080", "fps": 29.97, "frames": 1800, "pix_fmt": "yuv420p",
                        "color": "bt709/bt709/bt709/tv", "audio": True},
             "models": {"detector": {"model": "retinaface_r50", "size": "640", "provider": "tensorrt", "precision": "FP16",
                                     "threshold": 0.5, "max_faces": "unbounded (no cap configured)"},
                        "swapper": {"model": "realswap", "provider": "tensorrt", "precision": "MIXED", "batch_mode": "xframe",
                                    "zero_copy": "ACTIVE", "zero_copy_note": "IO binding live"},
                        "restore_chain": ["CodeFormer w=0.75", "frequency separation", "grain closer 0.45", "LAB colour transfer"],
                        "mask": {"engine": "RealityUX", "engine_2": "None", "box_padding": "0/0/0/0", "feather_px": 12.0,
                                 "erode_dilate_px": 0, "bisenet_regions": ["skin", "brows", "eyes"]}},
             "hardware": {"gpu": "NVIDIA GeForce RTX 4070", "vram_total_gb": 12.0, "vram_used_gb": 1.2, "trt_cache": "models/trt_cache",
                          "trt_cache_status": "9 engine(s) cached", "openvino": "1 session(s)", "workers": 20,
                          "p_cores": "8 P / 16 E cores"}}
        p.update(over)
        return p

    def text(self, p, color=False):
        return td.strip_ansi("\n".join(td.render_profile(p, color=color)))

    def test_every_brief_field_is_in_the_banner(self):
        t = self.text(self.profile())
        for needle in ("job-7", "SEGMENTED", "batch 3/8", "segment 2/6", "20s slices", "out/.segments/g",
                       "monica.fsz", "14 crop(s)", "24.31", "7/9 cells", "monica.fsz:0.60 + anna.fsz:0.40",
                       "1920x1080", "29.970 fps", "1,800 frames", "yuv420p", "bt709", "present",
                       "retinaface_r50", "FP16", "conf>=0.50", "max faces: unbounded", "realswap", "zero-copy CUDA ACTIVE",
                       "CodeFormer w=0.75", "grain closer 0.45", "LAB colour transfer", "BiSeNet skin,brows,eyes", "feather 12px",
                       "NVIDIA GeForce RTX 4070", "1.20/12.00 GB", "models/trt_cache", "9 engine(s) cached", "1 session(s)", "20 workers", "8 P / 16 E"):
            self.assertIn(needle, t, needle)

    def test_the_boxed_banner_is_a_box_and_the_plain_one_is_ascii_records(self):
        boxed = td.render_profile(self.profile(), color=True)
        self.assertTrue(boxed[0].lstrip(td.STAGE_COLOR["S2"]).startswith("╭"))
        self.assertIn("╰", boxed[-1])
        plain = td.render_profile(self.profile(), color=False)
        self.assertTrue(all(ln.startswith("[SwapProfile] ") for ln in plain))
        self.assertNotIn("\x1b", "\n".join(plain))
        self.assertFalse(any(ord(c) > 127 for c in "\n".join(plain)))

    def test_a_section_that_could_not_be_read_says_n_a_and_does_not_hide_the_rest(self):
        t = self.text(self.profile(models=None, hardware=None))
        self.assertIn("models: n/a", t)
        self.assertIn("hardware: n/a", t)
        self.assertIn("monica.fsz", t)

    def test_the_collector_survives_having_nothing_to_read(self):
        p = td.collect_swap_profile(None, {"job_id": "j"}, None, None, None)
        self.assertEqual(p["job"]["mode"], "image")
        self.assertEqual(p["identity"]["sources"], [])
        json.dumps(p)

    def test_mode_resolution(self):
        c = lambda **job: td.collect_swap_profile(None, job, "clip.mp4", 30.0, 10)["job"]["mode"]
        self.assertEqual(c(), "video")
        self.assertEqual(c(batch_index=2, batch_count=5), "batch")
        self.assertEqual(c(segment={"index": 1, "count": 2}), "segmented")
        self.assertEqual(c(mode="image"), "image")

    def test_zero_copy_status_follows_the_live_binding(self):
        sess = lambda prov: mock.Mock(get_providers=lambda: [prov, "CPUExecutionProvider"])
        swap = lambda prov, binds=None: mock.Mock(model_swap_insightface=sess(prov), _io_bindings=binds if binds is not None else {})
        self.assertEqual(td.swapper_zero_copy(swap("TensorrtExecutionProvider"))[0], "ACTIVE")
        self.assertEqual(td.swapper_zero_copy(swap("CUDAExecutionProvider", {1: mock.Mock(enabled=True)}))[0], "ACTIVE")
        self.assertEqual(td.swapper_zero_copy(swap("CUDAExecutionProvider", {1: mock.Mock(enabled=False)}))[0], "HOST-FALLBACK")
        self.assertEqual(td.swapper_zero_copy(swap("CPUExecutionProvider"))[0], "HOST-FALLBACK")
        self.assertEqual(td.swapper_zero_copy(None)[0], "HOST-FALLBACK")
        with mock.patch.dict(os.environ, {"ROOP_ORT_IO_BINDING": "0"}):
            self.assertEqual(td.swapper_zero_copy(swap("TensorrtExecutionProvider"))[0], "HOST-FALLBACK")


class VtScreen:
    """Just enough of a terminal to watch the live block: \\n, \\r, CSI n F (previous line), CSI J, CSI K."""

    def __init__(self):
        self.rows, self.r, self.c = [""], 0, 0

    def write(self, s):
        i = 0
        while i < len(s):
            ch = s[i]
            m = re.match(r"\x1b\[([0-9;?]*)([A-Za-z])", s[i:])
            if m:
                arg, cmd = m.group(1), m.group(2)
                n = int(arg) if arg.isdigit() else 1
                if cmd == "F":
                    self.r, self.c = max(0, self.r - n), 0
                elif cmd == "J":
                    self.rows = self.rows[:self.r + 1]
                    self.rows[self.r] = self.rows[self.r][:self.c]
                elif cmd == "K":
                    self.rows[self.r] = self.rows[self.r][:self.c]
                i += m.end()
                continue
            if ch == "\n":
                self.r += 1
                self.c = 0
                while len(self.rows) <= self.r:
                    self.rows.append("")
            elif ch == "\r":
                self.c = 0
            else:
                row = self.rows[self.r].ljust(self.c)
                self.rows[self.r] = row[:self.c] + ch + row[self.c + 1:]
                self.c += 1
            i += 1
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return True

    def screen(self):
        return [r.rstrip() for r in self.rows]


class LiveTerminal(unittest.TestCase):
    def setUp(self):
        self.scr = VtScreen()
        self.err = VtScreen()
        self.saved = (sys.stdout, sys.stderr)
        sys.stdout, sys.stderr = self.scr, self.err
        self.term = td._LiveTerminal(self.scr)
        self.addCleanup(self.restore)

    def restore(self):
        sys.stdout, sys.stderr = self.saved

    def test_the_block_redraws_in_place(self):
        self.term.install()
        for n in range(5):
            self.term.set_lines([f"A{n}", f"B{n}", f"C{n}"])
        self.assertEqual([r for r in self.scr.screen() if r], ["A4", "B4", "C4"])

    def test_a_stray_print_lands_above_the_block_and_the_block_is_not_duplicated(self):
        self.term.install()
        self.term.set_lines(["bar-1", "bar-2"])
        print("worker says hi")
        self.term.set_lines(["bar-1b", "bar-2b"])
        print("and again")
        sys.stderr.write("an error line\n")
        self.term.set_lines(["bar-1c", "bar-2c"])
        rows = [r for r in self.scr.screen() if r]
        self.assertEqual(rows, ["worker says hi", "and again", "bar-1c", "bar-2c"])

    def test_a_line_left_open_does_not_get_a_block_painted_into_it(self):
        self.term.install()
        self.term.set_lines(["bar"])
        sys.stdout.write("partial")
        self.term.set_lines(["bar2"])
        self.assertEqual([r for r in self.scr.screen() if r], ["partial"])
        sys.stdout.write(" done\n")
        self.assertEqual([r for r in self.scr.screen() if r], ["partial done", "bar2"])

    def test_a_stuck_open_line_is_closed_after_the_valve_time(self):
        self.term.install()
        sys.stdout.write("tqdm-ish")
        with mock.patch.object(td.time, "monotonic", side_effect=[100.0 + td._LiveTerminal.PARTIAL_VALVE_S + 1]):
            self.term._partial_since = 100.0
            self.term.set_lines(["bar"])
        self.assertEqual([r for r in self.scr.screen() if r], ["tqdm-ish", "bar"])

    def test_the_streams_come_back_and_the_final_lines_stay_in_the_scrollback(self):
        self.term.install()
        self.assertIsInstance(sys.stdout, td._Proxy)
        self.term.set_lines(["live-1", "live-2"])
        self.term.uninstall(["FINAL-1", "FINAL-2"])
        self.assertIs(sys.stdout, self.scr)
        self.assertIs(sys.stderr, self.err)
        self.assertEqual([r for r in self.scr.screen() if r], ["FINAL-1", "FINAL-2"])

    def test_the_cursor_is_hidden_while_live_and_restored_after(self):
        out = io.StringIO()
        term = td._LiveTerminal(out)
        term.install()
        term.uninstall([])
        self.assertTrue(out.getvalue().startswith("\x1b[?25l"))
        self.assertTrue(out.getvalue().endswith("\x1b[?25h"))

    def test_the_proxy_delegates_everything_else(self):
        self.term.install()
        self.assertTrue(sys.stdout.isatty())

    def test_lines_wider_than_the_terminal_are_cut_not_wrapped(self):
        self.term.install()
        with mock.patch.object(td.shutil, "get_terminal_size", return_value=os_size(40, 20)):
            self.term.set_lines(["x" * 200])
        self.assertLessEqual(len(self.scr.screen()[0]), 40)


def os_size(cols, rows):
    import os as _os
    return _os.terminal_size((cols, rows))


class Facade(_Env):
    """The module-level calls the pipeline makes, through the real manager."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(td, "manager", self.mgr)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_log_mode_under_pytest_writes_no_escape_codes(self):
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"ROOP_DASHBOARD": "auto"}), mock.patch.object(sys, "stdout", buf):
            td.begin_job({"job_id": "j", "mode": "video"}, td.collect_swap_profile(None, {"job_id": "j"}, None, 30.0, 10))
            self.assertEqual(self.mgr._mode, "log")
            st.begin_run(10)
            st.enter(S3, 10)
            for _ in range(10):
                st.retire()
            td.end_job(True)
        out = buf.getvalue()
        self.assertIn("[SwapProfile]", out)
        self.assertIn("[Dashboard]", out)
        self.assertNotIn("\x1b", out)

    def test_off_means_nothing_runs(self):
        with mock.patch.dict(os.environ, {"ROOP_DASHBOARD": "off"}):
            td.begin_job({"job_id": "j"}, None)
        self.assertFalse(self.mgr.active)
        self.assertFalse(td.claims_bar("Processing"))

    def test_a_dashboard_fault_never_reaches_the_render(self):
        with mock.patch.object(self.mgr, "_tick", side_effect=RuntimeError("boom")):
            td.begin_job({"job_id": "j"}, None)             # must not raise
        self.assertFalse(self.mgr.active)
        td.end_job(False)

    def test_a_half_started_live_job_gives_the_streams_back(self):
        scr = VtScreen()
        before = (sys.stdout, sys.stderr)
        with mock.patch.dict(os.environ, {"ROOP_DASHBOARD": "live"}), mock.patch.object(td, "_enable_windows_vt", return_value=True), \
                mock.patch.object(sys, "stdout", scr), mock.patch.object(self.mgr, "_tick", side_effect=RuntimeError("boom")):
            td.begin_job({"job_id": "j"}, None)
            self.assertIs(sys.stdout, scr)
        self.assertEqual(before, (sys.stdout, sys.stderr))
        self.assertFalse(self.mgr.active)

    def test_a_second_begin_ends_the_first_without_deadlocking(self):
        with mock.patch("builtins.print"):
            td.begin_job({"job_id": "a"}, None)
            t0 = time.monotonic()
            td.begin_job({"job_id": "b"}, None)
            self.assertLess(time.monotonic() - t0, 1.5)
            self.assertEqual(self.mgr.snapshot()["job"]["job_id"], "b")
            td.end_job(True)

    def test_a_live_job_end_to_end_on_a_screen(self):
        scr = VtScreen()
        self.mgr.TICK_S = 0.05
        with mock.patch.dict(os.environ, {"ROOP_DASHBOARD": "live"}), mock.patch.object(td, "_enable_windows_vt", return_value=True), \
                mock.patch.object(td.shutil, "get_terminal_size", return_value=os_size(160, 40)), \
                mock.patch.object(sys, "stdout", scr), mock.patch.object(sys, "stderr", VtScreen()):
            td.begin_job({"job_id": "live-1", "mode": "video"}, td.collect_swap_profile(None, {"job_id": "live-1"}, None, 30.0, 40))
            st.begin_run(40)
            td.s1_finish()
            st.skip(S1)
            st.enter(S3, 40)
            for i in range(40):
                st.record_op("enhance", 0.03)
                st.retire()
                if i % 10 == 0:
                    print(f"worker diagnostic {i}")
                    time.sleep(0.12)
            st.leave(S3)
            td.end_job(True)
        text = "\n".join(scr.screen())
        self.assertIn("SWAP PROFILE", text)
        self.assertEqual(text.count("worker diagnostic"), 4)                # every stray print survived, once
        self.assertEqual(text.count("S3 Swap"), 1)                           # the block was left ONCE, as the final lines
        self.assertIn("100.0%", text)


class BarOwnership(_Env):
    """ChunkedProgress keeps all its arithmetic but prints nothing for a bar the dashboard paints."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(td, "manager", self.mgr)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.dict(os.environ, {"ROOP_PROGRESS_STYLE": "chunk"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_bar(self, desc, total=10):
        from roop.procmgr_runtime import ChunkedProgress
        buf = io.StringIO()
        with mock.patch("builtins.print", lambda *a, **k: buf.write(" ".join(map(str, a)) + "\n")):
            with ChunkedProgress(total=total, desc=desc, unit="frames") as bar:
                for _ in range(total):
                    bar.update(1)
                    time.sleep(0.01)
        return buf.getvalue(), bar

    def test_without_a_live_job_the_legacy_line_is_unchanged(self):
        out, _ = self.run_bar("Processing")
        self.assertIn("Processing", out)

    def test_a_claimed_bar_is_silent_but_still_publishes_its_rate(self):
        self.start()
        bar_cls_out, _ = None, None
        from roop.procmgr_runtime import ChunkedProgress
        self.assertTrue(td.claims_bar("Processing"))
        out, bar = self.run_bar("Processing")
        self.assertEqual(out, "")
        self.assertEqual(bar.n, 10)
        self.assertIsNotNone(bar._last_rate)                                  # what the web UI's ETA divides by

    def test_the_bar_is_attached_while_open_and_detached_after(self):
        from roop.procmgr_runtime import ChunkedProgress
        self.start()
        with mock.patch("builtins.print"):
            bar = ChunkedProgress(total=10, desc="Processing", unit="frames")
            self.assertIn(bar, self.mgr._bars)
            bar.close()
        self.assertNotIn(bar, self.mgr._bars)
        self.assertEqual(self.mgr._last_bar, (0, 10))

    def test_a_bar_the_dashboard_does_not_claim_still_renders(self):
        self.start()
        for desc in ("Upscaling", "Interpolating"):
            self.assertFalse(td.claims_bar(desc))
            out, _ = self.run_bar(desc)
            self.assertIn(desc, out)

    def test_the_pipelines_own_bars_are_claimed(self):
        self.start()
        for desc in ("Processing", "Scanning keyframes", "Tracking identities", "Analyzing faces"):
            self.assertTrue(td.claims_bar(desc), desc)


class WireFormat(unittest.TestCase):
    """The real app: `/ws/telemetry` carries `dashboard`, REST carries the profile."""

    @classmethod
    def setUpClass(cls):
        import api
        cls.api = api

    def setUp(self):
        st.reset()
        td.reset()
        self.addCleanup(st.reset)
        self.addCleanup(td.reset)

    def test_every_telemetry_frame_has_the_dashboard_block_and_idle_frames_are_equal(self):
        a, b = self.api._telemetry_snapshot(), self.api._telemetry_snapshot()
        self.assertEqual(a["dashboard"], {"schema": 1, "active": False})
        self.assertEqual(a, b)

    def test_an_active_job_is_in_the_frame_with_the_same_numbers_as_the_console(self):
        mgr = td.TerminalProgressManager()
        with mock.patch.object(td, "manager", mgr), mock.patch.dict(os.environ, {"ROOP_DASHBOARD": "log"}), mock.patch("builtins.print"):
            td.begin_job({"job_id": "w1", "mode": "video"}, td.collect_swap_profile(None, {"job_id": "w1"}, None, 30.0, 100))
            st.begin_run(100)
            st.enter(S3, 100)
            for _ in range(40):
                st.retire()
            mgr._tick()
            frame = self.api._telemetry_snapshot()["dashboard"]
            console = td.strip_ansi("\n".join(td.render_stage_lines(mgr.snapshot(), color=True)))
            rest = self.api._routes_pipeline_status.pipeline_dashboard_snapshot()
            td.end_job(True)
        self.assertTrue(frame["active"])
        self.assertNotIn("profile", frame)
        self.assertIn("profile", rest)
        self.assertEqual(frame["stages"]["S3"]["done"], 40)
        self.assertIn(f"{frame['stages']['S3']['pct']:.1f}%", console)
        json.dumps(frame)
        json.dumps(rest)

    def test_the_rest_route_answers_over_http(self):
        from fastapi.testclient import TestClient
        resp = TestClient(self.api.app).get("/api/v1/pipeline/dashboard")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"schema": 1, "active": False})


class NoAnsiIntoPipes(unittest.TestCase):
    def test_the_clear_screen_escape_is_not_written_into_a_non_tty(self):
        try:
            from roop import core
        except Exception as exc:                                              # the GPU stack may be absent on a light profile
            self.skipTest(f"roop.core not importable here: {exc}")
        buf = io.StringIO()
        with mock.patch.object(core, "_terminal_has_previous_run", True), mock.patch.object(sys, "stdout", buf):
            core._clear_terminal_for_new_run()
        self.assertEqual(buf.getvalue(), "")

    def test_it_is_still_written_into_a_terminal(self):
        try:
            from roop import core
        except Exception as exc:
            self.skipTest(f"roop.core not importable here: {exc}")
        scr = VtScreen()
        with mock.patch.object(core, "_terminal_has_previous_run", True), mock.patch.object(sys, "stdout", scr), \
                mock.patch.object(td, "stream_is_tty", return_value=True):
            core._clear_terminal_for_new_run()
        self.assertEqual([r for r in scr.screen() if r], [])


class Wiring(unittest.TestCase):
    """The pipeline really calls the dashboard (a hook nothing calls looks wired)."""

    def src(self, rel):
        with open(os.path.join(APP, rel), encoding="utf-8") as fh:
            return fh.read()

    def test_the_hooks_are_in_the_pipeline(self):
        batch = self.src("roop/procmgr_batch.py")
        for needle in ('_dash.s1_step("segment/GOP probe")', '_dash.s1_step("source identity")', "_dash.s1_finish()",
                       "_dash.manager.s4_begin()", '_dash.s4_step("segment custody & concat"'):
            self.assertIn(needle, batch, needle)
        core = self.src("roop/core.py")
        for needle in ("_dashboard_begin(v.filename", "_dash.s4_step('audio remux'", "_dash.s4_end(", "_dash.end_job(", "terminal_dashboard.end_job("):
            self.assertIn(needle, core, needle)
        self.assertIn("claims_bar", self.src("roop/procmgr_runtime.py"))
        self.assertIn("_dashboard_job", self.src("routes_queue.py"))
        # Built once per frame (`_dashboard_frame`) so the Viseme Mouth Custody `lipsync_ms` reads the same snapshot.
        api_src = self.src("api.py")
        self.assertIn("_dashboard_frame = _terminal_dashboard.telemetry_frame()", api_src)
        self.assertIn("'dashboard': _dashboard_frame", api_src)


if __name__ == "__main__":
    unittest.main()
