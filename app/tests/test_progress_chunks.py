"""Terminal progress reporting: fixed wall-clock updates, not chunk commits.

A progress bar is a thing you rewrite in place, which needs a terminal to move
the cursor on AND nothing else printing. During a render neither holds: output
goes to Pinokio's captured log, and the swap loop prints its own diagnostics,
each of which terminates the bar's line. Measured on a real 48,501-frame render
before this: 340 bar lines across the last 671 frames — a median of one 451-
character line per frame, with every message worth reading buried under them.

So the guarantee worth testing is a COUNT: a render of N frames must produce
about N/chunk lines, and that must hold no matter how often the render loop
calls update(). These tests assert that property rather than any particular
wording, plus the two things that make the line trustworthy — it always ends on
100%, and it never says 100% twice.
"""

import io
import os
import re
import sys
import time
import unittest
from collections import deque
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.procmgr_runtime import ChunkedProgress, bar_write  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def strip(s):
    return ANSI.sub("", s)


class ProgressEnv(unittest.TestCase):
    """Base that isolates the env these read on every construction."""

    VARS = ("ROOP_PROGRESS_STYLE", "ROOP_PROGRESS_EVERY", "ROOP_PROGRESS_SECS",
            "ROOP_RESUME_CHUNK")

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self.VARS}
        os.environ["ROOP_PROGRESS_STYLE"] = "chunk"
        os.environ["ROOP_PROGRESS_EVERY"] = "100"
        os.environ["ROOP_PROGRESS_SECS"] = "9999"   # frame-driven unless asked

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def run_frames(self, total, every=None, desc="Processing", unit="frames",
                   sleep=0.0):
        if every is not None:
            os.environ["ROOP_PROGRESS_EVERY"] = str(every)
        buf = io.StringIO()
        with redirect_stdout(buf):
            with ChunkedProgress(total=total, desc=desc, unit=unit) as p:
                for _ in range(total):
                    if sleep:
                        time.sleep(sleep)
                    p.update(1)
        return [strip(l) for l in buf.getvalue().splitlines() if l.strip()]


class ChunkedOutputTest(ProgressEnv):
    def test_fast_render_emits_only_its_final_state(self):
        lines = self.run_frames(1000, every=1)
        self.assertEqual(len(lines), 1,
                         "a frame-count chunk must not force terminal redraws")

    def test_frame_chunk_environment_does_not_change_cadence(self):
        """ROOP_PROGRESS_EVERY is retained for compatibility but never controls FPS."""
        short = self.run_frames(500, every=100)
        long = self.run_frames(500, every=1)
        self.assertEqual(len(short), 1)
        self.assertEqual(len(long), 1)

    def test_terminal_line_has_progress_without_chunk_numbers(self):
        lines = self.run_frames(400, every=100)
        self.assertIn("Processing", lines[-1])
        self.assertNotIn("chunk", lines[-1].lower())

    def test_last_line_reaches_100_percent_exactly_once(self):
        lines = self.run_frames(450, every=100)     # does NOT divide evenly
        self.assertIn("100.0%", lines[-1])
        self.assertEqual(sum(1 for l in lines if "100.0%" in l), 1,
                         "100% reported more than once:\n" + "\n".join(lines))

    def test_no_duplicate_final_line_when_total_divides_evenly(self):
        """A completion update and close() must not duplicate the final state."""
        lines = self.run_frames(400, every=100)
        self.assertEqual(len(lines), 1)
        self.assertEqual(len(set(lines)), len(lines), "duplicate line emitted")

    def test_counts_and_percentage_are_right(self):
        lines = self.run_frames(300, every=100)
        self.assertIn("300/300", lines[0])
        self.assertIn("100.0%", lines[0])

    def test_slow_chunk_still_reports_on_a_timer(self):
        """A slow render reports every 500 ms, independent of frame chunks."""
        lines = self.run_frames(6, every=100_000, sleep=0.12)
        self.assertGreater(len(lines), 1,
                           "the time-based fallback never fired")

    def _drive(self, schedule, total=None):
        """Feed a bar an exact arrival schedule on a fake clock.

        Returns (reported_rate, true_rate). The schedule is a list of gaps in
        seconds between successive update(1) calls, so the true aggregate rate
        is len(schedule) / sum(schedule) by construction.
        """
        clock = [0.0]
        with redirect_stdout(io.StringIO()),                 patch("roop.procmgr_runtime.time.perf_counter", lambda: clock[0]):
            p = ChunkedProgress(total=total or len(schedule), desc="Processing",
                                unit="frames")
            p._last_t = 0.0
            p._completion_times = deque([(0.0, 0)])
            for gap in schedule:
                clock[0] += gap
                p.update(1)
            p._refresh_rate()
            reported = p.rolling_rate or 0.0
        return reported, len(schedule) / clock[0]

    def test_rate_ignores_how_arrivals_are_spaced(self):
        """Bursty and evenly-spaced arrivals at one true rate must agree.

        Both stages that own a bar here are fed by a worker POOL: a batch of
        detections lands within microseconds, then the loop blocks on the
        dispatch cap. Averaging per-update `n / dt` samples let those
        microsecond gaps dominate (E[1/dt] >> 1/E[dt]) and reported a real
        pre-pass running at 42.8 frames/s as 230.34, with "02:17 left" against
        a true ~12 minutes. On this synthetic 50 f/s schedule it read 39,692.
        """
        bursty = []
        for _ in range(400):
            bursty += [0.00002] * 7 + [0.16]        # 8 workers, then a block
        smooth = [sum(bursty) / len(bursty)] * len(bursty)

        burst_rate, burst_true = self._drive(bursty)
        smooth_rate, smooth_true = self._drive(smooth)

        self.assertAlmostEqual(burst_true, smooth_true, places=6)
        for reported, true in ((burst_rate, burst_true), (smooth_rate, smooth_true)):
            self.assertAlmostEqual(reported, true, delta=true * 0.05)
        self.assertAlmostEqual(burst_rate, smooth_rate, delta=smooth_rate * 0.05)

    def test_rate_ignores_free_skip_frames(self):
        """ROOP_TEMPORAL_STEP > 1 calls update(1) with no work in between.

        Those cost microseconds and used to report ~1000x the true rate.
        """
        schedule = []
        for _ in range(400):
            schedule += [0.00001, 0.00001, 0.05]    # two skips, one real frame
        reported, true = self._drive(schedule)
        self.assertAlmostEqual(reported, true, delta=true * 0.05)

    def test_eta_is_derived_from_the_honest_rate(self):
        """The published ETA divides by the rate, so it inherits its accuracy."""
        from roop.procmgr_runtime import _bar_eta_seconds
        clock = [0.0]
        with redirect_stdout(io.StringIO()),                 patch("roop.procmgr_runtime.time.perf_counter", lambda: clock[0]):
            p = ChunkedProgress(total=1000, desc="Processing", unit="frames")
            p._last_t = 0.0
            p._completion_times = deque([(0.0, 0)])
            for _ in range(50):                      # bursts of 8 at 50 f/s
                for gap in ([0.00002] * 7 + [0.16]):
                    clock[0] += gap
                    p.update(1)
            p._refresh_rate()
            eta = _bar_eta_seconds(p)
        # 600 frames remain at ~50 f/s: ~12 s, not the fraction of a second the
        # old rate implied.
        self.assertAlmostEqual(eta, (1000 - p.n) / 50.0, delta=1.0)

    def test_rate_uses_the_rolling_completion_ema(self):
        """``display = .15 * current + .85 * previous`` exactly."""
        with redirect_stdout(io.StringIO()):
            with ChunkedProgress(total=100, desc="Processing", unit="frames") as p:
                p._completion_times = deque([(0.0, 0), (1.0, 20)])
                p._refresh_rate()
                self.assertAlmostEqual(p.rolling_rate, 20.0)
                p._completion_times = deque([(1.0, 20), (2.0, 60)])
                p._refresh_rate()
                self.assertAlmostEqual(p.rolling_rate, 23.0)

    def test_draws_on_500ms_boundaries_not_completion_count(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            with ChunkedProgress(total=4, desc="Processing", unit="frames") as p:
                p._last_t = 0.0
                p._completion_times = deque([(0.0, 0)])
                with patch("roop.procmgr_runtime.time.perf_counter",
                           side_effect=(0.49, 0.50, 0.99, 1.00)):
                    for _ in range(4):
                        p.update(1)
        lines = [strip(line) for line in buf.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(lines), 2, lines)
        self.assertIn("2/4", lines[0])
        self.assertIn("4/4", lines[1])

    def test_unknown_total_is_reported_without_crashing(self):
        """Stages fed from a pipe (upscale, interpolate) may not know the total."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            with ChunkedProgress(total=None, desc="Upscaling", unit="frame") as p:
                for _ in range(250):
                    p.update(1)
        lines = [strip(l) for l in buf.getvalue().splitlines() if l.strip()]
        self.assertTrue(lines)
        self.assertIn("Upscaling", lines[0])
        self.assertIn("250 frame", lines[0])

    def test_zero_frame_stage_says_nothing(self):
        lines = self.run_frames(0, every=100)
        self.assertEqual(lines, [])


class DropInBehaviourTest(ProgressEnv):
    """ProcessMgr reads progress.n and format_dict['rate'] to drive the web UI's
    progress and ETA. Suppressing the DRAWING must not disturb the arithmetic."""

    def test_n_tracks_updates_in_chunk_mode(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            with ChunkedProgress(total=250, desc="Processing", unit="frames") as p:
                for _ in range(250):
                    p.update(1)
                self.assertEqual(p.n, 250)
                self.assertEqual(p.format_dict.get("total"), 250)

    def test_set_postfix_is_carried_into_the_line(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            with ChunkedProgress(total=100, desc="Processing", unit="frames") as p:
                for _ in range(100):
                    p.set_postfix({"memory_usage": "3.26GB",
                                   "execution_threads": "8"}, refresh=False)
                    p.update(1)
        self.assertIn("memory_usage=3.26GB", strip(buf.getvalue()))

    def test_bar_style_draws_a_bar_and_prints_no_chunk_lines(self):
        """On a real terminal this must still be an ordinary tqdm bar."""
        os.environ["ROOP_PROGRESS_STYLE"] = "bar"
        stream = io.StringIO()
        buf = io.StringIO()
        with redirect_stdout(buf):
            with ChunkedProgress(total=100, desc="Processing", unit="frames",
                                 file=stream, mininterval=0) as p:
                for _ in range(100):
                    p.update(1)
        self.assertNotIn("chunk", strip(buf.getvalue()),
                         "bar style should not print chunk lines")
        self.assertIn("Processing", strip(stream.getvalue()))

    def test_unknown_style_falls_back_to_auto(self):
        os.environ["ROOP_PROGRESS_STYLE"] = "nonsense"
        # Under the test runner stdout/stderr are not terminals, so auto means
        # chunked — the assertion is that it resolves rather than raising.
        lines = self.run_frames(200, every=100)
        self.assertEqual(len(lines), 1)

    def test_resume_segment_does_not_change_display_cadence(self):
        """Resume segmentation must not change the wall-clock display cadence."""
        os.environ.pop("ROOP_PROGRESS_EVERY", None)
        os.environ["ROOP_RESUME_CHUNK"] = "250"
        buf = io.StringIO()
        with redirect_stdout(buf):
            with ChunkedProgress(total=1000, desc="Processing", unit="frames") as p:
                for _ in range(1000):
                    p.update(1)
        self.assertEqual(len([l for l in buf.getvalue().splitlines() if l.strip()]), 1)


class BarWriteTest(unittest.TestCase):
    def test_writes_the_message(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            bar_write("part 12 written")
        self.assertIn("part 12 written", buf.getvalue())

    def test_never_raises_on_an_unencodable_character(self):
        """run.py puts stdout into UTF-8, but if that ever fails a single ✓ in a
        status line would raise — and a diagnostic must not kill an hour-long
        render."""
        class Cp1252Stream(io.StringIO):
            encoding = "cp1252"

            def write(self, s):
                s.encode("cp1252")      # raises exactly as the real console does
                return super().write(s)

        with redirect_stdout(Cp1252Stream()):
            bar_write("✓ part 12 written")     # must not raise

    def test_survives_a_broken_stream(self):
        class Broken(io.StringIO):
            def write(self, _s):
                raise IOError("pipe closed")

        with redirect_stdout(Broken()):
            bar_write("anything")               # must not raise


if __name__ == "__main__":
    unittest.main()
