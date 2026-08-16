"""Terminal progress reporting: one line per chunk, not one per frame.

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
from contextlib import redirect_stdout

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
    def test_one_line_per_chunk_not_per_frame(self):
        lines = self.run_frames(1000, every=100)
        self.assertEqual(len(lines), 10,
                         "expected one line per 100-frame chunk, got:\n" + "\n".join(lines))

    def test_scales_with_the_render_not_with_update_calls(self):
        """The property that matters: a ten-times-longer render must cost ten
        times the lines, not ten times per frame."""
        short = self.run_frames(500, every=100)
        long = self.run_frames(5000, every=100)
        self.assertEqual(len(short), 5)
        self.assertEqual(len(long), 50)

    def test_every_line_says_which_chunk_of_how_many(self):
        lines = self.run_frames(400, every=100)
        for i, line in enumerate(lines, start=1):
            self.assertIn(f"chunk {i}/4", line, f"line {i} did not name its chunk: {line}")
            self.assertIn("Processing", line)

    def test_last_line_reaches_100_percent_exactly_once(self):
        lines = self.run_frames(450, every=100)     # does NOT divide evenly
        self.assertIn("100.0%", lines[-1])
        self.assertEqual(sum(1 for l in lines if "100.0%" in l), 1,
                         "100% reported more than once:\n" + "\n".join(lines))

    def test_no_duplicate_final_line_when_total_divides_evenly(self):
        """A total that lands exactly on a chunk boundary gets its final line
        from the update AND from close() unless close() checks."""
        lines = self.run_frames(400, every=100)
        self.assertEqual(len(lines), 4)
        self.assertEqual(len(set(lines)), len(lines), "duplicate line emitted")

    def test_counts_and_percentage_are_right(self):
        lines = self.run_frames(300, every=100)
        self.assertIn("100/300", lines[0])
        self.assertIn(" 33.3%", lines[0])
        self.assertIn("300/300", lines[-1])

    def test_slow_chunk_still_reports_on_a_timer(self):
        """A chunk of a thousand frames can take minutes; a terminal silent that
        long reads as a hang, so time triggers a line too."""
        os.environ["ROOP_PROGRESS_SECS"] = "0.05"
        lines = self.run_frames(6, every=100_000, sleep=0.03)
        self.assertGreater(len(lines), 1,
                           "the time-based fallback never fired")

    def test_rate_reflects_the_chunk_not_the_lifetime_average(self):
        """A stretch that ran slower than the last one is the whole reason to
        print per-chunk lines, so the rate has to be the chunk's own."""
        os.environ["ROOP_PROGRESS_EVERY"] = "20"
        buf = io.StringIO()
        with redirect_stdout(buf):
            with ChunkedProgress(total=40, desc="Processing", unit="frames") as p:
                for i in range(40):
                    time.sleep(0.05 if i < 20 else 0.0)
                    p.update(1)
        rates = [float(m.group(1)) for m in
                 (re.search(r"([\d.]+) frames/s", strip(l))
                  for l in buf.getvalue().splitlines() if l.strip()) if m]
        self.assertEqual(len(rates), 2, f"expected two chunk lines, got {rates}")
        self.assertGreater(rates[1], rates[0] * 2,
                           f"the fast chunk did not report a faster rate: {rates}")

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
        self.assertIn("100 frame", lines[0])

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
        self.assertEqual(len(lines), 2)

    def test_chunk_size_defaults_to_the_resume_segment(self):
        """A line in the terminal and a tab in the console's part strip should
        cover the same stretch of the render."""
        os.environ.pop("ROOP_PROGRESS_EVERY", None)
        os.environ["ROOP_RESUME_CHUNK"] = "250"
        buf = io.StringIO()
        with redirect_stdout(buf):
            with ChunkedProgress(total=1000, desc="Processing", unit="frames") as p:
                for _ in range(1000):
                    p.update(1)
        self.assertEqual(len([l for l in buf.getvalue().splitlines() if l.strip()]), 4)


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
