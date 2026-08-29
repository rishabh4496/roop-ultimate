"""The Phase 2 baseline parser must not read the wrong number as the FPS.

This guards a defect that nearly reached `PERFORMANCE_BASELINE.md`, which is the
locked reference every later phase and both GPUs are compared against. The first
version of `parse_run` searched the whole log with `([\\d.]+)\\s*fps` and matched
the SOURCE CLIP's geometry line, reporting 30.0 fps for a render that ran at
12.20. Nothing about "30 fps" looks wrong -- only the arithmetic against the
frame count and the elapsed seconds gives it away, which is why the parser now
cross-checks and refuses to record a disagreement.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import baseline_controlled as bc


class ParseRunFpsTest(unittest.TestCase):

    GOOD = ("Processing clip_15-04-58.mp4 took 49.18 secs, 12.20 frames/s\n"
            "[bench] output: 600 frames\n")

    def test_reads_the_authoritative_line(self):
        r = bc.parse_run(self.GOOD)
        self.assertEqual(r["frames"], 600)
        self.assertAlmostEqual(r["processing_seconds"], 49.18, places=2)
        self.assertAlmostEqual(r["fps"], 12.20, places=2)
        self.assertAlmostEqual(r["fps_check"], 12.20, places=1)

    def test_source_clip_frame_rate_is_not_mistaken_for_the_fps(self):
        """The original defect: the clip's own 30fps elsewhere in the log."""
        noisy = ("s1.mp4 1280x720 30.0fps 19672 frames\n"
                 "some other line mentioning 25 fps\n") + self.GOOD
        r = bc.parse_run(noisy)
        self.assertAlmostEqual(r["fps"], 12.20, places=2,
                               msg="fps must come from the encoder's own line, "
                                   "not from any 'fps' token in the log")

    def test_disagreement_is_refused_rather_than_recorded(self):
        """A printed rate that contradicts frames/seconds must stop the run.

        A cross-check that is computed and never compared is decoration; this
        asserts it actually fires.
        """
        bad = ("Processing clip.mp4 took 49.18 secs, 30.00 frames/s\n"
               "[bench] output: 600 frames\n")
        with self.assertRaises(SystemExit) as ctx:
            bc.parse_run(bad)
        self.assertIn("fps disagreement", str(ctx.exception))
        self.assertIn("12.20", str(ctx.exception))

    def test_small_rounding_difference_is_tolerated(self):
        near = ("Processing clip.mp4 took 49.18 secs, 12.21 frames/s\n"
                "[bench] output: 600 frames\n")
        r = bc.parse_run(near)          # must not raise
        self.assertAlmostEqual(r["fps"], 12.21, places=2)

    def test_swap_audit_is_carried_through(self):
        txt = self.GOOD + (
            "  box 0 from the left (harjot): 477 frames, swapped 473\n"
            "      WRONG FACESET APPLIED on 0 of 206 swaps attributed to harjot\n"
            "      WRONG FACESET APPLIED on 0 of 436 swaps attributed to gargee\n")
        r = bc.parse_run(txt)
        self.assertEqual(r["wrong_faceset"], 0)
        self.assertEqual(r["attributed_swaps"], 642)


if __name__ == "__main__":
    unittest.main()
