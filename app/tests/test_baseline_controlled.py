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


class AdaptiveDowngradeTests(unittest.TestCase):
    """A run that quietly dropped half its stack must not read as comparable.

    The first RTX 3060 baseline recorded `provider: tensorrt` and
    `enhancer: GPEN 256 Pro` -- the values it REQUESTED from config.yaml --
    while the sub-7GB policy had already disabled TensorRT, dropped the enhancer
    to None, degraded RealityUX and forced CPU decode. Matching the locked
    fixture is necessary for comparability but not sufficient: an arm doing less
    work is not this machine's answer to the 4070's number.
    """

    LOG = (
        "[Backend] sub-7GB GPU: TensorRT disabled by the laptop RSS safety "
        "policy; using CUDA/CPU providers.\n"
        "[RuntimeOptimizer] sub-7GB RSS safety: enhancer 'GPEN 256 Pro' -> "
        "'None'; measured enhancer path exceeds the strict 2.5GB RSS gate.\n"
        "[Mask_RealityUX] sub-7GB CUDA profile: retaining the authoritative "
        "XSeg mask and skipping the auxiliary BiSeNet parser.\n"
        "[RuntimeOptimizer] sub-7GB decode safety: NVDEC -> CPU; measured "
        "NVDEC path increases RSS without an end-to-end speed win.\n")

    def test_detects_every_downgrade(self):
        found = bc.parse_adaptive_downgrades(self.LOG)
        self.assertEqual(sorted(found),
                         ["decode", "enhancer", "mask_engine", "provider"])
        self.assertIn("GPEN 256 Pro", found["enhancer"])
        self.assertIn("None", found["enhancer"])

    def test_clean_log_reports_nothing(self):
        self.assertEqual(
            bc.parse_adaptive_downgrades(
                "[Runtime] all good\n[Track] 4 tracks over 600 frames\n"),
            {})

    def test_enhancer_name_is_read_not_assumed(self):
        found = bc.parse_adaptive_downgrades(
            "sub-7GB RSS safety: enhancer 'UltraMax' -> 'None';")
        self.assertIn("UltraMax", found["enhancer"])


if __name__ == "__main__":
    unittest.main()


class FixtureDeterminismTest(unittest.TestCase):
    """The locked baseline's capture must not depend on machine speed.

    The auto-capture scan used to be bounded by wall clock. On 2026-08-31 an
    RTX 4070 counterbalanced set scanned 646/629/598/409 frames in the same
    30-second box, and the short arm therefore selected seed frame 2930 with
    separation 0.990 where every other arm selected 4930/1.039 -- a different
    pair of source captures under the same fixture name. Across targets it is
    worse: the 3060 is about half the speed, so it buys about half the scan.
    """

    def test_capture_frame_is_pinned_in_the_workload(self):
        self.assertIn("capture_frame", bc.WORKLOAD)
        self.assertIsInstance(bc.WORKLOAD["capture_frame"], int)
        self.assertGreater(bc.WORKLOAD["capture_frame"], 0)

    def test_harness_passes_an_explicit_capture_frame(self):
        """Never `--capture -1`, and never a wall-clock budget beside it.

        Asserted against the QUOTED CLI forms, so that prose explaining the
        defect does not itself trip the guard -- the first version of this test
        failed on its own comment.
        """
        with open(bc.__file__, encoding="utf-8") as fh:
            source = fh.read()
        self.assertNotIn('"--capture", "-1"', source)
        self.assertNotIn('"--capture-budget"', source)
        self.assertIn('"--capture", str(WORKLOAD["capture_frame"])', source)


if __name__ == "__main__":
    unittest.main()


class Phase10PassThroughTests(unittest.TestCase):
    """The controlled harness must be able to express the Phase 10 arm.

    `two_face_video.py` has owned `--target-conditioned-appearance` since the
    feature landed, but `baseline_controlled.py` never forwarded it, so the
    documented 3060 validation arm could not be run through the controlled
    harness at all -- the same shape as the 2026-08-23 finding that the bench
    populated no `merger_*` global, and the Phase 15 controller that was never
    reachable. An unreachable arm silently measures the baseline twice.
    """

    def _source(self):
        import io
        return io.open(bc.__file__, encoding="utf-8").read()

    def test_harness_declares_the_appearance_override(self):
        self.assertIn("--target-appearance-mode", self._source())

    def test_on_forwards_the_child_flag(self):
        self.assertIn("--target-conditioned-appearance", self._source())

    def test_off_does_not_forge_an_explicit_disable(self):
        """The child cannot express "off"; off must be silence, not strength 0.

        Passing `--target-conditioned-strength 0.0` would read as an off arm
        while leaving `target_conditioned_appearance` at whatever the config
        said, which is a different configuration than the one the arm claims.
        """
        src = self._source()
        self.assertNotIn('"--target-conditioned-strength", "0.0"', src)


class ResolvedFeatureStateTests(unittest.TestCase):
    """Record the state the CHILD resolved, never the one we asked for."""

    ECHO = ("[bench] provider=tensorrt threads=8 tracking=1 "
            "swap_model_mask=0 merger_clarity=0.0 "
            "identity_detail=0.35 target_appearance=True "
            "target_strength=0.75 target_alpha=0.3 "
            "temporal_compositing=False\n"
            "video took 140.0 secs, 4.29 frames/s\n"
            "[bench] output: 600 frames\n")

    def test_reads_back_the_opt_in_feature_state(self):
        out = bc.parse_run(self.ECHO)
        self.assertEqual(out.get("feature_identity_detail"), 0.35)
        self.assertIs(out.get("feature_target_appearance"), True)
        self.assertEqual(out.get("feature_target_strength"), 0.75)
        self.assertEqual(out.get("feature_target_alpha"), 0.3)

    def test_an_off_arm_is_recorded_as_off_not_as_absent(self):
        off = self.ECHO.replace("target_appearance=True",
                                "target_appearance=False")
        out = bc.parse_run(off)
        self.assertIs(out.get("feature_target_appearance"), False)

    def test_absent_echo_records_nothing_rather_than_guessing(self):
        out = bc.parse_run("video took 140.0 secs, 4.29 frames/s\n"
                           "[bench] output: 600 frames\n")
        self.assertNotIn("feature_target_appearance", out)
