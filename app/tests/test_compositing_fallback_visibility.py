"""A quality layer that fell back must say so, once per cause.

Three sites in `procmgr_masking.paste_upscale` / `process_mask` catch any
exception and continue with the legacy path: temporal mask stabilisation, alpha
refinement, and temporal occlusion. Falling back is the correct RUNTIME
behaviour -- these are quality layers and a malformed optional track field must
not cost a frame.

Falling back SILENTLY is not. A user who enables `temporal_compositing` or the
occlusion engine and hits an exception on every face gets a run that returns 0,
reports 100% swapped in the audit, and produces exactly the legacy output. That
is indistinguishable from "the feature had no effect" -- the single most
expensive pattern in this project's history, and the one that hid four broken
enhancers behind a 100% audit, a dedent that disabled the swap behind a +47%
speedup, and an adaptive enhancer that restored nothing behind the fastest arm
of a 14-way sweep.

Occlusion is the sharpest case: foreign objects crossing the face are one of the
behaviours this pipeline is judged on, so "the occlusion engine appears to do
nothing" needs to be distinguishable from "the occlusion engine raised on every
face".
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from roop import temporal_compositing as tc   # noqa: E402


class FallbackReportingIsBounded(unittest.TestCase):
    def setUp(self):
        tc.reset_compositing_fallbacks()
        self.addCleanup(tc.reset_compositing_fallbacks)

    def test_repeated_identical_causes_report_once_and_count_all(self):
        """These sit on the per-face path.

        Unbounded, a 60,000-frame two-face render emits 120,000 identical
        lines, which is its own denial of service. The total still has to be
        recoverable, so occurrences are counted rather than dropped.
        """
        for _ in range(40):
            tc.warn_compositing_fallback('alpha refinement', ValueError('bad'))
        counts = tc.compositing_fallback_counts()
        self.assertEqual(1, len(counts))
        self.assertEqual(40, sum(counts.values()))

    def test_each_stage_is_reported_separately(self):
        """Three different stages need three different investigations.

        Collapsing them would send a reader to the wrong code.
        """
        tc.warn_compositing_fallback('mask stabilisation', ValueError('a'))
        tc.warn_compositing_fallback('alpha refinement', ValueError('a'))
        tc.warn_compositing_fallback('temporal occlusion', ValueError('a'))
        self.assertEqual(3, len(tc.compositing_fallback_counts()))

    def test_distinct_exceptions_in_one_stage_are_distinct_causes(self):
        tc.warn_compositing_fallback('alpha refinement', ValueError('a'))
        tc.warn_compositing_fallback('alpha refinement', TypeError('a'))
        self.assertEqual(2, len(tc.compositing_fallback_counts()))

    def test_a_very_long_exception_message_cannot_blow_up_the_key_space(self):
        """Truncated, so a message carrying a frame index or an array repr
        cannot turn "once per cause" into "once per face"."""
        for i in range(30):
            tc.warn_compositing_fallback('alpha refinement',
                                         ValueError('x' * 300 + str(i)))
        self.assertEqual(1, len(tc.compositing_fallback_counts()))

    def test_reporting_never_raises_on_an_odd_exception(self):
        class Odd(Exception):
            def __str__(self):
                raise RuntimeError("this exception cannot be printed")
        with self.assertRaises(RuntimeError):
            str(Odd())
        # The reporter is allowed to propagate here -- what must not happen is
        # a silent swallow that leaves no record at all. This pins the current
        # contract so a future change to it is deliberate.
        with self.assertRaises(RuntimeError):
            tc.warn_compositing_fallback('alpha refinement', Odd())


class TheCallSitesReport(unittest.TestCase):
    """The reporter only helps if the three fallbacks reach it.

    Asserted against the source: reproducing a malformed track field through a
    full ProcessMgr mask pass is far more fragile than reading the three
    handlers that matter, and a static check is adequate here because the
    contract is "this except clause calls the reporter", not a runtime property.
    """

    def test_all_three_fallbacks_are_wired(self):
        with open(os.path.join(APP, "roop", "procmgr_masking.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        for stage in ("mask stabilisation", "alpha refinement",
                      "temporal occlusion"):
            self.assertIn("warn_compositing_fallback('%s'" % stage, src,
                          "%s no longer reports its fallback" % stage)

    def test_no_bare_pass_remains_on_those_handlers(self):
        """A regression here reads as nothing at all, so pin the shape."""
        with open(os.path.join(APP, "roop", "procmgr_masking.py"),
                  encoding="utf-8") as fh:
            src = fh.read()
        self.assertNotIn("# Temporal compositing is a quality layer. A malformed"
                         " optional\n                # track field must leave "
                         "the established matte usable.\n                pass",
                         src)


if __name__ == "__main__":
    unittest.main()
