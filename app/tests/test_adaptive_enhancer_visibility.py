"""An adaptive run that restored nothing must say so, and say why.

MEASURED, not hypothesised. On the RTX 4070, the shipped BALANCED profile on
this project's own locked fixture (`double/d4.mp4`, RealSwap / RealityUX /
TensorRT) chose `none` for 60 of 60 faces. Every instrument called it a pass:

  * return code 0, valid output video;
  * swap audit `swapped 100.0%` -- it counts faces handed to the swap;
  * a `ROOP_PROFILE` `enhance` stage with 60 calls -- the WRAPPER ran, and
    returns immediately when it selects `none`, so the count proves nothing
    about restoration;
  * 1.95 fps, the FASTEST arm of a 14-enhancer sweep, ahead of `--enhancer
    None` at 1.87 -- fast because it was not enhancing.

The old summary line was `decisions={'none': 60} last_quality={}`, which gives
the outcome and not the cause, and `last_quality` is empty in exactly this case
because it is only written after a candidate actually runs. So the one number
that explains the behaviour was missing precisely when it was needed.

The behaviour itself is the selector's stated policy and is NOT changed here.
It engages on harder material: on `double/d6.mp4` the same profile chose
`gpen_realistic` for 18 of 73 faces, with the rest refused by
`extreme-angle-geometry-first` rather than by the quality cut, over a quality
band of 0.42-0.47 against d4's 0.77-0.82. No threshold was re-tuned, because
this campaign measured the distribution the gate reads and not whether
restoration would have improved those faces -- and four gate changes in this
project's history were implemented and reverted for exactly that gap.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for p in (APP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from roop.adaptive_enhancer import (NONE, AdaptiveEnhancer,  # noqa: E402
                                    Decision, choose_enhancer)


def _metrics(quality, **over):
    m = {"quality": quality, "pose": 0.9, "temporal_stability": 0.9,
         "confidence": 0.9, "occlusion": 0.0, "low_light_tier": "NORMAL",
         "output_quality": 0.8}
    m.update(over)
    return m


class TelemetryExplainsTheDecision(unittest.TestCase):
    def _enh(self, decisions):
        enh = AdaptiveEnhancer()
        enh._decisions = list(decisions)
        return enh

    def test_reasons_are_counted_not_discarded(self):
        enh = self._enh([Decision(NONE, "high-quality-face-minimal-enhancement",
                                  _metrics(0.80)) for _ in range(3)])
        self.assertEqual({"high-quality-face-minimal-enhancement": 3},
                         enh.telemetry()["reasons"])

    def test_quality_band_reports_the_population_the_gate_read(self):
        """min/p50/max, because a single number cannot show a whole population.

        The actionable fact about the d4 run was not "it refused" but "every
        face scored 0.77-0.82 against a 0.68 cut" -- the population is nowhere
        near the threshold, so this is not a marginal calibration question.
        """
        enh = self._enh([Decision(NONE, "r", _metrics(q))
                         for q in (0.77, 0.80, 0.82)])
        band = enh.telemetry()["quality_band"]
        self.assertEqual(3, band["n"])
        self.assertAlmostEqual(0.77, band["min"], places=4)
        self.assertAlmostEqual(0.82, band["max"], places=4)

    def test_quality_band_is_none_when_nothing_was_decided(self):
        self.assertIsNone(AdaptiveEnhancer().telemetry()["quality_band"])

    def test_a_decision_without_a_quality_metric_does_not_crash_the_summary(self):
        """The summary must survive a malformed metrics dict.

        A diagnostic that raises while reporting a problem removes the only
        evidence of the problem.
        """
        enh = self._enh([Decision(NONE, "r", {}),
                         Decision(NONE, "r", {"quality": "not-a-number"})])
        self.assertIsNone(enh.telemetry()["quality_band"])
        self.assertEqual({"r": 2}, enh.telemetry()["reasons"])


class FallbackReportingIsBounded(unittest.TestCase):
    """Both fallback sites sit on the per-face path.

    Unbounded, that is two lines per face: 120,000 on a 60,000-frame two-face
    render, which is its own failure. The running total still has to reach the
    user, so it is counted and reported at Release.
    """

    def test_repeated_identical_failures_are_counted_once_reported_once(self):
        enh = AdaptiveEnhancer()
        for _ in range(25):
            enh._report_fallback("ultramax", "failed", RuntimeError("boom"))
        counts = enh.fallback_counts()
        self.assertEqual(1, len(counts))
        self.assertEqual(25, sum(counts.values()))

    def test_distinct_causes_are_kept_apart(self):
        enh = AdaptiveEnhancer()
        enh._report_fallback("ultramax", "failed", RuntimeError("boom"))
        enh._report_fallback("ultramax", "unavailable", ImportError("nope"))
        self.assertEqual(2, len(enh.fallback_counts()))

    def test_fallbacks_reach_the_telemetry_summary(self):
        enh = AdaptiveEnhancer()
        enh._report_fallback("gpen_realistic", "failed", ValueError("x"))
        self.assertEqual(1, sum(enh.telemetry()["fallbacks"].values()))


class TheRefusalPopulationIsWhatWasMeasured(unittest.TestCase):
    """Pin the gate this campaign actually observed firing.

    Not a re-derivation of the policy -- a regression anchor. If a future edit
    moves BALANCED's cut above the measured d4 band, this fails and the change
    has to be justified against real footage rather than intuition.
    """

    def test_the_measured_d4_band_is_refused_by_every_profile(self):
        for profile in ("FAST", "BALANCED", "REALISTIC", "MAX QUALITY"):
            for q in (0.7665, 0.7994, 0.8188):     # d4 min / p50 / max
                path, reason = choose_enhancer(_metrics(q), profile)
                self.assertEqual(NONE, path,
                                 "%s no longer refuses q=%s" % (profile, q))
                self.assertEqual("high-quality-face-minimal-enhancement", reason)

    def test_the_measured_d6_band_still_selects_a_restorer(self):
        """The selector must not be refusing everything everywhere.

        d6's band (0.42-0.47) is where it engaged, choosing gpen_realistic for
        18 faces. If this stops selecting, "adaptive" has become "off".
        """
        path, reason = choose_enhancer(_metrics(0.4341), "BALANCED")
        self.assertNotEqual(NONE, path)
        self.assertIn("restoration", reason)


if __name__ == "__main__":
    unittest.main()
