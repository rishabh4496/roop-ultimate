"""The autotuner must not promote a candidate that noise alone could produce.

Gate A finding 12 (2026-08-30) recorded that `MIN_IMPROVEMENT = 1%` sits below
this project's measured run-to-run spread on both validation targets. It was
confirmed on hardware on 2026-08-31: two runs of the same deterministic search
on the same RTX 4070 returned "0.0%, promote nothing" and "+3.59%, promote
trt_context_count=1", the latter from twelve candidates spanning 5.45-5.77 fps.

The acceptance threshold is therefore measured on the live machine from
repeated baseline runs instead of being a constant tuned on one GPU.
"""
import unittest

from roop.runtime_optimizer import (HardwareProfile, RuntimeAutotuner,
                                    RuntimeTuning, WorkloadProfile)


class _Harness:
    """Feeds a scripted fps sequence to the tuner's measure callback."""

    def __init__(self, fps_sequence):
        self.fps = list(fps_sequence)
        self.calls = 0

    def __call__(self, candidate, warmup_frames):
        value = self.fps[min(self.calls, len(self.fps) - 1)]
        self.calls += 1
        return {"end_to_end_fps": value, "stable": True,
                "quality_regression": False, "startup_seconds": 0.0}


def _fixtures():
    hardware = HardwareProfile(gpu_name="RTX 4070", vram_total_gb=12.0,
                               ram_total_gb=32.0)
    workload = WorkloadProfile(input_width=1280, input_height=720)
    return RuntimeTuning(), hardware, workload


class AutotunerNoiseFloorTest(unittest.TestCase):

    def test_noisy_baseline_raises_the_acceptance_threshold(self):
        """A 3.6% candidate must lose when the baseline itself spans 5%."""
        base, hardware, workload = _fixtures()
        # Three baseline replicates spanning 5.45..5.77 (~5.7%), then every
        # candidate at 5.77 -- the exact shape of the 2026-08-31 4070 run.
        harness = _Harness([5.57, 5.45, 5.77] + [5.77] * 20)
        _, report = RuntimeAutotuner().tune(base, hardware, workload,
                                            measure=harness)
        self.assertGreaterEqual(report["baseline_replicates"], 2)
        self.assertGreater(report["measured_noise_spread"], 0.05)
        self.assertGreater(report["min_improvement_used"],
                           RuntimeAutotuner.MIN_IMPROVEMENT)
        self.assertLessEqual(report["improvement_pct"], 0.0)

    def test_quiet_baseline_still_accepts_a_real_win(self):
        """A stable machine must still be able to promote a genuine gain."""
        base, hardware, workload = _fixtures()
        harness = _Harness([10.00, 10.00, 10.00] + [13.00] * 20)
        _, report = RuntimeAutotuner().tune(base, hardware, workload,
                                            measure=harness)
        self.assertLess(report["measured_noise_spread"], 0.01)
        self.assertGreater(report["improvement_pct"], 10.0)

    def test_threshold_never_falls_below_the_constant_floor(self):
        base, hardware, workload = _fixtures()
        harness = _Harness([10.0] * 24)
        _, report = RuntimeAutotuner().tune(base, hardware, workload,
                                            measure=harness)
        self.assertGreaterEqual(report["min_improvement_used"],
                                RuntimeAutotuner.MIN_IMPROVEMENT)

    def test_baseline_is_measured_more_than_once(self):
        base, hardware, workload = _fixtures()
        harness = _Harness([10.0] * 24)
        RuntimeAutotuner().tune(base, hardware, workload, measure=harness)
        self.assertGreaterEqual(harness.calls,
                                RuntimeAutotuner.BASELINE_REPLICATES)


if __name__ == "__main__":
    unittest.main()


class AutotunerConfirmationTest(unittest.TestCase):
    """A win must survive being measured a second time.

    Measured on the RTX 4070 on 2026-08-31: three baseline replicates spanned
    only 1.64% while the twelve candidates spanned 5.11-5.67 fps, so the
    replicate spread alone still let a lucky single run through. The winner is
    re-measured before it is promoted.
    """

    def test_lucky_single_run_is_rejected_on_reconfirmation(self):
        base, hardware, workload = _fixtures()
        # baseline x3, then candidates; one candidate spikes once and reverts.
        harness = _Harness([5.49, 5.42, 5.51,
                            5.42, 5.51, 5.56, 5.67] + [5.40] * 20)
        tuning, report = RuntimeAutotuner().tune(base, hardware, workload,
                                                 measure=harness)
        if report.get("confirmation") is not None:
            self.assertFalse(report["confirmation"]["confirmed"])
        self.assertLessEqual(report["improvement_pct"], 0.0)

    def test_repeatable_win_is_still_promoted(self):
        base, hardware, workload = _fixtures()
        harness = _Harness([10.0, 10.0, 10.0] + [14.0] * 20)
        _, report = RuntimeAutotuner().tune(base, hardware, workload,
                                            measure=harness)
        self.assertIsNotNone(report["confirmation"])
        self.assertTrue(report["confirmation"]["confirmed"])
        self.assertGreater(report["improvement_pct"], 10.0)
