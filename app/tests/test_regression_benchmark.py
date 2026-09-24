"""roop.benchmark.regression: the pure gates and metrics, plus the CLI wiring.

The render itself needs the GPU stack and is exercised by running
`python run.py --benchmark --benchmark-mode regression`; these tests pin the
parts whose mistakes would read as a pass: the verdict, the metrics and the
flag plumbing.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from roop.benchmark import regression as rg  # noqa: E402


def _report(**over):
    report = {
        "frames_in": 300, "frames_out": 300,
        "quality": {"face_ssim_mean": 0.99, "face_ssim_min": 0.95,
                    "face_psnr_mean": 40.0, "frame_psnr_mean": 45.0},
        "swap": {"changed_coverage": 0.98, "decided_coverage": 0.99},
        "perf": {"e2e_fps": 10.0},
    }
    for key, value in over.items():
        if isinstance(value, dict):
            report[key] = dict(report[key], **value)
        else:
            report[key] = value
    return report


class MetricTests(unittest.TestCase):
    def test_identical_images_are_perfect(self):
        img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        self.assertEqual(rg.psnr(img, img), 100.0)
        self.assertAlmostEqual(rg.ssim(img, img), 1.0, places=6)

    def test_noise_lowers_both(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        noisy = np.clip(img.astype(np.int16) + rng.integers(-20, 21, img.shape), 0, 255).astype(np.uint8)
        self.assertLess(rg.psnr(img, noisy), 40.0)
        self.assertLess(rg.ssim(img, noisy), 0.99)

    def test_percentile_interpolates(self):
        self.assertEqual(rg.percentile([], 99), 0.0)
        self.assertEqual(rg.percentile([5.0], 99), 5.0)
        self.assertAlmostEqual(rg.percentile(list(range(101)), 99), 99.0)
        self.assertAlmostEqual(rg.percentile([0.0, 10.0], 50), 5.0)

    def test_expand_box_clamps_to_frame(self):
        self.assertEqual(rg.expand_box((0, 0, 100, 100), 90, 90), (0, 0, 90, 90))
        self.assertEqual(rg.expand_box((100, 100, 200, 200), 1000, 1000), (85, 85, 215, 215))


class EngineListTests(unittest.TestCase):
    def test_every_shape_map_mask_engines_returns(self):
        # The first live run died on KeyError 'm': one engine comes back as a
        # plain string, and iterating it gave ['m', 'a', 's', 'k', ...].
        self.assertEqual(rg.engine_list("mask_xseg"), ["mask_xseg"])
        self.assertEqual(rg.engine_list(["mask_xseg", "mask_occluder"]),
                         ["mask_xseg", "mask_occluder"])
        self.assertEqual(rg.engine_list(None), [])


class SignatureTests(unittest.TestCase):
    def test_perf_knobs_do_not_change_the_key(self):
        a = SimpleNamespace(swap_model="realswap", max_threads=10)
        b = SimpleNamespace(swap_model="realswap", max_threads=20)
        self.assertEqual(rg.signature_hash(rg.config_signature(a)),
                         rg.signature_hash(rg.config_signature(b)))

    def test_look_settings_change_the_key(self):
        a = SimpleNamespace(swap_model="realswap")
        b = SimpleNamespace(swap_model="hyperswap_1b")
        self.assertNotEqual(rg.signature_hash(rg.config_signature(a)),
                            rg.signature_hash(rg.config_signature(b)))


class VerdictTests(unittest.TestCase):
    T = rg.DEFAULT_THRESHOLDS

    def test_unchanged_passes(self):
        verdict = rg.compare_to_baseline(_report(), _report(), self.T)
        self.assertTrue(verdict["passed"], verdict)

    def test_dropped_frames_fail(self):
        verdict = rg.compare_to_baseline(_report(frames_out=299), _report(), self.T)
        self.assertFalse(verdict["passed"])

    def test_each_quality_floor_fails_on_its_own(self):
        for metric, bad in (("face_ssim_mean", 0.5), ("face_ssim_min", 0.5),
                            ("face_psnr_mean", 20.0), ("frame_psnr_mean", 20.0)):
            with self.subTest(metric=metric):
                verdict = rg.compare_to_baseline(_report(quality={metric: bad}), _report(), self.T)
                self.assertFalse(verdict["passed"])

    def test_unmeasured_quality_is_a_failure_not_a_pass(self):
        verdict = rg.compare_to_baseline(_report(quality={"face_ssim_mean": None}), _report(), self.T)
        self.assertFalse(verdict["passed"])

    def test_swap_coverage_drop_fails(self):
        # The project's recurring defect: a stage that stops swapping while
        # everything else (and the fps) looks fine or better.
        verdict = rg.compare_to_baseline(
            _report(swap={"changed_coverage": 0.5}, perf={"e2e_fps": 20.0}), _report(), self.T)
        self.assertFalse(verdict["passed"])

    def test_fps_drop_only_warns(self):
        verdict = rg.compare_to_baseline(_report(perf={"e2e_fps": 5.0}), _report(), self.T)
        self.assertTrue(verdict["passed"])
        self.assertEqual(len(verdict["warnings"]), 1)


class StageClockTests(unittest.TestCase):
    def test_counts_calls_and_frame_samples(self):
        clock = rg.StageClock()
        for _ in range(3):
            clock("detect", 0.01)
            clock("frame_total", 0.1)
        clock("encode", 0.002)
        table = clock.stage_table()
        self.assertEqual(table["detect"]["calls"], 3)
        self.assertEqual(len(clock.frame_ms), 3)
        self.assertEqual(table["swap"]["calls"], 0)   # present, and visibly zero
        self.assertEqual(len(clock.encode_times), 1)

    def test_prepass_detection_counts_as_detect(self):
        # temporal_detection moves detection into the tracking pre-pass, where
        # _prof names it 'detection' (inside 'track_detect'). The first live
        # run failed "detect never executed" on a render that detected fine.
        clock = rg.StageClock()
        clock("track_detect", 0.02)
        clock("detection", 0.015)
        clock("detect", 0.01)
        row = clock.stage_table()["detect"]
        self.assertEqual(row["calls"], 2)
        self.assertAlmostEqual(row["busy_s"], 0.025)


class WiringTests(unittest.TestCase):
    def test_args_reach_the_suite(self):
        args = SimpleNamespace(benchmark_frames=120, benchmark_clip="c.mp4",
                               benchmark_source="s.png", benchmark_threads=4,
                               benchmark_update_baseline=True)
        with mock.patch.object(rg, "run_regression_cli", return_value=0) as cli:
            self.assertEqual(rg.run_regression_from_args(args), 0)
        cli.assert_called_once_with(frames=120, clip="c.mp4", source="s.png",
                                    threads=4, update_baseline=True)

    def test_both_entry_points_route_regression(self):
        # Source-level: run.py parses sys.argv at import (see
        # test_full_benchmark_e2e.test_both_entry_points_share_one_cli_renderer).
        for relative in ("run.py", "roop/core.py"):
            source = (APP / relative).read_text(encoding="utf-8")
            with self.subTest(file=relative):
                self.assertIn("'regression'", source)
                self.assertIn("run_regression_from_args", source)
                for flag in ("--benchmark-frames", "--benchmark-clip", "--benchmark-source",
                             "--benchmark-threads", "--benchmark-update-baseline"):
                    self.assertIn(flag, source)

    def test_stage_sink_is_called_by_prof(self):
        from roop import procmgr_runtime
        seen = []
        procmgr_runtime.set_stage_sink(lambda stage, dt: seen.append(stage))
        try:
            with procmgr_runtime._prof("detect"):
                pass
        finally:
            procmgr_runtime.set_stage_sink(None)
        self.assertEqual(seen, ["detect"])


if __name__ == "__main__":
    unittest.main()
