"""Contracts every Phase harness must keep, so a phase cannot pass by doing nothing.

Each test here is anchored to a defect that was live in this tree and that every
other check reported clean through:

  * `phase8_expression_bench` exited 1 on a fully successful grading run, and 0
    only when it measured nothing, because the status it tested lived one level
    down in `result["metrics"]`.
  * `phase16_integrity` exited 0 on an empty file set -- `npass == len(rows)` is
    trivially true at 0 == 0 -- so a stale or mistyped --glob reported PASS
    having opened no file at all.
  * `phase12_benchmark` reported a confident `inference_throughput_fps: 0.0` for
    a swap stage that never ran, and divided by zero when the stage's profile
    rounded to 0.000s.
  * `phase13_benchmark` floored the encode denominator at 1e-9, turning a
    zero-length encode stage into ~1e12 fps.
  * `phase14_autotune` hardcoded `startup_seconds` to 0.0 while
    `RuntimeOptimizer.score` charges a real penalty for it, so the term was
    inert and every candidate tied on it.
  * `phase5_quality_matrix` printed "arms without a valid quality result" and
    still returned 0.
  * eight test modules were written in pytest's bare-function style, which
    `unittest discover` collects as zero tests while printing OK.

The last one is why this file is defensive rather than incidental: a regression
in any of these is invisible unless something asserts it.
"""

import ast
import importlib
import io
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _source(name):
    with io.open(os.path.join(HERE, name), encoding="utf-8") as fh:
        return fh.read()


class TestPhase8ExitCode(unittest.TestCase):
    """A complete measurement must exit 0; an unmeasured one must not."""

    def _run(self, grade_result):
        module = importlib.import_module("phase8_expression_bench")
        original_init, original_grade = module._init_detector, module.grade
        original_argv = sys.argv
        try:
            module._init_detector = lambda *a, **k: None
            module.grade = lambda *a, **k: dict(grade_result)
            sys.argv = ["phase8_expression_bench.py",
                        "--target-video", "t.mp4", "--output-video", "o.mp4"]
            return module.main()
        finally:
            module._init_detector, module.grade = original_init, original_grade
            sys.argv = original_argv

    def test_a_complete_grading_run_exits_zero(self):
        self.assertEqual(
            self._run({"status": "complete", "graded_frames": 60, "channels": {}}), 0,
            "a successful grading run must not report failure")

    def test_insufficient_detections_exits_nonzero(self):
        self.assertNotEqual(
            self._run({"status": "insufficient_detections", "graded_frames": 0,
                       "channels": {}}), 0)

    def test_unopenable_video_exits_nonzero(self):
        self.assertNotEqual(
            self._run({"status": "error", "reason": "could_not_open_video"}), 0)

    def test_status_is_promoted_to_the_level_the_exit_code_reads(self):
        # The bug was structural: main() tested result["status"] while grade()
        # only ever wrote result["metrics"]["status"].
        src = _source("phase8_expression_bench.py")
        self.assertIn('"status": metrics.get("status"', src)


class TestPhase16IntegrityEmptySweep(unittest.TestCase):
    def test_an_empty_sweep_does_not_pass(self):
        module = importlib.import_module("phase16_integrity")
        original_argv = sys.argv
        try:
            sys.argv = ["phase16_integrity.py", "--glob",
                        os.path.join(HERE, "__no_such_file_*.mp4")]
            rc = module.main()
        finally:
            sys.argv = original_argv
        self.assertNotEqual(rc, 0, "inspecting zero files is not a pass")

    def test_row_label_keeps_the_filename(self):
        module = importlib.import_module("phase16_integrity")
        a = module._label(os.path.join("out", "arm", "work", "clip.mp4"))
        b = module._label(os.path.join("out", "arm", "work", "clip_23-06-27.mp4"))
        self.assertNotEqual(a, b, "sibling outputs must not print as one row")
        self.assertIn("clip.mp4", a)

    def test_long_paths_are_trimmed_from_the_left(self):
        module = importlib.import_module("phase16_integrity")
        label = module._label("/" + "d" * 200 + "/the_actual_file_name.mp4")
        self.assertLessEqual(len(label), 52)
        self.assertTrue(label.endswith("the_actual_file_name.mp4"), label)


class TestPhase12StageThroughput(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("phase12_benchmark")

    def test_a_stage_that_never_ran_has_no_throughput(self):
        self.assertIsNone(self.module._stage_fps({}, "swap"))
        self.assertIsNone(self.module._stage_fps({"enhance": {}}, "swap"))

    def test_a_zero_duration_stage_does_not_divide_by_zero(self):
        self.assertIsNone(
            self.module._stage_fps({"swap": {"calls": 10, "total_s": 0.0}}, "swap"))

    def test_a_measured_stage_reports_calls_per_second(self):
        self.assertAlmostEqual(
            self.module._stage_fps({"swap": {"calls": 120, "total_s": 4.0}}, "swap"),
            30.0)

    def test_the_table_no_longer_fabricates_a_zero(self):
        rows = self.module._table([
            {"phase12_arm": {"name": "baseline"}, "returncode": 0,
             "run": {"fps": 10.0}, "telemetry": {}, "stages": {}},
        ])
        self.assertIsNone(rows[0]["inference_throughput_fps"])
        self.assertIsNone(rows[0]["enhancement_throughput_fps"])


class TestPhase13EncodeThroughput(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("phase13_benchmark")

    def _rows(self, results):
        return self.module._table(results)

    def test_zero_length_encode_stage_is_unmeasured_not_infinite(self):
        rows = self._rows([
            {"phase13": {"codec": "libx264", "segment_size": "auto"},
             "returncode": 0, "run": {"fps": 8.0}, "telemetry": {},
             "wall_seconds": 75.0,
             "stages": {"encode": {"calls": 600, "total_s": 0.0}}},
        ])
        self.assertIsNone(rows[0]["encode_throughput_fps"])

    def test_measured_encode_stage_reports_frames_per_second(self):
        rows = self._rows([
            {"phase13": {"codec": "libx264", "segment_size": "auto"},
             "returncode": 0, "run": {"fps": 8.0}, "telemetry": {},
             "wall_seconds": 75.0,
             "stages": {"encode": {"calls": 600, "total_s": 1.5}}},
        ])
        self.assertAlmostEqual(rows[0]["encode_throughput_fps"], 400.0)

    def test_a_substituted_baseline_is_named_not_hidden(self):
        rows = self._rows([
            {"phase13": {"codec": "hevc_nvenc", "segment_size": "auto"},
             "returncode": 0, "run": {"fps": 9.0}, "telemetry": {},
             "wall_seconds": 66.0, "stages": {}},
        ])
        self.assertFalse(rows[0]["baseline_is_reference_codec"])
        self.assertEqual(rows[0]["baseline_codec"], "hevc_nvenc")

    def test_the_real_reference_row_is_marked_as_such(self):
        rows = self._rows([
            {"phase13": {"codec": "libx264", "segment_size": "auto"},
             "returncode": 0, "run": {"fps": 8.0}, "telemetry": {},
             "wall_seconds": 75.0, "stages": {}},
            {"phase13": {"codec": "hevc_nvenc", "segment_size": "auto"},
             "returncode": 0, "run": {"fps": 9.2}, "telemetry": {},
             "wall_seconds": 65.0, "stages": {}},
        ])
        self.assertTrue(all(row["baseline_is_reference_codec"] for row in rows))
        self.assertEqual(rows[1]["baseline_codec"], "libx264")
        self.assertAlmostEqual(rows[1]["improvement_pct"], 15.0)


class TestPhase14StartupIsMeasured(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("phase14_autotune")

    def test_startup_is_wall_clock_minus_the_render(self):
        # 600 frames at 10 fps is a 60 s render; a 78 s child spent 18 s starting.
        self.assertAlmostEqual(
            self.module._startup_seconds({"fps": 10.0, "frames": 600}, 78.0),
            18.0, places=6)

    def test_startup_is_never_negative(self):
        self.assertEqual(
            self.module._startup_seconds({"fps": 10.0, "frames": 600}, 55.0), 0.0)

    def test_unreported_throughput_leaves_startup_unstated(self):
        self.assertIsNone(self.module._startup_seconds({"fps": 0, "frames": 600}, 78.0))
        self.assertIsNone(self.module._startup_seconds({}, 78.0))

    def test_the_scoring_input_is_no_longer_a_hardcoded_zero(self):
        src = _source("phase14_autotune.py")
        self.assertNotIn('"startup_seconds": 0.0,', src,
                         "startup_seconds is a live penalty in "
                         "RuntimeOptimizer.score; a constant makes it inert")
        self.assertIn("_startup_seconds(run, elapsed)", src)

    def test_startup_actually_moves_the_optimizer_score(self):
        # Guards the premise: if the penalty were inert, hardcoding 0.0 would
        # not have mattered and this fix would be noise.
        from roop.runtime_optimizer import (HardwareProfile, RuntimeAutotuner,
                                            TuneMeasurement)
        hardware = HardwareProfile(gpu_name="test", vram_total_gb=12.0,
                                   ram_total_gb=32.0)
        fast_start = TuneMeasurement(end_to_end_fps=10.0, startup_seconds=0.0)
        slow_start = TuneMeasurement(end_to_end_fps=10.0, startup_seconds=120.0)
        self.assertGreater(RuntimeAutotuner.score(fast_start, hardware),
                           RuntimeAutotuner.score(slow_start, hardware))


class TestPhase5ReportsIncompleteMatrices(unittest.TestCase):
    def test_an_arm_without_a_verdict_fails_the_run(self):
        src = _source("phase5_quality_matrix.py")
        tail = src[src.index("bad = [k for k, r in rows.items()"):]
        self.assertIn("return 1", tail,
                      "an ERROR/timeout arm must not exit 0")
        self.assertIn("MATRIX INCOMPLETE", src)


class TestEveryTestModuleIsActuallyCollected(unittest.TestCase):
    """`unittest discover` must not report OK over a file it collected nothing from.

    Eight modules here are written in pytest's bare-function style. unittest
    ignores those, so each printed "Ran 0 tests ... OK" -- 30 real assertions,
    including three Phase harness contracts, absent from every green count on
    record. The fix is a `load_tests` hook per module; this guard fails if a new
    module arrives without one.
    """

    def test_no_module_hides_its_tests_from_unittest(self):
        offenders = []
        for name in sorted(os.listdir(HERE)):
            if not (name.startswith("test_") and name.endswith(".py")):
                continue
            src = _source(name)
            try:
                tree = ast.parse(src)
            except SyntaxError:  # pragma: no cover - would fail elsewhere first
                continue
            has_functions = any(
                isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
                for node in tree.body)
            if not has_functions:
                continue
            has_case = "unittest.TestCase" in src
            has_hook = any(isinstance(node, ast.FunctionDef)
                           and node.name == "load_tests" for node in tree.body)
            if not (has_case or has_hook):
                offenders.append(name)
        self.assertEqual(
            offenders, [],
            "these modules define module-level test functions that "
            "`unittest discover` silently collects as ZERO tests; add the "
            "`load_tests` hook from tests/unittest_shim.py: "
            + ", ".join(offenders))

    def test_the_shim_collects_the_functions_it_is_given(self):
        from unittest_shim import module_function_suite
        calls = []

        def test_alpha():
            calls.append("alpha")

        def test_beta(tmp_path):
            calls.append(tmp_path.is_dir())

        def not_a_test():  # pragma: no cover - must be ignored
            calls.append("no")

        namespace = {"__name__": __name__, "test_alpha": test_alpha,
                     "test_beta": test_beta, "not_a_test": not_a_test}
        suite = module_function_suite(namespace)
        self.assertEqual(suite.countTestCases(), 2)
        result = unittest.TestResult()
        suite.run(result)
        self.assertEqual(result.errors + result.failures, [])
        self.assertEqual(calls, ["alpha", True])

    def test_the_shim_ignores_imported_helpers(self):
        from unittest_shim import module_function_suite

        def test_from_elsewhere():  # pragma: no cover
            raise AssertionError("must not run")
        test_from_elsewhere.__module__ = "some.other.module"

        suite = module_function_suite(
            {"__name__": __name__, "test_from_elsewhere": test_from_elsewhere})
        self.assertEqual(suite.countTestCases(), 0)

    def test_an_unsupported_fixture_skips_loudly_rather_than_passing(self):
        from unittest_shim import module_function_suite

        def test_needs_a_fixture(monkeypatch):  # pragma: no cover
            raise AssertionError("must not run")

        suite = module_function_suite(
            {"__name__": __name__, "test_needs_a_fixture": test_needs_a_fixture})
        result = unittest.TestResult()
        suite.run(result)
        self.assertEqual(len(result.skipped), 1)
        self.assertIn("monkeypatch", result.skipped[0][1])


class TestPhaseHarnessesStayPortable(unittest.TestCase):
    """No harness may hardcode a drive letter; PINOKIO_HOME is resolved."""

    DRIVE = re.compile(r"[\"'][A-Za-z]:[\\/](?:pinokio|Users)", re.I)

    def test_no_phase_harness_bakes_in_a_machine_specific_root(self):
        offenders = []
        for name in sorted(os.listdir(HERE)):
            if not re.match(r"(bench_)?phase\d+.*\.py$", name):
                continue
            for lineno, line in enumerate(_source(name).splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue
                if self.DRIVE.search(line):
                    offenders.append("%s:%d %s" % (name, lineno, line.strip()))
        self.assertEqual(offenders, [], "; ".join(offenders))


if __name__ == "__main__":
    unittest.main()
