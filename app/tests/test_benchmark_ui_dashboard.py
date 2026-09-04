"""Tests for the benchmark's presentation layer and its API surface.

The dashboard is a persuasion surface, so most of these assert what it must
REFUSE to say: an estimate presented as a measurement, a lossy temp-frame
format applied without consent, "applied" for a setting that needs a restart,
or a confident bottleneck badge built from telemetry nobody reported.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.benchmark.ui_dashboard import (
    BENCHMARK_MODE_CHOICES,
    DECLINE_NOTICE,
    FACE_COMPLEXITY_CHOICES,
    REFERENCE_FPS,
    BenchmarkSession,
    DashboardReport,
    PreBenchmarkPrompt,
    apply_recommended_settings,
    build_comparison,
    classify_badge,
    compute_score,
    decline_recommended_settings,
    resolve_selection,
)


class FakeConfig:
    """Stands in for roop.globals.CFG."""

    def __init__(self, **values):
        self.max_threads = 4
        self.output_image_format = "png"
        self.perf_gpu_mem_limit = ""
        self.provider = "cuda"
        self.output_video_codec = "libx264"
        self.__dict__.update(values)
        self.saved = 0

    def save(self):
        self.saved += 1


def result_payload(**overrides):
    payload = {
        "run_id": "run-1", "timestamp": "2026-09-04T00:00:00Z",
        "device_specs": {"gpu_name": "RTX 4070", "vram_total_mb": 12288,
                         "cpu_logical_cores": 32, "ram_total_gb": 32.0},
        "active_models": {"swapper": "realswap", "enhancer": "GPEN 256 Pro",
                          "mask_engine": "RealityUX"},
        "workload": {"mode": "solo", "target_faces": 1, "name": "1 Face"},
        "metrics": {"avg_fps": 11.2, "p1_low_fps": 8.4, "peak_vram_mb": 7000.0,
                    "avg_latency_ms": 89.0, "p99_latency_ms": 120.0,
                    "frames_processed": 150, "peak_cpu_pct": 40.0,
                    "avg_gpu_pct": 95.0, "avg_cpu_pct": 35.0},
        "thermal_stability": {"retention_pct": 99.0, "throttling_detected": False},
        "recommended_settings": {"execution_threads": 8},
        "applied": False,
    }
    payload.update(overrides)
    return payload


class SelectionTests(unittest.TestCase):
    def test_the_three_face_choices_map_to_engine_workloads(self):
        for choice in FACE_COMPLEXITY_CHOICES:
            with self.subTest(choice=choice["value"]):
                resolved = resolve_selection(choice["value"], "quick")
                self.assertEqual(resolved["workload_mode"], choice["mode"])

    def test_full_mode_measures_more_frames_than_quick(self):
        quick = resolve_selection("1", "quick")
        full = resolve_selection("1", "full")
        self.assertGreater(full["frame_window"], quick["frame_window"])
        self.assertGreater(full["estimated_seconds"], quick["estimated_seconds"])

    def test_an_unknown_selection_degrades_instead_of_raising(self):
        """It runs in a worker thread, where an exception is invisible."""
        resolved = resolve_selection("nonsense", "nonsense")
        self.assertEqual(resolved["workload_mode"], "solo")
        self.assertEqual(resolved["mode"], "quick")

    def test_mode_choices_declare_the_seconds_they_promise(self):
        for choice in BENCHMARK_MODE_CHOICES:
            self.assertGreater(choice["seconds"], 0)
            self.assertGreater(choice["frames"], 0)


class PromptTests(unittest.TestCase):
    def test_the_summary_names_all_three_locked_models(self):
        text = PreBenchmarkPrompt.summarize(
            {"swapper": "realswap", "enhancer": "GPEN 256 Pro",
             "mask_engine": "RealityUX"})
        for expected in ("realswap", "GPEN 256 Pro", "RealityUX"):
            self.assertIn(expected, text)

    def test_a_pipeline_that_cannot_be_read_blocks_the_run(self):
        """Benchmarking an unknown configuration measures nothing useful."""
        class Broken:
            def inspect_active_models(self):
                raise RuntimeError("models not loaded")

        prompt = PreBenchmarkPrompt.build(Broken())
        self.assertFalse(prompt.can_run)
        self.assertTrue(prompt.warnings)

    def test_a_readable_pipeline_can_run(self):
        class Fine:
            def inspect_active_models(self):
                return {"swapper": "a", "enhancer": "b", "mask_engine": "c"}

        prompt = PreBenchmarkPrompt.build(Fine())
        self.assertTrue(prompt.can_run)
        self.assertIn("decline_notice", prompt.as_dict())


class ScoreTests(unittest.TestCase):
    def test_the_reference_rate_scores_one_thousand(self):
        score, basis = compute_score(
            {"avg_fps": REFERENCE_FPS, "p1_low_fps": REFERENCE_FPS},
            {"retention_pct": 100.0}, {"mode": "solo"})
        self.assertEqual(score, 1000)
        self.assertIn(str(int(REFERENCE_FPS)), basis)

    def test_the_basis_states_what_the_number_is_relative_to(self):
        _, basis = compute_score({"avg_fps": 10.0}, {}, {"mode": "solo"})
        self.assertIn("Comparable only", basis)

    def test_a_stuttering_run_scores_below_a_smooth_one(self):
        smooth, _ = compute_score({"avg_fps": 12.0, "p1_low_fps": 11.5},
                                  {"retention_pct": 100.0}, {"mode": "solo"})
        stutter, _ = compute_score({"avg_fps": 12.0, "p1_low_fps": 3.0},
                                   {"retention_pct": 100.0}, {"mode": "solo"})
        self.assertLess(stutter, smooth)

    def test_thermal_fade_lowers_the_score(self):
        stable, _ = compute_score({"avg_fps": 12.0, "p1_low_fps": 12.0},
                                  {"retention_pct": 100.0}, {"mode": "solo"})
        fading, _ = compute_score({"avg_fps": 12.0, "p1_low_fps": 12.0},
                                  {"retention_pct": 70.0}, {"mode": "solo"})
        self.assertLess(fading, stable)

    def test_a_cold_first_window_is_not_rewarded(self):
        """Retention above 100% is warm-up, not a bonus."""
        normal, _ = compute_score({"avg_fps": 12.0, "p1_low_fps": 12.0},
                                  {"retention_pct": 100.0}, {"mode": "solo"})
        warmed, _ = compute_score({"avg_fps": 12.0, "p1_low_fps": 12.0},
                                  {"retention_pct": 140.0}, {"mode": "solo"})
        self.assertEqual(warmed, normal)

    def test_a_heavier_workload_is_not_penalised_for_being_heavier(self):
        """Otherwise every machine scores worst on the test that stresses it."""
        solo, _ = compute_score({"avg_fps": 12.0, "p1_low_fps": 12.0},
                                {"retention_pct": 100.0}, {"mode": "solo"})
        group, _ = compute_score({"avg_fps": 4.0, "p1_low_fps": 4.0},
                                 {"retention_pct": 100.0}, {"mode": "group"})
        self.assertEqual(group, solo)

    def test_no_throughput_scores_zero_rather_than_a_number(self):
        score, basis = compute_score({"avg_fps": 0.0}, {}, {})
        self.assertEqual(score, 0)
        self.assertIn("no measured throughput", basis)


class BadgeTests(unittest.TestCase):
    def test_a_healthy_gpu_bound_run_reads_as_good(self):
        badge, tone, _, kind, _ = classify_badge(
            {"avg_fps": 12.0, "avg_gpu_pct": 95.0, "avg_cpu_pct": 30.0,
             "peak_vram_mb": 6000.0},
            {}, {"vram_total_mb": 12288})
        self.assertEqual(kind, "GPU compute bound")
        self.assertEqual(tone, "good")
        self.assertIn("GPU Bound", badge)

    def test_memory_pressure_is_critical_not_good(self):
        _, tone, _, kind, _ = classify_badge(
            {"avg_fps": 2.0, "avg_gpu_pct": 99.0, "avg_cpu_pct": 20.0,
             "peak_vram_mb": 11400.0},
            {}, {"vram_total_mb": 12288})
        self.assertEqual(kind, "GPU VRAM bound")
        self.assertEqual(tone, "critical")

    def test_absent_telemetry_does_not_produce_a_confident_badge(self):
        _, tone, _, kind, _ = classify_badge({"avg_fps": 5.0}, {},
                                             {"vram_total_mb": 12288})
        self.assertEqual(kind, "unknown")
        self.assertEqual(tone, "neutral")


class ComparisonTests(unittest.TestCase):
    def test_current_values_come_from_the_live_config(self):
        config = FakeConfig(max_threads=4)
        rows = build_comparison({"execution_threads": 8}, config, {"avg_fps": 11.2})
        threads = next(r for r in rows if r.key == "max_threads")
        self.assertEqual(threads.current, "4")
        self.assertEqual(threads.recommended, "8")
        self.assertTrue(threads.changed)

    def test_an_unmeasured_projection_is_labelled_not_measured(self):
        rows = build_comparison({"execution_threads": 8}, FakeConfig(),
                                {"avg_fps": 11.2})
        fps = next(r for r in rows if r.key == "expected_fps")
        self.assertEqual(fps.recommended, "not measured")
        self.assertEqual(fps.evidence, "not measured")
        self.assertEqual(fps.delta, "")

    def test_a_supplied_projection_carries_its_delta_and_its_caveat(self):
        rows = build_comparison(
            {"execution_threads": 8, "expected_fps": 17.8}, FakeConfig(),
            {"avg_fps": 11.2})
        fps = next(r for r in rows if r.key == "expected_fps")
        self.assertEqual(fps.recommended, "17.8 FPS")
        self.assertEqual(fps.delta, "+59%")
        self.assertNotEqual(fps.evidence, "measured")
        self.assertIn("not been rendered", fps.note)

    def test_a_lossy_temp_format_is_flagged_as_a_quality_trade(self):
        rows = build_comparison({"temp_frame_format": "jpg"}, FakeConfig(),
                                {"avg_fps": 11.2})
        row = next(r for r in rows if r.key == "output_image_format")
        self.assertIn("encoder's input", row.note)
        self.assertIn("quality", row.evidence)

    def test_restart_only_settings_say_so(self):
        rows = build_comparison({"gpu_memory_limit_mb": 6144}, FakeConfig(),
                                {"avg_fps": 11.2})
        row = next(r for r in rows if r.key == "perf_gpu_mem_limit")
        self.assertTrue(row.requires_restart)
        self.assertIn("next application start", row.note)

    def test_an_unset_current_value_reads_as_Unset(self):
        rows = build_comparison({"gpu_memory_limit_mb": 6144},
                                FakeConfig(perf_gpu_mem_limit=""),
                                {"avg_fps": 11.2})
        row = next(r for r in rows if r.key == "perf_gpu_mem_limit")
        self.assertEqual(row.current, "Unset")


class ReportTests(unittest.TestCase):
    def test_a_full_report_carries_every_dashboard_field(self):
        report = DashboardReport.from_result(result_payload(), config=FakeConfig())
        self.assertGreater(report.score, 0)
        self.assertEqual(report.average_fps, 11.2)
        self.assertEqual(report.p1_low_fps, 8.4)
        self.assertTrue(report.badge)
        self.assertTrue(report.comparison)
        self.assertEqual(set(report.presets),
                         {"max_throughput", "balanced", "stable_low_power"})
        self.assertEqual(report.decline_notice, DECLINE_NOTICE)

    def test_throttling_is_surfaced_as_a_warning(self):
        report = DashboardReport.from_result(
            result_payload(thermal_stability={"retention_pct": 71.0,
                                              "throttling_detected": True}),
            config=FakeConfig())
        self.assertTrue(report.throttling_detected)
        self.assertTrue(any("thermally limited" in w for w in report.warnings))

    def test_the_report_warns_that_the_recommendation_was_not_rendered(self):
        report = DashboardReport.from_result(result_payload(), config=FakeConfig())
        self.assertTrue(any("not itself rendered" in w for w in report.warnings))

    def test_the_cli_rendering_shows_the_same_numbers(self):
        report = DashboardReport.from_result(result_payload(), config=FakeConfig())
        text = report.summary_text()
        self.assertIn(str(report.score), text)
        self.assertIn("11.20", text)
        self.assertIn("Setting", text)

    def test_the_report_serialises_for_the_api(self):
        payload = DashboardReport.from_result(result_payload(),
                                              config=FakeConfig()).as_dict()
        self.assertIsInstance(payload["comparison"], list)
        self.assertIsInstance(payload["comparison"][0], dict)


class ApplyDeclineTests(unittest.TestCase):
    def test_apply_writes_live_settings_and_saves(self):
        config = FakeConfig(max_threads=4)
        result = apply_recommended_settings({"execution_threads": 8}, config=config)
        self.assertEqual(config.max_threads, 8)
        self.assertEqual(config.saved, 1)
        self.assertIn("max_threads", result["applied"])
        self.assertFalse(result["restart_required"])

    def test_restart_only_settings_are_pending_not_applied(self):
        """Reporting 'applied' would have the user measure a change that has
        not happened: run.py reads these once at process start."""
        config = FakeConfig()
        result = apply_recommended_settings({"gpu_memory_limit_mb": 6144},
                                            config=config)
        self.assertIn("perf_gpu_mem_limit", result["pending"])
        self.assertNotIn("perf_gpu_mem_limit", result["applied"])
        self.assertTrue(result["restart_required"])

    def test_a_lossy_temp_format_is_refused_without_explicit_consent(self):
        config = FakeConfig(output_image_format="png")
        result = apply_recommended_settings({"temp_frame_format": "jpg"},
                                            config=config)
        self.assertEqual(config.output_image_format, "png")
        self.assertIn("output_image_format", result["skipped"])
        self.assertIn("re-compress", result["skipped"]["output_image_format"])

    def test_a_lossy_temp_format_is_applied_when_explicitly_accepted(self):
        config = FakeConfig(output_image_format="png")
        apply_recommended_settings({"temp_frame_format": "jpg"}, config=config,
                                   allow_lossy_temp_frames=True)
        self.assertEqual(config.output_image_format, "jpg")

    def test_a_value_already_set_is_skipped_rather_than_reported_applied(self):
        config = FakeConfig(max_threads=8)
        result = apply_recommended_settings({"execution_threads": 8}, config=config)
        self.assertIn("max_threads", result["skipped"])
        self.assertEqual(result["applied"], {})

    def test_apply_without_a_config_changes_nothing_and_says_so(self):
        result = apply_recommended_settings({"execution_threads": 8}, config=None)
        # config=None falls through to the live CFG, which is absent in tests.
        self.assertIn(result["status"], ("error", "applied"))

    def test_decline_returns_the_exact_notice(self):
        result = decline_recommended_settings(run_id="run-1")
        self.assertFalse(result["applied"])
        self.assertEqual(result["notice"], DECLINE_NOTICE)
        self.assertIn("Optimization Profiles", result["message"])

    def test_decline_persists_a_run_it_is_handed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "benchmark_history.json")
            result = decline_recommended_settings(result=result_payload(),
                                                  storage_path=path)
            self.assertEqual(result["status"], "declined")
            self.assertTrue(os.path.exists(path))


class SessionTests(unittest.TestCase):
    class FakeRunner:
        def __init__(self, frames=6):
            self.frames = frames

        def inspect_active_models(self):
            return {"swapper": "a", "enhancer": "b", "mask_engine": "c"}

        def run(self, workload=None, frame_window=0, persist=True,
                progress_cb=None, **kw):
            total = frame_window or self.frames
            for i in range(1, total + 1):
                if progress_cb:
                    progress_cb(i, total, 10.0 + i * 0.1)
            return result_payload()

    def _session(self, runner=None):
        runner = runner or self.FakeRunner()
        return BenchmarkSession(runner_factory=lambda: runner)

    def _wait(self, session, timeout=5.0):
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            snap = session.snapshot()
            if not snap.running and (snap.done or snap.error or snap.cancelled):
                return snap
            time.sleep(0.01)
        self.fail("session did not finish")

    def test_a_run_publishes_progress_and_then_a_report(self):
        session = self._session()
        session.start("1", "quick")
        snap = self._wait(session)
        self.assertTrue(snap.done)
        self.assertEqual(snap.frames_remaining, 0)
        self.assertGreater(snap.average_fps, 0)
        self.assertTrue(snap.fps_series)
        report = session.report()
        self.assertIsNotNone(report)
        self.assertGreater(report.score, 0)

    def test_a_second_run_is_refused_while_one_is_in_flight(self):
        import threading
        gate = threading.Event()

        class Blocking(SessionTests.FakeRunner):
            def run(self, workload=None, frame_window=0, persist=True,
                    progress_cb=None, **kw):
                gate.wait(5.0)
                return result_payload()

        session = self._session(Blocking())
        self.assertEqual(session.start("1", "quick")["status"], "started")
        second = session.start("1", "quick")
        gate.set()
        self.assertEqual(second["status"], "running")
        self._wait(session)

    def test_a_failing_run_reports_an_error_instead_of_hanging(self):
        class Broken(SessionTests.FakeRunner):
            def run(self, **kw):
                raise RuntimeError("no CUDA device")

        session = self._session(Broken())
        session.start("1", "quick")
        snap = self._wait(session)
        self.assertFalse(snap.running)
        self.assertIn("no CUDA device", snap.error)
        self.assertIsNone(session.report())

    def test_cancelling_stops_the_run(self):
        class LongRunner(SessionTests.FakeRunner):
            # Ignores frame_window on purpose: the point is a run long enough
            # that a cancel lands mid-flight.
            def run(self, workload=None, frame_window=0, persist=True,
                    progress_cb=None, **kw):
                for i in range(1, 100001):
                    if progress_cb:
                        progress_cb(i, 100000, 10.0)
                return result_payload()

        session = self._session(LongRunner())
        session.start("1", "quick")
        session.cancel()
        snap = self._wait(session)
        self.assertFalse(snap.running)
        self.assertEqual(snap.phase, "cancelled")
        self.assertIsNone(session.report())

    def test_the_snapshot_series_are_copies(self):
        """The worker appends while FastAPI serialises; aliasing fails mid-response."""
        session = self._session()
        session.start("1", "quick")
        self._wait(session)
        snap = session.snapshot()
        snap.fps_series.append(999.0)
        self.assertNotIn(999.0, session.snapshot().fps_series)


def _walk_routes(routes):
    """Yield every route, descending into included routers.

    This FastAPI build wraps ``include_router`` results in an
    ``_IncludedRouter`` object instead of flattening them into ``app.routes``,
    so a flat scan sees only the ``@app.get`` handlers and silently skips every
    routes_*.py module. Anything asserting a property of "all routes" has to
    recurse or it is asserting it of about half of them.
    """
    for route in routes:
        if getattr(route, "path", None) is not None:
            yield route
        # This build exposes an included router as `_IncludedRouter`, whose
        # own routes hang off `original_router`.
        for attr in ("routes", "router", "original_router", "app"):
            nested = getattr(route, attr, None)
            if nested is None:
                continue
            nested = getattr(nested, "routes", nested)
            if isinstance(nested, (list, tuple)):
                for item in _walk_routes(nested):
                    yield item


class RouteTests(unittest.TestCase):
    """The transport layer must not require query params or invent numbers."""

    def _routes(self):
        try:
            import api
        except Exception as exc:
            self.skipTest("api.py not importable here: %s" % exc)
        return list(_walk_routes(api.app.routes))

    def test_every_benchmark_route_is_registered(self):
        paths = {getattr(r, "path", "") for r in self._routes()}
        for expected in ("/api/benchmark/prompt", "/api/benchmark/start",
                         "/api/benchmark/progress", "/api/benchmark/result",
                         "/api/benchmark/apply", "/api/benchmark/decline",
                         "/api/benchmark/profiles"):
            self.assertIn(expected, paths)

    def test_no_benchmark_GET_requires_a_query_parameter(self):
        """An undecorated helper's positional args become required query params,
        which turns a bare GET into a 422 that the UI reports as a dead server."""
        offenders = []
        for route in self._routes():
            path = getattr(route, "path", "") or ""
            if not path.startswith("/api/benchmark"):
                continue
            if "GET" not in (getattr(route, "methods", None) or set()):
                continue
            dependant = getattr(route, "dependant", None)
            for field in getattr(dependant, "query_params", ()) or ():
                required = getattr(field, "required", None)
                if required is None:
                    required = field.field_info.is_required()
                if required:
                    offenders.append((path, field.name))
        self.assertEqual([], offenders)

    def test_the_benchmark_endpoints_answer_a_bare_request(self):
        try:
            import api
            from fastapi.testclient import TestClient
        except Exception as exc:
            self.skipTest("api.py not importable here: %s" % exc)
        client = TestClient(api.app)
        for path in ("/api/benchmark/progress", "/api/benchmark/result",
                     "/api/benchmark/profiles"):
            with self.subTest(path=path):
                self.assertEqual(client.get(path).status_code, 200)


if __name__ == "__main__":
    unittest.main()
