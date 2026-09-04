"""End-to-end verification of the full benchmark user journey.

    select workload -> run -> report -> DECLINE (settings preserved)
                                     -> load from history
                                     -> APPLY (roop.globals.CFG updated)

WHAT IS REAL HERE, AND THE ONE THING THAT IS NOT
------------------------------------------------
Every layer is the real one: the real ``WorkloadSelector``, the real
``BenchmarkRunResult``, the real ``storage`` module writing a real
``benchmark_history.json``, the real ``DashboardReport``, the real
``apply_recommended_settings``, the real ``Settings`` object bound to
``roop.globals.CFG``, and -- in the HTTP journey -- the real FastAPI app.

The ONE seam that is stubbed is ``BenchmarkRunner.run``, because executing it
would render video on the GPU. That is deliberate and it is stated rather than
hidden: a suite that renders is a suite nobody runs, and this project's own
rule is that only one render happens at a time. The stub returns a genuine
``BenchmarkRunResult`` built from the same fields the runner fills, so
everything downstream of the render is exercised for real.

Set ``ROOP_E2E_REAL_BENCHMARK=1`` to swap that stub for the real runner and
perform the journey against actual hardware.

SAFETY
------
This test writes settings. It must never touch the user's ``config.yaml`` or
their real ``benchmark_history.json``, so every case binds
``roop.globals.CFG`` to a ``Settings`` instance on a temp path and passes an
explicit ``storage_path``. ``_SandboxedConfig`` asserts the sandbox actually
took, because a test that silently wrote the real config would be worse than
no test at all.
"""
import copy
import os
import sys
import tempfile
import time
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals
import settings as settings_module
from roop.benchmark.runner import BenchmarkRunResult, FrameTelemetry
from roop.benchmark.storage import (
    get_latest_optimal_settings,
    load_benchmark_history,
    update_setting_status,
)
from roop.benchmark.ui_dashboard import (
    BenchmarkSession,
    DashboardReport,
    PreBenchmarkPrompt,
    apply_recommended_settings,
    decline_recommended_settings,
    list_saved_profiles,
    normalize_recommendation,
    resolve_selection,
)

REAL_RUN = os.environ.get("ROOP_E2E_REAL_BENCHMARK", "").strip().lower() in (
    "1", "true", "yes", "on")


# --------------------------------------------------------------------------
# the stubbed seam: a real BenchmarkRunResult without a GPU render
# --------------------------------------------------------------------------

class StubRunner:
    """Produces what BenchmarkRunner.run produces, without rendering.

    The recommendation below uses the runner's OWN key spellings
    (``temp_format``, an ORT-style ``execution_provider``) rather than the
    canonical ones. That is the point: this is the seam between two modules
    written in different sessions, and a fixture that quietly used the
    canonical names would test the dashboard against itself.
    """

    def __init__(self, threads=8, provider="CUDAExecutionProvider",
                 temp_format="png", fps=11.2, frames=150):
        self.threads = threads
        self.provider = provider
        self.temp_format = temp_format
        self.fps = fps
        self.frames = frames
        self.runs = []

    def inspect_active_models(self):
        return {"swapper": "realswap", "enhancer": "GPEN 256 Pro",
                "mask_engine": "RealityUX"}

    def run(self, workload=None, frame_window=0, persist=True,
            storage_path=None, progress_cb=None, **kw):
        total = frame_window or self.frames
        self.runs.append({"workload": workload, "frames": total,
                          "storage_path": storage_path})
        telemetry = []
        for index in range(1, total + 1):
            fps = self.fps + (index % 5) * 0.05
            telemetry.append(FrameTelemetry(
                frame_index=index, duration_ms=1000.0 / fps, fps=fps,
                vram_used_mb=7000.0, cpu_util_pct=35.0, gpu_util_pct=95.0,
                faces_detected=1, faces_swapped=1))
            if progress_cb:
                progress_cb(index, total, fps)
        result = BenchmarkRunResult(
            # A UUID4, because that is what the real runner emits and what
            # storage requires; a non-UUID id is replaced on save.
            run_id=str(uuid.uuid4()),
            timestamp="2026-09-05T00:00:00+00:00",
            device_specs={"gpu": {"available": True, "name": "RTX 4070"},
                          "cpu": {"logical_threads": 32},
                          "gpu_name": "RTX 4070", "vram_total_mb": 12288,
                          "cpu_logical_cores": 32, "ram_total_gb": 32.0},
            active_models=self.inspect_active_models(),
            workload={"mode": str(workload or "solo"), "target_faces": 1,
                      "name": "1 Face (Solo target)"},
            metrics={"avg_fps": self.fps, "p1_low_fps": self.fps * 0.75,
                     "peak_vram_mb": 7000.0, "peak_cpu_pct": 40.0,
                     "avg_latency_ms": 1000.0 / self.fps,
                     "p99_latency_ms": 1000.0 / (self.fps * 0.75),
                     "total_duration_s": total / self.fps,
                     "frames_processed": total,
                     "avg_gpu_pct": 95.0, "avg_cpu_pct": 35.0},
            thermal_stability={"fps_first_30": self.fps, "fps_last_30": self.fps,
                               "retention_pct": 99.0, "fps_delta": 0.0,
                               "frame_time_variance": 0.4,
                               "throttling_detected": False},
            recommended_settings={
                "execution_threads": self.threads,
                "execution_provider": self.provider,
                "temp_format": self.temp_format,
                "provider_options": {"workload_mode": "solo", "target_faces": 1},
            },
            applied=False,
            frame_telemetry=telemetry)
        if persist:
            result.save(storage_path)
        return result


class _SandboxedConfig:
    """Bind roop.globals.CFG to a throwaway Settings for one test."""

    def __enter__(self):
        self._dir = tempfile.TemporaryDirectory(prefix="roop_e2e_")
        self.config_path = os.path.join(self._dir.name, "config.yaml")
        self.history_path = os.path.join(self._dir.name, "benchmark_history.json")
        self._previous = getattr(roop.globals, "CFG", None)
        self.cfg = settings_module.Settings(self.config_path)
        roop.globals.CFG = self.cfg
        # The sandbox is load-bearing: apply() calls save(), so a failure here
        # would rewrite the developer's real configuration.
        assert self.cfg.config_file == self.config_path
        assert not os.path.exists(self.config_path), "sandbox pre-polluted"
        return self

    def __exit__(self, *exc):
        roop.globals.CFG = self._previous
        self._dir.cleanup()
        return False

    def snapshot(self):
        return {name: copy.deepcopy(value)
                for name, value in vars(self.cfg).items()
                if not name.startswith("_") and name != "config_file"}


class FullJourneyTests(unittest.TestCase):
    """The whole journey, in the order a user performs it."""

    def _run_benchmark(self, sandbox, runner, faces="1", mode="quick"):
        session = BenchmarkSession(runner_factory=lambda: runner)
        selection = resolve_selection(faces, mode)
        # The session persists through the runner, which needs the sandboxed
        # history path; the stub honours storage_path exactly as the real
        # runner does.
        original_run = runner.run

        def run_into_sandbox(**kw):
            kw["storage_path"] = sandbox.history_path
            return original_run(**kw)

        runner.run = run_into_sandbox
        started = session.start(faces=faces, mode=mode)
        self.assertEqual(started["status"], "started")
        deadline = time.time() + 20.0
        while time.time() < deadline:
            snap = session.snapshot()
            if not snap.running and (snap.done or snap.error or snap.cancelled):
                break
            time.sleep(0.01)
        else:
            self.fail("benchmark did not finish")
        self.assertEqual(snap.error, "")
        self.assertTrue(snap.done)
        return session, selection, snap

    # -- the journey ------------------------------------------------------
    def test_the_full_user_journey(self):
        with _SandboxedConfig() as sandbox:
            runner = StubRunner(threads=8)
            sandbox.cfg.max_threads = 4        # the "before" the user sees

            # 1. SELECT WORKLOAD -------------------------------------------
            prompt = PreBenchmarkPrompt.build(runner)
            self.assertTrue(prompt.can_run)
            self.assertIn("realswap", prompt.model_summary)
            selection = resolve_selection("1", "quick")
            self.assertEqual(selection["workload_mode"], "solo")

            # 2. RUN --------------------------------------------------------
            session, selection, snap = self._run_benchmark(sandbox, runner)
            self.assertGreater(snap.average_fps, 0.0)
            self.assertEqual(snap.frames_remaining, 0)
            self.assertEqual(snap.frame, selection["frame_window"])

            # 3. REPORT -----------------------------------------------------
            report = session.report()
            self.assertIsNotNone(report)
            self.assertGreater(report.score, 0)
            self.assertAlmostEqual(report.average_fps, 11.2, places=2)
            self.assertGreater(report.p1_low_fps, 0.0)
            self.assertTrue(report.badge)
            threads_row = next(r for r in report.comparison
                               if r.key == "max_threads")
            self.assertEqual(threads_row.current, "4")
            self.assertEqual(threads_row.recommended, "8")
            self.assertTrue(threads_row.changed)

            # 4. DECLINE -- nothing may change ------------------------------
            before = sandbox.snapshot()
            outcome = decline_recommended_settings(
                run_id=report.run_id, storage_path=sandbox.history_path)
            self.assertFalse(outcome["applied"])
            self.assertIn("Optimization Profiles", outcome["message"])
            self.assertEqual(sandbox.snapshot(), before,
                             "Decline must not touch a single live setting")
            self.assertEqual(sandbox.cfg.max_threads, 4)
            # ...but the run is kept, and its status records the decision.
            # Storage models this as three states rather than a boolean, so
            # "declined" is distinguishable from "not decided yet".
            history = load_benchmark_history(sandbox.history_path)
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["status"], "declined")

            # 5. LOAD FROM HISTORY -----------------------------------------
            profiles = list_saved_profiles(storage_path=sandbox.history_path)
            self.assertEqual(len(profiles), 1)
            self.assertEqual(profiles[0]["run_id"], report.run_id)
            self.assertGreater(profiles[0]["score"], 0)
            self.assertFalse(profiles[0]["applied"])
            stored = get_latest_optimal_settings(sandbox.history_path)
            self.assertEqual(stored["execution_threads"], 8)

            # 6. APPLY -- roop.globals.CFG must actually change -------------
            applied = apply_recommended_settings(
                recommended=stored, run_id=report.run_id,
                storage_path=sandbox.history_path)
            self.assertEqual(applied["status"], "applied")
            self.assertEqual(roop.globals.CFG.max_threads, 8,
                             "Apply must reach the live globals, not a copy")
            self.assertIn("max_threads", applied["applied"])
            self.assertTrue(os.path.exists(sandbox.config_path),
                            "the applied settings must be persisted")

            # ...and the history records that it was accepted.
            history = load_benchmark_history(sandbox.history_path)
            self.assertEqual(history[0]["status"], "accepted")
            self.assertTrue(list_saved_profiles(
                storage_path=sandbox.history_path)[0]["applied"])

    def test_decline_then_apply_later_from_history_alone(self):
        """The declined run must remain applicable after the session is gone."""
        with _SandboxedConfig() as sandbox:
            sandbox.cfg.max_threads = 4
            runner = StubRunner(threads=12)
            session, _, _ = self._run_benchmark(sandbox, runner)
            report = session.report()
            decline_recommended_settings(run_id=report.run_id,
                                         storage_path=sandbox.history_path)
            self.assertEqual(sandbox.cfg.max_threads, 4)

            # Everything below uses only what was persisted -- no session.
            profile = list_saved_profiles(storage_path=sandbox.history_path)[0]
            applied = apply_recommended_settings(
                recommended=profile["recommended_settings"],
                run_id=profile["run_id"], storage_path=sandbox.history_path)
            self.assertEqual(applied["status"], "applied")
            self.assertEqual(roop.globals.CFG.max_threads, 12)

    def test_the_engine_key_spellings_survive_the_journey(self):
        """The seam between the runner and the settings layer.

        The runner emits ``temp_format`` and an ORT-style
        ``execution_provider``; the settings layer stores
        ``output_image_format`` and a short provider name. If that translation
        is missing the settings silently never apply -- which looks exactly
        like a successful run that changed nothing.
        """
        with _SandboxedConfig() as sandbox:
            sandbox.cfg.provider = "cpu"
            sandbox.cfg.output_image_format = "jpg"
            runner = StubRunner(threads=8, provider="CUDAExecutionProvider",
                                temp_format="png")
            session, _, _ = self._run_benchmark(sandbox, runner)
            report = session.report()

            fmt_row = next((r for r in report.comparison
                            if r.key == "output_image_format"), None)
            self.assertIsNotNone(
                fmt_row, "the engine's temp_format never reached the table")
            self.assertEqual(fmt_row.recommended, "png")

            applied = apply_recommended_settings(
                recommended=report.recommended_settings, run_id=report.run_id,
                storage_path=sandbox.history_path)
            self.assertEqual(roop.globals.CFG.output_image_format, "png")
            # The provider must land in the form the app actually checks
            # (`if self.provider in ['cuda', 'tensorrt']`), not as an ORT
            # provider class name.
            self.assertEqual(roop.globals.CFG.provider, "cuda")
            self.assertIn("provider", applied["pending"],
                          "provider is read once at process start")

    def test_a_lossy_temp_format_does_not_slip_through_the_journey(self):
        """The engine may recommend it; the journey must still ask."""
        with _SandboxedConfig() as sandbox:
            sandbox.cfg.output_image_format = "png"
            runner = StubRunner(temp_format="jpg")
            session, _, _ = self._run_benchmark(sandbox, runner)
            report = session.report()
            applied = apply_recommended_settings(
                recommended=report.recommended_settings, run_id=report.run_id,
                storage_path=sandbox.history_path)
            self.assertEqual(roop.globals.CFG.output_image_format, "png")
            self.assertIn("output_image_format", applied["skipped"])

    def test_two_runs_accumulate_and_the_latest_wins(self):
        with _SandboxedConfig() as sandbox:
            first, _, _ = self._run_benchmark(sandbox, StubRunner(threads=6))
            second, _, _ = self._run_benchmark(sandbox, StubRunner(threads=14))
            self.assertNotEqual(first.report().run_id, second.report().run_id)
            history = load_benchmark_history(sandbox.history_path)
            self.assertEqual(len(history), 2)
            self.assertEqual(
                get_latest_optimal_settings(sandbox.history_path)["execution_threads"],
                14)
            profiles = list_saved_profiles(storage_path=sandbox.history_path)
            self.assertEqual(profiles[0]["run_id"], second.report().run_id,
                             "history is listed newest first")

    def test_a_failed_run_leaves_settings_and_history_untouched(self):
        class Broken(StubRunner):
            def run(self, **kw):
                raise RuntimeError("no CUDA device")

        with _SandboxedConfig() as sandbox:
            before = sandbox.snapshot()
            session = BenchmarkSession(runner_factory=lambda: Broken())
            session.start("1", "quick")
            deadline = time.time() + 10.0
            while time.time() < deadline and session.snapshot().running:
                time.sleep(0.01)
            snap = session.snapshot()
            self.assertIn("no CUDA device", snap.error)
            self.assertIsNone(session.report())
            self.assertEqual(sandbox.snapshot(), before)
            self.assertEqual(load_benchmark_history(sandbox.history_path), [])


class ActiveModelReadbackTests(unittest.TestCase):
    """The modal must name the models the application actually renders with.

    Found on hardware 2026-09-05: the report read "Swapper=DFL XSeg,
    Enhancer=None" while config.yaml held swap_model: realswap and
    selected_enhancer: UltraMax. `face_swap_mode` is the face SELECTION mode,
    not a model, and it was being read first; the other two globals are never
    populated outside a live render. Every field produced a plausible string,
    so nothing surfaced it.
    """

    def test_the_models_reported_are_the_ones_in_the_live_config(self):
        from roop.benchmark.runner import BenchmarkRunner

        with _SandboxedConfig() as sandbox:
            sandbox.cfg.swap_model = "realswap"
            sandbox.cfg.selected_enhancer = "UltraMax"
            sandbox.cfg.mask_engine = "RealityUX"
            models = BenchmarkRunner().inspect_active_models()
            self.assertEqual(models["swapper"], "realswap")
            self.assertEqual(models["enhancer"], "UltraMax")
            self.assertEqual(models["mask_engine"], "RealityUX")

    def test_the_face_selection_mode_is_never_reported_as_the_swapper(self):
        from roop.benchmark.runner import BenchmarkRunner

        with _SandboxedConfig() as sandbox:
            sandbox.cfg.swap_model = "hyperswap_1a_256"
            roop.globals.face_swap_mode = "DFL XSeg"   # its real-world default
            try:
                models = BenchmarkRunner().inspect_active_models()
            finally:
                roop.globals.face_swap_mode = "selected"
            self.assertEqual(models["swapper"], "hyperswap_1a_256")

    def test_the_configured_swapper_is_the_one_the_run_is_built_with(self):
        """The consequence, not just the label.

        _prepare_process_options builds the real ProcessOptions from these
        values, and its "dfl"/selection-mode arm maps anything unrecognised to
        inswapper. So the wrong readback did not merely mislabel the run -- it
        benchmarked inswapper with no enhancer while the user runs realswap
        with UltraMax, and every recommendation came from that.
        """
        from roop.benchmark.runner import BenchmarkRunner
        from roop.benchmark.asset_manager import WorkloadSelector

        with _SandboxedConfig() as sandbox:
            sandbox.cfg.swap_model = "realswap"
            sandbox.cfg.mask_engine = "RealityUX"
            runner = BenchmarkRunner()
            captured = {}

            import roop.core
            original = roop.core.get_processing_plugins

            def capture(masking_engine=None, swap_model=None, **kw):
                captured["swap_model"] = swap_model
                captured["masking_engine"] = masking_engine
                return original(masking_engine=masking_engine,
                                swap_model=swap_model, **kw)

            roop.core.get_processing_plugins = capture
            try:
                runner._prepare_process_options(
                    runner.inspect_active_models(),
                    WorkloadSelector.get_workload("solo"))
            finally:
                roop.core.get_processing_plugins = original
            self.assertEqual(captured["swap_model"], "realswap",
                             "the benchmark rendered a different swapper than "
                             "the one configured")

    def test_changing_the_configured_models_changes_the_summary(self):
        """A summary that cannot change is a label, not a readback."""
        from roop.benchmark.runner import BenchmarkRunner

        with _SandboxedConfig() as sandbox:
            sandbox.cfg.swap_model = "realswap"
            sandbox.cfg.selected_enhancer = "None"
            first = PreBenchmarkPrompt.build(BenchmarkRunner()).model_summary
            sandbox.cfg.selected_enhancer = "GPEN 256 Pro"
            second = PreBenchmarkPrompt.build(BenchmarkRunner()).model_summary
            self.assertNotEqual(first, second)
            self.assertIn("GPEN 256 Pro", second)


class RunIdConsistencyTests(unittest.TestCase):
    """The id the dashboard reports must be the id history stores.

    Storage enforces a UUID4 and substitutes one for anything else. If the
    result object kept its original id, every later correlation by run_id --
    marking a run applied, re-applying it from the profiles list -- would match
    nothing while reporting success.
    """

    def _result(self, run_id):
        runner = StubRunner()
        result = runner.run(workload="solo", frame_window=2, persist=False)
        result.run_id = run_id
        return result

    def test_a_non_uuid_id_is_REJECTED_rather_than_silently_replaced(self):
        """Storage validates instead of substituting.

        The earlier behaviour minted a fresh UUID for any unrecognised id,
        which meant the in-memory result and the history row could disagree
        and every later correlation by run_id matched nothing while reporting
        success. Refusing the write surfaces the problem instead.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "history.json")
            result = self._result("not-a-uuid")
            with self.assertRaises(ValueError):
                result.save(path)
            self.assertEqual(load_benchmark_history(path), [])

    def test_a_uuid_id_survives_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "history.json")
            original = str(uuid.uuid4())
            result = self._result(original)
            result.save(path)
            self.assertEqual(result.run_id, original)

    def test_a_canonicalised_id_is_adopted_so_later_matches_still_work(self):
        """What save()'s adoption is still worth now that storage validates.

        Storage returns ``str(uuid.UUID(value))``, which lower-cases and
        re-hyphenates. An id differing from the stored one only in case would
        fail an equality match, so the result adopts the canonical form.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "history.json")
            shouty = str(uuid.uuid4()).upper()
            result = self._result(shouty)
            persisted = result.save(path)
            self.assertEqual(result.run_id, persisted)
            self.assertEqual(result.run_id, shouty.lower())
            self.assertTrue(update_setting_status(result.run_id, True, path))
            self.assertEqual(load_benchmark_history(path)[0]["status"],
                             "accepted")


class NormalizationTests(unittest.TestCase):
    """The translation the seam above depends on, unit-level."""

    def test_the_runner_key_spelling_is_accepted(self):
        result = normalize_recommendation({"temp_format": "png"})
        self.assertEqual(result["temp_frame_format"], "png")

    def test_the_canonical_spelling_is_left_alone(self):
        result = normalize_recommendation({"temp_frame_format": "png"})
        self.assertEqual(result["temp_frame_format"], "png")

    def test_ort_provider_names_become_the_short_form_the_app_stores(self):
        for ort, short in (("CUDAExecutionProvider", "cuda"),
                           ("CPUExecutionProvider", "cpu"),
                           ("TensorrtExecutionProvider", "tensorrt")):
            with self.subTest(provider=ort):
                self.assertEqual(
                    normalize_recommendation({"execution_provider": ort})
                    ["execution_provider"], short)

    def test_a_short_provider_name_is_already_correct(self):
        self.assertEqual(
            normalize_recommendation({"execution_provider": "cuda"})
            ["execution_provider"], "cuda")

    def test_an_unknown_provider_is_passed_through_untouched(self):
        """Better a value the app rejects loudly than one silently rewritten."""
        self.assertEqual(
            normalize_recommendation({"execution_provider": "rocm"})
            ["execution_provider"], "rocm")

    def test_normalization_does_not_mutate_its_input(self):
        original = {"temp_format": "png"}
        normalize_recommendation(original)
        self.assertEqual(original, {"temp_format": "png"})


class HttpJourneyTests(unittest.TestCase):
    """The same journey over HTTP, which is the path the React panel takes."""

    def setUp(self):
        try:
            import api
            from fastapi.testclient import TestClient
        except Exception as exc:
            self.skipTest("api.py not importable here: %s" % exc)
        self.api = api
        self.client = TestClient(api.app)

    def test_the_panel_can_walk_the_journey_over_the_api(self):
        import roop.benchmark.ui_dashboard as dashboard

        with _SandboxedConfig() as sandbox:
            sandbox.cfg.max_threads = 4
            runner = StubRunner(threads=8)
            original_run = runner.run

            def run_into_sandbox(**kw):
                kw["storage_path"] = sandbox.history_path
                return original_run(**kw)

            runner.run = run_into_sandbox
            session = BenchmarkSession(runner_factory=lambda: runner)
            previous = dashboard._SESSION
            dashboard._SESSION = session
            try:
                # prompt
                prompt = self.client.get("/api/benchmark/prompt")
                self.assertEqual(prompt.status_code, 200)
                self.assertIn("face_choices", prompt.json())

                # start
                started = self.client.post("/api/benchmark/start",
                                           json={"faces": "1", "mode": "quick"})
                self.assertEqual(started.status_code, 200)

                # poll to completion
                deadline = time.time() + 20.0
                while time.time() < deadline:
                    body = self.client.get("/api/benchmark/progress").json()
                    if not body["running"]:
                        break
                    time.sleep(0.02)
                else:
                    self.fail("the run never finished over HTTP")

                # result
                result = self.client.get("/api/benchmark/result").json()
                self.assertTrue(result["ready"])
                self.assertGreater(result["score"], 0)
                self.assertTrue(result["comparison"])

                # decline: settings preserved
                declined = self.client.post(
                    "/api/benchmark/decline",
                    json={"run_id": result["run_id"]}).json()
                self.assertIn("Optimization Profiles", declined["message"])
                self.assertEqual(roop.globals.CFG.max_threads, 4)

                # apply: globals updated
                applied = self.client.post(
                    "/api/benchmark/apply",
                    json={"run_id": result["run_id"],
                          "recommended_settings": result["recommended_settings"]})
                self.assertEqual(applied.status_code, 200)
                self.assertEqual(roop.globals.CFG.max_threads, 8)
            finally:
                dashboard._SESSION = previous

    def test_starting_a_benchmark_during_a_render_is_refused(self):
        self.api._progress["processing"] = True
        try:
            response = self.client.post("/api/benchmark/start",
                                        json={"faces": "1", "mode": "quick"})
            self.assertEqual(response.status_code, 409)
        finally:
            self.api._progress["processing"] = False


@unittest.skipUnless(REAL_RUN, "set ROOP_E2E_REAL_BENCHMARK=1 to render for real")
class RealHardwareJourneyTests(unittest.TestCase):
    """The same journey against the real runner. Opt-in: it renders video."""

    def test_the_journey_on_real_hardware(self):
        from roop.benchmark.runner import BenchmarkRunner

        with _SandboxedConfig() as sandbox:
            runner = BenchmarkRunner()
            prompt = PreBenchmarkPrompt.build(runner)
            self.assertTrue(prompt.can_run, prompt.warnings)
            result = runner.run(workload="solo", frame_window=30, persist=True,
                                storage_path=sandbox.history_path)
            report = DashboardReport.from_result(result)
            self.assertGreater(report.average_fps, 0.0,
                               "a real run produced no throughput")
            self.assertGreater(report.score, 0)
            # The run really happened, so the recommendation really applies.
            applied = apply_recommended_settings(
                recommended=report.recommended_settings, run_id=report.run_id,
                storage_path=sandbox.history_path)
            self.assertIn(applied["status"], ("applied",))
            self.assertTrue(load_benchmark_history(sandbox.history_path))


if __name__ == "__main__":
    unittest.main()
