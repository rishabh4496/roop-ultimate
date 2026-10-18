"""The pipeline status channel and the governor's absolute VRAM floor.

A render's first minutes (TensorRT engine build, canary) have no frame counter, so the React
client's 15 s deadline fired on a healthy backend. These tests pin that those phases are visible
on `/api/v1/pipeline/status` while they run, that the socket pushes changes, and that the governor
throttles, flushes and reports when free VRAM falls under its floor.
"""
import os
import sys
import threading
import time
import unittest
from unittest.mock import patch

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
sys.path.insert(0, os.path.join(APP, "tests"))

from roop import pipeline_status as ps  # noqa: E402


class Registry(unittest.TestCase):
    def setUp(self):
        ps.reset()

    def test_a_stage_is_active_while_it_runs_and_finished_after(self):
        with ps.stage(ps.STAGE_COMPILING_TRT, model="gpen"):
            snap = ps.snapshot()
            self.assertTrue(snap["compiling"])
            self.assertEqual(snap["active"][0]["model"], "gpen")
        snap = ps.snapshot()
        self.assertFalse(snap["compiling"])
        self.assertEqual([e["state"] for e in snap["events"]], ["started", "finished"])

    def test_an_exception_ends_the_stage_as_failed_and_propagates(self):
        with self.assertRaises(ValueError):
            with ps.stage(ps.STAGE_RUNNING_CANARY, model="x"):
                raise ValueError("boom")
        snap = ps.snapshot()
        self.assertFalse(snap["canary"])
        self.assertEqual(snap["events"][-1]["state"], "failed")

    def test_sequence_only_moves_forward_on_change(self):
        a = ps.seq()
        ps.snapshot()
        self.assertEqual(ps.seq(), a, "reading must not look like a change")
        ps.governor_throttle(reason="t")
        self.assertGreater(ps.seq(), a)

    def test_event_ring_is_bounded(self):
        for i in range(500):
            ps.governor_throttle(reason="r", n=i)
        self.assertLessEqual(len(ps.snapshot(events=1000)["events"]), 200)

    def test_segment_progress_counts_completed_slices(self):
        ps.segment_update("g", 1, 3, "COMPLETED")
        ps.segment_update("g", 2, 3, "PROCESSING")
        g = ps.snapshot()["segments"]["g"]
        self.assertEqual((g["count"], g["done"]), (3, 1))
        self.assertEqual(g["segments"]["2"]["state"], "PROCESSING")


class Instrumentation(unittest.TestCase):
    """The real call sites report; a registry nobody writes to is the failure this repo keeps hitting."""

    def setUp(self):
        ps.reset()

    def test_warmup_is_announced_as_compiling_while_the_first_inference_runs(self):
        from roop import predictor
        seen = []

        class Meta:
            name, type, shape = "input", "tensor(float)", [1, 3, 64, 64]

        class Session:
            def get_inputs(self):
                return [Meta()]

            def run(self, _out, _feed):
                seen.append(ps.snapshot()["active"])
                return []

        predictor.reset()
        self.assertTrue(predictor.warmup_session(Session(), tag="unit:model", once=False))
        self.assertEqual(len(seen), 1)
        self.assertEqual([a["kind"] for a in seen[0]], [ps.STAGE_COMPILING_TRT])
        self.assertEqual(seen[0][0]["model"], "unit:model")
        self.assertFalse(ps.snapshot()["compiling"])

    def test_a_failing_warmup_still_closes_the_stage(self):
        from roop import predictor

        class Meta:
            name, type, shape = "input", "tensor(float)", [1, 3, 64, 64]

        class Session:
            def get_inputs(self):
                return [Meta()]

            def run(self, *_):
                raise RuntimeError("engine build failed")

        predictor.reset()
        self.assertFalse(predictor.warmup_session(Session(), tag="unit:bad", once=False))
        self.assertFalse(ps.snapshot()["compiling"])
        self.assertEqual(ps.snapshot()["events"][-1]["state"], "failed")

    def test_the_canary_is_announced_while_it_runs(self):
        from roop import aux_canary
        seen = []

        def impl(stem, session, model_file, reference_factory=None):
            seen.append(ps.snapshot()["canary"])
            return aux_canary.CanaryResult("aux:" + stem, None, "stub")

        with patch.object(aux_canary, "_check_session", impl):
            aux_canary.check_session("restoreformer_plus_plus", object(), "m.onnx")
        self.assertEqual(seen, [True])
        self.assertFalse(ps.snapshot()["canary"])
        self.assertEqual(ps.snapshot()["events"][0]["model"], "restoreformer_plus_plus")


class Rest(unittest.TestCase):
    def setUp(self):
        ps.reset()
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import routes_pipeline_status as r
        app = FastAPI()
        app.include_router(r.router)
        self.client = TestClient(app)

    def test_get_returns_the_snapshot(self):
        with ps.stage(ps.STAGE_COMPILING_TRT, model="m"):
            body = self.client.get("/api/v1/pipeline/status").json()
        self.assertTrue(body["compiling"])
        self.assertEqual(body["active"][0]["kind"], "STAGE_COMPILING_TRT")
        self.assertFalse(self.client.get("/api/v1/pipeline/status").json()["compiling"])

    def test_events_parameter_is_clamped(self):
        for _ in range(5):
            ps.governor_throttle(reason="r")
        self.assertEqual(len(self.client.get("/api/v1/pipeline/status?events=2").json()["events"]), 2)
        self.assertEqual(len(self.client.get("/api/v1/pipeline/status?events=-5").json()["events"]), 0)

    def test_socket_sends_once_on_connect_then_only_on_change(self):
        with self.client.websocket_connect("/ws/v1/pipeline/status") as ws:
            first = ws.receive_json()
            self.assertFalse(first["compiling"])

            def later():
                time.sleep(0.4)
                ps.begin(ps.STAGE_COMPILING_TRT, model="m")
                time.sleep(0.4)
                ps.governor_throttle(reason="vram_floor", vram_free_gb=1.0)

            t = threading.Thread(target=later)
            t.start()
            second = ws.receive_json()
            self.assertTrue(second["compiling"])
            third = ws.receive_json()
            self.assertEqual(third["events"][-1]["kind"], ps.VRAM_GOVERNOR_THROTTLE)
            self.assertEqual(third["governor"]["throttles"], 1)
            t.join()

    def test_the_real_app_serves_both_routes(self):
        """Through api.app itself (its middleware, its router wrapping), not a private FastAPI."""
        import api
        from fastapi.testclient import TestClient
        client = TestClient(api.app)
        body = client.get("/api/v1/pipeline/status")
        self.assertEqual(body.status_code, 200, body.text)
        self.assertIn("seq", body.json())
        with client.websocket_connect("/ws/v1/pipeline/status") as ws:
            self.assertIn("active", ws.receive_json())


class GovernorFloor(unittest.TestCase):
    def setUp(self):
        ps.reset()
        from roop.runtime_optimizer import RuntimeTuning
        from test_runtime_scheduler import _hardware, _workload
        from roop.runtime_scheduler import UnifiedRuntimeScheduler
        self.sched = UnifiedRuntimeScheduler(
            _hardware(6.0), _workload(),
            RuntimeTuning(worker_count=4, queue_depth=3, in_flight_frames=4, ram_buffer_mb=1024))

    def _observe(self, free, total=6.0, times=2):
        sample = {"time": time.time(), "vram_free_gb": free, "vram_total_gb": total,
                  "ram_available_gb": 12.0, "ram_total_gb": 16.0}
        with patch.object(self.sched, "_resource_snapshot", return_value=sample):
            for _ in range(times):
                self.sched.observe({}, force=True)

    def test_default_floor_is_1_2_gb_and_overridable(self):
        from roop.runtime_scheduler import governor_floor_gb
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ROOP_GOVERNOR_MIN_FREE_GB", None)
            self.assertEqual(governor_floor_gb(), 1.2)
        for raw, want in (("0", 0.0), ("2.5", 2.5), ("junk", 1.2), ("-3", 0.0)):
            with patch.dict(os.environ, {"ROOP_GOVERNOR_MIN_FREE_GB": raw}):
                self.assertEqual(governor_floor_gb(), want, raw)

    def test_under_the_floor_throttles_flushes_and_reports(self):
        """1.0 GB free of 6: above the proportional margin (0.72) so only the floor can fire."""
        before = self.sched.effective_inflight
        flushes = []
        with patch.dict(os.environ, {"ROOP_GOVERNOR_MIN_FREE_GB": "1.2"}), \
                patch.object(self.sched, "_flush_vram_cache", lambda: flushes.append(1)):
            self._observe(1.0)
        self.assertEqual(self.sched.effective_inflight, before - 1)
        self.assertTrue(flushes, "the cache was not flushed")
        snap = ps.snapshot()
        event = snap["events"][-1]
        self.assertEqual(event["kind"], ps.VRAM_GOVERNOR_THROTTLE)
        self.assertEqual(event["reason"], "vram_floor")
        self.assertEqual((event["in_flight_before"], event["in_flight_after"]), (before, before - 1))
        self.assertEqual(snap["governor"]["throttles"], 1)

    def test_above_the_floor_nothing_happens(self):
        before = self.sched.effective_inflight
        with patch.dict(os.environ, {"ROOP_GOVERNOR_MIN_FREE_GB": "1.2"}):
            self._observe(3.0, times=4)
        self.assertEqual(self.sched.effective_inflight, before)
        self.assertEqual(ps.snapshot()["governor"]["throttles"], 0)

    def test_floor_off_leaves_only_the_proportional_margin(self):
        before = self.sched.effective_inflight
        with patch.dict(os.environ, {"ROOP_GOVERNOR_MIN_FREE_GB": "0"}):
            self._observe(1.0)
        self.assertEqual(self.sched.effective_inflight, before)

    def test_flush_is_rate_limited_and_survives_a_missing_cuda(self):
        calls = []
        fake = type("T", (), {"cuda": type("C", (), {
            "is_available": staticmethod(lambda: True),
            "empty_cache": staticmethod(lambda: calls.append(1))})})
        with patch.dict(sys.modules, {"torch": fake}):
            self.sched._flush_vram_cache()
            self.sched._flush_vram_cache()
        self.assertEqual(len(calls), 1)


class _FakeSession:
    """Just enough of an onnxruntime session for `warmup_session` and `session_precision`."""

    def __init__(self, trt_options, on_run=None):
        self._opts = trt_options
        self._on_run = on_run

    def get_provider_options(self):
        return {} if self._opts is None else {"TensorrtExecutionProvider": self._opts}

    def get_inputs(self):
        meta = type("M", (), {"name": "x", "shape": [1, 3, 8, 8], "type": "tensor(float)"})
        return [meta]

    def run(self, _outputs, _feed):
        if self._on_run:
            self._on_run()


class StagePrecision(unittest.TestCase):
    """The React banner names FP32 vs FP16, so the payload has to carry the precision the ENGINE builds with."""

    def setUp(self):
        ps.reset()

    def test_precision_is_read_from_the_provider_options_the_session_was_built_with(self):
        sp = ps.session_precision
        self.assertEqual(sp(_FakeSession({"trt_fp16_enable": "True"})), "FP16")
        self.assertEqual(sp(_FakeSession({"trt_fp16_enable": True})), "FP16")
        self.assertEqual(sp(_FakeSession({"trt_fp16_enable": "False"})), "FP32")
        self.assertEqual(sp(_FakeSession({})), "FP32")
        self.assertEqual(sp(_FakeSession({"trt_bf16_enable": "1", "trt_fp16_enable": "False"})), "BF16")

    def test_a_session_off_tensorrt_or_unreadable_says_n_a(self):
        self.assertEqual(ps.session_precision(_FakeSession(None)), "n/a")
        self.assertEqual(ps.session_precision(object()), "n/a")

    def test_warmup_announces_the_model_and_its_precision_while_it_compiles(self):
        from roop import predictor
        seen = []
        session = _FakeSession({"trt_fp16_enable": "False"}, on_run=lambda: seen.extend(ps.snapshot()["active"]))
        tag = "stage5_test_model_%d" % time.time_ns()
        self.assertTrue(predictor.warmup_session(session, tag, once=False))
        self.assertEqual(len(seen), 1, "the stage was active DURING session.run")
        self.assertEqual((seen[0]["kind"], seen[0]["model"], seen[0]["precision"]),
                         (ps.STAGE_COMPILING_TRT, tag, "FP32"))
        done = ps.snapshot()["events"][-1]
        self.assertEqual((done["state"], done["precision"]), ("finished", "FP32"), "and the finish event keeps it")

    def test_the_canary_announces_its_precision_too(self):
        from roop import aux_canary
        seen = []
        session = _FakeSession({"trt_fp16_enable": "True"})
        with patch.object(aux_canary, "_check_session",
                          lambda *a, **k: seen.extend(ps.snapshot()["active"]) or "verdict"):
            self.assertEqual(aux_canary.check_session("xseg", session, "xseg.onnx"), "verdict")
        self.assertEqual((seen[0]["kind"], seen[0]["model"], seen[0]["precision"]),
                         (ps.STAGE_RUNNING_CANARY, "xseg", "FP16"))


if __name__ == "__main__":
    unittest.main()
