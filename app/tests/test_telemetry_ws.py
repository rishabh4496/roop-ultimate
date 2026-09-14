"""Telemetry transport tests, against the REAL app.

These cover the properties the transport has to hold for a render to be safe:

  * the socket serves, greets with current state, and answers a liveness ping
  * the SAMPLER pushes a change without the client asking (the whole point)
  * a worker THREAD -- which is how the swap pipeline runs, with no event loop
    of its own -- can reach a connected client
  * a wedged or vanished client cannot hurt the run
  * /api/progress is byte-for-byte unaffected, since it stays the source of
    truth and the client's fallback

Run standalone:
    app/env/Scripts/python.exe -m unittest tests.test_telemetry_ws -v
(from the app/ directory), or via the repo pytest config.
"""
import os
import sys
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
import routes_telemetry  # noqa: E402


class TelemetryTransportTests(unittest.TestCase):
    """The socket itself: handshake, greeting, liveness."""

    def setUp(self):
        self.client = TestClient(api.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_status_endpoint_reports_the_hub_is_live(self):
        body = self.client.get("/api/telemetry/status").json()
        self.assertTrue(body["enabled"],
                        "lifespan must bind the event loop; without it no "
                        "worker thread can ever reach a client")
        self.assertGreater(body["sample_hz"], 0)

    def test_connect_is_greeted_with_current_state(self):
        # A client attaching mid-render must render correctly immediately,
        # rather than showing an empty bar until something next changes.
        with self.client.websocket_connect("/ws/telemetry") as ws:
            hello = ws.receive_json()
        self.assertEqual(hello["event"], "hello")
        for key in ("processing", "progress", "current_frame", "total_frames", "fps"):
            self.assertIn(key, hello)

    def test_ping_is_answered(self):
        # Liveness has to be provable through an intermediary that would
        # otherwise reap an idle tunnel.
        with self.client.websocket_connect("/ws/telemetry") as ws:
            ws.receive_json()          # hello
            ws.send_text("ping")
            self.assertEqual(ws.receive_json()["event"], "pong")

    def test_status_counts_connections(self):
        before = self.client.get("/api/telemetry/status").json()["total_connections"]
        with self.client.websocket_connect("/ws/telemetry") as ws:
            ws.receive_json()
        after = self.client.get("/api/telemetry/status").json()["total_connections"]
        self.assertEqual(after, before + 1)


class TelemetryPushTests(unittest.TestCase):
    """The push path: the server must send without being asked."""

    def setUp(self):
        self.client = TestClient(api.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self._saved = dict(api._progress)
        self._saved_stats = dict(api._run_stats)

    def tearDown(self):
        api._progress.clear()
        api._progress.update(self._saved)
        api._run_stats.clear()
        api._run_stats.update(self._saved_stats)

    def test_sampler_pushes_a_change_unprompted(self):
        """The client sends nothing and still learns the run advanced."""
        with self.client.websocket_connect("/ws/telemetry") as ws:
            ws.receive_json()  # hello

            api._progress.update({"processing": True, "progress": 0.25,
                                  "desc": "75 / 300", "error": ""})
            api._run_stats.update({"start": time.time() - 10,
                                   "frames_done": 75, "frames_total": 300})

            # The sampler runs at SAMPLE_HZ; allow a few intervals.
            deadline = time.time() + 5
            frame = None
            while time.time() < deadline:
                candidate = ws.receive_json()
                if candidate.get("current_frame") == 75:
                    frame = candidate
                    break
            self.assertIsNotNone(frame, "sampler never pushed the change")
            self.assertEqual(frame["total_frames"], 300)
            self.assertTrue(frame["processing"])
            self.assertGreater(frame["fps"], 0, "fps must be derived, not zero")

    def test_worker_thread_can_reach_a_client(self):
        """The load-bearing case.

        The swap pipeline is a plain threading.Thread with no event loop, so
        this is the only way it could ever push. If this breaks, telemetry
        silently degrades to polling for every real render.
        """
        with self.client.websocket_connect("/ws/telemetry") as ws:
            ws.receive_json()  # hello

            sent = {}

            def worker():
                sent["ok"] = routes_telemetry.hub.broadcast_threadsafe(
                    {"event": "progress", "current_frame": 4242,
                     "total_frames": 9000, "fps": 31.5})

            t = threading.Thread(target=worker)
            t.start()
            t.join(timeout=10)

            self.assertTrue(sent.get("ok"), "broadcast_threadsafe reported failure")

            deadline = time.time() + 5
            got = None
            while time.time() < deadline:
                candidate = ws.receive_json()
                if candidate.get("current_frame") == 4242:
                    got = candidate
                    break
            self.assertIsNotNone(got, "worker thread's frame never arrived")
            self.assertEqual(got["fps"], 31.5)


class TelemetryResilienceTests(unittest.TestCase):
    """Telemetry must never be able to damage a render."""

    def test_broadcast_with_no_clients_is_a_cheap_noop(self):
        # Nothing attached is the common case (the UI is often closed while a
        # long render runs); it must not raise or block.
        self.assertFalse(routes_telemetry.hub.broadcast_threadsafe({"event": "progress"}))

    def test_broadcast_without_a_loop_fails_soft(self):
        hub = routes_telemetry.TelemetryHub()
        self.assertIsNone(hub.loop)
        # A pipeline thread calling this before the server is up must get False,
        # never an exception that would propagate into the render.
        self.assertFalse(hub.broadcast_threadsafe({"event": "progress"}))

    def test_a_dead_client_is_dropped_not_retried(self):
        import asyncio

        class ExplodingSocket:
            async def send_text(self, _payload):
                raise RuntimeError("client went away")

        hub = routes_telemetry.TelemetryHub()
        dead = ExplodingSocket()
        hub._clients.add(dead)

        sent = asyncio.run(hub.broadcast({"event": "progress"}))
        self.assertEqual(sent, 0)
        self.assertEqual(hub.client_count(), 0,
                         "a client that raises must be evicted, or a half-open "
                         "socket accumulates sends for the whole render")

    def test_a_failing_snapshot_does_not_kill_the_sampler(self):
        import asyncio

        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient")
            return {"processing": False, "n": calls["n"]}

        async def drive():
            loop = asyncio.get_running_loop()
            hub = routes_telemetry.TelemetryHub()
            hub.bind_loop(loop)

            class Sink:
                def __init__(self):
                    self.frames = []

                async def send_text(self, payload):
                    self.frames.append(payload)

            sink = Sink()
            hub._clients.add(sink)
            hub.start_sampler(flaky)
            await asyncio.sleep(routes_telemetry.SAMPLE_INTERVAL * 8)
            await hub.stop_sampler()
            return sink.frames

        frames = asyncio.run(drive())
        self.assertTrue(frames, "sampler died on the first snapshot exception")


class ProgressEndpointUnchangedTests(unittest.TestCase):
    """The fallback must keep working exactly as before."""

    def test_progress_still_serves_its_full_shape(self):
        with TestClient(api.app) as client:
            body = client.get("/api/progress").json()
        # These are the fields the React client reads; telemetry must not have
        # altered or thinned the endpoint it falls back to.
        for key in ("processing", "paused", "progress", "desc", "error",
                    "output", "live_seq", "eta_s", "started_at", "log",
                    "parts", "status_line", "runtime"):
            self.assertIn(key, body, f"/api/progress lost {key!r}")

    def test_snapshot_has_no_side_effects_on_the_log(self):
        # The sampler calls this 4x a second. If it mutated the rolling log the
        # way get_progress() does, it would multiply log churn by 4.
        before = len(api._log_lines)
        for _ in range(20):
            api._telemetry_snapshot()
        self.assertEqual(len(api._log_lines), before,
                         "the telemetry snapshot must not append to the log")


if __name__ == "__main__":
    unittest.main(verbosity=2)
