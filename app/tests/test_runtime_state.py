"""Contract tests for the backend-owned structured runtime state."""

import json
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import runtime_state  # noqa: E402
from roop.procmgr_runtime import pause_controller  # noqa: E402


class RuntimeStateContractTest(unittest.TestCase):

    def setUp(self):
        runtime_state.reset_resource_cache()
        pause_controller.cancel()

    def test_progress_and_fps_are_structured_from_backend_status(self):
        with patch.object(runtime_state, "_manager", return_value=None), \
             patch.object(runtime_state, "_active_provider", return_value="cuda"), \
             patch.object(runtime_state, "_resource_snapshot", return_value={
                 "gpu": "NVIDIA Test GPU",
                 "vram": {"used_gb": 2.0, "free_gb": 4.0, "total_gb": 6.0},
                 "cpu": {"utilization_pct": 20.0, "logical_threads": 8},
                 "memory": {"process_rss_gb": 1.0, "used_gb": 4.0,
                            "available_gb": 12.0, "total_gb": 16.0,
                            "utilization_pct": 25.0},
             }):
            value = runtime_state.snapshot(
                progress={"processing": True, "paused": False, "progress": 0.5,
                          "desc": "Processing 12 / 24 (3.5 fps)", "error": ""},
                run_stats={"start": 100.0, "frames_done": 12, "frames_total": 24},
                eta_s=4.0, live_seq=7)

        self.assertEqual(value["schema_version"], 1)
        self.assertEqual(value["frame_progress"]["done"], 12)
        self.assertEqual(value["frame_progress"]["total"], 24)
        self.assertEqual(value["fps"], 3.5)
        self.assertEqual(value["eta_s"], 4.0)
        self.assertEqual(value["frame_progress"]["live_seq"], 7)
        self.assertEqual(value["status"]["code"], "PROCESSING")
        json.dumps(value)

    def test_missing_facts_use_contract_sentinels(self):
        with patch.object(runtime_state, "_manager", return_value=None), \
             patch.object(runtime_state, "_active_provider", return_value=runtime_state.UNKNOWN), \
             patch.object(runtime_state, "_resource_snapshot", return_value={
                 "gpu": runtime_state.UNKNOWN,
                 "vram": {"used_gb": runtime_state.UNKNOWN,
                           "free_gb": runtime_state.UNKNOWN,
                           "total_gb": runtime_state.UNKNOWN},
                 "cpu": {"utilization_pct": runtime_state.UNKNOWN,
                         "logical_threads": runtime_state.UNKNOWN},
                 "memory": {"process_rss_gb": runtime_state.UNKNOWN,
                            "used_gb": runtime_state.UNKNOWN,
                            "available_gb": runtime_state.UNKNOWN,
                            "total_gb": runtime_state.UNKNOWN,
                            "utilization_pct": runtime_state.UNKNOWN},
             }):
            value = runtime_state.snapshot()

        self.assertEqual(value["provider"], runtime_state.UNKNOWN)
        self.assertEqual(value["model"], runtime_state.UNKNOWN)
        self.assertEqual(value["precision"], runtime_state.UNKNOWN)
        self.assertEqual(value["gpu"], runtime_state.UNKNOWN)
        self.assertEqual(value["vram"]["total_gb"], runtime_state.UNKNOWN)
        self.assertEqual(value["pool"]["swap"], runtime_state.UNKNOWN)
        self.assertEqual(value["workers"]["active"], runtime_state.UNKNOWN)
        self.assertEqual(value["queue"]["capacity"], runtime_state.UNKNOWN)
        self.assertEqual(value["warnings"], runtime_state.UNKNOWN)
        self.assertEqual(value["errors"], [])

    def test_zero_is_preserved_when_zero_is_verified(self):
        with patch.object(runtime_state, "_manager", return_value=None), \
             patch.object(runtime_state, "_active_provider", return_value="cpu"), \
             patch.object(runtime_state, "_resource_snapshot", return_value={
                 "gpu": runtime_state.NOT_APPLICABLE,
                 "vram": {"used_gb": runtime_state.NOT_APPLICABLE,
                           "free_gb": runtime_state.NOT_APPLICABLE,
                           "total_gb": runtime_state.NOT_APPLICABLE},
                 "cpu": {"utilization_pct": 0.0, "logical_threads": 4},
                 "memory": {"process_rss_gb": 0.5, "used_gb": 2.0,
                            "available_gb": 6.0, "total_gb": 8.0,
                            "utilization_pct": 25.0},
             }):
            value = runtime_state.snapshot(progress={"processing": False})

        self.assertEqual(value["cpu"]["utilization_pct"], 0.0)
        self.assertEqual(value["precision"], runtime_state.NOT_APPLICABLE)

    def test_runtime_endpoint_is_registered(self):
        import api
        routes = [route for route in api.app.routes
                  if getattr(route, "path", None) == "/api/runtime/state"
                  and "GET" in (getattr(route, "methods", None) or set())]
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0].endpoint.__name__, "get_runtime_state")

    def test_pause_state_distinguishes_request_from_acknowledged_pause(self):
        pause_controller.start()
        roop_globals = __import__("roop.globals", fromlist=["processing"])
        roop_globals.processing = True
        self.assertTrue(pause_controller.begin(lambda: roop_globals.processing))
        pause_controller.request()
        with patch.object(runtime_state, "_manager", return_value=None), \
             patch.object(runtime_state, "_active_provider", return_value="cpu"), \
             patch.object(runtime_state, "_resource_snapshot", return_value={
                 "gpu": "UNKNOWN",
                 "vram": {},
                 "cpu": {},
                 "memory": {},
             }):
            value = runtime_state.snapshot(progress={"processing": True,
                                                      "pause_requested": True,
                                                      "paused": False})
        self.assertEqual(value["status"]["code"], "PAUSE_REQUESTED")
        self.assertFalse(value["pause"]["acknowledged"])
        pause_controller.end()
        self.assertTrue(pause_controller.snapshot()["acknowledged"])
        pause_controller.cancel()
        roop_globals.processing = False


if __name__ == "__main__":
    unittest.main()
