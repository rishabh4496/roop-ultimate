"""B-04: saving a different execution provider must say it needs a restart.

`roop.globals.execution_providers` is fixed at startup and every processor keeps the session it built from it, so a
provider saved from the UI changes the config file but not the render. `/api/settings` used to answer a bare
{"status": "success"}.
"""
import os
import sys
import types
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.environ.setdefault("ROOP_SKIP_STARTUP", "1")

import roop.globals as roop_globals  # noqa: E402


class ProviderRestartTests(unittest.TestCase):
    def setUp(self):
        try:
            import api
        except Exception as exc:
            self.skipTest("api module not importable here: %s" % type(exc).__name__)
        self.api = api
        self._providers = list(getattr(roop_globals, "execution_providers", []) or [])
        self._cfg = roop_globals.CFG

    def tearDown(self):
        roop_globals.execution_providers = self._providers
        roop_globals.CFG = self._cfg

    def _running(self, *names):
        roop_globals.execution_providers = list(names)

    def test_a_different_provider_needs_a_restart(self):
        self._running("TensorrtExecutionProvider", "CUDAExecutionProvider")
        self.assertTrue(self.api._provider_restart_needed("cuda"))
        self.assertTrue(self.api._provider_restart_needed("CPU"))

    def test_the_provider_already_running_does_not(self):
        self._running("TensorrtExecutionProvider", "CUDAExecutionProvider")
        self.assertFalse(self.api._provider_restart_needed("tensorrt"))
        self.assertFalse(self.api._provider_restart_needed("TensorRT"))

    def test_auto_and_blank_cannot_be_compared_so_they_are_not_reported(self):
        self._running("CUDAExecutionProvider")
        for value in ("auto", "", None):
            self.assertFalse(self.api._provider_restart_needed(value), repr(value))

    def test_nothing_loaded_yet_means_nothing_to_restart(self):
        self._running()
        self.assertFalse(self.api._provider_restart_needed("cpu"))

    def test_dml_and_directml_are_the_same_provider(self):
        self._running("DmlExecutionProvider")
        self.assertFalse(self.api._provider_restart_needed("dml"))
        self._running("DirectMLExecutionProvider")
        self.assertFalse(self.api._provider_restart_needed("dml"))

    def test_the_endpoint_reports_it(self):
        self._running("CUDAExecutionProvider")
        cfg = types.SimpleNamespace(provider="cuda", save=lambda: None)
        roop_globals.CFG = cfg
        reply = self.api.save_settings({"provider": "cpu"})
        self.assertEqual(reply.get("status"), "success")
        self.assertEqual(reply.get("restart_required"), ["provider"])

    def test_the_endpoint_stays_quiet_when_nothing_changed(self):
        self._running("CUDAExecutionProvider")
        roop_globals.CFG = types.SimpleNamespace(provider="cuda", save=lambda: None)
        self.assertEqual(self.api.save_settings({"provider": "cuda"}), {"status": "success"})


if __name__ == "__main__":
    unittest.main()
