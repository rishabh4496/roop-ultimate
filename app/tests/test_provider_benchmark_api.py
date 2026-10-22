"""The provider-matrix benchmark profile ('providers') through the API worker that runs it.

`bench.run_benchmark` is stubbed (this file checks the transport, not the GPU): what must hold is that a
report-only profile is NEVER run with `apply_to_config=True`, its result reaches the status endpoint's state,
and the other profiles keep applying. The matrix itself is tested in test_provider_matrix.py.
"""
import os
import sys
import unittest
from unittest import mock

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.environ.setdefault("ROOP_SKIP_STARTUP", "1")

import api  # noqa: E402
from roop import bench  # noqa: E402


class WorkerApplyFlag(unittest.TestCase):
    def setUp(self):
        with api._benchmark_lock:
            api._benchmark_state.update(running=True, result=None, error="", logs=[], progress=0)

    def tearDown(self):
        with api._benchmark_lock:
            api._benchmark_state.update(running=False, result=None, error="", logs=[], progress=0)

    def _run(self, profile):
        stub = mock.Mock(return_value={"status": "success", "profile": profile})
        with mock.patch.object(bench, "run_benchmark", stub):
            api._run_benchmark_worker(profile, 1.0)
        return stub

    def test_the_providers_profile_is_registered_as_report_only(self):
        self.assertIn("providers", bench.PROFILES)
        self.assertTrue(bench.PROFILES["providers"]["providers_only"])
        for name, cfg in bench.PROFILES.items():
            if name != "providers":
                self.assertFalse(cfg.get("providers_only"), name)

    def test_providers_never_applies_to_the_config(self):
        stub = self._run("providers")
        self.assertIs(stub.call_args.kwargs["apply_to_config"], False)
        self.assertEqual(stub.call_args.kwargs["profile"], "providers")

    def test_the_other_profiles_still_apply(self):
        for name in ("quick", "full"):
            self.assertIs(self._run(name).call_args.kwargs["apply_to_config"], True, name)

    def test_the_result_reaches_the_status_state(self):
        self._run("providers")
        status = api.get_benchmark_status()
        self.assertEqual(status["result"]["profile"], "providers")
        self.assertFalse(status["running"])
        self.assertEqual(status["progress"], 100)


if __name__ == "__main__":
    unittest.main()
