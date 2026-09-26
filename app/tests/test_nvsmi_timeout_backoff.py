"""nvidia-smi HUD probe: a timeout backs off, it does not disable the probe.

Under a full render nvidia-smi can take longer than its 3 s timeout. That used
to count toward the 3-strike permanent disable (the GPU HUD died for the rest
of the session) and, because a failure never set the cache time, every UI poll
spawned another probe that hung for 3 s.
"""
import subprocess
import unittest
from unittest import mock

import routes_diagnostics as rd


class NvsmiTimeoutBackoffTest(unittest.TestCase):
    def setUp(self):
        rd._nvsmi_cache.clear()
        rd._nvsmi_cache.update({"t": 0.0, "data": {}, "fails": 0})

    def _timeout(self, *a, **k):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=3)

    def test_timeouts_never_disable_the_probe(self):
        with mock.patch.object(rd.subprocess, "run", side_effect=self._timeout):
            for _ in range(5):
                rd._nvsmi_cache["retry_at"] = 0.0      # let each call probe
                self.assertEqual(rd._nvidia_smi_stats(), {})
        self.assertEqual(rd._nvsmi_cache["fails"], 0)

    def test_timeout_backs_off_instead_of_respawning(self):
        with mock.patch.object(rd.subprocess, "run", side_effect=self._timeout) as run:
            rd._nvidia_smi_stats()
            rd._nvidia_smi_stats()
            rd._nvidia_smi_stats()
        self.assertEqual(run.call_count, 1)

    def test_timeout_keeps_last_good_reading(self):
        ok = subprocess.CompletedProcess([], 0, stdout="55, 20, 61, 150.5, 2700, 200\n")
        with mock.patch.object(rd.subprocess, "run", return_value=ok):
            first = rd._nvidia_smi_stats()
        self.assertEqual(first["gpu_util"], 55.0)
        rd._nvsmi_cache["t"] = 0.0                      # expire the TTL
        with mock.patch.object(rd.subprocess, "run", side_effect=self._timeout):
            self.assertEqual(rd._nvidia_smi_stats(), first)

    def test_missing_binary_still_disables_after_three(self):
        with mock.patch.object(rd.subprocess, "run", side_effect=FileNotFoundError):
            for _ in range(3):
                rd._nvsmi_cache["retry_at"] = 0.0
                rd._nvidia_smi_stats()
        self.assertEqual(rd._nvsmi_cache["fails"], 3)


if __name__ == "__main__":
    unittest.main()
