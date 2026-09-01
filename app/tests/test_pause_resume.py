"""Stage 8A contract tests for the real processing pause boundary."""

import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals as roop_globals  # noqa: E402
from roop.procmgr_runtime import pause_controller  # noqa: E402


class PauseResumeContractTest(unittest.TestCase):

    def setUp(self):
        pause_controller.cancel()
        pause_controller.start()
        roop_globals.processing = True

    def tearDown(self):
        roop_globals.processing = False
        pause_controller.cancel()

    def _run_and_pause(self, pause_at):
        roop_globals.processing = True
        pause_controller.cancel()
        pause_controller.start()
        processed = []
        written = []
        requested = threading.Event()

        def processing():
            return bool(roop_globals.processing)

        def worker():
            for index in range(8):
                if not pause_controller.begin(processing):
                    break
                try:
                    processed.append(index)
                    time.sleep(0.01)
                    pause_controller.pending_output(1)
                finally:
                    pause_controller.end()
                written.append(index)
                pause_controller.pending_output(-1)
                pause_controller.checkpoint(processing, wait_for_ack=False)
                if pause_controller.snapshot()["acknowledged"]:
                    pause_controller.wait_until_resumed(processing)
            roop_globals.processing = False

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        deadline = time.time() + 2.0
        while len(processed) < pause_at and time.time() < deadline:
            time.sleep(0.005)
        self.assertGreaterEqual(len(processed), pause_at,
                                "worker did not reach the requested point")
        pause_state = pause_controller.request()
        requested.set()
        self.assertTrue(requested.is_set())
        self.assertTrue(pause_state["requested"])
        self.assertFalse(pause_state["acknowledged"])

        deadline = time.time() + 2.0
        while not pause_controller.snapshot()["acknowledged"] and time.time() < deadline:
            time.sleep(0.005)
        self.assertTrue(pause_controller.snapshot()["acknowledged"])
        frozen = (list(processed), list(written))
        time.sleep(0.05)
        self.assertEqual(frozen, (processed, written))

        pause_controller.resume()
        thread.join(2.0)
        self.assertFalse(thread.is_alive(), "resume left a worker blocked")
        self.assertEqual(processed, list(range(8)))
        self.assertEqual(written, list(range(8)))
        state = pause_controller.snapshot()
        self.assertFalse(state["requested"])
        self.assertFalse(state["acknowledged"])
        self.assertEqual(state["active_work"], 0)
        self.assertEqual(state["pending_output"], 0)

    def test_pause_at_early_middle_and_late_processing_points(self):
        for point in (1, 4, 7):
            with self.subTest(point=point):
                self._run_and_pause(point)

    def test_stop_wakes_a_paused_worker(self):
        entered = threading.Event()

        def processing():
            return bool(roop_globals.processing)

        def worker():
            self.assertTrue(pause_controller.begin(processing))
            entered.set()
            pause_controller.end()
            pause_controller.checkpoint(processing)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self.assertTrue(entered.wait(1.0))
        pause_controller.request()
        roop_globals.processing = False
        pause_controller.cancel()
        thread.join(1.0)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
