"""RateWindow: the current frame rate the HUD's ms/frame is derived from."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routes_telemetry import RateWindow  # noqa: E402


class RateWindowTests(unittest.TestCase):
    def test_needs_two_samples(self):
        w = RateWindow(3.0)
        self.assertIsNone(w.add(0.0, 0))
        self.assertAlmostEqual(w.add(1.0, 10), 10.0)

    def test_reports_the_recent_rate_not_the_run_average(self):
        w = RateWindow(3.0)
        t = 0.0
        for i in range(40):                 # 10 s at 10 fps
            w.add(t, i * 10 // 4 * 4)       # noisy-ish steps
            t += 0.25
        done = 100
        for _ in range(16):                 # then 4 s at 2 fps
            done += 0.5
            rate = w.add(t, int(done))
            t += 0.25
        self.assertLess(rate, 3.0, "the window still carries the fast phase")

    def test_backwards_counter_starts_a_new_window(self):
        w = RateWindow(3.0)
        w.add(0.0, 500)
        w.add(1.0, 510)
        self.assertIsNone(w.add(2.0, 3))    # a new run re-based the counter
        self.assertAlmostEqual(w.add(3.0, 13), 10.0)

    def test_duplicate_timestamps_do_not_divide_by_zero(self):
        w = RateWindow(3.0)
        w.add(1.0, 1)
        self.assertIsNone(w.add(1.0, 5))


if __name__ == "__main__":
    unittest.main()
