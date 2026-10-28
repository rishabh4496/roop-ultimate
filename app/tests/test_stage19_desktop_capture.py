"""Stage 19: Desktop Window & Region Capture and Livecam Subsystem Unit Tests.

Covers:
1. Native Desktop and Window Capture Provider (Win32 GDI / OpenInputDesktop / Synthetic fallback)
2. Interactive Bounding-Box Cropping & Clamping Custody
3. Face Engine DesktopWindowCaptureProvider contract
4. LiveCam API Endpoints (/api/livecam/sources, /api/livecam/windows, /api/livecam/monitors, /api/livecam/start, /api/livecam/status, /api/livecam/stop)
"""
from __future__ import annotations

import os
import sys
import unittest
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
if APP not in sys.path:
    sys.path.insert(0, APP)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from roop.desktop_capture import (
    DesktopCaptureProvider,
    get_desktop_capture_provider,
)
from face_engine.media.capturer import DesktopWindowCaptureProvider


class TestDesktopCaptureProvider(unittest.TestCase):
    """Verifies native OS screen/window capture and synthetic fallback."""

    def setUp(self):
        self.provider = DesktopCaptureProvider()

    def test_enumerate_monitors(self):
        monitors = self.provider.enumerate_monitors()
        self.assertIsInstance(monitors, list)
        self.assertGreaterEqual(len(monitors), 1)
        mon = monitors[0]
        self.assertIn("index", mon)
        self.assertIn("name", mon)
        self.assertIn("width", mon)
        self.assertIn("height", mon)
        self.assertGreater(mon["width"], 0)
        self.assertGreater(mon["height"], 0)

    def test_enumerate_windows(self):
        windows = self.provider.enumerate_windows()
        self.assertIsInstance(windows, list)
        # Should return a list (either live active windows or fallback)
        for w in windows:
            self.assertIn("hwnd", w)
            self.assertIn("title", w)
            self.assertIn("width", w)
            self.assertIn("height", w)

    def test_capture_monitor_full_and_crop(self):
        frame = self.provider.capture_monitor(0)
        self.assertIsNotNone(frame)
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(frame.shape[2], 3)

        # Interactive crop
        crop = {"x": 50, "y": 50, "width": 200, "height": 150}
        cropped = self.provider.capture_monitor(0, crop=crop)
        self.assertIsNotNone(cropped)
        self.assertEqual(cropped.shape[0], 150)
        self.assertEqual(cropped.shape[1], 200)
        self.assertEqual(cropped.shape[2], 3)

    def test_capture_window_and_crop(self):
        # Even with hwnd 0 (desktop root) or arbitrary hwnd, should return a valid frame safely
        frame = self.provider.capture_window(0)
        self.assertIsNotNone(frame)
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape[2], 3)

        crop = {"x": 10, "y": 10, "width": 120, "height": 80}
        cropped = self.provider.capture_window(0, crop=crop)
        self.assertIsNotNone(cropped)
        self.assertEqual(cropped.shape[0], 80)
        self.assertEqual(cropped.shape[1], 120)

    def test_crop_out_of_bounds_clamping(self):
        # Clamps to image boundary rather than raising an index error
        crop = {"x": -50, "y": -50, "width": 99999, "height": 99999}
        clamped = self.provider.capture_monitor(0, crop=crop)
        self.assertIsNotNone(clamped)
        self.assertGreater(clamped.shape[0], 0)
        self.assertGreater(clamped.shape[1], 0)

    def test_synthetic_fallback(self):
        fallback = self.provider._generate_synthetic_frame(
            "Test Synthetic Feed", width=640, height=480, crop={"x": 10, "y": 10, "width": 100, "height": 80}
        )
        self.assertEqual(fallback.shape, (80, 100, 3))


class TestFaceEngineDesktopCapturer(unittest.TestCase):
    """Verifies face_engine integration layer contract."""

    def setUp(self):
        self.capturer = DesktopWindowCaptureProvider()

    def test_capturer_methods(self):
        mons = self.capturer.enumerate_monitors()
        self.assertIsInstance(mons, list)
        self.assertGreaterEqual(len(mons), 1)

        wins = self.capturer.enumerate_windows()
        self.assertIsInstance(wins, list)

        frame = self.capturer.read_frame("monitor", 0, crop={"x": 0, "y": 0, "width": 160, "height": 120})
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (120, 160, 3))


class TestLivecamApiEndpoints(unittest.TestCase):
    """Verifies FastAPI livecam route contract."""

    @classmethod
    def setUpClass(cls):
        from starlette.testclient import TestClient
        import api
        cls.client = TestClient(api.app)

    def test_livecam_sources(self):
        res = self.client.get("/api/livecam/sources")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("webcams", data)
        self.assertIn("monitors", data)
        self.assertIn("windows", data)
        self.assertGreaterEqual(len(data["monitors"]), 1)

    def test_livecam_windows(self):
        res = self.client.get("/api/livecam/windows")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("windows", data)
        self.assertIsInstance(data["windows"], list)

    def test_livecam_monitors(self):
        res = self.client.get("/api/livecam/monitors")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("monitors", data)
        self.assertIsInstance(data["monitors"], list)
        self.assertGreaterEqual(len(data["monitors"]), 1)

    def test_livecam_start_desktop_and_status_and_stop(self):
        # Test desktop monitor start
        payload = {
            "source_type": "monitor",
            "monitor_index": 0,
            "crop": {"x": 0, "y": 0, "width": 320, "height": 240},
            "detector_interval": 6,
        }
        res_start = self.client.post("/api/livecam/start", json=payload)
        self.assertEqual(res_start.status_code, 200)

        # Check status
        res_status = self.client.get("/api/livecam/status")
        self.assertEqual(res_status.status_code, 200)
        st = res_status.json()
        self.assertTrue(st.get("active"))
        self.assertEqual(st.get("source_type"), "monitor")
        self.assertEqual(st.get("crop"), {"x": 0, "y": 0, "width": 320, "height": 240})

        # Stop
        res_stop = self.client.post("/api/livecam/stop")
        self.assertEqual(res_stop.status_code, 200)
        st_after = self.client.get("/api/livecam/status").json()
        self.assertFalse(st_after.get("active"))


if __name__ == "__main__":
    unittest.main()
