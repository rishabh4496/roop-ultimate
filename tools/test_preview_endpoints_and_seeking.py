"""Test preview endpoints, accurate seeking, and fallback extraction."""
import os
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app")
sys.path.insert(0, APP)

from fastapi.testclient import TestClient
import roop.globals as roop_globals
from roop.ffmpeg_path import ffmpeg_binary
from roop import capturer
from api import app, list_files_process
from roop.ProcessEntry import ProcessEntry

class TestPreviewEndpointsAndSeeking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        roop_globals.output_path = cls.tmpdir.name

        # Create a small test video of 10 frames (160x120, 10 fps)
        cls.test_vid = os.path.join(cls.tmpdir.name, "preview_test.mp4")
        subprocess.run([
            ffmpeg_binary(), "-hide_banner", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=10",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            cls.test_vid
        ], check=True)

        list_files_process.clear()
        entry = ProcessEntry(cls.test_vid, 0, 10, 10.0)
        list_files_process.append(entry)
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        capturer.release_video()
        cls.tmpdir.cleanup()

    def test_accurate_frame_seeking(self):
        """1. Seek to specific frames and verify valid ndarray is returned."""
        fr1 = capturer.get_video_frame(self.test_vid, 1)
        self.assertIsNotNone(fr1)
        self.assertEqual(fr1.shape, (120, 160, 3))

        fr5 = capturer.get_video_frame(self.test_vid, 5)
        self.assertIsNotNone(fr5)
        self.assertEqual(fr5.shape, (120, 160, 3))

    def test_cap_read_none_triggers_fallback(self):
        """2. When cap.read() returns (True, None), verify capturer falls back safely."""
        capturer.release_video()
        orig_read = capturer.cv2.VideoCapture.read

        def faulty_read(self):
            return True, None # simulate corrupt/None frame from cv2

        with patch.object(capturer.cv2.VideoCapture, "read", faulty_read):
            # Target frame 3
            fr = capturer.get_video_frame(self.test_vid, 3)
            # The fallback (ffmpeg pipe or fallback extractor) should produce a valid frame
            self.assertIsNotNone(fr, "Fallback should have produced a valid frame when cv2 returned None")
            self.assertEqual(fr.shape, (120, 160, 3))

    def test_target_preview_endpoint(self):
        """3. GET /api/target/preview returns 200 with image/jpeg."""
        resp = self.client.get("/api/target/preview?index=0&frame=1")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("image/jpeg", resp.headers.get("content-type", ""))
        self.assertGreater(len(resp.content), 0)

    def test_live_frame_endpoint(self):
        """4. GET /api/live_frame returns 204 or 200 without blocking event loop."""
        resp = self.client.get("/api/live_frame")
        self.assertIn(resp.status_code, (200, 204))

if __name__ == "__main__":
    unittest.main()
