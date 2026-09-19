"""Test fractional FPS extraction and OpenCV handle cleanup."""
import os
import sys
import tempfile
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app")
sys.path.insert(0, APP)

import roop.globals as roop_globals
from roop.ffmpeg_path import ffmpeg_binary, ffprobe_binary
from roop import util_ffmpeg
from roop import utilities as util

class TestFractionalFPSAndHandles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        roop_globals.output_path = cls.tmpdir.name

        # 1. Generate 23.976 FPS (24000/1001) NTSC video
        cls.ntsc_24 = os.path.join(cls.tmpdir.name, "ntsc_23_976.mp4")
        subprocess.run([
            ffmpeg_binary(), "-hide_banner", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=24000/1001",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            cls.ntsc_24
        ], check=True)

        # 2. Generate 29.97 FPS (30000/1001) NTSC video
        cls.ntsc_30 = os.path.join(cls.tmpdir.name, "ntsc_29_97.mp4")
        subprocess.run([
            ffmpeg_binary(), "-hide_banner", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=30000/1001",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            cls.ntsc_30
        ], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_fractional_fps_detection(self):
        """1. detect_fps_fractional returns exact r_frame_rate strings."""
        frac24 = util.detect_fps_fractional(self.ntsc_24)
        self.assertEqual(frac24, "24000/1001")

        frac30 = util.detect_fps_fractional(self.ntsc_30)
        self.assertEqual(frac30, "30000/1001")

        fps24 = util.detect_fps(self.ntsc_24)
        self.assertAlmostEqual(fps24, 23.976024, places=5)

    def test_create_video_preserves_fractional_framerate(self):
        """2. create_video passes exact fractional r_frame_rate to FFmpeg."""
        dest = os.path.join(self.tmpdir.name, "compiled_24fps.mp4")
        frames_dir = os.path.join(self.tmpdir.name, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        import cv2, numpy as np
        roop_globals.CFG = type('Cfg', (), {'output_image_format': 'png'})()
        for i in range(1, 10):
            cv2.imwrite(os.path.join(frames_dir, f"{i:06d}.png"), np.zeros((120, 160, 3), dtype=np.uint8))

        util_ffmpeg.create_video(self.ntsc_24, dest, fps=23.976, temp_directory_path=frames_dir)
        self.assertTrue(os.path.isfile(dest))

        # Probe the output r_frame_rate
        cmd = [
            ffprobe_binary(), "-v", "0", "-of", "csv=p=0",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            dest
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        self.assertEqual(out, "24000/1001", "Output framerate should match exact fractional source")

    def test_detect_dimensions_releases_handle(self):
        """3. detect_dimensions does not leak cv2.VideoCapture handle."""
        w, h = util.detect_dimensions(self.ntsc_24)
        self.assertEqual(w, 320)
        self.assertEqual(h, 240)
        # On Windows, if handle was leaked, os.remove or rename would fail
        temp_copy = os.path.join(self.tmpdir.name, "copy.mp4")
        import shutil
        shutil.copyfile(self.ntsc_24, temp_copy)
        w2, h2 = util.detect_dimensions(temp_copy)
        self.assertEqual(w2, 320)
        # Verify file can be deleted immediately (not locked by open cv2.VideoCapture)
        os.remove(temp_copy)
        self.assertFalse(os.path.exists(temp_copy))

if __name__ == "__main__":
    unittest.main()
