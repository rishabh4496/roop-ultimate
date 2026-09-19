"""Test FFmpeg odd dimension padding, silent video protection, and error reporting."""
import os
import sys
import tempfile
import json
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app")
sys.path.insert(0, APP)

import roop.globals as roop_globals
from roop.ffmpeg_path import ffmpeg_binary, ffprobe_binary
from roop import util_ffmpeg

class TestFFmpegRobustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        roop_globals.output_path = cls.tmpdir.name
        roop_globals.log_level = "error"
        roop_globals.video_quality = 18

        # 1. Generate an odd dimension video (321x241) with audio
        cls.odd_video = os.path.join(cls.tmpdir.name, "odd_321x241.mp4")
        cmd_odd = [
            ffmpeg_binary(), "-hide_banner", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=321x241:rate=15",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-c:v", "libx264", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            cls.odd_video,
        ]
        subprocess.run(cmd_odd, check=True)

        # 2. Generate a silent video (0 audio streams) with odd dimensions (161x121)
        cls.silent_odd_video = os.path.join(cls.tmpdir.name, "silent_161x121.mp4")
        cmd_silent = [
            ffmpeg_binary(), "-hide_banner", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=161x121:rate=15",
            "-c:v", "libx264", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt", "yuv420p", "-an",
            cls.silent_odd_video,
        ]
        subprocess.run(cmd_silent, check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def _probe(self, path):
        cmd = [
            ffprobe_binary(), "-v", "error",
            "-show_entries", "stream=width,height,codec_name,pix_fmt,codec_type",
            "-of", "json", path,
        ]
        out = subprocess.check_output(cmd, text=True)
        data = json.loads(out)
        streams = data.get("streams", [])
        v = [s for s in streams if s.get("codec_type") == "video"][0]
        a = [s for s in streams if s.get("codec_type") == "audio"]
        return v, a

    def test_finalize_web_video_with_odd_dimensions(self):
        """1. Odd dimension raw video (321x241) must be padded to even without crashing."""
        dest = os.path.join(self.tmpdir.name, "out_odd.mp4")
        ok = util_ffmpeg.finalize_web_video(self.odd_video, dest, audio_source=self.odd_video)
        self.assertTrue(ok, "finalize_web_video failed on odd dimensions")
        v, a = self._probe(dest)
        self.assertEqual(v["codec_name"], "h264")
        self.assertEqual(v["pix_fmt"], "yuv420p")
        # Assert width and height are divisible by 2 (padded)
        self.assertEqual(v["width"] % 2, 0, f"Width {v['width']} is not divisible by 2")
        self.assertEqual(v["height"] % 2, 0, f"Height {v['height']} is not divisible by 2")
        self.assertEqual(len(a), 1, "Audio should be present")
        self.assertEqual(a[0]["codec_name"], "aac")

    def test_finalize_web_video_with_silent_video(self):
        """2. Silent video (no audio stream) must not crash or abort the process."""
        dest = os.path.join(self.tmpdir.name, "out_silent.mp4")
        ok = util_ffmpeg.finalize_web_video(self.silent_odd_video, dest, audio_source=self.silent_odd_video)
        self.assertTrue(ok, "finalize_web_video failed on silent video")
        v, a = self._probe(dest)
        self.assertEqual(v["codec_name"], "h264")
        self.assertEqual(v["pix_fmt"], "yuv420p")
        self.assertEqual(v["width"] % 2, 0)
        self.assertEqual(v["height"] % 2, 0)
        self.assertEqual(len(a), 0, "Muted video must have 0 audio streams cleanly")

    def test_cut_video_with_silent_video(self):
        """3. cut_video on silent input must not fail with -map 0:a?."""
        dest = os.path.join(self.tmpdir.name, "cut_silent.mp4")
        util_ffmpeg.cut_video(self.silent_odd_video, dest, 0, 10, reencode=True)
        self.assertTrue(os.path.isfile(dest), "cut_video failed on silent video")
        v, a = self._probe(dest)
        self.assertEqual(v["codec_name"], "h264")
        self.assertEqual(len(a), 0)

    def test_resize_video_with_odd_dimensions_and_silent(self):
        """4. resize_video with odd target dims on silent input."""
        dest = os.path.join(self.tmpdir.name, "resized_odd.mp4")
        ok = util_ffmpeg.resize_video(self.silent_odd_video, dest, 321, 241)
        self.assertTrue(ok, "resize_video failed")
        v, a = self._probe(dest)
        self.assertEqual(v["codec_name"], "h264")
        self.assertEqual(v["width"] % 2, 0)
        self.assertEqual(v["height"] % 2, 0)
        self.assertEqual(len(a), 0)

if __name__ == "__main__":
    unittest.main()
