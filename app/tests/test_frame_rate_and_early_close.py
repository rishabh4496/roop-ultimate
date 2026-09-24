"""Frame-rate selection, the writers' exact ``-r``, and the reader's early close.

1. ``detect_fps`` read ``r_frame_rate`` after 52acd20. On a VFR clip (4 s @30
   + 4 s @15: r=30/1, avg=2700/119, 180 frames over 7.93 s) the render came
   out 6.0 s long and ``restore_audio``'s ``-shortest`` cut 2 s of audio. The
   average rate is the one that preserves duration.
2. The writers passed ``str(float)`` to ``-r``; they now pass an exact
   rational so NTSC stays 24000/1001.
3. Closing ``read_frames()`` before EOF ran ``communicate()``, which drained
   the rest of the decode for up to 10 s and then logged the kill as a decode
   failure.
"""

import io
import os
import subprocess
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.ffmpeg_path import frame_rate_arg
from roop.video_stream import NVHardwareVideoReader, NVHardwareVideoWriter


class FrameRateArgTest(unittest.TestCase):
    def test_rounded_ntsc_rates_snap_back_to_their_exact_fraction(self):
        self.assertEqual(frame_rate_arg(23.976024), "24000/1001")
        self.assertEqual(frame_rate_arg(29.97003), "30000/1001")
        self.assertEqual(frame_rate_arg(59.94006), "60000/1001")

    def test_integer_and_vfr_average_rates_are_exact(self):
        self.assertEqual(frame_rate_arg(25), "25/1")
        self.assertEqual(frame_rate_arg(round(2700 / 119, 6)), "2700/119")

    def test_an_arbitrary_rate_is_not_quantised_coarsely(self):
        num, den = map(int, frame_rate_arg(17.123457).split("/"))
        self.assertLess(abs(num / den - 17.123457), 1e-6)

    def test_unusable_input_passes_through(self):
        self.assertEqual(frame_rate_arg("bogus"), "bogus")
        self.assertEqual(frame_rate_arg(0), "0")

    def test_nvenc_writer_command_carries_the_rational(self):
        writer = NVHardwareVideoWriter.__new__(NVHardwareVideoWriter)
        writer.output_path = os.path.abspath("out.mp4")
        writer.width, writer.height, writer.fps = 64, 32, 23.976024
        writer.audio_source = None
        writer.preset, writer.bitrate, writer.cq = None, None, 19
        writer.threads, writer.ffmpeg_params, writer.colorspace = None, [], "off"
        cmd = writer._command("hevc_nvenc")
        self.assertEqual(cmd[cmd.index("-r") + 1], "24000/1001")


def _ffprobe_stdout(text):
    return subprocess.CompletedProcess([], 0, stdout=text, stderr="")


class DetectFpsTest(unittest.TestCase):
    def setUp(self):
        from roop import utilities
        self.util = utilities

    def _detect(self, stdout):
        with mock.patch.object(self.util.subprocess, "run",
                               return_value=_ffprobe_stdout(stdout)):
            return (self.util.detect_fps("clip.mp4"),
                    self.util.detect_fps_fractional("clip.mp4"))

    def test_vfr_clip_uses_the_average_rate(self):
        fps, frac = self._detect("r_frame_rate=30/1\navg_frame_rate=2700/119\n")
        self.assertEqual(frac, "2700/119")
        self.assertAlmostEqual(fps, 2700 / 119, places=5)

    def test_doubled_nominal_rate_does_not_win(self):
        fps, frac = self._detect("r_frame_rate=48000/1001\navg_frame_rate=24000/1001\n")
        self.assertEqual(frac, "24000/1001")

    def test_cfr_clip_keeps_the_exact_nominal_rate(self):
        # avg_frame_rate from a rounded container duration, within 1e-4.
        fps, frac = self._detect("r_frame_rate=24000/1001\navg_frame_rate=2997/125\n")
        self.assertEqual(frac, "24000/1001")

    def test_missing_average_falls_back_to_nominal(self):
        fps, frac = self._detect("r_frame_rate=25/1\navg_frame_rate=0/0\n")
        self.assertEqual(frac, "25/1")
        self.assertEqual(fps, 25.0)


class _EndlessProc:
    """A decoder that would keep producing frames: communicate() would block."""

    def __init__(self, frame_size):
        self.stdout = io.BytesIO(b"\x02" * (frame_size * 8))
        self.stderr = io.BytesIO(b"")
        self.returncode = None
        self.terminated = False

    def communicate(self, timeout=None):
        raise AssertionError("early close must not drain the decoder")

    def poll(self):
        return None if not self.terminated else -15

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    kill = terminate

    def wait(self, timeout=None):
        return self.returncode


class EarlyCloseTest(unittest.TestCase):
    def test_closing_before_eof_terminates_without_draining(self):
        reader = NVHardwareVideoReader("nonexistent.mp4", 4, 2)
        proc = _EndlessProc(reader.frame_size)
        reader.proc = proc
        reader._start = lambda: None
        gen = reader.read_frames()
        next(gen)
        next(gen)
        with self.assertLogs("roop.video", level="ERROR") as logs:
            gen.close()
            reader_logger_sentinel = __import__("logging").getLogger("roop.video")
            reader_logger_sentinel.error("sentinel")
        self.assertTrue(proc.terminated)
        self.assertIsNone(reader.proc)
        self.assertEqual([r.getMessage() for r in logs.records], ["sentinel"])


if __name__ == "__main__":
    unittest.main()
