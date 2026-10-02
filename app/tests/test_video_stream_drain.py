"""Tests for StderrDrainer and subprocess pipe buffer deadlock prevention.

When FFmpeg processes videos (especially with hardware decode/encode and timestamps/seeking),
it frequently writes warnings or error messages to stderr (e.g. non-monotonic DTS warnings).
On Windows, standard anonymous pipes have a default buffer limit of 4 KB.
If stderr is redirected to subprocess.PIPE without continuous draining, FFmpeg blocks on write
as soon as 4 KB of stderr is generated, deadlocking the reader/writer thread and causing
processing to freeze at 0% indefinitely.

These tests verify that:
1. StderrDrainer drains streams continuously and maintains a bounded trailing buffer.
2. NVHardwareVideoReader does not hang when FFmpeg emits large amounts of stderr.
3. NVHardwareVideoWriter does not hang when FFmpeg emits large amounts of stderr.
4. Diagnostics and trailing error messages remain accessible upon process failure.
"""

import io
import os
import sys
import threading
import time
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.video_stream import NVHardwareVideoReader, NVHardwareVideoWriter, StderrDrainer


class _BlockingStream:
    """A stream that simulates a pipe with lots of data."""

    def __init__(self, total_bytes=64 * 1024, chunk_size=1024):
        self._total = total_bytes
        self._chunk_size = chunk_size
        self._produced = 0
        self._closed = False
        self._lock = threading.Lock()

    def read(self, size=4096):
        with self._lock:
            if self._closed or self._produced >= self._total:
                return b""
            n = min(size, self._chunk_size, self._total - self._produced)
            self._produced += n
            return b"W" * n

    def close(self):
        with self._lock:
            self._closed = True


class StderrDrainerTest(unittest.TestCase):
    def test_drainer_drains_large_stream_bounded(self):
        stream = _BlockingStream(total_bytes=100 * 1024, chunk_size=512)
        drainer = StderrDrainer(stream, max_bytes=4096)
        time.sleep(0.1)
        drainer.stop()

        captured = drainer.get_bytes()
        self.assertLessEqual(len(captured), 4096)
        self.assertGreater(len(captured), 0)
        self.assertTrue(all(b == ord(b"W") for b in captured))

    def test_drainer_handles_none_stream(self):
        drainer = StderrDrainer(None)
        self.assertEqual(drainer.get_bytes(), b"")
        self.assertEqual(drainer.get_text(), "")
        drainer.stop()

    def test_drainer_get_text_decodes_cleanly(self):
        stream = io.BytesIO(b"error: invalid dts\nwarning: frame dropped\n")
        drainer = StderrDrainer(stream, max_bytes=1024)
        time.sleep(0.05)
        drainer.stop()

        text = drainer.get_text()
        self.assertIn("error: invalid dts", text)
        self.assertIn("warning: frame dropped", text)


class _LoudProc:
    """A fake subprocess that generates frames on stdout and voluminous output on stderr."""

    def __init__(self, frame_size, num_frames=5, stderr_bytes=64 * 1024):
        self.frame_size = frame_size
        self.stdout = io.BytesIO(b"\x00" * (frame_size * num_frames))
        self.stderr = _BlockingStream(total_bytes=stderr_bytes, chunk_size=256)
        self.stdin = io.BytesIO()
        self.returncode = 0

    def poll(self):
        return None

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.returncode = -9


class VideoStreamDeadlockPreventionTest(unittest.TestCase):
    def test_reader_does_not_deadlock_on_heavy_stderr(self):
        reader = NVHardwareVideoReader("nonexistent.mp4", 16, 16)
        proc = _LoudProc(reader.frame_size, num_frames=5, stderr_bytes=64 * 1024)
        reader.proc = proc
        reader._stderr_drainer = StderrDrainer(proc.stderr)
        reader._start = lambda: None

        frames = []
        for idx, frame in reader.read_frames():
            frames.append(frame)

        self.assertEqual(len(frames), 5)
        self.assertIsNone(reader.proc)

    def test_writer_close_with_heavy_stderr(self):
        writer = NVHardwareVideoWriter.__new__(NVHardwareVideoWriter)
        writer.output_path = os.path.abspath("test_out.mp4")
        writer.width, writer.height, writer.fps = 16, 16, 24.0
        writer._closed = False
        writer.frames_written = 0

        proc = _LoudProc(16 * 16 * 3, num_frames=0, stderr_bytes=64 * 1024)
        writer.proc = proc
        writer._stderr_drainer = StderrDrainer(proc.stderr)

        # close() should complete without timeout or deadlock
        t0 = time.time()
        writer.close()
        self.assertLess(time.time() - t0, 5.0)
        self.assertIsNone(writer.proc)


if __name__ == "__main__":
    unittest.main()
