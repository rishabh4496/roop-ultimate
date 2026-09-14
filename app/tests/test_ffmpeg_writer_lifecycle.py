"""The encoder must not turn a late ffmpeg exit into a truncated success."""

import os
import sys
import tempfile
import unittest
from threading import RLock

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)


class _Stream:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _Process:
    def __init__(self, returncode, stderr=b""):
        self.returncode = returncode
        self.stdin = _Stream()
        self.stderr = _Stream()
        self._stderr = stderr
        self.communicated = False

    def communicate(self):
        self.communicated = True
        return None, self._stderr

    def wait(self, timeout=None):
        return self.returncode


class FfmpegWriterLifecycleTests(unittest.TestCase):
    def test_close_surfaces_a_late_encoder_exit(self):
        from roop.ffmpeg_writer import FFMPEG_VideoWriter

        writer = object.__new__(FFMPEG_VideoWriter)
        writer.proc = _Process(17, b"Error submitting packet")
        writer.codec = "hevc_nvenc"
        writer.filename = "output.mp4"

        with self.assertRaisesRegex(IOError, "Error submitting packet"):
            writer.close()
        self.assertIsNone(writer.proc)

    def test_close_accepts_a_clean_encoder_exit(self):
        from roop.ffmpeg_writer import FFMPEG_VideoWriter

        writer = object.__new__(FFMPEG_VideoWriter)
        proc = _Process(0)
        writer.proc = proc
        writer.codec = "libx264"
        writer.filename = "output.mp4"

        writer.close()

        self.assertTrue(proc.communicated)
        self.assertIsNone(writer.proc)


class SegmentedWriterLifecycleTests(unittest.TestCase):
    def test_finalize_discards_uncommitted_active_segment(self):
        from roop.segment_writer import SegmentedVideoWriter

        class _FailingWriter:
            def close(self):
                raise IOError("encoder exited")

        with tempfile.TemporaryDirectory() as directory:
            filename = ".failed-segment.mp4"
            path = os.path.join(directory, filename)
            with open(path, "wb") as handle:
                handle.write(b"partial")
            writer = object.__new__(SegmentedVideoWriter)
            writer._writer = _FailingWriter()
            writer._dir = directory
            writer._cur_seg_file = filename
            writer._cur_frames = 2
            writer._write_lock = RLock()

            with self.assertRaisesRegex(IOError, "encoder exited"):
                writer._finalize_segment()
            self.assertFalse(os.path.exists(path))
        self.assertIsNone(writer._writer)
        self.assertIsNone(writer._cur_seg_file)
        self.assertEqual(writer._cur_frames, 0)

    def test_close_does_not_hide_active_segment_failure(self):
        from roop.segment_writer import SegmentedVideoWriter

        writer = object.__new__(SegmentedVideoWriter)
        writer._write_lock = RLock()
        writer._finalize_segment = lambda: (_ for _ in ()).throw(
            IOError("encoder exited after a partial segment"))

        with self.assertRaisesRegex(IOError, "partial segment"):
            writer.close()


if __name__ == "__main__":
    unittest.main()
