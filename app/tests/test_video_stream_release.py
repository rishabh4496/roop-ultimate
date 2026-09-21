"""A pipe released from another thread is END OF STREAM, not a decode failure.

`ProcessMgr._run_stab_parallel`'s cleanup calls `cap.release()` on purpose, to
interrupt a blocking pipe read before joining the reader thread. `release()`
nulls `self.proc`, and `read_frames` used to re-read `self.proc.stdout` on every
iteration, so the reader thread died with
``AttributeError: 'NoneType' object has no attribute 'stdout'`` -- which
`_reader` recorded as a failure and re-raised after cleanup. A user's Stop at
13.9% ended in a traceback (start_react.js log, 2026-09-21).
"""

import io
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.video_stream import NVHardwareVideoReader


class _FakeProc:
    """Enough of a Popen to feed `read_frames`: two frames, then EOF."""

    def __init__(self, frame_size, frames=2):
        self.stdout = io.BytesIO(b"\x01" * (frame_size * frames))
        self.stderr = io.BytesIO(b"")
        self.returncode = 0

    def communicate(self, timeout=None):
        return b"", b""

    def poll(self):
        return None

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 0


class ReleaseDuringReadTest(unittest.TestCase):
    def _reader(self):
        reader = NVHardwareVideoReader("nonexistent.mp4", 4, 2)
        reader.proc = _FakeProc(reader.frame_size)
        reader._start = lambda: None       # the fake is already "started"
        return reader

    def test_release_mid_stream_ends_the_generator_cleanly(self):
        reader = self._reader()
        gen = reader.read_frames()
        idx, frame = next(gen)
        self.assertEqual(idx, 0)
        self.assertEqual(frame.shape, (2, 4, 3))

        reader.release()                    # what cleanup does on cancellation

        with self.assertRaises(StopIteration):
            next(gen)

    def test_read_contract_reports_eof_after_release(self):
        reader = self._reader()
        ok, frame = reader.read()
        self.assertTrue(ok)
        self.assertIsInstance(frame, np.ndarray)

        reader.release()

        ok, frame = reader.read()
        self.assertFalse(ok)
        self.assertIsNone(frame)

    def test_an_unreleased_stream_still_reads_to_its_real_end(self):
        reader = self._reader()
        frames = [f for _, f in reader.read_frames()]
        self.assertEqual(len(frames), 2)


if __name__ == "__main__":
    unittest.main()
