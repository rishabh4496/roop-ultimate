"""The NVDEC decode pipe has a large kernel buffer, and still behaves like `subprocess.PIPE`.

A default Windows pipe buffer let FFmpeg get only a few KB ahead of the reader, so each
frame took dozens of raw reads, each one re-taking the GIL against Stage 3's ten
HIGHEST-priority frame workers; the workers then starved for input. See
`roop/video_stream.py:_decode_pipe_bytes` for the measurements.

These tests drive the real `_start` with a child process standing in for FFmpeg (it writes
numbered frames to stdout), so they exercise the actual pipe, not a fake.
"""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import video_stream
from roop.video_stream import NVHardwareVideoReader

W, H = 64, 36
FRAMES = 12


def _writer_command(frames=FRAMES, w=W, h=H):
    # Frame k is filled with byte k, so order and completeness are checkable.
    code = ("import sys\n"
            f"n, size = {frames}, {w * h * 3}\n"
            "out = sys.stdout.buffer\n"
            "for k in range(n):\n"
            "    out.write(bytes([k % 256]) * size)\n"
            "out.flush()\n")
    return [sys.executable, "-I", "-c", code]


class _Reader(NVHardwareVideoReader):
    def __init__(self, frames=FRAMES):
        super().__init__("unused.mp4", W, H)
        self._frames_to_write = frames

    def _command(self):
        return _writer_command(self._frames_to_write)


def _pipe_buffer_sizes(file_obj):
    """(out_buffer, in_buffer) of the pipe behind `file_obj` (Windows)."""
    import ctypes
    import ctypes.wintypes as W_
    import msvcrt
    handle = msvcrt.get_osfhandle(file_obj.fileno())
    flags, out_size, in_size, max_inst = W_.DWORD(), W_.DWORD(), W_.DWORD(), W_.DWORD()
    ok = ctypes.windll.kernel32.GetNamedPipeInfo(
        W_.HANDLE(handle), ctypes.byref(flags), ctypes.byref(out_size),
        ctypes.byref(in_size), ctypes.byref(max_inst))
    if not ok:
        raise OSError(ctypes.get_last_error(), "GetNamedPipeInfo failed")
    return out_size.value, in_size.value


class DecodePipeSizeTest(unittest.TestCase):
    def test_three_frames_within_the_bounds(self):
        self.assertEqual(video_stream._decode_pipe_bytes(1272 * 720 * 3), 8 * 1024 * 1024)
        self.assertEqual(video_stream._decode_pipe_bytes(1920 * 1080 * 3), 3 * 1920 * 1080 * 3)
        self.assertEqual(video_stream._decode_pipe_bytes(3840 * 2160 * 3), 32 * 1024 * 1024)
        self.assertEqual(video_stream._decode_pipe_bytes(16), 8 * 1024 * 1024)


class RealPipeTest(unittest.TestCase):
    def test_every_frame_arrives_in_order_and_the_stream_ends(self):
        reader = _Reader()
        got = [int(frame[0, 0, 0]) for _, frame in reader.read_frames()]
        # EOF was reached at all: had this process kept a copy of the write end, the
        # read after the last frame would block forever instead of returning 0 bytes.
        self.assertEqual(got, list(range(FRAMES)))

    def test_frames_are_complete(self):
        reader = _Reader(frames=3)
        frames = [frame for _, frame in reader.read_frames()]
        self.assertEqual(len(frames), 3)
        for k, frame in enumerate(frames):
            self.assertEqual(frame.shape, (H, W, 3))
            self.assertTrue(np.all(frame == k))

    @unittest.skipUnless(os.name == "nt", "the large decode pipe is Windows-only")
    def test_the_pipe_has_the_large_kernel_buffer(self):
        reader = _Reader()
        reader._start()
        try:
            out_size, in_size = _pipe_buffer_sizes(reader.proc.stdout)
            want = video_stream._decode_pipe_bytes(reader.frame_size)
            # The read end of an anonymous pipe reports the quota it was created with.
            self.assertGreaterEqual(max(out_size, in_size), want)
        finally:
            reader.release()

    def test_release_mid_stream_ends_cleanly(self):
        reader = _Reader(frames=200)
        gen = reader.read_frames()
        idx, frame = next(gen)
        self.assertEqual(idx, 0)
        reader.release()
        with self.assertRaises(StopIteration):
            next(gen)

    def test_without_the_large_pipe_it_falls_back_to_subprocess_pipe(self):
        original = video_stream._open_large_pipe
        video_stream._open_large_pipe = lambda size: None
        try:
            reader = _Reader(frames=4)
            got = [int(frame[0, 0, 0]) for _, frame in reader.read_frames()]
        finally:
            video_stream._open_large_pipe = original
        self.assertEqual(got, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
