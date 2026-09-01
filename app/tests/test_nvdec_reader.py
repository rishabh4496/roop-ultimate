"""Regression tests for the safe NVDEC/host-frame boundary.

The application intentionally hands ordinary mutable BGR NumPy arrays to its
existing consumers. These tests make the bounded prefetch reader prove that it
keeps frame order and does not reuse a live array underneath a later frame.
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.nvdec_reader import (  # noqa: E402
    FFmpegVideoReader,
    _auto_pix_fmt,
    _auto_prefetch_depth,
)


def _ensure_ffmpeg_on_path():
    """Resolve ffmpeg the way the application does, not by hoping it is on PATH.

    These two tests spawn a real ffmpeg through `FFmpegVideoReader`. Pinokio
    ships ffmpeg under `PINOKIO_HOME/bin`, NOT on the interactive PATH, so a
    bare `python -m unittest` run raised `FileNotFoundError [WinError 2]` from
    `subprocess` and both tests errored on every machine -- long enough that the
    failures were being carried in the session logs as "pre-existing
    environment errors". They are not environmental: the app finds ffmpeg
    perfectly well at runtime (every render in this repo encodes with
    hevc_nvenc). The tests simply skipped the resolution step the app performs.
    """
    import shutil as _sh
    if _sh.which("ffmpeg"):
        return True
    try:
        from roop.runtime_optimizer import HardwareProfiler
        found = HardwareProfiler._resolve_ffmpeg()
    except Exception:
        found = None
    if not found:
        return False
    os.environ["PATH"] = (os.path.dirname(found) + os.pathsep
                          + os.environ.get("PATH", ""))
    return True


class NvdecReaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _ensure_ffmpeg_on_path():
            raise unittest.SkipTest(
                "ffmpeg could not be resolved from PATH or PINOKIO_HOME/bin")
        cls.tmp = tempfile.mkdtemp(prefix="roop_nvdec_reader_")
        cls.video = os.path.join(cls.tmp, "labelled.mp4")
        writer = cv2.VideoWriter(cls.video, cv2.VideoWriter_fourcc(*"mp4v"),
                                 25, (64, 48))
        if not writer.isOpened():
            shutil.rmtree(cls.tmp, ignore_errors=True)
            raise unittest.SkipTest("OpenCV mp4v writer unavailable")
        for i in range(6):
            writer.write(np.full((48, 64, 3), 20 + i * 20, dtype=np.uint8))
        writer.release()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_auto_nv12_is_limited_to_safe_source_formats(self):
        with mock.patch.dict(os.environ, {"ROOP_NVDEC_NV12": "1"}), \
                mock.patch("roop.nvdec_reader._source_pix_fmt", return_value="yuv420p"):
            self.assertEqual(_auto_pix_fmt(self.video, 64, 48), "nv12")
        with mock.patch("roop.nvdec_reader._source_pix_fmt", return_value="yuv420p10le"):
            self.assertEqual(_auto_pix_fmt(self.video, 64, 48), "bgr24")
        with mock.patch("roop.nvdec_reader._source_pix_fmt", return_value="yuv422p"):
            self.assertEqual(_auto_pix_fmt(self.video, 64, 48), "bgr24")
        self.assertEqual(_auto_pix_fmt(self.video, 64, 48), "bgr24")

    def test_prefetch_depth_is_explicitly_bounded(self):
        old = os.environ.get("ROOP_NVDEC_PREFETCH")
        try:
            os.environ["ROOP_NVDEC_PREFETCH"] = "99"
            self.assertEqual(_auto_prefetch_depth(), 4)
            os.environ["ROOP_NVDEC_PREFETCH"] = "-1"
            self.assertEqual(_auto_prefetch_depth(), 0)
        finally:
            if old is None:
                os.environ.pop("ROOP_NVDEC_PREFETCH", None)
            else:
                os.environ["ROOP_NVDEC_PREFETCH"] = old

    def test_prefetched_bgr_frames_keep_order_and_ownership(self):
        reader = FFmpegVideoReader(self.video, 64, 48, 25, hwaccel=None,
                                   pix_fmt="bgr24", prefetch_depth=2)
        frames = []
        try:
            for _ in range(6):
                ok, frame = reader.read()
                self.assertTrue(ok)
                frames.append(frame)
            ok, frame = reader.read()
            self.assertFalse(ok)
            self.assertIsNone(frame)
            ok, frame = reader.read()
            self.assertFalse(ok)
            self.assertIsNone(frame)
        finally:
            reader.release()
        self.assertEqual(reader.buffer_count, 2)
        for actual, expected in zip((int(np.median(f)) for f in frames),
                                    [20, 40, 60, 80, 100, 120]):
            self.assertAlmostEqual(actual, expected, delta=6)
        self.assertEqual(len({id(f) for f in frames}), len(frames))
        self.assertTrue(all(f.flags.writeable for f in frames))

    def test_nv12_boundary_returns_mutable_bgr(self):
        reader = FFmpegVideoReader(self.video, 64, 48, 25, hwaccel=None,
                                   pix_fmt="nv12", prefetch_depth=1)
        try:
            ok, frame = reader.read()
        finally:
            reader.release()
        self.assertTrue(ok)
        self.assertEqual(frame.shape, (48, 64, 3))
        self.assertTrue(frame.flags.writeable)


if __name__ == "__main__":
    unittest.main()
