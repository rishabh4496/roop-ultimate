"""A merge the output drive cannot hold must fail loudly and keep the parts.

2026-09-27: G: filled during the final stream-copy concat. `_concat()` returned
False, `close()` returned normally, the render logged "Done", and a truncated
1.23 GB temp video (of 3.15 GB of parts) was left for downstream. The only
error the user saw was an ENOSPC traceback from the project checkpoint.
"""
import errno
import json
import os
import shutil
import tempfile
import unittest
from collections import namedtuple
from unittest import mock

import roop.globals
from roop import segment_writer
from roop.segment_writer import SegmentedVideoWriter, manifest_path

_Usage = namedtuple("usage", "total used free")


def _writer(directory, sizes):
    """A writer holding finalized parts, built without opening any encoder."""
    w = SegmentedVideoWriter.__new__(SegmentedVideoWriter)
    w.target_video = os.path.join(directory, "clip__temp.mp4")
    w._dir = directory
    w._seg_prefix = ".clip__temp.seg"
    w._seg_ext = ".mp4"
    w._writer = None
    w._cur_seg_file = None
    w._cur_frames = 0
    w._checkpoint_callback = None
    import threading
    w._write_lock = threading.RLock()
    w.segments = []
    for i, size in enumerate(sizes):
        name = f".clip__temp.seg{i:04d}.mp4"
        with open(os.path.join(directory, name), "wb") as fh:
            fh.write(b"\0" * size)
        w.segments.append({"file": name, "frames": 10, "bytes": size})
    with open(manifest_path(w.target_video), "w", encoding="utf-8") as fh:
        json.dump({"segments": w.segments}, fh)
    return w


class SegmentConcatDiskFull(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="segconcat-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        patcher = mock.patch.object(roop.globals, "processing", True, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _parts_on_disk(self, w):
        return all(os.path.exists(os.path.join(self.dir, s["file"])) for s in w.segments)

    def test_insufficient_space_refuses_before_ffmpeg_and_keeps_parts(self):
        w = _writer(self.dir, [4096, 4096])
        with mock.patch.object(segment_writer.shutil, "disk_usage",
                               return_value=_Usage(10**12, 10**12, 1024)), \
                mock.patch.object(segment_writer.subprocess, "run") as run:
            with self.assertRaises(OSError) as ctx:
                w.close()
        self.assertEqual(ctx.exception.errno, errno.ENOSPC)
        self.assertIn("not enough space", str(ctx.exception))
        run.assert_not_called()
        self.assertTrue(self._parts_on_disk(w))
        self.assertTrue(os.path.exists(manifest_path(w.target_video)))

    def test_failed_concat_raises_removes_partial_output_and_keeps_parts(self):
        w = _writer(self.dir, [4096, 4096])

        def _half_written(cmd, **_):
            with open(cmd[-1], "wb") as fh:          # ffmpeg died mid-write
                fh.write(b"\0" * 100)
            return mock.Mock(returncode=1, stderr=b"No space left on device")

        with mock.patch.object(segment_writer.subprocess, "run", side_effect=_half_written):
            with self.assertRaises(IOError):
                w.close()
        self.assertFalse(os.path.exists(w.target_video))
        self.assertTrue(self._parts_on_disk(w))
        self.assertTrue(os.path.exists(manifest_path(w.target_video)))

    def test_successful_concat_still_cleans_up(self):
        w = _writer(self.dir, [4096, 4096])
        files = [s["file"] for s in w.segments]

        def _ok(cmd, **_):
            with open(cmd[-1], "wb") as fh:
                fh.write(b"\0" * 8192)
            return mock.Mock(returncode=0, stderr=b"")

        with mock.patch.object(segment_writer.subprocess, "run", side_effect=_ok):
            w.close()
        self.assertTrue(os.path.exists(w.target_video))
        self.assertFalse(any(os.path.exists(os.path.join(self.dir, f)) for f in files))
        self.assertFalse(os.path.exists(manifest_path(w.target_video)))

    def test_single_part_promotion_needs_no_free_space(self):
        # A rename on the same drive allocates nothing; it must not be refused.
        w = _writer(self.dir, [4096])
        with mock.patch.object(segment_writer.shutil, "disk_usage",
                               return_value=_Usage(10**12, 10**12, 0)):
            w.close()
        self.assertTrue(os.path.exists(w.target_video))


if __name__ == "__main__":
    unittest.main()
