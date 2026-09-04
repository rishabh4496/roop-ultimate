"""The output writer must convert AND tag its colour space on every path.

The defect these cover: the scale-for-odd-dimensions branch and the colour
conversion were an if/elif, so an odd-sized render skipped the conversion
entirely and shipped a file whose color_space / color_primaries /
color_transfer / color_range were all `unknown` -- leaving the matrix to the
player's guess.  Measured through real encode/decode
(tests/measure_color_roundtrip.py, 1280x720 chart):

    ODD, scale only     mean|d| 1.139  max 5  all tags unknown   <- shipped
    ODD, scale+convert  mean|d| 1.028  max 4  bt709 on all four

These are argv tests rather than render tests because the argv is where the
defect lived: both arms produced a perfectly valid video file.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals  # noqa: E402
from roop.ffmpeg_writer import FFMPEG_VideoWriter  # noqa: E402


def build(size, colorspace=None, env=None):
    """The argv the writer would spawn, without spawning it."""
    writer = FFMPEG_VideoWriter.__new__(FFMPEG_VideoWriter)
    writer.filename = "out.mp4"
    writer.codec = "libx264"
    writer.ext = "mp4"
    writer._size = size
    writer._fps = 25
    writer._crf = 20
    writer._audiofile = None
    writer._preset = "faster"
    writer._bitrate = None
    writer._logfile = None
    writer._threads = None
    writer._ffmpeg_params = None
    writer._colorspace = colorspace
    with mock.patch.dict(os.environ, env or {}, clear=False):
        return writer._build_cmd("libx264")


def vf_of(cmd):
    return cmd[cmd.index("-vf") + 1] if "-vf" in cmd else None


TAGS = ("-colorspace", "-color_primaries", "-color_trc", "-color_range")


class ColourFilterTest(unittest.TestCase):

    def test_even_dimensions_convert(self):
        vf = vf_of(build((640, 480)))
        self.assertEqual(vf, "colorspace=bt709:iall=bt601-6-625:fast=1")

    def test_odd_dimensions_scale_AND_convert(self):
        """The regression. Before the fix this was `scale=...` alone."""
        vf = vf_of(build((641, 481)))
        self.assertEqual(vf, "scale=640:480,colorspace=bt709:iall=bt601-6-625:fast=1")

    def test_scale_runs_before_the_conversion(self):
        vf = vf_of(build((641, 481)))
        self.assertLess(vf.index("scale="), vf.index("colorspace="),
                        "converting before resizing would convert pixels the "
                        "resampler then re-mixes across the matrix boundary")

    def test_every_converting_path_is_tagged(self):
        """A converted file with no tags leaves the matrix to the player's guess."""
        for size in ((640, 480), (641, 481), (640, 481), (641, 480)):
            cmd = build(size)
            for tag in TAGS:
                self.assertIn(tag, cmd, "%r lost %s" % (size, tag))
            self.assertEqual(cmd[cmd.index("-colorspace") + 1], "bt709")
            self.assertEqual(cmd[cmd.index("-color_range") + 1], "tv")

    def test_tags_name_the_same_matrix_the_filter_produces(self):
        cmd = build((640, 480))
        self.assertIn("colorspace=bt709", vf_of(cmd))
        for tag in ("-colorspace", "-color_primaries", "-color_trc"):
            self.assertEqual(cmd[cmd.index(tag) + 1], "bt709")

    def test_colorspace_off_disables_both_the_filter_and_the_tags(self):
        """`off` means the caller already owns the display-space conversion.

        Tagging bt709 while applying no conversion would be a worse lie than
        leaving the file untagged.
        """
        for value in ("off", "none", "passthrough", "0", "false"):
            cmd = build((640, 480), colorspace=value)
            self.assertIsNone(vf_of(cmd), value)
            for tag in TAGS:
                self.assertNotIn(tag, cmd, "%s still tagged %s" % (value, tag))

    def test_colorspace_off_still_scales_odd_dimensions(self):
        """yuv420p needs even dimensions; that is geometry, not colour."""
        cmd = build((641, 481), colorspace="off")
        self.assertEqual(vf_of(cmd), "scale=640:480")
        for tag in TAGS:
            self.assertNotIn(tag, cmd)

    def test_environment_override_is_honoured(self):
        cmd = build((640, 480), env={"ROOP_FFMPEG_COLORSPACE": "off"})
        self.assertIsNone(vf_of(cmd))
        cmd = build((640, 480), env={"ROOP_FFMPEG_COLORSPACE": "bt709"})
        self.assertIn("colorspace=bt709", vf_of(cmd))

    def test_explicit_constructor_argument_beats_the_environment(self):
        cmd = build((640, 480), colorspace="bt709",
                    env={"ROOP_FFMPEG_COLORSPACE": "off"})
        self.assertIn("colorspace=bt709", vf_of(cmd))

    def test_pix_fmt_stays_yuv420p(self):
        # Two -pix_fmt args: bgr24 describes the rawvideo INPUT, yuv420p the
        # encoded output. The output one is the last.
        cmd = build((640, 480))
        indices = [i for i, a in enumerate(cmd) if a == "-pix_fmt"]
        self.assertEqual(cmd[indices[0] + 1], "bgr24")
        self.assertEqual(cmd[indices[-1] + 1], "yuv420p")


if __name__ == "__main__":
    unittest.main()
