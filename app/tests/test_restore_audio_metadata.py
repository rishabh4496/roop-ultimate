"""The audio mux must carry the ORIGINAL container's global metadata.

WHY THIS EXISTS.  `restore_audio` is the final pass of every video render: it
takes the freshly encoded video (input 0) and stream-copies the source's audio
(input 1) into the delivered file.  It mapped the two streams explicitly and
said nothing about metadata, so ffmpeg applied its default -- map global
metadata from input 0, the file it had just created seconds earlier, whose only
tags are the ones the encoder invented.  Everything the original container
carried was therefore dropped on every render, silently.

Measured on the physical RTX 4070 host, `double/d4.mp4` through the real
function:

    original                 TAG:creation_time=2026-08-07T23:10:50.000000Z
    output, before the fix   (absent)
    output, after the fix    TAG:creation_time=2026-08-07T23:10:50.000000Z

Nothing failed and nothing warned: the file plays, the audio is in sync, the
return code is 0.  This is the same shape as the defects the swap audit keeps
missing -- an operation reports success while quietly not doing the part nobody
asserted on.

THE HALF THAT MUST NOT REGRESS.  `-map_metadata 1` maps GLOBAL metadata only.
Stream-level side data -- specifically the display matrix that encodes rotation
-- is left untouched, and it has to stay that way.  The decode pipe runs
`-noautorotate` (roop/nvdec_reader.py), so frames are detected, swapped and
encoded in RAW orientation; stamping the source's rotation onto the result
would make a player rotate footage that was never un-rotated, turning correct
output sideways on exactly the phone/action-cam clips that carry the tag.
Verified on hardware that rotation side data is absent from the output both
before and after this change.  Do not widen this to `-map_metadata:s:v`.
"""

import os
import sys
import unittest
from unittest import mock

_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from roop import util_ffmpeg  # noqa: E402


def _captured_command(trim_start=0, trim_end=60):
    """Run restore_audio with ffmpeg stubbed out, return the argv it built."""
    seen = {}

    def _fake_run(commands):
        seen["cmd"] = list(commands)
        return True

    with mock.patch.object(util_ffmpeg, "run_ffmpeg", _fake_run), \
         mock.patch.object(util_ffmpeg.util, "detect_fps", lambda _p: 30.0), \
         mock.patch.object(util_ffmpeg.util, "constant_frame_rate", lambda f: f), \
         mock.patch.object(util_ffmpeg.util, "audio_sample_rate", lambda _p: 48000):
        util_ffmpeg.restore_audio("processed.mp4", "original.mp4",
                                  trim_start, trim_end, "final.mp4")
    return seen["cmd"]


class RestoreAudioMetadataTests(unittest.TestCase):
    def test_global_metadata_is_mapped_from_the_original(self):
        cmd = _captured_command()
        self.assertIn("-map_metadata", cmd,
                      "restore_audio dropped the original container's metadata")
        self.assertEqual(cmd[cmd.index("-map_metadata") + 1], "1",
                         "metadata must come from input 1 (the original); "
                         "input 0 is the file ffmpeg just encoded")

    def test_rotation_side_data_is_not_mapped(self):
        """Widening this to per-stream would rotate already-unrotated frames."""
        cmd = _captured_command()
        for arg in cmd:
            self.assertFalse(
                arg.startswith("-map_metadata:"),
                f"per-stream metadata mapping ({arg}) would carry the source's "
                "display matrix onto frames decoded with -noautorotate")

    def test_the_stream_mapping_and_stream_copy_are_unchanged(self):
        """Metadata must ride along with, not displace, the existing mux."""
        cmd = _captured_command()
        for expected in ("0:v:0", "1:a:0?"):
            self.assertIn(expected, cmd)
        # Both streams still copied, never re-encoded -- a re-encode here would
        # cost a full transcode and lose quality on the delivered file.
        self.assertEqual(cmd[cmd.index("-c:v") + 1], "copy")
        self.assertEqual(cmd[cmd.index("-c:a") + 1], "copy")

    def test_metadata_flag_precedes_the_output_path(self):
        """An ffmpeg option after the output filename applies to nothing."""
        cmd = _captured_command()
        self.assertLess(cmd.index("-map_metadata"), cmd.index("final.mp4"))


if __name__ == "__main__":
    unittest.main()
