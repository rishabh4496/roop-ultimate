"""The render path must resolve ffmpeg, not assume it is on PATH.

WHY THIS EXISTS.  `ffmpeg_writer.FFMPEG_BINARY` was the literal string
`"ffmpeg"`, and `util_ffmpeg` built its command lines with the same bare name.
Pinokio's shell puts ffmpeg on PATH, so under the launcher this worked and every
prior validation passed.  Outside that shell -- the health worker's launch
probe, a benchmark child process, a plain terminal -- the encoder pre-flight
failed and aborted the run:

    Video encoder 'hevc_nvenc' is not working, aborting.
    ffmpeg binary 'ffmpeg' was not found on PATH.

Measured on the physical RTX 3060 host: a 900-frame render reported
`progress: 1.0` and `desc: 'Done'` within seconds, wrote no output file, and
both queued jobs were marked FAILED -- on a machine with a working ffmpeg that
exposes NVENC.  The failure is silent in the worst way: the run "finishes".

The repository had already been bitten by this twice (the hardware profile's
`shutil.which('ffmpeg')` reporting no NVDEC/NVENC, and the health worker's
`shutil.which('npm')`), each fixed with a private copy of the search.  These
tests pin the single shared resolver and the absence of bare invocations.
"""

import os
import sys
import unittest
from unittest import mock

_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from roop import ffmpeg_path  # noqa: E402


class FfmpegResolutionTests(unittest.TestCase):
    def test_path_lookup_wins_when_available(self):
        with mock.patch.object(ffmpeg_path.shutil, "which",
                               lambda _n: r"C:\tools\ffmpeg.exe"):
            self.assertEqual(ffmpeg_path.ffmpeg_binary(refresh=True),
                             r"C:\tools\ffmpeg.exe")

    def test_falls_back_to_the_pinokio_toolchain_when_not_on_path(self):
        """The exact case that broke: PATH has no ffmpeg, the toolchain does."""
        home = os.path.join(os.sep, "fake-pinokio")
        expected = os.path.join(home, "bin", "miniforge", "Library", "bin", "ffmpeg.exe")
        with mock.patch.object(ffmpeg_path.shutil, "which", lambda _n: None), \
                mock.patch.dict(os.environ, {"PINOKIO_HOME": home}, clear=False), \
                mock.patch.object(ffmpeg_path.os.path, "isdir", lambda p: p == home), \
                mock.patch.object(ffmpeg_path.os.path, "isfile",
                                  lambda p: p == expected):
            self.assertEqual(ffmpeg_path.ffmpeg_binary(refresh=True), expected)

    def test_unknown_environment_keeps_the_previous_behaviour(self):
        """Falling back to the bare name can only add working cases."""
        with mock.patch.object(ffmpeg_path.shutil, "which", lambda _n: None), \
                mock.patch.object(ffmpeg_path.os.path, "isfile", lambda _p: False):
            self.assertEqual(ffmpeg_path.ffmpeg_binary(refresh=True), "ffmpeg")

    def test_the_writer_uses_the_resolver(self):
        from roop import ffmpeg_writer
        self.assertTrue(ffmpeg_writer.FFMPEG_BINARY)
        # Either an absolute path we resolved, or the documented bare fallback.
        self.assertTrue(os.path.isabs(ffmpeg_writer.FFMPEG_BINARY)
                        or ffmpeg_writer.FFMPEG_BINARY == "ffmpeg")

    def test_no_render_path_module_invokes_a_bare_ffmpeg(self):
        """The regression guard: a new bare 'ffmpeg' must fail this test."""
        offenders = []
        for name in ("ffmpeg_writer.py", "util_ffmpeg.py"):
            path = os.path.join(_APP, "roop", name)
            with open(path, encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    stripped = line.strip()
                    if stripped.startswith("#") or "ffmpeg_path" in stripped:
                        continue
                    for literal in ("'ffmpeg',", '"ffmpeg",',
                                    "['ffmpeg'", '["ffmpeg"'):
                        if literal in stripped:
                            offenders.append(f"{name}:{number}: {stripped[:90]}")
        self.assertEqual(offenders, [],
                         "these invoke ffmpeg by bare name; use "
                         "roop.ffmpeg_path.ffmpeg_binary() instead")

    def test_profiler_and_render_path_agree(self):
        from roop.runtime_optimizer import HardwareProfiler
        resolved = ffmpeg_path.ffmpeg_binary(refresh=True)
        profiled = HardwareProfiler._resolve_ffmpeg()
        if os.path.isabs(resolved) and os.path.isfile(resolved):
            self.assertEqual(profiled, resolved)


if __name__ == "__main__":
    unittest.main()
