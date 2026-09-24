"""A trimmed render's video must not start late against its audio.

`restore_audio` cut the source audio with an INPUT-side `-ss` and `-c:a copy`.
For stream copy that is not a cut: the demuxer seeks the file to the VIDEO
keyframe before the trim point, keeps the audio from there with negative
timestamps, and `-avoid_negative_ts make_zero` shifts every stream by that much.
On b1.mp4 trimmed at frame 200 the delivered file's video began at 3.788 s with
the audio at 0 (found 2026-09-24: the output player's compare view put the two
sides on different scenes, although the clocks agreed to 16 ms).

The fix cuts the audio in its own audio-only pass with an OUTPUT-side `-ss`
(exact to the packet, still a stream copy), then muxes. Untrimmed renders keep
the single command -- test_restore_audio_metadata.py pins that one.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import util_ffmpeg  # noqa: E402


def _commands(trim_start, trim_end, cut_ok=True, cut_bytes=b"x"):
    seen = []

    def fake_run(commands):
        seen.append(list(commands))
        out = commands[-1]
        if out.endswith(".__audio_cut.mka") and cut_ok:
            with open(out, "wb") as fh:
                fh.write(cut_bytes)
        return cut_ok or not out.endswith(".__audio_cut.mka")

    tmp = tempfile.mkdtemp()
    final = os.path.join(tmp, "final.mp4")
    with mock.patch.object(util_ffmpeg, "run_ffmpeg", fake_run), \
         mock.patch.object(util_ffmpeg, "_stream_duration", lambda _p, _s: 4.8), \
         mock.patch.object(util_ffmpeg.util, "detect_fps", lambda _p: 25.0), \
         mock.patch.object(util_ffmpeg.util, "constant_frame_rate", lambda f: f), \
         mock.patch.object(util_ffmpeg.util, "audio_sample_rate", lambda _p: 48000):
        ok = util_ffmpeg.restore_audio("processed.mp4", "original.mp4", trim_start, trim_end, final)
    leftover = [n for n in os.listdir(tmp) if n.endswith(".mka")]
    shutil.rmtree(tmp, ignore_errors=True)
    return ok, seen, leftover


class TrimmedAudioIsCutSeparately(unittest.TestCase):
    def test_trimmed_render_cuts_audio_with_an_output_side_seek(self):
        ok, seen, leftover = _commands(200, 320)
        self.assertTrue(ok)
        self.assertEqual(len(seen), 2, "expected an audio cut pass then the mux")
        cut, mux = seen
        # -ss AFTER the input = output-side = exact; before it would be the bug.
        self.assertLess(cut.index("-i"), cut.index("-ss"))
        self.assertEqual(float(cut[cut.index("-ss") + 1]), 8.0)        # 200 / 25
        self.assertAlmostEqual(float(cut[cut.index("-t") + 1]), 4.8)   # 120 / 25
        self.assertEqual(cut[cut.index("-c:a") + 1], "copy")
        self.assertIn("-vn", cut)
        # The mux takes the cut audio, keeps metadata from the ORIGINAL (input
        # 1), stream-copies both, and does NOT seek the original at all.
        self.assertIn("2:a:0?", mux)
        self.assertEqual(mux[mux.index("-map_metadata") + 1], "1")
        self.assertEqual(mux[mux.index("-c:v") + 1], "copy")
        self.assertEqual(mux[mux.index("-c:a") + 1], "copy")
        self.assertNotIn("-ss", mux)
        # Bounded by the video's own length; `-shortest` dropped the last
        # frames (interleave buffering against a packet-aligned audio cut).
        self.assertNotIn("-shortest", mux)
        self.assertAlmostEqual(float(mux[mux.index("-t") + 1]), 4.8)
        self.assertEqual(leftover, [], "the temporary audio cut was left behind")

    def test_untrimmed_render_keeps_the_single_command(self):
        ok, seen, _ = _commands(0, 120)
        self.assertTrue(ok)
        self.assertEqual(len(seen), 1)
        self.assertIn("1:a:0?", seen[0])

    def test_a_failed_cut_falls_back_to_the_single_command(self):
        """No audio stream (or an odd container): still deliver the file."""
        ok, seen, leftover = _commands(200, 320, cut_ok=False)
        self.assertTrue(ok)
        self.assertEqual(len(seen), 2)
        self.assertIn("1:a:0?", seen[1])
        self.assertEqual(leftover, [])


def _tool(name):
    from roop.ffmpeg_path import ffmpeg_binary
    exe = ffmpeg_binary()
    if not exe:
        return None
    cand = os.path.join(os.path.dirname(exe), name + (".exe" if os.name == "nt" else ""))
    return cand if os.path.exists(cand) else shutil.which(name)


class RealMuxStartsTogether(unittest.TestCase):
    """Through real ffmpeg, on a clip whose keyframes are 2 s apart."""

    def test_video_and_audio_start_together(self):
        ffmpeg, ffprobe = _tool("ffmpeg"), _tool("ffprobe")
        if not ffmpeg or not ffprobe:
            self.skipTest("ffmpeg/ffprobe not available")
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        src = os.path.join(tmp, "src.mp4")
        inter = os.path.join(tmp, "inter.mp4")
        final = os.path.join(tmp, "final.mp4")
        run = lambda args: subprocess.run([ffmpeg, "-v", "error", "-y"] + args, check=True)  # noqa: E731
        # 10 s @25 fps, keyframe every 50 frames, with AAC audio.
        run(["-f", "lavfi", "-i", "testsrc=size=160x90:rate=25:duration=10",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
             "-c:v", "libx264", "-g", "50", "-keyint_min", "50", "-sc_threshold", "0",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", src])
        # The "rendered" trim: frames 137..236 (5.48 s .. 9.44 s), mid-GOP.
        run(["-ss", "5.48", "-i", src, "-t", "4.0", "-an", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", inter])
        with mock.patch.object(util_ffmpeg.util, "detect_fps", lambda _p: 25.0), \
             mock.patch.object(util_ffmpeg.util, "constant_frame_rate", lambda f: f):
            self.assertTrue(util_ffmpeg.restore_audio(inter, src, 137, 237, final))
        probe = subprocess.run([ffprobe, "-v", "error", "-count_packets", "-show_entries",
                                "stream=codec_type,start_time,nb_read_packets", "-of", "csv=p=0", final],
                               capture_output=True, text=True, check=True).stdout.split()
        rows = [line.split(",") for line in probe]
        starts = {r[0]: float(r[1]) for r in rows}
        packets = {r[0]: int(r[2]) for r in rows}
        self.assertEqual(packets["video"], 100, f"rendered frames were dropped: {packets}")
        self.assertIn("audio", starts)
        self.assertLess(abs(starts["video"] - starts["audio"]), 0.05,
                        f"video/audio start offset: {starts}")
        self.assertLess(starts["video"], 0.1, f"video starts late: {starts}")


if __name__ == "__main__":
    unittest.main()
