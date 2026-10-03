"""An untrimmed render must keep EVERY video frame when the audio is a hair short.

Found 2026-10-03 by tests/test_performance_regression.py: a 90-frame, 29.97 fps render
came back with 89 frames. `restore_audio`'s single-command branch (used whenever the
trim starts at 0) ended with `-shortest`; source audio routinely ends a fraction of a
frame before its video (an AAC packet is 21.3 ms), and `-shortest` then drops the last
video frame. The trimmed branch had already been fixed for the same defect (b1.mp4,
120 -> 117) with a bound by the render's own video length; this branch had not.

Real ffmpeg, real files: the defect is in how ffmpeg interleaves, so a stubbed argv
cannot show it.
"""
import os
import subprocess
import sys

import pytest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import util_ffmpeg  # noqa: E402
from roop.ffmpeg_path import ffmpeg_binary, ffprobe_binary  # noqa: E402

def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())


FRAMES, RATE = 90, "30000/1001"                     # 3.003 s
# A stream-copied AAC cut ends on a packet boundary: up to one packet past -t plus the
# encoder-priming packet. Two packets (2 x 21.3 ms); the defect it guards is SECONDS.
AAC_SLACK = 0.045


def _ff(*args):
    subprocess.run([ffmpeg_binary(), "-y", "-v", "error", *args], check=True)


def _probe(path, entry, select):
    out = subprocess.run([ffprobe_binary(), "-v", "error", "-count_frames", "-select_streams",
                          select, "-show_entries", f"stream={entry}", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    return float(out.splitlines()[0])


def _make(tmp_path, audio_seconds):
    video = tmp_path / "video.mp4"
    _ff("-f", "lavfi", "-i", f"testsrc2=s=160x120:r={RATE}", "-frames:v", str(FRAMES),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video))
    original = tmp_path / "original.mp4"
    _ff("-i", str(video), "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
        "-map", "0:v", "-map", "1:a", "-t", f"{audio_seconds:.6f}", "-c:v", "copy",
        "-c:a", "aac", str(original))
    return video, original


@pytest.mark.parametrize("audio_seconds", [2.986667,     # one AAC packet short (the s3 cut)
                                           2.9, 4.5])    # a frame+ short, and longer than the video
def test_every_video_frame_survives_the_audio_mux(tmp_path, audio_seconds):
    video, original = _make(tmp_path, audio_seconds)
    intermediate = tmp_path / "intermediate.mp4"        # what the encoder wrote: video only
    _ff("-i", str(video), "-an", "-c:v", "copy", str(intermediate))
    final = tmp_path / "final.mp4"
    assert util_ffmpeg.restore_audio(str(intermediate), str(original), None, None, str(final))
    assert _probe(final, "nb_read_frames", "v:0") == FRAMES
    video_s = _probe(final, "duration", "v:0")
    audio_s = _probe(final, "duration", "a:0")
    assert audio_s <= video_s + AAC_SLACK                  # audio never runs on past the video


def test_stopped_render_still_bounds_the_audio(tmp_path):
    """A render stopped at frame 45 of a 90-frame trim: video is the short side."""
    video, original = _make(tmp_path, 4.5)
    intermediate = tmp_path / "intermediate.mp4"
    _ff("-i", str(video), "-an", "-frames:v", "45", "-c:v", "libx264", str(intermediate))
    final = tmp_path / "final.mp4"
    assert util_ffmpeg.restore_audio(str(intermediate), str(original), 0, FRAMES, str(final))
    assert _probe(final, "nb_read_frames", "v:0") == 45
    assert _probe(final, "duration", "a:0") <= _probe(final, "duration", "v:0") + AAC_SLACK
