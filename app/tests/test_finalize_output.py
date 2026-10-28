"""Stage 4 finalization: mux, verify, and delete the intermediate ONLY after the verification passed.

Real ffmpeg, real files. The defect these guard is not an argv: `core.batch_process_regular` used to
delete the intermediate video whenever `os.path.isfile(destination)`, so a mux that failed half way (a
partial destination) lost the render, and nothing marked the job failed. A stubbed argv cannot show that.

The lip-sync test does not read a timestamp: it cross-correlates the delivered file's audio against the
SAME seconds of the source (deterministic noise, so one alignment is unambiguous) and asserts the offset is
under one video frame.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import finalize_output as fo  # noqa: E402
from roop.ffmpeg_path import ffmpeg_binary  # noqa: E402


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())


FPS = 30
SR = 48000
# Written into the H.264 VUI by x264 itself. `-colorspace/-color_primaries/-color_trc` on the command line
# tagged only the matrix and range on this build (primaries and transfer stayed `unknown`), which let a first
# version of the colour test pass with None == None; `_video` now asserts the fixture really carries them.
BT709 = ["-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=tv"]
BT2020 = ["-x264-params", "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=tv"]


def _ff(*args):
    subprocess.run([ffmpeg_binary(), "-y", "-v", "error", *args], check=True)


def _video(path, frames, tags=BT709):
    _ff("-f", "lavfi", "-i", f"testsrc2=s=160x120:r={FPS}", "-frames:v", str(frames),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", *tags, str(path))
    tagged = fo.color_tags(fo.probe_container(str(path)))
    assert all(tagged.values()), f"the fixture is not tagged: {tagged}"
    return str(path)


def _source(path, seconds=6.0):
    """A source clip: video + deterministic-noise AAC audio."""
    _ff("-f", "lavfi", "-i", f"testsrc2=s=160x120:r={FPS}", "-f", "lavfi",
        "-i", f"anoisesrc=d={seconds}:c=white:r={SR}:a=0.5:seed=7", "-t", str(seconds),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(path))
    return str(path)


def _pcm(path, start=0.0, seconds=1.0):
    out = subprocess.run([ffmpeg_binary(), "-v", "error", "-ss", f"{start:.6f}", "-i", str(path), "-t",
                          f"{seconds:.6f}", "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def _lag_seconds(ref, got, max_s=0.2):
    """Seconds `got` is LATE relative to `ref` (positive) by FFT cross-correlation."""
    n = min(len(ref), len(got))
    ref, got = ref[:n].astype(np.float64), got[:n].astype(np.float64)
    size = 1 << (2 * n - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(got, size) * np.conj(np.fft.rfft(ref, size)), size)
    window = int(max_s * SR)
    lags = np.concatenate([np.arange(0, window), np.arange(-window, 0)])
    vals = np.concatenate([corr[:window], corr[size - window:]])
    return float(lags[int(np.argmax(vals))]) / SR


def _finalize(tmp_path, frames=90, trim=(None, None), tags=BT709, **kw):
    temp = _video(tmp_path / "out__temp.mp4", frames, tags)
    src = _source(tmp_path / "source.mp4")
    dest = str(tmp_path / "final.mp4")
    return temp, src, dest, fo.finalize_render(
        temp, dest, audio_source=src, trim_start=trim[0], trim_end=trim[1], expected_frames=frames, **kw)


def test_untrimmed_render_is_verified_then_the_intermediate_is_deleted(tmp_path):
    temp, src, dest, r = _finalize(tmp_path)
    assert r["ok"], r["problems"]
    assert r["temp_deleted"] and not os.path.exists(temp)
    assert r["verify"]["frames"] == 90 and r["verify"]["frames_source"] == "header"
    assert r["verify"]["audio_present"] and r["expect_audio"]
    assert r["orphans"] == []
    assert r["verify"]["probe_ms"] < 2000          # a header read, not a decode


@pytest.mark.parametrize("trim", [(30, 90), (45, 105)])
def test_trimmed_render_audio_is_aligned_to_under_one_frame(tmp_path, trim):
    start, end = trim
    temp, src, dest, r = _finalize(tmp_path, frames=end - start, trim=trim)
    assert r["ok"], r["problems"]
    assert abs(r["verify"]["audio_start_offset_ms"]) <= 1000 / FPS
    # The delivered file's first second of audio is the source's [start/fps, start/fps + 1 s].
    lag = _lag_seconds(_pcm(src, start / FPS, 1.0), _pcm(dest, 0.0, 1.0))
    assert abs(lag) < 1.0 / FPS, f"audio is {lag * 1000:.1f} ms off the video (> 1 frame)"


@pytest.mark.parametrize("name,tags", [("bt709", BT709), ("bt2020", BT2020)])
def test_container_colour_tags_survive_the_mux(tmp_path, name, tags):
    temp, src, dest, r = _finalize(tmp_path, tags=tags)
    assert r["ok"], r["problems"]
    assert r["verify"]["color_preserved"]
    color = r["verify"]["color"]
    assert color["color_primaries"] == name
    assert color["color_space"] in ("bt709", "bt2020nc")


def test_colour_tag_loss_is_caught(tmp_path):
    """A mux that strips the tags must fail verification, with the intermediate kept."""
    def strip_tags(temp, source, s, e, dest):
        # ffmpeg copies colour properties from decoded frames into the encoder whatever flags are given, so a
        # re-encode of the TAGGED intermediate stays tagged. A mux that loses the tags is simulated with
        # untagged frames of the same length plus the source audio: frame count and audio are intact, so the
        # ONLY thing wrong with this file is the colour metadata.
        _ff("-f", "lavfi", "-i", f"testsrc2=s=160x120:r={FPS}", "-i", source, "-map", "0:v", "-map", "1:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy", "-frames:v", "60", "-t", "2.0", dest)
        return True
    temp = _video(tmp_path / "t__temp.mp4", 60, BT2020)
    dest = str(tmp_path / "f.mp4")
    r = fo.finalize_render(temp, dest, audio_source=_source(tmp_path / "source.mp4"), trim_start=None,
                           trim_end=None, expected_frames=60, skip_audio=False, mux=strip_tags)
    assert not r["ok"] and any("colour tags changed" in p for p in r["problems"]), r["problems"]
    assert os.path.exists(temp) and not os.path.exists(dest)


def test_a_mux_that_dies_leaves_the_intermediate_and_no_partial_output(tmp_path):
    """The old behaviour: a PARTIAL destination existed, `isfile` was true, the intermediate was deleted."""
    temp = _video(tmp_path / "t__temp.mp4", 60)
    src = _source(tmp_path / "source.mp4")
    dest = str(tmp_path / "f.mp4")

    def dies(temp_v, source, s, e, destination):
        with open(destination, "wb") as fh:
            with open(temp_v, "rb") as rf:
                fh.write(rf.read()[:4000])      # a truncated mp4
        return False

    r = fo.finalize_render(temp, dest, audio_source=src, trim_start=None, trim_end=None,
                           expected_frames=60, mux=dies)
    assert not r["ok"] and not r["temp_deleted"]
    assert os.path.exists(temp), "the only complete copy of the render was deleted"
    assert not os.path.exists(dest), "a partial file is left looking like a result"
    assert r["kept_intermediate"] == temp
    assert any("mux failed" in p for p in r["problems"])


def test_a_mux_that_reports_success_over_garbage_is_still_rejected(tmp_path):
    temp = _video(tmp_path / "t__temp.mp4", 60)
    src = _source(tmp_path / "source.mp4")
    dest = str(tmp_path / "f.mp4")

    def liar(temp_v, source, s, e, destination):
        with open(destination, "wb") as fh:
            fh.write(b"\x00" * 512)
        return True

    r = fo.finalize_render(temp, dest, audio_source=src, trim_start=None, trim_end=None,
                           expected_frames=60, mux=liar)
    assert not r["ok"] and os.path.exists(temp) and not os.path.exists(dest)
    assert any("ffprobe" in p for p in r["problems"])


def test_a_mux_that_drops_a_frame_is_caught_against_the_intermediate(tmp_path):
    temp = _video(tmp_path / "t__temp.mp4", 60)
    src = _source(tmp_path / "source.mp4")
    dest = str(tmp_path / "f.mp4")

    def drops(temp_v, source, s, e, destination):
        _ff("-i", temp_v, "-c:v", "libx264", "-pix_fmt", "yuv420p", *BT709, "-frames:v", "59", destination)
        return True

    r = fo.finalize_render(temp, dest, audio_source=src, trim_start=None, trim_end=None, expected_frames=None,
                           mux=drops)
    assert not r["ok"] and os.path.exists(temp)
    assert any("changed the frame count: 60 -> 59" in p for p in r["problems"])


def test_a_frame_count_that_does_not_match_the_trim_is_caught(tmp_path):
    temp, src, dest, r = _finalize(tmp_path, frames=60)
    assert r["ok"]
    temp2 = _video(tmp_path / "b__temp.mp4", 60)
    r2 = fo.finalize_render(temp2, str(tmp_path / "b.mp4"), audio_source=src, trim_start=None, trim_end=None,
                            expected_frames=61)
    assert not r2["ok"] and any("!= the 61" in p for p in r2["problems"])
    assert os.path.exists(temp2) and not os.path.exists(tmp_path / "b.mp4")


def test_a_stopped_render_is_checked_against_its_intermediate_not_the_trim(tmp_path):
    temp = _video(tmp_path / "s__temp.mp4", 40)
    src = _source(tmp_path / "source.mp4")
    dest = str(tmp_path / "s.mp4")
    r = fo.finalize_render(temp, dest, audio_source=src, trim_start=None, trim_end=None,
                           expected_frames=600, stopped=True)
    assert r["ok"], r["problems"]
    assert r["verify"]["frames"] == 40 and r["expected_frames"] is None


def test_silent_render_is_moved_and_verified(tmp_path):
    temp = _video(tmp_path / "m__temp.mp4", 30)
    dest = str(tmp_path / "m.mp4")
    r = fo.finalize_render(temp, dest, audio_source=None, trim_start=None, trim_end=None, expected_frames=30,
                           skip_audio=True)
    assert r["ok"] and not os.path.exists(temp) and os.path.exists(dest)
    assert r["verify"]["frames"] == 30


def test_helper_files_are_removed_and_leftover_parts_are_reported(tmp_path):
    temp = _video(tmp_path / "o__temp.mp4", 30)
    src = _source(tmp_path / "source.mp4")
    dest = str(tmp_path / "o.mp4")
    scratch = str(tmp_path / "o.__audio_cut.mka")
    part = str(tmp_path / ".o__temp.mp4.seg0001.mp4")
    for p in (scratch, part):
        with open(p, "wb") as f:
            f.write(b"x")
    r = fo.finalize_render(temp, dest, audio_source=src, trim_start=None, trim_end=None, expected_frames=30)
    assert r["ok"]
    assert not os.path.exists(scratch), "audio-cut scratch left behind"
    assert r["orphans"] == [part], "a leftover segment part must be reported, not hidden"


def test_unreadable_intermediate_is_kept_and_reported(tmp_path):
    temp = str(tmp_path / "bad__temp.mp4")
    with open(temp, "wb") as f:
        f.write(b"not a video")
    r = fo.finalize_render(temp, str(tmp_path / "bad.mp4"), audio_source=None, trim_start=None,
                           trim_end=None, expected_frames=10)
    assert not r["ok"] and os.path.exists(temp)
