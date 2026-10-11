"""finalize_web_video encodes on NVENC first and falls back to libx264.

The subprocess is faked, so this pins the command lines and the retry order and
does not depend on the host having NVENC. The real-ffmpeg odd-dimension / muted
cases live in tools/test_ffmpeg_robustness.py.
"""
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import util_ffmpeg


def _value_after(cmd, flag):
    return cmd[cmd.index(flag) + 1]


@pytest.fixture
def fake_run(monkeypatch, tmp_path):
    """Patch subprocess.run; returns (calls, set_results). Creates the output file on success."""
    calls = []
    results = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        code = results[len(calls) - 1] if len(calls) <= len(results) else 0
        if code == 0:
            open(cmd[-1], 'wb').close()
        return SimpleNamespace(returncode=code, stderr=b'Cannot load nvcuda.dll' if code else b'')

    monkeypatch.setattr(util_ffmpeg.subprocess, 'run', run)
    monkeypatch.setattr(util_ffmpeg.util, 'audio_sample_rate', lambda p: 48000)
    monkeypatch.setattr(util_ffmpeg.util, 'detect_fps', lambda p: 23.976024)
    monkeypatch.setattr(util_ffmpeg, '_stream_duration', lambda p, selector: 3.003)
    monkeypatch.delenv('ROOP_FINALIZE_NVENC', raising=False)
    src = tmp_path / 'raw.mp4'
    src.write_bytes(b'x')
    return calls, results, str(src), str(tmp_path / 'final.mp4')


def test_nvenc_command_matches_the_brief(fake_run):
    calls, _, src, dest = fake_run
    assert util_ffmpeg.finalize_web_video(src, dest, audio_source=src) is True
    assert len(calls) == 1
    cmd = calls[0]
    assert _value_after(cmd, '-c:v') == 'h264_nvenc'
    assert _value_after(cmd, '-preset') == 'p6'
    assert _value_after(cmd, '-tune') == 'hq'
    assert _value_after(cmd, '-rc') == 'vbr'
    assert _value_after(cmd, '-cq') == '19'
    assert _value_after(cmd, '-b:v') == '0'
    assert _value_after(cmd, '-pix_fmt') == 'yuv420p'
    assert _value_after(cmd, '-movflags') == '+faststart'
    assert _value_after(cmd, '-vf') == 'pad=ceil(iw/2)*2:ceil(ih/2)*2'
    assert _value_after(cmd, '-r') == '24000/1001'          # exact, not 23.976024
    assert _value_after(cmd, '-c:a') == 'aac' and _value_after(cmd, '-b:a') == '192k'
    maps = [cmd[i + 1] for i, a in enumerate(cmd) if a == '-map']
    assert maps == ['0:v:0', '1:a:0?']
    assert '-crf' not in cmd


def test_output_is_bounded_by_video_length_not_shortest(fake_run, monkeypatch):
    calls, _, src, dest = fake_run
    assert util_ffmpeg.finalize_web_video(src, dest, audio_source=src) is True
    assert _value_after(calls[0], '-t') == '3.003000' and '-shortest' not in calls[0]
    # Video length unknown: the old behaviour.
    calls.clear()
    monkeypatch.setattr(util_ffmpeg, '_stream_duration', lambda p, selector: None)
    assert util_ffmpeg.finalize_web_video(src, dest, audio_source=src) is True
    assert '-shortest' in calls[0] and '-t' not in calls[0]


def test_muted_result_is_stripped(fake_run, monkeypatch):
    calls, _, src, dest = fake_run
    monkeypatch.setattr(util_ffmpeg.util, 'audio_sample_rate', lambda p: None)
    assert util_ffmpeg.finalize_web_video(src, dest, audio_source=src) is True
    cmd = calls[0]
    assert '-an' in cmd and '-c:a' not in cmd and '1:a:0?' not in cmd


def test_falls_back_to_the_unchanged_libx264_command(fake_run):
    calls, results, src, dest = fake_run
    results.append(1)                                        # NVENC fails, libx264 succeeds
    assert util_ffmpeg.finalize_web_video(src, dest, audio_source=src, crf=21) is True
    assert [_value_after(c, '-c:v') for c in calls] == ['h264_nvenc', 'libx264']
    soft = calls[1]
    assert _value_after(soft, '-crf') == '21'                # the caller's crf, not the NVENC cq
    assert '-r' not in soft                                  # as before this change
    assert '-cq' not in soft and '-preset' not in soft
    assert os.path.isfile(dest)


def test_both_encoders_failing_returns_false_and_leaves_no_partial(fake_run):
    calls, results, src, dest = fake_run
    results.extend([1, 1])
    assert util_ffmpeg.finalize_web_video(src, dest, audio_source=src) is False
    assert len(calls) == 2
    assert not os.path.exists(dest)


def test_env_switch_skips_nvenc(fake_run, monkeypatch):
    calls, _, src, dest = fake_run
    monkeypatch.setenv('ROOP_FINALIZE_NVENC', '0')
    assert util_ffmpeg.finalize_web_video(src, dest, audio_source=src) is True
    assert [_value_after(c, '-c:v') for c in calls] == ['libx264']


def test_in_place_finalize_uses_a_temp_then_replaces(fake_run):
    calls, _, src, _ = fake_run
    assert util_ffmpeg.finalize_web_video(src, src, audio_source=src) is True
    assert calls[0][-1] == src + '.web.tmp.mp4'
    assert os.path.isfile(src) and not os.path.exists(src + '.web.tmp.mp4')
