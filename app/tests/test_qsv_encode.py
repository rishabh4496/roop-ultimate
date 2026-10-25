"""roop/qsv_encode.py: Intel QuickSync as a whole-render video encoder (NOT a per-segment split).

No QuickSync frame is encoded here: the bundled ffmpeg has no ``*_qsv`` encoder and no Intel GPU is enabled on either
development machine. What IS tested: the exact command line each QSV codec gets (and that NVENC / x265 are unchanged),
the real ffmpeg's answer for an encoder it does not have, the pre-flight decision table, the ``core`` and ``api``
call sites that consume it, and the ``SegmentedVideoWriter`` guard that keeps two encoders out of one file
(``tests/probe_mixed_encoder_concat.py``: an HEVC NVENC + x265 stream-copy join exits 0 and loses a third of the
frames). A real QSV encode stays UNVERIFIED until someone runs it on an Intel machine.
"""
import os
import subprocess
import sys
from unittest import mock

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import ffmpeg_writer, qsv_encode  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("ROOP_QSV", "ROOP_QSV_PRESET", "ROOP_NVENC_PRESET", "ROOP_NVENC_QUALITY_FIRST", "ROOP_NVENC_MULTIPASS",
                 "ROOP_NVENC_SPATIAL_AQ", "ROOP_NVENC_TEMPORAL_AQ", "ROOP_ENCODER_PRESET", "ROOP_RUNTIME_ENCODER_PRESET",
                 "ROOP_FFMPEG_THREADS", "ROOP_RUNTIME_FFMPEG_THREADS"):
        monkeypatch.delenv(name, raising=False)


def _argv(codec, quality=23):
    """The ffmpeg argv FFMPEG_VideoWriter would spawn for *codec*, without spawning anything."""
    with mock.patch.object(ffmpeg_writer.FFMPEG_VideoWriter, "_spawn", lambda self, c: setattr(self, "codec", c)):
        w = ffmpeg_writer.FFMPEG_VideoWriter("out.mp4", (320, 240), 25, codec=codec, crf=quality)
    return w._build_cmd(codec)


def _after(argv, flag):
    return argv[argv.index(flag) + 1]


class TestCommandLine:
    @pytest.mark.parametrize("codec", ["h264_qsv", "hevc_qsv", "av1_qsv"])
    def test_qsv_gets_icq_rate_control_and_nv12(self, codec):
        a = _argv(codec, 23)
        assert codec in a
        assert _after(a, "-global_quality") == "23"
        assert _after(a, "-preset") == "medium"
        for foreign in ("-crf", "-cq", "-rc", "-tune"):
            assert foreign not in a, f"{foreign} is not a QuickSync option"
        assert a[a.index("-pix_fmt", a.index(codec)) + 1] == "nv12"        # the OUTPUT pix_fmt, not the bgr24 input

    def test_quality_is_clamped_into_the_encoder_range(self):
        assert _after(_argv("hevc_qsv", 90), "-global_quality") == "51"
        assert _after(_argv("hevc_qsv", -4), "-global_quality") == "0"

    def test_preset_follows_env_and_a_typo_falls_back(self, monkeypatch):
        monkeypatch.setenv("ROOP_QSV_PRESET", "veryslow")
        assert _after(_argv("hevc_qsv"), "-preset") == "veryslow"
        monkeypatch.setenv("ROOP_QSV_PRESET", "turbo")
        assert _after(_argv("hevc_qsv"), "-preset") == "medium"

    def test_nvenc_and_software_command_lines_are_unchanged(self):
        n = _argv("hevc_nvenc", 23)
        assert _after(n, "-cq") == "23" and _after(n, "-rc") == "vbr" and "-global_quality" not in n
        assert n[n.index("-pix_fmt", n.index("hevc_nvenc")) + 1] == "yuv420p"
        x = _argv("libx265", 23)
        assert _after(x, "-crf") == "23" and "-global_quality" not in x
        assert x[x.index("-pix_fmt", x.index("libx265")) + 1] == "yuv420p"

    def test_the_other_rate_control_site_agrees(self):
        from roop.util_ffmpeg import _rate_control
        assert _rate_control("hevc_qsv", 23) == ["-global_quality", "23", "-preset", "medium"]
        assert _rate_control("libx264", 23) == ["-crf", "23"]

    def test_a_dead_qsv_encoder_has_a_software_fallback_entry(self):
        for codec, sw in (("h264_qsv", "libx264"), ("hevc_qsv", "libx265"), ("av1_qsv", "libx265")):
            assert ffmpeg_writer.FFMPEG_VideoWriter._SW_FALLBACK[codec] == sw


class TestProbe:
    def test_probe_builds_the_qsv_command(self):
        seen = {}

        class FakeProc:
            returncode = 0

            def communicate(self, timeout=None):
                return b"", b""

        def fake_popen(cmd, **kw):
            seen["cmd"] = cmd
            return FakeProc()

        with mock.patch.object(ffmpeg_writer.sp, "Popen", fake_popen):
            ok, _ = ffmpeg_writer.probe_encoder("hevc_qsv", 23)
        cmd = seen["cmd"]
        assert ok and _after(cmd, "-global_quality") == "23" and "-crf" not in cmd
        assert cmd[cmd.index("-pix_fmt") + 1] == "nv12"

    def test_the_real_ffmpeg_reports_a_missing_qsv_encoder_instead_of_hanging(self):
        try:
            listing = subprocess.run([ffmpeg_writer.FFMPEG_BINARY, "-hide_banner", "-encoders"], capture_output=True,
                                     timeout=30).stdout.decode("utf-8", "replace")
        except FileNotFoundError:
            pytest.skip("no ffmpeg binary on this machine (the CI light job installs one)")
        if " hevc_qsv " in listing:
            pytest.skip("this ffmpeg lists hevc_qsv; whether it can open depends on an Intel GPU (UNVERIFIED here)")
        ok, message = ffmpeg_writer.probe_encoder("hevc_qsv", 23)
        assert ok is False and message.strip(), "a missing encoder must come back as (False, reason)"


def _probe(results):
    """A probe whose per-codec answers are scripted; records the calls."""
    calls = []

    def probe(codec, quality):
        calls.append((codec, quality))
        return results[codec]
    probe.calls = calls
    return probe


class TestChoice:
    def test_a_working_encoder_is_left_alone_and_qsv_is_never_probed(self):
        p = _probe({"hevc_nvenc": (True, "")})
        c = qsv_encode.choose_encoder("hevc_nvenc", 14, p)
        assert (c.ok, c.codec, c.swapped) == (True, "hevc_nvenc", False)
        assert p.calls == [("hevc_nvenc", 14)]

    @pytest.mark.parametrize("nvenc,qsv", [("hevc_nvenc", "hevc_qsv"), ("h264_nvenc", "h264_qsv"),
                                           ("av1_nvenc", "av1_qsv")])
    def test_a_dead_nvenc_encoder_falls_to_its_own_family_qsv_codec(self, nvenc, qsv):
        p = _probe({nvenc: (False, "OpenEncodeSessionEx failed: out of memory"), qsv: (True, "")})
        c = qsv_encode.choose_encoder(nvenc, 14, p)
        assert (c.ok, c.codec, c.swapped) == (True, qsv, True)
        assert "out of memory" in c.message and qsv in c.message and "whole file" in c.message
        assert p.calls == [(nvenc, 14), (qsv, 14)]

    def test_both_dead_reports_both_reasons_and_does_not_swap(self):
        p = _probe({"hevc_nvenc": (False, "no NVENC"), "hevc_qsv": (False, "Unknown encoder 'hevc_qsv'")})
        c = qsv_encode.choose_encoder("hevc_nvenc", 14, p)
        assert (c.ok, c.codec, c.swapped) == (False, "hevc_nvenc", False)
        assert "no NVENC" in c.message and "hevc_qsv was tried" in c.message and "Unknown encoder" in c.message

    def test_roop_qsv_zero_removes_the_fallback_without_probing_qsv(self, monkeypatch):
        monkeypatch.setenv("ROOP_QSV", "0")
        p = _probe({"hevc_nvenc": (False, "no NVENC"), "hevc_qsv": (True, "")})
        c = qsv_encode.choose_encoder("hevc_nvenc", 14, p)
        assert not c.ok and "ROOP_QSV=0" in c.message
        assert [x[0] for x in p.calls] == ["hevc_nvenc"]

    @pytest.mark.parametrize("codec", ["libx264", "libx265", "libvpx-vp9", "hevc_qsv"])
    def test_a_non_nvenc_encoder_that_fails_is_not_rescued(self, codec):
        p = _probe({codec: (False, "broken")})
        c = qsv_encode.choose_encoder(codec, 14, p)
        assert (c.ok, c.swapped, c.message) == (False, False, "broken")
        assert len(p.calls) == 1


class TestCallSites:
    """The path the render takes: core's pre-flight and the codec list the API offers."""

    def test_core_preflight_swaps_the_global_encoder_for_this_render_only(self):
        core = pytest.importorskip("roop.core")
        import roop.globals as g
        statuses = []
        answers = {"hevc_nvenc": (False, "sessions exhausted"), "hevc_qsv": (True, "")}
        with mock.patch.object(g, "video_encoder", "hevc_nvenc"), mock.patch.object(g, "video_quality", 14), \
                mock.patch.object(ffmpeg_writer, "probe_encoder", lambda c, q: answers[c]), \
                mock.patch.object(core, "update_status", statuses.append):
            assert core.ensure_video_encoder() is True
            assert g.video_encoder == "hevc_qsv"
        assert len(statuses) == 1 and "QuickSync" in statuses[0]

    def test_core_preflight_aborts_with_the_reason_when_nothing_works(self):
        core = pytest.importorskip("roop.core")
        import roop.globals as g
        statuses = []
        answers = {"hevc_nvenc": (False, "sessions exhausted"), "hevc_qsv": (False, "Unknown encoder")}
        with mock.patch.object(g, "video_encoder", "hevc_nvenc"), mock.patch.object(g, "video_quality", 14), \
                mock.patch.object(ffmpeg_writer, "probe_encoder", lambda c, q: answers[c]), \
                mock.patch.object(core, "update_status", statuses.append):
            assert core.ensure_video_encoder() is False
            assert g.video_encoder == "hevc_nvenc"
        assert "not working, aborting" in statuses[0] and "sessions exhausted" in statuses[0]

    def test_core_preflight_with_a_healthy_encoder_says_nothing(self):
        core = pytest.importorskip("roop.core")
        import roop.globals as g
        statuses = []
        with mock.patch.object(g, "video_encoder", "libx264"), mock.patch.object(g, "video_quality", 14), \
                mock.patch.object(ffmpeg_writer, "probe_encoder", lambda c, q: (True, "")), \
                mock.patch.object(core, "update_status", statuses.append):
            assert core.ensure_video_encoder() is True
        assert statuses == []

    def test_the_api_offers_qsv_codecs_only_where_this_ffmpeg_lists_them(self):
        api = pytest.importorskip("api")
        blob = b" V....D libx264 x\n V....D h264_nvenc x\n V....D hevc_qsv x\n"
        fake = mock.Mock(return_value=mock.Mock(stdout=blob, stderr=b""))
        with mock.patch.dict(api._codec_cache, {"list": None}), mock.patch.object(api.subprocess, "run", fake):
            listed = api._available_video_codecs()
        assert "hevc_qsv" in listed and "h264_qsv" not in listed and "av1_qsv" not in listed


class TestOneEncoderPerFile:
    """The guard the mixed-encoder measurement says must stay."""

    def test_segmented_writer_refuses_to_change_encoder_between_segments(self, tmp_path, monkeypatch):
        import numpy as np
        from roop import segment_writer
        import roop.globals as g
        monkeypatch.setenv("ROOP_RESUME_CHUNK", "50")
        monkeypatch.setattr(g, "processing", True, raising=False)
        monkeypatch.setattr(segment_writer, "hardware_stream_enabled", lambda: False)
        made = []

        class Stub:
            def __init__(self, path, size, fps, codec, crf, audiofile=None, **kw):
                made.append(codec)
                # the second segment's encoder died and its own fallback quietly switched codec
                self.codec = codec if len(made) == 1 else "libx265"
                open(path, "wb").write(b"x" * 10)

            def write_frame(self, img):
                pass

            def close(self):
                pass

            def abort(self):
                pass

        monkeypatch.setattr(segment_writer, "FFMPEG_VideoWriter", Stub)
        w = segment_writer.SegmentedVideoWriter(str(tmp_path / "out.mp4"), (64, 64), 25, codec="hevc_nvenc", crf=23)
        frame = np.zeros((64, 64, 3), np.uint8)
        with pytest.raises(IOError, match="refusing to concat mixed-codec segments"):
            for _ in range(120):
                w.write_frame(frame)
        assert len(w.segments) == 1, "segment 1 stays committed so the run can resume"
