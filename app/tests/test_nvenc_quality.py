"""roop/nvenc_quality.py: the opt-in NVENC multipass / spatial AQ / temporal AQ options.

Pure tests run everywhere. ``HardwareBitstream`` drives a real NVENC and is marked ``gpu``; it skips (with the
reason) when PyNvVideoCodec / CUDA are absent. A skip is "unverified", not "passed".

The point of the hardware class: PyNvVideoCodec 2.2.3 ACCEPTS AND IGNORES every kwarg it does not know. A kwarg
that is accepted proves nothing, so the only valid test that an option is wired is that the bitstream CHANGES -
and that the brief's literal spellings (enableAQ / enableTemporalAQ) do NOT.
"""
import hashlib
import os
import sys
from unittest import mock

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import nvenc_quality as nq  # noqa: E402

_ENV = ("ROOP_NVENC_QUALITY_FIRST", "ROOP_NVENC_MULTIPASS", "ROOP_NVENC_SPATIAL_AQ", "ROOP_NVENC_TEMPORAL_AQ",
        "ROOP_NVENC_PRESET")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


class TestOptions:
    def test_default_is_off_and_adds_nothing(self):
        o = nq.options()
        assert not o.active and o.label() == "default"
        assert nq.ffmpeg_args("hevc_nvenc", o) == []
        assert nq.encoder_kwargs(o) == {}
        assert nq.preset_override(o) is None

    def test_bundle_is_the_requested_configuration(self, monkeypatch):
        monkeypatch.setenv("ROOP_NVENC_QUALITY_FIRST", "1")
        o = nq.options()
        assert (o.multipass, o.spatial_aq, o.temporal_aq, o.preset) == ("fullres", True, True, "p6")
        assert nq.ffmpeg_args("hevc_nvenc", o) == ["-multipass", "fullres", "-spatial-aq", "1", "-temporal-aq", "1"]
        assert nq.preset_override(o) == "p6"

    def test_individual_switches_override_the_bundle(self, monkeypatch):
        monkeypatch.setenv("ROOP_NVENC_QUALITY_FIRST", "1")
        monkeypatch.setenv("ROOP_NVENC_SPATIAL_AQ", "0")      # the measured loser
        monkeypatch.setenv("ROOP_NVENC_MULTIPASS", "off")
        o = nq.options()
        assert nq.ffmpeg_args("h264_nvenc", o) == ["-temporal-aq", "1"]

    def test_explicit_preset_beats_the_bundle(self, monkeypatch):
        monkeypatch.setenv("ROOP_NVENC_QUALITY_FIRST", "1")
        monkeypatch.setenv("ROOP_NVENC_PRESET", "p3")
        assert nq.preset_override() is None

    def test_a_typo_does_not_pick_a_different_mode(self, monkeypatch):
        monkeypatch.setenv("ROOP_NVENC_MULTIPASS", "fulres")
        assert nq.options().multipass == "off"
        monkeypatch.setenv("ROOP_NVENC_QUALITY_FIRST", "1")
        assert nq.options().multipass == "fullres"

    def test_software_codecs_get_no_nvenc_flags(self, monkeypatch):
        monkeypatch.setenv("ROOP_NVENC_QUALITY_FIRST", "1")
        for codec in ("libx264", "libx265", "", None):
            assert nq.ffmpeg_args(codec) == []

    def test_encoder_kwargs_use_only_the_proven_spellings(self, monkeypatch):
        monkeypatch.setenv("ROOP_NVENC_QUALITY_FIRST", "1")
        assert nq.encoder_kwargs() == {"multipass": "fullres", "aq": 1, "temporalaq": 1}
        assert not ({"enableAQ", "enableTemporalAQ", "spatial_aq", "temporal_aq"} & set(nq.encoder_kwargs()))


class TestWritersUseIt:
    """The options must reach the command line of the writers production actually runs."""

    def _ffmpeg_writer_cmd(self, codec):
        from roop.ffmpeg_writer import FFMPEG_VideoWriter
        w = FFMPEG_VideoWriter.__new__(FFMPEG_VideoWriter)
        w.filename, w.codec = os.path.join(HERE, "x.mp4"), codec
        w._size, w._fps, w._crf, w._audiofile, w._preset, w._bitrate = (640, 360), 24.0, 19, None, "faster", None
        w._logfile, w._colorspace, w._threads, w._ffmpeg_params, w._sar, w._dar = -1, None, None, None, None, None
        return w._build_cmd(codec)

    def _video_stream_cmd(self, codec):
        from roop.video_stream import NVHardwareVideoWriter
        w = NVHardwareVideoWriter.__new__(NVHardwareVideoWriter)
        w.output_path, w.width, w.height, w.fps = os.path.join(HERE, "x.mp4"), 640, 360, 24.0
        w.audio_source, w.preset, w.bitrate, w.cq, w.threads = None, "p4", None, 19, None
        w.ffmpeg_params, w.colorspace, w.sar, w.dar = [], None, None, None
        return w._command(codec)

    @staticmethod
    def _after(cmd, flag):
        return cmd[cmd.index(flag) + 1]

    @pytest.mark.parametrize("build", ["_ffmpeg_writer_cmd", "_video_stream_cmd"])
    def test_off_changes_nothing(self, build):
        cmd = getattr(self, build)("hevc_nvenc")
        for flag in ("-multipass", "-spatial-aq", "-temporal-aq"):
            assert flag not in cmd
        assert self._after(cmd, "-preset") in ("p4", "p5")      # the shipped presets, unchanged

    @pytest.mark.parametrize("build", ["_ffmpeg_writer_cmd", "_video_stream_cmd"])
    def test_bundle_reaches_the_command_line(self, build, monkeypatch):
        monkeypatch.setenv("ROOP_NVENC_QUALITY_FIRST", "1")
        cmd = getattr(self, build)("hevc_nvenc")
        assert self._after(cmd, "-preset") == "p6"
        assert self._after(cmd, "-multipass") == "fullres"
        assert self._after(cmd, "-spatial-aq") == "1"
        assert self._after(cmd, "-temporal-aq") == "1"

    @pytest.mark.parametrize("build", ["_ffmpeg_writer_cmd", "_video_stream_cmd"])
    def test_software_encode_is_untouched_by_the_bundle(self, build, monkeypatch):
        monkeypatch.setenv("ROOP_NVENC_QUALITY_FIRST", "1")
        cmd = getattr(self, build)("libx265")
        for flag in ("-multipass", "-spatial-aq", "-temporal-aq"):
            assert flag not in cmd


# --------------------------------------------------------------------------------------------------------------
# Hardware: does each kwarg change the bitstream?
# --------------------------------------------------------------------------------------------------------------
def _nvenc_or_skip():
    try:
        import torch
        import PyNvVideoCodec as nvc
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available - NVENC bitstream effect UNVERIFIED")
        return nvc, torch
    except ImportError as exc:                                # pragma: no cover
        pytest.skip(f"PyNvVideoCodec/torch missing ({exc}) - NVENC bitstream effect UNVERIFIED")


def _textured_nv12(torch, n=24, w=640, h=360):
    """Frames AQ must treat differently: a flat gradient (where AQ spends bits), noisy texture (where it saves
    them), and motion (so temporal AQ has something to look at)."""
    from roop import video_io_hw as hw
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:h, 0:w]
    base = (60 + 0.15 * xx + 0.1 * yy).astype(np.float32)
    out = []
    for i in range(n):
        f = np.repeat(base[:, :, None], 3, axis=2).copy()
        f[:, w // 2:, :] += rng.normal(0, 14, size=(h, w - w // 2, 1)).astype(np.float32)
        x0 = 20 + 9 * i
        f[100:220, x0:x0 + 90, :] = 200
        t = torch.from_numpy(np.clip(f, 0, 255).astype(np.uint8)).cuda().permute(2, 0, 1).contiguous()
        out.append(hw.rgb_to_nv12(t, hw.default_matrix(w, h)))
    torch.cuda.synchronize()
    return out


def _encode_digest(nvc, frames, w=640, h=360, **extra):
    base = dict(codec="hevc", gpu_id=0, fps=30, gop=30, preset="P6", rc="vbr", bitrate=1_500_000, maxbitrate=3_000_000)
    enc = nvc.CreateEncoder(w, h, "NV12", False, **{**base, **extra})
    digest = hashlib.sha1()
    for i, buf in enumerate(frames):
        p = nvc.NV_ENC_PIC_PARAMS()
        p.inputTimeStamp = i
        for pk in enc.Encode(buf, p):
            digest.update(bytes(pk["data"]))
    for pk in enc.EndEncode():
        digest.update(bytes(pk["data"]))
    return digest.hexdigest()


@pytest.fixture(scope="module")
def setup():
    nvc, torch = _nvenc_or_skip()
    frames = _textured_nv12(torch)
    return nvc, frames, _encode_digest(nvc, frames)


@pytest.mark.gpu
class TestHardwareBitstream:
    @pytest.mark.parametrize("kwargs", [{"multipass": "fullres"}, {"aq": 1}, {"temporalaq": 1}])
    def test_proven_spellings_change_the_bitstream(self, setup, kwargs):
        nvc, frames, baseline = setup
        assert _encode_digest(nvc, frames, **kwargs) != baseline, f"{kwargs} did not change the bitstream"

    @pytest.mark.parametrize("kwargs", [{"enableAQ": 1}, {"enableTemporalAQ": 1}, {"spatial_aq": 1},
                                         {"temporal_aq": 1}, {"totallybogus": 1}, {"multipass": "bogus"}])
    def test_ignored_spellings_are_byte_identical(self, setup, kwargs):
        """The trap. If a future PyNvVideoCodec starts honouring one of these, this fails: update nvenc_quality."""
        nvc, frames, baseline = setup
        assert _encode_digest(nvc, frames, **kwargs) == baseline, f"{kwargs} now changes the bitstream"

    def test_nv_zero_copy_writer_passes_the_options_through(self, setup, monkeypatch, tmp_path):
        """The writer must forward them: spy on CreateEncoder instead of trusting the call site."""
        nvc, _, _ = setup
        from roop import video_io_hw as hw
        monkeypatch.setenv("ROOP_NVENC_QUALITY_FIRST", "1")
        seen = {}
        real = nvc.CreateEncoder

        def spy(*a, **kw):
            seen.update(kw)
            return real(*a, **kw)

        with mock.patch.object(nvc, "CreateEncoder", spy):
            w = hw.NvZeroCopyWriter(str(tmp_path / "q.mp4"), 640, 360, 25.0, codec="hevc")
            info = w.close()
        assert seen["preset"] == "P6" and seen["multipass"] == "fullres"
        assert seen["aq"] == 1 and seen["temporalaq"] == 1
        assert "enableAQ" not in seen and "enableTemporalAQ" not in seen
        assert info["nvenc_quality"] == "p6+multipass=fullres+spatialAQ+temporalAQ"


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())
