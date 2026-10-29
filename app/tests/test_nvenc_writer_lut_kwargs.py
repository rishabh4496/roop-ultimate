"""SegmentedVideoWriter forwards lut_path/lut_bake to whichever writer it opens; the NVENC one must accept them."""
import inspect
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from roop.video_stream import NVHardwareVideoWriter  # noqa: E402


def test_nvenc_writer_accepts_every_segment_writer_option():
    # The exact keys SegmentedVideoWriter._writer_options splats into the constructor.
    options = ("preset", "bitrate", "threads", "ffmpeg_params", "colorspace", "sar", "dar", "color",
               "lut_path", "lut_bake")
    params = inspect.signature(NVHardwareVideoWriter.__init__).parameters
    assert [o for o in options if o not in params] == []


def _cmd(lut_path, lut_bake):
    w = NVHardwareVideoWriter.__new__(NVHardwareVideoWriter)
    w.output_path, w.width, w.height, w.fps = os.path.join(HERE, "x.mp4"), 640, 360, 24.0
    w.audio_source, w.preset, w.bitrate, w.cq, w.threads = None, "p4", None, 19, None
    w.ffmpeg_params, w.colorspace, w.sar, w.dar = [], None, None, None
    w.lut_path, w.lut_bake = lut_path, lut_bake
    return w._command("hevc_nvenc")


@pytest.mark.parametrize("bake,expect", [(True, True), (False, False)])
def test_lut_filter_only_when_baking(tmp_path, bake, expect):
    cube = tmp_path / "grade.cube"
    cube.write_text("LUT_3D_SIZE 2\n")
    vf = _cmd(str(cube), bake)
    joined = " ".join(vf)
    assert ("lut3d=" in joined) is expect
    if expect:
        assert "interp=tetrahedral" in joined
