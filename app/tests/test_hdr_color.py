"""roop.hdr_color / roop.hdr_pipeline: the managed HDR / high-bit-depth path.

Anchors are published values (and were cross-checked against colour-science
0.4.6 and OpenColorIO 2.5.2 when this was written: every curve to machine
precision, every gamut matrix to 1e-10). The ffmpeg round trip at the bottom
skips when ffmpeg / libx265 are missing; the NVENC encoders are proven by the
render, not here.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import hdr_color as hc  # noqa: E402
from roop import hdr_pipeline as hp  # noqa: E402


def _sig(tr, lin_rgb):
    """linear (source primaries) -> legal-normalised signal"""
    return tr.encode_signal(lin_rgb)


def _planes_from_linear(tr, lin709, h, w):
    lin = lin709.reshape(-1, 3) @ hc.gamut_matrix("bt709", tr.spec.primaries).T
    ycc = _sig(tr, lin) @ tr._sig_to_ycc[:, :3].T + tr._sig_to_ycc[:, 3]
    return np.clip(np.rint(ycc), 0, 65535).astype(np.uint16).T.reshape(3, h, w)


def _scene(h=96, w=128, peak=12.0, seed=0):
    """A smooth SDR-ish picture with a specular patch at `peak` x diffuse white."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    base = np.stack([0.15 + 0.6 * xx, 0.1 + 0.5 * yy, 0.3 + 0.2 * xx * yy], axis=-1)
    base = base + rng.uniform(-0.02, 0.02, base.shape)
    base = np.clip(base, 0.0, 1.0) ** 2.4
    base[4:20, 4:40] = peak
    return base


SPECS = [
    ("pq", "bt2020", "bt2020nc"),
    ("hlg", "bt2020", "bt2020nc"),
    ("slog3", "sgamut3cine", "bt709"),
    ("clog", "cinemagamut", "bt709"),
    ("clog2", "cinemagamut", "bt709"),
    ("clog3", "cinemagamut", "bt709"),
    ("bt1886", "bt709", "bt709"),
]


class TransferAnchors(unittest.TestCase):
    def test_published_code_values(self):
        # PQ: 203 nits (BT.2408 reference white) and 1000 nits
        self.assertAlmostEqual(float(hc._pq_encode(1.0)), 0.580688881042, places=9)
        self.assertAlmostEqual(float(hc._pq_encode(1000.0 / 203.0)), 0.751827096247, places=9)
        # HLG: 75% signal is reference white
        self.assertAlmostEqual(hc.HLG_REFERENCE_WHITE, 0.264962559786, places=8)
        # 18% grey in each camera Log (Sony / Canon v1.2, CV/1023)
        self.assertAlmostEqual(float(hc._slog3_encode(0.18)), 0.410557184751, places=9)
        self.assertAlmostEqual(float(hc._clog_encode(0.18)), 0.343389649295, places=8)
        self.assertAlmostEqual(float(hc._clog2_encode(0.18)), 0.398254692561, places=8)
        self.assertAlmostEqual(float(hc._clog3_encode(0.18)), 0.343389370374, places=8)

    def test_every_curve_inverts_over_the_code_range(self):
        sig = np.linspace(-64 / 876, (1023 - 64) / 876, 2048)
        for name, prim, mat in SPECS:
            tr = hc.HdrTransform(hc.ColorSpec(transfer=name, primaries=prim, matrix=mat))
            lin = tr.decode_signal(sig)
            back = tr.encode_signal(lin)
            # PQ/HLG clamp below black, and PQ above signal 1.0 (10000 nits, the
            # curve's own ceiling); compare where each curve is defined.
            ok = lin > 0 if name in ("pq", "hlg") else np.ones_like(sig, bool)
            if name == "pq":
                ok &= sig <= 1.0
            self.assertLess(np.abs(back - sig)[ok].max(), 2e-6, name)

    def test_bt709_to_acescg_is_the_cat02_matrix(self):
        ref = np.array([[0.61308289, 0.34116705, 0.04575005],
                        [0.07000355, 0.91806283, 0.01193362],
                        [0.02049078, 0.10676399, 0.87274523]])
        np.testing.assert_allclose(hc.gamut_matrix("bt709", "ap1"), ref, atol=1e-7)
        for p in hc.PRIMARIES:
            m = hc.gamut_matrix(p, "ap1") @ hc.gamut_matrix("ap1", p)
            np.testing.assert_allclose(m, np.eye(3), atol=1e-12)


class ViewTransform(unittest.TestCase):
    def test_rolloff_is_identity_below_knee_c1_and_hits_one_at_peak(self):
        for head in (6.35, 38.4, 49.26):
            x = np.linspace(0, hc.VIEW_KNEE, 50)
            np.testing.assert_allclose(hc.rolloff(x, head), x)
            self.assertAlmostEqual(float(hc.rolloff(head, head)), 1.0, places=12)
            eps = 1e-6
            slope = (hc.rolloff(hc.VIEW_KNEE + eps, head) - hc.rolloff(hc.VIEW_KNEE, head)) / eps
            self.assertAlmostEqual(float(slope), 1.0, places=4)
            y = np.linspace(0, 1, 1001)
            np.testing.assert_allclose(hc.rolloff(hc.rolloff_inverse(y, head), head), y, atol=1e-12)

    def test_sdr_source_is_an_identity_view(self):
        tr = hc.HdrTransform(hc.ColorSpec(transfer="bt1886", primaries="bt709", matrix="bt709"))
        self.assertTrue(tr.identity)
        w = np.linspace(0, 1, 256)
        np.testing.assert_allclose(tr.view_encode(tr.decode_signal(w)), w, atol=1e-12)

    def test_diffuse_white_and_grey_land_alike_for_every_source(self):
        for name, prim, mat in SPECS:
            tr = hc.HdrTransform(hc.ColorSpec(transfer=name, primaries=prim, matrix=mat))
            self.assertAlmostEqual(float(tr.view_encode(0.18) * 255), 124.8, places=1)


class FrameTransform(unittest.TestCase):
    def test_fast_path_matches_exact_maths(self):
        h, w = 96, 128
        for name, prim, mat in SPECS:
            tr = hc.HdrTransform(hc.ColorSpec(transfer=name, primaries=prim, matrix=mat))
            planes = _planes_from_linear(tr, _scene(h, w, peak=min(12.0, tr.headroom * 0.9)), h, w)
            fast = tr.to_working(planes)
            ycc = planes.reshape(3, -1).T.astype(np.float64)
            sig = ycc @ tr._ycc_to_sig[:, :3].T + tr._ycc_to_sig[:, 3]
            lin = tr.decode_signal(sig) @ tr.m_src_to_work.T
            exact = np.clip(np.rint(tr.view_encode(lin) * 255), 0, 255)[:, ::-1].reshape(h, w, 3)
            self.assertLessEqual(np.abs(fast.astype(int) - exact).max(), 1, name)

    def test_composite_keeps_untouched_pixels_bit_exact_and_carries_the_edit(self):
        h, w = 96, 128
        for name, prim, mat in SPECS:
            tr = hc.HdrTransform(hc.ColorSpec(transfer=name, primaries=prim, matrix=mat))
            planes = _planes_from_linear(tr, _scene(h, w, peak=min(12.0, tr.headroom * 0.9)), h, w)
            w_in = tr.to_working(planes)
            same, n = tr.composite(planes, w_in, w_in)
            self.assertEqual(n, 0)
            self.assertTrue(np.array_equal(same, planes), name)
            w_out = w_in.copy()
            w_out[40:80, 60:110] = 255 - w_out[40:80, 60:110]
            out, n = tr.composite(planes, w_in, w_out)
            self.assertEqual(n, int(np.any(w_in != w_out, axis=2).sum()))
            keep = np.ones((h, w), bool)
            keep[40:80, 60:110] = False
            self.assertTrue(np.array_equal(out[:, keep], planes[:, keep]), name)
            # The managed view of the result IS the pipeline's edit.
            self.assertLessEqual(np.abs(tr.to_working(out).astype(int) - w_out).max(), 1, name)

    def test_specular_highlight_survives_an_edit_next_to_it(self):
        h, w = 96, 128
        tr = hc.HdrTransform(hc.ColorSpec(transfer="pq", primaries="bt2020", matrix="bt2020nc"))
        planes = _planes_from_linear(tr, _scene(h, w, peak=12.0), h, w)
        w_in = tr.to_working(planes)
        w_out = w_in.copy()
        w_out[4:20, 30:60] //= 2        # half the patch is edited
        out, _ = tr.composite(planes, w_in, w_out)
        # the unedited half still holds ~2436 nits
        np.testing.assert_array_equal(out[:, 4:20, 4:30], planes[:, 4:20, 4:30])

    def test_blind_rebuild_inverts_the_view(self):
        tr = hc.HdrTransform(hc.ColorSpec(transfer="slog3", primaries="sgamut3cine", matrix="bt709"))
        planes = _planes_from_linear(tr, _scene(), 96, 128)
        w = tr.to_working(planes)
        self.assertLessEqual(np.abs(tr.to_working(tr.from_working(w)).astype(int) - w).max(), 1)


class Probe(unittest.TestCase):
    def setUp(self):
        self._env = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith("ROOP_HDR")}

    def tearDown(self):
        for k in list(os.environ):
            if k.startswith("ROOP_HDR"):
                del os.environ[k]
        os.environ.update(self._env)

    def test_pq_stream(self):
        s = hp._spec_from_probe(
            {"width": 3840, "height": 2160, "pix_fmt": "yuv420p10le", "color_range": "tv",
             "color_space": "bt2020nc", "color_transfer": "smpte2084", "color_primaries": "bt2020"},
            {"side_data_list": [
                {"side_data_type": "Mastering display metadata", "red_x": "34000/50000",
                 "red_y": "16000/50000", "green_x": "13250/50000", "green_y": "34500/50000",
                 "blue_x": "7500/50000", "blue_y": "3000/50000", "white_point_x": "15635/50000",
                 "white_point_y": "16450/50000", "min_luminance": "50/10000",
                 "max_luminance": "10000000/10000"},
                {"side_data_type": "Content light level metadata", "max_content": 1000, "max_average": 400}]},
            None, None)
        self.assertEqual((s.transfer, s.primaries, s.matrix, s.bit_depth, s.chroma, s.full_range),
                         ("pq", "bt2020", "bt2020nc", 10, "420", False))
        self.assertTrue(s.hdr)
        self.assertEqual(s.mastering_display,
                         "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)")
        self.assertEqual(s.content_light, "1000,400")

    def test_log_override_over_a_bt709_tag_takes_the_camera_gamut(self):
        stream = {"width": 1920, "height": 1080, "pix_fmt": "yuv422p10le", "color_range": "tv",
                  "color_space": "bt709", "color_transfer": "bt709", "color_primaries": "bt709"}
        s = hp._spec_from_probe(stream, {}, "slog3", None)
        self.assertEqual((s.transfer, s.primaries, s.chroma), ("slog3", "sgamut3cine", "422"))
        s = hp._spec_from_probe(stream, {}, "clog3", "bt2020")
        self.assertEqual((s.transfer, s.primaries), ("clog3", "bt2020"))

    def test_12bit_444_and_untagged(self):
        s = hp._spec_from_probe({"width": 4096, "height": 2160, "pix_fmt": "yuv444p12le"}, {}, None, None)
        self.assertEqual((s.bit_depth, s.chroma, s.transfer, s.primaries), (12, "444", "bt1886", "bt709"))
        self.assertTrue(s.high_bit_depth)

    def test_output_codec(self):
        saved = dict(hp._encoder_ok)
        try:
            hp._encoder_ok.update({"hevc_nvenc": True, "av1_nvenc": True})
            self.assertEqual(hp.output_codec("h264_nvenc"), "hevc_nvenc")
            self.assertEqual(hp.output_codec("libx264"), "libx265")
            self.assertEqual(hp.output_codec("av1_nvenc"), "av1_nvenc")
            # no AV1 NVENC (RTX 3060) -> HEVC; no NVENC at all -> x265
            hp._encoder_ok["av1_nvenc"] = False
            self.assertEqual(hp.output_codec("av1_nvenc"), "hevc_nvenc")
            hp._encoder_ok["hevc_nvenc"] = False
            self.assertEqual(hp.output_codec("av1_nvenc"), "libx265")
            self.assertEqual(hp.output_codec("h264_nvenc"), "libx265")
            hp._encoder_ok["hevc_nvenc"] = True
            os.environ["ROOP_HDR_CODEC"] = "libx265"
            self.assertEqual(hp.output_codec("hevc_nvenc"), "libx265")
        finally:
            hp._encoder_ok.clear()
            hp._encoder_ok.update(saved)

    def test_mode(self):
        self.assertEqual(hp.mode(), "auto")
        os.environ["ROOP_HDR"] = "off"
        self.assertEqual(hp.mode(), "off")

    def test_encoder_tags_travel_as_frame_properties(self):
        # FFmpeg 8.1 drops -color_trc / -color_primaries given as output
        # options; only setparams reaches the encoder. Guard the command.
        spec = hp._spec_from_probe(
            {"width": 1280, "height": 720, "pix_fmt": "yuv420p10le", "color_range": "tv",
             "color_space": "bt2020nc", "color_transfer": "arib-std-b67", "color_primaries": "bt2020"},
            {}, None, None)
        wr = object.__new__(hp.HdrVideoWriter)
        wr.spec, wr.width, wr.height, wr.fps = spec, 1280, 720, 25.0
        wr.quality, wr.preset, wr.output_path = 19, None, os.path.abspath("x.mp4")
        cmd = wr.command("hevc_nvenc")
        vf = cmd[cmd.index("-vf") + 1]
        self.assertIn("setparams=range=tv:color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc", vf)
        self.assertIn("format=p010le", vf)
        self.assertIn("main10", cmd)
        self.assertEqual(wr._pix_fmt("libx265"), "yuv420p10le")

    def test_software_encoder_rejects_nvenc_preset_vocabulary(self):
        spec = hp._spec_from_probe(
            {"width": 1280, "height": 720, "pix_fmt": "yuv420p10le",
             "color_range": "tv", "color_space": "bt2020nc",
             "color_transfer": "smpte2084", "color_primaries": "bt2020"},
            {}, None, None)
        wr = object.__new__(hp.HdrVideoWriter)
        wr.spec, wr.width, wr.height, wr.fps = spec, 1280, 720, 25.0
        wr.quality, wr.preset, wr.output_path = 19, None, os.path.abspath("x.mp4")
        with mock.patch.dict(os.environ, {"ROOP_ENCODER_PRESET": "p5"}, clear=False):
            cmd = wr.command("libx265")
        self.assertEqual(cmd[cmd.index("-preset") + 1], "faster")


def _ffmpeg_ok() -> bool:
    try:
        ff = hp.ffmpeg_binary()
        out = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, timeout=30)
        return b"libx265" in out.stdout
    except Exception:
        return False


@unittest.skipUnless(_ffmpeg_ok(), "ffmpeg with libx265 not available")
class RoundTrip(unittest.TestCase):
    """Reader -> edit -> writer through real ffmpeg on a synthetic PQ clip."""

    def test_pq_clip_round_trip(self):
        tmp = tempfile.mkdtemp(prefix="hdr_rt_")
        try:
            h, w, n = 96, 128, 12
            tr = hc.HdrTransform(hc.ColorSpec(transfer="pq", primaries="bt2020", matrix="bt2020nc"))
            src = os.path.join(tmp, "src.mp4")
            ff = hp.ffmpeg_binary()
            p = subprocess.Popen([ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                                  "-pix_fmt", "yuv444p16le", "-s", f"{w}x{h}", "-r", "24", "-i", "-",
                                  "-vf", "setparams=range=tv:color_primaries=bt2020:color_trc=smpte2084"
                                         ":colorspace=bt2020nc,format=yuv420p10le",
                                  "-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "lossless=1:log-level=error",
                                  src], stdin=subprocess.PIPE)
            for i in range(n):
                p.stdin.write(_planes_from_linear(tr, _scene(h, w, seed=i), h, w).tobytes())
            p.stdin.close()
            self.assertEqual(p.wait(), 0)
            spec = hp.probe(src)
            self.assertEqual((spec.transfer, spec.primaries, spec.bit_depth), ("pq", "bt2020", 10))

            out = os.path.join(tmp, "out.mp4")
            reader = hp.HdrFrameReader(src, spec, fps=24.0, start_frame=2, hwaccel="")
            class Lossless(hp.HdrVideoWriter):
                # Take the encoder's loss out of the measurement: what is left
                # is the pipeline's own error.
                def command(self, codec):
                    cmd = super().command(codec)
                    return cmd[:-1] + ["-x265-params", "lossless=1:log-level=error", cmd[-1]]

            writer = Lossless(out, w, h, 24.0, spec, src, start_frame=2,
                              codec="libx265", quality=0)
            writer._master._hwaccel = ""
            k = 0
            while True:
                ok, f = reader.read()
                if not ok:
                    break
                f[50:80, 60:100] = 255 - f[50:80, 60:100]
                writer.write_frame(f)
                k += 1
            reader.release()
            writer.close()
            self.assertEqual(k, n - 2)
            self.assertEqual(writer.stats["composited"], n - 2)
            self.assertEqual(writer.stats["misaligned"], 0)
            got = hp.probe(out)
            self.assertEqual((got.transfer, got.primaries, got.matrix, got.bit_depth),
                             ("pq", "bt2020", "bt2020nc", 10))
            a = hp.HdrFrameReader(src, spec, fps=24.0, start_frame=5, planes=True, hwaccel="")
            b = hp.HdrFrameReader(out, got, fps=24.0, start_frame=3, planes=True, hwaccel="")
            pa, pb = a.read_planes().astype(int), b.read_planes().astype(int)
            a.release(), b.release()
            # Luma outside the edit is the source's own code value (the only
            # lossy step left is 4:4:4 -> 4:2:0 chroma, which luma never sees);
            # the highlight keeps its code value.
            np.testing.assert_array_equal(pa[0, :44, :], pb[0, :44, :])
            np.testing.assert_array_equal(pa[0, :, :56], pb[0, :, :56])
            self.assertLessEqual(abs(pa[0, 6:18, 6:38].mean() - pb[0, 6:18, 6:38].mean()) / 64.0, 0.5)
            self.assertGreater(np.abs(pa[0, 55:75, 65:95] - pb[0, 55:75, 65:95]).mean() / 64.0, 20)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
