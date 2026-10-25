"""roop/video_io_hw.py: zero-copy NVDEC/NVENC, true-IDR keyframe decode, the QSV rung, the fallbacks, the telemetry.

Two classes. ``PureLogic`` needs no GPU and runs in the light CI profile (it uses the bundled ffmpeg for the
software rungs and skips if there is none). ``HardwarePaths`` drives a real NVDEC/NVENC and is marked ``gpu``; it
skips, with the reason, when PyNvVideoCodec / CUDA are absent - a skip is "unverified", not "passed".

Clips are generated here (testsrc2 + sine, x264 with open-GOP) so nothing depends on a user's footage. The open-GOP
clip is the regression for the real defect: its sync samples after the first are NAL type 1, not IDR, and decoding
them as keyframes returned the wrong picture (Love.mp4: 3 of 34 frames up to 232/255 off).

QSV: ``IntelQsvReader`` is covered for its probe, its fallback and its command line. A QSV DECODE is not tested -
the bundled ffmpeg has no ``*_qsv`` decoder and the machine has no Intel iGPU in use - so it is UNVERIFIED.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import video_io_hw as hw      # noqa: E402  (numpy + roop.degrade only; torch is imported lazily)

try:
    from roop.ffmpeg_writer import FFMPEG_BINARY
    _FFMPEG = str(FFMPEG_BINARY)
except Exception:                       # pragma: no cover - no ffmpeg
    _FFMPEG = ""

_TMP = None
_CLIPS = {}
# <PINOKIO_HOME>/roop-keep/Love.mp4: this repo is <PINOKIO_HOME>/api/roop-ultimate. Optional; the test skips without it.
LOVE = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "roop-keep", "Love.mp4")


def _ffmpeg_ok():
    return bool(_FFMPEG) and (os.path.exists(_FFMPEG) or shutil.which(_FFMPEG))


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, timeout=180)


def _gen_clip(name, *, open_gop, audio=True, size="640x360"):
    path = os.path.join(_TMP.name, name)
    cmd = [_FFMPEG, "-hide_banner", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
    cmd += ["-t", "6", "-c:v", "libx264", "-g", "30", "-bf", "2", "-pix_fmt", "yuv420p", "-x264-params",
            "keyint=30:min-keyint=30:scenecut=0" + (":open-gop=1" if open_gop else "")]
    cmd += ["-c:a", "aac", "-shortest"] if audio else []
    cmd.append(path)
    proc = _run(cmd)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")[-400:]
    return path


def _probe(path, entries, stream="v:0"):
    out = subprocess.check_output([hw._ffprobe_binary(), "-v", "error", "-select_streams", stream, "-count_frames",
                                   "-show_entries", f"stream={entries}", "-of", "json", path], timeout=60)
    return (json.loads(out).get("streams") or [{}])[0]


def setUpModule():
    global _TMP
    _TMP = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    if _ffmpeg_ok():
        _CLIPS["idr"] = _gen_clip("idr.mp4", open_gop=False)
        _CLIPS["open"] = _gen_clip("open.mp4", open_gop=True)


def tearDownModule():
    if _TMP is not None:
        _TMP.cleanup()


def _need_clips(test):
    return unittest.skipUnless(_ffmpeg_ok(), "no ffmpeg to generate the test clips")(test)


class PureLogic(unittest.TestCase):
    def setUp(self):
        hw.telemetry.reset()

    # ---- the NAL classifier ------------------------------------------------------------------------------
    def test_h264_idr_is_accepted_and_non_idr_sync_sample_is_not(self):
        sps_pps_idr = b"\x00\x00\x00\x01\x06xx" + b"\x00\x00\x00\x01\x67aa" + b"\x00\x00\x00\x01\x68bb" + \
                      b"\x00\x00\x01\x65" + b"\x88" * 40
        sei_nonidr = b"\x00\x00\x00\x01\x06xx" + b"\x00\x00\x01\x41" + b"\x9a" * 40      # NAL 1, flagged key
        self.assertTrue(hw.is_random_access_packet(sps_pps_idr, "h264"))
        self.assertFalse(hw.is_random_access_packet(sei_nonidr, "h264"))

    def test_hevc_idr_accepted_cra_and_trail_refused(self):
        def nal(t):
            return b"\x00\x00\x01" + bytes([(t << 1) & 0x7E, 1]) + b"\x80" * 20
        self.assertTrue(hw.is_random_access_packet(nal(19), "hevc"))
        self.assertTrue(hw.is_random_access_packet(nal(20), "hevc"))
        self.assertFalse(hw.is_random_access_packet(nal(21), "hevc"), "CRA can carry leading pictures")
        self.assertFalse(hw.is_random_access_packet(nal(1), "hevc"))

    def test_unclassified_codecs_defer_to_the_container_flag(self):
        self.assertIsNone(hw.is_random_access_packet(b"\x00\x00\x01\x65", "av1"))
        self.assertIsNone(hw.is_random_access_packet(b"", "vp9"))

    # ---- telemetry -----------------------------------------------------------------------------------------
    def test_telemetry_counts_what_it_is_told_and_flat_fields_are_coarse(self):
        t = hw.telemetry
        t.register_decoder("stage1", hw.BACKEND_NVDEC_DLPACK)
        t.register_decoder("stage3", hw.BACKEND_NVDEC_DLPACK)
        t.register_encoder("encode", hw.BACKEND_NVENC_DIRECT)
        for wait in (0.0, 0.0, 0.0, 0.01):
            t.note_read(wait)
        t.note_decoded(6 * 1024 * 1024, device=True)
        t.note_decoded(6 * 1024 * 1024, device=False)       # a host frame saves nothing
        t.note_encoded(6 * 1024 * 1024, device=True)
        t.note_moved(1000)
        s = t.snapshot()
        self.assertEqual(s["decode_backend"], "NVDEC-DLPack", "one backend across stages is not repeated")
        self.assertEqual(s["encode_backend"], "NVENC-Direct")
        self.assertEqual(s["pcie_saved_bytes"], 12 * 1024 * 1024)
        self.assertEqual(s["pcie_moved_bytes"], 1000)
        self.assertTrue(s["pcie_saved_is_estimate"])
        self.assertEqual(s["decode_queue_saturation"], 0.75)
        flat = t.flat()
        self.assertEqual(flat, {"io_decode": "NVDEC-DLPack", "io_encode": "NVENC-Direct", "io_pcie_saved_mb": 12,
                                "io_queue_sat": 75})

    def test_mixed_backends_are_all_named_and_idle_is_empty_not_zero_saturation(self):
        t = hw.telemetry
        self.assertEqual(t.flat(), {"io_decode": "", "io_encode": "", "io_pcie_saved_mb": 0, "io_queue_sat": None})
        t.register_decoder("stage1", hw.BACKEND_CPU)
        t.register_decoder("stage3", hw.BACKEND_NVDEC_PIPE)
        self.assertEqual(t.snapshot()["decode_backend"], "CPU-Fallback + NVDEC-Pipe")

    def test_encoder_label_follows_the_codec_name(self):
        self.assertEqual(hw.encoder_label("h264_nvenc"), hw.BACKEND_NVENC_PIPE)
        self.assertEqual(hw.encoder_label("libx265"), hw.BACKEND_CPU)
        self.assertEqual(hw.encoder_label("hevc_qsv"), hw.BACKEND_QSV)

    def test_the_decoder_the_pipeline_picks_is_what_gets_reported(self):
        """wrap_capture registers the backend it actually returned (this is the label the HUD shows for a render)."""
        from roop import nvdec_reader
        cap = mock.Mock()
        with mock.patch.dict(os.environ, {"ROOP_NVDEC": "0"}):
            self.assertIs(nvdec_reader.wrap_capture(cap, "x.mp4", 640, 360, 25.0, tag="stage3"), cap)
        self.assertEqual(hw.telemetry.snapshot()["decoders"], {"stage3": hw.BACKEND_CPU})

    # ---- the runtime report and the telemetry frame ---------------------------------------------------------
    def test_runtime_report_hardware_section_carries_video_io(self):
        from roop import runtime_state
        hw.telemetry.register_decoder("stage3", hw.BACKEND_NVDEC_PIPE)
        hw.telemetry.register_encoder("encode", hw.BACKEND_NVENC_PIPE)
        value = runtime_state._video_io()
        self.assertEqual(value["decode_backend"], "NVDEC-Pipe")
        self.assertEqual(value["encode_backend"], "NVENC-Pipe")
        self.assertIn("capabilities", value)
        # and the section really includes it (the consumer, not just the producer)
        src = open(os.path.join(APP, "roop", "runtime_state.py"), encoding="utf-8").read()
        self.assertIn('"video_io": _video_io()', src)

    def test_job_report_names_the_backend(self):
        from roop import stage_telemetry
        hw.telemetry.register_decoder("stage3", hw.BACKEND_NVDEC_DLPACK)
        report = stage_telemetry.build_summary()
        self.assertEqual(report["video_io"]["decode_backend"], "NVDEC-DLPack")

    def test_the_websocket_frame_carries_the_flat_fields_and_a_new_job_resets_them(self):
        src = open(os.path.join(APP, "api.py"), encoding="utf-8").read()
        self.assertIn("**_video_io_flat(),", src)
        self.assertIn("_video_io_reset()", src)
        js = open(os.path.join(ROOT, "react-ui", "src", "store", "telemetryStore.js"), encoding="utf-8").read()
        for key in ("io_decode", "io_encode", "io_pcie_saved_mb", "io_queue_sat"):
            self.assertIn(key, js)

    def test_hud_text_is_built_by_the_store_helper(self):
        node = shutil.which("node") or os.path.join(os.path.dirname(os.path.dirname(ROOT)),
                                                    "bin", "miniforge", "node.exe")      # <PINOKIO_HOME>/bin
        if not os.path.exists(node):
            self.skipTest("no node to run the store helper")
        store = os.path.join(ROOT, "react-ui", "src", "store", "telemetryStore.js")
        script = (f"import {{ videoIoTextOf }} from {json.dumps('file:///' + store.replace(os.sep, '/'))};"
                  "console.log(JSON.stringify([videoIoTextOf({}),"
                  "videoIoTextOf({io_decode:'NVDEC-DLPack',io_encode:'NVENC-Direct',io_pcie_saved_mb:2480,io_queue_sat:74}),"
                  "videoIoTextOf({io_decode:'CPU-Fallback',io_encode:'',io_pcie_saved_mb:0,io_queue_sat:null})]))")
        out = subprocess.run([node, "--input-type=module", "-e", script], capture_output=True, timeout=60,
                             cwd=os.path.join(ROOT, "react-ui"))
        if out.returncode != 0:
            self.skipTest("node could not import the store: " + out.stderr.decode("utf-8", "replace")[-200:])
        none, full, cpu = json.loads(out.stdout.decode().strip().splitlines()[-1])
        self.assertEqual(none, "")
        self.assertEqual(full, "Video I/O: NVDEC-DLPack → NVENC-Direct · PCIe saved ~2480 MB (est.) · decode queue 74%")
        self.assertEqual(cpu, "Video I/O: CPU-Fallback → —")

    # ---- QSV ----------------------------------------------------------------------------------------------
    def test_qsv_command_line_and_filter(self):
        cmd = hw.FfmpegPipeReader.build_command("in.mp4", "h264_qsv", True)
        self.assertEqual(cmd[cmd.index("-c:v") + 1], "h264_qsv")
        self.assertLess(cmd.index("-skip_frame"), cmd.index("-i"), "skip_frame is an INPUT option")
        self.assertIn("select=key", cmd)
        self.assertEqual(hw.qsv_decoder_name("hevc"), "hevc_qsv")
        self.assertIsNone(hw.qsv_decoder_name("prores"))

    @_need_clips
    def test_qsv_unavailable_here_is_reported_and_not_faked(self):
        with mock.patch.dict(os.environ, {"ROOP_QSV": "1"}):
            hw._qsv_cache.clear()
            if hw.qsv_probe(_CLIPS["idr"], "h264"):
                self.skipTest("this machine HAS a working QSV decoder; the hardware test is out of scope here")
            with self.assertRaises(hw.HwUnavailable):
                hw.IntelQsvReader(_CLIPS["idr"])

    # ---- the fallback ladders ------------------------------------------------------------------------------
    @_need_clips
    def test_keyframe_ladder_ends_at_software_ffmpeg_with_exact_indices(self):
        with mock.patch.object(hw, "_load_nvc", side_effect=hw.HwUnavailable("no nvc (test)")), \
                mock.patch.dict(os.environ, {"ROOP_QSV": "0"}):
            with hw.open_keyframe_reader(_CLIPS["open"]) as k:
                self.assertEqual(k.backend, hw.BACKEND_CPU)
                got = [f.index for f in k.keyframes_iter()]
        truth = hw.keyframe_display_indices(_CLIPS["open"])
        self.assertEqual(got, truth, "emission order is the decoder's, and the labels follow it")
        self.assertNotIn(-1, got, "every emitted keyframe must carry an exact index")
        self.assertIn(0, got)
        self.assertTrue(all(i % 30 == 0 for i in got), got)
        self.assertLess(len(got), 150)
        self.assertTrue(any("software ffmpeg" in n for n in hw.snapshot()["fallbacks"]))

    @_need_clips
    def test_container_flags_overcount_the_open_gop_clip_so_indices_come_from_the_decoder(self):
        """ffprobe flags 5 packets as sync samples; libavcodec emits fewer key pictures. The reader trusts the latter."""
        out = subprocess.check_output([hw._ffprobe_binary(), "-v", "error", "-select_streams", "v:0", "-show_entries",
                                       "packet=flags", "-of", "csv=p=0", _CLIPS["open"]])
        flagged = sum(1 for line in out.decode().split() if "K" in line)
        self.assertEqual(flagged, 5)
        self.assertLess(len(hw.keyframe_display_indices(_CLIPS["open"])), flagged)

    @_need_clips
    def test_main_ladder_ends_at_the_existing_pipeline_and_reads_every_frame(self):
        with mock.patch.object(hw, "_load_nvc", side_effect=hw.HwUnavailable("no nvc (test)")):
            r = hw.open_reader(_CLIPS["idr"], tag="stage3")
            self.assertIsInstance(r, hw.LegacyCaptureReader)
            frames = list(r)
            r.close()
        self.assertEqual(len(frames), 150)
        self.assertEqual(frames[0].bgr().shape, (360, 640, 3))
        self.assertIn(hw.snapshot()["decoders"]["stage3"], (hw.BACKEND_NVDEC_PIPE, hw.BACKEND_CPU))
        self.assertTrue(any("existing pipeline" in n for n in hw.snapshot()["fallbacks"]))

    @_need_clips
    def test_roop_hw_io_zero_forces_the_fallback_even_with_the_package(self):
        with mock.patch.dict(os.environ, {"ROOP_HW_IO": "0"}):
            self.assertIsInstance(hw.open_reader(_CLIPS["idr"]), hw.LegacyCaptureReader)
            with hw.open_keyframe_reader(_CLIPS["idr"]) as k:
                self.assertEqual(k.backend, hw.BACKEND_CPU)

    def test_open_writer_falls_back_for_odd_dimensions_and_for_a_missing_package(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            if not _ffmpeg_ok():
                self.skipTest("no ffmpeg")
            with mock.patch.object(hw, "_load_nvc", side_effect=hw.HwUnavailable("no nvc (test)")):
                w = hw.open_writer(os.path.join(d, "a.mp4"), 320, 180, 25.0)
                self.assertIsInstance(w, hw.CpuFallbackWriter)
                for _ in range(3):
                    w.write(np.zeros((180, 320, 3), np.uint8))
                info = w.close()
            self.assertEqual(info["frames"], 3)
            self.assertEqual(hw.snapshot()["encoders"]["encode"], hw.BACKEND_CPU)

    def test_trim_seconds_become_a_half_open_frame_window(self):
        self.assertEqual(hw.trim_range(1.0, 2.0, 25.0), (25, 50))
        self.assertEqual(hw.trim_range(None, 2.0, 25.0), (0, 50))
        self.assertEqual(hw.trim_range(1.0, None, 25.0), (25, None))
        self.assertEqual(hw.trim_range(1.0, None, 25.0, total=150), (25, 150))
        self.assertEqual(hw.trim_range(1.0, 99.0, 25.0, total=150), (25, 150), "trim_out past the end clamps")
        self.assertEqual(hw.trim_range(100.0, None, 25.0, total=150), (150, 150), "trim_in past the end is empty")
        self.assertEqual(hw.timecode_to_frame(1.001, 30000 / 1001), 30)
        with self.assertRaises(ValueError):
            hw.trim_range(3.0, 1.0, 25.0)
        with self.assertRaises(ValueError):
            hw.timecode_to_frame(1.0, 0)

    @_need_clips
    def test_the_fallback_rung_honours_the_same_window_as_the_hardware_rung(self):
        """LegacyCaptureReader is what every machine without NVDEC reads through: same [start, end) contract."""
        with mock.patch.dict(os.environ, {"ROOP_HW_IO": "0"}):
            r = hw.open_reader(_CLIPS["idr"], start_frame=40, end_frame=50)
            try:
                got = list(r)
            finally:
                r.close()
            self.assertEqual([f.index for f in got], list(range(40, 50)))
            r = hw.open_reader(_CLIPS["idr"], trim_in=2.0, trim_out=2.4)          # 25 fps -> frames 50..59
            try:
                self.assertEqual([f.index for f in r], list(range(50, 60)))
            finally:
                r.close()

    def test_matrix_rule_matches_what_a_decoder_assumes(self):
        self.assertEqual(hw.default_matrix(1920, 1080), hw._MATRIX_BT709)
        self.assertEqual(hw.default_matrix(1280, 720), hw._MATRIX_BT709)
        self.assertEqual(hw.default_matrix(640, 360), hw._MATRIX_BT601)

    @_need_clips
    def test_background_keyframe_scan_streams_frames_and_surfaces_errors(self):
        with mock.patch.object(hw, "_load_nvc", side_effect=hw.HwUnavailable("no nvc (test)")), \
                mock.patch.dict(os.environ, {"ROOP_QSV": "0"}):
            scan = hw.BackgroundKeyframeScan(_CLIPS["open"], depth=2)
            try:
                indices = [f.index for f in scan]
            finally:
                scan.close()
        self.assertEqual(sorted(indices), sorted(hw.keyframe_display_indices(_CLIPS["open"])))


@pytest.mark.gpu
class HardwarePaths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.nvc, cls.torch = hw._load_nvc()
        except hw.HwUnavailable as exc:
            raise unittest.SkipTest(f"UNVERIFIED - no PyNvVideoCodec/CUDA here: {exc}")
        if not _ffmpeg_ok():
            raise unittest.SkipTest("no ffmpeg to generate the test clips")

    def setUp(self):
        hw.telemetry.reset()

    def _psnr(self, a, b):
        mse = float(((a.float() - b.float()) ** 2).mean())
        return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)

    # ---- the reader ----------------------------------------------------------------------------------------
    def test_reader_yields_a_cuda0_chw_uint8_tensor(self):
        clip = _CLIPS["idr"]
        meta = _probe(clip, "width,height,nb_read_frames")
        with hw.NvZeroCopyReader(clip, buffer_size=16) as r:
            f = r.read()
            t = f.tensor
            self.assertEqual(tuple(t.shape), (3, 360, 640))
            self.assertEqual(t.dtype, self.torch.uint8)
            self.assertEqual(str(t.device), "cuda:0")
            self.assertTrue(t.is_contiguous())
            n = 1 + sum(1 for _ in r)
        self.assertEqual(n, int(meta["nb_read_frames"]))
        s = hw.snapshot()
        self.assertEqual(s["decode_backend"], hw.BACKEND_NVDEC_DLPACK)
        self.assertEqual(s["pcie_moved_bytes"], 0, "reading as tensors must not touch the host")
        self.assertEqual(s["pcie_saved_bytes"], n * 360 * 640 * 3)

    def test_default_buffer_is_sixteen_and_is_configurable(self):
        import inspect
        self.assertEqual(inspect.signature(hw.NvZeroCopyReader).parameters["buffer_size"].default, 16)
        with hw.NvZeroCopyReader(_CLIPS["idr"], buffer_size=4) as r:
            self.assertEqual(r.buffer_size, 4)
            self.assertEqual(len(r.read_batch(4)), 4)
            with self.assertRaises(Exception):
                r._dec.get_batch_frames(8)      # the decoder itself caps a batch at buffer_size

    def test_colours_are_rgb_in_order_and_match_cv2(self):
        import cv2
        clip = _CLIPS["idr"]
        cap = cv2.VideoCapture(clip)
        ok, bgr = cap.read()
        cap.release()
        self.assertTrue(ok)
        with hw.NvZeroCopyReader(clip) as r:
            f = r.read()
            mine = f.bgr()
            swapped = mine[..., ::-1]
        mad = float(np.abs(mine.astype(int) - bgr.astype(int)).mean())
        mad_swapped = float(np.abs(swapped.astype(int) - bgr.astype(int)).mean())
        self.assertLess(mad, 4.0, f"decode differs from cv2 by {mad:.2f}/255")
        self.assertGreater(mad_swapped, mad * 3, "a channel-order bug would look like this")
        self.assertEqual(hw.snapshot()["pcie_moved_bytes"], mine.nbytes, "bgr() is the one counted host copy")

    def test_a_held_tensor_goes_stale_loudly_and_detach_keeps_it(self):
        with hw.NvZeroCopyReader(_CLIPS["idr"], buffer_size=16) as r:
            first = r.read()
            kept = first.detach()
            snapshot = kept.tensor.clone()
            for _ in range(10):
                r.read()
            with self.assertRaises(hw.StaleFrameError):
                first.tensor
            self.assertTrue(bool(self.torch.equal(kept.tensor, snapshot)), "a detached copy must survive")

    def test_the_recycling_the_guard_exists_for_is_real(self):
        """If this ever fails the decoder stopped recycling surfaces and StaleFrameError is over-cautious."""
        with hw.NvZeroCopyReader(_CLIPS["idr"], buffer_size=16) as r:
            held = []
            for _ in range(20):
                f = r.read()
                held.append((f.tensor, f.tensor.clone()))
            # compared while the reader is still open: after close() a raw view points at FREED device memory
            # (an illegal-memory-access that poisons the whole CUDA context - found by running this test).
            changed = sum(1 for live, copy in held if not bool(self.torch.equal(live, copy)))
        self.assertGreater(changed, 0)

    def test_start_frame_seek_lands_on_the_requested_frame(self):
        with hw.NvZeroCopyReader(_CLIPS["idr"]) as full:
            for _ in range(40):
                full.read()
            truth = full.read().tensor.clone()
        with hw.NvZeroCopyReader(_CLIPS["idr"], start_frame=40) as r:
            got = r.read()
            self.assertEqual(got.index, 40)
            self.assertTrue(bool(self.torch.equal(got.tensor, truth)), "start_frame=40 did not land on frame 40")

    def test_trim_window_is_bit_exact_and_hands_out_nothing_past_trim_out(self):
        """[trim_in, trim_out) = [45, 50): frame 45 sits 15 pictures after the preceding IDR (30), so the decoder must
        decode 30..44 and DISCARD them; nothing after 49 may be handed out (ThreadedDecoder reads ahead inside its own
        2-surface ring, which the reader never exposes - the counter below is frames handed out)."""
        with hw.NvZeroCopyReader(_CLIPS["idr"]) as full:
            truth = {f.index: f.tensor.clone() for f in full if 45 <= f.index < 50}
        hw.telemetry.reset()
        with hw.NvZeroCopyReader(_CLIPS["idr"], start_frame=45, end_frame=50, buffer_size=16) as r:
            got = []
            while True:
                batch = r.read_batch(3)
                if not batch:
                    break
                got.extend((f.index, f.tensor.clone()) for f in batch)
            self.assertIsNone(r.read(), "read() past trim_out")
        self.assertEqual([i for i, _ in got], [45, 46, 47, 48, 49])
        for index, tensor in got:
            self.assertTrue(bool(self.torch.equal(tensor, truth[index])), f"frame {index} differs from a full decode")
        self.assertEqual(hw.snapshot()["pcie_saved_bytes"], 5 * 360 * 640 * 3,
                         "a post-trim frame was handed out")

    def test_the_first_batch_after_open_is_exact(self):
        """Regression: the frame fetched at open is recycled by the decoder call that completes the batch, so
        `read_batch` on a fresh reader returned a wrong first frame (single `read()` and later batches were fine).
        Found by the trim-window test; reproduced at start_frame 0, 30, 45 before the fix."""
        with hw.NvZeroCopyReader(_CLIPS["idr"]) as full:
            truth = [f.tensor.clone() for f, _ in zip(full, range(8))]
        for start in (0, 30, 45):
            with hw.NvZeroCopyReader(_CLIPS["idr"], start_frame=start) as r:
                batch = r.read_batch(4)
                self.assertEqual([f.index for f in batch], [start + i for i in range(4)])
                got = [f.tensor.clone() for f in batch]
            if start == 0:
                for i in range(4):
                    self.assertTrue(bool(self.torch.equal(got[i], truth[i])), f"first batch, frame {i}")
            else:
                with hw.NvZeroCopyReader(_CLIPS["idr"]) as full:
                    want = {f.index: f.tensor.clone() for f in full if start <= f.index < start + 4}
                for f, t in zip(batch, got):
                    self.assertTrue(bool(self.torch.equal(t, want[f.index])), f"start {start}, frame {f.index}")

    def test_an_inverted_window_is_refused(self):
        with self.assertRaises(ValueError):
            hw.NvZeroCopyReader(_CLIPS["idr"], start_frame=50, end_frame=40)

    # ---- sparse keyframes ----------------------------------------------------------------------------------
    def _full_decode(self, clip, wanted):
        out = {}
        with hw.NvZeroCopyReader(clip) as r:
            for f in r:
                if f.index in wanted:
                    out[f.index] = f.tensor.clone()
        return out

    def test_sparse_reader_decodes_only_idr_and_matches_a_full_decode_bit_for_bit(self):
        for name in ("idr", "open"):
            with self.subTest(clip=name):
                clip = _CLIPS[name]
                hw.telemetry.reset()
                with hw.NvSparseKeyframeReader(clip) as k:
                    table = k.keyframe_table()
                    frames = list(k.keyframes())
                self.assertEqual([f.index for f in frames], [i for i, _ in table])
                if name == "open":
                    self.assertEqual(len(table), 1, "only the first sync sample of the open-GOP clip is an IDR")
                else:
                    self.assertEqual(len(table), 5)
                self.assertEqual(hw.snapshot()["frames_decoded"], len(table), "decoded ONLY the keyframes")
                truth = self._full_decode(clip, {i for i, _ in table})
                for f in frames:
                    self.assertTrue(bool(self.torch.equal(f.tensor, truth[f.index])), f"{name} frame {f.index}")
                self.assertEqual(hw.snapshot()["pcie_moved_bytes"], 0)

    @unittest.skipUnless(os.path.exists(LOVE), "Love.mp4 (the footage where the defect was found) is not here")
    def test_the_container_key_flag_alone_decodes_the_wrong_pictures_on_love_mp4(self):
        """The defect the NAL filter fixes, on the clip that showed it: 36 sync samples, 29 of them IDR. Feeding all
        36 to the decoder returned 6 pictures that differ from a full decode (up to 232/255); the IDR filter, 0."""
        nvc, torch = self.nvc, self.torch

        def key_flag_frames():
            dmx = nvc.CreateDemuxer(LOVE)
            dec = nvc.CreateDecoder(gpuid=0, codec=dmx.GetNvCodecId(), usedevicememory=True,
                                    outputColorType=nvc.OutputColorType.RGBP,
                                    latency=nvc.DisplayDecodeLatencyType.LOW)
            flush = nvc.PacketData()
            flush.bsl = 0
            got = {}
            while True:
                p = dmx.Demux()
                if p.bsl == 0:
                    break
                if p.key:
                    for f in dec.Decode(p):
                        got[int(f.timestamp)] = torch.from_dlpack(f).clone()
            for f in dec.Decode(flush):         # drain once at the end, as the first prototype did
                got[int(f.timestamp)] = torch.from_dlpack(f).clone()
            return got

        flagged = key_flag_frames()
        with hw.NvSparseKeyframeReader(LOVE) as k:
            _, pts_all = k._scan()
            rank = {pts: i for i, pts in enumerate(sorted(pts_all))}
            table = k.keyframe_table()
            mine = {f.index: f.tensor.clone() for f in k.keyframes()}
        self.assertEqual(len(flagged), 36)
        self.assertEqual(len(table), 29)
        truth = self._full_decode(LOVE, {rank[pts] for pts in flagged})
        wrong_with_flag = [pts for pts, t in flagged.items() if not bool(torch.equal(t, truth[rank[pts]]))]
        self.assertGreater(len(wrong_with_flag), 0, "the container flag was expected to decode wrong pictures here")
        self.assertEqual(sorted(mine), sorted(i for i, _ in table))
        wrong_idr = [i for i, t in mine.items() if not bool(torch.equal(t, truth[i]))]
        self.assertEqual(wrong_idr, [], "IDR-only output must equal a full decode exactly")

    def test_software_keyframe_frames_are_the_pictures_their_index_claims(self):
        """Index alignment: each software keyframe must look like the full-decode frame at that display index."""
        import cv2
        with mock.patch.object(hw, "_load_nvc", side_effect=hw.HwUnavailable("forced")):
            pass
        with hw.FfmpegPipeReader(_CLIPS["open"], keyframes_only=True) as p:
            got = list(p)
        cap = cv2.VideoCapture(_CLIPS["open"])
        for f in got:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f.index)
            ok, truth = cap.read()
            self.assertTrue(ok)
            mad = float(np.abs(f.bgr().astype(int) - truth.astype(int)).mean())
            self.assertLess(mad, 4.0, f"keyframe labelled {f.index} differs from that frame by {mad:.1f}")
        cap.release()
        self.assertGreaterEqual(len(got), 1)

    def test_keyframe_ladder_prefers_nvdec_when_qsv_is_absent(self):
        with mock.patch.dict(os.environ, {"ROOP_QSV": "0"}):
            with hw.open_keyframe_reader(_CLIPS["idr"]) as k:
                self.assertEqual(k.backend, hw.BACKEND_NVDEC_DLPACK)
                self.assertEqual(len(list(k.keyframes_iter())), 5)

    # ---- the writer ----------------------------------------------------------------------------------------
    def _frames(self, clip, n):
        out = []
        with hw.NvZeroCopyReader(clip) as r:
            for _ in range(n):
                f = r.read()
                if f is None:
                    break
                out.append(f.tensor.clone())
        return out

    def test_writer_round_trips_h264_hevc_av1_from_vram(self):
        frames = self._frames(_CLIPS["idr"], 60)
        for codec in ("h264", "hevc", "av1"):
            with self.subTest(codec=codec):
                hw.telemetry.reset()
                path = os.path.join(_TMP.name, f"rt_{codec}.mp4")
                with hw.NvZeroCopyWriter(path, 640, 360, 25.0, codec=codec) as w:
                    for t in frames:
                        w.write(t)
                self.assertEqual(w.frames_written, 60)
                self.assertEqual(_probe(path, "nb_read_frames")["nb_read_frames"], "60")
                back = self._frames(path, 60)
                self.assertEqual(len(back), 60)
                psnr = min(self._psnr(a, b) for a, b in zip(frames, back))
                self.assertGreater(psnr, 36.0, f"{codec} round trip {psnr:.1f} dB")
                s = hw.snapshot()
                self.assertEqual(s["encoders"]["encode"], hw.BACKEND_NVENC_DIRECT)
                self.assertEqual(s["frames_encoded"], 60)

    def test_writer_colour_matrix_is_consistent_for_standard_definition(self):
        sd = _gen_clip("sd.mp4", open_gop=False, audio=False, size="320x180")
        frames = self._frames(sd, 30)
        path = os.path.join(_TMP.name, "sd_rt.mp4")
        with hw.NvZeroCopyWriter(path, 320, 180, 25.0) as w:
            self.assertEqual(w.matrix, hw._MATRIX_BT601)
            for t in frames:
                w.write(t)
        back = self._frames(path, 30)
        psnr = min(self._psnr(a, b) for a, b in zip(frames, back))
        self.assertGreater(psnr, 36.0, f"SD round trip {psnr:.1f} dB (a 601/709 mismatch costs ~3.7 dB)")

    def test_ntsc_frame_rate_is_exact(self):
        frames = self._frames(_CLIPS["idr"], 48)
        path = os.path.join(_TMP.name, "ntsc.mp4")
        with hw.NvZeroCopyWriter(path, 640, 360, 24000 / 1001) as w:
            for t in frames:
                w.write(t)
        meta = _probe(path, "r_frame_rate,nb_read_frames")
        self.assertEqual(meta["r_frame_rate"], "24000/1001")
        self.assertEqual(meta["nb_read_frames"], "48")

    def test_writer_takes_hwc_bgr_and_refuses_host_arrays(self):
        frames = self._frames(_CLIPS["idr"], 4)
        path = os.path.join(_TMP.name, "hwc.mp4")
        with hw.NvZeroCopyWriter(path, 640, 360, 25.0) as w:
            w.write(frames[0].flip(0).permute(1, 2, 0).contiguous(), layout="HWC_BGR")
            with self.assertRaises(TypeError):
                w.write(np.zeros((360, 640, 3), np.uint8))
            with self.assertRaises(ValueError):
                w.write(self.torch.zeros((3, 100, 100), dtype=self.torch.uint8, device="cuda"))
        self.assertEqual(w.frames_written, 1)

    def test_odd_dimensions_fall_back_to_the_ffmpeg_writer(self):
        w = hw.open_writer(os.path.join(_TMP.name, "odd.mp4"), 321, 181, 25.0)
        try:
            self.assertIsInstance(w, hw.CpuFallbackWriter)
        finally:
            w.close()
        self.assertTrue(any("NVENC-Direct" in n for n in hw.snapshot()["fallbacks"]))

    def test_finalize_stream_copies_the_source_audio_and_verifies(self):
        src = _CLIPS["idr"]
        frames = self._frames(src, 150)
        video_only = os.path.join(_TMP.name, "va.mp4")
        final = os.path.join(_TMP.name, "final.mp4")
        w = hw.NvZeroCopyWriter(video_only, 640, 360, 25.0)
        for t in frames:
            w.write(t)
        result = w.finalize(final, audio_source=src, expected_frames=150)
        self.assertTrue(result["ok"], result.get("problems"))
        self.assertTrue(result["temp_deleted"])
        self.assertFalse(os.path.exists(video_only), "the intermediate is deleted only after verification")
        self.assertEqual(_probe(final, "nb_read_frames")["nb_read_frames"], "150")
        self.assertEqual(_probe(final, "codec_name", stream="a:0").get("codec_name"), "aac")

    def test_decode_then_encode_in_vram_moves_nothing_over_pcie(self):
        hw.telemetry.reset()
        path = os.path.join(_TMP.name, "vram.mp4")
        with hw.NvZeroCopyReader(_CLIPS["idr"]) as r, hw.NvZeroCopyWriter(path, 640, 360, 25.0) as w:
            for f in r:
                w.write(f)
        s = hw.snapshot()
        self.assertEqual(s["pcie_moved_bytes"], 0)
        self.assertEqual(s["pcie_saved_bytes"], 2 * 150 * 640 * 360 * 3)
        self.assertEqual((s["decode_backend"], s["encode_backend"]), ("NVDEC-DLPack", "NVENC-Direct"))
        self.assertIsNotNone(s["decode_queue_saturation"])

    def test_sparse_keyframe_reader_samples_gop_midpoints_when_gop_exceeds_60(self):
        clip_path = os.path.join(_TMP.name, "long_gop.mp4")
        cmd = [_FFMPEG, "-hide_banner", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25",
               "-t", "4", "-c:v", "libx264", "-g", "75", "-pix_fmt", "yuv420p", "-x264-params", "keyint=75:min-keyint=75:scenecut=0",
               clip_path]
        proc = _run(cmd)
        self.assertEqual(proc.returncode, 0)
        with hw.NvSparseKeyframeReader(clip_path, adaptive_midpoints=True, max_gop_gap=60) as r:
            table = r.keyframe_table()
            self.assertGreaterEqual(len(table), 1)
            sample_tbl = r.sample_table(max_gop_gap=60)
            self.assertGreater(len(sample_tbl), len(table), "Adaptive midpoint frames must be sampled when GOP > 60")
            has_mid = any(not is_k for _, _, is_k in sample_tbl)
            self.assertTrue(has_mid, "Sample table must contain non-keyframe midpoint samples")
            frames = list(r.sample_frames(enable_stats=True))
            self.assertEqual(len(frames), len(sample_tbl))
            self.assertIsNotNone(frames[0].decode_stats)
            self.assertIn("qp_luma", frames[0].decode_stats)


if __name__ == "__main__":
    unittest.main()
