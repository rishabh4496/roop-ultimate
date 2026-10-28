"""Stage 17 through the real ASGI app: the request refusals, the upload-confined waveform route, and the media editor's
audio half (a real silent mp4, a real uploaded wav, the delivered file decoded and measured).

`/api/swap` and `/api/preview` only reach `_normalize_mouth_custody` after the app's own readiness and target checks, which
need loaded media; the helper is driven directly and the three entry points are pinned to call it (as test_lipsync_audio
pins ProcessMgr's gate), so a request that skips the check cannot be added quietly.
"""
from __future__ import annotations

import array
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
import wave
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)
os.environ.setdefault("ROOP_REACT_CLIENT", "1")


def _wav_bytes(seconds=1.5, tone_from=0.0, rate=16000):
    path = tempfile.mktemp(suffix=".wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / rate)) if i / rate >= tone_from else 0)
                               for i in range(int(seconds * rate))))
    with open(path, "rb") as fh:
        data = fh.read()
    os.remove(path)
    return data


def _ffmpeg():
    from roop.ffmpeg_path import ffmpeg_binary
    exe = ffmpeg_binary()
    if not exe or not (os.path.isfile(exe) or shutil.which(exe)):
        raise unittest.SkipTest("no ffmpeg")
    return exe


def _onset_s(path, threshold=3000):
    done = subprocess.run([_ffmpeg(), "-v", "error", "-i", path, "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                          capture_output=True)
    if done.returncode != 0 or not done.stdout:
        return None
    pcm = array.array("h")
    pcm.frombytes(done.stdout[: len(done.stdout) // 2 * 2])
    for i, v in enumerate(pcm):
        if abs(v) > threshold:
            return i / 16000.0
    return None


class MouthCustodyApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import api
        import api_access
        import roop.globals as roop_globals
        from fastapi.testclient import TestClient
        cls.api, cls.roop_globals = api, roop_globals
        api_access.set_policy(api_access.AccessPolicy(share=False))
        cls.client = TestClient(api.app)

    @classmethod
    def tearDownClass(cls):
        import api_access
        api_access.set_policy(None)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.out = os.path.join(self.tmp, "output")
        os.makedirs(self.out)
        self._saved_out = getattr(self.roop_globals, "output_path", None)
        self.roop_globals.output_path = self.out
        self.addCleanup(lambda: setattr(self.roop_globals, "output_path", self._saved_out))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # prepare_environment() points output_path at <cwd>/output and reads CFG, neither of which this test owns.
        patcher = mock.patch("ui.main.prepare_environment", lambda: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def upload_wav(self, **kw):
        r = self.client.post("/api/lipsync/audio/add", files=[("file", ("dub.wav", _wav_bytes(**kw), "audio/wav"))])
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["path"]

    # ── the request fields ────────────────────────────────────────────────
    def test_the_helper_refuses_a_bad_mode_a_bad_slip_and_a_missing_track_with_422(self):
        for bad in ({"mouth_custody_mode": "lipsync"}, {"audio_offset_ms": 501}, {"audio_offset_ms": "x"},
                    {"override_audio_path": os.path.join(self.tmp, "missing.wav")}):
            response = self.api._normalize_mouth_custody(dict(bad))
            self.assertIsNotNone(response, bad)
            self.assertEqual(response.status_code, 422, bad)

    def test_the_helper_accepts_and_derives_for_a_good_request(self):
        payload = {"mouth_custody_mode": "audio_driven_sync", "audio_offset_ms": -120}
        self.assertIsNone(self.api._normalize_mouth_custody(payload))
        self.assertEqual((payload["lipsync_enabled"], payload["restore_original_mouth"], payload["audio_offset_ms"]), (True, False, -120.0))

    def test_every_render_entry_point_normalises_the_request_first(self):
        with open(os.path.join(APP, "api.py"), encoding="utf-8") as fh:
            src = fh.read()
        for entry in ("def trigger_swap(", "def preview(", "def _start_existing_project("):
            body = src[src.index(entry):]
            body = body[:body.index("\n@app.", 10) if "\n@app." in body[10:] else len(body)]
            self.assertIn("_normalize_mouth_custody(", body, entry + " must validate the mouth-custody fields")
        run = src[src.index("def _run_swap("):]
        self.assertIn("_mouth_custody.normalize_payload(payload", run[:6000])

    def test_apply_lipsync_settings_puts_the_new_fields_on_globals(self):
        payload = {"mouth_custody_mode": "audio_driven_sync", "audio_offset_ms": 250, "lipsync_frame_offset": 300,
                   "override_audio_path": None}
        self.api._apply_lipsync_settings(payload)
        g = self.roop_globals
        try:
            self.assertEqual((g.mouth_custody_mode, g.audio_offset_ms, g.lipsync_frame_offset, g.override_audio_path),
                             ("audio_driven_sync", 250.0, 300, None))
        finally:
            g.mouth_custody_mode, g.audio_offset_ms, g.lipsync_frame_offset = "off", 0.0, 0
            g.lipsync_enabled = False

    # ── the telemetry frame ───────────────────────────────────────────────
    def test_the_telemetry_frame_carries_the_four_fields_idle_and_running(self):
        g = self.roop_globals
        idle = self.api._mouth_custody_flat({}, False)
        self.assertEqual(idle, {"lipsync_ms": None, "mouth_mode": "", "mouth_fallback": "", "audio_offset_ms": 0.0})
        g.mouth_custody_mode, g.mouth_custody_fallback, g.audio_offset_ms = "audio_driven_sync", "", -40.0
        try:
            dash = {"telemetry": {"latency_ms": {"lipsync": 4.234}}}
            running = self.api._mouth_custody_flat(dash, True)
            self.assertEqual(running, {"lipsync_ms": 4.23, "mouth_mode": "audio_driven_sync", "mouth_fallback": "", "audio_offset_ms": -40.0})
            g.mouth_custody_fallback = "no usable driving audio (target has no audio stream)"
            self.assertEqual(self.api._mouth_custody_flat({"telemetry": {"latency_ms": {"lipsync": None}}}, True)["lipsync_ms"], None)
            self.assertIn("no usable driving audio", self.api._mouth_custody_flat(dash, True)["mouth_fallback"])
        finally:
            g.mouth_custody_mode, g.mouth_custody_fallback, g.audio_offset_ms = "off", "", 0.0

    def test_the_real_snapshot_has_the_keys(self):
        snap = self.api._telemetry_snapshot()
        for key in ("lipsync_ms", "mouth_mode", "mouth_fallback", "audio_offset_ms", "dashboard"):
            self.assertIn(key, snap)
        self.assertEqual((snap["mouth_mode"], snap["lipsync_ms"]), ("", None), "idle is constant, so the change-only sampler stays quiet")

    # ── GET /api/audio/waveform ───────────────────────────────────────────
    def test_an_uploaded_track_has_a_waveform_and_nothing_else_does(self):
        _ffmpeg()
        path = self.upload_wav(seconds=1.5, tone_from=0.5)
        body = {}
        for _ in range(60):
            body = self.client.get("/api/audio/waveform", params={"path": path, "bins": 150}).json()
            if body.get("ready"):
                break
            time.sleep(0.25)
        self.assertTrue(body.get("ready"), body)
        self.assertAlmostEqual(body["duration"], 1.5, delta=0.15)
        import base64
        peaks = base64.b64decode(body["peaks"])
        self.assertEqual(len(peaks), body["bins"])
        quiet, loud = max(peaks[: int(len(peaks) * 0.25)]), max(peaks[int(len(peaks) * 0.6):])
        self.assertGreater(loud, quiet + 40, "silence for 0.5 s then a tone must read as quiet then loud")

        elsewhere = os.path.join(self.tmp, "elsewhere.wav")
        with open(elsewhere, "wb") as fh:
            fh.write(_wav_bytes(0.2))
        for outside in (elsewhere, os.path.join(self.api.API_TEMP, "..", "..", "config.yaml"), os.path.join(self.api.API_TEMP, "nope.wav"), ""):
            r = self.client.get("/api/audio/waveform", params={"path": outside})
            self.assertEqual(r.status_code, 404, outside)

    # ── POST /api/extras/apply: the audio half, end to end ────────────────
    def test_the_media_editor_dubs_a_silent_clip_and_slips_the_track(self):
        exe = _ffmpeg()
        clip = os.path.join(self.tmp, "silent.mp4")
        done = subprocess.run([exe, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=2",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", clip], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        track = self.upload_wav(seconds=3.0, tone_from=0.0)
        with open(clip, "rb") as fh:
            video = fh.read()

        def apply(offset):
            return self.client.post("/api/extras/apply", files=[("file", ("silent.mp4", video, "video/mp4"))],
                                    data={"override_audio_path": track, "audio_offset_ms": str(offset), "fps": "30"})

        r = apply(250)
        self.assertEqual(r.status_code, 200, r.text)
        out = r.json()["absolute_path"]
        self.assertTrue(os.path.isfile(out))
        onset = _onset_s(out)
        self.assertIsNotNone(onset, "the delivered file must carry the dub")
        self.assertAlmostEqual(onset, 0.25, delta=0.06, msg="a +250 ms slip starts the audio 250 ms after the first frame")

        r0 = apply(0)
        self.assertEqual(r0.status_code, 200, r0.text)
        self.assertAlmostEqual(_onset_s(r0.json()["absolute_path"]) or 0.0, 0.0, delta=0.06)

    def test_the_media_editor_refuses_a_bad_slip_and_a_bad_track_without_processing(self):
        for data in ({"audio_offset_ms": "900"}, {"override_audio_path": os.path.join(self.tmp, "missing.wav")}):
            r = self.client.post("/api/extras/apply", files=[("file", ("x.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, "image/png"))], data=data)
            self.assertEqual(r.status_code, 422, (data, r.text))
        self.assertEqual(os.listdir(self.out), [], "a refused request writes nothing")


if __name__ == "__main__":
    unittest.main()
