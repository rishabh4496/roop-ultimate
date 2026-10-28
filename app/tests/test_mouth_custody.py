"""Stage 17: Viseme Mouth Custody - the mode switch, the audio slip, the silent-video fallback, timestamp alignment.

Three layers, each proving something the next one relies on:

  * the pure rules in `roop.mouth_custody` (modes, slip clamp, request normalisation, fallback, mux trim arithmetic);
  * the REAL remux: a real ffmpeg cuts a real override track and the decoded OUTPUT is measured for where its audio
    starts. Arithmetic on `start_s` proves nothing about a file; the onset in the decoded samples does;
  * the distributed GOP worker hand-off (`_project_for_chunk`), which used to give every chunk the track from t = 0.

The defect this stage found is pinned by name: an override track's trim was taken from `detect_fps(<audio file>)`
(the 24.0 fallback), so a 30 fps render trimmed at frame 150 played its dub from 6.25 s instead of 5.0 s.
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
import unittest
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import mouth_custody as mc  # noqa: E402  (pure: no numpy, no ffmpeg)


class ModeAndOffsetRulesTest(unittest.TestCase):
    def test_modes_normalise_and_unknown_text_is_an_error_not_a_silent_off(self):
        self.assertEqual(mc.normalize_mode("AUDIO_DRIVEN_SYNC"), mc.AUDIO_DRIVEN_SYNC)
        self.assertEqual(mc.normalize_mode("source-mouth-restore"), mc.SOURCE_MOUTH_RESTORE)
        self.assertEqual(mc.normalize_mode(None), mc.OFF)
        self.assertEqual(mc.normalize_mode(""), mc.OFF)
        with self.assertRaises(mc.MouthCustodyError):
            mc.normalize_mode("lipsync")

    def test_the_slip_is_clamped_for_internal_use_and_refused_for_api_input(self):
        self.assertEqual(mc.clamp_offset_ms(900), 500.0)
        self.assertEqual(mc.clamp_offset_ms(-900), -500.0)
        self.assertEqual(mc.clamp_offset_ms(float("nan")), 0.0)
        self.assertEqual(mc.clamp_offset_ms("junk"), 0.0)
        self.assertEqual(mc.validate_offset_ms(-500), -500.0)
        for bad in (501, -501, "x", float("inf")):
            with self.assertRaises(mc.MouthCustodyError, msg=repr(bad)):
                mc.validate_offset_ms(bad)

    def test_audio_time_is_t_minus_offset_and_never_negative(self):
        # + offset = the sound is heard LATER, so the mouth at t matches the audio that was at t - offset.
        self.assertAlmostEqual(mc.audio_time_s(10.0, 250), 9.75)
        self.assertAlmostEqual(mc.audio_time_s(10.0, -250), 10.25)
        self.assertEqual(mc.audio_time_s(0.1, 500), 0.0)
        self.assertEqual(mc.audio_time_s(3.0, 0), 3.0)

    def test_mux_trim_turns_a_slip_that_reaches_before_the_file_into_head_silence(self):
        self.assertEqual(mc.mux_trim_s(5.0, 0), (5.0, 0.0))
        self.assertEqual(mc.mux_trim_s(5.0, 250), (4.75, 0.0))
        self.assertEqual(mc.mux_trim_s(5.0, -250), (5.25, 0.0))
        cut, lead = mc.mux_trim_s(0.0, 500)
        self.assertEqual(cut, 0.0)
        self.assertAlmostEqual(lead, 0.5)

    def test_the_visual_and_the_audible_slip_are_the_same_shift(self):
        """The mouth at output time u uses source audio at trim + u - d; so does the remux: cut_from + (u - lead)."""
        for trim, d in ((5.0, 250.0), (5.0, -250.0), (0.0, 500.0), (0.2, 500.0)):
            cut, lead = mc.mux_trim_s(trim, d)
            for u in (0.0, 0.6, 1.0):
                heard = cut + max(0.0, u - lead)                     # what the remuxed track plays at u (silence before lead)
                seen = mc.audio_time_s(trim + u, d)                   # what the mouth is driven by at u
                if u >= lead:
                    self.assertAlmostEqual(heard, seen, places=6, msg=(trim, d, u))


class NormalizePayloadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.wav = os.path.join(self.tmp.name, "dub.wav")
        open(self.wav, "wb").close()

    def run_norm(self, payload, cfg=None):
        cfg = cfg or {}
        return mc.normalize_payload(dict(payload), lambda k, d: cfg.get(k, d))

    def test_an_old_client_with_no_mode_keeps_exactly_its_old_meaning(self):
        self.assertEqual(self.run_norm({})["mouth_custody_mode"], mc.OFF)
        p = self.run_norm({"lipsync_enabled": True})
        self.assertEqual((p["mouth_custody_mode"], p["lipsync_enabled"], p["restore_original_mouth"]), (mc.AUDIO_DRIVEN_SYNC, True, False))
        p = self.run_norm({"restore_original_mouth": True})
        self.assertEqual((p["mouth_custody_mode"], p["lipsync_enabled"], p["restore_original_mouth"]), (mc.SOURCE_MOUTH_RESTORE, False, True))
        # lip-sync outranks restore, as ProcessMgr already settled that contest
        self.assertEqual(self.run_norm({"lipsync_enabled": True, "restore_original_mouth": True})["mouth_custody_mode"], mc.AUDIO_DRIVEN_SYNC)

    def test_the_config_legacy_flags_apply_only_when_the_request_is_silent_about_them(self):
        self.assertEqual(self.run_norm({}, {"lipsync_enabled": True})["mouth_custody_mode"], mc.AUDIO_DRIVEN_SYNC)
        self.assertEqual(self.run_norm({"lipsync_enabled": False}, {"lipsync_enabled": True})["mouth_custody_mode"], mc.OFF)

    def test_an_explicit_mode_overrides_the_legacy_keys_both_ways(self):
        p = self.run_norm({"mouth_custody_mode": "off", "lipsync_enabled": True, "restore_original_mouth": True})
        self.assertEqual((p["lipsync_enabled"], p["restore_original_mouth"]), (False, False))
        p = self.run_norm({"mouth_custody_mode": "source_mouth_restore", "lipsync_enabled": True})
        self.assertEqual((p["lipsync_enabled"], p["restore_original_mouth"]), (False, True))
        p = self.run_norm({"mouth_custody_mode": "audio_driven_sync"})
        self.assertEqual((p["lipsync_enabled"], p["restore_original_mouth"]), (True, False))

    def test_audio_mode_does_not_force_restore_on_so_a_render_with_sound_is_unchanged(self):
        # Mask_RealityUX and the anatomical engine read restore_original_mouth directly (see the module docstring).
        self.assertFalse(self.run_norm({"mouth_custody_mode": "audio_driven_sync"})["restore_original_mouth"])

    def test_override_audio_path_is_the_one_remux_source_and_drives_lipsync_as_an_upload(self):
        p = self.run_norm({"mouth_custody_mode": "audio_driven_sync", "override_audio_path": self.wav})
        self.assertEqual(p["override_audio_path"], self.wav)
        self.assertEqual((p["lipsync_audio_source"], p["lipsync_audio_path"]), ("upload", self.wav))
        p = self.run_norm({"mouth_custody_mode": "off", "override_audio_path": self.wav})
        self.assertEqual(p["override_audio_path"], self.wav, "a dub with lip-sync OFF still remuxes")
        self.assertEqual(self.run_norm({"mouth_custody_mode": "audio_driven_sync"})["lipsync_audio_source"], "original")

    def test_the_legacy_upload_pair_is_folded_into_override_audio_path(self):
        p = self.run_norm({"lipsync_enabled": True, "lipsync_audio_source": "upload", "lipsync_audio_path": self.wav})
        self.assertEqual(p["override_audio_path"], self.wav)

    def test_bad_requests_are_errors(self):
        for bad in ({"mouth_custody_mode": "nope"}, {"audio_offset_ms": 501},
                    {"override_audio_path": os.path.join(self.tmp.name, "missing.wav")}):
            with self.assertRaises(mc.MouthCustodyError, msg=repr(bad)):
                self.run_norm(bad)
        txt = os.path.join(self.tmp.name, "notes.txt")
        open(txt, "w").close()
        with self.assertRaises(mc.MouthCustodyError):
            self.run_norm({"override_audio_path": txt})
        mp4 = os.path.join(self.tmp.name, "take2.mp4")
        open(mp4, "wb").close()
        self.assertEqual(self.run_norm({"override_audio_path": mp4})["override_audio_path"], mp4)

    def test_normalising_twice_changes_nothing(self):
        once = self.run_norm({"mouth_custody_mode": "audio_driven_sync", "override_audio_path": self.wav, "audio_offset_ms": 120})
        self.assertEqual(mc.normalize_payload(dict(once)), once)


class SilentVideoFallbackTest(unittest.TestCase):
    def test_only_audio_driven_sync_falls_back_and_it_falls_back_to_source_mouth_restore(self):
        self.assertEqual(mc.fallback_for(mc.AUDIO_DRIVEN_SYNC, False), mc.SOURCE_MOUTH_RESTORE)
        self.assertIsNone(mc.fallback_for(mc.AUDIO_DRIVEN_SYNC, True))
        self.assertIsNone(mc.fallback_for(mc.SOURCE_MOUTH_RESTORE, False))
        self.assertIsNone(mc.fallback_for(mc.OFF, False))

    def test_processmgr_decides_its_fallback_through_that_function(self):
        with open(os.path.join(APP, "roop", "ProcessMgr.py"), encoding="utf-8") as fh:
            src = fh.read()
        i = src.index("_mouth_fallback = mouth_custody.fallback_for(")
        self.assertIn("_lipsync_audio', None) is not None) is not None", src[i:i + 400])
        # ...and the three places that consult the restore path all consult the fallback too.
        for needle in ("or _mouth_fallback)", "and not lipsync_wins and not _mouth_fallback"):
            self.assertIn(needle, src)


# ── the real remux ───────────────────────────────────────────────────────────────────────────────────────

def _ffmpeg():
    from roop.ffmpeg_path import ffmpeg_binary
    exe = ffmpeg_binary()
    if not exe or not (os.path.isfile(exe) or shutil.which(exe)):
        raise unittest.SkipTest("no ffmpeg")
    return exe


def _run(args):
    done = subprocess.run([_ffmpeg(), "-v", "error", "-y"] + list(args), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def _write_wav(path, seconds, *, tone_from, rate=16000):
    """Silence until ``tone_from`` seconds, then a loud 440 Hz tone: the onset is a timestamp you can measure."""
    frames = bytearray()
    for i in range(int(seconds * rate)):
        t = i / rate
        frames += struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * t)) if t >= tone_from else 0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))


def _onset_s(path, threshold=3000):
    """Where the first loud sample of the file's FIRST audio stream is, in seconds from the file's start."""
    done = subprocess.run([_ffmpeg(), "-v", "error", "-i", path, "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                           "-f", "s16le", "-"], capture_output=True)
    if done.returncode != 0 or not done.stdout:
        return None
    pcm = array.array("h")
    pcm.frombytes(done.stdout[: len(done.stdout) // 2 * 2])
    for i, v in enumerate(pcm):
        if abs(v) > threshold:
            return i / 16000.0
    return None


def _has_audio(path):
    from roop.ffmpeg_path import ffprobe_binary
    out = subprocess.run([ffprobe_binary(), "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_name",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    return bool(out.strip())


def _frames(path):
    from roop.ffmpeg_path import ffprobe_binary
    out = subprocess.run([ffprobe_binary(), "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    return int(out.strip().split()[0])


class RemuxAlignmentTest(unittest.TestCase):
    """30 fps target, 10 s long. The render is the 2 s window frames 150..210 (t = 5.0..7.0 s). The override track is
    silent until 6.0 s and then a tone, so in the delivered file the tone must start at 1.0 s."""
    TOL = 0.06                                              # AAC priming / one packet

    @classmethod
    def setUpClass(cls):
        _ffmpeg()
        cls.tmp = tempfile.mkdtemp(prefix="roop_mc_")
        t = cls.tmp
        cls.target = os.path.join(t, "target.mp4")
        _run(["-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=10", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", cls.target])
        cls.window = os.path.join(t, "window.mp4")           # what the render produced: 60 frames, no audio
        _run(["-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", cls.window])
        cls.wav = os.path.join(t, "dub.wav")
        _write_wav(cls.wav, 10.0, tone_from=6.0)
        cls.tone0 = os.path.join(t, "tone0.wav")             # tone from the very first sample
        _write_wav(cls.tone0, 4.0, tone_from=0.0)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def mux(self, audio, start, end, name, offset=0.0, window=None):
        from roop import util_ffmpeg
        out = os.path.join(self.tmp, name)
        ok = util_ffmpeg.restore_audio(window or self.window, audio, start, end, out, timing_source=self.target, audio_offset_ms=offset)
        self.assertTrue(ok, "mux failed")
        return out

    def test_a_trimmed_override_track_starts_where_the_video_says_not_where_24_fps_says(self):
        out = self.mux(self.wav, 150, 210, "trim.mp4")
        onset = _onset_s(out)
        self.assertIsNotNone(onset)
        self.assertAlmostEqual(onset, 1.0, delta=self.TOL)
        self.assertEqual(_frames(out), 60, "the video is a stream copy: every frame survives")

    def test_the_defect_this_replaces_cut_the_track_at_the_audio_files_fallback_fps(self):
        """`detect_fps` of an audio file is 24.0, so the old arithmetic put the trim at 150 / 24 = 6.25 s (a tone from
        its first sample) instead of 5.0 s. Pinned against the fallback itself so it cannot be reintroduced quietly."""
        from roop import utilities
        self.assertEqual(utilities.constant_frame_rate(utilities.detect_fps(self.wav)), 24.0)
        self.assertNotAlmostEqual(150 / 24.0, 150 / 30.0, delta=1.0)

    def test_a_positive_slip_plays_the_audio_later_and_a_negative_one_earlier(self):
        base = _onset_s(self.mux(self.wav, 150, 210, "s0.mp4"))
        later = _onset_s(self.mux(self.wav, 150, 210, "s_plus.mp4", offset=250))
        earlier = _onset_s(self.mux(self.wav, 150, 210, "s_minus.mp4", offset=-250))
        self.assertAlmostEqual(later - base, 0.25, delta=self.TOL)
        self.assertAlmostEqual(base - earlier, 0.25, delta=self.TOL)

    def test_a_slip_that_reaches_before_the_start_of_the_file_becomes_head_silence(self):
        out = self.mux(self.tone0, 0, 60, "lead.mp4", offset=500)
        self.assertAlmostEqual(_onset_s(out), 0.5, delta=self.TOL)

    def test_the_slip_extremes_are_exact_at_both_ends(self):
        plus = _onset_s(self.mux(self.wav, 150, 210, "e_plus.mp4", offset=500))
        minus = _onset_s(self.mux(self.wav, 150, 210, "e_minus.mp4", offset=-500))
        self.assertAlmostEqual(plus, 1.5, delta=self.TOL)
        self.assertAlmostEqual(minus, 0.5, delta=self.TOL)

    def test_a_wav_override_is_delivered_as_aac_in_the_mp4(self):
        out = self.mux(self.tone0, 0, 60, "wav_mp4.mp4")
        self.assertTrue(_has_audio(out))

    def test_untouched_defaults_still_take_the_historical_stream_copy_path(self):
        from roop import util_ffmpeg
        calls = []
        orig = util_ffmpeg.mux_with_audio_track
        util_ffmpeg.mux_with_audio_track = lambda *a, **k: calls.append(a) or True
        try:
            # a video with no audio stream is fine: the stream-copy branch runs and simply maps nothing
            util_ffmpeg.restore_audio(self.window, self.target, 0, 60, os.path.join(self.tmp, "hist.mp4"))
        finally:
            util_ffmpeg.mux_with_audio_track = orig
        self.assertEqual(calls, [])

    # silent-video fallback, at the file level
    def test_a_silent_target_with_no_override_delivers_a_silent_file_with_every_frame(self):
        out = os.path.join(self.tmp, "silent.mp4")
        from roop import util_ffmpeg
        self.assertTrue(util_ffmpeg.restore_audio(self.window, self.target, 0, 60, out))
        self.assertFalse(_has_audio(out))
        self.assertEqual(_frames(out), 60)

    def test_an_override_track_gives_a_silent_target_its_dialogue(self):
        out = self.mux(self.tone0, 0, 60, "dubbed_silent.mp4")
        self.assertTrue(_has_audio(out))
        self.assertAlmostEqual(_onset_s(out), 0.0, delta=self.TOL)

    def test_finalize_render_routes_an_override_through_the_video_timing(self):
        """The integration point: finalize_render must hand `restore_audio` the video as the timing source."""
        from roop import finalize_output
        seen = {}

        def fake_mux(temp, audio, start, end, dest, **kw):
            seen.update(kw)
            shutil.copyfile(temp, dest)
            return True
        out = os.path.join(self.tmp, "fin.mp4")
        shutil.copyfile(self.window, os.path.join(self.tmp, "fin_in.mp4"))
        finalize_output.finalize_render(os.path.join(self.tmp, "fin_in.mp4"), out, audio_source=self.wav, video_source=self.target,
                                        trim_start=150, trim_end=210, expected_frames=60, audio_offset_ms=120.0,
                                        mux=fake_mux, source_has_audio=lambda p: False)
        self.assertEqual(seen.get("timing_source"), self.target)
        self.assertEqual(seen.get("audio_offset_ms"), 120.0)

        seen.clear()
        shutil.copyfile(self.window, os.path.join(self.tmp, "fin_in2.mp4"))
        finalize_output.finalize_render(os.path.join(self.tmp, "fin_in2.mp4"), os.path.join(self.tmp, "fin2.mp4"),
                                        audio_source=self.target, video_source=self.target, trim_start=150, trim_end=210,
                                        expected_frames=60, mux=fake_mux, source_has_audio=lambda p: False)
        self.assertEqual(seen, {}, "no override, no slip: the historical five-argument call")


class DistributedChunkAlignmentTest(unittest.TestCase):
    """A GOP worker renders a stream-copied `-an` slice as its own clip from frame 0. Each chunk must read the WHOLE
    source's audio from its own absolute start; before Stage 17 they all read from t = 0."""

    def _chunk_settings(self, tmp, settings, start_frame, frames=300):
        from fractions import Fraction
        from roop import distributed_render as dr, project_io
        target = os.path.join(tmp, "target.mp4")
        open(target, "wb").write(b"x")
        project = os.path.join(tmp, "p.roop")
        document = {"id": "p", "media": {"target": project_io.media_reference(target, project, kind="target"), "sources": []},
                    "timeline": {"frame_start": 0, "frame_end": 900}, "automation": {}, "settings": settings}
        chunk = dr.Chunk(index=1, start_frame=start_frame, end_frame=start_frame + frames,
                         start_time=Fraction(start_frame, 30), end_time=Fraction(start_frame + frames, 30))
        destination = os.path.join(tmp, "chunk.roop")
        dr._project_for_chunk(document, project, target, chunk, os.path.join(tmp, "out.mp4"), destination)
        return project_io.load_project(destination), target

    def test_an_audio_driven_chunk_is_given_the_whole_source_and_its_absolute_start(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            doc, target = self._chunk_settings(tmp, {"mouth_custody_mode": "audio_driven_sync", "audio_offset_ms": 120}, 300)
            s = doc["settings"]
            self.assertEqual(s["lipsync_frame_offset"], 300)
            self.assertEqual(s["override_audio_path"], target)
            self.assertEqual(s["audio_offset_ms"], 120)
            self.assertTrue(s["skip_audio"], "the worker never remuxes: the join does, once")

    def test_an_override_track_stays_the_override_for_every_chunk(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            dub = os.path.join(tmp, "dub.wav")
            open(dub, "wb").close()
            doc, _ = self._chunk_settings(tmp, {"mouth_custody_mode": "audio_driven_sync", "override_audio_path": dub}, 600)
            self.assertEqual(doc["settings"]["override_audio_path"], dub)
            self.assertEqual(doc["settings"]["lipsync_frame_offset"], 600)

    def test_legacy_lipsync_enabled_projects_are_chunked_the_same_way(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            doc, target = self._chunk_settings(tmp, {"lipsync_enabled": True}, 90)
            self.assertEqual((doc["settings"]["lipsync_frame_offset"], doc["settings"]["override_audio_path"]), (90, target))

    def test_modes_that_do_not_listen_are_left_alone(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            for settings in ({}, {"mouth_custody_mode": "source_mouth_restore"}):
                doc, _ = self._chunk_settings(tmp, settings, 300)
                self.assertNotIn("lipsync_frame_offset", doc["settings"])
                self.assertFalse(doc["settings"].get("override_audio_path"))

    def test_chunk_local_frame_n_reads_the_same_audio_second_as_source_frame_start_plus_n(self):
        from roop.lipsync_audio import frame_time
        fps = 30.0
        for chunk_start in (0, 300, 899):
            for n in (0, 1, 149, 299):
                local = mc.audio_time_s(frame_time(0 + chunk_start, n, fps), 0)        # what the worker computes
                absolute = frame_time(0, chunk_start + n, fps)                         # what a whole-clip render computes
                self.assertAlmostEqual(local, absolute, places=9)

    def test_chunks_tile_the_audio_without_a_gap_or_an_overlap(self):
        from roop.lipsync_audio import frame_time
        fps, spans, edge = 30.0, [(0, 300), (300, 600), (600, 900)], None
        for start, end in spans:
            first, last = frame_time(start, 0, fps), frame_time(start, end - start - 1, fps)
            if edge is not None:
                self.assertAlmostEqual(first - edge, 1 / fps, places=9)
            edge = last


if __name__ == "__main__":
    unittest.main()
