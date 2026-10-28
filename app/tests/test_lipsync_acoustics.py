"""Stage 17: the viseme fallback opens the mouth FROM THE AUDIO, at the slipped time.

Found by the real-render proof (tests/verify_stage17_render.py): the slip changed which audio second every frame was fed
(proved), yet the delivered pixels were identical to the un-slipped render. ProcessMgr hands `Lipsync_Wav2Lip.prepare` a bare
(80, 16) mel chunk; `prepare`'s ndarray branch set `phoneme_energy` and never `viseme_openness`, and `_synthesize_viseme`
returns the crop untouched below 0.05 - so without the network's weights the fallback could not open a mouth for ANY audio.

`AudioFeatureCache.acoustic_for_time` carries the clip-normalised curves at a time, ProcessMgr passes them with the chunk, and
these tests pin: silence is closed, a vowel-like sound is open, the lookup follows the time it is given (the slip), and the
bare-chunk behaviour that hid the defect is gone from the production call.
"""
from __future__ import annotations

import math
import os
import struct
import sys
import tempfile
import unittest
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.lipsync_audio import build_mel_audio_cache  # noqa: E402

RATE = 16000


def _speechlike_wav(path, quiet_s=1.0, vowel_s=1.0):
    """Silence, then a loud vowel-like sound (strong 500 Hz / 1500 Hz formants, little high-frequency energy)."""
    frames = bytearray()
    for i in range(int((quiet_s + vowel_s) * RATE)):
        t = i / RATE
        v = 0.0 if t < quiet_s else 9000 * (math.sin(2 * math.pi * 500 * t) + 0.6 * math.sin(2 * math.pi * 1500 * t))
        frames += struct.pack("<h", int(max(-32000, min(32000, v))))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(bytes(frames))
    return path


class AcousticForTimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.cache = build_mel_audio_cache(_speechlike_wav(os.path.join(cls.tmp.name, "v.wav")), 30.0)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_silence_reads_closed_and_a_vowel_reads_open(self):
        e_quiet, o_quiet = self.cache.acoustic_for_time(0.5)
        e_vowel, o_vowel = self.cache.acoustic_for_time(1.5)
        self.assertEqual(o_quiet, 0.0)
        self.assertGreater(o_vowel, 0.2)
        self.assertGreater(e_vowel, e_quiet)

    def test_the_lookup_follows_the_time_it_is_given_so_the_slip_changes_it(self):
        from roop import mouth_custody as mc
        t = 1.3                                                    # inside the vowel
        open_now = self.cache.acoustic_for_time(mc.audio_time_s(t, 0))[1]
        # +500 ms: the sound heard at 1.3 s is the file's 0.8 s - still silence
        open_slipped = self.cache.acoustic_for_time(mc.audio_time_s(t, 500))[1]
        self.assertGreater(open_now, 0.2)
        self.assertEqual(open_slipped, 0.0)

    def test_times_outside_the_clip_are_clamped_not_errors(self):
        self.assertIsNotNone(self.cache.acoustic_for_time(-3.0))
        self.assertIsNotNone(self.cache.acoustic_for_time(999.0))

    def test_a_cache_with_no_mel_curves_answers_none(self):
        from roop.lipsync_audio import AudioFeatureCache
        self.assertIsNone(AudioFeatureCache(features=np.zeros((4, 8), dtype=np.float32)).acoustic_for_time(0.1))


class VisemeFallbackTest(unittest.TestCase):
    """The consumer: with no network loaded, the dict ProcessMgr now builds opens the mouth and a closed one does not."""

    def setUp(self):
        from roop.processors.Lipsync_Wav2Lip import Lipsync_Wav2Lip
        self.model = Lipsync_Wav2Lip()                              # session None: the built-in viseme path
        self.assertIsNone(self.model.session)
        rng = np.random.default_rng(3)
        self.frame = rng.integers(60, 200, (240, 240, 3), dtype=np.uint8)
        self.face = {"bbox": [40, 40, 200, 200]}
        self.chunk = np.zeros((80, 16), dtype=np.float32)

    def run_with(self, energy, openness):
        feat = {"mel_chunk": self.chunk, "phoneme_energy": energy, "viseme_openness": openness}
        prepared = self.model.prepare(self.frame, self.face, feat)
        return prepared["crop"], self.model.infer(prepared)

    def test_a_closed_mouth_leaves_the_crop_and_an_open_one_changes_it(self):
        crop, closed = self.run_with(0.0, 0.0)
        self.assertTrue(np.array_equal(closed, crop))
        crop, opened = self.run_with(0.8, 0.9)
        self.assertFalse(np.array_equal(opened, crop), "openness 0.9 must change the mouth region")

    def test_the_bare_chunk_the_pipeline_used_to_pass_could_never_open_it(self):
        """Pinned as the defect: a 2-D chunk alone gives openness 0 and an unchanged crop, whatever is in the chunk."""
        loud = np.full((80, 16), 3.0, dtype=np.float32)
        prepared = self.model.prepare(self.frame, self.face, loud)
        self.assertEqual(prepared["viseme_openness"], 0.0)
        self.assertTrue(np.array_equal(self.model.infer(prepared), prepared["crop"]))

    def test_openness_scales_the_aperture(self):
        _, small = self.run_with(0.4, 0.3)
        _, large = self.run_with(0.9, 1.0)
        self.assertFalse(np.array_equal(small, large))


class ProcessMgrPassesTheAcousticsTest(unittest.TestCase):
    def test_the_production_call_builds_the_dict_from_the_slipped_time(self):
        with open(os.path.join(APP, "roop", "ProcessMgr.py"), encoding="utf-8") as fh:
            src = fh.read()
        block = src[src.index("audio_feat = audio_cache.features_for_time(t_s)"):]
        block = block[:block.index("restorer.prepare(")]
        self.assertIn("audio_cache.acoustic_for_time(t_s)", block)
        self.assertIn("'viseme_openness': _acoustic[1]", block)
        # t_s is the slipped time, computed before the lookup
        before = src[:src.index("audio_feat = audio_cache.features_for_time(t_s)")]
        self.assertIn("mouth_custody.audio_time_s(", before[-900:])


if __name__ == "__main__":
    unittest.main()
