"""Three guards against the memory failure that killed a 40934-frame render.

Reported from a 16 GB / RTX 3060 (6 GB) machine: the run died at frame 5001,
twice, having burned 33 minutes each time. The log named three things at once
and only one of them was the cause.

    hevc_nvenc ... CreateInputBuffer failed: out of memory (10)
    Task finished with error code: -12 (Cannot allocate memory)
    MemoryError((256, 256, 3), dtype('float64'))

Error -12 is AVERROR(ENOMEM) raised inside ffmpeg's OWN filter/output threads,
and numpy missing a 1.5 MB allocation is not a GPU symptom: the wall was SYSTEM
RAM, and NVENC's "out of memory" was downstream of it. Frame 5001 is not a
coincidence either — ROOP_RESUME_CHUNK is 1000, so that is the sixth segment
boundary, where SegmentedVideoWriter opens a BRAND NEW encoder. The pre-flight
probe_encoder() cannot catch this: it runs once at frame 0 when memory is free.

Three separate defects fall out of that, one per class below.
"""

import os
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

FW_SRC = open(os.path.join(APP, 'roop', 'ffmpeg_writer.py'), encoding='utf-8').read()
SEG_SRC = open(os.path.join(APP, 'roop', 'segment_writer.py'), encoding='utf-8').read()
PM_SRC = open(os.path.join(APP, 'roop', 'ProcessMgr.py'), encoding='utf-8').read()
API_SRC = open(os.path.join(APP, 'api.py'), encoding='utf-8').read()


# ─────────────────────────────────────────────────────────────────────────────
class EncoderFallback(unittest.TestCase):
    """A hardware encoder that dies before its first frame must not kill the run.

    It is a RESOURCE failure, not a bad file: the same frames encode fine on the
    CPU. Losing 33 minutes of completed work to it is the avoidable part.
    """

    def setUp(self):
        import roop.ffmpeg_writer as fw       # module-level imports are os+subprocess
        self.fw = fw
        self.W = fw.FFMPEG_VideoWriter

    def _writer(self, codec, frames_written=0, fell_back=False):
        """A writer with no child process — enough to exercise the decision."""
        w = object.__new__(self.W)
        w.codec = codec
        w.proc = None
        w._frames_written = frames_written
        w._fell_back = fell_back
        w._spawned = []
        w._spawn = lambda c: w._spawned.append(c)
        return w

    def test_every_hardware_codec_the_ui_offers_has_a_fallback(self):
        """The dropdown is the population this map has to cover.

        Read from api.py rather than restated here, so adding av1_nvenc to the UI
        without adding it here fails instead of shipping a codec that dies hard.
        """
        m = re.search(r'_VIDEO_CODECS\s*=\s*\[([^\]]*)\]', API_SRC)
        self.assertIsNotNone(m, "could not find _VIDEO_CODECS in api.py")
        codecs = re.findall(r"'([^']+)'|\"([^\"]+)\"", m.group(1))
        codecs = [a or b for a, b in codecs]
        self.assertTrue(codecs, "parsed an empty codec list")
        hw = [c for c in codecs if any(t in c for t in ('nvenc', 'qsv', 'amf'))]
        self.assertTrue(hw, "expected at least one hardware encoder in the UI list")
        for c in hw:
            self.assertIn(c, self.W._SW_FALLBACK, f"{c} is selectable but has no CPU fallback")

    def test_fallbacks_are_software(self):
        """A fallback to another hardware encoder would fail the same way."""
        for hw, sw in self.W._SW_FALLBACK.items():
            self.assertFalse(any(t in sw for t in ('nvenc', 'qsv', 'amf')),
                             f"{hw} falls back to {sw}, which is also hardware")

    def test_retries_a_dead_hardware_encoder_once(self):
        w = self._writer('hevc_nvenc')
        self.assertTrue(w._retry_as_software())
        self.assertEqual(w._spawned, ['libx265'])
        self.assertTrue(w._fell_back)

    def test_never_retries_twice(self):
        """A software encoder that also dies is a real failure. Surface it."""
        w = self._writer('hevc_nvenc')
        self.assertTrue(w._retry_as_software())
        self.assertFalse(w._retry_as_software())
        self.assertEqual(len(w._spawned), 1)

    def test_never_switches_codec_mid_file(self):
        """THE constraint. Frames already in this segment were encoded as HEVC;
        appending x265 packets to them would corrupt the part that a resume is
        about to trust. Only a virgin encoder may be swapped."""
        w = self._writer('hevc_nvenc', frames_written=1)
        self.assertFalse(w._retry_as_software())
        self.assertEqual(w._spawned, [])

    def test_software_encoder_failure_is_not_retried(self):
        """libx265 dying is not a resource-shaped failure with an obvious plan B."""
        w = self._writer('libx265')
        self.assertFalse(w._retry_as_software())

    def test_write_frame_tries_the_fallback_before_giving_up(self):
        """Both death paths — already-exited and broken-pipe — must route through
        it, or the fallback exists and never fires."""
        body = FW_SRC.split('def write_frame', 1)[1].split('def close', 1)[0]
        self.assertEqual(body.count('_retry_as_software()'), 2,
                         "write_frame must attempt the fallback on BOTH the "
                         "poll()-is-dead path and the IOError path")
        self.assertIn('self._frames_written += 1', body,
                      "nothing increments the counter the guard reads")


# ─────────────────────────────────────────────────────────────────────────────
class SegmentValidity(unittest.TestCase):
    """A committed part is a promise that its frames survived. Check it is true.

    The crash committed `part 6 · frames 5001-5002 · 0 MB` from an encoder that
    had never opened — ffmpeg said "Nothing was written into output file" — while
    the writer counted the two frames it had handed to the dead pipe. Resume then
    inherits an empty file as valid and feeds it to the final concat.
    """

    def setUp(self):
        from roop.segment_writer import _segments_that_exist
        self.f = _segments_that_exist
        self.d = tempfile.mkdtemp(prefix='segvalid_')

    def _mk(self, name, size):
        with open(os.path.join(self.d, name), 'wb') as fh:
            fh.write(b'\0' * size)

    def test_stops_at_an_empty_segment(self):
        self._mk('a.mp4', 4096)
        self._mk('b.mp4', 0)          # the encoder that failed to open
        self._mk('c.mp4', 4096)
        m = {"segments": [{"file": "a.mp4", "frames": 1000},
                          {"file": "b.mp4", "frames": 2},
                          {"file": "c.mp4", "frames": 1000}]}
        segs, done = self.f(m, self.d)
        self.assertEqual([s["file"] for s in segs], ["a.mp4"])
        self.assertEqual(done, 1000, "an empty part must not count as progress")

    def test_healthy_segments_are_all_kept(self):
        for n in ('a.mp4', 'b.mp4', 'c.mp4'):
            self._mk(n, 4096)
        m = {"segments": [{"file": n, "frames": 1000} for n in ('a.mp4', 'b.mp4', 'c.mp4')]}
        segs, done = self.f(m, self.d)
        self.assertEqual(len(segs), 3)
        self.assertEqual(done, 3000)

    def test_missing_file_still_stops_the_prefix(self):
        self._mk('a.mp4', 4096)
        m = {"segments": [{"file": "a.mp4", "frames": 10},
                          {"file": "gone.mp4", "frames": 10}]}
        segs, done = self.f(m, self.d)
        self.assertEqual(len(segs), 1)
        self.assertEqual(done, 10)

    def test_finalize_refuses_to_commit_an_empty_part(self):
        """Fix it at the source too, not only on the way back in — otherwise the
        manifest still records a part that never existed."""
        body = SEG_SRC.split('def _finalize_segment', 1)[1].split('def close', 1)[0]
        self.assertIn('getsize', body,
                      "_finalize_segment commits on frame count alone")
        self.assertIn('_seg_bytes > 0', body,
                      "the size must gate the append, not just be measured")


# ─────────────────────────────────────────────────────────────────────────────
class StabChunkBudget(unittest.TestCase):
    """The chunk budget must count every live copy of a chunk, not one.

    ROOP_STAB_CHUNK_MB reads as "the decoded-frame memory budget", and at its
    1536 MB default that sounds like 1.5 GB. _run_stab_parallel actually holds
    SIX chunk-sized buffers at once, so the real reservation was ~9 GB — fine on
    32 GB, impossible on 16 GB alongside models, CUDA host memory and ffmpeg.
    """

    def setUp(self):
        from roop.ProcessMgr import ProcessMgr
        self.PM = ProcessMgr

    def _budget(self, avail_gb, share=None):
        """Call the real method with a stub self and a faked psutil reading."""
        import roop.ProcessMgr as pmmod

        class _NS:
            _STAB_LIVE_CHUNKS = self.PM._STAB_LIVE_CHUNKS

        class _FakeVM:
            available = int(avail_gb * 1024 ** 3)

        real = pmmod.psutil.virtual_memory
        old_share = os.environ.get('ROOP_STAB_RAM_SHARE')
        if share is not None:
            os.environ['ROOP_STAB_RAM_SHARE'] = str(share)
        elif 'ROOP_STAB_RAM_SHARE' in os.environ:
            del os.environ['ROOP_STAB_RAM_SHARE']
        pmmod.psutil.virtual_memory = lambda: _FakeVM()
        try:
            return self.PM._default_stab_chunk_mb(_NS())
        finally:
            pmmod.psutil.virtual_memory = real
            if old_share is None:
                os.environ.pop('ROOP_STAB_RAM_SHARE', None)
            else:
                os.environ['ROOP_STAB_RAM_SHARE'] = old_share

    def test_never_raises_the_old_default(self):
        """A machine that was fine before must behave EXACTLY as before — this is
        a fix for constrained boxes, not a retune for everyone."""
        for gb in (24, 32, 64, 128):
            self.assertEqual(self._budget(gb), 1536.0, f"{gb} GB free changed behaviour")

    def test_lowers_on_a_constrained_machine(self):
        """The reported machine: 16 GB total, so single-digit GB actually free."""
        for gb in (4, 8, 9, 11):
            b = self._budget(gb)
            self.assertLess(b, 1536.0, f"{gb} GB free still reserves the full budget")
            self.assertGreater(b, 0)

    def test_all_six_copies_stay_inside_the_share(self):
        """The property that makes the number mean something: what is reserved is
        the budget TIMES the number of live chunks."""
        for gb in (4, 8, 11, 16):
            total_gb = self._budget(gb) * self.PM._STAB_LIVE_CHUNKS / 1024.0
            self.assertLessEqual(round(total_gb, 3), round(gb * 0.40, 3) + 1e-6,
                                 f"{gb} GB free: reserves {total_gb:.2f} GB of frames")

    def test_share_is_configurable_and_clamped(self):
        self.assertGreater(self._budget(8, share=0.80), self._budget(8, share=0.20))
        self.assertEqual(self._budget(64, share='nonsense'), 1536.0)
        # Absurd values must not produce an absurd budget.
        self.assertLessEqual(self._budget(8, share=9.0) * 6 / 1024.0, 8 * 0.90 + 1e-6)

    def test_an_explicit_env_value_runs_exactly_as_set(self):
        """Same rule the pool knobs settled on: a control that silently runs a
        different number than the one it was given is a control that lies."""
        body = PM_SRC.split('def _stab_parallel_geometry', 1)[1].split('def _run_stab_parallel', 1)[0]
        self.assertIn("_env_budget", body)
        self.assertNotIn("min(budget_mb", body,
                         "an explicit ROOP_STAB_CHUNK_MB must not be clamped")

    def test_the_live_chunk_count_matches_the_code(self):
        """The constant is only worth having if it tracks reality. These are the
        queues whose depths it sums; if one changes, this number must too."""
        body = PM_SRC.split('def _run_stab_parallel', 1)[1]
        self.assertIn('prefetch_q = Queue(2)', body)
        self.assertIn('_write_q = Queue(1)', body)
        # reader-in-progress 1 + prefetch 2 + in-hand 1 + results 1 + write_q 1
        self.assertEqual(self.PM._STAB_LIVE_CHUNKS, 1 + 2 + 1 + 1 + 1)


if __name__ == '__main__':
    unittest.main()
