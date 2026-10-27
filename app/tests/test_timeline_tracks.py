"""What the React timeline's lanes read from the backend (``roop/timeline_tracks.py`` + the two routes).

The lanes draw whatever these return, so every wrong answer here is a lane that looks right and is
wrong: a presence pill that runs a stride too long, a face-free stretch that is off by the trim
offset, a waveform that is flat because the audio was read at the wrong scale. The groups below pin
the arithmetic against a brute-force oracle, the waveform against REAL ffmpeg output, and the routes
through the real ASGI app.

Not covered here, and said so in the changelog: nothing in this file runs a render, so that the
pipeline PUBLISHES its Stage 1 / Stage 2 maps during a real job is shown by the call sites
(test_pipeline_publishes_the_maps) and not by a render.
"""
import base64
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.environ.setdefault('ROOP_SKIP_STARTUP', '1')

from roop import timeline_tracks as tt  # noqa: E402


class RangeArithmetic(unittest.TestCase):
    def test_merge_complement_agree_with_a_per_frame_oracle(self):
        rng = np.random.default_rng(7)
        for _ in range(300):
            n = int(rng.integers(5, 120))
            ranges = []
            for _ in range(int(rng.integers(0, 8))):
                a = int(rng.integers(0, n))
                ranges.append((a, min(n - 1, a + int(rng.integers(0, 15)))))
            covered = np.zeros(n, bool)
            for s, e in ranges:
                covered[s:e + 1] = True
            merged = tt.merge_ranges(ranges)
            comp = tt.complement_ranges(ranges, 0, n - 1)
            got = np.zeros(n, bool)
            for s, e in merged:
                got[s:e + 1] = True
            inv = np.zeros(n, bool)
            for s, e in comp:
                inv[s:e + 1] = True
            self.assertTrue((got == covered).all(), (ranges, merged))
            self.assertTrue((inv == ~covered).all(), (ranges, comp))
            for (a, b), (c, d) in zip(merged, merged[1:]):
                self.assertGreater(c, b + 1, 'touching ranges must have merged')

    def test_ui_numbering_is_one_based_and_absolute(self):
        # clip-relative 0..9 of a window starting at absolute (0-based) frame 100 is UI frames 101..110
        self.assertEqual(tt.ui_ranges([(0, 9)], 100), [[101, 110]])
        self.assertEqual(tt.ui_ranges([(0, 0)], 0), [[1, 1]])

    def test_frames_to_ranges(self):
        self.assertEqual(tt.frames_to_ranges([5, 3, 4, 9, 10, 20]), [(3, 5), (9, 10), (20, 20)])
        self.assertEqual(tt.frames_to_ranges([]), [])

    def test_reuse_frames_needs_every_face_marked(self):
        faces = {0: [{'reuse_flow_warp': True}], 1: [{'reuse_flow_warp': True}, {'reuse_flow_warp': False}],
                 2: [], 3: [{}], 4: [{'reuse_flow_warp': True}, {'reuse_flow_warp': True}]}
        self.assertEqual(sorted(tt.reuse_frames(faces)), [0, 4])


class PresenceFromSamples(unittest.TestCase):
    def test_a_miss_between_two_hits_ends_the_stretch(self):
        # looked at every 10 frames; the person was seen at 0,10,20 and again at 50,60 (30 and 40 were misses)
        samples = list(range(0, 100, 10))
        out = tt.presence_intervals([0, 10, 20, 50, 60], samples, total_frames=100)
        self.assertEqual(len(out), 2)
        # widened by half a stride into the unseen gap on each side, clipped to the clip
        self.assertEqual(out[0], [1, 26])        # 0 -> 0, 20 + 5 = 25 (0-based) -> UI 26
        self.assertEqual(out[1], [46, 66])       # 50 - 5 = 45 -> 46, 60 + 5 = 65 -> 66

    def test_consecutive_hits_are_one_stretch(self):
        samples = list(range(0, 50, 5))
        out = tt.presence_intervals(samples, samples, total_frames=50)
        # Seen at every look, so one stretch. With no sample after the last one (45) the end is widened
        # by half a stride only - the 4 unlooked-at frames at the tail are not claimed.
        self.assertEqual(out, [[1, 48]])

    def test_the_person_never_seen_has_no_intervals(self):
        self.assertEqual(tt.presence_intervals([], [0, 10, 20]), [])

    def test_widening_is_clipped_to_the_clip(self):
        out = tt.presence_intervals([0, 90], [0, 90], total_frames=100)
        self.assertEqual(out[0][0], 1)
        self.assertLessEqual(out[-1][1], 100)

    def test_matches_a_per_frame_truth_to_within_half_a_stride(self):
        """Ground truth: a person on screen in known stretches; sample every 8th frame; the estimate
        must cover every sampled hit and never reach further than half a stride past the last one."""
        truth = [(20, 70), (140, 200), (260, 275)]
        total, stride = 300, 8
        samples = list(range(0, total, stride))
        hits = [f for f in samples if any(a <= f <= b for a, b in truth)]
        out = tt.presence_intervals(hits, samples, total_frames=total)
        got = np.zeros(total + 1, bool)
        for s, e in out:
            got[s - 1:e] = True
        for h in hits:
            self.assertTrue(got[h], f'hit {h} not inside any interval')
        for s, e in out:
            inside = [h for h in hits if s - 1 <= h <= e - 1]
            self.assertTrue(inside)
            self.assertLessEqual(inside[0] - (s - 1), stride // 2 + 1)
            self.assertLessEqual((e - 1) - inside[-1], stride // 2 + 1)

    def test_frame_base_moves_the_whole_answer(self):
        a = tt.presence_intervals([0, 10], [0, 10, 20], total_frames=0, frame_base=0)
        b = tt.presence_intervals([0, 10], [0, 10, 20], total_frames=0, frame_base=100)
        self.assertEqual([[s + 100, e + 100] for s, e in a], b)


class RenderMaps(unittest.TestCase):
    def setUp(self):
        tt.begin_run(None, 0, 0)

    def test_nothing_published_answers_no_data(self):
        out = tt.render_maps('x.mp4')
        self.assertEqual((out['video_matches'], out['face_free'], out['reuse']), (False, [], []))

    def test_face_free_is_the_complement_of_stage_1_in_ui_numbering(self):
        vid = os.path.abspath('clip.mp4')
        tt.begin_run(vid, frame_start=100, frame_count=50)
        tt.publish_active_ranges([(0, 9), (20, 29)])             # clip-relative: faces at 0-9 and 20-29
        tt.publish_reuse_frames([4, 5, 6, 25])
        out = tt.render_maps(vid)
        self.assertTrue(out['video_matches'])
        self.assertEqual(out['range'], [101, 150])
        # face-free (relative): 10-19 and 30-49  ->  UI 111-120 and 131-150
        self.assertEqual(out['face_free'], [[111, 120], [131, 150]])
        self.assertEqual(out['reuse'], [[105, 107], [126, 126]])
        self.assertTrue(out['stage1_ran'])

    def test_stage_1_not_run_is_unknown_not_all_active(self):
        vid = os.path.abspath('clip.mp4')
        tt.begin_run(vid, 0, 50)
        tt.publish_active_ranges(None)
        out = tt.render_maps(vid)
        self.assertEqual(out['face_free'], [])
        self.assertFalse(out['stage1_ran'])

    def test_maps_belong_to_their_video(self):
        tt.begin_run(os.path.abspath('a.mp4'), 0, 10)
        tt.publish_active_ranges([(0, 4)])
        self.assertFalse(tt.render_maps(os.path.abspath('b.mp4'))['video_matches'])
        self.assertEqual(tt.render_maps(os.path.abspath('b.mp4'))['face_free'], [])

    def test_a_new_run_forgets_the_old_one(self):
        vid = os.path.abspath('a.mp4')
        tt.begin_run(vid, 0, 10)
        tt.publish_reuse_frames([1, 2])
        tt.begin_run(vid, 0, 10)
        self.assertEqual(tt.render_maps(vid)['reuse'], [])

    def test_publishing_without_a_run_is_a_no_op(self):
        tt._maps.clear()
        tt.publish_active_ranges([(0, 1)])
        tt.publish_reuse_frames([1])
        self.assertEqual(tt._maps, {})


class WaveformEnvelope(unittest.TestCase):
    def test_peak_per_ten_ms_on_a_sqrt_scale(self):
        pcm = np.zeros(2000, dtype='<i2')            # one second at 2 kHz
        pcm[:1000] = 32767                           # first half loud
        env = tt.envelope_from_pcm(pcm)
        self.assertEqual(len(env), 100)
        self.assertTrue((env[:50] == 255).all())
        self.assertTrue((env[50:] == 0).all())
        quiet = tt.envelope_from_pcm(np.full(2000, 1000, dtype='<i2'))
        # 1000/32767 = -30 dB: linear would be 8/255; the sqrt scale lifts it to ~44 so speech is visible
        self.assertTrue(30 < int(quiet[0]) < 60, int(quiet[0]))

    def test_slicing_max_pools_and_never_invents_detail(self):
        env = np.arange(100, dtype=np.uint8)
        pooled = tt.slice_peaks(env, 0.0, 1.0, 10)
        self.assertEqual(list(pooled), [9, 19, 29, 39, 49, 59, 69, 79, 89, 99])
        stretched = tt.slice_peaks(env, 0.0, 0.1, 40)        # 10 envelope bins asked for 40 values
        self.assertEqual(len(stretched), 40)
        self.assertEqual(set(stretched.tolist()), set(range(10)))
        self.assertTrue((tt.slice_peaks(env, 5.0, 6.0, 8) == 0).all(), 'past the audio is silence')
        self.assertTrue((tt.slice_peaks(np.zeros(0, np.uint8), 0, 1, 8) == 0).all())

    def test_b64_round_trip(self):
        peaks = np.array([0, 1, 200, 255], dtype=np.uint8)
        self.assertEqual(list(base64.b64decode(tt.peaks_b64(peaks))), [0, 1, 200, 255])


def _ffmpeg():
    from roop.ffmpeg_path import ffmpeg_binary
    path = ffmpeg_binary()
    return path if (shutil.which(path) or os.path.isfile(path)) else None


@unittest.skipUnless(_ffmpeg(), 'ffmpeg not available')
class WaveformFromRealFfmpeg(unittest.TestCase):
    """A 3 s clip: 1 s silence, 1 s of a loud 440 Hz tone, 1 s silence. The envelope must put the loud
    stretch where the tone is, in time, at the documented 100 Hz - measured from the real decode."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix='tl_wave_')
        ff = _ffmpeg()
        cls.with_audio = os.path.join(cls.dir, 'tone.mp4')
        cls.silent = os.path.join(cls.dir, 'noaudio.mp4')
        tone = "aevalsrc='if(between(t,1,2),0.9*sin(2*PI*440*t),0)':s=44100:d=3"
        subprocess.run([ff, '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=64x64:r=25:d=3',
                        '-f', 'lavfi', '-i', tone, '-shortest', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        '-c:a', 'aac', cls.with_audio], check=True, timeout=120)
        subprocess.run([ff, '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=64x64:r=25:d=2',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', cls.silent], check=True, timeout=120)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def _wait(self, path):
        for _ in range(200):
            state, env = tt.envelope_state(path)
            if state != 'building':
                return state, env
            time.sleep(0.1)
        self.fail('envelope never finished building')

    def test_the_loud_second_is_where_the_tone_is(self):
        state, env = self._wait(self.with_audio)
        self.assertEqual(state, 'ready')
        self.assertAlmostEqual(len(env) / tt.ENVELOPE_HZ, 3.0, delta=0.15)
        silence = env[10:90]                      # 0.1 - 0.9 s
        loud = env[115:185]                       # 1.15 - 1.85 s
        tail = env[215:290]
        self.assertLess(int(silence.max()), 20, 'the first second is silent')
        self.assertGreater(int(loud.min()), 200, 'the tone is loud for the whole of its second')
        self.assertLess(int(tail.max()), 20, 'and silent again after')
        # the onset is within 60 ms of 1.0 s (AAC priming / frame granularity), and the offset of 2.0 s
        above = np.nonzero(env > 128)[0]
        self.assertAlmostEqual(above[0] / tt.ENVELOPE_HZ, 1.0, delta=0.06)
        self.assertAlmostEqual((above[-1] + 1) / tt.ENVELOPE_HZ, 2.0, delta=0.06)

    def test_a_file_with_no_audio_says_none(self):
        state, env = self._wait(self.silent)
        self.assertEqual(state, 'none')

    def test_a_missing_file_is_an_error_not_a_hang(self):
        self.assertEqual(tt.envelope_state(os.path.join(self.dir, 'nope.mp4'))[0], 'error')


class FaceBankRecordsWhereEachPersonWasSeen(unittest.TestCase):
    def test_characters_carry_their_sampled_frames(self):
        from roop.face_bank import FaceBank
        bank = FaceBank()

        def member(frame, emb):
            return {'face': object(), 'embedding': np.asarray(emb, np.float32), 'frame_idx': frame, 'det_score': 0.9,
                    'off_axis': 5.0, 'crop': np.zeros((8, 8, 3), np.uint8), 'yaw': 0.0, 'pitch': 0.0}
        collected = [member(0, [1, 0]), member(10, [1, 0]), member(10, [1, 0.01]), member(40, [0, 1]), member(50, [0, 1])]
        chars = bank._aggregate_characters(collected, [0, 0, 0, 1, 1])
        by_size = sorted(chars, key=lambda c: -c['cluster_size'])
        self.assertEqual(by_size[0]['frames'], [0, 10])          # two detections in frame 10 are one frame
        self.assertEqual(by_size[1]['frames'], [40, 50])
        bank.clear()
        self.assertEqual(bank.sample_frames, set())

    def test_both_scan_paths_record_every_frame_they_looked_at(self):
        src = open(os.path.join(APP, 'roop', 'face_bank.py'), encoding='utf-8').read()
        self.assertEqual(src.count('self.sample_frames.add('), 2,
                         'the hardware path and the capture path must each record the frames they examine')


class ThePipelinePublishesItsMaps(unittest.TestCase):
    """Call sites, not a render: begin before Stage 1, Stage 1's ranges after it, Stage 2's reuse plan
    after the pre-pass. A real-render check is listed as NOT done in the changelog."""

    def _src(self, name):
        return open(os.path.join(APP, 'roop', name), encoding='utf-8').read()

    def test_stage_1_and_the_run_boundary(self):
        src = self._src('procmgr_batch.py')
        begin = src.index('_timeline_tracks.begin_run(source_video, frame_start, frame_count)')
        ranges = src.index('_timeline_tracks.publish_active_ranges(ranges)')
        self.assertLess(src.index('_st.begin_run(frame_count)'), begin)
        self.assertLess(begin, ranges, 'a run must begin (clearing the last one) before Stage 1 publishes')

    def test_stage_2_reuse_plan(self):
        src = self._src('procmgr_tracking.py')
        self.assertIn('_timeline_tracks.publish_reuse_frames(_timeline_tracks.reuse_frames(self._temporal_faces))', src)


class TheRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import api
        from fastapi.testclient import TestClient
        from roop.face_bank import get_face_bank
        cls.api = api
        cls.client = TestClient(api.app)
        cls.get_bank = staticmethod(get_face_bank)

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='tl_routes_')
        self.video = os.path.join(self.dir, 'clip.mp4')
        open(self.video, 'wb').close()
        entry = mock.Mock()
        entry.filename = self.video
        entry.total_frames = 100
        entry.media_id = 'm-1'
        self.entry = entry
        self.patches = [
            mock.patch.object(self.api, 'list_files_process', [entry]),
            mock.patch.object(self.api, '_target_index_for_ref', lambda *a, **k: (0, 'm-1')),
        ]
        for p in self.patches:
            p.start()
        bank = self.get_bank()
        self._saved = (bank.characters, bank.last_video_path, set(bank.scene_cuts), set(bank.sample_frames), bank.total_frames)
        bank.characters = []
        bank.last_video_path = None
        bank.scene_cuts = set()
        bank.sample_frames = set()
        tt.begin_run(None, 0, 0)

    def tearDown(self):
        for p in self.patches:
            p.stop()
        bank = self.get_bank()
        bank.characters, bank.last_video_path, bank.scene_cuts, bank.sample_frames, bank.total_frames = self._saved
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_tracks_are_empty_and_honest_before_any_scan(self):
        body = self.client.get('/api/target/timeline_tracks?index=0').json()
        self.assertEqual((body['scene_cuts'], body['presence'], body['presence_source']), ([], [], 'none'))
        self.assertFalse(body['render']['video_matches'])

    def test_tracks_after_a_face_bank_scan_are_in_ui_numbering(self):
        bank = self.get_bank()
        bank.last_video_path = self.video
        bank.scene_cuts = {30, 60}                                 # 0-based keyframe indices
        bank.sample_frames = set(range(0, 100, 10))
        bank.characters = [{'person_id': 'tp_0', 'name': 'Actor 1', 'frames': [0, 10, 20, 70, 80]},
                           {'person_id': 'tp_1', 'name': 'Actor 2', 'frames': [40, 50]}]
        body = self.client.get('/api/target/timeline_tracks?index=0').json()
        self.assertEqual(body['scene_cuts'], [31, 61], 'a 0-based cut index is the 1-based first frame of the shot + 1')
        self.assertEqual(body['presence_source'], 'face_bank_sampled')
        self.assertEqual(body['sample_stride'], 10)
        p0, p1 = body['presence']
        self.assertEqual((p0['person_id'], len(p0['intervals'])), ('tp_0', 2))
        self.assertEqual(p1['intervals'], [[36, 56]])

    def test_a_bank_scanned_on_another_video_is_ignored(self):
        bank = self.get_bank()
        bank.last_video_path = os.path.join(self.dir, 'other.mp4')
        bank.characters = [{'person_id': 'tp_0', 'name': 'A', 'frames': [0]}]
        bank.scene_cuts = {5}
        body = self.client.get('/api/target/timeline_tracks?index=0').json()
        self.assertEqual((body['scene_cuts'], body['presence']), ([], []))

    def test_the_run_maps_ride_along_when_they_belong_to_this_video(self):
        tt.begin_run(self.video, frame_start=0, frame_count=100)
        tt.publish_active_ranges([(0, 39), (60, 99)])
        tt.publish_reuse_frames([70, 71])
        body = self.client.get('/api/target/timeline_tracks?index=0').json()
        self.assertTrue(body['render']['video_matches'])
        self.assertEqual(body['render']['face_free'], [[41, 60]])
        self.assertEqual(body['render']['reuse'], [[71, 72]])

    def test_an_unknown_target_is_a_404(self):
        with mock.patch.object(self.api, '_target_index_for_ref', lambda *a, **k: (None, None)):
            self.assertEqual(self.client.get('/api/target/timeline_tracks?index=9').status_code, 404)
            self.assertEqual(self.client.get('/api/target/waveform?index=9').status_code, 404)

    def test_waveform_for_a_still_image_is_none_not_building(self):
        with mock.patch.object(self.api.util, 'is_video', lambda f: False):
            body = self.client.get('/api/target/waveform?index=0').json()
        self.assertEqual((body['ready'], body['state']), (True, 'none'))

    @unittest.skipUnless(_ffmpeg(), 'ffmpeg not available')
    def test_waveform_is_building_then_ready_and_slices(self):
        ff = _ffmpeg()
        subprocess.run([ff, '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=64x64:r=25:d=2',
                        '-f', 'lavfi', '-i', 'sine=frequency=330:duration=2', '-shortest',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', self.video], check=True, timeout=120)
        with mock.patch.object(self.api.util, 'is_video', lambda f: True):
            body = {}
            for _ in range(200):
                body = self.client.get('/api/target/waveform?index=0&bins=64').json()
                if body.get('ready'):
                    break
                self.assertEqual(body['state'], 'building')
                time.sleep(0.1)
            self.assertEqual(body['state'], 'ready')
            self.assertEqual(body['bins'], 64)
            self.assertEqual(len(base64.b64decode(body['peaks'])), 64)
            self.assertAlmostEqual(body['duration'], 2.0, delta=0.15)
            window = self.client.get('/api/target/waveform?index=0&t0=0.5&t1=1.0&bins=16').json()
            self.assertEqual((window['t0'], window['t1'], window['bins']), (0.5, 1.0, 16))
            # ffmpeg's `sine` source is 1/8 of full scale; the envelope is sqrt-scaled 0..255, so the
            # expected peak is 255 * sqrt(0.125) = 90.2 - the documented scale, measured end to end.
            self.assertAlmostEqual(max(base64.b64decode(window['peaks'])), 90, delta=8)


if __name__ == '__main__':
    unittest.main()
