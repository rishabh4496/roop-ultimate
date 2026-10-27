"""Segment custody: the chunk matrix, re-rendering ONE slice, and a join that checks what it made.

Built on the same harness as test_queue_segments.py: the runner loop is real, only `_run_swap` is
faked by a renderer that writes `segment_destination(...)` the way core does, and every clip is a real
ffmpeg encode so the joins and the duration checks are measured, not assumed.

What is already pinned elsewhere and NOT repeated here: deterministic naming
(`.segments/<group>/seg_NNNN_<start>_<end>.mp4`), retention after the join, and `clear_output` not
wiping earlier slices (test_segment_output.py, test_queue_segments.py).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
sys.path.insert(0, os.path.join(APP, "tests"))

import api  # noqa: E402,F401  (module level on purpose: importing api inside setUp re-injects the real
#               target list over the harness's and every job fails with "target media is no longer loaded")
import roop.globals as g  # noqa: E402
from roop import pipeline_status, segment_output as so  # noqa: E402
from roop.ffmpeg_path import ffmpeg_binary, ffprobe_binary  # noqa: E402
import routes_queue as q  # noqa: E402
from test_queue import QueueTestBase, _Entry  # noqa: E402
from test_segment_output import _frames, _make_clip  # noqa: E402


class _Cfg:
    clear_output = True
    output_video_format = "mp4"


def _audio_packet_sizes(path, n=6):
    r = subprocess.run([ffprobe_binary(), "-v", "error", "-select_streams", "a:0", "-show_entries",
                        "packet=size", "-of", "json", path], capture_output=True, text=True)
    return [int(p["size"]) for p in json.loads(r.stdout or "{}").get("packets", [])][:n]


class Custody(QueueTestBase):
    def setUp(self):
        super().setUp()
        pipeline_status.reset()
        self.out = tempfile.mkdtemp(prefix="custody_out_")
        self.src = os.path.join(self.out, "clip.mp4")
        _make_clip(self.src, 30, audio=True)
        entry = _Entry(self.src, total=30)
        entry.media_id = "m1"
        entry.fps = 10
        self.entries.append(entry)
        import api
        self.api = api
        self._saved = (getattr(g, "CFG", None), getattr(g, "output_path", None), getattr(g, "skip_audio", False))
        g.CFG, g.output_path, g.skip_audio = _Cfg(), self.out, False
        self.api.roop_globals.CFG = g.CFG
        self.api.roop_globals.output_path = self.out
        self.rendered = []
        q._run_swap = self._render_slice

    def tearDown(self):
        g.CFG, g.output_path, g.skip_audio = self._saved
        g._segment_spec = None
        shutil.rmtree(self.out, ignore_errors=True)
        super().tearDown()

    def _render_slice(self, payload):
        self.ran.append(dict(payload))
        self.api._clear_output_folder()
        spec = g._segment_spec
        self.rendered.append(spec.index)
        dest = so.segment_destination(g.output_path, spec, ".mp4")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        _make_clip(dest, spec.end_frame - spec.start_frame + 1, audio=True)
        self.progress["progress"] = 1.0
        self.progress["processing"] = False

    def _split(self, marked=True, ranges=((1, 10), (11, 20), (21, 30))):
        jobs = []
        for i, (s, e) in enumerate(ranges, 1):
            job = {"target_name": "clip.mp4", "target_media_id": "m1", "source_index": 0,
                   "payload": {"swap_model": "inswapper"}, "frame_start": s, "frame_end": e}
            if marked:
                job["segment"] = {"group": "G1", "index": i, "count": len(ranges)}
            jobs.append(job)
        q.queue_add_batch({"jobs": jobs})

    def _group(self, name="G1"):
        return next(x for x in q.queue_segments()["groups"] if x["group"] == name)

    def _seg(self, index):
        return so.segment_destination(self.out, so.SegmentSpec("G1", index, 3, 0, 0), ".mp4").replace(
            "seg_%04d_0_0" % index, "seg_%04d_%d_%d" % (index, (index - 1) * 10 + 1, index * 10))

    # ── the chunk matrix ──
    def test_the_matrix_reports_every_slice_its_file_and_the_verified_join(self):
        self._split()
        self._drain(timeout=60)
        grp = self._group()
        self.assertEqual((grp["count"], grp["present"], grp["completed"], grp["can_join"]), (3, 3, 3, True))
        self.assertIsNone(grp["join_blocked"])
        for s in grp["segments"]:
            self.assertEqual(s["state"], "COMPLETED")
            self.assertTrue(s["exists"] and s["bytes"] > 0)
            self.assertEqual(s["name"], "seg_%04d_%d_%d.mp4" % (s["index"], s["start_frame"], s["end_frame"]))
            self.assertEqual(s["frames"], 10)
        self.assertEqual([s["index"] for s in grp["segments"]], [1, 2, 3])
        j = grp["joined"]
        self.assertTrue(j["exists"])
        self.assertEqual(j["url"], "/outputs/clip_joined.mp4")
        v = j["result"]["verify"]
        self.assertIs(v["ok"], True, v)
        self.assertAlmostEqual(v["joined_s"], v["expected_s"], delta=v["tolerance_s"])

    def test_the_matrix_before_anything_has_run_says_what_is_waiting(self):
        self._split()
        grp = self._group()
        self.assertEqual([s["state"] for s in grp["segments"]], ["QUEUED"] * 3)
        self.assertFalse(grp["can_join"])
        self.assertIn("not finished", grp["join_blocked"])
        self.assertFalse(grp["joined"]["exists"])

    def test_unmarked_ranged_jobs_on_one_target_are_a_group_too(self):
        self._split(marked=False, ranges=((1, 15), (16, 30)))
        groups = q.queue_segments()["groups"]
        self.assertEqual(len(groups), 1)
        self.assertTrue(groups[0]["group"].startswith("auto:"))
        self.assertEqual(groups[0]["count"], 2)

    def test_a_plain_job_is_not_a_group(self):
        self._add("clip.mp4")
        self.assertEqual(q.queue_segments()["groups"], [])

    # ── re-render one slice ──
    def test_rerendering_one_slice_touches_only_that_slice_and_rejoins(self):
        self._split()
        self._drain(timeout=60)
        before = {i: (os.path.getmtime(self._seg(i)), os.path.getsize(self._seg(i))) for i in (1, 2, 3)}
        first_join = os.path.getmtime(os.path.join(self.out, "clip_joined.mp4"))
        job2 = self._group()["segments"][1]["job_id"]
        time.sleep(1.1)                                   # mtime resolution
        self.rendered.clear()

        res = q.queue_segment_rerender({"id": job2})
        grp = next(x for x in res["groups"] if x["group"] == "G1")
        self.assertEqual([s["state"] for s in grp["segments"]], ["COMPLETED", "QUEUED", "COMPLETED"])
        self.assertFalse(grp["can_join"], "a set with a slice being redone is not joinable")
        self.assertIsNone(self._group()["joined"]["result"], "the old join no longer describes the set")
        self.assertEqual(self.rendered, [], "queuing must not render by itself")

        self._drain(timeout=60)
        self.assertEqual(self.rendered, [2], "only the chosen slice was rendered")
        after = {i: (os.path.getmtime(self._seg(i)), os.path.getsize(self._seg(i))) for i in (1, 2, 3)}
        self.assertEqual(after[1], before[1], "slice 1 was touched")
        self.assertEqual(after[3], before[3], "slice 3 was touched")
        self.assertGreater(after[2][0], before[2][0], "slice 2 was not rewritten")
        self.assertEqual(sorted(os.listdir(so.segments_dir(self.out, "G1"))),
                         ["seg_0001_1_10.mp4", "seg_0002_11_20.mp4", "seg_0003_21_30.mp4"])
        grp = self._group()
        self.assertTrue(grp["can_join"])
        self.assertGreater(os.path.getmtime(os.path.join(self.out, "clip_joined.mp4")), first_join)
        self.assertIs(grp["joined"]["result"]["verify"]["ok"], True)
        self.assertTrue(grp["joined"]["result"]["automatic"])
        self.assertEqual(_frames(os.path.join(self.out, "clip_joined.mp4")), 30)

    def test_a_failed_slice_can_be_redone_and_the_set_then_joins(self):
        self._split()
        real = self._render_slice

        def flaky(payload):
            if g._segment_spec.index == 2 and not getattr(self, "_failed_once", False):
                self._failed_once = True
                raise RuntimeError("boom")
            real(payload)
        q._run_swap = flaky
        self._drain(timeout=60)
        grp = self._group()
        self.assertEqual([s["state"] for s in grp["segments"]], ["COMPLETED", "FAILED", "COMPLETED"])
        self.assertIn("boom", grp["segments"][1]["error"])
        self.assertFalse(grp["joined"]["exists"], "an incomplete set must not have been joined")
        q.queue_segment_rerender({"id": grp["segments"][1]["job_id"]})
        self._drain(timeout=60)
        grp = self._group()
        self.assertEqual([s["state"] for s in grp["segments"]], ["COMPLETED"] * 3)
        self.assertTrue(grp["joined"]["exists"])

    def test_rerender_refuses_what_it_cannot_do(self):
        self._split()
        queued = self._group()["segments"][0]["job_id"]
        r = q.queue_segment_rerender({"id": queued})
        self.assertEqual(r.status_code, 409)                 # already waiting: nothing to redo
        self.assertEqual(q.queue_segment_rerender({"id": "nope"}).status_code, 404)
        self._add("other.mp4")
        plain = q._queue["jobs"][-1]["id"]
        self.assertEqual(q.queue_segment_rerender({"id": plain}).status_code, 404, "a non-segment job")

    # ── join now ──
    def test_join_now_is_refused_until_every_slice_has_finished(self):
        self._split()
        r = q.queue_segments_join({"group": "G1"})
        self.assertEqual(r.status_code, 409)
        self.assertIn("not finished", json.loads(r.body)["message"])
        self.assertEqual(q.queue_segments_join({"group": "nope"}).status_code, 404)
        self.assertFalse(os.path.exists(os.path.join(self.out, "clip_joined.mp4")))

    def test_join_now_rebuilds_a_deleted_join_with_the_sources_own_audio_stream(self):
        self._split()
        self._drain(timeout=60)
        joined = os.path.join(self.out, "clip_joined.mp4")
        os.remove(joined)
        res = q.queue_segments_join({"group": "G1"})
        self.assertTrue(res["ok"])
        self.assertEqual(res["url"], "/outputs/clip_joined.mp4")
        self.assertTrue(res["source_audio"])
        self.assertEqual(res["audio"], "copy", "an AAC source rides into the mp4 untouched")
        self.assertFalse(res["automatic"])
        self.assertIs(res["verify"]["ok"], True, res["verify"])
        self.assertEqual(_frames(joined), 30)
        # "stream copy" measured, not claimed: the audio packets are byte-for-byte the source's.
        n = 5
        self.assertEqual(_audio_packet_sizes(joined, n), _audio_packet_sizes(self.src, n))

    def test_the_duration_check_catches_a_join_that_is_the_wrong_length(self):
        self._split()
        self._drain(timeout=60)
        original = so.expected_join_duration
        so.expected_join_duration = lambda *a, **k: 5.0       # the source "should" be 5 s; the join is 3 s
        try:
            res = q.queue_segments_join({"group": "G1"})
        finally:
            so.expected_join_duration = original
        self.assertTrue(res["ok"], "the file was made; only its length is wrong")
        self.assertIs(res["verify"]["ok"], False)
        self.assertIn("expected 5.00s", res["verify"]["reason"])
        self.assertLess(res["verify"]["delta_s"], -1.5)

    def test_a_missing_slice_file_fails_the_join_and_deletes_nothing(self):
        self._split()
        self._drain(timeout=60)
        os.remove(self._seg(3))
        r = q.queue_segments_join({"group": "G1"})
        self.assertEqual(r.status_code, 500)
        self.assertIn("seg_0003", json.loads(r.body)["message"])
        self.assertTrue(os.path.isfile(self._seg(1)) and os.path.isfile(self._seg(2)))
        grp = self._group()
        self.assertFalse(grp["can_join"], "the matrix must show the missing file, not a green set")
        self.assertFalse(grp["segments"][2]["exists"])


class AudioAndDuration(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="custody_unit_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_audio_codec_choice(self):
        aac = os.path.join(self.dir, "a.mp4")
        _make_clip(aac, 10, audio=True)
        self.assertEqual(so.audio_codec_for_join(aac), "copy")
        pcm = os.path.join(self.dir, "p.mkv")
        r = subprocess.run([ffmpeg_binary(), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10",
                            "-f", "lavfi", "-i", "sine=frequency=440", "-frames:v", "10", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", "-shortest", pcm], capture_output=True)
        if r.returncode:
            self.skipTest("ffmpeg cannot write pcm/mkv here")
        self.assertEqual(so.audio_codec_for_join(pcm), "aac", "mp4 cannot hold PCM: re-encode, do not fail")
        self.assertEqual(so.audio_codec_for_join(None), "aac")
        silent = os.path.join(self.dir, "s.mp4")
        _make_clip(silent, 10, audio=False)
        self.assertEqual(so.audio_codec_for_join(silent), "aac")

    def test_expected_duration_prefers_the_source_when_the_slices_cover_it(self):
        src = os.path.join(self.dir, "src.mp4")
        _make_clip(src, 30, audio=True, fps=10)
        self.assertAlmostEqual(so.expected_join_duration([(1, 10), (11, 30)], 10, src, True), 3.0, delta=0.15)
        self.assertAlmostEqual(so.expected_join_duration([(1, 10), (21, 25)], 10, src, False), 1.5, places=3)
        self.assertIsNone(so.expected_join_duration([(1, 10)], None, None, False))
        self.assertAlmostEqual(so.probe_fps(src), 10.0, places=3)

    def test_verify_joined_reports_not_measured_as_neither_pass_nor_fail(self):
        clip = os.path.join(self.dir, "c.mp4")
        _make_clip(clip, 20, audio=False, fps=10)
        self.assertIsNone(so.verify_joined(clip, None, 10)["ok"])
        self.assertIs(so.verify_joined(clip, 2.0, 10)["ok"], True)
        self.assertIs(so.verify_joined(clip, 3.0, 10)["ok"], False)
        self.assertIs(so.verify_joined(os.path.join(self.dir, "missing.mp4"), 2.0, 10)["ok"], False)

    def test_the_concat_command_never_trims_video_to_the_audio(self):
        cmd = so.concat_command("m.txt", "o.mp4", "src.mp4", source_audio=True, audio_codec="copy")
        self.assertIn("-c:a copy", " ".join(cmd))
        self.assertNotIn("-shortest", cmd, "-shortest cost 4 of 30 frames when the source audio ended early")


if __name__ == "__main__":
    unittest.main()
