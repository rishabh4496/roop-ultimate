"""The queue as a split-render runner: segments are named, retained and joined; state is on disk.

The runner loop is real; only `_run_swap` is faked, by a renderer that does what core does for a
slice (write `segment_destination(...)` from `roop.globals._segment_spec`) and what api does before
it (`_clear_output_folder`, with `clear_output` ON, the default that used to wipe earlier slices).
"""
import json
import os
import sys
import tempfile
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
sys.path.insert(0, os.path.join(APP, "tests"))

import roop.globals as g  # noqa: E402
from roop import pipeline_status, segment_output as so  # noqa: E402
import routes_queue as q  # noqa: E402
from test_queue import QueueTestBase, _Entry  # noqa: E402
from test_segment_output import _frames, _has_audio, _make_clip  # noqa: E402


class _Cfg:
    def __init__(self, clear_output=True):
        self.clear_output = clear_output
        self.output_video_format = "mp4"


class Markers(unittest.TestCase):
    def test_clean_segment(self):
        self.assertEqual(q._clean_segment({"group": "g", "index": 2, "count": 4}),
                         {"group": "g", "index": 2, "count": 4})
        for bad in (None, "x", {}, {"group": "", "index": 1, "count": 2},
                    {"group": "g", "index": 0, "count": 2}, {"group": "g", "index": 3, "count": 2},
                    {"group": "g", "index": 1, "count": 1}, {"group": "g", "index": "a", "count": 2}):
            self.assertIsNone(q._clean_segment(bad), bad)


class _E:
    def __init__(self, total):
        self.total_frames = total
        self.startframe = 0
        self.endframe = total


class UiRangesTileTheClip(unittest.TestCase):
    """A job's range is the UI's (1-based, inclusive); the entry's is the pipeline's (0-based start,
    exclusive end). `_apply_segment` used to pass the start through, so every job lost its first frame
    and a 3-way split of 600 frames rendered 597 (frames 0, 200 and 400 in nobody's slice) -
    found by `tests/verify_three_stage_pipeline.py`, invariant 5, with a real render."""

    def _slices(self, total, count):
        step = -(-total // count)                              # ceil, as stageSegments does
        out = []
        for i in range(count):
            fs, fe = i * step + 1, min((i + 1) * step, total)
            if fs > total:
                break
            entry = _E(total)
            q._apply_segment(entry, {"frame_start": fs, "frame_end": fe})
            out.append((entry.startframe, entry.endframe))
        return out

    def test_three_way_split_of_600_is_contiguous_and_complete(self):
        self.assertEqual(self._slices(600, 3), [(0, 200), (200, 400), (400, 600)])

    def test_any_split_covers_every_frame_exactly_once(self):
        for total in (600, 599, 601, 30, 7, 1000):
            for count in (2, 3, 4, 7):
                covered = []
                for a, b in self._slices(total, count):
                    covered.extend(range(a, b))
                self.assertEqual(covered, list(range(total)), (total, count))

    def test_a_whole_file_job_from_the_ui_starts_on_frame_zero(self):
        entry = _E(500)
        q._apply_segment(entry, {"frame_start": 1, "frame_end": 500})
        self.assertEqual((entry.startframe, entry.endframe), (0, 500))

    def test_a_job_with_no_range_leaves_the_entry_alone(self):
        entry = _E(500)
        entry.startframe = 40
        q._apply_segment(entry, {})
        self.assertEqual((entry.startframe, entry.endframe), (40, 500))


class SpecAndPersistence(QueueTestBase):
    def test_marked_job_uses_its_marker(self):
        self._add("a.mp4", frame_start=1, frame_end=50,
                  segment={"group": "G", "index": 1, "count": 2})
        job = q._queue["jobs"][0]
        with q._lock:
            spec = q._segment_spec_for(job)
        self.assertEqual((spec.group, spec.index, spec.count, spec.start_frame, spec.end_frame),
                         ("G", 1, 2, 1, 50))

    def test_unmarked_ranged_jobs_on_one_file_still_get_distinct_slice_names(self):
        self._add("a.mp4", frame_start=51, frame_end=100)
        self._add("a.mp4", frame_start=1, frame_end=50)
        self._add("b.mp4", frame_start=1, frame_end=50)       # alone on its file
        self._add("c.mp4")                                    # no range at all
        jobs = q._queue["jobs"]
        with q._lock:
            specs = [q._segment_spec_for(j) for j in jobs]
        self.assertEqual([s.index for s in specs[:2]], [2, 1])   # by frame_start, not queue order
        self.assertIsNone(specs[2])
        self.assertIsNone(specs[3])

    def test_the_state_on_disk_is_the_state_the_job_is_in_for_all_ten(self):
        self._add("a.mp4")
        job = q._queue["jobs"][0]
        for state in q.JOB_STATES:
            q._set_state(job, state, persist=True)
            with open(q.QUEUE_FILE, encoding="utf-8") as fh:
                on_disk = json.load(fh)["jobs"][0]
            self.assertEqual(on_disk["state"], state)
            self.assertEqual(on_disk["progress"]["phase"], state)
        self.assertEqual(len(q.JOB_STATES), 10)

    def test_processing_is_on_disk_while_the_render_runs(self):
        """Before: PROCESSING reached the file only when a project checkpoint was created, so a
        crash mid-render reloaded the job as PREPARING."""
        seen = []

        def render(payload):
            with open(q.QUEUE_FILE, encoding="utf-8") as fh:
                seen.append(json.load(fh)["jobs"][0]["state"])
            self.progress["progress"] = 1.0
            self.progress["processing"] = False

        q._run_swap = render
        self.entries.append(_Entry("a.mp4"))
        self._add("a.mp4")
        self._drain()
        self.assertEqual(seen, ["PROCESSING"])


class ClearOutput(unittest.TestCase):
    def setUp(self):
        import api
        self.api = api
        self.dir = tempfile.mkdtemp(prefix="clearout_")
        with open(os.path.join(self.dir, "keep.mp4"), "wb") as fh:
            fh.write(b"x")
        self.saved = (getattr(api.roop_globals, "CFG", None), api.roop_globals.output_path,
                      getattr(api.roop_globals, "_segment_spec", None))
        api.roop_globals.CFG = _Cfg(True)
        api.roop_globals.output_path = self.dir

    def tearDown(self):
        self.api.roop_globals.CFG, self.api.roop_globals.output_path, \
            self.api.roop_globals._segment_spec = self.saved

    def test_an_ordinary_run_still_clears(self):
        self.assertTrue(self.api._clear_output_folder())
        self.assertEqual(os.listdir(self.dir), [])

    def test_a_slice_of_a_split_render_never_clears(self):
        self.api.roop_globals._segment_spec = so.SegmentSpec("g", 2, 3, 11, 20)
        self.assertFalse(self.api._clear_output_folder())
        self.assertEqual(os.listdir(self.dir), ["keep.mp4"])

    def test_a_project_with_committed_segments_never_clears(self):
        self.assertFalse(self.api._clear_output_folder(has_checkpoint=True))
        self.assertEqual(os.listdir(self.dir), ["keep.mp4"])

    def test_an_ordinary_run_clears_everything_except_finished_slices(self):
        """An ordinary job queued between two slices used to take the finished slices with it."""
        slice_dir = so.segments_dir(self.dir, "G1")
        os.makedirs(slice_dir)
        kept = os.path.join(slice_dir, "seg_0001_1_10.mp4")
        with open(kept, "wb") as fh:
            fh.write(b"slice")
        os.makedirs(os.path.join(self.dir, "folder"))
        with open(os.path.join(self.dir, "folder", "old.mp4"), "wb") as fh:
            fh.write(b"x")
        self.assertTrue(self.api._clear_output_folder())
        self.assertEqual(os.listdir(self.dir), [".segments"])
        self.assertTrue(os.path.isfile(kept))

    def test_off_means_off(self):
        self.api.roop_globals.CFG = _Cfg(False)
        self.assertFalse(self.api._clear_output_folder())
        self.assertEqual(os.listdir(self.dir), ["keep.mp4"])


class SplitThroughTheRunner(QueueTestBase):
    def setUp(self):
        super().setUp()
        pipeline_status.reset()
        self.out = tempfile.mkdtemp(prefix="split_out_")
        self.src = os.path.join(self.out, "clip.mp4")
        _make_clip(self.src, 30, audio=True)
        entry = _Entry(self.src, total=30)
        entry.media_id = "m1"
        self.entries.append(entry)
        import api
        self.api = api
        self._saved = (getattr(g, "CFG", None), getattr(g, "output_path", None),
                       getattr(g, "skip_audio", False))
        g.CFG, g.output_path, g.skip_audio = _Cfg(True), self.out, False
        # api._clear_output_folder reads the module the api imported as `roop_globals`.
        self.api.roop_globals.CFG = g.CFG
        self.api.roop_globals.output_path = self.out
        q._run_swap = self._render_slice

    def tearDown(self):
        g.CFG, g.output_path, g.skip_audio = self._saved
        g._segment_spec = None
        super().tearDown()

    def _render_slice(self, payload):
        self.ran.append(dict(payload))
        self.api._clear_output_folder()            # what _run_swap does first, clear_output ON
        spec = g._segment_spec
        if spec is None:
            raise AssertionError("a split slice reached the renderer without its spec")
        dest = so.segment_destination(g.output_path, spec, ".mp4")
        os.makedirs(os.path.dirname(dest), exist_ok=True)      # core.batch_process_regular does the same
        _make_clip(dest, spec.end_frame - spec.start_frame + 1, audio=True)
        self.progress["progress"] = 1.0
        self.progress["processing"] = False

    def _queue_split(self):
        jobs = [{"target_name": "clip.mp4", "target_media_id": "m1", "source_index": 0,
                 "payload": {"swap_model": "inswapper"}, "frame_start": s, "frame_end": e,
                 "segment": {"group": "G1", "index": i, "count": 3}}
                for i, (s, e) in enumerate([(1, 10), (11, 20), (21, 30)], 1)]
        q.queue_add_batch({"jobs": jobs})

    def test_every_slice_survives_and_the_join_is_added_beside_them(self):
        self._queue_split()
        self._drain(timeout=60)
        states = [j["state"] for j in q._queue["jobs"]]
        self.assertEqual(states, ["COMPLETED"] * 3, [j["error"] for j in q._queue["jobs"]])
        names = sorted(os.listdir(self.out))
        seg_dir = so.segments_dir(self.out, "G1")
        self.assertEqual(sorted(os.listdir(seg_dir)),
                         ["seg_0001_1_10.mp4", "seg_0002_11_20.mp4", "seg_0003_21_30.mp4"])
        self.assertIn("clip_concat_list.txt", names)
        self.assertIn("clip_joined.mp4", names)
        self.assertNotIn("clip_segment0001_000001_000010.mp4", names, "slices live under .segments, not beside the joined file")
        joined = os.path.join(self.out, "clip_joined.mp4")
        self.assertEqual(_frames(joined), 30)
        self.assertTrue(_has_audio(joined))
        result = q._queue["jobs"][0]["group_join"]
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["source_audio"], "slices cover the whole source: use its own audio")
        self.assertEqual(result["segments"], [os.path.join(seg_dir, n) for n in sorted(os.listdir(seg_dir))])
        for name in os.listdir(seg_dir):
            self.assertGreater(os.path.getsize(os.path.join(seg_dir, name)), 0)

    def test_the_status_channel_saw_each_slice_and_the_join(self):
        self._queue_split()
        self._drain(timeout=60)
        snap = pipeline_status.snapshot(events=200)
        group = snap["segments"]["G1"]
        self.assertEqual(group["count"], 3)
        self.assertEqual(group["done"], 3)
        self.assertTrue(group["joined"]["ok"])
        kinds = [e["kind"] for e in snap["events"]]
        self.assertEqual(kinds.count(pipeline_status.SEGMENT_PROGRESS), 6)    # 3 x (PROCESSING, COMPLETED)
        self.assertEqual(kinds.count(pipeline_status.SEGMENTS_JOINED), 1)

    def test_a_failed_slice_blocks_the_join_and_keeps_the_finished_ones(self):
        calls = []

        def flaky(payload):
            calls.append(1)
            if len(calls) == 2:
                self.progress["error"] = "boom"
                self.progress["processing"] = False
                return
            self.progress["error"] = ""
            self._render_slice(payload)

        q._run_swap = flaky
        self._queue_split()
        self._drain(timeout=60)
        states = [j["state"] for j in q._queue["jobs"]]
        self.assertEqual(states, ["COMPLETED", "FAILED", "COMPLETED"])
        self.assertNotIn("clip_joined.mp4", os.listdir(self.out))
        kept = os.listdir(so.segments_dir(self.out, "G1"))
        self.assertIn("seg_0001_1_10.mp4", kept)
        self.assertIn("seg_0003_21_30.mp4", kept)
        self.assertNotIn("seg_0002_11_20.mp4", kept)


if __name__ == "__main__":
    unittest.main()
