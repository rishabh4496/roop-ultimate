"""Split-render segments: unique names, retained files, a verified stream-copy join.

The defect (queue "Split into N segments"): every slice named its video from the output
template alone, so slices collided and, with `clear_output` on, each slice also wiped the folder.
These tests drive real ffmpeg on tiny clips, because a join that "works" on mocks is how an
unplayable file ships.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

from roop import segment_output as so  # noqa: E402
from roop.ffmpeg_path import ffmpeg_binary, ffprobe_binary  # noqa: E402


def _make_clip(path, frames, audio=True, fps=10, size="64x48"):
    cmd = [ffmpeg_binary(), "-v", "error", "-y", "-f", "lavfi", "-i",
           f"testsrc=size={size}:rate={fps}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100"]
    cmd += ["-frames:v", str(frames), "-c:v", "libx264", "-g", "5", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    r = subprocess.run(cmd + [path], capture_output=True, text=True)
    if r.returncode or not os.path.isfile(path):
        raise unittest.SkipTest("ffmpeg cannot encode test clips here: " + r.stderr[-200:])


def _frames(path):
    r = subprocess.run([ffprobe_binary(), "-v", "error", "-count_frames", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_read_frames", "-of", "json", path],
                       capture_output=True, text=True)
    return int(json.loads(r.stdout)["streams"][0]["nb_read_frames"])


def _has_audio(path):
    r = subprocess.run([ffprobe_binary(), "-v", "error", "-select_streams", "a:0", "-show_entries",
                        "stream=codec_name", "-of", "json", path], capture_output=True, text=True)
    return bool(json.loads(r.stdout or "{}").get("streams"))


class Naming(unittest.TestCase):
    def test_layout_is_hidden_folder_per_split_then_numbered_range(self):
        spec = so.SegmentSpec("job-7", 2, 3, 201, 400)
        dest = so.segment_destination("out", spec)
        self.assertEqual(dest.replace("\\", "/"), "out/.segments/job-7/seg_0002_201_400.mp4")

    def test_zero_padded_unique_and_sortable(self):
        names = [so.segment_filename(i, s, e) for i, (s, e) in
                 enumerate([(1, 200), (201, 400), (401, 600)], 1)]
        self.assertEqual(names[0], "seg_0001_1_200.mp4")
        self.assertEqual(len(set(names)), 3)
        self.assertEqual(sorted(names), names)

    def test_round_trip(self):
        n = so.segment_filename(12, 4001, 5000, "mp4")
        self.assertEqual(so.parse_segment_filename(n), {"index": 12, "start": 4001, "end": 5000, "ext": "mp4"})
        self.assertIsNone(so.parse_segment_filename("clip_2026-10-18.mp4"))
        self.assertIsNone(so.parse_segment_filename("clip_segment0001_000001_000010.mp4"))   # the old layout

    def test_rejects_nonsense_ranges(self):
        for bad in ((0, 1, 2), (1, -1, 5), (1, 9, 3)):
            with self.assertRaises(ValueError):
                so.segment_filename(*bad)

    def test_a_hostile_group_id_cannot_escape_the_segments_folder(self):
        root = os.path.abspath(os.path.join("out", ".segments"))
        for group in ('..\\evil:"x"', "../../etc", "a/b", "", "auto:clip.mp4", "..", "."):
            dest = os.path.abspath(so.segment_destination("out", so.SegmentSpec(group, 1, 2, 1, 2)))
            self.assertEqual(os.path.dirname(os.path.dirname(dest)), root, group)

    def test_ids_that_sanitise_alike_still_get_different_folders(self):
        self.assertNotEqual(so.safe_group_id("auto:a/b"), so.safe_group_id("auto:a_b"))
        self.assertEqual(so.safe_group_id("3f2a9c"), "3f2a9c")           # ordinary ids pass through

    def test_two_splits_of_clips_with_one_file_name_do_not_meet(self):
        """The collision the earlier `{stem}_segment...` layout still had: same file name, same ranges."""
        a = so.segment_destination("out", so.SegmentSpec("split-A", 1, 2, 1, 10))
        b = so.segment_destination("out", so.SegmentSpec("split-B", 1, 2, 1, 10))
        self.assertNotEqual(a, b)

    def test_the_old_template_naming_collided_and_segment_naming_does_not(self):
        """The bug, reproduced against the real template function."""
        import roop.globals as g
        from roop import utilities as util

        class Cfg:
            output_template = "{file}"        # any template without a time/index token

        saved_cfg, saved_out = getattr(g, "CFG", None), getattr(g, "output_path", None)
        g.CFG, g.output_path = Cfg(), "out"
        try:
            old = {util.replace_template("clip__temp.mp4", index=0) for _ in range(3)}
        finally:
            g.CFG, g.output_path = saved_cfg, saved_out
        self.assertEqual(len(old), 1, "three segments mapped to ONE path: only the last survives")
        specs = [so.SegmentSpec("g", i, 3, s, e) for i, (s, e) in enumerate([(1, 10), (11, 20), (21, 30)], 1)]
        new = {so.segment_destination("out", sp) for sp in specs}
        self.assertEqual(len(new), 3)


class Coverage(unittest.TestCase):
    def test_whole_source(self):
        self.assertTrue(so.covers_whole_source([(1, 50), (51, 100)], 100))
        self.assertTrue(so.covers_whole_source([(51, 100), (1, 50)], 100))   # order-free

    def test_gaps_and_partial_ranges_do_not_count(self):
        self.assertFalse(so.covers_whole_source([(1, 40), (51, 100)], 100))   # gap
        self.assertFalse(so.covers_whole_source([(11, 100)], 100))            # starts late
        self.assertFalse(so.covers_whole_source([(1, 90)], 100))              # ends early
        self.assertFalse(so.covers_whole_source([], 100))
        self.assertFalse(so.covers_whole_source([(1, 10)], 0))


class RealFfmpeg(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="segout_")

    def _segments(self, counts=(12, 8, 10), audio=True, group="G"):
        paths, start = [], 1
        for i, n in enumerate(counts, 1):
            p = so.segment_destination(self.dir, so.SegmentSpec(group, i, len(counts), start, start + n - 1))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            _make_clip(p, n, audio=audio)
            paths.append(p)
            start += n
        return paths

    def test_verify_reports_missing_empty_and_non_video(self):
        good = self._segments((5,))[0]
        empty = os.path.join(self.dir, "empty.mp4")
        open(empty, "wb").close()
        junk = os.path.join(self.dir, "junk.mp4")
        with open(junk, "wb") as fh:
            fh.write(b"not a video" * 50)
        reasons = so.verify_segments([good, empty, junk, os.path.join(self.dir, "gone.mp4")])
        self.assertEqual(len(reasons), 3)
        self.assertTrue(any("zero bytes" in r for r in reasons))
        self.assertTrue(any("no valid video stream: junk" in r for r in reasons))
        self.assertTrue(any("missing" in r for r in reasons))
        self.assertEqual(so.verify_segments([good]), [])

    def test_manifest_is_written_only_for_verified_segments(self):
        paths = self._segments()
        manifest = so.write_concat_manifest(paths, so.concat_list_path(self.dir, "clip"))
        self.assertTrue(manifest.endswith("clip_concat_list.txt"))
        lines = open(manifest, encoding="utf-8").read().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(all(l.startswith("file '") for l in lines))
        bad = os.path.join(self.dir, "zero.mp4")
        open(bad, "wb").close()
        target = os.path.join(self.dir, "other_concat_list.txt")
        with self.assertRaises(RuntimeError):
            so.write_concat_manifest(paths + [bad], target)
        self.assertFalse(os.path.exists(target), "a manifest for unjoinable files must not exist")

    def test_join_keeps_every_segment_and_sums_the_frames(self):
        paths = self._segments((12, 8, 10))
        before = {p: os.path.getsize(p) for p in paths}
        out = so.join_segments(paths, so.joined_path(self.dir, "clip"),
                               so.concat_list_path(self.dir, "clip"))
        self.assertEqual(_frames(out), 30)
        self.assertTrue(_has_audio(out), "the slices' own audio must be carried")
        for p in paths:
            self.assertTrue(os.path.isfile(p), "segment deleted by the join: " + p)
            self.assertEqual(os.path.getsize(p), before[p])
        self.assertTrue(os.path.isfile(so.concat_list_path(self.dir, "clip")))
        self.assertFalse(os.path.exists(out + ".part.mp4"))

    def test_two_splits_of_the_same_ranges_retain_all_six_and_each_joins_cleanly(self):
        """Regression: multi-segment renders keep every segment and join to the right frame count.
        Same ranges, same clip, two split ids - the case a `{stem}_segment...` name still overwrote."""
        a = self._segments((12, 8, 10), group="split-A")
        b = self._segments((12, 8, 10), group="split-B")
        self.assertEqual(len({os.path.abspath(p) for p in a + b}), 6, "six distinct files, none overwritten")
        for group, paths in (("A", a), ("B", b)):
            out = so.join_segments(paths, os.path.join(self.dir, f"final_{group}.mp4"),
                                   os.path.join(self.dir, f"concat_{group}.txt"))
            self.assertEqual(_frames(out), 30, group)
            self.assertTrue(_has_audio(out), group)
        self.assertTrue(all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in a + b),
                        "a join (or the other split's join) removed a segment")

    def test_join_with_the_sources_own_audio(self):
        paths = self._segments((12, 8, 10), audio=False)
        source = os.path.join(self.dir, "source.mp4")
        _make_clip(source, 30, audio=True)
        out = so.join_segments(paths, os.path.join(self.dir, "full.mp4"),
                               so.concat_list_path(self.dir, "clip"),
                               source_media=source, source_audio=True)
        self.assertEqual(_frames(out), 30)
        self.assertTrue(_has_audio(out))

    def test_a_failed_join_leaves_no_partial_output_and_no_deleted_segments(self):
        paths = self._segments((6, 6))
        broken = os.path.join(self.dir, "broken.mp4")
        with open(broken, "wb") as fh:
            fh.write(b"\x00" * 4096)
        out = os.path.join(self.dir, "joined.mp4")
        with self.assertRaises(RuntimeError):
            so.join_segments(paths + [broken], out, os.path.join(self.dir, "m.txt"))
        self.assertFalse(os.path.exists(out))
        self.assertTrue(all(os.path.isfile(p) for p in paths))

    def test_command_shape_matches_the_documented_form(self):
        cmd = so.concat_command("m.txt", "o.mp4", "src.mp4", source_audio=True)
        joined = " ".join(cmd)
        for token in ("-f concat", "-safe 0", "-i m.txt", "-i src.mp4", "-map 0:v:0", "-map 1:a?",
                      "-c:v copy", "-c:a aac", "-movflags +faststart"):
            self.assertIn(token, joined)
        self.assertIn("-map 0:a?", " ".join(so.concat_command("m.txt", "o.mp4")))


if __name__ == "__main__":
    unittest.main()
