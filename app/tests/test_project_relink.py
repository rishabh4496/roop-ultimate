"""UI overhaul Stage 8: relinking a missing input of a persistent project checkpoint.

The same-file rule is the point of the feature: a relink that accepted a different clip would resume a
different render under this project's name. These tests pin the refusals as well as the success.
"""

import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import project_checkpoint as pc  # noqa: E402
from tests.test_project_checkpoint import _Config  # noqa: E402


class ProjectRelinkTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="project_relink_")
        self.old_dir = pc.PROJECTS_DIR
        pc.PROJECTS_DIR = self.tmp.name
        self.source = os.path.join(self.tmp.name, "ann.png")
        self.target = os.path.join(self.tmp.name, "target.mp4")
        for path, data in ((self.source, b"source-bytes"), (self.target, b"target-bytes")):
            with open(path, "wb") as fh:
                fh.write(data)
        self.record = pc.new_project(
            job_id="job-relink", name="target.mp4",
            payload={"swap_model": "inswapper", "output_method": "File"},
            sources=[pc.file_identity(self.source)],
            target=pc.file_identity(self.target), frame_start=0, frame_end=10,
            output={"directory": self.tmp.name, "format": "mp4", "codec": "libx264",
                    "quality": 14, "method": "File", "template": "{file}_{time}"},
            cfg=_Config(), target_faces=[], app_version="main@test")

    def tearDown(self):
        pc.PROJECTS_DIR = self.old_dir
        self.tmp.cleanup()

    def _moved_copy(self, name, data):
        moved = os.path.join(self.tmp.name, "moved", name)
        os.makedirs(os.path.dirname(moved), exist_ok=True)
        with open(moved, "wb") as fh:
            fh.write(data)
        return moved

    def test_missing_source_relinks_to_the_same_bytes_at_a_new_path(self):
        moved = self._moved_copy("ann.png", b"source-bytes")
        os.remove(self.source)
        self.assertTrue(pc.validate(pc.load(self.record["id"]), _Config()))

        updated = pc.relink_input(pc.load(self.record["id"]), role="source", index=0, path=moved)

        self.assertEqual(updated["inputs"]["sources"][0]["path"], os.path.abspath(moved))
        self.assertEqual(pc.validate(pc.load(self.record["id"]), _Config()), [])

    def test_a_different_file_is_refused_and_the_record_is_untouched(self):
        other = self._moved_copy("ann.png", b"a different face entirely")
        os.remove(self.source)
        before = pc.load(self.record["id"])["inputs"]["sources"][0]["path"]

        with self.assertRaises(ValueError) as ctx:
            pc.relink_input(pc.load(self.record["id"]), role="source", index=0, path=other)

        self.assertIn("SHA-256", str(ctx.exception))
        self.assertEqual(pc.load(self.record["id"])["inputs"]["sources"][0]["path"], before)

    def test_target_relinks_the_same_way(self):
        moved = self._moved_copy("target.mp4", b"target-bytes")
        os.remove(self.target)
        updated = pc.relink_input(pc.load(self.record["id"]), role="target", index=None, path=moved)
        self.assertEqual(updated["inputs"]["target"]["path"], os.path.abspath(moved))
        self.assertEqual(pc.validate(pc.load(self.record["id"]), _Config()), [])

    def test_bad_role_index_and_path_are_refused(self):
        record = pc.load(self.record["id"])
        with self.assertRaises(ValueError):
            pc.relink_input(record, role="output", index=0, path=self.source)
        with self.assertRaises(ValueError):
            pc.relink_input(record, role="source", index=7, path=self.source)
        with self.assertRaises(ValueError):
            pc.relink_input(record, role="source", index=0, path="")
        with self.assertRaises(ValueError):
            pc.relink_input(record, role="source", index=0,
                            path=os.path.join(self.tmp.name, "does-not-exist.png"))

    def test_the_missing_message_is_the_one_the_ui_parses(self):
        # checkpointBadge.js reads "<label> is missing: <path>" with label "source N" or "target". Pin that the
        # backend still writes exactly that, so the relink button and its index cannot silently drift apart.
        os.remove(self.source)
        reasons = pc.validate(pc.load(self.record["id"]), _Config())
        pattern = re.compile(r"^(source (\d+)|target) is missing: (.*)$")
        match = next((pattern.match(r) for r in reasons if "is missing" in r), None)
        self.assertIsNotNone(match, reasons)
        self.assertEqual(match.group(1), "source 1")
        self.assertEqual(match.group(2), "1")
        self.assertEqual(match.group(3), os.path.abspath(self.source))


if __name__ == "__main__":
    unittest.main()
