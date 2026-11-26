"""/api/output lists renders, not the render path's own bookkeeping.

The resume journal (roop/journal.py) lives in the output folder and was shown on Home and in
the Outputs tab as a finished file, with a Delete button beside it.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

import roop.globals as roop_globals  # noqa: E402
import routes_output  # noqa: E402


class OutputListingHidesInternalFilesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = getattr(roop_globals, "output_path", None)
        roop_globals.output_path = self._tmp.name
        for name in ("clip_swapped.mp4", "still.png", "notes.txt",
                     "journal.json", "journal.json.tmp", "render.mp4.part", ".hidden.mp4"):
            Path(self._tmp.name, name).write_bytes(b"x")
        os.mkdir(os.path.join(self._tmp.name, "a_folder"))

    def tearDown(self):
        roop_globals.output_path = self._saved
        self._tmp.cleanup()

    def test_journal_and_partial_files_are_not_listed(self):
        names = {f["name"] for f in routes_output.list_output()["files"]}
        self.assertEqual({"clip_swapped.mp4", "still.png", "notes.txt"}, names)

    def test_media_kinds_are_still_classified(self):
        kinds = {f["name"]: f["kind"] for f in routes_output.list_output()["files"]}
        self.assertEqual("video", kinds["clip_swapped.mp4"])
        self.assertEqual("image", kinds["still.png"])
        self.assertEqual("file", kinds["notes.txt"])


if __name__ == "__main__":
    unittest.main()
