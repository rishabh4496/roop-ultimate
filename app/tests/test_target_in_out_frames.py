"""In/Out frame numbering at /api/target/set_frame and in the target list.

Clients show 1-based, inclusive frames (the frame /api/preview decodes, and a queue job's
frame_start/frame_end). The engine stores a 0-based seek (CAP_PROP_POS_FRAMES) plus an exclusive end
and renders [startframe, endframe). set_frame used to store In raw as the seek, so every render
started one frame after the In point the user picked: measured on the live engine 2026-11-28,
In 300 / Out 329 wrote 29 frames whose first matched source frame 301, not 300.
"""

import os
import sys
import unittest
from unittest import mock

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

import api  # noqa: E402


class _Entry:
    def __init__(self, frames=1000):
        self.filename = "clip.mp4"
        self.media_id = "m-inout"
        self.startframe = 0
        self.endframe = frames
        self.total_frames = frames
        self.fps = 30


class InOutFrames(unittest.TestCase):
    def setUp(self):
        self.old_entries = list(api.list_files_process)
        api.list_files_process.clear()
        self.entry = _Entry()
        api.list_files_process.append(self.entry)
        p1 = mock.patch.object(api, "_activate_target_from_payload", return_value=(0, "m-inout", None))
        p2 = mock.patch.object(api, "_target_list_payload", return_value={"ok": True})
        p1.start(); p2.start()
        self.addCleanup(p1.stop); self.addCleanup(p2.stop)

    def tearDown(self):
        api.list_files_process.clear()
        api.list_files_process.extend(self.old_entries)

    def _set(self, which, frame):
        api.target_set_frame({"which": which, "frame": frame, "target_media_id": "m-inout"})

    def test_in_point_is_the_frame_on_screen(self):
        self._set("start", 300)
        self._set("end", 329)
        self.assertEqual(self.entry.startframe, 299, "In 300 must seek to 0-based 299 (UI frame 300)")
        self.assertEqual(self.entry.endframe, 329)
        self.assertEqual(self.entry.endframe - self.entry.startframe, 30, "In 300..Out 329 is 30 frames")

    def test_from_the_first_frame(self):
        self.entry.startframe = 50
        self._set("start", 1)
        self.assertEqual(self.entry.startframe, 0)
        self._set("start", 0)  # tools send 0 to mean "from the start"
        self.assertEqual(self.entry.startframe, 0)

    def test_in_cannot_pass_out_and_out_cannot_pass_in(self):
        self._set("end", 100)
        self._set("start", 500)
        self.assertEqual(self.entry.startframe, 99, "In clamps to the Out frame: one frame left")
        self._set("end", 10)
        self.assertEqual(self.entry.endframe, 100, "Out never ends before In's frame")
        self.assertGreaterEqual(self.entry.endframe - self.entry.startframe, 1)

    def test_out_clamps_to_the_clip(self):
        self._set("end", 5000)
        self.assertEqual(self.entry.endframe, 1000)

    def test_target_list_reports_1_based_inclusive(self):
        self.entry.startframe = 299
        self.entry.endframe = 329
        d = api._target_entry_dict(self.entry)
        self.assertEqual((d["start_frame"], d["end_frame"]), (300, 329))
        self.assertEqual((d["startframe"], d["endframe"]), (299, 329), "the raw engine values stay as they are")
        self.entry.startframe = 0
        self.assertEqual(api._target_entry_dict(self.entry)["start_frame"], 1)


if __name__ == "__main__":
    unittest.main()
