"""Stage 13 asked for a "no-face passthrough": frames the pre-pass marked face-free should skip every GPU stage.

That path already exists in `ProcessMgr.swap_faces` (the `_temporal_faces` cache, ProcessMgr.py ~3743-3752: inside the covered
range a missing entry means nobody was there, so no detector, matcher, swapper, mask or enhancer runs). These tests drive it,
so "exists" rests on a path that executes rather than on a reading (AGENTS.md, "Never mark a stage ... on a reading").
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _manager(covered, faces):
    from roop.ProcessMgr import ProcessMgr
    pm = ProcessMgr(progress=MagicMock())
    pm.options = MagicMock()
    pm.options.swap_mode = "all"
    pm._temporal_mode = True
    pm._temporal_faces = faces
    pm._temporal_covered = covered
    return pm


class NoFacePassthrough(unittest.TestCase):
    def _run(self, pm, frame, idx):
        boom = MagicMock(side_effect=AssertionError("a GPU/analysis stage ran on a face-free frame"))
        with patch("roop.ProcessMgr.get_all_faces", boom), patch("roop.ProcessMgr.get_first_face", boom), \
                patch.object(type(pm), "process_face", boom):
            result = pm.swap_faces(frame, frame_idx=idx)
        self.assertEqual(boom.call_count, 0)
        return result

    def test_covered_frame_without_cache_entry_returns_the_frame_untouched(self):
        pm = _manager(covered=100, faces={7: [MagicMock()]})
        frame = np.full((72, 128, 3), 77, np.uint8)
        before = frame.copy()
        found, out = self._run(pm, frame, 5)
        self.assertEqual(found, 0)
        self.assertTrue(out is frame)
        self.assertTrue(np.array_equal(out, before))

    def test_frame_past_the_covered_range_is_NOT_skipped(self):
        """Past the pre-pass range a missing entry means 'never looked at', so detection must still run."""
        pm = _manager(covered=10, faces={})
        frame = np.zeros((72, 128, 3), np.uint8)
        with patch("roop.ProcessMgr.get_all_faces", return_value=[]) as detect:
            pm.swap_faces(frame, frame_idx=50)
        self.assertGreaterEqual(detect.call_count, 1)


if __name__ == "__main__":
    unittest.main()
