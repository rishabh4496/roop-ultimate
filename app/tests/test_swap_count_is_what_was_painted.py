"""B-07: `swap_faces` reports the faces it PAINTS, not the faces it matched.

`num_faces_found` was bumped as each face was matched, so it also counted
  * a face the nth-face mode then refused for want of a source (`all_input`, more faces than sources),
  * a face whose source faceset holds no faces (`process_face` returns the frame untouched for it).
A frame in which nothing was painted therefore read as swapped to the no-face policy and to the audit.
These drive the real `ProcessMgr.swap_faces` with `process_face` replaced by a recorder.
"""
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("ROOP_SKIP_STARTUP", "1")


def _face(x):
    from insightface.app.common import Face
    f = Face()
    f.bbox = np.array([x, 10, x + 40, 60], np.float32)
    f.kps = np.array([[x + 10, 25], [x + 30, 25], [x + 20, 35], [x + 12, 48], [x + 28, 48]], np.float32)
    f.det_score = 0.9
    return f


def _source(n_faces):
    return types.SimpleNamespace(faces=[object() for _ in range(n_faces)])


def _manager(mode, sources, faces):
    from roop.ProcessMgr import ProcessMgr
    pm = ProcessMgr(progress=MagicMock())
    pm.options = MagicMock()
    pm.options.swap_mode = mode
    pm.options.selected_index = 0
    pm.input_face_datas = sources
    pm._temporal_mode = True
    pm._temporal_faces = {3: faces}
    pm._temporal_covered = 100
    return pm


class SwapCountIsWhatWasPainted(unittest.TestCase):
    def setUp(self):
        try:
            import insightface.app.common  # noqa: F401
        except Exception as exc:
            self.skipTest("insightface not importable here: %s" % type(exc).__name__)
        self.frame = np.full((120, 200, 3), 90, np.uint8)

    def _run(self, pm):
        painted = []

        def record(self_, src_index, face, temp_frame, plate=None, region=None):
            painted.append(src_index)
            return temp_frame

        with patch.object(type(pm), "process_face", record):
            found, _ = pm.swap_faces(self.frame, frame_idx=3)
        return found, painted

    def test_an_empty_source_faceset_paints_nothing_and_counts_nothing(self):
        pm = _manager("all", [_source(0)], [_face(10)])
        found, painted = self._run(pm)
        self.assertEqual((found, painted), (0, []))

    def test_first_found_with_an_empty_source_counts_nothing(self):
        pm = _manager("first", [_source(0)], [_face(10)])
        found, painted = self._run(pm)
        self.assertEqual((found, painted), (0, []))

    def test_a_usable_source_still_counts(self):
        pm = _manager("all", [_source(1)], [_face(10), _face(100)])
        found, painted = self._run(pm)
        self.assertEqual(found, 2)
        self.assertEqual(painted, [0, 0])

    def test_all_input_refuses_the_faces_it_has_no_source_for(self):
        # Two people, one source: the second is refused. It used to be counted as swapped anyway.
        pm = _manager("all_input", [_source(1)], [_face(10), _face(100)])
        found, painted = self._run(pm)
        self.assertEqual(found, 1)
        self.assertEqual(painted, [0])

    def test_all_input_skips_only_the_empty_source(self):
        pm = _manager("all_input", [_source(0), _source(2)], [_face(10), _face(100)])
        found, painted = self._run(pm)
        self.assertEqual(found, 1)
        self.assertEqual(painted, [1])


if __name__ == "__main__":
    unittest.main()
