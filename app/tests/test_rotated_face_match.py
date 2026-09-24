"""Autorotate must re-detect THIS face, not whichever face is leftmost in the cut.

The rotated cut is padded 45% of the face size each side, so two faces in
contact are both in it. `process_face` took the leftmost detection and
`_unrotate_face_to_parent` wrote its kps, bbox, landmarks and embedding over
the target. On d2.mp4 (2026-09-24) one woman was aligned, swapped and
enhancer-stabilized from the other's keypoints, which painted a pale
hard-edged patch across her cheek.
"""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.ProcessMgr import ProcessMgr


class _Face:
    def __init__(self, bbox, kps=None):
        self.bbox = np.asarray(bbox, np.float32)
        self.kps = None if kps is None else np.asarray(kps, np.float32)


# A 200x100 frame-space cut starting at (100, 50), turned 180 degrees.
CUT_W, CUT_H, SX, SY = 200, 100, 100, 50


def _rotated_box(frame_box):
    """Where a frame-space box lands inside the 180-degree-rotated cut."""
    x0, y0, x1, y1 = frame_box
    return [CUT_W - 1 - (x1 - SX), CUT_H - 1 - (y1 - SY),
            CUT_W - 1 - (x0 - SX), CUT_H - 1 - (y0 - SY)]


class MatchRotatedFaceTest(unittest.TestCase):
    def test_picks_the_target_not_the_leftmost_neighbour(self):
        target = _Face([110, 60, 190, 140])           # left half of the cut
        neighbour_frame = [210, 60, 290, 140]          # right half: the other person
        # Rotated 180, the NEIGHBOUR is the leftmost detection in the cut.
        in_cut = [_Face(_rotated_box(neighbour_frame)), _Face(_rotated_box(target.bbox))]
        self.assertLess(in_cut[0].bbox[0], in_cut[1].bbox[0])
        got = ProcessMgr._match_rotated_face(target, in_cut, 'rotate_180',
                                             CUT_W, CUT_H, SX, SY)
        self.assertIs(got, in_cut[1])

    def test_declines_when_nothing_in_the_cut_is_this_face(self):
        target = _Face([110, 60, 190, 140])
        only_neighbour = [_Face(_rotated_box([210, 60, 290, 140]))]
        self.assertIsNone(ProcessMgr._match_rotated_face(
            target, only_neighbour, 'rotate_180', CUT_W, CUT_H, SX, SY))

    def test_unrotate_writes_the_matched_geometry_back_exactly(self):
        target = _Face([110, 60, 190, 140], kps=np.zeros((5, 2)))
        frame_kps = np.array([[130, 90], [170, 90], [150, 105], [135, 125], [165, 125]],
                             np.float32)
        cut_kps = np.stack([CUT_W - 1 - (frame_kps[:, 0] - SX),
                            CUT_H - 1 - (frame_kps[:, 1] - SY)], axis=1)
        rot = _Face(_rotated_box(target.bbox), kps=cut_kps)
        ProcessMgr._unrotate_face_to_parent(target, rot, 'rotate_180', CUT_W, CUT_H, SX, SY)
        np.testing.assert_allclose(target.kps, frame_kps, atol=1e-4)
        np.testing.assert_allclose(target.bbox, [110, 60, 190, 140], atol=1e-4)


if __name__ == "__main__":
    unittest.main()
