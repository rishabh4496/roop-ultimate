"""tools/compare_recognition_swap.py: the decision metrics, track stitching and compositing maths (no GPU, no video)."""
import os
import sys
import unittest

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(APP, "tools"))
import compare_recognition_swap as c  # noqa: E402


def box(x, y=0, w=100, h=100):
    return [x, y, x + w, y + h]


class TestMetrics(unittest.TestCase):
    def test_double_match_counts_only_distinct_faces_in_one_frame(self):
        frames = np.array([0, 0, 1, 1, 2, 2, 3])
        boxes = np.array([box(0), box(300), box(0), box(300), box(0), box(10), box(0)], np.float32)
        label = np.array([1, 1, 1, 0, 1, 1, 1], bool)
        rate, bad, of = c.double_match_rate(frames, boxes, label)
        self.assertEqual((bad, of), (1, 2))                       # frame 0 violates; frame 1 fine; frame 2 is a duplicate pair; frame 3 has one face
        self.assertAlmostEqual(rate, 0.5)

    def test_double_match_with_no_multi_face_frames(self):
        self.assertEqual(c.double_match_rate(np.array([0, 1]), np.array([box(0), box(0)], np.float32), np.array([1, 1], bool)), (0.0, 0, 0))

    def test_flips_ignore_short_tracks_and_normalise_per_thousand(self):
        label = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + [0, 1, 0], bool)
        long_alternating, long_steady, short = list(range(0, 10)), list(range(10, 20)), [20, 21, 22]
        rate, flips, faces = c.flips_per_1000([long_alternating, long_steady, short], label)
        self.assertEqual((flips, faces), (9, 20))
        self.assertAlmostEqual(rate, 450.0)

    def test_track_recall_excludes_the_exemplars(self):
        label = np.array([1, 1, 1, 0, 0, 1], bool)
        self.assertEqual(c.track_recall([0, 1, 2, 3, 4, 5], exclude=[0, 1], label=label), (0.5, 4))
        self.assertEqual(c.track_recall([0, 1], exclude=[0, 1], label=label), (0.0, 0))

    def test_agreement(self):
        self.assertEqual(c.agreement(np.array([1, 0, 1, 1], bool), np.array([1, 1, 1, 0], bool)), 0.5)


class TestTracks(unittest.TestCase):
    def test_stitching_joins_across_a_short_gap_but_not_across_a_cut_or_a_jump(self):
        frames = np.array([0, 1, 2, 5, 6, 7, 20, 21])
        boxes = np.array([box(0)] * 6 + [box(0), box(0)], np.float32)
        cut = np.zeros(30, bool)
        tracks = [[0, 1, 2], [3, 4, 5], [6, 7]]
        joined = c.stitch_tracks(tracks, frames, boxes, cut)
        self.assertEqual(sorted(len(t) for t in joined), [2, 6])         # 0-2 and 5-7 join (gap 3); 20-21 is >5 frames away
        cut[4] = True
        self.assertEqual(sorted(len(t) for t in c.stitch_tracks(tracks, frames, boxes, cut)), [2, 3, 3])
        far = np.array([box(0)] * 3 + [box(500)] * 3 + [box(0)] * 2, np.float32)
        self.assertEqual(sorted(len(t) for t in c.stitch_tracks(tracks, frames, far, np.zeros(30, bool))), [2, 3, 3])

    def test_pick_subject_takes_the_longest_track_of_large_faces_and_spaces_exemplars(self):
        boxes = np.array([box(0, w=120, h=120)] * 40 + [box(500, w=30, h=30)] * 60, np.float32)
        det = np.full(100, 0.9)
        tracks = [list(range(40)), list(range(40, 100))]                 # the longer track is made of tiny faces
        k, ex = c.pick_subject(tracks, boxes, det, min_px=80, n_exemplars=8)
        self.assertEqual(k, 0)
        self.assertEqual(len(ex), 8)
        self.assertEqual(ex, sorted(ex))
        self.assertTrue(set(ex) <= set(range(40)))
        self.assertGreater(ex[-1] - ex[0], 30)                            # spread over the track, not the first eight

    def test_rows_by_frame(self):
        self.assertEqual(c.rows_by_frame(np.array([2, 2, 5])), {2: [0, 1], 5: [2]})


class TestCompositing(unittest.TestCase):
    def test_roi_box_is_expanded_and_clamped(self):
        self.assertEqual(c.roi_box([100, 100, 200, 200], 1280, 720, 0.25), (75, 75, 225, 225))
        self.assertEqual(c.roi_box([0, 0, 100, 100], 1280, 720, 0.25), (0, 0, 125, 125))
        self.assertEqual(c.roi_box([1200, 650, 1280, 720], 1280, 720, 0.25), (1180, 632, 1280, 720))

    def test_feather_mask_is_soft_and_centred(self):
        m = c.feather_mask(120, 100)[:, :, 0]
        self.assertGreater(m[60, 50], 0.99)
        self.assertLess(m[0, 0], 0.01)
        self.assertTrue(0.05 < m[60, 5] < 0.95 or m[60, 5] < 0.5)       # the edge is a gradient, not a step

    def test_paste_changes_the_centre_and_leaves_the_corners_alone(self):
        base = np.full((200, 200, 3), 50, np.uint8)
        roi = np.full((100, 100, 3), 250, np.uint8)
        out = c.paste_roi(base.copy(), roi, 50, 50)
        self.assertGreater(int(out[100, 100, 0]), 240)
        self.assertEqual(int(out[55, 55, 0]), 50)
        self.assertEqual(int(out[10, 10, 0]), 50)
        np.testing.assert_array_equal(out[:40], base[:40])


class TestSpeed(unittest.TestCase):
    def test_pipeline_fps_combines_detection_and_recognition(self):
        det = np.full(100, 0.020)                                           # 20 ms per frame
        rec = {"fast": np.full(200, 0.001), "slow": np.full(200, 0.004)}    # 2 faces a frame
        st = c.speed_table(det, rec, skip=10)
        self.assertAlmostEqual(st["fast"]["ms_per_face"], 1.0)
        self.assertAlmostEqual(st["fast"]["faces_per_s"], 1000.0)
        self.assertAlmostEqual(st["fast"]["pipeline_fps"], 1.0 / (0.020 + 0.002), places=6)
        self.assertAlmostEqual(st["slow"]["pipeline_fps"], 1.0 / (0.020 + 0.008), places=6)

    def test_configuration_is_consistent(self):
        self.assertEqual(set(c.THRESHOLD), set(c.MODELS))
        self.assertEqual(set(c.TITLE), set(c.MODELS))
        self.assertNotIn("antelopev2", c.DISTINCT)
        self.assertEqual(c.THRESHOLD["antelopev2"], c.THRESHOLD["glintr100"])
        from roop.recognition_registry import get_registered_models
        self.assertEqual(set(c.MODELS), set(get_registered_models()))


if __name__ == "__main__":
    unittest.main()
