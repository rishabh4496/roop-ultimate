"""Focused regressions for motion continuity, object occlusion, and identity routing."""

import os
import sys
import unittest

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.face_reference import EmbeddingSlidingWindow, MultiIdentityReferenceRouter, dual_threshold_match
from roop.processors.frame import face_swapper
from roop.tracker import FaceTracker


class TemporalContinuityRegression(unittest.TestCase):
    def test_large_mask_residual_is_not_a_raw_one_frame_pop(self):
        smoother = face_swapper.TemporalMaskSmoother(alpha=0.8)
        smoother._dense_flow = lambda current, previous: np.zeros(
            current.shape + (2,), dtype=np.float32)
        crop = np.zeros((64, 64, 3), dtype=np.uint8)
        first = np.zeros((64, 64), dtype=np.float32)
        second = np.ones((64, 64), dtype=np.float32)

        smoother.smooth(first, crop, track_id=7)
        out = smoother.smooth(second, crop, track_id=7)

        self.assertLess(float(out.mean()), 1.0)
        self.assertGreater(float(out.mean()), 0.75)

    def test_fast_track_coasts_beyond_normal_short_gap(self):
        tracker = FaceTracker(max_age=10, max_coast=2, min_hits_to_coast=1)
        face = {
            "bbox": np.asarray((20, 20, 60, 60), dtype=np.float32),
            "embedding": np.eye(1, 512, 0, dtype=np.float32).reshape(-1),
            "det_score": 0.95,
        }
        tracker.update([face], frame_index=0)
        tracker.tracks[0].state[4:6] = (6.0, 0.0)

        coast_runs = []
        for frame_index in (1, 2, 3):
            tracker.update([], frame_index=frame_index)
            coast_runs.append(tracker.coast(frame_index=frame_index))

        self.assertTrue(all(coast_runs))
        self.assertEqual(tracker.tracks[0].coasted_run, 3)


class ForeignObjectRegression(unittest.TestCase):
    def test_thin_non_skin_colored_object_gets_geometry_occlusion_signal(self):
        crop = np.full((256, 256, 3), 140, dtype=np.uint8)
        face_mask = np.zeros((256, 256), dtype=np.float32)
        cv2.ellipse(face_mask, (128, 128), (100, 110), 0, 0, 360, 1.0, -1)
        # Same approximate luminance as the background, but a narrow high-edge
        # object. The fallback must not depend on a fixed skin color.
        crop[124:132, 70:190] = (100, 150, 170)

        occ = face_swapper._heuristic_occlusion_mask(crop, face_mask=face_mask)
        object_region = occ[124:132, 80:180]
        forehead = occ[80:100, 110:145]

        self.assertGreater(float(object_region.mean()), 0.10)
        self.assertLess(float(forehead.mean()), 0.10)


class IdentityContinuityRegression(unittest.TestCase):
    @staticmethod
    def _unit(index):
        result = np.zeros(512, dtype=np.float32)
        result[index] = 1.0
        return result

    def test_one_contaminated_window_entry_cannot_win_by_max_similarity(self):
        identity = self._unit(0)
        other = self._unit(1)
        window = EmbeddingSlidingWindow(maxlen=8)
        for _ in range(7):
            window.add(identity)
        window.add(other)

        matched, score, _, reason = dual_threshold_match(
            other, identity, sliding_window=window)

        self.assertFalse(matched)
        self.assertLess(score, 0.62)
        self.assertEqual(reason, "below_threshold")

    def test_crossing_uses_track_position_and_freezes_identity_memory(self):
        identity_a = self._unit(0)
        identity_b = self._unit(1)
        router = MultiIdentityReferenceRouter({
            "a": {"embedding": identity_a},
            "b": {"embedding": identity_b},
        })
        first = [
            {"bbox": np.asarray((0, 0, 60, 60), dtype=np.float32),
             "embedding": identity_a, "_track_id": 1, "_emb_contam": 0.0},
            {"bbox": np.asarray((60, 0, 120, 60), dtype=np.float32),
             "embedding": identity_b, "_track_id": 2, "_emb_contam": 0.0},
        ]
        self.assertEqual(router.route(first, frame_index=0), ["a", "b"])
        lengths = {name: len(state.sliding_window)
                   for name, state in router.identities.items()}

        crossing = [
            {"bbox": np.asarray((20, 0, 80, 60), dtype=np.float32),
             "embedding": identity_b, "_track_id": 1, "_emb_contam": 0.4},
            {"bbox": np.asarray((40, 0, 100, 60), dtype=np.float32),
             "embedding": identity_a, "_track_id": 2, "_emb_contam": 0.4},
        ]
        self.assertEqual(router.route(crossing, frame_index=1), ["a", "b"])
        self.assertEqual(
            lengths,
            {name: len(state.sliding_window)
             for name, state in router.identities.items()},
        )


if __name__ == "__main__":
    unittest.main()
