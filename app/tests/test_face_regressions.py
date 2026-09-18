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
from roop.procmgr_tracking import is_synthetic_face
from roop.ProcessMgr import _restore_partial_occlusion


class TemporalContinuityRegression(unittest.TestCase):
    def test_synthetic_temporal_faces_are_not_swap_candidates(self):
        self.assertTrue(is_synthetic_face({'_interpolated': True}))
        self.assertTrue(is_synthetic_face({'_coasted': True}))
        self.assertFalse(is_synthetic_face({'_track_id': 7}))

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
    def test_partial_occlusion_restores_target_face_hull(self):
        angles = np.linspace(0.0, 2.0 * np.pi, 106, endpoint=False)
        landmarks = np.column_stack((
            64.0 + 42.0 * np.cos(angles),
            64.0 + 52.0 * np.sin(angles),
        )).astype(np.float32)
        face = {
            "occlusion_state": "partial",
            "landmark_2d_106": landmarks,
            "kps": np.asarray(
                ((48, 50), (80, 50), (64, 64), (52, 82), (76, 82)),
                dtype=np.float32,
            ),
            "bbox": np.asarray((20, 12, 108, 116), dtype=np.float32),
        }
        generated = np.zeros((128, 128, 3), dtype=np.uint8)
        plate = np.full_like(generated, 255)

        restored = _restore_partial_occlusion(generated, plate, face)

        self.assertGreater(int(restored[64, 64, 0]), 200)
        self.assertEqual(int(restored[4, 4, 0]), 0)

    def test_partial_occlusion_keeps_foreground_object_from_plate(self):
        angles = np.linspace(0.0, 2.0 * np.pi, 106, endpoint=False)
        landmarks = np.column_stack((
            64.0 + 42.0 * np.cos(angles),
            64.0 + 52.0 * np.sin(angles),
        )).astype(np.float32)
        face = {
            "occlusion_state": "partial",
            "landmark_2d_106": landmarks,
            "kps": np.asarray(
                ((48, 50), (80, 50), (64, 64), (52, 82), (76, 82)),
                dtype=np.float32,
            ),
            "bbox": np.asarray((20, 12, 108, 116), dtype=np.float32),
        }
        generated = np.zeros((128, 128, 3), dtype=np.uint8)
        plate = np.full_like(generated, 255)
        plate[60:68, 20:108] = (7, 11, 13)

        restored = _restore_partial_occlusion(generated, plate, face)

        self.assertEqual(tuple(int(v) for v in restored[64, 64]), (7, 11, 13))

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

    def test_missing_or_broad_face_mask_does_not_erase_face_detail(self):
        crop = np.full((256, 256, 3), 140, dtype=np.uint8)
        cv2.ellipse(crop, (128, 128), (90, 105), 0, 0, 360,
                    (130, 150, 175), -1)
        cv2.ellipse(crop, (98, 118), (12, 5), 0, 0, 360,
                    (25, 25, 25), -1)
        cv2.ellipse(crop, (158, 118), (12, 5), 0, 0, 360,
                    (25, 25, 25), -1)
        cv2.ellipse(crop, (128, 165), (28, 8), 0, 0, 360,
                    (25, 25, 25), 2)

        no_mask = face_swapper._heuristic_occlusion_mask(crop)
        broad_mask = face_swapper._heuristic_occlusion_mask(
            crop, face_mask=np.ones((256, 256), dtype=np.float32))

        self.assertLess(float(np.mean(no_mask > 0.2)), 0.05)
        self.assertLess(float(np.mean(broad_mask > 0.2)), 0.05)


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
