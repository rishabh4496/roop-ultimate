"""Does a face survive an occlusion, and does it stop surviving one in time?

Every test here drives `roop.tracker` directly with synthetic detections, so a
pass means the mechanism works -- NOT that a rendered clip improved. The
rendered evidence is `tests/occlusion_ground_truth.py`; these are the contracts
that harness cannot state.

The negative tests carry as much weight as the positive ones. A coasted face is
invented, carries the track's mean embedding, and therefore passes every
downstream identity gate by construction; the only thing between a prediction
and a swap painted on the background is the guard set, so each guard gets a test
that FAILS if the guard is removed.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.tracker import (MAX_COAST_FRAMES, MAX_LOST_FRAMES, STATE_COASTED,
                          FaceTracker)


def _face(cx, cy=100.0, width=40.0, height=52.0, embedding=(1.0, 0.0, 0.0, 0.0),
          score=0.9):
    return {
        'bbox': np.asarray((cx - width / 2, cy - height / 2,
                            cx + width / 2, cy + height / 2), dtype=np.float32),
        'embedding': np.asarray(embedding, dtype=np.float32),
        'kps': np.asarray([[cx - 8, cy - 8], [cx + 8, cy - 8], [cx, cy],
                           [cx - 6, cy + 10], [cx + 6, cy + 10]], dtype=np.float32),
        'det_score': np.float32(score),
    }


FRAME = (400, 800, 3)


class CoastingTest(unittest.TestCase):

    def _established(self, tracker, frames=6, step=5.0):
        """Walk a face right for `frames` frames so it is a real track."""
        for i in range(frames):
            tracker.update([_face(100.0 + step * i)], i)
        return frames

    def test_face_survives_a_detector_dropout(self):
        """The reported bug: one frame of detector silence must not blink the swap."""
        tracker = FaceTracker()
        n = self._established(tracker)
        tracker.update([], n)
        coasted = tracker.coast(n, frame_shape=FRAME)

        self.assertEqual(len(coasted), 1)
        face = coasted[0]
        self.assertTrue(face['_coasted'])
        self.assertEqual(face['occlusion_state'], STATE_COASTED)
        # Flagged as gap-filled too, so the existing swap audit counts it. A
        # coasted face invisible to that report is the defect this project
        # keeps finding.
        self.assertTrue(face['_interpolated'])
        self.assertEqual(face['_coast_age'], 1)

    def test_the_prediction_follows_the_motion_it_learned(self):
        """Constant velocity, so the coasted box must keep moving, not freeze."""
        tracker = FaceTracker()
        n = self._established(tracker, frames=8, step=5.0)
        last_cx = 100.0 + 5.0 * (n - 1)

        centres = []
        for k in range(3):
            tracker.update([], n + k)
            coasted = tracker.coast(n + k, frame_shape=FRAME)
            self.assertEqual(len(coasted), 1)
            box = coasted[0]['bbox']
            centres.append(float(box[0] + box[2]) * 0.5)

        self.assertGreater(centres[0], last_cx,
                           'a frozen box is not a prediction')
        self.assertLess(centres[0], centres[1])
        self.assertLess(centres[1], centres[2])

    def test_landmarks_travel_with_the_predicted_box(self):
        """The swap aligns from kps; a coasted face with stale kps is useless."""
        tracker = FaceTracker()
        n = self._established(tracker, frames=8, step=5.0)
        before = tracker.tracks[0].kps.copy()
        tracker.update([], n)
        face = tracker.coast(n, frame_shape=FRAME)[0]
        after = np.asarray(face['kps'], dtype=np.float32)

        self.assertEqual(after.shape, before.shape)
        self.assertGreater(float(after[:, 0].mean()), float(before[:, 0].mean()))
        # The box centre and the keypoint centroid must move together, or the
        # crop is built somewhere the landmarks are not.
        box = face['bbox']
        self.assertAlmostEqual(float(after[2, 0]),
                               float(box[0] + box[2]) * 0.5, delta=2.0)

    def test_coasting_stops_at_the_limit(self):
        """MAX_COAST_FRAMES is a hard stop, not a suggestion."""
        tracker = FaceTracker()
        n = self._established(tracker)
        produced = 0
        for k in range(MAX_COAST_FRAMES + 8):
            tracker.update([], n + k)
            produced += len(tracker.coast(n + k, frame_shape=FRAME))
        self.assertEqual(produced, MAX_COAST_FRAMES)
        self.assertGreater(tracker.stats['coast_expired'], 0)

    def test_the_track_outlives_the_coast(self):
        """Coasting and track lifetime answer different questions."""
        self.assertLess(MAX_COAST_FRAMES, MAX_LOST_FRAMES)
        tracker = FaceTracker()
        n = self._established(tracker)
        for k in range(MAX_COAST_FRAMES + 3):
            tracker.update([], n + k)
            tracker.coast(n + k, frame_shape=FRAME)
        # Stopped coasting, still available for re-association.
        self.assertIn(0, tracker.tracks)

    def test_the_track_is_retired_after_max_lost_frames(self):
        tracker = FaceTracker()
        n = self._established(tracker)
        for k in range(MAX_LOST_FRAMES + 2):
            tracker.update([], n + k)
        self.assertNotIn(0, tracker.tracks)
        self.assertGreater(tracker.stats['retired'], 0)

    def test_a_reappearing_face_resets_the_coast_budget(self):
        """A blink-out, a re-detection, then a second blink-out is two occlusions."""
        tracker = FaceTracker()
        n = self._established(tracker)
        for k in range(3):
            tracker.update([], n + k)
            tracker.coast(n + k, frame_shape=FRAME)
        tracker.update([_face(100.0 + 5.0 * (n + 3))], n + 3)
        self.assertEqual(tracker.tracks[0].coasted_run, 0)

        produced = 0
        for k in range(MAX_COAST_FRAMES + 2):
            tracker.update([], n + 4 + k)
            produced += len(tracker.coast(n + 4 + k, frame_shape=FRAME))
        self.assertEqual(produced, MAX_COAST_FRAMES)

    # -- the guards -----------------------------------------------------------

    def test_a_one_frame_track_is_never_coasted(self):
        """A single observation is as likely a false detection as a person."""
        tracker = FaceTracker()
        tracker.update([_face(100.0)], 0)
        tracker.update([], 1)
        self.assertEqual(tracker.coast(1, frame_shape=FRAME), [])
        self.assertGreater(tracker.stats['coast_refused_young'], 0)

    def test_a_face_that_walked_off_screen_is_not_coasted(self):
        """A face that LEFT is not a face that was covered."""
        tracker = FaceTracker()
        for i in range(8):
            tracker.update([_face(700.0 + 30.0 * i)], i)   # heading off the right edge
        produced = 0
        for k in range(6):
            tracker.update([], 8 + k)
            produced += len(tracker.coast(8 + k, frame_shape=FRAME))
        self.assertGreater(tracker.stats['coast_refused_outside'], 0)
        self.assertLess(produced, 6)

    def test_a_prediction_is_refused_where_a_real_face_already_is(self):
        """Two people in contact: never invent one on top of the other."""
        tracker = FaceTracker()
        n = self._established(tracker, frames=6, step=5.0)
        neighbour = _face(100.0 + 5.0 * n, embedding=(0.0, 1.0, 0.0, 0.0))
        tracker.update([], n)
        coasted = tracker.coast(n, frame_shape=FRAME, occupied=[neighbour])
        self.assertEqual(coasted, [])
        self.assertGreater(tracker.stats['coast_refused_collide'], 0)

    def test_confidence_decays_over_the_coast(self):
        """Anything reading det_score must see a prediction weakening."""
        tracker = FaceTracker()
        n = self._established(tracker)
        scores = []
        for k in range(5):
            tracker.update([], n + k)
            scores.append(float(tracker.coast(n + k, frame_shape=FRAME)[0]['det_score']))
        self.assertTrue(all(b < a for a, b in zip(scores, scores[1:])), scores)
        self.assertLess(scores[0], 0.9)

    def test_max_coast_zero_is_a_complete_no_op(self):
        """The disable switch has to actually disable, for a clean A/B arm."""
        tracker = FaceTracker(max_coast=0)
        n = self._established(tracker)
        for k in range(6):
            tracker.update([], n + k)
            self.assertEqual(tracker.coast(n + k, frame_shape=FRAME), [])
        self.assertEqual(tracker.stats['coasted'], 0)

    # -- backwards compatibility ---------------------------------------------

    def test_update_returns_only_what_it_was_given(self):
        """Existing callers must be unaffected by the upgrade.

        `ProcessMgr` and the Gradio path both rely on `update` returning the
        detection list. If coasting leaked into it, every caller would silently
        start swapping predictions whether or not it asked to.
        """
        tracker = FaceTracker()
        n = self._established(tracker)
        self.assertEqual(tracker.update([], n), [])
        self.assertEqual(len(tracker.update([_face(400.0)], n + 1)), 1)

    def test_face_analyser_still_exports_the_tracker(self):
        from roop import face_analyser, tracker
        self.assertIs(face_analyser.FaceTracker, tracker.FaceTracker)
        self.assertIs(face_analyser.FaceTrack, tracker.FaceTrack)

    def test_update_with_coasting_returns_both_lists(self):
        tracker = FaceTracker()
        n = self._established(tracker)
        allf, coasted = tracker.update_with_coasting([], n, frame_shape=FRAME)
        self.assertEqual(len(allf), 1)
        self.assertEqual(len(coasted), 1)
        self.assertIs(allf[0], coasted[0])


if __name__ == '__main__':
    unittest.main()
