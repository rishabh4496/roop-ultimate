"""Gap-fill continuity and claim order — the two rules that stop a swap being
painted onto the background while the real face goes unswapped.

The temporal pre-pass fills a track's detection misses by interpolating between
the observations either side of the gap, gated only on the gap LENGTH. The
scan's primary association is IoU-gated and so cannot teleport, but its Re-ID
fallback matches on embedding alone with no spatial constraint — so the two
anchors can sit on opposite sides of the frame, and filling that in manufactures
a face for every frame in between, sliding across the background.

An invented face defeats every identity check downstream by construction: its
embedding IS the track mean, so its distance to the track is 0 and its distance
to the captured target is the track's own. It is swapped wherever it was put,
and since a source can only be used once per frame, the real face in that frame
is then refused. Both halves are asserted here:

  * _bridgeable accepts ordinary motion and refuses a teleport;
  * the claim order puts real detections, closest first, ahead of gap-filled
    ones, so a false detection cannot take the source from the real face.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.procmgr_tracking import TrackingMixin              # noqa: E402
from roop.procmgr_runtime import _INTERP_MAX_TRAVEL          # noqa: E402


class _Obs(dict):
    """Stand-in for an insightface Face: attribute + dict access over a bbox."""

    def __init__(self, x, y, w=100.0):
        super().__init__()
        self['bbox'] = np.array([x, y, x + w, y + w], dtype=np.float32)

    def __getattr__(self, name):
        return self[name]


bridgeable = TrackingMixin._bridgeable


class BridgeableTest(unittest.TestCase):

    def test_still_face_bridges(self):
        """A face that did not move across a 5-frame miss is the ordinary case."""
        self.assertTrue(bridgeable(_Obs(400, 300), _Obs(400, 300), 5))

    def test_normal_motion_bridges(self):
        """Half a face width per frame is already fast; it must still bridge."""
        span = 6
        travel = _INTERP_MAX_TRAVEL * span * 100.0 * 0.9      # 90% of the budget
        self.assertTrue(bridgeable(_Obs(400, 300), _Obs(400 + travel, 300), span))

    def test_teleport_is_refused(self):
        """A Re-ID reconnection on the far side of the frame is not motion."""
        self.assertFalse(bridgeable(_Obs(1500, 300), _Obs(120, 500), 4))

    def test_budget_scales_with_gap_length(self):
        """The same jump is implausible over 2 frames and fine over 10."""
        jump = _INTERP_MAX_TRAVEL * 6 * 100.0
        self.assertFalse(bridgeable(_Obs(0, 0), _Obs(jump, 0), 2))
        self.assertTrue(bridgeable(_Obs(0, 0), _Obs(jump, 0), 10))

    def test_budget_scales_with_face_size(self):
        """A big (close-up) face may cross more pixels than a small one."""
        jump = 300.0
        self.assertFalse(bridgeable(_Obs(0, 0, w=40), _Obs(jump, 0, w=40), 4))
        self.assertTrue(bridgeable(_Obs(0, 0, w=400), _Obs(jump, 0, w=400), 4))

    def test_size_jump_is_refused(self):
        """A track that swaps a close-up for a distant face changed subject."""
        self.assertFalse(bridgeable(_Obs(400, 300, w=300), _Obs(400, 300, w=40), 3))

    def test_missing_geometry_does_not_refuse(self):
        """No bbox to judge on → behave as before rather than drop the fill."""
        self.assertTrue(bridgeable(object(), object(), 3))


class ClaimOrderTest(unittest.TestCase):
    """The ordering ProcessMgr applies before faces claim a source.

    Mirrors the sort key at its use site: real detections before gap-filled
    ones, then closest to a captured target first.
    """

    @staticmethod
    def _order(faces, dist):
        return sorted(faces, key=lambda f: (bool(f.get('_interpolated')), dist[id(f)]))

    def test_real_face_claims_before_a_false_detection(self):
        real, blob = {}, {}
        dist = {id(real): 0.38, id(blob): 0.91}
        self.assertIs(self._order([blob, real], dist)[0], real)

    def test_real_face_outranks_a_gap_filled_one_that_scores_better(self):
        """A gap-filled face carries the TRACK MEAN as its embedding, so it
        scores better than any real frame of the same person. It must still not
        take the source away from a real detection."""
        real = {}
        filled = {'_interpolated': True}
        dist = {id(real): 0.62, id(filled): 0.05}
        self.assertIs(self._order([filled, real], dist)[0], real)

    def test_order_is_a_permutation(self):
        """Ordering only — nothing is added or dropped."""
        faces = [{}, {'_interpolated': True}, {}]
        dist = {id(f): d for f, d in zip(faces, (0.9, 0.1, 0.4))}
        out = self._order(faces, dist)
        self.assertEqual(len(out), len(faces))
        self.assertEqual({id(f) for f in out}, {id(f) for f in faces})


if __name__ == '__main__':
    unittest.main()
