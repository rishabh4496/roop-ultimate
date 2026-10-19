"""Stage 16: the pre-pass association cost.

`legacy_cost` is the cost `_consume` has always used; these pin its four
branches so lifting it out of the closure cannot have changed it. `composite_cost`
is the candidate. Each composite test names the failure it prevents.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.association_cost import (CostWeights, composite_cost, giou,
                                   identity_reliability, iou, legacy_cost,
                                   propagate_kps)

IOU_MIN, EMB_MAX = 0.2, 0.7


class LegacyCostTest(unittest.TestCase):

    def test_clean_face_adds_appearance(self):
        self.assertAlmostEqual(legacy_cost(0.8, 0.1, False, IOU_MIN, EMB_MAX), 0.3)

    def test_dirty_face_is_appearance_blind(self):
        self.assertAlmostEqual(legacy_cost(0.8, 0.9, True, IOU_MIN, EMB_MAX), 0.2)

    def test_low_iou_is_not_a_candidate(self):
        self.assertIsNone(legacy_cost(0.19, 0.0, False, IOU_MIN, EMB_MAX))
        self.assertIsNone(legacy_cost(0.19, 0.0, True, IOU_MIN, EMB_MAX))

    def test_appearance_gate_applies_to_clean_faces_only(self):
        self.assertIsNone(legacy_cost(0.9, 0.71, False, IOU_MIN, EMB_MAX))
        self.assertIsNotNone(legacy_cost(0.9, 0.71, True, IOU_MIN, EMB_MAX))


class GeometryTest(unittest.TestCase):

    def test_giou_ranks_disjoint_boxes_by_distance(self):
        a = (0, 0, 10, 10)
        near, far = (12, 0, 22, 10), (60, 0, 70, 10)
        self.assertEqual(iou(a, near), 0.0)
        self.assertEqual(iou(a, far), 0.0)            # IoU cannot tell them apart
        self.assertGreater(giou(a, near), giou(a, far))

    def test_giou_identical_is_one(self):
        self.assertAlmostEqual(giou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_propagate_kps_follows_the_box(self):
        k = np.array([[10, 10], [20, 10], [15, 15], [11, 20], [19, 20]], np.float32)
        last = np.array([0, 0, 30, 30], np.float32)
        moved = np.array([100, 50, 130, 80], np.float32)          # translate only
        out = propagate_kps(k, last, moved)
        np.testing.assert_allclose(out, k + [100, 50], atol=1e-4)
        scaled = np.array([0, 0, 60, 60], np.float32)             # 2x
        np.testing.assert_allclose(propagate_kps(k, last, scaled), k * 2, atol=1e-4)

    def test_propagate_kps_without_inputs(self):
        self.assertIsNone(propagate_kps(None, np.zeros(4), np.ones(4)))


def _kps(cx, cy, s=10.0):
    return np.array([[cx - s, cy - s], [cx + s, cy - s], [cx, cy],
                     [cx - s * .6, cy + s], [cx + s * .6, cy + s]], np.float32)


class CompositeCostTest(unittest.TestCase):

    BOX = (100.0, 100.0, 160.0, 170.0)

    def _cost(self, **kw):
        base = dict(det_bbox=self.BOX, det_kps=_kps(130, 135), det_pose=np.array([0, 0, 0.]),
                    cos_dist=0.2, contam=0.0, dirty=False, prior_bbox=self.BOX,
                    prior_kps=_kps(130, 135), prior_pose=np.array([0, 0, 0.]),
                    iou_min=IOU_MIN, emb_max=EMB_MAX)
        base.update(kw)
        return composite_cost(**base)

    def test_a_perfect_prior_costs_only_the_identity_term(self):
        self.assertAlmostEqual(self._cost(), 0.2 * CostWeights().identity, places=5)

    def test_dirty_face_is_no_longer_appearance_blind(self):
        """The defect: two tracks with the same overlap, one the right person.
        Legacy cost ties them; composite must prefer the identity-closer one."""
        right = self._cost(dirty=True, contam=0.25, cos_dist=0.35)
        wrong = self._cost(dirty=True, contam=0.25, cos_dist=0.85)
        self.assertLess(right, wrong)
        self.assertEqual(legacy_cost(0.8, 0.35, True, IOU_MIN, EMB_MAX),
                         legacy_cost(0.8, 0.85, True, IOU_MIN, EMB_MAX))

    def test_identity_signal_fades_with_contamination(self):
        w = CostWeights()
        spread = lambda c: (self._cost(dirty=True, contam=c, cos_dist=0.85)
                            - self._cost(dirty=True, contam=c, cos_dist=0.35))
        self.assertGreater(spread(0.25), spread(0.45))
        self.assertGreater(spread(0.45), spread(0.7))
        self.assertGreater(spread(0.7), 0.0)             # floor: never fully blind
        self.assertEqual(identity_reliability(0.0, w), 1.0)
        self.assertEqual(identity_reliability(1.0, w), w.contam_floor)

    def test_gates_match_legacy(self):
        far = (400.0, 400.0, 460.0, 470.0)
        self.assertIsNone(self._cost(prior_bbox=far))
        self.assertIsNone(self._cost(cos_dist=0.75))
        self.assertIsNotNone(self._cost(cos_dist=0.75, dirty=True, contam=0.5))

    def test_keypoint_term_separates_two_equal_boxes(self):
        near = self._cost(prior_kps=_kps(131, 135))
        off = self._cost(prior_kps=_kps(150, 150))
        self.assertLess(near, off)

    def test_pose_term_separates_a_turned_head(self):
        same = self._cost(prior_pose=np.array([2.0, 0, 0]))
        turned = self._cost(prior_pose=np.array([70.0, 0, 0]))
        self.assertLess(same, turned)

    def test_missing_kps_or_pose_degrade_to_the_remaining_terms(self):
        full = self._cost()
        self.assertAlmostEqual(self._cost(det_kps=None), full, places=6)
        self.assertAlmostEqual(self._cost(prior_pose=None), full, places=6)
        self.assertAlmostEqual(self._cost(det_pose=np.array([np.nan, 0, 0.])), full, places=6)

    def test_zero_weights_reduce_to_identity_plus_giou(self):
        w = CostWeights(kpt=0.0, pose=0.0)
        a = self._cost(w=w, prior_kps=_kps(150, 150), prior_pose=np.array([80., 0, 0]))
        b = self._cost(w=w)
        self.assertAlmostEqual(a, b, places=6)


if __name__ == '__main__':
    unittest.main()
