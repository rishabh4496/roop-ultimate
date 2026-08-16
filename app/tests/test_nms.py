"""Non-max suppression: one face detected twice vs two faces touching.

Plain NMS answers both questions with one IoU number, and at the 0.40 default
the second one loses — a head partly behind another at similar scale exceeds the
threshold and is deleted outright. It is never detected, so it is never tracked,
matched or swapped: the swap is simply absent for those frames, and nothing
downstream can recover it.

The shared rule additionally requires the two boxes to be CONCENTRIC before one
suppresses the other. Two tests carry the fix (offset boxes survive), and the
rest exist to stop it becoming a duplicate-box generator: an equivalence test
pins the refactor as behaviour-preserving with the rule off, and the geometry
cases pin how far the rule can widen suppression.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.nms import nms_keep, bind_instance_nms, CENTER_FRAC   # noqa: E402


def _box(x, y, size=100.0, score=0.9):
    return [x, y, x + size, y + size, score]


def _reference_nms(dets, thresh, offset):
    """The implementation that was in retinaface/scrfd/yoloface, verbatim in
    behaviour, to prove the shared one is a refactor and not a rewrite."""
    dets = np.asarray(dets, np.float32)
    x1, y1, x2, y2, scores = (dets[:, 0], dets[:, 1], dets[:, 2],
                              dets[:, 3], dets[:, 4])
    areas = (x2 - x1 + offset) * (y2 - y1 + offset)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + offset)
        h = np.maximum(0.0, yy2 - yy1 + offset)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][ovr <= thresh]
    return keep


def _iou(a, b, offset=0.0):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1) + offset)
    ih = max(0.0, min(ay2, by2) - max(ay1, by1) + offset)
    inter = iw * ih
    ua = ((ax2 - ax1 + offset) * (ay2 - ay1 + offset)
          + (bx2 - bx1 + offset) * (by2 - by1 + offset) - inter)
    return inter / ua


class TouchingFacesTest(unittest.TestCase):
    """The fix. Both of these are deleted by plain NMS at 0.40."""

    def test_two_offset_faces_both_survive(self):
        """One head partly behind another: same scale, centres a third of a face
        apart, IoU just over 0.5 — comfortably past the 0.40 threshold."""
        a, b = _box(0, 0, score=0.95), _box(33, 0, score=0.90)
        self.assertGreater(_iou(a, b), 0.40, 'fixture must actually trip NMS')
        self.assertEqual(len(nms_keep([a, b], 0.40)), 2)

    def test_plain_nms_would_have_dropped_it(self):
        """Load-bearing: the same pair with the rule disabled loses a face."""
        a, b = _box(0, 0, score=0.95), _box(33, 0, score=0.90)
        self.assertEqual(len(nms_keep([a, b], 0.40, center_frac=0)), 1)

    def test_vertical_offset_counts_the_same(self):
        """Separation is a distance, not an x-offset — a face below another
        must be treated like one beside it."""
        a, b = _box(0, 0, score=0.95), _box(0, 33, score=0.90)
        self.assertEqual(len(nms_keep([a, b], 0.40)), 2)

    def test_the_higher_scoring_face_is_never_the_one_dropped(self):
        a, b = _box(0, 0, score=0.60), _box(10, 0, score=0.99)
        keep = nms_keep([a, b], 0.40)
        self.assertEqual(keep[0], 1)


class DuplicateBoxesStillCollapseTest(unittest.TestCase):
    """The property the threshold exists for. Breaking these means two swaps
    painted on one face."""

    def test_identical_boxes(self):
        self.assertEqual(len(nms_keep([_box(0, 0, score=0.9),
                                       _box(0, 0, score=0.8)], 0.40)), 1)

    def test_near_coincident_boxes(self):
        """Adjacent anchors firing on one face: a few percent apart."""
        self.assertEqual(len(nms_keep([_box(0, 0, score=0.9),
                                       _box(6, 4, score=0.8)], 0.40)), 1)

    def test_concentric_boxes_of_different_size(self):
        """Multi-scale anchors: a smaller box inside a larger one, same centre."""
        big = _box(0, 0, size=100, score=0.9)
        small = _box(15, 15, size=70, score=0.8)
        self.assertEqual(len(nms_keep([big, small], 0.40)), 1)

    def test_a_contained_box_collapses_however_it_is_placed(self):
        """The other route to a high IoU is one box inside another. Pushed
        diagonally into a corner — the worst case, maximising the separation —
        it must still collapse once the overlap is real."""
        big = _box(0, 0, size=100, score=0.9)
        corner = _box(25, 25, size=75, score=0.8)          # r = 0.75, IoU 0.56
        self.assertGreater(_iou(big, corner), 0.50)
        self.assertEqual(len(nms_keep([big, corner], 0.40)), 1)

    def test_the_containment_bound_and_where_it_stops_holding(self):
        """Honest limit, pinned so it cannot drift. For a contained box of
        relative size r the largest reachable separation is sqrt(2)(1-r)/(1+r),
        and IoU > 0.40 only needs r > 0.63 — which allows ~0.32. So a
        corner-pushed contained box DOES survive in the narrow band between IoU
        0.40 and 0.49, and cannot above it. Both ends asserted."""
        big = _box(0, 0, size=100, score=0.9)
        escapes = _box(35, 35, size=65, score=0.8)          # r = 0.65, IoU 0.42
        self.assertGreater(_iou(big, escapes), 0.40)
        self.assertLess(_iou(big, escapes), 0.49)
        self.assertEqual(len(nms_keep([big, escapes], 0.40)), 2)

        closed = _box(30, 30, size=70, score=0.8)           # r = 0.70, IoU 0.49
        self.assertGreaterEqual(_iou(big, closed), 0.48)
        self.assertEqual(len(nms_keep([big, closed], 0.40)), 1)


class WideningIsBoundedTest(unittest.TestCase):

    def test_heavy_overlap_still_collapses(self):
        """The rule widens the effective threshold to ~0.60 for offset pairs,
        not to infinity: two boxes sharing this much area are one face."""
        a, b = _box(0, 0, score=0.9), _box(18, 0, score=0.8)
        self.assertGreater(_iou(a, b), 0.60)
        self.assertEqual(len(nms_keep([a, b], 0.40)), 1)

    def test_separation_is_measured_in_face_widths(self):
        """The same geometry at a different scale must give the same answer, or
        the rule would behave differently on close-ups and distant faces."""
        small = nms_keep([_box(0, 0, size=40, score=0.9),
                          _box(13.2, 0, size=40, score=0.8)], 0.40)
        large = nms_keep([_box(0, 0, size=400, score=0.9),
                          _box(132, 0, size=400, score=0.8)], 0.40)
        self.assertEqual(len(small), len(large))
        self.assertEqual(len(small), 2)


class RefactorEquivalenceTest(unittest.TestCase):
    """With the rule off, the shared implementation must be exactly what each
    detector had before — otherwise this is a rewrite of four engines, not a
    consolidation."""

    def _random_dets(self, rng, n):
        xs = rng.uniform(0, 500, n)
        ys = rng.uniform(0, 500, n)
        sizes = rng.uniform(20, 200, n)
        scores = rng.uniform(0.3, 1.0, n)
        return np.stack([xs, ys, xs + sizes, ys + sizes, scores], axis=1).astype(np.float32)

    def test_matches_the_previous_implementation(self):
        rng = np.random.default_rng(20260804)
        for offset in (0.0, 1.0):
            for thresh in (0.3, 0.4, 0.5, 0.7):
                for n in (0, 1, 2, 5, 20):
                    dets = self._random_dets(rng, n)
                    self.assertEqual(
                        nms_keep(dets, thresh, offset=offset, center_frac=0),
                        _reference_nms(dets, thresh, offset),
                        f'offset={offset} thresh={thresh} n={n}')

    def test_empty_input(self):
        self.assertEqual(nms_keep([], 0.4), [])
        self.assertEqual(nms_keep(np.zeros((0, 5), np.float32), 0.4), [])

    def test_single_box(self):
        self.assertEqual(nms_keep([_box(0, 0)], 0.4), [0])


class EngineWiringTest(unittest.TestCase):
    """Every engine must answer to the same rule. The close-up rescue once went
    live on one engine of five because each detector reads its own settings;
    these assert the wiring rather than trusting it."""

    def test_yoloface_delegates(self):
        from roop.yoloface import _nms
        boxes = np.array([_box(0, 0)[:4], _box(33, 0)[:4]], np.float32)
        scores = np.array([0.95, 0.90], np.float32)
        self.assertEqual(len(_nms(boxes, scores, iou_thresh=0.40)), 2)

    def test_retinaface_r50_delegates(self):
        from roop.retinaface import RetinaFace3Output
        det = RetinaFace3Output.__new__(RetinaFace3Output)
        det.nms_thresh = 0.40
        dets = np.array([_box(0, 0, score=0.95), _box(33, 0, score=0.90)], np.float32)
        self.assertEqual(len(det.nms(dets)), 2)

    def test_bind_instance_nms_replaces_the_method(self):
        class _Stock:
            nms_thresh = 0.40

            def nms(self, dets):
                return _reference_nms(dets, self.nms_thresh, 1.0)

        det = _Stock()
        dets = np.array([_box(0, 0, score=0.95), _box(33, 0, score=0.90)], np.float32)
        self.assertEqual(len(det.nms(dets)), 1, 'stock behaviour, for contrast')
        bind_instance_nms(det)
        self.assertEqual(len(det.nms(dets)), 2)

    def test_bind_is_instance_scoped(self):
        """Two detectors from the same class must not affect each other, and the
        CLASS must be left alone — insightface is shared with other code."""
        class _Stock:
            nms_thresh = 0.40

            def nms(self, dets):
                return _reference_nms(dets, self.nms_thresh, 1.0)

        patched, stock = _Stock(), _Stock()
        bind_instance_nms(patched)
        dets = np.array([_box(0, 0, score=0.95), _box(33, 0, score=0.90)], np.float32)
        self.assertEqual(len(stock.nms(dets)), 1)
        self.assertEqual(len(_Stock().nms(dets)), 1)

    def test_bind_tolerates_a_detector_without_nms(self):
        """A hybrid engine leaves fa.det_model None, and not every routed model
        exposes nms — binding must not invent one or raise."""
        self.assertIsNone(bind_instance_nms(None))

        class _NoNms:
            pass

        obj = _NoNms()
        bind_instance_nms(obj)
        self.assertFalse(hasattr(obj, 'nms'))

    def test_yunet_relaxes_opencv_and_suppresses_itself(self):
        """YuNet's NMS is inside OpenCV and cannot be replaced, so it must be
        told not to decide — otherwise it pre-deletes the pair before the shared
        rule ever sees it."""
        from roop import yunet
        self.assertGreater(yunet._RAW_NMS, 0.60,
                           'must not pre-delete a pair the rule would keep')
        # detect() needs the real OpenCV model and a frame, so the wiring is
        # asserted at the source: relaxing OpenCV without then suppressing here
        # would leak duplicate boxes, and suppressing without relaxing would
        # leave the pair already deleted. Both halves must be present.
        src = open(os.path.join(os.path.dirname(__file__), '..', 'roop',
                                'yunet.py'), encoding='utf-8').read()
        self.assertIn('setNMSThreshold(raw_nms)', src)
        self.assertIn('nms_keep(bboxes, nms_thresh', src)


class DefaultsTest(unittest.TestCase):

    def test_center_frac_default(self):
        self.assertEqual(CENTER_FRAC, 0.25)


if __name__ == '__main__':
    unittest.main()
