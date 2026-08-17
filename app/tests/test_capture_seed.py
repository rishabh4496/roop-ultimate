"""capture_seed — choosing WHICH frame the target people get captured from.

The bank the user captures IS the identity model: every later gate (the manual
capture gates in /api/target/add_angle, track-to-source binding, the swap-time
vetoes) is a distance measured against it, so a bank whose two people are not
separable cannot be rescued by any threshold downstream. Measured on d6.mp4,
same clip and same code, only the seed frame differing: a kissing frame gives
two people 0.119 apart (11 of 13 later captures ambiguous), the frame this
module picks gives 1.001 (0 of 13 refused by the absolute cutoff).

The properties asserted here are the ones that failure depended on, not the
numbers from that clip:

  * a frame where any two people overlap must lose to one where none do, and
    with three people one bad pair has to sink the whole frame — the seed fixes
    everyone's identity at once;
  * an anatomically impossible landmark set and a weak detection must both be
    refused as reference captures, since each produced a real mis-binding;
  * separation must read the closest pair across people, not the average, and
    must ignore distances WITHIN one person (whose angles are meant to be close);
  * failures must return, not raise — this backs an HTTP endpoint now, and its
    bench ancestor called SystemExit.

Geometry and arithmetic only: no model, no video, no GPU.
"""

import ast
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import capture_seed                                     # noqa: E402
from roop.capture_seed import (landmarks_plausible, min_pair_gap,  # noqa: E402
                               pick_seed, separation, MIN_GAP_FRAC)


class FakeFace(dict):
    """Enough of a Face for the geometry paths: bbox/kps/det_score/embedding.

    A dict subclass with attribute access, because roop's Face is indexed both
    ways (`face["bbox"]` and `face.bbox`) and capture_seed uses the attribute
    form throughout.
    """

    def __init__(self, x0, y0, x1, y1, det_score=0.99, kps=None, embedding=None):
        super().__init__()
        self['bbox'] = np.array([x0, y0, x1, y1], dtype=np.float32)
        w, h = x1 - x0, y1 - y0
        # A plain, upright, anatomically ordered face: eyes high, nose middle,
        # mouth low — the arrangement _landmarks_plausible exists to require.
        self['kps'] = np.array(kps if kps is not None else [
            [x0 + 0.30 * w, y0 + 0.35 * h],
            [x0 + 0.70 * w, y0 + 0.35 * h],
            [x0 + 0.50 * w, y0 + 0.55 * h],
            [x0 + 0.35 * w, y0 + 0.75 * h],
            [x0 + 0.65 * w, y0 + 0.75 * h],
        ], dtype=np.float32)
        self['det_score'] = det_score
        self['embedding'] = embedding

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def face(x0, **kw):
    """A 100px-wide face at x0, vertically wherever."""
    return FakeFace(x0, 0, x0 + 100, 120, **kw)


class GapGeometry(unittest.TestCase):
    def test_gap_is_measured_in_face_widths(self):
        # 100px boxes with 50px of air between them: half a face width.
        self.assertAlmostEqual(min_pair_gap([face(0), face(150)]), 0.5, places=6)

    def test_overlapping_boxes_read_as_zero_not_negative(self):
        # max(...,0) in the gap term: touching and deeply overlapping both floor
        # at 0, which is what MIN_GAP_FRAC compares against.
        self.assertEqual(min_pair_gap([face(0), face(50)]), 0.0)
        self.assertEqual(min_pair_gap([face(0), face(10)]), 0.0)

    def test_order_does_not_matter(self):
        self.assertEqual(min_pair_gap([face(300), face(0)]),
                         min_pair_gap([face(0), face(300)]))

    def test_worst_pair_decides_with_three_people(self):
        # Two are far apart, but one pair overlaps. The seed binds all three
        # identities at once, so the overlapping pair has to sink the frame —
        # taking the best or the mean pair here is what would let a fused pair
        # through.
        faces = [face(0), face(110), face(1000)]
        self.assertEqual(min_pair_gap(faces), 0.1)
        self.assertLess(min_pair_gap(faces), MIN_GAP_FRAC)


class LandmarkPlausibility(unittest.TestCase):
    def test_ordinary_upright_face_passes(self):
        f = face(0)
        self.assertTrue(landmarks_plausible(f.bbox, f.kps, det_score=0.9))

    def test_mouth_above_eyes_is_refused(self):
        # The real d1.mp4 failure: a fully-back head tilt (chin at the sky, no
        # face visible) that still returned a bbox, 5 keypoints and a flattering
        # off-axis reading — with "mouth" above "eyes".
        f = face(0)
        flipped = f.kps.copy()
        flipped[:, 1] = 120 - flipped[:, 1]
        self.assertFalse(landmarks_plausible(f.bbox, flipped, det_score=0.9))

    def test_weak_detection_is_refused_however_good_the_geometry(self):
        # The other half of that investigation: normal-looking landmarks on a
        # det_score 0.434 box. Geometry alone would have accepted it.
        f = face(0)
        self.assertTrue(landmarks_plausible(f.bbox, f.kps, det_score=0.9))
        self.assertFalse(landmarks_plausible(f.bbox, f.kps, det_score=0.434))

    def test_degenerate_box_is_refused(self):
        f = face(0)
        self.assertFalse(landmarks_plausible(np.array([5, 5, 5, 5]), f.kps, det_score=0.9))

    def test_too_few_keypoints_is_refused(self):
        f = face(0)
        self.assertFalse(landmarks_plausible(f.bbox, f.kps[:3], det_score=0.9))


class SeedChoice(unittest.TestCase):
    def rows(self):
        return [
            (10, [face(0), face(105)]),      # nearly touching
            (20, [face(0), face(400)]),      # clearly apart  <- should win
            (30, [face(0), face(140)]),      # a little apart
        ]

    def test_picks_the_most_separated_frame_not_the_first_acceptable_one(self):
        # The bench ancestor returned the FIRST frame clearing the bar, which on
        # d6 gives a separation of 0.70 where the best frame gives 0.810.
        seed, warn = pick_seed(self.rows(), expect=2)
        self.assertIsNotNone(seed)
        self.assertEqual(seed[0], 20)
        self.assertIsNone(warn)

    def test_frames_with_the_wrong_head_count_are_ignored(self):
        rows = self.rows() + [(40, [face(0), face(400), face(800)])]
        seed, _ = pick_seed(rows, expect=3)
        self.assertEqual(seed[0], 40)

    def test_warns_but_still_returns_when_nothing_is_clearly_apart(self):
        # A clip of two people in constant contact must still yield a capture —
        # refusing outright is worse than capturing with a stated caveat.
        seed, warn = pick_seed([(7, [face(0), face(105)])], expect=2)
        self.assertIsNotNone(seed)
        self.assertEqual(seed[0], 7)
        self.assertIsNotNone(warn)

    def test_overlapping_everywhere_warns_about_overlap_specifically(self):
        seed, warn = pick_seed([(7, [face(0), face(40)])], expect=2)
        self.assertIsNotNone(seed)
        self.assertIn('overlap', warn)

    def test_no_usable_frame_returns_none_rather_than_raising(self):
        seed, warn = pick_seed([(1, [face(0)])], expect=2)
        self.assertIsNone(seed)
        self.assertTrue(warn)


class Separation(unittest.TestCase):
    @staticmethod
    def emb(*v):
        a = np.array(v, dtype=np.float32)
        return a / np.linalg.norm(a)

    def test_reads_the_closest_pair_across_people(self):
        # Three people, two of them near-identical. The number has to report the
        # bad pair: an average would hide it behind the two good ones, and it is
        # precisely the closest pair that makes swaps mix people up.
        t = [FakeFace(0, 0, 1, 1, embedding=self.emb(1, 0, 0)),
             FakeFace(0, 0, 1, 1, embedding=self.emb(0, 1, 0)),
             FakeFace(0, 0, 1, 1, embedding=self.emb(0.999, 0.045, 0))]
        self.assertLess(separation(t, [0, 1, 2]), 0.01)

    def test_ignores_distances_within_one_person(self):
        # Two angles of the SAME person are supposed to be close; only
        # cross-person pairs say anything about separability.
        t = [FakeFace(0, 0, 1, 1, embedding=self.emb(1, 0, 0)),
             FakeFace(0, 0, 1, 1, embedding=self.emb(0.999, 0.045, 0)),
             FakeFace(0, 0, 1, 1, embedding=self.emb(0, 1, 0))]
        self.assertGreater(separation(t, [0, 0, 1]), 0.5)

    def test_one_person_has_no_separation(self):
        t = [FakeFace(0, 0, 1, 1, embedding=self.emb(1, 0, 0))]
        self.assertIsNone(separation(t, [0]))

    def test_faces_without_an_embedding_are_skipped_not_crashed_on(self):
        t = [FakeFace(0, 0, 1, 1, embedding=self.emb(1, 0, 0)),
             FakeFace(0, 0, 1, 1, embedding=None)]
        self.assertIsNone(separation(t, [0, 1]))


class ContractsThatBackAnEndpoint(unittest.TestCase):
    def test_nothing_in_the_module_calls_sys_exit(self):
        # The bench ancestor raised SystemExit when it could not find a frame,
        # which inside a request handler kills the response rather than
        # answering it. Walked as an AST rather than grepped, because the module
        # docstring legitimately mentions SystemExit while describing what it
        # replaced — a substring check passes or fails on prose.
        with open(capture_seed.__file__, encoding='utf-8') as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise):
                exc = node.exc
                name = getattr(exc, 'id', None) or getattr(getattr(exc, 'func', None), 'id', None)
                self.assertNotEqual(name, 'SystemExit',
                                    'capture_seed backs an HTTP endpoint and must return, not exit')
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Attribute) and fn.attr == 'exit':
                    self.fail('capture_seed must not call exit() — it backs an HTTP endpoint')

    def test_frame_numbers_are_documented_as_zero_based(self):
        # get_video_frame is 1-based and subtracts one itself, so a caller
        # re-decoding a returned frame index must add 1. The bug this prevents
        # is silent (a crop from the previous frame looks almost right), so the
        # convention has to stay written down next to the values.
        self.assertIn('0-based', capture_seed.auto_capture.__doc__)


if __name__ == '__main__':
    unittest.main()
