"""Carrying identity across a CUT, and the shape-change half of the outcome guard.

Both come from the same report — "one faceset swaps and the other does not, at
different angles, even when the face is not a close-up, and both still flicker".
Attributed per frame, that was two things and neither was the swap:

  * the second and later SHOTS of a person get no source. Their track's mean is
    0.47 and 0.68 from a frontal capture — one over the assignment gate, one
    over the anchor margin — while the same tracks are 0.29 and 0.58 from that
    person's OWN track in the first shot, and 0.97+ from anybody else's. The
    photo cannot separate them; the tracks separate them with a gap.
    _track_inherit already made exactly this comparison, but only for a fragment
    INSIDE the owner's span, which a later shot never is.

  * the outcome guard discarded a quarter of the correct swaps, because on a
    profile the interocular distance it divides by has collapsed to a few
    pixels. See SWAP_SHAPE_TOL.

What these tests defend is the refusing, not the accepting. A missed inheritance
costs what the pipeline did before; a wrong one hands a stretch of frames to the
wrong person's faceset.
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.procmgr_tracking import TrackingMixin          # noqa: E402
from roop import procmgr_tracking as _pt                 # noqa: E402
from roop.utilities import compute_cosine_distance as cd  # noqa: E402
from roop.face_util import keypoint_shape_change, swap_moved_the_face  # noqa: E402


def _emb(idx):
    a = np.zeros(512, dtype=np.float32)
    a[idx] = 1.0
    return a


def _at(base, d, axis):
    """A unit vector `d` in cosine distance from `base`, along a free axis."""
    perp = np.zeros_like(base)
    perp[axis] = 1.0
    perp = perp - base * float(np.dot(perp, base))
    perp /= np.linalg.norm(perp)
    cos = 1.0 - d
    v = base * cos + perp * float(np.sqrt(max(0.0, 1.0 - cos * cos)))
    return (v / np.linalg.norm(v)).astype(np.float32)


PERSON_A = _emb(0)
PERSON_B = _emb(1)

# Shot 1: both people match their captured stills outright.
A1 = _at(PERSON_A, 0.21, 5)
B1 = _at(PERSON_B, 0.17, 6)
# Shot 2: a close-up of each, built as a perturbation OF that person's first
# track — which is what a second shot of the same face actually is, and what
# makes the photo distance drift while the track distance stays small.
A2 = _at(A1, 0.40, 7)
B2 = _at(B1, 0.58, 8)
# Shot 3, person A only: near A2 and no longer near A1, so it can only be
# reached by going through A2.
A3 = _at(A2, 0.45, 9)


class _Captured:
    def __init__(self, embedding):
        self.embedding = embedding


class _Options:
    face_distance_threshold = 0.75
    selected_index = 0


class _Mgr(TrackingMixin):
    def __init__(self, captured, groups):
        self.target_face_datas = [_Captured(e) for e in captured]
        self.target_face_groups = list(groups)
        self.options = _Options()

    def _publish_live(self, frame):
        pass


def _track(tid, emb, first, last):
    return {'id': tid, 'emb_mean': emb, 'first_seen': first, 'last_seen': last}


def _shots(*spans):
    """per_frame for tracks that each occupy one disjoint stretch."""
    per_frame = {}
    for tid, lo, hi in spans:
        for f in range(lo, hi + 1):
            per_frame.setdefault(f, []).append((np.zeros(2, np.float32), tid))
    return per_frame


def _merge(per_frame, more):
    """Add another track's frames WITHOUT replacing what is already there.

    dict.update overwrites the whole entry for a frame, which silently deletes
    the other tracks from it — and a concurrency test whose fixture has quietly
    become non-concurrent passes for the wrong reason.
    """
    for f, ents in more.items():
        per_frame.setdefault(f, []).extend(ents)
    return per_frame


def _assign(tracks, per_frame, captured=(PERSON_A, PERSON_B), groups=(0, 1)):
    return _Mgr(list(captured), groups)._assign_track_sources(tracks, per_frame)


class TheFixture(unittest.TestCase):
    """Assert the material before asserting anything about the code.

    An identity test whose fixture does not actually sit in the band it claims
    is measuring the fixture. These are the numbers read off the reported clip:
    a later shot 0.4-0.7 from the photo, under 0.6 from its own person's other
    track, and 0.85+ from everyone else's.
    """

    def test_the_later_shots_are_over_the_photo_gates(self):
        self.assertGreater(cd(PERSON_A, A2), 0.45)     # over the anchor floor
        self.assertGreater(cd(PERSON_B, B2), 0.60)     # over the assign gate

    def test_but_close_to_their_own_persons_first_track(self):
        self.assertLess(cd(A1, A2), 0.60)
        self.assertLess(cd(B1, B2), 0.60)

    def test_and_far_from_the_other_person(self):
        for later, other in ((A2, B1), (B2, A1)):
            self.assertGreater(cd(later, other), 0.85)

    def test_the_third_shot_needs_the_second_to_reach_it(self):
        self.assertGreater(cd(A1, A3), 0.60)
        self.assertLess(cd(A2, A3), 0.60)


class InheritingAcrossACut(unittest.TestCase):

    def _clip(self):
        tracks = [_track(0, A1, 0, 99), _track(1, B1, 0, 99),
                  _track(2, A2, 100, 199), _track(3, B2, 100, 199)]
        per_frame = _shots((0, 0, 99), (1, 0, 99), (2, 100, 199), (3, 100, 199))
        return tracks, per_frame

    def test_both_later_tracks_get_their_own_person(self):
        src, _m, _r, inherited = _assign(*self._clip())
        self.assertEqual(src[0], 0)
        self.assertEqual(src[1], 1)
        self.assertEqual(src[2], 0, 'the second shot of person A')
        self.assertEqual(src[3], 1, 'the second shot of person B')
        self.assertIn(2, inherited)
        self.assertIn(3, inherited)

    def test_it_propagates_through_the_shot_it_just_assigned(self):
        """A3 is out of reach of A1 and only reachable via A2, so one round of
        the second pass leaves it unassigned. Identity along a clip is
        transitive and the pass has to be run to a fixed point."""
        tracks, per_frame = self._clip()
        tracks.append(_track(4, A3, 200, 299))
        _merge(per_frame, _shots((4, 200, 299)))
        src, _m, _r, inherited = _assign(tracks, per_frame)
        self.assertEqual(src[4], 0)
        self.assertEqual(inherited[4][0], 2, 'reached through the second shot')

    def test_a_track_equidistant_from_both_people_is_refused(self):
        """No margin, no inheritance. This is the bystander shape: somebody who
        is not either selected person is not much nearer one than the other,
        whatever the absolute numbers happen to be.

        Only the two first-shot tracks are present, so "equidistant" is exact
        rather than an accident of which later track happened to be nearer.
        """
        middle = (A2 + B2) / 2.0
        middle = (middle / np.linalg.norm(middle)).astype(np.float32)
        # It has to be genuinely ambiguous on BOTH comparisons, or the test is
        # measuring whichever gate happened to catch it: over the photo gate for
        # each person, and the same distance from each person's nearest track.
        self.assertGreater(min(cd(middle, PERSON_A), cd(middle, PERSON_B)), 0.60)
        self.assertAlmostEqual(cd(middle, A2), cd(middle, B2), places=5)
        tracks, per_frame = self._clip()
        tracks.append(_track(4, middle, 200, 299))
        _merge(per_frame, _shots((4, 200, 299)))
        src, _m, _r, _i = _assign(tracks, per_frame)
        self.assertIsNone(src[4])

    def test_a_concurrent_second_track_is_still_a_second_body(self):
        """The same person cannot hold two tracks over the same frames, however
        well the appearance matches — that is a duplicate detection or somebody
        else, and it is the check that stops one source landing on two faces."""
        tracks, per_frame = self._clip()
        tracks.append(_track(4, A2, 100, 199))
        _merge(per_frame, _shots((4, 100, 199)))
        src, _m, _r, _i = _assign(tracks, per_frame)
        # Which of the two identical tracks wins is arbitrary; that only ONE of
        # them holds person A over those frames is the property. One source per
        # person per frame is what stops a swap landing on two faces at once.
        self.assertEqual([src[2], src[4]].count(0), 1)

    def test_with_one_person_selected_nothing_changes(self):
        """THE regression guard. With a single captured person the margin is
        vacuous — there is nobody to be further from — and what would be left is
        a bare absolute bar on a disjoint track, which is exactly the bystander
        that the containment rule exists to refuse. The path must not exist at
        all in that case.
        """
        tracks = [_track(0, A1, 0, 99), _track(2, A2, 100, 199)]
        per_frame = _shots((0, 0, 99), (2, 100, 199))
        src, _m, _r, inherited = _assign(tracks, per_frame,
                                         captured=(PERSON_A,), groups=(0,))
        self.assertIsNone(src[2])
        self.assertEqual(inherited, {})

    def test_the_kill_switch(self):
        with mock.patch.object(_pt, '_TRACK_INHERIT_MARGIN', 0.0):
            src, _m, _r, inherited = _assign(*self._clip())
        self.assertIsNone(src[2])
        self.assertIsNone(src[3])
        self.assertEqual(inherited, {})


class TheOutcomeGuardsShapeTerm(unittest.TestCase):
    """Real (plate, result) pairs lifted from the two measured sets.

    Constructions were tried first and are not good enough here: a synthetic
    "frontal face pasted on a profile at the same orientation and extent"
    measures 0.71, which is BELOW the 0.730 maximum of the 229 correct swaps —
    so that particular hypothetical is not separable by this metric at any
    threshold, and pretending otherwise with a hand-built fixture would assert
    something the measurements do not support. These two pairs are observations.
    """

    # The correct swap with the WORST displacement reading of the 229: 4.75
    # interocular units, four and a half times the tolerance, on a face that
    # plainly did not move. This is the frame the old rule threw away.
    GOOD_PLATE = np.array([[760.3, 155.5], [770.1, 152.9], [597.2, 184.2],
                           [661.8, 417.4], [676.7, 416.2]])
    GOOD_RESULT = np.array([[767.6, 107.5], [816.4, 125.6], [621.6, 194.2],
                            [588.8, 404.6], [640.1, 410.9]])
    # The lowest member of the wrecked mode in the studio sweep.
    BAD_PLATE = np.array([[400.4, 311.5], [416.8, 320.9], [446.0, 338.8],
                          [398.8, 396.7], [420.8, 407.9]])
    BAD_RESULT = np.array([[428.8, 355.2], [435.5, 358.6], [472.1, 331.4],
                           [429.3, 331.8], [427.6, 332.8]])

    KPS = np.array([[100.0, 100.0], [140.0, 100.0], [120.0, 125.0],
                    [105.0, 150.0], [135.0, 150.0]])

    def test_a_pure_translation_changes_no_shape(self):
        moved = self.KPS + np.array([37.0, -11.0])
        self.assertLess(keypoint_shape_change(self.KPS, moved), 1e-9)

    def test_the_two_classes_sit_either_side_of_the_threshold(self):
        from roop.face_util import SWAP_SHAPE_TOL
        self.assertLess(keypoint_shape_change(self.GOOD_PLATE, self.GOOD_RESULT),
                        SWAP_SHAPE_TOL)
        self.assertGreater(keypoint_shape_change(self.BAD_PLATE, self.BAD_RESULT),
                           SWAP_SHAPE_TOL)

    def test_the_correct_swap_survives_although_it_moved_a_long_way(self):
        """Both conditions must hold. Displacement alone discarded this frame,
        and 28.4% of the set it came from."""
        with mock.patch('roop.face_util.detect_boxes_in_roi',
                        return_value=[_Det(self.GOOD_RESULT)]):
            self.assertFalse(swap_moved_the_face(
                np.zeros((600, 900, 3), np.uint8), self.GOOD_PLATE,
                (560.0, 100.0, 800.0, 440.0)))

    def test_the_wrecked_swap_is_still_discarded(self):
        with mock.patch('roop.face_util.detect_boxes_in_roi',
                        return_value=[_Det(self.BAD_RESULT)]):
            self.assertTrue(swap_moved_the_face(
                np.zeros((600, 900, 3), np.uint8), self.BAD_PLATE,
                (380.0, 290.0, 470.0, 420.0)))

    def test_the_shape_term_can_be_turned_off(self):
        with mock.patch('roop.face_util.detect_boxes_in_roi',
                        return_value=[_Det(self.GOOD_RESULT)]),              mock.patch('roop.face_util.SWAP_SHAPE_TOL', 0.0):
            self.assertTrue(swap_moved_the_face(
                np.zeros((600, 900, 3), np.uint8), self.GOOD_PLATE,
                (560.0, 100.0, 800.0, 440.0)))


class _Det:
    def __init__(self, kps):
        self.kps = np.asarray(kps, dtype=np.float64)
        x0, y0 = self.kps.min(axis=0)
        x1, y1 = self.kps.max(axis=0)
        self.bbox = np.array([x0 - 20, y0 - 20, x1 + 20, y1 + 20], np.float32)


if __name__ == '__main__':
    unittest.main()
