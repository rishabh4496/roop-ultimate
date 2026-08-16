"""Which tracklets are allowed to inherit a selected person's source.

The reported failure: with two people on screen and ONE of them selected, the
swap is correct while both are visible, but as soon as the target leaves the
frame the OTHER person starts getting swapped — as if the frame always has to
find somebody to swap.

It is decided here, in the pre-pass, not in the swap loop. A person may own
several tracklets (tracking fragments constantly), so every track under the
absolute gate is bound, and only tracks that run CONCURRENTLY with one the
person already owns are refused — one person cannot be in two places at once.
The converse does not follow: a bystander's fragment that lies entirely inside a
stretch where the target is off screen is concurrent with nothing, so that guard
never looks at it, and it inherits the target's source for exactly those frames.

The margin gate closes it: a person's later tracks must sit near its closest
one. Asserted below, together with the properties that must survive it — the
target's own fragments still bind, and a refusal is never a dropped face.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.procmgr_tracking import TrackingMixin              # noqa: E402
from roop.procmgr_runtime import _TRACK_ASSIGN_MAX           # noqa: E402


def _emb(*vals):
    """A unit vector whose cosine distance to the others is controllable."""
    v = np.asarray(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def _at_distance(base, d, axis=-1):
    """A unit vector exactly cosine-distance *d* from unit vector *base*.

    The offset is taken along *axis*, which must be a dimension no identity in
    the fixture uses — otherwise moving away from one person walks straight
    towards another and the fixture, not the code, decides the outcome.
    """
    perp = np.zeros_like(base)
    perp[axis] = 1.0
    perp = perp - base * float(np.dot(perp, base))
    perp /= np.linalg.norm(perp)
    cos = 1.0 - d
    v = base * cos + perp * float(np.sqrt(max(0.0, 1.0 - cos * cos)))
    return (v / np.linalg.norm(v)).astype(np.float32)


class _Captured:
    """Stand-in for a captured target face (only its embedding is read)."""

    def __init__(self, embedding):
        self.embedding = embedding


class _Options:
    def __init__(self, threshold=0.75, selected_index=0):
        self.face_distance_threshold = threshold
        self.selected_index = selected_index


class _Mgr(TrackingMixin):
    """The assignment needs only the captured faces, their grouping and options."""

    def __init__(self, captured, groups, threshold=0.75):
        self.target_face_datas = [_Captured(e) for e in captured]
        self.target_face_groups = groups
        self.options = _Options(threshold)


def _track(tid, emb, first, last):
    return {'id': tid, 'emb_mean': emb, 'first_seen': first, 'last_seen': last}


# Identities live on their own axes; the last axis is reserved as the "away
# from everybody" direction _at_distance perturbs along.
TARGET = _emb(1.0, 0.0, 0.0, 0.0, 0.0)


class BystanderInheritsSourceTest(unittest.TestCase):
    """One selected person, a second person on screen."""

    def _assign(self, tracks, threshold=0.75):
        mgr = _Mgr([TARGET], [0], threshold)
        track_src, _assign_max, refused, _inh = mgr._assign_track_sources(tracks)
        return track_src, refused

    def test_disjoint_bystander_track_is_refused(self):
        """The reported bug: the bystander's fragment covers frames 200-400,
        where the target is absent, so it overlaps nothing the target owns and
        the concurrency guard cannot see it. It scrapes under the absolute gate
        (0.55 < 0.60) but sits far from the target's own 0.30."""
        tracks = [
            _track(1, _at_distance(TARGET, 0.30), 0, 199),      # the target
            _track(2, _at_distance(TARGET, 0.55), 200, 400),    # the bystander
        ]
        track_src, refused = self._assign(tracks)
        self.assertEqual(track_src[1], 0, 'the target must keep its source')
        self.assertIsNone(track_src[2], 'the bystander must not inherit it')
        self.assertEqual(refused, 1)

    def test_targets_own_fragment_still_binds(self):
        """The same shape — a disjoint second track — but it really is the
        target, re-acquired after an occlusion. It must still lock, or the fix
        would have traded a wrong-face bug for a flicker bug."""
        tracks = [
            _track(1, _at_distance(TARGET, 0.30), 0, 199),
            _track(2, _at_distance(TARGET, 0.38), 200, 400),
        ]
        track_src, refused = self._assign(tracks)
        self.assertEqual(track_src[1], 0)
        self.assertEqual(track_src[2], 0)
        self.assertEqual(refused, 0)

    def test_concurrent_lookalike_still_refused(self):
        """The pre-existing concurrency guard is untouched: a track running at
        the same time as the target's is a second body whatever it measures."""
        tracks = [
            _track(1, _at_distance(TARGET, 0.30), 0, 400),
            _track(2, _at_distance(TARGET, 0.34), 0, 400),
        ]
        track_src, _ = self._assign(tracks)
        self.assertEqual(track_src[1], 0)
        self.assertIsNone(track_src[2])

    def test_anchor_is_the_closest_track_not_the_first_seen(self):
        """Candidates are ranked by distance, so a poor early track cannot set a
        loose anchor for everything after it."""
        tracks = [
            _track(1, _at_distance(TARGET, 0.52), 0, 99),       # early, poor
            _track(2, _at_distance(TARGET, 0.30), 100, 199),    # the real target
            _track(3, _at_distance(TARGET, 0.55), 200, 400),    # bystander
        ]
        track_src, _ = self._assign(tracks)
        self.assertEqual(track_src[2], 0, 'closest track anchors the person')
        self.assertIsNone(track_src[3])

    def test_absolute_gate_still_applies_first(self):
        """A stranger well clear of the absolute gate never becomes a candidate,
        so the margin never even sees it."""
        tracks = [_track(1, _at_distance(TARGET, 0.95), 0, 400)]
        track_src, refused = self._assign(tracks)
        self.assertIsNone(track_src[1])
        self.assertEqual(refused, 0, 'refused by the absolute gate, not the margin')

    def test_floor_protects_a_fragment_when_the_anchor_is_unusually_good(self):
        """The margin is relative, so a very good anchor makes it very strict —
        a clean frontal capture matching a clean frontal track anchors near 0.10,
        which would refuse that same person's profile-heavy fragment at 0.40, a
        distance nothing else in the pipeline calls a stranger. The floor stops
        the margin binding there. This matters because ROOP_TRACK_REID_MAX
        deliberately produces MORE fragments, each of which lands here."""
        tracks = [
            _track(1, _at_distance(TARGET, 0.10), 0, 199),
            _track(2, _at_distance(TARGET, 0.40), 200, 400),
        ]
        track_src, refused = self._assign(tracks)
        self.assertEqual(track_src[2], 0, 'anchor + margin would have refused it')
        self.assertEqual(refused, 0)

    def test_floor_does_not_reach_the_band_the_margin_targets(self):
        """The floor must not undo the fix: a bystander in the 0.5-0.6 band is
        still refused however good the target's anchor is."""
        tracks = [
            _track(1, _at_distance(TARGET, 0.10), 0, 199),
            _track(2, _at_distance(TARGET, 0.55), 200, 400),
        ]
        track_src, refused = self._assign(tracks)
        self.assertIsNone(track_src[2])
        self.assertEqual(refused, 1)

    def test_single_track_is_unaffected(self):
        """One track means no anchor to compare against — the common case must
        behave exactly as before."""
        tracks = [_track(1, _at_distance(TARGET, 0.58), 0, 400)]
        track_src, refused = self._assign(tracks)
        self.assertEqual(track_src[1], 0)
        self.assertEqual(refused, 0)


class MultiPersonTest(unittest.TestCase):
    """Two selected people: the margin is per person, never across them."""

    def test_each_person_anchors_independently(self):
        other = _emb(0.0, 1.0, 0.0, 0.0, 0.0)
        mgr = _Mgr([TARGET, other], [0, 1])
        tracks = [
            _track(1, _at_distance(TARGET, 0.30), 0, 199),
            _track(2, _at_distance(other, 0.30), 0, 199),
        ]
        track_src, _assign_max, refused, _inh = mgr._assign_track_sources(tracks)
        self.assertEqual(track_src[1], 0)
        self.assertEqual(track_src[2], 1)
        self.assertEqual(refused, 0)

    def test_multi_angle_capture_uses_the_closest_angle(self):
        """A person captured at several angles is matched on its best one, so
        adding angles must not shift the anchor away from that person."""
        profile = _at_distance(TARGET, 0.45)
        mgr = _Mgr([TARGET, profile], [0, 0])
        tracks = [_track(1, _at_distance(profile, 0.05), 0, 400)]
        track_src, _assign_max, _refused, _inh = mgr._assign_track_sources(tracks)
        self.assertEqual(track_src[1], 0)


class GateOrderTest(unittest.TestCase):

    def test_distances_are_measured_as_the_gate_expects(self):
        """Guard on the test's own fixture: _at_distance must produce the
        distance it claims, or every assertion above is testing nothing."""
        from roop.utilities import compute_cosine_distance
        for d in (0.30, 0.55, 0.95):
            self.assertAlmostEqual(
                compute_cosine_distance(TARGET, _at_distance(TARGET, d)), d, places=5)

    def test_absolute_gate_constant_is_tighter_than_per_frame_matching(self):
        """The margin only ever narrows an already-tight gate; if the absolute
        one were loosened to the per-frame threshold the anchor logic would be
        carrying far more weight than it was tuned for."""
        self.assertLessEqual(_TRACK_ASSIGN_MAX, 0.75)


if __name__ == '__main__':
    unittest.main()
