"""User identity pins in the pre-pass binding (Stage 11).

A pin is the user saying "this box, on this frame, is that person". The automatic gates in
`_assign_track_sources` exist to ESTIMATE that, so a pinned track is bound before they run. These
tests drive the real method (through the same minimal manager the sibling track-assignment tests
use) and check the three things a pin must do:

  * bind a track the automatic gates refuse (its embedding is out of everyone's range);
  * keep the person's other tracks honest - a track running at the same time as the pinned one is
    still refused for that person (one person cannot be in two places);
  * change NOTHING when there are no pins, or when a pin names no track.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals as G                                       # noqa: E402
from roop.procmgr_tracking import TrackingMixin                # noqa: E402


def _emb(*v):
    a = np.asarray(v, np.float32)
    return a / np.linalg.norm(a)


def _at_distance(base, d):
    """A unit vector at cosine distance d from `base`, offset along the reserved last axis."""
    perp = np.zeros_like(base)
    perp[-1] = 1.0
    perp = perp - base * float(np.dot(perp, base))
    perp /= np.linalg.norm(perp)
    cos = 1.0 - d
    v = base * cos + perp * float(np.sqrt(max(0.0, 1.0 - cos * cos)))
    return (v / np.linalg.norm(v)).astype(np.float32)


A = _emb(1, 0, 0, 0, 0, 0)            # person 0 -> source 0
B = _emb(0, 1, 0, 0, 0, 0)            # person 1 -> source 1


class _Captured:
    def __init__(self, e):
        self.embedding = e


class _Options:
    face_distance_threshold = 0.75
    selected_index = 0


class _Mgr(TrackingMixin):
    def __init__(self):
        self.target_face_datas = [_Captured(A), _Captured(B)]
        self.target_face_groups = [0, 1]
        self.options = _Options()


_X = {1: 100.0, 2: 300.0, 3: 500.0}      # each track stands still at its own x


def _track(tid, emb, first, last):
    x = _X[tid]
    return {'id': tid, 'emb_mean': emb, 'first_seen': first, 'last_seen': last,
            'bbox': np.array([x - 30, 90.0, x + 30, 150.0], np.float32)}


def _scan(tracks):
    """{frame: [(centroid, track_id)]} - each track stands still at its own x."""
    xs = _X
    out = {}
    for t in tracks:
        for f in range(t['first_seen'], t['last_seen'] + 1):
            out.setdefault(f, []).append((np.array([xs[t['id']], 120.0], np.float32), t['id']))
    return out


class PinBindingTest(unittest.TestCase):
    def setUp(self):
        self._saved = {k: getattr(G, k, None) for k in ('identity_pins', 'roto_frame_base')}
        G.roto_frame_base = 1                     # pipeline frame 0 is UI frame 1

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(G, k, v)

    def _bind(self, tracks, pins):
        G.identity_pins = pins
        mgr = _Mgr()
        track_src, _gate, _refused, _inh = mgr._assign_track_sources(tracks, _scan(tracks))
        return mgr, track_src

    def _tracks(self):
        return [
            _track(1, _at_distance(A, 0.30), 0, 100),         # person 0, bound by the automatic gates
            # far from EVERYONE (beyond every gate) and running AFTER person 0 has left, so the
            # existing assignment-by-elimination pass (which needs a concurrent, spatially apart
            # track) cannot explain it either: only the user can say who this is
            _track(2, _at_distance(B, 0.95), 120, 220),
        ]

    def test_without_a_pin_the_unmatched_track_is_refused(self):
        _mgr, src = self._bind(self._tracks(), [])
        self.assertEqual(src[1], 0)
        self.assertIsNone(src[2])

    def test_a_pin_binds_the_track_the_gates_refused(self):
        pins = [{'frame': 171, 'bbox': [270.0, 90.0, 330.0, 150.0], 'centre': (300.0, 120.0),
                 'source_index': 1, 'person_id': 'tp_1'}]
        mgr, src = self._bind(self._tracks(), pins)
        self.assertEqual(src[2], 1, 'the pinned track must carry the pinned source')
        self.assertEqual(src[1], 0, 'the other person is untouched')
        self.assertEqual(list(mgr._pinned_tracks), [2])
        self.assertEqual(mgr._pin_report, {'pins': 1, 'bound': 1, 'unresolved': 0})

    def test_a_pinned_person_still_cannot_be_in_two_places(self):
        # track 3 sits right on person 1 (0.30) and runs the same frames as the pinned track 2.
        # Unpinned, the automatic gates hand it to person 1; with track 2 pinned to person 1 the
        # concurrency gate must refuse it.
        tracks = self._tracks() + [_track(3, _at_distance(B, 0.30), 120, 220)]
        _mgr, free = self._bind(tracks, [])
        self.assertEqual(free[3], 1)
        pins = [{'frame': 171, 'bbox': [270.0, 90.0, 330.0, 150.0], 'centre': (300.0, 120.0),
                 'source_index': 1, 'person_id': 'tp_1'}]
        _mgr, pinned = self._bind(tracks, pins)
        self.assertEqual(pinned[2], 1)
        # NOT person 1 (one person cannot be in two places). The existing elimination pass then
        # explains it as the only other person: it runs alongside person 1's pinned track.
        self.assertNotEqual(pinned[3], 1, 'a second concurrent track must not also become person 1')

    def test_a_pin_that_names_no_track_changes_nothing(self):
        base = self._bind(self._tracks(), [])[1]
        pins = [{'frame': 51, 'bbox': [900.0, 900.0, 960.0, 960.0], 'centre': (930.0, 930.0),
                 'source_index': 1, 'person_id': 'tp_1'}]
        mgr, src = self._bind(self._tracks(), pins)
        self.assertEqual(src, base)
        self.assertEqual(mgr._pin_report, {'pins': 1, 'bound': 0, 'unresolved': 1})

    def test_a_pin_to_a_source_no_selected_person_maps_to_is_unresolved(self):
        pins = [{'frame': 171, 'bbox': [270.0, 90.0, 330.0, 150.0], 'centre': (300.0, 120.0),
                 'source_index': 7, 'person_id': None}]
        mgr, src = self._bind(self._tracks(), pins)
        self.assertIsNone(src[2])
        self.assertEqual(mgr._pin_report['unresolved'], 1)

    def test_no_pins_is_bit_identical_to_before(self):
        G.identity_pins = []
        mgr = _Mgr()
        a = mgr._assign_track_sources(self._tracks(), _scan(self._tracks()))
        G.identity_pins = None
        b = _Mgr()._assign_track_sources(self._tracks(), _scan(self._tracks()))
        self.assertEqual(a[0], b[0])
        self.assertEqual(mgr._pinned_tracks, {})

    def test_the_frame_base_maps_the_ui_frame_onto_the_scan(self):
        # a clip trimmed to start at pipeline frame 40 (UI frame 41): UI frame 211 is scan key 170
        G.roto_frame_base = 41
        pins = [{'frame': 211, 'bbox': [270.0, 90.0, 330.0, 150.0], 'centre': (300.0, 120.0),
                 'source_index': 1, 'person_id': 'tp_1'}]
        _mgr, src = self._bind(self._tracks(), pins)
        self.assertEqual(src[2], 1)


if __name__ == '__main__':
    unittest.main()
