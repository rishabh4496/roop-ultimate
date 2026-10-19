"""Stage 16: a track closer to a registered bystander than to the target is not
the target.

The pre-existing gates bind a track to a person on ABSOLUTE distance to the
captured stills, plus a margin against that person's own best track. Neither
knows who ELSE is in the scene. A lookalike (or a face the recogniser reads
half-way between two people) can sit inside the absolute gate while being far
closer to a stranger who is plainly in the clip. The bystander bank is that
knowledge; `_assign_track_sources` consults it.

Every refusal here is paired with the same scene where the track really is the
target (it must still bind) and with the switch off (the old behaviour).
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
pytestmark = pytest.mark.gpu        # imports the tracking mixin (ML stack), like its siblings

import roop.procmgr_tracking as pt                                   # noqa: E402
from roop.procmgr_tracking import TrackingMixin                      # noqa: E402


def _unit(*v):
    a = np.asarray(v, np.float32)
    return a / np.linalg.norm(a)


# five orthogonal identity axes: target, stranger, scratch
TARGET = _unit(1, 0, 0, 0, 0)
STRANGER = _unit(0, 1, 0, 0, 0)


def _mix(w_target, w_stranger):
    """Unit vector with cosine w_target to TARGET and w_stranger to STRANGER."""
    rest = max(0.0, 1.0 - w_target ** 2 - w_stranger ** 2)
    return _unit(w_target, w_stranger, 0, np.sqrt(rest), 0)


class _Captured:
    def __init__(self, e):
        self.embedding = e


class _Options:
    face_distance_threshold = 0.75
    selected_index = 0


class _Mgr(TrackingMixin):
    def __init__(self):
        self.target_face_datas = [_Captured(TARGET)]
        self.target_face_groups = [0]
        self.options = _Options()


def _track(tid, emb, first, last, n):
    return {'id': tid, 'emb_mean': emb, 'first_seen': first, 'last_seen': last,
            'emb_n': n, 'emb_dirty': False}


def _assign(tracks):
    m = _Mgr()
    track_src, *_ = m._assign_track_sources(tracks)
    return m, track_src


class BystanderVetoTest(unittest.TestCase):

    def setUp(self):
        self._saved = pt._BYSTANDER_VETO
        pt._BYSTANDER_VETO = True

    def tearDown(self):
        pt._BYSTANDER_VETO = self._saved

    def _scene(self, lookalike):
        # a long clean stranger, and a shorter track that is the one under test
        return [_track(1, STRANGER, 0, 400, 60), _track(2, lookalike, 100, 300, 30)]

    def test_a_lookalike_closer_to_the_stranger_is_refused(self):
        # cos 0.50 to the target (distance 0.50, INSIDE the 0.60 gate) but 0.80
        # to a stranger who is plainly in the clip.
        m, src = _assign(self._scene(_mix(0.50, 0.80)))
        self.assertIsNone(src[2])
        self.assertEqual(m._bystander_refused, 1)

    def test_the_same_track_binds_with_the_veto_off(self):
        """Control: the refusal above is the veto's, not some other gate's."""
        pt._BYSTANDER_VETO = False
        m, src = _assign(self._scene(_mix(0.50, 0.80)))
        self.assertEqual(src[2], 0)
        self.assertEqual(m._bystander_refused, 0)

    def test_the_targets_own_fragment_still_binds(self):
        m, src = _assign(self._scene(_mix(0.80, 0.10)))
        self.assertEqual(src[2], 0)
        self.assertEqual(m._bystander_refused, 0)

    def test_without_a_registered_bystander_nothing_changes(self):
        """A lone track has nobody to be closer to: the old gates decide alone."""
        m, src = _assign([_track(2, _mix(0.50, 0.80), 100, 300, 30)])
        self.assertEqual(src[2], 0)

    def test_a_short_stranger_does_not_register(self):
        scene = [_track(1, STRANGER, 0, 400, 4), _track(2, _mix(0.50, 0.80), 100, 300, 30)]
        m, src = _assign(scene)
        self.assertEqual(src[2], 0, 'a 4-observation track is not evidence of a bystander')

    def test_a_contaminated_stranger_does_not_register(self):
        scene = self._scene(_mix(0.50, 0.80))
        scene[0]['emb_dirty'] = True
        m, src = _assign(scene)
        self.assertEqual(src[2], 0, 'a chimera seeded in a contact must not veto anyone')

    def test_an_unexplained_track_is_not_a_candidate_either_way(self):
        """The stranger itself sits outside the gate; the veto never has to act on it."""
        m, src = _assign(self._scene(_mix(0.80, 0.10)))
        self.assertIsNone(src[1])


if __name__ == '__main__':
    unittest.main()
