"""Stage 16: the flow-bridged gap fill in `_build_temporal_faces`.

The scan loop records, per dropped frame, where optical flow says the face went
(`track['flow'][frame] = FlowResult`) and, at the closing detection, where the
same chain lands (`track['flow'][closing_frame]`). `_flow_bridge_geometry` turns
that into landmarks for the hole ONLY if it arrives at the closing detection.

These tests hand the builder flow results made by hand, so what is asserted is
the DECISION (bridge / refuse) and the arithmetic (shape from flow, end pinned
to the anchor), not LK itself -- that is `test_flow_coast.py`.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
pytestmark = pytest.mark.gpu        # imports insightface/the ML stack, like its siblings

from insightface.app.common import Face                                  # noqa: E402
import roop.procmgr_tracking as pt                                       # noqa: E402
from roop.flow_coast import FlowResult                                   # noqa: E402
from roop.procmgr_tracking import TrackingMixin                          # noqa: E402


class _Options:
    stabilize_face = False
    stabilize_method = 'one_euro'
    stabilize_min_cutoff = 0.05
    stabilize_beta = 0.02


class _Mgr(TrackingMixin):
    def __init__(self):
        self.options = _Options()
        self.input_face_datas = []
        self._track_pose_source_map = {}


def _emb():
    v = np.random.default_rng(1).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def _lm(x, y, n=106):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([x + 40 * np.cos(t), y + 55 * np.sin(t)], 1).astype(np.float32)


def _face(x, y, w=120.0):
    h = w * 0.5
    return Face(bbox=np.array([x - h, y - h, x + h, y + h], np.float32),
                kps=np.array([[x - 30, y - 20], [x + 30, y - 20], [x, y],
                              [x - 25, y + 35], [x + 25, y + 35]], np.float32),
                landmark_2d_106=_lm(x, y), det_score=0.9, embedding=_emb())


def _flow(x, y, w=120.0, consistent=True):
    h = w * 0.5
    f = _face(x, y, w)
    return FlowResult(consistent, f['landmark_2d_106'].copy(), f['kps'].copy(),
                      np.array([x - h, y - h, x + h, y + h], np.float32),
                      np.eye(2, 3, dtype=np.float32), 100, 90, 0.9, 0.2, 'ok')


# real detections at x=100 for frames 0..3 and x=140 for frames 8..11; hole 4..7
X0, X1 = 100.0, 140.0
HOLE = [4, 5, 6, 7]


def _track(flow=None):
    obs = {i: _face(X0, 200.0) for i in range(0, 4)}
    obs.update({i: _face(X1, 200.0) for i in range(8, 12)})
    t = {'id': 0, 'obs': obs, 'emb_mean': _emb(), 'emb_n': len(obs)}
    if flow is not None:
        t['flow'] = flow
    return t


def _good_flow(end_x=X1):
    """A path that moves 8 px a frame (nonlinear vs Hermite's smooth-step), then lands."""
    fl = {g: _flow(X0 + 8.0 * (g - 3), 200.0) for g in HOLE}
    fl[8] = _flow(end_x, 200.0)
    return fl


def _build(track):
    m = _Mgr()
    out = m._build_temporal_faces([track], 10)
    return m, out


class FlowBridgeTest(unittest.TestCase):

    def test_a_consistent_flow_path_that_lands_on_the_anchor_is_used(self):
        m, out = _build(_track(_good_flow()))
        for g in HOLE:
            f = out[g][0]
            self.assertTrue(f.get('_flow_bridged'), f'frame {g}')
            self.assertTrue(f.get('_interpolated'), 'the audit counts gap-filled faces by this')
        self.assertEqual(m._flow_bridged, len(HOLE))
        # shape comes from flow: x advances 8 px/frame plus a linear share of the
        # (here zero) residual, which Hermite's smooth-step would not reproduce
        xs = [float(np.mean(out[g][0]['landmark_2d_106'][:, 0])) for g in HOLE]
        self.assertAlmostEqual(xs[1] - xs[0], 8.0, delta=0.05)

    def test_the_end_is_pinned_to_the_closing_detection(self):
        # flow lands 3 px short of the anchor; the shortfall is spread linearly
        m, out = _build(_track(_good_flow(end_x=X1 - 3.0)))
        last = float(np.mean(out[7][0]['landmark_2d_106'][:, 0]))
        expected = (X0 + 8.0 * 4) + 3.0 * 4 / 5.0       # frame j=3 of 4 gets 4/5 of the residual
        self.assertAlmostEqual(last, expected, delta=0.05)

    def test_flow_that_missed_the_anchor_is_refused_and_hermite_stands(self):
        m, out = _build(_track(_good_flow(end_x=X1 - 60.0)))     # 60 px off on a ~60 px IOD
        for g in HOLE:
            self.assertFalse(out[g][0].get('_flow_bridged'))
            self.assertTrue(out[g][0].get('_interpolated'))
        self.assertEqual(m._flow_bridged, 0)
        self.assertEqual(m._flow_refused, len(HOLE))

    def test_one_inconsistent_step_refuses_the_whole_gap(self):
        fl = _good_flow()
        fl[6] = _flow(X0 + 8.0 * 3, 200.0, consistent=False)
        m, out = _build(_track(fl))
        self.assertEqual(m._flow_bridged, 0)

    def test_a_missing_step_refuses_the_whole_gap(self):
        fl = _good_flow()
        del fl[5]
        m, out = _build(_track(fl))
        self.assertEqual(m._flow_bridged, 0)

    def test_no_closing_flow_result_means_no_bridge(self):
        fl = _good_flow()
        del fl[8]
        m, out = _build(_track(fl))
        self.assertEqual(m._flow_bridged, 0)

    def test_no_flow_data_is_exactly_the_old_behaviour(self):
        _, with_none = _build(_track(None))
        _, with_empty = _build(_track({}))
        for g in HOLE:
            np.testing.assert_array_equal(with_none[g][0]['bbox'], with_empty[g][0]['bbox'])
            self.assertFalse(with_none[g][0].get('_flow_bridged'))

    def test_the_switch_restores_the_previous_output_exactly(self):
        _, base = _build(_track(None))
        saved = pt._FLOW_COAST
        pt._FLOW_COAST = False
        try:
            m, off = _build(_track(_good_flow()))
        finally:
            pt._FLOW_COAST = saved
        self.assertEqual(m._flow_bridged, 0)
        for g in HOLE:
            np.testing.assert_array_equal(off[g][0]['landmark_2d_106'],
                                          base[g][0]['landmark_2d_106'])
            np.testing.assert_array_equal(off[g][0]['bbox'], base[g][0]['bbox'])

    def test_flow_actually_changes_the_landmarks(self):
        """The mutation check: a bridge that silently did nothing would pass the
        'refused' tests above. Flow and Hermite must DIFFER on this path."""
        _, base = _build(_track(None))
        _, used = _build(_track(_good_flow()))
        diffs = [float(np.abs(used[g][0]['landmark_2d_106'] - base[g][0]['landmark_2d_106']).max())
                 for g in HOLE]
        self.assertGreater(max(diffs), 1.0)


# A hole one frame longer than gap_max (10): anchors at 3 and 15, hole 4..14.
LONG_HOLE = list(range(4, 15))


def _long_track(flow=None):
    obs = {i: _face(X0, 200.0) for i in range(0, 4)}
    obs.update({i: _face(X1, 200.0) for i in range(15, 19)})
    t = {'id': 0, 'obs': obs, 'emb_mean': _emb(), 'emb_n': len(obs)}
    if flow is not None:
        t['flow'] = flow
    return t


def _long_flow(end_x=X1):
    fl = {g: _flow(X0 + (X1 - X0) * (g - 3) / 12.0, 200.0) for g in LONG_HOLE}
    fl[15] = _flow(end_x, 200.0)
    return fl


class LongHoleTest(unittest.TestCase):
    """Past gap_max the hole used to belong to the Kalman coast. A verified flow
    bridge may take it; an unverified one must leave it exactly as it was."""

    def test_a_verified_flow_path_fills_a_hole_longer_than_gap_max(self):
        m, out = _build(_long_track(_long_flow()))
        for g in LONG_HOLE:
            f = out[g][0]
            self.assertTrue(f.get('_flow_bridged'), f'frame {g}')
            self.assertFalse(f.get('_coasted'), 'a flow bridge is not a Kalman coast')
        self.assertEqual(m._flow_bridged, len(LONG_HOLE))

    def test_an_unverified_long_hole_is_left_to_the_coast(self):
        _, base = _build(_long_track(None))
        m, out = _build(_long_track(_long_flow(end_x=X1 - 80.0)))
        self.assertEqual(m._flow_bridged, 0)
        for g in LONG_HOLE:
            self.assertEqual(bool(out[g][0].get('_coasted')), bool(base[g][0].get('_coasted')))
            np.testing.assert_array_equal(out[g][0]['bbox'], base[g][0]['bbox'])

    def test_a_hole_beyond_the_flow_reach_is_left_to_the_coast(self):
        obs = {i: _face(X0, 200.0) for i in range(0, 4)}
        obs.update({i: _face(X1, 200.0) for i in range(30, 34)})
        fl = {g: _flow(X0 + 1.0 * (g - 3), 200.0) for g in range(4, 30)}
        fl[30] = _flow(X1, 200.0)
        t = {'id': 0, 'obs': obs, 'emb_mean': _emb(), 'emb_n': len(obs), 'flow': fl}
        m, out = _build(t)
        self.assertEqual(m._flow_bridged, 0)


if __name__ == '__main__':
    unittest.main()
