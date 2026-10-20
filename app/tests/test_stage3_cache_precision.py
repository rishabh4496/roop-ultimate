"""The faces Stage 2 hands to Stage 3 keep float32 and image-space coordinates.

`_build_temporal_faces` gap-fills (interpolates) and smooths landmarks. Doing that arithmetic in
Python floats or float64 would silently widen the cached arrays, and every consumer downstream
(alignment, the paste hull, the mouth mask) was tuned on float32 input. On a vertical or
non-square frame an x/y mix-up in the interpolation would show as a landmark off by hundreds of
pixels, so the gap-filled face is also checked against the straight line between its anchors.
Measured on real footage (docs/perf/stage3_plan/geom_*.json): every cached array float32, 266/266
faces on 1280x720 and 90/90 on a 9:16 crop; this pins it for the synthetic case that runs in CI.
"""
import contextlib
import io
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import roop.globals                                              # noqa: E402
from roop.procmgr_tracking import TrackingMixin                  # noqa: E402

EMB = np.zeros(512, np.float32)
EMB[0] = 1.0


from insightface.app.common import Face as _InsightFace     # noqa: E402


def _Face(cx, cy, size=100.0):
    rng = np.random.RandomState(3)
    pts5 = np.array([[-.3, -.2], [.3, -.2], [0, 0], [-.25, .25], [.25, .25]], np.float32) * size
    lm106 = rng.rand(106, 2).astype(np.float32) * size - size / 2
    lm68 = np.concatenate([rng.rand(68, 2).astype(np.float32) * size - size / 2,
                           rng.rand(68, 1).astype(np.float32)], axis=1)
    c = np.array([cx, cy], np.float32)
    return _InsightFace(dict(
        bbox=np.array([cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2], np.float32),
        kps=(pts5 + c).astype(np.float32), det_score=np.float32(0.9),
        landmark_2d_106=(lm106 + c).astype(np.float32),
        landmark_3d_68=(lm68 + np.array([cx, cy, 0], np.float32)).astype(np.float32),
        pose=np.array([1, 2, 3], np.float32), embedding=EMB.copy()))


class _Mgr(TrackingMixin):
    def __init__(self):
        self.target_face_datas = [type('C', (), {'embedding': EMB})()]
        self.target_face_groups = [0]
        self.options = type('O', (), {'face_distance_threshold': 0.75, 'selected_index': 0})()
        self.progress_gradio = None
        self._track_assignments = {}
        self._track_scanned = 0

    def _publish_live(self, frame):
        pass


def build(script, n):
    mgr = _Mgr()
    was, roop.globals.processing = roop.globals.processing, True
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                mock.patch('roop.face_util.get_all_faces', lambda fr, *a, **k: list(script.get(fr, []))):
            tracks = mgr._precompute_tracks(None, 0, n, n, awebp_frames=list(range(n)),
                                            step=1, collect_obs=True)
            return mgr._build_temporal_faces(tracks, 10)
    finally:
        roop.globals.processing = was


class CachePrecision(unittest.TestCase):
    def _check(self, w, h):
        # a face drifting right with a 3-frame detection miss in the middle
        xs = {f: (w * 0.3 + 4 * f) for f in range(0, 20)}
        script = {f: [_Face(xs[f], h * 0.4)] for f in range(0, 20) if f not in (8, 9, 10)}
        faces = build(script, 20)
        filled = [f for fr in faces.values() for f in fr if f.get('_interpolated')]
        real = [f for fr in faces.values() for f in fr if not f.get('_interpolated')]
        self.assertEqual(len(filled), 3, 'the 3-frame miss was not gap-filled')
        for f in filled + real:
            for name in ('bbox', 'kps', 'landmark_2d_106', 'landmark_3d_68', 'pose'):
                self.assertEqual(np.asarray(f[name]).dtype, np.float32, f'{name} widened')
            self.assertEqual(np.asarray(f['landmark_2d_106']).shape, (106, 2))
            self.assertEqual(np.asarray(f['landmark_3d_68']).shape, (68, 3))
        for idx, fr in faces.items():
            for f in fr:
                if f.get('_interpolated'):
                    cx = float((f['bbox'][0] + f['bbox'][2]) / 2)
                    cy = float((f['bbox'][1] + f['bbox'][3]) / 2)
                    self.assertAlmostEqual(cx, xs[idx], delta=2.0)      # on the line between anchors
                    self.assertAlmostEqual(cy, h * 0.4, delta=2.0)
                    lm = np.asarray(f['landmark_2d_106'])
                    self.assertLess(abs(float(lm[:, 0].mean()) - xs[idx]), 12.0)
                    self.assertLess(abs(float(lm[:, 1].mean()) - h * 0.4), 12.0)

    def test_landscape(self):
        self._check(1280, 720)

    def test_vertical_9_16(self):
        self._check(404, 720)

    def test_square(self):
        self._check(720, 720)


if __name__ == '__main__':
    unittest.main()
