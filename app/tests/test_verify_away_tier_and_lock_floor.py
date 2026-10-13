"""Two ways a frontal face got painted onto a head that shows none.

Measured on D:\\out.mp4 (2026-10-14, harjot + ashna, two women), both while the
OTHER woman in the same frames swapped cleanly:

  A. frames 340-447 -- one woman head-down and turned away. The 5-point solver
     reads 90.3-92.5 deg of yaw, the swap crop lands on her ear, and the render
     shows a doubled eyebrow, a nose past the head's outline and a smeared mouth.
     The outcome guard measured it at displacement 0.65-0.88 / shape 0.40-0.54,
     which is under BOTH shipped tolerances (1.0 and 0.75), so it passed.
  B. frames 448-479 -- the other woman turns into an out-of-focus back of a head.
     The detector reports a "face" on the ear at 0.64-0.82, her embedding is
     0.89-0.96 from her own identity (gate 0.75), and the identity lock swaps her
     anyway: a whole ghost face on the ear. The guard cannot see this one
     (displacement 0.3-0.4 on half the frames).

A is fixed by judging heads turned past profile at a stricter tolerance pair
(`VERIFY_AWAY_*`); B by a detector-confidence floor on locked faces
(`LOCK_MIN_DET`). The fixtures here are the real plate keypoints of frame 350
with results constructed to read where the real swap read; the end-to-end proof
is the out.mp4 render, recorded in docs/CHANGELOG.md.
"""
import ast
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from roop import procmgr_runtime as rt                       # noqa: E402
from roop.face_util import (                                 # noqa: E402
    SWAP_MOVED_TOL, SWAP_SHAPE_TOL, keypoint_shape_change, swap_moved_the_face,
)

_PROCMGR = os.path.join(os.path.dirname(__file__), '..', 'roop', 'ProcessMgr.py')

# Frame 350, the head-down woman (production detector, refine_landmarks on).
PLATE_KPS = np.array([[477.0, 404.0], [465.0, 428.0], [499.0, 470.0],
                      [454.0, 506.0], [444.0, 521.0]])
PLATE_BOX = (350.0, 286.0, 507.0, 569.0)


class _Det:
    def __init__(self, kps):
        self.kps = np.asarray(kps, dtype=np.float64)
        x0, y0 = self.kps.min(axis=0)
        x1, y1 = self.kps.max(axis=0)
        self.bbox = np.array([x0 - 20, y0 - 20, x1 + 20, y1 + 20], np.float32)


def _scale(kps):
    interocular = float(np.linalg.norm(kps[0] - kps[1]))
    extent = float(np.linalg.norm(kps - kps.mean(axis=0), axis=1).mean())
    return max(interocular, extent * 0.70)


def _result_reading(plate, result):
    moved = float(np.linalg.norm(result - plate, axis=1).mean() / _scale(plate))
    return moved, keypoint_shape_change(plate, result)


def _result_at(moved_range, shape_range):
    """A deterministic result constellation whose guard reading falls in the
    given bands. Searched rather than hand-tuned so the fixture states the
    reading it stands for, and fails loudly if no such constellation exists."""
    for seed in range(5000):
        d = np.random.RandomState(seed).uniform(-1.0, 1.0, PLATE_KPS.shape)
        d *= (np.mean(moved_range) * _scale(PLATE_KPS)) / np.linalg.norm(d, axis=1).mean()
        result = PLATE_KPS + d
        moved, shape = _result_reading(PLATE_KPS, result)
        if (moved_range[0] <= moved <= moved_range[1]
                and shape_range[0] <= shape <= shape_range[1]):
            return result
    raise AssertionError(f'no constellation reads moved={moved_range} shape={shape_range}')


def _guard(result_kps, **kw):
    with mock.patch('roop.face_util.detect_boxes_in_roi',
                    return_value=[_Det(result_kps)]):
        return swap_moved_the_face(np.zeros((720, 1280, 3), np.uint8),
                                   PLATE_KPS, PLATE_BOX, **kw)


class TestAwayTier(unittest.TestCase):
    # What the render measured for failure A, and what a clean swap read.
    SMUDGED = ((0.65, 0.88), (0.40, 0.54))
    CLEAN_PROFILE = ((0.20, 0.35), (0.12, 0.22))      # frames 202/203, yaw 85

    def test_the_shipped_tolerances_pass_the_smudged_swap(self):
        """Documents the defect: this is the reading that used to get through."""
        result = _result_at(*self.SMUDGED)
        moved, shape = _result_reading(PLATE_KPS, result)
        self.assertLess(moved, SWAP_MOVED_TOL)
        self.assertLess(shape, SWAP_SHAPE_TOL)
        self.assertFalse(_guard(result), 'the shipped guard started catching it '
                                         'by itself; the away tier may be redundant')

    def test_the_away_tier_discards_it(self):
        result = _result_at(*self.SMUDGED)
        self.assertTrue(_guard(result, tol=rt.VERIFY_AWAY_MOVED_TOL,
                               shape_tol=rt.VERIFY_AWAY_SHAPE_TOL))

    def test_the_away_tier_leaves_a_clean_profile_alone(self):
        result = _result_at(*self.CLEAN_PROFILE)
        self.assertFalse(_guard(result, tol=rt.VERIFY_AWAY_MOVED_TOL,
                                shape_tol=rt.VERIFY_AWAY_SHAPE_TOL))

    def test_the_tier_is_a_tightening_not_a_loosening(self):
        self.assertLess(rt.VERIFY_AWAY_MOVED_TOL, SWAP_MOVED_TOL)
        self.assertLess(rt.VERIFY_AWAY_SHAPE_TOL, SWAP_SHAPE_TOL)
        self.assertGreater(rt.VERIFY_AWAY_YAW, 85.4,
                           'every other face of the clip reads <= 85.4 deg; a '
                           'lower trigger puts ordinary profiles in the tier')
        self.assertLess(rt.VERIFY_AWAY_YAW, 90.3,
                        'the smudged stretch reads 90.3-92.5 deg')

    def test_shape_tol_default_is_the_module_constant(self):
        result = _result_at((0.65, 0.88), (0.40, 0.54))
        self.assertEqual(_guard(result, shape_tol=None), _guard(result))


class TestWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(_PROCMGR, encoding='utf-8') as fh:
            cls.src = fh.read()
        cls.tree = ast.parse(cls.src)

    def _fn(self, name):
        return next(n for n in ast.walk(self.tree)
                    if isinstance(n, ast.FunctionDef) and n.name == name)

    def test_process_face_picks_the_tier_from_the_jaw_free_yaw(self):
        body = ast.get_source_segment(self.src, self._fn('process_face'))
        self.assertIn('VERIFY_AWAY_YAW', body)
        self.assertIn('_head_angles()[0]', body,
                      'the tier must read the jaw-free yaw, not the 5-point one')
        call = next(n for n in ast.walk(self._fn('process_face'))
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == '_verify_after')
        self.assertIn('shape_tol', {k.arg for k in call.keywords},
                      'the stricter shape tolerance is computed and then dropped')

    def test_names_used_by_the_tier_are_importable_in_processmgr(self):
        """A NameError inside the guarded block is swallowed (`_swallowed`) and
        reads as 'no effect' -- the failure class this project keeps hitting."""
        import roop.ProcessMgr as PM
        for name in ('VERIFY_AWAY_YAW', 'VERIFY_AWAY_MOVED_TOL',
                     'VERIFY_AWAY_SHAPE_TOL', 'LOCK_MIN_DET', 'SWAP_MOVED_TOL'):
            self.assertTrue(hasattr(PM, name), f'ProcessMgr has no {name}')


class TestLockFloor(unittest.TestCase):
    def test_floor_sits_between_the_back_of_head_and_every_real_face(self):
        # back-of-head detections 0.64-0.82; the lowest REAL swapped face of the
        # clip outside three transition flashes is 0.84.
        self.assertGreater(rt.LOCK_MIN_DET, 0.82)
        self.assertLessEqual(rt.LOCK_MIN_DET, 0.85)

    def test_the_lock_path_applies_the_floor_and_falls_back_to_matching(self):
        with open(_PROCMGR, encoding='utf-8') as fh:
            src = fh.read()
        self.assertIn('skipped (low-confidence detection on a locked track)', src)
        i = src.index('low_conf_lock = False')
        j = src.index("_audit_hit('swapped (identity lock)')")
        self.assertLess(i, j, 'the floor must run BEFORE the lock swaps the face')
        seg = src[i:j]
        self.assertIn('LOCK_MIN_DET', seg)
        self.assertIn('src_index = None', seg,
                      'a face under the floor must lose the lock, not just be counted')
        # ...and must not be mislabelled as "no track entry matched" afterwards.
        self.assertIn('if veto is None and not low_conf_lock:', src)

    def test_floor_can_be_turned_off_and_a_bad_value_falls_back(self):
        # Not a module reload: that would reset the audit counters other tests
        # share. The constants are all read through this helper.
        with mock.patch.dict(os.environ, {'ROOP_LOCK_MIN_DET_PROBE': '0'}):
            self.assertEqual(rt._env_float('ROOP_LOCK_MIN_DET_PROBE', 0.85), 0.0)
        with mock.patch.dict(os.environ, {'ROOP_LOCK_MIN_DET_PROBE': 'abc'}):
            self.assertEqual(rt._env_float('ROOP_LOCK_MIN_DET_PROBE', 0.85), 0.85)
        self.assertEqual(rt._env_float('ROOP_LOCK_MIN_DET_UNSET_X', 0.85), 0.85)


if __name__ == '__main__':
    unittest.main()
