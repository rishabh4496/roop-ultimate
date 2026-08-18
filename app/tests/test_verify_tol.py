"""Per-model tolerance for the swap outcome guard.

face_util.swap_moved_the_face throws away a swap that put the face somewhere
the plate's face was not. Its threshold used to be one shared constant, which
is one too few: how far a CLEAN swap moves the keypoints is a property of the
swapper, and hififace's clean band stops at less than half of where inswapper's
and hyperswap's do (see the measurement table in FaceSwapInsightFace.py). So
hififace now carries its own, tighter value and everything else keeps the
shared default.

Three things have to stay true, and each of them has already been a bug shape
somewhere else in this file's neighbourhood:

  * a model WITHOUT a measured value must fall back to the shared default, not
    inherit whichever model was loaded before it (the same stale-per-processor
    trap `model_has_mask` had);
  * the value must actually reach swap_moved_the_face — a threshold that is
    published, read, and then dropped on the floor is the quiet failure here,
    because the guard keeps working and simply keeps using 1.0;
  * the tolerance must be a REDUCTION. It was introduced from a measurement
    saying hififace trips the guard on frames the others do not, and a value
    above the default would silently invert that.
"""
import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from roop.face_util import SWAP_MOVED_TOL                       # noqa: E402
from roop.processors.FaceSwapInsightFace import (               # noqa: E402
    SWAP_MODELS, FaceSwapInsightFace, verify_tol_for,
)

_PROCMGR = os.path.join(os.path.dirname(__file__), '..', 'roop', 'ProcessMgr.py')


def _fn(name):
    with open(_PROCMGR, encoding='utf-8') as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f'{name} not found in ProcessMgr.py')


class TestSpec(unittest.TestCase):

    def test_only_hififace_overrides_the_default(self):
        overridden = {k: v['verify_tol'] for k, v in SWAP_MODELS.items()
                      if v.get('verify_tol') is not None}
        self.assertEqual(list(overridden), ['hififace'],
                         'the override is measured per model; adding one '
                         'without a measurement behind it is how the shared '
                         'default stopped meaning anything')

    def test_override_is_tighter_than_the_shared_default(self):
        for name, tol in ((k, v.get('verify_tol')) for k, v in SWAP_MODELS.items()):
            if tol is None:
                continue
            self.assertGreater(tol, 0.0, f'{name}: a tolerance of 0 rejects '
                                         f'every swap, including honest ones')
            self.assertLess(tol, SWAP_MOVED_TOL,
                            f'{name}: the per-model value exists to TIGHTEN '
                            f'the guard; above the default it would loosen it')

    def test_fresh_processor_reports_no_override(self):
        # A processor that never loaded a model must not force a threshold.
        self.assertIsNone(verify_tol_for(FaceSwapInsightFace()))

    def test_reader_is_none_safe(self):
        # process_face reads through this with whatever `swap_p` resolved to,
        # and that is None when no swap processor is configured.
        self.assertIsNone(verify_tol_for(None))

    def test_release_clears_the_override(self):
        p = FaceSwapInsightFace()
        p.model_verify_tol = SWAP_MODELS['hififace']['verify_tol']
        p.model_swap_insightface = None
        p.Release()
        self.assertIsNone(p.model_verify_tol,
                          'a stale tolerance surviving Release would apply '
                          'hififace\'s threshold to the next model loaded')

    def test_initialize_assigns_unconditionally(self):
        """Initialize must SET the attribute for every model, not only for the
        ones that have a value — otherwise switching hififace -> inswapper on a
        reused processor leaves 0.65 in place."""
        path = os.path.join(os.path.dirname(__file__), '..', 'roop',
                            'processors', 'FaceSwapInsightFace.py')
        with open(path, encoding='utf-8') as fh:
            src = fh.read()
        tree = ast.parse(src)
        init = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == 'Initialize')
        assigns = [n for n in ast.walk(init) if isinstance(n, ast.Assign)
                   for t in n.targets
                   if isinstance(t, ast.Attribute) and t.attr == 'model_verify_tol']
        self.assertTrue(assigns, 'Initialize never assigns model_verify_tol')
        for a in assigns:
            self.assertIsInstance(
                a.value, ast.Call,
                'expected spec.get("verify_tol"), which yields None for the '
                'models that have no measured value')


class TestWiring(unittest.TestCase):

    def test_verify_after_forwards_tol(self):
        """The published value must reach the guard. Patched in ProcessMgr's own
        namespace because that is the binding _verify_after actually calls."""
        import roop.ProcessMgr as PM
        import numpy as np

        seen = {}

        def fake(result, kps, bbox, tol=None, rotation_action=None):
            seen['tol'] = tol
            seen['rotation_action'] = rotation_action
            return False        # "did not move" -> swap stands, result returned

        orig = PM.swap_moved_the_face
        PM.swap_moved_the_face = fake
        try:
            frame = np.zeros((40, 40, 3), np.uint8)
            snap = (np.zeros((5, 2), np.float32), np.array([0, 0, 10, 10]),
                    (0, 0, 10, 10), frame[0:10, 0:10].copy())
            PM.ProcessMgr._verify_after(frame, snap, tol=0.65)
            self.assertEqual(seen.get('tol'), 0.65)
            # The turn must reach the guard too, or the re-detection reads a
            # rolled face with the raw detector and discards good swaps —
            # measured as 2 un-swapped frames in a 0-360 roll sweep.
            seen.clear()
            PM.ProcessMgr._verify_after(frame, snap, rotation_action="rotate_180")
            self.assertEqual(seen.get('rotation_action'), "rotate_180")
            seen.clear()
            PM.ProcessMgr._verify_after(frame, snap)
            self.assertIsNone(seen.get('tol'),
                              'the default must stay None so face_util picks '
                              'the shared constant')
        finally:
            PM.swap_moved_the_face = orig

    def test_process_face_passes_the_loaded_model_s_tol(self):
        """The call site must read the tolerance off the swap processor rather
        than hardcoding one — a literal here would pin every model to hififace's
        value the moment anything else gets a measurement."""
        call = None
        for node in ast.walk(_fn('process_face')):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == '_verify_after'):
                call = node
        self.assertIsNotNone(call, '_verify_after is no longer called')
        kw = {k.arg: k.value for k in call.keywords}
        self.assertIn('tol', kw, '_verify_after called without a tolerance; '
                                 'the per-model value would be inert')
        self.assertIsInstance(kw['tol'], ast.Call,
                              'tol should come from the reader, not a literal')


if __name__ == '__main__':
    unittest.main()
