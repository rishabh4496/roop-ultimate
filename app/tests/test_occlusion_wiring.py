"""Is the occlusion work actually REACHED on the shipped default configuration?

This file exists because of the failure mode this project keeps rediscovering:
code that is complete, correct, covered by unit tests, and never executed.
`roop/occlusion_mask.py` was exactly that -- a finished implementation of
`M_composite = M_face_blend * (1 - M_occluder)`, with the algebra written out in
its docstring, that no caller anywhere invoked. The suite was green through all
of it, because every test called the function directly.

So these are wiring assertions, not behaviour assertions. They fail when a call
site disappears, which no amount of unit coverage on the callee can catch. The
call sites are found by walking the AST rather than by grepping, so a mention
inside a comment or a docstring cannot satisfy them.
"""

import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

APP = Path(__file__).resolve().parents[1]


def _calls_in(path, function_name):
    """Every function name called anywhere inside `function_name`'s body.

    Methods count by their attribute name, so `self._coast_track_gaps(...)`
    registers as `_coast_track_gaps`.
    """
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != function_name:
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            target = inner.func
            if isinstance(target, ast.Attribute):
                found.add(target.attr)
            elif isinstance(target, ast.Name):
                found.add(target.id)
    return found


class OccluderIsInjectedTest(unittest.TestCase):
    """Spec 3: the foreground occluder must reach the shipped default chain."""

    def test_process_mgr_initialize_injects_the_occluder(self):
        calls = _calls_in(APP / 'roop' / 'ProcessMgr.py', 'initialize')
        self.assertIn('inject_occlusion_engine', calls,
                      'ProcessMgr.initialize no longer injects the occluder; '
                      'the default chain has nothing trained to find a hand, '
                      'a mug or a microphone in front of a face.')

    def test_the_default_chain_gains_an_occluder(self):
        """RealityUX + no second engine is what config.yaml actually ships."""
        from roop.occlusion_mask import OCCLUSION_ENGINE, inject_occlusion_engine

        default = {'faceswap': {'swap_model': 'realswap'},
                   'ultramax': {},
                   'mask_realityux': {}}
        updated, note = inject_occlusion_engine(dict(default), enabled=True)
        self.assertIn(OCCLUSION_ENGINE, updated)
        self.assertTrue(note)
        # Order IS execution order, and the product only comes out right when
        # the occluder runs after the face mask. See occlusion_mask.py.
        self.assertEqual(list(updated)[-1], OCCLUSION_ENGINE)

    def test_an_occluder_already_in_the_chain_is_not_doubled(self):
        from roop.occlusion_mask import inject_occlusion_engine

        chain = {'faceswap': {}, 'mask_occluder': {}}
        updated, note = inject_occlusion_engine(dict(chain), enabled=True)
        self.assertEqual(list(updated), list(chain))
        self.assertEqual(note, '')

    def test_a_mask_only_chain_is_left_alone(self):
        """The preview mask editor has no swap to protect."""
        from roop.occlusion_mask import inject_occlusion_engine

        chain = {'mask_realityux': {}}
        updated, _ = inject_occlusion_engine(dict(chain), enabled=True)
        self.assertEqual(list(updated), list(chain))

    def test_disabling_it_is_a_true_no_op(self):
        from roop.occlusion_mask import inject_occlusion_engine

        chain = {'faceswap': {}, 'mask_realityux': {}}
        updated, note = inject_occlusion_engine(dict(chain), enabled=False)
        self.assertEqual(list(updated), list(chain))
        self.assertEqual(note, '')


class CoastingIsReachedTest(unittest.TestCase):
    """Spec 1: persistence has to run on the path the user actually renders."""

    def test_the_temporal_prepass_coasts_its_gaps(self):
        calls = _calls_in(APP / 'roop' / 'procmgr_tracking.py',
                          '_build_temporal_faces')
        self.assertIn('_coast_track_gaps', calls,
                      'the whole-clip pre-pass -- the SHIPPED default, '
                      'temporal_detection: true -- no longer coasts, so an '
                      'occlusion longer than the gap limit blinks the swap off '
                      'again.')

    def test_the_per_frame_path_coasts_before_giving_up(self):
        calls = _calls_in(APP / 'roop' / 'ProcessMgr.py', 'swap_faces')
        self.assertIn('coast', calls,
                      'the per-frame detection path no longer coasts; a frame '
                      'where the detector found nothing goes straight to the '
                      '"no face at all" return.')

    def test_coasting_runs_before_the_no_face_return(self):
        """Ordering is the fix. After that return it can never fire.

        `swap_faces` bails out the moment `faces` is empty, which is precisely
        the frame an occlusion produces. A coast placed after that line is
        unreachable on exactly the frames it exists for -- and would still pass
        a test that only asked whether it was called.
        """
        source = (APP / 'roop' / 'ProcessMgr.py').read_text(encoding='utf-8')
        coast_at = source.index('self._dispatch_face_tracker.coast(')
        bail_at = source.index("_audit_hit('frames with no face detected at all')")
        self.assertLess(coast_at, bail_at)


class OcclusionStateIsStampedTest(unittest.TestCase):
    """Spec 2: the pipeline has to carry an `occlusion_state` flag."""

    def test_process_mask_stamps_the_state(self):
        calls = _calls_in(APP / 'roop' / 'procmgr_masking.py', 'process_mask')
        self.assertIn('_stamp_occlusion_state', calls)

    def test_the_stamp_names_the_states_the_tracker_defines(self):
        """One vocabulary, so a consumer cannot be reading a stale spelling."""
        from roop import tracker

        self.assertEqual(
            {tracker.STATE_VISIBLE, tracker.STATE_PARTIAL, tracker.STATE_COASTED},
            {'visible', 'partial', 'coasted'})

    def test_the_stamp_is_confined_to_the_occluder_family(self):
        """A face-shape masker is HIGH on background too, so it must not drive this.

        Reading `occlusion_state` off RealityUX or XSeg would mark every jaw and
        hairline landmark occluded and put essentially every frame into the
        partial state -- a flag that is always on carries no information.
        """
        source = (APP / 'roop' / 'procmgr_masking.py').read_text(encoding='utf-8')
        stamp_at = source.index('_stamp_occlusion_state(target_face, img_mask, M)')
        guard_at = source.index("if p_name in ('mask_occluder', 'mask_xseg3'):")
        dense_at = source.index("if p_name in dense_maskers and kps is not None")
        self.assertLess(guard_at, stamp_at)
        self.assertLess(stamp_at, dense_at)


class SwapperSurfaceTest(unittest.TestCase):
    """Spec 2: `face_swapper.py`'s own surface, and its honesty about scope."""

    def test_it_exposes_tracklet_persistence(self):
        from roop.processors.frame import face_swapper

        self.assertTrue(callable(face_swapper.track_faces))
        self.assertTrue(callable(face_swapper.process_frame_tracked))

    def test_the_occluder_mask_precedes_the_landmark_consumers(self):
        """Blink and mouth retention must not read hallucinated landmarks.

        The mask is the only thing in that function that knows which landmarks
        are behind an object. Computed after them -- as it was -- it cannot
        inform them, and the target's eyelid gets pasted through the hand.
        """
        source = (APP / 'roop' / 'processors' / 'frame' /
                  'face_swapper.py').read_text(encoding='utf-8')
        mask_at = source.index('occlusion_mask = compute_occlusion_mask(')
        repair_at = source.index('pts_crop, occlusion_state = _repair_occluded_landmarks(')
        dynamics_at = source.index('swapped_crop, _ = apply_facial_dynamics(')
        self.assertLess(mask_at, repair_at)
        self.assertLess(repair_at, dynamics_at)

    def test_clearing_temporal_state_also_clears_the_tracklets(self):
        """A tracker carried across clips coasts the last clip's people into this one."""
        from roop.processors.frame import face_swapper

        face_swapper._GLOBAL_TRACKER.update(
            [{'bbox': (0.0, 0.0, 10.0, 10.0)}], 0)
        self.assertTrue(face_swapper._GLOBAL_TRACKER.tracks)
        face_swapper.clear_temporal_state()
        self.assertFalse(face_swapper._GLOBAL_TRACKER.tracks)

    def test_it_says_it_is_not_the_render_path(self):
        """Its header must keep pointing at ProcessMgr, or the next reader edits dead code."""
        source = (APP / 'roop' / 'processors' / 'frame' /
                  'face_swapper.py').read_text(encoding='utf-8')
        self.assertIn('It is not', source)
        self.assertIn('roop.ProcessMgr', source)


if __name__ == '__main__':
    unittest.main()
