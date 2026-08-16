"""Guards on the two per-face costs that turned a GPU-bound pipeline CPU-bound.

Both were correctness fixes that shipped without a cost gate: the outcome guard
re-ran a FULL face analysis on every swapped face to read two of its fields, and
autorotate pulled landmark_3d_68 into the per-face loop of every detection on
every clip. Measured on an RTX 4070 under TensorRT with retinaface_r50 @640,
4 worker threads:

    the guard's re-detect, full analysis -> detector only   +49.6% throughput
    detection, 68-point model in-loop -> on demand          +28.7% throughput

Neither number survives an innocent-looking edit, so the shape of both fixes is
pinned here rather than the numbers.
"""

import ast
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

FACE_UTIL = os.path.join(APP, 'roop', 'face_util.py')
PROCMGR = os.path.join(APP, 'roop', 'ProcessMgr.py')


def _source(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


def _func(name, src):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise AssertionError(f'{name} not found')


def _code(name, src):
    """The function with its docstring and comments stripped.

    A test that greps a body for an identifier is otherwise answered by the
    prose explaining why that identifier is NOT used there, which is the
    opposite of what it meant to assert.
    """
    tree = ast.parse(_func(name, src))
    fn = tree.body[0]
    if (fn.body and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str)):
        fn.body = fn.body[1:]
    return ast.unparse(fn)


class OutcomeGuardIsDetectorOnly(unittest.TestCase):
    """The guard reads `bbox` and `kps`. Everything else it used to compute —
    a 512-d embedding and 174 landmark points per swapped face — was thrown
    away on the next line."""

    def setUp(self):
        self.src = _source(FACE_UTIL)

    def test_guard_uses_the_detector_only_path(self):
        body = _func('swap_moved_the_face', self.src)
        self.assertIn('detect_boxes_in_roi', body)
        self.assertNotIn('get_all_faces_in_roi', body,
                         'the outcome guard is back on the full analysis path; '
                         'it reads only bbox/kps, so recognition and both '
                         'landmark models are pure cost')

    def test_guard_reads_nothing_the_detector_does_not_produce(self):
        """If it ever needs an embedding or landmarks again, aux=False is wrong
        and this test is the place that says so."""
        body = _code('swap_moved_the_face', self.src)
        for attr in ('embedding', 'normed_embedding',
                     'landmark_2d_106', 'landmark_3d_68'):
            self.assertNotIn(attr, body,
                             f'guard reads {attr}, which detect_boxes_in_roi '
                             f'does not fill in')

    def test_detector_only_path_skips_the_aux_models(self):
        body = _func('detect_boxes_in_roi', self.src)
        self.assertIn('aux=False', body)

    def test_aux_flag_reaches_every_engine(self):
        """Four hybrid engines plus SCRFD. An engine left on the aux path is an
        engine where this saves nothing."""
        raw = _func('_detect_faces_raw', self.src)
        self.assertRegex(raw, r'def _detect_faces_raw\([^)]*aux=True',
                         'aux must be a parameter of _detect_faces_raw')
        for engine in ('_hybrid_yolo_faces', '_hybrid_retinaface_faces',
                       '_hybrid_yunet_faces'):
            for call in re.findall(re.escape(engine) + r'\([^)]*\)', raw):
                self.assertIn('aux=aux', call, f'{call} drops the aux flag')
        # SCRFD: fa.get() always runs the aux loop, so the detector-only case
        # must not go through it.
        self.assertIn('fa.det_model.detect(', raw,
                      'SCRFD has no detector-only branch, so aux=False silently '
                      'still pays for the aux models on the default engine')

    def test_hybrid_wrapper_honours_aux(self):
        body = _func('_hybrid_detector_faces', self.src)
        self.assertRegex(body, r'def _hybrid_detector_faces\([^)]*aux=True')
        self.assertIn('if aux:', body)


class OutcomeGuardIsPreFiltered(unittest.TestCase):
    """A detection per swapped face, on footage whose faces are frontal, cannot
    find anything: the failure is a face painted on a head pointing AWAY."""

    def setUp(self):
        self.src = _source(PROCMGR)

    def test_gate_exists_and_is_consulted(self):
        self.assertIn('_verify_worth_it', self.src)
        self.assertRegex(self.src,
                         r'if _vs is not None and self\._verify_worth_it\(')

    def test_rolled_faces_are_always_checked(self):
        body = _func('_verify_worth_it', self.src)
        self.assertRegex(body, r'if rotation_action is not None:\s*\n\s*return True')

    def test_unreadable_pose_falls_back_to_checking(self):
        body = _func('_verify_worth_it', self.src)
        self.assertRegex(body, r'except Exception:\s*\n\s*return True')

    def test_threshold_keeps_margin_under_the_measured_failure(self):
        """The guard fires from |yaw| 43.6 deg upward on the calibration clips
        (tests/angle_video.py, yaw +-90). solve_pose_5pt is 15-20 deg off per
        person, so a gate above ~35 has no room for that error."""
        from roop.procmgr_runtime import VERIFY_MIN_OFFAXIS
        self.assertGreater(VERIFY_MIN_OFFAXIS, 0)
        self.assertLessEqual(VERIFY_MIN_OFFAXIS, 35.0,
                             'gate is too close to the lowest reading (43.6) '
                             'that needs checking')

    def test_the_guard_is_profiled(self):
        """It is a detection. Every other model stage reports into STAGE TIMING,
        and an unprofiled one is a cost with nothing to blame it on."""
        self.assertRegex(self.src, r"_prof\('verify'\)")

    def test_the_guard_takes_the_gpu_guard(self):
        """A detection from a swap worker thread. Without this, a card small
        enough for the analyser pool to be disabled runs N threads through one
        shared ORT session."""
        self.assertRegex(
            self.src,
            r"_prof\('verify'\), _gpu_guard\(pooled=analysis_pooled\(\)\)")


class Lm68IsLazyWhenOnlyAutorotateWantsIt(unittest.TestCase):

    def test_pose_features_keep_it_in_the_loop(self):
        """They read the landmarks of every face they touch; for them the
        on-demand path would be slower, not faster."""
        body = _func('initialize', _source(PROCMGR))
        self.assertIn('pose_features_need_68', body)
        self.assertRegex(body, r'lm68_lazy\s*=\s*\(not pose_features_need_68')

    def test_model_is_still_requested_so_it_can_be_run_on_demand(self):
        """Lazy means OUT OF THE LOOP, not absent — dropping it from the module
        list would leave nothing to call."""
        body = _func('initialize', _source(PROCMGR))
        self.assertRegex(body, r"modules\.insert\(0, .landmark_3d_68.\)")
        self.assertIn('autorotate_faces', body)

    def test_pool_rebuilds_when_laziness_changes(self):
        """The model is popped out of fa.models at BUILD time, so a pool built
        under one setting is wrong under the other."""
        body = _func('_ensure_face_analyser', _source(FACE_UTIL))
        self.assertIn('_ANALYSER_LM68_LAZY', body)
        self.assertIn('cur_lm68_lazy', body)

    def test_detect_measures_before_anything_reads_the_axis(self):
        """_upright_remeasure's entire gate is the orientation axis, so filling
        it in afterwards would be filling it in too late.

        Lives in _enrich_detected_faces, not _detect_faces itself: that
        ordering-critical block was factored out so a caller that ran its own
        raw detector pass (e.g. get_all_faces_hires's higher-resolution retry)
        gets the identical enrichment pipeline instead of a divergent copy."""
        body = _code('_enrich_detected_faces', _source(FACE_UTIL))
        self.assertLess(body.index('_lm68_should_measure'),
                        body.index('_upright_remeasure('),
                        'the 68-point axis is resolved after the thing that '
                        'reads it')

    def test_escalation_has_both_a_ramp_and_a_probe(self):
        """The ramp catches a head rotating into the blind band; only the probe
        catches a cut straight to one."""
        body = _func('_lm68_should_measure', _source(FACE_UTIL))
        self.assertIn('LM68_PROBE', body)
        self.assertIn('LM68_ARM_DEG', body)
        self.assertIn('_lm68_arm', body)

    def test_ramp_threshold_is_far_below_the_blind_band(self):
        """The keypoints go wrong from ~140 deg. Arming has to happen while
        they are still right, or the latch is never set."""
        from roop import face_util
        self.assertGreater(face_util.LM68_ARM_DEG, 0)
        self.assertLess(face_util.LM68_ARM_DEG, 90.0)

    def test_arm_threshold_uses_the_keypoint_axis_not_the_68_one(self):
        """Deciding whether to run the model with the output of the model is
        the one way to make this a no-op."""
        body = _func('_lm68_should_measure', _source(FACE_UTIL))
        self.assertIn('_axis_from_kps', body)
        self.assertNotIn('_axis_from_68', body)

    def test_probe_arms_the_latch_on_disagreement(self):
        """See test_detect_measures_before_anything_reads_the_axis: this logic
        now lives in _enrich_detected_faces, shared with get_all_faces_hires."""
        body = _func('_enrich_detected_faces', _source(FACE_UTIL))
        self.assertIn('LM68_DISAGREE_DEG', body)
        self.assertIn('_lm68_arm', body)

    def test_remeasure_compares_like_with_like(self):
        """The outcome test compares a candidate's tilt with the original's. A
        68-point reading against a keypoint reading is not a comparison — on the
        faces this exists for they differ by up to 172 deg."""
        body = _func('_upright_remeasure', _source(FACE_UTIL))
        self.assertIn('used_68', body)
        self.assertIn('ensure_landmark_3d_68', body)

    def test_state_resets_between_runs(self):
        body = _func('initialize', _source(PROCMGR))
        self.assertIn('reset_lm68_state', body)

    def test_ensure_is_a_noop_without_a_lazy_model(self):
        """When the model is in the pipeline the faces already carry the
        landmarks, and when nobody asked there is no model — neither may raise."""
        from roop import face_util

        class Fake:
            landmark_3d_68 = None
            kps = None

        self.assertEqual(face_util.ensure_landmark_3d_68(None, []), [])
        faces = [Fake()]
        self.assertIs(face_util.ensure_landmark_3d_68(None, faces), faces)


class AxisHelpersStaySeparable(unittest.TestCase):
    """face_down_axis prefers the 68-point midline; the lazy path needs the two
    sources callable independently to decide whether to ask for it."""

    def test_both_sources_exist(self):
        from roop import face_util
        self.assertTrue(callable(face_util._axis_from_68))
        self.assertTrue(callable(face_util._axis_from_kps))

    def test_face_down_axis_still_prefers_68(self):
        body = _func('face_down_axis', _source(FACE_UTIL))
        self.assertLess(body.index('_axis_from_68'), body.index('_axis_from_kps'))

    def test_kps_axis_matches_the_documented_construction(self):
        import numpy as np
        from roop import face_util

        class Fake:
            pass

        f = Fake()
        # eyes level at y=0, mouth corners below at y=40 -> chin points +y
        f.kps = np.array([[0, 0], [40, 0], [20, 20], [5, 40], [35, 40]],
                         dtype=np.float32)
        dx, dy = face_util._axis_from_kps(f)
        self.assertAlmostEqual(dx, 0.0, places=4)
        self.assertGreater(dy, 0.0)
        self.assertAlmostEqual(face_util._tilt_from_axis((dx, dy)), 0.0, places=4)

    def test_tilt_reads_180_for_an_inverted_face(self):
        from roop import face_util
        self.assertAlmostEqual(abs(face_util._tilt_from_axis((0.0, -50.0))),
                               180.0, places=4)


if __name__ == '__main__':
    unittest.main()
