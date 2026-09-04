"""Regression guards for correctness fixes found during the five-stage audit.

These source-level checks deliberately avoid loading the model stack. They pin
small integration seams whose failure would otherwise show up only in a long
video render or in UI guidance.
"""

import ast
import os
import unittest


APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*parts):
    with open(os.path.join(APP, *parts), encoding='utf-8') as fh:
        return fh.read()


class AuditRegressionTests(unittest.TestCase):
    def test_in_memory_video_frame_count_is_end_exclusive(self):
        tree = ast.parse(_read('roop', 'ProcessMgr.py'))
        assignments = [node for node in ast.walk(tree)
                       if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == 'frame_count'
                               for target in node.targets)]
        video_assignment = next(
            node for node in assignments
            if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Sub)
            and isinstance(node.value.left, ast.Name) and node.value.left.id == 'frame_end'
            and isinstance(node.value.right, ast.Name) and node.value.right.id == 'frame_start')
        self.assertIsInstance(video_assignment.value.op, ast.Sub)

    def test_backend_does_not_disable_tls_verification_globally(self):
        self.assertNotIn('_create_unverified_context', _read('run.py'))

    def test_calibration_signature_uses_effective_worker_count(self):
        source = _read('api.py')
        self.assertIn('threads=roop_globals.execution_threads', source)
        self.assertNotIn('threads=roop_globals.CFG.max_threads', source)

    def test_pool_help_matches_the_runtime_policy(self):
        source = _read('..', 'react-ui', 'src', 'components', 'Settings.jsx')
        self.assertIn('15.5GB+ = 4', source)
        self.assertNotIn('15.5GB+ = 8', source)

    def test_stopped_run_cannot_enter_success_history_path(self):
        source = _read('api.py')
        stop_guard = source.index('if _stop_requested["flag"]:',
                                  source.index('batch_process_regular('))
        upscale = source.index('# ── AI upscale second pass', stop_guard)
        history = source.index('_record_run_history(', upscale)
        guard = source[stop_guard:upscale]
        self.assertIn('_progress["desc"] = "Stopped"', guard)
        self.assertIn('return', guard)
        self.assertGreater(history, upscale)

    def test_sample_runner_uses_requested_ashna_rhythm_facesets(self):
        source = _read('tests', 'run_all_samples.py')
        self.assertIn('load_library_faceset("rhythm")', source)
        self.assertIn('load_library_faceset("ashna")', source)
        self.assertNotIn('load_library_faceset("harjot")', source)
        self.assertNotIn('load_library_faceset("shambhavi")', source)

    def test_realswap_eye_band_default_matches_measured_safe_opacity(self):
        source = _read('roop', 'processors', 'FaceSwapInsightFace.py')
        self.assertIn("ROOP_REALSWAP_BAND_ALPHA', '0.5'", source)
        self.assertNotIn("ROOP_REALSWAP_BAND_ALPHA', '1.0'", source)

    def test_mask_feather_does_not_erode_half_the_blend_radius(self):
        source = _read('roop', 'procmgr_masking.py')
        self.assertIn('erosion_px = max(1, blend_px // 4)', source)
        self.assertNotIn('erosion_px = max(1, blend_px // 2)', source)

    def test_realswap_suppresses_secondary_band_at_extreme_yaw(self):
        source = _read('roop', 'processors', 'FaceSwapInsightFace.py')
        self.assertIn('solve_pose_jaw_5pt', source)
        self.assertIn("ROOP_REALSWAP_LATERAL_SKIP_DEG', '65'", source)
        self.assertIn('self._lateral_skips += 1', source)
        self.assertIn('if yaw is not None and abs(yaw) >= self._LATERAL_SKIP_DEG:', source)

    def test_batch_process_regular_accepts_stabilization_parameters(self):
        tree = ast.parse(_read('roop', 'core.py'))
        fn = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == 'batch_process_regular')
        arg_names = [arg.arg for arg in fn.args.args]
        for param in ('stabilize_landmarks', 'stabilize_hf_texture', 'stabilize_hf_texture_weight'):
            self.assertIn(param, arg_names, f'batch_process_regular must accept {param}')
        self.assertIsNotNone(fn.args.kwarg, 'batch_process_regular must accept **kwargs')

    def test_preview_mode_disables_session_pools(self):
        import roop.globals
        from roop import session_pool as sp
        orig = getattr(roop.globals, 'is_preview', False)
        try:
            roop.globals.is_preview = True
            self.assertFalse(sp.pooling_enabled())
            self.assertFalse(sp.detmask_pooling_enabled())
            self.assertFalse(sp.expression_pooling_enabled())
            self.assertEqual(sp.pool_size(), 1)
            self.assertEqual(sp.detmask_pool_size(), 1)
            self.assertEqual(sp.detector_pool_size(), 1)
            self.assertEqual(sp.expression_pool_size(), 1)
        finally:
            roop.globals.is_preview = orig

    def test_release_face_analyser_resets_pool(self):
        source = _read('roop', 'face_util.py')
        self.assertIn('_cleanup_fa_pool(old_pool)', source)
        self.assertIn('FACE_ANALYSER_POOL = []', source)
        try:
            from roop import face_util
            face_util.release_face_analyser()
            self.assertIsNone(face_util.FACE_ANALYSER)
            self.assertEqual(len(face_util.FACE_ANALYSER_POOL), 0)
        except ImportError:
            pass


if __name__ == '__main__':
    unittest.main()
