"""Regression guards for correctness fixes found during the five-stage audit.

These source-level checks deliberately avoid loading the model stack. They pin
small integration seams whose failure would otherwise show up only in a long
video render or in UI guidance.
"""

import ast
import os
import threading
import unittest

import numpy as np


APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*parts):
    with open(os.path.join(APP, *parts), encoding='utf-8') as fh:
        return fh.read()


class AuditRegressionTests(unittest.TestCase):
    def test_in_memory_video_frame_count_is_end_exclusive(self):
        # Batch orchestration was extracted from ProcessMgr. Keep this guard on
        # the owner of run_batch_inmem so a refactor does not turn a valid source
        # check into a false failure.
        tree = ast.parse(_read('roop', 'procmgr_batch.py'))
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

    def test_sample_runner_uses_requested_person_b_person_d_facesets(self):
        source = _read('tests', 'run_all_samples.py')
        self.assertIn('load_library_faceset("person_d")', source)
        self.assertIn('load_library_faceset("person_b")', source)
        self.assertNotIn('load_library_faceset("person_a")', source)
        self.assertNotIn('load_library_faceset("person_f")', source)

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

    def test_roi_retry_uses_the_configured_detector_entrypoint(self):
        source = _read('roop', 'ProcessMgr.py')
        start = source.index('def _detect_face_in_roi')
        end = source.index('\n\ndef ', start)
        body = source[start:end]
        self.assertIn('faces = get_all_faces(crop) or []', body)
        self.assertNotIn('fa.get(crop)', body)

    def test_hybrid_detector_rejects_or_clips_invalid_roi_geometry(self):
        try:
            from roop import face_util
        except ImportError:
            self.skipTest('runtime dependencies are not installed')

        class _AuxModel:
            def get(self, frame, face):
                face.embedding = np.ones(4, dtype=np.float32)

        class _Analyser:
            models = {'recognition': _AuxModel()}

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        bboxes = np.array([
            [-20.0, -30.0, 1305.0, 760.0, 0.95],
            [500.0, 500.0, 499.0, 510.0, 0.90],
            [0.0, 0.0, np.nan, 10.0, 0.85],
        ], dtype=np.float32)
        kpss = np.array([
            [[-10.0, -10.0], [1300.0, 20.0], [640.0, 360.0],
             [100.0, 730.0], [1200.0, 730.0]],
            [[500.0, 500.0], [500.0, 500.0], [500.0, 500.0],
             [500.0, 500.0], [500.0, 500.0]],
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
        ], dtype=np.float32)

        faces = face_util._hybrid_detector_faces(
            frame, _Analyser(), bboxes, kpss, aux=True)

        self.assertEqual(len(faces), 1)
        self.assertTrue(np.isfinite(faces[0].bbox).all())
        self.assertTrue((faces[0].bbox >= 0).all())
        self.assertLessEqual(float(faces[0].bbox[2]), frame.shape[1])
        self.assertLessEqual(float(faces[0].bbox[3]), frame.shape[0])
        self.assertTrue((faces[0].kps[:, 0] < frame.shape[1]).all())
        self.assertTrue((faces[0].kps[:, 1] < frame.shape[0]).all())

    def test_clahe_state_is_not_shared_between_detection_workers(self):
        try:
            from roop import face_util
        except ImportError:
            self.skipTest('runtime dependencies are not installed')

        owner = face_util._clahe_for_current_worker()
        other = []
        worker = threading.Thread(
            target=lambda: other.append(face_util._clahe_for_current_worker()))
        worker.start()
        worker.join()

        self.assertEqual(len(other), 1)
        self.assertIsNot(owner, other[0])

    def test_last_output_cleared_on_job_start(self):
        source = _read('api.py')
        self.assertIn('_last_output.update({"path": "", "kind": ""})', source)

    def test_duration_s_exposed_in_progress_and_run_stats(self):
        source = _read('api.py')
        self.assertIn('"duration_s": dur_s', source)
        self.assertIn('"duration_s": 0.0', source)
        self.assertIn('_run_stats["duration_s"] = round(max(0.0, time.time() - _run_stats["start"]), 1)', source)


if __name__ == '__main__':
    unittest.main()
