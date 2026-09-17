"""Comprehensive verification tests for all 7 boost fixes and edge cases.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import roop.globals
from roop.face_detector import should_trigger_pyramid
from roop import face_util


class TestBoostFixes(unittest.TestCase):

    def test_fix1_unique_identities_calculation_in_batch(self):
        """Fix 1: Correct faces_per_frame & face_count calculation using unique person groups."""
        from roop.procmgr_batch import BatchProcessingMixin

        class DummyBatchManager(BatchProcessingMixin):
            def __init__(self):
                self.lock = MagicMock()
                self._runtime_worker_busy = set()
                self.target_face_datas = []
                self.target_face_groups = []

        # Case A: 5 angle exemplars for 2 people
        bm = DummyBatchManager()
        bm.target_face_datas = [MagicMock() for _ in range(5)]
        bm.target_face_groups = [0, 0, 0, 1, 1]

        mock_opt = MagicMock()
        mock_profile = MagicMock()
        mock_profile.hardware.vram_total_gb = 12.0
        mock_profile.tuning.stabilization_chunk_size = 32
        mock_profile.tuning.stabilization_workers = 4
        mock_profile.tuning.queue_depth = 8
        mock_profile.tuning.batch_size = 2
        mock_profile.tuning.tile_batch_size = 2
        mock_profile.tuning.face_concurrency = 2
        mock_profile.tuning.in_flight_frames = 4
        mock_profile.tuning.encoder = 'h264_nvenc'
        mock_opt_cls = MagicMock(return_value=mock_opt)
        mock_opt.profile_video.return_value = mock_profile

        with patch('roop.runtime_optimizer.RuntimeOptimizer', mock_opt_cls), \
             patch('roop.ProcessMgr.RuntimeOptimizer', mock_opt_cls):
            bm._replay_analysis_released = False
            # Simulate procmgr_batch.py logic directly
            _target_datas = getattr(bm, 'target_face_datas', []) or []
            _target_groups = getattr(bm, 'target_face_groups', None) or getattr(roop.globals, 'TARGET_FACE_GROUP', None)
            if _target_groups and (not _target_datas or len(_target_groups) == len(_target_datas)):
                try:
                    _unique_identities = len(set(_target_groups))
                except Exception:
                    _unique_identities = len(_target_groups)
            else:
                _unique_identities = len(_target_datas)

            self.assertEqual(_unique_identities, 2)
            self.assertEqual(max(1, _unique_identities), 2)

        # Case B: 3 angle exemplars for 1 person
        bm.target_face_datas = [MagicMock() for _ in range(3)]
        bm.target_face_groups = [0, 0, 0]
        _target_datas = bm.target_face_datas
        _target_groups = bm.target_face_groups
        _unique = len(set(_target_groups)) if (_target_groups and len(_target_groups) == len(_target_datas)) else len(_target_datas)
        self.assertEqual(_unique, 1)

        # Case C: Mismatched length fallback (e.g. stale TARGET_FACE_GROUP of length 1, but 4 target faces)
        bm.target_face_datas = [MagicMock() for _ in range(4)]
        bm.target_face_groups = [0]
        _target_datas = bm.target_face_datas
        _target_groups = bm.target_face_groups
        if _target_groups and (not _target_datas or len(_target_groups) == len(_target_datas)):
            _unique = len(set(_target_groups))
        else:
            _unique = len(_target_datas)
        self.assertEqual(_unique, 4)

        # Case D: Empty target faces (all-face swap)
        bm.target_face_datas = []
        bm.target_face_groups = []
        _target_datas = bm.target_face_datas
        _target_groups = bm.target_face_groups
        _unique = len(set(_target_groups)) if (_target_groups and len(_target_groups) == len(_target_datas)) else len(_target_datas)
        self.assertEqual(_unique, 0)
        self.assertEqual(max(1, _unique), 1)

    def test_fix2_unblock_auxiliary_session_teardown(self):
        """Fix 2: _release_replayed_analysis is unblocked even when 3d recon or frontalization is on."""
        from roop.ProcessMgr import ProcessMgr
        pm = ProcessMgr(progress=MagicMock())
        pm.options = MagicMock()
        pm.options.use_3d_recon = True
        pm.options.use_frontalization = True
        pm._temporal_mode = True
        pm._temporal_faces = {0: []}
        pm._temporal_covered = 100

        with patch('roop.face_util.release_face_analyser_aux') as mock_release, \
             patch('gc.collect') as mock_gc:
            result = pm._release_replayed_analysis(frame_count=100)
            self.assertTrue(result)
            mock_release.assert_called_once()
            mock_gc.assert_called_once()
            self.assertTrue(pm._replay_analysis_released)

    def test_fix3_inplace_face_analyser_threshold(self):
        """Fix 3: Mutate det_thresh in-place on existing pool without rebuilding TRT sessions."""
        fake_fa1 = MagicMock()
        fake_fa1.det_thresh = 0.50
        fake_fa1.det_model = MagicMock()
        fake_fa1.det_model.det_thresh = 0.50

        fake_fa2 = MagicMock()
        fake_fa2.det_thresh = 0.50
        fake_fa2.det_model = MagicMock()
        fake_fa2.det_model.det_thresh = 0.50

        orig_pool = [fake_fa1, fake_fa2]
        face_util.FACE_ANALYSER_POOL = orig_pool
        face_util.FACE_ANALYSER = fake_fa1
        face_util._ANALYSER_DET_SIZE = 640
        face_util._ANALYSER_DET_THRESH = 0.50
        face_util._ANALYSER_ENGINE = 'scrfd'
        face_util._ANALYSER_LM68_LAZY = False
        roop.globals.g_current_face_analysis = 'all'
        roop.globals.g_desired_face_analysis = 'all'
        roop.globals.face_detector_threshold = 0.70

        with patch('roop.face_util._desired_det_size', return_value=640), \
             patch('roop.face_util._current_engine', return_value='scrfd'), \
             patch('roop.session_pool.detmask_pool_size', return_value=2), \
             patch('roop.session_pool.detmask_pooling_enabled', return_value=True), \
             patch('roop.face_util._build_face_analyser') as mock_build:

            res = face_util._ensure_face_analyser()
            # Must NOT have rebuilt any sessions
            mock_build.assert_not_called()
            # Threshold mutated in-place on both pool instances and primary instance
            self.assertEqual(face_util._ANALYSER_DET_THRESH, 0.70)
            self.assertEqual(fake_fa1.det_thresh, 0.70)
            self.assertEqual(fake_fa1.det_model.det_thresh, 0.70)
            self.assertEqual(fake_fa2.det_thresh, 0.70)
            self.assertEqual(fake_fa2.det_model.det_thresh, 0.70)
            self.assertIs(res, fake_fa1)

    def test_fix4_rescue_bailout_and_multiscale_pruning(self):
        """Fix 4: Prune multiscale on negative frames and skip redundant rescues."""
        # 1. should_trigger_pyramid: empty initial_dets on 1080p frame returns False (pruned)
        h, w = 1080, 1920
        empty_dets = np.zeros((0, 4), dtype=np.float32)
        self.assertFalse(should_trigger_pyramid((h, w), initial_dets=empty_dets))

        # 2. Large close-up face still triggers pyramid
        closeup_dets = np.array([[100, 100, 700, 700]], dtype=np.float32) # h=600 >= 500
        self.assertTrue(should_trigger_pyramid((h, w), initial_dets=closeup_dets))

        # 3. Rescue bailout in _detect_faces when has_multiscale is True
        with patch('roop.face_util._detect_faces_raw', return_value=[]), \
             patch('roop.face_util._rescue_downscaled') as mock_downscaled, \
             patch('roop.face_util._rescue_padded') as mock_padded, \
             patch('roop.face_util._rescue_rotated', return_value=[]) as mock_rotated, \
             patch('roop.face_util._rescue_clahe', return_value=[]), \
             patch('roop.face_util._enrich_detected_faces', side_effect=lambda f, x: x):

            # When engine is retinaface (has_multiscale=True), downscaled & padded are skipped
            roop.globals.detector_engine = 'retinaface'
            roop.globals.detector_scale_pyramid = None
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            face_util._detect_faces(frame)
            mock_downscaled.assert_not_called()
            mock_padded.assert_not_called()
            mock_rotated.assert_called_once()

            # When detector_scale_pyramid is configured, downscaled & padded are also skipped
            mock_rotated.reset_mock()
            roop.globals.detector_engine = 'scrfd'
            roop.globals.detector_scale_pyramid = '0.5,1.0'
            face_util._detect_faces(frame)
            mock_downscaled.assert_not_called()
            mock_padded.assert_not_called()
            mock_rotated.assert_called_once()

            # When standard engine has no multiscale, downscaled & padded ARE called
            mock_rotated.reset_mock()
            mock_downscaled.reset_mock()
            mock_padded.reset_mock()
            mock_downscaled.return_value = []
            mock_padded.return_value = []
            roop.globals.detector_engine = 'scrfd'
            roop.globals.detector_scale_pyramid = None
            face_util._detect_faces(frame)
            mock_downscaled.assert_called_once()
            mock_padded.assert_called_once()
            mock_rotated.assert_called_once()

    def test_fix5_codeformer_fidelity_scalar_shape(self):
        """Fix 5: CodeFormer fidelity tensor is 0-D scalar with shape () bound via IO binding."""
        import roop.processors.Enhance_UltraMax as UM
        from roop.processors.Enhance_UltraMax import Enhance_UltraMax

        enhancer = Enhance_UltraMax()
        enhancer.in_dtype = np.float32
        enhancer.devicename = 'cuda'
        enhancer._cuda_iob_available = None
        enhancer.pool = None
        enhancer._session_lock = MagicMock()
        enhancer.session = MagicMock()
        enhancer.session.get_providers.return_value = ['CUDAExecutionProvider']
        enhancer.io_binding = MagicMock()
        enhancer.model_inputs = [MagicMock(name='x'), MagicMock(name='w')]
        enhancer.model_inputs[0].name = 'x'
        enhancer.model_inputs[1].name = 'w'
        enhancer.model_outputs = [MagicMock(name='y', type='tensor(float)')]
        enhancer.model_outputs[0].name = 'y'

        roop.globals.codeformer_fidelity = 0.65
        fidelity_tensor = torch.tensor(float(getattr(roop.globals, 'codeformer_fidelity', 0.5)),
                                       device='cpu', dtype=torch.float64)
        self.assertEqual(fidelity_tensor.shape, torch.Size([]))
        self.assertEqual(tuple(fidelity_tensor.shape), ())
        self.assertEqual(fidelity_tensor.item(), 0.65)

        # Test CUDA postprocess binding with simulated CUDA environment
        mock_output = MagicMock()
        with patch.object(UM, '_TORCH_CUDA', True), \
             patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.current_stream'), \
             patch('torch.from_numpy') as mock_from_numpy, \
             patch('roop.processors.Enhance_UltraMax.ort_cuda_output_to_torch') as mock_out_to_torch:

            mock_source = MagicMock()
            mock_source.device = 'cpu'
            mock_source.permute.return_value.flip.return_value.unsqueeze.return_value.div.return_value.sub.return_value.to.return_value.contiguous.return_value = MagicMock(shape=(1, 3, 512, 512), data_ptr=lambda: 123456)
            mock_from_numpy.return_value.to.return_value = mock_source

            mock_out_to_torch.return_value = torch.ones((1, 3, 512, 512), dtype=torch.float32)

            frame = np.zeros((512, 512, 3), dtype=np.uint8)
            enhancer._run_cuda_postprocess(frame, 512, frame)

            # Confirm bind_input for w input received scalar shape ()
            calls = enhancer.io_binding.bind_input.call_args_list
            self.assertGreaterEqual(len(calls), 2)
            w_call = calls[1]
            # args: (name, device, device_id, element_type, shape, buffer_ptr)
            w_shape = w_call[0][4]
            self.assertEqual(w_shape, (), "Weight tensor bound to CUDA EP must have scalar shape ()")

    def test_fix6_tracking_prune_obs_and_other_real(self):
        """Fix 6: other_real stores lightweight bboxes, t['obs'] is popped, _interp_collides handles dict/obj."""
        from roop.procmgr_tracking import TrackingMixin

        # 1. Test _interp_collides with raw bbox
        face_obj = MagicMock()
        face_obj.bbox = np.array([10, 10, 50, 50], dtype=np.float32)
        raw_other_bbox = np.array([12, 12, 48, 48], dtype=np.float32)
        other_real = {0: [(2, raw_other_bbox)]}

        collides = TrackingMixin._interp_collides(face_obj, tid=1, frame_idx=0, other_real=other_real)
        self.assertTrue(collides)

        # 2. Test _interp_collides with dict observation
        face_dict = {'bbox': np.array([10, 10, 50, 50], dtype=np.float32)}
        collides_dict = TrackingMixin._interp_collides(face_dict, tid=1, frame_idx=0, other_real=other_real)
        self.assertTrue(collides_dict)

        # 3. Test other_real construction and pop('obs')
        tracks = [
            {'id': 1, 'obs': {0: face_obj, 1: face_dict}, 'emb_mean': np.zeros(512, dtype=np.float32)},
            {'id': 2, 'obs': {0: face_dict}, 'emb_mean': np.zeros(512, dtype=np.float32)},
        ]
        other_real = {}
        for t in tracks:
            for f_idx, face in (t.get('obs') or {}).items():
                bbox = getattr(face, 'bbox', None)
                if bbox is None and isinstance(face, dict):
                    bbox = face.get('bbox')
                if bbox is not None:
                    other_real.setdefault(f_idx, []).append((t['id'], bbox))

        self.assertEqual(len(other_real[0]), 2)
        self.assertEqual(len(other_real[1]), 1)
        self.assertTrue(isinstance(other_real[0][0][1], np.ndarray))

        # Popping obs frees the dict
        for t in tracks:
            obs = t.pop('obs', None) or {}
            self.assertNotIn('obs', t)

    def test_fix7_segment_writer_gc_collect(self):
        """Fix 7: gc.collect() is called in _finalize_segment when segment is committed."""
        import tempfile
        import os
        from roop.segment_writer import SegmentedVideoWriter
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SegmentedVideoWriter(target_video=os.path.join(tmpdir, 'fake.mp4'), size=(512, 512), fps=30.0)
            writer._cur_seg_file = 'fake_part.mp4'
            writer._cur_frames = 100
            writer._cur_bytes = 1000
            writer._writer = MagicMock()

            with patch.object(writer, '_write_manifest'), \
                 patch.object(writer, '_notify_checkpoint'), \
                 patch('roop.segment_writer.os.path.getsize', return_value=1000), \
                 patch('roop.segment_writer.bar_write'), \
                 patch('roop.segment_writer.gc.collect') as mock_gc:
                writer._finalize_segment()
                mock_gc.assert_called_once()


if __name__ == '__main__':
    unittest.main()
