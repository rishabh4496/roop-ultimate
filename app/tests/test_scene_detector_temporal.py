"""Unit tests for Scene-Cut Detector & Hard State Reset and single-image automatic bypass.

Verifies:
1. Fast, lightweight shot-boundary detector (HSV histogram / chi-square downscaled frame difference).
2. Hard scene cut triggering full reset of all temporal buffers (Kalman tracks, One-Euro filters,
   EMA masks, restorer flow history, and appearance stats).
3. Single-image swaps automatically bypassing all temporal stabilization and buffers.
4. Comprehensive verification that `flush_pipeline_temporal_buffers` handles all active temporal components cleanly.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

import roop.globals
from roop.scene_detector import (
    ShotBoundaryDetector,
    FastShotBoundaryDetector,
    flush_pipeline_temporal_buffers,
    ContentAwareSceneDetector,
)
from roop.ProcessMgr import ProcessMgr


class TestShotBoundaryDetector(unittest.TestCase):
    """Test fast lightweight shot-boundary detector."""

    def setUp(self):
        self.detector = ShotBoundaryDetector(
            downscale_size=(64, 36),
            spatial_cut_thresh=58.0,
            spatial_min_thresh=23.0,
            chisqr_thresh=0.65,
        )

    def test_alias_equivalence(self):
        """FastShotBoundaryDetector must be an alias of ShotBoundaryDetector."""
        self.assertIs(FastShotBoundaryDetector, ShotBoundaryDetector)

    def test_first_frame_initializes_baseline(self):
        """First frame should record baseline and return False (not a cut)."""
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        is_cut = self.detector.observe(frame)
        self.assertFalse(is_cut)
        self.assertIsNotNone(self.detector.previous)
        self.assertEqual(len(self.detector.scene_cuts), 0)

    def test_static_and_sub_threshold_frames_are_not_cuts(self):
        """Identical or slightly perturbed frames must not be flagged as cuts."""
        f1 = np.full((100, 100, 3), 100, dtype=np.uint8)
        f2 = np.full((100, 100, 3), 103, dtype=np.uint8)  # slight variation
        self.assertFalse(self.detector.observe(f1))
        self.assertFalse(self.detector.observe(f2))

    def test_drastic_scene_cut_detected(self):
        """A sudden high spatial/color shift must be flagged as a cut."""
        f_dark = np.zeros((100, 100, 3), dtype=np.uint8)
        f_bright = np.full((100, 100, 3), 255, dtype=np.uint8)

        self.assertFalse(self.detector.observe(f_dark))
        is_cut = self.detector.observe(f_bright)
        self.assertTrue(is_cut)

    def test_observe_frame_records_frame_index(self):
        """observe_frame must track frame_idx in scene_cuts on cut detection."""
        f_red = np.zeros((100, 100, 3), dtype=np.uint8)
        f_red[:, :, 2] = 255  # BGR -> Red
        f_green = np.zeros((100, 100, 3), dtype=np.uint8)
        f_green[:, :, 1] = 255  # BGR -> Green

        self.detector.observe_frame(f_red, 0)
        self.assertEqual(len(self.detector.scene_cuts), 0)

        is_cut = self.detector.observe_frame(f_green, 15)
        self.assertTrue(is_cut)
        self.assertIn(15, self.detector.scene_cuts)

    def test_reset_clears_state(self):
        """reset() must clear previous frame baseline and scene_cuts."""
        f1 = np.zeros((100, 100, 3), dtype=np.uint8)
        f2 = np.full((100, 100, 3), 255, dtype=np.uint8)
        self.detector.observe_frame(f1, 0)
        self.detector.observe_frame(f2, 1)
        self.assertEqual(len(self.detector.scene_cuts), 1)

        self.detector.reset()
        self.assertIsNone(self.detector.previous)
        self.assertEqual(len(self.detector.scene_cuts), 0)

    def test_invalid_and_empty_frames(self):
        """Detector should safely reject None or empty frames without raising."""
        self.assertFalse(self.detector.observe(None))
        self.assertFalse(self.detector.observe(np.array([], dtype=np.uint8)))


class TestFlushPipelineTemporalBuffers(unittest.TestCase):
    """Test comprehensive temporal buffer flushing across all pipeline components."""

    def test_flush_without_mgr(self):
        """When mgr is None, global face_swapper smoother and patch cache are still cleared."""
        with patch('roop.processors.frame.face_swapper.clear_temporal_state') as mock_fs, \
             patch('roop.temporal_smart_skip.clear_track_patches') as mock_ss:
            res = flush_pipeline_temporal_buffers(mgr=None)
            self.assertTrue(res['face_swapper_temporal'])
            self.assertTrue(res['smart_skip_patches'])
            mock_fs.assert_called_once()
            mock_ss.assert_called_once()

    def test_flush_all_active_temporal_components_on_mgr(self):
        """Verify all active temporal components on ProcessMgr are cleanly reset."""
        mgr = MagicMock()
        mgr._landmark_smoother = MagicMock()
        mgr._hf_stabilizer = MagicMock()
        mgr._temporal_enhancer = MagicMock()
        mgr._target_appearance = MagicMock()
        mgr._interaction_masker = MagicMock()
        mgr._temporal_color = MagicMock()
        mgr._temporal_identity = MagicMock()
        mgr._temporal_occlusion = MagicMock()
        mgr._temporal_expression = MagicMock()
        mgr._temporal_compositing = MagicMock()
        mgr._temporal_quality = MagicMock()
        mgr._video_temporal = MagicMock()
        mgr._reset_dispatch_tracker = MagicMock()

        mgr.kps_stabilizer = MagicMock()
        mgr.enh_stabilizer = MagicMock()
        mgr.mask_stabilizer = MagicMock()

        mgr._cur_kps_stab = MagicMock(return_value=MagicMock())
        mgr._cur_enh_stab = MagicMock(return_value=MagicMock())
        mgr._cur_mask_stab = MagicMock(return_value=MagicMock())

        tls_mock = MagicMock()
        tls_mock.video_temporal_state = MagicMock()
        tls_mock.kps = MagicMock()
        tls_mock.enh = MagicMock()
        tls_mock.mask = MagicMock()
        tls_mock.last_found_bboxes = np.array([[10, 10, 50, 50]])
        tls_mock.frame_faces = [MagicMock()]
        mgr._tls = tls_mock
        mgr.last_swapped_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mgr.face_masks = {'0': np.zeros((100, 100))}

        res = flush_pipeline_temporal_buffers(mgr)

        # Verify all return statuses
        self.assertTrue(res['face_swapper_temporal'])
        self.assertTrue(res['smart_skip_patches'])
        self.assertTrue(res['landmark_smoother'])
        self.assertTrue(res['hf_stabilizer'])
        self.assertTrue(res['temporal_enhancer'])
        self.assertTrue(res['target_appearance'])
        self.assertTrue(res['occlusion_mask'])
        self.assertTrue(res['temporal_color'])
        self.assertTrue(res['temporal_identity'])
        self.assertTrue(res['temporal_occlusion'])
        self.assertTrue(res['temporal_expression'])
        self.assertTrue(res['temporal_compositing'])
        self.assertTrue(res['temporal_quality'])
        self.assertTrue(res['video_temporal'])
        self.assertTrue(res['dispatch_tracker'])
        self.assertTrue(res['stabilizers'])

        # Verify calls
        mgr._landmark_smoother.reset.assert_called_once()
        mgr._hf_stabilizer.reset.assert_called_once()
        mgr._temporal_enhancer.flush.assert_called_once()
        mgr._target_appearance.reset.assert_called_once()
        mgr._interaction_masker.reset.assert_called_once()
        mgr._temporal_color.reset.assert_called_once()
        mgr._temporal_identity.reset.assert_called_once()
        mgr._temporal_occlusion.reset.assert_called_once()
        mgr._temporal_expression.reset.assert_called_once()
        mgr._temporal_compositing.reset.assert_called_once()
        mgr._temporal_quality.reset.assert_called_once()
        mgr._video_temporal.reset.assert_called_once()
        mgr._reset_dispatch_tracker.assert_called_once()

        # Verify stabilizers reset
        mgr.kps_stabilizer.reset.assert_called()
        mgr.enh_stabilizer.reset.assert_called()
        mgr.mask_stabilizer.reset.assert_called()

        # Verify frame references cleared
        self.assertIsNone(mgr.last_swapped_frame)
        self.assertEqual(len(mgr.face_masks), 0)
        self.assertIsNone(mgr._tls.video_temporal_state)
        self.assertIsNone(mgr._tls.last_found_bboxes)
        self.assertIsNone(mgr._tls.frame_faces)

    def test_flush_resilience_to_exceptions(self):
        """Any component raising an error during reset must be swallowed cleanly."""
        mgr = MagicMock()
        mgr._landmark_smoother = MagicMock()
        mgr._landmark_smoother.reset.side_effect = RuntimeError("Broken smoother")
        mgr._hf_stabilizer = MagicMock()
        mgr._hf_stabilizer.reset.side_effect = TypeError("Invalid state")

        res = flush_pipeline_temporal_buffers(mgr)
        self.assertIsInstance(res, dict)
        self.assertFalse(res['landmark_smoother'])
        self.assertFalse(res['hf_stabilizer'])


class TestSingleImageAutomaticBypass(unittest.TestCase):
    """Test that single-image swaps automatically bypass all temporal stabilization and buffers."""

    def setUp(self):
        roop.globals.processing = True
        self.mgr = ProcessMgr(progress=MagicMock())
        self.mgr.is_preview = True

    def tearDown(self):
        roop.globals.processing = False

    def test_is_single_image_detection(self):
        """is_single_image returns True when frame_idx is None or target_path is an image file."""
        self.assertTrue(self.mgr.is_single_image(frame_idx=None))

        # With video mode flags False and image extension
        with patch('roop.globals.target_path', 'portrait.jpg'):
            self.assertTrue(self.mgr.is_single_image(frame_idx=0))

        with patch('roop.globals.target_path', 'video.mp4'):
            self.assertFalse(self.mgr.is_single_image(frame_idx=0))

    def test_bypass_temporal_guards_return_none(self):
        """When bypass_temporal is set on TLS, temporal stabilizers and engines must return None."""
        self.mgr._tls.bypass_temporal = True

        self.assertIsNone(self.mgr._cur_kps_stab())
        self.assertIsNone(self.mgr._cur_enh_stab())
        self.assertIsNone(self.mgr._cur_mask_stab())

        # All temporal engines return None
        for name in ('interaction_masker', 'temporal_identity', 'temporal_occlusion',
                     'temporal_expression', 'target_appearance', 'temporal_compositing',
                     'temporal_quality', 'landmark_smoother', 'hf_stabilizer', 'temporal_enhancer'):
            self.assertIsNone(self.mgr._temporal_engine(name))

        # Video temporal state returns None
        self.assertIsNone(self.mgr._video_temporal_state())

        # Dispatch tracker returns (None, False)
        tracker, coast = self.mgr._dispatch_tracker()
        self.assertIsNone(tracker)
        self.assertFalse(coast)

        # Cleanup
        self.mgr._tls.bypass_temporal = False

    def test_process_frame_single_image_automatic_bypass_and_restoration(self):
        """Single-image process_frame must activate bypass, flush buffers, and restore state."""
        frame = np.full((128, 128, 3), 128, dtype=np.uint8)

        # Provide a dummy input face so process_frame does not return early
        self.mgr.input_face_datas = [MagicMock()]
        self.mgr.swap_faces = MagicMock(return_value=(1, frame))

        with patch('roop.scene_detector.flush_pipeline_temporal_buffers') as mock_flush:
            res = self.mgr.process_frame(frame, frame_idx=None)
            self.assertIsNotNone(res)

            # flush_pipeline_temporal_buffers must be called on single image entry and exit
            self.assertGreaterEqual(mock_flush.call_count, 2)

            # swap_faces must be called with stabilize=False for single image
            self.mgr.swap_faces.assert_called_once()
            args, kwargs = self.mgr.swap_faces.call_args
            self.assertFalse(kwargs.get('stabilize'))

            # bypass_temporal flag on TLS must be restored to False
            self.assertFalse(getattr(self.mgr._tls, 'bypass_temporal', False))


class TestHardSceneCutResetIntegration(unittest.TestCase):
    """Test that a detected hard cut triggers immediate temporal buffer flushing."""

    def test_shot_boundary_triggers_flush(self):
        """When ShotBoundaryDetector detects a cut, flush_pipeline_temporal_buffers is triggered."""
        sbd = ShotBoundaryDetector()
        f_dark = np.zeros((64, 64, 3), dtype=np.uint8)
        f_bright = np.full((64, 64, 3), 255, dtype=np.uint8)

        mgr = MagicMock()
        mgr._landmark_smoother = MagicMock()
        mgr._hf_stabilizer = MagicMock()

        # Frame 0: dark
        cut0 = sbd.observe_frame(f_dark, 0)
        self.assertFalse(cut0)

        # Frame 1: bright (hard cut)
        cut1 = sbd.observe_frame(f_bright, 1)
        self.assertTrue(cut1)

        # Trigger flush
        flushed = flush_pipeline_temporal_buffers(mgr)
        self.assertTrue(flushed['face_swapper_temporal'])
        self.assertTrue(flushed['smart_skip_patches'])
        mgr._landmark_smoother.reset.assert_called_once()
        mgr._hf_stabilizer.reset.assert_called_once()


if __name__ == '__main__':
    unittest.main()
