"""`swap_moved_the_face` (the post-swap verification) detects through whichever engine is active.

With `detector_engine == 'retinaface_r50_gpu'` (the face_engine on-device RetinaFace) the
verification's re-detection must run on THAT detector, on the padded ROI crop, with the detector
alone -- not on buffalo_l's SCRFD (`fa.det_model`) and without recognition / landmark models. It
already does: `swap_moved_the_face` -> `detect_boxes_in_roi` -> `_detect_faces_raw(crop, aux=False)`,
which dispatches on the engine. Measured on d4 (600 frames, r50_gpu): every one of the 451 main-pass
detector executions -- the 237 (+51 warm-up) from `detect_boxes_in_roi` among them -- is counted under
`raw.engine.retinaface_r50_gpu` with aux=False.

That was established by reading counters from a render, so nothing in the suite kept it true; a
refactor of the dispatch in `_detect_faces_raw` would have silently put verification back on the
wrong detector (it would still "work", slower and on a second network). This pins it, with the
scrfd path as the control, and pins that the rotation the pipeline applied still reaches the
detector, which is what keeps a rolled face from being un-swapped.
"""
import contextlib
import os
import sys
from unittest import mock

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

import roop.globals                                                   # noqa: E402
from roop import face_util as fu                                      # noqa: E402

FRAME = np.random.RandomState(0).randint(0, 255, (480, 640, 3), dtype=np.uint8)
BBOX = np.array([250.0, 150.0, 390.0, 310.0], np.float32)
KPS = np.array([[285, 205], [355, 205], [320, 245], [292, 280], [350, 280]], np.float32)


def _roi_offset(rotation_action=None):
    win = fu._roi_window(FRAME, BBOX, 0.6, 160)
    assert win is not None
    crop, cx1, cy1 = win
    return crop, cx1, cy1


def _detections(kps_shift=0.0, offset=(0, 0)):
    """What a detector returns for the plate's face, in the coordinates of the crop it was given.

    `kps_shift` displaces only the keypoints, not the box: that is "the same face, features moved",
    which is what verification looks for. (A box that moved away reads as the NEIGHBOUR and is
    ignored by design.)"""
    ox, oy = offset
    box = np.array([[BBOX[0] - ox, BBOX[1] - oy, BBOX[2] - ox, BBOX[3] - oy, 0.99]], np.float32)
    kps = (KPS - np.array([ox, oy], np.float32) + np.array([kps_shift, 0.0], np.float32))[None]
    return box, kps


class _FakeAnalyser:
    def __init__(self):
        self.models = {'recognition': mock.MagicMock(), 'landmark_2d_106': mock.MagicMock()}
        self.det_model = mock.MagicMock()
        self.det_model.input_size = (512, 512)
        self.det_model.det_thresh = 0.5


@pytest.fixture()
def rig(monkeypatch):
    fa = _FakeAnalyser()

    @contextlib.contextmanager
    def lease():
        yield fa
    monkeypatch.setattr(fu, 'lease_face_analyser', lease)
    saved = getattr(roop.globals, 'detector_engine', 'scrfd')
    yield fa
    roop.globals.detector_engine = saved


def _verify(rotation_action=None):
    return fu.swap_moved_the_face(FRAME, KPS, BBOX, rotation_action=rotation_action)


class TestVerifyRoutesThroughTheActiveEngine:

    def test_gpu_engine_active_detects_on_the_gpu_detector_alone(self, rig):
        roop.globals.detector_engine = 'retinaface_r50_gpu'
        crop, cx1, cy1 = _roi_offset()
        rig.det_model.detect.side_effect = AssertionError('verification used the FaceAnalysis detector')
        with mock.patch('roop.retinaface_gpu_engine.detect', return_value=_detections(offset=(cx1, cy1))) as gpu:
            moved = _verify()
        assert moved is False
        assert gpu.call_count == 1
        given = gpu.call_args.args[0]
        assert given.shape == crop.shape and given.shape[:2] != FRAME.shape[:2], 'not the padded ROI'
        assert np.array_equal(given, crop)
        rig.det_model.detect.assert_not_called()
        for model in rig.models.values():
            model.get.assert_not_called()              # detector alone: no recognition / landmarks

    def test_the_gpu_detector_s_answer_decides(self, rig):
        roop.globals.detector_engine = 'retinaface_r50_gpu'
        _, cx1, cy1 = _roi_offset()
        far = _detections(kps_shift=90.0, offset=(cx1, cy1))
        # The shape gate (SWAP_SHAPE_TOL) is a separate, separately tested rule; a rigid shift of
        # the whole constellation is "moved" only when it is off.
        with mock.patch.object(fu, 'SWAP_SHAPE_TOL', 0.0), \
                mock.patch('roop.retinaface_gpu_engine.detect', return_value=far):
            assert _verify() is True                   # the swap moved the face: undone
        with mock.patch('roop.retinaface_gpu_engine.detect', return_value=_detections(offset=(cx1, cy1))):
            assert _verify() is False

    def test_a_miss_on_the_gpu_detector_keeps_the_swap(self, rig):
        roop.globals.detector_engine = 'retinaface_r50_gpu'
        empty = (np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32))
        with mock.patch('roop.retinaface_gpu_engine.detect', return_value=empty):
            assert _verify() is False

    def test_the_applied_rotation_reaches_the_gpu_detector(self, rig):
        roop.globals.detector_engine = 'retinaface_r50_gpu'
        crop, cx1, cy1 = _roi_offset()
        with mock.patch('roop.retinaface_gpu_engine.detect',
                        return_value=(np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32))) as gpu:
            _verify(rotation_action='rotate_180')
        assert np.array_equal(gpu.call_args.args[0], np.ascontiguousarray(crop[::-1, ::-1])), \
            'the ROI was not turned upright before detecting'

    def test_scrfd_active_is_the_control_and_never_calls_the_gpu_engine(self, rig):
        roop.globals.detector_engine = 'scrfd'
        _, cx1, cy1 = _roi_offset()
        box, kps = _detections(offset=(cx1, cy1))
        rig.det_model.detect.return_value = (box, kps)
        with mock.patch('roop.retinaface_gpu_engine.detect',
                        side_effect=AssertionError('the GPU engine ran while scrfd was selected')):
            assert _verify() is False
        assert rig.det_model.detect.call_count == 1
        for model in rig.models.values():
            model.get.assert_not_called()
