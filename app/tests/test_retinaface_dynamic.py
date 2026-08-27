"""Regression tests for the dynamic-shape RetinaFace r50 export."""

import os
import sys
import unittest
from unittest.mock import MagicMock

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)


class TestRetinaFaceDynamicShape(unittest.TestCase):
    def test_dynamic_spatial_shape_uses_640_profile(self):
        from roop.retinaface import RetinaFace3Output

        session = MagicMock()
        inp = MagicMock()
        inp.shape = ['b', 3, 'h', 'w']
        inp.name = 'input'
        session.get_inputs.return_value = [inp]
        session.get_outputs.return_value = []

        detector = RetinaFace3Output(session=session)
        self.assertEqual(detector.input_size, (640, 640))

    def test_none_spatial_shape_uses_640_profile(self):
        from roop.retinaface import RetinaFace3Output

        session = MagicMock()
        inp = MagicMock()
        inp.shape = [1, 3, None, None]
        inp.name = 'input'
        session.get_inputs.return_value = [inp]
        session.get_outputs.return_value = []

        detector = RetinaFace3Output(session=session)
        self.assertEqual(detector.input_size, (640, 640))

    def test_r50_face_confidence_is_column_one(self):
        from roop.retinaface import RetinaFace3Output

        session = MagicMock()
        inp = MagicMock()
        inp.shape = ['b', 3, 'h', 'w']
        inp.name = 'input'
        session.get_inputs.return_value = [inp]
        session.get_outputs.return_value = [MagicMock(name='loc'),
                                            MagicMock(name='conf'),
                                            MagicMock(name='landms')]
        loc = np.zeros((1, 16800, 4), np.float32)
        conf = np.zeros((1, 16800, 2), np.float32)
        conf[:, :, 0] = 0.01
        conf[:, :, 1] = 0.99
        landms = np.zeros((1, 16800, 10), np.float32)
        session.run.return_value = [loc, conf, landms]

        detector = RetinaFace3Output(session=session)
        boxes, kps = detector.detect(np.zeros((720, 1280, 3), np.uint8))
        self.assertGreater(len(boxes), 0)
        self.assertEqual(kps.shape[1:], (5, 2))


if __name__ == '__main__':
    unittest.main()
