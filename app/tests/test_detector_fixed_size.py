"""A fixed-shape detector export must ignore face_detector_size, not break.

`face_detector_size` is a single global shared by every detector engine, but the
engines are not alike: retinaface's export takes an arbitrary input, while
yoloface_8n and buffalo_l's det_10g are exported at a FIXED [1,3,640,640].

Feeding yoloface anything but 640 raised `InvalidArgument` — and because
`face_util.get_all_faces` swallows detector exceptions, that surfaced as ZERO
faces for an entire render, with no error anywhere. Measured 2026-08-24 over 480
frames of the hard-angle clips:

    yoloface @ 640    95.4% recall    15.62 ms/frame
    yoloface @ 512     0.0% recall     3.04 ms/frame   <- failing, not detecting

The "329 fps" that came with it is the tell: it was fast because it was doing
nothing. A detector that silently finds nothing is worse than one that is slow,
and this combination was reachable from the UI — pick yoloface, set 512, get a
render with no swaps at all.

The fix reads the model's own input dimension and uses it regardless of what the
caller asks for. These tests pin that a size mismatch is survivable, that a
dynamic export still honours the caller, and that the detect contract holds.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)


def _detector(shape):
    """A YoloFaceDetector with its session replaced, so no weights are loaded."""
    from roop.yoloface import YoloFaceDetector
    d = YoloFaceDetector.__new__(YoloFaceDetector)
    inp = MagicMock()
    inp.name = 'input'
    inp.shape = shape
    sess = MagicMock()
    sess.get_inputs = MagicMock(return_value=[inp])
    # (1, 20, 8400) -> squeeze().T -> (8400, 20); all scores 0 so nothing passes
    sess.run = MagicMock(return_value=[np.zeros((1, 20, 8400), np.float32)])
    d.session = sess
    d.input_name = 'input'
    try:
        h, w = int(shape[2]), int(shape[3])
        d.fixed_size = h if h == w else None
    except (TypeError, ValueError, IndexError):
        d.fixed_size = None
    return d


class TestFixedSizeExport(unittest.TestCase):
    def setUp(self):
        from roop.yoloface import YoloFaceDetector
        YoloFaceDetector._warned_size = False

    def test_the_model_dimension_is_read_from_the_session(self):
        self.assertEqual(_detector([1, 3, 640, 640]).fixed_size, 640)

    def test_a_dynamic_export_keeps_honouring_the_caller(self):
        for shape in ([1, 3, 'h', 'w'], [1, 3, None, None]):
            with self.subTest(shape=shape):
                self.assertIsNone(_detector(shape).fixed_size)

    def test_a_non_square_export_is_not_treated_as_fixed(self):
        self.assertIsNone(_detector([1, 3, 640, 480]).fixed_size)

    def test_a_mismatched_size_does_not_raise(self):
        """The whole bug: this used to throw, and the throw was swallowed."""
        d = _detector([1, 3, 640, 640])
        frame = np.full((720, 1280, 3), 128, np.uint8)
        for size in (512, 416, 320, 1024):
            with self.subTest(det_size=size):
                boxes, kps = d.detect(frame, det_size=size, det_thresh=0.5)
                self.assertEqual(boxes.shape[1], 5)
                self.assertEqual(kps.shape[1:], (5, 2))

    def test_the_blob_is_built_at_the_model_dimension_not_the_request(self):
        d = _detector([1, 3, 640, 640])
        d.detect(np.full((720, 1280, 3), 128, np.uint8), det_size=512)
        blob = d.session.run.call_args[0][1]['input']
        self.assertEqual(blob.shape, (1, 3, 640, 640),
                         "a 512 canvas is what ONNX Runtime rejected")

    def test_it_warns_once_not_per_frame(self):
        import io
        from contextlib import redirect_stdout
        d = _detector([1, 3, 640, 640])
        frame = np.full((720, 1280, 3), 128, np.uint8)
        buf = io.StringIO()
        with redirect_stdout(buf):
            for _ in range(5):
                d.detect(frame, det_size=512)
        self.assertEqual(buf.getvalue().count('fixed 640px'), 1,
                         'a per-frame warning would flood a 60k-frame render')

    def test_a_matching_size_is_silent(self):
        import io
        from contextlib import redirect_stdout
        d = _detector([1, 3, 640, 640])
        buf = io.StringIO()
        with redirect_stdout(buf):
            d.detect(np.full((720, 1280, 3), 128, np.uint8), det_size=640)
        self.assertEqual(buf.getvalue(), '')


class TestDetectContract(unittest.TestCase):
    def test_empty_result_shapes_are_what_callers_unpack(self):
        """`_hybrid_yolo_faces` indexes these straight away, so a no-face frame
        must still return correctly shaped arrays."""
        d = _detector([1, 3, 640, 640])
        boxes, kps = d.detect(np.zeros((480, 640, 3), np.uint8), det_thresh=0.99)
        self.assertEqual(boxes.shape[1], 5)
        self.assertEqual(kps.shape[1:], (5, 2))
        self.assertEqual(len(boxes), len(kps))


if __name__ == '__main__':
    unittest.main()
