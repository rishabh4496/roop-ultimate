"""A swallowed detector exception must not be silent.

`face_util.get_all_faces` catches every exception and returns `[]`. That
return value is correct and must not change -- callers all over the pipeline
treat "no faces in this frame" as an ordinary outcome. What was wrong is that
the failure was INVISIBLE, and it has cost this project two separate
investigations:

* `yoloface_8n.onnx` is a fixed `[1,3,640,640]` export, so every other
  `det_size` raised `InvalidArgument` on every frame. The run produced a video
  with no swaps, returned 0, and reported nothing. The only tell was that it
  was *fast* -- 329 fps, because it was doing nothing.
* A bench that called this helper without initialising roop graded 0 of 600
  frames on valid footage and reported `insufficient_detections` -- blaming the
  footage for a detector that had never been started.

These contracts pin both halves: the empty-list contract, and the visibility.
"""

import io
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import roop.face_util as fu


class _Capture:
    def __enter__(self):
        self._old = sys.stdout
        self.buf = io.StringIO()
        sys.stdout = self.buf
        return self

    def __exit__(self, *a):
        sys.stdout = self._old

    @property
    def text(self):
        return self.buf.getvalue()


class DetectorFailureVisibilityTest(unittest.TestCase):

    def setUp(self):
        self._real = fu._detect_faces
        fu.reset_detector_failures()

    def tearDown(self):
        fu._detect_faces = self._real
        fu.reset_detector_failures()

    def _raise(self, exc):
        def boom(_frame):
            raise exc
        fu._detect_faces = boom

    def test_return_contract_is_unchanged(self):
        """Still an empty list. Callers depend on this; do not "improve" it."""
        self._raise(ValueError("boom"))
        with _Capture():
            self.assertEqual(fu.get_all_faces(None), [])

    def test_failure_is_announced(self):
        self._raise(ValueError("input must be [1,3,640,640]"))
        with _Capture() as cap:
            fu.get_all_faces(None)
        self.assertIn("DETECTOR FAILED", cap.text)
        self.assertIn("640", cap.text)

    def test_warning_is_bounded_to_one_line_per_signature(self):
        """A per-frame warning on a 60k-frame render is its own outage."""
        self._raise(ValueError("same every time"))
        with _Capture() as cap:
            for _ in range(500):
                fu.get_all_faces(None)
        self.assertEqual(cap.text.count("DETECTOR FAILED"), 1)

    def test_a_different_failure_is_still_reported(self):
        """Bounding must not hide a second, different fault."""
        self._raise(ValueError("first"))
        with _Capture() as cap1:
            fu.get_all_faces(None)
        self._raise(RuntimeError("second, different"))
        with _Capture() as cap2:
            fu.get_all_faces(None)
        self.assertIn("DETECTOR FAILED", cap1.text)
        self.assertIn("DETECTOR FAILED", cap2.text)
        self.assertEqual(len(fu.detector_failure_signatures()), 2)

    def test_success_path_is_untouched(self):
        class _F:
            def __init__(self, x):
                self.bbox = [x, 0, x + 10, 10]
        fu._detect_faces = lambda _f: [_F(30), _F(10), _F(20)]
        with _Capture() as cap:
            faces = fu.get_all_faces(None)
        self.assertEqual([f.bbox[0] for f in faces], [10, 20, 30])
        self.assertNotIn("DETECTOR FAILED", cap.text)
        self.assertEqual(fu.detector_failure_signatures(), [])

    def test_empty_detection_is_not_reported_as_a_failure(self):
        """No faces in frame is normal; only an EXCEPTION is a failure."""
        fu._detect_faces = lambda _f: []
        with _Capture() as cap:
            self.assertEqual(fu.get_all_faces(None), [])
        self.assertNotIn("DETECTOR FAILED", cap.text)
        self.assertEqual(fu.detector_failure_signatures(), [])


if __name__ == "__main__":
    unittest.main()
