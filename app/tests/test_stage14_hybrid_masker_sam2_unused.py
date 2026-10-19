"""Stage 14 asked for a SAM 2 sliding-window memory bank in `face_engine.pipeline.hybrid_masker.HybridVideoMasker`.

`step()` never calls SAM 2. The "SAM 2 face hull mask" is a landmark ellipse (`generate_hull_mask`, `cv2.ellipse`), the
`_sam2_predictor` that `initialize()` builds is not touched again, and the "16-frame FIFO window" is a deque of frame INDICES, not of
memory embeddings. These tests pin that, so no one prunes a memory bank that does not exist (or believes the masker tracks with
SAM 2). They fail the day `step()` starts driving the predictor, which is the moment this stage's work becomes real.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

torch = __import__("torch")
from face_engine.pipeline.hybrid_masker import HybridVideoMasker  # noqa: E402


def _masker(predictor):
    m = HybridVideoMasker(device="cpu")
    m.device = "cpu"
    m._initialized = True
    m._sam2_predictor = predictor
    m.extract_occluder_mask = lambda crop: torch.zeros(1, 1, 256, 256)
    return m


KPS = np.array([[96, 100], [160, 100], [128, 140], [104, 180], [152, 180]], np.float32)


class HybridMaskerNeverRunsSam2(unittest.TestCase):
    def test_step_makes_no_call_on_the_sam2_predictor(self):
        predictor = MagicMock()
        m = _masker(predictor)
        for i in range(40):
            m.step(i, np.zeros((256, 256, 3), np.float32), KPS)
        self.assertEqual(predictor.method_calls, [])
        self.assertEqual(predictor.call_count, 0)

    def test_the_mask_is_the_landmark_ellipse_whatever_the_predictor_would_say(self):
        a = _masker(MagicMock()).step(0, np.zeros((256, 256, 3), np.float32), KPS)
        b = _masker(None).step(0, np.zeros((256, 256, 3), np.float32), KPS)
        self.assertTrue(torch.equal(a, b))
        self.assertGreater(float(a.sum()), 0.0)

    def test_the_fifo_window_holds_frame_indices_only(self):
        m = _masker(MagicMock())
        for i in range(100):
            m.step(i, np.zeros((256, 256, 3), np.float32), KPS)
        self.assertEqual(len(m._fifo_history), m.max_fifo_window)
        self.assertTrue(all(isinstance(v, int) for v in m._fifo_history))


if __name__ == "__main__":
    unittest.main()
