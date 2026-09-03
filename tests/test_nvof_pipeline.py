"""Self-verification of the three-tier temporal motion-vector pipeline.

Real NVOF and CUDA are optional build features, so the tests use finite,
deterministic stand-ins for their output vectors.  This verifies the public
contract on every developer machine while production capability checks select
the actual NVOF / CUDA / CPU implementation.
"""

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / 'app'
for path in (str(REPO_ROOT), str(APP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from roop.processors.frame import face_swapper


class NvofPipelineTest(unittest.TestCase):
    """Three simulated frames must produce valid vectors in each flow tier."""

    FRAME_SIZE = 96
    FLOW_SIZE = face_swapper.TemporalMaskSmoother.FLOW_SIZE

    def setUp(self):
        base = np.zeros((self.FRAME_SIZE, self.FRAME_SIZE, 3), dtype=np.uint8)
        cv2.circle(base, (36, 48), 18, (80, 160, 220), -1)
        self.frames = [base]
        for shift in (3, 6):
            transform = np.float32([[1, 0, shift], [0, 1, 0]])
            self.frames.append(cv2.warpAffine(base, transform,
                                              (self.FRAME_SIZE, self.FRAME_SIZE)))
        self.masks = []
        for frame in self.frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.masks.append((gray > 0).astype(np.float32))

    def _backward_motion(self, current, previous):
        """Return the known -3px/frame source coordinate displacement."""
        flow = np.zeros((current.shape[0], current.shape[1], 2), dtype=np.float32)
        flow[..., 0] = -3.0 * current.shape[1] / self.FRAME_SIZE
        return flow

    def test_motion_vectors_are_finite_for_three_frames_in_every_tier(self):
        tiers = {
            'nvof': '_flow_nvof',
            'cuda_farneback': '_flow_cuda_farneback',
            'dis': '_flow_dis_engine',
        }
        for tier, method in tiers.items():
            with self.subTest(tier=tier):
                smoother = face_swapper.TemporalMaskSmoother(
                    alpha=0.85, flow_tier=tier)
                if tier == 'dis':
                    fake_engine = type('FakeDIS', (), {
                        'calc': lambda _, current, previous, output:
                        self._backward_motion(current, previous),
                    })()
                    patched = patch.object(smoother, method,
                                           return_value=fake_engine)
                else:
                    patched = patch.object(smoother, method,
                                           side_effect=self._backward_motion)

                with patched:
                    outputs = [smoother.smooth(self.masks[0], self.frames[0], 7)]
                    vectors_1 = smoother.calculate_motion_vectors(
                        self.frames[1], self.frames[0])
                    outputs.append(smoother.smooth(self.masks[1], self.frames[1], 7))
                    vectors_2 = smoother.calculate_motion_vectors(
                        self.frames[2], self.frames[1])
                    outputs.append(smoother.smooth(self.masks[2], self.frames[2], 7))

                for vectors in (vectors_1, vectors_2):
                    self.assertEqual(vectors.shape, (self.FLOW_SIZE, self.FLOW_SIZE, 2))
                    self.assertEqual(vectors.dtype, np.float32)
                    self.assertFalse(np.isnan(vectors).any())
                for output in outputs:
                    self.assertEqual(output.shape, (self.FRAME_SIZE, self.FRAME_SIZE))
                    self.assertFalse(np.isnan(output).any())
                seam = smoother._states[7]['seam']
                self.assertEqual(seam.shape, (self.FRAME_SIZE, self.FRAME_SIZE))
                self.assertFalse(np.isnan(seam).any())
                self.assertEqual(smoother.last_flow_tier, tier)

    def test_auto_tier_prefers_nvof_then_cuda_then_dis(self):
        smoother = face_swapper.TemporalMaskSmoother(flow_tier='auto')
        flow = self._backward_motion(self.frames[1], self.frames[0])
        with patch.object(smoother, '_flow_nvof', return_value=flow), \
             patch.object(smoother, '_flow_cuda_farneback', return_value=None):
            smoother.calculate_motion_vectors(self.frames[1], self.frames[0])
        self.assertEqual(smoother.last_flow_tier, 'nvof')

        with patch.object(smoother, '_flow_nvof', return_value=None), \
             patch.object(smoother, '_flow_cuda_farneback', return_value=flow):
            smoother.calculate_motion_vectors(self.frames[1], self.frames[0])
        self.assertEqual(smoother.last_flow_tier, 'cuda_farneback')

        fake_engine = type('FakeDIS', (), {
            'calc': lambda _, current, previous, output:
            self._backward_motion(current, previous),
        })()
        with patch.object(smoother, '_flow_nvof', return_value=None), \
             patch.object(smoother, '_flow_cuda_farneback', return_value=None), \
             patch.object(smoother, '_flow_dis_engine', return_value=fake_engine):
            smoother.calculate_motion_vectors(self.frames[1], self.frames[0])
        self.assertEqual(smoother.last_flow_tier, 'dis')


if __name__ == '__main__':
    unittest.main()
