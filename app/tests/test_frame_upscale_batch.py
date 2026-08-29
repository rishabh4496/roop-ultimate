"""Frame_Upscale tile-batch correctness and bounded fallback tests."""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.processors.Frame_Upscale import Frame_Upscale  # noqa: E402


class _Input:
    name = "input"


class _Output:
    name = "output"


class _BatchSession:
    def __init__(self, reject_batches=False):
        self.calls = []
        self.reject_batches = reject_batches

    def run(self, _outputs, feeds):
        value = feeds["input"]
        self.calls.append(int(value.shape[0]))
        if self.reject_batches and value.shape[0] > 1:
            raise RuntimeError("static batch-one reshape")
        return [np.repeat(np.repeat(value, 2, axis=2), 2, axis=3)]


class FrameUpscaleBatchTests(unittest.TestCase):
    @staticmethod
    def _processor(session):
        proc = Frame_Upscale()
        proc.model_upscale = session
        proc.model_inputs = [_Input()]
        proc.scale = 2
        proc.prev_type = "test"
        return proc

    def test_large_gpu_batches_tiles_and_preserves_order_and_shape(self):
        frame = np.zeros((100, 140, 3), dtype=np.uint8)
        frame[:50, :70] = (10, 20, 30)
        frame[50:, 70:] = (200, 150, 100)
        batched_session = _BatchSession()
        single_session = _BatchSession()
        with patch.dict(os.environ, {
            "ROOP_VRAM_GB": "12",
            "ROOP_UPSCALE_TILE": "64",
            "ROOP_UPSCALE_TILE_BATCH": "2",
        }, clear=False):
            batched = self._processor(batched_session).RunThreadSafe(frame)
            with patch.dict(os.environ, {"ROOP_UPSCALE_TILE_BATCH": "1"},
                            clear=False):
                single = self._processor(single_session).RunThreadSafe(frame)
        self.assertEqual(batched.shape, (200, 280, 3))
        np.testing.assert_array_equal(batched, single)
        self.assertIn(2, batched_session.calls)
        self.assertTrue(all(size <= 2 for size in batched_session.calls))

    def test_small_gpu_is_conservative_and_static_batch_falls_back(self):
        frame = np.zeros((70, 70, 3), dtype=np.uint8)
        session = _BatchSession(reject_batches=True)
        proc = self._processor(session)
        with patch.dict(os.environ, {
            "ROOP_VRAM_GB": "6",
            "ROOP_UPSCALE_TILE": "64",
        }, clear=False):
            self.assertEqual(proc._tile_batch_size(), 1)
            output = proc.RunThreadSafe(frame)
        self.assertEqual(output.shape, (140, 140, 3))
        self.assertNotIn(2, session.calls)

    def test_batch_rejection_disables_only_tile_batching_for_the_run(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        session = _BatchSession(reject_batches=True)
        proc = self._processor(session)
        with patch.dict(os.environ, {
            "ROOP_VRAM_GB": "12",
            "ROOP_UPSCALE_TILE": "64",
            "ROOP_UPSCALE_TILE_BATCH": "2",
        }, clear=False):
            first = proc.RunThreadSafe(frame)
            second = proc.RunThreadSafe(frame)
        self.assertEqual(first.shape, (200, 200, 3))
        self.assertEqual(second.shape, (200, 200, 3))
        self.assertTrue(proc._tile_batch_unsupported)
        self.assertEqual(session.calls[0], 2)
        self.assertNotIn(2, session.calls[1:])

    def test_tile_size_uses_detected_vram_when_no_override_exists(self):
        proc = self._processor(_BatchSession())
        with patch.dict(os.environ, {"ROOP_VRAM_GB": "6"}, clear=False):
            os.environ.pop("ROOP_UPSCALE_TILE", None)
            os.environ.pop("ROOP_RUNTIME_UPSCALE_TILE", None)
            self.assertEqual(proc._tile_size(), 128)
        with patch.dict(os.environ, {"ROOP_VRAM_GB": "12"}, clear=False):
            os.environ.pop("ROOP_UPSCALE_TILE", None)
            os.environ.pop("ROOP_RUNTIME_UPSCALE_TILE", None)
            self.assertEqual(proc._tile_size(), 256)

    def test_explicit_runtime_tile_hint_wins_over_vram_tier(self):
        proc = self._processor(_BatchSession())
        with patch.dict(os.environ, {
            "ROOP_VRAM_GB": "6",
            "ROOP_RUNTIME_UPSCALE_TILE": "192",
        }, clear=False):
            os.environ.pop("ROOP_UPSCALE_TILE", None)
            self.assertEqual(proc._tile_size(), 192)


if __name__ == "__main__":
    unittest.main()
