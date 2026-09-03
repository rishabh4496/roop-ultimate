"""Regression tests for PinnedBufferPool.acquire's non-blocking default.

A buffer only returns to the pool through `release()`.  `roop/nvdec_reader.py`
is the pool's only caller and never releases -- decoded frames escape to
ProcessMgr and stay there -- so the pool is permanently empty after its first
`capacity` acquires.

With the old `timeout=0.5` default, every acquire past that point waited the
full half second for a refill that could not arrive and then allocated anyway.
That is 2.0 fps, and it measured 2.0 fps in isolation and 1.96 fps end to end
against ~900 fps for the same ffmpeg pipe.  Nothing failed: the frames were
correct, just slow, so the return code, the swap audit and the whole suite
stayed green while the render was decode-starved.

These tests pin the property that broke: an exhausted pool must not stall.
"""

import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.buffer_pool import PinnedBufferPool  # noqa: E402


SHAPE = (16, 16, 3)


class TestPinnedBufferPoolAcquire(unittest.TestCase):

    def test_exhausted_pool_does_not_stall(self):
        """The defect: N acquires past capacity cost N * timeout seconds."""
        pool = PinnedBufferPool(SHAPE, capacity=2)
        for _ in range(pool.capacity):          # drain the pre-allocated buffers
            pool.acquire()

        started = time.time()
        for _ in range(8):
            pool.acquire()
        elapsed = time.time() - started

        # The old default would need ~4.0s here (8 * 0.5).  Allow generous
        # headroom for allocation on a loaded machine and still fail loudly.
        self.assertLess(
            elapsed, 0.5,
            'acquire() on an exhausted pool stalled: %.2fs for 8 acquires. '
            'A pool nothing releases into must not wait for a refill.' % elapsed)

    def test_exhausted_pool_still_returns_a_usable_buffer(self):
        """Falling back to allocation is correct -- only the waiting was wrong."""
        pool = PinnedBufferPool(SHAPE, capacity=1)
        pool.acquire()

        buf = pool.acquire()
        self.assertIsInstance(buf, np.ndarray)
        self.assertEqual(buf.shape, SHAPE)
        self.assertEqual(buf.dtype, np.uint8)
        buf[:] = 7                              # must be writable
        self.assertTrue((buf == 7).all())

    def test_pooled_buffers_are_reused_when_released(self):
        """A caller that does honour the protocol still gets its buffer back."""
        pool = PinnedBufferPool(SHAPE, capacity=1)
        first = pool.acquire()
        pool.release(first)
        self.assertIs(pool.acquire(), first)

    def test_explicit_timeout_is_still_honoured(self):
        """The wait remains available where a release protocol exists."""
        pool = PinnedBufferPool(SHAPE, capacity=1)
        pool.acquire()

        started = time.time()
        pool.acquire(timeout=0.2)
        self.assertGreaterEqual(time.time() - started, 0.15)


if __name__ == '__main__':
    unittest.main()
