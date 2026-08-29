import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from roop.swap_batcher import SwapBatcher  # noqa: E402


class SwapBatcherContractTests(unittest.TestCase):
    def test_short_result_releases_every_waiter_with_an_error(self):
        batcher = SwapBatcher(
            lambda requests: ["only-one"],
            lambda: _NoopContext(),
            max_batch=2,
            max_wait_ms=0,
        )
        try:
            first = batcher.submit("a", "b", None)
            second = batcher.submit("c", "d", None)
            for request in (first, second):
                self.assertTrue(request.ev.wait(2), "request waiter was stranded")
                with self.assertRaisesRegex(RuntimeError, "returned 1 outputs for 2 requests"):
                    batcher.wait(request)
        finally:
            batcher.stop()


class _NoopContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


if __name__ == '__main__':
    unittest.main()
