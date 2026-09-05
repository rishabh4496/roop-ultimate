"""The stabilizer's RAM budget must count the buffers the code actually holds.

`_run_stab_parallel` keeps several chunk-sized frame buffers alive at once, and
`_default_stab_chunk_mb` divides the RAM share by that count so the total stays
inside what the machine has free. The divisor was the constant 6, which was
right when the queues were literally `Queue(2)` and `Queue(1)`.

Both are now sized from the unified scheduler's `queue_capacity`:

    prefetch_q = Queue(max(1, int(_scheduler_capacity)))
    _write_q   = Queue(max(1, int(_scheduler_capacity)))

and nothing updated the divisor. At the capacity of 3 a 12GB/32GB box profiles,
the real count is 3 + 2*3 = NINE, so the budget reserved for six while the
render held half as much again -- on a machine already at its RAM ceiling, that
difference IS the ceiling. Host RAM is the binding constraint on both target
machines (a 16GB box died at 12% of a 40934-frame render with ffmpeg's threads
failing malloc), so an under-count here is not a rounding error.

These pin the arithmetic to the queues rather than to a number.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop.ProcessMgr import ProcessMgr                            # noqa: E402


class _Scheduler:
    def __init__(self, queue_capacity):
        self.queue_capacity = queue_capacity


class _Bare:
    """A ProcessMgr stand-in: these two helpers touch nothing else."""
    _STAB_LIVE_CHUNKS = ProcessMgr._STAB_LIVE_CHUNKS
    _stab_live_chunks = ProcessMgr._stab_live_chunks
    _default_stab_chunk_mb = ProcessMgr._default_stab_chunk_mb

    def __init__(self, queue_capacity):
        self._runtime_scheduler = (_Scheduler(queue_capacity)
                                   if queue_capacity is not None else None)


class LiveChunkCountFollowsTheQueues(unittest.TestCase):

    def test_it_counts_both_queues_plus_the_three_working_buffers(self):
        # reader + in-flight chunk + results = 3, then prefetch_q and _write_q.
        for capacity, expected in ((1, 5), (2, 7), (3, 9), (4, 11)):
            self.assertEqual(_Bare(capacity)._stab_live_chunks(), expected,
                             f'capacity {capacity}')

    def test_the_historical_constant_is_what_the_old_queues_implied(self):
        """Queue(2) + Queue(1) + 3 == 6. Keeping the fallback honest matters:
        it is what runs when no scheduler is present."""
        self.assertEqual(_Bare(None)._stab_live_chunks(), 6)
        self.assertEqual(ProcessMgr._STAB_LIVE_CHUNKS, 6)

    def test_a_junk_capacity_falls_back_rather_than_dividing_by_nonsense(self):
        for bad in (None, 'three', object()):
            self.assertEqual(_Bare(bad)._stab_live_chunks(), 6, repr(bad))

    def test_zero_or_negative_capacity_never_under_counts(self):
        """Queue(max(1, ...)) floors at one slot, so the accounting must too."""
        for bad in (0, -5):
            self.assertEqual(_Bare(bad)._stab_live_chunks(), 5, repr(bad))

    def test_a_deeper_queue_shrinks_the_per_chunk_budget(self):
        """The whole point: more live buffers means each must be smaller, or
        the total exceeds the RAM the share was supposed to cap."""
        shallow = _Bare(2)._default_stab_chunk_mb()
        deep = _Bare(4)._default_stab_chunk_mb()
        self.assertLess(deep, shallow)

    def test_the_reserved_total_is_the_same_whatever_the_queue_depth(self):
        """budget x live is the quantity the share actually bounds."""
        totals = [_Bare(c)._default_stab_chunk_mb() * _Bare(c)._stab_live_chunks()
                  for c in (2, 3, 4)]
        # Allow for the hard cap and the 96 MB floor clamping an extreme.
        for other in totals[1:]:
            self.assertAlmostEqual(other, totals[0], delta=totals[0] * 0.02)


if __name__ == '__main__':
    unittest.main()
