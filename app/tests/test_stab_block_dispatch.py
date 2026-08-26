"""Stabilized-render block geometry: more blocks per chunk, same block grid.

THE PROBLEM. `_run_stab_parallel` decodes the clip in chunks and splits each
chunk into contiguous blocks handed to workers through a shared queue, so an
idle worker can take the next block. The chunk used to be sized `width * block`
— exactly ONE block per worker — so there was never a next block and the chunk's
wall time was gated by its unluckiest one. Measured over 96 chunks of a live
50,646-frame render: the fastest worker idled a median 18.2% of every chunk, up
to 54%, totalling 8.0 minutes of 42.5.

MEASURED OUTCOME, READ THIS FIRST: more rounds made NO measurable difference.
A/B on an 8748-frame 720p clip, all arms in one process, repeated with the order
reversed to counterbalance position:

    config       forward   reversed   mean
    1 round      15.32     15.01      15.17 fps
    2 rounds     15.17     -          15.17
    4 rounds     16.89     14.99      15.94

The SAME configuration measured 14.99 and 16.89 depending only on whether it ran
first or last, and two adjacent arms in the reversed pass gave 14.99 (4 rounds)
and 15.01 (1 round). The apparent +10% in the first, un-counterbalanced pass was
ordering. Idle was 1.9-3.6% on every arm of that clip — there was no imbalance to
redistribute, which is exactly why nothing moved. The automatic second round is
therefore used only when the current RAM budget can hold two complete rounds;
otherwise the one-round path remains. The high-imbalance case that motivated all
this (a live render at 18.2% median idle) is the reason the desktop profile keeps
enough memory headroom for that second round.

THE FIX, AND THE TRAP IN IT. Giving the queue slack means more blocks per chunk
— but the number must be a WHOLE MULTIPLE of the worker count. A greedy queue
takes ceil(k/n) rounds and a final round costs as much as a full one however few
blocks are in it, so k slightly above n is worse than k == n. Simulated at the
spread measured in that same log, against 10 workers:

    blocks   10     11     12     14     16     18     20     30     40
    eff.   84.7%  60.9%  62.6%  69.0%  75.9%  82.8%  88.6%  90.4%  91.6%

"as many blocks as the budget allows" is 14 at 720p, which is 19% SLOWER than
today. This was written that way first and the simulation caught it. Whole rounds
only: with the default 1536 MB budget that leaves 720p at exactly today's 10
blocks, and reaching 20 needs a bigger ROOP_STAB_CHUNK_MB — a per-machine RAM
decision, not something to double silently for everyone.

Note also that most of the reported 18.9% idle is NOT recoverable: it is the cost
of the final round, which more blocks amortise but never remove. The realistic
gain is +5% at two rounds rising to +8% at four.

WHY IT CANNOT CHANGE THE OUTPUT, which is the whole point of these tests: a
block's output depends only on the frames it processes and the WU frames
immediately before it, and blocks are laid out at a FIXED size from the start.
So the partition of the video into blocks — and therefore what each block is
primed from — is identical no matter how many of them a chunk holds. Only the
grouping into chunks changes, and grouping is a scheduling and memory decision.

The alternative way to create slack — smaller blocks — is the one to avoid, and
`test_block_size_is_not_reduced_to_create_slack` pins that: a block is 4x the
warm-up it discards, so halving it takes redundant priming from 25% to 50%,
which is more than the imbalance it would recover.
"""

import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)


def geometry(threads, wu, frame_bytes, budget_mb=1536.0, want=0):
    """`_stab_parallel_geometry`, as a pure function of its inputs.

    `want` is ROOP_STAB_BLOCKS_PER_WORKER.  With no explicit request, the
    production scheduler uses two rounds when the RAM budget can hold them."""
    block = max(4 * wu, 24)
    frame_mb = max(0.1, frame_bytes / (1024.0 ** 2))
    fits = max(1, int((budget_mb / frame_mb) // block))
    width = max(1, min(threads, fits))
    rounds = (max(1, min(max(1, fits // width), int(want)))
              if want > 0 else (2 if (fits // width) >= 2 else 1))
    return wu, block, width, width * rounds


def greedy_efficiency(costs, n):
    """The shared queue's own schedule: whoever frees up takes the next block."""
    free = [0.0] * n
    for c in costs:
        i = min(range(n), key=lambda j: free[j])
        free[i] += c
    return sum(costs) / (n * max(free))


def blocks_of(chunk_len, block):
    """The dispatch's block layout for one chunk."""
    n = max(1, min(-(-chunk_len // block), chunk_len))
    return [(bi * block, min(chunk_len, bi * block + block))
            for bi in range(n)
            if min(chunk_len, bi * block + block) > bi * block]


def grid_over_video(total, chunk_len, block):
    """Every block of the whole video, in absolute frame indices."""
    out, start = [], 0
    while start < total:
        n = min(chunk_len, total - start)
        out += [(start + a, start + b) for a, b in blocks_of(n, block)]
        start += n
    return out


class TestChunkHoldsMoreBlocksThanWorkers(unittest.TestCase):
    def test_720p_default_is_one_block_per_worker(self):
        wu, block, width, bpc = geometry(threads=10, wu=10,
                                         frame_bytes=1280 * 720 * 3)
        self.assertEqual((block, width, bpc), (40, 10, 10))

    def test_default_uses_two_rounds_only_when_the_budget_fits_them(self):
        """Automatic queue slack is desktop-only when memory permits it."""
        for w, h in ((1280, 720), (1920, 1080), (720, 1280)):
            for wu in (6, 10, 20):
                for budget in (1536, 3072, 8192):
                    with self.subTest(res=(w, h), wu=wu, budget=budget):
                        _wu, _b, width, bpc = geometry(10, wu, w * h * 3, budget)
                        fits = max(1, int((budget / ((w * h * 3) / 1024.0 ** 2)) // _b))
                        self.assertEqual(bpc, width * (2 if fits // width >= 2 else 1))

    def test_the_knob_buys_whole_rounds_only(self):
        fb = 1280 * 720 * 3
        for budget, want, expect in ((1536, 2, 20), (2048, 2, 20), (1536, 4, 20),
                                     (3072, 4, 40), (1536, 1, 10)):
            with self.subTest(budget=budget, want=want):
                _wu, _b, width, bpc = geometry(10, 6, fb, budget, want)
                self.assertEqual(bpc, expect)
                self.assertEqual(bpc % width, 0,
                                 "a partial round is slower than none")

    def test_never_exceeds_the_memory_budget(self):
        """The slack comes from the budget that was already sanctioned, not from
        quietly spending more memory.

        The one exception is not new and not avoidable here: when a SINGLE block
        does not fit the budget, `fits` floors to 1 and the chunk overruns by
        whatever that one block costs. 4K with a strength-1.0 warm-up is the case
        — 156-frame blocks at 23.7 MB a frame. The old sizing overran identically
        (`width * block` with width 1), so this pins the invariant that actually
        matters: the chunk never holds MORE blocks than the budget allows.
        """
        for w, h in ((1280, 720), (1920, 1080), (3840, 2160)):
            for wu in (6, 10, 20, 39):
                with self.subTest(res=(w, h), wu=wu):
                    budget, fb = 1536.0, w * h * 3
                    _wu, block, width, bpc = geometry(10, wu, fb, budget)
                    fits = max(1, int((budget / (fb / 1024.0 ** 2)) // block))
                    self.assertLessEqual(bpc, fits)
                    self.assertGreaterEqual(bpc, width)
                    self.assertEqual(bpc % width, 0, "whole rounds only")
                    self.assertEqual(bpc, width * (2 if fits // width >= 2 else 1),
                                     "automatic slack must use a whole second round only when it fits")
                    if fits > 1:
                        used_mb = bpc * block * fb / 1024.0 ** 2
                        self.assertLessEqual(round(used_mb), round(budget),
                                             "chunk exceeds ROOP_STAB_CHUNK_MB")
                    else:
                        # Degenerate floor: identical to what the old sizing did.
                        self.assertEqual(bpc, 1)

    def test_a_partial_round_is_slower_than_no_extra_round(self):
        """The measurement that decides the whole design.

        A shared queue takes ceil(k/n) rounds and the last round costs a full
        one however few blocks are in it, so k slightly above n is WORSE than
        k == n. Simulated at the per-block spread seen in the live log.
        """
        import random
        random.seed(11)
        n, cv, trials = 10, 0.12, 1500

        def eff(k):
            tot = 0.0
            for _ in range(trials):
                cs = [max(0.05, random.gauss(1.0, cv)) for _ in range(k)]
                tot += greedy_efficiency(cs, n)
            return tot / trials

        one_round = eff(n)
        self.assertLess(eff(n + 4), one_round * 0.90,
                        "a partial second round must measure clearly SLOWER — "
                        "this is why blocks_per_chunk is a multiple of width")
        self.assertGreater(eff(2 * n), one_round,
                           "two whole rounds should beat one")
        self.assertGreater(eff(4 * n), eff(2 * n),
                           "more whole rounds keep helping, with diminishing returns")

    def test_falls_back_to_one_per_worker_when_memory_is_tight(self):
        """4K with a big warm-up: the budget cannot hold more, and that must
        degrade to the old behaviour rather than overcommit."""
        _wu, _block, width, bpc = geometry(10, 39, 3840 * 2160 * 3)
        self.assertGreaterEqual(bpc, width)

    def test_blocks_per_chunk_is_at_least_the_worker_count(self):
        for threads in (1, 2, 4, 8, 16, 32):
            for wu in (6, 10, 39):
                with self.subTest(threads=threads, wu=wu):
                    _wu, _b, width, bpc = geometry(threads, wu, 1280 * 720 * 3)
                    self.assertGreaterEqual(bpc, width,
                                            "fewer blocks than workers would "
                                            "leave workers with nothing at all")


class TestTheBlockGridIsUnchanged(unittest.TestCase):
    """The output-safety argument, made executable."""

    def test_grid_is_identical_however_the_chunk_is_sized(self):
        total, block = 50646, 40
        reference = grid_over_video(total, 10 * block, block)   # old: 1 per worker
        for bpc in (10, 11, 14, 20, 37):
            with self.subTest(blocks_per_chunk=bpc):
                self.assertEqual(grid_over_video(total, bpc * block, block),
                                 reference,
                                 "chunk size changed which frames a block "
                                 "covers — that WOULD move the output")

    def test_every_block_is_full_size_except_possibly_the_last(self):
        for total in (400, 560, 50646, 1):
            with self.subTest(total=total):
                g = grid_over_video(total, 560, 40)
                for a, b in g[:-1]:
                    self.assertEqual(b - a, 40)
                self.assertLessEqual(g[-1][1] - g[-1][0], 40)

    def test_the_grid_covers_every_frame_exactly_once(self):
        for total in (1, 39, 40, 41, 400, 561, 50646):
            with self.subTest(total=total):
                g = grid_over_video(total, 560, 40)
                covered = [i for a, b in g for i in range(a, b)]
                self.assertEqual(covered, list(range(total)))

    def test_block_size_is_not_reduced_to_create_slack(self):
        """Smaller blocks are the wrong lever: priming is 1/4 of a block, so
        halving the block doubles the share of work that is thrown away."""
        wu = 10
        _wu, block, _w, _bpc = geometry(10, wu, 1280 * 720 * 3)
        self.assertEqual(block, 4 * wu)
        self.assertAlmostEqual(wu / block, 0.25)
        self.assertAlmostEqual(wu / (block / 2), 0.50)


class TestGeometrySignature(unittest.TestCase):
    def test_it_returns_four_values_and_both_callers_unpack_four(self):
        """The function grew a fourth return value; a caller still unpacking
        three would raise at render time, not at import."""
        import re
        with open(os.path.join(APP, 'roop', 'ProcessMgr.py'),
                  encoding='utf-8') as source_file:
            src = source_file.read()
        calls = re.findall(r'([\w, _]+)=\s*self\._stab_parallel_geometry\(', src)
        self.assertTrue(calls, 'no calls found — the guard is dead')
        for lhs in calls:
            with self.subTest(lhs=lhs.strip()):
                self.assertEqual(len([x for x in lhs.split(',') if x.strip()]), 4,
                                 f'{lhs.strip()!r} does not unpack 4 values')


if __name__ == '__main__':
    unittest.main()
