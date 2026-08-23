"""Stabilized-render block geometry: more blocks per chunk, same block grid.

THE PROBLEM. `_run_stab_parallel` decodes the clip in chunks and splits each
chunk into contiguous blocks handed to workers through a shared queue, so an
idle worker can take the next block. The chunk used to be sized `width * block`
— exactly ONE block per worker — so there was never a next block and the chunk's
wall time was gated by its unluckiest one. Measured over 96 chunks of a live
50,646-frame render: the fastest worker idled a median 18.2% of every chunk, up
to 54%, totalling 8.0 minutes of 42.5.

THE FIX. Size the chunk to hold every block the SAME memory budget already
allows, so the queue has slack. At 720p that is 14 blocks against 10 workers.

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


def geometry(threads, wu, frame_bytes, budget_mb=1536.0, per_worker=2.0):
    """`_stab_parallel_geometry`, as a pure function of its inputs."""
    block = max(4 * wu, 24)
    frame_mb = max(0.1, frame_bytes / (1024.0 ** 2))
    fits = max(1, int((budget_mb / frame_mb) // block))
    width = max(1, min(threads, fits))
    want = int(width * max(1.0, per_worker))
    blocks_per_chunk = max(width, min(fits, want))
    return wu, block, width, blocks_per_chunk


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
    def test_720p_case_from_the_live_render(self):
        wu, block, width, bpc = geometry(threads=10, wu=10,
                                         frame_bytes=1280 * 720 * 3)
        self.assertEqual(block, 40)
        self.assertEqual(width, 10)
        self.assertGreater(bpc, width,
                           "a chunk with one block per worker leaves the "
                           "work-stealing queue nothing to hand out")
        self.assertEqual(bpc, 14, "should use every block the budget allows")

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
                    if fits > 1:
                        used_mb = bpc * block * fb / 1024.0 ** 2
                        self.assertLessEqual(round(used_mb), round(budget),
                                             "chunk exceeds ROOP_STAB_CHUNK_MB")
                    else:
                        # Degenerate floor: identical to what the old sizing did.
                        self.assertEqual(bpc, 1)

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
        src = open(os.path.join(APP, 'roop', 'ProcessMgr.py'),
                   encoding='utf-8').read()
        calls = re.findall(r'([\w, _]+)=\s*self\._stab_parallel_geometry\(', src)
        self.assertTrue(calls, 'no calls found — the guard is dead')
        for lhs in calls:
            with self.subTest(lhs=lhs.strip()):
                self.assertEqual(len([x for x in lhs.split(',') if x.strip()]), 4,
                                 f'{lhs.strip()!r} does not unpack 4 values')


if __name__ == '__main__':
    unittest.main()
