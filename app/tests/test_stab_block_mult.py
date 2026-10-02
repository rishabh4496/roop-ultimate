"""ROOP_STAB_BLOCK_MULT: how many times its warm-up a stabilization block is.

A block discards `warm_up` frames of work, so its redundant priming is 1/mult of
its work (25% at the default 4x). A bigger multiple needs more RAM per chunk; the
geometry's own shrink steps back off when too few blocks fit, so a larger value
cannot take effect on a machine that cannot hold it. The default must stay 4.
"""
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.procmgr_stabilization import StabilizationSchedulingMixin  # noqa: E402


def geometry(threads=12, wu=6, frame_bytes=480 * 854 * 3, budget_mb=1839.0, small=False, env=None):
    stub = types.SimpleNamespace(
        _stab_warmup=wu, _stab_warmup_frames=lambda: wu,
        _stab_min_block_multiple=1, _runtime_stab_small=small,
        _stab_frame_bytes=frame_bytes,
        _default_stab_chunk_mb=lambda hard_cap=None: budget_mb)
    with mock.patch.dict(os.environ, {k: v for k, v in (env or {}).items()}, clear=False):
        if not env or 'ROOP_STAB_BLOCK_MULT' not in env:
            os.environ.pop('ROOP_STAB_BLOCK_MULT', None)
        return StabilizationSchedulingMixin._stab_parallel_geometry(stub, threads)


class BlockMultTest(unittest.TestCase):

    def test_small_frames_with_ram_automatically_get_the_8x_block(self):
        """480x854, 1839 MB budget: two whole rounds of 48-frame blocks fit."""
        wu, block, width, bpc = geometry()
        self.assertEqual((wu, block, width, bpc), (6, 48, 12, 24))

    def test_720p_and_1080p_keep_the_4x_block(self):
        """A bigger block would shrink the width/blocks there and cost more than it saves."""
        for w, h in ((1280, 720), (1920, 1080), (3840, 2160)):
            with self.subTest(res=(w, h)):
                _wu, block, _width, _bpc = geometry(frame_bytes=w * h * 3)
                # Never LARGER than today's 4x block there (4K is shrunk further by the
                # pre-existing adaptive steps, which this change does not touch).
                self.assertLessEqual(block, 24)
                if (w, h) in ((1280, 720), (1920, 1080)):
                    self.assertEqual(block, 24)

    def test_the_auto_block_only_applies_when_two_whole_rounds_fit(self):
        # 480x854 with a quarter of the budget: 8x no longer fits two rounds -> 4x.
        self.assertEqual(geometry(budget_mb=460.0)[1], 24)
        self.assertEqual(geometry(budget_mb=1839.0)[1], 48)

    def test_explicit_4_keeps_the_old_behaviour_everywhere(self):
        self.assertEqual(geometry(env={'ROOP_STAB_BLOCK_MULT': '4'})[1], 24)

    def test_without_ram_for_two_big_rounds_the_block_is_never_larger_than_4x(self):
        """The automatic 8x block must not appear where it does not fit (1080p etc.);
        whatever the pre-existing adaptive shrink does there is unchanged."""
        for wu in (4, 6, 10):
            with self.subTest(wu=wu):
                self.assertLessEqual(geometry(wu=wu, frame_bytes=1920 * 1080 * 3)[1],
                                     max(4 * wu, 24))

    def test_a_larger_multiple_gives_a_larger_block(self):
        _wu, block, width, bpc = geometry(env={'ROOP_STAB_BLOCK_MULT': '8'})
        self.assertEqual(block, 48)
        self.assertEqual(width, 12)

    def test_it_cannot_overcommit_ram(self):
        """1080p frames, small budget: the shrink steps back off the big block."""
        _wu, block, width, bpc = geometry(frame_bytes=1920 * 1080 * 3, budget_mb=400.0,
                                          env={'ROOP_STAB_BLOCK_MULT': '16'})
        frame_mb = 1920 * 1080 * 3 / 1048576.0
        self.assertLessEqual(block * width * frame_mb, 400.0 * 1.0001 + block * frame_mb)
        self.assertLess(block, 16 * 6)             # it did shrink

    def test_blocks_per_chunk_stay_a_whole_multiple_of_width(self):
        for mult in ('2', '4', '8', '12'):
            with self.subTest(mult=mult):
                _wu, _b, width, bpc = geometry(env={'ROOP_STAB_BLOCK_MULT': mult})
                self.assertEqual(bpc % width, 0)

    def test_junk_and_out_of_range_values_are_safe(self):
        # Junk is treated as "not set": the same automatic block as no variable at all.
        self.assertEqual(geometry(env={'ROOP_STAB_BLOCK_MULT': 'junk'})[1], geometry()[1])
        self.assertEqual(geometry(env={'ROOP_STAB_BLOCK_MULT': '1'})[1], 24)     # clamped to 2 -> max(12, 24)
        self.assertEqual(geometry(env={'ROOP_STAB_BLOCK_MULT': '999'})[1],
                         geometry(env={'ROOP_STAB_BLOCK_MULT': '16'})[1])

    def test_the_small_card_profile_ignores_it(self):
        self.assertEqual(geometry(small=True, env={'ROOP_STAB_BLOCK_MULT': '16'})[1],
                         geometry(small=True)[1])


if __name__ == '__main__':
    unittest.main()
