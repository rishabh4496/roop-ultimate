"""Cross-frame swap batching vs the swap model's own face mask.

These two features are mutually exclusive by construction. The batcher
coalesces crops from several worker threads into one inference, so a mask
coming back out cannot be attributed to the frame that asked for it — the
cross-frame path therefore publishes no mask at all.

That trade was invisible while cross-frame batching was opt-in. It stopped
being invisible when perf_batch_swap='auto' (the shipped default) began
exporting ROOP_BATCH_SWAP_XFRAME=1, because the collision now resolves against
the user by default: raise "swap model's own face mask" off 0 and nothing
happens, with nothing on screen to say why. The slider's own help text calls it
"the first thing to raise if a hififace or hyperswap paste reaches up into
hair", so the setting most likely to be reached for is the one that silently
does nothing.

Batching is the optional half — it trades speed for a mask the user asked for
by name — so batching is what has to yield.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import roop.globals                                    # noqa: E402
from roop import ProcessMgr as _pm                     # noqa: E402
from roop.ProcessMgr import ProcessMgr                 # noqa: E402


class _Swapper:
    type = 'swap'
    pool = None

    def __init__(self, has_mask):
        self.model_has_mask = has_mask

    def RunBatchMulti(self, requests):
        return []


class _Stub:
    """Just enough `self` for the real _make_swap_batcher to run."""

    def __init__(self, has_mask):
        self.processors = [_Swapper(has_mask)]


class BatcherYieldsToTheMask(unittest.TestCase):

    def setUp(self):
        self._env = dict(os.environ)
        self._batch = _pm._BATCH_SWAP
        self._strength = getattr(roop.globals, 'swap_model_mask_strength', 0.0)
        # The shipped default state: both halves enabled.
        os.environ['ROOP_BATCH_SWAP_XFRAME'] = '1'
        _pm._BATCH_SWAP = True

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        _pm._BATCH_SWAP = self._batch
        roop.globals.swap_model_mask_strength = self._strength

    def _make(self, has_mask, strength, threads=4):
        roop.globals.swap_model_mask_strength = strength
        return ProcessMgr._make_swap_batcher(_Stub(has_mask), threads)

    def test_batcher_is_built_when_the_mask_is_off(self):
        """The perf default must survive — this is the common case."""
        self.assertIsNotNone(self._make(has_mask=True, strength=0.0))

    def test_batcher_yields_when_the_mask_is_requested(self):
        self.assertIsNone(
            self._make(has_mask=True, strength=25.0),
            'cross-frame batching cannot attribute the mask, so asking for the '
            'mask has to turn batching off — not the other way round',
        )

    def test_a_maskless_swapper_still_batches(self):
        """inswapper et al emit no mask, so the setting cannot cost them speed."""
        self.assertIsNotNone(self._make(has_mask=False, strength=25.0))

    def test_a_string_strength_does_not_crash_the_run(self):
        """Settings arrive off a JSON payload; this must not raise mid-render."""
        self.assertIsNone(self._make(has_mask=True, strength='25'))

    def test_a_none_strength_reads_as_off(self):
        self.assertIsNotNone(self._make(has_mask=True, strength=None))

    def test_single_thread_never_batches(self):
        self.assertIsNone(self._make(has_mask=False, strength=0.0, threads=1))

    def test_xframe_off_never_batches(self):
        os.environ['ROOP_BATCH_SWAP_XFRAME'] = '0'
        self.assertIsNone(self._make(has_mask=False, strength=0.0))


if __name__ == '__main__':
    unittest.main()
