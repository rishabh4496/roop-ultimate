"""The sequential fallback inside RunBatch / RunBatchMulti.

When a batched swap inference fails (a static-batch ONNX export, or a
TensorRT engine built for batch 1), the swapper re-runs the crops one at a
time instead of failing the frame. The pixels were always covered by that
fallback; the MASKS were not, and that is the interesting half.

`_stash_masks` writes into a single-slot thread-local, and `Run` calls it once
per crop. So a fallback loop that just calls Run in sequence leaves exactly one
mask behind — the last crop's — while the caller (ProcessMgr, around the
`take_masks()` after RunBatch) does ONE take_masks() and pairs the result to
the crops BY POSITION. One mask for N crops then either misattributes every
face's mask or fails reassembly outright, and it does so only on the path that
exists to rescue a broken batch: the failure is invisible until the day the
fallback fires.

So the contract under test is: after a fallback, take_masks() returns one mask
per crop, in crop order — the same shape the batched path publishes — and a
PARTIAL set collapses to None rather than to a short list that would silently
shift every mask onto the wrong face.
"""
import os
import sys
import threading
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from roop.processors.FaceSwapInsightFace import FaceSwapInsightFace  # noqa: E402


class _Swapper(FaceSwapInsightFace):
    """The real RunBatch/RunBatchMulti/Run/_stash_masks over a stub session.

    Subclassed rather than constructed so the test drives the shipping methods
    without loading a 500MB ONNX model or a GPU.
    """

    def __init__(self, batch_works, emits_mask, size=8):
        self.pool = None
        self._mask_tls = threading.local()
        self.image_input_name = 'target'
        self.embed_input_name = 'source'
        self.loaded_model_key = 'stub'
        self._batch_unsupported = False
        self._batch_works = batch_works
        self._emits_mask = emits_mask
        self._size = size
        self.single_calls = 0
        self.batch_attempts = 0

    def _compute_source_input(self, source_face):
        return np.zeros((1, 512), dtype=np.float32)

    def _infer(self, feed):
        n = feed[self.image_input_name].shape[0]
        if n > 1:
            self.batch_attempts += 1
            if not self._batch_works:
                raise RuntimeError('static batch: expected 1, got %d' % n)
        else:
            self.single_calls += 1
        img = np.full((n, 3, self._size, self._size), float(n), dtype=np.float32)
        if not self._emits_mask:
            return [img]
        # One distinct mask per crop, so a misattribution is visible.
        mask = np.stack([np.full((1, self._size, self._size), i + 1, dtype=np.float32)
                         for i in range(n)])
        return [img, mask]


def _crops(n, size=8):
    return [np.full((1, 3, size, size), i, dtype=np.float32) for i in range(n)]


class _AlwaysFailsSession:
    def run(self, _out, _feed):
        raise RuntimeError('trt broken')


class _WorksSession:
    def __init__(self, image_input_name):
        self.image_input_name = image_input_name

    def run(self, _out, feed):
        n = feed[self.image_input_name].shape[0]
        return [np.zeros((n, 3, 8, 8), dtype=np.float32)]


class _TrtGateSwapper(FaceSwapInsightFace):
    """Exercises the REAL _infer(), unlike _Swapper above (which stubs _infer
    entirely and so never touches _rebuild_without_trt/_trt_disabled at all).
    `_rebuild_without_trt` itself is stubbed here — its own internals are
    untouched by this fix and are not what's under test — only whether
    _infer() is willing to CALL it, gated by batch size."""

    def __init__(self):
        self.pool = None
        self.image_input_name = 'target'
        self.embed_input_name = 'source'
        self._trt_disabled = False
        self.rebuild_calls = 0
        self.model_swap_insightface = _AlwaysFailsSession()

    def _rebuild_without_trt(self):
        self.rebuild_calls += 1
        self._trt_disabled = True
        self.model_swap_insightface = _WorksSession(self.image_input_name)
        return True


class FallbackKeepsTheMaskContract(unittest.TestCase):

    def test_batched_path_publishes_one_mask_per_crop(self):
        """The baseline the fallback has to match."""
        sw = _Swapper(batch_works=True, emits_mask=True)
        outs = sw.RunBatch(object(), object(), _crops(4))
        self.assertEqual(len(outs), 4)
        masks = sw.take_masks()
        self.assertEqual(len(masks), 4)

    def test_fallback_publishes_one_mask_per_crop(self):
        sw = _Swapper(batch_works=False, emits_mask=True)
        outs = sw.RunBatch(object(), object(), _crops(4))

        self.assertEqual(len(outs), 4, 'every crop must still come back')
        self.assertEqual(sw.single_calls, 4, 'fallback runs one inference per crop')

        masks = sw.take_masks()
        self.assertIsNotNone(masks, 'the fallback dropped the swap model mask')
        self.assertEqual(len(masks), 4,
                         'one mask per crop — a single-slot thread-local means '
                         'only the last Run survives unless the loop drains it')

    def test_fallback_masks_stay_in_crop_order(self):
        """Position IS the pairing, so order is the whole contract."""
        sw = _Swapper(batch_works=False, emits_mask=True)
        sw.RunBatch(object(), object(), _crops(3))
        masks = sw.take_masks()
        # Each single-crop inference emits mask value 1 for its one crop, so
        # every entry is a real (H,W) plane rather than a repeat of one object.
        self.assertEqual(len(masks), 3)
        for m in masks:
            self.assertEqual(np.asarray(m).ndim, 2, 'a mask per crop, not a batch axis')
        self.assertEqual(len({id(m) for m in masks}), 3,
                         'the same mask object repeated means one crop won')

    def test_masks_are_cleared_after_reading(self):
        sw = _Swapper(batch_works=False, emits_mask=True)
        sw.RunBatch(object(), object(), _crops(2))
        self.assertIsNotNone(sw.take_masks())
        self.assertIsNone(sw.take_masks(), 'take_masks must consume exactly once')

    def test_a_maskless_model_publishes_none_not_a_list_of_none(self):
        """`if _m:` upstream must not see a truthy list of Nones."""
        sw = _Swapper(batch_works=False, emits_mask=False)
        outs = sw.RunBatch(object(), object(), _crops(3))
        self.assertEqual(len(outs), 3)
        self.assertIsNone(sw.take_masks())

    def test_partial_masks_collapse_to_none(self):
        """A gap must not shift every later mask onto the wrong face."""
        sw = _Swapper(batch_works=False, emits_mask=True)
        sw._republish_masks([np.zeros((4, 4)), None, np.zeros((4, 4))])
        self.assertIsNone(sw.take_masks())

    def test_run_batch_multi_fallback_keeps_the_contract(self):
        sw = _Swapper(batch_works=False, emits_mask=True)
        reqs = [(object(), object(), c) for c in _crops(3)]
        outs = sw.RunBatchMulti(reqs)

        self.assertEqual(len(outs), 3)
        self.assertEqual(sw.single_calls, 3)
        masks = sw.take_masks()
        self.assertIsNotNone(masks, 'RunBatchMulti dropped the swap model mask')
        self.assertEqual(len(masks), 3)

    def test_fallback_output_shape_matches_the_batched_path(self):
        """[3,H,W] per crop either way, or explode_pixel_boost mis-tiles."""
        batched = _Swapper(batch_works=True, emits_mask=True).RunBatch(
            object(), object(), _crops(3))
        fell_back = _Swapper(batch_works=False, emits_mask=True).RunBatch(
            object(), object(), _crops(3))
        self.assertEqual([np.asarray(o).shape for o in batched],
                         [np.asarray(o).shape for o in fell_back])

    def test_a_permanently_broken_model_stops_retrying_the_batch_path(self):
        """A static-batch export fails the SAME way on every call — once
        RunBatch has seen that, later calls must go straight to the sequential
        fallback instead of paying for another doomed inference + exception on
        every remaining frame of the run."""
        sw = _Swapper(batch_works=False, emits_mask=True)
        sw.RunBatch(object(), object(), _crops(4))
        self.assertEqual(sw.batch_attempts, 1, 'the first call is allowed to try')
        self.assertTrue(sw._batch_unsupported)

        outs = sw.RunBatch(object(), object(), _crops(3))
        self.assertEqual(sw.batch_attempts, 1, 'a known-broken model must not retry')
        self.assertEqual(len(outs), 3)
        masks = sw.take_masks()
        self.assertEqual(len(masks), 3, 'the fallback mask contract still applies')

    def test_run_batch_multi_also_stops_retrying_after_one_failure(self):
        sw = _Swapper(batch_works=False, emits_mask=True)
        sw.RunBatchMulti([(object(), object(), c) for c in _crops(3)])
        self.assertEqual(sw.batch_attempts, 1)

        outs = sw.RunBatchMulti([(object(), object(), c) for c in _crops(2)])
        self.assertEqual(sw.batch_attempts, 1, 'a known-broken model must not retry')
        self.assertEqual(len(outs), 2)


class BatchFailureMustNotDisableTrtForSingleFrames(unittest.TestCase):
    """A model whose export only breaks at batch>1 (e.g. hyperswap's internal
    reshape baked to batch=1) must not lose TensorRT for every later
    single-frame call just because ONE batch>1 attempt failed — measured at a
    25x latency regression (23.8ms -> 602.9ms/call) before this fix, because
    _infer() used to call _rebuild_without_trt() for ANY failure regardless
    of batch size."""

    def test_a_batch_gt1_failure_never_calls_the_trt_disabling_rebuild(self):
        sw = _TrtGateSwapper()
        feed = {'target': np.zeros((2, 3, 8, 8), np.float32),
                'source': np.zeros((2, 512), np.float32)}
        with self.assertRaises(RuntimeError):
            sw._infer(feed)
        self.assertEqual(sw.rebuild_calls, 0,
                         'a batch>1 failure must re-raise immediately, never '
                         'attempting the TRT-disabling rebuild')
        self.assertFalse(sw._trt_disabled)

    def test_a_genuine_batch1_failure_still_gets_the_trt_fallback(self):
        """The rebuild-and-retry behaviour itself must still work for what it
        was designed for: a real single-frame TRT failure (e.g. GHOST)."""
        sw = _TrtGateSwapper()
        feed = {'target': np.zeros((1, 3, 8, 8), np.float32),
                'source': np.zeros((1, 512), np.float32)}
        out = sw._infer(feed)
        self.assertEqual(sw.rebuild_calls, 1)
        self.assertTrue(sw._trt_disabled)
        self.assertEqual(out[0].shape[0], 1)

    def test_a_batch_gt1_failure_does_not_poison_a_later_batch1_call(self):
        """The regression case: a prior batch>1 failure must not leave the
        NEXT (unrelated) batch=1 call already condemned to a disabled-TRT
        session it never got a fair shot at."""
        sw = _TrtGateSwapper()
        batch_feed = {'target': np.zeros((2, 3, 8, 8), np.float32),
                     'source': np.zeros((2, 512), np.float32)}
        with self.assertRaises(RuntimeError):
            sw._infer(batch_feed)
        self.assertFalse(sw._trt_disabled,
                         'TRT must still be enabled after only a batch>1 failure')

        single_feed = {'target': np.zeros((1, 3, 8, 8), np.float32),
                      'source': np.zeros((1, 512), np.float32)}
        sw._infer(single_feed)
        self.assertEqual(sw.rebuild_calls, 1,
                         'the single-frame call gets its OWN fair attempt — '
                         'exactly one rebuild, triggered by ITS OWN failure, '
                         'not inherited from the earlier batch failure')


if __name__ == '__main__':
    unittest.main()
