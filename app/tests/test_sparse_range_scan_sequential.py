"""Stage 1 sequential NVDEC scan: the demuxer, the shot ledger and the edge expansion (no GPU, no video).

``run_demux_hw`` is driven by a reader double that hands out constant-colour CPU tensors, so what is asserted is the
scheduling: where the samples fall, where the cuts are, and that a shot with a positive sample stops being sampled.
The real-decoder behaviour (exact cut frames from a decoded file, a short stream failing safe) is in
``test_sparse_range_scan.NvdecSequentialScanTest``.
"""
import unittest

import numpy as np
import torch

from roop.sparse_range_scan import (
    ShotLedger, build_active_ranges, dilation_pad, gpu_luma_thumbs, luma_delta, luma_thumb, run_demux_hw,
)

DARK, BRIGHT, MID = 20, 220, 120


class FakeFrame:
    def __init__(self, index, value, fail_bgr=False):
        self.index = index
        self.tensor = torch.full((3, 36, 64), float(value)).to(torch.uint8)
        self._fail = fail_bgr
        self._value = value

    def bgr(self):
        if self._fail:
            raise RuntimeError("download failed")
        return np.full((36, 64, 3), self._value, np.uint8)


class FakeReader:
    """What NvZeroCopyReader exposes to the demuxer: read_batch(n) over a list of per-frame values."""

    def __init__(self, values, fail_at=()):
        self.values = list(values)
        self.pos = 0
        self.batches = []
        self.fail_at = set(fail_at)

    def read_batch(self, n):
        self.batches.append(n)
        out = [FakeFrame(self.pos + k, v, (self.pos + k) in self.fail_at)
               for k, v in enumerate(self.values[self.pos:self.pos + n])]
        self.pos += len(out)
        return out


def _run(values, stride, ledger=None, on_put=None, stopped=lambda: False, **kw):
    got = []

    def put(item):
        got.append(item)
        if on_put is not None:
            return on_put(item)
        return True

    info = run_demux_hw(FakeReader(values, kw.pop('fail_at', ())), len(values), stride, put, stopped, ledger, **kw)
    return info, got


class GpuThumbTest(unittest.TestCase):
    def test_matches_the_cv2_luma_thumbnail_scale(self):
        a = np.full((90, 160, 3), DARK, np.uint8)
        b = np.full((90, 160, 3), BRIGHT, np.uint8)
        cpu = luma_delta(luma_thumb(a), luma_thumb(b))
        ta = torch.from_numpy(a[..., ::-1].copy()).permute(2, 0, 1)
        tb = torch.from_numpy(b[..., ::-1].copy()).permute(2, 0, 1)
        thumbs = gpu_luma_thumbs([ta, tb], torch)
        self.assertEqual(tuple(thumbs.shape), (2, 36, 64))
        self.assertAlmostEqual(float((thumbs[0] - thumbs[1]).abs().mean()), cpu, delta=1.0)


class SequentialDemuxTest(unittest.TestCase):
    def test_every_frame_is_decoded_in_order_and_sampled_on_the_grid(self):
        info, got = _run([DARK] * 40, 6)
        self.assertEqual(info['decoded'], 40)
        self.assertEqual([i for _, i, _ in got], [0, 6, 12, 18, 24, 30, 36, 39])   # grid + the last frame
        self.assertEqual(info['cuts'], 0)
        self.assertEqual(info['scene_cuts'], [])
        self.assertEqual(info['decode_backend'], 'nvdec_sequential')

    def test_a_cut_is_found_on_its_exact_frame_and_its_first_frame_is_sampled(self):
        info, got = _run([DARK] * 14 + [BRIGHT] * 26, 6)
        self.assertEqual(info['scene_cuts'], [14])
        self.assertIn(14, [i for _, i, _ in got])                  # the new shot is sampled where it starts ...
        self.assertEqual([i for _, i, _ in got if i >= 14], [14, 20, 26, 32, 38, 39])   # ... and the grid restarts there

    def test_a_flash_shorter_than_the_stride_is_never_stepped_over(self):
        # a 2-frame flash at 7-8: the cut INTO it (7) and the cut OUT (9) each start a shot that is sampled at once
        info, got = _run([DARK] * 7 + [BRIGHT] * 2 + [DARK] * 31, 6)
        self.assertEqual(info['scene_cuts'], [7, 9])
        sampled = {i for _, i, _ in got}
        self.assertIn(7, sampled)
        self.assertIn(9, sampled)

    def test_a_gradual_change_is_not_a_cut(self):
        values = [20 + 2 * i for i in range(60)]                    # 2 levels per frame: far below CUT_DELTA_Y
        info, _ = _run(values, 6)
        self.assertEqual(info['scene_cuts'], [])

    def test_a_cut_on_a_chunk_boundary_is_compared_with_the_previous_chunk(self):
        info, _ = _run([DARK] * 16 + [BRIGHT] * 16, 6, chunk=16)
        self.assertEqual(info['scene_cuts'], [16])

    def test_a_positive_sample_ends_sampling_of_its_shot_but_not_of_the_next_one(self):
        ledger = ShotLedger()
        # shot 1 = frames 0-19, shot 2 = frames 20-39. The detector says "positive" at frame 0 (its first sample).
        def on_put(item):
            if item[1] == 0:
                ledger.mark_positive(0)
            return True
        info, got = _run([DARK] * 20 + [BRIGHT] * 20, 4, ledger, on_put)
        sampled = [i for _, i, _ in got]
        self.assertEqual([i for i in sampled if i < 20], [0])      # nothing more from the resolved shot
        self.assertIn(20, sampled)                                  # the next shot is its own question
        self.assertIn(24, sampled)

    def test_the_last_frame_of_an_unresolved_shot_is_sampled_but_not_of_a_resolved_one(self):
        ledger = ShotLedger()
        info, got = _run([DARK] * 31, 6, ledger)
        self.assertEqual([i for _, i, _ in got][-2:], [24, 30])
        ledger2 = ShotLedger()
        _, got2 = _run([DARK] * 31, 6, ledger2, lambda item: (ledger2.mark_positive(item[1]) or True))
        self.assertEqual([i for _, i, _ in got2], [0])

    def test_stop_flag_ends_the_scan_and_says_so(self):
        n = {'calls': 0}

        def stopped():
            n['calls'] += 1
            return n['calls'] > 2
        info, got = _run([DARK] * 200, 6, stopped=stopped, chunk=16)
        self.assertTrue(info['stopped'])
        self.assertLessEqual(info['decoded'], 32)

    def test_put_refusing_a_sample_stops_the_scan(self):
        info, got = _run([DARK] * 100, 6, on_put=lambda item: False)
        self.assertTrue(info['stopped'])
        self.assertEqual(len(got), 1)

    def test_a_stream_that_ends_early_reports_how_far_it_got(self):
        reader = FakeReader([DARK] * 20)
        got = []
        info = run_demux_hw(reader, 100, 6, lambda item: got.append(item) or True, lambda: False)
        self.assertEqual(info['decoded'], 20)
        self.assertFalse(info['stopped'])

    def test_a_failed_download_is_a_read_failure_not_a_crash(self):
        info, got = _run([DARK] * 20, 6, fail_at={6})
        self.assertEqual(info['read_failures'], 1)
        bad = [item for item in got if item[1] == 6]
        self.assertEqual(len(bad), 1)
        self.assertIsNone(bad[0][2])                                # the worker turns a None frame into an error

    def test_reads_are_chunked_not_per_frame(self):
        reader = FakeReader([DARK] * 100)
        run_demux_hw(reader, 100, 6, lambda item: True, lambda: False, None, chunk=16)
        self.assertEqual(reader.batches[0], 16)
        self.assertLessEqual(len(reader.batches), 7)


class ShotLedgerTest(unittest.TestCase):
    def test_skip_counts_only_samples_of_a_positive_shot(self):
        ledger = ShotLedger()
        ledger.register(0, 0)
        ledger.register(4, 0)
        ledger.register(8, 1)
        self.assertFalse(ledger.skip(4))
        ledger.mark_positive(0)
        self.assertTrue(ledger.skip(4))
        self.assertFalse(ledger.skip(8))
        self.assertEqual(ledger.skipped, 1)
        self.assertTrue(ledger.resolved(0))
        self.assertFalse(ledger.resolved(1))

    def test_unknown_sample_is_not_skipped(self):
        ledger = ShotLedger()
        ledger.mark_positive(99)
        self.assertFalse(ledger.skip(99))


class ExpandToEdgesTest(unittest.TestCase):
    def test_default_keeps_the_pad_on_an_open_side(self):
        ranges, _ = build_active_ranges([40], 100, 6, scene_cuts=[75])
        self.assertEqual(ranges, [(30, 74)])

    def test_first_and_last_shot_expand_to_the_clip_edge(self):
        ranges, mask = build_active_ranges([40], 100, 6, scene_cuts=[], expand_to_edges=True)
        self.assertEqual(ranges, [(0, 99)])                         # one shot: a positive anywhere covers it
        self.assertTrue(mask.all())

    def test_edges_only_open_sides_expand(self):
        ranges, _ = build_active_ranges([40], 100, 6, scene_cuts=[25, 75], expand_to_edges=True)
        self.assertEqual(ranges, [(25, 74)])
        ranges, _ = build_active_ranges([10], 100, 6, scene_cuts=[25, 75], expand_to_edges=True)
        self.assertEqual(ranges, [(0, 24)])
        ranges, _ = build_active_ranges([90], 100, 6, scene_cuts=[25, 75], expand_to_edges=True)
        self.assertEqual(ranges, [(75, 99)])

    def test_trim_bounds_are_the_edges(self):
        ranges, _ = build_active_ranges([40], 100, 6, trim_in=10, trim_out=80, expand_to_edges=True)
        self.assertEqual(ranges, [(10, 80)])

    def test_no_positive_is_still_empty(self):
        ranges, mask = build_active_ranges([], 100, 6, expand_to_edges=True)
        self.assertEqual(ranges, [])
        self.assertFalse(mask.any())


if __name__ == '__main__':
    unittest.main()
