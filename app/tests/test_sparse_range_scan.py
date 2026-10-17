"""Unit tests for Sparse Auto-Range Pre-Scan."""

import unittest
import numpy as np

import roop.globals
from roop.sparse_range_scan import cluster_hits_into_ranges
from settings import Settings


class SparseRangeScanClusteringTest(unittest.TestCase):
    def test_cluster_single_hit(self):
        hits = [60]
        total = 200
        stride = 30
        padding = 15
        ranges, mask = cluster_hits_into_ranges(hits, total, stride, padding)
        self.assertEqual(ranges, [(45, 75)])
        self.assertEqual(len(mask), total)
        self.assertTrue(np.all(mask[45:76]))
        self.assertFalse(np.any(mask[:45]))
        self.assertFalse(np.any(mask[76:]))

    def test_cluster_contiguous_hits_merged(self):
        hits = [30, 60, 90]
        total = 300
        stride = 30
        padding = 15
        ranges, mask = cluster_hits_into_ranges(hits, total, stride, padding)
        self.assertEqual(ranges, [(15, 105)])
        self.assertEqual(np.sum(mask), 105 - 15 + 1)

    def test_cluster_disjoint_ranges(self):
        hits = [30, 60, 200, 230]
        total = 300
        stride = 30
        padding = 15
        ranges, mask = cluster_hits_into_ranges(hits, total, stride, padding)
        self.assertEqual(ranges, [(15, 75), (185, 245)])
        self.assertTrue(mask[30])
        self.assertTrue(mask[200])
        self.assertFalse(mask[120])

    def test_cluster_boundary_clamping(self):
        hits = [5, 295]
        total = 300
        stride = 30
        padding = 20
        ranges, mask = cluster_hits_into_ranges(hits, total, stride, padding)
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], 299)

    def test_cluster_empty_hits_failsafe_full(self):
        total = 500
        ranges, mask = cluster_hits_into_ranges([], total, 30, 15)
        self.assertEqual(ranges, [(0, 499)])
        self.assertTrue(np.all(mask))

    def test_zero_total_frames(self):
        ranges, mask = cluster_hits_into_ranges([10], 0, 30, 15)
        self.assertEqual(ranges, [])
        self.assertEqual(len(mask), 0)


class SparseRangeScanSettingsWiringTest(unittest.TestCase):
    def test_sparse_auto_range_default_in_settings(self):
        s = Settings.__new__(Settings)
        data = {}
        s.sparse_auto_range = s.default_get(data, 'sparse_auto_range', False)
        self.assertFalse(s.sparse_auto_range)

    def test_sparse_auto_range_enabled_in_settings(self):
        s = Settings.__new__(Settings)
        data = {'sparse_auto_range': True}
        s.sparse_auto_range = s.default_get(data, 'sparse_auto_range', False)
        self.assertTrue(s.sparse_auto_range)

    def test_sparse_auto_range_default_in_globals(self):
        self.assertFalse(getattr(roop.globals, 'sparse_auto_range', True))


if __name__ == '__main__':
    unittest.main()
