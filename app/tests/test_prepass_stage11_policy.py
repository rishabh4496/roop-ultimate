"""Pins the ladder-stride replay in `probe_prepass_stage11.py` (Stage 11, 2026-10-19).

The probe's verdict ("striding the rescue ladder through faceless runs is lossless only WITH rewind, and then
saves <= ~10%") rests on `simulate_one`, so its accounting is tested here on hand-made rows
(first_n, full_n, t_first, t_full)."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from probe_prepass_stage11 import simulate_one  # noqa: E402

T1, T2 = 1.0, 2.0


def row(first, full):
    return (first, full, T1, T2)


class LadderStrideReplay(unittest.TestCase):
    def test_always_faces_costs_the_same_and_loses_nothing(self):
        rows = [row(1, 1)] * 20
        cost, laddered, lost = simulate_one(rows, k=2, n=8, rewind=True)
        self.assertEqual((cost, laddered, lost), (20 * T2, 20, 0))

    def test_faceless_run_skips_the_ladder_after_k_futile_frames(self):
        rows = [row(0, 0)] * 12
        cost, laddered, lost = simulate_one(rows, k=2, n=8, rewind=True)
        # frames 0,1 (streak < k) and 8 (every n) get the ladder; the other 9 pay the plain pass only.
        self.assertEqual(laddered, 3)
        self.assertEqual(cost, 3 * T2 + 9 * T1)
        self.assertEqual(lost, 0)

    def test_rewind_recovers_a_ladder_only_face_found_after_the_gap(self):
        # ladder-only face on frame 5 (skipped), plain pass sees a face on frame 6 -> rewind runs frame 5's ladder.
        rows = [row(0, 0)] * 5 + [row(0, 1)] + [row(1, 1)]
        _, laddered, lost = simulate_one(rows, k=2, n=8, rewind=True)
        self.assertEqual(lost, 0)
        self.assertGreater(laddered, 3)

    def test_without_rewind_the_same_face_is_lost(self):
        rows = [row(0, 0)] * 5 + [row(0, 1)] + [row(1, 1)]
        _, _, lost = simulate_one(rows, k=2, n=8, rewind=False)
        self.assertEqual(lost, 1)

    def test_a_ladder_only_face_that_vanishes_before_any_signal_is_lost_even_with_rewind(self):
        rows = [row(0, 0)] * 5 + [row(0, 1)] + [row(0, 0)] * 3
        _, _, lost = simulate_one(rows, k=2, n=8, rewind=True)
        self.assertEqual(lost, 1)


if __name__ == "__main__":
    unittest.main()
