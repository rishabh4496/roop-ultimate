"""Stage 16: the differential-margin lock gate and the bystander bank.

Pure numpy; no GPU, no detector. Every refusal test is paired with an
acceptance test on the SAME inputs one notch the other side of the line, so a
gate that always says yes (or always says no) fails one of them.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.bystander_bank import (MARGIN, NON_TARGET_BYSTANDER, TARGET, TAU_HOLD,
                                 TAU_LOCK, UNKNOWN, IdentityBank, TargetBank,
                                 TrackLedger, bystanders_from_tracks,
                                 differential_margin_gate)


def _vec(seed, dim=512):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _mix(a, b, wa):
    """A vector with cosine ~wa to `a` and the rest toward `b`."""
    v = wa * a + (1.0 - wa) * b
    return v / np.linalg.norm(v)


class DifferentialMarginGateTest(unittest.TestCase):

    def test_binds_when_above_lock_and_clear_of_every_bystander(self):
        d = differential_margin_gate(0.60, 0.10)
        self.assertTrue(d.bound)
        self.assertEqual(d.label, TARGET)

    def test_margin_is_strict_on_the_inside(self):
        # 0.60 vs 0.49: margin 0.11 < 0.12 -> refused; 0.60 vs 0.47: 0.13 -> bound.
        refused = differential_margin_gate(0.60, 0.49)
        self.assertFalse(refused.bound)
        self.assertEqual(refused.label, UNKNOWN)
        self.assertIn('margin', refused.reason)
        self.assertTrue(differential_margin_gate(0.60, 0.47).bound)

    def test_lock_threshold_is_enforced_when_no_bystander_exists(self):
        self.assertFalse(differential_margin_gate(0.41, -1.0).bound)
        self.assertTrue(differential_margin_gate(0.43, -1.0).bound)

    def test_closer_to_a_bystander_is_hard_labelled_even_above_tau(self):
        d = differential_margin_gate(0.55, 0.60)
        self.assertFalse(d.bound)
        self.assertEqual(d.label, NON_TARGET_BYSTANDER)

    def test_hysteresis_holds_a_locked_track_below_tau_lock(self):
        # 0.30 is under the lock bar (0.42) but over the hold floor (0.22).
        self.assertFalse(differential_margin_gate(0.30, 0.05, locked=False).bound)
        held = differential_margin_gate(0.30, 0.05, locked=True)
        self.assertTrue(held.bound)
        self.assertEqual(held.reason, 'held')

    def test_hold_needs_flow_consistency(self):
        self.assertTrue(differential_margin_gate(
            0.30, 0.05, locked=True, flow_consistent=True).bound)
        self.assertFalse(differential_margin_gate(
            0.30, 0.05, locked=True, flow_consistent=False).bound)

    def test_hold_floor(self):
        self.assertFalse(differential_margin_gate(0.21, -1.0, locked=True).bound)
        self.assertTrue(differential_margin_gate(0.23, -1.0, locked=True).bound)

    def test_defaults_are_the_briefs(self):
        self.assertEqual((TAU_LOCK, TAU_HOLD, MARGIN), (0.42, 0.22, 0.12))


class IdentityBankTest(unittest.TestCase):

    def test_same_person_merges_different_people_do_not(self):
        a, b = _vec(1), _vec(2)
        bank = IdentityBank(merge_sim=0.55)
        self.assertEqual(bank.add(a), 0)
        self.assertEqual(bank.add(_mix(a, _vec(3), 0.95)), 0)
        self.assertEqual(bank.add(b), 1)
        self.assertEqual(len(bank), 2)

    def test_capacity_is_bounded(self):
        bank = IdentityBank(merge_sim=0.99, max_entries=4)
        for i in range(20):
            bank.add(_vec(100 + i))
        self.assertLessEqual(len(bank), 4)

    def test_rejects_unusable_embeddings(self):
        bank = IdentityBank()
        self.assertEqual(bank.add(None), -1)
        self.assertEqual(bank.add(np.zeros(512)), -1)
        self.assertEqual(bank.add(np.full(512, np.nan)), -1)
        self.assertEqual(len(bank), 0)

    def test_empty_bank_reports_no_bystander(self):
        self.assertEqual(IdentityBank().max_similarity(_vec(1)), (-1.0, -1))

    def test_target_bank_keeps_every_angle(self):
        front, profile = _vec(10), _vec(11)       # unrelated: two angles
        bank = TargetBank([front, profile])
        self.assertGreater(bank.similarity(front), 0.99)
        self.assertGreater(bank.similarity(profile), 0.99)
        # a single mean of the two would score ~0.7 on each -- the point.
        mean = (front + profile) / np.linalg.norm(front + profile)
        self.assertLess(float(mean @ front), 0.8)


class TrackLedgerTest(unittest.TestCase):

    def test_one_weak_frame_does_not_demote_a_confirmed_target(self):
        led = TrackLedger()
        self.assertTrue(led.observe(1, 0.60, 0.05).bound)
        self.assertEqual(led.label(1), TARGET)
        d = led.observe(1, 0.28, 0.05)             # contact frame
        self.assertTrue(d.bound)
        self.assertEqual(led.label(1), TARGET)

    def test_bystander_verdict_is_sticky_until_released(self):
        led = TrackLedger(bystander_frames=3, release_frames=4)
        for _ in range(3):
            led.observe(7, 0.30, 0.55)
        self.assertEqual(led.label(7), NON_TARGET_BYSTANDER)
        # a single passing frame must not promote it
        self.assertFalse(led.observe(7, 0.70, 0.05).bound)
        self.assertEqual(led.label(7), NON_TARGET_BYSTANDER)
        for _ in range(3):
            d = led.observe(7, 0.70, 0.05)
        self.assertTrue(d.bound)
        self.assertEqual(led.label(7), TARGET)

    def test_interrupted_release_run_restarts(self):
        led = TrackLedger(bystander_frames=1, release_frames=3)
        led.observe(2, 0.3, 0.6)
        led.observe(2, 0.7, 0.05)
        led.observe(2, 0.7, 0.05)
        led.observe(2, 0.3, 0.6)                    # breaks the run
        led.observe(2, 0.7, 0.05)
        self.assertEqual(led.label(2), NON_TARGET_BYSTANDER)

    def test_tracks_are_independent(self):
        led = TrackLedger(bystander_frames=1)
        led.observe(1, 0.3, 0.6)
        led.observe(2, 0.7, 0.05)
        self.assertEqual(led.label(1), NON_TARGET_BYSTANDER)
        self.assertEqual(led.label(2), TARGET)


class BystandersFromTracksTest(unittest.TestCase):

    def setUp(self):
        self.target = TargetBank([_vec(1)])
        self.other = _vec(2)

    def _track(self, emb, n=20, dirty=False):
        return {'emb_mean': emb, 'emb_n': n, 'emb_dirty': dirty}

    def test_clean_sustained_stranger_is_registered(self):
        bank = bystanders_from_tracks([self._track(self.other)], self.target)
        self.assertEqual(len(bank), 1)
        self.assertGreater(bank.max_similarity(self.other)[0], 0.99)

    def test_target_like_track_is_never_a_bystander(self):
        near_target = _mix(_vec(1), _vec(9), 0.7)
        bank = bystanders_from_tracks([self._track(near_target)], self.target)
        self.assertEqual(len(bank), 0)

    def test_contaminated_track_is_never_a_bystander(self):
        bank = bystanders_from_tracks([self._track(self.other, dirty=True)], self.target)
        self.assertEqual(len(bank), 0)

    def test_short_fragment_is_never_a_bystander(self):
        bank = bystanders_from_tracks([self._track(self.other, n=2)], self.target)
        self.assertEqual(len(bank), 0)

    def test_fragments_of_one_person_merge(self):
        frags = [self._track(self.other, n=30),
                 self._track(_mix(self.other, _vec(5), 0.95), n=8)]
        bank = bystanders_from_tracks(frags, self.target)
        self.assertEqual(len(bank), 1)

    def test_gate_with_registered_bystander_end_to_end(self):
        bank = bystanders_from_tracks([self._track(self.other)], self.target)
        # a face that is half target, half bystander
        face = _mix(_vec(1), self.other, 0.5)
        s_t = self.target.similarity(face)
        s_b, j = bank.max_similarity(face)
        d = differential_margin_gate(s_t, s_b, bystander_index=j)
        self.assertFalse(d.bound)                  # ambiguous -> refused
        pure = differential_margin_gate(self.target.similarity(_vec(1)),
                                        bank.max_similarity(_vec(1))[0])
        self.assertTrue(pure.bound)


if __name__ == '__main__':
    unittest.main()
