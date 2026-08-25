"""The audit must say what a gate change would BUY, not just how often it fired.

Two lines in the SWAP AUDIT are the ones users act on:

    fallback missed (over match threshold)   <- "loosen the match threshold"
    frames with no face detected at all      <- "lower the detector threshold"

Both pieces of advice are only correct if the refused population sits NEAR the
gate, and neither was ever checked against that population. This project has
implemented and reverted four gate changes for exactly that reason, so the
report now carries the recovery curve and says so out loud when the population
is nowhere near the gate.

Also guarded here: the four different failures that used to be reported as one
"over match threshold" line, and the no-face line's denominator (a FRAME count
that was being divided by a FACE count).
"""
import io
import os
import re
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from roop import procmgr_runtime as rt


class AuditDistributions(unittest.TestCase):

    def setUp(self):
        rt._audit_reset()
        rt.audit_detect_frame_begin()

    def tearDown(self):
        rt._audit_reset()

    def _report(self):
        buf = io.StringIO()
        with mock.patch('sys.stdout', buf):
            rt._audit_report()
        return buf.getvalue()

    # -- over-threshold recovery curve ------------------------------------

    def test_silent_when_nothing_was_refused_for_distance(self):
        rt._audit_hit('faces seen', 10)
        rt._audit_hit('swapped (identity lock)', 10)
        self.assertNotIn('match threshold', self._report())

    def test_near_misses_are_reported_as_recoverable(self):
        rt._audit_hit('faces seen', 100)
        rt._audit_hit('swapped (identity lock)', 50)
        for _ in range(50):
            rt._audit_hit('fallback missed (over match threshold)')
            rt.audit_over_threshold(0.77, 0.75)     # 1.027x the gate
        out = self._report()
        self.assertIn('within 1.05x the threshold', out)
        self.assertRegex(out, r'within 1\.05x the threshold \(0\.79\):\s+50')
        self.assertNotIn('NOT NEAR THE GATE', out)

    def test_far_misses_are_called_out_as_not_a_slider(self):
        rt._audit_hit('faces seen', 100)
        rt._audit_hit('swapped (identity lock)', 50)
        for _ in range(50):
            rt._audit_hit('fallback missed (over match threshold)')
            rt.audit_over_threshold(1.40, 0.75)     # 1.87x the gate
        out = self._report()
        self.assertIn('NOT NEAR THE GATE', out)
        self.assertIn('intake or', out)

    def test_curve_is_monotone_and_expressed_against_its_own_gate(self):
        rt._audit_hit('faces seen', 10)
        rt._audit_hit('swapped (identity lock)', 5)
        for d in (0.76, 0.80, 0.90, 1.10, 1.60):
            rt._audit_hit('fallback missed (over match threshold)')
            rt.audit_over_threshold(d, 0.75)
        out = self._report()
        counts = [int(m) for m in
                  re.findall(r'the threshold \([0-9.]+\):\s+(\d+)', out)]
        self.assertEqual(counts, sorted(counts), "recovery curve must be monotone")
        # 1.60 is 2.13x a 0.75 gate, so the widest bucket must NOT claim it.
        # A curve that saturates at n regardless would overstate every recovery
        # it reports, which is the one thing it exists not to do.
        self.assertEqual(counts[-1], 4)
        self.assertEqual(counts[0], 1, "only 0.76 is inside 1.05x of 0.75")

    def test_junk_distances_are_ignored_not_crashed(self):
        rt.audit_over_threshold(None, 0.75)
        rt.audit_over_threshold(float('nan'), 0.75)
        rt.audit_over_threshold(0.9, 0)          # meaningless gate
        rt.audit_over_threshold('x', 0.75)
        self.assertEqual(len(rt._audit_over), 0)

    def test_reset_clears_the_distribution_too(self):
        rt.audit_over_threshold(0.9, 0.75)
        rt._audit_reset()
        self.assertEqual(len(rt._audit_over), 0)

    # -- detector misses ---------------------------------------------------

    def test_detector_near_misses_reported(self):
        rt._audit_hit('faces seen', 10)
        rt._audit_hit('swapped (identity lock)', 10)
        for _ in range(20):
            rt._audit_hit('frames with no face detected at all')
            rt.audit_detect_frame_begin()
            rt.audit_detect_best_rejected(0.47)
            rt.audit_detect_miss(0.5)
        out = self._report()
        self.assertIn('best REJECTED candidate', out)
        self.assertIn('would return at threshold 0.45', out)
        self.assertNotIn('SAW ALMOST NOTHING', out)

    def test_detector_saw_nothing_is_called_out(self):
        rt._audit_hit('faces seen', 10)
        rt._audit_hit('swapped (identity lock)', 10)
        for _ in range(20):
            rt._audit_hit('frames with no face detected at all')
            rt.audit_detect_frame_begin()
            rt.audit_detect_best_rejected(0.02)
            rt.audit_detect_miss(0.5)
        out = self._report()
        self.assertIn('SAW ALMOST NOTHING', out)
        self.assertIn('ENGINE', out)

    def test_best_rejected_is_the_max_over_the_frames_attempts(self):
        # full-frame detect, then an ROI re-detect: the frame's evidence is the
        # best of them, not the last one tried.
        rt.audit_detect_frame_begin()
        rt.audit_detect_best_rejected(0.10)
        rt.audit_detect_best_rejected(0.44)
        rt.audit_detect_best_rejected(0.20)
        rt.audit_detect_miss(0.5)
        self.assertAlmostEqual(rt._audit_det_miss[0][0], 0.44)

    def test_miss_without_any_score_records_nothing(self):
        rt.audit_detect_frame_begin()
        rt.audit_detect_miss(0.5)
        self.assertEqual(len(rt._audit_det_miss), 0)

    # -- denominators ------------------------------------------------------

    def test_no_face_line_is_denominated_in_frames_not_faces(self):
        rt._audit_hit('faces seen', 100)
        rt._audit_hit('swapped (identity lock)', 100)
        for _ in range(300):
            rt.audit_frame_seen()
        rt._audit_hit('frames with no face detected at all', 60)
        out = self._report()
        self.assertIn('60 frames of 300 (20.0% of frames)', out)
        # and it must NOT appear in the face-denominated table above
        self.assertNotRegex(out, r'frames with no face detected at all\s+60\s+60\.0%')

    def test_no_frame_count_degrades_gracefully(self):
        rt._audit_hit('faces seen', 100)
        rt._audit_hit('swapped (identity lock)', 100)
        rt._audit_hit('frames with no face detected at all', 60)
        out = self._report()
        self.assertIn('60 frames had NO face detected', out)


class RefusalBucketsAreDistinct(unittest.TestCase):
    """The four ways to reach the fallback with nothing to paste are four
    different problems and only one of them is a threshold."""

    BUCKETS = (
        'fallback missed (over match threshold)',
        "fallback missed (this person's source already used this frame)",
        'fallback missed (no usable embedding to compare)',
        'fallback missed (no candidate person left)',
    )

    def _src(self):
        return io.open(os.path.join(APP, 'roop', 'ProcessMgr.py'),
                       encoding='utf-8').read()

    def test_process_mgr_names_all_four(self):
        src = self._src()
        for bucket in self.BUCKETS:
            self.assertIn(bucket, src, "missing refusal bucket: %s" % bucket)

    def test_distance_is_only_collected_for_the_threshold_bucket(self):
        """Collecting it for the other three would put unrelated faces into the
        recovery curve and overstate what loosening the gate buys."""
        src = self._src()
        self.assertEqual(src.count('_audit_over_threshold('), 1)
        i = src.index('_audit_over_threshold(')
        window = src[max(0, i - 400):i]
        self.assertIn('fallback missed (over match threshold)', window)


class VetoDecidedRefusalsAreNotThresholdRefusals(unittest.TestCase):
    """A distance veto confines the fallback to one person and its own gate is
    deliberately LOOSER than the match threshold — so the confined re-test can
    never pass, and reporting it as a threshold refusal sends the reader to a
    slider that had no say. Measured on d3.mp4: 166 of 166.
    """

    def _src(self):
        return io.open(os.path.join(APP, 'roop', 'ProcessMgr.py'),
                       encoding='utf-8').read()

    def test_the_bucket_exists(self):
        self.assertIn('refused by the track veto, before any threshold', self._src())

    def test_it_is_tested_before_the_generic_threshold_branch(self):
        """Ordering is the whole guard: the generic `elif n_scored` matches every
        face this branch is for, so placing it second makes this one dead."""
        src = self._src()
        veto_i = src.index("_audit_hit('refused by the track veto, before any threshold')")
        thr_i = src.index("_audit_hit('fallback missed (over match threshold)')")
        self.assertLess(veto_i, thr_i,
                        "the veto branch must be reached before the generic one")

    def test_the_condition_compares_the_two_gates_rather_than_hardcoding(self):
        """0.85 > 0.75 holds for the defaults, but both are user-settable
        (ROOP_TRACK_VETO / max_face_distance) and AdaFace rescales them. If the
        match threshold is raised past the veto gate the fallback CAN rescue,
        and this branch must stop firing."""
        src = self._src()
        i = src.index("_audit_hit('refused by the track veto, before any threshold')")
        head = src.rindex('elif', 0, i)
        cond = src[head:src.index(':', head)]
        self.assertIn('vetoed_gate', cond)
        self.assertIn('id_threshold', cond)
        self.assertIn('>=', cond)

    def test_the_gate_is_recorded_at_every_distance_veto(self):
        src = self._src()
        for kind in ('VETO_FAR_FROM_OWN', 'VETO_SINGLE_ABS'):
            i = src.index('veto_kind = %s' % kind)
            self.assertIn('vetoed_gate = _ada.scale', src[i:i + 200],
                          "%s must record the gate it used" % kind)

    def test_non_distance_vetoes_do_not_record_a_gate(self):
        """VETO_SOURCE_REUSED and VETO_OTHER_FITS are not distance decisions, so
        a face vetoed by them reaching the fallback IS a real threshold test."""
        src = self._src()
        for kind in ('VETO_SOURCE_REUSED', 'VETO_OTHER_FITS'):
            i = src.index('veto_kind = %s' % kind)
            self.assertNotIn('vetoed_gate =', src[i:i + 200],
                             "%s must not record a distance gate" % kind)


if __name__ == '__main__':
    unittest.main()
